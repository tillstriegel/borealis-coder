"""Explicit delegated evidence crosses handoffs without opening private history."""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from borealis_coder.agent import build_runner
from borealis_coder.models import ModelResponse, Role, ToolCall, Usage
from borealis_coder.providers.mock import MockProvider
from tests.helpers import make_config

FACT = 'RETAINED_ONLY_FACT: the useful source value is 731.'


def response(text='', calls=()):
    return ModelResponse(text=text, tool_calls=list(calls), usage=Usage(requests=1))


class DelegatedArtifactHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_mixed_success_and_error_artifacts_preserve_the_completed_handoff(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/'source.txt').write_text('preview filler\n' * 2000 + FACT + '\n')
            runner = await build_runner(root, config=make_config(
                root, context={'tool_output_chars': 2000}), interactive=False)
            self.addAsyncCleanup(runner.close)
            provider = runner.providers[0].provider
            assert isinstance(provider, MockProvider)
            handoffs, references, reads, findings = [], {}, {}, []

            def handler(request, call):
                last = request.messages[-1]
                if request.metadata.get('delegated'):
                    if last.role != Role.TOOL:
                        return response(calls=[
                            ToolCall(id='good-source', name='read_file', arguments={
                                'path': 'source.txt', 'start_line': None,
                                'end_line': None, 'max_chars': 100000}),
                            ToolCall(id='error-source', name='git_diff', arguments={
                                'path': None, 'staged': False, 'stat': False}),
                        ])
                    return response('Source inspected; Git reported that this workspace is not a repository.')
                if last.role != Role.TOOL:
                    return response(calls=[ToolCall(id='delegate-mixed', name='delegate_task', arguments={
                        'task': 'Read the source and inspect its Git diff.',
                        'max_turns': 4, 'capability': 'read-only'})])
                if last.tool_name == 'delegate_task':
                    handoffs.append(last)
                    if last.is_error:
                        return response('The child handoff failed unexpectedly.')
                    references.update({item['tool']: item for item in last.metadata['evidence']
                                       if item.get('output_artifact', {}).get('artifact_id')})
                    return response(calls=[ToolCall(id=f'parent-{name}', name='read_artifact', arguments={
                        'artifact_id': references[name]['output_artifact']['artifact_id'],
                        'offset': 0, 'limit': 1000, 'query': FACT if name == 'read_file' else '',
                    }) for name in ('read_file', 'git_diff')])
                if last.tool_name == 'read_artifact':
                    reads.update({message.tool_call_id: message for message in request.messages
                                  if message.role == Role.TOOL and message.tool_name == 'read_artifact'})
                    return response(calls=[ToolCall(id='reject-error-finding', name='record_finding', arguments={
                        'tool_call_id': references['git_diff']['tool_call_id'], 'quote': 'git'})])
                if last.tool_name == 'record_finding':
                    findings.append(last)
                    return response('Both evidence artifacts remain readable; failed output is not a successful finding.')
                return response('Unexpected tool result.')

            provider.handler = handler
            result = await asyncio.wait_for(runner.run('Inspect mixed delegated evidence.'), 15)
            self.assertEqual(len(handoffs), 1)
            self.assertFalse(handoffs[0].is_error, handoffs[0].content)
            self.assertEqual(handoffs[0].metadata['completion_status'], 'completed')
            self.assertEqual(set(references), {'read_file', 'git_diff'})
            self.assertEqual(set(reads), {'parent-read_file', 'parent-git_diff'})
            self.assertTrue(all(not message.is_error for message in reads.values()))
            self.assertIn(FACT, reads['parent-read_file'].content)
            self.assertIn('git', reads['parent-git_diff'].content.lower())
            self.assertEqual(len(findings), 1)
            self.assertTrue(findings[0].is_error)
            self.assertIn('successful', findings[0].content)
            self.assertEqual(result.stop_reason.value, 'end_turn', result.error)
            errors = [row for row in runner.sessions.tool_calls(result.session_id)
                      if row['tool_name'] == 'git_diff']
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]['status'], 'error')

    async def test_returned_artifact_can_be_read_and_quoted_by_each_receiving_parent(self):
        for nested in (False, True):
            with self.subTest(nested=nested), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                (root/'source.txt').write_text('preview filler\n' * 2000 + FACT + '\n' + 'tail filler\n' * 2000)
                # Keep structured search results intact; the source fact still lies
                # beyond both ends of the bounded file preview, asserted below.
                runner = await build_runner(root, config=make_config(root, context={'tool_output_chars': 8000}), interactive=False)
                self.addAsyncCleanup(runner.close)
                provider = runner.providers[0].provider
                assert isinstance(provider, MockProvider)
                references, searches, reads, findings, leaf_previews = {}, {}, {}, {}, []

                def handler(request, call, nested=nested, references=references, searches=searches, reads=reads,
                            findings=findings, leaf_previews=leaf_previews):
                    depth = request.metadata.get('delegation_depth', 0)
                    last = request.messages[-1]
                    if last.role != Role.TOOL:
                        if depth == 0 or (nested and depth == 1):
                            return response(calls=[ToolCall(id=f'delegate-{depth}', name='delegate_task', arguments={
                                'task': 'Read the oversized source and return its evidence references.',
                                'max_turns': 6, 'capability': 'read-only'})])
                        return response(calls=[ToolCall(id=f'source-{depth}', name='read_file', arguments={
                            'path': 'source.txt', 'start_line': None, 'end_line': None, 'max_chars': 100000})])
                    if last.tool_name == 'read_file':
                        leaf_previews.append(last.content)
                        return response('Read source.txt; the retained artifact contains the full evidence.')
                    if last.tool_name == 'delegate_task':
                        refs = [item for item in last.metadata.get('evidence', [])
                                if item.get('tool') == 'read_file' and item.get('output_artifact', {}).get('artifact_id')]
                        if not refs:
                            return response('No usable source reference was returned.')
                        references[depth] = refs[0]
                        return response(calls=[ToolCall(id=f'search-{depth}', name='search_history', arguments={
                            'query': FACT, 'after': 0, 'offset': 0, 'artifact_after': None})])
                    if last.tool_name == 'search_history':
                        payload = json.loads(last.content.split('\n', 1)[1])
                        matches = [item for item in payload['artifacts']
                                   if item['artifact_id'] == references[depth]['output_artifact']['artifact_id']]
                        searches[depth] = matches
                        if not matches:
                            return response('The retained artifact cannot be recovered through history search.')
                        return response(calls=[ToolCall(id=f'artifact-{depth}', name='read_artifact', arguments={
                            'artifact_id': matches[0]['artifact_id'],
                            'offset': 0, 'limit': 1000, 'query': FACT})])
                    if last.tool_name == 'read_artifact':
                        reads[depth] = last
                        return response(calls=[ToolCall(id=f'finding-{depth}', name='record_finding', arguments={
                            'tool_call_id': references[depth]['tool_call_id'], 'quote': FACT})])
                    if last.tool_name == 'record_finding':
                        findings[depth] = last
                        return response('Verified retained source value 731; evidence review complete.')
                    return response('Unexpected tool result.')

                provider.handler = handler
                result = await asyncio.wait_for(runner.run('Inspect delegated retained evidence.'), 15)
                expected_receivers = {0, 1} if nested else {0}
                self.assertEqual(set(searches), expected_receivers, result.text)
                for matches in searches.values():
                    self.assertEqual(len(matches), 1, result.text)
                    self.assertIn(FACT, matches[0]['preview'])
                self.assertEqual(set(reads), expected_receivers, result.text)
                self.assertEqual(set(findings), expected_receivers, result.text)
                self.assertEqual(result.text, 'Verified retained source value 731; evidence review complete.')
                self.assertTrue(leaf_previews)
                self.assertTrue(all(FACT not in preview for preview in leaf_previews))
                for depth in expected_receivers:
                    self.assertFalse(reads[depth].is_error, reads[depth].content)
                    self.assertIn(FACT, reads[depth].content)
                    self.assertFalse(findings[depth].is_error, findings[depth].content)
                    self.assertEqual(findings[depth].metadata['retained_finding']['quote'], FACT)
                    self.assertEqual(findings[depth].metadata['retained_finding']['output_artifact'],
                                     references[depth]['output_artifact'])

    def test_handoff_is_receipt_bound_private_idempotent_and_durable(self):
        import json

        from borealis_coder.errors import SessionError
        from borealis_coder.safety.redaction import Redactor
        from borealis_coder.sessions import SessionStore

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root/'sessions.sqlite3'
            store = SessionStore(path)
            self.addCleanup(store.close)
            session = store.create_session(workspace=root, provider='mock', model='offline')
            donor, sibling = 'root-run.one', 'root-run.two'
            for scope, delegation in ((donor, 'one'), (sibling, 'two')):
                store.ensure_delegated_conversation(session.id, scope, delegation, 1, delegation_call_id=delegation)
                store.start_tool_call(session.id, 'root-run', delegation, 'delegate_task', {})

            def source(scope, call_id, content, *, max_bytes=2000000):
                store.start_tool_call(session.id, scope, call_id, 'read_file', {})
                artifact = store.save_output_artifact(scope, root, content, source=call_id,
                                                      redactor=Redactor(), max_bytes=max_bytes)
                store.complete_tool_call(session.id, call_id, output='Short source preview.', is_error=False,
                                         metadata={'output_artifact': artifact})
                return {'tool_call_id': call_id, 'tool': 'read_file', 'output_artifact': artifact}

            returned = source(donor, 'one.read', 'Retained snowman ☃ and data. ' * 20, max_bytes=64)
            private = source(donor, 'one.private', 'Unreturned private output')
            structured = source(donor, 'one.json', json.dumps({'output': 'Decoded snowman ☃\nnext line'}))
            foreign = source(sibling, 'two.read', 'Sibling private output')
            root_artifact = store.save_output_artifact(session.id, root, 'Ancestor private output',
                                                       source='root-read', redactor=Redactor())
            returned_id = returned['output_artifact']['artifact_id']
            private_id = private['output_artifact']['artifact_id']
            structured_id = structured['output_artifact']['artifact_id']
            before = store._connection.execute(
                'SELECT session_id,artifact_id,content,metadata_json FROM output_artifacts ORDER BY artifact_id').fetchall()
            before = [tuple(row) for row in before]
            with self.assertRaises(SessionError):
                store.read_output_artifact(session.id, root, returned_id)
            # A known but unreturned ID cannot be authorized by changing its source receipt.
            with self.assertRaises(SessionError):
                store.handoff_output_artifacts(donor, session.id, root, 'one',
                                              [{'tool_call_id': 'one.read', 'output_artifact': private['output_artifact']}])
            with self.assertRaises(SessionError):
                store.read_output_artifact(session.id, root, private_id)
            with self.assertRaises(SessionError):
                store.handoff_output_artifacts(donor, sibling, root, 'one', [returned])
            with self.assertRaises(SessionError):
                store.handoff_output_artifacts(donor, session.id, root, 'two', [returned])
            with self.assertRaises(SessionError):
                store.handoff_output_artifacts(donor, session.id, root, 'one', [foreign])
            with self.assertRaises(SessionError):
                store.handoff_output_artifacts(donor, session.id, root/'wrong-workspace', 'one', [returned])
            # A mixed invalid batch cannot leave a partial handoff behind.
            with self.assertRaises(SessionError):
                store.handoff_output_artifacts(donor, session.id, root, 'one', [returned, foreign])
            with self.assertRaises(SessionError):
                store.read_output_artifact(session.id, root, returned_id)
            with self.assertRaises(SessionError):
                store.ensure_delegated_conversation(session.id, donor, 'one', 1, delegation_call_id='two')
            for _ in range(2):
                store.handoff_output_artifacts(donor, session.id, root, 'one', [returned, structured])
            matches = store.search_output_artifacts(session.id, root, 'snowman')
            self.assertEqual([row['artifact_id'] for row in matches], sorted([returned_id, structured_id]))
            self.assertTrue(all('snowman' in row['preview'] for row in matches))
            continuation = store.search_output_artifacts(session.id, root, 'snowman',
                                                         after=matches[0]['artifact_id'])
            self.assertEqual(continuation, matches[1:])
            self.assertEqual(store.search_output_artifacts(session.id, root, returned_id)[0]['artifact_id'], returned_id)
            self.assertEqual(store.search_output_artifacts(session.id, root, 'Ancestor private output')[0]['artifact_id'],
                             root_artifact['artifact_id'])
            for scope, query in ((session.id, 'Unreturned private output'), (session.id, 'Sibling private output'),
                                 (sibling, returned_id), (donor, 'Sibling private output'),
                                 (donor, 'Ancestor private output')):
                self.assertEqual(store.search_output_artifacts(scope, root, query), [])
            with self.assertRaises(SessionError):
                store.search_output_artifacts(session.id, root/'wrong-workspace', 'snowman')
            value = store.read_output_artifact(session.id, root, returned_id, limit=1000)
            self.assertEqual({key: value[key] for key in returned['output_artifact']}, returned['output_artifact'])
            self.assertEqual(value['content'], store.read_output_artifact(donor, root, returned_id, limit=1000)['content'])
            self.assertEqual(value['status'], 'quota_limited')
            self.assertLessEqual(value['stored_bytes'], 64)
            self.assertTrue(store.output_artifact_has_quote(session.id, root, returned_id, 'Retained snowman ☃'))
            self.assertTrue(store.output_artifact_has_quote(session.id, root, structured_id,
                                                          'Decoded snowman ☃\nnext line', json_envelope=True))
            self.assertFalse(store.output_artifact_has_quote(session.id, root, structured_id,
                                                           'Decoded snowman ☃\nnext line'))
            for scope, artifact_id in ((session.id, private_id), (sibling, returned_id),
                                       (donor, foreign['output_artifact']['artifact_id']),
                                       (donor, root_artifact['artifact_id'])):
                with self.assertRaises(SessionError):
                    store.read_output_artifact(scope, root, artifact_id)
                with self.assertRaises(SessionError):
                    store.output_artifact_has_quote(scope, root, artifact_id, 'private')
            with self.assertRaises(SessionError):
                store.read_output_artifact(session.id, root/'wrong-workspace', returned_id)
            after = store._connection.execute(
                'SELECT session_id,artifact_id,content,metadata_json FROM output_artifacts ORDER BY artifact_id').fetchall()
            self.assertEqual([tuple(row) for row in after], before)
            self.assertEqual(store._connection.execute('SELECT COUNT(*) FROM delegated_artifact_handoffs').fetchone()[0], 2)
            reopened = SessionStore(path)
            self.addCleanup(reopened.close)
            reopened.handoff_output_artifacts(donor, session.id, root, 'one', [returned])
            self.assertEqual(reopened.search_output_artifacts(session.id, root, 'snowman'), matches)
            self.assertEqual(reopened.read_output_artifact(session.id, root, returned_id, limit=1000), value)
            self.assertEqual([s.id for s in reopened.list_sessions()], [session.id])
            self.assertEqual(reopened.finding_source(session.id, root, 'one.read')['metadata']['output_artifact'],
                             returned['output_artifact'])
            reopened.delete_session(session.id)
            self.assertEqual(store._connection.execute('SELECT COUNT(*) FROM delegated_artifact_handoffs').fetchone()[0], 0)

    def test_existing_unshipped_schema_seven_reopens_with_private_artifacts(self):
        from borealis_coder.errors import SessionError
        from borealis_coder.safety.redaction import Redactor
        from borealis_coder.sessions import SessionStore

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root/'sessions.sqlite3'
            store = SessionStore(path)
            session = store.create_session(workspace=root, provider='mock', model='offline')
            store.ensure_delegated_conversation(session.id, 'root-run.child', 'child', 1)
            artifact = store.save_output_artifact('root-run.child', root, 'Private existing output',
                                                  source='child.read', redactor=Redactor())
            # The original unshipped schema 7 has all current tables except this one.
            store._connection.execute('DROP TABLE delegated_artifact_handoffs')
            store._connection.commit()
            store.close()
            reopened = SessionStore(path)
            self.addCleanup(reopened.close)
            self.assertEqual(reopened._connection.execute("SELECT value FROM schema_meta WHERE key='version'").fetchone()[0], '7')
            self.assertEqual(reopened.read_output_artifact('root-run.child', root, artifact['artifact_id'])['content'],
                             'Private existing output')
            with self.assertRaises(SessionError):
                reopened.read_output_artifact(session.id, root, artifact['artifact_id'])
