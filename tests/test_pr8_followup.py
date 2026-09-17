from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from borealis_coder.agent import build_runner
from borealis_coder.agent.budget import Budget
from borealis_coder.agent.runner import ProviderRoute
from borealis_coder.config import ProviderConfig
from borealis_coder.errors import BudgetExceeded, ProviderUnavailableError, SessionError
from borealis_coder.mcp.client import MCPToolDefinition
from borealis_coder.mcp.manager import MCPTool
from borealis_coder.models import Message, ModelResponse, Role, ToolCall, Usage
from borealis_coder.providers.base import Provider
from borealis_coder.providers.openai import _strict_schema_compatible
from borealis_coder.tools import build_builtin_registry
from borealis_coder.tools.base import ToolRegistry, validate_schema
from borealis_coder.tools.history import SearchHistoryTool
from borealis_coder.tools.search import GrepTool
from tests.helpers import make_config, make_context


class AttemptProvider(Provider):
    name = 'test'

    def __init__(self, model, *, context=None, failures=0, retries=0):
        super().__init__(ProviderConfig(model=model, context_tokens=context, max_retries=retries,
                                       initial_backoff_seconds=0))
        self.failures = failures
        self.seen = []

    async def complete(self, request):
        self.seen.append(request)
        usage = Usage(requests=1, input_tokens=10, output_tokens=2, cost_usd=.2, cost_status='known',
                      native_usage={'billed_attempt': len(self.seen), 'route': self.config.model})
        if len(self.seen) <= self.failures:
            raise ProviderUnavailableError('billed failure', retryable=True, usage=usage)
        return ModelResponse(text='done', usage=usage)


class FollowupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config = make_config(self.root, agent={'max_cost_usd': 0, 'max_turns': 1,
            'max_input_tokens': 100000, 'max_output_tokens': 100,
            'compaction_safety_margin_tokens': 128, 'deterministic_compaction': True},
            cache={'response_cache_enabled': False})
        self.runner = await build_runner(self.root, config=self.config, interactive=False)
        self.addAsyncCleanup(self.runner.close)
        self.store = self.runner.sessions
        self.session = self.store.create_session(workspace=self.root, provider='test', model='primary')

    def routes(self, primary, fallback):
        self.runner.providers = [ProviderRoute('primary', 'primary', primary),
                                 ProviderRoute('fallback', 'fallback', fallback)]

    async def test_small_primary_capacity_does_not_block_large_fallback(self):
        self.config.agent.max_model_requests = 1
        primary = AttemptProvider('primary', context=500)
        fallback = AttemptProvider('fallback', context=100000)
        self.routes(primary, fallback)
        result = await self.runner.run('Retain the full instruction. ' * 100, session_id=self.session.id)
        self.assertEqual(result.text, 'done')
        self.assertEqual(len(primary.seen), 0)
        self.assertEqual(len(fallback.seen), 1)
        self.assertIn('Retain the full instruction.', fallback.seen[0].messages[-1].content)

    async def test_primary_compaction_does_not_leak_into_fallback(self):
        primary = AttemptProvider('primary', context=9000, failures=1)
        fallback = AttemptProvider('fallback', context=100000)
        self.routes(primary, fallback)
        self.store.append_message(self.session.id, Message(role=Role.USER, content='Keep the API stable.'))
        original = []
        for index in range(24):
            message = Message(role=Role.ASSISTANT, content=f'original-{index} ' + 'context ' * 350)
            original.append(message)
            self.store.append_message(self.session.id, message)
        result = await self.runner.run('Finish.', session_id=self.session.id)
        self.assertEqual(result.text, 'done')
        self.assertEqual(len(primary.seen), 1)
        self.assertEqual(len(fallback.seen), 1)
        self.assertLess(len(primary.seen[0].messages), len(original))
        fallback_messages = {message.id: message.content for message in fallback.seen[0].messages}
        for message in original:
            self.assertEqual(fallback_messages[message.id], message.content)
        self.assertNotIn('compaction_artifact_id', fallback.seen[0].metadata)

    async def test_operator_limits_remain_authoritative(self):
        primary, fallback = AttemptProvider('primary'), AttemptProvider('fallback', context=100000)
        self.routes(primary, fallback)
        self.config.agent.max_input_tokens = 500
        result = await self.runner.run('Cannot fit.', session_id=self.session.id)
        self.assertEqual(result.stop_reason.value, 'budget')
        self.assertFalse(primary.seen or fallback.seen)
        prepared_routes = []
        async def failed_budget(route):
            prepared_routes.append(route.name)
            raise BudgetExceeded('cost', 'global cost budget')
        with self.assertRaises(BudgetExceeded):
            await self.runner._complete_request(failed_budget, self.session.id, 'run', asyncio.Event(),
                'assistant', usage_sink=AsyncMock(), budget=Budget.start(self.config.agent))
        self.assertEqual(prepared_routes, ['primary'])
        self.assertFalse(primary.seen or fallback.seen)

    async def test_external_optional_schema_stays_omittable_and_builtins_stay_strict(self):
        schema = {'type': 'object', 'properties': {'path': {'type': 'string'},
            'revision': {'type': 'string'}}, 'required': ['path'], 'additionalProperties': False}
        client = AsyncMock()
        client.call_tool.return_value = {'content': [{'type': 'text', 'text': 'ok'}]}
        tool = MCPTool('test', MCPToolDefinition(name='read', description='read', input_schema=schema, annotations={}),
                       client, read_only=True)
        self.assertEqual(tool.schema()['parameters'], schema)
        validate_schema({'path': 'file.py'}, tool.schema()['parameters'])
        result = await ToolRegistry([tool]).execute(ToolCall(name=tool.name, arguments={'path': 'file.py'}),
                                                   make_context(self.root, self.config))
        self.assertFalse(result.is_error, result.output)
        self.assertEqual(client.call_tool.call_args.args[1], {'path': 'file.py'})
        for built_in in build_builtin_registry().schemas():
            self.assertTrue(_strict_schema_compatible(built_in['parameters']), built_in['name'])

    async def test_long_history_match_and_source_pagination(self):
        content = 'prefix ' * 1500 + 'MATCH-beyond-prefix' + ' tail' * 2500
        message = Message(role=Role.USER, content=content)
        self.store.append_message(self.session.id, message)
        rows = self.store.search_history(self.session.id, self.root, 'MATCH-beyond-prefix')
        self.assertIn('MATCH-beyond-prefix', rows[0]['preview'])
        self.assertLessEqual(len(rows[0]['preview']), 500)
        for source in (message.id, 'user:1'):
            offset, chunks = 0, []
            while True:
                row = self.store.search_history(self.session.id, self.root, source, offset=offset)[0]
                self.assertLessEqual(len(row['preview']), 8000)
                chunks.append(row['preview'])
                if row['next_offset'] is None:
                    break
                self.assertGreater(row['next_offset'], offset)
                offset = row['next_offset']
            self.assertEqual(''.join(chunks), content)
        context = make_context(self.root, self.config)
        context.session_id = self.session.id
        context.metadata['session_store'] = self.store
        result = await SearchHistoryTool().execute({'query': 'user:1', 'after': 0, 'offset': 10000}, context)
        self.assertIn('Historical', result.output.replace('historical', 'Historical'))
        payload = json.loads(result.output.split('\n', 1)[1])
        self.assertIn('MATCH-beyond-prefix', payload['messages'][0]['preview'])
        with self.assertRaises(SessionError):
            self.store.search_history(self.session.id, self.root / 'other', message.id)

    async def test_grep_cursor_tracks_only_matching_scope(self):
        context = make_context(self.root, self.config)
        tool = GrepTool()
        args = {'pattern': 'needle', 'path': '.', 'glob': '*.py', 'regex': False, 'case_sensitive': True,
                'context_lines': 0, 'max_results': 1}
        for mutation in ('excluded', 'modify', 'add', 'remove'):
            with self.subTest(mutation=mutation):
                first, second, extra = self.root/'a.py', self.root/'b.py', self.root/'c.py'
                extra.unlink(missing_ok=True)
                first.write_text('needle first')
                second.write_text('needle second')
                page = await tool.execute(args, context)
                cursor = page.metadata['next_cursor']
                self.assertTrue(cursor)
                if mutation == 'excluded':
                    (self.root/'build.log').write_text('changed log')
                elif mutation == 'modify':
                    second.write_text('needle changed')
                elif mutation == 'add':
                    extra.write_text('needle new')
                else:
                    second.unlink()
                continued = await tool.execute({**args, 'cursor': cursor}, context)
                if mutation == 'excluded':
                    self.assertFalse(continued.is_error, continued.output)
                    self.assertIn('needle second', continued.output)
                    self.assertNotIn('needle first', continued.output)
                    self.assertFalse(continued.metadata['next_cursor'])
                else:
                    self.assertTrue(continued.is_error)
                    self.assertIn('stale', continued.output)

    async def test_native_attempts_survive_retry_and_fallback_without_duplicate_settlement(self):
        for fallback_route in (False, True):
            with self.subTest(fallback=fallback_route):
                session = self.store.create_session(workspace=self.root, provider='test', model='primary')
                primary = AttemptProvider('primary', failures=1, retries=0 if fallback_route else 1)
                backup = AttemptProvider('fallback')
                backup.name = 'other_provider'
                self.routes(primary, backup)
                result = await self.runner.run('Finish.', session_id=session.id)
                self.assertEqual(result.text, 'done')
                usage = self.store.usage(session.id)
                self.assertEqual(usage.requests, 2)
                self.assertEqual(usage.input_tokens, 20)
                self.assertEqual(usage.output_tokens, 4)
                self.assertAlmostEqual(usage.cost_usd, .4)
                attempts = [event.data for _, event in self.store._export_events(session.id)
                            if event.type == 'model.attempt_usage']
                self.assertEqual(len(attempts), 2)
                self.assertEqual([item['provider'] for item in attempts],
                                 ['test', 'other_provider' if fallback_route else 'test'])
                self.assertEqual([item['model'] for item in attempts],
                                 ['primary', 'fallback' if fallback_route else 'primary'])
                self.assertEqual(attempts[0]['native_usage'], {'billed_attempt': 1, 'route': 'primary'})
                self.assertEqual(attempts[1]['native_usage'], {'billed_attempt': 1 if fallback_route else 2,
                    'route': 'fallback' if fallback_route else 'primary'})

    async def test_native_terminal_usage_survives_cancellation_at_route_completion(self):
        primary, backup = AttemptProvider('primary'), AttemptProvider('fallback')
        self.routes(primary, backup)
        ready = asyncio.Event()
        original_emit = self.runner.events.emit

        async def emit(event_type, **data):
            if event_type == 'model.route_completed':
                ready.set()
                await asyncio.Event().wait()
            return await original_emit(event_type, **data)

        self.runner.events.emit = emit
        task = asyncio.create_task(self.runner.run('Finish.', session_id=self.session.id))
        await asyncio.wait_for(ready.wait(), timeout=5)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        usage = self.store.usage(self.session.id)
        self.assertEqual(usage.requests, 1)
        self.assertAlmostEqual(usage.cost_usd, .2)
        attempts = [event.data for _, event in self.store._export_events(self.session.id)
                    if event.type == 'model.attempt_usage']
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]['native_usage'], {'billed_attempt': 1, 'route': 'primary'})
