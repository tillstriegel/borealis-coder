from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from borealis_coder.agent import build_runner
from borealis_coder.agent.budget import Budget, ContextBudget, prepare_route_request
from borealis_coder.agent.runner import PreparedProviderRequest
from borealis_coder.config import AgentConfig, ProviderConfig
from borealis_coder.context.builder import PromptContext
from borealis_coder.errors import BudgetExceeded, SessionError
from borealis_coder.models import (
    Message,
    ModelResponse,
    ProviderRequest,
    Role,
    ToolCall,
    ToolResult,
    Usage,
)
from borealis_coder.providers.mock import MockProvider
from borealis_coder.providers.openai import OpenAIProvider
from borealis_coder.safety.redaction import Redactor
from borealis_coder.sessions.store import SessionStore
from borealis_coder.tools.base import FunctionTool, ToolRegistry, object_schema
from borealis_coder.tools.search import GrepTool
from borealis_coder.tools.shell import ShellTool
from tests.helpers import make_config, make_context


class EvidenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.path = self.root / 'sessions.sqlite3'
        self.store = SessionStore(self.path)
        self.session = self.store.create_session(workspace=self.root, provider='mock', model='test')
        self.context = make_context(self.root)
        self.context.session_id = self.session.id
        self.context.metadata['session_store'] = self.store
        self.addCleanup(lambda: self.store.close())

    async def test_middle_output_survives_restart_with_redaction_and_scope(self):
        secret = 'sk-' + 'a' * 32
        content = 'start\n' + 'padding\n' * 5000 + 'diagnostic ' + secret + '\n' + 'tail\n' * 5000
        tool = FunctionTool(name='evidence', description='fixture', parameters=object_schema({}),
                            function=lambda *_: ToolResult(content))
        result = await ToolRegistry([tool]).execute(ToolCall(name='evidence', arguments={}), self.context)
        artifact = result.metadata['output_artifact']
        self.assertIn(artifact['artifact_id'], result.output)
        self.assertNotIn(secret, result.output)
        self.store.close()
        self.store = SessionStore(self.path)
        found = self.store.read_output_artifact(self.session.id, self.root, artifact['artifact_id'], query='diagnostic', limit=100)
        self.assertIn('diagnostic [REDACTED]', found['content'])
        self.assertEqual(found['status'], 'complete')
        other = self.store.create_session(workspace=self.root, provider='mock', model='test')
        with self.assertRaises(SessionError):
            self.store.read_output_artifact(other.id, self.root, artifact['artifact_id'])
        with self.assertRaises(SessionError):
            self.store.read_output_artifact(self.session.id, self.root / 'other', artifact['artifact_id'])
        self.store.delete_session(self.session.id)
        self.assertEqual(self.store._connection.execute('SELECT count(*) FROM output_artifacts').fetchone()[0], 0)

    async def test_shell_capture_precedes_process_truncation(self):
        self.context.config.safety.max_process_output_chars = 100
        self.context.config.context.tool_output_chars = 100
        result = await ShellTool().execute({'command': "printf '%05000dDIAGNOSTIC%05000d' 0 0", 'cwd': '.', 'timeout_seconds': 10}, self.context)
        artifact = result.metadata['output_artifact']
        found = self.store.read_output_artifact(self.session.id, self.root, artifact['artifact_id'], query='DIAGNOSTIC')
        self.assertTrue(found['content'].startswith('DIAGNOSTIC'))
        self.assertEqual(found['status'], 'complete')

    async def test_cancelled_capture_is_explicitly_partial(self):
        from borealis_coder.safety.sandbox import output_capture

        class InterruptedProcess:
            guarantees_bounded_lifecycle = True

            async def run(self, *args, **kwargs):
                capture = output_capture.get()
                assert capture
                capture('stdout', 'partial diagnostic')
                raise asyncio.CancelledError()

        self.context.process = InterruptedProcess()  # type: ignore[assignment]
        with self.assertRaises(asyncio.CancelledError):
            await ShellTool().execute({'command': 'true', 'cwd': '.'}, self.context)
        row = self.store._connection.execute('SELECT metadata_json FROM output_artifacts').fetchone()
        self.assertIn('interrupted', row[0])

    async def test_read_preview_preserves_the_edit_hash(self):
        import hashlib

        from borealis_coder.tools.filesystem import ReadFileTool

        data = b'long file line\n' * 1000
        (self.root / 'file.txt').write_bytes(data)
        result = await ReadFileTool().execute({'path': 'file.txt', 'max_chars': 100}, self.context)
        self.assertIn('sha256: ' + hashlib.sha256(data).hexdigest(), result.output)
        self.assertIn('output_artifact', result.metadata)

    async def test_failed_process_collection_is_distinct_from_cancellation(self):
        from borealis_coder.safety.sandbox import output_capture

        class FailedProcess:
            async def run(self, *args, **kwargs):
                capture = output_capture.get()
                assert capture
                capture('stderr', 'diagnostic before collection failed')
                raise OSError('collection failed')

        self.context.process = FailedProcess()  # type: ignore[assignment]
        with self.assertRaises(OSError):
            await self.context.run_process('test', cwd=self.root, timeout=1)
        row = self.store._connection.execute('SELECT metadata_json FROM output_artifacts').fetchone()
        self.assertIn('collection_failed', row[0])

    def test_quota_and_missing_are_explicit(self):
        artifact = self.store.save_output_artifact(self.session.id, self.root, 'abcdef', source='call', redactor=Redactor(), max_bytes=3)
        found = self.store.read_output_artifact(self.session.id, self.root, artifact['artifact_id'])
        self.assertEqual(found['content'], 'abc')
        self.assertEqual(found['status'], 'quota_limited')
        self.assertEqual(found['observed_bytes'], 6)
        with self.assertRaises(SessionError):
            self.store.read_output_artifact(self.session.id, self.root, '../../etc/passwd')

    async def test_search_pages_and_stale_scope(self):
        (self.root / 'a.txt').write_text('needle first\nneedle second\nneedle third\n')
        (self.root / 'binary.txt').write_bytes(b'\0needle')
        arguments = {'pattern': 'needle', 'path': '.', 'glob': '*.txt', 'regex': False, 'case_sensitive': True, 'context_lines': 0, 'max_results': 1}
        tool = GrepTool()
        first = await tool.execute(arguments, self.context)
        second = await tool.execute({**arguments, 'cursor': first.metadata['next_cursor']}, self.context)
        third = await tool.execute({**arguments, 'cursor': second.metadata['next_cursor']}, self.context)
        self.assertIn('a.txt:1:needle first', first.output)
        self.assertNotIn('a.txt:1:needle first', second.output)
        self.assertIn('a.txt:2:needle second', second.output)
        self.assertIn('a.txt:3:needle third', third.output)
        self.assertFalse(third.metadata['complete'])
        self.assertIn('unsupported_content', third.output)
        (self.root / 'a.txt').write_text('needle changed')
        stale = await tool.execute({**arguments, 'cursor': first.metadata['next_cursor']}, self.context)
        self.assertTrue(stale.is_error)
        self.assertIn('restart', stale.output)

    async def test_helper_parallel_reads_are_bounded_and_read_only(self):
        from types import SimpleNamespace

        from borealis_coder.agent.runner import ProviderRoute
        from borealis_coder.models import Effect
        from borealis_coder.tools.delegate import DelegateTaskTool

        self.context.config.agent.max_input_tokens = 30_000
        self.context.config.agent.max_output_tokens = 1000
        self.context.config.agent.compaction_safety_margin_tokens = 128
        provider = MockProvider(ProviderConfig(model='small', context_tokens=6000))
        budget = Budget.start(self.context.config.agent)
        writes = []
        registry = ToolRegistry([
            FunctionTool(name='read_evidence', description='read evidence', parameters=object_schema({'part': {'type': 'integer'}}),
                         function=lambda *_: ToolResult('diagnostic evidence ' * 1000), concurrent=True),
            FunctionTool(name='write_test', description='must not run', parameters=object_schema({}),
                         function=lambda *_: writes.append(True) or ToolResult('written'), effect=Effect.WRITE),
        ])
        def handler(request, call):
            if call == 1:
                return ModelResponse(tool_calls=[
                    *[ToolCall(name='read_evidence', arguments={'part': n}) for n in range(10)],
                    ToolCall(name='write_test', arguments={}),
                ])
            actual = ContextBudget.calculate(self.context.config.agent, system=request.system, messages=request.messages, tools=request.tools, provider='mock')
            self.assertLessEqual(actual.estimated_total(request.messages) + request.max_output_tokens + 128, 6000)
            self.assertIn('Scoped user requirements', request.system)
            return ModelResponse(text='Read evidence; no unresolved questions.')
        provider.handler = handler
        async def settle(usage, *, reservation=0):
            budget.release_cost(reservation)
            budget.add_usage(usage)
        self.context.metadata.update(provider_routes=[ProviderRoute('mock', 'small', provider)], tool_registry=registry,
                                     context_builder=SimpleNamespace(system_prompt=lambda **_: 'Read-only investigation'),
                                     budget=budget, usage_sink=settle, before_model_request=budget.before_model_request)
        result = await DelegateTaskTool().execute({'task': 'Inspect the evidence.', 'max_turns': 3}, self.context)
        self.assertFalse(result.is_error)
        self.assertEqual(writes, [])
        self.assertEqual(budget.model_requests, 2)
        self.assertEqual(result.metadata['usage']['requests'], 2)
        self.assertIn('Evidence references', result.output)


class RequestSafeguardsTests(unittest.TestCase):
    def setUp(self):
        self.config = AgentConfig(max_input_tokens=30_000, max_output_tokens=1000, compaction_safety_margin_tokens=128)
        self.provider = MockProvider(ProviderConfig(model='main', model_limits={'small': {'context_tokens': 6000}}))

    def test_smaller_route_compacts_and_preserves_scoped_requirement(self):
        messages = [Message(role=Role.USER, content='Investigate only; preserve the API.')]
        for index in range(20):
            call = ToolCall(id=str(index), name='read_file', arguments={'path': f'{index}.txt'})
            messages += [Message(role=Role.ASSISTANT, tool_calls=[call]), Message(role=Role.TOOL, tool_call_id=call.id, tool_name=call.name, content='evidence ' * 900)]
        request = prepare_route_request(ProviderRequest(model='small', system='Read-only helper.', messages=messages, max_output_tokens=1000), self.provider, self.config)
        budget = ContextBudget.calculate(self.config, system=request.system, tools=[], messages=request.messages, provider='mock')
        self.assertLessEqual(budget.estimated_total(request.messages) + 1000 + 128, 6000)
        self.assertIn('preserve the API', request.system)
        self.assertLess(len(request.messages), len(messages))

    def test_fixed_context_and_schema_cannot_be_silently_dropped(self):
        with self.assertRaises(BudgetExceeded):
            prepare_route_request(ProviderRequest(model='small', system='protected ' * 3000, messages=[]), self.provider, self.config)
        with self.assertRaises(BudgetExceeded):
            prepare_route_request(ProviderRequest(model='small', system='', messages=[], response_schema={'description': 'x' * 40_000}), self.provider, self.config)

    def test_bytes_are_independent_and_unknown_limits_are_explicit(self):
        request = ProviderRequest(model='main', system='ok', messages=[Message(role=Role.USER, content='🙂' * 1000)], max_output_tokens=100)
        prepared = prepare_route_request(request, self.provider, self.config)
        self.assertIsNone(prepared.metadata['resolved_limits']['request_byte_limit'])
        self.provider.config.request_byte_limit = 500
        with self.assertRaises(BudgetExceeded):
            prepare_route_request(request, self.provider, self.config)

    def test_price_status_zero_unknown_and_native_charge(self):
        provider = OpenAIProvider(ProviderConfig())
        unknown = provider.price_usage(Usage(input_tokens=100, requests=1))
        self.assertEqual(unknown.cost_status, 'unknown')
        provider.config.input_cost_per_million = 0
        provider.config.output_cost_per_million = 0
        free = provider.price_usage(Usage(input_tokens=100, requests=1))
        self.assertEqual(free.cost_status, 'estimated')
        total = Usage().add(free).add(unknown)
        self.assertEqual(total.cost_status, 'incomplete')
        reported = provider._usage_from_chat({'prompt_tokens': 10, 'cost': 0.03})
        self.assertEqual(reported.cost_status, 'known')
        self.assertEqual(reported.cost_usd, 0.03)
        self.assertEqual(reported.native_usage['cost'], 0.03)
        self.assertEqual(provider._usage_from_chat({'prompt_tokens': 10}).cost_status, 'incomplete')

    def test_model_specific_price_and_cached_categories(self):
        provider = OpenAIProvider(ProviderConfig(model='main', input_cost_per_million=2, output_cost_per_million=4,
            cached_input_cost_per_million=0, model_prices={'small': {'input_cost_per_million': 1, 'output_cost_per_million': 2}}))
        main = provider.price_usage(Usage(input_tokens=1000, cached_input_tokens=500, output_tokens=100))
        self.assertAlmostEqual(main.cost_usd, 0.0014)
        provider.request_model.set('small')
        small = provider.price_usage(Usage(input_tokens=1000, output_tokens=100))
        self.assertAlmostEqual(small.cost_usd, 0.0012)
        provider.request_model.set('unpriced')
        self.assertEqual(provider.price_usage(Usage(input_tokens=10)).cost_status, 'unknown')

    def test_strict_budget_rejects_unknown_and_reserves_parallel_cost(self):
        budget = Budget.start(AgentConfig(max_cost_usd=0.025))
        provider = OpenAIProvider(ProviderConfig(model='main'))
        request = ProviderRequest(model='main', system='', messages=[], max_output_tokens=10_000)
        with self.assertRaisesRegex(BudgetExceeded, 'requires.*pricing'):
            budget.reserve_cost(provider, request)
        provider.config.input_cost_per_million = 1
        provider.config.output_cost_per_million = 2
        reservation = budget.reserve_cost(provider, request)
        with self.assertRaisesRegex(BudgetExceeded, 'concurrent'):
            budget.reserve_cost(provider, request)
        budget.release_cost(reservation)
        budget.add_usage(Usage(requests=1, cost_status='incomplete'))
        with self.assertRaisesRegex(BudgetExceeded, 'unreconciled'):
            budget.reserve_cost(provider, request)


class TaskStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_followups_compaction_revocation_and_restart(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root, context={'compact_tool_output_tokens': 20})
            runner = await build_runner(root, config=config, interactive=False)
            session = runner.sessions.create_session(workspace=root, provider='mock', model='deterministic')
            for index, text in enumerate(['Build the importer. Preserve CSV support.', 'Also add tests.', 'Handle empty rows.', 'Use UTF-8.', 'Revoke the CSV requirement. Only support TSV.']):
                runner.sessions.append_message(session.id, Message(role=Role.USER, content=text))
                call = ToolCall(name='read_file', arguments={'path': f'{index}.txt'})
                runner.sessions.append_message(session.id, Message(role=Role.ASSISTANT, tool_calls=[call]))
                runner.sessions.append_message(session.id, Message(role=Role.TOOL, tool_call_id=call.id, tool_name=call.name, content='untrusted diagnostic ' * 500))
            original = runner.sessions.messages(session.id)
            async def usage_sink(_usage: Usage) -> None:
                pass

            async def prepare() -> PreparedProviderRequest:
                return await runner._prepare_provider_request(prompt_context=PromptContext(stable='system'), messages=runner.sessions.messages(session.id), schemas=[], final_turn=False, verification_finalization_pending=False, adaptive_cache=False, conversation_cache=True, usage_sink=usage_sink, cancel=asyncio.Event(), session_id=session.id, run_id='test', last_prune_signature=None)
            first = await prepare()
            self.assertTrue(first.compacted)
            for text in ['Build the importer', 'Also add tests', 'Only support TSV']:
                self.assertIn(text, first.request.system)
            self.assertEqual(runner.sessions.messages(session.id), original)
            await runner.close()
            runner = await build_runner(root, config=config, interactive=False)
            try:
                second = await prepare()
                self.assertIn('Revoke the CSV requirement', second.request.system)
                self.assertEqual(runner.sessions.task_state(session.id)['objective_source'], original[0].id)
                revocation = Message(role=Role.USER, content='Revoke user:2. Do not add tests.')
                runner.sessions.append_message(session.id, revocation)
                state = runner.sessions.task_state(session.id)
                self.assertEqual(state['superseded'][original[3].id], revocation.id)
                refreshed = await prepare()
                self.assertIn('"status":"superseded"', refreshed.request.system)
                source = runner.sessions.search_history(session.id, root, 'user:1')
                self.assertEqual(source[0]['message_id'], original[0].id)
                config.agent.max_input_tokens = 100
                with self.assertRaisesRegex(BudgetExceeded, 'Protected task'):
                    await prepare()
            finally:
                await runner.close()


class RequestPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_fallback_reprepares_and_accounts_each_route_once(self):
        from borealis_coder.agent.runner import ProviderRoute
        from borealis_coder.errors import ProviderUnavailableError
        from borealis_coder.providers.base import Provider

        class PricedProvider(Provider):
            name = 'priced'

            def __init__(self, config, fail=False):
                super().__init__(config)
                self.fail = fail
                self.seen = []

            async def complete(self, request):
                self.seen.append(request)
                usage = self.price_usage(Usage(input_tokens=100, output_tokens=10, requests=1))
                if self.fail:
                    raise ProviderUnavailableError('try backup', usage=usage)
                return ModelResponse(text='done', usage=usage)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root, agent={'max_input_tokens': 30_000, 'max_output_tokens': 1000, 'compaction_safety_margin_tokens': 128, 'max_cost_usd': 1}, cache={'response_cache_enabled': False})
            runner = await build_runner(root, config=config, interactive=False)
            self.addAsyncCleanup(runner.close)
            primary = PricedProvider(ProviderConfig(model='main', input_cost_per_million=2, output_cost_per_million=4, max_retries=0), fail=True)
            backup = PricedProvider(ProviderConfig(model='small', context_tokens=6000, input_cost_per_million=1, output_cost_per_million=2, max_retries=0))
            runner.providers = [ProviderRoute('main', 'main', primary), ProviderRoute('backup', 'small', backup)]
            session = runner.sessions.create_session(workspace=root, provider='main', model='main')
            messages = [Message(role=Role.USER, content='Preserve the API.')]
            for _ in range(20):
                messages.extend([Message(role=Role.ASSISTANT, content='history ' * 800), Message(role=Role.USER, content='Continue.')])
            request = ProviderRequest(model='main', system='system', messages=messages, max_output_tokens=1000)
            budget = Budget.start(config.agent)
            budget.before_model_request()
            async def settle(usage):
                budget.add_usage(usage)
            response, route = await runner._complete_request(request, session.id, 'run', asyncio.Event(), 'assistant', usage_sink=settle, budget=budget)
            budget.add_usage(response.usage)
            self.assertEqual(route.name, 'backup')
            self.assertEqual(budget.model_requests, 2)
            self.assertEqual(response.usage.requests, 2)
            self.assertAlmostEqual(response.usage.cost_usd, 0.00036)
            self.assertEqual(response.usage.cost_status, 'estimated')
            self.assertEqual(budget.pending_cost_usd, 0)
            self.assertLess(len(backup.seen[0].messages), len(messages))
            self.assertIn('Preserve the API.', backup.seen[0].system)

    async def test_summarizer_uses_its_own_price_and_limit(self):
        from borealis_coder.agent.runner import ProviderRoute
        from borealis_coder.providers.base import Provider

        class PricedProvider(Provider):
            name = 'priced'
            async def complete(self, request):
                return ModelResponse(text='summary', usage=self.price_usage(Usage(input_tokens=100, output_tokens=10, requests=1)))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root, agent={'deterministic_compaction': False, 'small_model': 'small'})
            runner = await build_runner(root, config=config, interactive=False)
            self.addAsyncCleanup(runner.close)
            provider = PricedProvider(ProviderConfig(model='main', input_cost_per_million=10, output_cost_per_million=20,
                model_prices={'small': {'input_cost_per_million': 1, 'output_cost_per_million': 2}}))
            runner.providers = [ProviderRoute('priced', 'main', provider)]
            recorded = Usage()
            async def settle(usage):
                recorded.add(usage)
            summarize = runner._summarizer(settle, asyncio.Event())
            assert summarize
            await summarize('history')
            self.assertAlmostEqual(recorded.cost_usd, 0.00012)
            self.assertEqual(recorded.requests, 1)
            provider.config.model_limits = {'small': {'context_tokens': 100}}
            with self.assertRaises(BudgetExceeded):
                await summarize('history')
            self.assertEqual(recorded.requests, 1)
