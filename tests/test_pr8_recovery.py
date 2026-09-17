from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from borealis_coder.agent import build_runner
from borealis_coder.agent.budget import Budget
from borealis_coder.agent.runner import ProviderRoute
from borealis_coder.config import ProviderConfig
from borealis_coder.context.builder import PromptContext
from borealis_coder.errors import (
    BudgetExceeded,
    ProviderError,
    ProviderUnavailableError,
    SessionError,
)
from borealis_coder.models import Message, ProviderRequest, Role, ToolCall, ToolResult
from borealis_coder.providers.openai import OpenAIProvider
from borealis_coder.sessions.store import SessionStore
from borealis_coder.tools.base import FunctionTool, ToolRegistry, object_schema
from borealis_coder.tools.delegate import DelegateTaskTool
from borealis_coder.tools.history import SearchHistoryTool
from tests.helpers import make_config, make_context


class RecoveryHTTP:
    def __init__(self, *, block=False, empty_second=False, failure=False):
        self.calls = []
        self.model = "test"
        self.ready = asyncio.Event()
        self.release = asyncio.Event()
        self.block = block
        self.empty_second = empty_second
        self.failure = failure
        self.http_error = False
        self.first_usage: dict[str, Any] | None = None

    async def response(self, payload):
        self.calls.append(payload)
        attempt = len(self.calls)
        if attempt == 2 and self.block:
            self.ready.set()
            await self.release.wait()
        if attempt == 2 and self.failure:
            raise OSError('connection lost')
        text = '' if attempt == 1 or self.empty_second else 'done'
        return {'model': self.model, 'choices': [{'message': {'content': text}, 'delta': {'content': text},
            'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 2,
            'cost': .6 if attempt == 1 else .2}}

    async def post_json(self, url, *, headers, payload):
        return SimpleNamespace(data=await self.response(payload))

    async def stream_sse(self, url, *, headers, payload):
        yield SimpleNamespace(data=json.dumps(await self.response(payload)))

    def close(self):
        pass


class ResponsesRecoveryHTTP(RecoveryHTTP):
    async def response(self, payload):
        data = await super().response(payload)
        usage = {'input_tokens': 10, 'output_tokens': 2, 'cost': data['usage']['cost']}
        if len(self.calls) == 1:
            if self.first_usage is not None:
                usage = self.first_usage
            if self.http_error:
                raise ProviderError('reasoning summary is unsupported', status_code=400,
                                    details={'usage': usage})
            return {'type': 'response.failed', 'response': {'status': 'failed',
                'error': {'message': 'reasoning summary is unsupported'}, 'usage': usage}}
        return {'type': 'response.completed', 'response': {'status': 'completed',
            'model': self.model, 'usage': usage, 'output': [{'type': 'message',
            'content': [{'type': 'output_text', 'text': 'done'}]}]}}

    async def post_json(self, url, *, headers, payload):
        return SimpleNamespace(data=(await self.response(payload))['response'])


class InterruptedUsageHTTP(RecoveryHTTP):
    async def stream_sse(self, url, *, headers, payload):
        self.calls.append(payload)
        for cost in (.1, .6):
            yield SimpleNamespace(data=json.dumps({'model': self.model, 'choices': [],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'cost': cost}}))
        self.ready.set()
        await self.release.wait()
        raise OSError('stream interrupted after usage')


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config = make_config(self.root, agent={'max_cost_usd': 1, 'max_output_tokens': 100,
                                                   'reasoning_effort': 'high'},
                                  cache={'response_cache_enabled': False})
        self.runner = await build_runner(self.root, config=self.config, interactive=False)
        self.addAsyncCleanup(self.runner.close)

    def setup_request(self, *, limit: float = 1, block=False, empty_second=False, failure=False):
        self.config.agent.max_cost_usd = limit
        self.budget = Budget.start(self.config.agent)
        self.session = self.runner.sessions.create_session(workspace=self.root, provider='openai', model='test')
        self.provider = OpenAIProvider(ProviderConfig(model='test', api_style='chat_completions',
            input_cost_per_million=0, output_cost_per_million=6000, max_retries=0))
        self.http = RecoveryHTTP(block=block, empty_second=empty_second, failure=failure)
        transport = patch.object(self.provider, "http", self.http)
        transport.start()
        self.addCleanup(transport.stop)
        self.runner.providers = [ProviderRoute('openai', 'test', self.provider)]
        self.request = ProviderRequest(model='test', system='Inspect', messages=[],
                                       max_output_tokens=100, reasoning_effort='high')
        self.settlements = 0

    async def settle(self, usage):
        self.settlements += 1
        self.budget.add_usage(usage)
        self.runner.sessions.add_usage(self.session.id, usage)

    async def run_request(self, streaming):
        if streaming:
            self.budget.before_model_request()
            response, _ = await self.runner._complete_request(self.request, self.session.id, 'run',
                asyncio.Event(), 'assistant', usage_sink=self.settle, budget=self.budget)
            await self.settle(response.usage)
        else:
            context = make_context(self.root, self.config)
            context.session_id = self.session.id
            context.metadata.update(provider_routes=self.runner.providers, tool_registry=ToolRegistry([]),
                context_builder=SimpleNamespace(system_prompt=lambda **_: 'Inspect'), usage_sink=self.settle,
                budget=self.budget, before_model_request=self.budget.before_model_request)
            await DelegateTaskTool().execute({'task': 'Inspect', 'max_turns': 1}, context)

    def assert_settled(self, cost, requests, status):
        self.assertEqual(self.settlements, 1)
        usage = self.runner.sessions.usage(self.session.id)
        self.assertAlmostEqual(usage.cost_usd, cost)
        self.assertEqual(usage.requests, requests)
        self.assertEqual(usage.cost_status, status)
        assert self.budget.usage is not None
        self.assertAlmostEqual(self.budget.usage.cost_usd, cost)
        self.assertEqual(self.budget.usage.requests, requests)
        self.assertEqual(self.budget.pending_cost_usd, 0)
        self.assertFalse(self.budget._held_usage)

    async def test_recovery_rejected_before_dispatch_when_only_40_cents_remain(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                self.setup_request()
                with self.assertRaises(BudgetExceeded):
                    await self.run_request(streaming)
                self.assertEqual(len(self.http.calls), 1)
                self.assert_settled(.6, 1, 'known')
                self.assertEqual(self.budget.model_requests, 1)

    async def test_recovery_cancellation_preserves_known_charge(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                self.setup_request(limit=2, block=True)
                task = asyncio.create_task(self.run_request(streaming))
                await asyncio.wait_for(self.http.ready.wait(), timeout=3)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assert_settled(.6, 2, 'incomplete')
                self.assertEqual(self.budget.model_requests, 2)

    async def test_concurrent_reservation_sees_internal_charge_and_success_settles_once(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                self.setup_request(limit=1.5, block=True)
                task = asyncio.create_task(self.run_request(streaming))
                await asyncio.wait_for(self.http.ready.wait(), timeout=3)
                self.assertAlmostEqual(self.budget.pending_cost_usd, 1.2)
                with self.assertRaises(BudgetExceeded):
                    self.budget.reserve_cost(self.provider, self.request)
                self.http.release.set()
                await task
                self.assert_settled(.8, 2, 'known')
                self.assertNotIn('reasoning_effort', self.http.calls[1])
                native = [event for _, event in self.runner.sessions._export_events(self.session.id)
                          if event.type == 'model.attempt_usage']
                self.assertEqual(len(native), 2)

    async def test_recovery_preserves_explicit_response_model_billing_identity(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                self.setup_request(limit=2)
                self.http.model = 'snapshot'
                self.provider.config.model_prices['snapshot'] = {
                    'input_cost_per_million': 0, 'output_cost_per_million': 6000,
                }
                await self.run_request(streaming)
                self.assert_settled(.8, 2, 'known')
                native = [event.data for _, event in self.runner.sessions._export_events(self.session.id)
                          if event.type == 'model.attempt_usage']
                self.assertEqual([item['model'] for item in native], ['snapshot', 'snapshot'])

    async def test_empty_recovery_does_not_double_count_first_attempt(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                self.setup_request(limit=2, empty_second=True)
                with self.assertRaises(ProviderUnavailableError) as raised:
                    await self.run_request(streaming)
                if streaming:
                    self.assertIsNotNone(raised.exception.usage)
                    await self.settle(raised.exception.usage)
                self.assert_settled(.8, 2, 'known')

    async def test_recovery_transport_failure_preserves_known_and_incomplete_attempts(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                self.setup_request(limit=2, failure=True)
                with self.assertRaises(ProviderUnavailableError) as raised:
                    await self.run_request(streaming)
                if streaming:
                    await self.settle(raised.exception.usage)
                self.assert_settled(.6, 2, 'incomplete')


    def responses_transport(self, **kwargs):
        self.provider.config.api_style = 'responses'
        self.http = ResponsesRecoveryHTTP(**kwargs)
        transport = patch.object(self.provider, 'http', self.http)
        transport.start()
        self.addCleanup(transport.stop)

    async def test_responses_recovery_observes_each_attempt_and_checks_shared_budget(self):
        for streaming in (False, True):
            for limit in (1, 2):
                with self.subTest(streaming=streaming, limit=limit):
                    self.setup_request(limit=limit)
                    self.responses_transport()
                    if limit == 1:
                        with self.assertRaises(BudgetExceeded):
                            await self.run_request(streaming)
                        self.assertEqual(len(self.http.calls), 1)
                        self.assert_settled(.6, 1, 'known')
                    else:
                        await self.run_request(streaming)
                        self.assertEqual(len(self.http.calls), 2)
                        self.assert_settled(.8, 2, 'known')
                        self.assertNotIn('summary', self.http.calls[1]['reasoning'])
                        native = [event.data for _, event in self.runner.sessions._export_events(self.session.id)
                                  if event.type == 'model.attempt_usage']
                        self.assertEqual([item['native_usage']['cost'] for item in native], [.6, .2])
                    self.assertEqual(self.budget.model_requests, len(self.http.calls))

    async def test_responses_recovery_retains_estimated_and_incomplete_cost_status(self):
        for streaming in (False, True):
            for first_usage in ({'input_tokens': 10, 'output_tokens': 2}, {}):
                with self.subTest(streaming=streaming, first_usage=first_usage):
                    self.setup_request(limit=2)
                    self.responses_transport()
                    self.http.first_usage = first_usage
                    if first_usage:
                        await self.run_request(streaming)
                        self.assert_settled(.212, 2, 'estimated')
                    else:
                        with self.assertRaises(BudgetExceeded):
                            await self.run_request(streaming)
                        self.assert_settled(0, 1, 'incomplete')
                        self.assertEqual(len(self.http.calls), 1)

    async def test_responses_http_error_usage_survives_recovery(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                self.setup_request(limit=2)
                self.responses_transport()
                self.http.http_error = True
                await self.run_request(streaming)
                self.assert_settled(.8, 2, 'known')

    async def test_responses_recovery_respects_model_request_limit(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                self.setup_request(limit=2)
                self.config.agent.max_model_requests = 1
                self.responses_transport()
                with self.assertRaises(BudgetExceeded):
                    await self.run_request(streaming)
                self.assertEqual(len(self.http.calls), 1)
                self.assert_settled(.6, 1, 'known')

    async def test_responses_interrupted_recovery_preserves_first_charge(self):
        for streaming in (False, True):
            for cancelled in (False, True):
                with self.subTest(streaming=streaming, cancelled=cancelled):
                    self.setup_request(limit=2)
                    self.responses_transport(block=True, failure=True)
                    task = asyncio.create_task(self.run_request(streaming))
                    await asyncio.wait_for(self.http.ready.wait(), timeout=3)
                    if cancelled:
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    else:
                        self.http.release.set()
                        with self.assertRaises(ProviderUnavailableError) as raised:
                            await task
                        if streaming:
                            await self.settle(raised.exception.usage)
                    self.assert_settled(.6, 2, 'incomplete')
                    self.assertEqual(self.budget.model_requests, 2)

    async def test_chat_interrupted_after_usage_preserves_latest_snapshot_once(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                self.setup_request(limit=2)
                self.http = InterruptedUsageHTTP()
                self.http.model = 'snapshot'
                self.provider.config.model_prices['snapshot'] = {
                    'input_cost_per_million': 0, 'output_cost_per_million': 6000,
                }
                with patch.object(self.provider, 'http', self.http):
                    task = asyncio.create_task(self.run_request(True))
                    await asyncio.wait_for(self.http.ready.wait(), timeout=3)
                    if cancelled:
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    else:
                        self.http.release.set()
                        with self.assertRaises(ProviderUnavailableError) as raised:
                            await task
                        await self.settle(raised.exception.usage)
                self.assert_settled(.6, 1, 'incomplete')
                usage = self.runner.sessions.usage(self.session.id)
                self.assertEqual((usage.input_tokens, usage.output_tokens), (10, 2))
                native = [event.data for _, event in self.runner.sessions._export_events(self.session.id)
                          if event.type == 'model.attempt_usage']
                self.assertEqual(len(native), 1)
                self.assertEqual(native[0]['native_usage']['cost'], .6)
                self.assertEqual(native[0]['model'], 'snapshot')


class EvidenceRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config = make_config(self.root, agent={'max_cost_usd': 0, 'deterministic_compaction': True},
                                  context={'tool_output_chars': 8000})
        self.runner = await build_runner(self.root, config=self.config, interactive=False)
        self.addAsyncCleanup(self.runner.close)
        self.store = self.runner.sessions
        self.session = self.store.create_session(workspace=self.root, provider='mock', model='test')

    async def test_quoted_revocations_do_not_change_sources_or_protected_state_on_resume(self):
        examples = [
            'Document this example:\n```text\nRevoke user:1\n```',
            '```text\nRevoke user:1\nSupersede user:2\n```',
            '~~~\nRevoke user:1\n~~~', '> Revoke user:1',
            '"Revoke user:1"', '    Revoke user:1',
            'Revoke user:1 is an example, not a command.',
            'Example:\nRevoke user:1\nSupersede user:2',
        ]
        first = Message(role=Role.USER, content='Keep the original API contract.')
        self.store.append_message(self.session.id, first)
        for example in examples:
            self.store.append_message(self.session.id, Message(role=Role.USER, content=example))
        store = SessionStore(self.store.path)
        try:
            state = store.task_state(self.session.id)
            self.assertTrue(all(item['status'] == 'active' for item in state['instructions']))
            self.assertEqual(state['instruction_sources'][0], first.id)
            self.assertEqual(len(state['instruction_sources']), len(examples)+1)
        finally:
            store.close()
        prepared = await self.runner._prepare_provider_request(prompt_context=PromptContext(stable='system'),
            messages=self.store.messages(self.session.id), schemas=[], final_turn=False,
            verification_finalization_pending=False, adaptive_cache=False, conversation_cache=False,
            usage_sink=AsyncMock(), cancel=asyncio.Event(), session_id=self.session.id,
            run_id='run', last_prune_signature=None, overflow_retry_count=1)
        self.assertIn(first.content, prepared.request.system)
        self.assertIn('user:1', prepared.request.system)
        self.assertTrue(prepared.compacted)

    async def test_standalone_revocations_still_work(self):
        for command in ('Revoke user:1', 'Supersede user:1', 'Revoke user:1. Use the new API.'):
            with self.subTest(command=command):
                messages = [Message(role=Role.USER, content='Original'), Message(role=Role.USER, content=command)]
                state = self.store.task_state(self.session.id, messages)
                self.assertEqual(state['instructions'][0]['status'], 'superseded')
                self.assertEqual(state['instructions'][0]['superseded_by'], messages[1].id)

    async def test_tool_reference_pages_reconstruct_event_only_evidence_after_resume(self):
        output = 'prefix ' * 180 + 'middle evidence\n' + 'tail ' * 360
        tool = FunctionTool(name='inspect', description='Read-only evidence', parameters=object_schema({}),
                            function=lambda *_: ToolResult(output))
        context = make_context(self.root, self.config)
        context.session_id = self.session.id
        context.metadata['session_store'] = self.store
        context.events = self.runner.events
        call = ToolCall(name='inspect', arguments={})
        result = await ToolRegistry([tool]).execute(call, context)
        self.assertNotIn('output_artifact', result.metadata)
        await self.runner.events.flush()
        store = SessionStore(self.store.path)
        try:
            context.metadata['session_store'] = store
            offset, parts = 0, []
            while True:
                result = await SearchHistoryTool().execute({'query': call.id, 'after': 0, 'offset': offset}, context)
                page = json.loads(result.output.split('\n', 1)[1])['tools'][0]
                self.assertEqual(page['source'], call.id)
                self.assertEqual(page['offset'], offset)
                self.assertEqual(page['total_chars'], len(output))
                self.assertLessEqual(len(page['preview']), 1000)
                parts.append(page['preview'])
                if page['next_offset'] is None:
                    break
                offset = page['next_offset']
            self.assertEqual(''.join(parts), output)
            found = store.search_tool_history(self.session.id, self.root, 'middle evidence', offset=99999)[0]
            self.assertIn('middle evidence', found['preview'])
            self.assertFalse(store.search_tool_history(self.session.id, self.root, call.id, after=page['sequence']))
            other = store.create_session(workspace=self.root, provider='mock', model='test')
            self.assertEqual(store.search_tool_history(other.id, self.root, call.id), [])
            with self.assertRaises(SessionError):
                store.search_tool_history(self.session.id, self.root / 'other', call.id)
        finally:
            store.close()
