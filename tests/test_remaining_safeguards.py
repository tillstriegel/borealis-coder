from __future__ import annotations

import asyncio
import io
import tempfile
import unittest
from email.message import Message as Headers
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from borealis_coder.agent import build_runner
from borealis_coder.agent.budget import Budget, prepare_route_request
from borealis_coder.agent.runner import ProviderRoute
from borealis_coder.config import AgentConfig, ProviderConfig
from borealis_coder.errors import BudgetExceeded, ProviderUnavailableError
from borealis_coder.models import ModelResponse, ProviderRequest, Usage
from borealis_coder.providers.base import Provider
from borealis_coder.providers.openrouter import OpenRouterProvider
from borealis_coder.tools.base import ToolRegistry
from borealis_coder.tools.delegate import DelegateTaskTool
from borealis_coder.tools.fetch import FetchUrlTool
from borealis_coder.tools.search import GrepTool, _walk_files
from tests.helpers import make_config, make_context


def charge(cost):
    return Usage(requests=1, cost_usd=cost, cost_status='known')


class WaitingRetryProvider(Provider):
    name = 'test'

    def __init__(self, ready, release, *, retry=True, incomplete=False):
        super().__init__(ProviderConfig(model='test', input_cost_per_million=0, output_cost_per_million=4000,
                                       max_retries=1 if retry else 0, initial_backoff_seconds=0))
        self.ready, self.release = ready, release
        self.calls = 0
        self.retry = retry
        self.incomplete = incomplete

    async def complete(self, request):
        self.calls += 1
        if self.retry and self.calls == 1:
            usage = Usage(requests=1, cost_status='incomplete') if self.incomplete else charge(.4)
            raise ProviderUnavailableError('billed attempt', retryable=True, usage=usage)
        self.ready.set()
        await self.release.wait()
        return ModelResponse(text='done', usage=charge(.4))


class RemainingSafeguardsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config = make_config(self.root, agent={'max_cost_usd': 1, 'max_output_tokens': 100,
                                                   'deterministic_compaction': False, 'small_model': 'test'},
                                  cache={'response_cache_enabled': False})
        self.runner = await build_runner(self.root, config=self.config, interactive=False)
        self.addAsyncCleanup(self.runner.close)
        self.session = self.runner.sessions.create_session(workspace=self.root, provider='test', model='test')
        self.request = ProviderRequest(model='test', system='', messages=[], max_output_tokens=100)
        self.budget = Budget.start(self.config.agent)
        self.settlements = 0

    async def settle(self, usage):
        self.settlements += 1
        self.budget.add_usage(usage)
        self.runner.sessions.add_usage(self.session.id, usage)

    def route(self, provider):
        self.runner.providers = [ProviderRoute('test', 'test', provider)]

    async def main_request(self):
        response, _ = await self.runner._complete_request(self.request, self.session.id, 'run', asyncio.Event(),
                                                         'assistant', usage_sink=self.settle, budget=self.budget)
        await self.settle(response.usage)

    async def delegated_request(self):
        context = make_context(self.root, self.config)
        context.metadata.update(provider_routes=self.runner.providers, tool_registry=ToolRegistry([]),
                                context_builder=SimpleNamespace(system_prompt=lambda **_: 'Inspect'),
                                usage_sink=self.settle, before_model_request=self.budget.before_model_request, budget=self.budget)
        await DelegateTaskTool().execute({'task': 'Inspect', 'max_turns': 1}, context)

    async def summary_request(self):
        summarize = self.runner._summarizer(self.settle, asyncio.Event(), before_model_request=self.budget.before_model_request)
        assert summarize
        await summarize('history')

    async def test_retry_cost_is_visible_to_concurrent_reservations_on_every_path(self):
        for path in ('main_request', 'delegated_request', 'summary_request'):
            for cancel in (False, True):
                with self.subTest(path=path, cancel=cancel):
                    self.budget = Budget.start(self.config.agent)
                    self.settlements = 0
                    session = self.runner.sessions.create_session(workspace=self.root, provider='test', model='test')
                    self.session = session
                    ready, release = asyncio.Event(), asyncio.Event()
                    provider = WaitingRetryProvider(ready, release)
                    self.route(provider)
                    task = asyncio.create_task(getattr(self, path)())
                    try:
                        await asyncio.wait_for(ready.wait(), 2)
                        self.assertEqual(provider.calls, 2)
                        self.assertAlmostEqual(self.budget.pending_cost_usd, .8)
                        with self.assertRaises(BudgetExceeded):
                            self.budget.reserve_cost(provider, self.request)
                        if cancel:
                            task.cancel()
                            with self.assertRaises(asyncio.CancelledError):
                                await task
                        else:
                            release.set()
                            await task
                    finally:
                        release.set()
                        if not task.done():
                            task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                    durable = self.runner.sessions.usage(session.id)
                    self.assertEqual(self.settlements, 1)
                    self.assertEqual(durable.requests, 2)
                    self.assertEqual(durable.cost_status, 'incomplete' if cancel else 'known')
                    self.assertAlmostEqual(durable.cost_usd, .4 if cancel else .8)
                    self.assertEqual(self.budget.usage, durable)
                    self.assertEqual(self.budget.pending_cost_usd, 0)
                    self.assertEqual(self.budget._held_usage, {})

    async def test_retry_and_fallback_share_one_pending_usage_hold(self):
        ready, release = asyncio.Event(), asyncio.Event()
        first = WaitingRetryProvider(ready, release)
        first.config.max_retries = 0
        fallback = WaitingRetryProvider(ready, release, retry=False)
        self.runner.providers = [ProviderRoute('first', 'test', first), ProviderRoute('second', 'test', fallback)]
        task = asyncio.create_task(self.main_request())
        try:
            await asyncio.wait_for(ready.wait(), 2)
            self.assertAlmostEqual(self.budget.pending_cost_usd, .8)
            with self.assertRaises(BudgetExceeded):
                self.budget.reserve_cost(fallback, self.request)
        finally:
            release.set()
            await task
        self.assertEqual(self.settlements, 1)
        self.assertEqual(self.runner.sessions.usage(self.session.id).requests, 2)
        self.assertAlmostEqual(self.runner.sessions.usage(self.session.id).cost_usd, .8)
        self.assertEqual(self.budget.pending_cost_usd, 0)

    async def test_successful_terminal_usage_survives_cancelled_event_handoff(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                self.budget = Budget.start(self.config.agent)
                self.settlements = 0
                self.session = self.runner.sessions.create_session(workspace=self.root, provider='test', model='test')
                ready, release = asyncio.Event(), asyncio.Event()
                release.set()
                self.route(WaitingRetryProvider(asyncio.Event(), release, retry=False))
                original_emit = self.runner.events.emit
                async def block(event_type, ready=ready, original_emit=original_emit, **kwargs):
                    if event_type == 'model.route_completed':
                        ready.set()
                        await asyncio.Event().wait()
                    return await original_emit(event_type, **kwargs)
                if cancel:
                    with patch.object(self.runner.events, 'emit', side_effect=block):
                        task = asyncio.create_task(self.main_request())
                        await asyncio.wait_for(ready.wait(), 2)
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                else:
                    await self.main_request()
                durable = self.runner.sessions.usage(self.session.id)
                self.assertEqual(durable, self.budget.usage)
                self.assertEqual(durable.requests, 1)
                self.assertEqual(durable.cost_status, 'known')
                self.assertAlmostEqual(durable.cost_usd, .4)
                self.assertEqual(self.settlements, 1)
                self.assertEqual(self.budget.pending_cost_usd, 0)

    async def test_terminal_usage_is_retained_if_event_delivery_fails(self):
        release = asyncio.Event()
        release.set()
        self.route(WaitingRetryProvider(asyncio.Event(), release, retry=False))
        original_emit = self.runner.events.emit
        async def fail(event_type, **kwargs):
            if event_type == 'model.route_completed':
                raise ProviderUnavailableError('event delivery failed')
            return await original_emit(event_type, **kwargs)
        with patch.object(self.runner.events, 'emit', side_effect=fail), self.assertRaises(ProviderUnavailableError) as caught:
            await self.main_request()
        assert caught.exception.usage
        await self.settle(caught.exception.usage)
        self.assertEqual(self.settlements, 1)
        self.assertEqual(self.runner.sessions.usage(self.session.id).requests, 1)
        self.assertAlmostEqual(self.runner.sessions.usage(self.session.id).cost_usd, .4)
        self.assertEqual(self.budget.pending_cost_usd, 0)

    async def test_unknown_attempt_blocks_other_reservations_and_retry(self):
        provider = WaitingRetryProvider(asyncio.Event(), asyncio.Event(), incomplete=True)
        self.route(provider)
        with self.assertRaisesRegex(BudgetExceeded, 'unreconciled'):
            await self.main_request()
        self.assertEqual(provider.calls, 1)
        self.assertEqual(self.runner.sessions.usage(self.session.id).cost_status, 'incomplete')
        self.assertEqual(self.budget.pending_cost_usd, 0)
        with self.assertRaisesRegex(BudgetExceeded, 'unreconciled'):
            self.budget.reserve_cost(provider, self.request)
        self.budget.config.max_cost_usd = 0
        self.assertEqual(self.budget.reserve_cost(provider, self.request), 0)

    async def test_openrouter_wire_overrides_are_rejected_before_dispatch(self):
        keys = ('instructions', 'system', 'input', 'messages', 'model', 'models', 'tools', 'functions',
                'tool_choice', 'function_call', 'parallel_tool_calls', 'response_format', 'text',
                'previous_response_id', 'conversation', 'max_tokens', 'max_output_tokens', 'max_completion_tokens')
        provider = OpenRouterProvider(ProviderConfig(model='test', api_style='responses'))
        self.addAsyncCleanup(provider.close)
        provider.http.post_json = AsyncMock()  # type: ignore[method-assign]
        self.route(provider)
        for key in keys:
            with self.subTest(key=key):
                provider.config.extra_body = {key: 'replacement'}
                with self.assertRaisesRegex(BudgetExceeded, "extra_body"):
                    await self.main_request()
        provider.http.post_json.assert_not_awaited()
        provider.config.extra_body = {'user': 'request-owner'}
        request = prepare_route_request(self.request, provider, AgentConfig())
        payload = provider._responses_payload(request)
        self.assertEqual(payload['user'], 'request-owner')
        self.assertEqual(payload['instructions'], request.system)

    async def test_fetch_artifact_distinguishes_collection_limit_from_preview(self):
        context = make_context(self.root)
        context.session_id = self.session.id
        context.metadata['session_store'] = self.runner.sessions
        headers = Headers()
        headers['Content-Type'] = 'text/plain; charset=utf-8'
        for source, partial in ((b'a' * 400 + b'NEVER-COLLECTED', True), (b'a' * 400, False), (b'a' * 150, False), (('🙂' * 100).encode() + b'NEVER-COLLECTED', True)):
            with self.subTest(size=len(source)):
                response = SimpleNamespace(read=io.BytesIO(source).read, headers=headers, status=200)
                connection = SimpleNamespace(request=lambda *a, **kw: None, getresponse=lambda response=response: response, close=lambda: None)
                public = [(2, 1, 6, '', ('93.184.216.34', 80))]
                with patch('borealis_coder.tools.fetch.socket.getaddrinfo', return_value=public), patch('borealis_coder.tools.fetch.http.client.HTTPConnection', return_value=connection):
                    result = await FetchUrlTool().execute({'url': 'http://example.test', 'max_chars': 100}, context)
                artifact = result.metadata['output_artifact']
                stored = self.runner.sessions.read_output_artifact(self.session.id, self.root, artifact['artifact_id'])
                self.assertEqual(stored['status'], 'partial' if partial else 'complete')
                self.assertEqual(stored['collection_status'], stored['status'])
                self.assertEqual(stored['stored_bytes'], min(len(source), 400))
                self.assertEqual(stored['observed_bytes'], min(len(source), 401))
                self.assertNotIn('NEVER-COLLECTED', stored['content'])
                self.assertLessEqual(len(result.output), 100)

    async def test_search_snapshot_describes_the_exact_scan_list(self):
        (self.root / 'old.txt').write_text('no match')
        context = make_context(self.root)
        calls = 0
        def walk(root, ctx):
            nonlocal calls
            files = list(_walk_files(root, ctx))
            calls += 1
            if calls == 1:
                (root / 'new.txt').write_text('needle')
            return iter(files)
        with patch('borealis_coder.tools.search._walk_files', side_effect=walk):
            result = await GrepTool().execute({'pattern': 'needle', 'path': '.', 'glob': '*.txt', 'regex': False,
                                               'case_sensitive': True, 'context_lines': 0, 'max_results': 10}, context)
        self.assertTrue(result.is_error)
        self.assertIn('changed', result.output)
        self.assertNotIn('completed search scope', result.output)
        self.assertFalse(result.metadata.get('complete', False))
