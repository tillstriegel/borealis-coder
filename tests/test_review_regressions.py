from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from borealis_coder.agent import build_runner
from borealis_coder.agent.budget import Budget
from borealis_coder.agent.runner import ProviderRoute
from borealis_coder.config import AgentConfig, ProviderConfig, load_config
from borealis_coder.errors import (
    BudgetExceeded,
    Cancelled,
    ProviderContextOverflowError,
    ProviderUnavailableError,
)
from borealis_coder.models import (
    Event,
    Message,
    ModelResponse,
    ProviderRequest,
    Role,
    ToolCall,
    ToolResult,
    Usage,
)
from borealis_coder.providers.base import Provider, ProviderStreamEvent
from borealis_coder.providers.http import SSEEvent
from borealis_coder.providers.openai import OpenAICompatibleProvider, OpenAIProvider
from borealis_coder.providers.openrouter import OpenRouterProvider
from borealis_coder.tools.base import ToolRegistry, bound_tool_output
from borealis_coder.tools.delegate import DelegateTaskTool
from borealis_coder.util import json_dumps
from tests.helpers import make_config, make_context


def billed(amount):
    return Usage(requests=1, cost_usd=amount, cost_status='known')


class ScriptedProvider(Provider):
    name = 'test'

    def __init__(self, *actions, model='main', retries=0, output_rate=6000):
        super().__init__(ProviderConfig(model=model, max_retries=retries, initial_backoff_seconds=0,
                                       input_cost_per_million=0, output_cost_per_million=output_rate))
        self.actions = list(actions)
        self.seen: list[ProviderRequest] = []

    async def complete(self, request):
        self.seen.append(request)
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action


class ReviewRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config = make_config(self.root, cache={'response_cache_enabled': False},
                                  agent={'deterministic_compaction': False, 'small_model': 'main'})
        self.runner = await build_runner(self.root, config=self.config, interactive=False)
        self.addAsyncCleanup(self.runner.close)
        self.session = self.runner.sessions.create_session(workspace=self.root, provider='test', model='main')
        self.request = ProviderRequest(model='main', system='system', messages=[], max_output_tokens=100)
        self.recorded = Usage()

    async def settle(self, usage, *, reservation=0):
        self.recorded.add(usage)

    def routes(self, *providers):
        self.runner.providers = [ProviderRoute(str(i), p.config.model, p) for i, p in enumerate(providers)]

    async def complete(self, budget=None, cancel=None):
        return await self.runner._complete_request(self.request, self.session.id, 'run', cancel or asyncio.Event(),
                                                   'assistant', usage_sink=self.settle, budget=budget)

    def delegated(self, provider, budget):
        context = make_context(self.root, self.config)
        context.metadata.update(provider_routes=[ProviderRoute('test', 'main', provider)],
                                tool_registry=ToolRegistry([]), context_builder=SimpleNamespace(system_prompt=lambda **_: 'Investigate.'),
                                budget=budget, before_model_request=budget.before_model_request, usage_sink=self.settle)
        return context

    def test_shipped_defaults_allow_unpriced_live_routes_but_explicit_limits_fail(self):
        # Load shipped defaults, without make_config's provider or budget overrides.
        with patch.object(Path, 'is_file', return_value=False), patch('borealis_coder.config._environment_overrides', return_value={}):
            config = load_config(self.root, ensure_storage=False)
            strict = load_config(self.root, overrides={'agent': {'max_cost_usd': 25}}, ensure_storage=False)
        for name in ('openai', 'openrouter', 'anthropic', 'gemini', 'chatgpt', 'openai_compatible'):
            with self.subTest(provider=name):
                provider = OpenAIProvider(config.providers[name])
                provider.name = name
                self.addAsyncCleanup(provider.close)
                request = ProviderRequest(model=provider.config.model, system='', messages=[])
                self.assertIsNone(provider.prices(request.model))
                self.assertEqual(Budget.start(config.agent).reserve_cost(provider, request), 0)
                with self.assertRaisesRegex(BudgetExceeded, 'pricing'):
                    Budget.start(strict.agent).reserve_cost(provider, request)
        self.assertEqual(strict.agent.max_cost_usd, 25)

    async def test_failed_route_plus_fallback_reservation_is_cumulative(self):
        first = ScriptedProvider(ProviderUnavailableError('failed', usage=billed(.6)))
        backup = ScriptedProvider(ModelResponse(text='done', usage=billed(.6)))
        self.routes(first, backup)
        budget = Budget.start(AgentConfig(max_cost_usd=1))
        with self.assertRaises(BudgetExceeded):
            await self.complete(budget)
        self.assertEqual(len(first.seen), 1)
        self.assertEqual(backup.seen, [])
        self.assertAlmostEqual(self.recorded.cost_usd, .6)
        self.assertEqual(self.recorded.requests, 1)
        self.assertEqual(budget.pending_cost_usd, 0)

    async def test_three_routes_include_settled_and_concurrent_costs(self):
        for concurrent, expected_dispatches in ((0, 3), (.3, 2)):
            with self.subTest(concurrent=concurrent):
                providers = [ScriptedProvider(ProviderUnavailableError('failed', usage=billed(.3)), output_rate=3000)
                             for _ in range(2)] + [ScriptedProvider(ModelResponse(text='done', usage=billed(.3)), output_rate=3000)]
                self.routes(*providers)
                budget = Budget.start(AgentConfig(max_cost_usd=1.05))
                budget.add_usage(billed(.1))
                other = budget.reserve_cost(providers[0], self.request) if concurrent else 0
                self.recorded = Usage()
                if concurrent:
                    with self.assertRaises(BudgetExceeded):
                        await self.complete(budget)
                    self.assertAlmostEqual(self.recorded.cost_usd, .6)
                else:
                    response, route = await self.complete(budget)
                    budget.add_usage(response.usage)
                    self.assertEqual(route.name, '2')
                    self.assertEqual(response.usage.requests, 3)
                    self.assertAlmostEqual(response.usage.cost_usd, .9)
                    self.assertAlmostEqual(budget.usage.cost_usd, 1.0)  # type: ignore[union-attr]
                    self.assertTrue(self.recorded.is_empty)
                self.assertEqual(sum(len(p.seen) for p in providers), expected_dispatches)
                self.assertAlmostEqual(budget.pending_cost_usd, other)
                budget.release_cost(other)

    async def test_failed_cost_is_visible_to_other_concurrent_requests(self):
        first = ScriptedProvider(ProviderUnavailableError('failed', usage=billed(.6)))
        backup = ScriptedProvider(ModelResponse(text='done', usage=billed(.1)), output_rate=1000)
        self.routes(first, backup)
        budget = Budget.start(AgentConfig(max_cost_usd=1))
        checked = []
        async def observer(event):
            if event.type == 'model.route_failed':
                with self.assertRaises(BudgetExceeded):
                    budget.reserve_cost(first, self.request)
                checked.append(True)
        self.runner.events.subscribe(observer)
        response, _ = await self.complete(budget)
        self.assertEqual(checked, [True])
        self.assertAlmostEqual(response.usage.cost_usd, .7)
        self.assertEqual(budget.pending_cost_usd, 0)

    async def test_cancellation_after_billed_failure_marks_each_request_path_incomplete(self):
        for path in ('provider', 'main', 'fallback', 'delegate', 'summarizer'):
            with self.subTest(path=path):
                failure = ProviderUnavailableError('retry', retryable=True, usage=billed(.2))
                provider = ScriptedProvider(failure, asyncio.CancelledError(), retries=1)
                self.recorded = Usage()
                self.routes(provider)
                with self.assertRaises(asyncio.CancelledError):
                    if path == 'provider':
                        await provider.with_retries(lambda provider=provider: provider.complete(self.request), request=self.request,
                                                    failed_usage_collector=self.recorded)
                    elif path in ('main', 'fallback'):
                        if path == 'fallback':
                            provider.config.max_retries = 0
                            self.routes(provider, ScriptedProvider(asyncio.CancelledError()))
                        await self.complete()
                    elif path == 'delegate':
                        await DelegateTaskTool().execute({'task': 'Inspect', 'max_turns': 1}, self.delegated(provider, Budget.start(self.config.agent)))
                    else:
                        summarize = self.runner._summarizer(self.settle, asyncio.Event())
                        assert summarize
                        await summarize('history')
                self.assertEqual(self.recorded.cost_status, 'incomplete')
                self.assertAlmostEqual(self.recorded.cost_usd, .2)
                self.assertEqual(self.recorded.requests, 2)

    async def test_terminal_stream_usage_is_not_marked_incomplete_on_cancellation(self):
        cancel = asyncio.Event()
        class TerminalProvider(ScriptedProvider):
            async def stream(self, request):
                cancel.set()
                yield ProviderStreamEvent(type='completed', response=ModelResponse(text='done', usage=billed(.2)))
        self.routes(TerminalProvider())
        with self.assertRaises(Cancelled):
            await self.complete(cancel=cancel)
        self.assertEqual(self.recorded.cost_status, 'known')
        self.assertEqual(self.recorded.requests, 1)
        self.assertAlmostEqual(self.recorded.cost_usd, .2)

    async def test_cancellation_during_retry_backoff_does_not_invent_attempt(self):
        provider = ScriptedProvider(ProviderUnavailableError('retry', retryable=True, usage=billed(.2)), retries=1)
        with patch('borealis_coder.providers.base.asyncio.sleep', side_effect=asyncio.CancelledError), self.assertRaises(asyncio.CancelledError):
            await provider.with_retries(lambda provider=provider: provider.complete(self.request), failed_usage_collector=self.recorded)
        self.assertEqual(self.recorded.requests, 1)
        self.assertEqual(self.recorded.cost_status, 'known')

    async def test_settled_helper_and_summary_usage_remains_known_on_cancellation(self):
        for path in ('delegate', 'summarizer'):
            with self.subTest(path=path):
                provider = ScriptedProvider(ProviderUnavailableError('retry', retryable=True, usage=billed(.2)),
                                            ModelResponse(text='done', usage=billed(.1)), retries=1)
                self.recorded = Usage()
                self.routes(provider)
                cancel = asyncio.Event()
                async def settle_and_cancel(usage, cancel=cancel, path=path, **kwargs):
                    self.recorded.add(usage)
                    cancel.set()
                    if path == 'delegate':
                        raise asyncio.CancelledError
                if path == 'delegate':
                    context = self.delegated(provider, Budget.start(self.config.agent))
                    context.metadata['usage_sink'] = settle_and_cancel
                    with self.assertRaises(asyncio.CancelledError):
                        await DelegateTaskTool().execute({'task': 'Inspect', 'max_turns': 1}, context)
                else:
                    summarize = self.runner._summarizer(settle_and_cancel, cancel)
                    assert summarize
                    with self.assertRaises(Cancelled):
                        await summarize('history')
                self.assertEqual(self.recorded.cost_status, 'known')
                self.assertEqual(self.recorded.requests, 2)
                self.assertAlmostEqual(self.recorded.cost_usd, .3)

    async def test_delegate_overflow_retries_same_logical_turn(self):
        for retries, max_turns, before_synthesis, exhausted in ((1, 1, False, False), (2, 1, False, False),
                                                              (1, 2, True, False), (2, 1, False, True)):
            with self.subTest(retries=retries, max_turns=max_turns, exhausted=exhausted):
                self.config.agent.compaction_max_overflow_retries = retries
                actions = [ModelResponse(tool_calls=[ToolCall(name='unavailable', arguments={})], usage=billed(0))] if before_synthesis else []
                actions += [ProviderContextOverflowError('too large') for _ in range(retries + int(exhausted))]
                actions += [ModelResponse(text='finished', usage=billed(0))]
                provider = ScriptedProvider(*actions)
                budget = Budget.start(self.config.agent)
                context = self.delegated(provider, budget)
                if exhausted:
                    with self.assertRaises(ProviderContextOverflowError):
                        await DelegateTaskTool().execute({'task': 'Inspect', 'max_turns': max_turns}, context)
                else:
                    result = await DelegateTaskTool().execute({'task': 'Inspect', 'max_turns': max_turns}, context)
                    self.assertIn('finished', result.output)
                    self.assertEqual(result.metadata['turns'], max_turns)
                self.assertEqual(budget.model_requests, retries + 1 + int(before_synthesis))
                requests = provider.seen[int(before_synthesis):]
                limits = [r.metadata['resolved_limits']['effective_context_tokens'] for r in requests]
                self.assertEqual(limits, sorted(set(limits), reverse=True))
                self.assertTrue(all('final turn' in r.system for r in requests))

    def test_history_search_uses_literal_decoded_content(self):
        store = self.runner.sessions
        queries = ['print("hello")', r'C:\src\main.py', 'first\nsecond', 'left\tright']
        for query in queries:
            message = Message(role=Role.USER, content='prefix ' + query + ' suffix')
            store.append_message(self.session.id, message)
            store.append_event(Event(type='tool.completed', session_id=self.session.id,
                                     data={'output': message.content, 'tool_call_id': message.id}))
            with self.subTest(query=query):
                messages = store.search_history(self.session.id, self.root, query)
                tools = store.search_tool_history(self.session.id, self.root, query)
                self.assertEqual([m['message_id'] for m in messages], [message.id])
                self.assertEqual([t['source'] for t in tools], [message.id])
                self.assertIn(query, tools[0]['preview'])
                self.assertEqual(store.search_history(self.session.id, self.root, message.id)[0]['message_id'], message.id)
                self.assertEqual(store.search_tool_history(self.session.id, self.root, message.id)[0]['source'], message.id)
                self.assertEqual(store.search_history(self.session.id, self.root, query, after=messages[0]['sequence']), [])
                self.assertEqual(store.search_tool_history(self.session.id, self.root, query, after=tools[0]['sequence']), [])

    def test_output_ceiling_includes_complete_reference_and_is_idempotent(self):
        context = make_context(self.root)
        context.session_id = self.session.id
        context.metadata['session_store'] = self.runner.sessions
        for limit in (0, 1, 10, 28, 50, 100, 1000):
            for content in ('short', 'evidence\n' * 1000):
                with self.subTest(limit=limit, content=content[:5]):
                    result = bound_tool_output(ToolResult(content), context, limit)
                    output = result.output
                    self.assertLessEqual(len(output), limit)
                    artifact = result.metadata.get('output_artifact')
                    if artifact:
                        artifact_id = artifact['artifact_id']
                        self.assertEqual(self.runner.sessions.read_output_artifact(self.session.id, self.root, artifact_id)['content'], content[:8000])
                        if limit >= len(artifact_id):
                            self.assertIn(artifact_id, output)
                    bound_tool_output(result, context, limit)
                    self.assertEqual(result.output, output)
                    self.assertLessEqual(len(result.output), limit)
        # Also exercise results whose artifact was captured by a process driver.
        artifact = self.runner.sessions.save_output_artifact(self.session.id, self.root, 'evidence', source='call', redactor=context.events.redactor)
        result = bound_tool_output(ToolResult('preview', metadata={'output_artifact': artifact}), context, 50)
        self.assertLessEqual(len(result.output), 50)
        self.assertIn(artifact['artifact_id'], result.output)


class ProviderReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_alias_snapshot_override_and_actual_fallback_pricing(self):
        for cls in (OpenAIProvider, OpenRouterProvider):
            provider = cls(ProviderConfig(model='main', input_cost_per_million=2, output_cost_per_million=4,
                                          model_fallbacks=['other'], model_prices={'override': {'input_cost_per_million': 10, 'output_cost_per_million': 20},
                                                                               'other': {'input_cost_per_million': 1, 'output_cost_per_million': 2}}))
            self.addAsyncCleanup(provider.close)
            for reported, expected in (('main-2026-09-01', .006), ('override', .03), ('other', .003)):
                for style in ('chat', 'responses'):
                    with self.subTest(provider=cls.__name__, reported=reported, style=style):
                        data = {'model': reported, 'choices': [{'message': {'content': 'done'}, 'finish_reason': 'stop'}],
                                'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'done'}]}],
                                'usage': {'input_tokens': 1000, 'output_tokens': 1000, 'prompt_tokens': 1000, 'completion_tokens': 1000}}
                        parser = provider._parse_chat if style == 'chat' else provider._parse_responses
                        response = parser(data, retain_raw=False)
                        self.assertAlmostEqual(response.usage.cost_usd, expected)
                        self.assertEqual(response.usage.cost_status, 'estimated')
                        self.assertEqual(response.model, reported)
                        self.assertEqual(provider.request_model.get(), 'main')
            provider.config.model_prices.pop('other')
            usage = provider._usage_from_chat({'prompt_tokens': 1000, 'completion_tokens': 1000}, model='other')
            self.assertEqual(usage.cost_status, 'unknown')

    async def test_anthropic_and_gemini_preserve_alias_and_honor_price_override(self):
        from borealis_coder.providers.anthropic import AnthropicProvider
        from borealis_coder.providers.gemini import GeminiProvider

        for cls in (AnthropicProvider, GeminiProvider):
            provider = cls(ProviderConfig(model='main', input_cost_per_million=2, output_cost_per_million=4,
                                          model_prices={'override': {'input_cost_per_million': 10, 'output_cost_per_million': 20}}))
            self.addAsyncCleanup(provider.close)
            for model, expected in (('main-snapshot', .006), ('override', .03)):
                with self.subTest(provider=cls.__name__, model=model):
                    response = provider._parse({'model': model, 'status': 'completed', 'stop_reason': 'end_turn',
                                                'usage': {'input_tokens': 1000, 'output_tokens': 1000,
                                                          'total_input_tokens': 1000, 'total_output_tokens': 1000}}, retain_raw=False)
                    self.assertAlmostEqual(response.usage.cost_usd, expected)
                    self.assertEqual(response.model, model)

    async def test_chat_alias_counts_the_payload_actually_sent(self):
        request = ProviderRequest(model='main', system='日本語', messages=[Message(role=Role.USER, content='Grüße 🙂')],
                                  tools=[{'name': 'read', 'parameters': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}}],
                                  response_schema={'type': 'object', 'properties': {'text': {'type': 'string'}}, 'required': ['text'], 'additionalProperties': False})
        for cls in (OpenAIProvider, OpenAICompatibleProvider, OpenRouterProvider):
            for style in ('chat', 'chat_completions'):
                with self.subTest(provider=cls.__name__, style=style):
                    provider = cls(ProviderConfig(model='main', api_style=style, input_cost_per_million=2, output_cost_per_million=4))
                    self.addAsyncCleanup(provider.close)
                    sent = []
                    async def post(url, sent=sent, **kwargs):
                        self.assertTrue(url.endswith('/chat/completions'))
                        sent.append(kwargs['payload'])
                        return SimpleNamespace(data={'model': 'main', 'choices': [{'message': {'content': 'done'}, 'finish_reason': 'stop'}],
                                                     'usage': {'prompt_tokens': 1, 'completion_tokens': 1}})
                    async def stream(url, sent=sent, **kwargs):
                        self.assertTrue(url.endswith('/chat/completions'))
                        sent.append(kwargs['payload'])
                        yield SSEEvent(event='', data=json_dumps({'model': 'main', 'choices': [{'delta': {'content': 'done'}, 'finish_reason': 'stop'}],
                                                                 'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}))
                    provider.http.post_json = post  # type: ignore[method-assign]
                    provider.http.stream_sse = stream  # type: ignore[method-assign]
                    await provider.complete(request)
                    events = [event async for event in provider.stream(request)]
                    response = events[-1].response
                    assert response
                    self.assertEqual(response.model, 'main')
                    self.assertAlmostEqual(response.usage.cost_usd, .000006)
                    self.assertEqual(provider.request_model.get(), 'main')
                    sizes = [len(json_dumps(payload).encode('utf-8')) for payload in sent]
                    self.assertEqual(provider.request_bytes(request), max(sizes))
                    self.assertEqual([p['stream'] for p in sent], [False, True])
                    self.assertTrue(all(p['tools'] and p['response_format'] for p in sent))

    async def test_unpriced_native_response_uses_requested_alias(self):
        provider = ScriptedProvider(ModelResponse(text='done', model='main-snapshot', usage=Usage(requests=1, input_tokens=1000)), output_rate=0)
        provider.config.input_cost_per_million = 2
        request = ProviderRequest(model='main', system='', messages=[])
        response = await provider.with_retries(lambda: provider.complete(request), request=request)
        self.assertAlmostEqual(response.usage.cost_usd, .002)
        self.assertEqual(response.model, 'main-snapshot')
