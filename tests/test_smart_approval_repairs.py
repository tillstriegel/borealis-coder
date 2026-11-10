"""Regressions for the four independently reviewed approval boundaries."""

from __future__ import annotations

import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from borealis_coder.agent import build_runner
from borealis_coder.agent.budget import Budget
from borealis_coder.auth import ChatGPTCredentials
from borealis_coder.config import AgentConfig, ProviderConfig
from borealis_coder.errors import (
    ApprovalDenied,
    BudgetExceeded,
    ProviderAuthenticationError,
    ProviderError,
)
from borealis_coder.models import ModelResponse, ProviderRequest, Usage
from borealis_coder.providers.anthropic import AnthropicProvider
from borealis_coder.providers.base import Provider
from borealis_coder.providers.chatgpt import ChatGPTProvider
from borealis_coder.providers.gemini import GeminiProvider
from borealis_coder.providers.http import HttpResponse, SSEEvent
from borealis_coder.providers.mock import MockProvider
from borealis_coder.providers.openai import OpenAIProvider
from borealis_coder.providers.openrouter import OpenRouterProvider
from borealis_coder.providers.registry import ProviderRegistry
from borealis_coder.safety import ApprovalRequest, PolicyAction, PolicyDecision
from borealis_coder.safety.smart import SmartApprovalJudge
from borealis_coder.util import finish_on_cancellation
from tests.helpers import make_config, make_context
from tests.test_smart_approvals import harness, receipts, shell_call

ALLOW = '{"decision":"allow"}'
DENY = '{"decision":"deny"}'


def chat_response():
    return {
        "model": "fixture-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": ALLOW},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.00001},
    }


def responses_response():
    return {
        "model": "fixture-model",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": ALLOW}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


class WireRootProvider(OpenRouterProvider):
    """Keep the actual judge serializer/parser; substitute root model I/O only."""

    def __init__(self, config):
        super().__init__(config)
        self.root_calls = 0

    async def complete(self, request):
        if request.metadata.get("purpose") == "smart_approval":
            return await super().complete(request)
        self.root_calls += 1
        return ModelResponse(
            tool_calls=[shell_call()] if self.root_calls == 1 else [],
            text="" if self.root_calls == 1 else "Finished.",
            usage=Usage(requests=1, cost_status="known"),
        )

    async def stream(self, request):
        async for event in Provider.stream(self, request):
            yield event


class ChatGPTWireRoot(ChatGPTProvider):
    def __init__(self, config):
        super().__init__(config)
        self.root_calls = 0

    async def complete(self, request):
        if request.metadata.get("purpose") == "smart_approval":
            return await super().complete(request)
        self.root_calls += 1
        return ModelResponse(
            tool_calls=[shell_call()] if self.root_calls == 1 else [],
            text="" if self.root_calls == 1 else "Finished.",
            usage=Usage(requests=1, cost_status="known"),
        )

    async def stream(self, request):
        stream = (
            ChatGPTProvider.stream(self, request)
            if request.metadata.get("purpose") == "smart_approval"
            else Provider.stream(self, request)
        )
        async for event in stream:
            yield event


def fake_auth(provider):
    refreshes = []
    credentials = ChatGPTCredentials(access_token="synthetic-access", account_id="fixture-account")

    async def ensure_valid(*, force_refresh=False):
        refreshes.append(force_refresh)
        return credentials

    provider.credentials = SimpleNamespace(ensure_valid=ensure_valid, load=lambda: credentials)
    return refreshes


class LateIntentRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_steering_reaches_judge_and_next_run_gets_fresh_intent(self):
        async with harness() as (root, runner, provider, _):
            entered = asyncio.Event()
            release = asyncio.Event()
            original_worker = runner._run_locked
            original_complete = provider.complete

            async def gated_worker(*args, **kwargs):
                entered.set()
                await release.wait()
                return await original_worker(*args, **kwargs)

            async def complete(request):
                if request.metadata.get("purpose") == "smart_approval":
                    intent = json.loads(request.messages[0].content)["user_intent"]
                    provider.answers[:] = [
                        ModelResponse(
                            text=DENY if "Do not create sentinel." in intent else ALLOW,
                            stop_reason="stop",
                        )
                    ]
                return await original_complete(request)

            runner._run_locked = gated_worker
            provider.complete = complete
            session = runner.sessions.create_session(
                workspace=root,
                provider="mock",
                model=provider.config.model,
            )
            task = asyncio.create_task(runner.run("Create sentinel.", session_id=session.id))
            try:
                await asyncio.wait_for(entered.wait(), 3)
                self.assertTrue(runner.accepts_steering(session.id))
                runner.steer(session.id, "Do not create sentinel.")
            finally:
                release.set()
                await asyncio.wait_for(task, 3)
            self.assertFalse((root / "sentinel").exists())
            self.assertEqual(
                json.loads(provider.judgments[0].messages[0].content)["user_intent"],
                ["Create sentinel.", "Do not create sentinel."],
            )
            self.assertEqual(runner._approval_intents, {})
            provider.batches.append([shell_call(call_id="fresh-run")])
            await runner.run("Create sentinel.", session_id=session.id)
            self.assertTrue((root / "sentinel").exists())
            self.assertEqual(
                json.loads(provider.judgments[-1].messages[0].content)["user_intent"],
                ["Create sentinel."],
            )
            self.assertEqual(runner._approval_intents, {})

    async def test_startup_cancellation_cleans_up_approval_intent(self):
        async with harness() as (root, runner, provider, _):
            entered = asyncio.Event()
            release = asyncio.Event()
            original_worker = runner._run_locked

            async def blocked_worker(*args, **kwargs):
                entered.set()
                await release.wait()
                return await original_worker(*args, **kwargs)

            runner._run_locked = blocked_worker
            session = runner.sessions.create_session(
                workspace=root,
                provider="mock",
                model=provider.config.model,
            )
            task = asyncio.create_task(runner.run("Create sentinel.", session_id=session.id))
            await asyncio.wait_for(entered.wait(), 3)
            self.assertTrue(runner.accepts_steering(session.id))
            task.cancel()
            await asyncio.wait_for(task, 3)
            self.assertEqual(runner._approval_intents, {})
            self.assertFalse(runner.accepts_steering(session.id))
            self.assertFalse((root / "sentinel").exists())
            self.assertEqual(provider.judgments, [])

    async def test_cancelled_receipt_delivery_still_records_late_intent_denial(self):
        async with harness() as (root, runner, _, _):

            async def revoke_and_cancel(event):
                if event.type == "approval.decided" and event.data["decision"] == "allow":
                    runner.steer(event.session_id, "Do not create sentinel.")
                    runner.cancel(event.session_id)
                    await asyncio.sleep(0.03)

            runner.events.subscribe(revoke_and_cancel)
            result = await runner.run("Create sentinel.")
            self.assertFalse((root / "sentinel").exists())
            self.assertEqual(result.stop_reason, "cancelled")
            self.assertEqual(receipts(runner)[-1]["decision"], "deny")
            self.assertEqual(receipts(runner)[-1]["reason"], "intent_changed")

    async def test_intent_change_during_receipt_delivery_denies_execution(self):
        async with harness() as (root, runner, _, humans):

            async def revoke(event):
                if event.type == "approval.decided" and event.data["decision"] == "allow":
                    await asyncio.sleep(0)
                    runner.steer(event.session_id, "Do not create sentinel.")

            runner.events.subscribe(revoke)
            await runner.run("Create sentinel.")
            self.assertFalse((root / "sentinel").exists())
            decisions = receipts(runner)
            self.assertEqual(decisions[-1]["decision"], "deny")
            self.assertEqual(decisions[-1]["reason"], "intent_changed")
            self.assertTrue(decisions[-1]["supersedes"])
            self.assertEqual(humans, [])


class SettlementRepairTests(unittest.IsolatedAsyncioTestCase):
    async def check_settlement(self, *, incomplete, cancel=False):
        with tempfile.TemporaryDirectory() as directory:
            context = make_context(Path(directory))
            config = ProviderConfig(
                model="fixture-model", input_cost_per_million=0, output_cost_per_million=1000
            )
            provider = MockProvider(config)
            provider.name = "priced"
            observed = Usage(
                requests=1,
                cost_usd=0 if incomplete else 1.2,
                cost_status="incomplete" if incomplete else "known",
            )

            async def complete(request):
                with provider.request_attempt() as record:
                    record(observed)
                return ModelResponse(text=ALLOW, stop_reason="stop", usage=observed)

            provider.complete = complete
            budget = Budget.start(AgentConfig(max_cost_usd=1.0))
            settled = []

            async def sink(usage):
                settled.append(usage)
                budget.add_usage(usage)
                await asyncio.sleep(0)

            judge = SmartApprovalJudge(
                provider, "fixture-model", budget, sink, context, ["Create sentinel."]
            )
            approval = ApprovalRequest(
                "shell",
                "untrusted",
                PolicyDecision(PolicyAction.ASK, "risky"),
                "",
                arguments=shell_call().arguments,
            )
            admissions = []
            protected_calls = 0
            caller = asyncio.current_task()

            async def probe():
                candidate = ProviderRequest(
                    model="fixture-model", system="", messages=[], max_output_tokens=200
                )
                try:
                    reservation = budget.reserve_cost(provider, candidate)
                except BudgetExceeded:
                    admissions.append("blocked")
                else:
                    admissions.append("admitted")
                    budget.release_cost(reservation)

            async def protected(operation):
                nonlocal protected_calls
                protected_calls += 1
                if protected_calls == 1:
                    # The competing task is ready before protected settlement starts.
                    task = asyncio.create_task(probe())
                    if cancel:
                        assert caller is not None
                        caller.cancel()
                    try:
                        return await finish_on_cancellation(operation)
                    finally:
                        await task
                return await finish_on_cancellation(operation)

            error = asyncio.CancelledError if cancel else ApprovalDenied
            with (
                patch("borealis_coder.safety.smart.finish_on_cancellation", protected),
                self.assertRaises(error),
            ):
                await judge.enforce(approval)
            self.assertEqual(admissions, ["blocked"])
            self.assertEqual(len(settled), 1)
            assert budget.usage is not None
            self.assertEqual(budget.usage.requests, 1)
            self.assertEqual(budget.usage.cost_usd, observed.cost_usd)
            self.assertEqual(budget.pending_cost_usd, 0)
            self.assertEqual(budget._held_usage, {})

    async def test_known_charge_never_becomes_unheld_before_settlement(self):
        await self.check_settlement(incomplete=False)

    async def test_incomplete_usage_never_becomes_unheld_before_settlement(self):
        await self.check_settlement(incomplete=True)

    async def test_cancelled_settlement_keeps_holds_until_once_only_accounting(self):
        await self.check_settlement(incomplete=True, cancel=True)


class ChatGPTReplayRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_smart_auth_failure_cannot_refresh_or_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(
                root,
                agent={"provider": "chatgpt", "max_model_requests": 3},
                safety={"approval": "smart", "network": True},
            )
            provider = ChatGPTWireRoot(config.providers["chatgpt"])
            refreshes = fake_auth(provider)
            attempts = []

            async def stream_sse(*args, **kwargs):
                attempts.append(kwargs["payload"])
                if len(attempts) == 1:
                    raise ProviderAuthenticationError("synthetic auth failure")
                yield SSEEvent(
                    event="response.completed",
                    data=json.dumps(
                        {"type": "response.completed", "response": responses_response()}
                    ),
                )

            provider.http.stream_sse = stream_sse
            registry = ProviderRegistry()
            registry.register("chatgpt", lambda *_: provider)
            runner = await build_runner(
                root, config=config, provider_registry=registry, interactive=False
            )
            try:
                result = await runner.run("Create sentinel.")
                self.assertFalse((root / "sentinel").exists())
                self.assertEqual(len(attempts), 1)
                self.assertEqual(refreshes, [False])
                self.assertEqual(provider.root_calls, 2)
                self.assertEqual(result.usage.requests, 3)
                self.assertEqual(receipts(runner)[-1]["decision"], "deny")
            finally:
                await runner.close()

    async def test_ordinary_auth_replay_uses_request_admission_and_accounting(self):
        provider = ChatGPTProvider(ProviderConfig(type="chatgpt"))
        refreshes = fake_auth(provider)
        attempts = []

        async def stream_sse(*args, **kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise ProviderAuthenticationError("synthetic auth failure")
            yield SSEEvent(
                event="response.completed",
                data=json.dumps({"type": "response.completed", "response": responses_response()}),
            )

        provider.http.stream_sse = stream_sse
        budget = Budget.start(AgentConfig(max_model_requests=3))
        total = Usage()
        budget.before_model_request()

        def observe(usage):
            total.add(usage)

        try:
            with provider.observe_requests(observe, None, budget.before_model_request):
                response = await provider.complete(
                    ProviderRequest(model="fixture-model", system="", messages=[])
                )
            self.assertEqual(response.text, ALLOW)
            self.assertEqual(len(attempts), 2)
            self.assertEqual(refreshes, [False, True])
            self.assertEqual(budget.model_requests, 2)
            self.assertEqual(total.requests, 2)
        finally:
            await provider.close()


class EnvelopeRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_anthropic_tool_free_judge_omits_tool_choice_on_wire(self):
        with tempfile.TemporaryDirectory() as directory:
            context = make_context(Path(directory))
            provider = AnthropicProvider(
                ProviderConfig(
                    type="anthropic",
                    model="fixture-model",
                    base_url="https://api.anthropic.com/v1",
                )
            )
            budget = Budget.start(AgentConfig())
            payloads = []

            async def post_json(*args, **kwargs):
                payload = kwargs["payload"]
                payloads.append(payload)
                self.assertEqual(payload["tools"], [])
                self.assertNotIn("tool_choice", payload)
                data = {
                    "type": "message",
                    "role": "assistant",
                    "model": "fixture-model",
                    "content": [{"type": "text", "text": ALLOW}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
                return HttpResponse(
                    status=200, headers={}, data=data, raw=json.dumps(data).encode()
                )

            async def sink(usage):
                budget.add_usage(usage)

            provider.http.post_json = post_json
            judge = SmartApprovalJudge(
                provider,
                "fixture-model",
                budget,
                sink,
                context,
                ["Create sentinel."],
            )
            try:
                await judge.enforce(
                    ApprovalRequest(
                        "shell",
                        "untrusted",
                        PolicyDecision(PolicyAction.ASK, "risky"),
                        "",
                        arguments=shell_call().arguments,
                    )
                )
                self.assertEqual(len(payloads), 1)
                assert budget.usage is not None
                self.assertEqual(budget.usage.requests, 1)
            finally:
                await provider.close()

    async def test_anthropic_tools_keep_nonparallel_choice_control(self):
        provider = AnthropicProvider(ProviderConfig(type="anthropic"))
        try:
            request = ProviderRequest(
                model="fixture-model",
                system="",
                messages=[],
                parallel_tool_calls=False,
                tools=[{"name": "read", "parameters": {"type": "object", "properties": {}}}],
            )
            for stream in (False, True):
                self.assertEqual(
                    provider._payload(request, stream=stream)["tool_choice"],
                    {
                        "type": "auto",
                        "disable_parallel_tool_use": True,
                    },
                )
        finally:
            await provider.close()

    async def check_wire(self, data, *, allowed, extra_body=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(
                root,
                agent={"provider": "openrouter", "model": "fixture-model"},
                providers={"openrouter": {"api_key_env": ""}},
                safety={"approval": "smart", "network": True},
            )
            provider = WireRootProvider(config.providers["openrouter"])
            provider.config.extra_body = extra_body or {}
            wire_requests = []

            async def post_json(*args, **kwargs):
                wire_requests.append(kwargs["payload"])
                return HttpResponse(
                    status=200, headers={}, data=data, raw=json.dumps(data).encode()
                )

            provider.http.post_json = post_json
            registry = ProviderRegistry()
            registry.register("openrouter", lambda *_: provider)
            runner = await build_runner(
                root, config=config, provider_registry=registry, interactive=False
            )
            try:
                result = await runner.run("Create sentinel.")
                self.assertEqual((root / "sentinel").exists(), allowed)
                self.assertEqual(len(wire_requests), 1)
                self.assertEqual(result.usage.requests, 3)
                self.assertAlmostEqual(result.usage.cost_usd, 0.00001)
                self.assertEqual(receipts(runner)[-1]["decision"], "allow" if allowed else "deny")
            finally:
                await runner.close()

    async def test_single_wire_allow_still_executes(self):
        await self.check_wire(chat_response(), allowed=True)

    async def test_openrouter_native_finish_metadata_remains_supported(self):
        for native_finish_reason in ("stop", None):
            with self.subTest(native_finish_reason=native_finish_reason):
                data = chat_response()
                data["choices"][0]["native_finish_reason"] = native_finish_reason
                await self.check_wire(data, allowed=True)

    async def test_explicit_single_completion_setting_remains_supported(self):
        await self.check_wire(chat_response(), allowed=True, extra_body={"n": 1})

    async def test_conflicting_completions_cannot_select_first_allow(self):
        data = chat_response()
        other = copy.deepcopy(data["choices"][0])
        other["index"] = 1
        other["message"]["content"] = DENY
        data["choices"].append(other)
        await self.check_wire(data, allowed=False)

    async def test_ignored_chat_tool_or_protocol_fields_cannot_authorize(self):
        for field, value in [
            ("function_call", {"name": "shell", "arguments": "{}"}),
            ("refusal", "I cannot authorize this."),
            ("tool_calls", [{"type": "custom", "custom": {"name": "shell", "input": "danger"}}]),
        ]:
            with self.subTest(field=field):
                data = chat_response()
                data["choices"][0]["message"][field] = value
                await self.check_wire(data, allowed=False)

    async def test_unknown_chat_content_cannot_be_ignored(self):
        data = chat_response()
        data["choices"][0]["message"]["content"] = [
            {"type": "text", "text": ALLOW},
            {"type": "function_call", "name": "shell"},
        ]
        await self.check_wire(data, allowed=False)

    async def test_ordinary_multiple_completions_remain_compatible(self):
        provider = OpenRouterProvider(ProviderConfig(type="openrouter"))
        data = chat_response()
        data["choices"].append({"message": {"content": DENY}, "finish_reason": "stop"})
        try:
            self.assertEqual(provider._parse_chat(data, retain_raw=False).text, ALLOW)
        finally:
            await provider.close()

    async def test_equivalent_adapters_reject_ignored_nontext_outputs(self):
        for protocol in ("responses", "anthropic", "gemini"):
            for unexpected in (False, True):
                with self.subTest(protocol=protocol, unexpected=unexpected):
                    if protocol == "responses":
                        provider = OpenAIProvider(
                            ProviderConfig(type="openai", model="fixture-model")
                        )
                        data = responses_response()
                        if unexpected:
                            data["output"].append(
                                {"type": "web_search_call", "status": "completed"}
                            )
                    elif protocol == "anthropic":
                        provider = AnthropicProvider(
                            ProviderConfig(type="anthropic", model="fixture-model")
                        )
                        data = {
                            "type": "message",
                            "role": "assistant",
                            "model": "fixture-model",
                            "content": [{"type": "text", "text": ALLOW}],
                            "stop_reason": "end_turn",
                            "usage": {"input_tokens": 10, "output_tokens": 5},
                        }
                        if unexpected:
                            data["content"].append(
                                {"type": "server_tool_use", "name": "web_search"}
                            )
                    else:
                        provider = GeminiProvider(
                            ProviderConfig(type="gemini", model="fixture-model")
                        )
                        data = {
                            "model": "fixture-model",
                            "status": "completed",
                            "steps": [
                                {
                                    "type": "model_output",
                                    "content": [{"type": "text", "text": ALLOW}],
                                }
                            ],
                            "usage": {"total_input_tokens": 10, "total_output_tokens": 5},
                        }
                        if unexpected:
                            data["steps"].append({"type": "computer_call", "name": "shell"})

                    async def post_json(*args, data=data, **kwargs):
                        return HttpResponse(
                            status=200, headers={}, data=data, raw=json.dumps(data).encode()
                        )

                    provider.http.post_json = post_json
                    request = ProviderRequest(
                        model="fixture-model",
                        system="",
                        messages=[],
                        metadata={"purpose": "smart_approval"},
                    )
                    try:
                        if unexpected:
                            with self.assertRaises(ProviderError):
                                await provider.complete_once(request)
                        else:
                            response = await provider.complete_once(request)
                            self.assertEqual(response.text, ALLOW)
                    finally:
                        await provider.close()

    async def test_multicompletion_setting_is_rejected_before_judge_dispatch(self):
        for count in (2, 0, "1", True):
            with self.subTest(count=count):
                async with harness() as (root, runner, provider, _):

                    def change(event, count=count):
                        if event.type == "tool.policy":
                            provider.config.extra_body = {"n": count}

                    runner.events.subscribe(change)
                    result = await runner.run("Create sentinel.")
                    self.assertFalse((root / "sentinel").exists())
                    self.assertEqual(provider.judgments, [])
                    self.assertEqual(result.usage.requests, 2)

    async def test_chatgpt_stream_rejects_ignored_tool_output(self):
        provider = ChatGPTProvider(ProviderConfig(type="chatgpt"))
        fake_auth(provider)
        data = responses_response()
        data["output"].append({"type": "web_search_call", "status": "completed"})

        async def stream_sse(*args, **kwargs):
            yield SSEEvent(
                event="response.completed",
                data=json.dumps({"type": "response.completed", "response": data}),
            )

        provider.http.stream_sse = stream_sse
        try:
            with self.assertRaises(ProviderError):
                await provider.complete_once(
                    ProviderRequest(
                        model="fixture-model",
                        system="",
                        messages=[],
                        metadata={"purpose": "smart_approval"},
                    )
                )
        finally:
            await provider.close()
