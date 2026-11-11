"""Optional request-count limits retain accounting and the other run limits."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from borealis_coder.agent import build_runner, validate_tool_call_order
from borealis_coder.agent.budget import Budget
from borealis_coder.agent.runner import _remaining_budget_guidance
from borealis_coder.config import AgentConfig, ProviderConfig
from borealis_coder.errors import BudgetExceeded, ProviderUnavailableError
from borealis_coder.models import Event, ModelResponse, ToolCall, Usage
from borealis_coder.providers.mock import MockProvider
from borealis_coder.providers.registry import ProviderRegistry
from borealis_coder.swarm_view import SwarmView
from tests.helpers import make_config
from tests.test_smart_approvals import answer, harness, receipts
from tests.test_swarm import delegate, response


class OptionalBudgetTests(unittest.TestCase):
    def test_unlimited_keeps_reservations_and_counts_beyond_default(self):
        budget = Budget.start(AgentConfig(max_model_requests=0))
        for owner in ("root", "child", "grandchild"):
            self.assertTrue(budget.reserve_closure(owner))
        self.assertIsNone(budget.remaining_requests())
        self.assertIsNone(budget.remaining_requests("child"))
        for _ in range(3001):
            budget.before_model_request("child", closure=False)
        self.assertEqual(budget.model_requests, 3001)
        self.assertIsNone(budget.remaining_requests())
        self.assertTrue(budget.has_request_capacity())
        self.assertFalse(budget.last_request)
        self.assertEqual(budget._request_closures, {"root", "child", "grandchild"})
        budget.before_model_request("child")
        self.assertEqual(budget._request_closures, {"root", "grandchild"})
        budget.add_usage(Usage(requests=3002, cost_usd=0.25))
        assert budget.usage is not None
        self.assertEqual(budget.usage.requests, 3002)
        self.assertEqual(budget.usage.cost_usd, 0.25)

    def test_default_request_limit_is_finite(self):
        budget = Budget.start(AgentConfig())
        self.assertEqual(budget.remaining_requests(), 2500)
        for _ in range(2500):
            budget.before_model_request()
        self.assertTrue(budget.last_request)
        self.assertFalse(budget.has_request_capacity())
        with self.assertRaises(BudgetExceeded) as caught:
            budget.before_model_request()
        self.assertEqual(caught.exception.kind, "model_requests")
        self.assertEqual(budget.model_requests, 2500)

    def test_unlimited_still_checks_time_cost_and_turns(self):
        for kind in ("time", "cost", "turns"):
            with self.subTest(kind=kind):
                budget = Budget.start(AgentConfig(max_model_requests=0, max_turns=1, max_cost_usd=1))
                if kind == "time":
                    budget.started_ms -= 3_600_000
                elif kind == "cost":
                    budget.add_usage(Usage(cost_usd=1))
                else:
                    budget.before_turn()
                with self.assertRaises(BudgetExceeded) as caught:
                    if kind == "turns":
                        budget.before_turn()
                    else:
                        budget.before_model_request()
                self.assertEqual(caught.exception.kind, kind)
                self.assertEqual(budget.model_requests, 0)

    def test_unlimited_guidance_and_status_are_strict_json_safe(self):
        budget = Budget.start(AgentConfig(max_model_requests=0))
        self.assertEqual(_remaining_budget_guidance(budget), "")
        budget.turns = budget.config.max_turns
        guidance = _remaining_budget_guidance(budget)
        self.assertIn("unlimited", guidance)
        self.assertNotIn("requests including this request: 0", guidance)
        view = SwarmView()
        event = Event(type="swarm.started", run_id="root", data={
            "task": "Inspect", "limits": {}, "max_model_requests": 0,
        })
        view.adopt(event)
        view.visible = True
        view.requests = 43
        for width in (60, 120):
            rendered = "".join(fragment[1] for fragment in view.render(width, 20))
            self.assertIn("requests 43/unlimited", rendered)
        json.dumps({"event": event.data, "remaining": budget.remaining_requests()}, allow_nan=False)


class OptionalBudgetRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_unlimited_fallback_after_billed_retries(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root, agent={
                "max_model_requests": 0, "provider": "broken", "provider_fallbacks": ["mock"],
            })
            config.providers["broken"] = ProviderConfig(
                type="broken", model="fixture", max_retries=1,
                initial_backoff_seconds=0, max_backoff_seconds=0,
            )
            broken = MockProvider(config.providers["broken"])

            def fail(request, count):
                raise ProviderUnavailableError("Billed failure", retryable=True,
                                               usage=Usage(requests=1, cost_usd=0.02))

            broken.handler = fail
            registry = ProviderRegistry()
            registry.register("broken", lambda *_: broken)
            runner = await build_runner(root, config=config, interactive=False, provider_registry=registry)
            budget = Budget.start(config.agent)
            try:
                with patch.object(Budget, "start", return_value=budget):
                    result = await runner.run("Inspect through the fallback route.")
                self.assertEqual(result.stop_reason.value, "end_turn", result.error)
                self.assertIn("Offline mock", result.text)
                self.assertEqual(broken.calls, 2)
                self.assertEqual(result.usage.requests, 3)
                self.assertEqual(budget.model_requests, 3)
                self.assertAlmostEqual(result.usage.cost_usd, 0.04)
                self.assertEqual(result.usage.to_dict(), runner.sessions.usage(result.session_id).to_dict())
            finally:
                await runner.close()

    async def test_unlimited_turn_cap_still_blocks_final_turn_write(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runner = await build_runner(root, config=make_config(root, agent={
                "max_model_requests": 0, "max_turns": 1,
            }), interactive=False)
            provider = runner.providers[0].provider
            assert isinstance(provider, MockProvider)
            events = []
            runner.events.subscribe(events.append)

            def handler(request, count):
                self.assertEqual(request.tools, [])
                self.assertIn("unlimited", request.system)
                return ModelResponse(tool_calls=[ToolCall(id="forbidden-write", name="write_file", arguments={
                    "path": "must-not-exist.txt", "content": "No", "expected_sha256": None,
                })])

            provider.handler = handler
            try:
                result = await runner.run("Attempt a final-turn write.")
                self.assertEqual(result.stop_reason.value, "max_turns")
                self.assertEqual(provider.calls, 1)
                self.assertEqual(result.usage.requests, 1)
                self.assertFalse((root / "must-not-exist.txt").exists())
                checkpoints = [e for e in events if e.type == "run.budget_checkpoint"]
                self.assertTrue(checkpoints)
                self.assertIsNone(checkpoints[0].data["remaining_model_requests"])
                json.dumps([e.data for e in events], allow_nan=False)
            finally:
                await runner.close()

    async def test_root_dispatch_beyond_forty_with_actual_writes(self):
        for limit in (None, 0):
            for surface in ("direct", "ptc"):
                with self.subTest(limit=limit, surface=surface), tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    # Forty-one intentional writes isolate the request quota from
                    # repeated-call and read-only guards. Other limits stay enabled.
                    config = make_config(root, agent={} if limit is None else {"max_model_requests": limit},
                                         script={"tool_surface": surface})
                    runner = await build_runner(root, config=config, interactive=False)
                    provider = runner.providers[0].provider
                    assert isinstance(provider, MockProvider)
                    events = []
                    runner.events.subscribe(events.append)

                    def handler(request, count, surface=surface):
                        if count > 41:
                            return response(text="Created 41 evidence files.")
                        self.assertTrue(request.tools)
                        arguments = {"path": f"evidence-{count}.txt", "content": f"Evidence {count}", "expected_sha256": None}
                        return response(calls=[ToolCall(
                            id=f"write-{count}", name="script" if surface == "ptc" else "write_file",
                            arguments={"source": "text(await tools.write_file(" + json.dumps(arguments) + "));", "background": False} if surface == "ptc" else arguments,
                        )])

                    provider.handler = handler
                    try:
                        result = await runner.run("Create each evidence file in a separate step.")
                        self.assertEqual(result.stop_reason.value, "end_turn", result.error)
                        self.assertEqual(provider.calls, 42)
                        self.assertEqual(result.usage.requests, 42)
                        self.assertAlmostEqual(result.usage.cost_usd, 0.42)
                        for count in range(1, 42):
                            self.assertEqual((root / f"evidence-{count}.txt").read_text(), f"Evidence {count}")
                        validate_tool_call_order(runner.sessions.messages(result.session_id))
                        self.assertEqual(result.usage.to_dict(), runner.sessions.usage(result.session_id).to_dict())
                        json.dumps([e.data for e in events], allow_nan=False)
                    finally:
                        await runner.close()

    async def test_unlimited_nested_delegation_waiting_and_retry(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "evidence.txt").write_text("Verified source")
            config = make_config(root, agent={"swarm_max_concurrent": 1})
            # Isolate runtime admission from the separately tested config loader.
            config.agent.max_model_requests = 0
            runner = await build_runner(root, config=config, interactive=False)
            provider = runner.providers[0].provider
            assert isinstance(provider, MockProvider)
            provider.config.max_retries = 1
            provider.config.initial_backoff_seconds = 0
            provider.config.max_backoff_seconds = 0
            waiting = asyncio.Event()
            first = True
            original = provider.complete
            events = []

            def observe(event):
                events.append(event)
                if event.type == "delegate.state" and event.data["task"] == "right" and event.data["state"] == "waiting":
                    waiting.set()

            runner.events.subscribe(observe)

            async def controlled(request):
                nonlocal first
                if request.metadata.get("delegated") and first:
                    first = False
                    await asyncio.wait_for(waiting.wait(), 3)
                return await original(request)

            failed = False

            def handler(request, count):
                nonlocal failed
                depth = request.metadata.get("delegation_depth", 0)
                if request.messages[-1].role.value == "tool":
                    return response(text="Evidence: " + request.messages[-1].content)
                self.assertTrue(request.tools, request.system)
                if depth:
                    self.assertIn("Shared requests: unlimited", request.system)
                    self.assertNotIn("Shared unreserved requests: 0", request.system)
                if depth == 0:
                    return response(calls=[delegate("left", 4, f"left-{count}"), delegate("right", 4, f"right-{count}")])
                if depth == 1 and request.messages[0].content == "left":
                    if not failed:
                        failed = True
                        raise ProviderUnavailableError("Billed retry", retryable=True, usage=Usage(requests=1, cost_usd=0.02))
                    return response(calls=[delegate("grandchild", 3, f"grandchild-{count}")])
                return response(calls=[ToolCall(id=f"read-{count}", name="read_file", arguments={"path": "evidence.txt", "start_line": 1, "end_line": 10, "max_chars": None})])

            provider.handler = handler
            budget = Budget.start(config.agent)
            try:
                with patch.object(provider, "complete", side_effect=controlled), patch.object(Budget, "start", return_value=budget):
                    result = await asyncio.wait_for(runner.run("Investigate both branches"), 10)
                self.assertEqual(result.stop_reason.value, "end_turn", result.error)
                self.assertIn("Verified source", result.text)
                self.assertEqual(provider.calls, 9)
                self.assertEqual(budget.model_requests, 9)
                self.assertEqual(result.usage.requests, 9)
                self.assertAlmostEqual(result.usage.cost_usd, 0.1)
                self.assertFalse(budget._request_closures)
                self.assertEqual(budget.pending_cost_usd, 0)
                self.assertTrue(waiting.is_set())
                validate_tool_call_order(runner.sessions.messages(result.session_id))
                self.assertEqual(result.usage.to_dict(), runner.sessions.usage(result.session_id).to_dict())
                self.assertTrue(all(e.data["limits"]["active_requests"] <= 1 for e in events if e.type.startswith("delegate.")))
            finally:
                await runner.close()

    async def test_unlimited_judge_still_allows_denies_and_settles(self):
        for decision in ("allow", "deny"):
            with self.subTest(decision=decision):
                async with harness(answers=[answer(decision)]) as (root, runner, provider, humans):
                    runner.config.agent.max_model_requests = 0
                    result = await runner.run("Create sentinel.")
                    self.assertEqual((root / "sentinel").exists(), decision == "allow")
                    if decision == "allow":
                        self.assertEqual((root / "sentinel").read_text(), "allowed")
                    self.assertEqual(len(provider.judgments), 1)
                    self.assertEqual(result.usage.requests, 3)
                    self.assertEqual(receipts(runner)[0]["decision"], decision)
                    self.assertEqual(result.usage.to_dict(), runner.sessions.usage(result.session_id).to_dict())
                    self.assertEqual(humans, [])
