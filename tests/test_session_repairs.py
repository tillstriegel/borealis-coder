"""Synthetic regressions for restricted delegation and bounded closure (no live model)."""
from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from borealis_coder.agent import build_runner, validate_tool_call_order
from borealis_coder.agent.budget import Budget
from borealis_coder.agent.runner import ProviderRoute
from borealis_coder.config import ProviderConfig
from borealis_coder.errors import BudgetExceeded, ProviderError, ProviderUnavailableError
from borealis_coder.models import ModelResponse, ProviderRequest, Role, ToolCall, Usage
from borealis_coder.providers.http import HttpResponse, SSEEvent
from borealis_coder.providers.mock import MockProvider
from borealis_coder.providers.openai import OpenAICompatibleProvider
from borealis_coder.providers.openrouter import OpenRouterProvider
from borealis_coder.safety.redaction import Redactor
from tests.helpers import make_config
from tests.test_compaction_v2 import _echo_evidence, _tool_cycle
from tests.test_swarm import delegate, response


def read_call():
    return ToolCall(id="read", name="read_file", arguments={
        "path": "evidence.txt", "start_line": None, "end_line": None, "max_chars": None,
    })


class SessionRepairTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "evidence.txt").write_text("Synthetic source fact: alpha.\n")
        self.runner = await build_runner(self.root, config=make_config(self.root), interactive=False)
        self.addAsyncCleanup(self.runner.close)
        provider = self.runner.providers[0].provider
        assert isinstance(provider, MockProvider)
        self.provider = provider
        self.events = []
        self.runner.events.subscribe(self.events.append)

    def settled(self, result):
        rows = self.runner.sessions.tool_calls(result.session_id)
        self.assertTrue(rows)
        self.assertNotIn("running", {r["status"] for r in rows})
        validate_tool_call_order(self.runner.sessions.messages(result.session_id))
        stored = self.runner.sessions.usage(result.session_id).to_dict()
        actual = result.usage.to_dict()
        # Native per-attempt records live in durable events, not the SQL totals.
        stored.pop("attempt_usage")
        actual.pop("attempt_usage")
        self.assertEqual(stored, actual)
        starts = [e.data["delegation_id"] for e in self.events if e.type == "delegate.started"]
        ends = [e.data["delegation_id"] for e in self.events if e.type in {
            "delegate.completed", "delegate.failed", "delegate.cancelled"}]
        self.assertCountEqual(starts, ends)
        self.assertEqual(result.changed_files, [])
        return rows

    async def test_restricted_prompt_discovery_and_real_read_agree(self):
        child_requests = []

        def handler(request, call):
            if not request.metadata.get("delegated"):
                return response(calls=[delegate(turns=4)]) if call == 1 else response(text="Combined findings.")
            child_requests.append(request)
            n = len(child_requests)
            if n == 1:
                return response(calls=[ToolCall(id="catalog", name="discover_tools", arguments={"names": [], "offset": 0})])
            if n == 2:
                catalog = json.loads(request.messages[-1].content)
                names = {entry["name"] for entry in catalog["tools"]}
                self.assertNotIn("script", names)
                self.assertNotIn("fetch_url", names)
                self.assertIn("read_file", names)
                return response(calls=[ToolCall(id="exact", name="discover_tools", arguments={"names": ["script", "fetch_url", "read_file"], "offset": 0})])
            if n == 3:
                entries = json.loads(request.messages[-1].content)["tools"]
                self.assertEqual([e.get("unavailable", False) for e in entries], [True, True, False])
                return response(calls=[read_call()])
            self.assertIn("alpha", request.messages[-1].content)
            return response(text="evidence.txt:1 establishes alpha.")

        self.provider.handler = handler
        result = await self.runner.run("Inspect via a child")
        self.assertEqual(result.stop_reason.value, "end_turn")
        self.assertTrue(child_requests)
        for req in child_requests:
            self.assertNotIn("Use script as the normal", req.system)
            self.assertNotIn("inside Script", req.system)
            self.assertIn("direct", req.system.lower())
            self.assertFalse(any("inside script" in tool["description"].lower() for tool in req.tools))
        self.assertEqual([e.type for e in self.events if e.type == "delegate.completed"], ["delegate.completed"])
        self.settled(result)

    async def test_literal_null_cursor_stays_invalid_then_json_null_reads_evidence(self):
        children = 0

        def handler(request, call):
            nonlocal children
            if not request.metadata.get("delegated"):
                return response(calls=[delegate(turns=3)]) if call == 1 else response(text="Qualified grep findings combined.")
            children += 1
            if children == 2:
                self.assertTrue(request.messages[-1].is_error)
                self.assertIn("stale or invalid", request.messages[-1].content)
            if children < 3:
                return response(calls=[ToolCall(id="grep" + str(children), name="grep", arguments={
                    "pattern": "alpha", "path": ".", "glob": "*.txt", "regex": False,
                    "case_sensitive": True, "context_lines": 0, "max_results": 10,
                    "cursor": "null" if children == 1 else None,
                })])
            self.assertFalse(request.messages[-1].is_error)
            self.assertIn("evidence.txt:1:Synthetic source fact: alpha.", request.messages[-1].content)
            return response(text="evidence.txt:1 alpha matched after restarting with JSON null.")

        self.provider.handler = handler
        result = await self.runner.run("Check strict cursor handling")
        self.assertEqual(result.stop_reason.value, "end_turn", result.error)
        rows = [r for r in self.settled(result) if r["tool_name"] == "grep"]
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["status"] for r in rows], ["error", "completed"])
        self.assertEqual([r["arguments"]["cursor"] for r in rows], ["null", None])

    async def test_protocol_or_empty_is_repaired_without_becoming_findings(self):
        for bad in ("\n <\uff5cDSML\uff5c calls><invoke name='read_file'/>", " <TOOL_CALLS>payload", "<function_call>payload", "  "):
            with self.subTest(bad=bad):
                self.events.clear()
                child_calls = 0

                def handler(request, call, bad=bad):
                    nonlocal child_calls
                    if not request.metadata.get("delegated"):
                        return response(calls=[delegate(turns=3)]) if request.messages[-1].role == Role.USER else response(text="Combined qualified findings.")
                    child_calls += 1
                    if child_calls == 1:
                        return response(calls=[read_call()])
                    if child_calls == 2:
                        return response(text=bad)
                    self.assertEqual(request.tools, [])
                    self.assertNotIn(bad.strip() or "not-present", " ".join(m.content for m in request.messages))
                    return response(text="evidence.txt:1 establishes alpha.")

                self.provider.handler = handler
                result = await self.runner.run("Inspect with bounded answer repair")
                self.assertEqual(child_calls, 3)
                terminal = next(e for e in self.events if e.type == "delegate.completed")
                self.assertIn("alpha", terminal.data["result"])
                self.assertNotIn("DSML", terminal.data["result"])
                self.settled(result)

    async def test_failed_synthesis_keeps_observations_not_protocol(self):
        for terminal in ("dsml", "error", "incomplete", "tools", "blank", "prose"):
            with self.subTest(terminal=terminal):
                self.events.clear()
                children = 0

                def handler(request, call, terminal=terminal):
                    nonlocal children
                    if not request.metadata.get("delegated"):
                        if request.messages[-1].role == Role.USER:
                            return response(calls=[delegate(turns=2)])
                        receipt = request.messages[-1]
                        self.assertEqual(receipt.is_error, terminal != "prose")
                        self.assertIn("alpha", receipt.content)
                        self.assertNotIn("<\uff5cDSML", receipt.content)
                        return response(text="Qualified partial result: alpha observed; child status retained.")
                    children += 1
                    if children == 1:
                        return response(calls=[read_call()])
                    if terminal == "error":
                        raise ProviderError("Synthetic failure", usage=Usage(requests=1))
                    if terminal == "tools":
                        return response(text="not a final answer", calls=[read_call()])
                    if terminal == "incomplete":
                        return ModelResponse(text="partial private candidate", stop_reason="length", usage=Usage(requests=1))
                    if terminal == "blank":
                        return response(text=" ")
                    if terminal == "prose":
                        return response(text="evidence.txt:1 alpha. The literal `<tool_call>` is a protocol example.")
                    return response(text="<\uff5cDSML\uff5c calls><invoke/>")

                self.provider.handler = handler
                result = await self.runner.run("Retain useful observations")
                self.assertEqual(children, 2)
                self.assertEqual(result.stop_reason.value, "end_turn", result.error)
                self.settled(result)

    async def test_two_children_grandchild_preserve_root_closure_at_small_budget(self):
        for concurrency in (1, 2):
            with self.subTest(concurrency=concurrency):
                self.events.clear()
                self.runner.config.agent.max_model_requests = 10
                self.runner.config.agent.swarm_max_concurrent = concurrency
                seen = []

                def handler(request, call, seen=seen):
                    seen.append(request)
                    depth = request.metadata.get("delegation_depth", 0)
                    if not depth:
                        if request.messages[-1].role == Role.USER:
                            return response(calls=[delegate("left", 6, "left"), delegate("right", 6, "right")])
                        self.assertTrue(any("alpha" in m.content for m in request.messages if m.role == Role.TOOL))
                        return response(text="Combined qualified result: alpha observed; unfinished checks are incomplete.")
                    if not request.tools:
                        return response(text="Partial findings: alpha observed; request bounds prevent further checks.")
                    if len(request.messages) == 1:
                        return response(calls=[delegate("focused", 3)]) if depth == 1 and request.messages[0].content == "left" else response(calls=[read_call()])
                    if depth == 2:
                        return response(text="<\uff5cDSML\uff5c calls><invoke/>")
                    return response(calls=[ToolCall(id="more" + str(len(request.messages)), name="list_directory", arguments={"path": ".", "depth": 1, "include_hidden": False, "max_entries": len(request.messages)})])

                self.provider.handler = handler
                result = await asyncio.wait_for(self.runner.run("Two branches, one focused child, then combine"), 5)
                self.assertEqual(result.stop_reason.value, "end_turn")
                self.assertIn("alpha", result.text)
                self.assertLessEqual(len(seen), 10)
                self.assertEqual(result.usage.requests, len(seen))
                self.assertTrue(any(r.metadata.get("delegation_depth") == 2 for r in seen))
                self.assertFalse(any("DSML" in e.data.get("result", "") for e in self.events if e.type == "delegate.completed"))
                self.settled(result)

    async def test_global_budget_stop_still_delivers_attributable_child_evidence(self):
        self.runner.config.agent.max_model_requests = 4

        def handler(request, call):
            if call == 1:
                return response(calls=[delegate(turns=2)])
            if request.metadata.get("delegated"):
                return response(calls=[read_call()]) if request.tools else response(text="evidence.txt:1 alpha confirmed.")
            return response(text="<tool_call>not findings")

        self.provider.handler = handler
        result = await self.runner.run("Preserve findings when parent fails closure")
        self.assertEqual(result.stop_reason.value, "budget")
        self.assertIn("alpha", result.text)
        self.assertIn("evidence.txt", result.text)
        self.assertNotIn("<tool_call>", result.text)
        self.assertEqual(result.usage.requests, 4)
        self.settled(result)

    async def test_tiny_request_allowances_refuse_or_close_without_starving_root(self):
        for ceiling in (2, 3, 4):
            with self.subTest(ceiling=ceiling):
                self.events.clear()
                self.runner.config.agent.max_model_requests = ceiling
                seen = []

                def handler(request, call, seen=seen):
                    seen.append(request)
                    if not request.metadata.get("delegated"):
                        return response(calls=[delegate(turns=6)]) if request.messages[-1].role == Role.USER else response(text="Qualified result: child receipt retained; unresolved work remains.")
                    if request.tools:
                        return response(calls=[read_call()])
                    facts = [m.content for m in request.messages if m.role == Role.TOOL]
                    return response(text="Partial: alpha observed." if any("alpha" in fact for fact in facts) else "Incomplete: no observations established.")

                self.provider.handler = handler
                result = await self.runner.run("Inspect within a tiny allowance")
                self.assertEqual(result.stop_reason.value, "end_turn", result.error)
                self.assertEqual(len(seen), ceiling)
                self.assertEqual(result.usage.requests, ceiling)
                self.assertEqual(seen[-1].tools, [])
                self.assertEqual(len([e for e in self.events if e.type == "delegate.started"]), int(ceiling > 2))
                self.settled(result)

    async def test_production_tree_with_protocol_and_empty_openrouter_wire(self):
        provider = OpenRouterProvider(ProviderConfig(
            type="openrouter", base_url="https://synthetic.test", max_retries=0,
        ))
        self.addAsyncCleanup(provider.close)
        self.runner.providers = [ProviderRoute("openrouter", "synthetic-model", provider)]
        self.runner.config.agent.reasoning_effort = "high"
        self.runner.config.agent.max_model_requests = 13
        self.runner.config.agent.swarm_max_concurrent = 1
        payloads = []

        def wire(payload):
            payloads.append(payload)
            messages = payload["messages"]
            task = messages[1]["content"]
            text = ""
            calls = []
            if task == "Combine two branches":
                if messages[-1]["role"] == "user":
                    calls = [delegate("left", 4, "left"), delegate("right", 3, "right")]
                else:
                    receipts = [m["content"] for m in messages if m["role"] == "tool"]
                    self.assertEqual(len(receipts), 2)
                    self.assertTrue(all("alpha" in receipt for receipt in receipts))
                    self.assertTrue(any("Incomplete" in receipt for receipt in receipts))
                    text = "Combined qualified result: evidence.txt:1 alpha; right provider failed after its read."
            else:
                self.assertNotIn("Use script as the normal", messages[0]["content"])
                self.assertNotIn("script", {t["function"]["name"] for t in payload["tools"]})
                if len(messages) == 2:
                    calls = [ToolCall(id="catalog", name="discover_tools", arguments={"names": ["script", "read_file"], "offset": 0})] if task == "left" else [read_call()]
                elif task == "left":
                    if messages[-1].get("tool_call_id") == "catalog":
                        catalog = json.loads(messages[-1]["content"])["tools"]
                        self.assertTrue(catalog[0]["unavailable"])
                        self.assertEqual(catalog[1]["name"], "read_file")
                        calls = [delegate("focused", 3)]
                    else:
                        self.assertIn("alpha", messages[-1]["content"])
                        text = "Left findings: evidence.txt:1 alpha; focused read confirmed."
                elif task == "focused":
                    self.assertTrue(any("alpha" in m.get("content", "") for m in messages if m["role"] == "tool"))
                    text = "<\uff5cDSML\uff5c calls><invoke name='read_file'/>" if payload["tools"] else "Focused evidence.txt:1 alpha confirmed."
                else:
                    self.assertEqual(task, "right")
                    # Default-on upstream produces only private reasoning twice.
                    # The second attempt must explicitly disable it, then fail
                    # truthfully rather than accepting a fake conclusion.
                    if payloads[-2].get("messages", [])[1].get("content") == "right":
                        self.assertEqual(payload["reasoning"], {"effort": "none"})
            message: dict[str, Any] = {"content": text}
            if calls:
                message["tool_calls"] = [{"index": i, "id": call.id, "type": "function", "function": {
                    "name": call.name, "arguments": json.dumps(call.arguments),
                }} for i, call in enumerate(calls)]
            if task == "right" and not calls:
                message["reasoning_content"] = "PRIVATE_REASONING_MUST_NOT_ESCAPE"
            return {"model": "synthetic-model", "choices": [{"message": message, "delta": message,
                    "finish_reason": "tool_calls" if calls else "stop"}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "cost": 0.001}}

        async def post(*args, **kwargs):
            return HttpResponse(200, {}, wire(kwargs["payload"]), b"{}")

        async def sse(*args, **kwargs):
            yield SSEEvent("message", json.dumps(wire(kwargs["payload"])))
            yield SSEEvent("message", "[DONE]")

        provider.http.post_json = AsyncMock(side_effect=post)
        provider.http.stream_sse = sse
        result = await asyncio.wait_for(self.runner.run("Combine two branches"), 5)
        self.assertEqual(result.stop_reason.value, "end_turn", result.error)
        self.assertIn("alpha", result.text)
        self.assertEqual(len(payloads), 11)
        self.assertEqual(result.usage.requests, 11)
        self.assertAlmostEqual(result.usage.cost_usd, 0.011)
        native = [event for _, event in self.runner.sessions.events(result.session_id)
                  if event.type == "model.attempt_usage"]
        self.assertEqual(len(native), 11)
        rows = self.settled(result)
        reads = [r for r in rows if r["tool_name"] == "read_file"]
        self.assertEqual(len(reads), 2)
        self.assertTrue(all(r["status"] == "completed" for r in reads))
        self.assertEqual(len([e for e in self.events if e.type == "delegate.failed"]), 1)
        self.assertEqual(len([e for e in self.events if e.type == "delegate.completed"]), 2)
        public = json.dumps([e.to_dict() for e in self.events])
        self.assertNotIn("PRIVATE_REASONING_MUST_NOT_ESCAPE", public)
        self.assertFalse(any("DSML" in e.data.get("result", "") for e in self.events if e.type.startswith("delegate.")))

    async def test_child_retry_cannot_spend_parent_or_own_closure(self):
        self.runner.config.agent.max_model_requests = 5
        self.provider.config.max_retries = 3
        self.provider.config.initial_backoff_seconds = 0
        self.provider.config.max_backoff_seconds = 0
        observed = []

        def handler(request, call):
            observed.append(request)
            if not request.metadata.get("delegated"):
                return response(calls=[delegate(turns=4)]) if call == 1 else response(text="Root combines qualified child result.")
            if request.tools:
                raise ProviderUnavailableError("Synthetic billed retry", retryable=True,
                                               usage=Usage(requests=1, cost_usd=0.02, cost_status="known"))
            return response(text="Incomplete: provider investigation failed; no facts established.")

        self.provider.handler = handler
        budget = Budget.start(self.runner.config.agent)
        with patch.object(Budget, "start", return_value=budget):
            result = await self.runner.run("Retry within shared bounds")
        self.assertEqual(result.stop_reason.value, "end_turn", result.error)
        self.assertEqual(len(observed), 5)
        self.assertEqual(result.usage.requests, 5)
        self.assertAlmostEqual(result.usage.cost_usd, 0.07)
        self.assertEqual(budget.model_requests, 5)
        self.assertFalse(budget._request_closures)
        self.assertEqual(budget.pending_cost_usd, 0)
        self.assertFalse(budget._held_usage)
        self.settled(result)

    async def test_waiting_sibling_closes_after_a_retry_spends_investigative_capacity(self):
        self.runner.config.agent.max_model_requests = 6
        self.runner.config.agent.swarm_max_concurrent = 1
        self.provider.config.max_retries = 1
        self.provider.config.initial_backoff_seconds = 0
        self.provider.config.max_backoff_seconds = 0
        first_entered = threading.Event()
        second_waiting = asyncio.Event()
        left_calls = 0
        right_requests = []
        original = self.provider.complete
        original_prompt = self.runner.context_builder.system_prompt

        def prepare(*, query="", **kwargs):
            if query == "right":
                self.assertTrue(first_entered.wait(3))
            return original_prompt(query=query, **kwargs)

        def observe(event):
            if event.type == "delegate.state" and event.data["task"] == "right" and event.data["state"] == "waiting":
                second_waiting.set()

        self.runner.events.subscribe(observe)

        async def controlled(request):
            nonlocal left_calls
            if request.metadata.get("delegated") and request.messages[0].content == "left":
                left_calls += 1
                if left_calls == 1:
                    first_entered.set()
                    await asyncio.wait_for(second_waiting.wait(), 3)
            return await original(request)

        def handler(request, call):
            if not request.metadata.get("delegated"):
                return response(calls=[delegate("left", 4, "left"), delegate("right", 4, "right")]) if call == 1 else response(text="Combined partial: alpha observed; right had no investigative capacity.")
            if request.messages[0].content == "right":
                right_requests.append(request)
                self.assertEqual(request.tools, [])
                self.assertIn("Shared unreserved requests: 0", request.system)
                return response(text="Incomplete: no investigation capacity remained.")
            if left_calls == 1:
                raise ProviderUnavailableError("Synthetic billed attempt", retryable=True, usage=Usage(requests=1))
            if left_calls == 2:
                return response(calls=[read_call()])
            self.assertIn("alpha", request.messages[-1].content)
            return response(text="evidence.txt:1 alpha observed.")

        self.provider.handler = handler
        budget = Budget.start(self.runner.config.agent)
        with patch.object(Budget, "start", return_value=budget), patch.object(self.provider, "complete", side_effect=controlled), patch.object(self.runner.context_builder, "system_prompt", side_effect=prepare):
            result = await asyncio.wait_for(self.runner.run("Race sibling closure against recovery"), 5)
        self.assertEqual(result.stop_reason.value, "end_turn", result.error)
        self.assertEqual(len(right_requests), 1)
        self.assertEqual(result.usage.requests, 6)
        self.assertEqual(budget.model_requests, 6)
        self.assertFalse(budget._request_closures)
        self.assertEqual(budget.pending_cost_usd, 0)
        self.settled(result)

    async def test_summarizer_retries_are_also_inside_the_hard_request_ceiling(self):
        self.runner.config.agent.deterministic_compaction = False
        self.runner.config.agent.max_model_requests = 2
        self.provider.config.max_retries = 3
        self.provider.config.initial_backoff_seconds = 0
        self.provider.config.max_backoff_seconds = 0
        budget = Budget.start(self.runner.config.agent)

        def handler(request, call):
            raise ProviderUnavailableError("Synthetic billed summary failure", retryable=True,
                                           usage=Usage(requests=1, cost_usd=0.01, cost_status="known"))

        async def settle(usage):
            budget.add_usage(usage)

        self.provider.handler = handler
        summarize = self.runner._summarizer(settle, asyncio.Event(), before_model_request=budget.before_model_request)
        assert summarize is not None
        with self.assertRaises(BudgetExceeded):
            await summarize("Synthetic transcript requiring a summary")
        self.assertEqual(self.provider.calls, 2)
        self.assertEqual(budget.model_requests, 2)
        assert budget.usage is not None
        self.assertEqual(budget.usage.requests, 2)
        self.assertAlmostEqual(budget.usage.cost_usd, 0.02)
        self.assertEqual(budget.pending_cost_usd, 0)
        self.assertFalse(budget._held_usage)

    async def test_interrupted_observations_are_redacted_and_durable(self):
        for interruption in ("provider-cancel", "batch-cancel", "time", "cost", "healthy"):
            with self.subTest(interruption=interruption):
                self.events.clear()
                secret = 'SYNTHETIC_PRIVATE_FILENAME"suffix' if interruption == "provider-cancel" else "SYNTHETIC_PRIVATE_FILENAME"
                path = secret + ".txt"
                (self.root / path).write_text("Synthetic source fact: alpha.\n")
                self.runner.events.redactor = Redactor([secret])
                read_done = asyncio.Event()
                sibling_entered = asyncio.Event()
                release = asyncio.Event()
                children = 0
                original_execute = self.runner._execute_one
                original_complete = self.provider.complete

                def handler(request, call, path=path, interruption=interruption):
                    nonlocal children
                    if not request.metadata.get("delegated"):
                        return response(calls=[delegate(turns=3)]) if request.messages[-1].role == Role.USER else response(text="Combined alpha findings.")
                    children += 1
                    if children == 1:
                        read = read_call()
                        read.arguments["path"] = path
                        calls = [read]
                        if interruption not in {"provider-cancel", "healthy"}:
                            calls.append(ToolCall(id="sibling", name="list_directory", arguments={
                                "path": ".", "depth": 1, "include_hidden": False, "max_entries": 10,
                            }))
                        return response(calls=calls)
                    return response(text="Qualified alpha finding.")

                async def execute(call, context, cancel, *, sibling_entered=sibling_entered,
                                  read_done=read_done, interruption=interruption, release=release,
                                  original_execute=original_execute, **kwargs):
                    if call.name == "list_directory":
                        sibling_entered.set()
                        await asyncio.wait_for(read_done.wait(), 3)
                        if interruption in {"time", "cost"}:
                            raise BudgetExceeded(interruption, "Synthetic fatal " + interruption)
                        await release.wait()
                    result = await original_execute(call, context, cancel, **kwargs)
                    if call.name == "read_file":
                        read_done.set()  # Canonical read and nested result have completed.
                    return result

                async def complete(request, read_done=read_done, interruption=interruption,
                                   sibling_entered=sibling_entered, release=release,
                                   original_complete=original_complete):
                    if request.metadata.get("delegated") and read_done.is_set() and interruption == "provider-cancel":
                        sibling_entered.set()
                        await release.wait()
                    return await original_complete(request)

                self.provider.handler = handler
                with patch.object(self.runner, "_execute_one", side_effect=execute), patch.object(self.provider, "complete", side_effect=complete):
                    task = asyncio.create_task(self.runner.run("Retain interrupted observations"))
                    try:
                        await asyncio.wait_for(read_done.wait(), 3)
                        if interruption in {"provider-cancel", "batch-cancel"}:
                            await asyncio.wait_for(sibling_entered.wait(), 3)
                            task.cancel()
                        result = await asyncio.wait_for(task, 3)
                    finally:
                        release.set()
                        if not task.done():
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                rows = self.settled(result)
                read = next(row for row in rows if row["tool_name"] == "read_file")
                self.assertEqual(read["status"], "completed")
                self.assertEqual(read["metadata"]["path"], path)
                durable = self.runner.sessions.messages(result.session_id)[-1]
                self.assertEqual(durable.role, Role.ASSISTANT)
                self.assertEqual(durable.content, result.text)
                self.assertNotIn(secret, result.text)
                self.assertNotIn("SYNTHETIC_PRIVATE_FILENAME", result.text)
                if interruption == "healthy":
                    self.assertEqual(result.stop_reason.value, "end_turn")
                else:
                    self.assertEqual(result.stop_reason.value, "cancelled" if "cancel" in interruption else "budget")
                    self.assertIn("alpha", result.text)
                    self.assertIn("untrusted data, not a model conclusion", result.text)
                    self.assertIn(read["tool_call_id"], result.text)
                    self.assertIn("[REDACTED]", result.text)

    async def test_compaction_preserves_swarm_closure_through_helper_retry(self):
        await self._check_compaction_helper_retry(continued=False)

    async def test_compaction_rearms_root_closure_after_another_root_tool_turn(self):
        await self._check_compaction_helper_retry(continued=True)

    async def _check_compaction_helper_retry(self, *, continued):
        self.runner.config.agent.max_model_requests = 5 + int(continued)
        self.runner.config.context.compact_tool_output_tokens = 100_000
        session = self.runner.sessions.create_session(workspace=self.root, provider="mock", model="deterministic")
        for index in range(14):
            for message in _tool_cycle(f"prior-{index}", content=f"history-{index} " * 100):
                self.runner.sessions.append_message(session.id, message)
        self.provider.config.max_retries = 2
        self.provider.config.initial_backoff_seconds = 0
        self.provider.config.max_backoff_seconds = 0
        summaries = []
        children = 0

        def handler(request, call):
            nonlocal children
            if request.metadata.get("purpose") == "compaction_summary":
                summaries.append(request)
                raise ProviderUnavailableError("Synthetic billed helper retry", retryable=True,
                                               usage=Usage(requests=1, cost_usd=0.02, cost_status="known"))
            if request.metadata.get("delegated"):
                children += 1
                if children == 1:
                    return response(calls=[read_call()])
                if not continued:
                    self.runner.config.agent.deterministic_compaction = False
                    self.runner.config.context.compact_tool_output_tokens = 10
                return response(text="evidence.txt:1 alpha confirmed.")
            if call == 1:
                return response(calls=[delegate(turns=2)])
            if continued and call == 4:
                self.runner.config.agent.deterministic_compaction = False
                self.runner.config.context.compact_tool_output_tokens = 10
                return response(calls=[read_call()])
            self.assertEqual(request.tools, [])
            self.assertIn("alpha", request.system + str([m.content for m in request.messages]))
            return response(text="Qualified root result: alpha confirmed; helper summary unavailable.")

        self.provider.handler = handler
        budget = Budget.start(self.runner.config.agent)
        with patch.object(Budget, "start", return_value=budget):
            result = await self.runner.run("Inspect and combine", session_id=session.id)
        self.assertEqual(result.stop_reason.value, "end_turn", result.error)
        self.assertIn("alpha", result.text)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(budget.model_requests, 5 + int(continued))
        self.assertEqual(result.usage.requests, 5 + int(continued))
        self.assertAlmostEqual(result.usage.cost_usd, 0.06 + 0.01 * int(continued))
        self.assertFalse(budget._request_closures)
        self.assertEqual(budget.pending_cost_usd, 0)
        self.assertFalse(budget._held_usage)
        artifact = self.runner.sessions.compaction_artifacts(result.session_id)[-1]
        self.assertEqual(artifact.metadata["fallback_reason"], "summarizer_request_capacity")
        self.settled(result)

    async def test_root_final_dispatch_policy_reaches_wire_and_mutation_guard(self):
        for transition in ("retry", "recovery", "compaction"):
            for violate in (False, True):
                with self.subTest(transition=transition, violate=violate):
                    self.events.clear()
                    self.runner.config.agent.max_model_requests = 5
                    self.runner.config.agent.deterministic_compaction = True
                    self.runner.config.cache.response_cache_enabled = True
                    self.runner.config.context.compact_tool_output_tokens = 100_000
                    provider = OpenRouterProvider(ProviderConfig(type="openrouter", base_url="https://synthetic.test",
                        max_retries=1, initial_backoff_seconds=0, max_backoff_seconds=0))
                    self.addAsyncCleanup(provider.close)
                    self.runner.providers = [ProviderRoute("openrouter", "synthetic", provider)]
                    session = self.runner.sessions.create_session(workspace=self.root, provider="openrouter", model="synthetic")
                    if transition == "compaction":
                        for index in range(14):
                            for message in _tool_cycle(f"prior-{index}", content=f"history-{index} " * 100):
                                self.runner.sessions.append_message(session.id, message)
                    payloads = []
                    roots = []
                    children = 0
                    summaries = 0
                    target = self.root / "FORBIDDEN.txt"
                    target.unlink(missing_ok=True)

                    def wire(payload, payloads=payloads, roots=roots, transition=transition, violate=violate):
                        nonlocal children, summaries
                        payloads.append(payload)
                        system = payload["messages"][0]["content"]
                        text = ""
                        calls = []
                        if payload.get("response_format"):
                            summaries += 1
                            text = _echo_evidence(payload["messages"][-1]["content"])
                        elif "# Delegated role" in system:
                            children += 1
                            if children == 1:
                                calls = [read_call()]
                            else:
                                if transition == "compaction":
                                    self.runner.config.agent.deterministic_compaction = False
                                    self.runner.config.context.compact_tool_output_tokens = 10
                                text = "evidence.txt:1 alpha confirmed."
                        else:
                            roots.append(payload)
                            if len(roots) == 1:
                                calls = [delegate(turns=2)]
                            elif len(roots) == 2 and transition == "retry":
                                raise ProviderUnavailableError("Synthetic billed root retry", retryable=True,
                                    usage=Usage(requests=1, cost_usd=0.001, cost_status="known"))
                            elif len(roots) == 2 and transition == "recovery":
                                pass  # Empty, billed output invokes real adapter recovery.
                            else:
                                text = "Qualified root result: evidence.txt:1 alpha confirmed."
                                if violate:
                                    calls = [ToolCall(id="forbidden", name="write_file", arguments={
                                        "path": "FORBIDDEN.txt", "content": "must not execute", "expected_sha256": None,
                                    })]
                        message: dict[str, Any] = {"content": text}
                        if calls:
                            message["tool_calls"] = [{"index": i, "id": call.id, "type": "function", "function": {
                                "name": call.name, "arguments": json.dumps(call.arguments),
                            }} for i, call in enumerate(calls)]
                        return {"model": "synthetic", "choices": [{"message": message, "delta": message,
                            "finish_reason": "tool_calls" if calls else "stop"}],
                            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "cost": 0.001}}

                    async def post(*args, **kwargs):
                        return HttpResponse(200, {}, wire(kwargs["payload"]), b"{}")

                    async def sse(*args, **kwargs):
                        yield SSEEvent("message", json.dumps(wire(kwargs["payload"])))
                        yield SSEEvent("message", "[DONE]")

                    provider.http.post_json = AsyncMock(side_effect=post)
                    provider.http.stream_sse = sse
                    budget = Budget.start(self.runner.config.agent)
                    with patch.object(Budget, "start", return_value=budget), patch.object(
                        self.runner.sessions, "put_cached_response",
                        wraps=self.runner.sessions.put_cached_response,
                    ) as cache_put:
                        result = await self.runner.run("Inspect and combine", session_id=session.id)
                    # Retry/recovery changed the keyed tool-capable request at dispatch.
                    # A directly prepared synthesis request still caches under its own key.
                    if transition == "compaction" and not violate:
                        cache_put.assert_called_once()
                        cached = self.runner.sessions.get_cached_response(cache_put.call_args.args[0])
                        self.assertIsNotNone(cached)
                        assert cached is not None
                        self.assertEqual(cached["response"]["text"], result.text)
                    else:
                        cache_put.assert_not_called()
                    self.assertFalse(target.exists(), "Last-request structured tool call mutated the workspace")
                    self.assertFalse(roots[-1].get("tools"), "Final physical dispatch advertised tools")
                    self.assertFalse(roots[-1].get("parallel_tool_calls", False))
                    self.assertEqual(summaries, int(transition == "compaction"))
                    self.assertEqual(len(payloads), 5)
                    self.assertEqual(budget.model_requests, 5)
                    self.assertEqual(result.usage.requests, 5)
                    self.assertAlmostEqual(result.usage.cost_usd, 0.005)
                    self.assertEqual(budget.pending_cost_usd, 0)
                    self.assertFalse(budget._held_usage)
                    self.assertFalse(budget._request_closures)
                    self.assertEqual(result.stop_reason.value, "budget" if violate else "end_turn", result.error)
                    self.assertIn("alpha", result.text)
                    rows = self.settled(result)
                    self.assertFalse(any(r["tool_name"] == "write_file" for r in rows))
                    self.assertEqual(self.runner.sessions.messages(result.session_id)[-1].content, result.text)

    async def test_cancellation_releases_closures_after_billed_child_attempt(self):
        self.runner.config.agent.max_model_requests = 8
        self.provider.config.max_retries = 1
        entered = asyncio.Event()
        release = asyncio.Event()
        original = self.provider.complete

        def handler(request, call):
            return response(calls=[delegate()]) if call == 1 else response(text="Child answer.")

        async def controlled(request):
            if request.metadata.get("delegated"):
                entered.set()
                await release.wait()
            return await original(request)

        self.provider.handler = handler
        budget = Budget.start(self.runner.config.agent)
        with patch.object(Budget, "start", return_value=budget), patch.object(self.provider, "complete", side_effect=controlled):
            task = asyncio.create_task(self.runner.run("Cancel a reserved child"))
            try:
                await asyncio.wait_for(entered.wait(), 3)
                task.cancel()
                result = await asyncio.wait_for(task, 3)
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(result.stop_reason.value, "cancelled")
        self.assertEqual(budget.model_requests, 2)
        self.assertEqual(result.usage.requests, 2)
        self.assertFalse(budget._request_closures)
        self.assertEqual(budget.pending_cost_usd, 0)
        self.assertFalse(budget._held_usage)
        self.settled(result)


class ReasoningWireRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_on_empty_recovery_and_budget_refusal_settle_once(self):
        for stream in (False, True):
            for refuse in (False, True):
                with self.subTest(stream=stream, refuse=refuse):
                    provider = OpenRouterProvider(ProviderConfig(type="openrouter", base_url="https://synthetic.test", max_retries=0))
                    self.addAsyncCleanup(provider.close)
                    payloads = []
                    observed = Usage()
                    request = ProviderRequest(model="synthetic", system="Inspect", messages=[], tools=[], reasoning_effort=None)

                    def wire(payload, payloads=payloads):
                        payloads.append(payload)
                        text = "" if len(payloads) == 1 else "Recovered findings."
                        return {"choices": [{"message": {"content": text}, "delta": {"content": text}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 2, "completion_tokens": 1}}

                    async def post(*args, wire=wire, **kwargs):
                        return HttpResponse(200, {}, wire(kwargs["payload"]), b"{}")

                    async def sse(*args, wire=wire, **kwargs):
                        yield SSEEvent("message", json.dumps(wire(kwargs["payload"])))
                        yield SSEEvent("message", "[DONE]")

                    def before_recovery(refuse=refuse):
                        if refuse:
                            raise BudgetExceeded("model_requests", "Synthetic hard request ceiling")

                    async def run(stream=stream, provider=provider, request=request):
                        if stream:
                            events = [e async for e in provider.stream(request)]
                            return events[-1].response
                        return await provider.complete(request)

                    provider.http.post_json = AsyncMock(side_effect=post)
                    provider.http.stream_sse = sse
                    def observe_usage(usage, observed=observed):
                        observed.add(usage)

                    with provider.observe_requests(observe_usage, None, before_recovery):
                        if refuse:
                            with self.assertRaises(BudgetExceeded):
                                await run()
                        else:
                            answer = await run()
                            assert answer is not None
                            self.assertEqual(answer.text, "Recovered findings.")
                            self.assertEqual(answer.usage.requests, 2)
                    self.assertNotIn("reasoning", payloads[0])
                    self.assertEqual(len(payloads), 1 if refuse else 2)
                    self.assertEqual(observed.requests, len(payloads))
                    if not refuse:
                        self.assertEqual(payloads[1]["reasoning"], {"effort": "none"})

    async def test_empty_disables_but_unsupported_omits_on_actual_wire(self):
        for provider_type in (OpenRouterProvider, OpenAICompatibleProvider):
            for stream in (False, True):
                for cause in ("empty", "unsupported", "healthy", "initial-none"):
                    with self.subTest(provider=provider_type.name, stream=stream, cause=cause):
                        provider = provider_type(ProviderConfig(
                            type=provider_type.name, base_url="https://synthetic.test", max_retries=0,
                            extra_body={"reasoning": {"enabled": True}} if provider_type is OpenRouterProvider and cause in {"empty", "unsupported"} else {},
                        ))
                        self.addAsyncCleanup(provider.close)
                        payloads = []
                        attempts = []
                        request = ProviderRequest(model="synthetic", system="Inspect", messages=[], tools=[], reasoning_effort=None if cause == "initial-none" else "high")

                        def wire(payload, payloads=payloads, cause=cause):
                            payloads.append(payload)
                            if len(payloads) == 1 and cause == "unsupported":
                                raise ProviderError("reasoning parameter is not supported", status_code=400)
                            text = "" if len(payloads) == 1 and cause == "empty" else "Useful findings."
                            return {"choices": [{"message": {"content": text}, "delta": {"content": text}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 2, "completion_tokens": 1}}

                        async def post(*args, **kwargs):
                            return HttpResponse(200, {}, wire(kwargs["payload"]), b"{}")

                        async def sse(*args, **kwargs):
                            yield SSEEvent("message", json.dumps(wire(kwargs["payload"])))
                            yield SSEEvent("message", "[DONE]")

                        provider.http.post_json = AsyncMock(side_effect=post)
                        provider.http.stream_sse = sse
                        with provider.observe_requests(None, None, before_recovery=lambda attempts=attempts: attempts.append("recovery")):
                            if stream:
                                events = [event async for event in provider.stream(request)]
                                answer = events[-1].response
                            else:
                                answer = await provider.complete(request)
                        assert answer is not None
                        self.assertEqual(answer.text, "Useful findings.")
                        count = 2 if cause in {"empty", "unsupported"} else 1
                        self.assertEqual(len(payloads), count)
                        self.assertEqual(len(attempts), count - 1)
                        self.assertEqual(answer.usage.requests, 2 if cause == "empty" else 1)
                        key = "reasoning" if provider_type is OpenRouterProvider else "reasoning_effort"
                        if cause == "empty" and provider_type is OpenRouterProvider:
                            self.assertEqual(payloads[1][key], {"effort": "none"})
                        elif cause in {"empty", "unsupported"}:
                            self.assertNotIn(key, payloads[1])
                        elif cause == "initial-none":
                            self.assertNotIn(key, payloads[0])
                        else:
                            self.assertIn(key, payloads[0])
