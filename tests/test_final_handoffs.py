from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from borealis_coder.agent import build_runner, validate_tool_call_order
from borealis_coder.agent.budget import Budget
from borealis_coder.errors import ProviderContextOverflowError, ProviderError
from borealis_coder.events import Redactor
from borealis_coder.models import ModelResponse, Role, Usage
from borealis_coder.providers.mock import MockProvider
from borealis_coder.sessions.store import SessionStore
from borealis_coder.swarm_view import SwarmView
from tests.helpers import make_config
from tests.test_delegated_rereads import read
from tests.test_swarm import delegate, response, write


class FinalHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "shared.txt").write_text("Observed source fact, not a semantic conclusion.\n")
        self.runner = await build_runner(self.root, config=make_config(
            self.root, agent={"swarm_allow_writes": True, "swarm_max_turns": 6}), interactive=False)
        self.addAsyncCleanup(self.runner.close)
        provider = self.runner.providers[0].provider
        assert isinstance(provider, MockProvider)
        self.provider = provider
        self.events = []
        self.runner.events.subscribe(self.events.append)

    def assert_root_published(self, result):
        self.assertTrue(result.text.strip())
        messages = self.runner.sessions.messages(result.session_id)
        self.assertEqual(messages[-1].role, Role.ASSISTANT)
        self.assertEqual(messages[-1].content, result.text)
        validate_tool_call_order(messages)
        public = [e for e in self.events if e.type == "model.completed" and e.run_id == result.run_id]
        self.assertEqual(public[-1].data["text"], result.text)
        self.assertNotIn("running", {r["status"] for r in self.runner.sessions.tool_calls(result.session_id)})
        self.assertEqual(result.usage.to_dict(), self.runner.sessions.usage(result.session_id).to_dict())

    def child_receipt(self, result):
        return next(r for r in self.runner.sessions.tool_calls(result.session_id)
                    if r["tool_name"] == "delegate_task" and r["run_id"] == result.run_id)

    def assert_durable_child_diagnostic(self, result, cause, expected):
        receipt = self.child_receipt(result)
        self.assertTrue(receipt["is_error"])
        self.assertEqual(receipt["metadata"]["stop_cause"], cause)
        self.assertEqual(receipt["metadata"]["completion_status"], "incomplete")
        self.assertEqual(receipt["metadata"].get("provider_diagnostic"), expected)
        terminal = next(e for e in reversed(self.events)
                        if e.type in {"delegate.failed", "delegate.cancelled"})
        self.assertEqual(terminal.data["stop_cause"], cause)
        self.assertEqual(terminal.data.get("provider_diagnostic"), expected)
        self.assertEqual(terminal.data["result"], receipt["output"])
        self.assertEqual(terminal.data["usage"], receipt["metadata"]["usage"])
        view = SwarmView()
        for event in self.events:
            if event.session_id == result.session_id:
                view.adopt(event)
        detail = view.nodes[terminal.data["delegation_id"]].detail
        self.assertIn("Stop cause: " + cause, detail)
        if expected is None:
            self.assertNotIn("provider_diagnostic", receipt["metadata"])
            self.assertNotIn("Provider diagnostic:", detail)
            self.assertNotIn("context_overflow", result.text)
        else:
            self.assertIn(expected["code"], detail)
            self.assertIn(expected["code"], result.text)
        reopened = SessionStore(self.runner.sessions.path)
        try:
            row = next(r for r in reopened.tool_calls(result.session_id)
                       if r["tool_call_id"] == receipt["tool_call_id"])
            event = next(e for _, e in reopened.events(result.session_id)
                         if e.type == terminal.type)
            self.assertEqual(row, receipt)
            self.assertEqual(event.data, terminal.data)
        finally:
            reopened.close()
        self.assert_root_published(result)
        return receipt

    async def test_recovered_overflow_diagnostic_does_not_describe_later_terminal_failure(self):
        for overflow in (False, True):
            for ending in ("protocol", "blank", "cost"):
                with self.subTest(overflow=overflow, ending=ending):
                    self.events.clear()
                    self.runner.config.agent.max_cost_usd = 0.1 if ending == "cost" else 0
                    requests = []
                    calls_before = self.provider.calls

                    def handler(request, call, requests=requests, overflow=overflow, ending=ending):
                        if not request.metadata.get("delegated"):
                            return response(calls=[delegate(turns=3)]) if request.messages[-1].role != Role.TOOL else response(text=request.messages[-1].content)
                        requests.append(request)
                        if overflow and len(requests) == 1:
                            raise ProviderContextOverflowError("RECOVERED_PRIVATE_OVERFLOW", status_code=400,
                                code="context_length_exceeded", usage=Usage(requests=1, cost_usd=0.01, cost_status="known"))
                        if len(requests) == 1 + int(overflow):
                            return response(calls=[read("read")], cost=0.2 if ending == "cost" else 0.01)
                        return response(text="<tool_call>REJECTED_PRIVATE_PROTOCOL" if ending == "protocol" else " ")

                    self.provider.handler = handler
                    budget = Budget.start(self.runner.config.agent)
                    with patch.object(Budget, "start", return_value=budget):
                        result = await self.runner.run("Inspect within delegated limits")
                    cause = {"protocol": "protocol_output", "blank": "empty_answer", "cost": "budget_cost"}[ending]
                    receipt = self.assert_durable_child_diagnostic(result, cause, None)
                    expected_requests = (2 if ending == "cost" else 5) + int(overflow)
                    self.assertEqual(budget.model_requests, expected_requests)
                    self.assertEqual(self.provider.calls - calls_before, expected_requests)
                    self.assertEqual(result.usage.requests, expected_requests)
                    self.assertAlmostEqual(result.usage.cost_usd,
                                           0.21 + 0.01 * int(overflow) if ending == "cost" else 0.01 * expected_requests)
                    self.assertEqual(receipt["metadata"]["turns"], 0 if ending == "cost" else 3)
                    if overflow:
                        first, recovered = requests[:2]
                        self.assertIn("remaining child turns including this one: 3", first.system)
                        self.assertIn("remaining child turns including this one: 3", recovered.system)
                        self.assertLess(recovered.metadata["resolved_limits"]["effective_context_tokens"],
                                        first.metadata["resolved_limits"]["effective_context_tokens"])
                    reads = [r for r in self.runner.sessions.tool_calls(result.session_id) if r["tool_name"] == "read_file"]
                    self.assertEqual(len(reads), int(ending != "cost"))
                    public = json.dumps([receipt, [e.to_dict() for e in self.events], result.text])
                    self.assertNotIn("RECOVERED_PRIVATE_OVERFLOW", public)
                    self.assertNotIn("REJECTED_PRIVATE_PROTOCOL", public)
                    self.assertEqual(budget.pending_cost_usd, 0)
                    self.assertFalse(budget._held_usage)
                    self.assertFalse(budget._request_closures)

    async def test_billed_provider_diagnostic_survives_cost_stop_and_cancellation_settlement(self):
        expected = {"category": "provider_error", "provider": "mock", "status_code": 503,
                    "retryable": False, "code": "service_unavailable"}
        for ending in ("provider", "cost", "cancel"):
            with self.subTest(ending=ending):
                self.events.clear()
                self.runner.config.agent.max_cost_usd = 0.1 if ending == "cost" else 0
                cancel_once = [ending == "cancel"]
                calls_before = self.provider.calls

                def cancel_after_settlement(event, cancel_once=cancel_once):
                    if (cancel_once[0] and event.type == "swarm.updated"
                            and event.data["usage"]["requests"] == 2):
                        cancel_once[0] = False
                        self.assertTrue(self.runner.cancel(event.session_id))

                self.runner.events.subscribe(cancel_after_settlement)

                def handler(request, call):
                    if not request.metadata.get("delegated"):
                        return response(calls=[delegate(turns=3)]) if request.messages[-1].role != Role.TOOL else response(text=request.messages[-1].content)
                    raise ProviderError("BILLED_PRIVATE_RESPONSE", status_code=503,
                        details={"code": "service_unavailable", "body": "BILLED_PRIVATE_RESPONSE"},
                        usage=Usage(requests=1, cost_usd=0.2, cost_status="known"))

                self.provider.handler = handler
                budget = Budget.start(self.runner.config.agent)
                with patch.object(Budget, "start", return_value=budget):
                    result = await self.runner.run("Inspect billed provider failure")
                cause = {"provider": "provider_failure", "cost": "budget_cost", "cancel": "cancelled"}[ending]
                receipt = self.assert_durable_child_diagnostic(result, cause, expected)
                self.assertEqual(result.stop_reason.value, {"provider": "end_turn", "cost": "budget", "cancel": "cancelled"}[ending])
                attempts = 3 if ending == "provider" else 2
                self.assertEqual(budget.model_requests, attempts)
                self.assertEqual(self.provider.calls - calls_before, attempts)
                self.assertEqual(result.usage.requests, attempts)
                self.assertAlmostEqual(result.usage.cost_usd, 0.22 if ending == "provider" else 0.21)
                self.assertEqual(receipt["metadata"]["usage"], Usage(requests=1, cost_usd=0.2, cost_status="known").to_dict())
                public = json.dumps([receipt, [e.to_dict() for e in self.events], result.text])
                self.assertNotIn("BILLED_PRIVATE_RESPONSE", public)
                self.assertEqual(budget.pending_cost_usd, 0)
                self.assertFalse(budget._held_usage)
                self.assertFalse(budget._request_closures)

    async def test_safe_provider_diagnostic_survives_event_receipt_reopen_and_terminal(self):
        secret = 'PRIVATE_USER_TEXT\\n"token"'
        self.runner.events.redactor = Redactor([secret])
        for with_read in (True, False):
            with self.subTest(with_read=with_read):
                children = 0

                def handler(request, call, with_read=with_read):
                    nonlocal children
                    if not request.metadata.get("delegated"):
                        return response(calls=[delegate(turns=3)]) if request.messages[-1].role != Role.TOOL else response(text="Qualified result: " + request.messages[-1].content)
                    children += 1
                    if children == 1 and with_read:
                        return response(calls=[read("read")])
                    raise ProviderError(secret, status_code=503, retryable=False,
                                        details={"code": "service_unavailable", "body": secret,
                                                 "headers": {"Authorization": secret}},
                                        usage=Usage(requests=1))

                self.provider.handler = handler
                result = await self.runner.run("Inspect synthetic source")
                receipt = self.child_receipt(result)
                expected = {"category": "provider_error", "provider": "mock", "status_code": 503,
                            "retryable": False, "code": "service_unavailable"}
                self.assertEqual(receipt["metadata"].get("provider_diagnostic"), expected)
                terminal = next(e for e in reversed(self.events) if e.type == "delegate.failed")
                self.assertEqual(terminal.data.get("provider_diagnostic"), expected)
                self.assertIn("service_unavailable", result.text)
                view = SwarmView()
                for event in self.events:
                    if event.session_id == result.session_id:
                        view.adopt(event)
                self.assertIn("service_unavailable", view.nodes[terminal.data["delegation_id"]].detail)
                reopened = SessionStore(self.runner.sessions.path)
                try:
                    row = next(r for r in reopened.tool_calls(result.session_id)
                               if r["tool_call_id"] == receipt["tool_call_id"])
                    self.assertEqual(row["metadata"]["provider_diagnostic"], expected)
                    persisted = [e.to_dict() for _, e in reopened.events(result.session_id)]
                    self.assertEqual(next(e["data"]["provider_diagnostic"] for e in persisted
                                          if e["type"] == "delegate.failed"), expected)
                    public = json.dumps([persisted, row, result.text])
                    self.assertNotIn("PRIVATE_USER_TEXT", public)
                    self.assertNotIn("Authorization", public)
                finally:
                    reopened.close()
                self.assert_root_published(result)

    async def test_unknown_sensitive_diagnostic_fields_are_refused_before_escaping(self):
        secret = 'PRIVATE\\n"escaped"\nvalue'
        self.runner.events.redactor = Redactor([secret])
        for code in (secret, "sk-proj-" + "a" * 60, "service_unavailable\\n" + secret,
                     "unknown_" + "x" * 20_000, {"code": secret}):
            with self.subTest(code_type=type(code).__name__):
                def handler(request, call, code=code):
                    if not request.metadata.get("delegated"):
                        return response(calls=[delegate(turns=2)]) if request.messages[-1].role != Role.TOOL else response(text=request.messages[-1].content)
                    raise ProviderError(secret, status_code=401, details={"error": {"code": code},
                                        "headers": {"Authorization": secret}}, usage=Usage(requests=1))
                self.provider.handler = handler
                result = await self.runner.run("Synthetic privacy check")
                receipt = self.child_receipt(result)
                diagnostic = receipt["metadata"]["provider_diagnostic"]
                self.assertEqual(diagnostic, {"category": "provider_error", "provider": "mock",
                                              "status_code": 401, "retryable": False})
                self.assertLess(len(json.dumps(diagnostic)), 600)
                public = json.dumps(self.runner.sessions.tool_calls(result.session_id))
                public += json.dumps([e.to_dict() for _, e in self.runner.sessions.events(result.session_id)])
                self.assertNotIn("PRIVATE", public)
                self.assertNotIn("sk-proj-", public)
                self.assertNotIn("unknown_", public)
                self.assertIn("401", result.text)

    async def test_small_turn_limits_and_one_repair_never_execute_final_tools(self):
        for limit in (1, 2, 3, 6):
            for terminal in ("protocol", "incomplete", "tools", "blank", "healthy", "unrecoverable"):
                with self.subTest(limit=limit, terminal=terminal):
                    children = []
                    self.events.clear()
                    closure_start = max(1, limit - 2) if limit > 1 else 0

                    def handler(request, call, limit=limit, children=children,
                                closure_start=closure_start, terminal=terminal):
                        if not request.metadata.get("delegated"):
                            return response(calls=[delegate(turns=limit)]) if request.messages[-1].role != Role.TOOL else response(text="Qualified parent conclusion.")
                        children.append(request)
                        if request.tools:
                            return response(calls=[read(f"read-{len(children)}")])
                        self.assertIn("All tools are temporarily disabled", request.system)
                        self.assertIn("not evidence of permanent capability absence", request.system)
                        self.assertIn("Eligible investigative tools:", request.system)
                        if closure_start == 0:
                            self.assertIn("No completed tool observations", request.system)
                        self.assertNotIn("REJECTED_CANDIDATE", str(request.messages))
                        if len(children) > closure_start + 1 and terminal != "unrecoverable":
                            return response(text="shared.txt:1 inspected. No files changed; checks remain unverified.")
                        if terminal == "healthy":
                            return response(text="Healthy qualified conclusion.")
                        if terminal == "tools":
                            return response(calls=[write("forbidden.txt")])
                        if terminal == "incomplete":
                            return ModelResponse(text="REJECTED_CANDIDATE", stop_reason="length", usage=Usage(requests=1))
                        return response(text="<tool_call>REJECTED_CANDIDATE" if terminal in {"protocol", "unrecoverable"} else " ")

                    self.provider.handler = handler
                    result = await self.runner.run("Inspect within bounds")
                    receipt = self.child_receipt(result)
                    can_repair = limit >= 3
                    self.assertEqual(receipt["is_error"], terminal != "healthy" and (not can_repair or terminal == "unrecoverable"))
                    self.assertLessEqual(len(children), limit)
                    self.assertEqual(sum(bool(r.tools) for r in children), closure_start)
                    self.assertEqual(result.usage.requests, len(children) + 2)
                    self.assertFalse((self.root / "forbidden.txt").exists())
                    self.assertNotIn("REJECTED_CANDIDATE", receipt["output"])
                    self.assert_root_published(result)

    async def test_early_incomplete_delegate_transitions_to_explicit_bounded_model_closure(self):
        child_requests = []

        def handler(request, call):
            validate_tool_call_order(request.messages)
            if not request.metadata.get("delegated"):
                return response(text="Parent combined the evidence.") if request.messages[-1].role == Role.TOOL else response(calls=[delegate(turns=5)])
            child_requests.append(request)
            if len(child_requests) == 1:
                return response(calls=[read("read")])
            if len(child_requests) == 2:
                return ModelResponse(text="rejected incomplete candidate", stop_reason="length", usage=Usage(requests=1))
            self.assertEqual(request.tools, [])
            self.assertFalse(request.parallel_tool_calls)
            self.assertTrue(any(m.role == Role.USER and "final handoff" in m.content.lower()
                                for m in request.messages))
            self.assertFalse(any("rejected incomplete candidate" in m.content for m in request.messages))
            return response(text="shared.txt:1 was inspected; no files changed. Remaining checks are not performed.")

        self.provider.handler = handler
        result = await self.runner.run("Inspect source")
        receipt = self.child_receipt(result)
        self.assertEqual(len(child_requests), 3)
        self.assertFalse(receipt["is_error"], receipt["output"])
        self.assertIn("shared.txt:1 was inspected", receipt["output"])
        self.assertEqual(receipt["metadata"]["result_source"], "model")
        self.assertEqual(result.usage.requests, 5)
        self.assert_root_published(result)

    async def test_delegate_fallback_matrix_is_useful_honest_and_causal(self):
        for terminal, cause in (("blank", "empty_answer"), ("protocol", "protocol_output"),
                                ("incomplete", "incomplete_provider_response"),
                                ("tools", "tool_call_on_final_turn"), ("provider", "provider_failure")):
            with self.subTest(terminal=terminal):
                self.events.clear()
                children = 0

                def handler(request, call, terminal=terminal):
                    nonlocal children
                    if not request.metadata.get("delegated"):
                        return response(text="Parent combined qualified partial evidence.") if request.messages[-1].role == Role.TOOL else response(calls=[delegate("Inspect shared.txt and verify", turns=2)])
                    children += 1
                    if children == 1:
                        return response(calls=[read("read")])
                    if terminal == "provider":
                        raise ProviderError("Synthetic provider refusal", usage=Usage(requests=1))
                    if terminal == "tools":
                        return response(calls=[write("forbidden.txt")])
                    if terminal == "incomplete":
                        return ModelResponse(text="do not publish incomplete candidate", stop_reason="length", usage=Usage(requests=1))
                    return response(text="<tool_call>PRIVATE_REJECTED_PROTOCOL" if terminal == "protocol" else " \n")

                self.provider.handler = handler
                result = await self.runner.run("Inspect source")
                receipt = self.child_receipt(result)
                self.assertTrue(receipt["is_error"])
                self.assertEqual(receipt["metadata"].get("result_source"), "runtime")
                self.assertEqual(receipt["metadata"].get("stop_cause"), cause)
                self.assertEqual(receipt["metadata"].get("completion_status"), "incomplete")
                for token in ("Task:", "Inspect shared.txt", "Observed actions", "read_file", "shared.txt", "Remaining", "Verification"):
                    self.assertIn(token, receipt["output"])
                self.assertIn("Observed source fact", receipt["output"])
                self.assertNotIn("PRIVATE_REJECTED_PROTOCOL", receipt["output"])
                self.assertNotIn("do not publish incomplete candidate", receipt["output"])
                terminal_event = next(e for e in reversed(self.events) if e.type == "delegate.failed")
                self.assertEqual(terminal_event.data["result"], receipt["output"])
                self.assertEqual(terminal_event.data.get("stop_cause"), cause)
                self.assertEqual(children, 2)
                self.assertFalse((self.root / "forbidden.txt").exists())
                self.assert_root_published(result)

    async def test_root_direct_evidence_survives_provider_failure_and_remains_terminal(self):
        def handler(request, call):
            if call == 1:
                return response(calls=[read("root-read")])
            raise ProviderError("Synthetic root provider refusal", usage=Usage(requests=1))

        self.provider.handler = handler
        result = await self.runner.run("Inspect shared.txt without edits")
        self.assertEqual(result.stop_reason.value, "error")
        for token in ("Task:", "Observed actions", "root-read", "shared.txt", "Observed source fact", "Remaining", "Verification"):
            self.assertIn(token, result.text)
        self.assertEqual(result.usage.requests, 2)
        self.assert_root_published(result)

    async def test_cancelled_delegate_has_same_useful_durable_and_event_handoff(self):
        entered = asyncio.Event()
        original = self.provider.complete

        async def controlled(request):
            if request.metadata.get("delegated") and any(m.role == Role.TOOL for m in request.messages):
                entered.set()
                await asyncio.Event().wait()
            return await original(request)

        def handler(request, call):
            return response(calls=[read("read")]) if request.metadata.get("delegated") else response(calls=[delegate(turns=4)])

        self.provider.handler = handler
        with patch.object(self.provider, "complete", side_effect=controlled):
            task = asyncio.create_task(self.runner.run("Inspect then cancel"))
            await asyncio.wait_for(entered.wait(), 3)
            task.cancel()
            result = await asyncio.wait_for(task, 5)
        self.assertEqual(result.stop_reason.value, "cancelled")
        receipt = self.child_receipt(result)
        self.assertEqual(receipt["metadata"].get("completion_status"), "incomplete")
        self.assertEqual(receipt["metadata"].get("stop_cause"), "cancelled")
        self.assertIn("Observed source fact", receipt["output"])
        terminal = next(e for e in self.events if e.type == "delegate.cancelled")
        self.assertEqual(terminal.data["result"], receipt["output"])
        self.assert_root_published(result)

    async def test_early_root_incomplete_gets_text_only_closure_and_healthy_text_stays_verbatim(self):
        requests = []

        def handler(request, call):
            requests.append(request)
            if call == 1:
                return ModelResponse(text="unusable partial", stop_reason="length", usage=Usage(requests=1))
            self.assertEqual(request.tools, [])
            return response(text="The final requested answer, unchanged.")

        self.provider.handler = handler
        result = await self.runner.run("Give an answer")
        self.assertEqual(result.text, "The final requested answer, unchanged.")
        self.assertEqual(len(requests), 2)
        self.assert_root_published(result)

    async def test_root_bounded_invalid_answers_and_provider_failure_without_tools(self):
        for mode in ("blank", "protocol", "incomplete", "provider", "tools"):
            with self.subTest(mode=mode):
                self.events.clear()
                self.runner.config.agent.max_turns = 1

                def handler(request, call, mode=mode):
                    if mode == "provider":
                        raise ProviderError("Synthetic no-evidence refusal", usage=Usage(requests=1))
                    if mode == "incomplete":
                        return ModelResponse(text="", stop_reason="length", usage=Usage(requests=1))
                    return response(calls=[write("never.txt")]) if mode == "tools" else response(
                        text="<tool_call>REJECTED_PRIVATE" if mode == "protocol" else " ")

                self.provider.handler = handler
                result = await self.runner.run("Bounded no-evidence answer")
                self.assertNotEqual(result.stop_reason.value, "end_turn")
                self.assertIn("No completed tool observations", result.text)
                self.assertIn("Bounded no-evidence answer", result.text)
                self.assertNotIn("REJECTED_PRIVATE", result.text)
                self.assertFalse((self.root / "never.txt").exists())
                self.assertEqual(result.usage.requests, 1)
                self.assert_root_published(result)

    async def test_cancel_during_startup_uses_nonempty_authoritative_fallback(self):
        entered = asyncio.Event()

        async def blocked_recovery(rows, context):
            entered.set()
            await asyncio.Event().wait()

        with patch.object(self.runner, "_recover_nested_calls", side_effect=blocked_recovery):
            task = asyncio.create_task(self.runner.run("Startup task scope"))
            await asyncio.wait_for(entered.wait(), 3)
            task.cancel()
            result = await asyncio.wait_for(task, 5)
        self.assertEqual(result.stop_reason.value, "cancelled")
        self.assertIn("Startup task scope", result.text)
        self.assertEqual(result.usage.requests, 0)
        self.assertTrue(result.incomplete)
        self.assert_root_published(result)

    async def test_fatal_child_cost_keeps_same_partial_receipt_without_extra_dispatch(self):
        self.runner.config.agent.max_cost_usd = 0.1
        children = 0

        def handler(request, call):
            nonlocal children
            if not request.metadata.get("delegated"):
                return response(calls=[delegate(turns=4)])
            children += 1
            if children == 1:
                return response(calls=[read("read")])
            return response(text="Not accepted above cost ceiling", cost=0.2)

        self.provider.handler = handler
        result = await self.runner.run("Cost-bounded inspection")
        self.assertEqual(result.stop_reason.value, "budget")
        receipt = self.child_receipt(result)
        self.assertEqual(receipt["metadata"]["stop_cause"], "budget_cost")
        self.assertEqual(receipt["metadata"]["completion_status"], "incomplete")
        self.assertIn("Observed source fact", receipt["output"])
        terminal = next(e for e in self.events if e.type == "delegate.failed")
        self.assertEqual(terminal.data["result"], receipt["output"])
        self.assertEqual(children, 2)
        self.assertEqual(result.usage.requests, 3)
        self.assert_root_published(result)

    async def test_latest_boundary_evidence_and_quoted_secret_are_retained_safely(self):
        secret = 'SYNTHETIC_SECRET"quoted'
        private_path = secret + ".txt"
        (self.root / private_path).write_text("LATEST_BOUNDARY_FACT\n")
        self.runner.events.redactor = Redactor([secret])
        self.events.clear()
        children = 0

        def handler(request, call):
            nonlocal children
            if not request.metadata.get("delegated"):
                return response(text="Qualified parent conclusion.") if request.messages[-1].role == Role.TOOL else response(calls=[delegate(turns=2)])
            children += 1
            if children > 1:
                return response(text=" ")
            calls = [read("read-" + str(index)) for index in range(11)]
            final = read("last-boundary")
            final.arguments["path"] = private_path
            return response(calls=[*calls, final])

        self.provider.handler = handler
        result = await self.runner.run("Boundary evidence")
        receipt = self.child_receipt(result)
        self.assertIn("LATEST_BOUNDARY_FACT", receipt["output"])
        self.assertIn("last-boundary", receipt["output"])
        self.assertIn("[REDACTED]", receipt["output"])
        self.assertNotIn("SYNTHETIC_SECRET", receipt["output"])
        self.assertLessEqual(len(receipt["output"]), 8000)
        source = next(r for r in self.runner.sessions.tool_calls(result.session_id)
                      if r["tool_call_id"].endswith(".last-boundary"))
        self.assertEqual(source["metadata"]["path"], private_path)
        self.assertEqual(children, 2)
        self.assert_root_published(result)
