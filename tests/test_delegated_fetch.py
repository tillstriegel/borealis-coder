from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from borealis_coder.agent import build_runner, validate_tool_call_order
from borealis_coder.models import Effect, Role, ToolCall, ToolResult
from borealis_coder.providers.mock import MockProvider
from borealis_coder.safety import ApprovalManager
from borealis_coder.safety.policy import PolicyAction
from borealis_coder.tools.base import FunctionTool, MutationScope, ToolRegistry, object_schema
from borealis_coder.tools.delegate import DelegateTaskTool
from borealis_coder.tools.fetch import FetchUrlTool
from tests.helpers import make_config, make_context
from tests.test_swarm import delegate, response

PUBLIC_IP = "93.184.216.34"
PUBLIC_URL = "https://docs.example.test/evidence"
PUBLIC_TEXT = "Synthetic public evidence: the release has three stages."


def fetch(url=PUBLIC_URL, max_chars=2000):
    return ToolCall(name="fetch_url", arguments={"url": url, "max_chars": max_chars})


def discover(names=()):
    return ToolCall(name="discover_tools", arguments={"names": list(names), "offset": 0})


class DelegatedFetchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.approvals = []
        self.destinations = []
        self.lookups = []
        self.runner = await build_runner(
            self.root, config=make_config(
                self.root, safety={"network": True, "approval": "on-risk"},
                agent={"swarm_max_concurrent": 1},
            ), approval_callback=self.approve, interactive=True,
        )
        self.addAsyncCleanup(self.runner.close)
        provider = self.runner.providers[0].provider
        assert isinstance(provider, MockProvider)
        self.provider = provider
        self.events = []
        self.runner.events.subscribe(self.events.append)
        # Only the DNS and HTTP wire ports are synthetic. URL validation,
        # redirects, policy, approvals, output bounds and persistence stay real.
        self.resolver = self.enterContext(patch(
            "borealis_coder.tools.fetch.socket.getaddrinfo", side_effect=self.resolve,
        ))
        self.transport = self.enterContext(patch(
            "borealis_coder.tools.fetch._request_once", side_effect=self.request_once,
        ))

    def approve(self, request):
        self.assertEqual(request.tool_name, "fetch_url")
        self.assertEqual(request.decision.action, PolicyAction.ASK)
        self.assertEqual(request.decision.risk, "high")
        self.approvals.append(request)
        return "allow_once"

    def resolve(self, host, port, **kwargs):
        self.lookups.append((host, port))
        try:
            address = str(ipaddress.ip_address(host))
        except ValueError:
            address = PUBLIC_IP
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    def request_once(self, parsed, address, max_bytes, timeout):
        self.destinations.append((parsed.geturl(), address, max_bytes, timeout))
        return PUBLIC_TEXT.encode(), 200, "text/plain", "utf-8", None

    def assert_settled(self, result, requests):
        self.assertEqual(result.stop_reason.value, "end_turn", result.error)
        self.assertEqual(result.usage.requests, requests)
        self.assertEqual(result.usage.input_tokens, requests * 10)
        self.assertEqual(result.usage.output_tokens, requests * 2)
        self.assertAlmostEqual(result.usage.cost_usd, requests * 0.01)
        self.assertEqual(result.usage.to_dict(), self.runner.sessions.usage(result.session_id).to_dict())
        rows = self.runner.sessions.tool_calls(result.session_id)
        self.assertNotIn("running", {row["status"] for row in rows})
        self.assertEqual(len({row["tool_call_id"] for row in rows}), len(rows))
        validate_tool_call_order(self.runner.sessions.messages(result.session_id))
        starts = [e.data["delegation_id"] for e in self.events if e.type == "delegate.started"]
        ends = [e for e in self.events if e.type in {"delegate.completed", "delegate.failed", "delegate.cancelled"}]
        self.assertCountEqual(starts, [e.data["delegation_id"] for e in ends])
        self.assertTrue(all(e.data["limits"]["active_requests"] == 0 for e in ends))
        return rows

    async def test_runner_child_grandchild_discover_fetch_and_conclude(self):
        for surface in ("direct", "ptc"):
            with self.subTest(surface=surface):
                self.runner.config.script.tool_surface = surface
                self.events.clear()
                self.approvals.clear()
                self.destinations.clear()
                self.lookups.clear()
                seen = []
                child_schemas = []
                builtin = self.runner.tools.get("fetch_url")
                assert builtin is not None
                fetch_schema = builtin.schema()

                def handler(request, call, seen=seen, child_schemas=child_schemas,
                            fetch_schema=fetch_schema):
                    depth = request.metadata.get("delegation_depth", 0)
                    seen.append(depth)
                    validate_tool_call_order(request.messages)
                    if not depth:
                        return (response(text="Root conclusion: " + request.messages[-1].content)
                                if request.messages[-1].role == Role.TOOL else
                                response(calls=[delegate(turns=6)]))
                    self.assertTrue(request.metadata["delegation_allow_network"])
                    self.assertIn("Network access is enabled for built-in fetch_url only", request.system)
                    tools = [m for m in request.messages if m.role == Role.TOOL]
                    if not tools:
                        schemas = {t["name"]: t for t in request.tools}
                        child_schemas.append(schemas)
                        self.assertEqual(schemas["fetch_url"], fetch_schema)
                        self.assertNotIn("shell", schemas)
                        self.assertNotIn("script", schemas)
                        return response(calls=[discover()])
                    if len(tools) == 1:
                        catalog = json.loads(tools[-1].content)
                        self.assertIsNone(catalog["next_offset"])
                        self.assertIn("fetch_url", [t["name"] for t in catalog["tools"]])
                        return response(calls=[discover(["fetch_url"])])
                    if len(tools) == 2:
                        schema = json.loads(tools[-1].content)["tools"][0]
                        self.assertEqual(schema["parameters"], fetch_schema["parameters"])
                        self.assertEqual(schema["effect"], "network")
                        return response(calls=[fetch()])
                    if len(tools) == 3:
                        self.assertFalse(tools[-1].is_error, tools[-1].content)
                        self.assertEqual(tools[-1].content, PUBLIC_TEXT)
                        if depth == 1:
                            return response(calls=[delegate("grandchild", turns=6)])
                    if depth == 1:
                        self.assertEqual(request.tools, [])
                        self.assertIn("Eligible investigative tools:", request.system)
                        self.assertIn("fetch_url", request.system.split("Eligible investigative tools:")[1])
                    return response(text=f"Depth {depth} conclusion: " + tools[-1].content)

                self.provider.handler = handler
                result = await asyncio.wait_for(self.runner.run("Inspect public evidence recursively"), 10)
                self.assertTrue(child_schemas, "The real child must reach its first model request")
                self.assertIn("fetch_url", child_schemas[0])
                rows = self.assert_settled(result, 11)
                self.assertEqual(seen, [0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 0])
                self.assertIn(PUBLIC_TEXT, result.text)
                self.assertIn("Depth 2 conclusion", result.text)
                self.assertEqual(self.lookups, [("docs.example.test", 443)] * 2)
                self.assertEqual(self.destinations, [(PUBLIC_URL, PUBLIC_IP, 8000, 30)] * 2)
                self.assertEqual(len(self.approvals), 2)
                receipts = [r for r in rows if r["tool_name"] == "delegate_task"]
                self.assertEqual(len(receipts), 2)
                self.assertEqual(sorted(r["metadata"]["usage"]["requests"] for r in receipts), [4, 5])
                for receipt in receipts:
                    self.assertFalse(receipt["is_error"], receipt["output"])
                    self.assertIn(PUBLIC_TEXT, receipt["output"])
                    self.assertEqual(receipt["metadata"]["result_source"], "model")
                    self.assertEqual(receipt["metadata"]["capability"], "read-only")
                    self.assertIn("fetch_url", [e["tool"] for e in receipt["metadata"]["evidence"]])
                fetched = [r for r in rows if r["tool_name"] == "fetch_url"]
                self.assertEqual({r["metadata"]["delegation_depth"] for r in fetched}, {1, 2})
                self.assertTrue(all(r["metadata"]["status"] == 200 for r in fetched))
                self.assertTrue(all(r["metadata"]["network_allowed"] for r in receipts))
                self.assertTrue(all(e.data["network_allowed"] for e in self.events
                                    if e.type == "delegate.started"))

    def denial_handler(self, requests, *, enable_shared_config=False):
        def handler(request, call):
            depth = request.metadata.get("delegation_depth", 0)
            requests.append(request)
            if enable_shared_config and depth == 1:
                self.runner.config.safety.network = True
            observations = [m for m in request.messages if m.role == Role.TOOL]
            if not observations:
                return response(calls=[delegate(turns=6)] if not depth else [
                    discover(), discover(["fetch_url"]), fetch(),
                ])
            if depth == 1 and observations[-1].tool_name != "delegate_task":
                return response(calls=[delegate("restricted grandchild", turns=6)])
            return response(text="Restricted conclusion: " + observations[-1].content)
        return handler

    def assert_denied_requests(self, requests, expected_depths=(0, 1, 2)):
        self.assertEqual({r.metadata.get("delegation_depth", 0) for r in requests}, set(expected_depths))
        for request in requests:
            if not request.metadata.get("delegated"):
                continue
            self.assertFalse(request.metadata["delegation_allow_network"])
            self.assertNotIn("fetch_url", [t["name"] for t in request.tools])
            self.assertIn("Network access is disabled", request.system)
            if "Eligible investigative tools:" in request.system:
                self.assertNotIn("fetch_url", request.system.split("Eligible investigative tools:")[1])
            observations = [m for m in request.messages if m.role == Role.TOOL]
            for observation in observations:
                if observation.tool_name == "fetch_url":
                    self.assertTrue(observation.is_error)
                    self.assertIn("unavailable to this delegated capability", observation.content)
                elif observation.tool_name == "discover_tools":
                    entries = json.loads(observation.content)["tools"]
                    if len(entries) == 1:
                        self.assertEqual(entries, [{"name": "fetch_url", "unavailable": True}])
                    else:
                        self.assertNotIn("fetch_url", [e["name"] for e in entries])
        self.assertEqual(self.approvals, [])
        self.resolver.assert_not_called()
        self.transport.assert_not_called()

    async def test_no_network_hides_and_refuses_forged_fetch_at_both_depths(self):
        self.runner.config.safety.network = False
        requests = []
        self.provider.handler = self.denial_handler(requests)
        result = await self.runner.run("Try restricted descendants")
        rows = self.assert_settled(result, 7)
        self.assert_denied_requests(requests)
        denied = [r for r in rows if r["tool_name"] == "fetch_url"]
        self.assertEqual({r["metadata"]["delegation_depth"] for r in denied}, {1, 2})
        self.assertTrue(all(r["is_error"] for r in denied))
        self.assertTrue(all(not r["metadata"]["network_allowed"] for r in rows
                            if r["tool_name"] == "delegate_task"))

    async def test_restricted_tree_cannot_regain_access_when_shared_config_is_enabled(self):
        self.runner.config.safety.network = False
        requests = []
        self.provider.handler = self.denial_handler(requests, enable_shared_config=True)
        result = await self.runner.run("Retain inherited restriction")
        self.assert_settled(result, 7)
        self.assertTrue(self.runner.config.safety.network)
        self.assert_denied_requests(requests)

    def delegated_context(self, permission: object = True, *, depth=1):
        context = make_context(self.root, self.runner.config)
        context.policy.interactive = True
        context.approvals = ApprovalManager(self.approve)
        context.metadata.update(
            delegated=True, delegation_depth=depth, delegation_allow_network=permission,
            tool_registry=self.runner.tools, provider_routes=self.runner.providers,
        )
        return context

    async def test_allowed_root_with_explicitly_restricted_child_cannot_escalate(self):
        context = self.delegated_context(False, depth=0)
        # Confirm the root has actual fetch authority under the same config.
        context.metadata["delegated"] = False
        context.metadata.pop("delegation_allow_network")
        allowed = await self.runner.tools.execute(fetch(), context)
        self.assertFalse(allowed.is_error, allowed.output)
        self.assertEqual(allowed.output, PUBLIC_TEXT)
        self.approvals.clear()
        self.resolver.reset_mock()
        self.transport.reset_mock()
        context.metadata.update(delegated=True, delegation_allow_network=False)
        requests = []
        self.provider.handler = self.denial_handler(requests)
        # This restricted parent starts a child and grandchild. No runtime hook
        # is replaced; standalone registry execution uses the same guards.
        result = await DelegateTaskTool().execute({"task": "Restricted branch", "max_turns": 6}, context)
        self.assertFalse(result.is_error, result.output)
        self.assertFalse(result.metadata["network_allowed"])
        self.assertEqual(result.metadata["usage"]["requests"], 3)
        self.assert_denied_requests(requests, expected_depths=(1, 2))
        self.assertIn("unavailable to this delegated capability", result.output)

    async def test_direct_execution_fails_closed_for_missing_false_or_revoked_permission(self):
        for depth in (1, 2):
            for permission in (None, False, True, "true", 1):
                with self.subTest(depth=depth, permission=permission):
                    context = self.delegated_context(permission, depth=depth)
                    if permission is None:
                        context.metadata.pop("delegation_allow_network")
                    self.runner.config.safety.network = permission is not True
                    # Even an explicit inherited grant cannot override --no-network.
                    result = await self.runner.tools.execute(fetch(), context)
                    self.assertTrue(result.is_error)
                    self.assertIn("unavailable to this delegated capability", result.output)
                    named = await self.runner.tools.execute(discover(["fetch_url"]), context)
                    self.assertEqual(json.loads(named.output)["tools"], [
                        {"name": "fetch_url", "unavailable": True},
                    ])
        self.approvals.clear()
        self.resolver.assert_not_called()
        self.transport.assert_not_called()

    async def run_fetch_at_depth(self, depth, call=None):
        observations = []

        def handler(request, count):
            current_depth = request.metadata.get("delegation_depth", 0)
            if request.messages[-1].role == Role.TOOL:
                observations.append(request.messages[-1])
                return response(text="Observed result: " + request.messages[-1].content)
            return response(calls=[delegate(turns=3)] if current_depth < depth else [call or fetch()])

        self.provider.handler = handler
        result = await self.runner.run("Inspect one public URL")
        rows = self.assert_settled(result, 2 * depth + 2)
        fetched = [r for r in rows if r["tool_name"] == "fetch_url"]
        self.assertEqual(len(fetched), 1)
        self.assertEqual(fetched[0]["metadata"]["delegation_depth"], depth)
        self.assertTrue(any(m.tool_name == "fetch_url" for m in observations))
        return fetched[0]

    async def test_approval_denial_prevents_dns_and_transport(self):
        def deny(request):
            self.approve(request)
            return False

        self.runner.tool_context.approvals = ApprovalManager(deny)
        for depth in (1, 2):
            with self.subTest(depth=depth):
                self.events.clear()
                row = await self.run_fetch_at_depth(depth)
                self.assertTrue(row["is_error"])
                self.assertIn("Approval denied", row["output"])
        self.assertEqual(len(self.approvals), 2)
        self.resolver.assert_not_called()
        self.transport.assert_not_called()

    async def test_safety_modes_still_gate_fetch_before_dns(self):
        for mode, approval, interactive in (("plan", "on-risk", True),
                                            ("workspace-write", "on-risk", False)):
            with self.subTest(mode=mode, interactive=interactive):
                self.runner.config.safety.mode = mode
                self.runner.config.safety.approval = approval
                self.runner.tool_context.policy.interactive = interactive
                self.events.clear()
                row = await self.run_fetch_at_depth(2)
                self.assertTrue(row["is_error"])
                self.assertIn("plan mode" if mode == "plan" else "no approval channel", row["output"])
        self.assertEqual(self.approvals, [])
        self.resolver.assert_not_called()
        self.transport.assert_not_called()

    async def test_descendant_url_and_redirect_protections_remain_active(self):
        cases = (
            ("http://127.0.0.1/admin", "non-public address", 1, 0),
            ("http://10.0.0.1/private", "non-public address", 1, 0),
            ("http://169.254.169.254/metadata", "non-public address", 1, 0),
            ("http://[::1]/admin", "non-public address", 1, 0),
            ("https://user:password@docs.example.test/", "Credentials", 0, 0),
            (PUBLIC_URL, "non-public address", 2, 1),
        )
        for depth in (1, 2):
            for url, error, lookups, transports in cases:
                with self.subTest(depth=depth, url=url):
                    self.events.clear()
                    self.lookups.clear()
                    self.destinations.clear()
                    self.approvals.clear()
                    self.transport.side_effect = self.request_once
                    if url == PUBLIC_URL:
                        def redirect(parsed, address, max_bytes, timeout):
                            self.destinations.append((parsed.geturl(), address, max_bytes, timeout))
                            return b"", 302, "text/plain", "utf-8", "http://10.0.0.1/private"
                        self.transport.side_effect = redirect
                    row = await self.run_fetch_at_depth(depth, fetch(url))
                    self.assertTrue(row["is_error"])
                    self.assertIn(error, row["output"])
                    self.assertEqual(len(self.lookups), lookups)
                    self.assertEqual(len(self.destinations), transports)
                    self.assertTrue(all(d[1] == PUBLIC_IP for d in self.destinations))
                    self.assertEqual(len(self.approvals), 1)

    async def test_exact_builtin_scope_excludes_network_spoofs_subclasses_shell_and_script(self):
        executed = []

        def unsafe(arguments, context):
            executed.append("unsafe")
            return ToolResult("must not execute")

        class OverriddenFetch(FetchUrlTool):
            async def execute(self, arguments, context):
                return unsafe(arguments, context)

        tools = [FunctionTool(
            name="custom_network", description="Synthetic external tool",
            parameters=object_schema({}), function=unsafe, effect=Effect.NETWORK,
            mutation_scope=MutationScope.NONE,
        ), FunctionTool(
            name="fetch_url", description="Synthetic name spoof",
            parameters=object_schema({}), function=unsafe, effect=Effect.READ,
            mutation_scope=MutationScope.NONE,
        ), OverriddenFetch()]
        mutating_fetch = FetchUrlTool()
        mutating_fetch.mutation_scope = MutationScope.EXTERNAL
        tools.append(mutating_fetch)
        for name in ("shell", "script", "script_job"):
            tool = self.runner.tools.get(name)
            assert tool is not None
            tools.append(tool)
        discovery_tool = self.runner.tools.get("discover_tools")
        assert discovery_tool is not None
        for tool in tools:
            with self.subTest(tool=type(tool).__name__, name=tool.name):
                registry = ToolRegistry([tool, discovery_tool])
                context = self.delegated_context()
                context.metadata.update(tool_registry=registry, delegation_allow_writes=True)
                call = ToolCall(name=tool.name, arguments={
                    "fetch_url": fetch().arguments,
                    "shell": {"command": "curl https://docs.example.test", "timeout_seconds": None,
                              "cwd": ".", "description": "Synthetic forbidden shell call"},
                    "script": {"source": "text('forbidden');", "background": False},
                    "script_job": {"action": "status", "job_id": "synthetic", "seconds": 0},
                }.get(tool.name, {}))
                # The spoof has an empty schema; the real subclass needs fetch arguments.
                if type(tool) is FunctionTool:
                    call.arguments = {}
                result = await registry.execute(call, context)
                self.assertTrue(result.is_error)
                self.assertIn("unavailable to this delegated capability", result.output)
                catalog = await registry.execute(discover(), context)
                self.assertNotIn(tool.name, [t["name"] for t in json.loads(catalog.output)["tools"]])
                named = await registry.execute(discover([tool.name]), context)
                self.assertEqual(json.loads(named.output)["tools"], [{"name": tool.name, "unavailable": True}])
        self.assertEqual(executed, [])
        self.assertEqual(self.approvals, [])
        self.resolver.assert_not_called()
        self.transport.assert_not_called()

    async def test_network_and_file_write_permissions_are_independent(self):
        context = self.delegated_context()
        call = ToolCall(name="write_file", arguments={
            "path": "independent.txt", "content": "bounded edit", "expected_sha256": None,
        })
        denied = await self.runner.tools.execute(call, context)
        self.assertTrue(denied.is_error)
        self.assertIn("unavailable to this delegated capability", denied.output)
        self.assertFalse((self.root / "independent.txt").exists())
        context.metadata.update(delegation_allow_network=False, delegation_allow_writes=True)
        allowed = await self.runner.tools.execute(call, context)
        self.assertFalse(allowed.is_error, allowed.output)
        self.assertEqual((self.root / "independent.txt").read_text(), "bounded edit")
        denied_fetch = await self.runner.tools.execute(fetch(), context)
        self.assertTrue(denied_fetch.is_error)
        forged = delegate()
        forged.arguments["delegation_allow_network"] = True
        invalid = await self.runner.tools.execute(forged, context)
        self.assertTrue(invalid.is_error)
        self.assertIn("unknown keys", invalid.output)
        self.resolver.assert_not_called()
        self.transport.assert_not_called()

    async def test_final_and_repair_turns_cannot_execute_enabled_fetch(self):
        for turns in (1, 3):
            with self.subTest(turns=turns):
                self.events.clear()
                children = []

                def handler(request, call, turns=turns, children=children):
                    if not request.metadata.get("delegated"):
                        return (response(text="The child could not conclude.")
                                if request.messages[-1].role == Role.TOOL else
                                response(calls=[delegate(turns=turns)]))
                    children.append(request)
                    if turns == 3 and len(children) == 1:
                        return response(text=" ")  # Force the existing repair path.
                    return response(calls=[fetch()])  # Forge a call on a tool-free turn.

                self.provider.handler = handler
                result = await self.runner.run("Conclude within the existing bounds")
                rows = self.assert_settled(result, 3 if turns == 1 else 4)
                self.assertEqual(len(children), 1 if turns == 1 else 2)
                self.assertTrue(all(not r.tools for r in children[-1:]))
                self.assertIn("fetch_url", children[-1].system.split("Eligible investigative tools:")[1])
                self.assertEqual([r["tool_name"] for r in rows], ["delegate_task"])
                self.assertTrue(rows[0]["is_error"])
                self.assertEqual(rows[0]["metadata"]["stop_cause"], "tool_call_on_final_turn")
        self.assertEqual(self.approvals, [])
        self.resolver.assert_not_called()
        self.transport.assert_not_called()

    async def test_descendant_fetch_keeps_output_collection_bounds(self):
        self.transport.side_effect = lambda *args: (b"x" * 401, 200, "text/plain", "utf-8", None)
        row = await self.run_fetch_at_depth(2, fetch(max_chars=100))
        self.assertFalse(row["is_error"], row["output"])
        # The registry retains the bounded body and appends the durable artifact
        # reference under the shared output limit.
        self.assertLessEqual(len(row["output"].split("\nHistorical output artifact")[0]), 100)
        self.assertLessEqual(len(row["output"]), self.runner.config.context.tool_output_chars)
        self.assertEqual(row["metadata"]["collection_status"], "partial")
        self.assertEqual(row["metadata"]["observed_bytes"], 401)
        self.assertTrue(row["metadata"]["output_artifact"]["artifact_id"])
        self.assertEqual(row["metadata"]["output_artifact"]["stored_bytes"], 400)
        self.assertEqual(self.transport.call_args.args[1:], (PUBLIC_IP, 400, 30))

    async def test_mixed_dns_answers_and_redirect_limit_remain_enforced(self):
        public = (socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, 443))
        private = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))
        self.resolver.side_effect = lambda *args, **kwargs: [public, private]
        row = await self.run_fetch_at_depth(2)
        self.assertTrue(row["is_error"])
        self.assertIn("non-public address", row["output"])
        self.transport.assert_not_called()
        self.events.clear()
        self.resolver.reset_mock(side_effect=True)
        self.resolver.side_effect = self.resolve
        self.transport.side_effect = lambda *args: (b"", 302, "text/plain", "utf-8", PUBLIC_URL)
        row = await self.run_fetch_at_depth(2)
        self.assertTrue(row["is_error"])
        self.assertIn("Too many redirects (maximum 5)", row["output"])
        self.assertEqual(self.resolver.call_count, 6)
        self.assertEqual(self.transport.call_count, 6)
        self.assertTrue(all(call.args[1] == PUBLIC_IP for call in self.transport.call_args_list))

    async def test_parent_cancellation_during_descendant_approval_settles_receipts(self):
        pending = asyncio.Event()
        sessions = []

        async def approve(request):
            self.approve(request)
            pending.set()
            await asyncio.Event().wait()
            return True

        self.runner.tool_context.approvals = ApprovalManager(approve)
        self.runner.events.subscribe(lambda e: sessions.append(e.session_id) if e.type == "delegate.started" else None)

        def handler(request, call):
            return response(calls=[fetch()] if request.metadata.get("delegated") else [delegate()])

        self.provider.handler = handler
        task = asyncio.create_task(self.runner.run("Cancel pending approval"))
        try:
            await asyncio.wait_for(pending.wait(), 2)
            self.assertTrue(self.runner.cancel(sessions[0]))
            result = await asyncio.wait_for(task, 2)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(result.stop_reason.value, "cancelled")
        self.assertEqual(result.usage.requests, 2)
        self.assertAlmostEqual(result.usage.cost_usd, 0.02)
        self.assertEqual(result.usage.to_dict(), self.runner.sessions.usage(result.session_id).to_dict())
        rows = self.runner.sessions.tool_calls(result.session_id)
        self.assertCountEqual([r["tool_name"] for r in rows], ["delegate_task", "fetch_url"])
        self.assertNotIn("running", {r["status"] for r in rows})
        self.assertTrue(all(r["is_error"] for r in rows))
        self.assertTrue(any(e.type == "delegate.cancelled" for e in self.events))
        self.assertEqual(len(self.approvals), 1)
        self.resolver.assert_not_called()
        self.transport.assert_not_called()
