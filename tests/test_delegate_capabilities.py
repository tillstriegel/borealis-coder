from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from borealis_coder.agent import build_runner
from borealis_coder.agent.swarm import SwarmState
from borealis_coder.interactive import InteractiveCLI
from borealis_coder.models import Role, ToolCall
from borealis_coder.providers.mock import MockProvider
from borealis_coder.swarm_view import SwarmView
from tests.helpers import make_config, make_context
from tests.test_swarm import response, write


def delegate(capability, *, identifier="delegate", task="Implement child.txt", turns=4):
    return ToolCall(id=identifier, name="delegate_task", arguments={
        "task": task, "max_turns": turns, "capability": capability,
    })


class DelegateCapabilityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.runner = await build_runner(self.root, config=make_config(self.root), interactive=False)
        self.addAsyncCleanup(self.runner.close)
        provider = self.runner.providers[0].provider
        assert isinstance(provider, MockProvider)
        self.provider = provider
        self.events = []
        self.runner.events.subscribe(self.events.append)

    def starts(self):
        return [e for e in self.events if e.type == "delegate.started"]

    def view(self):
        view = SwarmView()
        for event in self.events:
            view.adopt(event)
        return view

    def assert_schema(self, schema, available):
        parameters = schema["parameters"]
        self.assertIn("capability", parameters["required"])
        field = parameters["properties"]["capability"]
        self.assertEqual(field["enum"], ["read-only", "write"])
        self.assertIn("every delegation", field["description"])
        self.assertIn("investigation/review", field["description"])
        self.assertIn("implementation/file edits", field["description"])
        self.assertIn("caller", schema["description"])
        self.assertIn(f"Available capability ceiling: {available}.", schema["description"])

    async def test_default_child_turn_limit_allows_work_past_legacy_cap(self):
        child_requests = []

        def handler(request, call):
            if request.metadata.get("delegated"):
                child_requests.append(request)
                if len(child_requests) <= 17:
                    tool_call = write(path=f"turn-{len(child_requests)}.txt")
                    tool_call.id = f"write-{len(child_requests)}"
                    return response(calls=[tool_call])
                return response(text="All child writes completed.")
            if request.messages[-1].role == Role.TOOL:
                return response(text="Child changes checked.")
            return response(calls=[delegate("write", turns=250)])

        self.provider.handler = handler
        result = await self.runner.run("Implement files across more than sixteen child turns")
        expected = {f"turn-{number}.txt" for number in range(1, 18)}
        self.assertEqual(set(result.changed_files), expected)
        self.assertTrue(all((self.root / name).read_text() == "child evidence\n" for name in expected))
        self.assertEqual(len(child_requests), 18)
        started = self.starts()
        self.assertEqual(len(started), 1)
        self.assertEqual(started[0].data["requested_max_turns"], 250)
        self.assertEqual(started[0].data["effective_max_turns"], 250)
        for request in child_requests:
            schema = next(tool for tool in request.tools if tool["name"] == "delegate_task")
            self.assertEqual(schema["parameters"]["properties"]["max_turns"]["maximum"], 250)

    async def test_default_parent_can_explicitly_delegate_real_write(self):
        def handler(request, call):
            if request.messages[-1].role == Role.TOOL:
                return response(text="Changes checked: " + request.messages[-1].content)
            if request.metadata.get("delegated"):
                self.assertIn("write_file", {t["name"] for t in request.tools})
                self.assertIn("Your maximum capability is write", request.system)
                return response(calls=[write()])
            return response(calls=[delegate("write")])

        self.provider.handler = handler
        result = await self.runner.run("Implement a disposable child file")
        self.assertTrue((self.root / "child.txt").exists(), result.text)
        self.assertEqual((self.root / "child.txt").read_text(), "child evidence\n")
        self.assertEqual(result.changed_files, ["child.txt"])
        self.assertEqual([e.data["capability"] for e in self.starts()], ["write"])
        row = next(r for r in self.runner.sessions.tool_calls(result.session_id) if r["tool_name"] == "delegate_task")
        self.assertEqual(row["metadata"]["capability"], "write")
        self.assertEqual(row["arguments"]["capability"], "write")
        self.assertEqual(self.view().nodes[row["metadata"]["delegation_id"]].capability, "write")

    async def test_explicit_reader_ignores_write_intent_and_blocks_forced_mutation(self):
        seen = []

        def handler(request, call):
            if request.messages[-1].role == Role.TOOL:
                if request.metadata.get("delegated"):
                    seen.append(request.messages[-1])
                return response(text="Read-only review concluded.")
            if request.metadata.get("delegated"):
                self.assertNotIn("write_file", {t["name"] for t in request.tools})
                self.assertIn("Your maximum capability is read-only", request.system)
                self.assertIn("capability", request.system)
                return response(calls=[write()])
            return response(calls=[delegate("read-only")])

        self.provider.handler = handler
        result = await self.runner.run("Implement the requested file")
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].is_error)
        self.assertIn("unavailable to this delegated capability", seen[0].content)
        self.assertFalse((self.root / "child.txt").exists())
        self.assertEqual(result.changed_files, [])
        self.assertEqual([e.data["capability"] for e in self.starts()], ["read-only"])
        row = next(r for r in self.runner.sessions.tool_calls(result.session_id) if r["tool_name"] == "delegate_task")
        self.assertEqual(row["metadata"]["capability"], "read-only")
        self.assertEqual(self.view().nodes[row["metadata"]["delegation_id"]].capability, "read-only")

    async def test_plan_and_operator_ceiling_reject_before_preparation_or_admission(self):
        for mode, ceiling in (("plan", True), ("workspace-write", False)):
            with self.subTest(mode=mode, ceiling=ceiling):
                self.events.clear()
                self.runner.config.safety.mode = mode
                self.runner.config.agent.swarm_allow_writes = ceiling
                self.provider.handler = lambda request, call: (
                    response(text="Refusal observed.") if request.messages[-1].role == Role.TOOL
                    else response(calls=[delegate("write")]))
                original = self.runner.context_builder.system_prompt
                with patch.object(self.runner.context_builder, "system_prompt", wraps=original) as prepare:
                    result = await self.runner.run("Check delegation ceiling")
                prepare.assert_not_called()  # The root uses its prebuilt context, not child preparation.
                self.assertEqual(self.starts(), [])
                refusal = next(e for e in self.events if e.type == "delegate.refused")
                self.assertEqual(refusal.data["limits"]["admitted"], 0)
                self.assertEqual(refusal.data["requested_capability"], "write")
                self.assertEqual(refusal.data["available_capability"], "read-only")
                row = next(r for r in self.runner.sessions.tool_calls(result.session_id) if r["tool_name"] == "delegate_task")
                self.assertEqual(row["status"], "error")
                self.assertIn("requested write", row["output"])
                self.assertIn("available read-only", row["output"])
                self.assertEqual(result.usage.requests, 2)
                self.assertFalse((self.root / "child.txt").exists())
                node = self.view().nodes[refusal.data["delegation_id"]]
                self.assertEqual(node.capability, "not-admitted")
                self.assertEqual(node.state, "refused")
                self.assertIn("requested write; available read-only", node.detail)

    async def test_refusal_leaves_last_descendant_available_for_reader(self):
        self.runner.config.agent.swarm_max_descendants = 1
        self.runner.config.agent.swarm_allow_writes = False
        self.provider.enqueue(
            response(calls=[delegate("write", identifier="denied")]),
            response(calls=[delegate("read-only", identifier="reader")]),
            response(text="Reader completed."),
            response(text="Root concluded."),
        )
        result = await self.runner.run("Refuse a writer, then admit a reader")
        self.assertEqual([e.data["capability"] for e in self.starts()], ["read-only"])
        self.assertEqual(self.starts()[0].data["limits"]["admitted"], 1)
        self.assertEqual(result.usage.requests, 4)
        self.assertFalse((self.root / "child.txt").exists())

    async def test_missing_inherited_permission_cannot_grant_write(self):
        for inherited in ({"delegated": True}, {"delegation_allow_writes": False}):
            with self.subTest(inherited=inherited):
                context = make_context(self.root, self.runner.config)
                state = SwarmState.start(context.config.agent)
                context.metadata.update(inherited, swarm_state=state)
                result = await self.runner.tools.execute(delegate("write"), context)
                self.assertTrue(result.is_error)
                self.assertIn("requested write; available read-only", result.output)
                self.assertEqual(state.admitted, 0)
                self.assertEqual(self.provider.calls, 0)
                self.assertFalse((self.root / "child.txt").exists())

    async def test_recursive_ceiling_at_each_generation(self):
        for selections in (("read-only", "write"), ("write", "read-only", "write"), ("write", "write", "write")):
            with self.subTest(selections=selections):
                self.events.clear()
                self.runner.config.agent.swarm_max_depth = 3
                (self.root / "child.txt").unlink(missing_ok=True)
                seen_depths = []

                def handler(request, call, seen_depths=seen_depths, selections=selections):
                    depth = request.metadata.get("delegation_depth", 0)
                    seen_depths.append(depth)
                    if request.messages[-1].role == Role.TOOL:
                        return response(text="Subtree checked: " + request.messages[-1].content)
                    if depth < len(selections):
                        return response(calls=[delegate(selections[depth])])
                    return response(calls=[write()])

                self.provider.handler = handler
                result = await self.runner.run("Check recursive authority")
                healthy = all(s == "write" for s in selections)
                expected = list(selections if healthy else selections[:-1])
                self.assertEqual([e.data["capability"] for e in self.starts()], expected)
                self.assertEqual(max(seen_depths), len(expected))
                self.assertEqual((self.root / "child.txt").exists(), healthy)
                self.assertEqual(result.changed_files, ["child.txt"] if healthy else [])
                if healthy:
                    self.assertEqual((self.root / "child.txt").read_text(), "child evidence\n")
                else:
                    refusal = next(e for e in self.events if e.type == "delegate.refused")
                    self.assertEqual(refusal.data["limits"]["admitted"], len(expected))
                    self.assertEqual(refusal.data["requested_capability"], "write")
                    self.assertEqual(refusal.data["available_capability"], "read-only")

    async def test_missing_and_malformed_capability_fail_closed(self):
        for value in ("missing", None, "read", "WRITE", True, {}, ["write"]):
            with self.subTest(value=value):
                self.events.clear()
                call = delegate(value)
                if value == "missing":
                    del call.arguments["capability"]  # Historic pending calls must not gain a default.
                self.provider.enqueue(response(calls=[call]), response(text="Invalid request rejected."))
                result = await self.runner.run("Reject malformed capability")
                self.assertEqual(self.starts(), [])
                self.assertFalse(any(e.type == "delegate.refused" for e in self.events))
                self.assertEqual(result.usage.requests, 2)
                row = next(r for r in self.runner.sessions.tool_calls(result.session_id) if r["tool_name"] == "delegate_task")
                self.assertEqual(row["status"], "error")
                self.assertEqual(row["metadata"]["error_type"], "ToolValidationError")
                self.assertFalse((self.root / "child.txt").exists())

    async def test_required_choice_reaches_root_child_and_discovery_on_both_surfaces(self):
        for surface in ("direct", "ptc"):
            for mode, ceiling, available in (("workspace-write", True, "write"), ("plan", True, "read-only"), ("full", False, "read-only")):
                with self.subTest(surface=surface, mode=mode, ceiling=ceiling):
                    self.runner.config.script.tool_surface = surface
                    self.runner.config.safety.mode = mode
                    self.runner.config.agent.swarm_allow_writes = ceiling
                    schemas = []

                    def handler(request, call, schemas=schemas, available=available):
                        child = request.metadata.get("delegated", False)
                        if request.tools:
                            schema = next(t for t in request.tools if t["name"] == "delegate_task")
                            self.assert_schema(schema, "read-only" if child else available)
                            schemas.append(schema)
                        if request.messages[-1].role == Role.TOOL:
                            if child:
                                discovered = json.loads(request.messages[-1].content)["tools"][0]
                                self.assert_schema(discovered, "read-only")
                                schemas.append(discovered)
                            return response(text="Schema checked.")
                        if child:
                            return response(calls=[ToolCall(name="discover_tools", arguments={"names": ["delegate_task"], "offset": 0})])
                        return response(calls=[delegate("read-only", task="Review delegation schema")])

                    self.provider.handler = handler
                    await self.runner.run("Inspect explicit capability schema")
                    self.assertGreaterEqual(len(schemas), 3)

    async def test_interactive_off_is_operator_ceiling_not_child_selection(self):
        cli = InteractiveCLI(workspace=self.root, config=self.runner.config, approval_callback=None)
        cli.runner = self.runner
        for setting in ("off", "on"):
            with self.subTest(setting=setting), patch.object(cli.ui, "notice"):
                await cli._command("/swarm-writes " + setting)
                assert cli.runner is not None
                self.addAsyncCleanup(cli.runner.close)
                self.assertEqual(cli.config.agent.swarm_allow_writes, setting == "on")
                provider = cli.runner.providers[0].provider
                assert isinstance(provider, MockProvider)
                provider.enqueue(
                    response(calls=[delegate("write" if setting == "off" else "read-only")]),
                    *([response(calls=[write()]), response(text="Reader rejected forced write.")] if setting == "on" else []),
                    response(text="Root observed the restriction."),
                )
                result = await cli.runner.run("Check interactive ceiling")
                rows = cli.runner.sessions.tool_calls(result.session_id)
                row = next(r for r in rows if r["tool_name"] == "delegate_task")
                if setting == "off":
                    self.assertEqual(row["status"], "error")
                    self.assertIn("requested write; available read-only", row["output"])
                else:
                    self.assertEqual(row["metadata"]["capability"], "read-only")
                    self.assertEqual(next(r for r in rows if r["tool_name"] == "write_file")["status"], "error")
                self.assertFalse((self.root / "child.txt").exists())
