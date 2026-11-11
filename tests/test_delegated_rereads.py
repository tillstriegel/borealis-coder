from __future__ import annotations

import asyncio
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from borealis_coder.agent import build_runner, validate_tool_call_order
from borealis_coder.models import Role, ToolCall
from borealis_coder.providers.mock import MockProvider
from borealis_coder.tools.delegate import DelegateTaskTool
from borealis_coder.util import sha256_text
from tests.helpers import make_config, make_context
from tests.test_swarm import delegate, response, write


def read(identifier):
    return ToolCall(id=identifier, name="read_file", arguments={
        "path": "shared.txt", "start_line": None, "end_line": None, "max_chars": None,
    })


class DelegatedRereadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.runner = await build_runner(self.root, config=make_config(
            self.root, agent={"swarm_allow_writes": True, "swarm_max_turns": 12}), interactive=False)
        self.addAsyncCleanup(self.runner.close)
        provider = self.runner.providers[0].provider
        assert isinstance(provider, MockProvider)
        self.provider = provider

    async def exercise(self, batches, *, standalone=False, capability="write"):
        observed = []
        child_turn = 0

        def handler(request, call):
            nonlocal child_turn
            validate_tool_call_order(request.messages)
            if not request.metadata.get("delegated"):
                return response(text="Root complete.") if request.messages[-1].role == Role.TOOL else response(calls=[delegate(turns=12, capability=capability)])
            known = {m.tool_call_id for m in observed}
            observed.extend(copy.deepcopy(m) for m in request.messages
                            if m.role == Role.TOOL and m.tool_call_id not in known)
            child_turn += 1
            if child_turn <= len(batches):
                return response(calls=batches[child_turn - 1])
            return response(text="Changes inspected; child complete.")

        self.provider.handler = handler
        if standalone:
            context = make_context(self.root, self.runner.config)
            context.metadata.update(provider_routes=self.runner.providers, tool_registry=self.runner.tools,
                                    context_builder=self.runner.context_builder)
            result = await DelegateTaskTool().execute({"task": "inspect and edit", "max_turns": 12, "capability": capability}, context)
            self.assertFalse(result.is_error, result.output)
        else:
            result = await asyncio.wait_for(self.runner.run("Inspect and edit"), 10)
            self.assertEqual(result.stop_reason.value, "end_turn", result.error)
            rows = self.runner.sessions.tool_calls(result.session_id)
            self.assertEqual(len({r["tool_call_id"] for r in rows}), len(rows))
            self.assertTrue(all(r["metadata"].get("delegation_id") for r in rows if r["run_id"] != result.run_id))
        return {m.tool_call_id: m for m in observed}

    async def test_read_write_same_read_and_guarded_write_in_runner_and_standalone(self):
        for standalone in (False, True):
            with self.subTest(standalone=standalone):
                (self.root / "shared.txt").write_text("before")
                observed = await self.exercise([
                    [read("read-before")],
                    [write("shared.txt", "after", sha256_text("before"))],
                    [read("read-after")],
                    [ToolCall(id="write-again", name="write_file", arguments={
                        "path": "shared.txt", "content": "final", "expected_sha256": sha256_text("after")})],
                ], standalone=standalone)
                self.assertFalse(observed["read-after"].is_error, observed["read-after"].content)
                self.assertEqual(observed["read-before"].metadata["sha256"], sha256_text("before"))
                self.assertEqual(observed["read-after"].metadata["sha256"], sha256_text("after"))
                self.assertIn("after", observed["read-after"].content)
                self.assertFalse(observed["write-again"].is_error)
                self.assertEqual((self.root / "shared.txt").read_text(), "final")

    async def test_stale_hash_failure_then_same_path_reread_and_guarded_write(self):
        (self.root / "shared.txt").write_text("before")
        observed = await self.exercise([
            [read("read-before")],
            [write("shared.txt", "after", sha256_text("before"))],
            [ToolCall(id="stale", name="write_file", arguments={
                "path": "shared.txt", "content": "final", "expected_sha256": sha256_text("before")})],
            [ToolCall(id="stale-again", name="write_file", arguments={
                "path": "shared.txt", "content": "final", "expected_sha256": sha256_text("before")})],
            [read("read-after")],
            [ToolCall(id="retry", name="write_file", arguments={
                "path": "shared.txt", "content": "final", "expected_sha256": sha256_text("after")})],
        ])
        self.assertTrue(observed["stale"].is_error)
        self.assertIn("Stale write rejected", observed["stale"].content)
        self.assertTrue(observed["stale-again"].is_error)
        self.assertIn("suppressed", observed["stale-again"].content)
        self.assertFalse(observed["read-after"].is_error, observed["read-after"].content)
        self.assertEqual(observed["read-after"].metadata["sha256"], sha256_text("after"))
        self.assertFalse(observed["retry"].is_error)
        self.assertEqual((self.root / "shared.txt").read_text(), "final")

    async def test_repeated_failure_suppressed_but_retry_after_mutation_allowed(self):
        for standalone in (False, True):
            with self.subTest(standalone=standalone):
                (self.root / "shared.txt").unlink(missing_ok=True)
                observed = await self.exercise([
                    [read("missing")], [read("missing-again")],
                    [write("shared.txt", "created")], [read("now-present")],
                ], standalone=standalone)
                self.assertTrue(observed["missing"].is_error)
                self.assertIn("does not exist", observed["missing"].content)
                self.assertTrue(observed["missing-again"].is_error)
                self.assertIn("suppressed", observed["missing-again"].content)
                self.assertFalse(observed["now-present"].is_error, observed["now-present"].content)
                self.assertEqual(observed["now-present"].metadata["sha256"], sha256_text("created"))

    async def test_read_only_success_can_be_repeated_without_mutation(self):
        self.runner.config.agent.swarm_allow_writes = False
        (self.root / "shared.txt").write_text("unchanged")
        observed = await self.exercise([[read("first")], [read("second")]], capability="read-only")
        self.assertFalse(observed["second"].is_error, observed["second"].content)
        self.assertEqual(observed["first"].metadata["sha256"], observed["second"].metadata["sha256"])

    async def test_descendant_edit_then_same_parent_read(self):
        (self.root / "shared.txt").write_text("before")
        observed = []

        def handler(request, call):
            validate_tool_call_order(request.messages)
            depth = request.metadata.get("delegation_depth", 0)
            tools = [m for m in request.messages if m.role == Role.TOOL]
            if depth == 0:
                return response(text="Root complete.") if tools else response(calls=[delegate(turns=6, capability="write")])
            if depth == 2:
                return response(text="Descendant edited.") if tools else response(calls=[write("shared.txt", "descendant", sha256_text("before"))])
            if not tools:
                return response(calls=[read("first")])
            if len(tools) == 1:
                return response(calls=[delegate("descendant", identifier="nested", capability="write")])
            if len(tools) == 2:
                return response(calls=[read("second")])
            observed.extend(tools)
            return response(text="Descendant revision inspected.")

        self.provider.handler = handler
        result = await asyncio.wait_for(self.runner.run("Nested edit"), 10)
        self.assertEqual(result.stop_reason.value, "end_turn", result.error)
        self.assertFalse(observed[-1].is_error, observed[-1].content)
        self.assertEqual(observed[-1].metadata["sha256"], sha256_text("descendant"))
        self.assertEqual((self.root / "shared.txt").read_text(), "descendant")

    async def test_gated_sibling_mutation_invalidates_failed_read_suppression(self):
        read_failed = asyncio.Event()
        written = asyncio.Event()
        observations = []
        original = self.provider.complete

        async def controlled(request):
            if request.metadata.get("delegated"):
                task = request.messages[0].content
                tools = [m for m in request.messages if m.role == Role.TOOL]
                if task == "reader" and tools:
                    observations.append(tools[-1])
                    if len(tools) == 1:
                        read_failed.set()
                        await written.wait()
                    else:
                        return response(text="Reader saw sibling file.")
                if task == "writer":
                    await read_failed.wait()
                    if tools:
                        written.set()
                        return response(text="Writer done.")
            return await original(request)

        def handler(request, call):
            if not request.metadata.get("delegated"):
                return response(text="Root complete.") if request.messages[-1].role == Role.TOOL else response(calls=[
                    delegate("reader", turns=4, identifier="reader"), delegate("writer", identifier="writer", capability="write")])
            if request.messages[0].content == "writer":
                return response(calls=[write("shared.txt", "sibling")])
            count = sum(m.role == Role.TOOL for m in request.messages)
            return response(calls=[read("first" if not count else "second")])

        self.provider.handler = handler
        with patch.object(self.provider, "complete", side_effect=controlled):
            result = await asyncio.wait_for(self.runner.run("Sibling repair"), 10)
        self.assertEqual(result.stop_reason.value, "end_turn", result.error)
        self.assertEqual(len(observations), 2)
        self.assertTrue(observations[0].is_error)
        self.assertFalse(observations[1].is_error, observations[1].content)
        self.assertEqual(observations[1].metadata["sha256"], sha256_text("sibling"))
