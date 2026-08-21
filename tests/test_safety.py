from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from borealis_coder.config import SafetyConfig
from borealis_coder.errors import PathViolation, ToolError
from borealis_coder.models import Effect
from borealis_coder.safety import ApprovalManager, ApprovalRequest, CheckpointManager, PolicyAction, PolicyEngine, WorkspaceRoots, assess_command
from borealis_coder.safety.commands import CommandRisk


class PathTests(unittest.TestCase):
    def test_traversal_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as outside:
            root = Path(td)
            roots = WorkspaceRoots(root)
            with self.assertRaises(PathViolation):
                roots.resolve("../outside.txt")
            if hasattr(os, "symlink"):
                try:
                    (root / "link").symlink_to(Path(outside), target_is_directory=True)
                except OSError:
                    return
                with self.assertRaises(PathViolation):
                    roots.resolve("link/secret.txt")

    def test_additional_root(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as extra:
            roots = WorkspaceRoots(Path(td), [Path(extra)])
            resolved = roots.resolve(Path(extra) / "x.txt")
            self.assertEqual(resolved.root, Path(extra).resolve())

    def test_protected_path_patterns(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            roots = WorkspaceRoots(root)
            with self.assertRaises(PathViolation):
                roots.assert_writable(root / ".borealis/checkpoints/x", [".borealis/checkpoints/**"])
            roots.assert_writable(root / "src/main.py", [".git/**"])

    def test_checkpoint_restores_an_additional_root(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as extra:
            root = Path(td)
            extra_root = Path(extra)
            file_path = extra_root / "state.txt"
            file_path.write_text("before", encoding="utf-8")
            manager = CheckpointManager(WorkspaceRoots(root, [extra_root]))
            checkpoint = manager.create([file_path], label="additional root")
            self.assertIsNotNone(checkpoint)
            file_path.write_text("after", encoding="utf-8")
            manager.restore(checkpoint.id)
            self.assertEqual(file_path.read_text(encoding="utf-8"), "before")

    def test_checkpoint_refuses_to_replace_a_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            future = root / "future.txt"
            manager = CheckpointManager(WorkspaceRoots(root))
            checkpoint = manager.create([future], label="absent file")
            self.assertIsNotNone(checkpoint)
            future.mkdir()
            with self.assertRaises(ToolError):
                manager.restore(checkpoint.id)
            self.assertTrue(future.is_dir())


class PolicyTests(unittest.IsolatedAsyncioTestCase):
    def test_command_classification(self):
        self.assertEqual(assess_command("git status").risk, CommandRisk.SAFE)
        self.assertEqual(assess_command("curl https://example.com").risk, CommandRisk.NETWORK)
        self.assertEqual(assess_command("rm -rf build").risk, CommandRisk.DESTRUCTIVE)
        self.assertEqual(assess_command("sudo true").risk, CommandRisk.FORBIDDEN)
        self.assertEqual(assess_command("npm test").risk, CommandRisk.SAFE)
        self.assertEqual(assess_command("cargo test").risk, CommandRisk.NETWORK)
        self.assertEqual(assess_command("cargo test --offline").risk, CommandRisk.SAFE)
        self.assertTrue(assess_command("python3 -c 'print(1)'").can_open_network)
        self.assertTrue(assess_command("env python3 -c 'print(1)'").can_open_network)

    def test_plan_mode_allows_plan_updates_only(self):
        policy = PolicyEngine(SafetyConfig(mode="plan", approval="never"), interactive=False)
        self.assertEqual(policy.decide(tool_name="update_plan", effect=Effect.CONTROL, arguments={}).action, PolicyAction.ALLOW)
        self.assertEqual(policy.decide(tool_name="write_file", effect=Effect.WRITE, arguments={}).action, PolicyAction.DENY)

    def test_network_and_commit_are_hard_gated(self):
        policy = PolicyEngine(SafetyConfig(network=False, allow_git_commit=False), interactive=True)
        self.assertEqual(policy.decide(tool_name="shell", effect=Effect.EXECUTE, arguments={"command":"git commit -m x"}).action, PolicyAction.DENY)
        self.assertEqual(policy.decide(tool_name="shell", effect=Effect.EXECUTE, arguments={"command":"curl https://x"}).action, PolicyAction.DENY)

    def test_native_network_false_rejects_arbitrary_code(self):
        policy = PolicyEngine(
            SafetyConfig(network=False, approval="never"),
            interactive=False,
        )
        decision = policy.decide(
            tool_name="shell",
            effect=Effect.EXECUTE,
            arguments={"command": "python3 -c 'import socket'"},
        )
        self.assertEqual(decision.action, PolicyAction.DENY)
        self.assertIn("cannot enforce network=false", decision.reason)

        isolated = PolicyEngine(
            SafetyConfig(network=False, approval="never"),
            interactive=False,
            process_network_isolated=True,
        )
        decision = isolated.decide(
            tool_name="shell",
            effect=Effect.EXECUTE,
            arguments={"command": "python3 -c 'print(1)'"},
        )
        self.assertEqual(decision.action, PolicyAction.ALLOW)

    async def test_approval_callback_and_cache(self):
        calls = 0
        async def callback(request):
            nonlocal calls
            calls += 1
            return "allow_always"
        manager = ApprovalManager(callback)
        decision = type("D", (), {"action": PolicyAction.ASK, "reason":"r", "cache_key":"k", "risk":"high"})()
        request = ApprovalRequest("x", "x", decision, "{}")
        await manager.enforce(request)
        await manager.enforce(request)
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
