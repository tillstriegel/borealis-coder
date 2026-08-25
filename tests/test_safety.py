from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from borealis_coder.config import SafetyConfig
from borealis_coder.errors import PathViolation, ToolError
from borealis_coder.models import Effect
from borealis_coder.safety import (
    ApprovalManager,
    ApprovalRequest,
    CheckpointManager,
    PolicyAction,
    PolicyDecision,
    PolicyEngine,
    WorkspaceRoots,
    assess_command,
)
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
            assert checkpoint is not None
            file_path.write_text("after", encoding="utf-8")
            manager.restore(checkpoint.id)
            self.assertEqual(file_path.read_text(encoding="utf-8"), "before")

    def test_checkpoint_refuses_to_replace_a_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            future = root / "future.txt"
            manager = CheckpointManager(WorkspaceRoots(root))
            checkpoint = manager.create([future], label="absent file")
            assert checkpoint is not None
            future.mkdir()
            with self.assertRaises(ToolError):
                manager.restore(checkpoint.id)
            self.assertTrue(future.is_dir())

    def test_checkpoint_restores_an_authorized_outside_path(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as outside:
            root = Path(td)
            file_path = Path(outside) / "state.txt"
            file_path.write_text("before", encoding="utf-8")
            roots = WorkspaceRoots(root, allow_outside=True)
            manager = CheckpointManager(roots)
            checkpoint = manager.create([file_path], label="outside root")
            assert checkpoint is not None

            file_path.write_text("after", encoding="utf-8")
            roots.allow_outside = False
            with self.assertRaises(PathViolation):
                manager.restore(checkpoint.id)
            self.assertEqual(file_path.read_text(encoding="utf-8"), "after")

            roots.allow_outside = True
            manager.restore(checkpoint.id)
            self.assertEqual(file_path.read_text(encoding="utf-8"), "before")

    def test_checkpoint_retention_respects_count_and_preserves_newest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "state.txt"
            source.write_text("one", encoding="utf-8")
            manager = CheckpointManager(
                WorkspaceRoots(root), retention_max_count=2, retention_max_bytes=10_000_000
            )
            first = manager.create([source], label="first")
            source.write_text("two", encoding="utf-8")
            second = manager.create([source], label="second")
            source.write_text("three", encoding="utf-8")
            newest = manager.create([source], label="newest")
            assert first is not None and second is not None and newest is not None

            self.assertEqual([item.id for item in manager.list()], [newest.id, second.id])
            source.write_text("changed", encoding="utf-8")
            manager.restore(newest.id)
            self.assertEqual(source.read_text(encoding="utf-8"), "three")

            records = manager._complete_records()
            manager.retention_max_bytes = max(size for _, _, size, _ in records)
            report = manager.prune()
            self.assertGreaterEqual(report["pruned_count"], 1)
            self.assertEqual(manager.list()[0].id, newest.id)

    def test_checkpoint_pruning_ignores_malformed_directories(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manager = CheckpointManager(WorkspaceRoots(root), retention_max_count=1)
            malformed = manager.directory / "interrupted"
            malformed.mkdir()
            (malformed / "manifest.json").write_text("not json", encoding="utf-8")
            source = root / "state.txt"
            source.write_text("value", encoding="utf-8")
            checkpoint = manager.create([source], label="valid")
            assert checkpoint is not None

            report = manager.prune()

            self.assertEqual(report["complete_checkpoints"], 1)
            self.assertTrue(malformed.is_dir())
            manager.restore(checkpoint.id)

    def test_checkpoint_pruning_skips_active_checkpoint_until_release(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "state.txt"
            source.write_text("first", encoding="utf-8")
            manager = CheckpointManager(
                WorkspaceRoots(root), retention_max_count=1
            )
            active = manager.create([source], label="active", active=True)
            assert active is not None
            source.write_text("second", encoding="utf-8")
            newest = manager.create([source], label="newest")
            assert newest is not None

            self.assertTrue((manager.directory / active.id).is_dir())
            manager.prune()
            self.assertTrue((manager.directory / active.id).is_dir())

            manager.release(active.id)

            self.assertFalse((manager.directory / active.id).exists())
            self.assertTrue((manager.directory / newest.id).is_dir())

    def test_checkpoint_retention_respects_age_and_preserves_newest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "state.txt"
            source.write_text("old", encoding="utf-8")
            manager = CheckpointManager(
                WorkspaceRoots(root), retention_max_age_seconds=60
            )
            old = manager.create([source], label="old")
            assert old is not None
            manifest_path = manager.directory / old.id / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["created_at"] = "2000-01-01T00:00:00+00:00"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            source.write_text("new", encoding="utf-8")
            newest = manager.create([source], label="newest")
            assert newest is not None

            self.assertFalse((manager.directory / old.id).exists())
            self.assertTrue((manager.directory / newest.id).is_dir())
            source.write_text("changed", encoding="utf-8")
            manager.restore(newest.id)
            self.assertEqual(source.read_text(encoding="utf-8"), "new")


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
        decision = PolicyDecision(PolicyAction.ASK, "r", "high", cache_key="k")
        request = ApprovalRequest("x", "x", decision, "{}")
        await manager.enforce(request)
        await manager.enforce(request)
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
