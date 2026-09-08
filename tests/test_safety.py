from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import patch

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
    def test_path_objects_preserve_literal_names_while_strings_expand_variables(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"BOREALIS_PATH_FIXTURE": "other"}):
            root = Path(td).resolve()
            literal = root / "$BOREALIS_PATH_FIXTURE.txt"
            other = root / "other.txt"
            literal.write_text("literal before")
            other.write_text("other untouched")
            roots = WorkspaceRoots(root)

            self.assertEqual(roots.resolve(literal, must_exist=True).path, literal)
            self.assertEqual(roots.resolve(Path(literal.name), must_exist=True).path, literal)
            self.assertEqual(roots.resolve(literal.name, must_exist=True).path, other)
            self.assertEqual(roots.resolve(literal).display, literal.name)
            manager = CheckpointManager(roots)
            checkpoint = manager.create([literal], label="literal file name")
            assert checkpoint is not None
            self.assertEqual(checkpoint.files[0]["path"], literal.name)
            literal.write_text("literal after")

            manager.restore(checkpoint.id)

            self.assertEqual(literal.read_text(), "literal before")
            self.assertEqual(other.read_text(), "other untouched")

    def test_failed_checkpoint_removes_partial_blobs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = [root / "a.txt", root / "b.txt"]
            for path in paths:
                path.write_text("12345678")
            manager = CheckpointManager(WorkspaceRoots(root), max_bytes=10)
            with self.assertRaisesRegex(ToolError, "byte limit"):
                manager.create(paths, label="too large")
            self.assertEqual(manager.list(), [])
            self.assertEqual(list(manager.directory.iterdir()), [])
            self.assertTrue(all(path.read_text() == "12345678" for path in paths))

    def test_checkpoint_manifest_failure_preserves_existing_recovery_points(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "state.txt"
            source.write_text("before")
            manager = CheckpointManager(WorkspaceRoots(root))
            previous = manager.create([source], label="previous")
            assert previous is not None
            with (
                patch("borealis_coder.safety.checkpoints.atomic_write_text", side_effect=OSError("disk full")),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                manager.create([source], label="failed")
            self.assertEqual([path.name for path in manager.directory.iterdir()], [previous.id])
            source.write_text("after")
            manager.restore(previous.id)
            self.assertEqual(source.read_text(), "before")

    def test_checkpoint_reads_only_the_remaining_byte_budget(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = root / "a.txt"
            second = root / "b.txt"
            first.write_bytes(b"12345678")
            second.write_bytes(b"more")
            manager = CheckpointManager(WorkspaceRoots(root), max_bytes=10)
            reads: list[int | None] = []

            class ObservedFile(io.BytesIO):
                def read(self, size: int | None = -1):
                    reads.append(size)
                    return super().read(size)

            original_open = Path.open

            def open_file(path, *args, **kwargs):
                mode = kwargs.get("mode", args[0] if args else "r")
                if path == second.resolve() and mode == "rb":
                    return ObservedFile(b"more than the remaining budget")
                return original_open(path, *args, **kwargs)

            with (
                patch.object(Path, "open", new=open_file),
                self.assertRaisesRegex(ToolError, "byte limit"),
            ):
                manager.create([first, second], label="bounded")
            self.assertEqual(reads, [3])

    @unittest.skipIf(os.name == "nt", "Long paths require Windows long-path support")
    def test_checkpoint_restores_files_with_long_relative_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / ("a" * 80) / ("b" * 80) / ("c" * 80) / "state.txt"
            target.parent.mkdir(parents=True)
            target.write_text("before")
            manager = CheckpointManager(WorkspaceRoots(root))
            checkpoint = manager.create([target], label="long path")
            assert checkpoint is not None
            target.write_text("after")
            manager.restore(checkpoint.id)
            self.assertEqual(target.read_text(), "before")

    def test_checkpoint_restores_legacy_blob_names(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "state.txt"
            source.write_text("before")
            manager = CheckpointManager(WorkspaceRoots(root))
            checkpoint = manager.create([source], label="legacy")
            assert checkpoint is not None
            directory = manager.directory / checkpoint.id
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            entry = manifest["files"][0]
            legacy_blob = "files/MDpzdGF0ZS50eHQ.bin"
            (directory / entry["blob"]).rename(directory / legacy_blob)
            entry["blob"] = legacy_blob
            manifest_path.write_text(json.dumps(manifest))
            source.write_text("after")
            manager.restore(checkpoint.id)
            self.assertEqual(source.read_text(), "before")
            self.assertEqual(manager.prune()["complete_checkpoints"], 1)

    def test_checkpoint_retention_hashes_blobs_in_bounded_chunks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "large.bin"
            source.write_bytes(b"x" * 1_000_000)
            manager = CheckpointManager(WorkspaceRoots(root))
            manager.create([source], label="large")
            reads: list[int] = []

            class BoundedFile(io.BufferedReader):
                def read(self, size: int | None = -1):
                    if size is None or size < 0:
                        raise AssertionError("Retention loaded an entire blob")
                    reads.append(size)
                    return super().read(size)

                def readinto(self, buffer):
                    reads.append(len(buffer))
                    return super().readinto(buffer)

            original_open = Path.open

            def open_file(path, *args, **kwargs):
                handle = original_open(path, *args, **kwargs)
                mode = kwargs.get("mode", args[0] if args else "r")
                if path.suffix == ".bin" and mode == "rb":
                    return BoundedFile(handle)
                return handle

            with patch.object(Path, "open", new=open_file):
                report = manager.prune(dry_run=True)
            self.assertEqual(report["complete_checkpoints"], 1)
            self.assertTrue(reads)
            self.assertLess(max(reads), source.stat().st_size)

    def test_checkpoint_retention_preserves_recovery_when_newer_blob_is_corrupt(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "state.txt"
            source.write_text("before")
            manager = CheckpointManager(WorkspaceRoots(root))
            previous = manager.create([source], label="previous")
            source.write_text("latest")
            latest = manager.create([source], label="latest")
            assert previous is not None and latest is not None
            blob = manager.directory / latest.id / latest.files[0]["blob"]
            blob.write_text("broken")
            manager.retention_max_count = 1
            report = manager.prune()
            self.assertEqual(report["complete_checkpoints"], 1)
            manager.restore(previous.id)
            self.assertEqual(source.read_text(), "before")

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

    def test_checkpoint_invalid_mode_fails_before_any_restore(self):
        for mode in ("invalid", True, -1, 0o200000, [], {}):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                paths = [root / "a.txt", root / "b.txt"]
                for path in paths:
                    path.write_text("before")
                manager = CheckpointManager(WorkspaceRoots(root), retention_max_count=2)
                previous = manager.create(paths, label="valid")
                malformed = manager.create(paths, label="invalid mode")
                assert previous is not None and malformed is not None
                manifest = manager.directory / malformed.id / "manifest.json"
                data = json.loads(manifest.read_text())
                data["files"][1]["mode"] = mode
                manifest.write_text(json.dumps(data))
                for path in paths:
                    path.write_text("after")

                with self.assertRaisesRegex(ToolError, "invalid file mode"):
                    manager.restore(malformed.id)
                self.assertEqual([path.read_text() for path in paths], ["after", "after"])

                manager.create(paths, label="newest")
                self.assertEqual(manager.prune(dry_run=True)["complete_checkpoints"], 2)
                manager.restore(previous.id)
                self.assertEqual([path.read_text() for path in paths], ["before", "before"])

    def test_checkpoint_restores_permission_only_and_legacy_missing_modes(self):
        for mode in (None, 0o600, 0o100600):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                source = root / "state.txt"
                source.write_text("before")
                manager = CheckpointManager(WorkspaceRoots(root))
                checkpoint = manager.create([source], label="compatible mode")
                assert checkpoint is not None
                manifest = manager.directory / checkpoint.id / "manifest.json"
                data = json.loads(manifest.read_text())
                if mode is None:
                    data["files"][0].pop("mode")
                else:
                    data["files"][0]["mode"] = mode
                manifest.write_text(json.dumps(data))
                source.write_text("after")

                manager.restore(checkpoint.id)

                self.assertEqual(source.read_text(), "before")
                if mode is not None and os.name != "nt":
                    self.assertEqual(source.stat().st_mode & 0o777, 0o600)
                self.assertEqual(manager.prune(dry_run=True)["complete_checkpoints"], 1)

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

    def test_checkpoint_follows_additional_root_after_reordering(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "primary"
            first = base / "one" / "shared"
            second = base / "two" / "shared"
            for directory in (root, first, second):
                directory.mkdir(parents=True)
            source = first / "state.txt"
            other = second / "state.txt"
            created = first / "new.txt"
            source.write_text("before")
            other.write_text("untouched")
            manager = CheckpointManager(WorkspaceRoots(root, [first, second]))
            checkpoint = manager.create([source, created], label="root reorder")
            assert checkpoint is not None
            source.write_text("after")
            created.write_text("new")
            (second / "new.txt").write_text("also untouched")

            resumed = CheckpointManager(WorkspaceRoots(root, [second, first]))
            resumed.restore(checkpoint.id)

            self.assertEqual(source.read_text(), "before")
            self.assertFalse(created.exists())
            self.assertEqual(other.read_text(), "untouched")
            self.assertEqual((second / "new.txt").read_text(), "also untouched")

    def test_checkpoint_directory_conflict_fails_before_any_restore(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first, second = root / "a.txt", root / "z.txt"
            for path in (first, second):
                path.write_text("before")
            manager = CheckpointManager(WorkspaceRoots(root))
            checkpoint = manager.create([first, second], label="directory conflict")
            assert checkpoint is not None
            first.write_text("after")
            second.unlink()
            second.mkdir()

            with self.assertRaisesRegex(ToolError, "non-file path"):
                manager.restore(checkpoint.id)

            self.assertEqual(first.read_text(), "after")
            self.assertTrue(second.is_dir())

    def test_checkpoint_rejects_redirected_paths_before_any_restore(self):
        for existed, parent_link, legacy in product((False, True), repeat=3):
            with (
                self.subTest(existed=existed, parent_link=parent_link, legacy=legacy),
                tempfile.TemporaryDirectory() as td,
            ):
                root = Path(td)
                first = root / "a-first.txt"
                first.write_text("first before")
                source = root / "source" / "state.txt"
                source.parent.mkdir()
                if existed:
                    source.write_text("source before")
                target = root / "target" / "state.txt"
                target.parent.mkdir()
                target.write_text("target untouched")
                manager = CheckpointManager(WorkspaceRoots(root))
                checkpoint = manager.create([first, source], label="redirected path")
                assert checkpoint is not None
                if legacy:
                    manifest = manager.directory / checkpoint.id / "manifest.json"
                    data = json.loads(manifest.read_text())
                    for entry in data["files"]:
                        entry.pop("root", None)
                        entry.pop("relative_path", None)
                    manifest.write_text(json.dumps(data))
                first.write_text("first after")
                source.unlink(missing_ok=True)
                try:
                    if parent_link:
                        source.parent.rmdir()
                        source.parent.symlink_to(target.parent, target_is_directory=True)
                    else:
                        source.symlink_to(target)
                except OSError as error:
                    self.skipTest(f"Symlinks are not available: {error}")

                with self.assertRaisesRegex(ToolError, "resolves to a different location"):
                    manager.restore(checkpoint.id)

                self.assertEqual(first.read_text(), "first after")
                self.assertEqual(target.read_text(), "target untouched")
                self.assertTrue((source.parent if parent_link else source).is_symlink())

    def test_checkpoint_keeps_original_resolved_symlink_target(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "target.txt"
            target.write_text("before")
            alias = root / "alias.txt"
            try:
                alias.symlink_to(target)
            except OSError as error:
                self.skipTest(f"Symlinks are not available: {error}")
            manager = CheckpointManager(WorkspaceRoots(root))
            checkpoint = manager.create([alias], label="existing alias")
            assert checkpoint is not None
            target.write_text("after")

            manager.restore(checkpoint.id)

            self.assertEqual(target.read_text(), "before")
            self.assertTrue(alias.is_symlink())

    def test_checkpoint_validates_parent_conflicts_but_allows_missing_directories(self):
        for parent_state in ("missing", "file"):
            with self.subTest(parent_state=parent_state), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                first = root / "a.txt"
                second = root / "folder" / "nested" / "z.txt"
                second.parent.mkdir(parents=True)
                for path in (first, second):
                    path.write_text("before")
                manager = CheckpointManager(WorkspaceRoots(root))
                checkpoint = manager.create([first, second], label="parent conflict")
                assert checkpoint is not None
                first.write_text("after")
                second.unlink()
                second.parent.rmdir()
                (root / "folder").rmdir()
                if parent_state == "file":
                    (root / "folder").write_text("keep this file")

                if parent_state == "file":
                    with self.assertRaisesRegex(ToolError, "parent path is not a directory"):
                        manager.restore(checkpoint.id)
                    self.assertEqual(first.read_text(), "after")
                    self.assertEqual((root / "folder").read_text(), "keep this file")
                else:
                    manager.restore(checkpoint.id)
                    self.assertEqual(first.read_text(), "before")
                    self.assertEqual(second.read_text(), "before")

    def test_checkpoint_missing_additional_root_fails_before_any_restore(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root, extra, replacement = (base / name for name in ("a", "b", "c"))
            for directory in (root, extra, replacement):
                directory.mkdir()
                (directory / "state.txt").write_text("before")
            paths = [root / "state.txt", extra / "state.txt"]
            manager = CheckpointManager(WorkspaceRoots(root, [extra]))
            checkpoint = manager.create(paths, label="removed root")
            assert checkpoint is not None
            for path in paths:
                path.write_text("after")

            resumed = CheckpointManager(WorkspaceRoots(root, [replacement]))
            with self.assertRaisesRegex(ToolError, "root is not configured"):
                resumed.restore(checkpoint.id)

            for path in paths:
                self.assertEqual(path.read_text(), "after")
            self.assertEqual((replacement / "state.txt").read_text(), "before")

    def test_legacy_checkpoint_restores_with_original_root_order(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            extra = root / "extra"
            extra.mkdir()
            source = extra / "state.txt"
            source.write_text("before")
            manager = CheckpointManager(WorkspaceRoots(root, [extra]))
            checkpoint = manager.create([source], label="legacy root")
            assert checkpoint is not None
            manifest = manager.directory / checkpoint.id / "manifest.json"
            data = json.loads(manifest.read_text())
            for entry in data["files"]:
                entry.pop("root_path")
            manifest.write_text(json.dumps(data))
            source.write_text("after")

            manager.restore(checkpoint.id)
            self.assertEqual(source.read_text(), "before")

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

    def test_checkpoint_creation_preserves_newest_when_timestamps_match(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "state.txt"
            source.write_text("first", encoding="utf-8")
            manager = CheckpointManager(
                WorkspaceRoots(root), retention_max_count=1
            )
            created_at = "2026-08-25T12:00:00.000+00:00"

            with (
                patch(
                    "borealis_coder.safety.checkpoints.utc_now",
                    return_value=created_at,
                ),
                patch(
                    "borealis_coder.safety.checkpoints.time.time_ns",
                    side_effect=[1, 2],
                ),
            ):
                first = manager.create([source], label="first")
                source.write_text("second", encoding="utf-8")
                newest = manager.create([source], label="newest")
            assert first is not None and newest is not None

            self.assertFalse((manager.directory / first.id).exists())
            self.assertTrue((manager.directory / newest.id).is_dir())
            source.write_text("changed", encoding="utf-8")
            manager.restore(newest.id)
            self.assertEqual(source.read_text(encoding="utf-8"), "second")

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

    def test_checkpoint_malformed_metadata_does_not_block_other_recovery_points(self):
        for field, value in (
            ("", []), ("", None), ("id", 42), ("created_at", 42),
            ("label", []), ("files", {}), ("files", None),
            ("files", [None]), ("files", [{}]),
            ("files", [{"path": "state.txt", "existed": "false"}]),
        ):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                source = root / "state.txt"
                source.write_text("before")
                manager = CheckpointManager(WorkspaceRoots(root), retention_max_count=2)
                previous = manager.create([source], label="valid")
                malformed = manager.create([source], label="malformed")
                assert previous is not None and malformed is not None
                manifest = manager.directory / malformed.id / "manifest.json"
                data = json.loads(manifest.read_text())
                if field:
                    data[field] = value
                else:
                    data = value
                manifest.write_text(json.dumps(data))
                source.write_text("after")

                self.assertEqual([item.id for item in manager.list()], [previous.id])
                with self.assertRaisesRegex(ToolError, "checkpoint"):
                    manager.restore(malformed.id)
                self.assertEqual(source.read_text(), "after")
                newest = manager.create([source], label="newest")
                assert newest is not None
                self.assertEqual({item.id for item in manager.list()}, {previous.id, newest.id})
                self.assertEqual(manager.prune(dry_run=True)["complete_checkpoints"], 2)
                self.assertTrue(manifest.is_file())
                manager.restore(previous.id)
                self.assertEqual(source.read_text(), "before")

    def test_checkpoint_invalid_path_references_cannot_displace_usable_recovery(self):
        invalid_fields = (
            ("root", "invalid"), ("root", True), ("root", -1),
            ("root", None), ("relative_path", None), ("relative_path", 123),
            ("relative_path", ""), ("relative_path", "../state.txt"),
            ("relative_path", str(Path.cwd() / "absolute.txt")),
            ("relative_path", "bad\x00path"),
            ("root_path", 123), ("root_path", "relative/root"),
            ("root_path", "bad\x00path"), ("path", ""), ("path", "bad\x00path"),
        )
        for field, value in invalid_fields:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                source = root / "state.txt"
                source.write_text("before")
                manager = CheckpointManager(WorkspaceRoots(root), retention_max_count=2)
                previous = manager.create([source], label="usable")
                malformed = manager.create([source], label="malformed reference")
                assert previous is not None and malformed is not None
                manifest = manager.directory / malformed.id / "manifest.json"
                data = json.loads(manifest.read_text())
                if value is None:
                    data["files"][0].pop(field)
                else:
                    data["files"][0][field] = value
                manifest.write_text(json.dumps(data))
                source.write_text("after")

                with self.assertRaisesRegex(ToolError, "checkpoint"):
                    manager.restore(malformed.id)
                self.assertEqual(source.read_text(), "after")
                self.assertFalse((root / "123").exists())
                self.assertEqual([item.id for item in manager.list()], [previous.id])
                manager.retention_max_count = 1
                report = manager.prune()
                self.assertEqual(report["complete_checkpoints"], 1)
                self.assertEqual(report["pruned_count"], 0)
                self.assertTrue(manifest.is_file())
                manager.restore(previous.id)
                self.assertEqual(source.read_text(), "before")

    def test_checkpoint_legacy_optional_metadata_and_empty_file_list(self):
        with tempfile.TemporaryDirectory() as td:
            manager = CheckpointManager(WorkspaceRoots(Path(td)))
            checkpoint = manager.create([], label="empty")
            assert checkpoint is not None
            manifest = manager.directory / checkpoint.id / "manifest.json"
            data = json.loads(manifest.read_text())
            data.pop("label")
            data.pop("creation_order")
            manifest.write_text(json.dumps(data))

            restored = manager.restore(checkpoint.id)

            self.assertEqual(restored.files, [])
            self.assertEqual(restored.label, "")
            self.assertEqual(restored.creation_order, "")
            self.assertEqual([item.id for item in manager.list()], [checkpoint.id])
            self.assertEqual(manager.prune(dry_run=True)["complete_checkpoints"], 1)

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
