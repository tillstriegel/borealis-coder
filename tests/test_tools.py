from __future__ import annotations

import asyncio
import io
import os
import subprocess
import tempfile
import threading
import tracemalloc
import unittest
from difflib import unified_diff
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from borealis_coder.errors import ToolError, ToolValidationError
from borealis_coder.models import ToolCall, ToolResult
from borealis_coder.providers.mock import MockProvider
from borealis_coder.tools import build_builtin_registry, validate_schema
from borealis_coder.tools.base import FunctionTool, ToolRegistry, object_schema
from borealis_coder.tools.fetch import _fetch_public_url, _validate_public_url
from borealis_coder.util import sha256_bytes, sha256_text
from tests.helpers import make_context


class SchemaTests(unittest.TestCase):
    def test_strict_tool_schemas(self):
        registry = build_builtin_registry()
        for tool in registry.schemas():
            schema = tool["parameters"]
            self.assertEqual(schema.get("type"), "object", tool["name"])
            self.assertFalse(schema.get("additionalProperties"), tool["name"])
            self.assertEqual(set(schema.get("properties", {})), set(schema.get("required", [])), tool["name"])

    def test_validator(self):
        schema = {"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":False}
        validate_schema({"x":1}, schema)
        with self.assertRaises(ToolValidationError):
            validate_schema({"x":"1"}, schema)
        with self.assertRaises(ToolValidationError):
            validate_schema({"x":1,"y":2}, schema)


class ToolExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_calls_keep_their_identity_and_share_run_changes(self):
        with tempfile.TemporaryDirectory() as td:
            context = make_context(Path(td))
            entered = 0
            ready = asyncio.Event()

            async def identify(arguments, call_context):
                nonlocal entered
                entered += 1
                if entered == 2:
                    ready.set()
                await ready.wait()
                call_context.changed_files.add(call_context.tool_call_id)
                call_context.metadata[call_context.tool_call_id] = True
                call_context.mutation_tracking = "incomplete"
                call_context.lifecycle_uncertainty_only = call_context.tool_call_id == "first"
                return ToolResult(call_context.tool_call_id)

            registry = ToolRegistry([FunctionTool(
                name="identify", description="Report the current call", parameters=object_schema({}),
                function=identify, concurrent=True,
            )])
            calls = [ToolCall(id=value, name="identify", arguments={}) for value in ("first", "second")]
            results = await asyncio.wait_for(
                asyncio.gather(*(registry.execute(call, context) for call in calls)), timeout=2,
            )

            self.assertEqual([result.output for result in results], [call.id for call in calls])
            self.assertEqual(context.changed_files, {"first", "second"})
            self.assertTrue(context.metadata["first"])
            self.assertTrue(context.metadata["second"])
            self.assertEqual(context.mutation_tracking, "incomplete")
            self.assertFalse(context.lifecycle_uncertainty_only)

    async def test_call_uncertainty_survives_failure_and_cancellation(self):
        for failure in (ToolError, asyncio.CancelledError):
            for lifecycle_only in (False, True):
                with self.subTest(failure=failure, lifecycle_only=lifecycle_only), tempfile.TemporaryDirectory() as td:
                    context = make_context(Path(td))

                    async def uncertain(arguments, call_context, failure=failure, lifecycle_only=lifecycle_only):
                        call_context.mutation_tracking = "incomplete"
                        call_context.lifecycle_uncertainty_only = lifecycle_only
                        call_context.changed_roots.add(call_context.workspace)
                        raise failure("interrupted")

                    registry = ToolRegistry([FunctionTool(
                        name="uncertain", description="Track an interrupted operation",
                        parameters=object_schema({}), function=uncertain,
                    )])
                    call = ToolCall(name="uncertain", arguments={})
                    if failure is asyncio.CancelledError:
                        with self.assertRaises(asyncio.CancelledError):
                            await registry.execute(call, context)
                    else:
                        result = await registry.execute(call, context)
                        self.assertTrue(result.is_error)
                    self.assertEqual(context.mutation_tracking, "incomplete")
                    self.assertEqual(context.lifecycle_uncertainty_only, lifecycle_only)
                    self.assertEqual(context.changed_roots, {context.workspace})


class FileToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.context = make_context(self.root)
        self.registry = build_builtin_registry()

    async def test_small_file_read_does_not_allocate_its_configured_ceiling(self):
        target = self.root / "small.txt"
        target.write_bytes(b"x" * 1024)
        self.context.config.context.max_file_bytes = 25_000_000
        tracemalloc.start()
        try:
            result = await self.call("read_file", {
                "path": target.name, "start_line": None, "end_line": None, "max_chars": None,
            })
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertFalse(result.is_error, result.output)
        self.assertEqual(result.metadata["sha256"], sha256_bytes(b"x" * 1024))
        self.assertLess(peak, 1_000_000)

    async def test_write_and_delete_stream_large_preimage_hashes(self):
        payload = b"x" * (2 * 1024 * 1024)
        digest = sha256_bytes(payload)
        target = self.root / "large.bin"
        self.context.config.safety.checkpoints = False
        self.context.checkpoints.enabled = False
        for name in ("write_file", "delete_file"):
            with self.subTest(tool=name):
                target.write_bytes(payload)
                arguments = {"path": target.name, "expected_sha256": digest}
                if name == "write_file":
                    arguments["content"] = "replacement"
                tracemalloc.start()
                try:
                    result = await self.call(name, arguments)
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                self.assertFalse(result.is_error, result.output)
                self.assertLess(peak, 1_000_000)
                if name == "write_file":
                    self.assertEqual(target.read_text(), "replacement")
                else:
                    self.assertFalse(target.exists())

    @unittest.skipIf(os.name == "nt", "Long paths require Windows long-path support")
    async def test_atomic_write_preserves_legal_long_file_names(self):
        target = self.root / ("x" * 250 + ".py")
        target.write_text("before")
        result = await self.call("write_file", {
            "path": target.name, "content": "after", "expected_sha256": sha256_text("before"),
        })
        self.assertFalse(result.is_error, result.output)
        self.assertEqual(target.read_text(), "after")

    async def test_write_rejects_a_deleted_empty_file(self):
        target = self.root / "empty.txt"
        target.touch()
        observed = await self.call("read_file", {
            "path": target.name, "start_line": 1, "end_line": None, "max_chars": 100,
        })
        self.assertFalse(observed.is_error, observed.output)
        target.unlink()
        result = await self.call("write_file", {
            "path": target.name, "content": "replacement",
            "expected_sha256": observed.metadata["sha256"],
        })
        self.assertTrue(result.is_error, result.output)
        self.assertIn("Stale write rejected", result.output)
        self.assertFalse(target.exists())

    async def test_text_edits_enforce_byte_limits_during_reads(self):
        target = self.root / "bounded.txt"
        target.write_text("old\n")
        self.context.config.context.max_file_bytes = 8
        calls = [
            ("replace_in_file", {
                "path": target.name, "old_text": "old", "new_text": "new",
                "expected_occurrences": 1, "expected_sha256": None,
            }),
            ("apply_patch", {
                "patch": "*** Begin Patch\n*** Update File: bounded.txt\n@@\n-old\n+new\n*** End Patch\n",
            }),
        ]
        reads: list[int | None] = []

        class ObservedFile(io.BytesIO):
            def read(self, size: int | None = -1):
                reads.append(size)
                return super().read(size)

        original_open = Path.open

        def open_file(path, *args, **kwargs):
            mode = kwargs.get("mode", args[0] if args else "r")
            if path == target.resolve() and mode == "rb":
                return ObservedFile(b"old\nmore than the configured byte limit")
            return original_open(path, *args, **kwargs)

        for name, arguments in calls:
            with self.subTest(tool=name):
                reads.clear()
                with patch.object(Path, "open", new=open_file):
                    result = await self.call(name, arguments)
                self.assertTrue(result.is_error, result.output)
                self.assertEqual(reads, [9])
                self.assertEqual(target.read_text(), "old\n")

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def call(self, name, arguments):
        return await self.registry.execute(ToolCall(name=name, arguments=arguments), self.context)

    async def test_context_tools_remain_cancellable_during_blocked_preparation(self):
        builder = self.context.metadata["context_builder"]
        provider = MockProvider(self.context.config.providers["mock"])
        self.context.metadata.update({
            "provider_routes": [SimpleNamespace(name="mock", model="test", provider=provider)],
            "tool_registry": self.registry,
        })
        cases = (
            ("repo_map", {"query": "query", "max_chars": 1000}, builder.repo_map, "build", "map"),
            ("read_skill", {"name": "example"}, builder.skills, "get", None),
            ("read_instructions", {"path": "."}, builder.instructions, "for_path", []),
            ("delegate_task", {"task": "inspect", "max_turns": 1}, builder, "system_prompt", "system"),
        )
        for name, arguments, owner, method, value in cases:
            with self.subTest(tool=name):
                started = threading.Event()
                release = threading.Event()
                finished = threading.Event()

                def blocked(*args, started=started, release=release, finished=finished, value=value, **kwargs):
                    started.set()
                    try:
                        if not release.wait(timeout=2):
                            raise AssertionError("Context preparation blocked the event loop")
                        return value
                    finally:
                        finished.set()

                with patch.object(owner, method, side_effect=blocked):
                    task = asyncio.create_task(self.call(name, arguments))
                    try:
                        self.assertTrue(await asyncio.to_thread(started.wait, 1))
                        self.assertFalse(task.done())
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await asyncio.wait_for(task, timeout=0.5)
                        self.assertFalse(finished.is_set())
                    finally:
                        release.set()
                        if not task.done():
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                        self.assertTrue(await asyncio.to_thread(finished.wait, 1))
        self.assertEqual(provider.calls, 0)

    async def test_write_read_replace_stale_and_rollback(self):
        result = await self.call("write_file", {"path":"a.txt","content":"one\n","expected_sha256":None})
        self.assertFalse(result.is_error, result.output)
        read = await self.call("read_file", {"path":"a.txt","start_line":None,"end_line":None,"max_chars":None})
        self.assertIn("1\tone", read.output)
        stale = await self.call("write_file", {"path":"a.txt","content":"bad","expected_sha256":None})
        self.assertTrue(stale.is_error)
        replaced = await self.call("replace_in_file", {
            "path":"a.txt", "old_text":"one", "new_text":"two",
            "expected_occurrences":1, "expected_sha256":sha256_text("one\n"),
        })
        self.assertFalse(replaced.is_error, replaced.output)
        checkpoint = replaced.metadata["checkpoint_id"]
        self.context.checkpoints.restore(checkpoint)
        self.assertEqual((self.root/"a.txt").read_text(), "one\n")

    async def test_read_empty_file_returns_hash_and_zero_lines(self):
        (self.root / "empty.txt").write_bytes(b"")
        result = await self.call("read_file", {
            "path": "empty.txt", "start_line": None, "end_line": None, "max_chars": None,
        })
        self.assertFalse(result.is_error, result.output)
        self.assertEqual(result.metadata["sha256"], sha256_text(""))
        self.assertEqual(result.metadata["line_count"], 0)

    async def test_read_past_end_returns_empty_selection(self):
        (self.root / "short.txt").write_text("one\n", encoding="utf-8")
        result = await self.call("read_file", {
            "path": "short.txt", "start_line": 5, "end_line": None, "max_chars": None,
        })
        self.assertFalse(result.is_error, result.output)
        self.assertEqual(result.metadata["line_count"], 1)
        self.assertNotIn("\tone", result.output)

        invalid = await self.call("read_file", {
            "path": "short.txt", "start_line": 5, "end_line": 2, "max_chars": None,
        })
        self.assertTrue(invalid.is_error)
        self.assertIn("end_line must be >= start_line", invalid.output)

    async def test_file_mutations_reject_changes_made_during_checkpoint(self):
        original_create = self.context.checkpoints.create

        async def raced_call(name, arguments, path, external_content):
            def create_then_change(*args, **kwargs):
                checkpoint = original_create(*args, **kwargs)
                path.write_text(external_content, encoding="utf-8")
                return checkpoint

            with patch.object(
                self.context.checkpoints,
                "create",
                side_effect=create_then_change,
            ):
                result = await self.call(name, arguments)
            self.assertTrue(result.is_error, result.output)
            self.assertIn("Stale", result.output)
            self.assertEqual(path.read_text(encoding="utf-8"), external_content)

        existing = self.root / "write.txt"
        existing.write_text("before", encoding="utf-8")
        await raced_call(
            "write_file",
            {
                "path": "write.txt",
                "content": "tool write",
                "expected_sha256": sha256_text("before"),
            },
            existing,
            "external write",
        )

        created = self.root / "create.txt"
        await raced_call(
            "write_file",
            {
                "path": "create.txt",
                "content": "tool create",
                "expected_sha256": None,
            },
            created,
            "external create",
        )

        replaced = self.root / "replace.txt"
        replaced.write_text("before", encoding="utf-8")
        await raced_call(
            "replace_in_file",
            {
                "path": "replace.txt",
                "old_text": "before",
                "new_text": "tool edit",
                "expected_occurrences": 1,
                "expected_sha256": sha256_text("before"),
            },
            replaced,
            "external edit",
        )

        deleted = self.root / "delete.txt"
        deleted.write_text("before", encoding="utf-8")
        await raced_call(
            "delete_file",
            {
                "path": "delete.txt",
                "expected_sha256": sha256_text("before"),
            },
            deleted,
            "external delete race",
        )
        self.assertEqual(self.context.changed_files, set())

    async def test_unified_patch_add_update_delete(self):
        (self.root/"a.txt").write_text("one\ntwo\n")
        patch = """--- a/a.txt\n+++ b/a.txt\n@@ -1,2 +1,2 @@\n one\n-two\n+three\n--- /dev/null\n+++ b/b.txt\n@@ -0,0 +1,1 @@\n+new\n"""
        result = await self.call("apply_patch", {"patch": patch})
        self.assertFalse(result.is_error, result.output)
        self.assertEqual((self.root/"a.txt").read_text(), "one\nthree\n")
        self.assertEqual((self.root/"b.txt").read_text(), "new\n")
        delete = """--- a/b.txt\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-new\n"""
        result = await self.call("apply_patch", {"patch": delete})
        self.assertFalse(result.is_error, result.output)
        self.assertFalse((self.root/"b.txt").exists())

    async def test_zero_context_patch_insertions_keep_their_position(self):
        for original, expected in (
            ("one\ntwo\n", "inserted\none\ntwo\n"),
            ("one\ntwo\n", "one\ninserted\ntwo\n"),
            ("one\ntwo\n", "one\ntwo\ninserted\n"),
            ("", "inserted\n"),
        ):
            with self.subTest(expected=expected):
                target = self.root / "insert.txt"
                target.write_text(original)
                diff = "".join(unified_diff(
                    original.splitlines(keepends=True), expected.splitlines(keepends=True),
                    "a/insert.txt", "b/insert.txt", n=0,
                ))
                result = await self.call("apply_patch", {"patch": diff})
                self.assertFalse(result.is_error, result.output)
                self.assertEqual(target.read_text(), expected)

    async def test_unified_patch_preserves_content_that_looks_like_headers(self):
        target = self.root / "headers.txt"
        target.write_text("-- original header\nkeep\n")
        diff = "".join(unified_diff(
            target.read_text().splitlines(keepends=True),
            ["++ replacement header\n", "keep\n"],
            "a/headers.txt", "b/headers.txt", n=0,
        ))
        result = await self.call("apply_patch", {"patch": diff})
        self.assertFalse(result.is_error, result.output)
        self.assertEqual(target.read_text(), "++ replacement header\nkeep\n")

    async def test_unified_added_file_preserves_plus_prefix_and_missing_newline(self):
        diff = "--- /dev/null\n+++ b/added.txt\n@@ -0,0 +1,1 @@\n+++counter\n\\ No newline at end of file\n"
        result = await self.call("apply_patch", {"patch": diff})
        self.assertFalse(result.is_error, result.output)
        self.assertEqual((self.root / "added.txt").read_bytes(), b"++counter")

    async def test_unified_patches_preserve_and_change_line_endings(self):
        cases = [
            (old.join(("one", "old", "end", "")), new.join(("one", "new", "end", "")))
            for old in ("\n", "\r\n") for new in ("\n", "\r\n")
        ]
        cases.append(("one\nold\r\nend\n", "one\nnew\r\nend\n"))
        for original, expected in cases:
            for header_newline in ("\n", "\r\n"):
                with self.subTest(original=original, expected=expected, headers=header_newline):
                    target = self.root / "line-endings.txt"
                    target.write_bytes(original.encode())
                    diff = "".join(unified_diff(
                        original.splitlines(keepends=True), expected.splitlines(keepends=True),
                        "a/line-endings.txt", "b/line-endings.txt", lineterm=header_newline,
                    ))
                    result = await self.call("apply_patch", {"patch": diff})
                    self.assertFalse(result.is_error, result.output)
                    self.assertEqual(target.read_bytes(), expected.encode())

    async def test_crlf_patch_transport_updates_lf_and_crlf_files(self):
        for envelope, newline in (
            (False, "\n"), (False, "\r\n"), (True, "\n"), (True, "\r\n"),
        ):
            with self.subTest(envelope=envelope, newline=newline):
                target = self.root / "transport.txt"
                target.write_bytes(f"old{newline}".encode())
                if envelope:
                    diff = "*** Begin Patch\n*** Update File: transport.txt\n@@\n-old\n+new\n*** End Patch\n"
                else:
                    diff = "--- a/transport.txt\n+++ b/transport.txt\n@@ -1 +1 @@\n-old\n+new\n"
                result = await self.call("apply_patch", {"patch": diff.replace("\n", "\r\n")})
                self.assertFalse(result.is_error, result.output)
                self.assertEqual(target.read_bytes(), f"new{newline}".encode())

    async def test_crlf_add_delete_and_missing_final_newline(self):
        added = "--- /dev/null\n+++ b/crlf.txt\n@@ -0,0 +1,2 @@\n+one\r\n+two\r\n"
        result = await self.call("apply_patch", {"patch": added})
        self.assertFalse(result.is_error, result.output)
        target = self.root / "crlf.txt"
        self.assertEqual(target.read_bytes(), b"one\r\ntwo\r\n")

        updated = (
            "--- a/crlf.txt\n+++ b/crlf.txt\n@@ -1,2 +1,2 @@\n one\r\n-two\r\n"
            "+three\n\\ No newline at end of file\n"
        )
        result = await self.call("apply_patch", {"patch": updated})
        self.assertFalse(result.is_error, result.output)
        self.assertEqual(target.read_bytes(), b"one\r\nthree")

        deleted = (
            "--- a/crlf.txt\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-one\r\n-three\n"
            "\\ No newline at end of file\n"
        )
        result = await self.call("apply_patch", {"patch": deleted})
        self.assertFalse(result.is_error, result.output)
        self.assertFalse(target.exists())

    async def test_truncated_new_file_hunk_prevents_all_changes(self):
        target = self.root / "existing.txt"
        target.write_text("old\n")
        diff = (
            "--- a/existing.txt\n+++ b/existing.txt\n@@ -1 +1 @@\n-old\n+new\n"
            "--- /dev/null\n+++ b/added.txt\n@@ -0,0 +1,2 @@\n+partial\n"
        )
        result = await self.call("apply_patch", {"patch": diff})
        self.assertTrue(result.is_error, result.output)
        self.assertEqual(target.read_text(), "old\n")
        self.assertFalse((self.root / "added.txt").exists())
        self.assertEqual(self.context.checkpoints.list(), [])

    async def test_envelope_paths_keep_literal_a_and_b_directories(self):
        root_file = self.root / "example.txt"
        root_file.write_text("old\n")
        for directory in ("a", "b"):
            with self.subTest(directory=directory):
                target = self.root / directory / "example.txt"
                target.parent.mkdir()
                target.write_text("old\n")
                diff = f"*** Begin Patch\n*** Update File: {directory}/example.txt\n@@\n-old\n+new\n*** End Patch\n"
                result = await self.call("apply_patch", {"patch": diff})
                self.assertFalse(result.is_error, result.output)
                self.assertEqual(target.read_text(), "new\n")
                self.assertEqual(root_file.read_text(), "old\n")

    async def test_patch_paths_do_not_expand_environment_variables(self):
        name = "$BOREALIS_PATH_FIXTURE.txt"
        target = self.root / name
        other = self.root / "other.txt"
        patches = (
            f"*** Begin Patch\n*** Update File: {name}\n@@\n-before\n+after\n*** End Patch\n",
            f"--- a/{name}\n+++ b/{name}\n@@ -1 +1 @@\n-before\n+after\n",
        )
        with patch.dict(os.environ, {"BOREALIS_PATH_FIXTURE": "other"}):
            for diff in patches:
                with self.subTest(patch=diff):
                    target.write_text("before\n")
                    other.write_text("before\n")

                    result = await self.call("apply_patch", {"patch": diff})

                    self.assertFalse(result.is_error, result.output)
                    self.assertEqual(target.read_text(), "after\n")
                    self.assertEqual(other.read_text(), "before\n")
                    checkpoint = self.context.checkpoints.list()[0]
                    self.assertEqual(checkpoint.files[0]["path"], name)
                    self.context.checkpoints.restore(checkpoint.id)
                    self.assertEqual(target.read_text(), "before\n")
                    self.assertEqual(other.read_text(), "before\n")

    async def test_incomplete_patch_envelope_is_rejected(self):
        result = await self.call("apply_patch", {
            "patch": "*** Begin Patch\n*** Add File: partial.txt\n+partial\n",
        })
        self.assertTrue(result.is_error, result.output)
        self.assertFalse((self.root / "partial.txt").exists())

    async def test_real_git_diff_applies_multiple_files_with_metadata(self):
        def git(*arguments):
            return subprocess.run(
                ["git", "-C", str(self.root), "-c", "core.hooksPath=/dev/null",
                 "-c", "core.fsmonitor=false", *arguments],
                check=True, capture_output=True,
            ).stdout.decode()

        originals = {"first.txt": "one\n", "second.txt": "-- old"}
        expected = {"first.txt": "one\ninserted\n", "second.txt": "++ new"}
        git("init", "-q")
        for name, text in originals.items():
            (self.root / name).write_text(text)
        git("add", "--", *originals)
        for name, text in expected.items():
            (self.root / name).write_text(text)
        diff = git("diff", "--no-ext-diff", "--no-textconv", "--unified=0")
        for name, text in originals.items():
            (self.root / name).write_text(text)
        result = await self.call("apply_patch", {"patch": diff})
        self.assertFalse(result.is_error, result.output)
        for name, text in expected.items():
            self.assertEqual((self.root / name).read_text(), text)

    async def test_unsupported_git_sections_reject_the_complete_patch(self):
        target = self.root / "ok.txt"
        target.write_text("old\n")
        valid = (
            "diff --git a/ok.txt b/ok.txt\nindex 1..2 100644\n"
            "--- a/ok.txt\n+++ b/ok.txt\n@@ -1 +1 @@\n-old\n+new\n"
        )
        unsupported = {
            "binary": "diff --git a/data.bin b/data.bin\nBinary files a/data.bin and b/data.bin differ\n",
            "binary payload": "diff --git a/data.bin b/data.bin\nGIT binary patch\nliteral 0\nHcmV?d00001\n",
            "mode": "diff --git a/run.sh b/run.sh\nold mode 100644\nnew mode 100755\n",
            "rename": "diff --git a/old.txt b/new.txt\nsimilarity index 100%\nrename from old.txt\nrename to new.txt\n",
            "copy": "diff --git a/old.txt b/new.txt\nsimilarity index 100%\ncopy from old.txt\ncopy to new.txt\n",
            "empty addition": "diff --git a/empty b/empty\nnew file mode 100644\nindex 0000000..e69de29\n",
            "executable addition": "diff --git a/run.sh b/run.sh\nnew file mode 100755\n--- /dev/null\n+++ b/run.sh\n@@ -0,0 +1 @@\n+exit 0\n",
            "symlink addition": "diff --git a/link b/link\nnew file mode 120000\n--- /dev/null\n+++ b/link\n@@ -0,0 +1 @@\n+target\n",
            "symlink update": "diff --git a/link b/link\nindex 1..2 120000\n--- a/link\n+++ b/link\n@@ -1 +1 @@\n-old\n+new\n",
            "combined": "diff --cc file.txt\nindex 1,2..3\n",
        }
        for name, section in unsupported.items():
            for first in (True, False):
                with self.subTest(section=name, first=first):
                    diff = section + valid if first else valid + section
                    result = await self.call("apply_patch", {"patch": diff})
                    self.assertTrue(result.is_error, result.output)
                    self.assertEqual(target.read_text(), "old\n")
                    self.assertEqual(self.context.changed_files, set())
                    self.assertEqual(self.context.checkpoints.list(), [])
                    self.assertFalse((self.root / "run.sh").exists())
                    self.assertFalse((self.root / "link").exists())

    async def test_unified_patch_rejects_rename_and_quoted_paths(self):
        (self.root / "old.txt").write_text("old\n")
        (self.root / "new.txt").write_text("old\n")
        for before, after in (("a/old.txt", "b/new.txt"), ('"a/quoted.txt"', '"b/quoted.txt"')):
            with self.subTest(before=before, after=after):
                result = await self.call("apply_patch", {
                    "patch": f"--- {before}\n+++ {after}\n@@ -1 +1 @@\n-old\n+new\n",
                })
                self.assertTrue(result.is_error, result.output)
                self.assertIn("cannot rename" if before == "a/old.txt" else "Quoted Git paths", result.output)
                self.assertEqual(self.context.checkpoints.list(), [])
        self.assertEqual((self.root / "old.txt").read_text(), "old\n")
        self.assertEqual((self.root / "new.txt").read_text(), "old\n")

    async def test_make_directory_tracks_new_directories_and_root(self):
        result = await self.call("make_directory", {"path": "parent/child"})

        self.assertFalse(result.is_error, result.output)
        self.assertTrue((self.root / "parent" / "child").is_dir())
        self.assertEqual(result.metadata["changed_files"], ["parent", "parent/child"])
        self.assertEqual(self.context.changed_files, {"parent", "parent/child"})
        self.assertEqual(self.context.changed_roots, {self.root.resolve()})

        self.context.changed_files.clear()
        self.context.changed_roots.clear()
        existing = await self.call("make_directory", {"path": "parent/child"})
        self.assertFalse(existing.is_error, existing.output)
        self.assertEqual(existing.metadata["changed_files"], [])
        self.assertEqual(self.context.changed_files, set())
        self.assertEqual(self.context.changed_roots, set())

    async def test_patch_prevalidation_is_atomic(self):
        (self.root/"a.txt").write_text("a\n")
        (self.root/"b.txt").write_text("b\n")
        patch = """--- a/a.txt\n+++ b/a.txt\n@@ -1,1 +1,1 @@\n-a\n+A\n--- a/b.txt\n+++ b/b.txt\n@@ -1,1 +1,1 @@\n-NOT_B\n+B\n"""
        result = await self.call("apply_patch", {"patch": patch})
        self.assertTrue(result.is_error)
        self.assertEqual((self.root/"a.txt").read_text(), "a\n")

    async def test_patch_rejects_change_during_checkpoint_before_any_commit(self):
        first = self.root / "a.txt"
        second = self.root / "b.txt"
        first.write_text("a\n", encoding="utf-8")
        second.write_text("b\n", encoding="utf-8")
        patch_text = """*** Begin Patch
*** Update File: a.txt
@@
-a
+A
*** Update File: b.txt
@@
-b
+B
*** End Patch"""
        original_create = self.context.checkpoints.create

        def create_then_change(*args, **kwargs):
            checkpoint = original_create(*args, **kwargs)
            second.write_text("external\n", encoding="utf-8")
            return checkpoint

        with patch.object(
            self.context.checkpoints,
            "create",
            side_effect=create_then_change,
        ):
            result = await self.call("apply_patch", {"patch": patch_text})

        self.assertTrue(result.is_error, result.output)
        self.assertIn("Stale patch rejected", result.output)
        self.assertEqual(first.read_text(encoding="utf-8"), "a\n")
        self.assertEqual(second.read_text(encoding="utf-8"), "external\n")
        self.assertEqual(self.context.changed_files, set())

    async def test_apply_patch_bare_header_and_multiple_hunks(self):
        (self.root / "a.txt").write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
        patch = """*** Begin Patch
*** Update File: a.txt
@@
 alpha
-beta
+BETA
@@
 gamma
-delta
+DELTA
*** End Patch"""

        result = await self.call("apply_patch", {"patch": patch})

        self.assertFalse(result.is_error, result.output)
        self.assertEqual(
            (self.root / "a.txt").read_text(encoding="utf-8"),
            "alpha\nBETA\ngamma\nDELTA\n",
        )

    async def test_apply_patch_bare_header_rejects_ambiguous_or_missing_context(self):
        (self.root / "a.txt").write_text("same\nvalue\nsame\nvalue\n", encoding="utf-8")
        ambiguous = """*** Begin Patch
*** Update File: a.txt
@@
 same
-value
+changed
*** End Patch"""
        missing = """*** Begin Patch
*** Update File: a.txt
@@
-absent
+changed
*** End Patch"""

        ambiguous_result = await self.call("apply_patch", {"patch": ambiguous})
        missing_result = await self.call("apply_patch", {"patch": missing})

        self.assertTrue(ambiguous_result.is_error)
        self.assertIn("ambiguous", ambiguous_result.output)
        self.assertTrue(missing_result.is_error)
        self.assertIn("not found", missing_result.output)
        self.assertEqual(
            (self.root / "a.txt").read_text(encoding="utf-8"),
            "same\nvalue\nsame\nvalue\n",
        )

    async def test_apply_patch_rejects_overlapping_numbered_hunks(self):
        (self.root / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
        patch = """*** Begin Patch
*** Update File: a.txt
@@ -1,2 +1,2 @@
 one
-two
+TWO
@@ -2,2 +2,2 @@
 two
-three
+THREE
*** End Patch"""

        result = await self.call("apply_patch", {"patch": patch})

        self.assertTrue(result.is_error)
        self.assertIn("Overlapping or reordered", result.output)
        self.assertEqual((self.root / "a.txt").read_text(), "one\ntwo\nthree\n")

    async def test_apply_patch_bare_header_multi_file_failure_is_atomic(self):
        (self.root / "a.txt").write_text("a\n", encoding="utf-8")
        (self.root / "b.txt").write_text("b\n", encoding="utf-8")
        patch = """*** Begin Patch
*** Update File: a.txt
@@
-a
+A
*** Update File: b.txt
@@
-missing
+B
*** End Patch"""

        result = await self.call("apply_patch", {"patch": patch})

        self.assertTrue(result.is_error)
        self.assertEqual((self.root / "a.txt").read_text(), "a\n")
        self.assertEqual(len(self.context.checkpoints.list()), 0)

    async def test_apply_patch_envelope_add_delete_and_no_trailing_newline(self):
        (self.root / "old.txt").write_text("old", encoding="utf-8")
        patch = """*** Begin Patch
*** Update File: old.txt
@@
-old
\\ No newline at end of file
+new
\\ No newline at end of file
*** Add File: added.txt
+added
*** End Patch"""

        result = await self.call("apply_patch", {"patch": patch})

        self.assertFalse(result.is_error, result.output)
        self.assertEqual((self.root / "old.txt").read_bytes(), b"new")
        self.assertEqual((self.root / "added.txt").read_text(), "added\n")

        delete = """*** Begin Patch
*** Delete File: added.txt
*** End Patch"""
        deleted = await self.call("apply_patch", {"patch": delete})
        self.assertFalse(deleted.is_error, deleted.output)
        self.assertFalse((self.root / "added.txt").exists())

    async def test_grep_glob_and_list(self):
        (self.root/"src").mkdir()
        for name in ("a.py", "b.py", "c.py"):
            (self.root / "src" / name).write_text("def alpha():\n    return 1\n")
        self.context.config.context.max_search_results = 1
        grep = await self.call("grep", {"pattern":"alpha","path":".","glob":"*.py","regex":False,"case_sensitive":True,"context_lines":0,"max_results":10})
        self.assertIn(":1:def alpha", grep.output)
        self.assertEqual(grep.metadata["matches"], 1)
        self.assertTrue(grep.metadata["truncated"])
        glob = await self.call("glob_files", {"pattern":"**/*.py","path":".","limit":10})
        self.assertEqual(len(glob.output.splitlines()), 1)
        self.assertTrue(glob.output.endswith(".py"))
        self.assertEqual(glob.metadata["matches"], 1)
        self.assertTrue(glob.metadata["truncated"])

    async def test_protected_paths_and_file_size_limits(self):
        protected = await self.call(
            "write_file",
            {"path": ".git/config", "content": "bad", "expected_sha256": None},
        )
        self.assertTrue(protected.is_error)
        self.context.config.context.max_file_bytes = 4
        too_large = await self.call(
            "write_file",
            {"path": "large.txt", "content": "12345", "expected_sha256": None},
        )
        self.assertTrue(too_large.is_error)
        self.assertFalse((self.root / "large.txt").exists())

    def test_fetch_rejects_non_public_destinations(self):
        with self.assertRaises(ToolError):
            _validate_public_url("http://127.0.0.1/admin")
        with self.assertRaises(ToolError):
            _validate_public_url("http://169.254.169.254/latest/meta-data")

    def test_fetch_uses_the_vetted_address_without_resolving_again(self):
        public_record = (
            2,
            1,
            6,
            "",
            ("93.184.216.34", 443),
        )
        with (
            patch(
                "borealis_coder.tools.fetch.socket.getaddrinfo",
                side_effect=[[public_record], OSError("unexpected second lookup")],
            ) as resolver,
            patch(
                "borealis_coder.tools.fetch._request_once",
                return_value=(b"ok", 200, "text/plain", "utf-8", None),
            ) as request,
        ):
            body, status, _, _ = _fetch_public_url("https://example.com", 100)
        self.assertEqual((body, status), ("ok", 200))
        resolver.assert_called_once()
        self.assertEqual(request.call_args.args[1], "93.184.216.34")


if __name__ == "__main__":
    unittest.main()
