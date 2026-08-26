from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from borealis_coder.errors import ToolError, ToolValidationError
from borealis_coder.models import ToolCall
from borealis_coder.tools import build_builtin_registry, validate_schema
from borealis_coder.tools.fetch import _fetch_public_url, _validate_public_url
from borealis_coder.util import sha256_text
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


class FileToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.context = make_context(self.root)
        self.registry = build_builtin_registry()

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def call(self, name, arguments):
        return await self.registry.execute(ToolCall(name=name, arguments=arguments), self.context)

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
