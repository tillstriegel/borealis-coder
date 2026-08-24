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
        self.assertEqual(self.context.changed_files, {"parent", "parent/child"})
        self.assertEqual(self.context.changed_roots, {self.root.resolve()})

        self.context.changed_files.clear()
        self.context.changed_roots.clear()
        existing = await self.call("make_directory", {"path": "parent/child"})
        self.assertFalse(existing.is_error, existing.output)
        self.assertEqual(self.context.changed_files, set())
        self.assertEqual(self.context.changed_roots, set())

    async def test_patch_prevalidation_is_atomic(self):
        (self.root/"a.txt").write_text("a\n")
        (self.root/"b.txt").write_text("b\n")
        patch = """--- a/a.txt\n+++ b/a.txt\n@@ -1,1 +1,1 @@\n-a\n+A\n--- a/b.txt\n+++ b/b.txt\n@@ -1,1 +1,1 @@\n-NOT_B\n+B\n"""
        result = await self.call("apply_patch", {"patch": patch})
        self.assertTrue(result.is_error)
        self.assertEqual((self.root/"a.txt").read_text(), "a\n")

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
