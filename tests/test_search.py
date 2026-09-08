from __future__ import annotations

import asyncio
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from borealis_coder.errors import ToolError
from borealis_coder.safety import WorkspaceRoots
from borealis_coder.tools.filesystem import ReadFileTool
from borealis_coder.tools.search import GrepTool, _RegexMatcher
from tests.helpers import make_context


class SearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.context = make_context(self.root)
        self.tool = GrepTool()

    async def search(self, **overrides):
        arguments = {
            "pattern": "needle",
            "path": ".",
            "glob": "*.txt",
            "regex": False,
            "case_sensitive": True,
            "context_lines": 0,
            "max_results": 100,
        }
        arguments.update(overrides)
        return await self.tool.execute(arguments, self.context)

    async def test_recursive_search_skips_external_symlinks(self):
        with tempfile.TemporaryDirectory() as external:
            outside = Path(external) / "private.txt"
            outside.write_text("needle private contents", encoding="utf-8")
            inside = self.root / "local.txt"
            inside.write_text("needle public contents", encoding="utf-8")
            try:
                (self.root / "external.txt").symlink_to(outside)
                (self.root / "alias.txt").symlink_to(inside)
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")

            result = await self.search()

            self.assertNotIn("private contents", result.output)
            self.assertIn("public contents", result.output)
            self.assertNotIn(str(outside), result.output)

            self.context.roots = WorkspaceRoots(self.root, [Path(external)])
            allowed = await self.search(path="external.txt")
            self.assertIn("private contents", allowed.output)

    async def test_result_limit_stops_directory_discovery(self):
        (self.root / "first.txt").write_text("needle", encoding="utf-8")

        def walk(*args, **kwargs):
            yield str(self.root), [], ["first.txt"]
            raise AssertionError("Search continued walking after reaching its limit")

        with patch("borealis_coder.tools.search.os.walk", side_effect=walk):
            result = await self.search(max_results=1)

        self.assertEqual(result.metadata["matches"], 1)
        self.assertTrue(result.metadata["truncated"])

    async def test_discovered_file_names_do_not_expand_environment_variables(self):
        (self.root / "$BOREALIS_PATH_FIXTURE.txt").write_text("needle", encoding="utf-8")
        (self.root / "other.txt").write_text("different contents", encoding="utf-8")
        with patch.dict(os.environ, {"BOREALIS_PATH_FIXTURE": "other"}):
            result = await self.search()

        self.assertEqual(result.output, "$BOREALIS_PATH_FIXTURE.txt:1:needle")
        self.assertEqual(result.metadata["matches"], 1)
        self.assertEqual(result.metadata["files_scanned"], 2)

    async def test_oversized_file_read_is_bounded(self):
        target = self.root / "large.txt"
        target.write_text("needle", encoding="utf-8")
        self.context.config.context.max_file_bytes = 64

        class BoundedStream(io.BytesIO):
            def read(self, size: int | None = -1):
                if size is None or not 0 <= size <= 65:
                    raise AssertionError("Search attempted an unbounded file read")
                return super().read(size)

        # The file can grow after discovery; the read itself must enforce the cap.
        with patch.object(Path, "open", return_value=BoundedStream(b"needle" * 1000)):
            result = await self.search(path="large.txt")

        self.assertEqual(result.output, "No matches")
        self.assertEqual(result.metadata["files_scanned"], 0)

        with (
            patch.object(Path, "open", return_value=BoundedStream(b"needle" * 1000)),
            self.assertRaisesRegex(ToolError, "File exceeds 64 byte read limit"),
        ):
            await ReadFileTool().execute({"path": "large.txt"}, self.context)

    async def test_output_limit_bounds_search_before_registry_truncation(self):
        (self.root / "long.txt").write_text("needle " + "x" * 1000, encoding="utf-8")
        self.context.config.context.tool_output_chars = 128

        result = await self.search()

        self.assertLessEqual(len(result.output), 128)
        self.assertIn("needle", result.output)
        self.assertTrue(result.metadata["truncated"])

    async def test_binary_files_are_skipped(self):
        (self.root / "binary.txt").write_bytes(b"needle\x00binary")
        result = await self.search()
        self.assertEqual(result.output, "No matches")
        self.assertEqual(result.metadata["files_scanned"], 0)

    async def test_regex_honors_large_configured_result_limits(self):
        (self.root / "many.txt").write_text("needle\n" * 20_000, encoding="utf-8")
        self.context.config.context.max_search_results = 20_000
        self.context.config.context.tool_output_chars = 1_000_000

        result = await self.search(regex=True, max_results=20_000)

        self.assertEqual(result.metadata["matches"], 20_000)
        self.assertIn("many.txt:20000:needle", result.output)

    async def test_search_yields_so_it_can_be_cancelled(self):
        (self.root / "many.txt").write_text("nothing\n" * 1000, encoding="utf-8")
        task = asyncio.create_task(self.search())
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_regex_preserves_python_semantics_and_reuses_one_worker(self):
        (self.root / "a.txt").write_text("prefix\nITEM:café café\nsuffix", encoding="utf-8")
        (self.root / "b.txt").write_text("ignore\nitem:abc abc\ntail", encoding="utf-8")
        processes = []
        original_create = asyncio.create_subprocess_exec

        async def create(*args, **kwargs):
            process = await original_create(*args, **kwargs)
            processes.append(process)
            return process

        with patch("borealis_coder.tools.search.asyncio.create_subprocess_exec", side_effect=create):
            result = await self.search(
                pattern=r"(?<=item:)(\w+)\s+\1$", regex=True,
                case_sensitive=False, context_lines=1, max_results=2,
            )

        self.assertEqual(result.output, (
            "a.txt:1-prefix\na.txt:2:ITEM:café café\na.txt:3-suffix\n"
            "b.txt:1-ignore\nb.txt:2:item:abc abc\nb.txt:3-tail\n… result limit reached …"
        ))
        self.assertEqual(result.metadata["matches"], 2)
        self.assertEqual(result.metadata["files_scanned"], 2)
        self.assertTrue(result.metadata["truncated"])
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].returncode)

    async def test_literal_search_does_not_start_a_worker(self):
        (self.root / "a.txt").write_text("literal (a+)+$", encoding="utf-8")
        with patch(
            "borealis_coder.tools.search.asyncio.create_subprocess_exec",
            side_effect=AssertionError("Literal search started a child process"),
        ):
            result = await self.search(pattern="(a+)+$")
        self.assertEqual(result.metadata["matches"], 1)

    async def test_invalid_regex_and_match_timeout_reap_the_worker(self):
        # This bounded input also finishes in finite time if isolation regresses.
        (self.root / "a.txt").write_text("a" * 26 + "!", encoding="utf-8")
        self.context.config.context.regex_timeout_seconds = 0.02
        original_create = asyncio.create_subprocess_exec
        processes = []

        async def create(*args, **kwargs):
            process = await original_create(*args, **kwargs)
            processes.append(process)
            return process

        for pattern, error in (("[", "Invalid regular expression"), ("(a+)+$", "timed out")):
            with (
                self.subTest(pattern=pattern),
                patch("borealis_coder.tools.search.asyncio.create_subprocess_exec", side_effect=create),
                self.assertRaisesRegex(ToolError, error),
            ):
                await asyncio.wait_for(self.search(pattern=pattern, regex=True), timeout=3)
        self.assertEqual(len(processes), 2)
        self.assertTrue(all(process.returncode is not None for process in processes))

    async def test_regex_cancellation_keeps_event_loop_responsive_and_reaps_worker(self):
        (self.root / "a.txt").write_text("a" * 26 + "!", encoding="utf-8")
        matching = asyncio.Event()
        processes = []
        original_search = _RegexMatcher.search

        async def search(matcher, *args, **kwargs):
            processes.append(matcher.process)
            matching.set()
            return await original_search(matcher, *args, **kwargs)

        with patch.object(_RegexMatcher, "search", new=search):
            task = asyncio.create_task(self.search(pattern="(a+)+$", regex=True))
            try:
                await asyncio.wait_for(matching.wait(), timeout=2)
                await asyncio.sleep(0.02)
                self.assertFalse(task.done())
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=1)
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].returncode)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires named pipes")
    async def test_search_does_not_open_named_pipes(self):
        os.mkfifo(self.root / "pipe.txt")
        with patch.object(Path, "open", side_effect=AssertionError("Opened a named pipe")):
            result = await self.search()
        self.assertEqual(result.output, "No matches")
