from __future__ import annotations

import io
import os
import subprocess
import tempfile
import threading
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from borealis_coder.context import (
    ContextBuilder,
    IgnoreMatcher,
    InstructionLoader,
    RepoMap,
    SkillCatalog,
    repository_files,
)
from borealis_coder.context.repomap import FileSummary, _python_symbols
from tests.helpers import make_config


class ContextTests(unittest.TestCase):
    def test_concurrent_repo_maps_keep_each_discovered_file_set(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            paths = [root / f"file_{index}.py" for index in range(48)]
            for index, path in enumerate(paths):
                path.write_text(f"def symbol_{index}(): pass\n")
            repo_map = RepoMap(root, IgnoreMatcher(root))
            local = threading.local()
            ready = threading.Barrier(4)

            def discover(*args, **kwargs):
                return local.paths

            def snapshot(offset):
                local.paths = [paths[(offset + index) % len(paths)] for index in range(12)]
                ready.wait(timeout=2)
                summaries = repo_map.snapshot()
                self.assertEqual({item.path for item in summaries}, {path.name for path in local.paths})
                self.assertTrue(all(item.symbols for item in summaries))

            with (
                patch("borealis_coder.context.repomap.repository_files", side_effect=discover),
                ThreadPoolExecutor(max_workers=4) as pool,
            ):
                for round_index in range(8):
                    pending = [pool.submit(snapshot, round_index + offset) for offset in (0, 12, 24, 36)]
                    for future in pending:
                        future.result(timeout=3)
                    self.assertEqual(len(repo_map._cache), 12)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        (self.root / "src").mkdir()
        (self.root / "src/main.py").write_text(
            "import json\nclass Engine:\n    def run(self):\n        return json.dumps({})\n"
        )
        (self.root / "src/other.ts").write_text("export function helper() { return 1 }\n")
        (self.root / "AGENTS.md").write_text("Always run tests.\n")
        (self.root / "src/AGENTS.md").write_text("Use Python typing.\n")
        skill = self.root / ".agents/skills/review"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: review\ndescription: Review code carefully\n---\n# Review\nInspect tests.\n"
        )
        (self.root / ".gitignore").write_text("ignored.txt\n")
        (self.root / "ignored.txt").write_text("secret")
        self.config = make_config(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_ignore_and_files(self):
        matcher = IgnoreMatcher(self.root, ignored_dirs=self.config.context.ignored_dirs)
        files = [
            item.relative_to(self.root).as_posix() for item in repository_files(self.root, matcher)
        ]
        self.assertNotIn("ignored.txt", files)
        self.assertIn("src/main.py", files)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "Named pipes require POSIX")
    def test_fallback_repository_discovery_skips_named_pipes(self):
        pipe = self.root / "src/pipe.py"
        os.mkfifo(pipe)
        link = self.root / "src/pipe_link.py"
        link.symlink_to(pipe)
        matcher = IgnoreMatcher(self.root, ignored_dirs=self.config.context.ignored_dirs)

        files = repository_files(self.root, matcher)
        self.assertNotIn(pipe, files)
        self.assertNotIn(link, files)
        output = RepoMap(self.root, matcher).build()
        self.assertIn("src/main.py", output)
        self.assertNotIn("pipe.py", output)
        self.assertNotIn("pipe_link.py", output)

    def test_context_caches_refresh_after_preserved_mtime_replacements(self):
        self._assert_context_refresh_with_preserved_mtime(atomic=True)

    @unittest.skipIf(os.name == "nt", "Windows ctime currently records creation time")
    def test_context_caches_refresh_after_preserved_mtime_inplace_edits(self):
        self._assert_context_refresh_with_preserved_mtime(atomic=False)

    def _assert_context_refresh_with_preserved_mtime(self, *, atomic):
        source = self.root / "src/main.py"
        source.write_text("class Before: pass\n")
        skill = self.root / ".agents/skills/review/SKILL.md"
        skill.write_text("---\nname: review\ndescription: Before\n---\nBefore.\n")
        ignored = self.root / ".borealisignore"
        ignored.write_text("src/aaa.py\n")
        (self.root / "src/aaa.py").write_text("class First: pass\n")
        (self.root / "src/bbb.py").write_text("class Second: pass\n")
        builder = ContextBuilder(self.root, self.config)
        initial = builder.build().text
        self.assertIn("symbols: Before", initial)
        self.assertIn("review: Before", initial)
        self.assertNotIn("src/aaa.py", initial)
        self.assertIn("src/bbb.py", initial)

        for path, text in (
            (source, "class After_: pass\n"),
            (skill, "---\nname: review\ndescription: After_\n---\nAfter_.\n"),
            (ignored, "src/bbb.py\n"),
        ):
            previous = path.stat()
            destination = path.with_suffix(".replacement") if atomic else path
            destination.write_text(text)
            os.utime(destination, ns=(previous.st_atime_ns, previous.st_mtime_ns))
            if atomic:
                destination.replace(path)
            self.assertEqual(path.stat().st_size, previous.st_size)
            self.assertEqual(path.stat().st_mtime_ns, previous.st_mtime_ns)

        refreshed = builder.build().text
        self.assertIn("symbols: After_", refreshed)
        self.assertNotIn("symbols: Before", refreshed)
        self.assertIn("review: After_", refreshed)
        self.assertIn("src/aaa.py", refreshed)
        self.assertNotIn("src/bbb.py", refreshed)

    def test_repository_discovery_refreshes_changed_ignore_files(self):
        matcher = IgnoreMatcher(self.root, ignored_dirs=self.config.context.ignored_dirs)
        repo_map = RepoMap(self.root, matcher)
        source = self.root / "src/private.py"
        source.write_text("class PrivateCredentialStore:\n    pass\n")
        self.assertIn("src/private.py", {item.path for item in repo_map.snapshot()})

        (self.root / ".borealisignore").write_text("src/private.py\n")
        refreshed = repo_map.snapshot()

        self.assertNotIn("src/private.py", {item.path for item in refreshed})
        self.assertNotIn(str(source.resolve()), repo_map._cache)

    def test_directory_ignore_excludes_tracked_descendants(self):
        generated = self.root / "generated"
        generated.mkdir()
        source = generated / "private.py"
        source.write_text("PRIVATE_GENERATED_SOURCE = True\n", encoding="utf-8")
        (self.root / ".borealisignore").write_text("generated/\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(
            ["git", "-C", str(self.root), "add", "generated/private.py"],
            check=True,
        )

        matcher = IgnoreMatcher(self.root, ignored_dirs=self.config.context.ignored_dirs)
        files = {
            item.relative_to(self.root).as_posix()
            for item in repository_files(self.root, matcher)
        }
        self.assertTrue(matcher.ignored(generated, is_dir=True))
        self.assertTrue(matcher.ignored(source, is_dir=False))
        self.assertNotIn("generated/private.py", files)

    def test_instructions_and_skills(self):
        loader = InstructionLoader(self.root, ["AGENTS.md"])
        applicable = loader.for_path(self.root / "src/main.py")
        self.assertEqual(
            [item.relative_path for item in applicable], ["AGENTS.md", "src/AGENTS.md"]
        )
        catalog = SkillCatalog(self.root, [".agents/skills"])
        review = catalog.get("review")
        assert review is not None
        self.assertEqual(review.description, "Review code carefully")
        skill_file = self.root / ".agents/skills/review/SKILL.md"
        skill_file.write_text(
            "---\nname: review\ndescription: Review changed code carefully\n---\nUpdated.\n"
        )
        updated = catalog.get("review")
        assert updated is not None
        self.assertEqual(updated.description, "Review changed code carefully")
        self.assertEqual(len(catalog._cache), 1)

    def test_repo_map_ranking_and_system_prompt(self):
        matcher = IgnoreMatcher(self.root, ignored_dirs=self.config.context.ignored_dirs)
        output = RepoMap(self.root, matcher).build(query="Engine run", max_chars=5000)
        self.assertIn("src/main.py", output)
        self.assertIn("Engine", output)
        prompt = ContextBuilder(self.root, self.config).system_prompt(query="Engine")
        self.assertIn("Always run tests", prompt)
        self.assertIn("Review code carefully", prompt)

        builder = ContextBuilder(self.root, self.config)
        first = builder.build(query="Engine")
        second = builder.build(query="unrelated request")
        self.assertEqual(first.stable, second.stable)
        self.assertEqual(first.stable_fingerprint, second.stable_fingerprint)
        self.assertEqual(first.cache_routing_key, second.cache_routing_key)
        self.assertNotEqual(first.dynamic, second.dynamic)
        self.assertEqual(len(first.cache_blocks), 3)
        self.assertEqual(
            "\n\n".join(str(block["text"]) for block in first.system_blocks),
            first.text,
        )

    def test_shared_skill_files_keep_distinct_alias_names_when_cached(self):
        shared = self.root / "shared-guide.md"
        shared.write_text("Shared project guidance.\n")
        for name in ("alpha", "beta"):
            directory = self.root / ".agents/alias-skills" / name
            directory.mkdir(parents=True)
            try:
                (directory / "SKILL.md").symlink_to(shared)
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")
        catalog = SkillCatalog(self.root, [".agents/alias-skills"])

        first = catalog.discover()
        second = catalog.discover()

        self.assertEqual(list(first), ["alpha", "beta"])
        self.assertEqual(second, first)
        for name in ("alpha", "beta"):
            skill = catalog.get(name)
            assert skill is not None
            self.assertEqual(skill.name, name)
            self.assertEqual(skill.path, shared.resolve())

    def test_skill_retarget_to_same_inode_updates_its_resource_path(self):
        first = self.root / "first" / "guide.md"
        second = self.root / "second" / "guide.md"
        first.parent.mkdir()
        second.parent.mkdir()
        first.write_text("Shared project guidance.\n")
        alias = self.root / ".agents/alias-skills/review/SKILL.md"
        alias.parent.mkdir(parents=True)
        try:
            os.link(first, second)
            alias.symlink_to(first)
        except OSError as error:
            self.skipTest(f"links unavailable: {error}")
        catalog = SkillCatalog(self.root, [".agents/alias-skills"])
        original = catalog.get("review")
        assert original is not None
        self.assertEqual(original.path, first.resolve())

        alias.unlink()
        alias.symlink_to(second)
        updated = catalog.get("review")

        assert updated is not None
        self.assertTrue(first.samefile(second))
        self.assertEqual(updated.body, original.body)
        self.assertEqual(updated.path, second.resolve())

    def test_prompt_maps_share_the_configured_character_budget(self):
        for index in range(80):
            (self.root / f"src/module{index}.py").write_text(f"class Class{index}: pass\n")
        for limit in (0, 1, 17, 18, 32, 100, 999, 1000, 1200, 3999, 4000, 28000):
            with self.subTest(limit=limit):
                config = make_config(self.root, context={
                    "repo_map_chars": limit, "include_git_status": False,
                })
                prompt = ContextBuilder(self.root, config).build(query="Class")
                stable_map = next((
                    block for block in prompt.cache_blocks
                    if block.startswith("# Repository map\n")
                ), "")
                _, separator, focus_map = prompt.dynamic.partition("# Request focus\n")
                self.assertLessEqual(len(stable_map) + len(separator) + len(focus_map), limit)
                self.assertIn("Always run tests", prompt.stable)
                self.assertIn("review:", prompt.stable)
                if limit == 1200:
                    self.assertGreater(len(stable_map), 1000)
                if limit >= 4000:
                    self.assertTrue(separator)
                    self.assertIn("Class", focus_map)

    def test_disabled_prompt_map_skips_source_discovery(self):
        config = make_config(self.root, context={"repo_map_chars": 0})
        builder = ContextBuilder(self.root, config)
        with patch.object(builder.repo_map, "snapshot", side_effect=AssertionError("source scan")):
            prompt = builder.build(query="Engine")
        self.assertNotIn("# Repository map", prompt.text)
        self.assertNotIn("# Request focus", prompt.text)
        self.assertIn("Always run tests", prompt.text)

    def test_context_build_reuses_one_fresh_discovery_snapshot(self):
        builder = ContextBuilder(self.root, self.config)
        with (
            patch.object(
                builder.instructions,
                "discover",
                wraps=builder.instructions.discover,
            ) as discover,
            patch.object(
                builder.repo_map,
                "snapshot",
                wraps=builder.repo_map.snapshot,
            ) as snapshot,
        ):
            first = builder.build(query="Engine")

        self.assertEqual(discover.call_count, 1)
        self.assertEqual(snapshot.call_count, 1)

        source = self.root / "src/main.py"
        source.write_text("class UpdatedEngine:\n    pass\n")
        second = builder.build(query="UpdatedEngine")
        self.assertIn("UpdatedEngine", second.stable)
        self.assertNotEqual(first.stable_fingerprint, second.stable_fingerprint)
        self.assertEqual(first.cache_routing_key, second.cache_routing_key)

    def test_context_build_runs_one_git_status_snapshot(self):
        builder = ContextBuilder(self.root, self.config)
        with (
            patch.object(
                builder,
                "_git_status",
                return_value=("## main\n M src/main.py", {"src/main.py"}),
            ) as status,
            patch.object(builder.repo_map, "_git_changed") as changed,
        ):
            prompt = builder.build(query="Engine")
        self.assertEqual(status.call_count, 1)
        changed.assert_not_called()
        self.assertIn("M src/main.py", prompt.dynamic)

    def test_repo_map_cache_replaces_old_file_versions(self):
        matcher = IgnoreMatcher(self.root, ignored_dirs=self.config.context.ignored_dirs)
        repo_map = RepoMap(self.root, matcher)
        repo_map.snapshot()
        original_size = len(repo_map._cache)
        source = self.root / "src/main.py"
        for index in range(20):
            source.write_text(f"class Engine{index}:\n    pass\n")
            repo_map.snapshot()
        self.assertEqual(len(repo_map._cache), original_size)
        source.unlink()
        repo_map.snapshot()
        self.assertEqual(len(repo_map._cache), original_size - 1)

    def test_repo_map_does_not_leak_workspace_syntax_warnings(self):
        source = self.root / "src/warning.py"
        source.write_text('pattern = "\\s+"\nclass WarningSource:\n    pass\n')
        matcher = IgnoreMatcher(self.root, ignored_dirs=self.config.context.ignored_dirs)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = RepoMap(self.root, matcher).build(query="WarningSource", max_chars=5000)

        self.assertIn("src/warning.py", output)
        self.assertFalse(any(item.category is SyntaxWarning for item in caught))

    def test_python_symbols_include_nested_statement_scopes(self):
        source = """import alpha
from beta import value

def top():
    if value:
        import gamma
        def nested():
            return [item * 2 for item in range(10)]
    try:
        from delta import other
    except ValueError:
        class Recovery:
            pass
    match value:
        case 1:
            async def branch():
                return other

class Outer:
    def method(self):
        from epsilon import result
        return result
"""
        symbols, imports = _python_symbols(source)
        self.assertEqual(symbols, ["top", "Outer", "method", "nested", "Recovery", "branch"])
        self.assertEqual(imports, ["alpha", "beta", "gamma", "delta", "epsilon"])

    def test_repo_map_enforces_its_character_budget(self):
        repo_map = RepoMap(self.root, IgnoreMatcher(self.root))
        summaries = [
            FileSummary(f"src/file_{index:03}.py", "Python", ["ExampleClass", "example_method"])
            for index in range(100)
        ]
        for limit in (100, 1000, 2000):
            with self.subTest(limit=limit):
                rendered = repo_map.render(summaries, max_chars=limit, rank_changed=False)
                self.assertLessEqual(len(rendered), limit)
                self.assertIn("repository map truncated", rendered)
        long_query = repo_map.render(summaries, query="x" * 2000, max_chars=1000, rank_changed=False)
        self.assertLessEqual(len(long_query), 1000)

    def test_repo_map_preserves_complete_entries_at_exact_budget(self):
        repo_map = RepoMap(self.root, IgnoreMatcher(self.root))
        cases = [
            [FileSummary("src/example.py", "Python", ["ExampleClass"])],
            [FileSummary("a.py", "Python"), FileSummary("b.py", "Python")],
        ]
        for summaries in cases:
            with self.subTest(files=len(summaries)):
                complete = repo_map.render(summaries, max_chars=1000, rank_changed=False)
                exact = repo_map.render(summaries, max_chars=len(complete), rank_changed=False)
                self.assertEqual(exact, complete)

    def test_instruction_read_enforces_limit_before_loading_contents(self):
        class BoundedText(io.StringIO):
            def read(self, size: int | None = -1):
                if size is None or not 0 <= size <= 16:
                    raise AssertionError("Instruction discovery attempted an unbounded read")
                return super().read(size)

        def open_instruction(*args, **kwargs):
            return BoundedText("Always run tests.\n" + "Additional guidance.\n" * 1000)

        loader = InstructionLoader(self.root, ["AGENTS.md"], max_chars=16)
        with patch.object(Path, "open", side_effect=open_instruction):
            documents = loader.discover()
        self.assertEqual(len(documents), 2)
        self.assertTrue(all(document.content == "Always run tests" for document in documents))

    def test_instruction_lookup_does_not_read_unrelated_guidance(self):
        unrelated = self.root / "unrelated" / "AGENTS.md"
        unrelated.parent.mkdir()
        unrelated.write_text("Unrelated guidance.")
        original_open = Path.open

        def open_instruction(path, *args, **kwargs):
            if path == unrelated:
                raise PermissionError("Unrelated guidance is unreadable")
            return original_open(path, *args, **kwargs)

        loader = InstructionLoader(self.root, ["AGENTS.md"])
        with patch.object(Path, "open", new=open_instruction):
            documents = loader.for_path(self.root / "src/main.py")
        self.assertEqual(
            [item.relative_path for item in documents], ["AGENTS.md", "src/AGENTS.md"],
        )

    def test_instruction_lookup_matches_discovery_scopes(self):
        for folder in ("src/nested", "node_modules/pkg", "src/build/pkg", "unrelated"):
            scope = self.root / folder
            scope.mkdir(parents=True)
            (scope / "AGENTS.md").write_text(f"Guidance for {folder}.")
            (scope / "RULES.md").write_text(f"Extra guidance for {folder}.")
        loader = InstructionLoader(self.root, ["RULES.md", "AGENTS.md"])
        discovered = loader.discover()
        for target in (
            self.root, self.root / "src", self.root / "src/main.py",
            self.root / "SRC/main.py",
            self.root / "src/nested/new/deep/file.py", self.root / "node_modules/pkg/file.py",
            self.root / "src/build/pkg", self.root / "unrelated", self.root.parent / "outside",
        ):
            with self.subTest(target=target):
                expected = [item for item in discovered if target.is_relative_to(item.scope)]
                self.assertEqual(loader.for_path(target), expected)

    def test_repository_map_read_stays_bounded_when_file_grows(self):
        class BoundedBytes(io.BytesIO):
            def read(self, size: int | None = -1):
                if size is None or not 0 <= size <= 129:
                    raise AssertionError("Repository map attempted an unbounded read")
                return super().read(size)

        source = self.root / "src/main.py"
        original_open = Path.open

        def open_source(path, *args, **kwargs):
            if path == source:
                return BoundedBytes(b"class TooLarge:\n    pass\n" + b"# comment\n" * 1000)
            return original_open(path, *args, **kwargs)

        repo_map = RepoMap(self.root, IgnoreMatcher(self.root), max_file_bytes=128)
        with patch.object(Path, "open", new=open_source):
            summaries = repo_map.snapshot()
        summary = next(item for item in summaries if item.path == "src/main.py")
        self.assertEqual(summary.symbols, [])

    def test_repository_map_preserves_universal_newline_symbol_matching(self):
        for index, separator in enumerate(("\n", "\r\n", "\r")):
            source = f"export function alpha() {{}}{separator}export function beta() {{}}"
            (self.root / f"src/newlines_{index}.js").write_bytes(source.encode("utf-8"))
        repo_map = RepoMap(self.root, IgnoreMatcher(self.root))
        summaries = {item.path: item for item in repo_map.snapshot()}
        for index in range(3):
            self.assertEqual(summaries[f"src/newlines_{index}.js"].symbols, ["alpha", "beta"])

    def test_external_instruction_and_skill_symlinks_are_ignored(self):
        with tempfile.TemporaryDirectory() as outside:
            secret = Path(outside) / "secret.txt"
            secret.write_text("OUTSIDE_SENTINEL", encoding="utf-8")
            external_instruction = self.root / "external" / "AGENTS.md"
            external_instruction.parent.mkdir()
            external_skill = self.root / ".agents" / "skills" / "external" / "SKILL.md"
            external_skill.parent.mkdir()
            try:
                external_instruction.symlink_to(secret)
                external_skill.symlink_to(secret)
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")
            prompt = ContextBuilder(self.root, self.config).system_prompt(query="sentinel")
            self.assertNotIn("OUTSIDE_SENTINEL", prompt)


if __name__ == "__main__":
    unittest.main()
