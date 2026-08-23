from __future__ import annotations

import subprocess
import tempfile
import unittest
import warnings
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
from tests.helpers import make_config


class ContextTests(unittest.TestCase):
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
