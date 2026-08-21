from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

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
        (self.root / "src/main.py").write_text("import json\nclass Engine:\n    def run(self):\n        return json.dumps({})\n")
        (self.root / "src/other.ts").write_text("export function helper() { return 1 }\n")
        (self.root / "AGENTS.md").write_text("Always run tests.\n")
        (self.root / "src/AGENTS.md").write_text("Use Python typing.\n")
        skill = self.root / ".agents/skills/review"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: review\ndescription: Review code carefully\n---\n# Review\nInspect tests.\n")
        (self.root / ".gitignore").write_text("ignored.txt\n")
        (self.root / "ignored.txt").write_text("secret")
        self.config = make_config(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_ignore_and_files(self):
        matcher = IgnoreMatcher(self.root, ignored_dirs=self.config.context.ignored_dirs)
        files = [item.relative_to(self.root).as_posix() for item in repository_files(self.root, matcher)]
        self.assertNotIn("ignored.txt", files)
        self.assertIn("src/main.py", files)

    def test_instructions_and_skills(self):
        loader = InstructionLoader(self.root, ["AGENTS.md"])
        applicable = loader.for_path(self.root / "src/main.py")
        self.assertEqual([item.relative_path for item in applicable], ["AGENTS.md", "src/AGENTS.md"])
        catalog = SkillCatalog(self.root, [".agents/skills"])
        review = catalog.get("review")
        assert review is not None
        self.assertEqual(review.description, "Review code carefully")

    def test_repo_map_ranking_and_system_prompt(self):
        matcher = IgnoreMatcher(self.root, ignored_dirs=self.config.context.ignored_dirs)
        output = RepoMap(self.root, matcher).build(query="Engine run", max_chars=5000)
        self.assertIn("src/main.py", output)
        self.assertIn("Engine", output)
        prompt = ContextBuilder(self.root, self.config).system_prompt(query="Engine")
        self.assertIn("Always run tests", prompt)
        self.assertIn("Review code carefully", prompt)

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
