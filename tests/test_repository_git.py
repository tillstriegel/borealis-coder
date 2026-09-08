from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from borealis_coder.context import ContextBuilder, IgnoreMatcher, RepoMap, repository_files
from tests.helpers import make_config


class RepositoryGitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.git("init", "-q")
        self.config = make_config(self.root)
        self.matcher = IgnoreMatcher(self.root, ignored_dirs=self.config.context.ignored_dirs)

    def git(self, *arguments):
        return subprocess.run(
            [
                "git", "-C", str(self.root),
                "-c", "core.hooksPath=/dev/null",
                "-c", "commit.gpgsign=false",
                "-c", "user.name=Borealis Test",
                "-c", "user.email=test@example.com",
                *arguments,
            ],
            check=True,
            capture_output=True,
        )

    def write_source(self, name):
        path = self.root / name
        path.write_text("class ImportantEngine:\n    pass\n", encoding="utf-8")
        return path

    def test_git_discovery_preserves_unicode_and_spaces(self):
        names = {"über.py", "two words.py"}
        for name in names:
            self.write_source(name)
        for tracked in (False, True):
            with self.subTest(tracked=tracked):
                if tracked:
                    self.git("add", "--", *names)
                paths = repository_files(self.root, self.matcher)
                self.assertEqual({path.name for path in paths}, names)
                summaries = RepoMap(self.root, self.matcher).snapshot()
                self.assertEqual({summary.path for summary in summaries}, names)
                self.assertTrue(all("ImportantEngine" in item.symbols for item in summaries))

    @unittest.skipIf(os.name == "nt", "filenames require POSIX")
    def test_git_discovery_preserves_control_characters_and_quotes(self):
        names = {'quote".py', "line\nbreak.py", "tab\tname.py", "carriage\rreturn.py"}
        for name in names:
            self.write_source(name)
        self.git("add", "--", *names)
        paths = repository_files(self.root, self.matcher)
        self.assertEqual({path.name for path in paths}, names)

    def test_changed_paths_preserve_unicode_and_spaces(self):
        names = {"über.py", "two words.py"}
        for name in names:
            self.write_source(name)
        self.git("add", "--", *names)
        builder = ContextBuilder(self.root, self.config)
        status, changed = builder._git_status()
        self.assertEqual(changed, names)
        self.assertIn("über.py", status)
        self.assertNotIn("\x00", status)
        self.assertEqual(set(builder.repo_map._git_changed()), names)

    def test_rename_status_ranks_destination_and_keeps_following_entries(self):
        self.write_source("old name.py")
        self.git("add", "--", "old name.py")
        self.git("commit", "-qm", "Test fixture")
        self.git("mv", "--", "old name.py", "über.py")
        self.write_source("z.py")
        builder = ContextBuilder(self.root, self.config)
        status, changed = builder._git_status()
        self.assertEqual(changed, {"über.py", "z.py"})
        self.assertIn("old name.py", status)
        self.assertIn("über.py", status)
        self.assertEqual(set(builder.repo_map._git_changed()), changed)

    @unittest.skipIf(os.name == "nt", "filenames require POSIX")
    def test_status_escapes_control_characters_for_display_only(self):
        name = "line\n M fake.py"
        self.write_source(name)
        builder = ContextBuilder(self.root, self.config)
        status, changed = builder._git_status()
        self.assertEqual(changed, {name})
        self.assertNotIn("\n M fake.py", status)
        self.assertIn("\\n M fake.py", status)
        rendered = builder.repo_map.build(rank_changed=False)
        self.assertNotIn("\n M fake.py", rendered)
        self.assertIn("\\n M fake.py", rendered)

    def test_ignore_rules_and_untracked_option_are_preserved(self):
        self.write_source("tracked.py")
        self.git("add", "--", "tracked.py")
        self.write_source("untracked.py")
        self.write_source("ignored.py")
        (self.root / ".borealisignore").write_text("ignored.py\n", encoding="utf-8")
        tracked = repository_files(self.root, self.matcher, include_untracked=False)
        self.assertEqual({path.name for path in tracked}, {"tracked.py"})
        all_files = repository_files(self.root, self.matcher)
        self.assertIn("untracked.py", {path.name for path in all_files})
        self.assertNotIn("ignored.py", {path.name for path in all_files})
