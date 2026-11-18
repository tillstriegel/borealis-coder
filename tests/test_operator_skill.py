from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from borealis_coder.cli import build_parser
from borealis_coder.context.skills import SkillCatalog

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "borealis" / "SKILL.md"


class OperatorSkillTests(unittest.TestCase):
    def commands(self) -> list[list[str]]:
        commands = [
            shlex.split(block.strip())
            for block in re.findall(r"```bash\n(.*?)\n\s*```", SKILL.read_text(), re.DOTALL)
        ]
        self.assertEqual(len(commands), 5)
        return commands

    def test_skill_discovery_and_documented_arguments(self) -> None:
        skill = SkillCatalog(ROOT, ["skills"]).get("borealis")
        self.assertIsNotNone(skill)
        assert skill is not None
        self.assertTrue(skill.description)
        self.assertIn("## Verification", skill.body)
        parser = build_parser()
        commands = self.commands()
        self.assertEqual(len(commands), 5)
        for command in commands:
            with self.subTest(command=command):
                self.assertEqual(command[0], "borealis")
                args = parser.parse_args(command[1:])
                if args.command == "run":
                    self.assertTrue(args.non_interactive)
                    self.assertTrue(args.json)
                    self.assertFalse(args.network)
                    self.assertEqual(args.approval, "on-risk")

    def test_documented_subprocess_result_and_resume_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            workspace = base / "workspace"
            workspace.mkdir()
            home = base / "home"
            home.mkdir()
            env = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("BOREALIS_")
            }
            env.update(
                HOME=str(home),
                USERPROFILE=str(home),
                XDG_CONFIG_HOME=str(home / ".config"),
                BOREALIS_DATA_DIR=str(base / "data"),
                BOREALIS_PROVIDER="mock",
                PYTHONPATH=str(ROOT / "src"),
            )
            session_id = ""
            for index, template in enumerate(self.commands()):
                command = [
                    str(workspace) if arg == "/path/to/project" else
                    session_id if arg == "SESSION_ID" else arg
                    for arg in template[1:]
                ]
                if index == 1:
                    # Exercise a real deterministic edit, not only mock end-turn prose.
                    command[-1] = "OFFLINE_WRITE_DEMO"
                process = subprocess.run(
                    [sys.executable, "-m", "borealis_coder", *command],
                    cwd=base,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                with self.subTest(command=command):
                    self.assertEqual(process.returncode, 0, process.stderr)
                    if command[0] == "run":
                        result = json.loads(process.stdout)
                        self.assertEqual(result["stop_reason"], "end_turn")
                        self.assertFalse(result["incomplete"])
                        self.assertIsNone(result["error"])
                        self.assertEqual(result["mutation_tracking"], "complete")
                        self.assertIn("run_id", result)
                        self.assertIn("usage", result)
                        self.assertIn("verification", result)
                        self.assertIn('"event"', process.stderr)
                        if index == 0:
                            self.assertEqual(result["changed_files"], [])
                            self.assertFalse((workspace / "borealis-demo.txt").exists())
                        if index == 1:
                            self.assertIn("borealis-demo.txt", result["changed_files"])
                            self.assertTrue((workspace / "borealis-demo.txt").is_file())
                            self.assertIsNotNone(result["verification"])
                        if index == 2:
                            self.assertEqual(result["session_id"], session_id)
                        session_id = result["session_id"]
                    elif command[1] == "show":
                        self.assertEqual(json.loads(process.stdout)["session"]["id"], session_id)
                    else:
                        exported = json.loads((base / "session.json").read_text())
                        self.assertEqual(exported["session"]["id"], session_id)
