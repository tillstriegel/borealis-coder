"""Deterministic subprocess smoke test for the installed-style interactive CLI."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_cli(
    args: list[str], *, stdin: str, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "borealis_coder", *args],
        cwd=ROOT,
        env=env,
        input=stdin,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="borealis-interactive-smoke-") as temp:
        base = Path(temp)
        workspace = base / "workspace"
        workspace.mkdir()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        env["BOREALIS_DATA_DIR"] = str(base / "data")
        env["BOREALIS_PROVIDER"] = "mock"
        env["BOREALIS_APPROVAL"] = "never"

        first = run_cli(
            [
                "--workspace",
                str(workspace),
                "--provider",
                "mock",
                "--no-verify",
                "--no-history",
            ],
            stdin="first turn\nsecond turn\n/session\n/quit\n",
            env=env,
        )
        assert first.returncode == 0, (first.stdout, first.stderr)
        assert "interactive mode" in first.stdout
        assert "\x1b[" not in first.stdout
        assert first.stdout.count("Offline mock response") >= 2
        match = re.search(r"(?:session:\s+|SESSION\s+)(sess_[A-Za-z0-9_-]+)", first.stdout)
        assert match, first.stdout
        session_id = match.group(1)

        resumed = run_cli(
            [
                "--workspace",
                str(workspace),
                "--provider",
                "mock",
                "--continue",
                "--no-verify",
                "--no-history",
            ],
            stdin="/session\n/quit\n",
            env=env,
        )
        assert resumed.returncode == 0, (resumed.stdout, resumed.stderr)
        assert session_id in resumed.stdout
        assert "(resumed)" in resumed.stdout

        direct = run_cli(
            [
                "direct initial prompt",
                "--workspace",
                str(workspace),
                "--provider",
                "mock",
                "--no-verify",
                "--no-history",
            ],
            stdin="/quit\n",
            env=env,
        )
        assert direct.returncode == 0, (direct.stdout, direct.stderr)
        assert "Offline mock response" in direct.stdout
        assert "TURN COMPLETE" in direct.stdout
        assert "end_turn" in direct.stdout

    print("Borealis interactive CLI smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
