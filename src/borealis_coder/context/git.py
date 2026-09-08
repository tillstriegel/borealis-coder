"""Git status for repository context, with lossless filename parsing."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def quote_path(path: str) -> str:
    """Keep ordinary paths readable and escape ambiguous display characters."""
    if path.isprintable() and not any(char.isspace() or char in '\\"' for char in path):
        return path
    return json.dumps(path, ensure_ascii=not path.isprintable())


def repository_status(root: Path) -> tuple[str, set[str]]:
    try:
        result = subprocess.run(
            [
                "git", "--no-optional-locks",
                "-c", "core.fsmonitor=false",
                "-c", "core.hooksPath=/dev/null",
                "-C", str(root),
                "status", "--porcelain=v1", "--branch", "-z",
            ],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "", set()
    if result.returncode:
        return "", set()
    rows: list[str] = []
    changed: set[str] = set()
    entries = iter(os.fsdecode(result.stdout).split("\x00"))
    for entry in entries:
        if entry.startswith("##"):
            rows.append(entry)
            continue
        if len(entry) < 4:
            continue
        status, path = entry[:2], entry[3:]
        changed.add(path)
        display = quote_path(path)
        if "R" in status or "C" in status:
            # NUL-delimited porcelain puts the destination before the source.
            source = next(entries, "")
            display = f"{quote_path(source)} -> {display}"
        rows.append(f"{status} {display}")
    return "\n".join(rows), changed
