"""Repository file discovery with git-aware and dependency-free fallback paths."""

from __future__ import annotations

import fnmatch
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class IgnoreRule:
    pattern: str
    negate: bool = False
    directory_only: bool = False
    anchored: bool = False


class IgnoreMatcher:
    def __init__(self, root: Path, *, ignored_dirs: list[str] | None = None) -> None:
        self.root = root.resolve()
        self.ignored_dirs = set(ignored_dirs or [])
        self.rules: list[IgnoreRule] = []
        self._paths = tuple(self.root / name for name in (".gitignore", ".borealisignore"))
        self._signature: tuple[tuple[int, int] | None, ...] | None = None
        self._lock = threading.Lock()
        self.refresh()

    def refresh(self) -> bool:
        """Reload changed ignore files and report whether the rules changed."""

        signature = self._current_signature()
        with self._lock:
            if signature == self._signature:
                return False
        rules: list[IgnoreRule] = []
        for path in self._paths:
            if path.is_file():
                try:
                    rules.extend(_parse_rules(path.read_text(encoding="utf-8", errors="replace")))
                except OSError:
                    continue
        with self._lock:
            self.rules = rules
            self._signature = signature
        return True

    def _current_signature(self) -> tuple[tuple[int, int] | None, ...]:
        signature: list[tuple[int, int] | None] = []
        for path in self._paths:
            try:
                stat = path.stat()
            except OSError:
                signature.append(None)
            else:
                signature.append((stat.st_mtime_ns, stat.st_size))
        return tuple(signature)

    def ignored(self, path: Path, *, is_dir: bool | None = None) -> bool:
        try:
            relative = path.resolve(strict=False).relative_to(self.root).as_posix()
        except ValueError:
            return True
        if not relative or relative == ".":
            return False
        parts = relative.split("/")
        if any(part in self.ignored_dirs for part in parts):
            return True
        state = False
        with self._lock:
            rules = tuple(self.rules)
        for rule in rules:
            if _matches_rule(relative, parts, rule, is_dir=is_dir):
                state = not rule.negate
        return state


def repository_files(
    root: Path, matcher: IgnoreMatcher, *, include_untracked: bool = True
) -> list[Path]:
    """Prefer git's ignore engine, then fall back to a bounded filesystem walk."""
    root = root.resolve()
    matcher.refresh()
    if (root / ".git").exists():
        args = [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(root),
            "ls-files",
            "--cached",
        ]
        if include_untracked:
            args += ["--others", "--exclude-standard"]
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
            if result.returncode == 0:
                paths = []
                for line in result.stdout.splitlines():
                    path = (root / line).resolve(strict=False)
                    if path.is_file() and not matcher.ignored(path, is_dir=False):
                        paths.append(path)
                return sorted(set(paths), key=lambda item: item.as_posix())
        except (OSError, subprocess.SubprocessError):
            pass
    paths: list[Path] = []
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        dirs[:] = sorted(
            item for item in dirs if not matcher.ignored(current_path / item, is_dir=True)
        )
        for name in sorted(files):
            path = current_path / name
            if not matcher.ignored(path, is_dir=False):
                paths.append(path)
    return paths


def _parse_rules(text: str) -> list[IgnoreRule]:
    rules: list[IgnoreRule] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        if negate:
            line = line[1:]
        anchored = line.startswith("/")
        if anchored:
            line = line[1:]
        directory_only = line.endswith("/")
        line = line.rstrip("/")
        if line:
            rules.append(IgnoreRule(line, negate, directory_only, anchored))
    return rules


def _matches_rule(
    relative: str,
    parts: list[str],
    rule: IgnoreRule,
    *,
    is_dir: bool | None,
) -> bool:
    candidates = (relative,)
    if rule.directory_only:
        directory_count = len(parts) if is_dir is not False else len(parts) - 1
        candidates = tuple(
            "/".join(parts[:index])
            for index in range(1, directory_count + 1)
        )
    for candidate in candidates:
        if rule.anchored and fnmatch.fnmatch(candidate, rule.pattern):
            return True
        if "/" in rule.pattern and not rule.anchored:
            if fnmatch.fnmatch(candidate, rule.pattern) or fnmatch.fnmatch(
                candidate,
                f"**/{rule.pattern}",
            ):
                return True
        elif not rule.anchored and any(
            fnmatch.fnmatch(part, rule.pattern)
            for part in candidate.split("/")
        ):
            return True
    return False
