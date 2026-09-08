"""Workspace-root path enforcement with symlink escape protection."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path

from ..errors import PathViolation


@dataclass(frozen=True, slots=True)
class ResolvedPath:
    path: Path
    root: Path
    display: str


class WorkspaceRoots:
    """Resolve paths against explicit roots and reject traversal/symlink escapes."""

    def __init__(
        self,
        primary: Path,
        additional: list[Path] | None = None,
        *,
        allow_outside: bool = False,
    ) -> None:
        primary = primary.expanduser().resolve()
        if not primary.is_dir():
            raise PathViolation(f"Workspace does not exist or is not a directory: {primary}")
        roots = [primary]
        for item in additional or []:
            resolved = item.expanduser().resolve()
            if not resolved.is_dir():
                raise PathViolation(f"Additional workspace root is not a directory: {resolved}")
            if resolved not in roots:
                roots.append(resolved)
        self.primary = primary
        self.roots = tuple(roots)
        self.allow_outside = allow_outside

    def resolve(
        self,
        value: str | Path,
        *,
        must_exist: bool = False,
        kind: str = "any",
    ) -> ResolvedPath:
        """Expand user-supplied strings; preserve literal filesystem Path values."""

        raw = value if isinstance(value, Path) else Path(os.path.expandvars(os.path.expanduser(value)))
        candidate = raw if raw.is_absolute() else self.primary / raw
        # strict=False still resolves existing symlink parents, which prevents writing
        # through an in-workspace symlink to an outside location.
        candidate = candidate.resolve(strict=False)
        root = self._containing_root(candidate)
        if root is None and not self.allow_outside:
            raise PathViolation(f"Path escapes configured workspace roots: {value}")
        display = self._display_resolved(candidate, root)
        if root is None:
            root = Path(candidate.anchor)
        if must_exist and not candidate.exists():
            raise PathViolation(f"Path does not exist: {display}")
        if kind == "file" and candidate.exists() and not candidate.is_file():
            raise PathViolation(f"Expected a file: {display}")
        if kind == "dir" and candidate.exists() and not candidate.is_dir():
            raise PathViolation(f"Expected a directory: {display}")
        return ResolvedPath(candidate, root, display)

    def _containing_root(self, candidate: Path) -> Path | None:
        matches: list[Path] = []
        for root in self.roots:
            try:
                candidate.relative_to(root)
                matches.append(root)
            except ValueError:
                continue
        return max(matches, key=lambda item: len(item.parts), default=None)

    def contains(self, path: Path) -> bool:
        return self._containing_root(path.resolve(strict=False)) is not None

    def assert_writable(self, path: Path, protected_patterns: list[str]) -> None:
        """Reject writes to harness recovery state and repository internals.

        Patterns are evaluated relative to the containing workspace root. This
        intentionally applies to every configured root, not only the primary one.
        """
        resolved = path.resolve(strict=False)
        root = self._containing_root(resolved)
        if root is None:
            if self.allow_outside:
                return
            raise PathViolation(f"Path escapes configured workspace roots: {path}")
        relative = resolved.relative_to(root).as_posix()
        for pattern in protected_patterns:
            normalized = pattern.strip().lstrip("/")
            if normalized and (
                fnmatch.fnmatch(relative, normalized)
                or fnmatch.fnmatch(relative, f"**/{normalized}")
            ):
                raise PathViolation(f"Path is protected from direct mutation: {self.display(resolved)}")

    def display(self, path: Path) -> str:
        resolved = path.resolve(strict=False)
        root = self._containing_root(resolved)
        return self._display_resolved(resolved, root)

    def _display_resolved(self, resolved: Path, root: Path | None) -> str:
        if root is None:
            return str(resolved)
        relative = resolved.relative_to(root)
        if root == self.primary:
            return relative.as_posix() or "."
        return f"{root.name}:{relative.as_posix()}"
