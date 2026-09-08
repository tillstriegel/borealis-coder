"""Hierarchical repository instruction discovery."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class InstructionDocument:
    path: Path
    relative_path: str
    content: str
    scope: Path


class InstructionLoader:
    def __init__(self, root: Path, names: list[str], *, max_chars: int = 60_000) -> None:
        self.root = root.resolve()
        self.names = names
        self.max_chars = max_chars

    def discover(self) -> list[InstructionDocument]:
        return self._discover(target=None)

    def _discover(self, *, target: Path | None) -> list[InstructionDocument]:
        result: list[InstructionDocument] = []
        ignored = {".git", ".borealis", "node_modules", ".venv", "venv", "dist", "build", "target", "__pycache__"}
        for current, dirs, files in os.walk(self.root):
            dirs[:] = sorted(item for item in dirs if item not in ignored)
            current_path = Path(current)
            if target is not None:
                remaining = target.relative_to(current_path).parts
                next_name = os.path.normcase(remaining[0]) if remaining else None
                dirs[:] = [name for name in dirs if os.path.normcase(name) == next_name]
            for name in self.names:
                if name not in files:
                    continue
                path = current_path / name
                try:
                    resolved = path.resolve(strict=True)
                    resolved.relative_to(self.root)
                except (OSError, ValueError):
                    continue
                if not resolved.is_file():
                    continue
                relative = path.relative_to(self.root).as_posix()
                with resolved.open(encoding="utf-8", errors="replace") as handle:
                    content = handle.read(self.max_chars)
                result.append(InstructionDocument(path, relative, content, path.parent))
        return sorted(result, key=lambda item: (len(item.path.parts), item.relative_path))

    def for_path(self, target: Path) -> list[InstructionDocument]:
        target = target.resolve(strict=False)
        if not target.is_relative_to(self.root):
            return []
        return self._discover(target=target)

    def root_text(
        self,
        documents: list[InstructionDocument] | None = None,
    ) -> str:
        chunks: list[str] = []
        for item in self.discover() if documents is None else documents:
            if item.scope == self.root:
                chunks.append(f"## {item.relative_path}\n\n{item.content.strip()}")
        return "\n\n".join(chunks)

    def catalog(
        self,
        documents: list[InstructionDocument] | None = None,
    ) -> str:
        source = self.discover() if documents is None else documents
        nested = [item.relative_path for item in source if item.scope != self.root]
        return "\n".join(f"- {item}" for item in nested)
