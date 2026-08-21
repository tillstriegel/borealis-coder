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
        result: list[InstructionDocument] = []
        ignored = {".git", ".borealis", "node_modules", ".venv", "venv", "dist", "build", "target", "__pycache__"}
        for current, dirs, files in os.walk(self.root):
            dirs[:] = sorted(item for item in dirs if item not in ignored)
            current_path = Path(current)
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
                content = resolved.read_text(encoding="utf-8", errors="replace")[: self.max_chars]
                result.append(InstructionDocument(path, relative, content, path.parent))
        return sorted(result, key=lambda item: (len(item.path.parts), item.relative_path))

    def for_path(self, target: Path) -> list[InstructionDocument]:
        target = target.resolve(strict=False)
        documents = self.discover()
        applicable: list[InstructionDocument] = []
        for item in documents:
            try:
                target.relative_to(item.scope)
            except ValueError:
                continue
            applicable.append(item)
        return sorted(applicable, key=lambda item: len(item.scope.parts))

    def root_text(self) -> str:
        chunks: list[str] = []
        for item in self.discover():
            if item.scope == self.root:
                chunks.append(f"## {item.relative_path}\n\n{item.content.strip()}")
        return "\n\n".join(chunks)

    def catalog(self) -> str:
        nested = [item.relative_path for item in self.discover() if item.scope != self.root]
        return "\n".join(f"- {item}" for item in nested)
