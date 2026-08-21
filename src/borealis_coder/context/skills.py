"""Project-local skill discovery and lazy loading."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    path: Path
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillCatalog:
    def __init__(self, root: Path, directories: list[str]) -> None:
        self.root = root.resolve()
        self.directories = directories
        self._skills: dict[str, Skill] | None = None

    def discover(self) -> dict[str, Skill]:
        if self._skills is not None:
            return self._skills
        skills: dict[str, Skill] = {}
        for directory in self.directories:
            base = (self.root / directory).resolve(strict=False)
            try:
                base.relative_to(self.root)
            except ValueError:
                continue
            if not base.is_dir():
                continue
            for path in sorted(base.rglob("SKILL.md")):
                try:
                    resolved = path.resolve(strict=True)
                    resolved.relative_to(self.root)
                except (OSError, ValueError):
                    continue
                if not resolved.is_file():
                    continue
                text = resolved.read_text(encoding="utf-8", errors="replace")
                metadata, body = _frontmatter(text)
                name = str(metadata.get("name") or path.parent.name).strip()
                description = str(metadata.get("description") or _first_paragraph(body)).strip()
                if name and name not in skills:
                    skills[name] = Skill(name, description, resolved, body, metadata)
        self._skills = skills
        return skills

    def get(self, name: str) -> Skill | None:
        return self.discover().get(name)

    def render_catalog(self) -> str:
        skills = self.discover()
        if not skills:
            return "(no project skills discovered)"
        return "\n".join(f"- {name}: {skill.description}" for name, skill in sorted(skills.items()))


def _frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    metadata: dict[str, Any] = {}
    for line in text[4:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"\'')
    return metadata, text[end + 5 :]


def _first_paragraph(text: str) -> str:
    cleaned = re.sub(r"^#+\s+.*$", "", text, flags=re.MULTILINE).strip()
    return cleaned.split("\n\n", 1)[0].replace("\n", " ")[:500]
