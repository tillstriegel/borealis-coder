"""Dependency-free text search with regex, glob, and context support."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from ..errors import ToolError
from ..models import Effect, ToolResult
from .base import Tool, ToolContext, nullable, object_schema


class GrepTool(Tool):
    name = "grep"
    description = "Search text files using literal or regular-expression matching."
    effect = Effect.READ
    concurrent = True
    parameters = object_schema({
        "pattern": {"type": "string", "minLength": 1},
        "path": {"type": "string"},
        "glob": nullable("string"),
        "regex": {"type": "boolean"},
        "case_sensitive": {"type": "boolean"},
        "context_lines": {"type": "integer", "minimum": 0, "maximum": 10},
        "max_results": {"type": "integer", "minimum": 1, "maximum": 2000},
    })

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        resolved = context.roots.resolve(arguments["path"] or ".", must_exist=True)
        pattern = str(arguments["pattern"])
        flags = 0 if arguments["case_sensitive"] else re.IGNORECASE
        try:
            expression = re.compile(pattern if arguments["regex"] else re.escape(pattern), flags)
        except re.error as error:
            raise ToolError(f"Invalid regular expression: {error}") from error
        files = [resolved.path] if resolved.path.is_file() else _walk_files(resolved.path, context)
        file_glob = arguments.get("glob")
        context_lines = int(arguments["context_lines"])
        max_results = int(arguments["max_results"])
        rows: list[str] = []
        match_count = 0
        scanned = 0
        for path in files:
            rel = path.relative_to(resolved.path).as_posix() if resolved.path.is_dir() else path.name
            if file_glob and not (fnmatch.fnmatch(rel, file_glob) or fnmatch.fnmatch(path.name, file_glob)):
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if len(data) > context.config.context.max_file_bytes or b"\x00" in data:
                continue
            scanned += 1
            lines = data.decode("utf-8", errors="replace").splitlines()
            emitted_context: set[int] = set()
            for index, line in enumerate(lines):
                if not expression.search(line):
                    continue
                match_count += 1
                start = max(0, index - context_lines)
                end = min(len(lines), index + context_lines + 1)
                display = context.roots.display(path)
                for ctx_index in range(start, end):
                    marker = ":" if ctx_index == index else "-"
                    key = hash((display, ctx_index))
                    if key not in emitted_context:
                        rows.append(f"{display}:{ctx_index + 1}{marker}{lines[ctx_index]}")
                        emitted_context.add(key)
                if match_count >= max_results:
                    rows.append("… result limit reached …")
                    return ToolResult("\n".join(rows), metadata={"matches": match_count, "files_scanned": scanned, "truncated": True})
        return ToolResult("\n".join(rows) or "No matches", metadata={"matches": match_count, "files_scanned": scanned})


def _walk_files(root: Path, context: ToolContext) -> list[Path]:
    ignored = set(context.config.context.ignored_dirs)
    result: list[Path] = []
    for current, dirs, files in os.walk(root):
        dirs[:] = [item for item in dirs if item not in ignored]
        result.extend(Path(current) / item for item in files)
    return result
