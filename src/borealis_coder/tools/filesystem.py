"""Deterministic, hash-aware filesystem tools."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any

from ..errors import ToolError
from ..models import Effect, ToolResult
from ..util import atomic_write_text, sha256_bytes, sha256_text, truncate_text
from .base import MutationScope, Tool, ToolContext, nullable, object_schema


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a UTF-8 text file with stable line numbers and a SHA-256 concurrency token."
    effect = Effect.READ
    concurrent = True
    parameters = object_schema({
        "path": {"type": "string", "minLength": 1},
        "start_line": nullable("integer", minimum=1),
        "end_line": nullable("integer", minimum=1),
        "max_chars": nullable("integer", minimum=100, maximum=1_000_000),
    })

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        resolved = context.roots.resolve(arguments["path"], must_exist=True, kind="file")
        data = resolved.path.read_bytes()
        if len(data) > context.config.context.max_file_bytes:
            raise ToolError(f"File exceeds {context.config.context.max_file_bytes} byte read limit")
        if b"\x00" in data:
            raise ToolError("Binary file; use a specialized MCP or client tool")
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        start = int(arguments.get("start_line") or 1)
        end = int(arguments.get("end_line") or len(lines))
        if end < start:
            raise ToolError("end_line must be >= start_line")
        selected = lines[start - 1 : end]
        numbered = "".join(f"{index:>6}\t{line}" for index, line in enumerate(selected, start=start))
        limit = int(arguments.get("max_chars") or context.config.context.tool_output_chars)
        numbered = truncate_text(numbered, limit)
        return ToolResult(
            f"path: {resolved.display}\nsha256: {sha256_bytes(data)}\nlines: {len(lines)}\n\n{numbered}",
            metadata={"path": resolved.display, "sha256": sha256_bytes(data), "line_count": len(lines)},
        )


class WriteFileTool(Tool):
    name = "write_file"
    description = "Atomically create or replace a UTF-8 text file. Use expected_sha256 to prevent stale writes."
    effect = Effect.WRITE
    mutation_scope = MutationScope.TRACKED
    default_risk = "medium"
    parameters = object_schema({
        "path": {"type": "string", "minLength": 1},
        "content": {"type": "string", "maxLength": 5_000_000},
        "expected_sha256": nullable("string", minLength=64, maxLength=64),
    })

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        resolved = context.roots.resolve(arguments["path"])
        context.roots.assert_writable(resolved.path, context.config.safety.protected_paths)
        expected = arguments.get("expected_sha256")
        existed = resolved.path.exists()
        before = resolved.path.read_bytes() if existed and resolved.path.is_file() else None
        if existed and not resolved.path.is_file():
            raise ToolError(f"Target is not a file: {resolved.display}")
        if existed and expected is None:
            raise ToolError(f"Existing file requires expected_sha256 from read_file: {resolved.display}")
        if expected is not None:
            actual = sha256_bytes(before or b"")
            if actual != expected:
                raise ToolError(f"Stale write rejected for {resolved.display}: expected {expected}, actual {actual}")
        content = str(arguments["content"])
        size = len(content.encode("utf-8"))
        if size > context.config.context.max_file_bytes:
            raise ToolError(
                f"Write would exceed {context.config.context.max_file_bytes} byte file limit"
            )
        checkpoint = context.checkpoints.create([resolved.path], label=f"write_file {resolved.display}")
        atomic_write_text(resolved.path, content)
        context.changed_files.add(resolved.display)
        context.changed_roots.add(resolved.root)
        return ToolResult(
            f"Wrote {size} bytes to {resolved.display}\nsha256: {sha256_text(content)}",
            metadata={"path": resolved.display, "sha256": sha256_text(content), "checkpoint_id": checkpoint.id if checkpoint else None, "created": not existed},
        )


class ReplaceInFileTool(Tool):
    name = "replace_in_file"
    description = "Atomically replace exact text in a file with occurrence-count and hash guards."
    effect = Effect.WRITE
    mutation_scope = MutationScope.TRACKED
    default_risk = "medium"
    parameters = object_schema({
        "path": {"type": "string", "minLength": 1},
        "old_text": {"type": "string", "minLength": 1, "maxLength": 2_000_000},
        "new_text": {"type": "string", "maxLength": 2_000_000},
        "expected_occurrences": {"type": "integer", "minimum": 1},
        "expected_sha256": nullable("string", minLength=64, maxLength=64),
    })

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        resolved = context.roots.resolve(arguments["path"], must_exist=True, kind="file")
        context.roots.assert_writable(resolved.path, context.config.safety.protected_paths)
        data = resolved.path.read_bytes()
        if len(data) > context.config.context.max_file_bytes:
            raise ToolError(f"File exceeds {context.config.context.max_file_bytes} byte edit limit")
        actual_sha = sha256_bytes(data)
        expected_sha = arguments.get("expected_sha256")
        if expected_sha is not None and actual_sha != expected_sha:
            raise ToolError(f"Stale edit rejected: expected {expected_sha}, actual {actual_sha}")
        text = data.decode("utf-8")
        old = str(arguments["old_text"])
        count = text.count(old)
        expected_count = int(arguments["expected_occurrences"])
        if count != expected_count:
            raise ToolError(f"Expected {expected_count} occurrence(s), found {count}; no changes made")
        updated = text.replace(old, str(arguments["new_text"]))
        if len(updated.encode("utf-8")) > context.config.context.max_file_bytes:
            raise ToolError(
                f"Edit would exceed {context.config.context.max_file_bytes} byte file limit"
            )
        checkpoint = context.checkpoints.create([resolved.path], label=f"replace_in_file {resolved.display}")
        atomic_write_text(resolved.path, updated)
        context.changed_files.add(resolved.display)
        context.changed_roots.add(resolved.root)
        return ToolResult(
            f"Replaced {count} occurrence(s) in {resolved.display}\nsha256: {sha256_text(updated)}",
            metadata={"path": resolved.display, "replacements": count, "sha256": sha256_text(updated), "checkpoint_id": checkpoint.id if checkpoint else None},
        )


class DeleteFileTool(Tool):
    name = "delete_file"
    description = "Delete one workspace file after checkpointing it. Directories are never deleted."
    effect = Effect.WRITE
    mutation_scope = MutationScope.TRACKED
    default_risk = "high"
    parameters = object_schema({
        "path": {"type": "string", "minLength": 1},
        "expected_sha256": {"type": "string", "minLength": 64, "maxLength": 64},
    })

    def approval_description(self, arguments: dict[str, Any]) -> str:
        return f"Delete file {arguments.get('path')}"

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        resolved = context.roots.resolve(arguments["path"], must_exist=True, kind="file")
        context.roots.assert_writable(resolved.path, context.config.safety.protected_paths)
        actual = sha256_bytes(resolved.path.read_bytes())
        if actual != arguments["expected_sha256"]:
            raise ToolError(f"Stale delete rejected: expected {arguments['expected_sha256']}, actual {actual}")
        checkpoint = context.checkpoints.create([resolved.path], label=f"delete_file {resolved.display}")
        resolved.path.unlink()
        context.changed_files.add(resolved.display)
        context.changed_roots.add(resolved.root)
        return ToolResult(f"Deleted {resolved.display}", metadata={"path": resolved.display, "checkpoint_id": checkpoint.id if checkpoint else None})


class MakeDirectoryTool(Tool):
    name = "make_directory"
    description = "Create a directory and missing parents inside the workspace."
    effect = Effect.WRITE
    mutation_scope = MutationScope.TRACKED
    default_risk = "low"
    parameters = object_schema({"path": {"type": "string", "minLength": 1}})

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        resolved = context.roots.resolve(arguments["path"])
        context.roots.assert_writable(resolved.path, context.config.safety.protected_paths)
        missing_directories: list[Path] = []
        candidate = resolved.path
        while not candidate.exists() and candidate != resolved.root:
            missing_directories.append(candidate)
            candidate = candidate.parent
        resolved.path.mkdir(parents=True, exist_ok=True)
        for path in missing_directories:
            context.changed_files.add(context.roots.display(path))
        if missing_directories:
            context.changed_roots.add(resolved.root)
        return ToolResult(f"Directory ready: {resolved.display}", metadata={"path": resolved.display})


class ListDirectoryTool(Tool):
    name = "list_directory"
    description = "List a directory tree with bounded depth and entry count."
    effect = Effect.READ
    concurrent = True
    parameters = object_schema({
        "path": {"type": "string"},
        "depth": {"type": "integer", "minimum": 1, "maximum": 8},
        "include_hidden": {"type": "boolean"},
        "max_entries": {"type": "integer", "minimum": 1, "maximum": 5000},
    })

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        resolved = context.roots.resolve(arguments["path"] or ".", must_exist=True, kind="dir")
        max_depth = int(arguments["depth"])
        max_entries = int(arguments["max_entries"])
        include_hidden = bool(arguments["include_hidden"])
        ignored = set(context.config.context.ignored_dirs)
        rows: list[str] = []
        base_depth = len(resolved.path.parts)
        for current, dirs, files in os.walk(resolved.path):
            current_path = Path(current)
            depth = len(current_path.parts) - base_depth
            dirs[:] = sorted(
                item for item in dirs
                if item not in ignored and (include_hidden or not item.startswith("."))
            )
            if depth >= max_depth:
                dirs[:] = []
            entries = [(item, True) for item in dirs] + [(item, False) for item in sorted(files)]
            for name, is_dir in entries:
                if not include_hidden and name.startswith("."):
                    continue
                path = current_path / name
                rel = path.relative_to(resolved.path).as_posix()
                rows.append(f"{'  ' * depth}{rel}{'/' if is_dir else ''}")
                if len(rows) >= max_entries:
                    rows.append("… entry limit reached …")
                    return ToolResult("\n".join(rows), metadata={"truncated": True, "entries": max_entries})
        return ToolResult("\n".join(rows) or "(empty directory)", metadata={"entries": len(rows)})


class GlobFilesTool(Tool):
    name = "glob_files"
    description = "Find workspace files matching a glob pattern."
    effect = Effect.READ
    concurrent = True
    parameters = object_schema({
        "pattern": {"type": "string", "minLength": 1},
        "path": {"type": "string"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 5000},
    })

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        resolved = context.roots.resolve(arguments["path"] or ".", must_exist=True, kind="dir")
        pattern = str(arguments["pattern"])
        limit = min(
            int(arguments["limit"]),
            context.config.context.max_search_results,
        )
        ignored = set(context.config.context.ignored_dirs)
        matches: list[str] = []
        for current, dirs, files in os.walk(resolved.path):
            dirs[:] = [item for item in dirs if item not in ignored]
            for name in files:
                path = Path(current) / name
                rel = path.relative_to(resolved.path).as_posix()
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
                    matches.append(context.roots.display(path))
                    if len(matches) >= limit:
                        return ToolResult("\n".join(sorted(matches)), metadata={"truncated": True, "matches": len(matches)})
        matches.sort()
        return ToolResult("\n".join(matches) or "No matches", metadata={"matches": len(matches)})
