"""Dependency-free text search with regex, glob, and context support."""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import json
import os
import re
import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from ..errors import PathViolation, ToolError
from ..models import Effect, ToolResult
from ..util import finish_on_cancellation, read_bytes_up_to, truncate_text
from .base import Tool, ToolContext, nullable, object_schema


class _RegexMatcher:
    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process

    async def exchange(self, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        assert self.process.stdin is not None and self.process.stdout is not None
        try:
            async with asyncio.timeout(timeout):
                self.process.stdin.write(json.dumps(payload, ensure_ascii=True).encode() + b"\n")
                await self.process.stdin.drain()
                line = await self.process.stdout.readline()
            if not line:
                raise ToolError("Regular-expression matching stopped unexpectedly")
            response = json.loads(line)
        except TimeoutError as error:
            raise ToolError("Regular-expression matching timed out; simplify the pattern or use literal search") from error
        except (OSError, ValueError) as error:
            raise ToolError("Regular-expression matching failed") from error
        if not isinstance(response, dict):
            raise ToolError("Regular-expression matching returned an invalid response")
        if "error" in response:
            raise ToolError(str(response["error"]))
        return response

    async def search(self, lines: list[str], limit: int, timeout: float) -> list[int]:
        response = await self.exchange(
            {"lines": lines, "limit": limit}, timeout=timeout,
        )
        return response["matches"]


@asynccontextmanager
async def _regex_matcher(
    pattern: str, flags: int, *, enabled: bool, max_matches: int,
) -> AsyncIterator[_RegexMatcher | None]:
    if not enabled:
        yield None
        return
    try:
        # Each result is a line index plus a JSON separator. Allow the whole
        # bounded response when callers configure more than the default results.
        response_limit = max_matches * (len(str(sys.maxsize)) + 2) + 64
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-I", str(Path(__file__).with_name("_regex_worker.py")),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=max(65_536, response_limit),
        )
    except OSError as error:
        raise ToolError("Could not start regular-expression matching") from error
    try:
        matcher = _RegexMatcher(process)
        await matcher.exchange({"pattern": pattern, "flags": flags}, timeout=5)
        yield matcher
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        await finish_on_cancellation(process.wait())


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
        expression = None if arguments["regex"] else re.compile(re.escape(pattern), flags)
        is_directory = resolved.path.is_dir()
        files = _walk_files(resolved.path, context) if is_directory else iter([resolved.path])
        file_glob = arguments.get("glob")
        context_lines = int(arguments["context_lines"])
        max_results = min(
            int(arguments["max_results"]),
            context.config.context.max_search_results,
        )
        rows: list[str] = []
        output_chars = 0
        output_limit = context.config.context.tool_output_chars
        file_limit = context.config.context.max_file_bytes
        match_count = 0
        scanned = 0
        async with _regex_matcher(
            pattern, flags, enabled=bool(arguments["regex"]), max_matches=max_results,
        ) as matcher:
            for path in files:
                await asyncio.sleep(0)
                rel = path.relative_to(resolved.path).as_posix() if is_directory else path.name
                if file_glob and not (fnmatch.fnmatch(rel, file_glob) or fnmatch.fnmatch(path.name, file_glob)):
                    continue
                try:
                    candidate = context.roots.resolve(path, must_exist=True, kind="file")
                    data = read_bytes_up_to(candidate.path, file_limit + 1)
                except (OSError, PathViolation):
                    continue
                if len(data) > file_limit or b"\x00" in data:
                    continue
                scanned += 1
                display = candidate.display
                lines = data.decode("utf-8", errors="replace").splitlines()
                emitted_context: set[int] = set()
                indices = (
                    await matcher.search(lines, max_results - match_count, context.config.context.regex_timeout_seconds)
                    if matcher is not None else range(len(lines))
                )
                for index in indices:
                    if index % 256 == 0:
                        await asyncio.sleep(0)
                    if expression is not None and not expression.search(lines[index]):
                        continue
                    match_count += 1
                    start = max(0, index - context_lines)
                    end = min(len(lines), index + context_lines + 1)
                    for ctx_index in range(start, end):
                        marker = ":" if ctx_index == index else "-"
                        if ctx_index not in emitted_context:
                            row = f"{display}:{ctx_index + 1}{marker}{lines[ctx_index]}"
                            output_chars += len(row) + bool(rows)
                            rows.append(row)
                            if output_chars > output_limit:
                                return ToolResult(
                                    truncate_text("\n".join(rows), output_limit),
                                    metadata={"matches": match_count, "files_scanned": scanned, "truncated": True},
                                )
                            emitted_context.add(ctx_index)
                    if match_count >= max_results:
                        rows.append("… result limit reached …")
                        return ToolResult(truncate_text("\n".join(rows), output_limit), metadata={"matches": match_count, "files_scanned": scanned, "truncated": True})
        return ToolResult("\n".join(rows) or "No matches", metadata={"matches": match_count, "files_scanned": scanned})


def _walk_files(root: Path, context: ToolContext) -> Iterator[Path]:
    ignored = set(context.config.context.ignored_dirs)
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(item for item in dirs if item not in ignored)
        for name in sorted(files):
            yield Path(current) / name
