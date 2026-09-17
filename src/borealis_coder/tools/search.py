"""Dependency-free text search with regex, glob, and context support."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import fnmatch
import hashlib
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
from ..util import finish_on_cancellation, json_dumps, read_bytes_up_to
from .base import Tool, ToolContext, bound_tool_output, nullable, object_schema


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
    nullable_defaults = ("cursor",)
    name = "grep"
    description = "Search text files using literal or regular-expression matching."
    effect = Effect.READ
    concurrent = True
    parameters = object_schema({
        "cursor": nullable("string", maxLength=512),
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
        files = list(files)

        def in_scope(path: Path) -> bool:
            rel = path.relative_to(resolved.path).as_posix() if is_directory else path.name
            return not file_glob or fnmatch.fnmatch(rel, file_glob) or fnmatch.fnmatch(path.name, file_glob)

        excluded = sum(not in_scope(path) for path in files)
        files = [path for path in files if in_scope(path)]
        def snapshot(current_files: list[Path]) -> str:
            entries = []
            for path in current_files:
                try:
                    stat = path.stat()
                    entries.append((str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino))
                except OSError:
                    entries.append((str(path), "unreadable"))
            scope = {k: v for k, v in arguments.items() if k != "cursor"}
            return hashlib.sha256(json_dumps([scope, context.config.context.ignored_dirs, context.config.context.max_file_bytes, entries]).encode()).hexdigest()

        identity = snapshot(files)
        offset = 0
        if arguments.get("cursor"):
            try:
                cursor = json.loads(base64.urlsafe_b64decode(arguments["cursor"]))
                offset = int(cursor["offset"])
                if cursor["snapshot"] != identity or offset < 0:
                    raise ValueError()
            except (ValueError, KeyError, TypeError):
                return ToolResult("Search continuation is stale or invalid; restart the search.", is_error=True)
        rows: list[str] = []
        output_chars = 0
        output_limit = context.config.context.tool_output_chars
        file_limit = context.config.context.max_file_bytes
        match_count = 0
        seen = 0
        scanned = 0
        skipped: dict[str, int] = {"glob_excluded": excluded} if excluded else {}
        truncated = False
        next_cursor = None
        identities: dict[str, str] = {}
        async with _regex_matcher(
            pattern, flags, enabled=bool(arguments["regex"]), max_matches=file_limit,
        ) as matcher:
            for path in files:
                await asyncio.sleep(0)
                try:
                    candidate = context.roots.resolve(path, must_exist=True, kind="file")
                    data = read_bytes_up_to(candidate.path, file_limit + 1)
                except (OSError, PathViolation):
                    skipped["unreadable_or_unauthorized"] = skipped.get("unreadable_or_unauthorized", 0) + 1
                    continue
                reason = "size" if len(data) > file_limit else "unsupported_content" if b"\x00" in data else ""
                text = ""
                try:
                    text = data.decode("utf-8") if not reason else ""
                except UnicodeDecodeError:
                    reason = "unsupported_content"
                if reason:
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue
                scanned += 1
                display = candidate.display
                lines = text.splitlines()
                indices = (
                    await matcher.search(lines, len(lines), context.config.context.regex_timeout_seconds)
                    if matcher is not None else range(len(lines))
                )
                emitted_context: set[int] = set()
                for index in indices:
                    if index % 256 == 0:
                        await asyncio.sleep(0)
                    if expression is not None and not expression.search(lines[index]):
                        continue
                    seen += 1
                    if seen <= offset:
                        continue
                    if match_count >= max_results or (rows and output_chars >= max(1, output_limit - 2000)):
                        truncated = True
                        break
                    match_count += 1
                    identities[display] = hashlib.sha256(data).hexdigest()
                    for ctx_index in range(max(0, index - context_lines), min(len(lines), index + context_lines + 1)):
                        if ctx_index in emitted_context:
                            continue
                        emitted_context.add(ctx_index)
                        marker = ":" if ctx_index == index else "-"
                        row = f"{display}:{ctx_index + 1}{marker}{lines[ctx_index]}"
                        rows.append(row)
                        output_chars += len(row) + 1
                if truncated:
                    break
        if snapshot([path for path in _walk_files(resolved.path, context) if in_scope(path)] if is_directory else files) != identity:
            return ToolResult("Search scope changed during collection; restart the search.", is_error=True)
        if truncated:
            next_cursor = base64.urlsafe_b64encode(json_dumps({"snapshot": identity, "offset": offset + match_count}).encode()).decode()
        incomplete_skips = sum(v for k, v in skipped.items() if k != "glob_excluded")
        metadata = {"matches": match_count, "files_scanned": scanned, "skipped": skipped,
                    "truncated": truncated, "complete": not truncated and not incomplete_skips,
                    "scope": arguments["path"], "glob": file_glob,
                    "excluded_directories": context.config.context.ignored_dirs,
                    "next_cursor": next_cursor, "snapshot": identity, "content_sha256": identities}
        negative = "No matches in the completed search scope" if metadata["complete"] else "No matches in the portion examined"
        body = ("\n".join(rows) or negative) + ("\nContent SHA-256: " + json_dumps(identities) if identities else "")
        coverage = {k: v for k, v in metadata.items() if k != "content_sha256"}
        preview_truncated = len(body) + len(json_dumps(coverage)) + 64 > output_limit
        metadata["preview_truncated"] = coverage["preview_truncated"] = preview_truncated
        return bound_tool_output(ToolResult("Search coverage: " + json_dumps(coverage) + "\n" + body,
                                            metadata=metadata), context, output_limit)


def _walk_files(root: Path, context: ToolContext) -> Iterator[Path]:
    ignored = set(context.config.context.ignored_dirs)
    def on_error(error: OSError) -> None:
        raise ToolError("Search could not enumerate part of the requested scope; coverage is incomplete") from error

    for current, dirs, files in os.walk(root, onerror=on_error):
        dirs[:] = sorted(item for item in dirs if item not in ignored)
        for name in sorted(files):
            yield Path(current) / name
