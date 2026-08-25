"""Deterministic patch application for unified diffs and apply_patch envelopes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import PatchError
from ..models import Effect, ToolResult
from ..util import atomic_write_text, sha256_text
from .base import MutationScope, Tool, ToolContext, object_schema


@dataclass(slots=True)
class HunkLine:
    kind: str
    text: str


@dataclass(slots=True)
class Hunk:
    old_start: int | None
    old_count: int
    new_start: int
    new_count: int
    lines: list[HunkLine] = field(default_factory=list)


@dataclass(slots=True)
class FilePatch:
    old_path: str | None
    new_path: str | None
    hunks: list[Hunk] = field(default_factory=list)
    add_content: str | None = None
    delete: bool = False


class ApplyPatchTool(Tool):
    name = "apply_patch"
    description = (
        "Apply an exact unified diff or *** Begin Patch envelope atomically. "
        "Envelope updates accept numbered headers or bare @@ headers, for example: "
        "*** Begin Patch\\n*** Update File: a.txt\\n@@\\n-old\\n+new\\n*** End Patch"
    )
    effect = Effect.WRITE
    mutation_scope = MutationScope.TRACKED
    default_risk = "medium"
    parameters = object_schema({
        "patch": {"type": "string", "minLength": 1, "maxLength": 5_000_000}
    })

    def risk(self, arguments: dict[str, Any]) -> str:
        patch = str(arguments.get("patch") or "")
        return "high" if "*** Delete File:" in patch or "+++ /dev/null" in patch else "medium"

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        patches = parse_patch(str(arguments["patch"]))
        if not patches:
            raise PatchError("Patch contains no file changes")
        targets: list[Path] = []
        planned: list[tuple[FilePatch, Path, str, str | None]] = []
        seen: set[Path] = set()
        for item in patches:
            path = item.new_path or item.old_path
            if not path:
                raise PatchError("File patch has no path")
            target = context.roots.resolve(_clean_path(path)).path
            context.roots.assert_writable(target, context.config.safety.protected_paths)
            if target in seen:
                raise PatchError(f"Patch contains duplicate target: {context.roots.display(target)}")
            seen.add(target)
            targets.append(target)
            display = context.roots.display(target)
            if item.add_content is not None:
                if target.exists():
                    raise PatchError(f"Add-file target already exists: {display}")
                if len(item.add_content.encode("utf-8")) > context.config.context.max_file_bytes:
                    raise PatchError(
                        f"Added file exceeds {context.config.context.max_file_bytes} byte limit: {display}"
                    )
                planned.append((item, target, "add", item.add_content))
                continue
            if not target.is_file():
                operation = "delete" if item.delete else "update"
                raise PatchError(f"Cannot {operation} missing file: {display}")
            original = target.read_text(encoding="utf-8")
            if len(original.encode("utf-8")) > context.config.context.max_file_bytes:
                raise PatchError(
                    f"File exceeds {context.config.context.max_file_bytes} byte edit limit: {display}"
                )
            if item.delete and not item.hunks:
                # The apply_patch envelope's Delete File directive names the
                # complete target; unified deletions still validate their hunks.
                planned.append((item, target, "delete", None))
                continue
            updated = apply_hunks(original, item.hunks, display)
            if item.delete:
                if updated != "":
                    raise PatchError(f"Delete patch for {display} did not remove the entire file")
                planned.append((item, target, "delete", None))
            else:
                if len(updated.encode("utf-8")) > context.config.context.max_file_bytes:
                    raise PatchError(
                        f"Updated file exceeds {context.config.context.max_file_bytes} byte limit: {display}"
                    )
                planned.append((item, target, "update", updated))

        # All hunks are validated before any file is changed. The checkpoint then
        # provides rollback if an unexpected filesystem error occurs during commit.
        checkpoint = context.checkpoints.create(
            targets,
            label=f"apply_patch ({len(patches)} files)",
            active=True,
        )
        outcomes: list[str] = []
        try:
            for _, target, operation, content in planned:
                display = context.roots.display(target)
                if operation == "delete":
                    target.unlink()
                    outcomes.append(f"deleted {display}")
                else:
                    assert content is not None
                    atomic_write_text(target, content)
                    outcomes.append(f"{'added' if operation == 'add' else 'updated'} {display} sha256={sha256_text(content)}")
                context.changed_files.add(display)
                context.changed_roots.add(context.roots.resolve(target).root)
        except Exception:
            if checkpoint is not None:
                context.checkpoints.restore(checkpoint.id)
                context.checkpoints.release(checkpoint.id)
            raise
        if checkpoint is not None:
            context.checkpoints.release(checkpoint.id)
        return ToolResult("\n".join(outcomes), metadata={
            "files": [context.roots.display(item) for item in targets],
            "checkpoint_id": checkpoint.id if checkpoint else None,
        })


def parse_patch(text: str) -> list[FilePatch]:
    normalized = text.replace("\r\n", "\n")
    if normalized.lstrip().startswith("*** Begin Patch"):
        return _parse_apply_patch(normalized)
    return _parse_unified(normalized)


def _parse_apply_patch(text: str) -> list[FilePatch]:
    lines = text.splitlines(keepends=True)
    result: list[FilePatch] = []
    index = 0
    while index < len(lines) and not lines[index].startswith("*** Begin Patch"):
        index += 1
    index += 1
    while index < len(lines):
        line = lines[index].rstrip("\n")
        if line == "*** End Patch":
            break
        if line.startswith("*** Add File: "):
            path = line[len("*** Add File: ") :].strip()
            index += 1
            content: list[str] = []
            while index < len(lines) and not lines[index].startswith("*** "):
                value = lines[index]
                if not value.startswith("+"):
                    raise PatchError(f"Add-file content must begin with '+': {value.rstrip()}")
                content.append(value[1:])
                index += 1
            result.append(FilePatch(None, path, add_content="".join(content)))
            continue
        if line.startswith("*** Delete File: "):
            path = line[len("*** Delete File: ") :].strip()
            result.append(FilePatch(path, None, delete=True))
            index += 1
            continue
        if line.startswith("*** Update File: "):
            path = line[len("*** Update File: ") :].strip()
            index += 1
            body: list[str] = []
            while index < len(lines) and not lines[index].startswith("*** "):
                body.append(lines[index])
                index += 1
            result.append(FilePatch(path, path, hunks=_parse_update_hunks(body, path)))
            continue
        if not line.strip():
            index += 1
            continue
        raise PatchError(f"Unexpected apply_patch directive: {line}")
    return result


def _parse_update_hunks(lines: list[str], path: str) -> list[Hunk]:
    """Parse apply_patch update hunks, including context-located bare headers."""

    numbered = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    hunks: list[Hunk] = []
    index = 0
    while index < len(lines):
        header = lines[index].rstrip("\n")
        match = numbered.match(lines[index])
        if match:
            hunk = Hunk(
                int(match.group(1)),
                int(match.group(2) or 1),
                int(match.group(3)),
                int(match.group(4) or 1),
            )
        elif header == "@@" or (header.startswith("@@ ") and header.endswith(" @@")):
            hunk = Hunk(None, 0, 0, 0)
        else:
            detail = header or "<blank line>"
            raise PatchError(
                f"Expected hunk header in {path}, got {detail!r}. "
                "Use '@@ -old,count +new,count @@' or bare '@@'."
            )
        index += 1
        while index < len(lines) and not lines[index].startswith("@@"):
            value = lines[index]
            if value.startswith("\\ No newline"):
                if not hunk.lines:
                    raise PatchError(f"No-newline marker has no preceding hunk line in {path}")
                hunk.lines[-1].text = hunk.lines[-1].text.removesuffix("\n")
                index += 1
                continue
            if not value or value[0] not in " +-":
                raise PatchError(f"Invalid hunk line in {path}: {value.rstrip()}")
            hunk.lines.append(HunkLine(value[0], value[1:]))
            index += 1
        if hunk.old_start is None:
            hunk.old_count = sum(line.kind in " -" for line in hunk.lines)
            hunk.new_count = sum(line.kind in " +" for line in hunk.lines)
            if not hunk.old_count:
                raise PatchError(
                    f"Bare '@@' hunk in {path} needs at least one context or removal line"
                )
        hunks.append(hunk)
    if not hunks:
        raise PatchError(
            f"Update for {path} contains no hunks. "
            "Use '@@ -old,count +new,count @@' or bare '@@'."
        )
    return hunks


def _parse_unified(text: str) -> list[FilePatch]:
    lines = text.splitlines(keepends=True)
    result: list[FilePatch] = []
    index = 0
    header = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    while index < len(lines):
        if not lines[index].startswith("--- "):
            index += 1
            continue
        old_path = lines[index][4:].strip().split("\t", 1)[0]
        index += 1
        if index >= len(lines) or not lines[index].startswith("+++ "):
            raise PatchError("Unified diff is missing +++ header")
        new_path = lines[index][4:].strip().split("\t", 1)[0]
        index += 1
        patch = FilePatch(
            None if old_path == "/dev/null" else old_path,
            None if new_path == "/dev/null" else new_path,
            delete=new_path == "/dev/null",
        )
        if old_path == "/dev/null":
            added: list[str] = []
            while index < len(lines) and not lines[index].startswith("--- "):
                match = header.match(lines[index])
                if match:
                    index += 1
                    continue
                value = lines[index]
                if value.startswith("+") and not value.startswith("+++"):
                    added.append(value[1:])
                elif value.startswith("\\ No newline"):
                    pass
                elif value.startswith((" ", "-")):
                    raise PatchError("New-file patch unexpectedly contains context or removals")
                index += 1
            patch.add_content = "".join(added)
            result.append(patch)
            continue
        while index < len(lines) and not lines[index].startswith("--- "):
            match = header.match(lines[index])
            if not match:
                if lines[index].strip():
                    raise PatchError(
                        "Expected numbered hunk header '@@ -old,count +new,count @@', "
                        f"got: {lines[index].rstrip()}. Bare '@@' is accepted only "
                        "inside a *** Begin Patch / *** Update File block."
                    )
                index += 1
                continue
            hunk = Hunk(
                int(match.group(1)), int(match.group(2) or 1),
                int(match.group(3)), int(match.group(4) or 1),
            )
            index += 1
            while index < len(lines) and not lines[index].startswith(("@@ ", "--- ")):
                value = lines[index]
                if value.startswith("\\ No newline"):
                    if not hunk.lines:
                        raise PatchError("No-newline marker has no preceding hunk line")
                    hunk.lines[-1].text = hunk.lines[-1].text.removesuffix("\n")
                    index += 1
                    continue
                if not value or value[0] not in " +-":
                    raise PatchError(f"Invalid hunk line: {value.rstrip()}")
                hunk.lines.append(HunkLine(value[0], value[1:]))
                index += 1
            patch.hunks.append(hunk)
        result.append(patch)
    return result


def apply_hunks(original: str, hunks: list[Hunk], display: str) -> str:
    source = original.splitlines(keepends=True)
    output: list[str] = []
    cursor = 0
    for hunk in hunks:
        if hunk.old_start is None:
            expected = [line.text for line in hunk.lines if line.kind in " -"]
            matches = [
                index
                for index in range(cursor, len(source) - len(expected) + 1)
                if source[index : index + len(expected)] == expected
            ]
            if not matches:
                raise PatchError(
                    f"Bare '@@' hunk context was not found in {display}; no changes made"
                )
            if len(matches) != 1:
                raise PatchError(
                    f"Bare '@@' hunk context is ambiguous in {display}: "
                    f"found {len(matches)} exact matches; add more context or use a numbered header"
                )
            start = matches[0]
        else:
            start = max(0, hunk.old_start - 1)
        if start < cursor or start > len(source):
            label = "located context" if hunk.old_start is None else f"old line {hunk.old_start}"
            raise PatchError(f"Overlapping or reordered hunk in {display}: {label}")
        output.extend(source[cursor:start])
        source_index = start
        old_seen = new_seen = 0
        for line in hunk.lines:
            if line.kind == " ":
                if source_index >= len(source) or source[source_index] != line.text:
                    actual = source[source_index] if source_index < len(source) else "<EOF>"
                    raise PatchError(f"Context mismatch in {display} at line {source_index + 1}: expected {line.text!r}, got {actual!r}")
                output.append(line.text)
                source_index += 1
                old_seen += 1
                new_seen += 1
            elif line.kind == "-":
                if source_index >= len(source) or source[source_index] != line.text:
                    actual = source[source_index] if source_index < len(source) else "<EOF>"
                    raise PatchError(f"Removal mismatch in {display} at line {source_index + 1}: expected {line.text!r}, got {actual!r}")
                source_index += 1
                old_seen += 1
            elif line.kind == "+":
                output.append(line.text)
                new_seen += 1
        if old_seen != hunk.old_count or new_seen != hunk.new_count:
            raise PatchError(f"Hunk count mismatch in {display}: header -{hunk.old_count}/+{hunk.new_count}, body -{old_seen}/+{new_seen}")
        cursor = source_index
    output.extend(source[cursor:])
    return "".join(output)


def _clean_path(path: str) -> str:
    path = path.strip()
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path
