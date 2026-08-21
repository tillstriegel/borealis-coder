"""File-level checkpoints created before deterministic mutations."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ToolError
from ..util import (
    atomic_write_bytes,
    atomic_write_text,
    ensure_private_directory,
    new_id,
    sha256_bytes,
    utc_now,
)
from .paths import WorkspaceRoots


@dataclass(slots=True)
class Checkpoint:
    id: str
    created_at: str
    label: str
    files: list[dict[str, Any]]


class CheckpointManager:
    def __init__(self, roots: WorkspaceRoots, *, enabled: bool = True, max_bytes: int = 25_000_000) -> None:
        self.roots = roots
        self.enabled = enabled
        self.max_bytes = max_bytes
        self.directory = roots.primary / ".borealis" / "checkpoints"
        ensure_private_directory(self.directory)

    def create(self, paths: list[Path], *, label: str) -> Checkpoint | None:
        if not self.enabled:
            return None
        unique = sorted({path.resolve(strict=False) for path in paths}, key=str)
        checkpoint_id = new_id("cp")
        target = self.directory / checkpoint_id
        files_dir = target / "files"
        files: list[dict[str, Any]] = []
        total = 0
        for path in unique:
            resolved_path = self.roots.resolve(path)
            resolved = resolved_path.path
            display = self.roots.display(resolved)
            root_index = self.roots.roots.index(resolved_path.root)
            relative_path = resolved.relative_to(resolved_path.root).as_posix()
            entry: dict[str, Any] = {
                "path": display,
                "root": root_index,
                "relative_path": relative_path,
                "existed": resolved.exists(),
            }
            if resolved.exists():
                if not resolved.is_file():
                    raise ToolError(f"Checkpoint only supports files: {display}")
                data = resolved.read_bytes()
                total += len(data)
                if total > self.max_bytes:
                    raise ToolError(f"Checkpoint exceeds configured {self.max_bytes} byte limit")
                reference = f"{root_index}:{relative_path}"
                blob_name = base64.urlsafe_b64encode(reference.encode()).decode().rstrip("=") + ".bin"
                blob = files_dir / blob_name
                blob.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_bytes(blob, data)
                entry.update({"blob": f"files/{blob_name}", "sha256": sha256_bytes(data), "mode": resolved.stat().st_mode})
            files.append(entry)
        checkpoint = Checkpoint(checkpoint_id, utc_now(), label, files)
        target.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target / "manifest.json", json.dumps({
            "id": checkpoint.id, "created_at": checkpoint.created_at,
            "label": checkpoint.label, "files": checkpoint.files,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return checkpoint

    def list(self) -> list[Checkpoint]:
        result: list[Checkpoint] = []
        for manifest in self.directory.glob("*/manifest.json"):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                result.append(Checkpoint(data["id"], data["created_at"], data.get("label", ""), data.get("files", [])))
            except (OSError, KeyError, json.JSONDecodeError):
                continue
        return sorted(result, key=lambda item: item.created_at, reverse=True)

    def restore(self, checkpoint_id: str) -> Checkpoint:
        target = (self.directory / checkpoint_id).resolve(strict=False)
        try:
            target.relative_to(self.directory.resolve())
        except ValueError as error:
            raise ToolError(f"Invalid checkpoint ID: {checkpoint_id}") from error
        manifest_path = target / "manifest.json"
        if not manifest_path.is_file():
            raise ToolError(f"Unknown checkpoint: {checkpoint_id}")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = Checkpoint(data["id"], data["created_at"], data.get("label", ""), data.get("files", []))
        validated: list[tuple[dict[str, Any], Path, bytes | None]] = []
        for entry in checkpoint.files:
            path = self._entry_path(entry)
            blob_data: bytes | None = None
            if entry.get("existed"):
                blob = (target / str(entry["blob"])).resolve(strict=False)
                try:
                    blob.relative_to(target)
                except ValueError as error:
                    raise ToolError(f"Checkpoint blob escapes its manifest: {entry['blob']}") from error
                if not blob.is_file():
                    raise ToolError(f"Checkpoint blob is missing: {entry['blob']}")
                blob_data = blob.read_bytes()
                if sha256_bytes(blob_data) != entry.get("sha256"):
                    raise ToolError(f"Checkpoint blob checksum mismatch: {entry['blob']}")
            elif path.exists() and not (path.is_file() or path.is_symlink()):
                raise ToolError(
                    f"Refusing to remove non-file path while restoring checkpoint: "
                    f"{self.roots.display(path)}"
                )
            validated.append((entry, path, blob_data))

        for entry, path, blob_data in validated:
            if entry.get("existed"):
                assert blob_data is not None
                atomic_write_bytes(path, blob_data, mode=entry.get("mode"))
            elif path.exists():
                if path.is_file() or path.is_symlink():
                    path.unlink()
        return checkpoint

    def _entry_path(self, entry: dict[str, Any]) -> Path:
        if "root" not in entry or "relative_path" not in entry:
            return self.roots.resolve(str(entry["path"])).path
        root_index = entry["root"]
        if not isinstance(root_index, int) or not 0 <= root_index < len(self.roots.roots):
            raise ToolError(f"Checkpoint has invalid root index: {root_index!r}")
        root = self.roots.roots[root_index]
        candidate = (root / str(entry["relative_path"])).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ToolError(
                f"Checkpoint path escapes configured root {root_index}: "
                f"{entry['relative_path']!r}"
            ) from error
        return candidate
