"""File-level checkpoints created before deterministic mutations."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..errors import ToolError
from ..util import (
    atomic_write_bytes,
    atomic_write_text,
    ensure_private_directory,
    new_id,
    read_bytes_up_to,
    sha256_bytes,
    utc_now,
)
from .paths import WorkspaceRoots

_ACTIVE_MARKER = ".active"


def _valid_mode(mode: object) -> bool:
    return mode is None or (
        isinstance(mode, int) and not isinstance(mode, bool) and 0 <= mode <= 0o177777
    )


@dataclass(slots=True)
class Checkpoint:
    id: str
    created_at: str
    label: str
    files: list[dict[str, Any]]
    creation_order: str = ""


def _read_manifest(manifest: Path) -> Checkpoint:
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ToolError(f"Cannot read checkpoint manifest: {manifest}") from error
    if (
        not isinstance(data, dict)
        or data.get("id") != manifest.parent.name
        or not isinstance(data.get("created_at"), str)
        or not isinstance(data.get("label", ""), str)
        or not isinstance(data.get("files"), list)
    ):
        raise ToolError(f"Invalid checkpoint manifest: {manifest}")
    for entry in data["files"]:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or not entry["path"]
            or "\x00" in entry["path"]
            or not isinstance(entry.get("existed"), bool)
        ):
            raise ToolError(f"Invalid checkpoint file entry: {manifest}")
        if "root" in entry or "relative_path" in entry or "root_path" in entry:
            root_index = entry.get("root")
            relative_path = entry.get("relative_path")
            if (
                not isinstance(root_index, int)
                or isinstance(root_index, bool)
                or root_index < 0
                or not isinstance(relative_path, str)
                or not relative_path
                or "\x00" in relative_path
                or Path(relative_path).is_absolute()
                or ".." in Path(relative_path).parts
            ):
                raise ToolError(f"Invalid checkpoint root reference: {manifest}")
            if "root_path" in entry:
                root_path = entry["root_path"]
                if (
                    not isinstance(root_path, str)
                    or "\x00" in root_path
                    or not Path(root_path).is_absolute()
                ):
                    raise ToolError(f"Invalid checkpoint root path: {manifest}")
    return Checkpoint(
        data["id"], data["created_at"], data.get("label", ""),
        data["files"], str(data.get("creation_order", "")),
    )


class CheckpointManager:
    def __init__(
        self,
        roots: WorkspaceRoots,
        *,
        enabled: bool = True,
        max_bytes: int = 25_000_000,
        retention_max_count: int = 50,
        retention_max_bytes: int = 250_000_000,
        retention_max_age_seconds: int = 0,
        create_directory: bool = True,
    ) -> None:
        self.roots = roots
        self.enabled = enabled
        self.max_bytes = max_bytes
        self.retention_max_count = retention_max_count
        self.retention_max_bytes = retention_max_bytes
        self.retention_max_age_seconds = retention_max_age_seconds
        self.directory = roots.primary / ".borealis" / "checkpoints"
        if create_directory:
            ensure_private_directory(self.directory)

    def create(
        self, paths: list[Path], *, label: str, active: bool = False
    ) -> Checkpoint | None:
        if not self.enabled:
            return None
        unique = sorted({path.resolve(strict=False) for path in paths}, key=str)
        checkpoint_id = new_id("cp")
        target = self.directory / checkpoint_id
        files_dir = target / "files"
        files: list[dict[str, Any]] = []
        total = 0
        target.mkdir(parents=True, exist_ok=False)
        try:
            for path in unique:
                resolved_path = self.roots.resolve(path)
                resolved = resolved_path.path
                display = self.roots.display(resolved)
                entry: dict[str, Any] = {
                    "path": display,
                    "existed": resolved.exists(),
                }
                if resolved_path.root in self.roots.roots:
                    root_index = self.roots.roots.index(resolved_path.root)
                    relative_path = resolved.relative_to(resolved_path.root).as_posix()
                    entry.update({"root": root_index, "relative_path": relative_path})
                    if root_index != 0:
                        entry["root_path"] = str(resolved_path.root)
                    reference = f"{root_index}:{relative_path}"
                else:
                    reference = f"outside:{resolved}"
                if resolved.exists():
                    if not resolved.is_file():
                        raise ToolError(f"Checkpoint only supports files: {display}")
                    data = read_bytes_up_to(resolved, max(0, self.max_bytes - total) + 1)
                    total += len(data)
                    if total > self.max_bytes:
                        raise ToolError(f"Checkpoint exceeds configured {self.max_bytes} byte limit")
                    blob_name = f"{sha256_bytes(reference.encode())}.bin"
                    blob = files_dir / blob_name
                    blob.parent.mkdir(parents=True, exist_ok=True)
                    atomic_write_bytes(blob, data)
                    entry.update({"blob": f"files/{blob_name}", "sha256": sha256_bytes(data), "mode": resolved.stat().st_mode})
                files.append(entry)
            checkpoint = Checkpoint(
                checkpoint_id,
                utc_now(),
                label,
                files,
                f"{time.time_ns():020d}:{checkpoint_id}",
            )
            if active:
                atomic_write_text(target / _ACTIVE_MARKER, "active\n")
            atomic_write_text(target / "manifest.json", json.dumps({
                "id": checkpoint.id, "created_at": checkpoint.created_at,
                "label": checkpoint.label, "files": checkpoint.files,
                "creation_order": checkpoint.creation_order,
            }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        except BaseException:
            if target.exists():
                shutil.rmtree(target)
            raise
        self.prune(preserve_id=checkpoint.id)
        return checkpoint

    def release(self, checkpoint_id: str) -> None:
        """Make an active checkpoint eligible for retention pruning."""

        target = self._checkpoint_directory(checkpoint_id)
        manifest = target / "manifest.json"
        if not manifest.is_file():
            raise ToolError(f"Unknown checkpoint: {checkpoint_id}")
        (target / _ACTIVE_MARKER).unlink(missing_ok=True)
        self.prune()

    def list(self) -> list[Checkpoint]:
        result: list[Checkpoint] = []
        if not self.directory.is_dir():
            return result
        for manifest in self.directory.glob("*/manifest.json"):
            try:
                result.append(_read_manifest(manifest))
            except ToolError:
                continue
        return sorted(
            result,
            key=lambda item: (item.created_at, item.creation_order, item.id),
            reverse=True,
        )

    def restore(self, checkpoint_id: str) -> Checkpoint:
        target = self._checkpoint_directory(checkpoint_id)
        manifest_path = target / "manifest.json"
        if not manifest_path.is_file():
            raise ToolError(f"Unknown checkpoint: {checkpoint_id}")
        checkpoint = _read_manifest(manifest_path)
        validated: list[tuple[dict[str, Any], Path, bytes | None]] = []
        for entry in checkpoint.files:
            path = self._entry_path(entry)
            if path.exists() and not (path.is_file() or path.is_symlink()):
                raise ToolError(
                    f"Refusing to replace non-file path while restoring checkpoint: "
                    f"{self.roots.display(path)}"
                )
            blob_data: bytes | None = None
            if entry.get("existed"):
                if not _valid_mode(entry.get("mode")):
                    raise ToolError(f"Checkpoint has invalid file mode: {entry.get('mode')!r}")
                for parent in path.parents:
                    if parent.exists():
                        if not parent.is_dir():
                            raise ToolError(
                                f"Cannot restore file because its parent path is not a directory: {parent}"
                            )
                        break
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
            validated.append((entry, path, blob_data))

        for entry, path, blob_data in validated:
            if entry.get("existed"):
                assert blob_data is not None
                atomic_write_bytes(path, blob_data, mode=entry.get("mode"))
            elif path.exists():
                if path.is_file() or path.is_symlink():
                    path.unlink()
        return checkpoint

    def prune(
        self, *, dry_run: bool = False, preserve_id: str | None = None
    ) -> dict[str, Any]:
        """Prune complete checkpoints oldest first while preserving the newest."""

        records = self._complete_records()
        if not records:
            return {
                "complete_checkpoints": 0,
                "bytes_before": 0,
                "pruned_count": 0,
                "pruned_bytes": 0,
                "checkpoint_ids": [],
                "dry_run": dry_run,
            }
        records.sort(
            key=lambda item: (
                item[0].created_at,
                item[0].creation_order,
                item[0].id,
            ),
            reverse=True,
        )
        if preserve_id is not None:
            preserved = next(
                (
                    index
                    for index, (checkpoint, _, _, _) in enumerate(records)
                    if checkpoint.id == preserve_id
                ),
                None,
            )
            if preserved is not None:
                records.insert(0, records.pop(preserved))
        total_bytes = sum(size for _, _, size, _ in records)
        remaining_count = len(records)
        remaining_bytes = total_bytes
        now = datetime.now(UTC)
        selected: list[tuple[Checkpoint, Path, int, datetime]] = []
        for checkpoint, directory, size, created_at in reversed(records[1:]):
            too_old = bool(
                self.retention_max_age_seconds
                and (now - created_at).total_seconds() > self.retention_max_age_seconds
            )
            over_count = remaining_count > self.retention_max_count
            over_bytes = remaining_bytes > self.retention_max_bytes
            if not (too_old or over_count or over_bytes):
                continue
            selected.append((checkpoint, directory, size, created_at))
            remaining_count -= 1
            remaining_bytes -= size
        if not dry_run:
            for _, directory, _, _ in selected:
                shutil.rmtree(directory)
        return {
            "complete_checkpoints": len(records),
            "bytes_before": total_bytes,
            "pruned_count": len(selected),
            "pruned_bytes": sum(size for _, _, size, _ in selected),
            "checkpoint_ids": [checkpoint.id for checkpoint, _, _, _ in selected],
            "dry_run": dry_run,
        }

    def _complete_records(self) -> list[tuple[Checkpoint, Path, int, datetime]]:
        records: list[tuple[Checkpoint, Path, int, datetime]] = []
        if not self.directory.is_dir():
            return records
        for directory in self.directory.iterdir():
            if not directory.is_dir() or directory.is_symlink():
                continue
            if (directory / _ACTIVE_MARKER).exists():
                continue
            manifest = directory / "manifest.json"
            try:
                checkpoint = _read_manifest(manifest)
                created_at = datetime.fromisoformat(checkpoint.created_at.replace("Z", "+00:00"))
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=UTC)
                size = self._complete_directory_size(directory, checkpoint)
                if size is None:
                    continue
            except (OSError, ToolError, ValueError):
                continue
            records.append((checkpoint, directory, size, created_at))
        return records

    def _checkpoint_directory(self, checkpoint_id: str) -> Path:
        target = (self.directory / checkpoint_id).resolve(strict=False)
        try:
            target.relative_to(self.directory.resolve())
        except ValueError as error:
            raise ToolError(f"Invalid checkpoint ID: {checkpoint_id}") from error
        return target

    @staticmethod
    def _complete_directory_size(directory: Path, checkpoint: Checkpoint) -> int | None:
        for entry in checkpoint.files:
            if not isinstance(entry, dict):
                return None
            if entry.get("existed"):
                if not _valid_mode(entry.get("mode")):
                    return None
                blob_value = entry.get("blob")
                if not isinstance(blob_value, str):
                    return None
                blob = (directory / blob_value).resolve(strict=False)
                try:
                    blob.relative_to(directory.resolve())
                except ValueError:
                    return None
                if not blob.is_file() or blob.is_symlink():
                    return None
                try:
                    with blob.open("rb") as handle:
                        digest = hashlib.file_digest(handle, "sha256").hexdigest()
                    if digest != entry.get("sha256"):
                        return None
                except OSError:
                    return None
        total = 0
        for path in directory.rglob("*"):
            if path.is_symlink() or (not path.is_file() and not path.is_dir()):
                return None
            if path.is_file():
                total += path.stat().st_size
        return total

    def _entry_path(self, entry: dict[str, Any]) -> Path:
        if "root" not in entry or "relative_path" not in entry:
            stored_path = Path(str(entry["path"]))
            original = stored_path if stored_path.is_absolute() else self.roots.primary / stored_path
            candidate = self.roots.resolve(stored_path).path
        else:
            root_index = entry["root"]
            if "root_path" in entry:
                root_path = entry["root_path"]
                if not isinstance(root_path, str) or not Path(root_path).is_absolute():
                    raise ToolError(f"Checkpoint has invalid root path: {root_path!r}")
                root = Path(root_path)
                if root not in self.roots.roots:
                    raise ToolError(f"Checkpoint workspace root is not configured: {root_path}")
            else:
                if not isinstance(root_index, int) or not 0 <= root_index < len(self.roots.roots):
                    raise ToolError(f"Checkpoint has invalid root index: {root_index!r}")
                root = self.roots.roots[root_index]
            original = root / str(entry["relative_path"])
            candidate = original.resolve(strict=False)
            try:
                candidate.relative_to(root)
            except ValueError as error:
                raise ToolError(
                    f"Checkpoint path escapes configured root {root_index}: "
                    f"{entry['relative_path']!r}"
                ) from error
        if candidate != original:
            raise ToolError(
                f"Checkpoint path now resolves to a different location: {original}"
            )
        return candidate
