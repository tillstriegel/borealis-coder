"""Small dependency-free utilities."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar, cast

T = TypeVar("T")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set | frozenset):
        return sorted(value)
    if hasattr(value, "value"):
        return cast(Any, value).value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def json_dumps(value: Any, *, pretty: bool = False) -> str:
    kwargs: dict[str, Any] = {
        "ensure_ascii": False,
        "default": json_default,
        "sort_keys": pretty,
    }
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    return json.dumps(value, **kwargs)


def json_loads(value: str | bytes) -> Any:
    return json.loads(value)


def atomic_write_bytes(path: Path, data: bytes, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temp, mode)
        elif path.exists():
            os.chmod(temp, path.stat().st_mode)
        os.replace(temp, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except (OSError, AttributeError):
            pass
    finally:
        temp.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str, *, mode: int | None = None) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), mode=mode)


def ensure_private_directory(path: Path) -> None:
    """Create a state directory and restrict it to the current user on POSIX."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        os.chmod(path, 0o700)


def ensure_private_file(path: Path) -> None:
    """Restrict an existing state file to the current user on POSIX."""
    if os.name == "posix" and path.exists():
        os.chmod(path, 0o600)


def truncate_text(text: str, limit: int, *, marker: str = "\n… output truncated …\n") -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    if limit < len(marker) + 20:
        return text[:limit]
    head = (limit - len(marker)) * 2 // 3
    tail = limit - len(marker) - head
    return text[:head] + marker + text[-tail:]


def estimate_tokens(text: str) -> int:
    """A conservative tokenizer-free estimate suitable for budget gates."""
    if not text:
        return 0
    ascii_chars = sum(ord(char) < 128 for char in text)
    ratio = 3.6 if ascii_chars / len(text) > 0.85 else 2.6
    return max(1, int(len(text) / ratio) + text.count("\n") // 8)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def coerce_scalar(value: str) -> Any:
    lower = value.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def set_nested(target: dict[str, Any], path: Iterable[str], value: Any) -> None:
    parts = list(path)
    if not parts:
        return
    cursor = target
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def slugify(value: str, *, max_length: int = 64) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-._")
    return (value or "item")[:max_length]


async def aiter_queue(queue: asyncio.Queue[T | None]) -> AsyncIterator[T]:
    while True:
        item = await queue.get()
        if item is None:
            return
        yield item


async def cancel_and_wait(task: asyncio.Task[Any] | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
