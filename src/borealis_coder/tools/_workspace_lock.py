"""Workspace-scoped serialization for built-in file mutations."""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Iterator
from pathlib import Path

from ..util import ensure_private_directory, ensure_private_file

_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[Path, threading.Lock] = {}


@contextlib.contextmanager
def workspace_transaction(workspace: Path) -> Iterator[None]:
    """Serialize one file transaction across threads and processes."""

    workspace = workspace.resolve()
    with _LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(workspace, threading.Lock())
    with thread_lock:
        lock_path = workspace / ".borealis" / "checkpoints" / ".workspace-mutations.lock"
        ensure_private_directory(lock_path.parent)
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            ensure_private_file(lock_path)
            with _advisory_file_lock(descriptor):
                yield
        finally:
            os.close(descriptor)


@contextlib.contextmanager
def _advisory_file_lock(descriptor: int) -> Iterator[None]:
    if os.name == "nt":
        import msvcrt

        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
