"""Workspace-scoped serialization for built-in file mutations."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import threading
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from ..util import ensure_private_directory, ensure_private_file

_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[Path, threading.Lock] = {}
_LOCK_POLL_SECONDS = 0.05


@contextlib.asynccontextmanager
async def workspace_transaction(workspace: Path) -> AsyncIterator[None]:
    """Wait without blocking, then serialize one file transaction."""

    workspace = workspace.resolve()
    with _LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(workspace, threading.Lock())
    while not thread_lock.acquire(blocking=False):
        await asyncio.sleep(_LOCK_POLL_SECONDS)
    try:
        lock_path = workspace / ".borealis" / "checkpoints" / ".workspace-mutations.lock"
        ensure_private_directory(lock_path.parent)
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            ensure_private_file(lock_path)
            async with _advisory_file_lock(descriptor):
                yield
        finally:
            os.close(descriptor)
    finally:
        thread_lock.release()


@contextlib.asynccontextmanager
async def _advisory_file_lock(descriptor: int) -> AsyncIterator[None]:
    if os.name == "nt":
        import msvcrt

        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        await _wait_for_file_lock(lambda: msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1))
        try:
            yield
        finally:
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    await _wait_for_file_lock(lambda: fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB))
    try:
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)


async def _wait_for_file_lock(acquire: Callable[[], None]) -> None:
    while True:
        try:
            acquire()
            return
        except OSError as error:
            if error.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
        await asyncio.sleep(_LOCK_POLL_SECONDS)
