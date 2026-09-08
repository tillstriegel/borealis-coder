from __future__ import annotations

import asyncio
import builtins
import contextlib
import errno
import io
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from borealis_coder import cli
from borealis_coder.interactive import InteractiveCLI
from borealis_coder.tools import _workspace_lock
from borealis_coder.tools.filesystem import WriteFileTool
from tests.helpers import make_context


class WorkspaceLockTests(unittest.IsolatedAsyncioTestCase):
    async def test_rollback_commands_wait_and_cancel_before_restoring(self):
        for command in ("cli", "interactive"):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                context = make_context(root)
                target = root / "file.txt"
                target.write_text("before")
                checkpoint = context.checkpoints.create([target], label="rollback test")
                assert checkpoint is not None
                target.write_text("after")
                args = cli.build_parser().parse_args([
                    "rollback", checkpoint.id, "--workspace", str(root),
                ])
                shell = InteractiveCLI(
                    workspace=root, config=context.config, approval_callback=None,
                )

                rollback = (
                    partial(cli._main, args) if command == "cli"
                    else partial(shell._command, f"/rollback {checkpoint.id}")
                )

                with (
                    patch.object(cli, "load_config", return_value=context.config),
                    patch.object(shell, "_require_runner", return_value=SimpleNamespace(tool_context=context)),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    async with _workspace_lock.workspace_transaction(root):
                        task = asyncio.create_task(rollback())
                        try:
                            await asyncio.sleep(0.02)
                            self.assertFalse(task.done(), "Rollback bypassed the mutation lock")
                            self.assertEqual(target.read_text(), "after")
                            task.cancel()
                            with self.assertRaises(asyncio.CancelledError):
                                await task
                            self.assertEqual(target.read_text(), "after")
                        finally:
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                    await asyncio.wait_for(rollback(), timeout=1)
                    self.assertEqual(target.read_text(), "before")

    async def _write(self, root):
        return await WriteFileTool().execute(
            {"path": "result.txt", "content": "written", "expected_sha256": None},
            make_context(root),
        )

    async def _cancel_waiting_write(self, root):
        task = asyncio.create_task(self._write(root))
        try:
            await asyncio.sleep(0.02)
            self.assertFalse(task.done(), "Write did not wait for the held lock")
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse((root / "result.txt").exists())
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_thread_lock_wait_is_cancellable_before_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            held = threading.Event()
            release = threading.Event()

            async def hold():
                async with _workspace_lock.workspace_transaction(root):
                    held.set()
                    await asyncio.to_thread(release.wait, 3)

            thread = threading.Thread(target=lambda: asyncio.run(hold()), daemon=True)
            thread.start()
            try:
                self.assertTrue(await asyncio.to_thread(held.wait, 2))
                await self._cancel_waiting_write(root)
                self.assertFalse(release.is_set())
            finally:
                release.set()
                await asyncio.to_thread(thread.join, 3)
            self.assertFalse(thread.is_alive())
            result = await self._write(root)
            self.assertFalse(result.is_error, result.output)
            self.assertEqual((root / "result.txt").read_text(), "written")

    async def test_process_lock_wait_is_cancellable_before_mutation(self):
        script = """
import os, sys
from pathlib import Path
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
try:
    if os.name == "nt":
        import msvcrt
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\\0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
    else:
        import fcntl
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    print("locked", flush=True)
    sys.stdin.readline()
finally:
    os.close(descriptor)
"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lock_path = root / ".borealis" / "checkpoints" / ".workspace-mutations.lock"
            with subprocess.Popen(
                [sys.executable, "-c", script, str(lock_path)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ) as process:
                assert process.stdout is not None
                failsafe = threading.Timer(3, process.terminate)
                failsafe.start()
                try:
                    self.assertEqual(await asyncio.to_thread(process.stdout.readline), b"locked\n")
                    await self._cancel_waiting_write(root)
                    self.assertIsNone(process.poll())
                finally:
                    failsafe.cancel()
                    try:
                        await asyncio.to_thread(process.communicate, b"\n", timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        await asyncio.to_thread(process.communicate)
            result = await self._write(root)
            self.assertFalse(result.is_error, result.output)
            self.assertEqual((root / "result.txt").read_text(), "written")

    async def test_file_lock_retries_contention_but_propagates_other_errors(self):
        calls = 0

        def acquire():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise OSError(errno.EACCES if calls == 1 else errno.EAGAIN, "locked")

        await _workspace_lock._wait_for_file_lock(acquire)
        self.assertEqual(calls, 3)

        def broken():
            raise OSError(errno.EBADF, "invalid descriptor")

        with self.assertRaisesRegex(OSError, "invalid descriptor"):
            await _workspace_lock._wait_for_file_lock(broken)

    async def test_windows_lock_uses_stable_byte_and_never_imports_fcntl(self):
        calls: list[tuple[int, int, int, int]] = []
        fake_msvcrt = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2)

        def locking(descriptor, mode, length):
            calls.append((descriptor, mode, length, os.lseek(descriptor, 0, os.SEEK_CUR)))

        fake_msvcrt.locking = locking
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "fcntl":
                self.fail("The Windows lock path imported fcntl")
            if name == "msvcrt":
                return fake_msvcrt
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with (
                patch.object(_workspace_lock.os, "name", "nt"),
                patch.object(builtins, "__import__", side_effect=guarded_import),
                self.assertRaisesRegex(RuntimeError, "injected"),
            ):
                async with _workspace_lock.workspace_transaction(root):
                    raise RuntimeError("injected")

            lock_path = (
                root
                / ".borealis"
                / "checkpoints"
                / ".workspace-mutations.lock"
            )
            self.assertEqual(lock_path.read_bytes(), b"\0")

        self.assertEqual([call[1:] for call in calls], [(1, 1, 0), (2, 1, 0)])
        with self.assertRaises(OSError):
            os.fstat(calls[0][0])

    async def test_posix_lock_uses_flock_and_never_imports_msvcrt(self):
        calls: list[tuple[int, int]] = []
        fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_UN=2, LOCK_NB=4)
        fake_fcntl.flock = lambda descriptor, mode: calls.append((descriptor, mode))
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "msvcrt":
                self.fail("The POSIX lock path imported msvcrt")
            if name == "fcntl":
                return fake_fcntl
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with (
                patch.object(_workspace_lock.os, "name", "posix"),
                patch.object(builtins, "__import__", side_effect=guarded_import),
            ):
                async with _workspace_lock.workspace_transaction(root):
                    pass

        self.assertEqual([mode for _, mode in calls], [5, 2])
        with self.assertRaises(OSError):
            os.fstat(calls[0][0])


if __name__ == "__main__":
    unittest.main()
