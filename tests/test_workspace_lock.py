from __future__ import annotations

import builtins
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from borealis_coder.tools import _workspace_lock


class WorkspaceLockTests(unittest.TestCase):
    def test_windows_lock_uses_stable_byte_and_never_imports_fcntl(self):
        calls: list[tuple[int, int, int, int]] = []
        fake_msvcrt = SimpleNamespace(LK_LOCK=1, LK_UNLCK=2)

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
                _workspace_lock.workspace_transaction(root),
            ):
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

    def test_posix_lock_uses_flock_and_never_imports_msvcrt(self):
        calls: list[tuple[int, int]] = []
        fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_UN=2)
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
                _workspace_lock.workspace_transaction(root),
            ):
                pass

        self.assertEqual([mode for _, mode in calls], [1, 2])
        with self.assertRaises(OSError):
            os.fstat(calls[0][0])


if __name__ == "__main__":
    unittest.main()
