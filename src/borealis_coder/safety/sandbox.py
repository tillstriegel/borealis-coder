"""Process execution drivers with bounded output, time, and resources."""

from __future__ import annotations

import asyncio
import codecs
import inspect
import os
import shutil
import signal
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from ..config import SafetyConfig, SandboxConfig
from ..errors import ToolError
from ..util import monotonic_ms
from .paths import WorkspaceRoots

# Receives ("stdout" | "stderr", decoded_chunk) as output arrives. May return an
# awaitable; exceptions raised by the consumer never abort the command itself.
OutputSink = Callable[[str, str], Awaitable[None] | None]
OUTPUT_TRUNCATION_MARKER = "\n… output truncated …\n"
_OUTPUT_OBSERVER_DRAIN_SECONDS = 1.0
_POSIX_PROCESS_SUPERVISOR = """
import os
import signal
import subprocess
import sys

status_fd = int(sys.argv[1])
shell = sys.argv[2] == "shell"
command = sys.argv[3] if shell else sys.argv[3:]
signal.signal(signal.SIGTERM, lambda *_: None)
try:
    completed = subprocess.run(command, shell=shell)
    status = completed.returncode
except OSError as error:
    print(error, file=sys.stderr)
    status = 127
os.write(status_fd, f"{status}\\n".encode())
while True:
    signal.pause()
"""


class _BoundedText:
    """Keep the same bounded head/tail view without retaining all process output."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._value = ""
        self._head = ""
        self._tail = ""
        self._truncated = False
        if limit >= len(OUTPUT_TRUNCATION_MARKER) + 20:
            available = limit - len(OUTPUT_TRUNCATION_MARKER)
            self._head_size = available * 2 // 3
            self._tail_size = available - self._head_size
        else:
            self._head_size = max(0, limit)
            self._tail_size = 0

    def append(self, text: str) -> None:
        if not text:
            return
        if self.limit <= 0:
            self._value += text
            return
        if self._truncated:
            if self._tail_size:
                self._tail = (self._tail + text)[-self._tail_size :]
            return
        combined = self._value + text
        if len(combined) <= self.limit:
            self._value = combined
            return
        self._truncated = True
        self._head = combined[: self._head_size]
        if self._tail_size:
            self._tail = combined[-self._tail_size :]
        self._value = ""

    def render(self) -> str:
        if not self._truncated:
            return self._value
        if not self._tail_size:
            return self._head
        return self._head + OUTPUT_TRUNCATION_MARKER + self._tail


@dataclass(slots=True)
class ProcessResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False
    stream_truncated: bool = False
    stream_complete: bool = True
    lifecycle_complete: bool = True

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def render(self) -> str:
        parts = [f"exit_code={self.exit_code}", f"duration_ms={self.duration_ms}"]
        if self.timed_out:
            parts.append("timed_out=true")
        if self.stdout:
            parts.append(f"stdout:\n{self.stdout}")
        if self.stderr:
            parts.append(f"stderr:\n{self.stderr}")
        return "\n".join(parts)


class ProcessDriver:
    guarantees_bounded_lifecycle: bool = False

    async def run(
        self,
        command: str | list[str],
        *,
        cwd: Path,
        timeout: int,
        env: dict[str, str] | None = None,
        shell: bool = False,
        on_output: OutputSink | None = None,
    ) -> ProcessResult:
        raise NotImplementedError


class NativeProcessDriver(ProcessDriver):
    def __init__(self, roots: WorkspaceRoots, safety: SafetyConfig, sandbox: SandboxConfig) -> None:
        self.roots = roots
        self.safety = safety
        self.sandbox = sandbox
        self.guarantees_bounded_lifecycle = os.name == "posix"

    async def run(
        self,
        command: str | list[str],
        *,
        cwd: Path,
        timeout: int,
        env: dict[str, str] | None = None,
        shell: bool = False,
        on_output: OutputSink | None = None,
    ) -> ProcessResult:
        cwd = self.roots.resolve(cwd, must_exist=True, kind="dir").path
        process_env = {key: value for key, value in os.environ.items() if key in self.safety.env_allowlist}
        process_env.update(env or {})
        preexec_fn = _resource_limiter(self.sandbox) if os.name == "posix" else None
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(asyncio.subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        started = monotonic_ms()
        status_task: asyncio.Task[int | None] | None = None
        if os.name == "posix":
            read_fd, write_fd = os.pipe()
            arguments = (
                ["shell", str(command)]
                if shell
                else ["exec", *(command if isinstance(command, list) else [command])]
            )
            try:
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    _POSIX_PROCESS_SUPERVISOR,
                    str(write_fd),
                    *arguments,
                    cwd=str(cwd),
                    env=process_env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                    preexec_fn=preexec_fn,
                    pass_fds=(write_fd,),
                )
            except BaseException:
                os.close(read_fd)
                raise
            finally:
                os.close(write_fd)
            status_task = asyncio.create_task(_read_process_status(read_fd))
        elif shell:
            process = await asyncio.create_subprocess_shell(
                str(command), cwd=str(cwd), env=process_env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                creationflags=creationflags,
            )
        else:
            argv = command if isinstance(command, list) else [command]
            process = await asyncio.create_subprocess_exec(
                *argv, cwd=str(cwd), env=process_env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                creationflags=creationflags,
            )
        timed_out = False
        limit = self.safety.max_process_output_chars
        stdout = _BoundedText(limit)
        stderr = _BoundedText(limit)
        emitted_chars = 0
        emitted_truncation = False
        output_queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()

        def queue_output(name: str, text: str) -> None:
            nonlocal emitted_chars, emitted_truncation
            if on_output is None or not text:
                return
            if limit > 0:
                available = max(0, limit - emitted_chars)
                chunk = text[:available]
                was_truncated = len(text) > available
            else:
                chunk = text
                was_truncated = False
            emitted_chars += len(chunk)
            if chunk:
                output_queue.put_nowait((name, chunk))
            if was_truncated and not emitted_truncation:
                emitted_truncation = True
                output_queue.put_nowait((name, OUTPUT_TRUNCATION_MARKER))

        async def dispatch_output() -> None:
            while True:
                item = await output_queue.get()
                if item is None:
                    return
                name, text = item
                assert on_output is not None
                try:
                    outcome = on_output(name, text)
                    if inspect.isawaitable(outcome):
                        await outcome
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass

        output_task = (
            asyncio.create_task(dispatch_output()) if on_output is not None else None
        )

        async def finish_output(*, drain: bool) -> bool:
            if output_task is None:
                return True
            output_queue.put_nowait(None)
            if drain:
                done, _ = await asyncio.wait(
                    {output_task}, timeout=_OUTPUT_OBSERVER_DRAIN_SECONDS
                )
                if done:
                    return True
            output_task.cancel()

            # Give ordinary cancellation-aware observers a chance to exit. Do
            # not await them without a bound: observers are outside the process
            # timeout contract and must not keep a command alive.
            await asyncio.sleep(0)
            return False

        if output_task is not None:
            def consume_output_task(task: asyncio.Task[None]) -> None:
                try:
                    task.result()
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

            output_task.add_done_callback(consume_output_task)

        async def pump(
            stream: asyncio.StreamReader | None, sink: _BoundedText, name: str
        ) -> None:
            if stream is None:
                return
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            while True:
                data = await stream.read(65_536)
                if not data:
                    break
                text = decoder.decode(data)
                sink.append(text)
                queue_output(name, text)
            final = decoder.decode(b"", final=True)
            sink.append(final)
            queue_output(name, final)

        io_tasks = [
            asyncio.create_task(pump(process.stdout, stdout, "stdout")),
            asyncio.create_task(pump(process.stderr, stderr, "stderr")),
        ]
        process_task = asyncio.create_task(process.wait())
        exit_code: int | None = None
        lifecycle_complete = os.name == "posix"
        abandon_io = False
        try:
            completion_tasks: set[asyncio.Task[object]] = {process_task}
            if status_task is not None:
                completion_tasks.add(status_task)
            done, _ = await asyncio.wait(
                completion_tasks,
                timeout=max(1, timeout),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if status_task is not None and status_task in done:
                exit_code = status_task.result()
                if exit_code is None:
                    lifecycle_complete = False
                    abandon_io = True
                else:
                    lifecycle_complete = await _kill_supervised_process_group(process)
            elif process_task in done:
                lifecycle_complete = False
                abandon_io = True
                exit_code = process.returncode
            else:
                timed_out = True
                await _terminate_process(
                    process,
                    grace_seconds=0.2 if status_task is not None else 2.0,
                )
            if abandon_io:
                for task in io_tasks:
                    task.cancel()
                await asyncio.gather(process_task, *io_tasks, return_exceptions=True)
            else:
                await asyncio.gather(process_task, *io_tasks)
            stream_complete = await finish_output(drain=True) and not abandon_io
        except asyncio.CancelledError:
            await _terminate_process(
                process,
                grace_seconds=0.2 if status_task is not None else 2.0,
            )
            await asyncio.gather(process_task, *io_tasks, return_exceptions=True)
            await finish_output(drain=False)
            raise
        finally:
            if status_task is not None:
                status_task.cancel()
                await asyncio.gather(status_task, return_exceptions=True)
        return ProcessResult(
            command=command if isinstance(command, str) else " ".join(command),
            exit_code=(
                exit_code
                if exit_code is not None
                else process.returncode
                if process.returncode is not None
                else -1
            ),
            stdout=stdout.render(),
            stderr=stderr.render(),
            duration_ms=monotonic_ms() - started,
            timed_out=timed_out,
            stream_truncated=emitted_truncation,
            stream_complete=stream_complete,
            lifecycle_complete=lifecycle_complete,
        )


class DockerProcessDriver(ProcessDriver):
    """Execute commands in an ephemeral, root-filesystem-read-only Docker container."""

    def __init__(self, roots: WorkspaceRoots, safety: SafetyConfig, sandbox: SandboxConfig) -> None:
        if shutil.which("docker") is None:
            raise ToolError("Docker sandbox selected but the docker executable is unavailable")
        self.roots = roots
        self.safety = safety
        self.sandbox = sandbox

    async def run(
        self,
        command: str | list[str],
        *,
        cwd: Path,
        timeout: int,
        env: dict[str, str] | None = None,
        shell: bool = False,
        on_output: OutputSink | None = None,
    ) -> ProcessResult:
        resolved = self.roots.resolve(cwd, must_exist=True, kind="dir")
        mount_points: dict[Path, Path] = {self.roots.primary: Path("/workspace")}
        for index, root in enumerate(self.roots.roots[1:], start=1):
            mount_points[root] = Path(f"/workspace_roots/root{index}")
        container_root = mount_points[resolved.root]
        container_cwd = container_root / resolved.path.relative_to(resolved.root)
        network = "bridge" if self.sandbox.docker_network and self.safety.network else "none"
        argv = [
            "docker", "run", "--rm", "--init", "--network", network,
            "--read-only",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m",
            "--memory", self.sandbox.docker_memory,
            "--cpus", str(self.sandbox.docker_cpus),
            "--pids-limit", "512",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
        ]
        if os.name == "posix" and hasattr(os, "getuid") and hasattr(os, "getgid"):
            argv += ["--user", f"{os.getuid()}:{os.getgid()}"]
        for host_root, container_path in mount_points.items():
            argv += ["-v", f"{host_root}:{container_path}:rw"]
        argv += ["-w", str(container_cwd)]
        for key, value in (env or {}).items():
            argv += ["-e", f"{key}={value}"]
        argv.append(self.sandbox.docker_image)
        if shell or isinstance(command, str):
            argv += ["sh", "-lc", str(command)]
        else:
            argv += command
        native = NativeProcessDriver(self.roots, self.safety, self.sandbox)
        return await native.run(
            argv, cwd=self.roots.primary, timeout=timeout, shell=False, on_output=on_output
        )


def build_process_driver(
    roots: WorkspaceRoots, safety: SafetyConfig, sandbox: SandboxConfig
) -> ProcessDriver:
    if sandbox.driver == "docker":
        return DockerProcessDriver(roots, safety, sandbox)
    return NativeProcessDriver(roots, safety, sandbox)


async def _terminate_process(
    process: asyncio.subprocess.Process,
    *,
    grace_seconds: float = 2.0,
) -> None:
    if process.returncode is not None:
        return
    try:
        if os.name == "posix" and process.pid:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
    except (ProcessLookupError, TimeoutError):
        try:
            if os.name == "posix" and process.pid:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass


async def _kill_supervised_process_group(process: asyncio.subprocess.Process) -> bool:
    """Kill a POSIX command group while its supervisor still owns the group ID."""

    if process.returncode is not None:
        return False
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return False
    await process.wait()
    return True


async def _read_process_status(fd: int) -> int | None:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[int | None] = loop.create_future()
    buffer = bytearray()
    os.set_blocking(fd, False)

    def read_ready() -> None:
        try:
            chunk = os.read(fd, 32)
        except BlockingIOError:
            return
        if not chunk:
            if not future.done():
                future.set_result(None)
            return
        buffer.extend(chunk)
        if b"\n" not in buffer:
            return
        try:
            status = int(bytes(buffer).splitlines()[0])
        except ValueError:
            status = None
        if not future.done():
            future.set_result(status)

    loop.add_reader(fd, read_ready)
    try:
        return await future
    finally:
        loop.remove_reader(fd)
        os.close(fd)


def _resource_limiter(config: SandboxConfig):  # type: ignore[no-untyped-def]
    def limit() -> None:
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (config.process_cpu_seconds, config.process_cpu_seconds))
            resource.setrlimit(resource.RLIMIT_FSIZE, (config.process_file_size_bytes, config.process_file_size_bytes))
            if hasattr(resource, "RLIMIT_CORE"):
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ImportError, OSError, ValueError):
            return
    return limit
