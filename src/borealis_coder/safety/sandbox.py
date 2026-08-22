"""Process execution drivers with bounded output, time, and resources."""

from __future__ import annotations

import asyncio
import codecs
import inspect
import os
import shutil
import signal
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
        if shell:
            process = await asyncio.create_subprocess_shell(
                str(command), cwd=str(cwd), env=process_env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix", preexec_fn=preexec_fn,
                creationflags=creationflags,
            )
        else:
            argv = command if isinstance(command, list) else [command]
            process = await asyncio.create_subprocess_exec(
                *argv, cwd=str(cwd), env=process_env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix", preexec_fn=preexec_fn,
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

        tasks = [
            asyncio.create_task(pump(process.stdout, stdout, "stdout")),
            asyncio.create_task(pump(process.stderr, stderr, "stderr")),
            asyncio.create_task(process.wait()),
        ]
        try:
            _, pending = await asyncio.wait(tasks, timeout=max(1, timeout))
            if pending:
                timed_out = True
                await _terminate_process(process)
            await asyncio.gather(*tasks)
            stream_complete = await finish_output(drain=True)
        except asyncio.CancelledError:
            await _terminate_process(process)
            await asyncio.gather(*tasks, return_exceptions=True)
            await finish_output(drain=False)
            raise
        return ProcessResult(
            command=command if isinstance(command, str) else " ".join(command),
            exit_code=process.returncode if process.returncode is not None else -1,
            stdout=stdout.render(),
            stderr=stderr.render(),
            duration_ms=monotonic_ms() - started,
            timed_out=timed_out,
            stream_truncated=emitted_truncation,
            stream_complete=stream_complete,
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


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    try:
        if os.name == "posix" and process.pid:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        await asyncio.wait_for(process.wait(), timeout=2)
    except (ProcessLookupError, TimeoutError):
        try:
            if os.name == "posix" and process.pid:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass


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
