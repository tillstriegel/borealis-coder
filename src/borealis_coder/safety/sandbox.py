"""Process execution drivers with bounded output, time, and resources."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
from dataclasses import dataclass
from pathlib import Path

from ..config import SandboxConfig, SafetyConfig
from ..errors import ToolError
from ..util import monotonic_ms, truncate_text
from .paths import WorkspaceRoots


@dataclass(slots=True)
class ProcessResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False

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
        try:
            stdout_b, stderr_b = await asyncio.wait_for(process.communicate(), timeout=max(1, timeout))
        except TimeoutError:
            timed_out = True
            await _terminate_process(process)
            stdout_b, stderr_b = await process.communicate()
        except asyncio.CancelledError:
            await _terminate_process(process)
            with contextlib.suppress(Exception):
                await process.communicate()
            raise
        limit = self.safety.max_process_output_chars
        stdout = truncate_text(stdout_b.decode("utf-8", errors="replace"), limit)
        stderr = truncate_text(stderr_b.decode("utf-8", errors="replace"), limit)
        return ProcessResult(
            command=command if isinstance(command, str) else " ".join(command),
            exit_code=process.returncode if process.returncode is not None else -1,
            stdout=stdout,
            stderr=stderr,
            duration_ms=monotonic_ms() - started,
            timed_out=timed_out,
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
        return await native.run(argv, cwd=self.roots.primary, timeout=timeout, shell=False)


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
