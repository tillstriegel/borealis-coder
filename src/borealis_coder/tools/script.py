"""Run-scoped programmatic tool calling with a supervised QuickJS subprocess."""
from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from ..errors import BudgetExceeded, Cancelled, ToolError
from ..models import Effect, ToolCall, ToolResult
from ..util import finish_on_cancellation, json_dumps, new_id
from .base import MutationScope, Tool, ToolContext, ToolRegistry, object_schema


def _reject_json_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def decode_packet(line: bytes, limit: int) -> dict[str, Any]:
    """Check size and nesting before JSON decoding or schema recursion."""
    if len(line) > limit or not line.endswith(b"\n"):
        raise ToolError("Script IPC limit exceeded or incomplete frame")
    depth = 0
    quoted = escaped = False
    for byte in line:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            if depth > 32:
                raise ToolError("Script IPC nesting limit exceeded")
        elif byte in (93, 125):
            depth -= 1
    try:
        value = json.loads(line, parse_constant=_reject_json_constant)
    except (ValueError, RecursionError) as error:
        raise ToolError("Invalid Script IPC JSON") from error
    if not isinstance(value, dict):
        raise ToolError("Invalid Script IPC object")
    return value


@dataclass(slots=True)
class ScriptJob:
    id: str
    parent_id: str
    output: str = ""
    state: str = "running"
    error: str = ""
    calls: int = 0
    errors: int = 0
    task: asyncio.Task[None] | None = None
    process: asyncio.subprocess.Process | None = None
    notified: bool = False
    children: set[asyncio.Task[Any]] = field(default_factory=set)
    findings: list[dict[str, Any]] = field(default_factory=list)
    plan: list[dict[str, Any]] | None = None
    changed_files: set[str] = field(default_factory=set)


NestedExecutor = Callable[[ToolCall, ToolContext], Awaitable[ToolResult]]


class ScriptScope:
    """Own every job, child call and worker until the exact run ends."""

    def __init__(self, context: ToolContext, registry: ToolRegistry, execute: NestedExecutor,
                 mutation_lock: asyncio.Lock, job_slots: set[str]) -> None:
        self.context = context
        self.registry = registry
        self.execute = execute
        self.config = context.config.script
        self.mutation_lock = mutation_lock
        self.job_slots = job_slots
        self.reads = asyncio.Semaphore(self.config.parallel_reads)
        self.jobs: dict[str, ScriptJob] = {}
        self.archive_remaining = self.config.archive_bytes
        self.closed = False
        self.fatal: BaseException | None = None

    @property
    def active(self) -> bool:
        return any(j.task is not None and not j.task.done() for j in self.jobs.values())

    def check(self) -> None:
        if self.fatal is not None:
            raise self.fatal
        if self.closed:
            raise ToolError("Script run has ended")
        budget = self.context.metadata.get("budget")
        if budget is not None:
            budget._check_time_and_cost()

    def _fail(self, error: BaseException) -> None:
        if self.fatal is not None:
            return
        self.fatal = error
        # Each owned job drains its calls and worker in _run's finally block.
        # Do not await siblings here: their cleanup can report another fatal error.
        current = asyncio.current_task()
        for job in self.jobs.values():
            if job.task is not None and job.task is not current and not job.task.done():
                job.task.cancel()

    async def start(self, source: str, parent_id: str, background: bool) -> ToolResult:
        self.check()
        if len(source.encode("utf-8")) > self.config.source_bytes:
            raise ToolError("Script source byte limit exceeded")
        if len(self.job_slots) >= self.config.max_jobs:
            raise ToolError("Script concurrent job limit reached")
        # Terminal jobs stay inspectable for this run, under a bounded count.
        if len(self.jobs) >= self.config.max_calls:
            raise ToolError("Script run job history limit reached")
        job = ScriptJob(new_id("script"), parent_id)
        self.jobs[job.id] = job
        self.job_slots.add(job.id)
        job.task = asyncio.create_task(self._run(job, source))
        if not background:
            await asyncio.wait({job.task}, timeout=min(300, self.config.foreground_seconds))
        self.check()
        if job.task.done():
            job.notified = True
        return self.result(job)

    def result(self, job: ScriptJob) -> ToolResult:
        status = {"job_id": job.id, "state": job.state, "calls": job.calls,
                  "tool_errors": job.errors, "error": job.error}
        output = self.context.events.redactor.text(job.output)
        return ToolResult(json_dumps(status) + ("\n" + output if output else ""),
                          is_error=job.state in {"error", "stopped"} or bool(job.errors),
                          metadata={"script_job": status, "retained_findings": job.findings,
                                    "changed_files": sorted(job.changed_files),
                                    **({"retained_plan": job.plan} if job.plan is not None else {})})

    async def control(self, job_id: str, action: str, seconds: float) -> ToolResult:
        self.check()
        if not math.isfinite(seconds):
            raise ToolError("Script wait seconds must be finite")
        job = self.jobs.get(job_id)
        if job is None:
            raise ToolError("Script job is unavailable in this run/session")
        assert job.task is not None
        if action == "stop":
            job.task.cancel()
            await finish_on_cancellation(asyncio.gather(job.task, return_exceptions=True))
        elif action == "wait":
            await asyncio.wait({job.task}, timeout=min(300, seconds, self.config.foreground_seconds))
        self.check()
        return self.result(job)

    def retain_result(self, result: ToolResult, context: ToolContext) -> ToolResult:
        try:
            if (not isinstance(result.output, str) or not isinstance(result.metadata, dict)
                    or type(result.is_error) is not bool
                    or (result.data is not None and not isinstance(result.data, dict))):
                raise ValueError("Invalid tool result")
            payload = {**result.to_dict(), "data": result.data}
            raw = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode()
            decode_packet(raw + b"\n", self.config.result_bytes)
            result = replace(result, output=context.events.redactor.text(result.output),
                             data=context.events.redactor.value(result.data),
                             metadata=context.events.redactor.value(result.metadata))
            encoded = json.dumps({**result.to_dict(), "data": result.data}, ensure_ascii=True, allow_nan=False).encode()
            decode_packet(encoded + b"\n", self.config.result_bytes)
        except (ValueError, TypeError, RecursionError, ToolError):
            result = ToolResult("Script tool result exceeds the size/type/depth limit", is_error=True)
            encoded = json_dumps(result.to_dict()).encode()
        store = context.metadata["session_store"]
        artifact = store.save_output_artifact(
            context.session_id, context.workspace, encoded.decode(), source=context.tool_call_id,
            redactor=context.events.redactor, max_bytes=min(self.archive_remaining, 2_000_000),
        )
        self.archive_remaining = max(0, self.archive_remaining - artifact["stored_bytes"])
        result.metadata["output_artifact"] = artifact
        if result.data is not None:
            result.data = {**result.data, "_evidence": {
                "tool_call_id": context.tool_call_id, "output_artifact": artifact,
            }}
        return result

    async def _invoke(self, job: ScriptJob, request: dict[str, Any]) -> dict[str, Any]:
        self.check()
        name = request["name"]
        tool = self.registry.get(name)
        if tool is not None and not tool.script_accessible:
            raise ToolError("Recursive Script, job control and delegation are unavailable inside Script")
        context = replace(self.context, tool_call_id="", metadata={
            **self.context.metadata, "script_mode": True, "script_retain_result": self.retain_result,
            "script_parent_id": job.parent_id, "script_job_id": job.id,
        })
        call = ToolCall(name=name, arguments=request["arguments"], id=new_id("nested"))
        # Any tool with side effects runs sequentially, even if a plugin declares concurrency.
        lock = self.reads if tool and tool.concurrent and tool.effect == Effect.READ and tool.effective_mutation_scope == MutationScope.NONE else self.mutation_lock
        async with lock:
            self.check()
            before = set(context.changed_files)
            try:
                result = await self.execute(call, context)
            finally:
                job.changed_files.update(context.changed_files - before)
            if (tool is not None and tool.effect == Effect.WRITE and not result.is_error
                    and isinstance(result.metadata.get("path"), str)):
                job.changed_files.add(result.metadata["path"])
        if result.is_error:
            job.errors += 1
        else:
            if name == "record_finding" and isinstance(result.metadata.get("retained_finding"), dict):
                job.findings = [*job.findings, result.metadata["retained_finding"]][-12:]
            if name == "update_plan" and isinstance(result.metadata.get("items"), list):
                job.plan = result.metadata["items"]
        return {"id": request["id"], "result": {**result.to_dict(), "data": result.data}}

    async def _run(self, job: ScriptJob, source: str) -> None:
        try:
            async with asyncio.timeout(self.config.wall_seconds):
                await self._exchange(job, source)
        except TimeoutError:
            job.state, job.error = "error", "Script wall time limit exceeded"
        except (BudgetExceeded, Cancelled) as error:
            self._fail(error)
            job.state, job.error = "error", "Owning run stopped"
        except asyncio.CancelledError:
            job.state, job.error = "stopped", "Script stopped; partial effects are retained"
            raise
        except Exception as error:
            job.state = "error"
            job.error = self.context.events.redactor.text(str(error))[:1000]
        finally:
            try:
                await finish_on_cancellation(self._cleanup(job))
            except Exception:
                self._fail(ToolError("Script cleanup failed; the owning run stopped"))
                job.state = "error"
                job.error = (job.error + "; " if job.error else "") + "Script cleanup failed"
                self.context.mutation_tracking = "incomplete"
                self.context.lifecycle_uncertainty_only = False
            finally:
                self._release_slot(job)
            if job.state == "running":
                job.state, job.error = "error", "Script ended without a terminal result"
            try:
                await self.context.events.emit("script.completed", session_id=self.context.session_id,
                                               run_id=self.context.run_id, **self.result(job).metadata)
            except Exception:
                self._fail(ToolError("Script completion could not be recorded"))
                job.state = "error"
                job.error = (job.error + "; " if job.error else "") + "Completion recording failed"

    def _release_slot(self, job: ScriptJob) -> None:
        if ((job.process is None or job.process.returncode is not None)
                and all(task.done() for task in job.children)):
            self.job_slots.discard(job.id)

    async def _cleanup(self, job: ScriptJob) -> None:
        process = job.process
        failure: OSError | TimeoutError | None = None
        if process is not None and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            except OSError as error:
                failure = error
        for task in job.children:
            task.cancel()
        await asyncio.gather(*job.children, return_exceptions=True)
        if process is not None:
            try:
                async with asyncio.timeout(5):
                    await process.wait()
            except (OSError, TimeoutError) as error:
                failure = failure or error
        if failure is not None:
            raise ToolError("Script worker cleanup did not complete") from failure

    async def _spawn(self, job: ScriptJob) -> None:
        # -I ignores PYTHONPATH, user site packages, and cwd imports. No provider credentials.
        env = {name: os.environ[name] for name in ("SYSTEMROOT", "WINDIR", "TMPDIR", "TMP", "TEMP") if name in os.environ}
        job.process = await asyncio.create_subprocess_exec(
            sys.executable, "-I", str(Path(__file__).with_name("_script_worker.py")),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, env=env, cwd=Path(__file__).parent,
            limit=self.config.ipc_bytes + 1,
        )

    async def _exchange(self, job: ScriptJob, source: str) -> None:
        await finish_on_cancellation(self._spawn(job))
        process = job.process
        assert process is not None
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(json.dumps(asdict(self.config)).encode() + b"\n")
        process.stdin.write(json.dumps(source, ensure_ascii=True).encode() + b"\n")
        await process.stdin.drain()
        reader = asyncio.create_task(process.stdout.readline())
        job.children.add(reader)
        calls: set[asyncio.Task[Any]] = set()
        seen: set[int] = set()
        while True:
            self.check()
            done, _ = await asyncio.wait({reader, *calls}, return_when=asyncio.FIRST_COMPLETED)
            if reader in done:
                job.children.discard(reader)
                packet = decode_packet(reader.result(), self.config.ipc_bytes)
                requests = packet.get("calls")
                output = packet.get("output")
                state = packet.get("state")
                if (not isinstance(requests, list) or not isinstance(output, list)
                        or any(not isinstance(text, str) for text in output)
                        or state not in {"running", "completed", "error"}
                        or not isinstance(packet.get("error"), str)
                        or type(packet.get("pending")) is not int):
                    raise ToolError("Invalid Script IPC packet")
                printed = "".join(output)
                if len((job.output + printed).encode("utf-8")) > self.config.output_bytes:
                    raise ToolError("Script printed output limit exceeded")
                job.output += printed
                if state == "error":
                    raise ToolError(packet["error"] or "Script execution failed")
                if job.calls + len(requests) > self.config.max_calls:
                    raise ToolError("Script nested call limit exceeded")
                for request in requests:
                    if (not isinstance(request, dict) or set(request) != {"id", "name", "arguments"}
                            or type(request["id"]) is not int or request["id"] in seen
                            or not isinstance(request["name"], str) or len(request["name"]) > 256
                            or not isinstance(request["arguments"], dict)):
                        raise ToolError("Invalid Script tool request")
                    seen.add(request["id"])
                    job.calls += 1
                    task = asyncio.create_task(self._invoke(job, request))
                    job.children.add(task)
                    calls.add(task)
                if state == "completed" and not calls and packet["pending"] == 0:
                    break
                reader = asyncio.create_task(process.stdout.readline())
                job.children.add(reader)
            for task in done:
                if task not in calls:
                    continue
                response = task.result()  # fatal host errors never become catchable JS errors
                calls.remove(task)
                job.children.discard(task)
                encoded = json.dumps(response, ensure_ascii=True, allow_nan=False).encode() + b"\n"
                if len(encoded) > self.config.ipc_bytes:
                    raise ToolError("Script result IPC limit exceeded")
                process.stdin.write(encoded)
                await process.stdin.drain()
        await process.wait()
        if process.returncode != 0:
            raise ToolError("Script worker failed")
        job.state = "completed"

    async def notifications(self, *, wait: bool = False) -> list[ToolResult]:
        if wait:
            tasks = [j.task for j in self.jobs.values() if j.task is not None and not j.task.done()]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        self.check()
        results = []
        for job in self.jobs.values():
            if job.task is not None and job.task.done() and not job.notified:
                job.notified = True
                results.append(self.result(job))
        return results

    async def close(self) -> None:
        self.closed = True
        tasks = [j.task for j in self.jobs.values() if j.task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        await finish_on_cancellation(asyncio.gather(*tasks, return_exceptions=True))
        # Retry idempotent termination if an earlier cleanup failed before reaping.
        for job in self.jobs.values():
            if job.id in self.job_slots:
                await finish_on_cancellation(self._cleanup(job))
                self._release_slot(job)


class ScriptTool(Tool):
    name = "script"
    script_accessible = False
    description = "Run JS with await tools.NAME(args), Promise.all and text(value). read_file: {content,path,sha256}; grep: {matches,coverage}; other tools: {output,metadata}. Tool errors reject. Only printed output returns."
    effect = Effect.READ
    mutation_scope = MutationScope.TRACKED
    parameters = object_schema({
        "source": {"type": "string", "minLength": 1},
        "background": {"type": "boolean"},
    })

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        scope = context.metadata.get("script_scope")
        if not context.config.script.enabled or not isinstance(scope, ScriptScope):
            raise ToolError("Script is unavailable; enable script.enabled in trusted config and install quickjs-ng")
        return await scope.start(arguments["source"], context.tool_call_id, arguments["background"])


class ScriptJobTool(Tool):
    name = "script_job"
    script_accessible = False
    description = "Inspect/wait/stop this run's Script jobs. Completion is notified automatically."
    effect = Effect.READ
    mutation_scope = MutationScope.TRACKED
    parameters = object_schema({
        "job_id": {"type": "string", "minLength": 1},
        "action": {"type": "string", "enum": ["status", "wait", "stop"]},
        "seconds": {"type": "number", "minimum": 0, "maximum": 300},
    })

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        scope = context.metadata.get("script_scope")
        if not isinstance(scope, ScriptScope):
            raise ToolError("Script job is unavailable in this run/session")
        return await scope.control(arguments["job_id"], arguments["action"], arguments["seconds"])
