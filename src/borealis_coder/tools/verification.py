"""Repository-aware verification planning and execution."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import PolicyError
from ..models import Effect, ToolResult
from ..safety import ApprovalRequest
from ..util import truncate_text
from .base import Tool, ToolContext, nullable, object_schema


@dataclass(slots=True)
class VerificationStep:
    name: str
    command: str
    timeout: int = 300


@dataclass(slots=True)
class VerificationReport:
    ok: bool
    steps: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "steps": self.steps}

    def render(self) -> str:
        rows = [f"verification_ok={str(self.ok).lower()}"]
        for step in self.steps:
            rows.append(f"\n## {step['name']}\ncommand: {step['command']}\nexit_code: {step['exit_code']}\nduration_ms: {step['duration_ms']}")
            if step.get("stdout"):
                rows.append("stdout:\n" + step["stdout"])
            if step.get("stderr"):
                rows.append("stderr:\n" + step["stderr"])
        return "\n".join(rows)


class VerificationPlanner:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def detect(self, *, max_seconds: int = 900) -> list[VerificationStep]:
        steps: list[VerificationStep] = []
        remaining = max_seconds

        def add(name: str, command: str, timeout: int) -> None:
            nonlocal remaining
            if remaining <= 0:
                return
            actual = min(timeout, remaining)
            steps.append(VerificationStep(name, command, actual))
            remaining -= actual

        if (self.workspace / "pyproject.toml").is_file() or (self.workspace / "setup.py").is_file():
            if shutil.which("ruff"):
                add("Python lint", "ruff check .", 180)
            if shutil.which("pyright"):
                add("Python types", "pyright", 300)
            if shutil.which("pytest"):
                add("Python tests", "pytest -q", 600)
            else:
                add("Python tests", "python -m unittest discover -s tests -v", 600)
        if (self.workspace / "package.json").is_file():
            package = _read_json(self.workspace / "package.json")
            scripts = package.get("scripts") or {}
            runner = "pnpm" if (self.workspace / "pnpm-lock.yaml").exists() and shutil.which("pnpm") else "npm"
            if "lint" in scripts:
                add("JavaScript lint", f"{runner} run lint", 300)
            if "typecheck" in scripts:
                add("TypeScript types", f"{runner} run typecheck", 300)
            elif "type-check" in scripts:
                add("TypeScript types", f"{runner} run type-check", 300)
            if "test" in scripts:
                add("JavaScript tests", f"{runner} test -- --runInBand" if runner == "npm" else f"{runner} test", 600)
            if "build" in scripts:
                add("JavaScript build", f"{runner} run build", 600)
        if (self.workspace / "Cargo.toml").is_file() and shutil.which("cargo"):
            add("Rust format", "cargo fmt --check", 180)
            add("Rust check", "cargo check --all-targets --locked --offline", 600)
            add("Rust tests", "cargo test --locked --offline", 900)
        if (self.workspace / "go.mod").is_file() and shutil.which("go"):
            add("Go tests", "go test ./...", 600)
            add("Go vet", "go vet ./...", 300)
        if (self.workspace / "Makefile").is_file() and not steps:
            add("Make test", "make test", 600)
        return steps[:6]

    async def run(self, context: ToolContext, steps: list[VerificationStep] | None = None) -> VerificationReport:
        steps = steps if steps is not None else self.detect(max_seconds=context.config.agent.auto_verify_max_seconds)
        report = VerificationReport(ok=True)
        if not steps:
            return report
        for step in steps:
            decision = context.policy.decide(
                tool_name="verify",
                effect=Effect.EXECUTE,
                arguments={"command": step.command},
                risk_hint="low",
            )
            await context.events.emit(
                "verification.policy",
                session_id=context.session_id,
                run_id=context.run_id,
                command=step.command,
                decision=decision.action.value,
                reason=decision.reason,
                risk=decision.risk,
            )
            try:
                await context.approvals.enforce(
                    ApprovalRequest(
                        tool_name="verify",
                        description=f"Run verification: {step.name}",
                        decision=decision,
                        arguments_preview=truncate_text(step.command, 4_000),
                    )
                )
            except PolicyError as error:
                report.ok = False
                report.steps.append({
                    "name": step.name,
                    "command": step.command,
                    "exit_code": -1,
                    "duration_ms": 0,
                    "timed_out": False,
                    "stdout": "",
                    "stderr": str(error),
                    "blocked": True,
                })
                break
            result = await context.process.run(
                step.command,
                cwd=self.workspace,
                timeout=step.timeout,
                shell=True,
            )
            report.steps.append({
                "name": step.name, "command": step.command, "exit_code": result.exit_code,
                "duration_ms": result.duration_ms, "timed_out": result.timed_out,
                "stdout": result.stdout, "stderr": result.stderr,
            })
            if not result.ok:
                report.ok = False
                break
        return report


class VerifyTool(Tool):
    name = "verify"
    description = "Run an explicit command or an automatically detected focused verification suite."
    effect = Effect.EXECUTE
    default_risk = "low"
    parameters = object_schema({
        "command": nullable("string", minLength=1, maxLength=10000),
        "timeout_seconds": nullable("integer", minimum=1, maximum=3600),
    })

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        planner = VerificationPlanner(context.workspace)
        command = arguments.get("command")
        if command:
            steps = [VerificationStep("Requested verification", str(command), int(arguments.get("timeout_seconds") or context.config.safety.command_timeout_seconds))]
        else:
            steps = planner.detect(max_seconds=context.config.agent.auto_verify_max_seconds)
        report = await planner.run(context, steps)
        return ToolResult(report.render(), is_error=not report.ok, metadata=report.to_dict())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}
