"""Central least-privilege policy decisions for tools and commands."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..config import SafetyConfig
from ..models import Effect
from .commands import CommandRisk, assess_command


class PolicyAction(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action: PolicyAction
    reason: str
    risk: str = "low"
    cache_key: str | None = None


class PolicyEngine:
    def __init__(
        self,
        config: SafetyConfig,
        *,
        interactive: bool = True,
        process_network_isolated: bool = False,
    ) -> None:
        self.config = config
        self.interactive = interactive
        self.process_network_isolated = process_network_isolated

    def decide(
        self,
        *,
        tool_name: str,
        effect: Effect,
        arguments: dict[str, Any],
        risk_hint: str | None = None,
    ) -> PolicyDecision:
        if effect == Effect.READ:
            if self.config.approval == "always":
                return self._ask(tool_name, "approval is required for every tool", "low")
            return PolicyDecision(PolicyAction.ALLOW, "read-only workspace operation")
        if effect == Effect.CONTROL and tool_name == "update_plan":
            return PolicyDecision(PolicyAction.ALLOW, "local plan-state update")

        if self.config.mode == "plan":
            return PolicyDecision(PolicyAction.DENY, "plan mode forbids mutations and execution", "high")

        if effect == Effect.NETWORK and not self.config.network:
            return PolicyDecision(PolicyAction.DENY, "network access is disabled", "high")
        if effect == Effect.EXECUTE and not self.config.allow_shell:
            return PolicyDecision(PolicyAction.DENY, "shell execution is disabled", "high")

        if tool_name == "shell":
            return self._command_decision(str(arguments.get("command") or ""))
        if tool_name == "verify" and arguments.get("command"):
            return self._command_decision(str(arguments.get("command") or ""))
        if tool_name == "git_commit" and not self.config.allow_git_commit:
            return self._ask_or_deny(tool_name, "git commits are disabled by default", "high")
        if tool_name == "git_push":
            if not self.config.allow_git_push or not self.config.network:
                return PolicyDecision(PolicyAction.DENY, "git push requires allow_git_push and network", "critical")
            return self._ask_or_deny(tool_name, "publishing remote changes", "critical")

        risk = risk_hint or ("medium" if effect == Effect.WRITE else "high")
        if tool_name.startswith("mcp__") and risk in {"high", "critical"}:
            return self._ask_or_deny(
                tool_name,
                f"{risk}-risk MCP operation",
                risk,
            )
        if self.config.approval == "always":
            return self._ask(tool_name, "approval is required for every tool", risk)
        if self.config.approval == "on-risk" and risk in {"high", "critical"}:
            return self._ask_or_deny(tool_name, f"{risk}-risk operation", risk)
        return PolicyDecision(PolicyAction.ALLOW, f"allowed {effect.value} operation", risk)

    def _command_decision(self, command: str) -> PolicyDecision:
        assessment = assess_command(command)
        key = f"shell:{assessment.executable}:{assessment.risk.name.lower()}"
        if assessment.risk == CommandRisk.FORBIDDEN:
            return PolicyDecision(PolicyAction.DENY, assessment.reason, "critical", key)
        if assessment.risk == CommandRisk.NETWORK and not self.config.network:
            return PolicyDecision(PolicyAction.DENY, f"{assessment.reason}; network is disabled", "high", key)
        if (
            not self.config.network
            and assessment.can_open_network
            and not self.process_network_isolated
        ):
            return PolicyDecision(
                PolicyAction.DENY,
                "command can open network connections and the native process driver "
                "cannot enforce network=false; use the Docker sandbox or enable network",
                "high",
                key,
            )
        lower_command = f" {command.lower()} "
        if assessment.executable == "git" and " push " in lower_command and not self.config.allow_git_push:
            return PolicyDecision(PolicyAction.DENY, "git push is disabled", "critical", key)
        if assessment.executable == "git" and " commit " in lower_command and not self.config.allow_git_commit:
            return PolicyDecision(PolicyAction.DENY, "git commit is disabled", "high", key)
        if assessment.risk >= CommandRisk.DESTRUCTIVE:
            return self._ask_or_deny("shell", assessment.reason, "critical", key)
        if assessment.risk in {CommandRisk.WRITE, CommandRisk.NETWORK}:
            return self._ask_or_deny("shell", assessment.reason, "high", key)
        if self.config.approval == "always":
            return self._ask("shell", assessment.reason, "low", key)
        return PolicyDecision(PolicyAction.ALLOW, assessment.reason, "low", key)

    def _ask_or_deny(self, name: str, reason: str, risk: str, key: str | None = None) -> PolicyDecision:
        if self.config.approval == "never" or not self.interactive:
            return PolicyDecision(PolicyAction.DENY, f"{reason}; no approval channel is available", risk, key)
        return self._ask(name, reason, risk, key)

    @staticmethod
    def _ask(name: str, reason: str, risk: str, key: str | None = None) -> PolicyDecision:
        return PolicyDecision(PolicyAction.ASK, reason, risk, key or name)
