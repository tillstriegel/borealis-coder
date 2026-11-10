"""Approval channels shared by CLI, ACP, and embedding applications."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ..config import SafetyConfig
from ..errors import ApprovalDenied
from .policy import PolicyAction, PolicyDecision


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    tool_name: str
    description: str
    decision: PolicyDecision
    arguments_preview: str
    arguments: dict[str, Any] | None = None
    tool_call_id: str = ""


ApprovalCallback = Callable[[ApprovalRequest], bool | str | Awaitable[bool | str]]


class ApprovalManager:
    def __init__(
        self, callback: ApprovalCallback | None = None, *, cache: bool = True,
        safety: SafetyConfig | None = None,
    ) -> None:
        self.callback = callback
        self.cache = cache
        self._approved: set[str] = set()
        self.safety = safety
        self.smart_callback: Callable[[ApprovalRequest], Awaitable[None]] | None = None

    async def enforce(self, request: ApprovalRequest) -> None:
        decision = request.decision
        if decision.action == PolicyAction.ALLOW:
            return
        if decision.action == PolicyAction.DENY:
            raise ApprovalDenied(decision.reason)
        if self.safety is not None and self.safety.approval == "smart":
            if self.smart_callback is None:
                raise ApprovalDenied("Smart approval unavailable: run with a configured judge provider.")
            await self.smart_callback(request)
            return
        key = decision.cache_key
        if key and key in self._approved:
            return
        if self.callback is None:
            raise ApprovalDenied(f"Approval required but no approval callback exists: {decision.reason}")
        answer = self.callback(request)
        if inspect.isawaitable(answer):
            answer = await answer
        normalized = str(answer).strip().lower() if not isinstance(answer, bool) else ("yes" if answer else "no")
        if normalized in {"yes", "y", "true", "allow", "allow_once", "allow_always"}:
            if self.cache and key and normalized in {"allow_always", "yes", "y", "true", "allow"}:
                self._approved.add(key)
            return
        raise ApprovalDenied(f"Approval denied: {decision.reason}")

    def clear(self) -> None:
        self._approved.clear()
