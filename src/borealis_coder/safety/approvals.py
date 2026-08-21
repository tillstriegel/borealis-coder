"""Approval channels shared by CLI, ACP, and embedding applications."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..errors import ApprovalDenied
from .policy import PolicyAction, PolicyDecision


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    tool_name: str
    description: str
    decision: PolicyDecision
    arguments_preview: str


ApprovalCallback = Callable[[ApprovalRequest], bool | str | Awaitable[bool | str]]


class ApprovalManager:
    def __init__(self, callback: ApprovalCallback | None = None, *, cache: bool = True) -> None:
        self.callback = callback
        self.cache = cache
        self._approved: set[str] = set()

    async def enforce(self, request: ApprovalRequest) -> None:
        decision = request.decision
        if decision.action == PolicyAction.ALLOW:
            return
        if decision.action == PolicyAction.DENY:
            raise ApprovalDenied(decision.reason)
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
