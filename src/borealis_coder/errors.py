"""Typed errors used across the runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class BorealisError(Exception):
    """Base class for expected Borealis errors."""


class ConfigurationError(BorealisError):
    """Raised when configuration is invalid or incomplete."""


class ProviderError(BorealisError):
    """Base class for provider failures."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.details = details


class ProviderAuthenticationError(ProviderError):
    """The provider rejected credentials."""


class ProviderRateLimitError(ProviderError):
    """The provider rate-limited a request."""


class ProviderContextOverflowError(ProviderError):
    """The request exceeded a provider context limit."""


class ProviderUnavailableError(ProviderError):
    """The provider is temporarily unavailable."""


class PolicyError(BorealisError):
    """An operation was denied by policy."""


class ApprovalDenied(PolicyError):
    """A user or non-interactive policy denied approval."""


class PathViolation(PolicyError):
    """A path escaped the configured workspace roots."""


class ToolError(BorealisError):
    """A tool failed in an expected way."""


class ToolValidationError(ToolError):
    """Tool arguments did not match the declared schema."""


class PatchError(ToolError):
    """A deterministic patch could not be applied safely."""


class SessionError(BorealisError):
    """A durable session operation failed."""


class BudgetExceeded(BorealisError):
    """An agent budget was exhausted."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class Cancelled(BorealisError):
    """The current run was cancelled."""


class ProtocolError(BorealisError):
    """A JSON-RPC, MCP, or ACP protocol contract was violated."""


@dataclass(slots=True)
class Diagnostic:
    """A machine-readable diagnostic item."""

    name: str
    ok: bool
    message: str
    details: dict[str, Any] | None = None
