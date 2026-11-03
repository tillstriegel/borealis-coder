"""Small allowlisted diagnostics suitable for durable, user-visible receipts."""
from __future__ import annotations

from typing import Any

from ..errors import (
    ProviderAuthenticationError,
    ProviderContextOverflowError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from ..events import Redactor

_CODES = frozenset({
    "empty_response", "empty_after_recovery", "malformed_response",
    "invalid_api_key", "insufficient_quota", "rate_limit_exceeded",
    "context_length_exceeded", "model_not_found", "service_unavailable",
    "server_error", "invalid_request_error", "content_filter",
})
_PROVIDERS = frozenset({"openai", "openai_compatible", "openrouter", "anthropic", "gemini", "chatgpt", "mock"})


def provider_diagnostic(error: ProviderError, provider: str, redactor: Redactor) -> dict[str, Any]:
    """Never copy error messages, arbitrary class names, headers or detail objects.

    Even plausible-looking provider codes can contain private text. Unknown codes
    are omitted rather than sanitized into a misleading public identity.
    """
    category = "provider_error"
    for kind, name in (
        (ProviderAuthenticationError, "authentication"),
        (ProviderContextOverflowError, "context_overflow"),
        (ProviderRateLimitError, "rate_limit"),
        (ProviderUnavailableError, "unavailable"),
    ):
        if isinstance(error, kind):
            category = name
            break
    result: dict[str, Any] = {"category": category, "retryable": bool(error.retryable)}
    if provider in _PROVIDERS:
        result["provider"] = provider
    if type(error.status_code) is int and 100 <= error.status_code <= 599:
        result["status_code"] = error.status_code
    code = error.code
    if code is None and isinstance(error.details, dict):
        details = error.details.get("error", error.details)
        if isinstance(details, dict):
            code = details.get("code")
    if isinstance(code, str) and code in _CODES:
        result["code"] = code
    # Redact structured values before a caller can JSON-escape them. All fields
    # are fixed-size allowlisted tokens, integers or booleans.
    return redactor.value(result)
