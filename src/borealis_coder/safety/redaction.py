"""Best-effort secret redaction for logs, events, and persisted errors."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

from ..util import json_dumps

_SECRET_NAME = re.compile(
    r"(?:api[_-]?key|access[_-]?token|auth(?:orization)?|password|passwd|secret|private[_-]?key|credential|cookie)",
    re.IGNORECASE,
)

_TOKEN_PATTERNS = [
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b(?:ghp|github_pat|glpat)-?[A-Za-z0-9_]{16,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\b(?:Bearer\s+)[A-Za-z0-9._~+/=-]{12,}\b", re.IGNORECASE),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.DOTALL),
]

_PRIVATE_KEY_BEGIN = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)
_PRIVATE_KEY_END = re.compile(
    r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)


class Redactor:
    def __init__(self, extra_values: list[str] | None = None) -> None:
        values = set(extra_values or [])
        for name, value in os.environ.items():
            if value and len(value) >= 8 and _SECRET_NAME.search(name):
                values.add(value)
        self._values = sorted(values, key=len, reverse=True)

    def text(self, value: str) -> str:
        result = value
        for secret in self._values:
            result = result.replace(secret, "[REDACTED]")
        for pattern in _TOKEN_PATTERNS:
            result = pattern.sub("[REDACTED]", result)
        return result

    def value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.value(item) for item in value)
        if isinstance(value, Mapping):
            output: dict[str, Any] = {}
            for key, item in value.items():
                if _SECRET_NAME.search(str(key)):
                    output[str(key)] = "[REDACTED]"
                else:
                    output[str(key)] = self.value(item)
            return output
        return value

    def json(self, value: Any) -> str:
        return json_dumps(self.value(value))


class StreamingRedactor:
    """Redact complete stream records without exposing split secrets."""

    def __init__(self, redactor: Redactor) -> None:
        self.redactor = redactor
        self._buffer = ""

    def feed(self, value: str) -> str:
        """Return redacted complete lines and retain an incomplete tail."""
        self._buffer += value
        boundary = max(self._buffer.rfind("\n"), self._buffer.rfind("\r"))
        if boundary < 0:
            return ""

        boundary = self._safe_boundary(boundary)
        if boundary < 0:
            return ""

        complete = self._buffer[: boundary + 1]
        self._buffer = self._buffer[boundary + 1 :]
        return self.redactor.text(complete)

    def flush(self) -> str:
        """Redact and return the incomplete stream tail."""
        if not self._buffer:
            return ""
        private_key_start = self._unclosed_private_key_start()
        if private_key_start is None:
            output = self.redactor.text(self._buffer)
        else:
            output = self.redactor.text(self._buffer[:private_key_start]) + "[REDACTED]"
        self._buffer = ""
        return output

    def _safe_boundary(self, boundary: int) -> int:
        """Keep private key blocks intact when selecting text to emit."""
        search_from = 0
        while True:
            start = _PRIVATE_KEY_BEGIN.search(self._buffer, search_from)
            if start is None or start.start() > boundary:
                return boundary
            end = _PRIVATE_KEY_END.search(self._buffer, start.end())
            if end is None:
                return max(
                    self._buffer.rfind("\n", 0, start.start()),
                    self._buffer.rfind("\r", 0, start.start()),
                )
            boundary = max(boundary, end.end() - 1)
            search_from = end.end()

    def _unclosed_private_key_start(self) -> int | None:
        starts = list(_PRIVATE_KEY_BEGIN.finditer(self._buffer))
        if not starts:
            return None
        ends = list(_PRIVATE_KEY_END.finditer(self._buffer))
        start = starts[-1].start()
        if not ends or ends[-1].start() < start:
            return start
        return None
