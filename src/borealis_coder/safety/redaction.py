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
