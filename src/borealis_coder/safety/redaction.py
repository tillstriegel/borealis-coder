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

_STREAMING_TOKEN_SUFFIXES = (
    re.compile(r"(?<!\w)sk-[A-Za-z0-9_-]*$"),
    re.compile(r"(?<!\w)(?:ghp|github_pat|glpat)-?[A-Za-z0-9_]*$"),
    re.compile(r"(?<!\w)AIza[0-9A-Za-z_-]*$"),
    re.compile(
        r"(?<!\w)Bearer(?:\s+[A-Za-z0-9._~+/=-]*)?$",
        re.IGNORECASE,
    ),
    re.compile(r"(?<!\w)AKIA[0-9A-Z]{0,16}$"),
)
_STREAMING_PREFIXES = (
    ("sk-", False),
    ("ghp", False),
    ("github_pat", False),
    ("glpat", False),
    ("AIza", False),
    ("Bearer", True),
    ("AKIA", False),
    ("-----BEGIN PRIVATE KEY-----", False),
    ("-----BEGIN RSA PRIVATE KEY-----", False),
    ("-----BEGIN EC PRIVATE KEY-----", False),
    ("-----BEGIN OPENSSH PRIVATE KEY-----", False),
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

    def _streaming_suffix_start(self, value: str) -> int | None:
        """Return the earliest suffix that could become a secret."""
        starts: list[int] = []
        for secret in self._values:
            maximum = min(len(value), len(secret) - 1)
            for size in range(maximum, 0, -1):
                if value.endswith(secret[:size]):
                    starts.append(len(value) - size)
                    break
        for pattern in _STREAMING_TOKEN_SUFFIXES:
            if match := pattern.search(value):
                starts.append(match.start())
        for prefix, ignore_case in _STREAMING_PREFIXES:
            candidate = value.lower() if ignore_case else value
            expected = prefix.lower() if ignore_case else prefix
            for size in range(min(len(candidate), len(expected) - 1), 0, -1):
                if not candidate.endswith(expected[:size]):
                    continue
                start = len(value) - size
                if start == 0 or not re.match(r"\w", value[start - 1]):
                    starts.append(start)
                break
        return min(starts) if starts else None


class StreamingRedactor:
    """Redact a stream while retaining only possible secret suffixes."""

    def __init__(self, redactor: Redactor) -> None:
        self.redactor = redactor
        self._buffer = ""

    def feed(self, value: str) -> str:
        """Return safe text immediately and retain possible split secrets."""
        self._buffer += value
        safe_end = len(self._buffer)
        private_key_start = self._unclosed_private_key_start()
        if private_key_start is not None:
            safe_end = private_key_start
        suffix_start = self.redactor._streaming_suffix_start(
            self._buffer[:safe_end]
        )
        if suffix_start is not None:
            safe_end = min(safe_end, suffix_start)
        if safe_end <= 0:
            return ""

        complete = self._buffer[:safe_end]
        self._buffer = self._buffer[safe_end:]
        return self.redactor.text(complete)

    def flush(self, *, mask_incomplete: bool = False) -> str:
        """Redact and return the incomplete stream tail."""
        if not self._buffer:
            return ""
        mask_start = self._unclosed_private_key_start()
        if mask_incomplete:
            suffix_start = self.redactor._streaming_suffix_start(self._buffer)
            if suffix_start is not None:
                mask_start = (
                    suffix_start
                    if mask_start is None
                    else min(mask_start, suffix_start)
                )
        if mask_start is None:
            output = self.redactor.text(self._buffer)
        else:
            output = self.redactor.text(self._buffer[:mask_start]) + "[REDACTED]"
        self._buffer = ""
        return output

    def _unclosed_private_key_start(self) -> int | None:
        starts = list(_PRIVATE_KEY_BEGIN.finditer(self._buffer))
        if not starts:
            return None
        ends = list(_PRIVATE_KEY_END.finditer(self._buffer))
        start = starts[-1].start()
        if not ends or ends[-1].start() < start:
            return start
        return None
