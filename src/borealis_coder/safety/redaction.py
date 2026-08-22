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

_TOKEN_PATTERNS = (
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b(?:ghp|github_pat|glpat)-?[A-Za-z0-9_]{16,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\b(?:Bearer\s+)[A-Za-z0-9._~+/=-]{12,}\b", re.IGNORECASE),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)
_PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?"
    r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    re.DOTALL,
)

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
_STREAMING_TOKEN_PREFIXES = (
    ("sk-", False),
    ("ghp", False),
    ("github_pat", False),
    ("glpat", False),
    ("AIza", False),
    ("Bearer", True),
    ("AKIA", False),
)
_STREAMING_PRIVATE_KEY_PREFIXES = (
    "-----BEGIN PRIVATE KEY-----",
    "-----BEGIN RSA PRIVATE KEY-----",
    "-----BEGIN EC PRIVATE KEY-----",
    "-----BEGIN OPENSSH PRIVATE KEY-----",
)


def _partial_prefix_start(
    value: str, prefix: str, *, ignore_case: bool = False
) -> int | None:
    candidate = value.lower() if ignore_case else value
    expected = prefix.lower() if ignore_case else prefix
    for size in range(min(len(candidate), len(expected) - 1), 0, -1):
        if candidate.endswith(expected[:size]):
            return len(value) - size
    return None


class Redactor:
    def __init__(self, extra_values: list[str] | None = None) -> None:
        values = set(extra_values or [])
        for name, value in os.environ.items():
            if value and len(value) >= 8 and _SECRET_NAME.search(name):
                values.add(value)
        self._values = sorted(values, key=len, reverse=True)

    def text(self, value: str, *, preceding_char: str = "") -> str:
        result = value
        for secret in self._values:
            result = result.replace(secret, "[REDACTED]")
        boundary_context = ""
        if preceding_char:
            boundary_context = (
                "x" if re.match(r"\w", preceding_char[-1]) else " "
            )
        for pattern in _TOKEN_PATTERNS:
            if boundary_context:
                framed = boundary_context + result
                result = pattern.sub("[REDACTED]", framed)[1:]
            else:
                result = pattern.sub("[REDACTED]", result)
        return _PRIVATE_KEY_PATTERN.sub("[REDACTED]", result)

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

    def _streaming_suffix_start(
        self, value: str, *, preceding_char: str = ""
    ) -> int | None:
        """Return the earliest suffix that could become a secret."""
        starts: list[int] = []
        private_key_ends = list(_PRIVATE_KEY_END.finditer(value))
        completed_private_key_end = (
            private_key_ends[-1]
            if private_key_ends and private_key_ends[-1].end() == len(value)
            else None
        )
        for secret in self._values:
            maximum = min(len(value), len(secret) - 1)
            for size in range(maximum, 0, -1):
                if value.endswith(secret[:size]):
                    starts.append(len(value) - size)
                    break
        for pattern in _STREAMING_TOKEN_SUFFIXES:
            if match := pattern.search(value):
                before = value[match.start() - 1] if match.start() else preceding_char
                if not before or not re.match(r"\w", before):
                    starts.append(match.start())
        for prefix, ignore_case in _STREAMING_TOKEN_PREFIXES:
            start = _partial_prefix_start(value, prefix, ignore_case=ignore_case)
            if start is None:
                continue
            before = value[start - 1] if start else preceding_char
            if not before or not re.match(r"\w", before):
                starts.append(start)
        for prefix in _STREAMING_PRIVATE_KEY_PREFIXES:
            start = _partial_prefix_start(value, prefix)
            if start is not None and (
                completed_private_key_end is None
                or start < completed_private_key_end.start()
            ):
                starts.append(start)
        return min(starts) if starts else None


class StreamingRedactor:
    """Redact a stream while retaining only possible secret suffixes."""

    def __init__(self, redactor: Redactor) -> None:
        self.redactor = redactor
        self._buffer = ""
        self._preceding_char = ""

    def feed(self, value: str) -> str:
        """Return safe text immediately and retain possible split secrets."""
        self._buffer += value
        safe_end = len(self._buffer)
        private_key_start = self._unclosed_private_key_start()
        if private_key_start is not None:
            safe_end = private_key_start
        suffix_start = self.redactor._streaming_suffix_start(
            self._buffer[:safe_end], preceding_char=self._preceding_char
        )
        if suffix_start is not None:
            safe_end = min(safe_end, suffix_start)
        if safe_end <= 0:
            return ""

        complete = self._buffer[:safe_end]
        self._buffer = self._buffer[safe_end:]
        output = self.redactor.text(
            complete, preceding_char=self._preceding_char
        )
        self._preceding_char = complete[-1]
        return output

    def flush(self, *, mask_incomplete: bool = False) -> str:
        """Redact and return the incomplete stream tail."""
        if not self._buffer:
            return ""
        mask_start = self._unclosed_private_key_start()
        if mask_incomplete:
            suffix_start = self.redactor._streaming_suffix_start(
                self._buffer, preceding_char=self._preceding_char
            )
            if suffix_start is not None:
                mask_start = (
                    suffix_start
                    if mask_start is None
                    else min(mask_start, suffix_start)
                )
        if mask_start is None:
            output = self.redactor.text(
                self._buffer, preceding_char=self._preceding_char
            )
        else:
            output = self.redactor.text(
                self._buffer[:mask_start], preceding_char=self._preceding_char
            ) + "[REDACTED]"
        self._preceding_char = self._buffer[-1]
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
