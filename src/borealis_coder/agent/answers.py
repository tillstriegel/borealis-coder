"""Terminal answer validation shared by primary and delegated conversations."""
from __future__ import annotations

import re


def protocol_answer(text: str) -> bool:
    return bool(re.match(r"^\s*<(?:[\uff5c|]DSML[\uff5c|]|tool_call(?:s)?\b|function_call\b)", text, re.I))


def usable_answer(text: str) -> bool:
    """Reject known protocol/continuation forms, not prose discussing markup."""
    value = text.strip()
    if not value or protocol_answer(value):
        return False
    intention = re.fullmatch(
        r"(?:Let me|I will|I'll) (?:read|inspect|check|look at) ([^\n]{1,160})",
        value, re.I,
    )
    if intention is None:
        return True
    # Quoted paths/commands are opaque; a subsequent prose clause remains valid.
    target = re.sub(r"`[^`]*`|\"[^\"]*\"|'[^']*'", "target", intention[1])
    return bool(re.search(r"[.!?:;]\s", target))
