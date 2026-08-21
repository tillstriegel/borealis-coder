"""Deterministic, auditable conversation compaction."""

from __future__ import annotations

from ..models import Message, Role
from ..util import truncate_text


def compact_messages(messages: list[Message], *, keep_recent: int = 18, summary_chars: int = 18_000) -> list[Message]:
    if len(messages) <= keep_recent + 2:
        return messages
    split = len(messages) - keep_recent
    # Avoid splitting an assistant tool-call message from its following tool results.
    while split > 1 and messages[split].role == Role.TOOL:
        split -= 1
    older = messages[:split]
    recent = messages[split:]
    summary_lines = [
        "<deterministic_conversation_summary>",
        "Older conversation content was compacted locally. This summary is an audit aid, not a new user request.",
    ]
    for message in older:
        if message.role == Role.USER:
            summary_lines.append("USER: " + truncate_text(message.content.strip(), 1200))
        elif message.role == Role.ASSISTANT:
            if message.content.strip():
                summary_lines.append("ASSISTANT: " + truncate_text(message.content.strip(), 1000))
            if message.tool_calls:
                summary_lines.append("TOOL CALLS: " + ", ".join(call.name for call in message.tool_calls))
        elif message.role == Role.TOOL:
            status = "ERROR" if message.is_error else "OK"
            summary_lines.append(f"TOOL {message.tool_name or message.tool_call_id} [{status}]: " + truncate_text(message.content.strip(), 700))
    summary_lines.append("</deterministic_conversation_summary>")
    summary = truncate_text("\n".join(summary_lines), summary_chars)
    return [Message(role=Role.USER, content=summary, metadata={"compacted": True, "source_messages": len(older)})] + recent
