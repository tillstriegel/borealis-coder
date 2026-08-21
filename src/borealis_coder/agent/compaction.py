"""Deterministic, auditable conversation compaction with LLM summarization."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from ..models import Message, Role
from ..util import truncate_text

# Summarizer receives the rendered transcript of the older messages and returns
# the replacement summary text. May return an awaitable. Implementations should
# be exception-free; any failure falls back to deterministic truncation.
Summarizer = Callable[[str], str | Awaitable[str]]


def render_transcript(
    messages: list[Message],
    *,
    user_chars: int = 1200,
    assistant_chars: int = 1000,
    tool_chars: int = 4_000,
) -> str:
    """Render older messages for summarization, keeping tool outputs in detail."""
    lines: list[str] = []
    for message in messages:
        if message.role == Role.USER:
            lines.append("USER: " + truncate_text(message.content.strip(), user_chars))
        elif message.role == Role.ASSISTANT:
            if message.content.strip():
                lines.append("ASSISTANT: " + truncate_text(message.content.strip(), assistant_chars))
            if message.tool_calls:
                lines.append("TOOL CALLS: " + ", ".join(call.name for call in message.tool_calls))
        elif message.role == Role.TOOL:
            status = "ERROR" if message.is_error else "OK"
            lines.append(
                f"TOOL {message.tool_name or message.tool_call_id} [{status}]: "
                + truncate_text(message.content.strip(), tool_chars)
            )
    return "\n".join(lines)


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
    return [
        Message(
            role=Role.USER,
            content=summary,
            metadata={"compacted": True, "strategy": "deterministic", "source_messages": len(older)},
        ),
        *recent,
    ]


async def compact_messages_with_summary(
    messages: list[Message],
    summarizer: Summarizer | None,
    *,
    keep_recent: int = 18,
    summary_chars: int = 18_000,
) -> list[Message]:
    """Compact with an LLM summary when a summarizer is available.

    The deterministic truncation path remains the offline fallback: if no
    summarizer is configured, or the summarizer raises or returns empty output,
    the auditable local summary is used instead. Either way the compacted
    message records the strategy used so it is visible in traces.
    """
    compacted = compact_messages(messages, keep_recent=keep_recent, summary_chars=summary_chars)
    if summarizer is None or compacted == messages:
        return compacted
    split = len(messages) - (len(compacted) - 1)
    older = messages[: max(split, 0)]
    transcript = render_transcript(older)
    if not transcript.strip():
        return compacted
    prompt = (
        "Summarize the following portion of a coding-agent conversation for later "
        "reference. Preserve: the user's goals and constraints, decisions made, "
        "files touched, and the full substance of any tool outputs (test failures, "
        "stack traces, command results) needed to continue the work. Be factual and "
        "complete; do not add new requests.\n\n" + transcript
    )
    try:
        outcome: str | Awaitable[str] = summarizer(prompt)
        if isinstance(outcome, Awaitable):
            summary = await outcome
        else:
            summary = outcome
    except Exception:
        return compacted
    summary = str(summary or "").strip()
    if not summary:
        return compacted
    summary = truncate_text(summary, summary_chars)
    return [
        Message(
            role=Role.USER,
            content=summary,
            metadata={
                "compacted": True,
                "strategy": "llm",
                "source_messages": len(older),
            },
        ),
        *compacted[1:],
    ]
