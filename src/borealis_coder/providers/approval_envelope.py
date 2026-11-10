"""Validate judge reply structure before adapter normalization can hide extra output."""

from __future__ import annotations

from typing import Any

from ..config import ProviderConfig
from ..errors import ProviderError
from ..models import ProviderRequest, Usage


def validate_approval_request(request: ProviderRequest, config: ProviderConfig) -> None:
    if request.metadata.get("purpose") == "smart_approval" and "n" in config.extra_body:
        count = config.extra_body["n"]
        if type(count) is not int or count != 1:
            raise ProviderError(
                "Smart approval requires one completion.",
                code="invalid_approval_request",
            )


def validate_approval_envelope(
    request: ProviderRequest,
    data: dict[str, Any],
    *,
    protocol: str,
    usage: Usage,
) -> None:
    if request.metadata.get("purpose") != "smart_approval":
        return
    validators = {
        "chat": _valid_chat,
        "responses": _valid_responses,
        "anthropic": _valid_anthropic,
        "gemini": _valid_gemini,
    }
    if not validators[protocol](data):
        # Never include raw envelope content, hidden reasoning, or provider diagnostics.
        raise ProviderError(
            "Smart approval requires one tool-free text response.",
            code="invalid_approval_response",
            usage=usage,
        )


def _text_block(block: Any, types: set[str]) -> bool:
    return (
        isinstance(block, dict)
        and block.get("type") in types
        and isinstance(block.get("text"), str)
        and set(block) <= {"type", "text", "annotations", "logprobs", "citations"}
    )


def _one_text(blocks: Any, types: set[str]) -> bool:
    return isinstance(blocks, list) and len(blocks) == 1 and _text_block(blocks[0], types)


def _valid_chat(data: dict[str, Any]) -> bool:
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return False
    choice = choices[0]
    if not isinstance(choice, dict) or set(choice) - {
        "index",
        "message",
        "finish_reason",
        "native_finish_reason",
        "logprobs",
    }:
        return False
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return False
    if set(message) - {
        "role",
        "content",
        "refusal",
        "tool_calls",
        "function_call",
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "annotations",
        "audio",
    }:
        return False
    if (
        message.get("tool_calls") not in (None, [])
        or message.get("function_call") is not None
        or message.get("refusal") not in (None, "")
        or message.get("audio") is not None
    ):
        return False
    content = message.get("content")
    return isinstance(content, str) or _one_text(content, {"text"})


def _valid_responses(data: dict[str, Any]) -> bool:
    output = data.get("output")
    if not isinstance(output, list):
        return False
    messages = []
    for item in output:
        if not isinstance(item, dict):
            return False
        if item.get("type") == "reasoning":
            continue
        if (
            item.get("type") != "message"
            or item.get("role") != "assistant"
            or item.get("status") not in (None, "completed")
            or set(item) - {"id", "type", "role", "status", "content", "phase"}
        ):
            return False
        messages.append(item)
    return len(messages) == 1 and _one_text(messages[0].get("content"), {"output_text", "text"})


def _valid_anthropic(data: dict[str, Any]) -> bool:
    content = data.get("content")
    if not isinstance(content, list) or data.get("role") != "assistant":
        return False
    texts = []
    for block in content:
        if not isinstance(block, dict):
            return False
        if block.get("type") in {"thinking", "redacted_thinking"}:
            continue
        if not _text_block(block, {"text"}):
            return False
        texts.append(block)
    return len(texts) == 1


def _valid_gemini(data: dict[str, Any]) -> bool:
    steps = data.get("steps")
    if not isinstance(steps, list):
        return False
    outputs = []
    for step in steps:
        if not isinstance(step, dict):
            return False
        if step.get("type") == "thought":
            continue
        if step.get("type") != "model_output" or set(step) - {"type", "id", "role", "content"}:
            return False
        outputs.append(step)
    return len(outputs) == 1 and _one_text(outputs[0].get("content"), {"text"})
