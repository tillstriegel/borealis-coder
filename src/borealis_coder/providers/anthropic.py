"""Anthropic Messages API adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ..errors import ProviderError
from ..models import Message, ModelResponse, ProviderRequest, Role, ToolCall, Usage
from ..util import json_dumps
from .base import Provider, ProviderStreamEvent
from .http import HttpClient


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, config, api_key: str = "") -> None:  # type: ignore[no-untyped-def]
        super().__init__(config, api_key)
        self.http = HttpClient(timeout_seconds=config.timeout_seconds)

    def _headers(self) -> dict[str, str]:
        headers = {
            "anthropic-version": "2023-06-01",
            **self.config.headers,
        }
        if self.api_key:
            headers.setdefault("x-api-key", self.api_key)
        return headers

    def _payload(self, request: ProviderRequest, *, stream: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "system": request.system,
            "messages": self._messages(request.messages),
            "tools": [
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("parameters") or {},
                }
                for tool in request.tools
            ],
            "max_tokens": request.max_output_tokens,
            "stream": stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if not request.parallel_tool_calls:
            payload["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        return payload

    async def complete(self, request: ProviderRequest) -> ModelResponse:
        async def operation() -> ModelResponse:
            url = self.config.base_url.rstrip("/") + "/messages"
            response = await self.http.post_json(
                url, headers=self._headers(), payload=self._payload(request)
            )
            if not isinstance(response.data, dict):
                raise ProviderError("Anthropic returned a non-object response")
            return self._parse(response.data, retain_raw=True)

        return await self.with_retries(operation)

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderStreamEvent]:
        url = self.config.base_url.rstrip("/") + "/messages"
        payload = self._payload(request, stream=True)
        text_parts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        usage = Usage(requests=1)
        message_id: str | None = None
        model: str | None = None
        stop_reason: str | None = None
        async for event in self.http.stream_sse(url, headers=self._headers(), payload=payload):
            try:
                data = json.loads(event.data)
            except json.JSONDecodeError:
                continue
            event_type = data.get("type") or event.event
            if event_type == "message_start":
                message = data.get("message") or {}
                message_id = message.get("id")
                model = message.get("model")
                initial = message.get("usage") or {}
                usage.input_tokens = int(initial.get("input_tokens", 0) or 0)
                usage.cached_input_tokens = int(initial.get("cache_read_input_tokens", 0) or 0)
                usage.cache_write_tokens = int(initial.get("cache_creation_input_tokens", 0) or 0)
            elif event_type == "content_block_start":
                index = int(data.get("index", 0))
                block = data.get("content_block") or {}
                if block.get("type") == "tool_use":
                    calls[index] = {
                        "id": str(block.get("id") or ""),
                        "name": str(block.get("name") or ""),
                        "arguments": "",
                        "input": block.get("input") or {},
                    }
            elif event_type == "content_block_delta":
                index = int(data.get("index", 0))
                delta = data.get("delta") or {}
                if delta.get("type") == "text_delta":
                    text = str(delta.get("text") or "")
                    text_parts.append(text)
                    yield ProviderStreamEvent(type="text_delta", text=text)
                elif delta.get("type") == "input_json_delta":
                    partial = str(delta.get("partial_json") or "")
                    call = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    call["arguments"] = str(call.get("arguments") or "") + partial
                    yield ProviderStreamEvent(
                        type="tool_call_delta",
                        data={
                            "index": index,
                            "id": call.get("id", ""),
                            "name": call.get("name", ""),
                            "delta": partial,
                        },
                    )
            elif event_type == "message_delta":
                delta = data.get("delta") or {}
                stop_reason = delta.get("stop_reason") or stop_reason
                delta_usage = data.get("usage") or {}
                usage.output_tokens = int(delta_usage.get("output_tokens", 0) or usage.output_tokens)
            elif event_type == "error":
                error = data.get("error") or {}
                raise ProviderError(str(error.get("message") or error))
        tool_calls: list[ToolCall] = []
        for _, item in sorted(calls.items()):
            raw = str(item.get("arguments") or "")
            arguments = _parse_arguments(raw) if raw else dict(item.get("input") or {})
            tool_calls.append(
                ToolCall(
                    id=str(item.get("id") or ""),
                    name=str(item.get("name") or ""),
                    arguments=arguments,
                    raw_arguments=raw or json_dumps(arguments),
                )
            )
        self.price_usage(usage)
        result = ModelResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=stop_reason,
            response_id=message_id,
            model=model,
        )
        yield ProviderStreamEvent(type="completed", response=result)

    @staticmethod
    def _messages(messages: list[Message]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for message in messages:
            if message.role == Role.SYSTEM:
                continue
            if message.role == Role.USER:
                _append_content(output, "user", [{"type": "text", "text": message.content}])
            elif message.role == Role.ASSISTANT:
                content: list[dict[str, Any]] = []
                if message.content:
                    content.append({"type": "text", "text": message.content})
                content.extend(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                    for call in message.tool_calls
                )
                _append_content(output, "assistant", content)
            elif message.role == Role.TOOL:
                _append_content(
                    output,
                    "user",
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id,
                            "content": message.content,
                            "is_error": message.is_error,
                        }
                    ],
                )
        return output

    def _parse(self, data: dict[str, Any], *, retain_raw: bool) -> ModelResponse:
        text: list[str] = []
        calls: list[ToolCall] = []
        for block in data.get("content", []) or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text.append(str(block.get("text") or ""))
            elif block.get("type") == "tool_use":
                arguments = block.get("input") or {}
                calls.append(
                    ToolCall(
                        id=str(block.get("id") or ""),
                        name=str(block.get("name") or ""),
                        arguments=dict(arguments) if isinstance(arguments, dict) else {"value": arguments},
                        raw_arguments=json_dumps(arguments),
                    )
                )
        usage_data = data.get("usage") or {}
        usage = self.price_usage(
            Usage(
                input_tokens=int(usage_data.get("input_tokens", 0) or 0),
                output_tokens=int(usage_data.get("output_tokens", 0) or 0),
                cached_input_tokens=int(usage_data.get("cache_read_input_tokens", 0) or 0),
                cache_write_tokens=int(usage_data.get("cache_creation_input_tokens", 0) or 0),
                requests=1,
            )
        )
        return ModelResponse(
            text="".join(text),
            tool_calls=calls,
            usage=usage,
            stop_reason=data.get("stop_reason"),
            response_id=data.get("id"),
            model=data.get("model"),
            raw=data if retain_raw else None,
        )


def _append_content(output: list[dict[str, Any]], role: str, content: list[dict[str, Any]]) -> None:
    if not content:
        return
    if output and output[-1].get("role") == role:
        existing = output[-1].get("content")
        if isinstance(existing, list):
            existing.extend(content)
            return
    output.append({"role": role, "content": content})


def _parse_arguments(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_raw": raw}
    return value if isinstance(value, dict) else {"value": value}
