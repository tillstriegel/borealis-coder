"""Google Gemini Interactions API adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlencode

from ..errors import ProviderError
from ..models import Message, ModelResponse, ProviderRequest, Role, ToolCall, Usage
from ..util import json_dumps
from .base import Provider, ProviderStreamEvent
from .http import HttpClient


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, config, api_key: str = "") -> None:  # type: ignore[no-untyped-def]
        super().__init__(config, api_key)
        self.http = HttpClient(timeout_seconds=config.timeout_seconds)

    def _headers(self) -> dict[str, str]:
        headers = dict(self.config.headers)
        if self.api_key:
            headers.setdefault("x-goog-api-key", self.api_key)
        return headers

    def _url(self, *, stream: bool = False) -> str:
        url = self.config.base_url.rstrip("/") + "/interactions"
        if stream:
            url += "?" + urlencode({"alt": "sse"})
        return url

    def _payload(self, request: ProviderRequest, *, stream: bool = False) -> dict[str, Any]:
        generation: dict[str, Any] = {
            "max_output_tokens": request.max_output_tokens,
            "tool_choice": "auto",
        }
        if request.temperature is not None:
            generation["temperature"] = request.temperature
        if request.reasoning_effort:
            effort = request.reasoning_effort.lower()
            generation["thinking_level"] = (
                effort if effort in {"minimal", "low", "medium", "high"} else "medium"
            )
        payload: dict[str, Any] = {
            "model": request.model,
            "system_instruction": request.system,
            "input": self._steps(request.messages),
            "tools": [
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters") or {},
                }
                for tool in request.tools
            ],
            "stream": stream,
            "store": False,
            "generation_config": generation,
        }
        if request.response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": request.response_schema,
            }
        return payload

    async def complete(self, request: ProviderRequest) -> ModelResponse:
        async def operation() -> ModelResponse:
            response = await self.http.post_json(
                self._url(), headers=self._headers(), payload=self._payload(request)
            )
            if not isinstance(response.data, dict):
                raise ProviderError("Gemini returned a non-object response")
            return self._parse(response.data, retain_raw=True)

        return await self.with_retries(operation)

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderStreamEvent]:
        payload = self._payload(request, stream=True)
        text_parts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        final_data: dict[str, Any] | None = None
        async for event in self.http.stream_sse(
            self._url(stream=True), headers=self._headers(), payload=payload
        ):
            try:
                data = json.loads(event.data)
            except json.JSONDecodeError:
                continue
            event_type = str(data.get("event_type") or data.get("type") or event.event)
            if event_type == "step.start":
                index = int(data.get("index", 0))
                step = data.get("step") or {}
                if step.get("type") == "function_call":
                    arguments = step.get("arguments")
                    calls[index] = {
                        "id": str(step.get("id") or ""),
                        "name": str(step.get("name") or ""),
                        "arguments": json_dumps(arguments) if isinstance(arguments, dict) else str(arguments or ""),
                    }
            elif event_type == "step.delta":
                index = int(data.get("index", 0))
                delta = data.get("delta") or {}
                if delta.get("type") == "text":
                    text = str(delta.get("text") or "")
                    text_parts.append(text)
                    yield ProviderStreamEvent(type="text_delta", text=text)
                elif delta.get("type") == "arguments":
                    partial = str(delta.get("partial_arguments") or "")
                    call = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    call["arguments"] += partial
                    yield ProviderStreamEvent(
                        type="tool_call_delta",
                        data={
                            "index": index,
                            "id": call.get("id", ""),
                            "name": call.get("name", ""),
                            "delta": partial,
                        },
                    )
            elif event_type in {"interaction.completed", "interaction.complete"}:
                final_data = data.get("interaction") if isinstance(data.get("interaction"), dict) else data
        if final_data:
            result = self._parse(final_data, retain_raw=True)
        else:
            result = ModelResponse(
                text="".join(text_parts),
                tool_calls=[_partial_call(item) for _, item in sorted(calls.items())],
            )
        yield ProviderStreamEvent(type="completed", response=result)

    @staticmethod
    def _steps(messages: list[Message]) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = []
        for message in messages:
            if message.role == Role.USER:
                steps.append(
                    {
                        "type": "user_input",
                        "content": [{"type": "text", "text": message.content}],
                    }
                )
            elif message.role == Role.ASSISTANT:
                if message.content:
                    steps.append(
                        {
                            "type": "model_output",
                            "content": [{"type": "text", "text": message.content}],
                        }
                    )
                for call in message.tool_calls:
                    steps.append(
                        {
                            "type": "function_call",
                            "id": call.id,
                            "name": call.name,
                            "arguments": call.arguments,
                        }
                    )
            elif message.role == Role.TOOL:
                steps.append(
                    {
                        "type": "function_result",
                        "call_id": message.tool_call_id,
                        "name": message.tool_name,
                        "is_error": message.is_error,
                        "result": [{"type": "text", "text": message.content}],
                    }
                )
        return steps

    def _parse(self, data: dict[str, Any], *, retain_raw: bool) -> ModelResponse:
        text: list[str] = []
        calls: list[ToolCall] = []
        for step in data.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            step_type = step.get("type")
            if step_type == "model_output":
                for block in step.get("content", []) or []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text.append(str(block.get("text") or ""))
            elif step_type == "function_call":
                arguments = step.get("arguments") or {}
                calls.append(
                    ToolCall(
                        id=str(step.get("id") or ""),
                        name=str(step.get("name") or ""),
                        arguments=dict(arguments) if isinstance(arguments, dict) else _parse_arguments(str(arguments)),
                        raw_arguments=(
                            json_dumps(arguments) if not isinstance(arguments, str) else arguments
                        ),
                    )
                )
        usage_data = data.get("usage") or {}
        usage = self.price_usage(
            Usage(
                input_tokens=int(usage_data.get("total_input_tokens", 0) or 0),
                output_tokens=int(usage_data.get("total_output_tokens", 0) or 0),
                cached_input_tokens=int(usage_data.get("total_cached_tokens", 0) or 0),
                reasoning_tokens=int(usage_data.get("total_thought_tokens", 0) or 0),
                requests=1,
            )
        )
        return ModelResponse(
            text="".join(text),
            tool_calls=calls,
            usage=usage,
            stop_reason=data.get("status"),
            response_id=data.get("id"),
            model=data.get("model"),
            raw=data if retain_raw else None,
        )


def _parse_arguments(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_raw": raw}
    return value if isinstance(value, dict) else {"value": value}


def _partial_call(item: dict[str, Any]) -> ToolCall:
    raw = str(item.get("arguments") or "{}")
    return ToolCall(
        id=str(item.get("id") or ""),
        name=str(item.get("name") or ""),
        arguments=_parse_arguments(raw),
        raw_arguments=raw,
    )
