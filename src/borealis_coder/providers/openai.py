"""OpenAI Responses API and OpenAI-compatible Chat Completions adapters."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

from ..config import ProviderConfig
from ..errors import ProviderError
from ..models import ContinuationState, ModelResponse, ProviderRequest, Role, ToolCall, Usage
from ..util import json_dumps
from .base import Provider, ProviderStreamEvent
from .http import HttpClient


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, config: ProviderConfig, api_key: str = "") -> None:
        super().__init__(config, api_key)
        self.http = HttpClient(timeout_seconds=config.timeout_seconds)

    @property
    def api_style(self) -> str:
        return self.config.api_style or "responses"

    async def complete(self, request: ProviderRequest) -> ModelResponse:
        if self.api_style in {"chat", "chat_completions"}:
            return await self.with_retries(lambda: self._complete_chat(request))
        return await self.with_retries(lambda: self._complete_responses(request))

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderStreamEvent]:
        if self.api_style in {"chat", "chat_completions"}:
            async for event in self._stream_chat(request):
                yield event
        else:
            async for event in self._stream_responses(request):
                yield event

    async def close(self) -> None:
        self.http.close()

    def _headers(self) -> dict[str, str]:
        headers = dict(self.config.headers)
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        return headers

    def _responses_payload(
        self, request: ProviderRequest, *, stream: bool = False
    ) -> dict[str, Any]:
        input_items = self._responses_input(request)
        payload: dict[str, Any] = {
            "model": request.model,
            "instructions": request.system,
            "input": input_items,
            "tools": [self._responses_tool(tool) for tool in request.tools],
            "max_output_tokens": request.max_output_tokens,
            "parallel_tool_calls": request.parallel_tool_calls,
            "store": False,
            "stream": stream,
        }
        prompt_cache_key = str(request.metadata.get("prompt_cache_key") or "").strip()
        if request.metadata.get("prompt_cache_enabled", True) and prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key
        system_blocks = self._explicit_cache_system_blocks(request)
        if system_blocks is not None:
            stable, dynamic = system_blocks
            content = _cache_content_blocks(stable, dynamic, block_type="input_text")
            payload.pop("instructions")
            input_items.insert(
                0,
                {"type": "message", "role": "developer", "content": content},
            )
            payload["prompt_cache_options"] = {"mode": "implicit"}
        if self.name == "openai":
            payload["include"] = ["reasoning.encrypted_content"]
        if request.reasoning_effort:
            payload["reasoning"] = {"effort": request.reasoning_effort}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.response_schema:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "borealis_output",
                    "strict": True,
                    "schema": request.response_schema,
                }
            }
        return payload

    async def _complete_responses(self, request: ProviderRequest) -> ModelResponse:
        url = self.config.base_url.rstrip("/") + "/responses"
        response = await self.http.post_json(
            url, headers=self._headers(), payload=self._responses_payload(request)
        )
        if not isinstance(response.data, dict):
            raise ProviderError("OpenAI returned a non-object response")
        return self._parse_responses(response.data, retain_raw=True)

    async def _stream_responses(
        self, request: ProviderRequest
    ) -> AsyncIterator[ProviderStreamEvent]:
        url = self.config.base_url.rstrip("/") + "/responses"
        payload = self._responses_payload(request, stream=True)
        text_parts: list[str] = []
        calls: dict[str, dict[str, Any]] = {}
        call_aliases: dict[str, str] = {}
        final_data: dict[str, Any] | None = None
        async for item in self.http.stream_sse(url, headers=self._headers(), payload=payload):
            if item.data == "[DONE]":
                continue
            try:
                data = json.loads(item.data)
            except json.JSONDecodeError:
                continue
            event_type = str(data.get("type") or item.event)
            if event_type == "response.output_text.delta":
                delta = str(data.get("delta") or "")
                text_parts.append(delta)
                yield ProviderStreamEvent(type="text_delta", text=delta)
            elif event_type in {"response.output_item.added", "response.output_item.done"}:
                output = data.get("item") or {}
                if output.get("type") == "function_call":
                    item_id = str(output.get("id") or "")
                    call_id = str(output.get("call_id") or "")
                    key = call_id or item_id
                    if item_id:
                        call_aliases[item_id] = key
                    if call_id:
                        call_aliases[call_id] = key
                    call = calls.setdefault(
                        key,
                        {"id": key, "name": "", "arguments": ""},
                    )
                    if item_id and item_id != key and item_id in calls:
                        pending = calls.pop(item_id)
                        call["arguments"] = pending.get("arguments") or call["arguments"]
                    call["name"] = str(output.get("name") or call["name"])
                    if output.get("arguments"):
                        call["arguments"] = str(output["arguments"])
            elif event_type == "response.function_call_arguments.delta":
                raw_key = str(data.get("call_id") or data.get("item_id") or "")
                key = call_aliases.get(raw_key, raw_key)
                call = calls.setdefault(key, {"id": key, "name": "", "arguments": ""})
                call["arguments"] += str(data.get("delta") or "")
                yield ProviderStreamEvent(
                    type="tool_call_delta",
                    data={
                        "id": key,
                        "name": call.get("name", ""),
                        "delta": data.get("delta", ""),
                    },
                )
            elif event_type in {"response.completed", "response.done"}:
                final_data = (
                    data.get("response") if isinstance(data.get("response"), dict) else data
                )
        if final_data:
            result = self._parse_responses(final_data, retain_raw=True)
            if not result.text:
                result.text = "".join(text_parts)
            final_calls = {call.id: call for call in result.tool_calls}
            for call_id, partial_data in calls.items():
                final_call = final_calls.get(call_id)
                if final_call is None:
                    continue
                partial_call = self._call_from_partial(partial_data)
                if not final_call.name:
                    final_call.name = partial_call.name
                if final_call.raw_arguments in {
                    None,
                    "",
                    "{}",
                } and partial_call.raw_arguments not in {None, "", "{}"}:
                    final_call.arguments = partial_call.arguments
                    final_call.raw_arguments = partial_call.raw_arguments
            result.tool_calls.extend(
                self._call_from_partial(item)
                for item in calls.values()
                if str(item.get("id") or "") not in final_calls
            )
        else:
            result_calls = [self._call_from_partial(item) for item in calls.values()]
            result = ModelResponse(text="".join(text_parts), tool_calls=result_calls)
        yield ProviderStreamEvent(type="completed", response=result)

    def _responses_input(self, request: ProviderRequest) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == Role.ASSISTANT:
                state = ContinuationState.from_metadata(
                    message.metadata.get("continuation_state"),
                    provider=self._continuation_provider(request),
                    model=request.model,
                    kind="openai.responses.reasoning",
                )
                if state is not None:
                    output.extend(state.items)
            if message.role == Role.USER:
                output.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": message.content}],
                    }
                )
            elif message.role == Role.ASSISTANT:
                if message.content:
                    output.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": message.content}],
                        }
                    )
                for call in message.tool_calls:
                    output.append(
                        {
                            "type": "function_call",
                            "call_id": call.id,
                            "name": call.name,
                            "arguments": call.raw_arguments or json_dumps(call.arguments),
                        }
                    )
            elif message.role == Role.TOOL:
                output.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": message.content,
                    }
                )
        return output

    @staticmethod
    def _responses_tool(tool: dict[str, Any]) -> dict[str, Any]:
        schema = dict(tool.get("parameters") or {"type": "object", "properties": {}})
        return {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": schema,
            "strict": _strict_schema_compatible(schema),
        }

    def _parse_responses(self, data: dict[str, Any], *, retain_raw: bool) -> ModelResponse:
        text: list[str] = []
        calls: list[ToolCall] = []
        for item in data.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                for block in item.get("content", []) or []:
                    if isinstance(block, dict) and block.get("type") in {
                        "output_text",
                        "text",
                    }:
                        text.append(str(block.get("text") or ""))
            elif item_type in {"function_call", "custom_tool_call"}:
                raw_arguments = item.get("arguments") or item.get("input") or "{}"
                calls.append(
                    ToolCall(
                        id=str(item.get("call_id") or item.get("id") or ""),
                        name=str(item.get("name") or ""),
                        arguments=_parse_arguments(raw_arguments),
                        raw_arguments=(
                            raw_arguments
                            if isinstance(raw_arguments, str)
                            else json_dumps(raw_arguments)
                        ),
                    )
                )
        usage_data = data.get("usage") or {}
        usage = self._usage_from_responses(usage_data)
        raw: dict[str, Any] | None = data if retain_raw else None
        state = _extract_encrypted_reasoning_state(data)
        return ModelResponse(
            text="".join(text),
            tool_calls=calls,
            usage=usage,
            stop_reason=data.get("status") or data.get("incomplete_details"),
            response_id=data.get("id"),
            model=data.get("model"),
            raw=raw,
            continuation_state=(
                ContinuationState(kind="openai.responses.reasoning", items=state) if state else None
            ),
        )

    def _chat_payload(self, request: ProviderRequest, *, stream: bool = False) -> dict[str, Any]:
        system_blocks = self._explicit_cache_system_blocks(request)
        if system_blocks is None:
            system_content: str | list[dict[str, Any]] = request.system
        else:
            stable, dynamic = system_blocks
            system_content = _cache_content_blocks(stable, dynamic, block_type="text")
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]
        for message in request.messages:
            if message.role in {Role.USER, Role.SYSTEM}:
                messages.append({"role": message.role.value, "content": message.content})
            elif message.role == Role.ASSISTANT:
                item: dict[str, Any] = {"role": "assistant", "content": message.content or None}
                if message.tool_calls:
                    item["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.raw_arguments or json_dumps(call.arguments),
                            },
                        }
                        for call in message.tool_calls
                    ]
                messages.append(item)
            elif message.role == Role.TOOL:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.content,
                    }
                )
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters") or {},
                        "strict": _strict_schema_compatible(tool.get("parameters") or {}),
                    },
                }
                for tool in request.tools
            ],
            "max_completion_tokens": request.max_output_tokens,
            "parallel_tool_calls": request.parallel_tool_calls,
            "stream": stream,
        }
        prompt_cache_key = str(request.metadata.get("prompt_cache_key") or "").strip()
        if (
            self.name == "openai"
            and request.metadata.get("prompt_cache_enabled", True)
            and prompt_cache_key
        ):
            payload["prompt_cache_key"] = prompt_cache_key
        if system_blocks is not None:
            payload["prompt_cache_options"] = {"mode": "implicit"}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.reasoning_effort:
            payload["reasoning_effort"] = request.reasoning_effort
        if request.response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "borealis_output",
                    "strict": True,
                    "schema": request.response_schema,
                },
            }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def _complete_chat(self, request: ProviderRequest) -> ModelResponse:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        response = await self.http.post_json(
            url, headers=self._headers(), payload=self._chat_payload(request)
        )
        if not isinstance(response.data, dict):
            raise ProviderError("OpenAI-compatible provider returned a non-object response")
        return self._parse_chat(response.data, retain_raw=True)

    async def _stream_chat(self, request: ProviderRequest) -> AsyncIterator[ProviderStreamEvent]:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = self._chat_payload(request, stream=True)
        text_parts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        usage = Usage()
        model: str | None = None
        response_id: str | None = None
        finish_reason: str | None = None
        async for item in self.http.stream_sse(url, headers=self._headers(), payload=payload):
            if item.data == "[DONE]":
                continue
            try:
                data = json.loads(item.data)
            except json.JSONDecodeError:
                continue
            model = data.get("model") or model
            response_id = data.get("id") or response_id
            if data.get("usage"):
                usage_data = data["usage"]
                usage = self._usage_from_chat(usage_data)
            for choice in data.get("choices", []) or []:
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    text = str(delta["content"])
                    text_parts.append(text)
                    yield ProviderStreamEvent(type="text_delta", text=text)
                for call_delta in delta.get("tool_calls", []) or []:
                    index = int(call_delta.get("index", 0))
                    call = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    call["id"] = call_delta.get("id") or call["id"]
                    function = call_delta.get("function") or {}
                    call["name"] = function.get("name") or call["name"]
                    call["arguments"] += str(function.get("arguments") or "")
                    yield ProviderStreamEvent(
                        type="tool_call_delta",
                        data={
                            "index": index,
                            "id": call["id"],
                            "name": call["name"],
                            "delta": function.get("arguments", ""),
                        },
                    )
        result = ModelResponse(
            text="".join(text_parts),
            tool_calls=[self._call_from_partial(item) for _, item in sorted(calls.items())],
            usage=usage,
            stop_reason=finish_reason,
            response_id=response_id,
            model=model,
        )
        yield ProviderStreamEvent(type="completed", response=result)

    def _parse_chat(self, data: dict[str, Any], *, retain_raw: bool) -> ModelResponse:
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("OpenAI-compatible response contained no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        calls: list[ToolCall] = []
        for item in message.get("tool_calls", []) or []:
            function = item.get("function") or {}
            raw_arguments = function.get("arguments") or "{}"
            calls.append(
                ToolCall(
                    id=str(item.get("id") or ""),
                    name=str(function.get("name") or ""),
                    arguments=_parse_arguments(raw_arguments),
                    raw_arguments=str(raw_arguments),
                )
            )
        usage_data = data.get("usage") or {}
        usage = self._usage_from_chat(usage_data)
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                str(item.get("text") or "") for item in content if isinstance(item, dict)
            )
        return ModelResponse(
            text=str(content),
            tool_calls=calls,
            usage=usage,
            stop_reason=choice.get("finish_reason"),
            response_id=data.get("id"),
            model=data.get("model"),
            raw=data if retain_raw else None,
        )

    def _usage_from_responses(self, usage_data: dict[str, Any]) -> Usage:
        details = usage_data.get("input_tokens_details") or {}
        output_details = usage_data.get("output_tokens_details") or {}
        return self.price_usage(
            Usage(
                input_tokens=int(usage_data.get("input_tokens", 0) or 0),
                output_tokens=int(usage_data.get("output_tokens", 0) or 0),
                cached_input_tokens=int(details.get("cached_tokens", 0) or 0),
                cache_write_tokens=int(details.get("cache_write_tokens", 0) or 0),
                reasoning_tokens=int(output_details.get("reasoning_tokens", 0) or 0),
                requests=1,
            ),
            cache_write_multiplier=1.25 if self.name == "openai" else 1.0,
        )

    def _usage_from_chat(self, usage_data: dict[str, Any]) -> Usage:
        details = usage_data.get("prompt_tokens_details") or {}
        output_details = usage_data.get("completion_tokens_details") or {}
        return self.price_usage(
            Usage(
                input_tokens=int(usage_data.get("prompt_tokens", 0) or 0),
                output_tokens=int(usage_data.get("completion_tokens", 0) or 0),
                cached_input_tokens=int(details.get("cached_tokens", 0) or 0),
                cache_write_tokens=int(details.get("cache_write_tokens", 0) or 0),
                reasoning_tokens=int(output_details.get("reasoning_tokens", 0) or 0),
                requests=1,
            ),
            cache_write_multiplier=1.25 if self.name == "openai" else 1.0,
        )

    def _explicit_cache_system_blocks(
        self,
        request: ProviderRequest,
    ) -> tuple[list[str], list[str]] | None:
        if (
            self.name != "openai"
            or not request.metadata.get("prompt_cache_enabled", True)
            or not _supports_explicit_cache_breakpoints(request.model)
        ):
            return None
        blocks = request.metadata.get("system_blocks")
        if not isinstance(blocks, list):
            return None
        stable: list[str] = []
        dynamic: list[str] = []
        found_dynamic = False
        all_text: list[str] = []
        for block in blocks:
            if not isinstance(block, dict) or not isinstance(block.get("text"), str):
                return None
            text = block["text"]
            all_text.append(text)
            if block.get("cacheable") and not found_dynamic:
                stable.append(text)
            else:
                found_dynamic = True
                dynamic.append(text)
        if not stable or "\n\n".join(all_text) != request.system:
            return None
        return stable, dynamic

    @staticmethod
    def _call_from_partial(item: dict[str, Any]) -> ToolCall:
        raw = str(item.get("arguments") or "{}")
        return ToolCall(
            id=str(item.get("id") or ""),
            name=str(item.get("name") or ""),
            arguments=_parse_arguments(raw),
            raw_arguments=raw,
        )


class OpenAICompatibleProvider(OpenAIProvider):
    name = "openai_compatible"

    @property
    def api_style(self) -> str:
        return self.config.api_style or "chat"


def _cache_content_blocks(
    stable: list[str],
    dynamic: list[str],
    *,
    block_type: str,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for index, text in enumerate([*stable, *dynamic]):
        block: dict[str, Any] = {
            "type": block_type,
            "text": text if index == 0 else "\n\n" + text,
        }
        if index < len(stable) and index < 4:
            block["prompt_cache_breakpoint"] = {"mode": "explicit"}
        content.append(block)
    return content


def _supports_explicit_cache_breakpoints(model: str) -> bool:
    normalized = model.rsplit("/", 1)[-1].lower()
    match = re.match(r"^gpt-(\d+)(?:\.(\d+))?(?:-|$)", normalized)
    if match is None:
        return False
    return (int(match.group(1)), int(match.group(2) or 0)) >= (5, 6)


def _extract_encrypted_reasoning_state(data: dict[str, Any]) -> list[dict[str, Any]]:
    state: list[dict[str, Any]] = []
    for item in data.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        encrypted = item.get("encrypted_content")
        if not isinstance(encrypted, str) or not encrypted:
            continue
        value: dict[str, Any] = {
            "type": "reasoning",
            "encrypted_content": encrypted,
            "summary": [],
        }
        if item.get("id"):
            value["id"] = str(item["id"])
        state.append(value)
    return state


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or value == "":
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {"_raw": str(value)}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _strict_schema_compatible(schema: dict[str, Any]) -> bool:
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        return False
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    if set(properties) != required:
        return False
    return all(
        _strict_schema_compatible(value)
        if isinstance(value, dict) and value.get("type") == "object"
        else True
        for value in properties.values()
    )
