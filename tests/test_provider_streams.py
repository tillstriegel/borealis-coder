from __future__ import annotations

import json
import unittest
from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from borealis_coder.config import ProviderConfig
from borealis_coder.errors import (
    ProviderAuthenticationError,
    ProviderContextOverflowError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from borealis_coder.models import Message, ModelResponse, ProviderRequest, Role, ToolCall, Usage
from borealis_coder.providers.anthropic import AnthropicProvider
from borealis_coder.providers.anthropic import _parse_arguments as anthropic_args
from borealis_coder.providers.base import Provider, classify_provider_error
from borealis_coder.providers.gemini import GeminiProvider
from borealis_coder.providers.gemini import _parse_arguments as gemini_args
from borealis_coder.providers.http import HttpResponse, SSEEvent
from borealis_coder.providers.openai import (
    OpenAICompatibleProvider,
    OpenAIProvider,
    _strict_schema_compatible,
)
from borealis_coder.providers.openai import (
    _parse_arguments as openai_args,
)
from borealis_coder.tools.base import object_schema


class FakeHttp:
    def __init__(self, *, events: list[SSEEvent] | None = None, data: object | None = None) -> None:
        self.events = events or []
        self.data = data if data is not None else {}
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def stream_sse(self, url: str, **kwargs: object) -> AsyncIterator[SSEEvent]:
        self.calls.append((url, dict(kwargs)))
        for event in self.events:
            yield event

    async def post_json(self, url: str, **kwargs: object) -> HttpResponse:
        self.calls.append((url, dict(kwargs)))
        return HttpResponse(200, {"content-type": "application/json"}, self.data, b"{}")


class DummyProvider(Provider):
    async def complete(self, request: ProviderRequest) -> ModelResponse:
        return ModelResponse(text="ok", usage=Usage(input_tokens=1, output_tokens=2))


class ProviderStreamTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        schema = object_schema({"path": {"type": "string"}})
        self.tool = {"name": "read_file", "description": "Read", "parameters": schema}
        self.messages = [
            Message(role=Role.SYSTEM, content="ignored provider-system"),
            Message(role=Role.USER, content="read it"),
            Message(
                role=Role.ASSISTANT,
                content="checking",
                tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a"})],
            ),
            Message(
                role=Role.TOOL,
                content="content",
                tool_call_id="c1",
                tool_name="read_file",
                is_error=False,
            ),
        ]
        self.request = ProviderRequest(
            model="model",
            system="system",
            messages=self.messages,
            tools=[self.tool],
            temperature=0.2,
            reasoning_effort="high",
            parallel_tool_calls=False,
            response_schema=object_schema({"answer": {"type": "string"}}),
        )

    async def test_base_stream_retry_pricing_and_error_classification(self) -> None:
        config = ProviderConfig(
            max_retries=2,
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
            input_cost_per_million=2,
            cached_input_cost_per_million=1,
            output_cost_per_million=4,
        )
        provider = DummyProvider(config)
        events = [item async for item in provider.stream(self.request)]
        self.assertEqual([item.type for item in events], ["text_delta", "completed"])

        attempts = 0

        async def eventually() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ProviderUnavailableError("temporary", retryable=True)
            return "done"

        with patch("borealis_coder.providers.base.random.uniform", return_value=1.0), patch(
            "borealis_coder.providers.base.asyncio.sleep", new=AsyncMock()
        ):
            self.assertEqual(await provider.with_retries(eventually), "done")
        self.assertEqual(attempts, 3)

        usage = provider.price_usage(
            Usage(input_tokens=1_000_000, cached_input_tokens=250_000, output_tokens=500_000)
        )
        self.assertAlmostEqual(usage.cost_usd, 3.75)
        self.assertIsInstance(classify_provider_error(401, "bad"), ProviderAuthenticationError)
        self.assertIsInstance(classify_provider_error(429, "slow"), ProviderRateLimitError)
        self.assertIsInstance(classify_provider_error(503, "down"), ProviderUnavailableError)
        self.assertIsInstance(
            classify_provider_error(400, "maximum context window exceeded"),
            ProviderContextOverflowError,
        )
        self.assertIs(type(classify_provider_error(400, "bad request")), ProviderError)

        async def fatal() -> str:
            raise ProviderAuthenticationError("no")

        with self.assertRaises(ProviderAuthenticationError):
            await provider.with_retries(fatal)

        async def os_failure() -> str:
            raise OSError("socket")

        with (
            patch(
                "borealis_coder.providers.base.asyncio.sleep",
                new=AsyncMock(),
            ),
            self.assertRaises(ProviderUnavailableError),
        ):
            await provider.with_retries(os_failure)

    async def test_openai_responses_stream_complete_and_partial(self) -> None:
        provider = OpenAIProvider(
            ProviderConfig(
                type="openai",
                base_url="https://example.test/v1",
                api_style="responses",
                input_cost_per_million=1,
                output_cost_per_million=2,
            ),
            "secret",
        )
        events = [
            SSEEvent("message", "not-json"),
            SSEEvent("message", json.dumps({"type": "response.output_text.delta", "delta": "hel"})),
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "type": "response.output_item.added",
                        "item": {
                            "type": "function_call",
                            "call_id": "call2",
                            "name": "read_file",
                            "arguments": "",
                        },
                    }
                ),
            ),
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "type": "response.function_call_arguments.delta",
                        "call_id": "call2",
                        "delta": '{"path":"b"}',
                    }
                ),
            ),
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "resp",
                            "model": "model",
                            "status": "completed",
                            "output": [
                                {
                                    "type": "message",
                                    "content": [{"type": "output_text", "text": "hello"}],
                                },
                                {
                                    "type": "function_call",
                                    "call_id": "call2",
                                    "name": "read_file",
                                    "arguments": '{"path":"b"}',
                                },
                            ],
                            "usage": {
                                "input_tokens": 10,
                                "output_tokens": 3,
                                "input_tokens_details": {"cached_tokens": 2},
                                "output_tokens_details": {"reasoning_tokens": 1},
                            },
                        },
                    }
                ),
            ),
            SSEEvent("message", "[DONE]"),
        ]
        provider.http = FakeHttp(events=events)  # type: ignore[assignment]
        streamed = [item async for item in provider.stream(self.request)]
        self.assertEqual([item.type for item in streamed], ["text_delta", "tool_call_delta", "completed"])
        self.assertEqual(streamed[1].data["name"], "read_file")
        final = streamed[-1].response
        self.assertIsNotNone(final)
        assert final is not None
        self.assertEqual(final.text, "hello")
        self.assertEqual(final.tool_calls[0].arguments, {"path": "b"})
        self.assertEqual(final.usage.reasoning_tokens, 1)
        self.assertIn("Authorization", provider._headers())

        sparse_completion_events = [
            SSEEvent(
                "message",
                json.dumps({"type": "response.output_text.delta", "delta": "sparse"}),
            ),
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "type": "response.output_item.added",
                        "item": {
                            "type": "function_call",
                            "id": "sparse-item",
                            "call_id": "sparse-call",
                            "name": "read_file",
                            "arguments": "",
                        },
                    }
                ),
            ),
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": "sparse-item",
                        "delta": '{"path":"sparse.txt"}',
                    }
                ),
            ),
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "sparse-response",
                            "status": "completed",
                            "usage": {"input_tokens": 5, "output_tokens": 2},
                        },
                    }
                ),
            ),
        ]
        provider.http = FakeHttp(events=sparse_completion_events)  # type: ignore[assignment]
        sparse = [item async for item in provider.stream(self.request)][-1].response
        assert sparse is not None
        self.assertEqual(sparse.text, "sparse")
        self.assertEqual(len(sparse.tool_calls), 1)
        self.assertEqual(sparse.tool_calls[0].id, "sparse-call")
        self.assertEqual(sparse.tool_calls[0].name, "read_file")
        self.assertEqual(sparse.tool_calls[0].arguments, {"path": "sparse.txt"})

        partial_events = [
            SSEEvent("message", json.dumps({"type": "response.output_text.delta", "delta": "partial"})),
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": "x",
                        "delta": "not-json",
                    }
                ),
            ),
        ]
        provider.http = FakeHttp(events=partial_events)  # type: ignore[assignment]
        partial = [item async for item in provider.stream(self.request)][-1].response
        assert partial is not None
        self.assertEqual(partial.text, "partial")
        self.assertEqual(partial.tool_calls[0].arguments, {"_raw": "not-json"})

    async def test_openai_chat_stream_and_complete_paths(self) -> None:
        provider = OpenAICompatibleProvider(
            ProviderConfig(type="openai_compatible", base_url="http://localhost:1234/v1"), ""
        )
        self.assertEqual(provider.api_style, "chat")
        chunks = [
            SSEEvent("message", "garbage"),
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "id": "chat1",
                        "model": "local",
                        "choices": [
                            {
                                "finish_reason": None,
                                "delta": {
                                    "content": "hi ",
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "tc",
                                            "function": {
                                                "name": "read_file",
                                                "arguments": '{"path":',
                                            },
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ),
            ),
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "choices": [
                            {
                                "finish_reason": "tool_calls",
                                "delta": {
                                    "content": "there",
                                    "tool_calls": [
                                        {"index": 0, "function": {"arguments": '"a"}'}}
                                    ],
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 8,
                            "completion_tokens": 2,
                            "prompt_tokens_details": {"cached_tokens": 3},
                        },
                    }
                ),
            ),
            SSEEvent("message", "[DONE]"),
        ]
        cast(Any, provider).http = FakeHttp(events=chunks)
        streamed = [item async for item in provider.stream(self.request)]
        final = streamed[-1].response
        assert final is not None
        self.assertEqual(final.text, "hi there")
        self.assertEqual(final.tool_calls[0].arguments, {"path": "a"})
        self.assertEqual(final.stop_reason, "tool_calls")
        self.assertEqual(final.response_id, "chat1")
        self.assertEqual(final.usage.cached_input_tokens, 3)

        cast(Any, provider).http = FakeHttp(
            data={
                "id": "c2",
                "model": "m",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": [{"text": "ok"}],
                            "tool_calls": [
                                {
                                    "id": "t",
                                    "function": {"name": "read_file", "arguments": "[]"},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )  # type: ignore[assignment]
        completed = await provider.complete(self.request)
        self.assertEqual(completed.text, "ok")
        self.assertEqual(completed.tool_calls[0].arguments, {"value": []})
        with self.assertRaises(ProviderError):
            provider._parse_chat({}, retain_raw=False)

        provider.http = FakeHttp(data=["not", "object"])  # type: ignore[assignment]
        with self.assertRaises(ProviderError):
            await provider.complete(self.request)

    async def test_openai_payload_helpers_and_response_complete_error(self) -> None:
        provider = OpenAIProvider(
            ProviderConfig(type="openai", base_url="https://x", api_style="responses"), ""
        )
        payload = provider._responses_payload(self.request, stream=True)
        self.assertEqual(payload["reasoning"], {"effort": "high"})
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertFalse(payload["parallel_tool_calls"])
        chat = provider._chat_payload(self.request, stream=True)
        self.assertTrue(chat["stream_options"]["include_usage"])
        self.assertEqual(chat["reasoning_effort"], "high")
        self.assertEqual(chat["response_format"]["type"], "json_schema")
        self.assertTrue(_strict_schema_compatible(self.tool["parameters"]))
        self.assertFalse(_strict_schema_compatible({"type": "object", "properties": {}}))
        self.assertFalse(
            _strict_schema_compatible(
                {
                    "type": "object",
                    "properties": {"nested": {"type": "object", "properties": {}}},
                    "required": ["nested"],
                    "additionalProperties": False,
                }
            )
        )
        self.assertEqual(openai_args(None), {})
        self.assertEqual(openai_args({"a": 1}), {"a": 1})
        self.assertEqual(openai_args("1"), {"value": 1})

        provider.http = FakeHttp(data="bad")  # type: ignore[assignment]
        with self.assertRaises(ProviderError):
            await provider.complete(self.request)

    async def test_anthropic_stream_complete_messages_and_error(self) -> None:
        provider = AnthropicProvider(
            ProviderConfig(
                type="anthropic",
                base_url="https://anthropic.test/v1",
                input_cost_per_million=1,
                output_cost_per_million=2,
            ),
            "key",
        )
        events = [
            SSEEvent("message", "bad-json"),
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "a1",
                            "model": "claude",
                            "usage": {
                                "input_tokens": 10,
                                "cache_read_input_tokens": 2,
                                "cache_creation_input_tokens": 1,
                            },
                        },
                    }
                ),
            ),
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "type": "content_block_start",
                        "index": 1,
                        "content_block": {
                            "type": "tool_use",
                            "id": "tool1",
                            "name": "read_file",
                            "input": {},
                        },
                    }
                ),
            ),
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "checking"},
                    }
                ),
            ),
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "input_json_delta", "partial_json": '{"path":"a"}'},
                    }
                ),
            ),
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "tool_use"},
                        "usage": {"output_tokens": 4},
                    }
                ),
            ),
        ]
        cast(Any, provider).http = FakeHttp(events=events)
        streamed = [item async for item in provider.stream(self.request)]
        tool_delta = next(item for item in streamed if item.type == "tool_call_delta")
        self.assertEqual(tool_delta.data["name"], "read_file")
        final = streamed[-1].response
        assert final is not None
        self.assertEqual(final.text, "checking")
        self.assertEqual(final.tool_calls[0].arguments, {"path": "a"})
        self.assertEqual(final.response_id, "a1")
        self.assertEqual(final.usage.cache_write_tokens, 1)
        self.assertEqual(provider._headers()["x-api-key"], "key")

        converted = provider._messages(self.messages)
        self.assertNotIn("system", [item["role"] for item in converted])
        self.assertEqual(converted[-1]["role"], "user")
        self.assertEqual(anthropic_args("[1]"), {"value": [1]})
        self.assertEqual(anthropic_args("bad"), {"_raw": "bad"})

        cast(Any, provider).http = FakeHttp(
            data={
                "id": "a2",
                "model": "claude",
                "stop_reason": "end_turn",
                "content": ["skip", {"type": "tool_use", "id": "x", "name": "t", "input": 3}],
                "usage": {},
            }
        )  # type: ignore[assignment]
        parsed = await provider.complete(self.request)
        self.assertEqual(parsed.tool_calls[0].arguments, {"value": 3})

        provider.http = FakeHttp(events=[SSEEvent("message", json.dumps({"type": "error", "error": {"message": "boom"}}))])  # type: ignore[assignment]
        with self.assertRaisesRegex(ProviderError, "boom"):
            _ = [item async for item in provider.stream(self.request)]
        provider.http = FakeHttp(data="bad")  # type: ignore[assignment]
        with self.assertRaises(ProviderError):
            await provider.complete(self.request)

    async def test_gemini_stream_complete_partial_and_helpers(self) -> None:
        provider = GeminiProvider(
            ProviderConfig(type="gemini", base_url="https://gemini.test/v1beta"), "key"
        )
        events = [
            SSEEvent("message", "not-json"),
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "event_type": "step.start",
                        "index": 1,
                        "step": {
                            "type": "function_call",
                            "id": "gcall",
                            "name": "read_file",
                            "arguments": "",
                        },
                    }
                ),
            ),
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "event_type": "step.delta",
                        "index": 0,
                        "delta": {"type": "text", "text": "yes"},
                    }
                ),
            ),
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "event_type": "step.delta",
                        "index": 1,
                        "delta": {"type": "arguments", "partial_arguments": '{"path":"z"}'},
                    }
                ),
            ),
        ]
        provider.http = FakeHttp(events=events)  # type: ignore[assignment]
        partial_events = [item async for item in provider.stream(self.request)]
        tool_delta = next(item for item in partial_events if item.type == "tool_call_delta")
        self.assertEqual(tool_delta.data["name"], "read_file")
        partial = partial_events[-1].response
        assert partial is not None
        self.assertEqual(partial.text, "yes")
        self.assertEqual(partial.tool_calls[0].arguments, {"path": "z"})

        final_events = [
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "type": "interaction.completed",
                        "interaction": {
                            "id": "i1",
                            "model": "gemini",
                            "status": "completed",
                            "steps": [
                                {"type": "model_output", "content": [{"type": "text", "text": "done"}]},
                                {"type": "function_call", "id": "f", "name": "read_file", "arguments": "bad"},
                            ],
                            "usage": {
                                "total_input_tokens": 4,
                                "total_output_tokens": 2,
                                "total_cached_tokens": 1,
                                "total_thought_tokens": 3,
                            },
                        },
                    }
                ),
            )
        ]
        provider.http = FakeHttp(events=final_events)  # type: ignore[assignment]
        completed = [item async for item in provider.stream(self.request)][-1].response
        assert completed is not None
        self.assertEqual(completed.text, "done")
        self.assertEqual(completed.tool_calls[0].arguments, {"_raw": "bad"})
        self.assertEqual(completed.usage.reasoning_tokens, 3)

        payload = provider._payload(self.request, stream=True)
        self.assertEqual(payload["generation_config"]["thinking_level"], "high")
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertIn("alt=sse", provider._url(stream=True))
        self.assertEqual(provider._headers()["x-goog-api-key"], "key")
        self.assertEqual(gemini_args("2"), {"value": 2})
        self.assertEqual(gemini_args("bad"), {"_raw": "bad"})

        provider.http = FakeHttp(data={"steps": [None], "usage": {}})  # type: ignore[assignment]
        response = await provider.complete(self.request)
        self.assertEqual(response.text, "")
        provider.http = FakeHttp(data="bad")  # type: ignore[assignment]
        with self.assertRaises(ProviderError):
            await provider.complete(self.request)


if __name__ == "__main__":
    unittest.main()
