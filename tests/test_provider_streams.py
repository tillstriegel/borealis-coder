from __future__ import annotations

import asyncio
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
from borealis_coder.providers.base import Provider, ProviderStreamEvent, classify_provider_error
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
from borealis_coder.providers.openrouter import OpenRouterProvider
from borealis_coder.tools.base import object_schema


class FakeHttp:
    def __init__(
        self,
        *,
        events: list[SSEEvent] | None = None,
        event_batches: list[list[SSEEvent]] | None = None,
        data: object | None = None,
    ) -> None:
        self.events = events or []
        self.event_batches = event_batches
        self.data = data if data is not None else {}
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def stream_sse(self, url: str, **kwargs: object) -> AsyncIterator[SSEEvent]:
        self.calls.append((url, dict(kwargs)))
        events = self.event_batches.pop(0) if self.event_batches is not None else self.events
        for event in events:
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
            cache_write_input_cost_per_million=2.5,
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

        with (
            patch("borealis_coder.providers.base.random.uniform", return_value=1.0),
            patch("borealis_coder.providers.base.asyncio.sleep", new=AsyncMock()),
        ):
            self.assertEqual(await provider.with_retries(eventually), "done")
        self.assertEqual(attempts, 3)

        usage = provider.price_usage(
            Usage(
                input_tokens=1_000_000,
                cached_input_tokens=250_000,
                cache_write_tokens=100_000,
                output_tokens=500_000,
            )
        )
        self.assertAlmostEqual(usage.cost_usd, 3.8)
        self.assertAlmostEqual(usage.cache_savings_usd, 0.2)
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

    async def test_concurrent_provider_requests_keep_independent_retry_budgets(self) -> None:
        provider = DummyProvider(ProviderConfig(
            max_retries=1, initial_backoff_seconds=0, max_backoff_seconds=0,
        ))
        attempts = {"first": 0, "second": 0}
        both_started = asyncio.Event()

        async def operation(name):
            attempts[name] += 1
            if attempts[name] == 1:
                if all(attempts.values()):
                    both_started.set()
                await both_started.wait()
                raise ProviderUnavailableError("Temporary failure", retryable=True)
            return name

        async def request(name):
            return await provider.with_retries(
                lambda: provider.with_retries(lambda: operation(name))
            )

        result = await asyncio.wait_for(asyncio.gather(request("first"), request("second")), timeout=2)
        self.assertEqual(result, ["first", "second"])
        self.assertEqual(attempts, {"first": 2, "second": 2})

    async def test_child_task_gets_its_own_provider_retry_budget(self) -> None:
        provider = DummyProvider(ProviderConfig(
            max_retries=1, initial_backoff_seconds=0, max_backoff_seconds=0,
        ))
        attempts = 0
        outer_calls = 0

        async def operation():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ProviderUnavailableError("Temporary failure", retryable=True)
            return "done"

        async def outer():
            nonlocal outer_calls
            outer_calls += 1
            return await asyncio.create_task(provider.with_retries(operation))

        self.assertEqual(await provider.with_retries(outer), "done")
        self.assertEqual((outer_calls, attempts), (1, 2))

    async def test_cancelled_provider_request_releases_its_retry_scope(self) -> None:
        provider = DummyProvider(ProviderConfig(
            max_retries=1, initial_backoff_seconds=0, max_backoff_seconds=0,
        ))

        async def cancelled():
            raise asyncio.CancelledError

        with self.assertRaises(asyncio.CancelledError):
            await provider.with_retries(cancelled)
        attempts = 0

        async def recovered():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ProviderUnavailableError("Temporary failure", retryable=True)
            return "done"

        self.assertEqual(await provider.with_retries(recovered), "done")
        self.assertEqual(attempts, 2)

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
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": "reasoning_1",
                        "output_index": 0,
                        "summary_index": 0,
                        "delta": "Checked ",
                    }
                ),
            ),
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": "reasoning_1",
                        "output_index": 0,
                        "summary_index": 0,
                        "delta": "the request.",
                    }
                ),
            ),
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": "reasoning_1",
                        "output_index": 0,
                        "summary_index": 1,
                        "delta": "Prepared the answer.",
                    }
                ),
            ),
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
                                    "type": "reasoning",
                                    "summary": [
                                        {
                                            "type": "summary_text",
                                            "text": "Checked the request.",
                                        },
                                        {
                                            "type": "summary_text",
                                            "text": "Prepared the answer.",
                                        }
                                    ],
                                },
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
        self.assertEqual(
            [item.type for item in streamed],
            [
                "reasoning_summary_delta",
                "reasoning_summary_delta",
                "reasoning_summary_delta",
                "text_delta",
                "tool_call_delta",
                "completed",
            ],
        )
        self.assertEqual(
            [item.text for item in streamed if item.type == "reasoning_summary_delta"],
            ["Checked ", "the request.", "\nPrepared the answer."],
        )
        tool_delta = next(item for item in streamed if item.type == "tool_call_delta")
        self.assertEqual(tool_delta.data["name"], "read_file")
        final = streamed[-1].response
        self.assertIsNotNone(final)
        assert final is not None
        self.assertEqual(final.text, "hello")
        self.assertEqual(
            final.reasoning_summary,
            "Checked the request.\nPrepared the answer.",
        )
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
            SSEEvent(
                "message", json.dumps({"type": "response.output_text.delta", "delta": "partial"})
            ),
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
        self.assertEqual(partial.tool_calls, [])

    async def test_openai_responses_stream_preserves_incomplete_status(self) -> None:
        provider = OpenAIProvider(
            ProviderConfig(
                type="openai",
                base_url="https://example.test/v1",
                api_style="responses",
            ),
            "secret",
        )
        provider.http = FakeHttp(  # type: ignore[assignment]
            events=[
                SSEEvent(
                    "message",
                    json.dumps(
                        {"type": "response.output_text.delta", "delta": "partial answer"}
                    ),
                ),
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "type": "response.incomplete",
                            "response": {
                                "id": "resp-incomplete",
                                "model": "model",
                                "status": "incomplete",
                                "incomplete_details": {"reason": "max_output_tokens"},
                                "output": [],
                                "usage": {"input_tokens": 10, "output_tokens": 5},
                            },
                        }
                    ),
                ),
            ]
        )

        streamed = [item async for item in provider.stream(self.request)]

        final = streamed[-1].response
        assert final is not None
        self.assertEqual(final.text, "partial answer")
        self.assertEqual(final.stop_reason, "incomplete")

    async def test_openai_responses_stream_hides_call_without_terminal_response(self) -> None:
        provider = OpenAIProvider(
            ProviderConfig(
                type="openai",
                base_url="https://example.test/v1",
                api_style="responses",
            ),
            "secret",
        )
        provider.http = FakeHttp(  # type: ignore[assignment]
            events=[
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "type": "response.output_item.done",
                            "item": {
                                "type": "function_call",
                                "id": "item-truncated",
                                "call_id": "call-truncated",
                                "name": "write_file",
                                "arguments": '{"path":"must-not-run.txt","content":"done"}',
                            },
                        }
                    ),
                ),
            ]
        )

        streamed = [item async for item in provider.stream(self.request)]

        final = streamed[-1].response
        assert final is not None
        self.assertEqual(final.stop_reason, "incomplete")
        self.assertEqual(final.tool_calls, [])

    async def test_openai_responses_stream_hides_incomplete_partial_call(self) -> None:
        provider = OpenAIProvider(
            ProviderConfig(
                type="openai",
                base_url="https://example.test/v1",
                api_style="responses",
            ),
            "secret",
        )
        partial_call = {
            "type": "function_call",
            "id": "item-partial",
            "call_id": "call-partial",
            "name": "write_file",
            "arguments": '{"path":"must-not-run.txt","content":"partial"}',
            "status": "in_progress",
        }
        provider.http = FakeHttp(  # type: ignore[assignment]
            events=[
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "type": "response.output_item.added",
                            "item": {
                                **partial_call,
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
                            "call_id": "call-partial",
                            "delta": partial_call["arguments"],
                        }
                    ),
                ),
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "type": "response.incomplete",
                            "response": {
                                "id": "resp-partial-call",
                                "model": "model",
                                "status": "incomplete",
                                "incomplete_details": {
                                    "reason": "max_output_tokens"
                                },
                                "output": [partial_call],
                                "usage": {"input_tokens": 10, "output_tokens": 5},
                            },
                        }
                    ),
                ),
            ]
        )

        streamed = [item async for item in provider.stream(self.request)]

        final = streamed[-1].response
        assert final is not None
        self.assertEqual(final.tool_calls, [])
        assert final.raw is not None
        self.assertEqual(final.raw["output"], [partial_call])

    async def test_openai_responses_stream_preserves_empty_incomplete_state(self) -> None:
        provider = OpenAIProvider(
            ProviderConfig(
                type="openai",
                base_url="https://example.test/v1",
                api_style="responses",
            ),
            "secret",
        )
        provider.http = FakeHttp(  # type: ignore[assignment]
            events=[
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "type": "response.incomplete",
                            "response": {
                                "id": "resp-incomplete-reasoning",
                                "model": "model",
                                "status": "incomplete",
                                "incomplete_details": {"reason": "max_output_tokens"},
                                "output": [
                                    {
                                        "type": "reasoning",
                                        "id": "reasoning_1",
                                        "encrypted_content": "opaque-state",
                                        "summary": [],
                                    }
                                ],
                                "usage": {"input_tokens": 10, "output_tokens": 5},
                            },
                        }
                    ),
                ),
            ]
        )

        streamed = [item async for item in provider.stream(self.request)]

        self.assertEqual([item.type for item in streamed], ["completed"])
        final = streamed[-1].response
        assert final is not None
        self.assertEqual(final.text, "")
        self.assertEqual(final.tool_calls, [])
        self.assertEqual(final.stop_reason, "incomplete")
        assert final.continuation_state is not None
        self.assertEqual(
            final.continuation_state.items,
            [
                {
                    "type": "reasoning",
                    "id": "reasoning_1",
                    "encrypted_content": "opaque-state",
                    "summary": [],
                }
            ],
        )

    async def test_responses_emits_summary_found_only_in_completed_event(self) -> None:
        provider = OpenAIProvider(
            ProviderConfig(type="openai", base_url="https://openai.test/v1"),
            "key",
        )
        cast(Any, provider).http = FakeHttp(
            events=[
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "type": "response.completed",
                            "response": {
                                "id": "summary-response",
                                "status": "completed",
                                "output": [
                                    {
                                        "type": "reasoning",
                                        "summary": [
                                            {
                                                "type": "summary_text",
                                                "text": "Only in the completed response.",
                                            }
                                        ],
                                    },
                                    {
                                        "type": "message",
                                        "content": [
                                            {"type": "output_text", "text": "answer"}
                                        ],
                                    },
                                ],
                            },
                        }
                    ),
                )
            ]
        )

        streamed = [item async for item in provider.stream(self.request)]

        self.assertEqual(
            [item.type for item in streamed],
            ["reasoning_summary_delta", "completed"],
        )
        self.assertEqual(streamed[0].text, "Only in the completed response.")
        assert streamed[-1].response is not None
        self.assertEqual(streamed[-1].response.reasoning_summary, streamed[0].text)

    async def test_responses_discards_summary_only_completed_attempt(self) -> None:
        provider = OpenAIProvider(
            ProviderConfig(type="openai", base_url="https://openai.test/v1"),
            "key",
        )
        cast(Any, provider).http = FakeHttp(
            events=[
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "type": "response.reasoning_summary_text.delta",
                            "item_id": "reasoning_1",
                            "output_index": 0,
                            "summary_index": 0,
                            "delta": "Abandoned summary.",
                        }
                    ),
                ),
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "type": "response.completed",
                            "response": {
                                "id": "summary-only",
                                "status": "completed",
                                "output": [
                                    {
                                        "type": "reasoning",
                                        "summary": [
                                            {
                                                "type": "summary_text",
                                                "text": "Abandoned summary.",
                                            }
                                        ],
                                    }
                                ],
                                "usage": {"input_tokens": 2, "output_tokens": 1},
                            },
                        }
                    ),
                ),
            ]
        )

        streamed = [event async for event in provider.stream(self.request)]

        self.assertEqual([event.type for event in streamed], ["completed"])
        assert streamed[-1].response is not None
        self.assertEqual(streamed[-1].response.reasoning_summary, "")
        self.assertEqual(streamed[-1].response.usage.input_tokens, 2)

    async def test_reasoning_summary_separators_are_not_duplicated(self) -> None:
        responses_provider = OpenAIProvider(
            ProviderConfig(type="openai", base_url="https://openai.test/v1"),
            "key",
        )
        cast(Any, responses_provider).http = FakeHttp(
            events=[
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "type": "response.reasoning_summary_text.delta",
                            "item_id": "reasoning_1",
                            "output_index": 0,
                            "summary_index": 0,
                            "delta": "First.",
                        }
                    ),
                ),
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "type": "response.reasoning_summary_text.delta",
                            "item_id": "reasoning_1",
                            "output_index": 0,
                            "summary_index": 1,
                            "delta": "Second.",
                        }
                    ),
                ),
                SSEEvent(
                    "message",
                    json.dumps({"type": "response.output_text.delta", "delta": "ok"}),
                ),
            ]
        )

        responses = [item async for item in responses_provider.stream(self.request)]

        responses_final = responses[-1].response
        assert responses_final is not None
        self.assertEqual(responses_final.reasoning_summary, "First.\nSecond.")
        self.assertEqual(
            [item.text for item in responses if item.type == "reasoning_summary_delta"],
            ["First.", "\nSecond."],
        )

        chat_provider = OpenAICompatibleProvider(
            ProviderConfig(type="openai_compatible", base_url="https://chat.test/v1"),
            "key",
        )
        cast(Any, chat_provider).http = FakeHttp(
            events=[
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "reasoning_details": [
                                            {
                                                "type": "reasoning.summary",
                                                "summary": "First.",
                                                "id": "summary_1",
                                                "index": 0,
                                            }
                                        ],
                                        "content": "ok",
                                    }
                                }
                            ]
                        }
                    ),
                ),
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "delta": {
                                        "reasoning_details": [
                                            {
                                                "type": "reasoning.summary",
                                                "summary": "Second.",
                                                "id": "summary_2",
                                                "index": 0,
                                            }
                                        ]
                                    },
                                }
                            ],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                        }
                    ),
                ),
            ]
        )

        chat = [item async for item in chat_provider.stream(self.request)]

        chat_final = chat[-1].response
        assert chat_final is not None
        self.assertEqual(chat_final.reasoning_summary, "First.\nSecond.")
        self.assertEqual(
            [item.text for item in chat if item.type == "reasoning_summary_delta"],
            ["First.", "\nSecond."],
        )

    async def test_responses_stream_preserves_refusal_as_visible_text(self) -> None:
        provider = OpenAIProvider(
            ProviderConfig(type="openai", base_url="https://openai.test/v1"),
            "key",
        )
        cast(Any, provider).http = FakeHttp(
            events=[
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "type": "response.refusal.delta",
                            "delta": "I cannot help with that.",
                        }
                    ),
                ),
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "type": "response.completed",
                            "response": {
                                "id": "refusal-response",
                                "status": "completed",
                                "output": [
                                    {
                                        "type": "message",
                                        "content": [
                                            {
                                                "type": "refusal",
                                                "refusal": "I cannot help with that.",
                                            }
                                        ],
                                    }
                                ],
                            },
                        }
                    ),
                ),
            ]
        )

        streamed = [item async for item in provider.stream(self.request)]

        self.assertEqual([item.type for item in streamed], ["text_delta", "completed"])
        assert streamed[-1].response is not None
        self.assertEqual(streamed[-1].response.text, "I cannot help with that.")

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
                                    "reasoning_details": [
                                        {
                                            "type": "reasoning.summary",
                                            "summary": "Checked ",
                                            "id": "summary_1",
                                            "index": 0,
                                        }
                                    ],
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
                                    "reasoning_details": [
                                        {
                                            "type": "reasoning.summary",
                                            "summary": "the request.",
                                            "id": "summary_1",
                                            "index": 0,
                                        }
                                    ],
                                    "content": "there",
                                    "tool_calls": [{"index": 0, "function": {"arguments": '"a"}'}}],
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
        self.assertEqual(
            [item.type for item in streamed],
            [
                "reasoning_summary_delta",
                "text_delta",
                "tool_call_delta",
                "reasoning_summary_delta",
                "text_delta",
                "tool_call_delta",
                "completed",
            ],
        )
        self.assertEqual(
            [item.text for item in streamed if item.type == "reasoning_summary_delta"],
            ["Checked ", "the request."],
        )
        final = streamed[-1].response
        assert final is not None
        self.assertEqual(final.text, "hi there")
        self.assertEqual(final.tool_calls[0].arguments, {"path": "a"})
        self.assertEqual(final.stop_reason, "tool_calls")
        self.assertEqual(final.response_id, "chat1")
        self.assertEqual(final.usage.cached_input_tokens, 3)
        self.assertEqual(final.reasoning_summary, "Checked the request.")

        cast(Any, provider).http = FakeHttp(
            data={
                "id": "c2",
                "model": "m",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": [{"text": "ok"}],
                            "reasoning_details": [
                                {
                                    "type": "reasoning.summary",
                                    "summary": "Prepared the response.",
                                    "id": "summary_2",
                                    "index": 0,
                                }
                            ],
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
        self.assertEqual(completed.reasoning_summary, "Prepared the response.")
        self.assertEqual(completed.tool_calls[0].arguments, {"value": []})
        with self.assertRaises(ProviderError):
            provider._parse_chat({}, retain_raw=False)

        provider.http = FakeHttp(data=["not", "object"])  # type: ignore[assignment]
        with self.assertRaises(ProviderError):
            await provider.complete(self.request)

    async def test_openai_chat_stream_hides_length_truncated_tool_call(self) -> None:
        provider = OpenAICompatibleProvider(
            ProviderConfig(type="openai_compatible", base_url="https://chat.test/v1"),
            "key",
        )
        cast(Any, provider).http = FakeHttp(
            events=[
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "id": "truncated-chat",
                            "choices": [
                                {
                                    "finish_reason": None,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": "truncated-call",
                                                "function": {
                                                    "name": "write_file",
                                                    "arguments": '{"path":"danger.txt"',
                                                },
                                            }
                                        ]
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
                                {"finish_reason": "length", "delta": {}}
                            ]
                        }
                    ),
                ),
                SSEEvent("message", "[DONE]"),
            ]
        )

        streamed = [item async for item in provider.stream(self.request)]

        final = streamed[-1].response
        assert final is not None
        self.assertEqual(final.stop_reason, "length")
        self.assertEqual(final.tool_calls, [])

    async def test_openai_chat_stream_hides_call_without_finish_reason(self) -> None:
        provider = OpenAICompatibleProvider(
            ProviderConfig(type="openai_compatible", base_url="https://chat.test/v1"),
            "key",
        )
        cast(Any, provider).http = FakeHttp(
            events=[
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "id": "truncated-chat",
                            "choices": [
                                {
                                    "finish_reason": None,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": "truncated-call",
                                                "function": {
                                                    "name": "write_file",
                                                    "arguments": (
                                                        '{"path":"must-not-run.txt",'
                                                        '"content":"done"}'
                                                    ),
                                                },
                                            }
                                        ]
                                    },
                                }
                            ],
                        }
                    ),
                ),
            ]
        )

        streamed = [item async for item in provider.stream(self.request)]

        final = streamed[-1].response
        assert final is not None
        self.assertEqual(final.stop_reason, "incomplete")
        self.assertEqual(final.tool_calls, [])

    async def test_chat_reasoning_streams_text_before_response_finishes(self) -> None:
        provider = OpenRouterProvider(
            ProviderConfig(
                type="openrouter",
                base_url="https://openrouter.test/api/v1",
                api_style="chat",
            ),
            "key",
        )

        class GatedHttp:
            def __init__(self) -> None:
                self.release = asyncio.Event()
                self.calls: list[tuple[str, dict[str, object]]] = []

            async def stream_sse(
                self,
                url: str,
                **kwargs: object,
            ) -> AsyncIterator[SSEEvent]:
                self.calls.append((url, dict(kwargs)))
                yield SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "id": "chat1",
                            "choices": [
                                {
                                    "finish_reason": None,
                                    "delta": {
                                        "reasoning_details": [
                                            {
                                                "type": "reasoning.summary",
                                                "summary": "Checked.",
                                                "id": "summary_1",
                                                "index": 0,
                                            }
                                        ],
                                        "content": "Live text.",
                                    },
                                }
                            ],
                        }
                    ),
                )
                await self.release.wait()
                yield SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "choices": [{"finish_reason": "stop", "delta": {}}],
                            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                        }
                    ),
                )
                yield SSEEvent("message", "[DONE]")

        fake = GatedHttp()
        provider.http = fake  # type: ignore[assignment]
        stream = provider.stream(self.request).__aiter__()
        try:
            summary = await asyncio.wait_for(stream.__anext__(), timeout=0.5)
            text = await asyncio.wait_for(stream.__anext__(), timeout=0.5)
            self.assertEqual(summary.type, "reasoning_summary_delta")
            self.assertEqual(summary.text, "Checked.")
            self.assertEqual(text.type, "text_delta")
            self.assertEqual(text.text, "Live text.")
            fake.release.set()
            rest = [event async for event in stream]
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()

        self.assertEqual([event.type for event in rest], ["completed"])
        assert rest[-1].response is not None
        self.assertEqual(rest[-1].response.text, "Live text.")

    async def test_chat_retries_empty_reasoning_response_without_reasoning_controls(self) -> None:
        provider = OpenRouterProvider(
            ProviderConfig(
                type="openrouter",
                base_url="https://openrouter.test/api/v1",
                api_style="chat",
                input_cost_per_million=1,
                output_cost_per_million=2,
            ),
            "key",
        )
        first = [
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "id": "empty",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "delta": {
                                    "reasoning_details": [
                                        {
                                            "type": "reasoning.text",
                                            "text": "Raw reasoning is not a summary.",
                                            "index": 0,
                                        }
                                    ]
                                },
                            }
                        ],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                    }
                ),
            ),
            SSEEvent("message", "[DONE]"),
        ]
        second = [
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "id": "recovered",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "delta": {"content": "Recovered answer."},
                            }
                        ],
                        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
                    }
                ),
            ),
            SSEEvent("message", "[DONE]"),
        ]
        fake = FakeHttp(event_batches=[first, second])
        provider.http = fake  # type: ignore[assignment]

        streamed = [event async for event in provider.stream(self.request)]

        final = streamed[-1].response
        assert final is not None
        self.assertEqual(final.text, "Recovered answer.")
        self.assertEqual(final.usage.input_tokens, 9)
        self.assertEqual(final.usage.output_tokens, 5)
        self.assertEqual(final.usage.requests, 2)
        self.assertAlmostEqual(final.usage.cost_usd, 0.000019)
        self.assertEqual(len(fake.calls), 2)
        first_payload = cast(dict[str, Any], fake.calls[0][1]["payload"])
        second_payload = cast(dict[str, Any], fake.calls[1][1]["payload"])
        self.assertEqual(first_payload["reasoning"], {"effort": "high"})
        self.assertNotIn("reasoning", second_payload)

    async def test_complete_chat_preserves_usage_across_reasoning_disabled_retry(self) -> None:
        provider = OpenRouterProvider(
            ProviderConfig(
                type="openrouter",
                base_url="https://openrouter.test/api/v1",
                api_style="chat",
                max_retries=0,
            ),
            "key",
        )
        first = ModelResponse(usage=Usage(input_tokens=5, output_tokens=3, requests=1))
        recovered = ModelResponse(
            text="Recovered answer.",
            usage=Usage(input_tokens=4, output_tokens=2, requests=1),
        )
        complete_once = AsyncMock(side_effect=[first, recovered])

        with patch.object(provider, "_complete_chat_once", new=complete_once):
            response = await provider.complete(self.request)

        self.assertEqual(response.text, "Recovered answer.")
        self.assertEqual(response.usage.input_tokens, 9)
        self.assertEqual(response.usage.output_tokens, 5)
        self.assertEqual(response.usage.requests, 2)
        self.assertEqual(complete_once.await_count, 2)
        fallback_request = complete_once.await_args_list[1].args[0]
        self.assertIsNone(fallback_request.reasoning_effort)

    async def test_complete_chat_preserves_usage_when_internal_fallback_fails(self) -> None:
        provider = OpenRouterProvider(
            ProviderConfig(
                type="openrouter",
                base_url="https://openrouter.test/api/v1",
                api_style="chat",
                max_retries=1,
            ),
            "key",
        )
        first = ModelResponse(usage=Usage(input_tokens=5, output_tokens=3, requests=1))
        recovered = ModelResponse(
            text="Recovered answer.",
            usage=Usage(input_tokens=4, output_tokens=2, requests=1),
        )
        complete_once = AsyncMock(
            side_effect=[
                first,
                ProviderUnavailableError("fallback down", retryable=True),
                recovered,
            ]
        )

        with patch.object(provider, "_complete_chat_once", new=complete_once):
            response = await provider.complete(self.request)

        self.assertEqual(response.text, "Recovered answer.")
        self.assertEqual(response.usage.input_tokens, 9)
        self.assertEqual(response.usage.output_tokens, 5)
        self.assertEqual(response.usage.requests, 2)
        self.assertEqual(complete_once.await_count, 3)
        self.assertIsNone(complete_once.await_args_list[1].args[0].reasoning_effort)
        self.assertEqual(complete_once.await_args_list[2].args[0].reasoning_effort, "high")

    async def test_chat_stream_error_carries_usage_from_failed_internal_fallback(self) -> None:
        provider = OpenRouterProvider(
            ProviderConfig(
                type="openrouter",
                base_url="https://openrouter.test/api/v1",
                api_style="chat",
                max_retries=0,
            ),
            "key",
        )
        first = [
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "id": "empty",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "delta": {
                                    "reasoning_details": [
                                        {
                                            "type": "reasoning.summary",
                                            "summary": "Billed summary.",
                                            "index": 0,
                                        }
                                    ]
                                },
                            }
                        ],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                    }
                ),
            ),
            SSEEvent("message", "[DONE]"),
        ]
        failed_fallback = [
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "error": {
                            "code": 503,
                            "message": "Fallback provider unavailable.",
                        }
                    }
                ),
            )
        ]
        fake = FakeHttp(event_batches=[first, failed_fallback])
        provider.http = fake  # type: ignore[assignment]

        with self.assertRaises(ProviderUnavailableError) as unavailable:
            _ = [event async for event in provider.stream(self.request)]

        self.assertIsNotNone(unavailable.exception.usage)
        assert unavailable.exception.usage is not None
        self.assertEqual(unavailable.exception.usage.input_tokens, 5)
        self.assertEqual(unavailable.exception.usage.output_tokens, 3)
        self.assertEqual(unavailable.exception.usage.requests, 1)
        self.assertEqual(len(fake.calls), 2)

    async def test_chat_buffers_summary_from_abandoned_attempt(self) -> None:
        provider = OpenRouterProvider(
            ProviderConfig(
                type="openrouter",
                base_url="https://openrouter.test/api/v1",
                api_style="chat",
                max_retries=0,
            ),
            "key",
        )
        summary_only = [
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "delta": {
                                    "reasoning_details": [
                                        {
                                            "type": "reasoning.summary",
                                            "summary": "Abandoned summary.",
                                            "index": 0,
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                ),
            ),
            SSEEvent("message", "[DONE]"),
        ]
        transient_failure = [
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "error": {
                            "code": 503,
                            "message": "Provider temporarily unavailable.",
                        }
                    }
                ),
            )
        ]
        fake = FakeHttp(event_batches=[summary_only, transient_failure])
        provider.http = fake  # type: ignore[assignment]
        streamed: list[ProviderStreamEvent] = []

        with self.assertRaises(ProviderUnavailableError) as unavailable:
            async for event in provider.stream(self.request):
                streamed.append(event)

        self.assertTrue(unavailable.exception.retryable)
        self.assertEqual(streamed, [])
        self.assertEqual(len(fake.calls), 2)

    async def test_chat_stream_classifies_in_band_error_without_fallback_request(self) -> None:
        provider = OpenRouterProvider(
            ProviderConfig(
                type="openrouter",
                base_url="https://openrouter.test/api/v1",
                api_style="chat",
                max_retries=0,
            ),
            "key",
        )
        fake = FakeHttp(
            events=[
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "error": {
                                "code": 429,
                                "message": "Provider rate limit exceeded.",
                            }
                        }
                    ),
                )
            ]
        )
        provider.http = fake  # type: ignore[assignment]

        with self.assertRaises(ProviderRateLimitError) as rate_limited:
            _ = [event async for event in provider.stream(self.request)]

        self.assertTrue(rate_limited.exception.retryable)
        self.assertEqual(rate_limited.exception.status_code, 429)
        self.assertEqual(
            rate_limited.exception.details["error"]["message"],
            "Provider rate limit exceeded.",
        )
        self.assertEqual(len(fake.calls), 1)

    async def test_responses_retries_when_reasoning_summaries_are_unsupported(self) -> None:
        provider = OpenAIProvider(
            ProviderConfig(type="openai", base_url="https://openai.test/v1"),
            "key",
        )
        failed = [
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "type": "response.failed",
                        "response": {
                            "status": "failed",
                            "error": {
                                "message": "This model does not support reasoning summaries."
                            },
                        },
                    }
                ),
            )
        ]
        recovered = [
            SSEEvent(
                "message",
                json.dumps(
                    {"type": "response.output_text.delta", "delta": "Recovered answer."}
                ),
            ),
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "status": "completed",
                            "output": [],
                            "usage": {"input_tokens": 4, "output_tokens": 2},
                        },
                    }
                ),
            ),
        ]
        fake = FakeHttp(event_batches=[failed, recovered])
        provider.http = fake  # type: ignore[assignment]

        streamed = [event async for event in provider.stream(self.request)]

        final = streamed[-1].response
        assert final is not None
        self.assertEqual(final.text, "Recovered answer.")
        self.assertEqual(len(fake.calls), 2)
        first_payload = cast(dict[str, Any], fake.calls[0][1]["payload"])
        second_payload = cast(dict[str, Any], fake.calls[1][1]["payload"])
        self.assertEqual(
            first_payload["reasoning"],
            {"effort": "high", "summary": "auto"},
        )
        self.assertEqual(second_payload["reasoning"], {"effort": "high"})

        summary_then_failed = [
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": "reasoning_1",
                        "output_index": 0,
                        "summary_index": 0,
                        "delta": "Before fallback.",
                    }
                ),
            ),
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "type": "response.failed",
                        "response": {
                            "status": "failed",
                            "error": {
                                "message": "This model does not support reasoning summaries."
                            },
                        },
                    }
                ),
            ),
        ]
        fake = FakeHttp(event_batches=[summary_then_failed, recovered])
        provider.http = fake  # type: ignore[assignment]

        streamed = [event async for event in provider.stream(self.request)]

        self.assertEqual(
            [event.type for event in streamed],
            ["text_delta", "completed"],
        )
        assert streamed[-1].response is not None
        self.assertEqual(streamed[-1].response.text, "Recovered answer.")
        self.assertEqual(len(fake.calls), 2)

        top_level_failed = [
            SSEEvent(
                "message",
                json.dumps(
                    {
                        "type": "error",
                        "code": "invalid_parameter",
                        "message": "Reasoning summaries are unsupported.",
                        "param": "reasoning.summary",
                    }
                ),
            )
        ]
        fake = FakeHttp(event_batches=[top_level_failed, recovered])
        provider.http = fake  # type: ignore[assignment]

        streamed = [event async for event in provider.stream(self.request)]

        assert streamed[-1].response is not None
        self.assertEqual(streamed[-1].response.text, "Recovered answer.")
        self.assertEqual(len(fake.calls), 2)

    async def test_responses_classifies_in_band_transient_errors(self) -> None:
        provider = OpenAIProvider(
            ProviderConfig(
                type="openai",
                base_url="https://openai.test/v1",
                max_retries=0,
            ),
            "key",
        )
        cast(Any, provider).http = FakeHttp(
            data={
                "status": "failed",
                "error": {"code": "server_error", "message": "Generation failed."},
            }
        )

        with self.assertRaises(ProviderUnavailableError) as unavailable:
            await provider.complete(self.request)

        self.assertTrue(unavailable.exception.retryable)
        self.assertEqual(
            unavailable.exception.details["error"]["code"],
            "server_error",
        )

        cast(Any, provider).http = FakeHttp(
            data={
                "type": "error",
                "code": "rate_limit_exceeded",
                "message": "Slow down.",
            }
        )

        with self.assertRaises(ProviderRateLimitError) as rate_limited:
            await provider.complete(self.request)

        self.assertTrue(rate_limited.exception.retryable)
        self.assertEqual(rate_limited.exception.details["code"], "rate_limit_exceeded")

    async def test_openai_payload_helpers_and_response_complete_error(self) -> None:
        provider = OpenAIProvider(
            ProviderConfig(type="openai", base_url="https://x", api_style="responses"), ""
        )
        payload = provider._responses_payload(self.request, stream=True)
        self.assertEqual(
            payload["reasoning"],
            {"effort": "high", "summary": "auto"},
        )
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
                json.dumps({"type": "content_block_stop", "index": 1}),
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
        self.assertEqual(final.usage.input_tokens, 13)
        self.assertEqual(provider._headers()["x-api-key"], "key")

        cached_request = ProviderRequest(
            model="claude",
            system="stable\n\ndynamic",
            messages=self.messages,
            metadata={
                "prompt_cache_enabled": True,
                "prompt_cache_ttl": "1h",
                "anthropic_conversation_cache": True,
                "system_blocks": [
                    {"text": "stable", "cacheable": True},
                    {"text": "dynamic", "cacheable": False},
                ],
            },
        )
        payload = provider._payload(cached_request)
        self.assertEqual(
            payload["system"][0]["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )
        self.assertNotIn("cache_control", payload["system"][1])
        self.assertEqual(
            payload["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )

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

        cast(Any, provider).http = FakeHttp(
            events=[
                SSEEvent("message", json.dumps({"type": "error", "error": {"message": "boom"}}))
            ]
        )
        with self.assertRaisesRegex(ProviderError, "boom"):
            _ = [item async for item in provider.stream(self.request)]
        cast(Any, provider).http = FakeHttp(data="bad")
        with self.assertRaises(ProviderError):
            await provider.complete(self.request)

    async def test_anthropic_stream_hides_max_tokens_truncated_tool_call(self) -> None:
        provider = AnthropicProvider(
            ProviderConfig(type="anthropic", base_url="https://anthropic.test/v1"),
            "key",
        )
        cast(Any, provider).http = FakeHttp(
            events=[
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {
                                "type": "tool_use",
                                "id": "truncated-call",
                                "name": "write_file",
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
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": '{"path":"danger.txt"',
                            },
                        }
                    ),
                ),
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": "max_tokens"},
                            "usage": {"output_tokens": 4},
                        }
                    ),
                ),
                SSEEvent("message", json.dumps({"type": "message_stop"})),
            ]
        )

        streamed = [item async for item in provider.stream(self.request)]

        final = streamed[-1].response
        assert final is not None
        self.assertEqual(final.stop_reason, "max_tokens")
        self.assertEqual(final.tool_calls, [])

    async def test_anthropic_stream_hides_context_window_truncated_tool_call(self) -> None:
        provider = AnthropicProvider(
            ProviderConfig(type="anthropic", base_url="https://anthropic.test/v1"),
            "key",
        )
        cast(Any, provider).http = FakeHttp(
            events=[
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {
                                "type": "tool_use",
                                "id": "truncated-call",
                                "name": "write_file",
                                "input": {"path": "danger.txt"},
                            },
                        }
                    ),
                ),
                SSEEvent(
                    "message",
                    json.dumps({"type": "content_block_stop", "index": 0}),
                ),
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": "model_context_window_exceeded"
                            },
                            "usage": {"output_tokens": 4},
                        }
                    ),
                ),
                SSEEvent("message", json.dumps({"type": "message_stop"})),
            ]
        )

        streamed = [item async for item in provider.stream(self.request)]

        final = streamed[-1].response
        assert final is not None
        self.assertEqual(final.stop_reason, "model_context_window_exceeded")
        self.assertEqual(final.tool_calls, [])

    async def test_anthropic_stream_hides_call_without_stop_reason(self) -> None:
        provider = AnthropicProvider(
            ProviderConfig(type="anthropic", base_url="https://anthropic.test/v1"),
            "key",
        )
        cast(Any, provider).http = FakeHttp(
            events=[
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {
                                "type": "tool_use",
                                "id": "truncated-call",
                                "name": "write_file",
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
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": (
                                    '{"path":"must-not-run.txt","content":"done"}'
                                ),
                            },
                        }
                    ),
                ),
                SSEEvent(
                    "message",
                    json.dumps({"type": "content_block_stop", "index": 0}),
                ),
            ]
        )

        streamed = [item async for item in provider.stream(self.request)]

        final = streamed[-1].response
        assert final is not None
        self.assertEqual(final.stop_reason, "incomplete")
        self.assertEqual(final.tool_calls, [])

    async def test_gemini_truncated_stream_hides_partial_call(self) -> None:
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
        self.assertEqual(partial.tool_calls, [])
        self.assertEqual(partial.stop_reason, "incomplete")

    async def test_gemini_stream_complete_and_helpers(self) -> None:
        provider = GeminiProvider(
            ProviderConfig(type="gemini", base_url="https://gemini.test/v1beta"), "key"
        )

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
                                {
                                    "type": "thought",
                                    "signature": "stream-signature",
                                    "content": [],
                                },
                                {
                                    "type": "model_output",
                                    "content": [{"type": "text", "text": "done"}],
                                },
                                {
                                    "type": "function_call",
                                    "id": "f",
                                    "name": "read_file",
                                    "arguments": "bad",
                                },
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
        assert completed.continuation_state is not None
        self.assertEqual(completed.continuation_state.items[0]["signature"], "stream-signature")

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

    async def test_gemini_stream_error_raises_provider_failure(self) -> None:
        provider = GeminiProvider(
            ProviderConfig(type="gemini", base_url="https://gemini.test/v1beta"), "key"
        )
        provider.http = FakeHttp(  # type: ignore[assignment]
            events=[
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "type": "interaction.failed",
                            "interaction": {"error": {"message": "generation failed"}},
                        }
                    ),
                )
            ]
        )
        with self.assertRaisesRegex(ProviderError, "generation failed"):
            _ = [item async for item in provider.stream(self.request)]

    async def test_gemini_non_success_statuses_preserve_usage_and_reject_calls(self) -> None:
        provider = GeminiProvider(ProviderConfig(type="gemini", base_url="https://gemini.test"))

        async def run(streaming: bool):
            if streaming:
                return [item async for item in provider.stream(self.request)][-1].response
            return await provider.complete(self.request)

        for status in ("failed", "cancelled", "in_progress", "queued", "incomplete", "budget_exceeded"):
            for streaming in (False, True):
                with self.subTest(status=status, streaming=streaming):
                    data = {
                        "status": status,
                        "steps": [
                            {"type": "model_output", "content": [{"type": "text", "text": "partial"}]},
                            {"type": "function_call", "id": "c", "name": "write_file", "arguments": {}},
                        ],
                        "usage": {"total_input_tokens": 7, "total_output_tokens": 3},
                    }
                    provider.http = FakeHttp(  # type: ignore[assignment]
                        data=data,
                        events=[SSEEvent("message", json.dumps({
                            "event_type": "interaction.completed", "interaction": data,
                        }))],
                    )

                    if status in {"incomplete", "budget_exceeded"}:
                        result = await run(streaming)
                        assert result is not None
                        self.assertTrue(result.incomplete)
                        self.assertEqual(result.text, "partial")
                        self.assertEqual(result.tool_calls, [])
                        self.assertIsNone(result.continuation_state)
                        usage = result.usage
                    else:
                        with self.assertRaises(ProviderError) as caught:
                            await run(streaming)
                        usage = caught.exception.usage
                    assert usage is not None
                    self.assertEqual((usage.input_tokens, usage.output_tokens, usage.requests), (7, 3, 1))

    async def test_gemini_partial_completion_metadata_retains_streamed_steps(self) -> None:
        provider = GeminiProvider(ProviderConfig(type="gemini", base_url="https://gemini.test"))
        payloads = [
            {"event_type": "step.start", "index": 0, "step": {
                "type": "thought", "summary": [{"type": "text", "text": "First thought."}],
            }},
            {"event_type": "step.delta", "index": 0, "delta": {
                "type": "thought_summary", "content": {"type": "text", "text": "Inspect the file."},
            }},
            {"event_type": "step.delta", "index": 0, "delta": {
                "type": "thought_signature", "signature": "signed-thought",
            }},
            {"event_type": "step.start", "index": 1, "step": {
                "type": "model_output", "content": [{"type": "text", "text": "prefix "}],
            }},
            {"event_type": "step.delta", "index": 1, "delta": {"type": "text", "text": "Hello"}},
            {"event_type": "step.delta", "index": 1, "delta": {"type": "text", "text": " world"}},
            {"event_type": "step.stop", "index": 1, "usage": {
                "total_input_tokens": 7, "total_output_tokens": 5, "total_thought_tokens": 2,
            }},
            {"event_type": "step.start", "index": 2, "step": {
                "type": "function_call", "id": "gcall", "name": "read_file", "arguments": {},
            }},
            {"event_type": "step.delta", "index": 2, "delta": {
                "type": "arguments_delta", "arguments": '{"path":',
            }},
            {"event_type": "step.delta", "index": 2, "delta": {
                "type": "arguments_delta", "arguments": '"file.txt"}',
            }},
            {"event_type": "interaction.completed", "interaction": {
                "id": "i1", "status": "requires_action", "model": "gemini-model",
                "usage": {"total_output_tokens": 8},
            }},
        ]
        provider.http = FakeHttp(  # type: ignore[assignment]
            events=[SSEEvent("message", json.dumps(item)) for item in payloads]
        )
        events = [item async for item in provider.stream(self.request)]
        result = events[-1].response
        assert result is not None
        self.assertEqual(result.text, "prefix Hello world")
        self.assertEqual(result.tool_calls[0].id, "gcall")
        self.assertEqual(result.tool_calls[0].arguments, {"path": "file.txt"})
        self.assertEqual((result.usage.input_tokens, result.usage.output_tokens), (7, 8))
        self.assertEqual(result.usage.reasoning_tokens, 2)
        self.assertEqual(
            "".join(item.data["delta"] for item in events if item.type == "tool_call_delta"),
            '{"path":"file.txt"}',
        )
        state = result.continuation_state
        assert state is not None
        self.assertEqual([step["type"] for step in state.items], ["thought", "model_output", "function_call"])
        self.assertEqual(state.items[0]["signature"], "signed-thought")
        self.assertEqual(state.items[0], {
            "type": "thought", "signature": "signed-thought",
            "summary": [
                {"type": "text", "text": "First thought."},
                {"type": "text", "text": "Inspect the file."},
            ],
        })
        self.assertEqual(state.items[2]["arguments"], {"path": "file.txt"})

    async def test_gemini_interrupted_stream_retains_reported_usage(self) -> None:
        for failed in (False, True):
            with self.subTest(failed=failed):
                provider = GeminiProvider(ProviderConfig(type="gemini", base_url="https://gemini.test"))
                payloads = [
                    {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": "partial"}},
                    {"event_type": "step.stop", "index": 0, "usage": {
                        "total_input_tokens": 9, "total_output_tokens": 4,
                    }},
                ]
                if failed:
                    payloads.append({"event_type": "error", "error": {"message": "generation failed"}})
                provider.http = FakeHttp(  # type: ignore[assignment]
                    events=[SSEEvent("message", json.dumps(item)) for item in payloads]
                )
                if failed:
                    with self.assertRaisesRegex(ProviderError, "generation failed") as caught:
                        _ = [item async for item in provider.stream(self.request)]
                    usage = caught.exception.usage
                else:
                    result = [item async for item in provider.stream(self.request)][-1].response
                    assert result is not None
                    self.assertTrue(result.incomplete)
                    self.assertEqual(result.text, "partial")
                    self.assertEqual(result.tool_calls, [])
                    usage = result.usage
                assert usage is not None
                self.assertEqual((usage.input_tokens, usage.output_tokens), (9, 4))

    async def test_gemini_transport_failure_retains_reported_usage(self) -> None:
        class BrokenHttp(FakeHttp):
            async def stream_sse(self, url: str, **kwargs: object) -> AsyncIterator[SSEEvent]:
                async for event in super().stream_sse(url, **kwargs):
                    yield event
                raise ProviderError("connection lost")

        provider = GeminiProvider(ProviderConfig(type="gemini", base_url="https://gemini.test"))
        provider.http = BrokenHttp(events=[SSEEvent("message", json.dumps({  # type: ignore[assignment]
            "event_type": "step.stop", "index": 0,
            "usage": {"total_input_tokens": 9, "total_output_tokens": 4},
        }))])
        with self.assertRaisesRegex(ProviderError, "connection lost") as caught:
            _ = [item async for item in provider.stream(self.request)]
        usage = caught.exception.usage
        assert usage is not None
        self.assertEqual((usage.input_tokens, usage.output_tokens, usage.requests), (9, 4, 1))


if __name__ == "__main__":
    unittest.main()
