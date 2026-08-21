"""Deterministic provider used for tests, demos, and offline evaluations."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from ..models import ModelResponse, ProviderRequest, ToolCall, Usage
from .base import Provider, ProviderStreamEvent

MockHandler = Callable[[ProviderRequest, int], ModelResponse]


class MockProvider(Provider):
    name = "mock"

    def __init__(self, config, api_key: str = "", handler: MockHandler | None = None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(config, api_key)
        self.handler = handler
        self.calls = 0
        self.responses: list[ModelResponse] = []

    def enqueue(self, *responses: ModelResponse) -> None:
        self.responses.extend(responses)

    async def complete(self, request: ProviderRequest) -> ModelResponse:
        self.calls += 1
        if self.handler:
            response = self.handler(request, self.calls)
        elif self.responses:
            response = self.responses.pop(0)
        else:
            response = self._default_response(request)
        response.usage.requests = max(1, response.usage.requests)
        return response

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderStreamEvent]:
        response = await self.complete(request)
        for chunk in _chunks(response.text, 24):
            yield ProviderStreamEvent(type="text_delta", text=chunk)
        yield ProviderStreamEvent(type="completed", response=response)

    @staticmethod
    def _default_response(request: ProviderRequest) -> ModelResponse:
        last = request.messages[-1].content if request.messages else ""
        if "OFFLINE_WRITE_DEMO" in last and not any(
            message.tool_calls for message in request.messages
        ):
            return ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_mock_write",
                        name="write_file",
                        arguments={
                            "path": "borealis-demo.txt",
                            "content": "Created by the deterministic Borealis mock provider.\n",
                            "expected_sha256": None,
                        },
                    )
                ],
                usage=Usage(input_tokens=50, output_tokens=25, requests=1),
                stop_reason="tool_use",
                model="deterministic",
            )
        return ModelResponse(
            text="Offline mock response. Configure a provider and API key for live model execution.",
            usage=Usage(input_tokens=30, output_tokens=14, requests=1),
            stop_reason="end_turn",
            model="deterministic",
        )


def _chunks(value: str, size: int) -> list[str]:
    return [value[index : index + size] for index in range(0, len(value), size)] or [""]
