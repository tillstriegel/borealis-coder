"""Provider abstraction and common retry/cost behavior."""

from __future__ import annotations

import abc
import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from ..config import ProviderConfig
from ..errors import (
    ProviderAuthenticationError,
    ProviderContextOverflowError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from ..models import ModelResponse, ProviderRequest, Usage

T = TypeVar("T")


@dataclass(slots=True)
class ProviderStreamEvent:
    type: str  # text_delta | tool_call_delta | completed | usage | error
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    response: ModelResponse | None = None


class Provider(abc.ABC):
    name = "provider"

    def __init__(self, config: ProviderConfig, api_key: str = "") -> None:
        self.config = config
        self.api_key = api_key

    @abc.abstractmethod
    async def complete(self, request: ProviderRequest) -> ModelResponse:
        raise NotImplementedError

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderStreamEvent]:
        response = await self.complete(request)
        if response.text:
            yield ProviderStreamEvent(type="text_delta", text=response.text)
        yield ProviderStreamEvent(type="completed", response=response)

    async def with_retries(self, operation: Callable[[], Awaitable[T]]) -> T:
        attempts = max(0, self.config.max_retries) + 1
        delay = max(0.0, self.config.initial_backoff_seconds)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return await operation()
            except ProviderError as error:
                last_error = error
                if not error.retryable or attempt + 1 >= attempts:
                    raise
            except (TimeoutError, OSError) as error:
                last_error = error
                if attempt + 1 >= attempts:
                    raise ProviderUnavailableError(str(error), retryable=True) from error
            sleep_for = min(self.config.max_backoff_seconds, delay)
            sleep_for *= random.uniform(0.8, 1.2)
            await asyncio.sleep(sleep_for)
            delay = max(0.25, delay * 2)
        assert last_error is not None
        raise last_error

    def price_usage(self, usage: Usage) -> Usage:
        uncached = max(0, usage.input_tokens - usage.cached_input_tokens)
        usage.cost_usd = (
            uncached * self.config.input_cost_per_million
            + usage.cached_input_tokens * self.config.cached_input_cost_per_million
            + usage.output_tokens * self.config.output_cost_per_million
        ) / 1_000_000
        return usage


def classify_provider_error(status: int | None, message: str, details: Any = None) -> ProviderError:
    lower = message.lower()
    if status in {401, 403}:
        return ProviderAuthenticationError(message, status_code=status, details=details)
    if status == 429:
        return ProviderRateLimitError(
            message, status_code=status, retryable=True, details=details
        )
    if status in {408, 409, 425} or (status is not None and status >= 500):
        return ProviderUnavailableError(
            message, status_code=status, retryable=True, details=details
        )
    if any(
        phrase in lower
        for phrase in (
            "context length",
            "context window",
            "too many tokens",
            "maximum context",
            "input token limit",
            "request too large",
        )
    ):
        return ProviderContextOverflowError(message, status_code=status, details=details)
    return ProviderError(message, status_code=status, details=details)
