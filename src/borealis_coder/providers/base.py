"""Provider abstraction and common retry/cost behavior."""

from __future__ import annotations

import abc
import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from contextvars import ContextVar
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
    type: str  # reasoning_summary_delta | text_delta | tool_call_delta | completed | usage | error
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    response: ModelResponse | None = None


class Provider(abc.ABC):
    name = "provider"

    def __init__(self, config: ProviderConfig, api_key: str = "") -> None:
        self.config = config
        self.api_key = api_key
        self._retry_task: ContextVar[asyncio.Task[Any] | None] = ContextVar(
            "provider_retry_task", default=None,
        )

    @abc.abstractmethod
    async def complete(self, request: ProviderRequest) -> ModelResponse:
        raise NotImplementedError

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderStreamEvent]:
        response = await self.complete(request)
        if response.reasoning_summary:
            yield ProviderStreamEvent(
                type="reasoning_summary_delta",
                text=response.reasoning_summary,
            )
        if response.text:
            yield ProviderStreamEvent(type="text_delta", text=response.text)
        yield ProviderStreamEvent(type="completed", response=response)

    async def close(self) -> None:
        """Release provider-owned runtime resources."""

        return None

    def _continuation_provider(self, request: ProviderRequest) -> str:
        route = request.metadata.get("provider_route")
        return route if isinstance(route, str) and route else self.name

    async def with_retries(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        failed_usage_collector: Usage | None = None,
    ) -> T:
        """Share one retry loop across nested wrappers in the same task."""
        task = asyncio.current_task()
        if task is not None and self._retry_task.get() is task:
            return await operation()
        token = self._retry_task.set(task)
        try:
            return await self._retry_operation(operation, failed_usage_collector=failed_usage_collector)
        finally:
            self._retry_task.reset(token)

    async def _retry_operation(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        failed_usage_collector: Usage | None = None,
    ) -> T:
        attempts = max(0, self.config.max_retries) + 1
        delay = max(0.0, self.config.initial_backoff_seconds)
        last_error: Exception | None = None
        prior_usage = Usage()
        for attempt in range(attempts):
            try:
                result = await operation()
                if isinstance(result, ModelResponse) and not prior_usage.is_empty:
                    result.usage = prior_usage.add(result.usage)
                return result
            except ProviderError as error:
                last_error = error
                if error.usage is not None:
                    if failed_usage_collector is not None:
                        failed_usage_collector.add(error.usage)
                    prior_usage.add(error.usage)
                if not error.retryable or attempt + 1 >= attempts:
                    if not prior_usage.is_empty:
                        error.usage = prior_usage
                    raise
            except (TimeoutError, OSError) as error:
                last_error = error
                if attempt + 1 >= attempts:
                    raise ProviderUnavailableError(
                        str(error),
                        retryable=True,
                        usage=prior_usage if not prior_usage.is_empty else None,
                    ) from error
            sleep_for = min(self.config.max_backoff_seconds, delay)
            sleep_for *= random.uniform(0.8, 1.2)
            await asyncio.sleep(sleep_for)
            delay = max(0.25, delay * 2)
        assert last_error is not None
        raise last_error

    def price_usage(
        self,
        usage: Usage,
        *,
        cache_write_multiplier: float = 1.0,
    ) -> Usage:
        cache_write_rate = self.config.cache_write_input_cost_per_million
        if not cache_write_rate:
            cache_write_rate = self.config.input_cost_per_million * cache_write_multiplier
        input_cost = (
            usage.uncached_input_tokens * self.config.input_cost_per_million
            + usage.cached_input_tokens * self.config.cached_input_cost_per_million
            + usage.cache_write_tokens * cache_write_rate
        ) / 1_000_000
        usage.cost_usd = input_cost + (
            usage.output_tokens * self.config.output_cost_per_million / 1_000_000
        )
        baseline_input_cost = usage.input_tokens * self.config.input_cost_per_million / 1_000_000
        usage.cache_savings_usd = baseline_input_cost - input_cost
        return usage


def classify_provider_error(status: int | None, message: str, details: Any = None) -> ProviderError:
    lower = message.lower()
    if status in {401, 403}:
        return ProviderAuthenticationError(message, status_code=status, details=details)
    if status == 429:
        return ProviderRateLimitError(message, status_code=status, retryable=True, details=details)
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
