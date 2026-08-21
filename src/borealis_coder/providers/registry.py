"""Provider construction and registration."""

from __future__ import annotations

import os
from collections.abc import Callable

from ..config import Config, ProviderConfig
from ..errors import ConfigurationError
from .anthropic import AnthropicProvider
from .chatgpt import ChatGPTProvider
from .base import Provider
from .gemini import GeminiProvider
from .mock import MockProvider
from .openai import OpenAICompatibleProvider, OpenAIProvider
from .openrouter import OpenRouterProvider

ProviderFactory = Callable[[ProviderConfig, str], Provider]


class ProviderRegistry:
    @classmethod
    def with_defaults(cls) -> "ProviderRegistry":
        """Return an independent registry containing all built-in providers."""
        return cls()

    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {
            "openai": OpenAIProvider,
            "openrouter": OpenRouterProvider,
            "chatgpt": ChatGPTProvider,
            "openai_compatible": OpenAICompatibleProvider,
            "anthropic": AnthropicProvider,
            "gemini": GeminiProvider,
            "mock": MockProvider,
        }

    def register(self, name: str, factory: ProviderFactory) -> None:
        if not name:
            raise ValueError("Provider name cannot be empty")
        self._factories[name] = factory

    def create(self, config: Config, name: str | None = None) -> tuple[str, str, Provider]:
        provider_name, provider_config = config.provider(name)
        factory = self._factories.get(provider_config.type)
        if factory is None:
            raise ConfigurationError(
                f"Provider {provider_name!r} uses unsupported type {provider_config.type!r}"
            )
        api_key = os.getenv(provider_config.api_key_env, "") if provider_config.api_key_env else ""
        if provider_config.type not in {"mock", "openai_compatible", "chatgpt"} and provider_config.api_key_env and not api_key:
            raise ConfigurationError(
                f"Environment variable {provider_config.api_key_env} is required for provider {provider_name!r}"
            )
        model = config.resolved_model(provider_name, provider_config)
        return provider_name, model, factory(provider_config, api_key)


DEFAULT_PROVIDER_REGISTRY = ProviderRegistry()
