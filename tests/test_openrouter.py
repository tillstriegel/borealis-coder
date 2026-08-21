from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from borealis_coder.config import AgentConfig, Config, ProviderConfig
from borealis_coder.errors import ConfigurationError
from borealis_coder.models import Message, ProviderRequest, Role
from borealis_coder.providers.openrouter import OpenRouterProvider
from borealis_coder.providers.registry import ProviderRegistry
from borealis_coder.tools.base import object_schema


class OpenRouterProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = ProviderRequest(
            model="anthropic/claude-sonnet-4.6",
            system="system",
            messages=[Message(role=Role.USER, content="fix it")],
            tools=[
                {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": object_schema({"path": {"type": "string"}}),
                }
            ],
            reasoning_effort="high",
        )
        self.config = ProviderConfig(
            type="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            api_style="chat",
            site_url="https://example.test/borealis",
            app_name="Borealis Coder Tests",
            model_fallbacks=["openai/gpt-5.4-mini", "google/gemini-3.6-pro"],
            provider_preferences={
                "allow_fallbacks": True,
                "data_collection": "deny",
                "zdr": True,
            },
        )

    def test_headers_and_openrouter_chat_extensions(self) -> None:
        provider = OpenRouterProvider(self.config, "sk-or-test")
        headers = provider._headers()
        self.assertEqual(headers["Authorization"], "Bearer sk-or-test")
        self.assertEqual(headers["HTTP-Referer"], "https://example.test/borealis")
        self.assertEqual(headers["X-Title"], "Borealis Coder Tests")

        payload = provider._chat_payload(self.request, stream=True)
        self.assertEqual(payload["model"], "anthropic/claude-sonnet-4.6")
        self.assertEqual(
            payload["models"], ["openai/gpt-5.4-mini", "google/gemini-3.6-pro"]
        )
        self.assertTrue(payload["provider"]["allow_fallbacks"])
        self.assertEqual(payload["provider"]["data_collection"], "deny")
        self.assertTrue(payload["provider"]["zdr"])
        self.assertEqual(payload["reasoning"], {"effort": "high"})
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(payload["usage"], {"include": True})
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["stream_options"], {"include_usage": True})

    def test_responses_style_is_available(self) -> None:
        config = ProviderConfig(
            type="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_style="responses",
            model_fallbacks=["openai/gpt-5.4-mini"],
        )
        provider = OpenRouterProvider(config, "key")
        payload = provider._responses_payload(self.request)
        self.assertEqual(provider.api_style, "responses")
        self.assertEqual(payload["reasoning"], {"effort": "high"})
        self.assertEqual(payload["models"], ["openai/gpt-5.4-mini"])
        self.assertEqual(payload["usage"], {"include": True})

    def test_extra_body_is_an_escape_hatch(self) -> None:
        config = ProviderConfig(
            type="openrouter",
            base_url="https://openrouter.ai/api/v1",
            extra_body={"plugins": [{"id": "custom"}], "seed": 7},
        )
        payload = OpenRouterProvider(config, "key")._chat_payload(self.request)
        self.assertEqual(payload["plugins"], [{"id": "custom"}])
        self.assertEqual(payload["seed"], 7)

    def test_openrouter_usage_prefers_reported_cost(self) -> None:
        provider = OpenRouterProvider(
            ProviderConfig(
                type="openrouter",
                input_cost_per_million=99,
                output_cost_per_million=99,
            ),
            "key",
        )
        usage = provider._usage_from_chat(
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 40},
                "completion_tokens_details": {"reasoning_tokens": 7},
                "cost": 0.001234,
            }
        )
        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.cached_input_tokens, 40)
        self.assertEqual(usage.reasoning_tokens, 7)
        self.assertAlmostEqual(usage.cost_usd, 0.001234)

    def test_registry_requires_openrouter_key_and_constructs_provider(self) -> None:
        config = Config(
            agent=AgentConfig(provider="openrouter", model="anthropic/claude-sonnet-4.6"),
            providers={"openrouter": self.config},
        )
        registry = ProviderRegistry()
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ConfigurationError):
            registry.create(config)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-live"}, clear=True):
            name, model, provider = registry.create(config)
        self.assertEqual(name, "openrouter")
        self.assertEqual(model, "anthropic/claude-sonnet-4.6")
        self.assertIsInstance(provider, OpenRouterProvider)


if __name__ == "__main__":
    unittest.main()
