from __future__ import annotations

import unittest

from borealis_coder.config import ProviderConfig
from borealis_coder.models import Message, ProviderRequest, Role, ToolCall
from borealis_coder.providers.anthropic import AnthropicProvider
from borealis_coder.providers.gemini import GeminiProvider
from borealis_coder.providers.openai import OpenAIProvider
from borealis_coder.tools.base import object_schema


class ProviderAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tool = {
            "name": "read_file",
            "description": "Read",
            "parameters": object_schema({"path": {"type": "string"}}),
        }
        self.messages = [
            Message(role=Role.USER, content="read it"),
            Message(
                role=Role.ASSISTANT,
                tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a"})],
            ),
            Message(role=Role.TOOL, content="content", tool_call_id="c1", tool_name="read_file"),
        ]
        self.request = ProviderRequest(
            model="model", system="system", messages=self.messages, tools=[self.tool]
        )

    def test_openai_responses_payload_and_parser(self):
        provider = OpenAIProvider(
            ProviderConfig(
                type="openai",
                base_url="https://x",
                api_style="responses",
                input_cost_per_million=2,
            ),
            "key",
        )
        self.request.metadata.update(
            {"prompt_cache_enabled": True, "prompt_cache_key": "stable-cache-key"}
        )
        payload = provider._responses_payload(self.request)
        self.assertEqual(payload["tools"][0]["type"], "function")
        self.assertTrue(payload["tools"][0]["strict"])
        self.assertEqual(payload["prompt_cache_key"], "stable-cache-key")
        self.request.metadata["prompt_cache_enabled"] = False
        self.assertNotIn("prompt_cache_key", provider._responses_payload(self.request))
        parsed = provider._parse_responses(
            {
                "id": "r1",
                "model": "m",
                "status": "completed",
                "output": [
                    {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque"},
                    {"type": "message", "content": [{"type": "output_text", "text": "done"}]},
                    {
                        "type": "function_call",
                        "call_id": "c2",
                        "name": "read_file",
                        "arguments": '{"path":"b"}',
                    },
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 3},
                },
            },
            retain_raw=False,
        )
        self.assertEqual(parsed.text, "done")
        self.assertEqual(parsed.tool_calls[0].arguments["path"], "b")
        self.assertEqual(parsed.usage.cached_input_tokens, 2)
        self.assertEqual(parsed.usage.cache_write_tokens, 3)
        self.assertAlmostEqual(parsed.usage.cost_usd, 0.0000175)
        refused = provider._parse_responses(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "refusal", "refusal": "I cannot help with that."}
                        ],
                    }
                ]
            },
            retain_raw=False,
        )
        self.assertEqual(refused.text, "I cannot help with that.")
        chat_usage = provider._usage_from_chat(
            {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "prompt_tokens_details": {
                    "cached_tokens": 2,
                    "cache_write_tokens": 3,
                },
            }
        )
        self.assertEqual(chat_usage.cache_write_tokens, 3)
        self.assertAlmostEqual(chat_usage.cost_usd, 0.0000175)

        retained = provider._parse_responses(
            {
                "output": [{"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque"}],
            },
            retain_raw=True,
        )
        assert retained.continuation_state is not None
        self.assertEqual(retained.continuation_state.items[0]["encrypted_content"], "opaque")
        metadata = retained.continuation_state.to_metadata(provider="openai", model="model")
        assert metadata is not None
        assistant = Message(
            role=Role.ASSISTANT,
            content="done",
            metadata={"continuation_state": metadata},
        )
        continuation_request = ProviderRequest(
            model="model",
            system="system",
            messages=[assistant],
        )
        self.assertEqual(provider._responses_input(continuation_request)[0]["type"], "reasoning")
        continuation_request.model = "different-model"
        self.assertEqual(provider._responses_input(continuation_request)[0]["type"], "message")
        continuation_request.model = "model"
        alias_metadata = retained.continuation_state.to_metadata(
            provider="corp_openai",
            model="model",
        )
        assert alias_metadata is not None
        assistant.metadata = {"continuation_state": alias_metadata}
        continuation_request.metadata = {"provider_route": "corp_openai"}
        self.assertEqual(provider._responses_input(continuation_request)[0]["type"], "reasoning")
        continuation_request.metadata = {"provider_route": "different_alias"}
        self.assertEqual(provider._responses_input(continuation_request)[0]["type"], "message")
        assistant.metadata = {
            "responses_state": retained.continuation_state.items,
        }
        self.assertEqual(provider._responses_input(continuation_request)[0]["type"], "message")

    def test_openai_gpt_5_6_marks_only_the_stable_system_prefix(self):
        self.request.model = "gpt-5.6"
        self.request.system = "core\n\nrepository guidance\n\nrepository map\n\ndynamic status"
        self.request.metadata = {
            "prompt_cache_enabled": True,
            "prompt_cache_key": "stable-cache-key",
            "system_blocks": [
                {"text": "core", "cacheable": True},
                {"text": "repository guidance", "cacheable": True},
                {"text": "repository map", "cacheable": True},
                {"text": "dynamic status", "cacheable": False},
            ],
        }
        responses = OpenAIProvider(
            ProviderConfig(type="openai", base_url="https://x", api_style="responses"),
            "key",
        )._responses_payload(self.request)
        self.assertNotIn("instructions", responses)
        self.assertEqual(responses["prompt_cache_options"], {"mode": "implicit"})
        self.assertEqual(responses["include"], ["reasoning.encrypted_content"])
        developer = responses["input"][0]
        self.assertEqual(developer["role"], "developer")
        self.assertEqual(
            [item.get("prompt_cache_breakpoint") for item in developer["content"]],
            [
                {"mode": "explicit"},
                {"mode": "explicit"},
                {"mode": "explicit"},
                None,
            ],
        )
        self.assertEqual(developer["content"][1]["text"], "\n\nrepository guidance")
        self.assertEqual(developer["content"][3]["text"], "\n\ndynamic status")

        chat = OpenAIProvider(
            ProviderConfig(type="openai", base_url="https://x", api_style="chat"),
            "key",
        )._chat_payload(self.request)
        self.assertEqual(chat["prompt_cache_key"], "stable-cache-key")
        self.assertEqual(chat["prompt_cache_options"], {"mode": "implicit"})
        self.assertEqual(
            sum("prompt_cache_breakpoint" in item for item in chat["messages"][0]["content"]),
            3,
        )

        self.request.metadata["prompt_cache_enabled"] = False
        disabled = OpenAIProvider(
            ProviderConfig(type="openai", base_url="https://x", api_style="responses"),
            "key",
        )._responses_payload(self.request)
        self.assertEqual(disabled["instructions"], self.request.system)
        self.assertNotIn("prompt_cache_key", disabled)
        self.assertNotIn("prompt_cache_options", disabled)

        self.request.metadata["prompt_cache_enabled"] = True
        self.request.model = "gpt-5.5"
        older = OpenAIProvider(
            ProviderConfig(type="openai", base_url="https://x", api_style="responses"),
            "key",
        )._responses_payload(self.request)
        self.assertEqual(older["instructions"], self.request.system)
        self.assertNotIn("prompt_cache_options", older)

    def test_anthropic_payload_and_parser(self):
        provider = AnthropicProvider(ProviderConfig(type="anthropic", base_url="https://x"), "key")
        payload = provider._payload(self.request)
        self.assertEqual(payload["tools"][0]["name"], "read_file")
        layered = ProviderRequest(
            model="claude",
            system="core\n\nguidance\n\nmap\n\ndynamic",
            messages=[],
            metadata={
                "prompt_cache_enabled": True,
                "anthropic_conversation_cache": True,
                "system_blocks": [
                    {"text": "core", "cacheable": True},
                    {"text": "guidance", "cacheable": True},
                    {"text": "map", "cacheable": True},
                    {"text": "dynamic", "cacheable": False},
                ],
            },
        )
        layered_payload = provider._payload(layered)
        self.assertEqual(
            sum("cache_control" in item for item in layered_payload["system"]),
            3,
        )
        self.assertEqual(layered_payload["system"][1]["text"], "\n\nguidance")
        self.assertEqual(layered_payload["system"][3]["text"], "\n\ndynamic")
        self.assertEqual(layered_payload["cache_control"], {"type": "ephemeral"})
        parsed = provider._parse(
            {
                "id": "a",
                "model": "claude",
                "stop_reason": "tool_use",
                "content": [
                    {"type": "text", "text": "ok"},
                    {"type": "tool_use", "id": "t", "name": "read_file", "input": {"path": "x"}},
                ],
                "usage": {
                    "input_tokens": 8,
                    "output_tokens": 3,
                    "cache_read_input_tokens": 2,
                    "cache_creation_input_tokens": 1,
                },
            },
            retain_raw=False,
        )
        self.assertEqual(parsed.text, "ok")
        self.assertEqual(parsed.tool_calls[0].name, "read_file")

    def test_incomplete_non_stream_responses_hide_tool_calls(self):
        openai = OpenAIProvider(
            ProviderConfig(type="openai", base_url="https://x", api_style="chat"),
            "key",
        )._parse_chat(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "partial",
                                    "function": {
                                        "name": "write_file",
                                        "arguments": '{"path":"danger.txt"',
                                    },
                                }
                            ]
                        },
                    }
                ]
            },
            retain_raw=False,
        )
        anthropic = AnthropicProvider(
            ProviderConfig(type="anthropic", base_url="https://x"),
            "key",
        )._parse(
            {
                "stop_reason": "max_tokens",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "partial",
                        "name": "write_file",
                        "input": {"path": "danger.txt"},
                    }
                ],
            },
            retain_raw=False,
        )

        self.assertEqual(openai.tool_calls, [])
        self.assertEqual(anthropic.tool_calls, [])

    def test_gemini_payload_and_parser(self):
        provider = GeminiProvider(ProviderConfig(type="gemini", base_url="https://x"), "key")
        payload = provider._payload(self.request)
        self.assertIn("input", payload)
        parsed = provider._parse(
            {
                "id": "g",
                "model": "gemini",
                "status": "completed",
                "steps": [
                    {"type": "thought", "signature": "signed-thought", "content": []},
                    {"type": "model_output", "content": [{"type": "text", "text": "yes"}]},
                    {
                        "type": "function_call",
                        "id": "f",
                        "name": "read_file",
                        "arguments": {"path": "z"},
                    },
                ],
                "usage": {"total_input_tokens": 9, "total_output_tokens": 2},
            },
            retain_raw=False,
        )
        self.assertEqual(parsed.text, "yes")
        self.assertEqual(parsed.tool_calls[0].arguments["path"], "z")
        assert parsed.continuation_state is not None
        self.assertEqual(
            [step["type"] for step in parsed.continuation_state.items],
            ["thought", "model_output", "function_call"],
        )
        self.assertEqual(parsed.continuation_state.items[0]["signature"], "signed-thought")
        continuation = parsed.continuation_state.to_metadata(
            provider="gemini",
            model="model",
        )
        assert continuation is not None
        replay = ProviderRequest(
            model="model",
            system="system",
            messages=[
                Message(
                    role=Role.ASSISTANT,
                    content=parsed.text,
                    tool_calls=parsed.tool_calls,
                    metadata={"continuation_state": continuation},
                ),
                Message(
                    role=Role.TOOL,
                    content="content",
                    tool_call_id="f",
                    tool_name="read_file",
                ),
            ],
        )
        replay_steps = provider._steps(replay)
        self.assertEqual(replay_steps[:3], parsed.continuation_state.items)
        self.assertEqual(replay_steps[3]["type"], "function_result")

        alias_continuation = parsed.continuation_state.to_metadata(
            provider="corp_gemini",
            model="model",
        )
        assert alias_continuation is not None
        replay.messages[0].metadata = {"continuation_state": alias_continuation}
        replay.metadata = {"provider_route": "corp_gemini"}
        alias_steps = provider._steps(replay)
        self.assertEqual(alias_steps[:3], parsed.continuation_state.items)


if __name__ == "__main__":
    unittest.main()
