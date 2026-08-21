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
        self.tool = {"name":"read_file","description":"Read","parameters":object_schema({"path":{"type":"string"}})}
        self.messages = [
            Message(role=Role.USER, content="read it"),
            Message(role=Role.ASSISTANT, tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path":"a"})]),
            Message(role=Role.TOOL, content="content", tool_call_id="c1", tool_name="read_file"),
        ]
        self.request = ProviderRequest(model="model", system="system", messages=self.messages, tools=[self.tool])

    def test_openai_responses_payload_and_parser(self):
        provider = OpenAIProvider(ProviderConfig(type="openai", base_url="https://x", api_style="responses"), "key")
        payload = provider._responses_payload(self.request)
        self.assertEqual(payload["tools"][0]["type"], "function")
        self.assertTrue(payload["tools"][0]["strict"])
        parsed = provider._parse_responses({
            "id":"r1","model":"m","status":"completed",
            "output":[
                {"type":"message","content":[{"type":"output_text","text":"done"}]},
                {"type":"function_call","call_id":"c2","name":"read_file","arguments":"{\"path\":\"b\"}"},
            ],
            "usage":{"input_tokens":10,"output_tokens":4,"input_tokens_details":{"cached_tokens":2}},
        }, retain_raw=False)
        self.assertEqual(parsed.text, "done")
        self.assertEqual(parsed.tool_calls[0].arguments["path"], "b")
        self.assertEqual(parsed.usage.cached_input_tokens, 2)

    def test_anthropic_payload_and_parser(self):
        provider = AnthropicProvider(ProviderConfig(type="anthropic", base_url="https://x"), "key")
        payload = provider._payload(self.request)
        self.assertEqual(payload["tools"][0]["name"], "read_file")
        parsed = provider._parse({
            "id":"a","model":"claude","stop_reason":"tool_use",
            "content":[{"type":"text","text":"ok"},{"type":"tool_use","id":"t","name":"read_file","input":{"path":"x"}}],
            "usage":{"input_tokens":8,"output_tokens":3,"cache_read_input_tokens":2,"cache_creation_input_tokens":1},
        }, retain_raw=False)
        self.assertEqual(parsed.text, "ok")
        self.assertEqual(parsed.tool_calls[0].name, "read_file")

    def test_gemini_payload_and_parser(self):
        provider = GeminiProvider(ProviderConfig(type="gemini", base_url="https://x"), "key")
        payload = provider._payload(self.request)
        self.assertIn("input", payload)
        parsed = provider._parse({
            "id":"g","model":"gemini","status":"completed",
            "steps":[
                {"type":"model_output","content":[{"type":"text","text":"yes"}]},
                {"type":"function_call","id":"f","name":"read_file","arguments":{"path":"z"}},
            ],
            "usage":{"total_input_tokens":9,"total_output_tokens":2},
        }, retain_raw=False)
        self.assertEqual(parsed.text, "yes")
        self.assertEqual(parsed.tool_calls[0].arguments["path"], "z")


if __name__ == "__main__":
    unittest.main()
