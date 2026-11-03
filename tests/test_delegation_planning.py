from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from borealis_coder.agent import build_runner
from borealis_coder.config import ProviderConfig
from borealis_coder.models import ModelResponse, Role, ToolCall
from borealis_coder.providers.mock import MockProvider
from borealis_coder.providers.openai import OpenAIProvider
from borealis_coder.tools.delegate import DelegateTaskTool, delegated_tool_allowed
from tests.helpers import make_config, make_context


class DelegationPlanningTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.runner = await build_runner(self.root, config=make_config(self.root), interactive=False)
        self.addAsyncCleanup(self.runner.close)
        provider = self.runner.providers[0].provider
        assert isinstance(provider, MockProvider)
        self.provider = provider

    def assert_schema(self, schema, cap):
        turns = schema['parameters']['properties']['max_turns']
        self.assertEqual(turns['minimum'], min(2, cap))
        self.assertEqual(turns['maximum'], cap)
        self.assertIn('report-only', schema['description'])
        self.assertIn('two turns', schema['description'])

    def assert_closure(self, request, observed=False):
        self.assertEqual(request.tools, [])
        self.assertFalse(request.parallel_tool_calls)
        self.assertIn('All tools are temporarily disabled', request.system)
        self.assertIn('not evidence of permanent capability absence', request.system)
        names = ', '.join(name for name in self.runner.tools.names()
                          if (tool := self.runner.tools.get(name)) is not None
                          and delegated_tool_allowed(tool, False))
        self.assertIn('Eligible investigative tools: ' + names, request.system)
        if not observed:
            self.assertIn('No completed tool observations', request.system)
            self.assertIn('must not claim', request.system)

    async def test_planner_drives_real_nested_discovery_and_persisted_model_conclusion(self):
        for cap in (2, 6):
            for surface in ('direct', 'ptc'):
                with self.subTest(cap=cap, surface=surface):
                    self.runner.config.agent.swarm_max_turns = cap
                    self.runner.config.script.tool_surface = surface
                    seen = []
                    requests = []

                    def handler(request, call, cap=cap, seen=seen, requests=requests):
                        depth = request.metadata.get('delegation_depth', 0)
                        seen.append(depth)
                        requests.append(request)
                        if not request.tools and request.messages[-1].role != Role.TOOL:
                            return ModelResponse(text="Report from supplied material only.")
                        if request.messages[-1].role == Role.TOOL:
                            if depth == 2:
                                catalog = json.loads(request.messages[-1].content)
                                self.assertIsNone(catalog['next_offset'])
                                self.assert_schema(catalog['tools'][0], cap)
                                self.assert_closure(request, observed=True)
                                return ModelResponse(text='Observed delegate_task maximum: ' + str(
                                    catalog['tools'][0]['parameters']['properties']['max_turns']['maximum']))
                            return ModelResponse(text=request.messages[-1].content)
                        schema = next(t for t in request.tools if t['name'] == 'delegate_task')
                        if depth < 2:
                            # A planner uses the advertised minimum, not a hard-coded allowance.
                            return ModelResponse(tool_calls=[ToolCall(name='delegate_task', arguments={
                                'task': 'Report observed delegation schema',
                                'max_turns': schema['parameters']['properties']['max_turns']['minimum'],
                            })])
                        return ModelResponse(tool_calls=[ToolCall(name='discover_tools', arguments={
                            'names': ['delegate_task'], 'offset': 0,
                        })])

                    self.provider.handler = handler
                    result = await self.runner.run('Report the observed nested schema')
                    self.assertEqual(seen, [0, 1, 2, 2, 1, 0])
                    for request in requests:
                        if request.tools:
                            self.assert_schema(next(t for t in request.tools if t["name"] == "delegate_task"), cap)
                            self.assertIn("report-only", request.system)
                    self.assertIn(f'Observed delegate_task maximum: {cap}', result.text)
                    rows = self.runner.sessions.tool_calls(result.session_id)
                    discovered = [r for r in rows if r['tool_name'] == 'discover_tools']
                    self.assertEqual(len(discovered), 1)
                    self.assertFalse(discovered[0]['is_error'])
                    receipts = [r for r in rows if r['tool_name'] == 'delegate_task']
                    self.assertEqual(len(receipts), 2)
                    for receipt in receipts:
                        self.assertEqual(receipt['metadata']['result_source'], 'model')
                        self.assertEqual(receipt['metadata']['tool_observation_status'], 'recorded')
                        self.assertEqual(receipt['metadata']['turns'], 2)
                    self.assertEqual(result.usage.requests, 6)

    async def test_explicit_one_and_operator_cap_one_are_report_only_without_inflation(self):
        for cap, requested in ((6, 1), (1, 1), (1, 2)):
            for surface in ('direct', 'ptc'):
                with self.subTest(cap=cap, requested=requested, surface=surface):
                    self.runner.config.agent.swarm_max_turns = cap
                    self.runner.config.script.tool_surface = surface
                    children = []

                    def handler(request, call, cap=cap, requested=requested, children=children):
                        if request.metadata.get('delegated'):
                            children.append(request)
                            self.assert_closure(request)
                            return ModelResponse(text='Summary of supplied material.')
                        if request.messages[-1].role == Role.TOOL:
                            return ModelResponse(text='Parent summary.')
                        self.assert_schema(next(t for t in request.tools if t['name'] == 'delegate_task'), cap)
                        return ModelResponse(tool_calls=[ToolCall(name='delegate_task', arguments={
                            'task': 'Summarize supplied material', 'max_turns': requested,
                        })])

                    self.provider.handler = handler
                    result = await self.runner.run('Summarize supplied material')
                    self.assertEqual(len(children), 1)
                    self.assertEqual(result.usage.requests, 3)
                    rows = self.runner.sessions.tool_calls(result.session_id)
                    self.assertEqual(len(rows), 1)
                    receipt = rows[0]
                    self.assertFalse(receipt['is_error'])
                    self.assertEqual(receipt['metadata']['delegation_mode'], 'report_only')
                    self.assertEqual(receipt['metadata']['tool_observation_status'], 'none')
                    self.assertEqual(receipt['metadata']['effective_max_turns'], 1)
                    self.assertEqual(receipt['metadata']['requested_max_turns'], requested)
                    self.assertIn('No tools executed', receipt['output'])
                    self.assertIn('unverified', receipt['output'])
                    self.assertIn('Summary of supplied material.', receipt['output'])

    async def test_shared_capacity_closure_is_explicit_and_unverified(self):
        self.runner.config.agent.max_model_requests = 3
        children = []

        def handler(request, call):
            if request.metadata.get('delegated'):
                children.append(request)
                self.assert_closure(request)
                return ModelResponse(text='Supplied material summary.')
            if request.messages[-1].role == Role.TOOL:
                return ModelResponse(text='Qualified parent summary.')
            return ModelResponse(tool_calls=[ToolCall(name='delegate_task', arguments={
                'task': 'Inspect available tools', 'max_turns': 6,
            })])

        self.provider.handler = handler
        result = await self.runner.run('Inspect tools within shared capacity')
        self.assertEqual(len(children), 1)
        self.assertEqual(result.usage.requests, 3)
        receipt = self.runner.sessions.tool_calls(result.session_id)[0]
        self.assertEqual(receipt['metadata']['delegation_mode'], 'investigation')
        self.assertEqual(receipt['metadata']['tool_observation_status'], 'none')
        self.assertIn('No tools executed', receipt['output'])

    def test_configuration_schema_views_do_not_mutate_runtime_validation(self):
        tool = DelegateTaskTool()
        configs = [make_config(self.root, agent={'swarm_max_turns': cap}) for cap in (1, 2, 6)]
        schemas = [tool.schema_for_config(config) for config in configs]
        for cap, schema in zip((1, 2, 6), schemas, strict=True):
            self.assert_schema(schema, cap)
        schemas[0]['parameters']['properties']['max_turns']['minimum'] = 99
        self.assert_schema(tool.schema_for_config(configs[0]), 1)
        self.assertEqual(tool.parameters['properties']['max_turns']['minimum'], 1)
        self.assertEqual(tool.parameters['properties']['max_turns']['maximum'], 16)

    async def test_chat_transport_carries_report_only_contract_without_tools(self):
        provider = OpenAIProvider(ProviderConfig(
            type="openai", api_style="chat", base_url="https://provider.test",
        ))
        self.addAsyncCleanup(provider.close)
        transport = AsyncMock(return_value=SimpleNamespace(data={
            "choices": [{"message": {"content": "Discovery verified."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }))
        provider.http.post_json = transport
        context = make_context(self.root)
        context.metadata.update({
            "provider_routes": [SimpleNamespace(name="synthetic", model="synthetic", provider=provider)],
            "tool_registry": self.runner.tools,
            "context_builder": self.runner.context_builder,
        })
        result = await DelegateTaskTool().execute({"task": "Report supplied material", "max_turns": 1}, context)
        self.assertEqual(transport.await_count, 1)
        assert transport.await_args is not None
        payload = transport.await_args.kwargs["payload"]
        self.assertEqual(payload["tools"], [])
        self.assertEqual(payload["tool_choice"], "none")
        self.assertFalse(payload["parallel_tool_calls"])
        system = payload["messages"][0]["content"]
        self.assertIn("All tools are temporarily disabled", system)
        self.assertIn("Eligible investigative tools:", system)
        self.assertIn("No completed tool observations", system)
        self.assertEqual(result.metadata["result_source"], "model")
        self.assertEqual(result.metadata["tool_observation_status"], "none")
        self.assertEqual(result.metadata["delegation_mode"], "report_only")
        # Runtime qualification is authoritative even when the model ignores the prompt.
        self.assertIn("No tools executed; discovery, fetch and inspection remain unverified.", result.output)
        self.assertIn("Discovery verified.", result.output)
