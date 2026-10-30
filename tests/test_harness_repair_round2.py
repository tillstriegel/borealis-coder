"""Focused offline regressions for the second boundary review."""
from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

from borealis_coder.agent import build_runner
from borealis_coder.agent.budget import Budget
from borealis_coder.errors import SessionError
from borealis_coder.models import Message, ModelResponse, Role, ToolCall, ToolResult, Usage
from borealis_coder.protocol import ACPServer
from borealis_coder.providers.mock import MockProvider
from borealis_coder.tools import FunctionTool, object_schema
from tests.helpers import make_config


def echo_evidence(request):
    prompt = request.messages[-1].content
    return prompt.split("<untrusted_structured_evidence>\n", 1)[1].split(
        "\n</untrusted_structured_evidence>", 1
    )[0]


class HarnessRepairRoundTwoTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    async def runner(self, **sections):
        runner = await build_runner(
            self.root, config=make_config(self.root, **sections), interactive=False
        )
        self.addAsyncCleanup(runner.close)
        provider = runner.providers[0].provider
        assert isinstance(provider, MockProvider)
        return runner, provider

    async def test_comma_intention_is_repaired_before_visible_publication(self):
        runner, provider = await self.runner()
        intention = "Let me inspect the configuration, then check the tests."
        answer = "The configuration enables retries; the tests cover three attempts."
        provider.enqueue(ModelResponse(text=intention), ModelResponse(text=answer))
        events = []
        runner.events.subscribe(events.append)
        result = await runner.run("Explain the retry configuration.")
        self.assertEqual(result.text, answer)
        self.assertEqual(provider.calls, 2)
        self.assertEqual(result.usage.requests, 2)
        self.assertFalse(any(intention in str(event.data) for event in events))
        self.assertNotIn(intention, [m.content for m in runner.sessions.messages(result.session_id)])
        self.assertEqual(
            "".join(e.data["text"] for e in events if e.type == "model.text_delta"), answer
        )

    async def test_quota_prefix_matches_only_exact_retained_source_values(self):
        runner, _ = await self.runner()
        store = runner.sessions
        session = store.create_session(workspace=self.root, provider="mock", model="deterministic")
        quote = 'Größe "quoted" \\ source\nline 😀'

        def save(payload, cap):
            return store.save_output_artifact(
                session.id, self.root, source="nested_synthetic", content=payload, max_bytes=cap,
                redactor=runner.events.redactor,
            )

        def matches(artifact, text, envelope=True):
            return store.output_artifact_has_quote(
                session.id, self.root, artifact["artifact_id"], text, json_envelope=envelope
            )

        payload = json.dumps({"output": "A" * 1000 + quote + "B" * 10000}, ensure_ascii=True)
        retained = save(payload, 2000)
        self.assertEqual(retained["status"], "quota_limited")
        self.assertTrue(matches(retained, quote))
        cut = payload.index("\\u00f6") + 4  # Cut inside the Unicode escape in Größe.
        self.assertFalse(matches(save(payload, cut), quote))
        literal = r"literal \u00f6 and \n stay literal"
        data = json.dumps({"metadata": {"only metadata": "private metadata"},
                           "data": {"only key": [literal, quote + "C" * 3000]}})
        nested = save(data, 700)
        self.assertTrue(matches(nested, literal))
        self.assertTrue(matches(nested, quote))
        self.assertFalse(matches(nested, literal.replace(r"\u00f6", "ö")))
        self.assertFalse(matches(nested, "only key"))
        self.assertFalse(matches(nested, "private metadata"))
        self.assertFalse(matches(nested, "only metadata"))
        for suffix in ("\\", "\\u", "\\u0", "\\u00", "\\u00f"):
            partial = save('{"data":["kept ' + suffix + '00"' + " " * 100, len('{"data":["kept ' + suffix))
            self.assertTrue(matches(partial, "kept "))
            self.assertFalse(matches(partial, "kept ö"))
        complete = save(json.dumps({"output": quote, "data": {"x": literal}}), 2000)
        self.assertTrue(matches(complete, quote))
        self.assertTrue(matches(complete, literal))
        raw = save(quote, 2000)
        self.assertTrue(matches(raw, quote, False))
        malformed = save('{"output":"unclosed', 2000)
        self.assertFalse(matches(malformed, "unclosed"))
        other = store.create_session(workspace=self.root, provider="mock", model="deterministic")
        with self.assertRaises(SessionError):
            store.output_artifact_has_quote(other.id, self.root, retained["artifact_id"], quote, json_envelope=True)
        with self.assertRaises(SessionError):
            store.output_artifact_has_quote(session.id, self.root / "other", retained["artifact_id"], quote, json_envelope=True)
        with self.assertRaises(SessionError):
            matches(retained, "x" * 601)

    def history(self, runner, session_id, *, large=False):
        runner.sessions.append_message(session_id, Message(role=Role.USER, content="Inspect retained source evidence."))
        for index in range(25):
            call = ToolCall(id=f"prior-{index}", name="read_file", arguments={"path": f"source-{index}.txt"})
            metadata: dict[str, Any] = {"path": f"source-{index}.txt"}
            if large and index == 0:
                call.name = "record_finding"
                call.arguments = {"tool_call_id": "historical-source", "quote": "Retained critical observation."}
                metadata["retained_finding"] = {"quote": "Retained critical observation.", "path": "source-0.txt"}
            runner.sessions.append_message(session_id, Message(role=Role.ASSISTANT, tool_calls=[call]))
            runner.sessions.append_message(session_id, Message(
                role=Role.TOOL, tool_call_id=call.id, tool_name=call.name,
                content=(f"source observation {index} " * (30 if large else 1)),
                metadata=metadata,
            ))

    async def test_helper_consumption_drains_already_completed_script_before_final_primary(self):
        await self._assert_completed_script_is_published(write=True)

    async def test_read_only_background_completion_has_one_visible_answer(self):
        await self._assert_completed_script_is_published(write=False)

    async def _assert_completed_script_is_published(self, *, write: bool):
        runner, provider = await self.runner(
            agent={"max_model_requests": 3, "deterministic_compaction": False, "auto_verify": False},
            context={"compact_tool_output_tokens": 1000},
        )
        session = runner.sessions.create_session(workspace=self.root, provider="mock", model="deterministic")
        self.history(runner, session.id)
        entered, release, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def gate(arguments, context):
            entered.set()
            await release.wait()
            return ToolResult("settled evidence")

        runner.tools.register(FunctionTool(name="fixture_gate", description="Controlled barrier", parameters=object_schema({}), function=gate))
        runner.tools.register(FunctionTool(name="fixture_evidence", description="Compaction trigger", parameters=object_schema({}), function=lambda a, c: ToolResult("Distinct observation " * 1000)))
        events, managers, requests = [], [], []
        def observe(event):
            events.append(event)
            if event.type == "script.completed":
                completed.set()
        runner.events.subscribe(observe)
        original_notifications = runner._script_notifications
        async def notifications(scripts, messages, *, wait=False):
            managers.append(scripts)
            return await original_notifications(scripts, messages, wait=wait)
        runner._script_notifications = notifications

        source = 'await tools.fixture_gate({}); '
        if write:
            source += 'await tools.write_file({path:"result.txt",content:"settled evidence",expected_sha256:null}); '
        source += 'text("settled evidence");'
        answer = (
            "The completed background job wrote result.txt." if write
            else "The read-only background job returned settled evidence."
        )

        async def respond(request):
            requests.append(request)
            if request.metadata.get("purpose") == "compaction_summary":
                self.assertEqual(len(requests), 2)
                await asyncio.wait_for(entered.wait(), 5)
                release.set()
                await asyncio.wait_for(completed.wait(), 5)
                async def finished():
                    while managers[-1].active:
                        await asyncio.sleep(0)
                await asyncio.wait_for(finished(), 5)
                return ModelResponse(text=echo_evidence(request), usage=Usage(requests=1, input_tokens=10))
            if len(requests) == 1:
                return ModelResponse(tool_calls=[
                    ToolCall(name="script", arguments={"background": True, "source": source}),
                    ToolCall(name="fixture_evidence", arguments={}),
                ], usage=Usage(requests=1, input_tokens=10))
            self.assertEqual(len(requests), 3)
            self.assertFalse(managers[-1].active)
            notice = next(m for m in request.messages if m.metadata.get("internal") == "script_completion")
            self.assertIn("settled evidence", notice.content)
            if write:
                self.assertIn("result.txt", str(notice.metadata))
            self.assertIn("including this request: 1", request.system)
            return ModelResponse(text=answer, usage=Usage(requests=1, input_tokens=10))

        provider.complete = AsyncMock(side_effect=respond)
        try:
            result = await asyncio.wait_for(runner.run("Continue and answer after completion.", session_id=session.id), 15)
        finally:
            release.set()
        self.assertEqual(result.stop_reason.value, "end_turn", result.to_dict())
        self.assertEqual(provider.complete.await_count, 3)
        self.assertEqual(result.usage.requests, 3)
        self.assertEqual(result.usage.input_tokens, 30)
        self.assertEqual(runner.sessions.usage(session.id).requests, 3)
        self.assertEqual(result.text, answer)
        if write:
            self.assertEqual((self.root / "result.txt").read_text(), "settled evidence")
            self.assertIn("result.txt", result.changed_files)
        else:
            self.assertFalse((self.root / "result.txt").exists())
            self.assertEqual(result.changed_files, [])
        self.assertEqual(sum(row["tool_name"] == "write_file" for row in runner.sessions.tool_calls(session.id)), int(write))
        visible = [e.data["text"] for e in events if e.type == "model.completed" and e.data.get("text")]
        self.assertEqual(visible, [result.text])
        public_answers = [
            message.content for message in runner.sessions.messages(session.id)
            if message.role == Role.ASSISTANT
            and not message.metadata.get("internal")
            and not message.tool_calls
            and message.content == result.text
        ]
        self.assertEqual(public_answers, [result.text])
        self.assertEqual([e.data["result"]["text"] for e in events if e.type == "run.completed"], [result.text])
        server = ACPServer()
        server._update = AsyncMock()
        for event in events:
            await server._event_update(session.id, runner, event)
        updates = [call.args[1] for call in server._update.await_args_list]
        answers = [update for update in updates if update["sessionUpdate"] == "agent_message"]
        chunks = [update for update in updates if update["sessionUpdate"] == "agent_message_chunk"]
        self.assertEqual([answer["content"] for answer in answers], [[{"type": "text", "text": result.text}]])
        self.assertEqual("".join(chunk["content"]["text"] for chunk in chunks), result.text)
        self.assertEqual({chunk["messageId"] for chunk in chunks}, {answers[0]["messageId"]})

    async def test_budget_change_reuses_evidence_but_real_instruction_change_invalidates(self):
        runner, provider = await self.runner(
            agent={"max_model_requests": 3, "deterministic_compaction": False},
            context={"compact_tool_output_tokens": 4000},
        )
        session = runner.sessions.create_session(workspace=self.root, provider="mock", model="deterministic")
        self.history(runner, session.id, large=True)
        provider.handler = lambda request, turn: ModelResponse(text=echo_evidence(request), usage=Usage(requests=1, input_tokens=11))
        prompt_context = runner.context_builder.build(query="continue")
        budget = Budget.start(runner.config.agent)
        settled = []
        async def usage_sink(usage):
            settled.append(usage)
        async def prepare(context=prompt_context, active_budget=budget):
            return await runner._prepare_provider_request(
                prompt_context=context, messages=runner.sessions.messages(session.id),
                schemas=runner.tools.schemas(), final_turn=False,
                verification_finalization_pending=False, adaptive_cache=False,
                conversation_cache=True, usage_sink=usage_sink, settled_usage_sink=usage_sink,
                cancel=asyncio.Event(),
                session_id=session.id, run_id="round-two", last_prune_signature=None,
                budget=active_budget, before_model_request=active_budget.before_model_request,
            )
        first = await prepare()
        self.assertEqual(provider.calls, 1)
        self.assertEqual(budget.model_requests, 1)
        self.assertIn("including this request: 2", first.request.system)
        budget.before_model_request()  # The preceding primary request used one allowance.
        second = await prepare()
        self.assertEqual(provider.calls, 1, "Budget guidance must not cause another summary")
        self.assertEqual(budget.model_requests, 2)
        self.assertEqual(len(settled), 1)
        self.assertEqual(first.request.metadata["compaction_artifact_id"], second.request.metadata["compaction_artifact_id"])
        self.assertIn("including this request: 1", second.request.system)
        self.assertNotIn("including this request: 2", second.request.system)
        self.assertIn("Retained critical observation.", second.request.system)
        self.assertIn("source-24.txt", str([m.to_dict() for m in second.request.messages]))
        artifact = runner.sessions.compaction_artifacts(session.id)[0]
        context = dict(artifact.metadata["provider_contexts"][0])
        context.update(system=second.request.system, system_blocks=second.request.metadata["system_blocks"], messages=[m.to_dict() for m in second.request.messages])
        context["metadata"] = {**context["metadata"], "system_blocks": second.request.metadata["system_blocks"]}
        digest = hashlib.sha256(json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(second.request.metadata["compacted_context_hashes"][runner.providers[0].name], digest)
        self.assertNotEqual(first.request.metadata["compacted_context_hashes"], second.request.metadata["compacted_context_hashes"])
        changed = replace(prompt_context, dynamic=prompt_context.dynamic + "\nNever edit retained sources.")
        third = await prepare(changed, Budget.start(runner.config.agent))
        self.assertNotEqual(third.request.metadata["compaction_artifact_id"], artifact.id)
        latest = runner.sessions.compaction_artifacts(session.id)[-1]
        self.assertNotEqual(latest.config_fingerprint, artifact.config_fingerprint)
        self.assertIn("Never edit retained sources.", third.request.system)
        # The summary cache may still reuse identical evidence. The final
        # primary request remains eligible under the original run allowance.
        provider.handler = lambda request, turn: ModelResponse(text="Retained evidence is sufficient.", usage=Usage(requests=1))
        budget.before_model_request()
        answer = await provider.complete(second.request)
        self.assertEqual(budget.model_requests, 3)
        self.assertEqual(answer.text, "Retained evidence is sufficient.")
