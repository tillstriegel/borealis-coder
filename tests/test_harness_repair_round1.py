"""Offline regressions for the independently reviewed harness repair delta."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from borealis_coder.agent import build_runner, extract_compaction_evidence
from borealis_coder.models import (
    Message,
    ModelResponse,
    ProviderRequest,
    Role,
    ToolCall,
    ToolResult,
    Usage,
)
from borealis_coder.providers.mock import MockProvider
from borealis_coder.tools import FunctionTool, object_schema
from tests.helpers import make_config


class HarnessRepairRoundOneTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    async def runner(self, root=None, **sections):
        root = root or self.root
        root.mkdir(exist_ok=True)
        runner = await build_runner(root, config=make_config(root, **sections), interactive=False)
        self.addAsyncCleanup(runner.close)
        provider = runner.providers[0].provider
        assert isinstance(provider, MockProvider)
        return runner, provider

    async def test_filename_path_and_command_intentions_are_repaired_before_publication(self):
        for index, intention in enumerate((
            "Let me inspect runner.py.", "I will read src/agent/runner.py:12.",
            "I'll check `python -m unittest tests.test_agent.AgentTests`.",
        )):
            with self.subTest(intention=intention):
                runner, provider = await self.runner(self.root / str(index))
                provider.enqueue(ModelResponse(text=intention), ModelResponse(text="The emitter is at runner.py:12; its behavior is untested."))
                events = []
                runner.events.subscribe(events.append)
                result = await runner.run("Locate the emitter.")
                self.assertEqual(provider.calls, 2)
                self.assertEqual(result.stop_reason.value, "end_turn")
                self.assertIn("runner.py:12", result.text)
                self.assertFalse(any(intention in event.data.get("text", "") for event in events))
                self.assertFalse(any(message.content == intention for message in runner.sessions.messages(result.session_id)))

    async def test_technical_prose_and_quoted_code_are_valid_conclusions(self):
        for index, answer in enumerate((
            "runner.py:12 emits the hint.", "Ja.", "已确认。", "<result>ok</result>",
            "```python\n# Let me inspect runner.py.\n```",
            "Let me check: runner.py returns 42.",
            "I will inspect runner.py. The existing test already proves the count is 42.",
        )):
            with self.subTest(answer=answer):
                runner, provider = await self.runner(self.root / str(index))
                provider.enqueue(ModelResponse(text=answer))
                result = await runner.run("Explain the finding briefly.")
                self.assertEqual(result.text, answer)
                self.assertEqual(provider.calls, 1)

    async def test_archived_quotes_use_exact_raw_or_decoded_envelope_text(self):
        quote = 'Größe "quoted" C:\\source\\file\nsecond line'
        for nested in (False, True):
            with self.subTest(nested=nested):
                runner, provider = await self.runner(self.root / str(nested), context={"tool_output_chars": 700})
                runner.tools.register(FunctionTool(
                    name="fixture_source", description="Synthetic archived evidence",
                    parameters=object_schema({}), function=lambda a, c: ToolResult("padding " * 1500 + quote + " trailing" * 1500),
                ))
                if nested:
                    source = ('const r = await tools.fixture_source({}); '
                              'await tools.record_finding({tool_call_id:r.metadata.output_artifact.source,quote:' + json.dumps(quote) + '}); text("Quote retained.");')
                    calls = [ToolCall(name="script", arguments={"source": source, "background": False})]
                    responses = [ModelResponse(tool_calls=calls), ModelResponse(text="The exact historical quote is retained.")]
                else:
                    responses = [
                        ModelResponse(tool_calls=[ToolCall(id="raw-source", name="fixture_source", arguments={})]),
                        ModelResponse(tool_calls=[ToolCall(name="record_finding", arguments={"tool_call_id": "raw-source", "quote": quote})]),
                        ModelResponse(text="The exact historical quote is retained."),
                    ]
                provider.enqueue(*responses)
                result = await runner.run("Keep the exact source quote.")
                rows = runner.sessions.tool_calls(result.session_id)
                finding = next(row for row in rows if row["tool_name"] == "record_finding")
                self.assertFalse(finding["is_error"], finding)
                self.assertEqual(finding["metadata"]["retained_finding"]["quote"], quote)
                raw = next(row for row in rows if row["tool_name"] == "fixture_source")
                self.assertNotIn(quote, raw["output"])

    async def test_nested_archive_does_not_confuse_literal_escape_with_newline(self):
        runner, provider = await self.runner(context={"tool_output_chars": 700})
        runner.tools.register(FunctionTool(name="fixture_source", description="Literal escapes", parameters=object_schema({}),
                                           function=lambda a, c: ToolResult("padding " * 1500 + "first\\nsecond")))
        source = ('const r=await tools.fixture_source({}); await tools.record_finding({'
                  'tool_call_id:r.metadata.output_artifact.source,quote:"first\\nsecond"});')
        provider.enqueue(ModelResponse(tool_calls=[ToolCall(name="script", arguments={"source": source, "background": False})]),
                         ModelResponse(text="The near-match was rejected."))
        result = await runner.run("Check exact quote matching.")
        row = next(row for row in runner.sessions.tool_calls(result.session_id) if row["tool_name"] == "record_finding")
        self.assertTrue(row["is_error"])
        self.assertNotIn("retained_finding", row["metadata"])

    @unittest.skipIf(os.name == "nt", "POSIX native read fixture")
    async def test_qualified_native_conclusion_has_one_visible_answer(self):
        runner, provider = await self.runner(agent={"auto_verify": False})
        provider.enqueue(
            ModelResponse(tool_calls=[ToolCall(name="shell", arguments={"command": "/usr/bin/printf observed", "cwd": ".", "timeout_seconds": None, "description": "Read fixture"})]),
            ModelResponse(text="The native probe returned observed."),
        )
        chunks, answers = [], {}
        def consume(event):
            if event.type == "model.text_delta":
                chunks.append(event.data["text"])
            if event.type == "model.completed" and event.data.get("text"):
                answers[event.data["message_id"]] = event.data["text"]
        runner.events.subscribe(consume)
        result = await runner.run("Inspect without edits.")
        self.assertEqual("".join(chunks).count("The native probe returned observed."), 1)
        self.assertEqual(list(answers.values()), [result.text])
        self.assertIn("Process lifecycle not guaranteed", result.text)
        durable = [m for m in runner.sessions.messages(result.session_id) if m.role == Role.ASSISTANT and not m.metadata.get("internal") and m.content]
        self.assertEqual([m.content for m in durable], [result.text])

    async def test_request_budget_checkpoint_answers_from_evidence_with_tools_available(self):
        runner, provider = await self.runner(agent={"max_model_requests": 8})
        requests = []
        def respond(request, turn):
            requests.append(request)
            if turn == 8:
                return ModelResponse(text="settings/app-05.json, settings/app-10.json, settings/app-20.json (3).")
            return ModelResponse(tool_calls=[ToolCall(name="fixture_evidence", arguments={"batch": turn})], usage=Usage(requests=1))
        runner.tools.register(FunctionTool(name="fixture_evidence", description="Synthetic completed evidence batches",
                                           parameters=object_schema({"batch": {"type": "integer"}}),
                                           function=lambda a, c: ToolResult("matches: app-05, app-10, app-20; count=3; batch=" + str(a["batch"]))))
        provider.handler = respond
        result = await runner.run("Find all enabled settings without timeout. Answer from complete evidence.")
        self.assertEqual(result.stop_reason.value, "end_turn", result.to_dict())
        self.assertEqual(result.usage.requests, 8)
        self.assertIn("Remaining logical requests including this request: 2", requests[6].system)
        self.assertIn("Remaining logical requests including this request: 1", requests[7].system)
        self.assertIn("Answer now", requests[6].system)
        self.assertTrue(requests[6].tools)
        self.assertTrue(requests[7].tools)
        self.assertEqual(result.changed_files, [])

    async def test_ignoring_budget_guidance_stops_without_an_extra_request(self):
        runner, provider = await self.runner(agent={"max_model_requests": 3})
        provider.handler = lambda request, turn: ModelResponse(tool_calls=[ToolCall(name="git_status", arguments={})])
        result = await runner.run("Inspect and answer.")
        self.assertEqual(provider.calls, 3)
        self.assertEqual(result.usage.requests, 3)
        self.assertTrue(result.incomplete)
        self.assertIn("maximum 3 model requests", result.text)

    async def test_small_request_cap_composes_settings_reads_then_answers_on_advisory(self):
        (self.root / "settings").mkdir()
        for index in range(24):
            setting = {"enabled": index in {5, 10, 20}, "filler": "RAW_FIXTURE_ONLY_729"}
            (self.root / f"settings/app-{index:02}.json").write_text(json.dumps(setting))
        runner, provider = await self.runner(agent={"max_model_requests": 8})
        runner.tools.register(FunctionTool(name="fixture_coverage", description="Synthetic distinct coverage batches", parameters=object_schema({"batch": {"type": "integer"}}), function=lambda a, c: ToolResult("Coverage batch " + str(a["batch"]))))
        source = '''const files=await tools.glob_files({pattern:"**/*.json",path:"settings",limit:100});
const rows=await Promise.all(files.output.split("\\n").map(path=>tools.read_file({path,start_line:null,end_line:null,max_chars:null})));
const matches=rows.filter(r=>{const v=JSON.parse(r.content);return v.enabled===true && !("timeout" in v);}).map(r=>r.path);
text({matches,count:matches.length});'''
        requests = []
        def respond(request, turn):
            requests.append(request)
            if turn == 1:
                return ModelResponse(tool_calls=[ToolCall(name="discover_tools", arguments={"names": ["glob_files", "read_file"], "offset": 0})])
            if turn == 2:
                return ModelResponse(tool_calls=[ToolCall(name="script", arguments={"source": source, "background": False})])
            if "# Remaining run budget" in request.system:
                return ModelResponse(text="settings/app-05.json, settings/app-10.json and settings/app-20.json (3).")
            return ModelResponse(tool_calls=[ToolCall(name="fixture_coverage", arguments={"batch": turn})])
        provider.handler = respond
        result = await runner.run("Inspect all 24 settings; report paths and count with enabled=true and no timeout. Ignore filler.")
        self.assertEqual(result.stop_reason.value, "end_turn", result.to_dict())
        self.assertEqual(provider.calls, 7)
        self.assertEqual(result.usage.requests, 7)
        self.assertTrue(requests[-1].tools)
        self.assertIn("(3)", result.text)
        rows = runner.sessions.tool_calls(result.session_id)
        self.assertEqual(sum(row["tool_name"] == "read_file" for row in rows), 24)
        self.assertFalse(any(row["is_error"] for row in rows))
        for request in requests[2:]:
            context = request.system + json.dumps([m.to_dict() for m in request.messages])
            self.assertNotIn("RAW_FIXTURE_ONLY_729", context)

    async def test_last_request_waits_for_background_result_with_tools_still_available(self):
        runner, provider = await self.runner(agent={"max_model_requests": 2})
        entered, release = asyncio.Event(), asyncio.Event()
        async def gate(a, c):
            entered.set()
            await release.wait()
            return ToolResult("settled evidence")
        runner.tools.register(FunctionTool(name="fixture_gate", description="Controlled last-request barrier", parameters=object_schema({}), function=gate))
        requests = []
        def respond(request, turn):
            requests.append(request)
            if turn == 1:
                return ModelResponse(tool_calls=[ToolCall(name="script", arguments={"source": 'const r=await tools.fixture_gate({});text(r.output);', "background": True})])
            self.assertTrue(release.is_set())
            notice = next(m for m in request.messages if m.metadata.get("internal") == "script_completion")
            self.assertIn("settled evidence", notice.content)
            return ModelResponse(text="The background evidence is settled.")
        provider.handler = respond
        run = asyncio.create_task(runner.run("Gather background evidence and answer within two requests."))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            self.assertEqual(provider.calls, 1)
            release.set()
            result = await asyncio.wait_for(run, 5)
            self.assertEqual(result.stop_reason.value, "end_turn", result.to_dict())
            self.assertEqual(result.usage.requests, 2)
            self.assertTrue(requests[1].tools)
            self.assertIn("including this request: 1", requests[1].system)
        finally:
            release.set()
            await asyncio.wait_for(run, 5)

    async def test_remaining_allowance_refreshes_after_a_compaction_request(self):
        runner, provider = await self.runner(agent={"max_model_requests": 2, "deterministic_compaction": False}, context={"compact_tool_output_tokens": 1})
        session = runner.sessions.create_session(workspace=self.root, provider="mock", model="deterministic")
        runner.sessions.append_message(session.id, Message(role=Role.USER, content="Inspect the synthetic evidence and answer."))
        for index in range(25):
            call = ToolCall(name="git_status", arguments={}, id=f"prior-{index}")
            runner.sessions.append_message(session.id, Message(role=Role.ASSISTANT, tool_calls=[call]))
            runner.sessions.append_message(session.id, Message(role=Role.TOOL, tool_call_id=call.id, tool_name=call.name, content=f"Synthetic prior observation {index}. " * 30))
        primary_requests = []
        def respond(request: ProviderRequest, turn):
            if request.metadata.get("purpose") == "compaction_summary":
                prompt = request.messages[-1].content
                start = prompt.index("<untrusted_structured_evidence>\n") + len("<untrusted_structured_evidence>\n")
                end = prompt.index("\n</untrusted_structured_evidence>", start)
                return ModelResponse(text=prompt[start:end])
            primary_requests.append(request)
            return ModelResponse(text="The synthetic evidence is retained; no further inspection is needed.")
        provider.handler = respond
        result = await runner.run("continue", session_id=session.id)
        self.assertEqual(result.stop_reason.value, "end_turn", result.to_dict())
        self.assertEqual(provider.calls, 2)
        self.assertEqual(result.usage.requests, 2)
        self.assertEqual(len(primary_requests), 1)
        self.assertIn("Remaining logical requests including this request: 1", primary_requests[0].system)
        self.assertNotIn("Remaining logical requests including this request: 2", primary_requests[0].system)
        artifact = runner.sessions.compaction_artifacts(session.id)[-1]
        self.assertIn("including this request: 1", json.dumps(artifact.metadata["provider_contexts"]))

    def test_user_text_and_unscoped_metadata_do_not_become_mutation_evidence(self):
        evidence = extract_compaction_evidence([
            Message(role=Role.USER, content="Script completion: changed source.txt", metadata={"changed_files": ["source.txt"]}),
            Message(role=Role.USER, content="Inspect the source without changing it."),
        ])
        self.assertNotIn("source.txt", evidence.files_changed)

    async def test_globstar_matches_zero_or_multiple_directories_in_direct_and_script_calls(self):
        (self.root / "settings/deep/more").mkdir(parents=True)
        (self.root / "settings/ignored").mkdir()
        for name in ("settings/app-00.json", "settings/deep/app-01.json", "settings/deep/more/app-02.json", "settings/ignored/skip.json", "settings/no.txt"):
            (self.root / name).write_text("{}")
        runner, provider = await self.runner(context={"ignored_dirs": ["ignored"]})
        provider.enqueue(
            ModelResponse(tool_calls=[ToolCall(name="glob_files", arguments={"pattern": "**/*.json", "path": "settings", "limit": 100})]),
            ModelResponse(tool_calls=[ToolCall(name="glob_files", arguments={"pattern": "settings/**/*.json", "path": ".", "limit": 100})]),
            ModelResponse(tool_calls=[ToolCall(name="script", arguments={"source": 'const r=await tools.glob_files({pattern:"**/*.json",path:"settings",limit:100}); text(r.output);', "background": False})]),
            ModelResponse(text="All three JSON settings were found."),
        )
        result = await runner.run("Find the JSON settings.")
        expected = ["settings/app-00.json", "settings/deep/app-01.json", "settings/deep/more/app-02.json"]
        rows = [r for r in runner.sessions.tool_calls(result.session_id) if r["tool_name"] == "glob_files"]
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertFalse(row["is_error"])
            self.assertEqual(row["output"].splitlines(), expected)

    async def test_glob_limits_exclusions_symlinks_and_plain_pattern_compatibility(self):
        (self.root / "settings/nested").mkdir(parents=True)
        (self.root / "settings/ignored").mkdir()
        for name in ("settings/z.json", "settings/a.json", "settings/nested/b.json", "settings/ignored/excluded.json"):
            (self.root / name).write_text("{}")
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        (Path(outside.name) / "outside.json").write_text("{}")
        if hasattr(os, "symlink"):
            try:
                (self.root / "settings/link.json").symlink_to(Path(outside.name) / "outside.json")
                (self.root / "settings/linked").symlink_to(Path(outside.name), target_is_directory=True)
            except OSError as error:
                self.skipTest(f"Symlink fixture is unavailable: {error}")
        runner, provider = await self.runner(context={"ignored_dirs": ["ignored"]})
        provider.enqueue(
            *[ModelResponse(tool_calls=[ToolCall(name="glob_files", arguments={"pattern": pattern, "path": "settings", "limit": limit})]) for pattern, limit in (("**/*.json", 2), ("*.json", 100), ("absent/**/*.json", 100), ("**/**/**/*.json", 100))],
            ModelResponse(text="Glob boundaries checked."),
        )
        result = await runner.run("Find bounded JSON paths without following links.")
        rows = [r for r in runner.sessions.tool_calls(result.session_id) if r["tool_name"] == "glob_files"]
        self.assertEqual(rows[0]["output"].splitlines(), ["settings/a.json", "settings/z.json"])
        self.assertTrue(rows[0]["metadata"]["truncated"])
        expected = ["settings/a.json", "settings/nested/b.json", "settings/z.json"]
        self.assertEqual(rows[1]["output"].splitlines(), expected)
        self.assertEqual(rows[2]["output"], "No matches")
        self.assertEqual(rows[2]["metadata"]["matches"], 0)
        self.assertEqual(rows[3]["output"].splitlines(), expected)

    async def test_background_completion_mutations_survive_compaction_and_reopen(self):
        for variant in ("complete", "partial", "no_change"):
            with self.subTest(variant=variant):
                await self.background_completion_case(variant)

    async def background_completion_case(self, variant):
        root = self.root / variant
        runner, provider = await self.runner(root, context={"compact_tool_output_tokens": 20})
        (root / "source.txt").write_text("original wiring\n")
        provider.enqueue(
            ModelResponse(tool_calls=[ToolCall(id="source-read", name="read_file", arguments={"path": "source.txt", "start_line": None, "end_line": None, "max_chars": None})]),
            ModelResponse(tool_calls=[ToolCall(name="record_finding", arguments={"tool_call_id": "source-read", "quote": "original wiring"})]),
            ModelResponse(text="The original wiring is recorded."),
        )
        first = await runner.run("Inspect source wiring and retain the observation.")
        runner.sessions.append_message(first.session_id, Message(role=Role.ASSISTANT, content="Original revision check passed.", metadata={"authoritative_verification": True}))
        entered, release, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def gate(a, c):
            entered.set()
            await release.wait()
            return ToolResult("released")
        runner.tools.register(FunctionTool(name="fixture_gate", description="Controlled barrier", parameters=object_schema({}), function=gate))
        runner.events.subscribe(lambda e: completed.set() if e.type == "script.completed" else None)
        source = 'await tools.fixture_gate({}); '
        if variant != "no_change":
            digest = __import__("hashlib").sha256((root / "source.txt").read_bytes()).hexdigest()
            source += 'await tools.write_file({path:"source.txt",content:"new wiring\\n",expected_sha256:' + json.dumps(digest) + '}); '
        source += 'throw Error("partial fixture failure");' if variant == "partial" else 'text("job complete");'
        requests = []
        async def respond(request):
            requests.append(request)
            if len(requests) == 1:
                return ModelResponse(tool_calls=[ToolCall(name="script", arguments={"source": source, "background": True})])
            if len(requests) == 2:
                await asyncio.wait_for(entered.wait(), 3)
                release.set()
                await asyncio.wait_for(completed.wait(), 3)
            return ModelResponse(text="Background work settled; the prior observation is historical.")
        provider.complete = AsyncMock(side_effect=respond)
        second = await runner.run("Continue with the controlled background job.", session_id=first.session_id)
        self.assertEqual((root / "source.txt").read_text(), "original wiring\n" if variant == "no_change" else "new wiring\n")
        notices = [m for m in runner.sessions.messages(second.session_id) if m.metadata.get("internal") == "script_completion"]
        self.assertEqual(len(notices), 1)
        self.assertEqual(notices[0].is_error, variant == "partial")
        evidence = extract_compaction_evidence(runner.sessions.messages(second.session_id))
        if variant != "no_change":
            self.assertIn("source.txt", evidence.files_changed)
            self.assertIn("stale", " ".join(evidence.important_decisions))
            self.assertIn("Stale:", " ".join(evidence.latest_verification))
        else:
            self.assertNotIn("source.txt", evidence.files_changed)
            self.assertNotIn("Stale:", " ".join(evidence.latest_verification))
        before_rows = runner.sessions.tool_calls(second.session_id)
        before_usage = runner.sessions.usage(second.session_id).requests
        self.assertTrue(runner.sessions.compaction_artifacts(second.session_id))
        config = runner.config
        await runner.close()
        reopened = await build_runner(root, config=config, interactive=False)
        self.addAsyncCleanup(reopened.close)
        resumed_provider = reopened.providers[0].provider
        assert isinstance(resumed_provider, MockProvider)
        resumed_requests = []
        resumed_provider.handler = lambda request, turn, seen=resumed_requests: seen.append(request) or ModelResponse(text="Historical state retained.")
        resumed = await reopened.run("continue", session_id=second.session_id)
        self.assertEqual(resumed.stop_reason.value, "end_turn")
        if variant != "no_change":
            self.assertIn("stale", resumed_requests[0].system)
            self.assertIn("Stale:", resumed_requests[0].system)
            self.assertIn("source.txt", resumed_requests[0].system)
        self.assertEqual(len(reopened.sessions.tool_calls(second.session_id)), len(before_rows))
        self.assertEqual(reopened.sessions.usage(second.session_id).requests, before_usage + 1)
