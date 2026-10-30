from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from borealis_coder.agent import build_runner, validate_tool_call_order
from borealis_coder.config import MCPServerConfig
from borealis_coder.errors import BudgetExceeded, ConfigurationError, SessionError, ToolError
from borealis_coder.models import Effect, ModelResponse, Role, StopReason, ToolCall, ToolResult
from borealis_coder.providers.mock import MockProvider
from borealis_coder.safety import PolicyEngine
from borealis_coder.tools.base import FunctionTool, object_schema
from borealis_coder.tools.script import decode_packet
from tests.helpers import make_config


class ScriptIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = make_config(self.root)
        self.runner = await build_runner(self.root, config=self.config, interactive=False)
        self.addAsyncCleanup(self.runner.close)
        provider = self.runner.providers[0].provider
        assert isinstance(provider, MockProvider)
        self.provider = provider

    async def run_script(self, source, *, background=False):
        self.provider.enqueue(
            ModelResponse(tool_calls=[ToolCall(name="script", arguments={"source": source, "background": background})]),
            ModelResponse(text="Done"),
        )
        return await self.runner.run("Inspect the synthetic fixtures")

    async def test_real_runner_filters_full_data_without_provider_payloads(self):
        secret_payload = "PRIVATE_FIXTURE_BODY_" + "x" * 30000
        (self.root / "one.txt").write_text(secret_payload + "\nneedle\n")
        (self.root / "two.txt").write_text(secret_payload + "\nother\n")
        requests = []
        source = '''
const found = await tools.grep({pattern:"PRIVATE_FIXTURE_BODY",path:".",glob:"*.txt",regex:false,case_sensitive:true,context_lines:0,max_results:10});
const files = await Promise.all(found.matches.map(m => tools.read_file({path:m.path,start_line:null,end_line:null,max_chars:null})));
text({files: files.length, selected:files.filter(f => f.content.includes("needle")).length});
'''
        def handler(request, count):
            requests.append(request)
            if count == 1:
                return ModelResponse(tool_calls=[ToolCall(name="script", arguments={"source": source, "background": False})])
            return ModelResponse(text="Two files, one selected")
        self.provider.handler = handler
        result = await self.runner.run("Count the fixture files")
        self.assertIsNone(result.error)
        self.assertEqual(self.provider.calls, 2)
        next_messages = requests[1].messages
        tool = next(m for m in next_messages if m.role == Role.TOOL)
        self.assertFalse(tool.is_error, tool.content)
        self.assertIn('"selected":1', tool.content.replace(" ", ""))
        self.assertNotIn(secret_payload, json.dumps([m.to_dict() for m in next_messages]))
        exported = self.runner.sessions.export(result.session_id)
        self.assertTrue(any(secret_payload in a["content"] for a in exported["output_artifacts"]))
        self.assertEqual(len(exported["tool_calls"]), 4)
        self.assertTrue(all(row["status"] == "completed" for row in exported["tool_calls"]))
        validate_tool_call_order(self.runner.sessions.messages(result.session_id))

    def script_message(self, result):
        return next(m for m in self.runner.sessions.messages(result.session_id)
                    if m.role == Role.TOOL and m.tool_name == "script")

    def register(self, name, function, *, effect=Effect.READ, concurrent=True, scope=None):
        self.runner.tools.register(FunctionTool(name=name, description="Synthetic fixture tool",
            parameters=object_schema({}), function=function, effect=effect,
            concurrent=concurrent, mutation_scope=scope))

    async def test_dependency_chain_preserves_literal_multiline_and_hash_checkpoint(self):
        original = "Original\n" * 5000
        (self.root / "input.txt").write_text(original)
        self.config.agent.auto_verify = True
        source = r'''
const before = await tools.read_file({path:"input.txt", start_line:null,end_line:null,max_chars:null});
const literal = String.raw`line one
"quotes", backslash: \ and ${"interpolation"}
last line
`;
const edit = await tools.write_file({path:"input.txt",content:before.content + literal,expected_sha256:before.sha256});
const after = await tools.read_file({path:"input.txt",start_line:null,end_line:null,max_chars:null});
text({same:after.content === before.content + literal,short:edit.output.length < 200,checkpoint:!!edit.metadata.checkpoint_id});
'''
        result = await self.run_script(source)
        message = self.script_message(result)
        self.assertFalse(message.is_error, message.content)
        self.assertIn('"same":true', message.content)
        self.assertIn('"short":true', message.content)
        self.assertIn('"checkpoint":true', message.content)
        self.assertEqual((self.root / "input.txt").read_text(), original + 'line one\n"quotes", backslash: \\ and interpolation\nlast line\n')
        self.assertIn("input.txt", result.changed_files)
        self.assertIsNotNone(result.verification)
        self.assertEqual(result.mutation_tracking, "complete")

    async def test_promise_all_is_concurrent_and_host_bounded(self):
        entered = 0
        maximum = 0
        barrier = asyncio.Event()
        release = asyncio.Event()
        async def read(arguments, context):
            nonlocal entered, maximum
            entered += 1
            maximum = max(maximum, entered)
            if entered == 2:
                barrier.set()
            await release.wait()
            entered -= 1
            return ToolResult("fixture")
        self.config.script.parallel_reads = 2
        self.register("fixture_read", read)
        task = asyncio.create_task(self.run_script('const values = await Promise.all(Array.from({length:9}, () => tools.fixture_read({}))); text(values.length);'))
        await asyncio.wait_for(barrier.wait(), 3)
        self.assertEqual(entered, 2)
        release.set()
        result = await asyncio.wait_for(task, 5)
        self.assertEqual(maximum, 2)
        self.assertFalse(self.script_message(result).is_error)
        self.assertIn("\n9\n", self.script_message(result).content)

    async def test_nested_mutations_serialize_even_for_concurrent_plugin(self):
        entered = 0
        maximum = 0
        async def mutate(arguments, context):
            nonlocal entered, maximum
            entered += 1
            maximum = max(maximum, entered)
            await asyncio.sleep(0)
            (context.workspace / "plugin.txt").write_text("changed")
            entered -= 1
            return ToolResult("short")
        self.register("mcp__fixture__mutate", mutate, effect=Effect.WRITE)
        result = await self.run_script('await Promise.all(Array.from({length:6}, () => tools.mcp__fixture__mutate({}))); text("done");')
        self.assertEqual(maximum, 1)
        self.assertIn("plugin.txt", result.changed_files)
        self.assertFalse(self.script_message(result).is_error)
        rows = self.runner.sessions.tool_calls(result.session_id)
        self.assertTrue(all(row["run_id"] == result.run_id for row in rows))
        self.assertTrue(all(row["metadata"].get("script_job_id") for row in rows if row["tool_call_id"].startswith("nested_")))

    async def test_caught_tool_errors_are_counted_and_schema_checks_still_apply(self):
        result = await self.run_script('''
for (const args of [{path:42}, {path:"missing.txt",start_line:null,end_line:null,max_chars:null}]) {
  try { await tools.read_file(args); } catch (e) { text(e.result.is_error); }
}
try { await tools.no_such_tool({}); } catch (e) { text(e.message); }
text("continued");
''')
        message = self.script_message(result)
        self.assertTrue(message.is_error)
        self.assertIn('"tool_errors":3', message.content)
        self.assertIn("continued", message.content)
        self.assertIn("Unknown tool", message.content)
        self.assertTrue(all(r["status"] == "error" for r in self.runner.sessions.tool_calls(result.session_id)))
        healthy = await self.run_script('text("healthy");')
        self.assertFalse(self.script_message(healthy).is_error)

    async def test_policy_gates_inside_script(self):
        self.config.safety.mode = "plan"
        result = await self.run_script('try { await tools.write_file({path:"blocked.txt",content:"no",expected_sha256:null}); } catch(e) { text(e.message); }')
        self.assertIn("plan mode", self.script_message(result).content)
        self.assertFalse((self.root / "blocked.txt").exists())
        self.config.safety.mode = "workspace-write"
        self.config.safety.allow_shell = False
        result = await self.run_script('''
const actions = [
  () => tools.shell({command:"echo no",cwd:".",timeout_seconds:1,description:"fixture"}),
  () => tools.fetch_url({url:"https://example.com",max_chars:100}),
  () => tools.git_commit({message:"no"}),
  () => tools.git_push({remote:"origin",branch:null})
];
for (const action of actions) { try { await action(); } catch(e) { text(e.message); } }
''')
        message = self.script_message(result)
        self.assertTrue(message.is_error)
        self.assertIn("shell execution is disabled", message.content)
        self.assertIn("network access is disabled", message.content)
        self.assertIn("git commits are disabled", message.content)
        self.config.safety.network = True
        push = await self.run_script('try { await tools.git_push({remote:"origin",branch:null}); } catch(e) {text(e.message);}')
        self.assertIn("git push requires", self.script_message(push).content)
        self.assertEqual(len(self.runner.sessions.tool_calls(result.session_id)), 5)

    async def test_paths_symlink_and_stale_write_stay_guarded(self):
        outside = self.root.parent / (self.root.name + "-outside.txt")
        outside.write_text("outside")
        self.addCleanup(outside.unlink)
        (self.root / "link.txt").symlink_to(outside)
        (self.root / "existing.txt").write_text("before")
        result = await self.run_script('''
for (const path of [".git/config", "link.txt", "../escape.txt", "existing.txt"]) {
 try { await tools.write_file({path,content:"no",expected_sha256:null}); } catch(e) { text(e.message); }
}
''')
        self.assertTrue(self.script_message(result).is_error)
        self.assertEqual((self.root / "existing.txt").read_text(), "before")
        self.assertEqual(outside.read_text(), "outside")
        self.assertFalse((self.root / ".git/config").exists())

    async def test_engine_has_no_native_host_access(self):
        result = await self.run_script('text([typeof process,typeof require,typeof fetch,typeof std,typeof os,typeof __import__]);')
        self.assertFalse(self.script_message(result).is_error)
        self.assertIn('["undefined","undefined","undefined","undefined","undefined","undefined"]', self.script_message(result).content)

    async def test_syntax_runtime_cpu_memory_output_and_call_limits_recover(self):
        self.config.script.wall_seconds = 0.4
        self.config.script.max_calls = 3
        cases = [
            "const = ;", 'throw Error("runtime fixture");',
            "while(true) {}", "let a=[]; while(true) a.push(new Array(100000).fill(1));",
            'text("x".repeat(30000));',
            'await Promise.all(Array.from({length:4},()=>tools.fixture_read({})));',
            'await tools.script({source:"text(1)",background:false});',
            'await new Promise(()=>{});',
        ]
        called = []
        self.register("fixture_read", lambda a,c: (called.append(1) or ToolResult("yes")))
        for source in cases:
            with self.subTest(source=source):
                result = await asyncio.wait_for(self.run_script(source), 4)
                self.assertTrue(self.script_message(result).is_error)
                self.assertFalse(self.runner._script_scopes)
        self.assertEqual(called, [])
        result = await self.run_script('text("recovered");')
        self.assertFalse(self.script_message(result).is_error)

    async def test_ipc_depth_and_result_limits_refuse_then_recover(self):
        for source in [
            'let x={}; let p=x; for(let i=0;i<40;i++) { p.x={};p=p.x; } await tools.read_file(x);',
            'await tools.read_file({path:"x".repeat(9000000)});',
        ]:
            result = await self.run_script(source)
            self.assertTrue(self.script_message(result).is_error)
        self.config.script.result_bytes = 1000
        self.register("large_fixture", lambda a,c: ToolResult("r"*2000))
        result = await self.run_script('try {await tools.large_fixture({});} catch(e) {text(e.message);}')
        self.assertIn("size/type/depth", self.script_message(result).content)
        result = await self.run_script('text("okay");')
        self.assertFalse(self.script_message(result).is_error)

    async def test_redacted_artifacts_remain_session_owned(self):
        token = "sk-" + "f"*24
        (self.root / "private.txt").write_text("fixture " + token)
        result = await self.run_script('const f=await tools.read_file({path:"private.txt",start_line:null,end_line:null,max_chars:null});text(f.content);')
        exported = self.runner.sessions.export(result.session_id)
        self.assertNotIn(token, json.dumps(exported))
        artifacts = exported["output_artifacts"]
        self.assertTrue(artifacts)
        self.assertIn("[REDACTED]", artifacts[0]["content"])
        another = self.runner.sessions.create_session(workspace=self.root,provider="mock",model="fixture")
        with self.assertRaises(SessionError):
            self.runner.sessions.read_output_artifact(another.id,self.root,artifacts[0]["artifact_id"])

    async def test_background_returns_handle_and_completion_wakes_without_polling(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        async def wait_fixture(args,context):
            entered.set()
            await release.wait()
            return ToolResult("released")
        self.register("wait_fixture",wait_fixture)
        requests = []
        def handler(request,count):
            requests.append(request)
            if count == 1:
                return ModelResponse(tool_calls=[ToolCall(name="script",arguments={"source":'await tools.wait_fixture({});text("background complete");',"background":True})])
            if count == 2:
                self.assertTrue(entered.is_set())
                release.set()
                return ModelResponse(text="early candidate")
            self.assertTrue(any(m.metadata.get("internal") == "script_completion" and "background complete" in m.content for m in request.messages))
            return ModelResponse(text="complete after notification")
        original_stream = self.provider.stream
        async def stream(request):
            if self.provider.calls == 1:
                await asyncio.wait_for(entered.wait(), 3)
            async for event in original_stream(request):
                yield event
        self.provider.stream = stream
        self.provider.handler=handler
        result = await asyncio.wait_for(self.runner.run("Run background fixture"),5)
        self.assertEqual(result.text,"complete after notification")
        self.assertEqual(self.provider.calls,3)
        first = next(m for m in requests[1].messages if m.role == Role.TOOL)
        self.assertEqual(first.metadata["script_job"]["state"],"running")
        self.assertFalse(any(c.name=="script_job" for r in requests for m in r.messages for c in m.tool_calls))
        self.assertFalse(self.runner._script_scopes)

    async def test_foreground_threshold_returns_live_job(self):
        self.config.script.foreground_seconds=0.01
        entered=asyncio.Event()
        release=asyncio.Event()
        self.register("wait_fixture",lambda a,c: wait())
        async def wait():
            entered.set()
            await release.wait()
            return ToolResult("ready")
        def handler(request,count):
            if count == 1:
                return ModelResponse(tool_calls=[ToolCall(name="script",arguments={"source":'await tools.wait_fixture({});text("late");',"background":False})])
            if count == 2:
                first=next(m for m in request.messages if m.role == Role.TOOL)
                self.assertEqual(first.metadata["script_job"]["state"],"running")
                release.set()
            return ModelResponse(text="done")
        self.provider.handler=handler
        result=await asyncio.wait_for(self.runner.run("Threshold fixture"),5)
        self.assertIsNone(result.error)
        self.assertTrue(entered.is_set())

    async def test_stop_drains_active_call_and_retains_partial_mutation(self):
        entered=asyncio.Event()
        ended=asyncio.Event()
        async def mutate(args,context):
            (context.workspace/"partial.txt").write_text("partial")
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                ended.set()
        self.register("mutate_fixture",mutate,effect=Effect.WRITE)
        def handler(request,count):
            if count==1:
                return ModelResponse(tool_calls=[ToolCall(name="script",arguments={"source":'await tools.mutate_fixture({});await tools.write_file({path:"later.txt",content:"bad",expected_sha256:null});',"background":True})])
            if count==2:
                first=next(m for m in request.messages if m.role==Role.TOOL)
                return ModelResponse(tool_calls=[ToolCall(name="script_job",arguments={"job_id":first.metadata["script_job"]["job_id"],"action":"stop","seconds":0})])
            return ModelResponse(text="stopped")
        # Ensure the child is admitted before the provider requests stop.
        original_stream=self.provider.stream
        async def stream(request):
            if self.provider.calls==1:
                await asyncio.wait_for(entered.wait(),3)
            async for event in original_stream(request):
                yield event
        self.provider.stream=stream
        self.provider.handler=handler
        result=await asyncio.wait_for(self.runner.run("Stop fixture"),5)
        self.assertTrue(ended.is_set())
        self.assertIn("partial.txt",result.changed_files)
        self.assertFalse((self.root/"later.txt").exists())
        self.assertFalse(any(r["status"]=="running" for r in self.runner.sessions.tool_calls(result.session_id)))

    async def test_runner_cancel_reaps_jobs_and_repeated_cancel_is_safe(self):
        entered=asyncio.Event()
        ended=asyncio.Event()
        async def block(args,context):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                ended.set()
        self.register("block_fixture",block)
        self.provider.enqueue(ModelResponse(tool_calls=[ToolCall(name="script",arguments={"source":'await tools.block_fixture({});',"background":False})]))
        task=asyncio.create_task(self.runner.run("Cancel fixture"))
        await asyncio.wait_for(entered.wait(),3)
        session_id=next(iter(self.runner._cancel))
        scopes=list(self.runner._script_scopes)
        processes=[j.process for s in scopes for j in s.jobs.values()]
        self.runner.cancel(session_id)
        self.runner.cancel(session_id)
        result=await asyncio.wait_for(task,5)
        self.assertEqual(result.stop_reason,StopReason.CANCELLED)
        self.assertTrue(ended.is_set())
        self.assertTrue(all(p is not None and p.returncode is not None for p in processes))
        self.assertFalse(self.runner._script_scopes)
        validate_tool_call_order(self.runner.sessions.messages(session_id))

    async def test_disabled_palette_and_workspace_config_authority(self):
        self.config.script.enabled=False
        runner=await build_runner(self.root,config=self.config,interactive=False)
        try:
            self.assertNotIn("script",runner.tools.names())
            self.assertNotIn("script_job",runner.tools.names())
        finally:
            await runner.close()
        (self.root/".borealis").mkdir(exist_ok=True)
        (self.root/".borealis/config.toml").write_text("[script]\nmax_calls=1000000\n")
        with self.assertRaises(ConfigurationError):
            make_config(self.root)

    async def test_cpu_budget_is_distinct_from_wall_deadline(self):
        self.config.script.cpu_seconds = 1
        self.config.script.wall_seconds = 10
        result = await asyncio.wait_for(self.run_script("while(true) {}"), 12)
        message = self.script_message(result)
        self.assertTrue(message.is_error)
        self.assertNotIn("wall time", message.content)
        healthy = await self.run_script('text("healthy after CPU interrupt");')
        self.assertFalse(self.script_message(healthy).is_error)

    async def test_timeout_drains_effects_and_does_not_run_continuation(self):
        self.config.script.wall_seconds = 0.2
        entered = asyncio.Event()
        ended = asyncio.Event()
        async def mutate(args, context):
            (context.workspace / "partial.txt").write_text("retained")
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                ended.set()
        self.register("timeout_fixture", mutate, effect=Effect.WRITE)
        result = await self.run_script('await tools.timeout_fixture({}); await tools.write_file({path:"after.txt",content:"no",expected_sha256:null});')
        self.assertTrue(entered.is_set())
        self.assertTrue(ended.is_set())
        self.assertIn("partial.txt", result.changed_files)
        self.assertFalse((self.root / "after.txt").exists())
        self.assertIn("wall time", self.script_message(result).content)
        self.assertFalse(self.runner._script_job_slots)
        self.assertTrue(all(r["status"] != "running" for r in self.runner.sessions.tool_calls(result.session_id)))

    async def test_budget_errors_cannot_be_caught_to_continue_effects(self):
        def exhausted(args, context):
            raise BudgetExceeded("fixture", "Synthetic run budget exhausted")
        self.register("budget_fixture", exhausted)
        result = await self.run_script('try {await tools.budget_fixture({});} catch(e) {} await tools.write_file({path:"escaped.txt",content:"no",expected_sha256:null});')
        self.assertEqual(result.stop_reason, StopReason.BUDGET)
        self.assertFalse((self.root / "escaped.txt").exists())
        self.assertFalse(self.runner._script_job_slots)
        self.assertTrue(all(r["status"] != "running" for r in self.runner.sessions.tool_calls(result.session_id)))
        validate_tool_call_order(self.runner.sessions.messages(result.session_id))

    async def test_run_deadline_and_close_reap_owned_subprocesses(self):
        for close_runner in (False, True):
            with self.subTest(close_runner=close_runner):
                entered = asyncio.Event()
                ended = asyncio.Event()
                async def block(args, context, entered=entered, ended=ended):
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        ended.set()
                name = "close_fixture" if close_runner else "deadline_fixture"
                self.register(name, block)
                self.config.agent.max_time_seconds = 5 if close_runner else 1
                self.provider.enqueue(ModelResponse(tool_calls=[ToolCall(name="script", arguments={"source":f'await tools.{name}({{}});',"background":False})]))
                running = asyncio.create_task(self.runner.run("Lifetime fixture"))
                await asyncio.wait_for(entered.wait(), 3)
                processes = [j.process for s in self.runner._script_scopes for j in s.jobs.values()]
                if close_runner:
                    await asyncio.wait_for(self.runner.close(), 5)
                result = await asyncio.wait_for(running, 5)
                self.assertTrue(ended.is_set())
                self.assertTrue(all(p is not None and p.returncode is not None for p in processes))
                self.assertEqual(result.stop_reason, StopReason.CANCELLED if close_runner else StopReason.BUDGET)
                self.assertFalse(self.runner._script_job_slots)

    async def test_stop_status_wait_and_exact_run_ownership(self):
        saved = ""
        def handler(request, count):
            nonlocal saved
            if count == 1:
                return ModelResponse(tool_calls=[ToolCall(name="script",arguments={"source":'text("complete");',"background":False})])
            if count == 2:
                saved = next(m for m in request.messages if m.role == Role.TOOL).metadata["script_job"]["job_id"]
                return ModelResponse(tool_calls=[ToolCall(name="script_job", arguments={"job_id":saved,"action":"wait","seconds":1})])
            return ModelResponse(text="done")
        self.provider.handler = handler
        first = await self.runner.run("Owned handle")
        self.provider.handler = None
        self.provider.enqueue(ModelResponse(tool_calls=[ToolCall(name="script_job",arguments={"job_id":saved,"action":"status","seconds":0})]), ModelResponse(text="refused"))
        second = await self.runner.run("Previous run cannot be controlled",session_id=first.session_id)
        last_tool = [m for m in self.runner.sessions.messages(second.session_id) if m.role==Role.TOOL][-1]
        self.assertTrue(last_tool.is_error)
        self.assertIn("unavailable in this run/session",last_tool.content)
        validate_tool_call_order(self.runner.sessions.messages(second.session_id))

    async def test_source_archive_and_concurrent_job_limits(self):
        self.config.script.source_bytes=10
        result=await self.run_script('text("too large source");')
        self.assertIn("source byte limit",self.script_message(result).content)
        self.config.script.source_bytes=256000
        self.config.script.archive_bytes=30
        self.register("archive_fixture",lambda a,c: ToolResult("long fixture value"*20))
        result=await self.run_script('await tools.archive_fixture({}); await tools.archive_fixture({}); text("okay");')
        artifacts=self.runner.sessions.export(result.session_id)["output_artifacts"]
        self.assertLessEqual(sum(a["stored_bytes"] for a in artifacts),30)
        self.assertTrue(all(a["status"]=="quota_limited" for a in artifacts))
        self.config.script.max_jobs=1
        entered=asyncio.Event()
        release=asyncio.Event()
        async def block(args,context):
            entered.set()
            await release.wait()
            return ToolResult("ready")
        self.register("slot_fixture",block)
        calls_start = self.provider.calls
        def handler(request,count):
            if count==calls_start+1:
                return ModelResponse(tool_calls=[ToolCall(name="script",arguments={"source":'await tools.slot_fixture({});',"background":True})])
            if count==calls_start+2:
                return ModelResponse(tool_calls=[ToolCall(name="script",arguments={"source":'text("second");',"background":True})])
            release.set()
            return ModelResponse(text="done")
        # The first worker owns its slot even before it reaches the tool barrier.
        self.provider.handler=handler
        result=await self.runner.run("Limit concurrent jobs")
        messages=[m for m in self.runner.sessions.messages(result.session_id) if m.role==Role.TOOL]
        self.assertTrue(messages[1].is_error)
        self.assertIn("concurrent job limit",messages[1].content)
        self.assertFalse(self.runner._script_job_slots)

    async def test_approval_channel_is_applied_to_nested_effects(self):
        self.config.safety.approval="on-risk"
        self.runner.tool_context.policy=PolicyEngine(self.config.safety,interactive=True)
        requests=[]
        self.runner.tool_context.approvals.callback=lambda request: (requests.append(request.tool_name) or False)
        tool=FunctionTool(name="approval_fixture",description="Synthetic gated effect",parameters=object_schema({}),
            function=lambda a,c: ToolResult("must not execute"),effect=Effect.WRITE,default_risk="high")
        self.runner.tools.register(tool)
        result=await self.run_script('try {await tools.approval_fixture({});} catch(e) {text(e.message);}')
        self.assertEqual(requests,["approval_fixture"])
        self.assertIn("Approval denied",self.script_message(result).content)

    async def test_real_mcp_stdio_tool_uses_same_bridge(self):
        from tests.test_mcp import SERVER
        self.config.mcp_servers["fixture"]=MCPServerConfig(command=sys.executable,args=["-c",SERVER],read_only_tools=["echo"])
        runner=await build_runner(self.root,config=self.config,interactive=False)
        try:
            provider=runner.providers[0].provider
            assert isinstance(provider,MockProvider)
            provider.enqueue(ModelResponse(tool_calls=[ToolCall(name="script",arguments={"source":'const r=await tools.mcp__fixture__echo({text:"literal\\nfixture"}); text(r.output);',"background":False})]), ModelResponse(text="done"))
            result=await runner.run("Synthetic MCP fixture")
            message=next(m for m in runner.sessions.messages(result.session_id) if m.role==Role.TOOL)
            self.assertFalse(message.is_error,message.content)
            self.assertIn("echo:literal\nfixture",message.content)
            self.assertEqual(len(runner.sessions.tool_calls(result.session_id)),2)
        finally:
            await runner.close()

    async def test_resume_marks_abandoned_nested_effects_without_replay(self):
        session=self.runner.sessions.create_session(workspace=self.root,provider="mock",model="fixture")
        self.runner.sessions.start_tool_call(session.id,"previous_run","nested_interrupted","write_file",{"path":"uncertain.txt"},metadata={"script_job_id":"previous_job","script_parent_id":"previous_script"})
        self.provider.enqueue(ModelResponse(text="inspect effects before retry"))
        result=await self.runner.run("Continue",session_id=session.id)
        row=self.runner.sessions.tool_calls(session.id)[0]
        self.assertEqual(row["status"],"error")
        self.assertTrue(row["metadata"]["unknown_outcome"])
        self.assertEqual(result.mutation_tracking,"incomplete")
        self.assertFalse((self.root/"uncertain.txt").exists())
        validate_tool_call_order(self.runner.sessions.messages(session.id))


    async def test_worker_crash_drains_active_tool_without_continuing_effects(self):
        entered=asyncio.Event()
        ended=asyncio.Event()
        async def mutate(args,context):
            (context.workspace/"partial.txt").write_text("retained")
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                ended.set()
        self.register("crash_fixture",mutate,effect=Effect.WRITE)
        running=asyncio.create_task(self.run_script('await tools.crash_fixture({});await tools.write_file({path:"escaped.txt",content:"no",expected_sha256:null});'))
        await asyncio.wait_for(entered.wait(),3)
        process=next(j.process for scope in self.runner._script_scopes for j in scope.jobs.values())
        assert process is not None
        process.kill()
        result=await asyncio.wait_for(running,5)
        self.assertTrue(ended.is_set())
        self.assertIsNotNone(process.returncode)
        self.assertTrue(self.script_message(result).is_error)
        self.assertFalse((self.root/"escaped.txt").exists())
        self.assertIn("partial.txt",result.changed_files)
        self.assertTrue(all(r["status"]!="running" for r in self.runner.sessions.tool_calls(result.session_id)))

    async def test_repeated_caller_cancellation_during_spawn_reaps_worker(self):
        launched=asyncio.Event()
        release=asyncio.Event()
        processes=[]
        real_spawn=asyncio.create_subprocess_exec
        async def delayed_spawn(*args,**kwargs):
            process=await real_spawn(*args,**kwargs)
            processes.append(process)
            launched.set()
            await release.wait()
            return process
        with patch("borealis_coder.tools.script.asyncio.create_subprocess_exec",side_effect=delayed_spawn):
            task=asyncio.create_task(self.run_script('await tools.write_file({path:"escaped.txt",content:"no",expected_sha256:null});'))
            await asyncio.wait_for(launched.wait(),3)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            release.set()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(task,5)
        self.assertTrue(all(p.returncode is not None for p in processes))
        self.assertFalse(self.runner._run_tasks)
        self.assertFalse(self.runner._script_job_slots)
        self.assertFalse((self.root/"escaped.txt").exists())

    async def test_background_mutation_settles_before_auto_verification_and_final_text(self):
        self.config.agent.auto_verify=True
        entered=asyncio.Event()
        release=asyncio.Event()
        ended=asyncio.Event()
        observed=[]
        async def gate(args,context):
            entered.set()
            await release.wait()
            return ToolResult("continue")
        self.register("gate_fixture",gate)
        def observe(event):
            if event.type=="tool.completed" and event.data["tool"]=="write_file":
                ended.set()
            if event.type=="verification.started":
                observed.append(ended.is_set())
            if event.type=="model.text_delta":
                self.assertNotIn("early candidate",event.data.get("text",""))
        self.runner.events.subscribe(observe)
        def handler(request,count):
            if count==1:
                return ModelResponse(tool_calls=[ToolCall(name="script",arguments={"source":'await tools.gate_fixture({}); await tools.write_file({path:"late.txt",content:"complete",expected_sha256:null});text("mutation settled");',"background":True})])
            if count==2:
                release.set()
                return ModelResponse(text="early candidate")
            return ModelResponse(text="finished after verification")
        self.provider.handler=handler
        result=await asyncio.wait_for(self.runner.run("Background write fixture"),5)
        self.assertTrue(ended.is_set())
        self.assertTrue(observed)
        self.assertTrue(all(observed))
        self.assertIn("late.txt",result.changed_files)
        self.assertIsNotNone(result.verification)
        self.assertNotEqual(result.text,"early candidate")
        self.assertEqual((self.root/"late.txt").read_text(),"complete")

    async def test_hundreds_of_real_nested_calls_share_one_model_invocation(self):
        called=[]
        self.register("count_fixture",lambda a,c: (called.append(c.tool_call_id) or ToolResult("fixture payload")))
        result=await self.run_script('const r=await Promise.all(Array.from({length:220},()=>tools.count_fixture({})));text({count:r.length});')
        self.assertEqual(len(called),220)
        self.assertEqual(len(set(called)),220)
        self.assertEqual(self.provider.calls,2)
        self.assertIn('"count":220',self.script_message(result).content)
        self.assertFalse(self.script_message(result).is_error)
        self.assertEqual(len(self.runner.sessions.tool_calls(result.session_id)),221)
        self.assertFalse(self.runner._script_job_slots)


    async def test_full_permitted_file_survives_json_escaping(self):
        (self.root/"escaped.txt").write_text("\x01"*1_900_000)
        result=await self.run_script('const f=await tools.read_file({path:"escaped.txt",start_line:null,end_line:null,max_chars:100});text(f.content.length);')
        message=self.script_message(result)
        self.assertFalse(message.is_error,message.content)
        self.assertIn("1900000",message.content)
        self.assertTrue(any(a["status"]=="quota_limited" for a in self.runner.sessions.export(result.session_id)["output_artifacts"]))

    async def test_structured_result_depth_is_checked_before_recursive_redaction(self):
        value={}
        cursor=value
        for _ in range(80):
            cursor["next"]={}
            cursor=cursor["next"]
        self.register("deep_fixture",lambda a,c: ToolResult("",data=value))
        result=await self.run_script('try {await tools.deep_fixture({});} catch(e) {text(e.message);}')
        self.assertTrue(self.script_message(result).is_error)
        self.assertIn("size/type/depth",self.script_message(result).content)
        self.assertTrue(all(r["status"]!="running" for r in self.runner.sessions.tool_calls(result.session_id)))

    async def test_script_alias_cannot_recurse(self):
        from borealis_coder.tools.script import ScriptTool
        alias=ScriptTool()
        alias.name="script_alias"
        self.runner.tools.register(alias)
        result=await self.run_script('await tools.script_alias({source:"text(1)",background:false});')
        self.assertTrue(self.script_message(result).is_error)
        self.assertIn("Recursive Script",self.script_message(result).content)
        self.assertEqual(len(self.runner.sessions.tool_calls(result.session_id)),1)

    async def test_compaction_and_session_replay_keep_script_results_ordered(self):
        self.config.agent.max_input_tokens=24000
        self.config.agent.max_output_tokens=1000
        observed=[]
        self.runner.events.subscribe(lambda e: observed.append(e.type))
        result=await self.run_script('text("selected Script evidence");')
        for index in range(10):
            self.provider.enqueue(ModelResponse(text=f"Historical response {index}: "+"fixture "*1000))
            continued=await self.runner.run(f"Continue fixture {index}",session_id=result.session_id)
            self.assertIsNone(continued.error,continued.error)
        self.assertIn("context.compacted",observed)
        durable=self.runner.sessions.messages(result.session_id)
        validate_tool_call_order(durable)
        self.assertEqual(len([m for m in durable if m.role==Role.TOOL]),1)
        self.assertIn("selected Script evidence",next(m.content for m in durable if m.role==Role.TOOL))


    async def test_collection_failure_records_partial_plugin_mutation_and_aborts_program(self):
        from borealis_coder.tools.script import ScriptScope
        def mutate(args,context):
            (context.workspace/"partial.txt").write_text("retained")
            return ToolResult("fixture")
        self.register("collect_fixture",mutate,effect=Effect.WRITE)
        with patch.object(ScriptScope,"retain_result",side_effect=OSError("fixture collection failed")):
            result=await self.run_script('await tools.collect_fixture({});await tools.write_file({path:"escaped.txt",content:"no",expected_sha256:null});')
        self.assertTrue(self.script_message(result).is_error)
        self.assertIn("partial.txt",result.changed_files)
        self.assertEqual(result.mutation_tracking,"incomplete")
        self.assertFalse((self.root/"escaped.txt").exists())
        self.assertTrue(all(r["status"]!="running" for r in self.runner.sessions.tool_calls(result.session_id)))

    async def test_cleanup_failure_keeps_primary_error_and_retries_only_termination(self):
        from borealis_coder.tools.script import ScriptScope
        original_cleanup=ScriptScope._cleanup
        jobs=[]
        attempts=0
        async def fail_once(scope,job):
            nonlocal attempts
            attempts+=1
            jobs.append(job)
            if attempts==1:
                raise OSError("synthetic cleanup failure")
            await original_cleanup(scope,job)
        # The worker is blocked on IPC when the host refuses its first packet.
        # It cannot exit before cleanup, so the termination retry is deterministic.
        with (patch.object(ScriptScope,"_cleanup",fail_once),
              patch("borealis_coder.tools.script.decode_packet",side_effect=ToolError("primary fixture failure"))):
            result=await self.run_script('await tools.read_file({path:"fixture.txt",start_line:null,end_line:null,max_chars:null});')
        self.assertEqual(result.stop_reason,StopReason.ERROR)
        self.assertGreaterEqual(attempts,2)
        self.assertIn("primary fixture failure",jobs[0].error)
        self.assertIn("cleanup failed",jobs[0].error)
        self.assertTrue(jobs[0].process and jobs[0].process.returncode is not None)
        self.assertFalse(self.runner._script_job_slots)
        self.assertTrue(all(t.done() for t in jobs[0].children))

    async def test_allowed_shell_executes_through_real_script_bridge(self):
        result=await self.run_script('const r=await tools.shell({command:"echo fixture-shell",cwd:".",timeout_seconds:3,description:"synthetic print"});text(r.output);')
        message=self.script_message(result)
        self.assertFalse(message.is_error,message.content)
        self.assertIn("fixture-shell",message.content)
        self.assertTrue(all(r["status"]=="completed" for r in self.runner.sessions.tool_calls(result.session_id)))

    async def test_direct_script_alias_can_await_mutations_without_holding_the_effect_lock(self):
        from borealis_coder.tools.script import ScriptTool
        alias=ScriptTool()
        alias.name="script_alias"
        self.runner.tools.register(alias)
        self.provider.enqueue(ModelResponse(tool_calls=[ToolCall(name="script_alias",arguments={"source":'await tools.write_file({path:"alias.txt",content:"done",expected_sha256:null});text("done");',"background":False})]), ModelResponse(text="done"))
        result=await asyncio.wait_for(self.runner.run("Synthetic Script alias"),5)
        self.assertIsNone(result.error)
        self.assertEqual((self.root/"alias.txt").read_text(),"done")
        self.assertIn("alias.txt",result.changed_files)


class ScriptProtocolTests(unittest.TestCase):
    def test_malformed_and_deep_ipc(self):
        for line in [b'[]\n',b'{bad}\n',b'{}',b'{"a":NaN}\n',b'{"x":'+b'['*40+b'0'+b']'*40+b'}\n']:
            with self.subTest(line=line),self.assertRaises(ToolError):
                decode_packet(line,1000)
        self.assertEqual(decode_packet(b'{"safe":[1,2]}\n',1000),{"safe":[1,2]})
