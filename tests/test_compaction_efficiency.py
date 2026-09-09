"""Compaction frequency and provider working-set regressions."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path

from borealis_coder.agent import build_runner, validate_tool_call_order
from borealis_coder.context import PromptContext
from borealis_coder.events import Event
from borealis_coder.models import Message, Role, ToolCall, Usage
from tests.helpers import make_config


def output_batch(turn: int, count: int, *, repeated_read: bool = False) -> list[Message]:
    calls = [
        ToolCall(
            id=f"call-{turn}-{index}",
            name="read_file" if repeated_read else "shell",
            arguments={"path": "file.py"} if repeated_read else {"command": f"check-{turn}-{index}"},
        )
        for index in range(count)
    ]
    return [
        Message(role=Role.ASSISTANT, tool_calls=calls),
        *[
            Message(
                role=Role.TOOL,
                tool_call_id=call.id,
                tool_name=call.name,
                content=("file contents\n" if repeated_read else f"output-{turn}-{index}\n") + "x" * 24_000,
                metadata={"path": "file.py", "sha256": "same-revision"} if repeated_read else {},
            )
            for index, call in enumerate(calls)
        ],
    ]


async def replay(batches: list[list[Message]]):
    """Prepare real runner requests offline without changing the source messages."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runner = await build_runner(root, config=make_config(root), interactive=False)
        session = runner.sessions.create_session(workspace=root, provider="mock", model="deterministic")
        events: list[Event] = []
        runner.events.subscribe(events.append)
        prepared_requests = []
        source = []

        async def usage_sink(_usage: Usage) -> None:
            raise AssertionError("Deterministic replay must not call a model")

        try:
            for batch in batches:
                for message in batch:
                    runner.sessions.append_message(session.id, message)
                    source.append(message.to_dict())
                prepared = await runner._prepare_provider_request(
                    prompt_context=PromptContext(stable="Complete the requested work."),
                    messages=runner.sessions.messages(session.id),
                    schemas=[],
                    final_turn=False,
                    verification_finalization_pending=False,
                    adaptive_cache=False,
                    conversation_cache=True,
                    usage_sink=usage_sink,
                    cancel=asyncio.Event(),
                    session_id=session.id,
                    run_id="efficiency-replay",
                    last_prune_signature=None,
                )
                validate_tool_call_order(prepared.request.messages)
                prepared_requests.append(prepared)
            assert source == [message.to_dict() for message in runner.sessions.messages(session.id)]
            artifacts = runner.sessions.compaction_artifacts(session.id)
            return prepared_requests, events, artifacts
        finally:
            await runner.close()


class CompactionEfficiencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_large_parallel_outputs_leave_room_for_another_batch(self):
        requests, events, artifacts = await replay([
            [Message(role=Role.USER, content="Inspect the project and fix the issue.")],
            *[output_batch(turn, 4) for turn in range(8)],
        ])
        fresh = [event for event in events if event.type == "context.compacted"]
        self.assertEqual(len(fresh), len(artifacts))
        self.assertLessEqual(len(fresh), 4)
        self.assertGreater(len(fresh), 0)
        for event in fresh:
            self.assertLess(event.data["tool_output_tokens_retained"], 10_000)
            self.assertGreater(event.data["tool_output_headroom"], 30_000)
            self.assertGreater(event.data["reduction_percentage"], 50)
        for previous, current in pairwise(requests):
            if previous.compaction_metadata and not previous.compaction_metadata["artifact_reused"]:
                self.assertTrue(current.compaction_metadata["artifact_reused"])
                self.assertEqual(
                    previous.request.metadata["compaction_artifact_id"],
                    current.request.metadata["compaction_artifact_id"],
                )
        # A new compaction compares against the previous provider view plus
        # new output, rather than claiming savings against all past history.
        self.assertLess(fresh[-1].data["estimated_tokens_before"], 70_000)
        self.assertGreater(
            fresh[-1].data["durable_estimated_tokens"],
            fresh[-1].data["estimated_tokens_before"],
        )

    async def test_superseded_reads_do_not_trigger_compaction(self):
        requests, events, artifacts = await replay([
            [Message(role=Role.USER, content="Read file.py.")],
            *[output_batch(turn, 1, repeated_read=True) for turn in range(15)],
        ])
        self.assertEqual(len(artifacts), 0)
        self.assertFalse(any(event.type == "context.compacted" for event in events))
        self.assertLess(requests[-1].estimated_tokens, 10_000)

    async def test_single_outputs_reuse_summary_without_new_compaction_events(self):
        _, events, artifacts = await replay([
            [Message(role=Role.USER, content="Inspect the project.")],
            *[output_batch(turn, 1) for turn in range(15)],
        ])
        fresh = [event for event in events if event.type == "context.compacted"]
        reused = [event for event in events if event.type == "context.reused"]
        self.assertEqual(len(fresh), len(artifacts))
        self.assertLessEqual(len(fresh), 2)
        self.assertGreater(len(reused), len(fresh))
        self.assertTrue(all(event.data["artifact_reused"] for event in reused))
