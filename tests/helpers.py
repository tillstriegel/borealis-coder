from __future__ import annotations

from pathlib import Path

from borealis_coder.config import Config, load_config
from borealis_coder.context import ContextBuilder
from borealis_coder.events import EventBus
from borealis_coder.safety import (
    ApprovalManager,
    CheckpointManager,
    PolicyEngine,
    WorkspaceRoots,
    build_process_driver,
)
from borealis_coder.tools import ToolContext


def make_config(root: Path, **sections):
    base = {
        "agent": {"provider": "mock", "auto_verify": False},
        "storage": {"directory": str(root / ".data")},
        "safety": {"approval": "never"},
    }
    for section, values in sections.items():
        base.setdefault(section, {}).update(values)
    return load_config(root, overrides=base)


def make_context(root: Path, config: Config | None = None) -> ToolContext:
    config = config or make_config(root)
    roots = WorkspaceRoots(root)
    builder = ContextBuilder(root, config)
    return ToolContext(
        workspace=root,
        roots=roots,
        config=config,
        events=EventBus(),
        policy=PolicyEngine(config.safety, interactive=False),
        approvals=ApprovalManager(None),
        process=build_process_driver(roots, config.safety, config.sandbox),
        checkpoints=CheckpointManager(roots, enabled=True, max_bytes=config.safety.checkpoint_max_bytes),
        session_id="sess_test",
        run_id="run_test",
        metadata={"context_builder": builder},
    )
