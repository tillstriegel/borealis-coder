"""Release smoke test: install-independent, deterministic, and offline."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from borealis_coder.agent import build_runner
from borealis_coder.config import load_config


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="borealis-smoke-") as temp:
        root = Path(temp)
        config = load_config(root, overrides={
            "agent": {"provider": "mock", "auto_verify": False},
            "storage": {"directory": str(root / ".data")},
            "safety": {"approval": "never"},
        })
        runner = await build_runner(root, config=config, interactive=False)
        try:
            result = await runner.run("OFFLINE_WRITE_DEMO")
            assert result.stop_reason.value == "end_turn", result.to_dict()
            assert (root / "borealis-demo.txt").is_file()
            assert runner.sessions.usage(result.session_id).requests == 2
            assert runner.tool_context.checkpoints.list()
        finally:
            await runner.close()
    print("Borealis smoke test passed")


if __name__ == "__main__":
    asyncio.run(main())
