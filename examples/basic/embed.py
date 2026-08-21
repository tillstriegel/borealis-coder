"""Embed Borealis in an async Python application."""
from __future__ import annotations

import asyncio
from pathlib import Path

from borealis_coder.agent import build_runner


async def main() -> None:
    runner = await build_runner(Path.cwd())
    runner.events.subscribe(lambda event: print(event.type, event.data))
    try:
        result = await runner.run("Find and fix the highest-impact failing test.")
        print(result.to_dict())
    finally:
        await runner.close()


if __name__ == "__main__":
    asyncio.run(main())
