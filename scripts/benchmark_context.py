"""Micro-benchmark the local repository-map/context pipeline."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from borealis_coder.config import load_config
from borealis_coder.context import ContextBuilder


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--query", default="agent tool session provider")
    args = parser.parse_args()
    root = args.workspace.resolve()
    config = load_config(root, overrides={"agent": {"provider": "mock"}})
    builder = ContextBuilder(root, config)
    start = time.perf_counter()
    value = builder.system_prompt(query=args.query)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"chars={len(value)} elapsed_ms={elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
