#!/usr/bin/env python3
"""Run the fixed offline compaction-v2 release-gate corpus."""

from __future__ import annotations

import json
from pathlib import Path

from borealis_coder.agent.compaction_eval import (
    evaluate_compaction_case,
    load_compaction_corpus,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cases = load_compaction_corpus(root / "evals" / "compaction_v2_corpus.json")
    reports = [evaluate_compaction_case(case) for case in cases]
    print(
        json.dumps(
            [
                {
                    "name": report.name,
                    "critical_fact_recall": report.critical_fact_recall,
                    "false_completion_claims": report.false_completion_claims,
                    "boundary_escape_cases": report.boundary_escape_cases,
                    "invalid_tool_sequences": report.invalid_tool_sequences,
                    "tokens_before": report.tokens_before,
                    "tokens_after": report.tokens_after,
                    "reduction_percentage": round(report.reduction_percentage, 2),
                    "below_target": report.below_target,
                    "deterministic": report.deterministic,
                    "latency_ms": round(report.latency_ms, 3),
                    "cost_usd": report.cost_usd,
                    "full_history_quality": report.full_history_quality,
                    "compacted_quality": report.compacted_quality,
                    "release_gate_passed": report.release_gate_passed,
                }
                for report in reports
            ],
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if all(report.release_gate_passed for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
