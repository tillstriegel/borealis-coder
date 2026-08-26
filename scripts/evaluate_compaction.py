#!/usr/bin/env python3
"""Run the fixed compaction-v2 structural or LLM release-gate corpus."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from pathlib import Path
from typing import Any, cast

from borealis_coder.agent.compaction_eval import (
    CompactionEvaluation,
    CompletionScorer,
    ReleaseSummarizer,
    evaluate_compaction_case,
    evaluate_compaction_release_case,
    load_compaction_corpus,
)


def _load_completion_scorer(reference: str) -> CompletionScorer:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Completion scorer must use the form module:function")
    scorer = getattr(importlib.import_module(module_name), attribute)
    if not callable(scorer):
        raise TypeError(f"Completion scorer is not callable: {reference}")
    return cast(CompletionScorer, scorer)


def _load_summarizer(reference: str) -> ReleaseSummarizer:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Summarizer must use the form module:function")
    summarizer = getattr(importlib.import_module(module_name), attribute)
    if not callable(summarizer):
        raise TypeError(f"Summarizer is not callable: {reference}")
    return cast(ReleaseSummarizer, summarizer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the compaction-v2 structural corpus and optional quality gate."
    )
    parser.add_argument(
        "--completion-scorer",
        help=(
            "Import an independent scorer as module:function. The callable receives "
            "(messages, case) and returns a score from 0 to 1."
        ),
    )
    parser.add_argument(
        "--summarizer",
        help=(
            "Import a release summarizer as module:function. The callable receives "
            "(prompt, case) and returns a ModelResponse with summary text and usage."
        ),
    )
    args = parser.parse_args(argv)
    if args.completion_scorer and not args.summarizer:
        parser.error("--completion-scorer requires --summarizer for the LLM release gate")
    if args.summarizer and not args.completion_scorer:
        parser.error("--summarizer requires --completion-scorer for the release gate")
    completion_scorer = (
        _load_completion_scorer(args.completion_scorer)
        if args.completion_scorer
        else None
    )
    summarizer = _load_summarizer(args.summarizer) if args.summarizer else None
    root = Path(__file__).resolve().parents[1]
    cases = load_compaction_corpus(root / "evals" / "compaction_v2_corpus.json")
    if summarizer is None or completion_scorer is None:
        reports = [evaluate_compaction_case(case) for case in cases]
    else:
        reports = asyncio.run(
            _evaluate_release_cases(
                cases,
                summarizer=summarizer,
                completion_scorer=completion_scorer,
            )
        )
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
                    "strategy": report.strategy,
                    "llm_compaction_evaluated": report.llm_compaction_evaluated,
                    "summarizer_calls": report.summarizer_calls,
                    "resume_artifact_reused": report.resume_artifact_reused,
                    "resume_summarizer_calls": report.resume_summarizer_calls,
                    "resume_cost_usd": report.resume_cost_usd,
                    "full_history_quality": report.full_history_quality,
                    "compacted_quality": report.compacted_quality,
                    "completion_quality_status": (
                        "evaluated"
                        if report.quality_gate_passed is not None
                        else "not_evaluated"
                    ),
                    "structural_gate_passed": report.structural_gate_passed,
                    "quality_gate_passed": report.quality_gate_passed,
                    "release_gate_passed": report.release_gate_passed,
                }
                for report in reports
            ],
            indent=2,
            sort_keys=True,
        )
    )
    if summarizer is None or completion_scorer is None:
        return 0 if all(report.structural_gate_passed for report in reports) else 1
    return 0 if all(report.release_gate_passed is True for report in reports) else 1


async def _evaluate_release_cases(
    cases: list[dict[str, Any]],
    *,
    summarizer: ReleaseSummarizer,
    completion_scorer: CompletionScorer,
) -> list[CompactionEvaluation]:
    reports: list[CompactionEvaluation] = []
    for case in cases:
        reports.append(
            await evaluate_compaction_release_case(
                case,
                summarizer=summarizer,
                completion_scorer=completion_scorer,
            )
        )
    return reports


if __name__ == "__main__":
    raise SystemExit(main())
