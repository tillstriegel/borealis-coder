# Compaction v2

Compaction v2 creates a bounded provider view. It does not edit the durable session history.

## Design requirements

- Durable messages are append-only. Compaction artifacts use a separate SQLite table.
- Every compacted summary is escaped and framed as untrusted quoted history.
- A synthetic summary is system context. It is never a current user message.
- An assistant tool call and all matching tool results form one atomic bundle.
- Malformed call and result relationships stop request preparation.
- Current objectives, recent user constraints, pending work, blockers, changed files, and verification evidence have priority.
- A successful compaction must fit the calculated target in tokens and bytes.
- Cancellation, provider errors, usage, and cost keep their normal accounting behavior.
- Each artifact records the exact summary, source IDs and hash, configuration fingerprint, retained IDs, routed provider contexts, usage, and parent artifact.
- An invalid or failed model summary falls back to the deterministic artifact.

## Provider view

The durable messages remain unchanged. Request preparation first validates and groups them as:

- user request;
- assistant tool call with all results;
- steering message;
- verification result;
- terminal assistant response.

The deterministic artifact contains these sections in a stable order:

1. Current objective
2. User constraints
3. Completed work
4. Files changed
5. Important decisions
6. Latest verification
7. Open failures and blockers
8. Pending work
9. Historical excerpts

Borealis records an unavailable marker when durable structured evidence does not support a section. It does not infer decisions from prose.

## Context budget

`ContextBudget` starts with the configured model input limit. It reserves output tokens, system and tool-schema tokens, provider framing, continuation state, and a safety margin. Compaction starts at `compact_at_ratio` and must end below `compaction_target_ratio` of the available input. Bundle selection does not reserve continuation metadata that it may remove. Borealis recalculates the reserve from retained messages and tightens the artifact again when needed.

After a provider overflow, Borealis increases the safety margin and retries with a smaller provider-message target. `compaction_max_overflow_retries` is a strict upper bound. The reduction order is historical excerpts, retained bundles, and diagnostic output. The final fallback preserves mandatory state and the latest actionable bundle or returns a context-budget error before another provider call.

## Durable reuse

An artifact is reusable only when its source-content hash, strategy, prompt version, model selection, configuration fingerprint, retained provider messages, and routed provider contexts match. The artifact stores the exact summary, every configured route's provider payload, and a compacted-context hash for each route. Each routed attempt records the artifact ID and matching context hash without logging summary text. Resume checks the artifact before it calls an LLM summarizer. A new artifact records the previous artifact as its parent when the old source range is an exact prefix. Only the new transcript suffix is sent to the summarizer.

LLM compaction uses a strict JSON schema and bounded complete requests. Output must retain every structured source field and may contain only source-backed historical excerpts. Empty, malformed, invented, over-budget, cancelled, or overflowing summaries use the deterministic artifact. Usage from completed or failed summary requests is recorded before an error is propagated.

## Evaluation and release gates

Run the fixed offline structural corpus:

```sh
PYTHONPATH=src python scripts/evaluate_compaction.py
```

The corpus covers multi-file work, requirement changes, repeated failures, large output, cancellation and resume, continuation metadata, steering, hostile content, repeated compaction, and provider overflow. It reports critical-fact recall, false completion claims, boundary escapes, tool ordering, token reduction, target compliance, latency, cost, and resume determinism. Without an independent completion scorer, completion quality and the full release gate are reported as `not_evaluated` instead of being inferred from the critical-fact checks.

Supply an independent model-backed or human-backed scorer to compare task-completion quality against full history:

```sh
PYTHONPATH=src python scripts/evaluate_compaction.py \
  --completion-scorer your_package.compaction_eval:score_completion
```

The scorer receives `(messages, case)` and returns a score from 0 to 1. The command evaluates the full release gate only when this scorer is present. Otherwise, its exit status covers the offline structural gate only.

The release gates are:

- 100% recall of marked critical facts;
- zero boundary escapes;
- zero invalid tool sequences;
- every successful size case below target;
- no second LLM charge for an unchanged resume;
- an independently measured completion-quality regression no greater than five percentage points.

## Rollout

`agent.compaction_version = 2` enables v2. A compatibility v1 path remains available for one release and uses the same secure framing.

1. Set `compaction_version = 1` and `compaction_shadow_v2 = true` to emit safe v2 comparison metrics while v1 remains active.
2. Set `compaction_version = 2` with `deterministic_compaction = true` to enable deterministic v2 and durable reuse.
3. Set `deterministic_compaction = false` only after both the structural gate and an independent completion-quality gate pass for the configured summarizer provider.
4. Remove v1 after the compatibility release.

`context.compacted` and `context.compaction_shadow` events contain sizes, counts, strategy, artifact version, reuse state, overflow retry count, fallback reason, usage, and latency. They do not contain summary text.
