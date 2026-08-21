# Changelog

All notable changes follow Keep a Changelog and Semantic Versioning.

## [0.1.3] - 2026-08-21

### Added
- A persistent Codex-style interactive CLI is now the default when running
  `borealis` without a subcommand.
- Durable session continuation through `borealis --continue`,
  `borealis resume --last`, and `/resume` inside the shell.
- Interactive commands for session history, provider/model switching, safety
  mode, approvals, network access, verification, sandbox selection, tools,
  diagnostics, checkpoints, and multiline prompts.
- Best-effort readline history and slash-command completion, with local history
  stored using owner-only permissions on Unix.
- Turn-level Ctrl+C cancellation that preserves the interactive process and
  durable session.
- The dependency-free Aurora Shell visual system with a responsive launch panel,
  session-aware prompt, assistant frames, activity rail, command panels, and
  compact turn receipts.
- Live elapsed-time pulses for pending model, tool, and verification work, with
  accessible uncolored and redirected-output fallbacks.
- Stable provider-cache context prefixes, explicit Anthropic prompt-cache
  breakpoints, cache-aware ChatGPT routing keys, and adaptive cache tuning.
- Bounded exact-response caching for successful text-only requests, with tool
  responses excluded from replay.
- Per-turn prompt/response cache metrics, net savings, and cache-write-aware
  provider cost accounting.
- A readline-native command selector that opens when an interactive message
  begins with `/`, with descriptions, prefix filtering, and Tab completion.

### Fixed
- Assistant output is always terminated cleanly before the next prompt.
- Final answers after one or more tool calls are no longer suppressed by text
  emitted during an earlier model exchange in the same user turn.
- Interactive status, tool, and response output share a deterministic terminal
  stream, avoiding prompt/footer interleaving in piped and captured sessions.

## [0.1.2] - 2026-08-21

### Added
- First-class `chatgpt` provider for ChatGPT-plan access through the Codex
  Responses backend.
- Discovery of secure file-backed Codex logins, `CODEX_ACCESS_TOKEN`, and a
  Borealis-managed Codex credential home.
- `borealis auth login`, `borealis auth status`, and `borealis auth logout`.
- OAuth token refresh with rotation, atomic persistence, cross-process locking,
  proactive expiry checks, and one safe pre-output authentication retry.
- Preservation of encrypted Responses reasoning state across tool turns without
  persisting plaintext reasoning summaries or complete native responses.

### Security
- ChatGPT bearer, account, and refresh-token headers are authoritative and cannot
  be replaced by repository-local provider headers.
- Custom ChatGPT API or OAuth endpoints require both an explicit configuration
  flag and the process-level `BOREALIS_ALLOW_CUSTOM_CHATGPT_ENDPOINTS=1` opt-in.
- Borealis never reads ChatGPT browser cookies or browser-session storage.

## [0.1.1] - 2026-08-20

### Added
- First-class OpenRouter provider with `OPENROUTER_API_KEY` auto-detection.
- OpenRouter attribution headers, native reasoning controls, model-level fallbacks,
  provider routing/privacy preferences, and an `extra_body` extension escape hatch.
- OpenRouter-reported request-cost and reasoning-token accounting when supplied.
- Chat Completions by default with optional OpenRouter Responses API mode.

## [0.1.0] - 2026-08-20

### Added
- Async, provider-portable coding-agent runtime.
- OpenAI Responses/Chat Completions, Anthropic Messages, Gemini Interactions,
  OpenAI-compatible, and deterministic mock providers.
- Policy engine, approvals, command risk classification, native/Docker process drivers,
  workspace containment, secret redaction, and automatic file checkpoints.
- Deterministic filesystem/search/patch/git/verification/repository-map/todo tools.
- Durable SQLite sessions, usage accounting, compaction, stuck detection, cancellation,
  JSONL traces, skills, plugins, MCP stdio/HTTP clients, and ACP v2 agent server.
- CLI, JSON automation mode, diagnostics, built-in evaluations, release and CI assets.
