# Frontier feature matrix

## Core requirements

| Requirement | Why it is required | Borealis implementation | Validation |
|---|---|---|---|
| Streaming, multi-turn agent loop | Coding tasks require repeated reasoning and tool feedback. | `agent/runner.py`, normalized provider events, explicit stop reasons. | Offline two-turn tool cycle and provider fixtures. |
| Interactive developer shell | Local coding work needs a readable, resumable loop rather than one process per prompt. | Bare `borealis` starts a persistent shell with streamed text, tool status, slash commands, history, runtime switching, multiline input, and cancellation. | CLI integration tests, piped-terminal smoke tests, and clean-wheel invocation. |
| Cancellation and steering | Users and clients must interrupt or redirect long tasks. | Session cancellation events and a turn-boundary steering queue. | `test_cancel`, `test_steering_is_injected_before_next_turn`. |
| Provider-neutral adapters | Models and APIs change; the runtime must remain portable. | OpenAI Responses, ChatGPT-plan access through Codex OAuth, first-class OpenRouter, OpenAI-compatible chat, Anthropic Messages, Gemini Interactions, mock provider, custom registry. | Payload/parser fixtures for every built-in live adapter plus ChatGPT auth/refresh fixtures. |
| Account authentication | Some coding surfaces are included in user subscriptions rather than API-key billing. | Secure file-backed Codex credential discovery, official Codex login bridge, token rotation, expiry checks, account headers, status/logout commands. | Credential, refresh, CLI, endpoint-hardening, and pre-output retry tests. |
| Retry and fallback | Transient failures and model availability are normal. | Error classification, capped exponential backoff, ordered provider routes. | `test_provider_fallback`. |
| Strict tool contracts | Malformed model output must fail before execution. | Full JSON-object schemas with all keys required and nullable optional semantics. | Schema validator and release gate. |
| Deterministic editing | Free-form shell edits are difficult to verify and roll back. | SHA-guarded writes, exact replace, atomic prevalidated unified patch, guarded delete. | Mutation and atomicity tests. |
| Parallel reads, ordered effects | Repository investigation benefits from concurrency; writes need consistency. | Scheduler partitions read-only calls and executes them with a semaphore; effects remain sequential. | Agent integration and tool tests. |
| Central effect policy | No tool, plugin, verification command, or MCP server may bypass safety. | Effect categories, `PolicyEngine`, `ApprovalManager`, hard gates. | Safety and shell policy tests. |
| Workspace and symlink containment | Path strings alone are not a security boundary. | Resolved allowed roots, traversal rejection, symlink-escape checks, optional additional roots. | Path security tests. |
| Sandboxed process execution | Models execute untrusted build and test commands. | Bounded native driver and stronger Docker driver with resource/network controls. | Shell bounds tests; Docker diagnostics. |
| Checkpoints and rollback | Autonomous writes need practical recovery independent of Git cleanliness. | Pre-mutation snapshots with byte limits and CLI rollback. | Write/rollback and smoke tests. |
| Repository-aware context | Full repository dumps waste tokens and obscure relevant code. | Git-aware discovery, ignore rules, symbol extraction, ranked repo map, focused search/read. | Context ranking tests and benchmark script. |
| Hierarchical instructions | Repository and directory rules materially affect correctness. | `AGENTS.md`, `BOREALIS.md`, `CLAUDE.md` discovery from root to active path. | Instruction tests. |
| Lazy skills | Rich procedural guidance should be available without polluting every prompt. | Skill discovery, compact descriptors, explicit `read_skill`. | Skill tests and example. |
| Context compaction | Long sessions exceed model windows. | Validated turn bundles, target-based structured artifacts, append-only history, durable reuse, and bounded overflow recovery. | Compaction v2 adversarial tests and fixed evaluation corpus. |
| Durable sessions and replay | Work must survive process/client restarts and support audits. | SQLite/WAL sessions, messages, events, tool calls, usage, KV, JSON export. | Session CRUD/export tests; ACP replay. |
| Budgets and stuck detection | Autonomous loops need hard operational limits. | Turn, input, output, time, cost, and repetition limits. | Unit paths and offline acceptance. |
| Git integration | Diffs and status are the most useful evidence of repository mutation. | Status, diff, log, gated commit and push tools. | Tool registry and policy tests. |
| Verification | “Done” should be supported by executable evidence. | Project-type detection and bounded verification; automatic post-mutation gate. | Smoke and runtime tests. |
| MCP tools | External tools and data sources need a standard extension boundary. | stdio and Streamable HTTP clients, paginated discovery, namespace and policy mapping. | Fake stdio server integration test. |
| ACP client protocol | Editors and front ends need session, streaming, permission, and cancellation semantics. | ACP v2 stdio JSON-RPC server with lifecycle, replay, updates, additional roots, MCP, steering. | ACP lifecycle integration test. |
| Plugins and embedding | A harness must be reusable beyond its own CLI. | Python library API, provider registry, entry-point tools, opt-in workspace plugins. | Import/smoke checks and examples. |
| Subagents | Focused parallel investigation can reduce main-context pressure. | Read-only delegated tasks with separate context, bounded report, usage accounting. | Tool registration and agent tests. |
| Structured observability | CLI, IDE, CI, and audit consumers need the same authoritative stream. | Typed `EventBus`, readable interactive renderer, redacted JSONL trace, SQLite events, JSON CLI mode, ACP rendering. | Session/event tests. |
| Secret hygiene | Provider and repository secrets can leak through logs and error payloads. | Environment-aware and pattern-based redaction; redacted config and traces. | Release secret scan. |
| Release engineering | Production use needs reproducible packaging and automated checks. | `pyproject.toml`, CI, release workflow, Dockerfile, smoke test, validation script, manifest. | Clean wheel installation and release validation. |

## Capability tiers

### Included in the core

The table above is the production baseline implemented directly in the repository.

### Integrated through standards rather than reimplemented

| Capability | Approach |
|---|---|
| Language-server diagnostics and navigation | Connect an LSP-aware MCP server or client-side ACP integration. Borealis does not bundle language servers. |
| Browser/computer interaction | Add an MCP server with explicit effect annotations and local policy rules. |
| Remote worker or multi-tenant control plane | Embed the runner behind an authenticated service or use ACP as the agent boundary. |
| Provider-hosted shell, search, or code execution | Prefer local audited tools; provider-hosted tools can be added through an adapter or MCP. |
| Long-lived background processes | Use an MCP job server or embedding application; the built-in shell is intentionally bounded and foreground-oriented. |

### Deliberately outside version 0.1

These are useful product features, but they are not necessary to call the core a competitive coding harness:

- an alternate-screen dashboard (Aurora Shell deliberately preserves scrollback)
- an account/subscription control plane
- native image/audio prompt ingestion
- encrypted cloud synchronization
- marketplace installation UI
- organization policy administration
- VM/Kubernetes worker provisioning

The architecture leaves explicit protocol and library boundaries for these features without making the core runtime depend on them.
