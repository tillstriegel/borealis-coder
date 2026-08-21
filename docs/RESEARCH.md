# Research: leading open-source coding harnesses

## Scope and method

This review defines a **coding harness** as a source-available runtime that lets a language model inspect a repository, invoke tools, modify code, and continue through multiple model/tool turns. Generic agent frameworks, autocomplete-only extensions, and products whose central runtime is proprietary were excluded.

Popularity is measured by GitHub stars because it is public, reproducible, and available across every candidate. Stars are not a quality score, usage count, or permanent ranking. The snapshot below was captured on **August 20, 2026**. Forks that were mainly variants of another reviewed harness were considered, but the architectural deep dive favors distinct runtimes.

The investigation inspected repository metadata, package/workspace boundaries, runtime loops, tool registries, policy and sandbox code, context management, persistence, provider adapters, extension protocols, editor/client protocols, and representative tests. It was not a claim that every line of every repository was audited.

## Popularity snapshot

| Rank | Repository | Stars | Primary implementation | Distinctive architectural contribution |
|---:|---|---:|---|---|
| 1 | [anomalyco/opencode](https://github.com/anomalyco/opencode) | 199,366 | TypeScript | Broad modular runtime: sessions, permissions, snapshots, worktrees, MCP/ACP, LSP, skills, plugins, background work. |
| 2 | [openai/codex](https://github.com/openai/codex) | 106,980 | Rust | Reusable core and app-server protocol; centralized approvals; OS-specific sandboxing; instruction and patch subsystems. |
| 3 | [google-gemini/gemini-cli](https://github.com/google-gemini/gemini-cli) | 106,588 | TypeScript | Event-driven scheduler, routing/fallback, confirmation bus, availability/billing handling, safety and telemetry. |
| 4 | [earendil-works/pi](https://github.com/earendil-works/pi) | 94,078 | TypeScript | Minimal event-stream agent primitive with steering, context transforms, cancellation, and a layered harness. |
| 5 | [OpenHands/OpenHands](https://github.com/OpenHands/OpenHands) | 84,546 | Python/TypeScript | Control plane plus reusable SDK, durable event logs, local/remote conversations, stuck detection, budgets, and remote runtimes. |
| 6 | [cline/cline](https://github.com/cline/cline) | 66,519 | TypeScript | Standalone SDK/runtime, provider construction, per-tool policy, lifecycle hooks, usage/cost/cache accounting, overflow recovery. |
| 7 | [aaif-goose/goose](https://github.com/aaif-goose/goose) | 53,031 | Rust | Operation/effect state machine, ACP/MCP, separated providers/context/local inference, durable reload between effects. |
| 8 | [Aider-AI/aider](https://github.com/Aider-AI/aider) | 48,339 | Python | Deterministic edit protocols, architect/editor separation, repository maps, automatic Git integration, extensive edit regression tests. |
| 9 | [continuedev/continue](https://github.com/continuedev/continue) | 35,550 | TypeScript | IDE-scale retrieval, indexing, context providers, autocomplete, diff/edit engines, broad provider adapters. |
| 10 | [charmbracelet/crush](https://github.com/charmbracelet/crush) | 27,512 | Go | Backend/client split, durable database services, prompt queueing, cancellation, LSP/MCP readiness, provider refresh and terminal polish. |

Near the cutoff were [QwenLM/qwen-code](https://github.com/QwenLM/qwen-code), [Kilo-Org/kilocode](https://github.com/Kilo-Org/kilocode), and [RooCodeInc/Roo-Code](https://github.com/RooCodeInc/Roo-Code). They informed the ecosystem scan; the detailed synthesis below uses the ten highest-starred distinct harnesses in the snapshot.

## Deep architectural findings

### 1. OpenCode

The current OpenCode source tree treats the harness as an operating system for agent work rather than a monolithic prompt loop. The runtime has explicit modules for sessions, messages, compaction, overflow, retries, run state, permissions, questions, tools, snapshots, Git worktrees, skills, plugins, MCP, ACP, LSP, IDE integration, storage, background work, and synchronization.

Important lessons:

- **Permissions are a subsystem.** Tool registration and execution do not own policy decisions.
- **Session processing is stateful and recoverable.** Durable tool status, retries, compaction, pruning, and revert are modeled separately.
- **Repository state has lifecycle.** Snapshots and worktrees make rollback and concurrent work explicit.
- **Extensions span several standards.** MCP extends tools; ACP exposes the agent; LSP enriches code intelligence; skills and plugins add local behavior.
- **Background and foreground work need different state.** Long-running tasks cannot be represented as a single blocking tool result forever.

Borealis adopts the modular boundaries, durable tool lifecycle, checkpointing, MCP/ACP split, skills, and policy separation. It keeps the initial implementation smaller by using a single-process async core and externalizing language-server or remote-control-plane functionality through protocols.

### 2. OpenAI Codex

Codex’s Rust workspace separates the orchestration core from CLI/UI surfaces and from an app-server protocol. It also has dedicated crates/modules for agent identity and graph state, analytics, instruction discovery, patch application, clients, platform execution, and delegated work.

The tool layer centralizes approvals across shell, patch, MCP, and network effects. Sandboxing is OS-specific rather than an assumption attached to a generic subprocess call.

Important lessons:

- **The CLI must not own the agent state.** A reusable protocol/runtime boundary enables editor, server, and local clients.
- **Patch application deserves a first-class implementation.** Editing quality and safety should be testable without a live model.
- **Approval semantics must cover every execution path.** MCP or delegated calls cannot bypass central policy.
- **Sandboxing is platform engineering.** Native subprocess limits and real OS isolation are materially different guarantees.
- **Instruction discovery is part of execution correctness.** Repository guidance should be resolved deterministically.

Borealis reflects these lessons through `AgentRunner`, an independent ACP server, strict `apply_patch`, central `PolicyEngine`, hierarchical instructions, and separate native/Docker drivers.

### 3. Gemini CLI

Gemini CLI’s core has explicit modules for routing, model fallback, availability, billing, a confirmation bus, context, hooks, IDE integration, MCP, output abstraction, policy, safety, sandboxing, a scheduler, services, skills, telemetry, and tools. Its scheduler is event-driven and treats tool calls as managed tasks rather than direct function calls embedded in rendering code.

Important lessons:

- **Graceful degradation is baseline behavior.** Availability, quota, fallback, and provider errors must yield actionable states.
- **Confirmation is an event, not terminal-only UI.** The same approval flow must work in CLI, IDE, or remote clients.
- **The scheduler owns concurrency.** Independent reads can run in parallel while mutations remain ordered.
- **Output must be abstract.** Structured events allow terminal, JSON, IDE, and telemetry consumers to share one runtime.

Borealis uses ordered provider routes, typed events, approval callbacks, ACP permission requests, parallel read scheduling, and distinct renderers.

### 4. Pi

Pi’s core loop is intentionally small: it streams events, transforms context before a turn, validates tool arguments, propagates aborts, and accepts steering/follow-up input. A higher harness layer adds sessions, reducers, compaction, skills, telemetry, environments, and tools.

Important lessons:

- **Keep the primitive loop legible.** Feature growth should happen through adapters and reducers rather than implicit global state.
- **Steering is a core interaction primitive.** A user should be able to alter an active task at a safe boundary.
- **Context preparation is replaceable.** The runner should request a projection of state, not assume one fixed prompt builder.
- **Events are the canonical integration surface.** UI code should not scrape model output.

Borealis preserves a compact normalized loop, queues steering, isolates context construction, and exposes all lifecycle information through `EventBus`.

### 5. OpenHands

OpenHands has evolved toward a control plane plus a reusable software-agent SDK. The SDK’s conversation layer includes local and remote implementations, durable event storage, cancellation, locks, stuck detection, budgets, condensers, secret handling, plugins, skills, MCP, subagents, confirmation policy, security analysis, visualizers, and observability.

Important lessons:

- **Conversation is a durable object.** It survives process and client boundaries.
- **Remote execution should share semantics with local execution.** The workspace/environment is an injected runtime, not hardcoded into the agent.
- **Stuck detection and budgets are operational necessities.** Frontier models can still loop or spend without bound.
- **Secrets need persistence semantics.** Redact or encrypt; never silently serialize raw credentials.
- **Subagents must be resource-accounted.** Delegation should not become invisible cost or unrestricted tool access.

Borealis implements durable sessions, cancellation, budgets, deterministic compaction, stuck-call detection, redacted traces, restricted read-only subagents, and an injectable process driver. A multi-tenant remote control plane is left to the embedding application.

### 6. Cline

Cline’s repository now exposes a reusable SDK with separate agent, core, LLM, shared, UI, CLI, hub, and editor packages. The agent runtime supports caller-built or provider-built models, wildcard and per-tool policies, before/after hooks, abort and controlled stops, usage/cost/cache metrics, request-size diagnostics, provider-error classification, and context-overflow recovery.

Important lessons:

- **Provider portability needs normalized accounting.** Tokens, cache reads/writes, reasoning tokens, and cost must survive adapter differences.
- **Context overflow should be recoverable and classified.** “Too large” is not one undifferentiated exception.
- **Hooks are useful when they cannot bypass policy.** Observability and application customization need lifecycle points.
- **A harness should be embeddable.** Editor UX is one client, not the architecture.

Borealis normalizes usage/cost, compacts before hard overflow, classifies provider errors, supports fallback, exposes events, entry-point tools, and an embedding API.

### 7. Goose

Goose’s Rust workspace separates agent state machines, context management, providers, provider types, MCP, ACP, local inference, SDK types, CLI, and test support. The agent core models ordered operations that either do not apply or produce effects; a runtime applies those effects, and durable session state can be reloaded between steps.

Important lessons:

- **Separate decision from effect application.** This makes cancellation, persistence, and alternative backends easier to reason about.
- **Protocol compatibility belongs in the runtime.** ACP and MCP should not be UI add-ons.
- **Local inference is a provider path, not a separate product.** OpenAI-compatible and native adapters should share the same agent semantics.
- **Effect objects improve testing.** State transitions can be tested without invoking every external system.

Borealis uses normalized provider events, effect categories, central policy before execution, durable state after each lifecycle event, MCP/ACP in core packages, and an OpenAI-compatible local route.

### 8. Aider

Aider’s strongest differentiation is edit quality. It supports multiple edit/coder modes, architect/editor separation, token-aware chat chunks, automatic Git integration, and repository maps that rank code symbols and dependencies. Its regression suite covers malformed edit blocks and edge cases in patch application.

Important lessons:

- **Editing is a deterministic compiler problem.** The model proposes a structured change; the harness parses, validates, and applies it.
- **Planning and editing can use different roles or models.** One model can reason broadly while another performs constrained mutation.
- **Repository maps are a high-leverage context primitive.** A compact symbol graph often beats raw file dumps.
- **Git is both evidence and recovery.** Diffs, status, and commits should be explicit tools.

Borealis uses strict mutation tools, atomic patch prevalidation, hash guards, checkpoints, ranked symbol maps, optional small-model delegation, and explicit Git status/diff/commit/push tools.

### 9. Continue

Continue’s core separates autocomplete, code rendering, commands, configuration, context providers, server integration, diffs, edits, indexing, retrieval, and a large provider adapter surface. The architecture is designed for continuously changing IDE state rather than only one terminal conversation.

Important lessons:

- **Retrieval should be cheap and incremental.** The model should receive relevant code, not the repository maximum.
- **Context has multiple producers.** Files, symbols, IDE state, diagnostics, documentation, and external systems should share a bounded interface.
- **Edit rendering and edit application are distinct.** A client may preview or accept changes.
- **Provider adapters should remain behind stable internal types.** UI and indexing code should not know wire formats.

Borealis implements a query-ranked repository map, focused search/read tools, bounded context providers, normalized adapters, and protocol events that allow clients to render tool and mutation progress.

### 10. Crush

Crush’s Go codebase separates backend, client, database, configuration, discovery, events, permissions, questions, history, file tracking, LSP, skills, and agent coordination. The coordinator handles prompt queueing, accepted-run handoff, cancellation, model/provider refresh, MCP readiness, retry event coalescing, and separate model roles.

Important lessons:

- **Terminal polish is runtime correctness.** Queues, cancellation, readiness, authentication refresh, and exactly-once terminal events matter.
- **Services should own durable domains.** Sessions, messages, permissions, questions, and history should not be one mutable singleton.
- **MCP readiness must be bounded.** A slow optional server should not freeze an interactive client.
- **Large and small models can serve different jobs.** Routing can reduce latency and cost.

Borealis has cancellable foreground sessions, steering queues, isolated MCP connection failures, model fallback, optional small-model delegation, durable domain tables, and one authoritative event stream. OAuth refresh and a rich TUI are outside the initial core.

## Converged frontier requirements

Across the ten codebases, the following capabilities repeatedly appear at mature boundaries:

1. A streaming, cancellable, event-driven model/tool loop.
2. Provider-neutral internal messages, tool calls, usage, error classes, and fallback.
3. Strict tool schemas and deterministic mutation protocols.
4. Central permission policy, approvals, filesystem containment, sandboxing, and rollback.
5. Repository-aware context, instructions, skills, retrieval, symbols, and compaction.
6. Durable sessions with replay, tool lifecycle, usage, and export.
7. Concurrency for independent reads and ordering for effects.
8. Git-native evidence and repository-aware verification.
9. MCP for tools/data and a client-agent protocol such as ACP.
10. Plugins, skills, hooks/events, and an embeddable library boundary.
11. Steering, cancellation, delegation, stuck detection, and hard budgets.
12. Operational diagnostics, structured logs, secret redaction, release automation, and test fixtures.

Borealis implements this set in a compact standard-library runtime. [FEATURE_MATRIX.md](FEATURE_MATRIX.md) maps each requirement to concrete modules and tests.

## Clean-room provenance

Borealis Coder was written from scratch. The review extracted architectural requirements, module boundaries, protocol expectations, and failure modes from public sources. No source files from the reviewed harnesses were copied into Borealis. Protocol implementations were written against public specifications and independently tested fixtures.
