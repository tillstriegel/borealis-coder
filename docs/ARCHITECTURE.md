# Architecture

## Design objective

Borealis Coder is organized around one constraint: **the model may propose actions, but the harness owns state, permissions, execution, durability, and evidence**. That keeps the language model replaceable and prevents provider-specific semantics from leaking into the safety boundary.

## Runtime layers

```text
Interactive CLI / one-shot CLI / ACP / embedded application
                 │
                 ▼
          AgentRunner + EventBus
       ┌─────────┼───────────────┐
       │         │               │
       ▼         ▼               ▼
 Provider routes ContextBuilder  SessionStore
       │         │               │
       │         ├─ instructions │─ SQLite/WAL
       │         ├─ skills       │─ messages/events
       │         ├─ repo map     │─ tool lifecycle
       │         └─ compaction   │─ usage/metadata
       │
       ▼
 strict model response → ToolRegistry
                         │
                 Policy + Approval
                         │
              ToolContext / ProcessDriver
        ┌───────────────┼─────────────────┐
        ▼               ▼                 ▼
 deterministic FS   native/Docker       MCP tools
 + checkpoints      command runner      + plugins
        │
        ▼
 repository-aware verification + final result
```

### Terminal and CLI layer

The local CLI has two surfaces over the same runtime:

- bare `borealis` starts `InteractiveCLI`, a persistent shell that reuses one
  runner, keeps a durable session across prompts, and can resume stored sessions
- `borealis run` performs one bounded task for scripts and CI

`ConsoleRenderer` subscribes to the authoritative EventBus. Rendering state is
scoped to each model exchange, because one user turn can contain several model
responses separated by tool calls. The completed model response remains the
source of truth when stream deltas are incomplete. Human text, tool progress,
verification, and the next prompt are finalized in deterministic order.

The shell adds readline history when available, slash commands, runtime route and
policy changes, multiline input, and turn cancellation. It does not create a
second conversation implementation; all messages, tools, usage, and terminal
state still pass through `AgentRunner` and `SessionStore`.

### `agent/`

`AgentRunner` is the foreground state machine. It:

1. Creates or resumes a durable session.
2. Builds the initial system/context prompt.
3. Selects the primary provider and ordered fallbacks.
4. Streams a model turn and persists usage.
5. Validates tool calls against strict schemas.
6. Executes independent read-only calls concurrently.
7. Executes effectful calls sequentially.
8. Appends tool results to the model conversation.
9. Injects queued steering before the next model turn.
10. Detects repeated tool-call loops and enforces turn/time/token/cost budgets.
11. Compacts in-memory context when needed without deleting durable history.
12. Runs verification after mutations and emits a terminal result.

The loop uses explicit `StopReason` values so callers can distinguish end turn, cancellation, budget exhaustion, tool failure, provider failure, and verification failure.

### `providers/`

Provider adapters implement a small async streaming interface and convert a normalized request into provider-specific wire formats. The core includes:

- OpenAI Responses API
- ChatGPT-plan access through Codex OAuth and the Codex Responses backend
- OpenRouter Chat Completions / Responses with routing extensions
- OpenAI-compatible Chat Completions
- Anthropic Messages
- Gemini Interactions
- deterministic mock provider

Adapters return normalized text deltas, tool calls, usage, finish reasons, and errors. Retry classification and exponential backoff live at the provider boundary; provider fallback lives in the runner.

### `tools/`

Every tool provides:

- a unique name and description
- a strict JSON Schema
- an effect category (`read`, `write`, `shell`, `network`, `git_commit`, `git_push`, and related classes)
- an async executor

The registry validates calls before policy evaluation. This means malformed or unknown arguments never reach the filesystem or subprocess layer.

Built-in mutation tools prefer deterministic operations:

- existing-file writes require an observed SHA-256
- exact replacements require an unambiguous old string and optional expected hash
- patches are parsed and prevalidated across all target files before any write
- writes use atomic replacement
- checkpoints record pre-mutation bytes and metadata

### `safety/`

The safety subsystem is deliberately independent from prompts and providers.

- `WorkspaceRoots` resolves paths and blocks traversal and symlink escapes.
- `PolicyEngine` decides allow/ask/deny from effect type, safety mode, command risk, and hard gates.
- `ApprovalManager` mediates terminal or ACP client approval and can cache decisions for one session.
- `CommandAssessment` classifies shell command chains, network use, destructive operations, privilege changes, commits, pushes, and unknown constructs.
- `NativeProcessDriver` applies environment filtering, timeouts, output limits, and Unix resource limits where supported.
- `DockerProcessDriver` mounts roots explicitly, constrains memory/CPU/processes, and disables network unless allowed.
- `CheckpointManager` records and restores bounded pre-mutation snapshots.
- `Redactor` removes configured and pattern-matched secrets from traces and persisted error payloads.

### `context/`

`ContextBuilder` creates a compact, task-specific view of the repository. It separates durable repository knowledge from transient model context:

- Git-aware file enumeration and ignore rules
- hierarchical instructions from root to active directory
- lazy skill discovery and loading
- language-aware symbol extraction
- query-ranked repository map
- optional Git status
- bounded tool results

The in-memory compactor summarizes structural facts, completed work, pending work, changed files, recent errors, and recent turns. The SQLite message log remains complete for replay and audit.

### `sessions/`

`SessionStore` uses SQLite with WAL and foreign-key enforcement. The schema stores:

- session identity, title, workspace, state, provider/model, timestamps, and metadata
- ordered normalized messages
- ordered runtime events
- tool calls and lifecycle status
- cumulative usage and cost
- session-scoped key/value state

Database writes are serialized through an async lock. Reads can be performed while the process remains active. Event persistence is authoritative: a failed event batch stays queued in order and fails the run instead of reporting false success. Optional trace and subscriber failures remain isolated from already committed SQLite batches. Export produces a portable JSON object containing all durable session evidence. Administrative session commands open this store directly and do not initialize providers, plugins, or MCP servers.

### `mcp/`

MCP clients support stdio and Streamable HTTP. Connection flow:

1. Launch or connect to the server.
2. Exchange `initialize` and `notifications/initialized`.
3. Page through `tools/list`.
4. Convert tool definitions to local strict tools.
5. Namespace names as `<server>__<tool>`.
6. Map MCP safety annotations to Borealis effect categories.
7. Invoke through `tools/call` while retaining local policy enforcement.

Each server has isolated lifecycle and failure state.

### `protocol/`

The ACP server uses bidirectional newline-delimited JSON-RPC over stdio. It maps ACP sessions to `AgentRunner` instances and durable session IDs. Notifications are rendered from the same event stream used by the CLI, which prevents protocol behavior from diverging from in-process behavior.

Implemented stable ACP v2 capabilities include session lifecycle, replay, additional roots, MCP configuration, prompt acceptance, cancellation, steering, client permissions, and structured updates.

## Event model

The runner emits typed events for:

- session and run lifecycle
- provider request/retry/fallback
- provider-supplied reasoning summaries, kept separate from assistant text
- assistant text
- plan changes
- tool pending/in-progress/completed/failed
- permission requests and decisions
- usage
- context compaction
- verification
- terminal state

Subscribers are isolated: one failing observer cannot crash the run. SQLite event persistence is required and fails closed; a redacted JSONL trace remains an independent optional sink.

## Concurrency model

Borealis uses `asyncio` throughout.

- One foreground run owns a session at a time.
- Steering messages enter a queue and are injected at a turn boundary.
- Read-only tool calls from one model response can run concurrently up to `agent.parallel_reads`.
- Mutations and shell effects remain ordered.
- Provider and subprocess cancellation propagate from the session cancellation event.
- Delegated read-only subagents receive a restricted registry and independent context, then return a bounded report to the parent tool call.

## Data and cache invariants

1. The model conversation is a projection, not the authoritative session record.
2. Durable messages are never deleted by context compaction.
3. File content is not silently reused after a mutation; hash guards make stale context explicit.
4. Tool output is bounded before insertion into model context.
5. Provider-specific response IDs are optional optimization metadata, never required to resume a session.
6. All effectful calls pass through one policy path, including MCP and verification commands.
7. Provider prompt caches receive a stable context prefix; request-ranked and mutable context is appended afterward.
8. Exact-response cache entries are content-addressed and bounded, and never contain tool calls.
9. Logical input usage includes uncached, cache-read, and cache-write tokens exactly once.
10. Provider continuation state is versioned, contains only provider-selected replay items, and is accepted only for the originating provider and requested model.
11. A lifecycle event is not considered durable until its SQLite batch commits successfully.

## Failure handling

- Transient provider failures retry with capped exponential backoff.
- A failed primary route can fall back to configured providers.
- Invalid tool calls become structured tool errors rather than unhandled exceptions.
- Partial patch validation fails before mutation; filesystem failure during commit triggers checkpoint restoration where possible.
- Repeated identical tool calls trigger a stuck-loop stop.
- Time, turn, token, output, and cost budgets stop the run deterministically.
- A disconnected MCP server removes only its own tools.
- Cancellation terminates active model/subprocess work and persists the terminal state.

## Deployment boundaries

The library, CLI, and ACP server share the same runtime. They can be deployed as:

- a local developer CLI
- an editor-managed ACP subprocess
- a CI task runner
- a Docker-isolated worker
- an embedded Python component

A multi-tenant control plane is intentionally outside the core. ACP or an application-specific service can own authentication, queueing, tenancy, and remote worker scheduling while embedding Borealis as the execution runtime.
