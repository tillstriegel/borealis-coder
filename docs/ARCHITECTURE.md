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

The shell adds prompt-toolkit input and history on interactive terminals, slash
commands, runtime route and policy changes, multiline input, and turn cancellation. It does not create a
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

Run results carry a `StopReason` (`end_turn`, `max_turns`, `budget`, `cancelled`,
`error`, or `stuck`) and a separate verification report. Callers inspect
`verification.checks_ok`, `verification.process_lifecycle_guaranteed`, and
`mutation_tracking` to assess verification and remaining uncertainty.

Parent and delegated runs share the same incomplete-response check. Delegated
investigations reject incomplete conclusions and tool calls, and include usage
reported by failed provider requests in the parent run's accounting. Delegated
usage is settled after each response before more work begins. Every logical model
request checks the shared time and cost limits, including at the exact cost ceiling.
Delegated tool loops preserve provider-selected continuation state with its route
and model identity, using the same metadata format as the parent loop.

When a tool reaches a budget, the runner records its error and observed workspace
changes, drains parallel reads, and records calls that did not start. The terminal
run event follows these durable results so the next run can resume valid history.

Provider-only read pruning compares actual output content and visible line ranges.
It indexes identical outputs and checks known mutations before scanning covering reads.
A truncated read cannot prove that it covers an earlier excerpt, even when its
requested range spans the whole file. Identical truncated outputs still deduplicate.

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

Nested retry wrappers for the same provider and asyncio task share one retry loop.
Concurrent and child tasks retain independent retry budgets. The retry scope is
released after success, failure, or cancellation.

Cancelling a JSON request or event stream stops pool waits and shuts down its active
socket, including while waiting for direct-response headers or reading a response
body. The request detaches its socket before returning a connection to the pool.
A pending DNS or proxy connection may not expose a socket yet; a bounded join and
daemon worker keep that wait from holding interpreter shutdown open. Direct
connections check cancellation again before sending after connection setup.
Streams detach socket ownership before releasing a redirected connection and
check cancellation before following the redirect. Cleanup waits for the worker's
bounded join even if the task receives repeated cancellation.

The Gemini adapter reconstructs streamed steps when terminal metadata omits them,
including function arguments and thought signatures needed for continuation.
Incomplete and budget-exhausted responses do not expose executable tool calls.
Failed interactions raise provider errors. Cumulative usage from step-stop events
and terminal metadata is retained in incomplete responses and provider failures.

### `tools/`

Every tool provides:

- a unique name and description
- a strict JSON Schema
- an effect category (`read`, `write`, `shell`, `network`, `git_commit`, `git_push`, and related classes)
- an async executor

The registry validates calls before policy evaluation. This means malformed or unknown arguments never reach the filesystem or subprocess layer.

Each tool invocation receives its own context with a stable call ID. File-change
sets and metadata remain shared for the run. Mutation uncertainty accumulates back
into the run context even if a call fails or is cancelled.

Built-in mutation tools prefer deterministic operations:

- existing-file writes require an observed SHA-256
- exact replacements require an unambiguous old string and optional expected hash
- patches are parsed and prevalidated across all target files before any write
- writes use atomic replacement
- checkpoints record pre-mutation bytes and metadata

File mutations wait for workspace thread and process locks without blocking the
event loop. Cancellation while waiting leaves files unchanged and releases local
lock resources. After acquisition, validation, checkpointing, and commit remain
within the same transaction.
CLI and interactive rollback acquire the same lock before restoring a checkpoint.
The synchronous checkpoint manager expects its mutation caller to hold this lock;
patch failure recovery already runs inside its existing transaction.

Unified patch hunks use their declared line counts to separate content from file
headers. Added files receive the same count and content checks as updates.
Zero-context insertions use the empty-range position, and no-newline markers
preserve file endings. Git's `a/` and `b/` prefixes are removed only from unified
headers; envelope paths are literal. Envelopes require `*** End Patch`.
Hunks preserve their supplied line endings. If exact matching fails for a CRLF
patch, the engine retries its LF-normalized form for transport compatibility.
Unsupported Git sections reject the entire patch before mutation. Binary changes,
mode changes, renames, copies, combined merge diffs, and sections without text hunks
are not supported. New-file Git patches must describe regular non-executable files.
Quoted Git paths must be expressed as literal paths in an envelope.

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

Repository discovery and status use Git's NUL-delimited filename output. Status
parsing is shared by prompt context and repository-map ranking; paths retain their
original characters for matching and are escaped when rendered for the model.
The filesystem fallback also excludes non-regular files, including named pipes.
Repository-map, ignore-rule, and skill caches compare device, file identity, size,
modification time, and change time. This detects preserved-timestamp replacements
and POSIX in-place edits without reading unchanged content on every cache check.
Instruction and source reads enforce their limits while loading data. Rendered
repository maps count headers and separators within their character budget.
Path-specific instruction lookup walks only ancestor scopes, retaining the same
ignored-directory and symlink rules as full instruction discovery.
Bounded binary reads accumulate small chunks, so a large configured ceiling does
not allocate that ceiling for each small file. Write and delete operations stream
their SHA-256 checks before checkpointing and again before committing changes.
The automatic stable map and request focus share `context.repo_map_chars`, including
their section headings. A zero budget skips source discovery for these maps;
instruction discovery, skill guidance, and an explicit `repo_map` tool remain available.

`ContextBuilder` creates a compact, task-specific view of the repository. It separates durable repository knowledge from transient model context:

- Git-aware file enumeration and ignore rules
- hierarchical instructions from root to active directory
- lazy skill discovery and loading
- language-aware symbol extraction
- query-ranked repository map
- optional Git status
- bounded tool results

The compactor builds validated turn bundles and a structured, untrusted historical artifact. It must meet a calculated provider target. Immutable versioned artifacts support exact resume reuse and incremental suffix compaction. The SQLite message log remains append-only and complete for replay and audit. See [COMPACTION.md](COMPACTION.md).

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
5. Namespace names as `mcp__<server>__<tool>`.
6. Map MCP safety annotations to Borealis effect categories.
7. Invoke through `tools/call` while retaining local policy enforcement.

Each server has isolated lifecycle and failure state.
Stdio requests apply their timeout to both sending and receiving. Incoming JSON
lines are limited to 16 MiB. Reader failures settle pending requests and reject new
requests; the client drains remaining output until shutdown to avoid pipe stalls.

### `protocol/`

The ACP server uses bidirectional newline-delimited JSON-RPC over stdio. It maps ACP sessions to `AgentRunner` instances and durable session IDs. Notifications are rendered from the same event stream used by the CLI, which prevents protocol behavior from diverging from in-process behavior.

Incoming JSON lines are limited to 16 MiB. Transport failure, disconnect, or
cancellation closes the input pipe, fails pending peer requests, cancels and drains
request handlers, and closes session runners. Peer request timeouts include writes.

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
