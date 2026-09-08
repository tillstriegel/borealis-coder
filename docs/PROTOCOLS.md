# Protocols

## MCP client

Borealis implements Model Context Protocol clients for stdio and Streamable HTTP tool servers.

### Lifecycle

1. Establish the configured transport.
2. Send `initialize` with implementation information and supported protocol version.
3. Send `notifications/initialized`.
4. Call `tools/list`, following pagination cursors.
5. Convert discovered tools into local registry entries.
6. Invoke with `tools/call`.
7. Close the process/session on runner shutdown.

Tool discovery preserves opaque pagination cursors, including empty strings, and
fails explicitly if a cursor repeats or has an invalid type.

### Names and policy

Remote tool names are normalized and namespaced as:

```text
mcp__<server-name>__<tool-name>
```

MCP annotations such as read-only, destructive, idempotent, or open-world behavior inform the initial effect classification. Borealis local policy remains authoritative. An MCP server cannot label a network/destructive tool safe and thereby bypass a hard gate configured by the operator.

Use `allowed_tools` to expose a narrow subset from a server.

### Stdio

The server process communicates with one JSON-RPC object per line, up to 16 MiB.
Request timeouts cover writes and replies. Borealis drains stderr in bounded chunks
to prevent pipe deadlock. A stdout read failure or oversized message fails pending and
later requests; remaining output drains until the process shuts down.
Shutdown gives the server a chance to exit after stdin closes, then stops remaining
children in its POSIX process group. Cancellation also stops the group and finishes
local process and pipe waiters through bounded cleanup.

### Streamable HTTP

The HTTP client supports JSON and event-stream responses, session IDs, request timeouts, and server-provided pagination. Authentication headers come from configuration and are redacted from printed state.

## ACP v2 agent server

`borealis acp` exposes the harness as an Agent Client Protocol v2 subprocess over bidirectional newline-delimited JSON-RPC.

Incoming JSON lines have a 16 MiB limit. Disconnects, read failures, and cancellation
close the input transport, fail pending peer requests, cancel and drain handlers,
and close session runners. Peer request timeouts cover writes and replies.
On POSIX, stdout pipes use async flow control instead of a blocking executor writer. A write
waits for its buffered bytes to drain; a later write first drains any frame left by
a timeout. Disconnects abort queued output and close the output transport, so a
client that stops reading cannot hold process shutdown open. Regular-file stdout
redirection remains supported. Windows retains its existing stdout writer.

### Initialization

The client calls `initialize`. Borealis returns:

- protocol version 2
- implementation name/title/version
- baseline session capabilities
- prompt text and embedded-resource support
- stdio and HTTP MCP support
- session listing/deletion and additional-directory support

Unsupported capabilities are omitted rather than implied.

### Session lifecycle

Implemented methods:

- `session/new`
- `session/list`
- `session/resume`
- `session/close`
- `session/delete`
- `session/prompt`
- `session/cancel`
- `session/update` handling where applicable

A session has a primary absolute `cwd`. Additional directories expand the effective root set only when provided by the client. The same root set is used by filesystem tools and process mounts.

Failed or cancelled session creation and resume requests close the runtime resources
opened for that request before returning the failure.
Resuming an active session cancels and drains its old prompt tasks before closing
their runtime and installing the replacement.
Resume, close, delete, and prompt requests are serialized per session. Cancellation
remains immediate, and lifecycle requests for different sessions can proceed independently.

### Prompt lifecycle

`session/prompt` accepts the message and returns an empty result. Foreground progress is emitted as `session/update` notifications:

- accepted user message with agent-owned message ID
- state `running`
- optional plan updates
- agent message chunks
- tool pending/in-progress/content/completed/failed
- usage updates
- final state `idle` with stop reason

When the model-turn limit prevents completion, the adapter sends an agent message with the
recovery instruction before the final `idle`/`max_turns` state update. The durable session can
then continue with another prompt.

When a prompt arrives while the session is running, Borealis queues it as steering for the next safe turn boundary rather than starting a second conflicting mutation loop.

### Permission requests

When policy returns “ask,” the server calls the client’s `session/request_permission` method. While blocked it emits `requires_action`; after a decision it resumes `running`. Decisions feed the same `ApprovalManager` used by the terminal UI.

### Replay

`session/resume` can restore without replay or replay durable history from the beginning. Message IDs remain stable for the replay stream. SQLite is the authoritative history source.

### Cancellation

`session/cancel` sets the session cancellation event. Active provider streams and subprocesses observe cancellation. The server emits an idle state with a cancellation stop reason and preserves completed lifecycle events.

## Protocol compatibility strategy

Protocol adapters are intentionally thin projections over `AgentRunner` and `EventBus`. New protocol fields should be added at the projection layer unless they represent a real runtime capability. This avoids implementing UI-only state twice or claiming a capability that the core cannot enforce.
