# Operations guide

## Deployment profiles

### Local developer

- Native process driver
- Interactive approval on risk
- Network disabled by default
- User-scoped SQLite storage
- Provider key in shell environment
- Persistent interactive shell started with `borealis`
- Owner-only local prompt history when readline is available

Start a normal local session with:

```bash
cd /path/to/project
borealis
```

Use `borealis --continue` or `borealis resume --last` after restarting the
process. Use `borealis run` only when a single non-interactive task is preferable.
Prompt history can be disabled with `--no-history`; durable model conversation
history remains governed by the session store.

For ChatGPT-plan authentication, run `borealis auth status` first. A secure,
readable file-backed Codex login is reused automatically. Otherwise run
`borealis auth login` (or `--device-code` on a headless host). Keep the
credential directory private to the worker account.

### Editor ACP subprocess

- Editor launches `borealis acp`
- Client supplies absolute workspace and optional additional roots
- Permission requests are handled by the editor
- Session IDs remain durable across reconnects
- MCP servers may be supplied per session

### CI worker

- Non-interactive mode
- Explicit provider/model/config
- Docker or externally isolated runner
- Network restricted to required provider and artifact endpoints
- Read-only credentials where possible
- JSON output and session export as build artifacts
- Hard cost, time, and turn budgets

### Remote service

- Run Borealis inside one isolated worker per task/session
- Keep authentication, authorization, queueing, tenant limits, and billing outside the core
- Mount only task-specific workspace volumes
- Use scoped provider/service credentials
- Stream EventBus data to the control plane
- Destroy worker state after exporting required evidence

## Storage

SQLite uses WAL mode. Back up the database and traces consistently or stop the writer before a raw file copy. The preferred portable backup is:

```bash
borealis sessions export SESSION_ID --output session.json
```

Checkpoint files live under the data directory and are bounded by configuration. They are recovery aids, not an archival backup system.

## Logging and observability

Three evidence streams are available:

1. Human interactive or one-shot terminal rendering
2. JSON events/final result with `borealis run --json`
3. Durable SQLite events and optional redacted JSONL trace

Useful operational fields include session ID, run ID, provider/model, turn, tool call ID, effect, duration, usage, cost, verification state, and stop reason.

The core sends no hosted telemetry. An embedding application can add an EventBus subscriber for OpenTelemetry, logs, metrics, or product analytics.

## Diagnostics

Run:

```bash
borealis doctor
borealis doctor --json
```

Diagnostics check configuration, provider credentials, storage access, Git, Docker when selected, and the effective workspace.

Before granting write authority to a new model or compatible API:

1. Run `borealis eval`.
2. Run a plan-mode repository summary.
3. Run a small write in a disposable Git worktree.
4. Inspect tool-call schemas and patch behavior.
5. Exercise cancellation and provider timeout.
6. Confirm cost and token accounting.
7. Enable broader effects incrementally.

## Failure recovery

- **Provider unavailable:** configure `agent.provider_fallbacks`, inspect retry events, and verify model/tool compatibility.
- **ChatGPT login expired/revoked:** run `borealis auth status`, then
  `borealis auth login`. A single pre-output authentication failure is refreshed
  automatically when a refresh token is available.
- **Context overflow:** lower initial context budgets, exclude generated files, reduce tool output, or resume after deterministic compaction.
- **Stuck loop:** inspect repeated tool calls and project instructions; lower `max_repeated_calls` for high-risk automation.
- **Verification failure:** use session history and Git diff; resume with the failing output as context.
- **Bad mutation:** list checkpoints and restore with `borealis rollback CHECKPOINT_ID`.
- **MCP failure:** remove/disable the server or narrow `allowed_tools`; core tools remain available.
- **Corrupt session database:** preserve the file, inspect SQLite integrity, and restore/export from backup. Do not silently replace it.

## Release artifacts

A release should include:

- source archive
- wheel and source distribution
- SHA-256 checksums
- validation manifest
- changelog and security documentation
- supported Python version statement

CI should build from a clean checkout, install the wheel in a fresh environment, run the offline acceptance suite, and verify the archive contains no caches, databases, traces, credentials, or local build state.

## Performance levers

- `agent.parallel_reads`: increase for high-latency MCP/read tools, lower for slow disks or constrained workers.
- `context.repo_map_chars`: increase for very large cross-cutting tasks; keep compact for routine fixes.
- `context.tool_output_chars`: avoid feeding full test logs or generated files back to the model.
- `agent.compact_at_ratio`: lower when providers reject near-limit requests or tool schemas are large.
- `agent.small_model`: route delegated investigations and optional LLM compaction to a cheaper model.
- provider caching and pricing fields: configure according to the current provider contract.

## Production qualification limits

The repository release gate is deterministic and offline. It validates adapter payload/parsing fixtures rather than making billable live API calls. Operators should maintain their own live canary tests for approved providers and models. No third-party penetration test or security audit is represented by the included validation manifest.
