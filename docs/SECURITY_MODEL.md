# Security model

## Executive statement

Borealis assumes that **model output, repository content, tool output, MCP servers, and shell commands can be hostile or wrong**. The harness therefore validates structure, constrains paths, classifies effects, requests approval, limits processes, records checkpoints, and redacts likely secrets before persistence.

Borealis does not claim that a Python process running directly on the host can safely execute arbitrary adversarial code. For untrusted repositories, dependencies, prompts, or models, use the Docker driver or a stronger external sandbox such as a dedicated VM/container worker with restricted credentials and network.

## Trust boundaries

| Component | Default trust | Consequence |
|---|---|---|
| User / embedding application | Trusted to select workspace, policy, credentials, and sandbox. | Configuration can intentionally broaden access. |
| Language model | Untrusted proposer. | Every tool call is schema-validated and policy-evaluated. |
| Repository files and instructions | Untrusted data with task relevance. | They inform the model but cannot alter policy code or bypass tool validation. |
| Built-in tool implementations | Trusted code in the installed package. | Release review and tests focus heavily on this boundary. |
| Workspace plugins | Untrusted arbitrary Python unless explicitly enabled. | Disabled by default; enabling grants in-process code execution. |
| Entry-point plugins | Trusted as installed Python packages. | Package installation is a supply-chain decision. |
| MCP servers | External processes/services with untrusted descriptions/results. | Tools are namespaced and still pass through local policy; server process security remains external. |
| Provider APIs | External services. | Requests may contain repository context; configure providers according to data policy. |
| Native subprocesses | Host processes with reduced environment/resources. | Not a kernel sandbox. |
| Docker subprocesses | Isolated container with explicit mounts/resources. | Security depends on Docker daemon, image, host configuration, and mount choices. |

## Protected assets

- Files outside configured workspace roots
- Uncommitted repository work
- Provider API keys and environment secrets
- Host network and credentials
- Git remote integrity
- Session history and traces
- Process availability, disk, memory, and cost budgets

## Filesystem controls

`WorkspaceRoots` resolves all paths before use. It rejects:

- `..` traversal beyond an allowed root
- absolute paths outside roots
- symlinks whose resolved target leaves a root
- additional roots not explicitly configured by the caller/client

String path inputs expand `~` and environment variables before containment checks.
Filesystem `Path` objects and patch filenames retain their literal names.

Recursive text search validates each discovered file against the same roots before
reading it. It skips external symlinks and non-regular files, caps each read at the
configured file limit, and stops at the configured result or output limit.
Regular-expression matching runs in a dedicated worker with a configurable per-file
deadline. Cancellation, timeout, and normal completion terminate and reap the worker.
The worker uses the installed Python interpreter in isolated mode to match text
supplied by the tool. Literal search remains in the main process.

Mutation controls add optimistic concurrency:

- `write_file` requires an expected SHA when replacing an existing file.
- `replace_in_file` can require the observed SHA and rejects ambiguous replacement counts.
- `apply_patch` validates every target and hunk before its commit phase.
- checkpoints capture original files before an effect.
- writes use atomic temporary-file replacement.

These mechanisms prevent many stale-context and partial-edit failures. They do not replace operating-system permissions.

## Effect policy

Tools declare an effect class. Policy evaluates the effect, safety mode, approval mode, hard feature gates, and command assessment.

Hard gates are independent of interactive approval:

- `safety.network` controls network effects.
- `safety.allow_git_commit` controls commits.
- `safety.allow_git_push` controls pushes.
- `safety.allow_git_hooks` controls whether repository hooks may run during agent-managed Git operations.
- `safety.allow_shell` controls arbitrary shell execution.
- `safety.allow_outside_workspace` controls non-root paths.
- `safety.protected_paths` reserves repository-control and recovery paths from file mutations.

An approval cannot silently override a disabled hard gate. The embedding application must change configuration deliberately.

## Smart approvals

`--approval smart` is an opt-in automatic reviewer for deterministic ASK
operations. Deterministic ALLOW and DENY decisions do not call the judge. A
separate request uses the configured primary provider/model with no tools or
conversation continuation. It contains operator policy, workspace roots, the
current run's user-authored request and steering, and redacted original tool
arguments. It does not include the conversation transcript or repository files.
Descriptions, commands, and other proposed arguments are untrusted data, not user
authorization. Give explicit task scope when resuming work; prior-run prompts are
not included in the review.

The judge must return exactly `{"decision":"allow"}` or `{"decision":"deny"}`.
Extra fields, duplicate fields, tool calls, incomplete responses, and unknown
completion states fail closed. Missing context/provider, errors, timeout,
cancellation, and exhausted root budgets cannot authorize execution. There is no
human fallback or smart approval cache. A new user request received during review
invalidates the result. Each review has a single provider attempt with a 20-second
ceiling and shares root request/time/cost accounting, including failed attempts.
Reserved final-answer requests remain available. Detected and explicit
verification commands use the same review before each command executes.

Judge replies must contain one text decision in the supported provider envelope;
multiple completions and unexpected tool or content blocks are rejected. If user
intent changes during delivery of an allow receipt, a final denial receipt links
to it with `supersedes`, and the operation does not execute.

Smart approval cannot override plan mode, disabled network/shell/Git gates,
workspace roots, or protected file-tool targets. In smart mode the explicit
`git_commit` tool also requires `allow_git_commit=true`; the human modes retain
their existing explicit-commit approval behavior. Enabling a Git gate still
requires review for risky commits/pushes. These controls do not add kernel path
isolation to native shell commands or improve the command classifier.

`approval.decided` events persist `source=AI`, the tool/call identity, allow/deny,
and a fixed reason code. They do not contain the judge prompt, answer, or hidden
reasoning. Existing redaction still applies to other tool events. Redaction and
AI judgments can be wrong. Smart mode is not a security boundary and does not
replace an OS sandbox or operator review of sensitive work.

## Script execution

`script` runs JavaScript in a fresh QuickJS context in a dedicated Python
subprocess. Python starts in isolated mode (`-I`) with a minimal environment;
provider credentials, user-site imports, and workspace imports are not inherited.
The engine receives no Python callbacks, host objects, module loader, file,
network, process, or environment APIs. A serialized bridge admits registered
tools through the same schema, policy, approval, checkpoint, and mutation
accounting path as direct calls. Script is a workflow container, so read-only
programs remain usable in plan mode; each nested effect has its own policy gate.

The trusted supervisor validates frame size, JSON types, depth and call identity,
limits calls and in-flight reads, serializes side effects, bounds printed output
and archives, and enforces a wall deadline across all IPC and child work. The
worker configures engine heap, stack and CPU limits; POSIX also applies an OS CPU
limit. Timeouts and fatal host cancellation/budget errors stop the worker without
returning a catchable JavaScript tool result. All admitted tasks are drained and
the subprocess is reaped before its slot is released. Ordinary refused tools
reject their promises and stay visible as job errors even when caught.

This is embedded-engine isolation, not an OS/kernel sandbox against engine
vulnerabilities. The Script worker runs on the host even when shell tools use
the Docker driver. Put the entire Borealis process in a restricted container or
VM when that boundary is required. Plugins remain trusted Python code and MCP
servers retain their own process authority. There is no unsafe execution fallback
when QuickJS cannot start. Disable Script with trusted `[script] enabled = false`.

Jobs are owned by one run/session, cannot control other runs, and stop before
run teardown. Completion uses events and model notices, with no polling loop.
There is no durable job daemon or mutation replay after restart. Nested lifecycle
records and redacted quota-bounded artifacts survive; interrupted records are
marked with an unknown outcome on resume. Partial mutations are retained, and
workflow rollback or atomicity is not promised.

## Command controls

The shell tool parses and classifies common shell chains and detects categories such as:

- network clients and package downloads
- destructive filesystem operations
- privilege/shell escalation
- Git commit/push and remote changes
- process management
- interpreters and opaque shell constructs

The native driver provides:

- explicit working directory
- environment allowlist plus caller-approved variables
- timeout and cancellation
- captured output with a character limit
- Unix CPU and file-size limits where available
- process-group termination

The native driver cannot remove network access from an arbitrary host process.
When `safety.network = false`, Borealis therefore allows only a small set of
network-inert inspection commands under the native driver. It rejects
interpreters, nested shells, package scripts, build tools, commands with plugin
or subprocess hooks, and unknown executables. Use the Docker driver with its
network disabled to run those commands offline, or deliberately enable network
access.

The Docker driver adds:

- explicit bind mounts
- configurable image
- memory and CPU limits
- PID limits and dropped capabilities
- no-new-privileges
- read-only container filesystem with temporary writable directories
- network disabled unless configured

Docker is the recommended built-in driver for autonomous execution. A dedicated VM or rootless sandbox remains preferable for high-risk workloads.

## Prompt injection

Repository instructions and tool output can contain text designed to manipulate the model. Borealis mitigates this structurally rather than claiming to “solve” prompt injection:

- policy is executable host code, never model text
- tools have fixed schemas and effect classes
- path and command checks run after the model response
- repository instructions are labeled as untrusted project guidance in the system prompt
- network, commit, push, and outside-root access are hard gates
- MCP tool annotations are advisory inputs to stricter local policy
- secrets are not intentionally inserted into the model prompt

A model may still make poor choices inside its allowed authority. Keep authority narrow.

## Secret handling

Configuration points to API-key environment variable names rather than storing keys by default. The redactor:

- collects values from environment variables whose names look secret-bearing
- removes configured extra secret values
- recognizes common token and private-key shapes
- recursively redacts values under secret-like object keys

Redaction is best effort. Do not place production secrets in a repository that an agent can read. Prefer scoped, short-lived credentials and a minimal environment allowlist.

### ChatGPT/Codex OAuth credentials

The `chatgpt` provider accepts Codex-compatible OAuth credentials rather than
browser sessions. Borealis does not inspect ChatGPT cookies, browser databases,
local-storage values, or browser profiles.

For file-backed credentials, Borealis:

- requires a regular file owned by the current user on POSIX systems
- rejects group/world-readable files by default
- parses JWT payloads only for expiry/account metadata; the upstream service
  remains the authority that validates the token
- refreshes near-expiry tokens under an in-process and cross-process lock
- atomically replaces rotated credentials with mode `0600`
- prints only masked account metadata and never access/refresh tokens
- persists normalized messages, redacted runtime events, usage, encrypted
  Responses continuation state, and eligible exact-response cache entries
- discards complete native provider responses and never replays plaintext
  reasoning summaries to the provider

The default API and refresh endpoints are fixed to the ChatGPT/Codex and OpenAI
OAuth hosts. Because a custom endpoint can exfiltrate bearer or refresh tokens,
repository configuration alone cannot enable one. Both
`allow_custom_chatgpt_endpoints = true` and the externally supplied
`BOREALIS_ALLOW_CUSTOM_CHATGPT_ENDPOINTS=1` are required, and custom endpoints
must use HTTPS without embedded credentials.

`borealis auth login` invokes the official Codex CLI in a separate,
Borealis-managed `CODEX_HOME` configured for a file credential store. Reusing
an existing `~/.codex/auth.json` is read-only except when Borealis must persist a
rotated access/refresh token. `borealis auth logout` targets the managed store by
default so it does not unexpectedly remove another Codex client’s login.

## MCP and plugin risks

MCP server commands execute outside the shell tool and should be treated as trusted configuration. Workspace-defined MCP servers are disabled unless `BOREALIS_ENABLE_WORKSPACE_MCP=1` is set outside the repository. A malicious stdio server can consume its configured environment and perform any action permitted to its process identity. Borealis only controls the model-visible tool calls, not the internal behavior of the server. Remote annotations can increase risk, but only the local `read_only_tools` allowlist can reduce it.

Workspace configuration also cannot replace provider endpoints or credential sources unless `BOREALIS_ALLOW_WORKSPACE_PROVIDER_ENDPOINTS=1` is set externally. Prefer user-level or explicit configuration for trusted custom endpoints.

Repository-owned workspace configuration may set only non-authoritative context
discovery fields by default. Provider selection, budgets, safety, sandbox,
storage, cache, telemetry, and context resource ceilings require
`BOREALIS_ALLOW_WORKSPACE_AUTHORITY=1` outside the repository. That general
opt-in does not enable MCP servers or provider endpoints; their narrower opt-ins
remain mandatory.

Workspace plugins are disabled unless `BOREALIS_ENABLE_WORKSPACE_PLUGINS=1`. When enabled, they execute Python in the Borealis process and have the process’s full authority. Use entry-point packages or MCP for reviewable production extensions.

## Persistence and privacy

By default Borealis stores normalized messages, assembled model events, lifecycle/policy/tool/verification/error events, usage, and redacted rotating JSONL traces under the configured data directory. Streaming text, reasoning, and tool-call deltas remain live-only unless `storage.persist_event_deltas` is enabled. Provider-supplied reasoning summaries are model-output events and may be sensitive. The default exact-response cache can also retain them for its configured lifetime. Encrypted reasoning continuation state remains opaque. Repository snippets and other model outputs may also be sensitive.

For restricted environments:

- set a protected storage directory
- disable `storage.trace_jsonl` when unnecessary
- disable `cache.response_cache_enabled` when exact-response reuse is unnecessary
- keep `storage.retain_raw_provider_responses = false`
- delete or export sessions according to retention policy
- preview retained-data cleanup with `borealis maintenance --dry-run` before applying it
- encrypt the storage volume at the operating-system level

The core does not implement application-level database encryption.

## Recommended production profile

```toml
[safety]
mode = "workspace-write"
approval = "on-risk"
network = false
allow_shell = true
allow_git_commit = false
allow_git_push = false
allow_git_hooks = false
allow_outside_workspace = false
protected_paths = [".git", ".git/**", ".borealis/checkpoints", ".borealis/checkpoints/**"]
checkpoints = true

[sandbox]
driver = "docker"
docker_network = false
docker_memory = "2g"
docker_cpus = 2.0

[storage]
trace_jsonl = true
retain_raw_provider_responses = false
```

Also isolate the worker account, use read-only provider credentials where possible, mount only the intended workspace, and review diffs before merging.

## Known limitations

- Native execution is containment and resource bounding, not strong isolation.
- Docker isolation depends on the host daemon and image supply chain.
- Command classification is conservative but cannot understand every shell language construct.
- Redaction cannot guarantee removal of unknown secret formats.
- Provider-side retention and training policies are outside Borealis.
- No independent third-party security audit has been completed for version 0.1.3.
- No application-level encryption or multi-tenant authorization is built into the core.

## Delegated agent authority

Recursive agents share the root's request/time/cost budgets and cannot change
operator depth, cumulative descendant, provider concurrency or child-turn limits.
Every `delegate_task` call requires `capability: "read-only"` or `"write"`.
The caller selects read-only for investigation/review and write for implementation
or file edits. Selection cannot exceed the parent's actual permissions: plan-mode
parents and read-only descendants cannot grant write access. A rejected escalation
starts no child, consumes no descendant admission and dispatches no child provider
request. There is no silent downgrade or inference from task text.

Trusted `agent.swarm_allow_writes` defaults to true and is only an operator ceiling;
an explicit false (or `/swarm-writes off` for future interactive turns) rejects
write delegation even from a writer. True does not select a child's capability.
Write children get only built-in write, replace, delete, mkdir and patch tools.
The registry enforces the selected capability at execution as well as schema
selection, including hallucinated tool calls. Descendants cannot gain more
authority than their parent. Plan mode, approvals, allowed roots, protected paths,
stale hashes and checkpoints remain effective. Missing or invalid capability
arguments fail schema validation; historical receipts remain historical data.

Delegates cannot use shell, Script/job control, or custom/external mutation tools.
Each nested call uses the root runner's durable lifecycle; shared file changes
reach root verification. Provider slots are released before tools, and mutation
locks cover individual operations rather than recursive conversations. Fatal
budgets and cancellation drain child work and settle billable usage before the
root finishes. Task/result text in the swarm view is bounded, redacted through the
EventBus, and rendered literally with control/bidi characters neutralized.
