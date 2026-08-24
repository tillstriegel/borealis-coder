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

By default Borealis stores normalized messages, events, tool lifecycle, usage, and redacted JSONL traces under the configured data directory. Provider-supplied reasoning summaries are model-output events and may be sensitive. The default exact-response cache can also retain them for its configured lifetime. Encrypted reasoning continuation state remains opaque. Repository snippets and other model outputs may also be sensitive.

For restricted environments:

- set a protected storage directory
- disable `storage.trace_jsonl` when unnecessary
- disable `cache.response_cache_enabled` when exact-response reuse is unnecessary
- keep `storage.retain_raw_provider_responses = false`
- delete or export sessions according to retention policy
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
