# Borealis Coder

**Borealis Coder is a provider-portable, policy-first, resumable AI coding harness for autonomous repository work.** It combines a compact asynchronous agent runtime with deterministic editing tools, durable SQLite sessions, context-efficient repository discovery, checkpoints, verification, MCP extensions, and an ACP v2 editor bridge.

The project is deliberately dependency-light: the runtime uses the Python standard library plus prompt-toolkit for stable interactive terminal input on Python 3.11–3.13. Provider SDKs are not required. This keeps the execution boundary inspectable and makes the harness practical for local tools, CI workers, containers, and embedded applications.

> **Release status:** `0.1.3` is a production-oriented beta/release candidate. The offline acceptance suite, unit/integration tests, wheel build, and clean-install checks are part of the release process. Live provider behavior still depends on the selected API, model, account, and network environment. Use Docker or another operating-system sandbox for untrusted repositories or models; the native driver is resource-bounded process execution, not a kernel security boundary.

## Why Borealis

Modern coding agents need more than an LLM call wrapped around a shell. Borealis treats the following as first-class runtime subsystems:

- **Agent orchestration:** streaming model turns, tool cycles, cancellation, steering, fallback routes, stuck-loop detection, budgets, and compacted context.
- **Deterministic code mutation:** SHA-guarded writes, exact replacements, transactional patch validation, bounded deletion, checkpoints, and rollback.
- **Repository intelligence:** Git-aware discovery, hierarchical `AGENTS.md` instructions, lazy skills, language-aware symbol extraction, ranked repository maps, search, and focused reads.
- **Least-privilege execution:** workspace-root containment, symlink-escape prevention, centralized policies, interactive approvals, command classification, network gates, and Docker isolation.
- **Durability and interoperability:** SQLite/WAL sessions, event traces, full-history export, MCP tools, ACP v2 sessions, and Python entry-point plugins.
- **Visible model progress:** provider-supplied reasoning summaries stream separately from assistant answers and are shown by default.
- **Evidence-based completion:** repository-aware verification, Git diff/status reporting, usage/cost accounting, and structured terminal or JSON output.

The design was synthesized from a code-level review of the most-starred open-source coding harnesses as of August 20, 2026. See [the research report](docs/RESEARCH.md) and [feature matrix](docs/FEATURE_MATRIX.md).

## Quick start

### Install from the repository

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Initialize a workspace:

```bash
cd /path/to/project
borealis init
```

Set a provider key, then start the persistent coding shell:

```bash
export OPENAI_API_KEY='...'
borealis
```

Enter tasks one after another. Borealis keeps the same durable session, streams
assistant responses, displays tool progress, and returns to the prompt after each
turn. You can also provide the first task directly:

```bash
borealis "Fix the failing tests, verify the result, and summarize the changes"
```

The explicit one-shot automation surface remains available through
`borealis run`.

Or use an eligible ChatGPT account through Codex-compatible authentication:

```bash
# Reuses a secure, readable file-backed Codex login when one exists.
borealis auth status

# Otherwise launches the official Codex sign-in flow into a Borealis-managed store.
borealis auth login

borealis \
  --provider chatgpt \
  --model gpt-5.6-terra
```

`borealis auth login --device-code` is available for headless systems. This route
uses Codex OAuth credentials and the Codex Responses backend. It does not inspect
ChatGPT browser cookies, local browser profiles, or private web-session storage.

Borealis automatically selects the first configured provider whose credentials are available. You can select a route explicitly:

```bash
borealis \
  --provider anthropic \
  --model claude-opus-5 \
  --mode workspace-write \
  --approval on-risk
```

### Run completely offline

The built-in deterministic mock provider exercises the full model/tool/session loop without network access:

```bash
borealis eval
```

For an inspectable end-to-end write cycle:

```bash
BOREALIS_PROVIDER=mock borealis run --non-interactive OFFLINE_WRITE_DEMO
```

## Command surface

| Command | Purpose |
|---|---|
| `borealis` | Open the persistent interactive coding shell (the primary local workflow). |
| `borealis chat` / `borealis interactive` | Explicit aliases for the interactive shell. |
| `borealis resume [SESSION_ID]` | Resume a saved session in the interactive shell; use `--last` for the latest. |
| `borealis run` | Execute one autonomous task for scripts and automation. |
| `borealis init` | Create `.borealis/config.toml`, ignore rules, skills, and plugin directories. |
| `borealis auth` | Inspect, create, or remove ChatGPT/Codex authentication. |
| `borealis doctor` | Check provider configuration, tools, Git, Docker, storage, and workspace access. |
| `borealis tools` | Print the effective built-in, plugin, and MCP tool palette. |
| `borealis config` | Print the merged configuration with secret-bearing fields redacted. |
| `borealis sessions` | List, inspect, export, or delete durable sessions. |
| `borealis rollback` | List or restore pre-mutation checkpoints. |
| `borealis maintenance` | Preview or apply event, trace, and checkpoint retention. |
| `borealis acp` | Serve Agent Client Protocol v2 over stdio. |
| `borealis eval` | Run deterministic offline acceptance checks. |

Examples:

```bash
# Open a new interactive session
borealis

# Resume the newest session in this workspace
borealis --continue
borealis resume --last

# Resume a specific session and immediately send a prompt
borealis resume sess_abc123 "Now implement the proposed change"

# Machine-readable one-shot events and final result
borealis run --json "Explain the current architecture"

# Inspect/export durable session history
borealis sessions show sess_abc123
borealis sessions export sess_abc123 --output session.json

# Stronger command isolation in the interactive shell
borealis --sandbox docker
```


## Interactive CLI

Running `borealis` without a subcommand opens a long-lived terminal shell over the
same `AgentRunner`, EventBus, policy engine, and SQLite session store used by
one-shot runs and ACP. A conversation therefore survives multiple prompts and can
be resumed after the process exits.

The interactive surface is the **Aurora Shell**: a responsive, color-aware inline
TUI backed by prompt-toolkit, with a session prompt, framed assistant responses, live
elapsed-time pulses, a model/tool/verification activity rail, structured slash
commands, and compact turn receipts. It preserves terminal scrollback and falls
back to clean uncolored output for pipes, basic terminals, and `NO_COLOR`.

Common commands inside the shell:

| Command | Effect |
|---|---|
| `/status` | Show provider, model, safety settings, active session, usage, and cost. |
| `/sessions [N]` | List recent sessions for the current workspace. |
| `/resume [ID or last]` | Resume a durable session without leaving the shell. |
| `/history [N]` | Show recent persisted messages from the active session. |
| `/new` or `/clear` | Begin a fresh conversation while retaining the old session. |
| `/provider [NAME MODEL]` | List providers or switch provider/model for this process. |
| `/model MODEL` | Switch models while keeping the current durable session. |
| `/mode`, `/approval`, `/network`, `/verify`, `/sandbox` | Inspect or change runtime controls. |
| `/tools [FILTER]`, `/doctor`, `/rollback [ID]` | Inspect tools, diagnose the runtime, or restore a checkpoint. |
| `/paste` | Enter a multiline task and finish it with a line containing only `/end`. |
| `/exit` or `/quit` | Close the shell. |

Press Ctrl+C while a turn is running to cancel that turn and retain the shell and
session. At the input prompt, Ctrl+C exits. Prefix a literal slash-leading prompt
with another slash, such as `//explain /api/users`.

Useful startup options include `--continue`, `--resume SESSION_ID`, `--no-stream`,
`--show-tool-output`, and `--no-history`. Prompt history is stored separately from
model conversation history under the configured Borealis data directory and uses
mode `0600` on Unix. See [the complete interactive CLI reference](docs/INTERACTIVE_CLI.md).

## Safety modes

Borealis applies policy before every tool invocation.

| Mode | Behavior |
|---|---|
| `plan` | Read-only repository work plus structured plan updates. |
| `workspace-write` | Reads and deterministic mutations inside configured workspace roots; risky effects require policy approval. |
| `full` | Enables the broadest effect surface, while preserving explicit network, commit, push, and path gates. |

The default is `workspace-write`, `approval = "on-risk"`, and `network = false`.

Important invariants:

1. Relative file paths resolve from the session workspace.
2. Resolved paths must remain inside an allowed root unless explicitly configured otherwise.
3. Symlink traversal cannot escape a root.
4. Existing-file writes require the caller to provide the SHA-256 observed during a prior read.
5. Multi-file patches are prevalidated before any mutation and checkpointed before application.
6. Protected recovery and repository-control paths such as `.git` and `.borealis/checkpoints` cannot be mutated through file tools.
7. Network commands, Git hooks, Git commits, and Git pushes have independent hard gates.
8. Provider keys and likely secret values are redacted from traces and persisted errors.

With the native process driver, `network = false` allows only a small set of
network-inert inspection commands. Other commands are blocked because a host
process cannot otherwise be prevented from opening sockets. Select the Docker
driver to run them inside a network-disabled container.

Read [SECURITY.md](SECURITY.md) and [the security model](docs/SECURITY_MODEL.md) before enabling autonomous execution in sensitive environments.

## Configuration

Configuration is merged in this order:

1. Built-in defaults
2. `~/.config/borealis/config.toml`
3. `<workspace>/.borealis/config.toml`
4. An explicit `--config` file
5. Environment variables
6. CLI overrides

Minimal workspace configuration:

```toml
[context]
include_git_status = true
instruction_names = ["AGENTS.md", "BOREALIS.md", "CLAUDE.md"]
skill_dirs = [".agents/skills", ".borealis/skills"]
```

Workspace configuration is repository-owned and cannot change provider
selection, budgets, safety, sandbox, storage, cache, telemetry, or context
resource ceilings by default. Put those settings in the user configuration or
an explicit `--config` file. To trust them in a workspace file, set
`BOREALIS_ALLOW_WORKSPACE_AUTHORITY=1` outside the repository. MCP servers and
provider endpoints still require their dedicated opt-ins.

Trusted user or explicit-file provider override:

```toml
[providers.openai]
type = "openai"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
model = "gpt-5.4-mini"
api_style = "responses"
timeout_seconds = 180
max_retries = 4
```

ChatGPT plan through Codex OAuth:

```toml
[agent]
provider = "chatgpt"
model = "gpt-5.6-terra"

[providers.chatgpt]
type = "chatgpt"
base_url = "https://chatgpt.com/backend-api/codex"
model = "gpt-5.6-terra"
api_style = "responses"
# auth_file = "~/.codex/auth.json" # optional explicit file-backed Codex login
```

Credential resolution checks `CODEX_ACCESS_TOKEN`, explicit configuration, an
existing file-backed Codex login, and the Borealis-managed login directory. Run
`borealis auth status --json` to inspect the selected source without printing any
token. Model availability and usage limits follow the signed-in ChatGPT account.
Because this subscription route does not expose API-dollar billing to Borealis,
local `cost_usd` remains zero; account-side usage limits remain authoritative.

OpenRouter:

```bash
export OPENROUTER_API_KEY='sk-or-v1-...'
borealis run --provider openrouter --model anthropic/claude-sonnet-4.6 "Fix the failing tests"
```

```toml
[agent]
provider = "openrouter"

[providers.openrouter]
type = "openrouter"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
model = "anthropic/claude-sonnet-4.6"
api_style = "chat"
app_name = "Borealis Coder"
# site_url = "https://your-project.example"
model_fallbacks = ["openai/gpt-5.4-mini"]

[providers.openrouter.provider_preferences]
allow_fallbacks = true
data_collection = "deny"
zdr = true
```

`api_style = "responses"` is also supported for OpenRouter models that expose the Responses API.
OpenRouter's own provider failover remains active by default; `model_fallbacks` adds model-level
fallbacks on top of it.

OpenAI-compatible local server:

```toml
[agent]
provider = "openai_compatible"

[providers.openai_compatible]
type = "openai_compatible"
base_url = "http://127.0.0.1:11434/v1"
model = "your-local-model"
api_style = "chat"
```

See [configuration reference](docs/CONFIGURATION.md) and [provider adapters](docs/PROVIDERS.md).

## Tools

The built-in palette includes:

- `read_file`, `list_directory`, `glob_files`, `grep`
- `write_file`, `replace_in_file`, `delete_file`, `make_directory`
- `apply_patch`
- `shell`
- `git_status`, `git_diff`, `git_log`, `git_commit`, `git_push`
- `fetch_url`
- `update_plan`
- `repo_map`, `read_instructions`, `read_skill`
- `verify`
- `delegate_task`

All tool schemas use strict JSON-object contracts. Read-only calls can execute concurrently; mutations execute sequentially to preserve deterministic state.

## Context and instructions

Borealis builds model context incrementally:

- It respects Git and `.borealisignore` exclusions.
- It discovers `AGENTS.md`, `BOREALIS.md`, and `CLAUDE.md` from the repository root down to the active path.
- It discovers project skills in `.agents/skills` and `.borealis/skills`, but injects only compact descriptions until a skill is explicitly loaded.
- It extracts symbols for common programming languages and ranks files against the task query.
- It gives the model search/read tools instead of front-loading the entire repository.
- It compacts validated turn bundles to a calculated provider target while retaining append-only full history and reusable versioned artifacts in SQLite. See [Compaction v2](docs/COMPACTION.md).
- It keeps a deterministic system-context prefix for provider prompt caching and appends Git status and request-ranked context afterward.
- It reports provider cache reads, writes, hit rate, and savings, and can reuse short-lived exact text-only responses without replaying tools. Validated provider continuation state is retained so cached responses remain resumable.
- Typing `/` at the interactive prompt opens a compact command selector with descriptions; continue typing to narrow it and use Tab to complete.

## Verification

`verify` detects common project types and runs focused checks under the same process policy as ordinary shell commands. Current built-in detection covers:

- Python (`ruff`, `pyright`, and `pytest` when installed; otherwise `unittest`)
- JavaScript/TypeScript (`npm` or `pnpm` lint, typecheck, test, and build scripts when present)
- Rust (`cargo fmt`, offline `cargo check`, and offline `cargo test`)
- Go (`go test ./...` and `go vet ./...`)
- Make-based repositories (`make test` when no more specific project plan was detected)

Automatic verification runs after mutations when `agent.auto_verify = true`. A failed verification changes the process exit status even when the model produced a nominal end-turn response.

## MCP extensions

Borealis can connect to Model Context Protocol servers over stdio or Streamable HTTP. Tools are namespaced by server name and pass through the local policy engine.

```toml
[mcp_servers.issues]
type = "stdio"
command = "python"
args = ["/absolute/path/to/server.py"]
timeout_seconds = 30
allowed_tools = ["search_issues", "read_issue"]
read_only_tools = ["search_issues", "read_issue"]
```

Workspace-defined MCP servers require the external
`BOREALIS_ENABLE_WORKSPACE_MCP=1` opt-in. Remote MCP annotations can increase
local risk, but only `read_only_tools` can lower a tool to read-only policy.

MCP server failures are isolated: a failed optional server does not corrupt the core registry or session state. See [protocol documentation](docs/PROTOCOLS.md).

## ACP editor integration

Run the stdio server:

```bash
borealis acp
```

The server implements the ACP v2 baseline plus advertised session-delete, additional-directory, and MCP capabilities for editor and client integrations:

- initialization and capability negotiation
- session create, list, resume, close, and delete
- optional full-history replay
- foreground prompt acceptance and streamed updates
- cancellation and mid-run steering
- plan, message, tool, usage, and state updates
- client-mediated permission requests
- additional workspace roots
- per-session MCP server configuration

## Embed the runtime

```python
import asyncio
from pathlib import Path

from borealis_coder import build_runner, load_config


async def main() -> None:
    workspace = Path(".").resolve()
    config = load_config(workspace)
    runner = await build_runner(workspace, config=config, interactive=False)
    try:
        result = await runner.run("Inspect the repository and propose three improvements")
        print(result.text)
        print(result.session_id, result.usage.total_tokens)
    finally:
        await runner.close()


asyncio.run(main())
```

## Extension points

- **Tools:** package entry points in the `borealis.tools` group.
- **Workspace plugins:** Python files in `.borealis/plugins`, disabled unless `BOREALIS_ENABLE_WORKSPACE_PLUGINS=1`.
- **Skills:** Markdown skill files in configured skill directories.
- **MCP:** local or remote tool servers.
- **Providers:** register a `ProviderFactory` in a custom `ProviderRegistry` passed to `build_runner`.
- **Clients:** use ACP v2 or subscribe directly to the in-process `EventBus`.

See [extending Borealis](docs/EXTENDING.md).

## Development and release validation

```bash
make test
make smoke
make check
```

The release gate checks:

- all tests and offline agent acceptance
- Python compilation
- strict tool schemas
- package metadata and required documentation
- secret-marker hygiene
- wheel/sdist construction and clean installation
- CLI import, `--version`, `doctor`, and `eval`

The generated validation manifest records the commands, environment, source statistics, artifact hashes, and known limitations.

## Architecture documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Research and benchmarked harnesses](docs/RESEARCH.md)
- [Frontier feature matrix](docs/FEATURE_MATRIX.md)
- [Security model](docs/SECURITY_MODEL.md)
- [Configuration reference](docs/CONFIGURATION.md)
- [Provider adapters](docs/PROVIDERS.md)
- [MCP and ACP protocols](docs/PROTOCOLS.md)
- [Extension guide](docs/EXTENDING.md)
- [Operations](docs/OPERATIONS.md)
- [Release checklist](docs/RELEASE_CHECKLIST.md)

## License

MIT. Borealis Coder is a clean-room implementation. It incorporates architectural lessons from public repositories and protocols, not copied source code.
