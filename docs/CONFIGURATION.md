# Configuration reference

## Precedence

Values are merged from lowest to highest precedence:

1. built-in defaults
2. `~/.config/borealis/config.toml`
3. `<workspace>/.borealis/config.toml`
4. explicit `--config PATH`
5. environment variables
6. CLI runtime overrides

Unknown sections and keys are rejected. This prevents misspellings from silently producing an unsafe or ineffective configuration.

The workspace file is repository-owned and untrusted by default. Without an
external opt-in it may set only `context.include_git_status`,
`context.instruction_names`, `context.skill_dirs`, and `context.ignored_dirs`.
Put authority-bearing settings in the user configuration or an explicit
`--config` file, or set `BOREALIS_ALLOW_WORKSPACE_AUTHORITY=1` outside the
repository. MCP servers and provider endpoints retain their separate, narrower
opt-ins.

## Environment variables

Direct variables:

| Variable | Target |
|---|---|
| `BOREALIS_PROVIDER` | `agent.provider` |
| `BOREALIS_MODEL` | `agent.model` |
| `BOREALIS_SMALL_MODEL` | `agent.small_model` |
| `BOREALIS_MODE` | `safety.mode` |
| `BOREALIS_APPROVAL` | `safety.approval` |
| `BOREALIS_NETWORK` | `safety.network` |
| `BOREALIS_SANDBOX` | `sandbox.driver` |
| `BOREALIS_DATA_DIR` | `storage.directory` |
| `BOREALIS_ENABLE_WORKSPACE_PLUGINS` | Enables workspace Python plugins when set to `1`. |
| `BOREALIS_ENABLE_WORKSPACE_MCP` | Trust workspace configuration to start MCP servers. |
| `BOREALIS_ALLOW_WORKSPACE_PROVIDER_ENDPOINTS` | Trust workspace configuration to set provider endpoints or credential sources. |
| `BOREALIS_ALLOW_WORKSPACE_AUTHORITY` | Trust workspace provider selection, budgets, safety, sandbox, storage, cache, telemetry, and context resource ceilings. Does not enable MCP or provider endpoints. |
| `BOREALIS_CHATGPT_AUTH_FILE` | Explicit file-backed Codex `auth.json` used by the `chatgpt` provider. |
| `BOREALIS_CHATGPT_HOME` | Borealis-managed Codex credential directory. |
| `BOREALIS_CHATGPT_ACCOUNT_ID` | Account/workspace ID for an externally supplied `CODEX_ACCESS_TOKEN`. |
| `BOREALIS_ALLOW_CUSTOM_CHATGPT_ENDPOINTS` | Process-level opt-in required in addition to the matching provider flag before sending ChatGPT credentials to custom endpoints. |
| `CODEX_ACCESS_TOKEN` | Externally managed ChatGPT/Codex bearer token. |
| `CODEX_HOME` | Existing Codex credential directory checked for `auth.json`. |

Arbitrary values can be overridden with double-underscore paths:

```bash
export BOREALIS_CFG__AGENT__MAX_TURNS=24
export BOREALIS_CFG__SAFETY__ALLOW_GIT_COMMIT=false
export BOREALIS_CFG__PROVIDERS__OPENAI__TIMEOUT_SECONDS=240
```

Scalar strings are coerced to booleans, integers, floats, or null where unambiguous.

## `[agent]`

| Key | Default | Meaning |
|---|---:|---|
| `provider` | `"auto"` | Provider route name. Auto checks API-key providers, then a usable ChatGPT/Codex login, then local compatible and mock routes. |
| `model` | `""` | Global model override. Empty uses the selected provider’s model. |
| `small_model` | `""` | Optional cheaper/faster model for delegated tasks and LLM compaction. |
| `provider_fallbacks` | `[]` | Ordered provider route names tried after the primary route fails. |
| `reasoning_effort` | `"medium"` | Adapter hint for providers/models that expose reasoning controls. |
| `max_turns` | `60` | Maximum model turns per run. |
| `max_input_tokens` | `180000` | Context budget used for compaction and hard checks. |
| `compact_at_ratio` | `0.82` | Compact when estimated context reaches this fraction of the input budget. |
| `max_output_tokens` | `16000` | Requested per-turn output ceiling. |
| `max_time_seconds` | `3600` | End-to-end run wall-time ceiling. |
| `max_cost_usd` | `25.0` | Cumulative run cost ceiling when adapter pricing is configured. |
| `parallel_reads` | `8` | Maximum concurrent read-only tool calls. |
| `max_repeated_calls` | `3` | Repeated identical call threshold for stuck detection. |
| `auto_verify` | `true` | Run project verification after mutations. |
| `auto_verify_max_seconds` | `900` | Verification time budget. |
| `deterministic_compaction` | `true` | Use local structured compaction rather than another model call. |

## `[safety]`

| Key | Default | Meaning |
|---|---:|---|
| `mode` | `"workspace-write"` | `plan`, `workspace-write`, or `full`. |
| `approval` | `"on-risk"` | `never`, `on-risk`, or `always`. |
| `network` | `false` | Hard gate for network effects. On the native driver, commands capable of opening their own sockets are rejected because host network isolation cannot be enforced. |
| `allow_shell` | `true` | Hard gate for the shell tool. |
| `allow_git_commit` | `false` | Hard gate for Git commits. |
| `allow_git_push` | `false` | Hard gate for pushes. |
| `allow_git_hooks` | `false` | Permit repository Git hooks during agent-managed Git commands. Disabled commands use no-op hook paths. |
| `allow_outside_workspace` | `false` | Permit paths beyond configured roots. |
| `command_timeout_seconds` | `300` | Default subprocess timeout. |
| `max_process_output_chars` | `40000` | Captured stdout/stderr ceiling. |
| `env_allowlist` | minimal host variables | Environment names inherited by subprocesses. |
| `checkpoints` | `true` | Snapshot files before mutation. |
| `checkpoint_max_bytes` | `25000000` | Maximum bytes captured in one checkpoint. |
| `approval_cache` | `"session"` | Cache matching approvals for the active session. |
| `protected_paths` | `.git`, checkpoint store | Glob patterns that file mutation tools may never write, delete, or patch. |

## `[sandbox]`

| Key | Default | Meaning |
|---|---:|---|
| `driver` | `"native"` | `native` or `docker`. |
| `docker_image` | `"python:3.13-slim"` | Image used by Docker commands. Use a project-specific pinned image in production. |
| `docker_memory` | `"2g"` | Container memory limit. |
| `docker_cpus` | `2.0` | Container CPU limit. |
| `docker_network` | `false` | Enable container network only when `safety.network` also permits it. |
| `process_cpu_seconds` | `600` | Native Unix CPU resource limit. |
| `process_file_size_bytes` | `200000000` | Native Unix output-file resource limit. |

## `[context]`

| Key | Default | Meaning |
|---|---:|---|
| `repo_map_chars` | `28000` | Character budget for the ranked repository map. |
| `max_file_bytes` | `2000000` | Largest file eligible for text inspection. |
| `max_search_results` | `200` | Search result ceiling. |
| `tool_output_chars` | `24000` | Model-facing tool-output ceiling. |
| `include_git_status` | `true` | Include compact Git status in the initial prompt. |
| `instruction_names` | `AGENTS.md`, `BOREALIS.md`, `CLAUDE.md` | Hierarchical instruction filenames. |
| `skill_dirs` | `.agents/skills`, `.borealis/skills` | Skill discovery paths. |
| `ignored_dirs` | common generated dirs | Directory names excluded during discovery. |

`.gitignore` and `.borealisignore` add repository-specific exclusions.

## `[storage]`

| Key | Default | Meaning |
|---|---:|---|
| `directory` | `~/.local/share/borealis` | Persistent data root. |
| `database` | `sessions.sqlite3` | SQLite filename. |
| `trace_jsonl` | `true` | Append redacted lifecycle events to JSONL. |
| `retain_raw_provider_responses` | `false` | Reserved opt-in for adapter debugging; normalized persistence remains the default. |

## `[telemetry]`

| Key | Default | Meaning |
|---|---:|---|
| `enabled` | `true` | Enable local lifecycle telemetry/event handling. No hosted telemetry is built in. |
| `console_level` | `"info"` | Console rendering threshold. |
| `json_events` | `false` | Prefer machine-readable output in embedding contexts. |

## `[providers.<name>]`

| Key | Meaning |
|---|---|
| `type` | `openai`, `openrouter`, `chatgpt`, `openai_compatible`, `anthropic`, `gemini`, `mock`, or a custom registered type. |
| `base_url` | Provider API base URL. |
| `api_key_env` | Environment variable containing the key. Empty is allowed for local compatible servers. |
| `model` | Default model ID for this route. |
| `api_style` | Wire style such as `responses`, `chat`, `messages`, `interactions`, or `mock`. |
| `timeout_seconds` | Per-request timeout. |
| `max_retries` | Number of transient retries. |
| `initial_backoff_seconds` | Initial retry delay. |
| `max_backoff_seconds` | Retry cap. |
| `headers` | Additional HTTP headers. Secret-like values are redacted from printed configuration. |
| `site_url` | OpenRouter app-attribution URL (`HTTP-Referer`). Ignored by other built-ins. |
| `app_name` | OpenRouter app-attribution title (`X-Title`). Ignored by other built-ins. |
| `model_fallbacks` | OpenRouter backup model slugs tried after the primary model. |
| `provider_preferences` | OpenRouter provider-routing object, e.g. `allow_fallbacks`, `only`, `order`, `data_collection`, or `zdr`. |
| `extra_body` | Provider-specific top-level request fields. Primarily an escape hatch for OpenRouter extensions; explicit values can override adapter defaults. |
| `input_cost_per_million` | Optional local price used for budget accounting. |
| `output_cost_per_million` | Optional output price. |
| `cached_input_cost_per_million` | Optional cached-input price. |
| `cache_write_input_cost_per_million` | Optional cache-creation price. Zero derives it from the normal input price and provider cache-write multiplier. |
| `auth_file` | ChatGPT only: explicit path to a secure file-backed Codex `auth.json`. |
| `codex_home` | ChatGPT only: Codex credential directory whose `auth.json` should be considered. |
| `codex_command` | ChatGPT only: official Codex CLI command used by `borealis auth login/logout`. |
| `oauth_client_id` | ChatGPT only: OAuth client override. Custom values require the dual endpoint opt-in. |
| `refresh_url` | ChatGPT only: OAuth refresh endpoint override. Custom values require the dual endpoint opt-in and HTTPS. |
| `refresh_before_expiry_seconds` | ChatGPT only: proactively refresh this many seconds before access-token expiry. |
| `allow_insecure_auth_file` | ChatGPT only: permit broader POSIX auth-file permissions. Unsafe; defaults to `false`. |
| `allow_custom_chatgpt_endpoints` | ChatGPT only: first half of the dual opt-in for non-default API/OAuth endpoints. The process must also set `BOREALIS_ALLOW_CUSTOM_CHATGPT_ENDPOINTS=1`. |

Model identifiers change over time. Treat built-in IDs as dated defaults, not a guarantee of account availability.

## `[cache]`

| Key | Default | Meaning |
|---|---:|---|
| `prompt_cache_enabled` | `true` | Emit supported provider prompt-cache directives. |
| `anthropic_ttl` | `"5m"` | Anthropic cache TTL: `5m` or `1h`. |
| `conversation_cache_enabled` | `true` | Cache the growing Anthropic conversation in addition to the explicit stable prefix. |
| `response_cache_enabled` | `true` | Reuse exact successful text-only model responses locally. |
| `response_cache_ttl_seconds` | `300` | Exact-response cache lifetime. Zero disables writes. |
| `response_cache_max_entries` | `256` | Global LRU-like entry bound in the session database. |
| `adaptive` | `true` | Fall back to stable-prefix-only caching and a shorter compaction window after sustained write-heavy low hit rates. |
| `low_hit_rate_threshold` | `0.15` | Prompt-cache hit rate that triggers adaptive mode. |
| `adaptive_min_requests` | `3` | Minimum provider requests before adaptation. |
| `adaptive_min_input_tokens` | `4096` | Minimum logical input tokens before adaptation. |

The exact-response cache key covers the effective provider configuration, model, system context,
semantic message history, tool schemas, and generation settings. Prompt text is not stored in the
cache table; only its SHA-256 key, the final text, accounting metadata, and any validated
provider-selected continuation items are persisted. Full raw provider responses are never cached.

### ChatGPT credential resolution

The `chatgpt` provider checks, in order:

1. `CODEX_ACCESS_TOKEN`
2. `providers.chatgpt.auth_file`
3. `BOREALIS_CHATGPT_AUTH_FILE`
4. `providers.chatgpt.codex_home/auth.json`
5. `CODEX_HOME/auth.json`
6. `~/.codex/auth.json`
7. the Borealis-managed Codex home

On POSIX systems, file-backed credentials must be owned by the current user and
must not be group/world accessible unless the unsafe override is explicitly
enabled. Tokens are never included in `borealis config`, `auth status`, session
metadata, or trace events.

## `[mcp_servers.<name>]`

| Key | Meaning |
|---|---|
| `type` | `stdio` or `http`. |
| `command` / `args` | stdio server process. Use an absolute executable path in ACP-managed sessions. |
| `env` | Explicit server environment additions. |
| `url` / `headers` | Streamable HTTP endpoint and headers. |
| `timeout_seconds` | Initialization and request timeout. |
| `enabled` | Disable without deleting configuration. |
| `allowed_tools` | Optional exact allowlist; empty exposes all discovered tools through local policy. |
| `read_only_tools` | Exact locally trusted list allowed to use read-only policy. Server annotations cannot reduce local authorization. |

## Complete example

See [`docs/config.example.toml`](config.example.toml).
