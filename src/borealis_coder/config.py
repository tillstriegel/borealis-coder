"""Layered TOML configuration with environment overrides."""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

from .errors import ConfigurationError
from .util import coerce_scalar, deep_merge, ensure_private_directory, set_nested

T = TypeVar("T")


@dataclass(slots=True)
class AgentConfig:
    provider: str = "auto"
    model: str = ""
    small_model: str = ""
    provider_fallbacks: list[str] = field(default_factory=list)
    reasoning_effort: str = "medium"
    max_turns: int = 60
    max_input_tokens: int = 180_000
    compact_at_ratio: float = 0.82
    max_output_tokens: int = 16_000
    max_time_seconds: int = 3_600
    max_cost_usd: float = 25.0
    parallel_reads: int = 8
    max_repeated_calls: int = 3
    auto_verify: bool = True
    auto_verify_max_seconds: int = 900
    deterministic_compaction: bool = True


@dataclass(slots=True)
class SafetyConfig:
    mode: str = "workspace-write"  # plan | workspace-write | full
    approval: str = "on-risk"  # never | on-risk | always
    network: bool = False
    allow_shell: bool = True
    allow_git_commit: bool = False
    allow_git_push: bool = False
    allow_git_hooks: bool = False
    allow_outside_workspace: bool = False
    command_timeout_seconds: int = 300
    max_process_output_chars: int = 40_000
    env_allowlist: list[str] = field(
        default_factory=lambda: [
            "PATH",
            "HOME",
            "USER",
            "TMPDIR",
            "TEMP",
            "TMP",
            "LANG",
            "LC_ALL",
            "TERM",
            "CI",
            "NO_COLOR",
        ]
    )
    checkpoints: bool = True
    checkpoint_max_bytes: int = 25_000_000
    approval_cache: str = "session"
    protected_paths: list[str] = field(
        default_factory=lambda: [
            ".git",
            ".git/**",
            ".borealis/checkpoints",
            ".borealis/checkpoints/**",
        ]
    )


@dataclass(slots=True)
class SandboxConfig:
    driver: str = "native"  # native | docker
    docker_image: str = "python:3.13-slim"
    docker_memory: str = "2g"
    docker_cpus: float = 2.0
    docker_network: bool = False
    process_cpu_seconds: int = 600
    process_file_size_bytes: int = 200_000_000


@dataclass(slots=True)
class ContextConfig:
    repo_map_chars: int = 28_000
    max_file_bytes: int = 2_000_000
    max_search_results: int = 200
    tool_output_chars: int = 24_000
    include_git_status: bool = True
    instruction_names: list[str] = field(
        default_factory=lambda: ["AGENTS.md", "BOREALIS.md", "CLAUDE.md"]
    )
    skill_dirs: list[str] = field(
        default_factory=lambda: [".agents/skills", ".borealis/skills"]
    )
    ignored_dirs: list[str] = field(
        default_factory=lambda: [
            ".git",
            ".borealis",
            ".venv",
            "venv",
            "node_modules",
            "dist",
            "build",
            "target",
            "coverage",
            "__pycache__",
        ]
    )


@dataclass(slots=True)
class StorageConfig:
    directory: str = "~/.local/share/borealis"
    database: str = "sessions.sqlite3"
    trace_jsonl: bool = True
    retain_raw_provider_responses: bool = False


@dataclass(slots=True)
class ProviderConfig:
    type: str = "openai"
    base_url: str = ""
    api_key_env: str = ""
    model: str = ""
    api_style: str = ""
    timeout_seconds: int = 180
    max_retries: int = 4
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    headers: dict[str, str] = field(default_factory=dict)
    site_url: str = ""
    app_name: str = ""
    model_fallbacks: list[str] = field(default_factory=list)
    provider_preferences: dict[str, Any] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    cached_input_cost_per_million: float = 0.0
    # ChatGPT/Codex OAuth settings. Ignored by other provider types.
    auth_file: str = ""
    codex_home: str = ""
    codex_command: str = "codex"
    oauth_client_id: str = ""
    refresh_url: str = ""
    refresh_before_expiry_seconds: int = 300
    allow_insecure_auth_file: bool = False
    # Custom ChatGPT/Codex endpoints can receive bearer or refresh tokens. To
    # prevent an untrusted workspace config from redirecting credentials, this
    # flag is honored only together with BOREALIS_ALLOW_CUSTOM_CHATGPT_ENDPOINTS=1.
    allow_custom_chatgpt_endpoints: bool = False


@dataclass(slots=True)
class MCPServerConfig:
    type: str = "stdio"  # stdio | http
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 30
    enabled: bool = True
    allowed_tools: list[str] = field(default_factory=list)
    read_only_tools: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TelemetryConfig:
    enabled: bool = True
    console_level: str = "info"
    json_events: bool = False


@dataclass(slots=True)
class Config:
    agent: AgentConfig = field(default_factory=AgentConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    mcp_servers: dict[str, MCPServerConfig] = field(default_factory=dict)
    source_files: list[str] = field(default_factory=list)

    @property
    def storage_dir(self) -> Path:
        value = Path(os.path.expandvars(os.path.expanduser(self.storage.directory)))
        return value.resolve()

    @property
    def database_path(self) -> Path:
        return self.storage_dir / self.storage.database

    def provider(self, name: str | None = None) -> tuple[str, ProviderConfig]:
        selected = name or self.agent.provider
        if selected == "auto":
            selected = self._auto_provider()
        if selected not in self.providers:
            raise ConfigurationError(f"Unknown provider: {selected}")
        return selected, self.providers[selected]

    def _auto_provider(self) -> str:
        priority = ["openai", "openrouter", "anthropic", "gemini", "chatgpt", "openai_compatible"]
        for name in priority:
            provider = self.providers.get(name)
            if not provider:
                continue
            if provider.api_key_env and os.getenv(provider.api_key_env):
                return name
            if provider.type == "chatgpt":
                from .auth import has_chatgpt_credentials

                if has_chatgpt_credentials(provider):
                    return name
            if provider.type == "openai_compatible" and provider.model and provider.base_url:
                return name
        return "mock"

    def resolved_model(self, provider_name: str, provider: ProviderConfig) -> str:
        model = self.agent.model or os.getenv("BOREALIS_MODEL", "") or provider.model
        if not model:
            raise ConfigurationError(
                f"No model configured for provider {provider_name!r}; use --model or BOREALIS_MODEL"
            )
        return model

    def to_dict(self, *, redact_secrets: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if redact_secrets:
            for provider in data.get("providers", {}).values():
                provider["headers"] = _redact_secret_mapping(provider.get("headers", {}))
                provider["extra_body"] = _redact_secret_mapping(provider.get("extra_body", {}))
            for server in data.get("mcp_servers", {}).values():
                server["headers"] = {
                    key: "[REDACTED]" if "auth" in key.lower() or "key" in key.lower() else value
                    for key, value in server.get("headers", {}).items()
                }
                server["env"] = {key: "[REDACTED]" for key in server.get("env", {})}
        return data


def _redact_secret_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            lower = str(key).lower()
            if any(token in lower for token in ("auth", "key", "token", "password", "secret", "credential", "cookie")):
                output[str(key)] = "[REDACTED]"
            else:
                output[str(key)] = _redact_secret_mapping(item)
        return output
    if isinstance(value, list):
        return [_redact_secret_mapping(item) for item in value]
    return value


DEFAULTS: dict[str, Any] = {
    "agent": asdict(AgentConfig()),
    "safety": asdict(SafetyConfig()),
    "sandbox": asdict(SandboxConfig()),
    "context": asdict(ContextConfig()),
    "storage": asdict(StorageConfig()),
    "telemetry": asdict(TelemetryConfig()),
    "providers": {
        "openai": asdict(
            ProviderConfig(
                type="openai",
                base_url="https://api.openai.com/v1",
                api_key_env="OPENAI_API_KEY",
                model="gpt-5.4-mini",
                api_style="responses",
            )
        ),
        "openrouter": asdict(
            ProviderConfig(
                type="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key_env="OPENROUTER_API_KEY",
                model="openai/gpt-5.4-mini",
                api_style="chat",
                site_url=os.getenv("OPENROUTER_SITE_URL", ""),
                app_name=os.getenv("OPENROUTER_APP_NAME", "Borealis Coder"),
            )
        ),
        "chatgpt": asdict(
            ProviderConfig(
                type="chatgpt",
                base_url="https://chatgpt.com/backend-api/codex",
                model="gpt-5.6-terra",
                api_style="responses",
            )
        ),
        "anthropic": asdict(
            ProviderConfig(
                type="anthropic",
                base_url="https://api.anthropic.com/v1",
                api_key_env="ANTHROPIC_API_KEY",
                model="claude-opus-5",
                api_style="messages",
            )
        ),
        "gemini": asdict(
            ProviderConfig(
                type="gemini",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                api_key_env="GEMINI_API_KEY",
                model="gemini-3.6-flash",
                api_style="interactions",
            )
        ),
        "openai_compatible": asdict(
            ProviderConfig(
                type="openai_compatible",
                base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://127.0.0.1:11434/v1"),
                api_key_env="OPENAI_COMPATIBLE_API_KEY",
                model=os.getenv("OPENAI_COMPATIBLE_MODEL", ""),
                api_style="chat",
            )
        ),
        "mock": asdict(
            ProviderConfig(type="mock", model="deterministic", api_style="mock", max_retries=0)
        ),
    },
    "mcp_servers": {},
}


def _construct(dataclass_type: type[T], values: dict[str, Any]) -> T:
    valid = {item.name for item in fields(dataclass_type)}
    unknown = sorted(set(values) - valid)
    if unknown:
        raise ConfigurationError(
            f"Unknown keys for {dataclass_type.__name__}: {', '.join(unknown)}"
        )
    return dataclass_type(**values)


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise ConfigurationError(f"Invalid TOML in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration root in {path} must be a table")
    return value


def _environment_overrides() -> dict[str, Any]:
    result: dict[str, Any] = {}
    direct = {
        "BOREALIS_PROVIDER": ("agent", "provider"),
        "BOREALIS_MODEL": ("agent", "model"),
        "BOREALIS_SMALL_MODEL": ("agent", "small_model"),
        "BOREALIS_MODE": ("safety", "mode"),
        "BOREALIS_APPROVAL": ("safety", "approval"),
        "BOREALIS_NETWORK": ("safety", "network"),
        "BOREALIS_SANDBOX": ("sandbox", "driver"),
        "BOREALIS_DATA_DIR": ("storage", "directory"),
        "NO_COLOR": ("telemetry", "no_color"),
    }
    for env_name, path in direct.items():
        if env_name in os.environ and env_name != "NO_COLOR":
            set_nested(result, path, coerce_scalar(os.environ[env_name]))
    prefix = "BOREALIS_CFG__"
    for name, value in os.environ.items():
        if name.startswith(prefix):
            path = [part.lower() for part in name[len(prefix) :].split("__") if part]
            set_nested(result, path, coerce_scalar(value))
    return result


def load_config(
    workspace: Path,
    *,
    explicit_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    workspace = workspace.resolve()
    merged: dict[str, Any] = dict(DEFAULTS)
    sources: list[str] = []
    user_config = Path("~/.config/borealis/config.toml").expanduser()
    workspace_config = workspace / ".borealis" / "config.toml"
    candidates = [user_config, workspace_config]
    if explicit_path:
        candidates.append(explicit_path.expanduser())
    for path in candidates:
        if path.is_file():
            values = _read_toml(path)
            if path == workspace_config:
                _validate_workspace_authority(values, path)
            merged = deep_merge(merged, values)
            sources.append(str(path.resolve()))
    merged = deep_merge(merged, _environment_overrides())
    if overrides:
        merged = deep_merge(merged, overrides)

    known_sections = {
        "agent",
        "safety",
        "sandbox",
        "context",
        "storage",
        "telemetry",
        "providers",
        "mcp_servers",
    }
    unknown_sections = sorted(set(merged) - known_sections)
    if unknown_sections:
        raise ConfigurationError(f"Unknown configuration sections: {', '.join(unknown_sections)}")

    providers = {
        name: _construct(ProviderConfig, dict(value))
        for name, value in dict(merged.get("providers") or {}).items()
    }
    mcp_servers = {
        name: _construct(MCPServerConfig, dict(value))
        for name, value in dict(merged.get("mcp_servers") or {}).items()
    }
    config = Config(
        agent=_construct(AgentConfig, dict(merged.get("agent") or {})),
        safety=_construct(SafetyConfig, dict(merged.get("safety") or {})),
        sandbox=_construct(SandboxConfig, dict(merged.get("sandbox") or {})),
        context=_construct(ContextConfig, dict(merged.get("context") or {})),
        storage=_construct(StorageConfig, dict(merged.get("storage") or {})),
        telemetry=_construct(TelemetryConfig, dict(merged.get("telemetry") or {})),
        providers=providers,
        mcp_servers=mcp_servers,
        source_files=sources,
    )
    validate_config(config)
    ensure_private_directory(config.storage_dir)
    return config


def _validate_workspace_authority(values: dict[str, Any], path: Path) -> None:
    if values.get("mcp_servers") and not _enabled("BOREALIS_ENABLE_WORKSPACE_MCP"):
        raise ConfigurationError(
            f"Workspace MCP configuration in {path} is disabled by default. "
            "Move it to the user config or set BOREALIS_ENABLE_WORKSPACE_MCP=1."
        )
    providers = values.get("providers") or {}
    endpoint_keys = {
        "api_key_env",
        "auth_file",
        "base_url",
        "codex_home",
        "headers",
        "oauth_client_id",
        "refresh_url",
    }
    if isinstance(providers, dict) and not _enabled(
        "BOREALIS_ALLOW_WORKSPACE_PROVIDER_ENDPOINTS"
    ):
        for name, provider in providers.items():
            if isinstance(provider, dict) and endpoint_keys.intersection(provider):
                raise ConfigurationError(
                    f"Workspace provider endpoint or credential configuration for {name!r} "
                    f"in {path} is disabled by default. Move it to the user config, use an "
                    "explicit config file, or set BOREALIS_ALLOW_WORKSPACE_PROVIDER_ENDPOINTS=1."
                )


def _enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def validate_config(config: Config) -> None:
    if config.agent.max_turns < 1:
        raise ConfigurationError("agent.max_turns must be positive")
    if config.agent.max_time_seconds < 1:
        raise ConfigurationError("agent.max_time_seconds must be positive")
    if not 0.5 <= config.agent.compact_at_ratio < 1.0:
        raise ConfigurationError("agent.compact_at_ratio must be between 0.5 and 1.0")
    if config.safety.mode not in {"plan", "workspace-write", "full"}:
        raise ConfigurationError("safety.mode must be plan, workspace-write, or full")
    if config.safety.approval not in {"never", "on-risk", "always"}:
        raise ConfigurationError("safety.approval must be never, on-risk, or always")
    if config.sandbox.driver not in {"native", "docker"}:
        raise ConfigurationError("sandbox.driver must be native or docker")
    for name, provider in config.providers.items():
        if provider.timeout_seconds <= 0:
            raise ConfigurationError(f"providers.{name}.timeout_seconds must be positive")
        if provider.refresh_before_expiry_seconds < 0:
            raise ConfigurationError(
                f"providers.{name}.refresh_before_expiry_seconds cannot be negative"
            )
        if provider.type == "chatgpt" and provider.api_style not in {"", "responses"}:
            raise ConfigurationError(
                f"providers.{name}.api_style must be responses for ChatGPT/Codex auth"
            )


SAMPLE_CONFIG = """# Borealis Coder workspace configuration

[agent]
provider = "auto"
# model = "gpt-5.4-mini"
max_turns = 60
max_cost_usd = 25.0
auto_verify = true

[safety]
mode = "workspace-write"
approval = "on-risk"
network = false
checkpoints = true

[sandbox]
driver = "native" # use "docker" for a stronger isolation boundary
# docker_image = "python:3.13-slim"

[context]
repo_map_chars = 28000
tool_output_chars = 24000

# ChatGPT plan via the official Codex login store:
# [agent]
# provider = "chatgpt"
# model = "gpt-5.6-terra"
# Run: borealis auth login
# Existing ~/.codex/auth.json credentials are discovered automatically.

# OpenRouter example:
# [providers.openrouter]
# type = "openrouter"
# model = "anthropic/claude-sonnet-4.6"
# model_fallbacks = ["openai/gpt-5.4-mini"]
#
# [providers.openrouter.provider_preferences]
# data_collection = "deny"
# zdr = true

# Add MCP servers as needed:
# [mcp_servers.files]
# type = "stdio"
# command = "npx"
# args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
"""
