"""ChatGPT-plan credentials compatible with the open-source Codex client.

Borealis deliberately does not read browser cookies or private browser-session
storage.  It consumes the documented local Codex credential record produced by
``codex login`` (or an explicit ``CODEX_ACCESS_TOKEN``), sends requests to the
Codex Responses endpoint, and refreshes file-backed OAuth tokens using the same
public client identifier and token endpoint as Codex.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import shutil
import ssl
import stat
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import ConfigurationError, ProviderAuthenticationError
from ..util import atomic_write_text, json_dumps

if TYPE_CHECKING:
    from ..config import ProviderConfig

CHATGPT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CHATGPT_TOKEN_REFRESH_URL = "https://auth.openai.com/oauth/token"
# Public OAuth client identifier used by the open-source Codex CLI.
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
MAX_AUTH_FILE_BYTES = 2_000_000


@dataclass(slots=True, frozen=True)
class ChatGPTCredentials:
    access_token: str
    account_id: str | None
    refresh_token: str | None = None
    id_token: str | None = None
    source: Path | None = None
    expires_at: int | None = None
    email: str | None = None
    plan: str | None = None
    fedramp: bool = False
    file_data: dict[str, Any] | None = None

    @property
    def refreshable(self) -> bool:
        return bool(self.refresh_token and self.source and self.file_data is not None)

    def expires_within(self, seconds: int, *, now: float | None = None) -> bool:
        if self.expires_at is None:
            return False
        current = time.time() if now is None else now
        return self.expires_at <= current + max(0, seconds)


@dataclass(slots=True, frozen=True)
class ChatGPTCredentialStatus:
    available: bool
    source: str | None = None
    account: str | None = None
    email: str | None = None
    plan: str | None = None
    expires_at: int | None = None
    refreshable: bool = False
    secure: bool | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "source": self.source,
            "account": self.account,
            "email": self.email,
            "plan": self.plan,
            "expires_at": self.expires_at,
            "refreshable": self.refreshable,
            "secure": self.secure,
            "message": self.message,
        }


def managed_codex_home() -> Path:
    override = os.getenv("BOREALIS_CHATGPT_HOME", "").strip()
    if override:
        return _expanded_path(override)
    if os.name == "nt" and os.getenv("LOCALAPPDATA"):
        return (Path(os.environ["LOCALAPPDATA"]) / "Borealis" / "chatgpt").resolve()
    xdg = os.getenv("XDG_DATA_HOME", "").strip()
    if xdg:
        return (_expanded_path(xdg) / "borealis" / "chatgpt").resolve()
    return (Path.home() / ".local" / "share" / "borealis" / "chatgpt").resolve()


def candidate_auth_files(config: ProviderConfig | None = None) -> list[Path]:
    """Return deterministic credential candidates without exposing their contents."""

    candidates: list[Path] = []
    explicit = getattr(config, "auth_file", "") if config is not None else ""
    if explicit:
        candidates.append(_expanded_path(explicit))
    env_file = os.getenv("BOREALIS_CHATGPT_AUTH_FILE", "").strip()
    if env_file:
        candidates.append(_expanded_path(env_file))
    configured_home = getattr(config, "codex_home", "") if config is not None else ""
    if configured_home:
        candidates.append(_expanded_path(configured_home) / "auth.json")
    env_home = os.getenv("CODEX_HOME", "").strip()
    if env_home:
        candidates.append(_expanded_path(env_home) / "auth.json")
    candidates.extend(
        [
            Path.home() / ".codex" / "auth.json",
            managed_codex_home() / "auth.json",
        ]
    )
    output: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = path.expanduser().resolve(strict=False)
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            output.append(resolved)
    return output


def has_chatgpt_credentials(config: ProviderConfig | None = None) -> bool:
    if os.getenv("CODEX_ACCESS_TOKEN", "").strip():
        return True
    manager = ChatGPTCredentialManager(config)
    return manager.status().available


class ChatGPTCredentialManager:
    """Load, validate, refresh, and safely report Codex-compatible OAuth data."""

    def __init__(
        self,
        config: ProviderConfig | None = None,
        *,
        refresh_transport: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
    ) -> None:
        self.config = config
        self._refresh_transport = refresh_transport or _post_refresh_json
        self._credentials: ChatGPTCredentials | None = None
        self._refresh_lock = asyncio.Lock()

    @property
    def refresh_window(self) -> int:
        return max(0, int(getattr(self.config, "refresh_before_expiry_seconds", 300)))

    @property
    def allow_insecure_file(self) -> bool:
        return bool(getattr(self.config, "allow_insecure_auth_file", False))

    @property
    def refresh_url(self) -> str:
        configured = str(getattr(self.config, "refresh_url", "") or "").strip()
        value = (
            configured
            or os.getenv("CODEX_REFRESH_TOKEN_URL_OVERRIDE", "").strip()
            or CHATGPT_TOKEN_REFRESH_URL
        )
        _validate_credential_endpoint(
            value,
            expected=CHATGPT_TOKEN_REFRESH_URL,
            config=self.config,
            label="ChatGPT token refresh endpoint",
        )
        return value

    @property
    def client_id(self) -> str:
        configured = str(getattr(self.config, "oauth_client_id", "") or "").strip()
        value = (
            configured
            or os.getenv("CODEX_APP_SERVER_LOGIN_CLIENT_ID", "").strip()
            or CODEX_OAUTH_CLIENT_ID
        )
        if value != CODEX_OAUTH_CLIENT_ID and not custom_chatgpt_endpoints_allowed(self.config):
            raise ConfigurationError(
                "A custom ChatGPT OAuth client ID was configured. For safety, custom ChatGPT "
                "authentication endpoints require both providers.chatgpt."
                "allow_custom_chatgpt_endpoints = true and "
                "BOREALIS_ALLOW_CUSTOM_CHATGPT_ENDPOINTS=1."
            )
        return value

    def load(self, *, force: bool = False) -> ChatGPTCredentials:
        if self._credentials is not None and not force:
            return self._credentials

        env_token = os.getenv("CODEX_ACCESS_TOKEN", "").strip()
        if env_token:
            claims = _combined_claims(None, env_token)
            account_id = (
                os.getenv("BOREALIS_CHATGPT_ACCOUNT_ID", "").strip()
                or _claim(claims, "chatgpt_account_id")
                or None
            )
            self._credentials = ChatGPTCredentials(
                access_token=env_token,
                account_id=account_id,
                expires_at=_integer_claim(_jwt_payload(env_token), "exp"),
                email=_claim(claims, "email"),
                plan=_claim(claims, "chatgpt_plan_type"),
                fedramp=bool(_claim(claims, "is_fedramp") is True),
            )
            return self._credentials

        explicit = bool(
            (getattr(self.config, "auth_file", "") if self.config else "")
            or os.getenv("BOREALIS_CHATGPT_AUTH_FILE", "").strip()
            or (getattr(self.config, "codex_home", "") if self.config else "")
        )
        errors: list[str] = []
        for path in candidate_auth_files(self.config):
            if not path.is_file():
                continue
            try:
                credentials = self._load_file(path)
            except (ConfigurationError, OSError, ValueError) as error:
                if explicit:
                    raise ConfigurationError(str(error)) from error
                errors.append(f"{path}: {error}")
                continue
            self._credentials = credentials
            return credentials

        detail = f" Checked: {', '.join(str(path) for path in candidate_auth_files(self.config))}."
        if errors:
            detail += " Rejected stores: " + "; ".join(errors)
        raise ConfigurationError(
            "No usable ChatGPT/Codex login was found. Run `borealis auth login` "
            "or `codex login`, then select provider `chatgpt`." + detail
        )

    async def ensure_valid(self, *, force_refresh: bool = False) -> ChatGPTCredentials:
        credentials = self.load(force=force_refresh)
        should_refresh = force_refresh or credentials.expires_within(self.refresh_window)
        if not should_refresh:
            return credentials
        if not credentials.refreshable:
            if credentials.expires_within(0):
                raise ProviderAuthenticationError(
                    "The ChatGPT access token has expired and cannot be refreshed. "
                    "Run `borealis auth login`."
                )
            return credentials
        return await self.refresh(force=force_refresh)

    async def refresh(self, *, force: bool = True) -> ChatGPTCredentials:
        async with self._refresh_lock:
            initial = self.load()
            if not initial.refreshable:
                raise ProviderAuthenticationError(
                    "The selected ChatGPT credentials do not contain a refresh token. "
                    "Run `borealis auth login`."
                )
            assert initial.source is not None
            lock = _CredentialFileLock(initial.source.with_suffix(initial.source.suffix + ".lock"))
            await asyncio.to_thread(lock.acquire)
            try:
                current = self.load(force=True)
                if current.access_token != initial.access_token and not current.expires_within(
                    self.refresh_window
                ):
                    return current
                if not force and not current.expires_within(self.refresh_window):
                    return current
                if not current.refresh_token:
                    raise ProviderAuthenticationError(
                        "The ChatGPT refresh token is missing. Run `borealis auth login`."
                    )
                try:
                    data = await self._refresh_transport(
                        self.refresh_url,
                        {
                            "client_id": self.client_id,
                            "grant_type": "refresh_token",
                            "refresh_token": current.refresh_token,
                        },
                    )
                except Exception as error:
                    raise ProviderAuthenticationError(
                        "ChatGPT credentials could not be refreshed. Run `borealis auth login` "
                        "if the refresh token was revoked or expired."
                    ) from error
                if not isinstance(data, dict):
                    raise ProviderAuthenticationError(
                        "The ChatGPT token service returned an invalid response."
                    )
                access_token = str(data.get("access_token") or current.access_token)
                refresh_token = str(data.get("refresh_token") or current.refresh_token)
                id_token = str(data.get("id_token") or current.id_token or "") or None
                if not access_token:
                    raise ProviderAuthenticationError(
                        "The ChatGPT token service did not return an access token."
                    )
                self._persist_refreshed(
                    current,
                    access_token=access_token,
                    refresh_token=refresh_token,
                    id_token=id_token,
                )
                return self.load(force=True)
            finally:
                await asyncio.to_thread(lock.release)

    def status(self) -> ChatGPTCredentialStatus:
        try:
            credentials = self.load(force=True)
        except Exception as error:
            return ChatGPTCredentialStatus(available=False, message=str(error))
        secure = None
        if credentials.source is not None:
            secure = _auth_file_is_secure(credentials.source)
        expired = credentials.expires_within(0)
        message = "expired" if expired else "ready"
        if expired and credentials.refreshable:
            message = "expired; refresh available"
        return ChatGPTCredentialStatus(
            available=not expired or credentials.refreshable,
            source=str(credentials.source) if credentials.source else "CODEX_ACCESS_TOKEN",
            account=_mask(credentials.account_id),
            email=credentials.email,
            plan=credentials.plan,
            expires_at=credentials.expires_at,
            refreshable=credentials.refreshable,
            secure=secure,
            message=message,
        )

    def _load_file(self, path: Path) -> ChatGPTCredentials:
        _validate_auth_file(path, allow_insecure=self.allow_insecure_file)
        if path.stat().st_size > MAX_AUTH_FILE_BYTES:
            raise ConfigurationError(f"ChatGPT auth file is unexpectedly large: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ConfigurationError(f"Invalid JSON in ChatGPT auth file {path}: {error}") from error
        if not isinstance(data, dict):
            raise ConfigurationError(f"ChatGPT auth file must contain a JSON object: {path}")
        mode = str(data.get("auth_mode") or "chatgpt").lower().replace("-", "_")
        tokens = data.get("tokens")
        if mode not in {"chatgpt", "chatgpt_auth_tokens"} or not isinstance(tokens, dict):
            raise ConfigurationError(f"Credential store is not a ChatGPT/Codex OAuth login: {path}")
        access_token = str(tokens.get("access_token") or "").strip()
        if not access_token:
            raise ConfigurationError(f"ChatGPT credential store has no access token: {path}")
        id_token = str(tokens.get("id_token") or "").strip() or None
        refresh_token = str(tokens.get("refresh_token") or "").strip() or None
        claims = _combined_claims(id_token, access_token)
        account_id = str(tokens.get("account_id") or "").strip() or _claim(
            claims, "chatgpt_account_id"
        )
        return ChatGPTCredentials(
            access_token=access_token,
            refresh_token=refresh_token,
            id_token=id_token,
            account_id=str(account_id) if account_id else None,
            source=path.resolve(),
            expires_at=_integer_claim(_jwt_payload(access_token), "exp"),
            email=_string_or_none(_claim(claims, "email")),
            plan=_string_or_none(_claim(claims, "chatgpt_plan_type")),
            fedramp=bool(_claim(claims, "is_fedramp") is True),
            file_data=data,
        )

    def _persist_refreshed(
        self,
        current: ChatGPTCredentials,
        *,
        access_token: str,
        refresh_token: str,
        id_token: str | None,
    ) -> None:
        if current.source is None or current.file_data is None:
            raise ProviderAuthenticationError("Environment-only ChatGPT credentials cannot be persisted")
        data = dict(current.file_data)
        tokens = dict(data.get("tokens") or {})
        tokens["access_token"] = access_token
        tokens["refresh_token"] = refresh_token
        if id_token:
            tokens["id_token"] = id_token
            claims = _combined_claims(id_token, access_token)
            account_id = _claim(claims, "chatgpt_account_id")
            if account_id:
                tokens["account_id"] = str(account_id)
        data["auth_mode"] = data.get("auth_mode") or "chatgpt"
        data["tokens"] = tokens
        data["last_refresh"] = datetime.now(UTC).isoformat()
        atomic_write_text(current.source, json_dumps(data, pretty=True) + "\n", mode=0o600)
        self._credentials = None


def launch_codex_login(
    *,
    codex_home: Path | None = None,
    codex_command: str = "codex",
    device_code: bool = False,
) -> Path:
    """Run the official Codex login flow in a Borealis-managed credential home."""

    home = (codex_home or managed_codex_home()).expanduser().resolve()
    home.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(home, 0o700)
    except OSError:
        pass
    config_path = home / "config.toml"
    if not config_path.exists():
        atomic_write_text(config_path, 'cli_auth_credentials_store = "file"\n', mode=0o600)

    command = shlex.split(codex_command)
    if not command:
        raise ConfigurationError("providers.chatgpt.codex_command cannot be empty")
    executable = shutil.which(command[0])
    if executable is None and not Path(command[0]).is_file():
        raise ConfigurationError(
            f"Codex CLI executable not found: {command[0]}. Install the official Codex CLI, "
            "then rerun `borealis auth login`."
        )
    argv = [*command, "login"]
    if device_code:
        argv.append("--device-auth")
    env = dict(os.environ)
    env["CODEX_HOME"] = str(home)
    result = subprocess.run(argv, env=env, check=False)
    if result.returncode != 0:
        raise ConfigurationError(f"Codex login exited with status {result.returncode}")
    auth_file = home / "auth.json"
    if not auth_file.is_file():
        raise ConfigurationError(
            f"Codex reported success but did not create the expected credential file: {auth_file}"
        )
    _validate_auth_file(auth_file, allow_insecure=False)
    return auth_file


def logout_managed_chatgpt(
    *,
    codex_home: Path | None = None,
    codex_command: str = "codex",
) -> Path:
    """Revoke/delete only Borealis-managed ChatGPT credentials by default."""

    home = (codex_home or managed_codex_home()).expanduser().resolve()
    auth_file = home / "auth.json"
    command = shlex.split(codex_command)
    executable_available = bool(command) and (
        shutil.which(command[0]) is not None or Path(command[0]).is_file()
    )
    if executable_available and auth_file.exists():
        env = dict(os.environ)
        env["CODEX_HOME"] = str(home)
        result = subprocess.run([*command, "logout"], env=env, check=False)
        if result.returncode == 0:
            return auth_file
    auth_file.unlink(missing_ok=True)
    return auth_file


class _CredentialFileLock:
    def __init__(self, path: Path, *, timeout: float = 10.0) -> None:
        self.path = path
        self.timeout = timeout
        self._held = False

    def acquire(self) -> None:
        deadline = time.monotonic() + self.timeout
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(f"{os.getpid()}\n")
                self._held = True
                return
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > 120:
                        self.path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise ProviderAuthenticationError(
                        f"Timed out waiting for ChatGPT credential lock: {self.path}"
                    )
                time.sleep(0.05)

    def release(self) -> None:
        if self._held:
            self.path.unlink(missing_ok=True)
            self._held = False


def _validate_auth_file(path: Path, *, allow_insecure: bool) -> None:
    try:
        info = path.stat()
    except OSError as error:
        raise ConfigurationError(f"Cannot read ChatGPT auth file {path}: {error}") from error
    if not stat.S_ISREG(info.st_mode):
        raise ConfigurationError(f"ChatGPT auth path is not a regular file: {path}")
    if os.name != "nt":
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise ConfigurationError(f"ChatGPT auth file is not owned by the current user: {path}")
        if not allow_insecure and info.st_mode & 0o077:
            raise ConfigurationError(
                f"ChatGPT auth file permissions are too broad: {path}. Run `chmod 600 {path}`."
            )


def _auth_file_is_secure(path: Path) -> bool:
    try:
        info = path.stat()
    except OSError:
        return False
    if os.name == "nt":
        return True
    owner_ok = not hasattr(os, "getuid") or info.st_uid == os.getuid()
    return owner_ok and not bool(info.st_mode & 0o077)


def _expanded_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve(strict=False)


def custom_chatgpt_endpoints_allowed(config: ProviderConfig | None) -> bool:
    configured = bool(getattr(config, "allow_custom_chatgpt_endpoints", False))
    process_opt_in = os.getenv("BOREALIS_ALLOW_CUSTOM_CHATGPT_ENDPOINTS", "").strip().lower()
    return configured and process_opt_in in {"1", "true", "yes", "on"}


def validate_chatgpt_api_base_url(
    value: str,
    config: ProviderConfig | None,
) -> str:
    normalized = value.rstrip("/") or CHATGPT_CODEX_BASE_URL
    expected = CHATGPT_CODEX_BASE_URL.rstrip("/")
    if normalized == expected:
        return expected
    _validate_credential_endpoint(
        normalized,
        expected=expected,
        config=config,
        label="ChatGPT/Codex API endpoint",
    )
    return normalized


def _validate_credential_endpoint(
    value: str,
    *,
    expected: str,
    config: ProviderConfig | None,
    label: str,
) -> None:
    if value.rstrip("/") == expected.rstrip("/"):
        return
    if not custom_chatgpt_endpoints_allowed(config):
        raise ConfigurationError(
            f"A custom {label} was configured. For safety, custom ChatGPT endpoints require "
            "both providers.chatgpt.allow_custom_chatgpt_endpoints = true and "
            "BOREALIS_ALLOW_CUSTOM_CHATGPT_ENDPOINTS=1."
        )
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ConfigurationError(
            f"Custom {label} must be an HTTPS URL without embedded credentials: {value}"
        )


def _jwt_payload(token: str | None) -> dict[str, Any]:
    if not token:
        return {}
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    encoded = parts[1]
    encoded += "=" * (-len(encoded) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _combined_claims(id_token: str | None, access_token: str | None) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for payload in (_jwt_payload(access_token), _jwt_payload(id_token)):
        combined.update(payload)
        nested = payload.get("https://api.openai.com/auth")
        if isinstance(nested, dict):
            combined.update(nested)
        profile = payload.get("https://api.openai.com/profile")
        if isinstance(profile, dict):
            combined.update(profile)
    return combined


def _claim(claims: dict[str, Any], name: str) -> Any:
    return claims.get(name)


def _integer_claim(claims: dict[str, Any], name: str) -> int | None:
    value = claims.get(name)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _mask(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 10:
        return value[:2] + "…" + value[-2:]
    return value[:6] + "…" + value[-4:]


async def _post_refresh_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    def send() -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json_dumps(payload).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "borealis-coder-chatgpt-auth",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=30, context=ssl.create_default_context()
            ) as response:
                raw = response.read(MAX_AUTH_FILE_BYTES)
        except urllib.error.HTTPError as error:
            # Do not include token-service response bodies: they may contain
            # account-specific diagnostics and are unnecessary to remediate auth.
            raise ProviderAuthenticationError(
                f"ChatGPT token refresh failed with HTTP {error.code}"
            ) from error
        except urllib.error.URLError as error:
            raise ProviderAuthenticationError(
                f"ChatGPT token refresh connection failed: {error.reason}"
            ) from error
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProviderAuthenticationError(
                "ChatGPT token refresh returned invalid JSON"
            ) from error
        if not isinstance(value, dict):
            raise ProviderAuthenticationError(
                "ChatGPT token refresh returned a non-object response"
            )
        return value

    return await asyncio.to_thread(send)
