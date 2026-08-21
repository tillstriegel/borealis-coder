from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from borealis_coder import cli
from borealis_coder.auth import (
    ChatGPTCredentialManager,
    ChatGPTCredentials,
    launch_codex_login,
)
from borealis_coder.config import AgentConfig, Config, ProviderConfig, load_config
from borealis_coder.errors import ConfigurationError, ProviderAuthenticationError
from borealis_coder.models import Message, ModelResponse, ProviderRequest, Role
from borealis_coder.providers.base import ProviderStreamEvent
from borealis_coder.providers.chatgpt import ChatGPTProvider
from borealis_coder.providers.openai import OpenAIProvider
from borealis_coder.providers.registry import ProviderRegistry
from borealis_coder.tools.base import object_schema


def jwt(payload: dict[str, object]) -> str:
    def encode(value: object) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{encode({'alg': 'none'})}.{encode(payload)}.signature"


def write_auth(path: Path, *, expires: int, access: str | None = None) -> None:
    access_token = access or jwt(
        {
            "exp": expires,
            "email": "till@example.test",
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct_1234567890",
                "chatgpt_plan_type": "plus",
            },
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "id_token": jwt(
                        {
                            "email": "till@example.test",
                            "https://api.openai.com/auth": {
                                "chatgpt_account_id": "acct_1234567890",
                                "chatgpt_plan_type": "plus",
                            },
                        }
                    ),
                    "access_token": access_token,
                    "refresh_token": "refresh-old",
                    "account_id": "acct_1234567890",
                },
                "last_refresh": "2026-08-20T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


class ChatGPTCredentialTests(unittest.IsolatedAsyncioTestCase):
    async def test_load_status_headers_and_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=True):
            auth = Path(td) / "auth.json"
            write_auth(auth, expires=int(time.time()) + 3600)
            config = ProviderConfig(
                type="chatgpt",
                base_url="https://chatgpt.com/backend-api/codex",
                model="gpt-5.6-terra",
                auth_file=str(auth),
                headers={
                    "Authorization": "Bearer attacker-controlled",
                    "ChatGPT-Account-ID": "wrong-account",
                },
            )
            manager = ChatGPTCredentialManager(config)
            credentials = manager.load()
            self.assertEqual(credentials.account_id, "acct_1234567890")
            self.assertEqual(credentials.email, "till@example.test")
            self.assertEqual(credentials.plan, "plus")
            self.assertTrue(manager.status().available)
            self.assertTrue(manager.status().secure)

            provider = ChatGPTProvider(config)
            provider.credentials = manager
            provider._active_credentials = credentials
            headers = provider._headers()
            self.assertTrue(headers["Authorization"].startswith("Bearer "))
            self.assertEqual(headers["ChatGPT-Account-ID"], "acct_1234567890")
            self.assertEqual(headers["Originator"], "borealis_coder")
            self.assertNotEqual(headers["Authorization"], "Bearer attacker-controlled")

            request = ProviderRequest(
                model="gpt-5.6-terra",
                system="system",
                messages=[Message(role=Role.USER, content="fix it")],
                tools=[
                    {
                        "name": "read_file",
                        "description": "Read",
                        "parameters": object_schema({"path": {"type": "string"}}),
                    }
                ],
                max_output_tokens=123,
                temperature=0.2,
                reasoning_effort="high",
                metadata={"session_id": "sess_abc"},
            )
            payload = provider._responses_payload(request, stream=True)
            self.assertEqual(payload["tool_choice"], "auto")
            self.assertEqual(payload["include"], ["reasoning.encrypted_content"])
            self.assertEqual(payload["prompt_cache_key"], "sess_abc")
            self.assertNotIn("max_output_tokens", payload)
            self.assertNotIn("temperature", payload)
            self.assertTrue(payload["stream"])

    async def test_refresh_is_atomic_and_rotates_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=True):
            auth = Path(td) / "auth.json"
            write_auth(auth, expires=int(time.time()) + 5)
            new_access = jwt(
                {
                    "exp": int(time.time()) + 7200,
                    "https://api.openai.com/auth": {
                        "chatgpt_account_id": "acct_new_1234",
                        "chatgpt_plan_type": "pro",
                    },
                }
            )
            calls: list[tuple[str, dict[str, object]]] = []

            async def transport(url: str, payload: dict[str, object]) -> dict[str, object]:
                calls.append((url, payload))
                return {
                    "access_token": new_access,
                    "refresh_token": "refresh-new",
                    "id_token": jwt(
                        {
                            "https://api.openai.com/auth": {
                                "chatgpt_account_id": "acct_new_1234",
                                "chatgpt_plan_type": "pro",
                            }
                        }
                    ),
                }

            config = ProviderConfig(
                type="chatgpt",
                auth_file=str(auth),
                refresh_before_expiry_seconds=300,
            )
            manager = ChatGPTCredentialManager(config, refresh_transport=transport)
            refreshed = await manager.ensure_valid()
            self.assertEqual(refreshed.access_token, new_access)
            self.assertEqual(refreshed.refresh_token, "refresh-new")
            self.assertEqual(refreshed.account_id, "acct_new_1234")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1]["grant_type"], "refresh_token")
            saved = json.loads(auth.read_text(encoding="utf-8"))
            self.assertEqual(saved["tokens"]["access_token"], new_access)
            self.assertEqual(saved["tokens"]["refresh_token"], "refresh-new")
            if os.name != "nt":
                self.assertEqual(auth.stat().st_mode & 0o777, 0o600)
            self.assertFalse(auth.with_suffix(".json.lock").exists())

    async def test_insecure_and_expired_unrefreshable_credentials_fail(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=True):
            auth = Path(td) / "auth.json"
            write_auth(auth, expires=int(time.time()) - 1)
            if os.name != "nt":
                os.chmod(auth, 0o644)
                with self.assertRaises(ConfigurationError):
                    ChatGPTCredentialManager(
                        ProviderConfig(type="chatgpt", auth_file=str(auth))
                    ).load()
                os.chmod(auth, 0o600)
            data = json.loads(auth.read_text(encoding="utf-8"))
            data["tokens"]["refresh_token"] = ""
            auth.write_text(json.dumps(data), encoding="utf-8")
            os.chmod(auth, 0o600)
            manager = ChatGPTCredentialManager(
                ProviderConfig(type="chatgpt", auth_file=str(auth))
            )
            with self.assertRaises(ProviderAuthenticationError):
                await manager.ensure_valid()

    async def test_environment_access_token_and_registry(self) -> None:
        access = jwt(
            {
                "exp": int(time.time()) + 3600,
                "https://api.openai.com/auth": {
                    "chatgpt_account_id": "acct_env_1234",
                    "chatgpt_plan_type": "team",
                },
            }
        )
        config = Config(
            agent=AgentConfig(provider="chatgpt", model="gpt-5.6-terra"),
            providers={
                "chatgpt": ProviderConfig(
                    type="chatgpt",
                    base_url="https://chatgpt.com/backend-api/codex",
                    model="gpt-5.6-terra",
                )
            },
        )
        with patch.dict(os.environ, {"CODEX_ACCESS_TOKEN": access}, clear=True):
            credentials = ChatGPTCredentialManager(config.providers["chatgpt"]).load()
            self.assertEqual(credentials.account_id, "acct_env_1234")
            name, model, provider = ProviderRegistry().create(config)
        self.assertEqual(name, "chatgpt")
        self.assertEqual(model, "gpt-5.6-terra")
        self.assertIsInstance(provider, ChatGPTProvider)

    async def test_encrypted_reasoning_state_is_preserved_without_raw_response(self) -> None:
        provider = ChatGPTProvider(ProviderConfig(type="chatgpt"))
        response = provider._parse_responses(
            {
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "encrypted_content": "opaque-ciphertext",
                        "summary": [{"type": "summary_text", "text": "private"}],
                    },
                    {"type": "message", "content": [{"type": "output_text", "text": "done"}]},
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
            retain_raw=True,
        )
        self.assertEqual(response.text, "done")
        self.assertEqual(
            response.raw,
            {
                "responses_state": [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "encrypted_content": "opaque-ciphertext",
                        "summary": [],
                    }
                ]
            },
        )
        assistant = Message(
            role=Role.ASSISTANT,
            content="done",
            metadata=response.raw or {},
        )
        values = provider._responses_input([assistant])
        self.assertEqual(values[0]["encrypted_content"], "opaque-ciphertext")
        self.assertNotIn("private", json.dumps(values))

    async def test_authentication_retry_only_before_stream_output(self) -> None:
        provider = ChatGPTProvider(ProviderConfig(type="chatgpt"))
        credentials = ChatGPTCredentials(
            access_token="access",
            account_id="acct_123",
        )
        refresh_calls: list[bool] = []

        class Manager:
            async def ensure_valid(self, *, force_refresh: bool = False) -> ChatGPTCredentials:
                refresh_calls.append(force_refresh)
                return credentials

            def load(self) -> ChatGPTCredentials:
                return credentials

        provider.credentials = Manager()  # type: ignore[assignment]
        attempts = 0

        async def parent_stream(self, request):  # type: ignore[no-untyped-def]
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ProviderAuthenticationError("expired")
            yield ProviderStreamEvent(
                type="completed",
                response=ModelResponse(text="done"),
            )

        request = ProviderRequest(model="gpt-5.6-terra", system="", messages=[])
        with patch.object(OpenAIProvider, "stream", parent_stream):
            events = [event async for event in provider.stream(request)]
        self.assertEqual(attempts, 2)
        self.assertEqual(refresh_calls, [False, True])
        response = events[-1].response
        assert response is not None
        self.assertEqual(response.text, "done")

        attempts = 0
        refresh_calls.clear()

        async def partial_parent_stream(self, request):  # type: ignore[no-untyped-def]
            nonlocal attempts
            attempts += 1
            yield ProviderStreamEvent(type="text_delta", text="partial")
            raise ProviderAuthenticationError("expired")

        with (
            patch.object(OpenAIProvider, "stream", partial_parent_stream),
            self.assertRaises(ProviderAuthenticationError),
        ):
            _ = [event async for event in provider.stream(request)]
        self.assertEqual(attempts, 1)
        self.assertEqual(refresh_calls, [False])

    async def test_custom_credential_endpoints_require_process_opt_in(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigurationError):
                ChatGPTProvider(
                    ProviderConfig(
                        type="chatgpt",
                        base_url="https://example.invalid/codex",
                        allow_custom_chatgpt_endpoints=True,
                    )
                )
            manager = ChatGPTCredentialManager(
                ProviderConfig(
                    type="chatgpt",
                    refresh_url="https://example.invalid/token",
                    allow_custom_chatgpt_endpoints=True,
                )
            )
            with self.assertRaises(ConfigurationError):
                _ = manager.refresh_url

        with patch.dict(
            os.environ,
            {"BOREALIS_ALLOW_CUSTOM_CHATGPT_ENDPOINTS": "1"},
            clear=True,
        ):
            provider = ChatGPTProvider(
                ProviderConfig(
                    type="chatgpt",
                    base_url="https://example.invalid/codex",
                    allow_custom_chatgpt_endpoints=True,
                )
            )
            self.assertEqual(provider.config.base_url, "https://example.invalid/codex")
            with self.assertRaises(ConfigurationError):
                ChatGPTProvider(
                    ProviderConfig(
                        type="chatgpt",
                        base_url="http://example.invalid/codex",
                        allow_custom_chatgpt_endpoints=True,
                    )
                )


class ChatGPTLoginAndCliTests(unittest.TestCase):
    def test_launch_codex_login_uses_isolated_home(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "auth-home"

            def run(argv, *, env, check):  # type: ignore[no-untyped-def]
                self.assertEqual(argv, ["codex", "login", "--device-auth"])
                self.assertFalse(check)
                self.assertEqual(env["CODEX_HOME"], str(home.resolve()))
                write_auth(home / "auth.json", expires=int(time.time()) + 3600)
                return SimpleNamespace(returncode=0)

            with patch("borealis_coder.auth.chatgpt.shutil.which", return_value="/bin/codex"), patch(
                "borealis_coder.auth.chatgpt.subprocess.run", side_effect=run
            ):
                auth_file = launch_codex_login(
                    codex_home=home, codex_command="codex", device_code=True
                )
            self.assertEqual(auth_file, (home / "auth.json").resolve())
            self.assertIn("cli_auth_credentials_store", (home / "config.toml").read_text())

    def test_cli_auth_status_login_and_logout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            auth = root / "auth.json"
            write_auth(auth, expires=int(time.time()) + 3600)
            env = {
                "BOREALIS_CHATGPT_AUTH_FILE": str(auth),
                "BOREALIS_DATA_DIR": str(root / "data"),
            }
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch.dict(os.environ, env, clear=True), redirect_stdout(stdout), redirect_stderr(
                stderr
            ):
                code = cli.main(["auth", "status", "--workspace", str(root), "--json"])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(stdout.getvalue())["available"])

            with patch("borealis_coder.cli.launch_codex_login", return_value=auth), redirect_stdout(
                stdout := io.StringIO()
            ):
                code = cli.main(
                    [
                        "auth",
                        "login",
                        "--codex-home",
                        str(root / "managed"),
                        "--device-code",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertIn("ChatGPT login ready", stdout.getvalue())

            with patch("borealis_coder.cli.logout_managed_chatgpt", return_value=auth), redirect_stdout(
                stdout := io.StringIO()
            ):
                code = cli.main(["auth", "logout", "--codex-home", str(root / "managed")])
            self.assertEqual(code, 0)
            self.assertIn("credentials removed", stdout.getvalue())

    def test_auto_provider_detects_existing_codex_login(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            auth = root / "auth.json"
            write_auth(auth, expires=int(time.time()) + 3600)
            with patch.dict(
                os.environ,
                {
                    "BOREALIS_CHATGPT_AUTH_FILE": str(auth),
                    "BOREALIS_DATA_DIR": str(root / "data"),
                },
                clear=True,
            ):
                config = load_config(root)
                name, provider = config.provider()
            self.assertEqual(name, "chatgpt")
            self.assertEqual(provider.type, "chatgpt")


if __name__ == "__main__":
    unittest.main()
