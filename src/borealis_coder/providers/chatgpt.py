"""ChatGPT-plan provider backed by Codex OAuth credentials and Responses."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ..auth import (
    ChatGPTCredentialManager,
    ChatGPTCredentials,
    validate_chatgpt_api_base_url,
)
from ..config import ProviderConfig
from ..errors import ProviderAuthenticationError, ProviderError
from ..models import ModelResponse, ProviderRequest
from .base import ProviderStreamEvent
from .openai import OpenAIProvider


class ChatGPTProvider(OpenAIProvider):
    """Use a ChatGPT subscription through the same Codex backend as Codex CLI.

    This adapter is intentionally Responses-only. Authentication is prepared before
    transport begins and a single transparent refresh/retry is allowed only when no
    stream content has been emitted, preventing duplicated partial output.
    """

    name = "chatgpt"

    def __init__(self, config: ProviderConfig, api_key: str = "") -> None:
        config.base_url = validate_chatgpt_api_base_url(config.base_url, config)
        super().__init__(config, api_key)
        self.credentials = ChatGPTCredentialManager(config)
        self._active_credentials: ChatGPTCredentials | None = None

    @property
    def api_style(self) -> str:
        return "responses"

    async def complete(self, request: ProviderRequest) -> ModelResponse:
        completed: ModelResponse | None = None
        async for event in self.stream(request):
            if event.type == "completed":
                completed = event.response
        if completed is None:
            raise ProviderError("ChatGPT/Codex stream ended without a completed response")
        return completed

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderStreamEvent]:
        for auth_attempt in range(2):
            self._active_credentials = await self.credentials.ensure_valid(
                force_refresh=auth_attempt > 0
            )
            emitted = False
            try:
                async for event in super().stream(request):
                    if event.type in {"text_delta", "tool_call_delta"}:
                        emitted = True
                    yield event
                return
            except ProviderAuthenticationError:
                if emitted or auth_attempt > 0:
                    raise
                # A 401/403 can arrive before the JWT expiry timestamp. Refresh once
                # and replay only because the stream emitted no user-visible content.
                continue
        raise ProviderAuthenticationError(
            "ChatGPT authentication failed after refreshing credentials. Run `borealis auth login`."
        )

    def _headers(self) -> dict[str, str]:
        credentials = self._active_credentials or self.credentials.load()
        headers = dict(self.config.headers)
        # Credential-bearing headers are authoritative. A project-local config
        # must not replace them with stale or attacker-controlled values.
        headers["Authorization"] = f"Bearer {credentials.access_token}"
        headers.setdefault("Originator", "borealis_coder")
        if credentials.account_id:
            headers["ChatGPT-Account-ID"] = credentials.account_id
        if credentials.fedramp:
            headers["X-OpenAI-Fedramp"] = "true"
        return headers

    def _responses_payload(
        self, request: ProviderRequest, *, stream: bool = False
    ) -> dict[str, Any]:
        payload = super()._responses_payload(request, stream=stream)
        # Match the Codex backend contract rather than the public API's optional
        # generation controls. ChatGPT plans enforce their own usage/output limits.
        payload.pop("max_output_tokens", None)
        payload.pop("temperature", None)
        payload["tool_choice"] = "auto"
        payload["include"] = ["reasoning.encrypted_content"]
        session_id = str(request.metadata.get("session_id") or "").strip()
        if (
            request.metadata.get("prompt_cache_enabled", True)
            and "prompt_cache_key" not in payload
            and session_id
        ):
            payload["prompt_cache_key"] = session_id
        if session_id:
            payload["client_metadata"] = {
                "borealis_session_id": session_id,
                "originator": "borealis_coder",
            }
        return payload

    def _parse_responses(self, data: dict[str, Any], *, retain_raw: bool) -> ModelResponse:
        response = super()._parse_responses(data, retain_raw=False)
        return response
