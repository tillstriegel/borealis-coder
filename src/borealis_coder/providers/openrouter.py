"""First-class OpenRouter adapter built on the normalized OpenAI wire layer."""

from __future__ import annotations

from typing import Any

from ..models import ProviderRequest, Usage
from .openai import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider with attribution, routing, fallback, and usage support.

    OpenRouter exposes OpenAI-compatible Chat Completions and Responses endpoints.
    Chat Completions is the default because it has the broadest cross-model support,
    while ``api_style = \"responses\"`` remains available for models that support it.
    """

    name = "openrouter"

    @property
    def api_style(self) -> str:
        return self.config.api_style or "chat"

    def _headers(self) -> dict[str, str]:
        headers = super()._headers()
        if self.config.site_url:
            headers.setdefault("HTTP-Referer", self.config.site_url)
        if self.config.app_name:
            headers.setdefault("X-Title", self.config.app_name)
        return headers

    def _chat_payload(self, request: ProviderRequest, *, stream: bool = False) -> dict[str, Any]:
        payload = super()._chat_payload(request, stream=stream)
        # OpenRouter normalizes reasoning controls across supported models via the
        # ``reasoning`` object rather than OpenAI's ``reasoning_effort`` field.
        payload.pop("reasoning_effort", None)
        if request.reasoning_effort:
            payload["reasoning"] = {"effort": request.reasoning_effort}
        return self._apply_extensions(payload, request)

    def _responses_payload(
        self, request: ProviderRequest, *, stream: bool = False
    ) -> dict[str, Any]:
        return self._apply_extensions(
            super()._responses_payload(request, stream=stream),
            request,
        )

    def _apply_extensions(
        self,
        payload: dict[str, Any],
        request: ProviderRequest,
    ) -> dict[str, Any]:
        # Explicit usage inclusion gives Borealis reliable accounting when a routed
        # upstream does not include usage by default.
        payload.setdefault("usage", {"include": True})
        session_id = str(request.metadata.get("session_id") or "").strip()
        if request.metadata.get("prompt_cache_enabled", True) and session_id:
            payload["session_id"] = session_id
        if self.config.model_fallbacks:
            payload["models"] = list(self.config.model_fallbacks)
        if self.config.provider_preferences:
            payload["provider"] = dict(self.config.provider_preferences)
        if self.config.extra_body:
            payload.update(self.config.extra_body)
        return payload

    def _usage_from_chat(self, usage_data: dict[str, Any]) -> Usage:
        usage = super()._usage_from_chat(usage_data)
        details = usage_data.get("completion_tokens_details") or {}
        usage.reasoning_tokens = int(
            details.get("reasoning_tokens", usage_data.get("reasoning_tokens", 0)) or 0
        )
        return self._apply_reported_cost(usage, usage_data)

    def _usage_from_responses(self, usage_data: dict[str, Any]) -> Usage:
        usage = super()._usage_from_responses(usage_data)
        return self._apply_reported_cost(usage, usage_data)

    @staticmethod
    def _apply_reported_cost(usage: Usage, usage_data: dict[str, Any]) -> Usage:
        # OpenRouter may report the authoritative routed request cost when usage
        # accounting is enabled. Prefer it over static local price configuration.
        cost = usage_data.get("cost")
        if isinstance(cost, (int, float)) and cost >= 0:
            usage.cost_usd = float(cost)
        return usage
