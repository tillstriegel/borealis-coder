# Provider adapters

## Normalized contract

The agent runtime does not consume provider-native response objects. Each adapter receives:

- normalized `Message` history
- strict tool definitions
- selected model and generation controls
- cancellation state
- optional provider metadata

It yields normalized events for:

- text deltas
- completed tool calls
- usage
- finish reason
- provider response metadata

This boundary allows sessions, tools, policy, ACP, and rendering to remain unchanged when providers differ.

## OpenAI Responses

Provider type: `openai`, API style: `responses`.

The adapter:

- sends system and conversation items to `/responses`
- emits flat strict function tools
- supports multiple tool calls in one turn
- parses response text and function-call argument fragments
- maps input, output, and cached token usage
- retries transient HTTP and service failures

The runtime does not require provider-hosted shell or patch tools because local tools preserve one policy and audit boundary.

## ChatGPT account through Codex OAuth

Provider type: `chatgpt`, API style: `responses`.

This route lets Borealis use an eligible ChatGPT account through the same
Codex authentication and Responses backend used by the open-source Codex CLI.
It is distinct from the public OpenAI API-key route:

- `borealis auth status` discovers secure, readable file-backed Codex credentials.
- `borealis auth login` launches the official Codex login flow into a
  Borealis-managed credential directory.
- `borealis auth login --device-code` supports headless environments.
- `CODEX_ACCESS_TOKEN` can supply an externally managed short-lived access token.
- access tokens are proactively refreshed when a refresh token is available.
- token rotation is persisted atomically under a cross-process credential lock.
- the account/workspace identifier is sent through `ChatGPT-Account-ID`.
- encrypted Responses reasoning state is preserved across local tool turns while
  plaintext reasoning summaries and complete native responses are discarded.
- one authentication retry is permitted only before any text or tool-call delta
  has been emitted, avoiding duplicate partial output.

Example:

```bash
borealis auth status
borealis auth login
borealis run --provider chatgpt --model gpt-5.6-terra "Fix and verify the failing tests"
```

```toml
[agent]
provider = "chatgpt"
model = "gpt-5.6-terra"

[providers.chatgpt]
type = "chatgpt"
base_url = "https://chatgpt.com/backend-api/codex"
model = "gpt-5.6-terra"
api_style = "responses"
refresh_before_expiry_seconds = 300
```

Borealis never reads ChatGPT browser cookies or browser-profile state. The
default credential endpoint and API endpoint are fixed to OpenAI/Codex hosts.
Custom endpoints require both
`allow_custom_chatgpt_endpoints = true` and the external process opt-in
`BOREALIS_ALLOW_CUSTOM_CHATGPT_ENDPOINTS=1`, because those endpoints receive
bearer or refresh tokens.

The provider reports token usage when supplied by the backend. It leaves
`cost_usd` at zero because a ChatGPT subscription route does not expose public
API-dollar pricing to the harness. Account-side model availability and usage
limits remain authoritative.

## OpenRouter

Provider type: `openrouter`. Default API style: `chat`. Optional API style: `responses`.

The adapter uses `https://openrouter.ai/api/v1` and `OPENROUTER_API_KEY` by default. It
adds first-class handling for OpenRouter rather than requiring a generic compatible route:

- optional `HTTP-Referer` / `X-Title` app attribution via `site_url` and `app_name`
- OpenRouter's normalized `reasoning = { effort = ... }` request shape
- OpenRouter-side model fallbacks through `model_fallbacks`
- provider routing controls through `provider_preferences`
- authoritative routed request cost when OpenRouter returns `usage.cost`
- reasoning-token accounting from completion usage details
- explicit `usage = { include = true }` so budgeting has the best available accounting
- `extra_body` as a forward-compatible escape hatch for newer OpenRouter extensions

Example:

```toml
[agent]
provider = "openrouter"
model = "anthropic/claude-sonnet-4.6"

[providers.openrouter]
type = "openrouter"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
api_style = "chat"
app_name = "Borealis Coder"
site_url = "https://example.com/borealis"
model_fallbacks = ["openai/gpt-5.4-mini", "google/gemini-3.6-pro"]

[providers.openrouter.provider_preferences]
allow_fallbacks = true
data_collection = "deny"
zdr = true
```

`model_fallbacks` contains backup models only; the primary remains `agent.model` (or the
provider's `model`). OpenRouter automatically performs provider-level failover for a model,
while this array adds model-level fallback. Routing controls such as `only`, `ignore`, `order`,
`sort`, `data_collection`, and `zdr` can be passed inside `provider_preferences` as supported by
your OpenRouter account and selected model.

Borealis also retains its own `agent.provider_fallbacks`, which operates one level higher and
can fall back from OpenRouter to a completely different Borealis provider route.

## OpenAI-compatible chat

Provider type: `openai_compatible`, API style: `chat`.

This route targets local or third-party servers exposing the Chat Completions shape. An API key is optional. Set `base_url` and `model` explicitly. Provider implementations vary; use `borealis doctor`, a small plan-mode task, and the acceptance workflow before granting write authority.

## Anthropic Messages

Provider type: `anthropic`, API style: `messages`.

The adapter:

- separates the system prompt from message history
- converts tool schemas to Anthropic tools
- maps tool-use and tool-result content blocks
- parses streamed text and structured tool calls
- normalizes input/output/cache usage where supplied
- marks the stable system-context block with an explicit Anthropic `cache_control` breakpoint
- can add a moving conversation breakpoint for multi-turn sessions

The stable block contains repository instructions, skills, and a deterministic repository map.
Request-ranked context, execution policy, and Git status follow it as an uncached suffix. This
keeps unrelated prompt changes from invalidating the reusable prefix.

## Gemini Interactions

Provider type: `gemini`, API style: `interactions`.

The adapter uses the Interactions surface and maps normalized messages, system instruction, flat function tools, tool calls/results, generation controls, and usage to Borealis events. Requests use stateless storage. Borealis therefore retains the model-generated interaction steps, including signed thought and function-call steps, and replays them exactly on the next turn. This continuation state is versioned and accepted only for the same provider and requested model.

Because provider APIs evolve, the adapter is covered by wire-format fixtures, but live calls should be part of deployment qualification.

## Mock provider

Provider type: `mock`.

The deterministic provider exists for:

- offline release smoke tests
- agent-loop regression tests
- tool lifecycle tests
- demos without credentials
- embedding tests that must not make network calls

It recognizes a small set of fixture prompts such as `OFFLINE_WRITE_DEMO`. It is not a general language model.

## Routing and fallback

`agent.provider` selects the primary route. `agent.provider_fallbacks` is an ordered list of additional configured routes.

Fallback occurs for provider-layer failures, not for a model’s ordinary end-turn judgment. A route change emits an event and preserves normalized conversation state. Tool calls are never replayed solely because a provider failed after the tool result was durably recorded.

## Pricing and budgets

Provider configurations can declare local prices per million input, cached-input, cache-write,
and output tokens. Borealis computes both actual cost and net prompt-cache savings from normalized
usage and enforces `agent.max_cost_usd`. When no explicit cache-write price is set, Borealis uses
the input price; the Anthropic adapter applies its 5-minute or 1-hour write multiplier.

Price fields default to zero because pricing changes and can depend on account, region, batch mode, or cache behavior. Production operators should set current prices explicitly and monitor the provider’s own billing controls.

The CLI reports prompt-cache hit rate, read/write tokens, net savings, exact-response-cache hits,
and avoided tokens. Exact final-text responses are cached locally for a short TTL. Safe
provider-selected continuation items are cached with the text so a cache hit remains resumable;
full provider responses are not stored. Tool-call, partial, failed, and cancelled responses are
never eligible.

## Adding a provider

Implement `Provider` and register a factory:

```python
from borealis_coder.agent import build_runner
from borealis_coder.providers import ProviderRegistry

registry = ProviderRegistry()
registry.register("custom", custom_factory)
runner = await build_runner(workspace, provider_registry=registry)
```

A provider implementation should:

1. Preserve tool call IDs.
2. Validate/parse JSON arguments without executing them.
3. Emit usage whenever available.
4. Classify transient, authentication, rate-limit, context-overflow, and terminal errors.
5. Honor cancellation and timeout.
6. Avoid persisting raw secrets or complete native responses by default.
7. Include deterministic fixture tests for request payload and response parsing.
