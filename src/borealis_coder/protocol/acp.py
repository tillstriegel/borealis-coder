"""ACP v2 stdio agent server backed by the same durable Borealis runtime."""

from __future__ import annotations

import asyncio
import base64
import binascii
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .. import __version__
from ..agent import AgentRunner, build_runner
from ..config import MCPServerConfig, load_config
from ..errors import ProtocolError, SessionError
from ..models import Event
from ..safety.approvals import ApprovalRequest
from ..sessions import SessionStore
from ..util import new_id
from .jsonrpc import JsonRpcConnection

ACP_PROTOCOL_VERSION = 2


class ACPServer:
    def __init__(self) -> None:
        self.connection = JsonRpcConnection(self.handle)
        self.runners: dict[str, AgentRunner] = {}
        self.tasks: dict[str, asyncio.Task[Any]] = {}
        self.session_locations: dict[str, Path] = {}
        self.initialized = False
        self.client_capabilities: dict[str, Any] = {}

    async def serve(self) -> None:
        await self.connection.serve_stdio()
        await self.close()

    async def close(self) -> None:
        for session_id, task in list(self.tasks.items()):
            if not task.done():
                runner = self.runners.get(session_id)
                if runner:
                    runner.cancel(session_id)
        await asyncio.gather(
            *(runner.close() for runner in set(self.runners.values())), return_exceptions=True
        )
        self.runners.clear()

    async def handle(self, method: str, params: dict[str, Any]) -> Any:
        if method == "initialize":
            return self._initialize(params)
        if not self.initialized:
            raise ProtocolError("initialize must be called first")
        handlers = {
            "session/new": self._session_new,
            "session/list": self._session_list,
            "session/resume": self._session_resume,
            "session/close": self._session_close,
            "session/delete": self._session_delete,
            "session/prompt": self._session_prompt,
            "session/cancel": self._session_cancel,
        }
        handler = handlers.get(method)
        if handler is None:
            raise ProtocolError(f"Method not found: {method}")
        return await handler(params)

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            requested = int(params.get("protocolVersion", 0))
        except (TypeError, ValueError) as error:
            raise ProtocolError("protocolVersion must be an integer") from error
        if requested < 1:
            raise ProtocolError("protocolVersion must be positive")
        self.client_capabilities = dict(params.get("capabilities") or {})
        self.initialized = True
        return {
            "protocolVersion": ACP_PROTOCOL_VERSION,
            "capabilities": {
                "session": {
                    "prompt": {"embeddedContext": {}},
                    "mcp": {"stdio": {}, "http": {}},
                    "delete": {},
                    "additionalDirectories": {},
                }
            },
            "info": {
                "name": "borealis-coder",
                "title": "Borealis Coder",
                "version": __version__,
            },
            "authMethods": [],
        }

    async def _session_new(self, params: dict[str, Any]) -> dict[str, Any]:
        cwd, additional = _roots(params)
        config = load_config(cwd)
        _merge_acp_mcp(config, params.get("mcpServers") or [])
        runner = await build_runner(
            cwd,
            config=config,
            interactive=True,
            approval_callback=lambda request: self._request_permission(None, request),
            additional_roots=additional,
        )
        route = runner.providers[0]
        session = await asyncio.to_thread(
            runner.sessions.create_session,
            workspace=cwd,
            provider=route.name,
            model=route.model,
            title="New coding session",
            metadata={"additional_directories": [str(item) for item in additional], "acp": True},
        )
        runner.tool_context.approvals.callback = lambda request: self._request_permission(
            session.id, request
        )
        self.runners[session.id] = runner
        self.session_locations[session.id] = runner.sessions.path
        self._subscribe(session.id, runner)
        return {"sessionId": session.id}

    async def _session_resume(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = _required_str(params, "sessionId")
        cwd, additional = _roots(params)
        config = load_config(cwd)
        _merge_acp_mcp(config, params.get("mcpServers") or [])
        runner = await build_runner(
            cwd,
            config=config,
            interactive=True,
            approval_callback=lambda request: self._request_permission(session_id, request),
            additional_roots=additional,
        )
        session = await asyncio.to_thread(runner.sessions.get_session, session_id)
        if Path(session.workspace).resolve() != cwd:
            await runner.close()
            raise ProtocolError("Session cwd does not match the stored workspace")
        old = self.runners.pop(session_id, None)
        if old:
            await old.close()
        self.runners[session_id] = runner
        self.session_locations[session_id] = runner.sessions.path
        self._subscribe(session_id, runner)
        replay = params.get("replayFrom") or {}
        if isinstance(replay, dict) and replay.get("type") == "start":
            await self._replay(session_id, runner)
        return {}

    async def _session_list(self, params: dict[str, Any]) -> dict[str, Any]:
        cwd_value = params.get("cwd")
        cwd = None
        if cwd_value:
            raw_cwd = Path(str(cwd_value)).expanduser()
            if not raw_cwd.is_absolute():
                raise ProtocolError("session/list cwd must be an absolute directory")
            cwd = raw_cwd.resolve()
            if not cwd.is_dir():
                raise ProtocolError("session/list cwd must be an existing directory")
        # Session databases are configured globally by default. Use an active runner
        # when possible; otherwise load config for cwd/current directory.
        runner = next(iter(self.runners.values()), None)
        temporary_store: SessionStore | None = None
        if runner is not None:
            store = runner.sessions
        else:
            base = cwd or Path.cwd().resolve()
            temporary_store = SessionStore(load_config(base).database_path)
            store = temporary_store
        try:
            cursor = _decode_cursor(params.get("cursor"))
            sessions = await asyncio.to_thread(
                store.list_sessions,
                workspace=cwd,
                limit=101 + cursor,
            )
            page = sessions[cursor : cursor + 100]
            for item in page:
                self.session_locations[item.id] = store.path
            result: dict[str, Any] = {
                "sessions": [
                    {
                        "sessionId": item.id,
                        "cwd": item.workspace,
                        "title": item.title,
                        "updatedAt": item.updated_at,
                        "additionalDirectories": item.metadata.get("additional_directories", []),
                        "_meta": {
                            "provider": item.provider,
                            "model": item.model,
                            "status": item.status,
                        },
                    }
                    for item in page
                ]
            }
            if len(sessions) > cursor + 100:
                result["nextCursor"] = _encode_cursor(cursor + 100)
            return result
        finally:
            if temporary_store is not None:
                await asyncio.to_thread(temporary_store.close)

    async def _session_close(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = _required_str(params, "sessionId")
        runner = self.runners.pop(session_id, None)
        task = self.tasks.pop(session_id, None)
        if runner:
            runner.cancel(session_id)
        if task and not task.done():
            try:
                await asyncio.wait_for(task, timeout=5)
            except TimeoutError:
                task.cancel()
        if runner:
            await runner.close()
        return {}

    async def _session_delete(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = _required_str(params, "sessionId")
        runner = self.runners.get(session_id)
        database_path = runner.sessions.path if runner else self.session_locations.get(session_id)
        if runner:
            await self._session_close({"sessionId": session_id})
        if database_path is None:
            database_path = load_config(Path.cwd().resolve()).database_path
        store = SessionStore(database_path)
        try:
            await asyncio.to_thread(store.delete_session, session_id)
        except SessionError as error:
            raise ProtocolError(str(error)) from error
        finally:
            await asyncio.to_thread(store.close)
        self.session_locations.pop(session_id, None)
        return {}

    async def _session_prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = _required_str(params, "sessionId")
        runner = self.runners.get(session_id)
        if runner is None:
            raise ProtocolError("Session is not active; call session/resume first")
        prompt = _prompt_text(params.get("prompt"))
        message_id = new_id("msg")
        if runner.accepts_steering(session_id):
            runner.steer(
                session_id,
                prompt,
                message_id=message_id,
                metadata={"acp": True},
            )
            await self._update(
                session_id,
                {
                    "sessionUpdate": "user_message",
                    "messageId": message_id,
                    "content": [{"type": "text", "text": prompt}],
                },
            )
            return {}
        await self._update(
            session_id,
            {
                "sessionUpdate": "user_message",
                "messageId": message_id,
                "content": [{"type": "text", "text": prompt}],
            },
        )
        await self._update(session_id, {"sessionUpdate": "state_update", "state": "running"})
        task = asyncio.create_task(
            self._run_prompt(runner, session_id, prompt, message_id),
            name=f"acp:{session_id}",
        )
        self.tasks[session_id] = task
        return {}

    async def _session_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = _required_str(params, "sessionId")
        runner = self.runners.get(session_id)
        if runner:
            runner.cancel(session_id)
        return {}

    async def _run_prompt(
        self,
        runner: AgentRunner,
        session_id: str,
        prompt: str,
        message_id: str,
    ) -> None:
        await asyncio.sleep(0)
        try:
            await runner.run(
                prompt,
                session_id=session_id,
                user_message_id=message_id,
                user_metadata={"acp": True},
                wait_for_active_run=True,
            )
        finally:
            if self.tasks.get(session_id) is asyncio.current_task():
                self.tasks.pop(session_id, None)

    def _subscribe(self, session_id: str, runner: AgentRunner) -> None:
        async def handler(event: Event) -> None:
            if event.session_id != session_id:
                return
            await self._event_update(session_id, runner, event)

        runner.events.subscribe(handler)

    async def _event_update(self, session_id: str, runner: AgentRunner, event: Event) -> None:
        data = event.data
        if event.type == "model.reasoning_delta" and data.get("text"):
            await self._update(
                session_id,
                {
                    "sessionUpdate": "agent_thought_chunk",
                    "messageId": data.get("message_id") or new_id("msg"),
                    "content": {"type": "text", "text": data["text"]},
                },
            )
        elif event.type == "model.text_delta" and data.get("text"):
            await self._update(
                session_id,
                {
                    "sessionUpdate": "agent_message_chunk",
                    "messageId": data.get("message_id") or new_id("msg"),
                    "content": {"type": "text", "text": data["text"]},
                },
            )
        elif event.type == "model.completed":
            if data.get("text"):
                await self._update(
                    session_id,
                    {
                        "sessionUpdate": "agent_message",
                        "messageId": data.get("message_id") or new_id("msg"),
                        "content": [{"type": "text", "text": data["text"]}],
                    },
                )
            for call in data.get("tool_calls", []) or []:
                await self._update(
                    session_id,
                    {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": call["id"],
                        "title": call["name"],
                        "kind": _tool_kind(call["name"]),
                        "status": "pending",
                        "rawInput": call.get("arguments") or {},
                    },
                )
            usage = await asyncio.to_thread(runner.sessions.usage, session_id)
            await self._update(
                session_id,
                {
                    "sessionUpdate": "usage_update",
                    "used": usage.total_tokens,
                    "size": runner.config.agent.max_input_tokens,
                    "cost": {"amount": usage.cost_usd, "currency": "USD"},
                },
            )
        elif event.type == "tool.started":
            await self._update(
                session_id,
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": data["tool_call_id"],
                    "title": data["tool"],
                    "kind": _tool_kind(data["tool"]),
                    "status": "in_progress",
                    "rawInput": data.get("arguments") or {},
                },
            )
        elif event.type == "tool.completed":
            await self._update(
                session_id,
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": data["tool_call_id"],
                    "title": data["tool"],
                    "kind": _tool_kind(data["tool"]),
                    "status": "failed" if data.get("is_error") else "completed",
                    "content": [
                        {
                            "type": "content",
                            "content": {"type": "text", "text": data.get("output") or ""},
                        }
                    ],
                },
            )
        elif event.type == "plan.updated":
            await self._update(
                session_id,
                {
                    "sessionUpdate": "plan_update",
                    "plan": {
                        "type": "items",
                        "planId": event.run_id or new_id("plan"),
                        "entries": [
                            {
                                "content": item["content"],
                                "status": item["status"],
                                "priority": "medium",
                            }
                            for item in data.get("items", [])
                        ],
                    },
                },
            )
        elif event.type == "run.completed":
            result = data.get("result") or {}
            if result.get("incomplete") and result.get("error"):
                await self._update(
                    session_id,
                    {
                        "sessionUpdate": "agent_message",
                        "messageId": new_id("msg"),
                        "content": [{"type": "text", "text": result["error"]}],
                    },
                )
            await self._update(
                session_id,
                {
                    "sessionUpdate": "state_update",
                    "state": "idle",
                    "stopReason": _acp_stop_reason(result.get("stop_reason")),
                },
            )

    async def _request_permission(self, session_id: str | None, request: ApprovalRequest) -> str:
        if not session_id:
            return "no"
        await self._update(
            session_id,
            {
                "sessionUpdate": "state_update",
                "state": "requires_action",
            },
        )
        try:
            result = await self.connection.request(
                "session/request_permission",
                {
                    "sessionId": session_id,
                    "toolCall": {
                        "toolCallId": new_id("permission"),
                        "title": request.description,
                        "kind": "other",
                        "status": "pending",
                        "rawInput": request.arguments_preview,
                    },
                    "options": [
                        {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
                        {
                            "optionId": "allow_always",
                            "name": "Allow for session",
                            "kind": "allow_always",
                        },
                        {"optionId": "reject_once", "name": "Reject", "kind": "reject_once"},
                    ],
                },
                timeout=3600,
            )
        finally:
            await self._update(
                session_id,
                {
                    "sessionUpdate": "state_update",
                    "state": "running",
                },
            )
        option = result.get("optionId") if isinstance(result, dict) else None
        return str(option or "no")

    async def _update(self, session_id: str, update: dict[str, Any]) -> None:
        await self.connection.notify("session/update", {"sessionId": session_id, "update": update})

    async def _replay(self, session_id: str, runner: AgentRunner) -> None:
        for message in await asyncio.to_thread(runner.sessions.messages, session_id):
            if message.role.value == "user":
                kind = "user_message"
            elif message.role.value == "assistant":
                kind = "agent_message"
            else:
                continue
            await self._update(
                session_id,
                {
                    "sessionUpdate": kind,
                    "messageId": message.id,
                    "content": [{"type": "text", "text": message.content}],
                },
            )


def _roots(params: dict[str, Any]) -> tuple[Path, list[Path]]:
    raw_cwd = Path(_required_str(params, "cwd")).expanduser()
    if not raw_cwd.is_absolute():
        raise ProtocolError("cwd must be an existing absolute directory")
    cwd = raw_cwd.resolve()
    if not cwd.is_dir():
        raise ProtocolError("cwd must be an existing absolute directory")
    additional: list[Path] = []
    for value in params.get("additionalDirectories", []) or []:
        raw_path = Path(str(value)).expanduser()
        if not raw_path.is_absolute():
            raise ProtocolError(f"additionalDirectories entry is invalid: {value}")
        path = raw_path.resolve()
        if not path.is_dir():
            raise ProtocolError(f"additionalDirectories entry is invalid: {value}")
        additional.append(path)
    return cwd, additional


def _merge_acp_mcp(config, values: list[dict[str, Any]]) -> None:  # type: ignore[no-untyped-def]
    for index, value in enumerate(values):
        name = str(value.get("name") or f"acp_{index}")
        transport = value.get("type")
        if transport == "stdio":
            command = str(value.get("command") or "")
            if not command or not Path(command).is_absolute():
                raise ProtocolError(f"MCP stdio command must be absolute for server {name!r}")
            env = {
                str(item.get("name")): str(item.get("value"))
                for item in value.get("env", [])
                if isinstance(item, dict) and item.get("name")
            }
            config.mcp_servers[name] = MCPServerConfig(
                type="stdio",
                command=command,
                args=[str(item) for item in value.get("args", [])],
                env=env,
            )
        elif transport == "http":
            url = str(value.get("url") or "")
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ProtocolError(f"MCP HTTP URL is invalid for server {name!r}")
            config.mcp_servers[name] = MCPServerConfig(
                type="http", url=url, headers=dict(value.get("headers") or {})
            )
        else:
            raise ProtocolError(f"Unsupported MCP transport for server {name!r}: {transport!r}")


def _prompt_text(value: Any) -> str:
    if not isinstance(value, list):
        raise ProtocolError("prompt must be a content-block array")
    chunks: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            chunks.append(str(item.get("text") or ""))
        elif item.get("type") == "resource":
            resource = item.get("resource") or {}
            chunks.append(
                f"\n<embedded_resource uri={resource.get('uri')!r}>\n{resource.get('text') or ''}\n</embedded_resource>"
            )
        elif item.get("type") in {"resource_link", "resourceLink"}:
            chunks.append(
                f"\nResource link: {item.get('uri') or item.get('resource', {}).get('uri')}"
            )
    prompt = "\n".join(chunks).strip()
    if not prompt:
        raise ProtocolError("prompt contains no supported content")
    return prompt


def _required_str(params: dict[str, Any], name: str) -> str:
    value = params.get(name)
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{name} is required")
    return value


def _tool_kind(name: str) -> str:
    if name in {"read_file", "grep", "glob_files", "repo_map", "list_directory"}:
        return "read"
    if name in {"write_file", "replace_in_file", "apply_patch", "delete_file"}:
        return "edit"
    if name == "shell" or name == "verify":
        return "execute"
    return "other"


def _acp_stop_reason(value: str | None) -> str:
    return {
        "end_turn": "end_turn",
        "cancelled": "cancelled",
        "max_turns": "max_turns",
        "budget": "max_turns",
        "error": "refusal",
        "stuck": "refusal",
    }.get(value or "", "end_turn")


def _encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(str(offset).encode()).decode()


def _decode_cursor(value: Any) -> int:
    if not value:
        return 0
    try:
        return max(0, int(base64.urlsafe_b64decode(str(value)).decode()))
    except (ValueError, TypeError, binascii.Error, UnicodeDecodeError):
        raise ProtocolError("Invalid session/list cursor") from None
