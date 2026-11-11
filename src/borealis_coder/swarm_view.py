"""Literal, bounded Aurora swarm state shared by live and serial views."""

from __future__ import annotations

import textwrap
import unicodedata
from dataclasses import dataclass

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.utils import get_cwidth

from .models import Event


def literal(value: object, limit: int = 4000) -> str:
    """Do not interpret untrusted escapes, line controls, or bidi overrides."""
    text = str(value)[:limit]
    return "".join(char if char == "\n" or not unicodedata.category(char).startswith("C")
                   else "�" for char in text).replace("\t", " ")


def clip(text: str, width: int) -> str:
    result = ""
    for char in text:
        if get_cwidth(result + char) > max(0, width):
            break
        result += char
    return result


@dataclass(slots=True)
class AgentNode:
    id: str
    parent: str
    run_id: str
    depth: int
    label: str
    capability: str
    state: str = "waiting"
    detail: str = ""
    collapsed: bool = False


class SwarmView:
    """An event-driven tree; no provider text is treated as terminal formatting."""

    def __init__(self) -> None:
        self.nodes: dict[str, AgentNode] = {}
        self.root_run = ""
        self.visible = False
        self.focused = False
        self.selected = ""
        self.detail_offset = 0
        self.limits: dict = {}
        self.usage: dict = {}
        self.requests = 0
        self.request_limit = 0
        self.omitted = 0

    def adopt(self, event: Event) -> bool:
        data = event.data
        if event.type == "swarm.started":
            self.nodes.clear()
            self.root_run = str(event.run_id)
            self.limits = data["limits"]
            self.request_limit = data["max_model_requests"]
            self.requests = 0
            self.usage = {}
            self.omitted = 0
            self.selected = "root"
            self.detail_offset = 0
            self.nodes["root"] = AgentNode("root", "", self.root_run, 0,
                                           literal(data.get("task", "Root"), 160).replace("\n", " "), "operator", "running")
            return True
        if event.type == "swarm.updated" and event.run_id == self.root_run:
            self.usage = data["usage"]
            self.requests = data["model_requests"]
            self.limits = data["limits"]
            return True
        if event.type.startswith("delegate."):
            if not str(data.get("delegated_run_id", "")).startswith(self.root_run + ".") or not self.root_run:
                return False
            identifier = data["delegation_id"]
            if event.type in {"delegate.started", "delegate.refused"}:
                if len(self.nodes) >= 256:
                    self.omitted += 1
                    return True
                if len(self.nodes) == 1:
                    self.visible = True
                self.nodes[identifier] = AgentNode(
                    identifier, data.get("parent_delegation_id") or "root",
                    data["delegated_run_id"], data["depth"], literal(data["task"], 160).replace("\n", " "),
                    data.get("capability", "not-admitted"), "refused" if event.type.endswith("refused") else "waiting",
                )
            node = self.nodes.get(identifier)
            if node is not None:
                states = {"delegate.completed": "done", "delegate.failed": "error",
                          "delegate.cancelled": "cancelled"}
                node.state = str(states.get(event.type) or data.get("state") or node.state)
                if "result" in data or "error" in data:
                    node.detail = literal(data.get("result") or data.get("error"))
                if data.get("evidence"):
                    node.detail += "\nEvidence: " + literal(data["evidence"], 1500)
            self.limits = data.get("limits", self.limits)
            self.requests = data.get("model_requests", self.requests)
            return True
        if event.type.startswith("tool."):
            node = next((item for item in self.nodes.values() if item.run_id == event.run_id), None)
            if node and node.id != "root" and node.state not in {"done", "error", "cancelled"}:
                node.state = "tool"
                node.detail = literal(data.get("tool", "tool"), 80)
                if event.type == "tool.completed":
                    node.detail += "\n" + literal(data.get("output", ""))
                return True
        if event.type == "run.completed" and event.run_id == self.root_run:
            result = data["result"]
            self.usage = result["usage"]
            node = self.nodes.get("root")
            if node:
                reason = result["stop_reason"]
                node.state = "done" if reason == "end_turn" else "cancelled" if reason == "cancelled" else "error"
                node.detail = literal(result.get("error") or result.get("text", ""))
            return True
        return False

    def rows(self) -> list[AgentNode]:
        children: dict[str, list[AgentNode]] = {}
        for node in self.nodes.values():
            children.setdefault(node.parent, []).append(node)
        result: list[AgentNode] = []

        def visit(parent: str) -> None:
            for node in children.get(parent, []):
                result.append(node)
                if not node.collapsed:
                    visit(node.id)
        visit("")
        return result

    def move(self, offset: int) -> None:
        rows = self.rows()
        if not rows:
            return
        index = next((i for i, item in enumerate(rows) if item.id == self.selected), 0)
        self.selected = rows[max(0, min(len(rows) - 1, index + offset))].id
        self.detail_offset = 0

    def collapse(self, collapsed: bool) -> None:
        if node := self.nodes.get(self.selected):
            node.collapsed = collapsed

    def render(self, width: int, height: int) -> StyleAndTextTuples:
        if not self.visible:
            return []
        height = max(1, height)
        rows = self.rows()
        if self.selected not in {item.id for item in rows}:
            self.selected = rows[0].id if rows else ""
        selected = self.nodes.get(self.selected)
        cost = self.usage.get("cost_usd", 0)
        status = self.usage.get("cost_status", "unknown")
        spend = f"${cost:.4f}" if status in {"known", "estimated"} else f"${cost:.4f} + unknown"
        limits = self.limits
        request_limit = "unlimited" if self.request_limit == 0 else str(self.request_limit)
        summary = (f"Agents {limits.get('admitted', 0)}/{limits.get('max_descendants', '?')} · "
                   f"provider {limits.get('active_requests', 0)}/{limits.get('max_concurrent', '?')} · "
                   f"requests {self.requests}/{request_limit} · {spend}")
        lines: list[tuple[str, str]] = [
            ("class:swarm.title", "╭─ SWARM  " + ("[focus] " if self.focused else "") + "F6 inspect · F7 hide"),
            ("class:swarm.muted", " " + summary if width >= 90 else
             f" Agents {limits.get('admitted', 0)}/{limits.get('max_descendants', '?')} · provider {limits.get('active_requests', 0)}/{limits.get('max_concurrent', '?')}"),
        ]
        if width < 90:
            lines.append(("class:swarm.muted", f" Root requests {self.requests}/{request_limit} · {spend}"))
        tree_height = max(1, min(len(rows), height - len(lines) - 4))
        selected_index = next((i for i, row in enumerate(rows) if row.id == self.selected), 0)
        start = max(0, selected_index - tree_height + 1)
        for node in rows[start:start + tree_height]:
            marker = "▸" if node.collapsed else "▾" if any(item.parent == node.id for item in rows) else "·"
            label = f"{'>' if node.id == self.selected else ' '} {'  ' * min(node.depth, 5)}{marker} {node.state} [{node.capability}] {node.label}"
            lines.append(("class:swarm.selected" if node.id == self.selected else "class:swarm.text", label))
        if selected:
            lines.append(("class:swarm.title", f" {selected.capability} · depth {selected.depth} · {selected.state} · {selected.id}"))
            details = [chunk for line in (selected.detail or "Waiting for evidence.").splitlines()
                       for chunk in (textwrap.wrap(line, max(1, width - 2)) or [""])]
            count = max(0, height - len(lines) - 1)
            offset = min(self.detail_offset, max(0, len(details) - count))
            self.detail_offset = offset
            lines.extend(("class:swarm.text", " " + line) for line in details[offset:offset + count])
        footer = "╰─ ↑↓ select · ←→ fold · PgUp/PgDn detail · Enter input" if self.focused else f"╰─ F6 inspect {len(rows)} entries · input stays editable"
        if self.omitted:
            footer += f" · {self.omitted} omitted"
        lines.append(("class:swarm.muted", footer))
        return [(style, clip(text, width) + ("\n" if index + 1 < min(len(lines), height) else ""))
                for index, (style, text) in enumerate(lines[:height])]
