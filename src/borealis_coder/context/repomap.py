"""Cheap local repository map using language-aware symbol extraction and ranking."""

from __future__ import annotations

import ast
import re
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from ..util import FileSignature, file_signature, read_bytes_up_to, truncate_text
from .git import quote_path, repository_status
from .ignore import IgnoreMatcher, repository_files


@dataclass(slots=True)
class FileSummary:
    path: str
    language: str
    symbols: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    size: int = 0
    score: float = 0.0

    def render(self) -> str:
        parts = [quote_path(self.path)]
        if self.symbols:
            parts.append("  symbols: " + ", ".join(self.symbols[:30]))
        if self.imports:
            parts.append("  imports: " + ", ".join(self.imports[:16]))
        return "\n".join(parts)


_EXT_LANG = {
    ".py": "Python",
    ".pyi": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".rs": "Rust",
    ".go": "Go",
    ".java": "Java",
    ".kt": "Kotlin",
    ".swift": "Swift",
    ".rb": "Ruby",
    ".php": "PHP",
    ".cs": "C#",
    ".c": "C",
    ".h": "C/C++",
    ".cpp": "C++",
    ".hpp": "C++",
    ".scala": "Scala",
    ".sh": "Shell",
    ".sql": "SQL",
}
_SYMBOL_PATTERNS = [
    re.compile(
        r"^\s*(?:export\s+)?(?:async\s+)?(?:def|function|fn|func)\s+([A-Za-z_$][\w$]*)",
        re.MULTILINE,
    ),
    re.compile(
        r"^\s*(?:export\s+)?(?:class|struct|enum|interface|trait|type)\s+([A-Za-z_$][\w$]*)",
        re.MULTILINE,
    ),
    re.compile(r"^\s*(?:pub\s+)?(?:const|static)\s+([A-Za-z_$][\w$]*)", re.MULTILINE),
]
_IMPORT_PATTERN = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w./@-]+)|use\s+([\w:]+)|require\(['\"]([^'\"]+))",
    re.MULTILINE,
)


class RepoMap:
    def __init__(
        self, root: Path, matcher: IgnoreMatcher, *, max_file_bytes: int = 2_000_000
    ) -> None:
        self.root = root.resolve()
        self.matcher = matcher
        self.max_file_bytes = max_file_bytes
        self._cache: dict[str, tuple[FileSignature, FileSummary]] = {}

    def build(
        self,
        *,
        query: str = "",
        max_chars: int = 28_000,
        rank_changed: bool = True,
    ) -> str:
        return self.render(
            self.snapshot(),
            query=query,
            max_chars=max_chars,
            rank_changed=rank_changed,
        )

    def snapshot(self) -> list[FileSummary]:
        """Collect current source summaries once for one or more map views."""

        summaries: list[FileSummary] = []
        active_cache: dict[str, tuple[FileSignature, FileSummary]] = {}
        for path in repository_files(self.root, self.matcher):
            if path.suffix.lower() not in _EXT_LANG and path.name not in {"Dockerfile", "Makefile"}:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size > self.max_file_bytes:
                continue
            key = str(path)
            signature = file_signature(stat)
            cached = self._cache.get(key)
            if cached is None or cached[0] != signature:
                summary = self._summarize(path, stat.st_size)
                cached = (signature, summary)
            else:
                summary = cached[1]
            active_cache[key] = cached
            summaries.append(summary)
        self._cache = active_cache
        return summaries

    def render(
        self,
        summaries: list[FileSummary],
        *,
        query: str = "",
        max_chars: int = 28_000,
        rank_changed: bool = True,
        changed_paths: set[str] | None = None,
    ) -> str:
        if max_chars <= 0:
            return ""
        terms = {item.lower() for item in re.findall(r"[A-Za-z_][\w.-]{2,}", query)}
        changed = (
            (set(self._git_changed()) if changed_paths is None else changed_paths)
            if rank_changed
            else set()
        )
        ranked: list[FileSummary] = []
        for summary in summaries:
            score = 0.2
            haystack = " ".join([summary.path, *summary.symbols, *summary.imports]).lower()
            for term in terms:
                if term in summary.path.lower():
                    score += 5
                score += 2 * sum(1 for symbol in summary.symbols if term in symbol.lower())
                if term in haystack:
                    score += 1
            if summary.path in changed:
                score += 3
            if summary.path.startswith(("src/", "lib/", "app/")):
                score += 0.5
            summary = FileSummary(
                summary.path,
                summary.language,
                summary.symbols,
                summary.imports,
                summary.size,
                score,
            )
            ranked.append(summary)
        ranked.sort(key=lambda item: (-item.score, item.path))
        ranking = f"query: {query}" if query else "stable path order"
        header = f"Repository map ({len(ranked)} source files; {ranking})"
        marker = "\n\n… repository map truncated …"
        if len(header) > max_chars:
            return truncate_text(header, max_chars, marker=marker)
        chunks = [header]
        used = len(header)
        for summary in ranked:
            rendered = "\n\n" + summary.render()
            if used + len(rendered) > max_chars:
                while len(chunks) > 1 and used + len(marker) > max_chars:
                    used -= len(chunks.pop())
                if used + len(marker) > max_chars:
                    return (header[:max(0, max_chars - len(marker))] + marker)[:max_chars]
                return "".join(chunks) + marker
            chunks.append(rendered)
            used += len(rendered)
        return "".join(chunks)

    def _summarize(self, path: Path, size: int) -> FileSummary:
        relative = path.relative_to(self.root).as_posix()
        language = _EXT_LANG.get(path.suffix.lower(), path.name)
        try:
            data = read_bytes_up_to(path, self.max_file_bytes + 1)
        except OSError:
            return FileSummary(relative, language, size=size)
        if len(data) > self.max_file_bytes:
            return FileSummary(relative, language, size=size)
        text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        if path.suffix.lower() in {".py", ".pyi"}:
            symbols, imports = _python_symbols(text, filename=relative)
        else:
            symbols = []
            for pattern in _SYMBOL_PATTERNS:
                symbols.extend(pattern.findall(text))
            imports = [
                next(item for item in match if item) for match in _IMPORT_PATTERN.findall(text)
            ]
        return FileSummary(relative, language, _dedupe(symbols)[:80], _dedupe(imports)[:40], size)

    def _git_changed(self) -> list[str]:
        _, changed = repository_status(self.root)
        return sorted(changed)


def _python_symbols(text: str, *, filename: str = "<unknown>") -> tuple[list[str], list[str]]:
    try:
        # Repository-map discovery is not a Python validation pass. Invalid
        # escape warnings in workspace files must not leak into the interactive
        # terminal while Borealis is assembling model context.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text, filename=filename)
    except SyntaxError:
        symbols = []
        for pattern in _SYMBOL_PATTERNS:
            symbols.extend(pattern.findall(text))
        return _dedupe(symbols), []
    symbols: list[str] = []
    imports: list[str] = []
    nodes: deque[ast.AST] = deque(tree.body)
    while nodes:
        node = nodes.popleft()
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            symbols.append(node.name)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
        # Definitions and imports occur in statement scopes, never expressions.
        # Keep handler/case containers so nested statements retain breadth-first order.
        nodes.extend(
            child for child in ast.iter_child_nodes(node)
            if isinstance(child, ast.stmt | ast.ExceptHandler | ast.match_case)
        )
    return _dedupe(symbols), _dedupe(imports)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in values if item))
