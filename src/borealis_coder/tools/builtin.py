"""Construction of the audited built-in tool palette."""

from __future__ import annotations

from .base import ToolRegistry
from .context import ReadInstructionsTool, ReadSkillTool, RepoMapTool
from .delegate import DelegateTaskTool
from .fetch import FetchUrlTool
from .filesystem import (
    DeleteFileTool, GlobFilesTool, ListDirectoryTool, MakeDirectoryTool,
    ReadFileTool, ReplaceInFileTool, WriteFileTool,
)
from .git import GitCommitTool, GitDiffTool, GitLogTool, GitPushTool, GitStatusTool
from .patch import ApplyPatchTool
from .search import GrepTool
from .shell import ShellTool
from .todo import UpdatePlanTool
from .verification import VerifyTool


def build_builtin_registry() -> ToolRegistry:
    return ToolRegistry([
        ReadFileTool(), ListDirectoryTool(), GlobFilesTool(), GrepTool(), RepoMapTool(),
        ReadInstructionsTool(), ReadSkillTool(), GitStatusTool(), GitDiffTool(), GitLogTool(),
        WriteFileTool(), ReplaceInFileTool(), ApplyPatchTool(), MakeDirectoryTool(), DeleteFileTool(),
        ShellTool(), VerifyTool(), FetchUrlTool(), UpdatePlanTool(), DelegateTaskTool(), GitCommitTool(), GitPushTool(),
    ])
