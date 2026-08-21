"""Repository discovery, instructions, skills, maps, and prompt assembly."""

from .builder import BASE_SYSTEM_PROMPT, ContextBuilder, PromptContext
from .ignore import IgnoreMatcher, repository_files
from .instructions import InstructionDocument, InstructionLoader
from .repomap import FileSummary, RepoMap
from .skills import Skill, SkillCatalog

__all__ = [
    "BASE_SYSTEM_PROMPT",
    "ContextBuilder",
    "FileSummary",
    "IgnoreMatcher",
    "InstructionDocument",
    "InstructionLoader",
    "PromptContext",
    "RepoMap",
    "Skill",
    "SkillCatalog",
    "repository_files",
]
