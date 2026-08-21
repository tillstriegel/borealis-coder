"""Conservative shell-command risk classification."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from enum import IntEnum


class CommandRisk(IntEnum):
    SAFE = 0
    WRITE = 1
    NETWORK = 2
    DESTRUCTIVE = 3
    FORBIDDEN = 4


@dataclass(frozen=True, slots=True)
class CommandAssessment:
    risk: CommandRisk
    reason: str
    executable: str = ""
    uses_shell_features: bool = False
    can_open_network: bool = False


_SHELL_META = re.compile(r"(?:[;&|`$<>]|\n|\r|\$\(|\*\*|\{[^}]*\})")
_NETWORK_TOOLS = {
    "curl", "wget", "ssh", "scp", "sftp", "ftp", "nc", "ncat", "telnet",
    "ping", "dig", "nslookup", "npm", "pnpm", "yarn", "pip", "pip3", "uv",
    "cargo", "go", "gem", "bundle", "composer", "apt", "apt-get", "brew",
    "docker", "podman", "kubectl", "helm", "gh",
}
_DESTRUCTIVE = {"rm", "rmdir", "shred", "mkfs", "fdisk", "dd", "truncate"}
_FORBIDDEN = {"shutdown", "reboot", "halt", "poweroff", "killall", "pkill"}
_READ_ONLY = {
    "cat", "head", "tail", "sed", "awk", "grep", "rg", "find", "fd", "ls",
    "pwd", "wc", "sort", "uniq", "cut", "tr", "file", "stat", "which",
    "whereis", "env", "printenv", "echo", "printf", "true", "false", "test",
    "python", "python3", "node", "ruby", "perl", "java", "git", "make",
    "pytest", "ruff", "mypy", "pyright", "npm", "pnpm", "yarn", "cargo", "go",
}

_NETWORK_INERT = {
    "cat", "head", "tail", "grep", "ls", "pwd", "wc", "uniq", "cut",
    "tr", "echo", "printf", "true", "false", "test", "stat",
}

_LOCAL_PACKAGE_COMMANDS = {
    "npm": {"run", "test"},
    "pnpm": {"run", "test", "exec"},
    "yarn": {"run", "test"},
    "cargo": {"fmt", "check", "test", "clippy", "metadata"},
}


def assess_command(command: str) -> CommandAssessment:
    command = command.strip()
    if not command:
        return CommandAssessment(CommandRisk.FORBIDDEN, "empty command")
    uses_shell = bool(_SHELL_META.search(command))
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as error:
        return CommandAssessment(
            CommandRisk.FORBIDDEN,
            f"invalid shell syntax: {error}",
            uses_shell_features=True,
            can_open_network=True,
        )
    if not tokens:
        return CommandAssessment(CommandRisk.FORBIDDEN, "empty command")
    executable = tokens[0].rsplit("/", 1)[-1].lower()
    lowered = [item.lower() for item in tokens]
    # Native processes have the caller's host network. Only commands with no
    # subprocess/plugin hooks are safe to run when that network cannot be
    # isolated. Everything else needs an OS sandbox or safety.network=true.
    can_open_network = uses_shell or executable not in _NETWORK_INERT

    if executable in _FORBIDDEN:
        return CommandAssessment(CommandRisk.FORBIDDEN, f"system-control command: {executable}", executable, uses_shell, can_open_network)
    if executable == "sudo" or any(item == "sudo" for item in lowered):
        return CommandAssessment(CommandRisk.FORBIDDEN, "privilege escalation is not allowed", executable, uses_shell, can_open_network)
    if executable in _DESTRUCTIVE:
        return CommandAssessment(CommandRisk.DESTRUCTIVE, f"destructive filesystem command: {executable}", executable, uses_shell, can_open_network)
    if executable == "git":
        subcommand = next((item for item in lowered[1:] if not item.startswith("-")), "")
        if subcommand in {"push", "fetch", "pull", "clone", "ls-remote", "submodule"}:
            return CommandAssessment(CommandRisk.NETWORK, f"git network operation: {subcommand}", executable, uses_shell, True)
        if subcommand in {"reset", "clean", "checkout", "restore", "rebase", "merge", "commit", "tag"}:
            return CommandAssessment(CommandRisk.WRITE, f"git mutation: {subcommand}", executable, uses_shell, can_open_network)
        return CommandAssessment(CommandRisk.SAFE, f"git read operation: {subcommand or 'status'}", executable, uses_shell, can_open_network)
    if executable in _LOCAL_PACKAGE_COMMANDS:
        subcommand = next((item for item in lowered[1:] if not item.startswith("-")), "")
        if subcommand in _LOCAL_PACKAGE_COMMANDS[executable]:
            if executable == "cargo" and subcommand in {"check", "test", "clippy", "metadata"}:
                if "--offline" not in lowered:
                    return CommandAssessment(
                        CommandRisk.NETWORK,
                        f"cargo {subcommand} may fetch dependencies without --offline",
                        executable,
                        uses_shell,
                        True,
                    )
            return CommandAssessment(
                CommandRisk.SAFE,
                f"recognized local verification command: {executable} {subcommand}",
                executable,
                uses_shell,
                can_open_network,
            )
    if executable in _NETWORK_TOOLS:
        # Package managers are network-capable even when some invocations are local.
        local_only = any(item in {"--offline", "--frozen", "--locked"} for item in lowered)
        if not local_only:
            return CommandAssessment(CommandRisk.NETWORK, f"network-capable command: {executable}", executable, uses_shell, True)
    if executable in {"chmod", "chown", "mv", "cp", "mkdir", "touch", "ln", "install"}:
        return CommandAssessment(CommandRisk.WRITE, f"filesystem mutation: {executable}", executable, uses_shell, can_open_network)
    if executable in {"bash", "sh", "zsh", "fish", "powershell", "pwsh", "cmd"}:
        return CommandAssessment(CommandRisk.WRITE, "nested shell execution", executable, True, True)
    if uses_shell:
        return CommandAssessment(CommandRisk.WRITE, "shell operators may compose or redirect commands", executable, True, True)
    if executable in _READ_ONLY:
        return CommandAssessment(CommandRisk.SAFE, f"recognized development command: {executable}", executable, uses_shell, can_open_network)
    return CommandAssessment(CommandRisk.WRITE, f"unrecognized executable: {executable}", executable, uses_shell, True)
