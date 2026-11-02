# Interactive CLI

## Purpose

The primary local Borealis workflow is a persistent, Codex-style terminal shell:

```bash
cd /path/to/repository
borealis
```

The shell keeps one `AgentRunner` alive, stores every turn in SQLite, streams
model text and tool activity, and returns to an input prompt after completion.
Exiting the process does not discard the conversation.

## Starting and resuming

```bash
# New conversation
borealis

# Send the first task immediately, then remain in the shell
borealis "Inspect the failing tests and fix the root cause"

# Resume the most recent session for this workspace
borealis --continue
borealis resume --last

# Resume a specific session
borealis --resume SESSION_ID
borealis resume SESSION_ID

# Resume and send an initial follow-up
borealis resume SESSION_ID "Now implement the plan"
```

Provider and safety flags work before the shell starts:

```bash
borealis \
  --provider chatgpt \
  --model gpt-5.6-terra \
  --mode workspace-write \
  --approval on-risk
```

## Output behavior

Interactive sessions use the Borealis **Aurora Shell**, a responsive inline TUI
that keeps normal terminal scrollback intact. Its mint, cyan, and violet palette,
framed launch summary, two-line session prompt, assistant blocks, activity rail,
turn receipts, and command panels form one consistent interface. The layout
adapts to narrower terminals and truncates long paths or labels in the middle so
the useful endpoints remain visible.

Color is enabled only for capable TTYs. `NO_COLOR` disables ANSI styling,
`CLICOLOR_FORCE=1` forces it when the receiving terminal understands ANSI, and
redirected output retains the same readable structure without escape sequences.
The colored input prompt uses readline-safe control markers, so wrapping and
cursor movement remain correct.

Assistant text is streamed by default. Every model exchange is rendered
independently, so a planning message before a tool call cannot suppress the final
answer after the tool result. Borealis reconciles the completed response with its
stream deltas and terminates the line before rendering tool state, the turn
footer, or the next prompt.

Interactive mode also reports live execution phases. It announces workspace
context preparation immediately, identifies each provider/model exchange, shows
when a tool call is being assembled, reports provider retries, renders policy,
tool, plan, fallback, and verification state, and closes every turn with usage
and change metadata. While a provider, tool, or verification step is pending, a
live Aurora pulse updates with the current phase and elapsed time. Redirected or
non-color output uses a persistent heartbeat every ten seconds instead.

These messages describe observable runtime state. Borealis does not expose
private model reasoning or stream raw tool arguments to the terminal.

## Live swarm view

The Aurora swarm tree appears automatically on delegation. It stays in a bounded
region beside the editable prompt and preserves normal terminal scrollback.
Rows show parent/child relationships, state, and `read-only` or `write` capability.
Selected detail shows bounded results, errors and evidence. The shared receipt
shows cumulative admitted descendants (`Agents`), active delegated provider work,
root request usage and authoritative root spend. The footer counts tree entries,
including the root and refusals but excluding descendants hidden by folding.
Unknown spend is labelled; per-agent usage
is never summed into the root receipt again. At most 256 nodes are retained in the
view, with an omission count. Session events retain the full lineage.

| Control | Action |
|---|---|
| F6 | Show/focus the swarm inspector; press again to return to input. |
| Up / Down | Select an agent while the inspector has focus. |
| Left / Right | Collapse / expand the selected branch. |
| Page Up / Page Down | Scroll selected evidence. |
| Enter / Escape | Return to the current input draft and cursor. |
| F7 or `/swarm` | Toggle the tree, including during active work. |
| `/swarm-writes [on or off]` | Inspect/change file-write permission for future turns in this process. |

Ctrl+C, approvals, EOF and queued follow-ups keep their existing behavior.
Resize preserves selection and the draft. Narrow screens clip rows and scroll the
selected region. `NO_COLOR` uses a monochrome view; dumb terminals and redirected
input use the serial shell. JSON mode never starts the live UI. Task and result
text cannot inject ANSI controls, terminal commands, or bidi overrides.

An offline synthetic demo uses the production runner, file tools, event model,
and prompt-toolkit compositor. It makes no live provider requests:

```bash
PYTHONPATH=src python scripts/swarm_demo.py
# Export actual screen cells, text and SVG at a chosen terminal size:
PYTHONPATH=src python scripts/swarm_demo.py --capture-dir /path/to/captures --width 80 --height 24
```

The fixture demonstrates nested/concurrent work, a workspace write, provider
failure, capacity refusal and cancellation. Demo files use an isolated temporary
workspace and disappear when the demo exits.

## Follow-up steering

On an interactive TTY, the prompt stays available while Borealis is working.
Enter a follow-up at the `follow-up · Enter to steer` prompt to queue it for the
active session. Borealis injects queued follow-ups before the next model turn,
including follow-ups entered while a tool is running. If a line arrives exactly
as the active turn finishes, the shell preserves it as the next user turn rather
than dropping it.

While the follow-up prompt is active, prompt-toolkit keeps the editable input line,
cursor position, and wrapped text stable while Borealis streams model and tool output
above it. Borealis pauses the animated activity pulse during input and coalesces rapid
terminal writes, so refreshes cannot erase or split the input buffer.

Except for `/swarm`, slash-prefixed input typed while a turn is active is treated as steering text,
not as an interactive command. Wait for the normal session prompt to run `/help`,
`/status`, `/exit`, or another shell command. Redirected stdin remains serial and
waits for each turn to finish before reading the next line.

```text
╭ AURORA SHELL  v0.1.3 ─────────────────────────────────────╮
│  ◢◤  BOREALIS  CODER                                      │
│      policy-first autonomous coding · interactive mode    │
├────────────────────────────────────────────────────────────┤
│  ROUTE       chatgpt/gpt-5.6-terra                        │
│  GUARDRAIL   workspace-write · approval on-risk           │
╰────────────────────────────────────────────────────────────╯

╭─ YOU  session new
╰─❯ Fix the failing test
│ ◌ Model  model working · chatgpt/gpt-5.6-terra · turn 1
│ ◇ Tool   read_file  tests/test_example.py
│ ◆ Tool complete  read_file · 4 ms

╭─ ✦ BOREALIS
I found the failing assertion and corrected the boundary condition.
╰─
◆ TURN COMPLETE  end_turn · 2 model turns · 1 changed file · verified
```

Options:

```text
--stream / --no-stream
--show-tool-output / --no-show-tool-output
--history / --no-history
```

`--no-stream` buffers assistant text until that model exchange completes.
`--show-tool-output` streams bounded shell output and prints bounded output from
other successful tools in addition to the usual status. Errors are always shown
in bounded form.

## Commands

| Command | Description |
|---|---|
| `/help` | Show the built-in command reference. |
| `/swarm` | Toggle the live tree; F6 focuses it during work. |
| `/swarm-writes [on or off]` | Show/change delegated file-write permission for future turns. |
| `/status` | Show route, safety, sandbox, verification, session, and cumulative usage. |
| `/session` | Show the current workspace, provider, model, and session ID. |
| `/sessions [N]` | List up to N recent sessions for this workspace. |
| `/resume [ID or last]` | Resume a session; defaults to the latest. |
| `/history [N]` | Print up to N recent persisted messages. |
| `/new`, `/clear` | Detach from the current session and start a fresh conversation. |
| `/provider` | List configured providers and their default models. |
| `/provider NAME [MODEL]` | Switch route for future turns in this process. |
| `/model [MODEL]` | Show or switch the active model. |
| `/mode [VALUE]` | Show or change the safety mode. |
| `/approval [VALUE]` | Show or change approval behavior. |
| `/network [on or off]` | Show or change model-initiated network access. |
| `/verify [on or off]` | Show or change automatic verification. |
| `/sandbox [VALUE]` | Show or change the process driver. |
| `/tools [FILTER]` | List effective built-in, plugin, and MCP tools. |
| `/doctor` | Run diagnostics using the current in-process configuration. |
| `/rollback [ID]` | List checkpoints or restore one. |
| `/paste` | Read a multiline task until a line containing only `/end`. |
| `/exit`, `/quit` | Close the shell. |

Use `//` to send a prompt that begins with a slash:

```text
//explain the /api/users route
```

## Cancellation

Press Ctrl+C while Borealis is working to request cancellation of the active
session turn. The runner propagates cancellation to provider streams and active
subprocesses, persists the terminal state, and returns control to the shell. A
second interrupt force-cancels the local task if graceful cancellation has not
completed.

Pressing Ctrl+C at the input prompt exits the CLI with status 130. EOF also exits
normally.

## History and privacy

The durable model conversation is stored in the configured SQLite session store.
Prompt history is separate and optional. On interactive terminals, prompt-toolkit
writes it to `cli-history` under the Borealis storage directory and changes
to mode `0600` on Unix.

Disable local prompt history for sensitive terminals:

```bash
borealis --no-history
```

Provider credentials are never written to prompt history by Borealis, but users
should still avoid pasting secrets into model prompts.

When a provider supplies a reasoning summary, Borealis displays it by default
under a separate `Reasoning summary` label. Initial summary text is shown before
the first assistant answer or tool-call delta for the committed attempt. Later
provider summary chunks can appear in that same labelled stream while the answer
continues. This is a provider-generated summary, not hidden chain-of-thought.
Encrypted reasoning used for stateless continuation remains opaque.

## Automation boundary

Use the interactive shell for local iterative work. Use `borealis run` when a
single invocation, JSONL events, deterministic exit status, or CI-friendly output
is required:

```bash
borealis run --json "Review the current diff"
```
