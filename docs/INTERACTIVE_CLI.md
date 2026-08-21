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

Assistant text is streamed by default. Every model exchange is rendered
independently, so a planning message before a tool call cannot suppress the final
answer after the tool result. Borealis reconciles the completed response with its
stream deltas and terminates the line before rendering tool state, the turn
footer, or the next prompt.

Interactive mode also reports live execution phases. It announces workspace
context preparation immediately, identifies each provider/model exchange, shows
when a tool call is being assembled, reports provider retries, renders tool
start/completion, and announces verification before it begins. These messages
describe observable runtime state; Borealis does not expose private model
reasoning or stream raw tool arguments to the terminal. After ten seconds with
no new model or tool event, a heartbeat reports the current phase and elapsed
turn time.

Options:

```text
--stream / --no-stream
--show-tool-output / --no-show-tool-output
--history / --no-history
```

`--no-stream` buffers assistant text until that model exchange completes.
`--show-tool-output` prints bounded successful tool output in addition to the
usual start/completion status. Errors are always shown in bounded form.

## Commands

| Command | Description |
|---|---|
| `/help` | Show the built-in command reference. |
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
Readline prompt history is separate and optional. When supported by the platform,
it is written to `cli-history` under the Borealis storage directory and changed
to mode `0600` on Unix.

Disable local prompt history for sensitive terminals:

```bash
borealis --no-history
```

Provider credentials are never written to prompt history by Borealis, but users
should still avoid pasting secrets into model prompts.

## Automation boundary

Use the interactive shell for local iterative work. Use `borealis run` when a
single invocation, JSONL events, deterministic exit status, or CI-friendly output
is required:

```bash
borealis run --json "Review the current diff"
```
