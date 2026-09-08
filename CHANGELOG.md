# Changelog

All notable changes follow Keep a Changelog and Semantic Versioning.

## [Unreleased]

### Added
- Compaction v2 with atomic conversation bundles, structured deterministic state,
  provider-aware target budgets, durable incremental artifacts, bounded LLM
  summarization, overflow recovery, safe observability, and a fixed release-gate corpus.

### Fixed
- Streamed request encoding errors reach the caller instead of leaving it waiting
  for a worker that has already failed.
- Cancelled streaming and non-streaming provider requests release active HTTP sockets and stop
  connection-pool waits. Pending connection setup cannot hold interpreter shutdown
  open, and a cancelled direct connection sends no late request after setup finishes.
  Stream redirects release socket ownership before pool reuse, so cancellation
  cannot abort the next borrower. Direct stream redirects check cancellation before
  opening the next request.
- Malformed checkpoint path and root references fail shared manifest validation,
  preventing unusable checkpoints from displacing valid recovery points during retention.
- Concurrent tools retain their own call identity, so delegated progress is attached
  to the correct parent call. File changes and mutation uncertainty remain shared
  across the run, including after tool failure or cancellation.
- Repository-map, skill, instruction, and delegated prompt preparation run off the
  event loop. Concurrent repository-map scans publish complete caches safely.
- Discovered file paths, checkpoint targets, and patch filenames preserve literal
  environment-variable syntax instead of selecting another file. Path validation
  reuses the resolved path when formatting its display name.
- Regular-expression search stays cancellable when a pattern backtracks excessively.
  Matching uses one isolated worker per query with a configurable per-file deadline;
  worker cleanup also handles timeouts and repeated cancellation.
  Worker response buffers honor larger configured result limits.
- Automatic npm verification runs the project's test script without adding
  Jest-only flags that fail with other test runners.
- Delegated provider retries honor the configured attempt limit instead of
  multiplying attempts through nested retry loops. Concurrent requests keep
  independent retry budgets, and cancellation releases the retry scope.
- Compaction summaries share one configured retry loop and preserve reported
  failed-attempt usage on cancellation.
- Streamed retries retain reported usage when a later attempt ends with a
  non-retryable provider error, including authentication and context overflow errors.
  Cancellation interrupts retry backoff and preserves usage reported by failed attempts,
  including when the run deadline expires or accounting receives repeated cancellation.
  Accounting for completed responses also finishes before cancellation propagates,
  keeping run totals consistent with stored usage.
- Gemini streaming reconstructs text, function arguments, and thought signatures
  when completion metadata omits the generated steps. Failed interactions cannot
  appear complete, incomplete responses hide tool calls, and cumulative usage survives
  incomplete streams, provider errors, and transport failures.
- Context pruning preserves earlier file excerpts when a later read omits content
  through truncation. Identical outputs still deduplicate; covering reads must
  contain an untruncated, consecutive line range. Direct duplicate lookups avoid
  repeated history scans when a session reads the same file many times.
  Literal backslashes in POSIX filenames remain distinct from directory separators
  when matching reads against later file changes.
- Bounded file reads accumulate small chunks instead of allocating the configured
  byte ceiling for small files. Write and delete operations stream their pre-edit
  and revalidation hashes, retaining stale-file checks without full duplicate reads.
- ACP stdout pipes on POSIX use async flow control, so blocked output no longer leaves an
  executor thread preventing shutdown after a timeout or disconnect. Timed-out
  frames remain complete when reading resumes, and later writes drain earlier data.
- MCP tool discovery rejects pagination cycles and invalid cursor types instead of
  looping or returning incomplete results. Opaque cursors retain their exact value,
  including empty strings.
- MCP stdio shutdown stops remaining children in its POSIX process group and
  finishes pipe readers after normal server exit or cancellation. Process cleanup
  shares the bounded lifecycle handling used by native commands.
- Automatic repository maps honor small character budgets across stable and
  request-focused sections, including headings. A zero budget skips source-map
  discovery while retaining repository instructions and skills.
- Context caches detect file replacements and POSIX metadata changes even when
  size and modification time stay unchanged. This refreshes repository symbols,
  ignore rules, and skill guidance after preserved-timestamp edits.
  Skill aliases that share one file retain their distinct names across cached scans,
  and retargeted aliases update the path used to resolve skill resources.
- Repository discovery outside Git skips named pipes and other non-regular files,
  preventing context assembly from blocking while opening a pipe with a source-file extension.
- Mixed Git patches reject unsupported sections before any mutation instead of
  reporting partial success. This includes binary, mode, rename, copy, merge, and
  hunkless changes. Quoted Git paths fail explicitly; envelopes accept literal paths.
- File mutations wait for workspace locks without blocking the event loop.
  Cancellation while waiting releases local resources before any file change,
  while acquired transactions remain serialized across threads and processes.
  CLI and interactive checkpoint restoration use the same mutation lock.
- Failed checkpoint creation removes its partial files, and checkpoint reads stop
  at the remaining byte budget. Fixed-size blob names support long source paths
  while existing checkpoints remain readable. Retention scans hash blobs in bounded
  chunks instead of loading each complete blob into memory.
- New checkpoints track additional workspace roots by path, so reordering roots
  cannot restore files into another root. Missing roots fail before any restoration.
  Restore also rejects file/directory conflicts and paths redirected through new symlinks
  before changing any files, including paths in legacy manifests.
  Invalid saved file modes fail before restoration and are excluded from retention counts.
  Malformed manifest metadata no longer breaks checkpoint listing or creation;
  an explicit restore reports an error before changing files.
- Atomic writes support long legal filenames. A deleted empty file is rejected as
  stale instead of being recreated under its previous empty-content hash.
  Text replacement and patch reads enforce file limits before loading excess data.
- Unified patches place zero-context insertions correctly, preserve header-like
  content and missing final newlines, and accept multi-file Git diff metadata.
  They preserve CRLF and mixed line endings, including explicit line-ending changes.
  Existing LF files still accept CRLF-formatted patches after exact matching fails.
  Added files use the same hunk validation as updates. Incomplete hunks and missing
  envelope terminators fail before mutation; envelope paths keep literal `a/` and
  `b/` directories.
- Runner shutdown attempts every resource cleanup even if an MCP or provider close
  fails, while still reporting the cleanup error.
- Failed or cancelled ACP session setup closes its runtime resources, including
  when a resume request names an unknown session.
  Resuming an active session stops its old prompt tasks before closing their runtime.
  Lifecycle requests for the same session are serialized to prevent overlapping
  resume, close, delete, and prompt requests from losing runtime ownership.
  Session listing owns its database connection so concurrent close requests cannot
  invalidate an in-progress query.
- ACP accepts incoming JSON messages up to 16 MiB. Disconnects, read failures, and
  cancellation close the input transport, drain handlers, and close session runners.
  Cancelling server shutdown while prompts finish still closes all session runners.
  Cancelled shutdowns no longer report unobserved future errors on Python 3.11.
  Peer request timeouts include writes, and malformed peer errors fail the request.
- MCP stdio request timeouts include blocked writes, failed sends release pending
  requests, and long stderr lines no longer stop output draining.
- MCP stdio accepts JSON messages up to 16 MiB instead of the stream reader's
  implicit 64 KiB limit. Oversized messages fail pending requests and allow clean
  shutdown; unrelated malformed messages cannot crash response dispatch.
- Instruction discovery enforces its character limit during reads, and repository
  maps enforce their byte limit even if a file grows after discovery.
  Path-specific instruction lookup visits only ancestor scopes, so unrelated
  directory trees and unreadable guidance cannot delay or break the lookup.
- Repository-map character limits include headers, separators, and truncation
  markers. Complete entries that fit the budget remain intact.
- Python repository maps skip expression subtrees during symbol extraction while
  retaining definitions and imports in nested statement scopes.
- Budget exits finalize started tool calls, cancel and drain parallel reads, and
  record skipped calls before completing the run. Observed workspace changes remain
  tracked, and resumption no longer treats those calls as abandoned work.
- Delegated investigations retain provider-reported usage on failures, enforce the
  shared cost limit after each response, and reject incomplete or blank conclusions.
  Cancelled delegated retries preserve reported failed-attempt usage in the parent session.
  They replay provider-selected continuation state after tool calls, including
  opaque reasoning state and signatures, with the configured provider route identity.
  Tool calls from incomplete delegated responses are not executed. Every logical
  model request checks the shared time and cost limits before starting.
- Repository discovery and changed-file ranking preserve Unicode, spaces, and rename
  destinations. Git status and repository maps escape unusual filename characters
  for display without changing the paths used for matching.
- Reading an empty file or a range past its end returns an empty selection with the
  current file hash. File reads enforce the byte limit before loading excess data.
- Text search stops directory discovery at its result limit, bounds reads and output
  while scanning, and yields between files and line batches so it can be cancelled.
- Read-only and no-change investigations now stop for synthesis after a bounded number
  of turns, and parent, delegated, and summarizer requests share one run-wide ceiling.
- Delegated investigations reserve a tool-free final turn and fail clearly when they do
  not return a textual conclusion.
- Compaction now reuses artifacts across token and byte triggers, prunes covered file
  reads, keeps retained tool output below its target, and deduplicates persisted provider
  messages through a backward-compatible schema migration.

### Security
- Recursive text search skips file symlinks outside the configured workspace roots
  and avoids opening non-regular files such as named pipes.
- Compacted user, assistant, and tool history is escaped through one untrusted-data
  boundary implementation. Durable messages are append-only across every write path.

## [0.1.3] - 2026-08-21

### Added
- A persistent Codex-style interactive CLI is now the default when running
  `borealis` without a subcommand.
- Durable session continuation through `borealis --continue`,
  `borealis resume --last`, and `/resume` inside the shell.
- Interactive commands for session history, provider/model switching, safety
  mode, approvals, network access, verification, sandbox selection, tools,
  diagnostics, checkpoints, and multiline prompts.
- Best-effort readline history and slash-command completion, with local history
  stored using owner-only permissions on Unix.
- Turn-level Ctrl+C cancellation that preserves the interactive process and
  durable session.
- The dependency-free Aurora Shell visual system with a responsive launch panel,
  session-aware prompt, assistant frames, activity rail, command panels, and
  compact turn receipts.
- Live elapsed-time pulses for pending model, tool, and verification work, with
  accessible uncolored and redirected-output fallbacks.
- Stable provider-cache context prefixes, explicit Anthropic prompt-cache
  breakpoints, cache-aware ChatGPT routing keys, and adaptive cache tuning.
- Bounded exact-response caching for successful text-only requests, with tool
  responses excluded from replay.
- Per-turn prompt/response cache metrics, net savings, and cache-write-aware
  provider cost accounting.
- A readline-native command selector that opens when an interactive message
  begins with `/`, with descriptions, prefix filtering, and Tab completion.

### Fixed
- Command menu text now follows the terminal foreground color, keeping slash
  commands readable in both light and dark color schemes.
- The final configured model turn now runs without tools. Runs that still reach
  the limit preserve resumable history, verify changed files when enabled, and
  report explicit recovery guidance across terminal and ACP clients.
- Assistant output is always terminated cleanly before the next prompt.
- Final answers after one or more tool calls are no longer suppressed by text
  emitted during an earlier model exchange in the same user turn.
- Interactive status, tool, and response output share a deterministic terminal
  stream, avoiding prompt/footer interleaving in piped and captured sessions.

## [0.1.2] - 2026-08-21

### Added
- First-class `chatgpt` provider for ChatGPT-plan access through the Codex
  Responses backend.
- Discovery of secure file-backed Codex logins, `CODEX_ACCESS_TOKEN`, and a
  Borealis-managed Codex credential home.
- `borealis auth login`, `borealis auth status`, and `borealis auth logout`.
- OAuth token refresh with rotation, atomic persistence, cross-process locking,
  proactive expiry checks, and one safe pre-output authentication retry.
- Preservation of encrypted Responses reasoning state across tool turns without
  persisting plaintext reasoning summaries or complete native responses.

### Security
- ChatGPT bearer, account, and refresh-token headers are authoritative and cannot
  be replaced by repository-local provider headers.
- Custom ChatGPT API or OAuth endpoints require both an explicit configuration
  flag and the process-level `BOREALIS_ALLOW_CUSTOM_CHATGPT_ENDPOINTS=1` opt-in.
- Borealis never reads ChatGPT browser cookies or browser-session storage.

## [0.1.1] - 2026-08-20

### Added
- First-class OpenRouter provider with `OPENROUTER_API_KEY` auto-detection.
- OpenRouter attribution headers, native reasoning controls, model-level fallbacks,
  provider routing/privacy preferences, and an `extra_body` extension escape hatch.
- OpenRouter-reported request-cost and reasoning-token accounting when supplied.
- Chat Completions by default with optional OpenRouter Responses API mode.

## [0.1.0] - 2026-08-20

### Added
- Async, provider-portable coding-agent runtime.
- OpenAI Responses/Chat Completions, Anthropic Messages, Gemini Interactions,
  OpenAI-compatible, and deterministic mock providers.
- Policy engine, approvals, command risk classification, native/Docker process drivers,
  workspace containment, secret redaction, and automatic file checkpoints.
- Deterministic filesystem/search/patch/git/verification/repository-map/todo tools.
- Durable SQLite sessions, usage accounting, compaction, stuck detection, cancellation,
  JSONL traces, skills, plugins, MCP stdio/HTTP clients, and ACP v2 agent server.
- CLI, JSON automation mode, diagnostics, built-in evaluations, release and CI assets.
