# Release checklist

## Source and provenance

- [ ] Version matches `pyproject.toml`, package `__version__`, and changelog.
- [ ] README and architecture/security/protocol documentation are present.
- [ ] Research snapshot date and popularity counts are explicit.
- [ ] No copied third-party source or incompatible license material is present.
- [ ] License, code of conduct, contribution guide, and security policy are current.

## Correctness

- [ ] `python -m compileall` passes.
- [ ] Unit and integration suite passes on Python 3.11, 3.12, and 3.13.
- [ ] Offline model/tool/session smoke test passes.
- [ ] Bare `borealis`, multi-turn input, `/continue`/resume, final-answer rendering, and Ctrl+C cancellation are smoke-tested.
- [ ] `python scripts/interactive_smoke_test.py` passes from the source tree and after wheel installation.
- [ ] Strict schemas validate for every effective built-in tool.
- [ ] Provider request/response fixtures pass.
- [ ] MCP stdio lifecycle test passes.
- [ ] ACP session/prompt/replay/cancel tests pass.
- [ ] Patch atomicity, path escape, stale hash, rollback, policy, and shell bounds tests pass.

## Security and privacy

- [ ] Secret-marker and private-key scan is clean.
- [ ] No API keys, local databases, traces, checkpoints, caches, or `.env` files are packaged.
- [ ] Workspace plugins remain disabled by default.
- [ ] Native sandbox limitations are prominent.
- [ ] Docker image and CI actions are pinned according to operator policy.
- [ ] Default network, commit, push, and outside-workspace gates remain closed.

## Packaging

- [ ] Wheel builds without downloading runtime dependencies.
- [ ] Source distribution builds.
- [ ] Wheel installs into a clean target/virtual environment.
- [ ] Installed `borealis --version`, `borealis eval`, and imports pass.
- [ ] Wheel metadata and file list are inspected.
- [ ] Source ZIP has one top-level directory and excludes transient files.
- [ ] SHA-256 checksums are generated.

## Operations

- [ ] `borealis doctor` passes for the release environment.
- [ ] Cancellation and timeout are exercised.
- [ ] Session export can be parsed independently.
- [ ] Checkpoint rollback is exercised.
- [ ] Known limitations are copied into the validation manifest and release notes.
- [ ] Live provider canaries are run separately for each supported production route.
