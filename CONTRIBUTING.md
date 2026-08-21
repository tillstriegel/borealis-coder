# Contributing

1. Open an issue for large behavioral or protocol changes.
2. Keep the runtime dependency-free unless a dependency removes more risk than it adds.
3. Add deterministic tests for every policy, parser, provider, or mutation change.
4. Run `make check` before submitting a pull request.
5. Never include credentials, private transcripts, or customer code in fixtures.

Commits should be small, explain intent, and preserve backward compatibility where practical.
Security-sensitive changes require tests covering both allow and deny paths.
