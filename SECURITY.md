# Security Policy

## Supported versions

Security fixes are provided for the latest minor release.

## Reporting

Do not open public issues for suspected vulnerabilities. Send a private report to the
maintainers with impact, reproduction steps, affected versions, and suggested mitigations.

## Execution boundary

The native driver is a policy and process-containment layer, not a kernel security boundary.
Use the Docker driver, a VM, or another hardened sandbox for untrusted repositories or models.
API keys are read from environment variables, redacted from traces, and never written to the
session database by Borealis. MCP servers and plugins execute with the privileges granted to
the Borealis process and must be reviewed independently.
