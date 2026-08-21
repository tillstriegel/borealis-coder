"""Validate release metadata, schemas, package import, and repository hygiene."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from borealis_coder import __version__
from borealis_coder.tools import build_builtin_registry


def main() -> int:
    errors: list[str] = []
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if f'version = "{__version__}"' not in pyproject:
        errors.append("package version does not match pyproject.toml")
    required = [
        "README.md",
        "LICENSE",
        "SECURITY.md",
        "CHANGELOG.md",
        "docs/ARCHITECTURE.md",
        "docs/INTERACTIVE_CLI.md",
    ]
    errors.extend(f"missing {name}" for name in required if not (ROOT / name).is_file())
    for schema in build_builtin_registry().schemas():
        params = schema["parameters"]
        if params.get("type") != "object" or params.get("additionalProperties") is not False:
            errors.append(f"tool {schema['name']} is not strict-schema compatible")
        if set(params.get("properties", {})) != set(params.get("required", [])):
            errors.append(f"tool {schema['name']} has optional keys outside nullable strict form")
    secret_patterns = {
        "OpenAI project key": re.compile(r"\bsk-proj-[A-Za-z0-9_-]{16,}\b"),
        "OpenRouter API key": re.compile(r"\bsk-or-v1-[A-Za-z0-9_-]{16,}\b"),
        "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "private key block": re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\s+[A-Za-z0-9+/=\r\n]{64,}",
            re.MULTILINE,
        ),
    }
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in {".git", "dist", "build", "__pycache__"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in secret_patterns.items():
            if pattern.search(text):
                errors.append(f"possible {label} in {path.relative_to(ROOT)}")
    report = {
        "version": __version__,
        "ok": not errors,
        "errors": errors,
        "python_files": len(list((ROOT / "src").rglob("*.py"))),
        "tests": len(list((ROOT / "tests").glob("test_*.py"))),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
