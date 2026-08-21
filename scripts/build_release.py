"""Build and structurally validate source and wheel distributions without extra tooling."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

from setuptools.build_meta import build_sdist, build_wheel

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    shutil.rmtree(DIST, ignore_errors=True)
    DIST.mkdir(parents=True)
    sdist_name = build_sdist(str(DIST))
    wheel_name = build_wheel(str(DIST))
    sdist = DIST / sdist_name
    wheel = DIST / wheel_name

    errors: list[str] = []
    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()
        for required in (
            "pyproject.toml",
            "README.md",
            "LICENSE",
            "SECURITY.md",
            "docs/ARCHITECTURE.md",
            "docs/INTERACTIVE_CLI.md",
            "docs/RESEARCH.md",
            "scripts/validate_release.py",
        ):
            if not any(name.endswith("/" + required) for name in names):
                errors.append(f"sdist missing {required}")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        suffixes = (
            ".dist-info/METADATA",
            ".dist-info/WHEEL",
            ".dist-info/entry_points.txt",
            "borealis_coder/py.typed",
        )
        for required in suffixes:
            if not any(name.endswith(required) for name in names):
                errors.append(f"wheel missing *{required}")

    artifacts = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in (sdist, wheel)
    ]
    report = {"ok": not errors, "errors": errors, "artifacts": artifacts}
    (DIST / "ARTIFACTS.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
