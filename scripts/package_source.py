"""Create a deterministic source ZIP with one top-level directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = "borealis-coder"
EXCLUDED_PARTS = {
    ".git",
    ".borealis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
}
EXCLUDED_NAMES = {".coverage", "SOURCE_FILES.json"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def included_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.name in EXCLUDED_NAMES or path.suffix in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_source_manifest(files: list[Path]) -> Path:
    manifest = {
        "format": 1,
        "root": TOP_LEVEL,
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    target = ROOT / "SOURCE_FILES.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def zip_info(archive_name: str, mode: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (mode & 0xFFFF) << 16
    return info


def build_zip(output: Path) -> dict[str, object]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = included_files()
    manifest = write_source_manifest(files)
    files = included_files() + [manifest]
    files = sorted(set(files), key=lambda item: item.relative_to(ROOT).as_posix())

    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        directory = zip_info(f"{TOP_LEVEL}/", stat.S_IFDIR | 0o755)
        archive.writestr(directory, b"")
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            mode = stat.S_IFREG | (0o755 if os.access(path, os.X_OK) else 0o644)
            archive.writestr(zip_info(f"{TOP_LEVEL}/{relative}", mode), path.read_bytes())
    temporary.replace(output)
    return {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "files": len(files),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "borealis-coder-production.zip",
    )
    args = parser.parse_args()
    print(json.dumps(build_zip(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
