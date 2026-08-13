#!/usr/bin/env python3
"""Fail on repository-layout drift that creates competing owners.

The audit deliberately distinguishes tracked source hygiene from local caches.
Local generated directories are allowed when ignored; checked-in generated
payloads and competing source owners are not.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable


REQUIRED_PATHS = (
    Path("benchmarks/README.md"),
    Path("benchmarks/cases"),
    Path("benchmarks/catalogs"),
    Path("benchmarks/idea_mining"),
    Path("docs/qa"),
    Path("docs/repository_layout.md"),
    Path("docs/research_agent_capability_inventory.md"),
    Path(".codegraph/.gitignore"),
)

FORBIDDEN_TRACKED_PARTS = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".playwright-cli",
    }
)

FORBIDDEN_TRACKED_ROOTS = frozenset(
    {
        "benchmark",
        "build",
        "dist",
        "output",
        "research_output",
        "scratchpad",
    }
)

FORBIDDEN_ROOT_FILES = frozenset({"design-qa.md"})


def _tracked_files(repo_root: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return tuple(Path(raw.decode("utf-8")) for raw in result.stdout.split(b"\0") if raw)


def audit_repository(
    repo_root: Path, *, tracked_files: Iterable[Path] | None = None
) -> tuple[str, ...]:
    """Return stable, human-readable hygiene failures for ``repo_root``."""

    root = repo_root.resolve()
    tracked = (
        tuple(tracked_files) if tracked_files is not None else _tracked_files(root)
    )
    findings: list[str] = []

    for required in REQUIRED_PATHS:
        if not (root / required).exists():
            findings.append(f"missing required repository owner: {required.as_posix()}")

    if (root / "benchmark").exists():
        findings.append("competing top-level benchmark/ owner exists; use benchmarks/")

    for filename in sorted(FORBIDDEN_ROOT_FILES):
        if (root / filename).exists():
            findings.append(f"root QA/ad-hoc file must be relocated: {filename}")

    for path in sorted(tracked, key=lambda item: item.as_posix()):
        if path.parts and path.parts[0] in FORBIDDEN_TRACKED_ROOTS:
            findings.append(f"generated or retired root is tracked: {path.as_posix()}")
        if FORBIDDEN_TRACKED_PARTS.intersection(path.parts):
            findings.append(f"cache payload is tracked: {path.as_posix()}")
        if path.name == ".DS_Store":
            findings.append(f"macOS metadata is tracked: {path.as_posix()}")
        if path.parts[:3] == ("src", "easyicu", "output"):
            findings.append(
                f"run artifact is inside importable source: {path.as_posix()}"
            )

    codegraph_ignore = root / ".codegraph" / ".gitignore"
    if codegraph_ignore.exists():
        ignore_text = codegraph_ignore.read_text(encoding="utf-8")
        if "*" not in ignore_text or "!.gitignore" not in ignore_text:
            findings.append(".codegraph/.gitignore must ignore local index payloads")

    return tuple(dict.fromkeys(findings))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    findings = audit_repository(args.repo_root)
    if findings:
        for finding in findings:
            print(f"ERROR: {finding}")
        return 1
    print("repository hygiene: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
