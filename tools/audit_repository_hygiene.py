#!/usr/bin/env python3
"""Fail on repository-layout drift that creates competing owners.

The audit deliberately distinguishes tracked source hygiene from local caches.
Local generated directories are allowed when ignored; checked-in generated
payloads and competing source owners are not.
"""

from __future__ import annotations

import argparse
import ast
import json
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
    Path("docs/research_agent_duplication_audit.md"),
    Path(".codegraph/.gitignore"),
    Path("tools/arch_baselines/research_agent_top_level_ownership.json"),
    Path("tools/arch_baselines/research_agent_duplicate_helpers.json"),
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

TOP_LEVEL_OWNERSHIP_MANIFEST = Path(
    "tools/arch_baselines/research_agent_top_level_ownership.json"
)
DUPLICATE_HELPER_BASELINE = Path(
    "tools/arch_baselines/research_agent_duplicate_helpers.json"
)


def _duplicate_helper_findings(root: Path) -> tuple[str, ...]:
    """Reject new local copies while allowing the baseline to shrink."""

    baseline_path = root / DUPLICATE_HELPER_BASELINE
    if not baseline_path.exists():
        return ()
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (f"duplicate-helper baseline is unreadable: {exc}",)
    declared = baseline.get("helpers")
    if not isinstance(declared, dict):
        return ("duplicate-helper baseline lacks a helpers object",)

    package_root = root / "src" / "easyicu" / "research_agent"
    counts: dict[str, dict[str, int]] = {str(name): {} for name in declared}
    findings: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            findings.append(f"research-agent helper scan failed: {path}: {exc}")
            continue
        relative = path.relative_to(package_root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in counts:
                continue
            counts[node.name][relative] = counts[node.name].get(relative, 0) + 1

    for name, actual_by_file in sorted(counts.items()):
        allowed = declared.get(name)
        if not isinstance(allowed, dict):
            findings.append(f"duplicate-helper baseline row is invalid: {name}")
            continue
        for relative, actual_count in sorted(actual_by_file.items()):
            allowed_count = allowed.get(relative)
            if not isinstance(allowed_count, int) or actual_count > allowed_count:
                findings.append(
                    "new local duplicate helper definition: "
                    f"{name} x{actual_count} in {relative}"
                )
    return tuple(findings)


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

    ownership_path = root / TOP_LEVEL_OWNERSHIP_MANIFEST
    if ownership_path.exists():
        try:
            ownership = json.loads(ownership_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            findings.append(f"top-level ownership manifest is unreadable: {exc}")
        else:
            declared = ownership.get("modules")
            if not isinstance(declared, dict):
                findings.append("top-level ownership manifest lacks a modules object")
            else:
                package_root = root / "src" / "easyicu" / "research_agent"
                actual = {
                    path.name for path in package_root.glob("*.py") if path.is_file()
                }
                declared_names = set(declared)
                for missing in sorted(actual - declared_names):
                    findings.append(
                        f"unowned research-agent top-level module: {missing}"
                    )
                for stale in sorted(declared_names - actual):
                    findings.append(f"stale top-level module ownership row: {stale}")
                for name, row in sorted(declared.items()):
                    if not isinstance(row, dict) or not str(row.get("owner") or ""):
                        findings.append(f"top-level module lacks an owner: {name}")
                    if not isinstance(row, dict) or not str(row.get("reason") or ""):
                        findings.append(f"top-level module lacks a reason: {name}")

    findings.extend(_duplicate_helper_findings(root))

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
