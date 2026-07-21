#!/usr/bin/env python3
"""Build a numbered, non-duplicating view over one research-agent run.

The source run remains immutable.  The package contains relative symlinks plus
SHA-indexed metadata, grouped as ``code/results/figures/reports/provenance``.
This gives humans the thesis-style numbered layout without copying large run
artifacts or weakening the canonical manifest/evidence authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping

_ID_RE = re.compile(r"^[A-Z0-9]+(?:-[A-Z0-9]+){2,}$")
_REPORT_NAMES = (
    "run_status.json",
    "results_report.md",
    "author_review_note.md",
    "manuscript_ready.md",
    "manuscript_scaffold_bound.md",
    "manuscript_critique.json",
)
_PROVENANCE_NAMES = (
    "manifest.json",
    "manifest_partial.json",
    "analysis_plan.json",
    "research_context.json",
    "experiment_spec.json",
    "run_input_capsule.json",
    "cost_summary.json",
)


class ExperimentPackageError(RuntimeError):
    """A run cannot be indexed without violating package authority."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    for name in ("manifest.json", "manifest_partial.json"):
        path = run_dir / name
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ExperimentPackageError("run manifest must be a JSON object")
            return path, payload
    raise ExperimentPackageError("run has no manifest.json or manifest_partial.json")


def _contained_file(run_dir: Path, candidate: Path) -> Path:
    resolved_run = run_dir.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_run)
    except ValueError as exc:
        raise ExperimentPackageError("artifact path escapes the source run") from exc
    if not candidate.is_file():
        raise ExperimentPackageError(f"indexed artifact is missing: {candidate}")
    return candidate


def _evidence_path(run_dir: Path, record: Mapping[str, Any]) -> Path | None:
    raw = str(record.get("relative_path") or "").strip()
    if not raw:
        return None
    return _contained_file(run_dir, run_dir / raw)


def _category(kind: str, path: Path) -> str:
    normalized = kind.strip().lower()
    if normalized == "figure" or path.suffix.lower() in {
        ".png",
        ".svg",
        ".pdf",
        ".tif",
        ".tiff",
    }:
        return "figures"
    return "results"


def _unique_name(category: str, path: Path, identity: str, used: set[str]) -> str:
    basename = path.name
    name = basename if basename not in used else f"{identity}__{basename}"
    if name in used:
        name = f"{identity[:16]}__{hashlib.sha256(str(path).encode()).hexdigest()[:8]}__{basename}"
    used.add(name)
    return name


def _relative_link(source: Path, destination: Path) -> None:
    destination.symlink_to(os.path.relpath(source, start=destination.parent))


def _record(
    *, category: str, role: str, source: Path, run_dir: Path, link_name: str
) -> dict[str, object]:
    return {
        "category": category,
        "role": role,
        "source_relative_path": source.relative_to(run_dir).as_posix(),
        "package_relative_path": f"{category}/{link_name}",
        "sha256": _sha256(source),
        "size_bytes": source.stat().st_size,
    }


def _iter_code(run_dir: Path) -> Iterable[tuple[str, Path]]:
    for path in sorted((run_dir / "steps").glob("*/analysis.py")):
        step_id = path.parent.name
        yield f"{step_id}__analysis.py", _contained_file(run_dir, path)


def _write_package_readme(path: Path, payload: Mapping[str, Any]) -> None:
    completion = payload.get("completion") or {}
    lines = [
        f"# {payload['experiment_id']}",
        "",
        f"- Benchmark item: `{payload.get('benchmark_item') or 'unknown'}`",
        f"- Run ID: `{payload.get('run_id') or 'unknown'}`",
        f"- Code commit: `{payload.get('code_commit') or 'unknown'}`",
        f"- Source run: `{payload['source_run']}`",
        "- Package mode: relative links; source run remains authoritative",
        "",
        "## Completion",
        "",
        f"- execution_ok: `{completion.get('execution_ok')}`",
        f"- artifact_valid: `{completion.get('artifact_valid')}`",
        f"- scientific_requirement_complete: `{completion.get('scientific_requirement_complete')}`",
        f"- paper_authorized: `{completion.get('paper_authorized')}`",
        "",
        "## Layout",
        "",
        "- `code/` — generated step scripts",
        "- `results/` — registered non-figure evidence",
        "- `figures/` — registered figure evidence",
        "- `reports/` — human-facing run/readiness reports",
        "- `provenance/` — manifest, plan, context, cost, and run identity",
        "- `package.json` — SHA-indexed machine-readable inventory",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _refresh_index(package_root: Path) -> None:
    packages: list[dict[str, object]] = []
    for package_json in sorted(package_root.glob("*/package.json")):
        payload = json.loads(package_json.read_text(encoding="utf-8"))
        packages.append(
            {
                key: payload.get(key)
                for key in (
                    "experiment_id",
                    "benchmark_item",
                    "run_id",
                    "code_commit",
                    "source_run",
                    "completion",
                )
            }
        )
    (package_root / "INDEX.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu.agent_experiment_index/1",
                "packages": packages,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown = ["# Agent experiment packages", ""]
    for package in packages:
        markdown.append(
            "- `{experiment_id}` — {benchmark_item}; run `{run_id}`; code `{code_commit}`".format(
                **{key: package.get(key) or "unknown" for key in package}
            )
        )
    (package_root / "INDEX.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def build_experiment_package(
    *, run_dir: Path, package_root: Path, experiment_id: str
) -> Path:
    """Create one numbered package and refresh the package-root index."""

    if _ID_RE.fullmatch(experiment_id) is None:
        raise ExperimentPackageError(
            "experiment id must be uppercase and structured, e.g. FIG2-E2-DEV-001"
        )
    run_dir = run_dir.resolve()
    package_root = package_root.resolve()
    manifest_path, manifest = _load_manifest(run_dir)
    destination = package_root / experiment_id
    if destination.exists():
        raise ExperimentPackageError(
            f"experiment package already exists: {destination}"
        )
    package_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{experiment_id}.", dir=package_root))
    try:
        for category in ("code", "results", "figures", "reports", "provenance"):
            (stage / category).mkdir()
        inventory: list[dict[str, object]] = []
        used: dict[str, set[str]] = {
            name: set()
            for name in ("code", "results", "figures", "reports", "provenance")
        }

        for link_name, source in _iter_code(run_dir):
            _relative_link(source, stage / "code" / link_name)
            inventory.append(
                _record(
                    category="code",
                    role="step_script",
                    source=source,
                    run_dir=run_dir,
                    link_name=link_name,
                )
            )
        for raw in manifest.get("evidence") or []:
            if not isinstance(raw, Mapping):
                continue
            source = _evidence_path(run_dir, raw)
            if source is None:
                continue
            expected_sha = str(raw.get("sha256") or "")
            observed_sha = _sha256(source)
            if expected_sha and expected_sha != observed_sha:
                raise ExperimentPackageError(
                    f"evidence digest mismatch: {raw.get('evidence_id')}"
                )
            category = _category(str(raw.get("kind") or ""), source)
            identity = str(raw.get("evidence_id") or observed_sha[:16])
            link_name = _unique_name(category, source, identity, used[category])
            _relative_link(source, stage / category / link_name)
            inventory.append(
                _record(
                    category=category,
                    role=identity,
                    source=source,
                    run_dir=run_dir,
                    link_name=link_name,
                )
            )
        for category, names in (
            ("reports", _REPORT_NAMES),
            ("provenance", _PROVENANCE_NAMES),
        ):
            for name in names:
                source = run_dir / name
                if not source.is_file():
                    continue
                link_name = _unique_name(category, source, name, used[category])
                _relative_link(source, stage / category / link_name)
                inventory.append(
                    _record(
                        category=category,
                        role=name,
                        source=source,
                        run_dir=run_dir,
                        link_name=link_name,
                    )
                )

        code_version = manifest.get("code_version")
        code_version = code_version if isinstance(code_version, Mapping) else {}
        readiness = manifest.get("readiness")
        readiness = readiness if isinstance(readiness, Mapping) else {}
        completion = {
            "execution_ok": readiness.get(
                "execution_ok", readiness.get("execution_complete")
            ),
            "artifact_valid": readiness.get(
                "artifact_valid", readiness.get("evidence_complete")
            ),
            "scientific_requirement_complete": readiness.get(
                "scientific_requirement_complete", readiness.get("analysis_validated")
            ),
            "paper_authorized": readiness.get(
                "paper_authorized", readiness.get("publication_ready")
            ),
        }
        payload = {
            "schema_version": "easyicu.agent_experiment_package/1",
            "experiment_id": experiment_id,
            "benchmark_item": (
                run_dir.parents[1].name if len(run_dir.parents) > 1 else None
            ),
            "run_id": manifest.get("run_id") or run_dir.name,
            "source_run": os.path.relpath(run_dir, start=package_root),
            "source_manifest": manifest_path.name,
            "source_manifest_sha256": _sha256(manifest_path),
            "code_commit": (
                code_version.get("git_sha")
                or code_version.get("commit")
                or code_version.get("git_commit")
            ),
            "code_branch": code_version.get("git_branch") or code_version.get("branch"),
            "code_dirty": (
                code_version.get("git_dirty")
                if "git_dirty" in code_version
                else code_version.get("dirty")
            ),
            "completion": completion,
            "inventory": sorted(
                inventory,
                key=lambda item: (
                    str(item["category"]),
                    str(item["package_relative_path"]),
                ),
            ),
        }
        (stage / "package.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_package_readme(stage / "PACKAGE.md", payload)
        stage.rename(destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    _refresh_index(package_root)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path("research_output/_packages"),
    )
    args = parser.parse_args()
    destination = build_experiment_package(
        run_dir=args.run_dir,
        package_root=args.package_root,
        experiment_id=args.experiment_id,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
