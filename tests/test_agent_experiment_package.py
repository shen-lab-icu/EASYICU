from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.agent_experiment_package import (
    ExperimentPackageError,
    build_experiment_package,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "bench" / "E2_example" / "aware" / "run_001"
    evidence = run / "evidence"
    step = run / "steps" / "01_model"
    evidence.mkdir(parents=True)
    step.mkdir(parents=True)
    (step / "analysis.py").write_text("print('ok')\n", encoding="utf-8")
    table = evidence / "ev_table__result.csv"
    table.write_text("term,estimate\nx,1.2\n", encoding="utf-8")
    figure = evidence / "ev_figure__figure.svg"
    figure.write_text("<svg/>\n", encoding="utf-8")
    (run / "run_status.json").write_text("{}\n", encoding="utf-8")
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run_001",
                "code_version": {"commit": "abc123", "branch": "main", "dirty": False},
                "readiness": {
                    "execution_complete": True,
                    "evidence_complete": False,
                    "analysis_validated": False,
                    "publication_ready": False,
                },
                "evidence": [
                    {
                        "evidence_id": "ev_table",
                        "kind": "table",
                        "relative_path": table.relative_to(run).as_posix(),
                        "sha256": _sha(table),
                    },
                    {
                        "evidence_id": "ev_figure",
                        "kind": "figure",
                        "relative_path": figure.relative_to(run).as_posix(),
                        "sha256": _sha(figure),
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run


def test_builds_numbered_categorized_package_without_copying(tmp_path: Path):
    run = _run(tmp_path)
    root = tmp_path / "packages"

    package = build_experiment_package(
        run_dir=run, package_root=root, experiment_id="FIG2-E2-DEV-001"
    )

    assert (package / "code" / "01_model__analysis.py").is_symlink()
    assert (package / "results" / "ev_table__result.csv").is_symlink()
    assert (package / "figures" / "ev_figure__figure.svg").is_symlink()
    assert (package / "reports" / "run_status.json").is_symlink()
    payload = json.loads((package / "package.json").read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "FIG2-E2-DEV-001"
    assert payload["code_commit"] == "abc123"
    assert payload["completion"] == {
        "execution_ok": True,
        "artifact_valid": False,
        "scientific_requirement_complete": False,
        "paper_authorized": False,
    }
    assert len(payload["inventory"]) == 5
    assert "FIG2-E2-DEV-001" in (root / "INDEX.md").read_text(encoding="utf-8")


def test_duplicate_experiment_id_is_rejected(tmp_path: Path):
    run = _run(tmp_path)
    root = tmp_path / "packages"
    build_experiment_package(
        run_dir=run, package_root=root, experiment_id="FIG2-E2-DEV-001"
    )

    with pytest.raises(ExperimentPackageError, match="already exists"):
        build_experiment_package(
            run_dir=run, package_root=root, experiment_id="FIG2-E2-DEV-001"
        )


def test_invalid_unstructured_id_is_rejected(tmp_path: Path):
    with pytest.raises(ExperimentPackageError, match="uppercase and structured"):
        build_experiment_package(
            run_dir=_run(tmp_path),
            package_root=tmp_path / "packages",
            experiment_id="latest",
        )


def test_tampered_evidence_is_rejected(tmp_path: Path):
    run = _run(tmp_path)
    (run / "evidence" / "ev_table__result.csv").write_text(
        "term,estimate\nx,999\n", encoding="utf-8"
    )

    with pytest.raises(ExperimentPackageError, match="digest mismatch"):
        build_experiment_package(
            run_dir=run,
            package_root=tmp_path / "packages",
            experiment_id="FIG2-E2-DEV-001",
        )


def test_artifact_escape_is_rejected(tmp_path: Path):
    run = _run(tmp_path)
    outside = tmp_path / "outside.csv"
    outside.write_text("x\n1\n", encoding="utf-8")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest["evidence"][0]["relative_path"] = "../../../../outside.csv"
    manifest["evidence"][0]["sha256"] = _sha(outside)
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ExperimentPackageError, match="escapes"):
        build_experiment_package(
            run_dir=run,
            package_root=tmp_path / "packages",
            experiment_id="FIG2-E2-DEV-001",
        )


def test_checked_in_registry_has_unique_ids_and_code_bindings():
    registry_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "agent_experiments"
        / "registry.json"
    )
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    experiments = payload["experiments"]
    ids = [row["experiment_id"] for row in experiments]

    assert payload["schema_version"] == "easyicu.agent_experiment_registry/1"
    assert len(ids) == len(set(ids))
    assert all(_id == _id.upper() for _id in ids)
    assert all(len(row["code_commit"]) == 40 for row in experiments)
    assert all(
        str(row["local_package"]).startswith("research_output/_packages/")
        for row in experiments
    )
