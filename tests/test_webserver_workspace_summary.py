from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from easyicu.webserver.app import app
from easyicu.webserver import agent_runs
from easyicu.webserver.agent_runs import _scan_artifact_payloads
from easyicu.webserver import dataio
from easyicu.webserver import provider_adapter
from easyicu.webserver import provider_gate
from easyicu.webserver import settings as settings_store
from easyicu.webserver import sources as source_store
from easyicu.webserver.dataio import (
    describe_export_source,
    summarize_crossdb_workspaces,
    summarize_export_workspace,
)


SIGNOFF_CONFIRMATIONS = [
    "evidence_reviewed",
    "claims_remain_locked",
    "no_patient_rows_persisted",
]


@pytest.fixture(autouse=True)
def _disable_real_provider_env_file(monkeypatch) -> None:
    monkeypatch.setenv("EASYICU_DISABLE_PROVIDER_ENV_FILE", "1")


def _write_csv_export(root: Path, database: str = "miiv") -> Path:
    root.mkdir()
    tables = {
        "demographics": pd.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "age": [50, 70, 60],
                "sex": ["F", "M", "F"],
            }
        ),
        "outcome": pd.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "death": ["", "1", "0"],
                "los_icu": [2.0, 5.0, 1.0],
            }
        ),
        "sofa2_score": pd.DataFrame(
            {
                "stay_id": [1, 1, 2],
                "charttime": ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 00:00"],
                "sofa2": [4, 5, 8],
            }
        ),
        "sepsis3_sofa2": pd.DataFrame(
            {
                "stay_id": [1, 2],
                "sep3_sofa2": ["true", ""],
            }
        ),
        "vitals": pd.DataFrame(
            {
                "stay_id": [1, 1, 2],
                "charttime": ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 00:00"],
                "hr": [90, 95, 80],
                "map": [70, 72, 75],
                "spo2": [97, 98, 96],
                "temp": [37.0, 37.2, 36.8],
            }
        ),
    }
    manifest_files = []
    for module, frame in tables.items():
        file_name = f"{module}.csv"
        frame.to_csv(root / file_name, index=False)
        manifest_files.append({"file": file_name, "module": module, "rows": len(frame)})

    (root / "_manifest.json").write_text(
        json.dumps(
            {
                "database": database,
                "generated": "2026-06-23T12:00:00",
                "files": manifest_files,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return root


def _write_preview_trap_export(root: Path) -> Path:
    root.mkdir()
    stays = list(range(1, 14))
    tables = {
        "demographics": pd.DataFrame(
            {
                "stay_id": stays,
                "age": [50] * 12 + [90],
                "sex": ["F"] * 13,
            }
        ),
        "outcome": pd.DataFrame(
            {
                "stay_id": stays,
                "death": [0] * 12 + [1],
                "los_icu": [1.0] * 12 + [13.0],
            }
        ),
        "sofa2_score": pd.DataFrame(
            {
                "stay_id": stays,
                "sofa2": [2] * 12 + [14],
            }
        ),
    }
    manifest_files = []
    for module, frame in tables.items():
        file_name = f"{module}.csv"
        frame.to_csv(root / file_name, index=False)
        manifest_files.append({"file": file_name, "module": module, "rows": len(frame)})
    (root / "_manifest.json").write_text(
        json.dumps({"database": "miiv", "files": manifest_files}, indent=2),
        encoding="utf-8",
    )
    return root


def _drop_export_module(root: Path, module: str) -> None:
    target = root / f"{module}.csv"
    if target.exists():
        target.unlink()
    manifest_path = root / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        row for row in manifest.get("files", [])
        if row.get("module") != module and row.get("file") != f"{module}.csv"
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_summarize_export_workspace_builds_bounded_real_snapshot(tmp_path: Path) -> None:
    export_dir = _write_csv_export(tmp_path / "export")

    result = summarize_export_workspace(str(export_dir))

    assert result["ok"] is True
    assert result["database"] == "miiv"
    assert result["summary"]["stays"] == 3
    assert result["summary"]["modules"] == 5
    assert result["summary"]["total_rows"] == 14
    assert result["summary"]["mean_age"] == 60
    assert result["summary"]["female_pct"] == 66.7
    assert result["summary"]["mortality"] == 33.3
    assert result["summary"]["sepsis_pct"] == 33.3
    assert result["summary"]["median_sofa2"] == 6.5
    assert result["tableRows"] == [
        {"stay_id": "1", "age": 50.0, "sex": "F", "sofa2": 5.0, "los_icu": 2.0, "outcome": "Survived"},
        {"stay_id": "2", "age": 70.0, "sex": "M", "sofa2": 8.0, "los_icu": 5.0, "outcome": "Deceased"},
        {"stay_id": "3", "age": 60.0, "sex": "F", "sofa2": None, "los_icu": 1.0, "outcome": "Survived"},
    ]
    assert result["patient"]["stay_id"] == "1"
    assert result["patient"]["sepsis3"] is True
    assert [series["key"] for series in result["series"]] == ["hr", "map", "spo2", "temp"]
    assert result["series"][0]["values"] == [90.0, 95.0]
    quality = {row["module"]: row for row in result["quality"]}
    assert quality["vitals"]["unique_stays"] == 2
    assert quality["vitals"]["coverage_pct"] == 66.7
    assert quality["vitals"]["coverage_basis"] == "unique_stay_id_intersection"
    assert quality["sepsis3_sofa2"]["status"] == "neutral"


def test_cohort_summary_uses_full_loaded_cohort_not_preview_rows(tmp_path: Path) -> None:
    export_dir = _write_preview_trap_export(tmp_path / "export")

    result = summarize_export_workspace(str(export_dir))

    assert result["ok"] is True
    assert len(result["tableRows"]) == 12
    assert {row["outcome"] for row in result["tableRows"]} == {"Survived"}
    assert result["cohort"]["survived"] == 12
    assert result["cohort"]["deceased"] == 1
    assert result["cohort"]["characteristics"][0] == ["Age, mean", 53.08, 50.0, 90.0]
    assert result["cohort"]["characteristics"][1] == ["SOFA-2, mean", 2.92, 2.0, 14.0]
    assert result["cohort"]["characteristics"][2] == ["ICU LOS, mean", 1.92, 1.0, 13.0]


def test_describe_export_source_uses_manifest_and_schema_without_full_table_reads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    export_dir = _write_csv_export(tmp_path / "export")

    def fail_count_rows(path: Path) -> int:
        raise AssertionError(f"row count should come from manifest: {path}")

    def fail_read_frame(path: Path):
        raise AssertionError(f"registry describe must not read full frame: {path}")

    monkeypatch.setattr(dataio, "_count_rows", fail_count_rows)
    monkeypatch.setattr(dataio, "_read_export_frame", fail_read_frame)

    result = describe_export_source(str(export_dir))

    assert result["ok"] is True
    assert result["summary"] == {"stays": 3, "modules": 5, "file_count": 5, "total_rows": 14}
    assert result["files"][0]["columns"]


def test_workspace_summary_endpoint_returns_snapshot_and_rejects_bad_paths(tmp_path: Path) -> None:
    export_dir = _write_csv_export(tmp_path / "export")
    client = TestClient(app)

    ok = client.post("/api/workspace/summary", json={"path": str(export_dir)})
    bad = client.post("/api/workspace/summary", json={"path": str(tmp_path / "missing")})

    assert ok.status_code == 200
    assert ok.json()["summary"]["stays"] == 3
    assert ok.json()["series"][0]["key"] == "hr"
    assert bad.status_code == 400
    assert bad.json()["detail"]["error"] == "not_a_directory"


def test_crossdb_summary_requires_two_valid_exports_and_compares_metrics(tmp_path: Path) -> None:
    miiv = _write_csv_export(tmp_path / "miiv", database="miiv")
    eicu = _write_csv_export(tmp_path / "eicu", database="eicu")

    result = summarize_crossdb_workspaces([str(miiv), str(eicu)])

    assert result["ok"] is True
    assert result["source_count"] == 2
    assert [source["label"] for source in result["sources"]] == ["MIIV", "EICU"]
    assert result["shared_modules"] == ["demographics", "outcome", "sepsis3_sofa2", "sofa2_score", "vitals"]
    assert result["compatibility_gate"]["status"] == "compatible"
    assert result["compatibility_gate"]["comparison_mode"] == "descriptive_only"
    assert result["compatibility_gate"]["matched_cohort"] is False
    assert result["compatibility_gate"]["inferential_statistics_allowed"] is False
    assert result["rows"][0] == {
        "key": "stays",
        "label": "Stays",
        "values": [3, 3],
        "delta": 0.0,
        "comparison": "descriptive_range",
    }
    assert result["rows"][5]["label"] == "Mortality %"
    assert result["rows"][5]["values"] == [33.3, 33.3]
    assert "p_value" not in json.dumps(result).lower()


def test_crossdb_summary_fails_closed_when_core_modules_are_not_comparable(tmp_path: Path) -> None:
    miiv = _write_csv_export(tmp_path / "miiv", database="miiv")
    eicu = _write_csv_export(tmp_path / "eicu", database="eicu")
    _drop_export_module(eicu, "outcome")

    result = summarize_crossdb_workspaces([str(miiv), str(eicu)])

    assert result["ok"] is False
    assert result["error"] == "crossdb_incompatible"
    gate = result["compatibility_gate"]
    assert gate["status"] == "incompatible"
    assert gate["comparison_mode"] == "descriptive_only"
    assert gate["matched_cohort_ready"] is False
    assert gate["inferential_statistics_allowed"] is False
    core_check = next(check for check in gate["checks"] if check["id"] == "core_modules_shared")
    assert core_check["passed"] is False
    assert core_check["missing_modules"] == ["outcome"]


def test_crossdb_summary_endpoint_is_fail_closed_until_two_exports(tmp_path: Path) -> None:
    miiv = _write_csv_export(tmp_path / "miiv", database="miiv")
    eicu = _write_csv_export(tmp_path / "eicu", database="eicu")
    no_outcome = _write_csv_export(tmp_path / "no_outcome", database="aumc")
    _drop_export_module(no_outcome, "outcome")
    client = TestClient(app)

    one = client.post("/api/workspaces/crossdb-summary", json={"paths": [str(miiv)]})
    two = client.post("/api/workspaces/crossdb-summary", json={"paths": [str(miiv), str(eicu)]})
    invalid = client.post("/api/workspaces/crossdb-summary", json={"paths": [str(miiv), str(tmp_path / "missing")]})
    incompatible = client.post("/api/workspaces/crossdb-summary", json={"paths": [str(miiv), str(no_outcome)]})

    assert one.status_code == 400
    assert one.json()["detail"]["error"] == "need_two_exports"
    assert two.status_code == 200
    assert two.json()["source_count"] == 2
    assert invalid.status_code == 400
    assert invalid.json()["detail"]["error"] == "invalid_export"
    assert incompatible.status_code == 400
    assert incompatible.json()["detail"]["error"] == "crossdb_incompatible"
    assert incompatible.json()["detail"]["compatibility_gate"]["status"] == "incompatible"


def test_export_source_registry_describes_and_persists_sources(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    miiv = _write_csv_export(tmp_path / "miiv", database="miiv")
    eicu = _write_csv_export(tmp_path / "eicu", database="eicu")

    desc = describe_export_source(str(miiv))
    assert desc["ok"] is True
    assert desc["summary"] == {"stays": 3, "modules": 5, "file_count": 5, "total_rows": 14}

    registered = source_store.register_source(str(miiv), active=True, crossdb=True)
    saved = source_store.save_registry({"sources": [{"path": str(eicu), "label": "External eICU"}], "crossdb_paths": [str(miiv), str(eicu)]})

    assert registered["ok"] is True
    assert saved["active_path"] == str(miiv)
    assert saved["crossdb_paths"] == [str(miiv), str(eicu)]
    assert [s["label"] for s in saved["sources"]] == ["External eICU", "MIIV"]


def test_export_source_registry_endpoint_registers_and_rejects_invalid_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    miiv = _write_csv_export(tmp_path / "miiv", database="miiv")
    client = TestClient(app)

    ok = client.post("/api/workspaces/register", json={"path": str(miiv)})
    bad = client.post("/api/workspaces/register", json={"path": str(tmp_path / "missing")})
    reg = client.get("/api/workspaces/registry")

    assert ok.status_code == 200
    assert ok.json()["active_path"] == str(miiv)
    assert bad.status_code == 400
    assert bad.json()["detail"]["error"] == "not_a_directory"
    assert reg.status_code == 200
    assert reg.json()["sources"][0]["summary"]["stays"] == 3


def test_export_source_registry_rename_and_remove_are_metadata_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    miiv = _write_csv_export(tmp_path / "miiv", database="miiv")
    eicu = _write_csv_export(tmp_path / "eicu", database="eicu")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [str(miiv), str(eicu)])

    renamed = source_store.rename_source(str(miiv), "Primary MIIV")
    removed = source_store.remove_source(str(miiv))
    reloaded = source_store.load_registry()

    assert renamed["ok"] is True
    assert renamed["action"] == "renamed_source_metadata"
    assert renamed["disk_touched"] is False
    assert any(s["path"] == str(miiv) and s["label"] == "Primary MIIV" for s in renamed["sources"])
    assert removed["ok"] is True
    assert removed["action"] == "unregistered_source_only"
    assert removed["disk_deleted"] is False
    assert removed["removed_path"] == str(miiv)
    assert miiv.exists()
    assert (miiv / "demographics.csv").exists()
    assert all(s["path"] != str(miiv) for s in reloaded["sources"])
    assert reloaded["active_path"] == str(eicu)

    restored = source_store.register_source(str(miiv), label="Restored MIIV", active=True, crossdb=True)

    assert restored["active_path"] == str(miiv)
    assert any(s["path"] == str(miiv) and s["label"] == "Restored MIIV" for s in restored["sources"])


def test_export_source_registry_endpoint_renames_and_removes_without_deleting_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    miiv = _write_csv_export(tmp_path / "miiv", database="miiv")
    eicu = _write_csv_export(tmp_path / "eicu", database="eicu")
    client = TestClient(app)

    client.post("/api/workspaces/register", json={"path": str(miiv)})
    client.post("/api/workspaces/register", json={"path": str(eicu)})
    rename = client.post("/api/workspaces/rename", json={"path": str(miiv), "label": "Primary source"})
    remove = client.post("/api/workspaces/remove", json={"path": str(miiv)})
    missing = client.post("/api/workspaces/remove", json={"path": str(tmp_path / "missing")})

    assert rename.status_code == 200
    assert any(s["path"] == str(miiv) and s["label"] == "Primary source" for s in rename.json()["sources"])
    assert remove.status_code == 200
    assert remove.json()["disk_deleted"] is False
    assert miiv.exists()
    assert (miiv / "outcome.csv").exists()
    assert all(s["path"] != str(miiv) for s in remove.json()["sources"])
    assert remove.json()["active_path"] == str(eicu)
    assert str(miiv) not in remove.json()["crossdb_paths"]
    assert missing.status_code == 400
    assert missing.json()["detail"]["error"] == "source_not_registered"


def test_agent_run_job_uses_active_registry_and_writes_bounded_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    client = TestClient(app)

    start = client.post(
        "/api/jobs/agent-run",
        json={
            "study_id": "sepsis",
            "mode": "analysis",
            "question": "cohort summary only",
            "project_root": str(tmp_path / "projects"),
        },
    )

    assert start.status_code == 200
    job_id = start.json()["job_id"]
    snapshot = _wait_for_job(client, job_id)
    assert snapshot["status"] == "done"
    assert [event["type"] for event in snapshot["events"] if event["type"] != "end"] == [
        "start",
        "progress",
        "progress",
        "gate",
        "artifact",
    ]
    result = snapshot["result"]
    assert result["summary"]["stays"] == 3
    assert result["gate"]["status"] == "analysis_only"
    assert result["gate"]["reportable"] is False
    assert result["uploads"] == 0
    assert result["tokens"] == 0
    gate_checks = {check["id"]: check for check in result["gate"]["checks"]}
    assert gate_checks["source_valid"]["passed"] is True
    assert gate_checks["source_valid"]["evidence"] == "registry description"
    assert gate_checks["no_patient_rows_persisted"]["passed"] is True
    assert gate_checks["no_patient_rows_persisted"]["evidence"] == "artifact_json_scan"
    assert gate_checks["no_patient_rows_persisted"]["scanned_artifacts"] == 4
    assert gate_checks["no_patient_rows_persisted"]["row_level_markers"] == []
    artifact_paths = [Path(item["path"]) for item in result["artifacts"]]
    assert {path.name for path in artifact_paths} == {
        "run_context.json",
        "cohort_summary.json",
        "quality_gate.json",
        "evidence_ledger.json",
    }
    for path in artifact_paths:
        text = path.read_text(encoding="utf-8")
        assert "tableRows" not in text
        assert '"series"' not in text
        assert '"patient"' not in text
    ledger = json.loads((Path(result["project_dir"]) / "evidence_ledger.json").read_text(encoding="utf-8"))
    assert ledger["privacy"]["patient_rows_persisted"] is False
    assert ledger["privacy"]["artifact_scan"]["passed"] is True
    assert ledger["privacy"]["artifact_scan"]["scanned_artifacts"] == 4


def test_agent_run_review_and_local_signoff_write_safe_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    client = TestClient(app)

    start = client.post(
        "/api/jobs/agent-run",
        json={
            "study_id": "sepsis",
            "mode": "analysis",
            "question": "cohort summary only",
            "project_root": str(tmp_path / "projects"),
        },
    )
    assert start.status_code == 200
    snapshot = _wait_for_job(client, start.json()["job_id"])
    assert snapshot["status"] == "done"
    run_dir = Path(snapshot["result"]["project_dir"])

    review = client.post("/api/agent-runs/review", json={"project_dir": str(run_dir)})
    assert review.status_code == 200
    review_payload = review.json()
    assert review_payload["signed"] is False
    assert review_payload["readiness"]["status"] == "awaiting_human_signoff"
    assert review_payload["readiness"]["signable"] is True
    assert review_payload["readiness"]["reportable"] is False
    assert review_payload["readiness"]["draft_unlocked"] is False
    assert "human_signoff.json" not in {item["name"] for item in review_payload["artifacts"]}

    signoff = client.post(
        "/api/agent-runs/signoff",
        json={
            "project_dir": str(run_dir),
            "reviewer": "local reviewer",
            "confirmations": SIGNOFF_CONFIRMATIONS,
            "note": "reviewed locally",
        },
    )

    assert signoff.status_code == 200
    signed_payload = signoff.json()
    assert signed_payload["signed"] is True
    assert signed_payload["readiness"]["status"] == "signed_analysis_only"
    assert signed_payload["readiness"]["reportable"] is False
    assert signed_payload["readiness"]["draft_unlocked"] is False
    assert "human_signoff.json" in {item["name"] for item in signed_payload["artifacts"]}

    signoff_path = run_dir / "human_signoff.json"
    assert signoff_path.exists()
    signoff_payload = json.loads(signoff_path.read_text(encoding="utf-8"))
    assert signoff_payload["status"] == "signed_analysis_only"
    assert signoff_payload["reportable"] is False
    assert signoff_payload["draft_unlocked"] is False
    assert signoff_payload["uploads"] == 0
    assert signoff_payload["tokens"] == 0
    assert signoff_payload["external_calls"] == 0
    signed_artifacts = {item["name"]: item for item in signoff_payload["signed_artifacts"]}
    assert set(signed_artifacts) == {
        "run_context.json",
        "cohort_summary.json",
        "quality_gate.json",
        "evidence_ledger.json",
    }
    for item in signed_artifacts.values():
        assert len(item["sha256"]) == 64
        assert item["bytes"] > 0
    assert _scan_artifact_payloads({"human_signoff.json": signoff_payload})["passed"] is True
    text = signoff_path.read_text(encoding="utf-8")
    assert "tableRows" not in text
    assert '"stay_id"' not in text
    assert '"patient"' not in text

    gate = json.loads((run_dir / "quality_gate.json").read_text(encoding="utf-8"))["gate"]
    gate_checks = {check["id"]: check for check in gate["checks"]}
    assert gate_checks["human_signoff"]["passed"] is False
    assert gate["reportable"] is False
    assert gate["draft_unlocked"] is False

    history = client.post(
        "/api/agent-runs/history",
        json={"project_root": str(tmp_path / "projects"), "study_id": "sepsis"},
    )
    assert history.status_code == 200
    history_payload = history.json()
    assert history_payload["count"] == 1
    assert history_payload["runs"][0]["signed"] is True
    assert history_payload["runs"][0]["readiness_status"] == "signed_analysis_only"
    assert history_payload["runs"][0]["integrity_status"] == "verified"
    assert history_payload["runs"][0]["signoff_stale"] is False

    artifact = client.post(
        "/api/agent-runs/artifact",
        json={"project_dir": str(run_dir), "artifact": "quality_gate.json"},
    )
    assert artifact.status_code == 200
    artifact_payload = artifact.json()
    assert artifact_payload["artifact"]["name"] == "quality_gate.json"
    assert artifact_payload["payload"]["gate"]["status"] == "analysis_only"
    assert artifact_payload["privacy_scan"]["passed"] is True
    assert artifact_payload["privacy_scan"]["row_level_markers"] == []

    rejected_artifact = client.post(
        "/api/agent-runs/artifact",
        json={"project_dir": str(run_dir), "artifact": "../quality_gate.json"},
    )
    assert rejected_artifact.status_code == 400
    assert rejected_artifact.json()["detail"]["error"] == "artifact_not_allowed"

    single_download = client.post(
        "/api/agent-runs/download-artifact",
        json={"project_dir": str(run_dir), "artifact": "evidence_ledger.json"},
    )
    assert single_download.status_code == 200
    assert b'"privacy"' in single_download.content
    assert "evidence_ledger.json" in single_download.headers["content-disposition"]

    bundle = client.post(
        "/api/agent-runs/download-bundle",
        json={"project_dir": str(run_dir)},
    )
    assert bundle.status_code == 200
    with zipfile.ZipFile(io.BytesIO(bundle.content)) as zf:
        assert set(zf.namelist()) == {
            "run_context.json",
            "cohort_summary.json",
            "quality_gate.json",
            "evidence_ledger.json",
            "human_signoff.json",
        }

    cohort_path = run_dir / "cohort_summary.json"
    cohort_payload = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort_payload["summary"]["stays"] = 999
    cohort_path.write_text(json.dumps(cohort_payload, indent=2), encoding="utf-8")

    stale_review = client.post("/api/agent-runs/review", json={"project_dir": str(run_dir)})
    assert stale_review.status_code == 200
    stale_payload = stale_review.json()
    assert stale_payload["signed"] is True
    assert stale_payload["signoff_stale"] is True
    assert stale_payload["readiness"]["status"] == "signoff_stale"
    assert stale_payload["readiness"]["reportable"] is False
    assert stale_payload["readiness"]["draft_unlocked"] is False
    assert stale_payload["signoff_integrity"]["status"] == "stale"
    assert stale_payload["signoff_integrity"]["tampered_artifacts"][0]["name"] == "cohort_summary.json"


def test_agent_run_signoff_requires_all_confirmations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    client = TestClient(app)

    start = client.post(
        "/api/jobs/agent-run",
        json={"study_id": "sepsis", "project_root": str(tmp_path / "projects")},
    )
    assert start.status_code == 200
    snapshot = _wait_for_job(client, start.json()["job_id"])
    run_dir = Path(snapshot["result"]["project_dir"])

    response = client.post(
        "/api/agent-runs/signoff",
        json={
            "project_dir": str(run_dir),
            "confirmations": ["evidence_reviewed"],
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"] == "missing_signoff_confirmations"
    assert set(detail["missing_confirmations"]) == {
        "claims_remain_locked",
        "no_patient_rows_persisted",
    }
    assert not (run_dir / "human_signoff.json").exists()


def test_agent_artifact_privacy_scan_flags_row_level_payloads() -> None:
    scan = _scan_artifact_payloads(
        {
            "run_context.json": {
                "summary": {"stays": 3},
                "tableRows": [{"stay_id": 1, "age": 50}],
            },
            "cohort_summary.json": {"cohort": {"survived": 2, "deceased": 1}},
            "quality_gate.json": {"quality": [{"module": "demo"}]},
        }
    )

    assert scan["passed"] is False
    assert scan["scanned_artifacts"] == 3
    markers = {(hit["path"], hit["marker"]) for hit in scan["row_level_markers"]}
    assert ("run_context.json.tableRows", "tableRows") in markers
    assert ("run_context.json.tableRows[0].stay_id", "stay_id") in markers


def test_full_agent_mock_run_writes_locked_strict_evidence_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    monkeypatch.setattr(settings_store, "load_settings", lambda: {"ai_enabled": False})
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    client = TestClient(app)

    start = client.post(
        "/api/jobs/agent-run",
        json={
            "study_id": "sepsis",
            "mode": "analysis",
            "run_type": "full",
            "llm_provider": "mock",
            "question": "draft only if evidence-bound",
            "project_root": str(tmp_path / "projects"),
        },
    )

    assert start.status_code == 200
    snapshot = _wait_for_job(client, start.json()["job_id"])
    assert snapshot["status"] == "done"
    assert [event["type"] for event in snapshot["events"] if event["type"] != "end"] == [
        "start",
        "progress",
        "progress",
        "progress",
        "gate",
        "artifact",
    ]
    result = snapshot["result"]
    assert result["run_type"] == "full"
    assert result["provider"]["client"] == "MockLLMClient"
    assert result["provider"]["external"] is False
    assert result["provider"]["credentials_loaded"] is False
    assert result["provider"]["credentials_attempted"] is False
    assert result["provider"]["client_constructed"] is False
    assert result["provider"]["provider_gate"] == "offline_mock"
    assert result["provider"]["canonical_opt_in_source"] == provider_gate.CANONICAL_OPT_IN_SOURCE
    assert result["provider"]["canonical_opt_in_passed"] is True
    assert result["provider"]["mock_calls"] == 1
    assert result["gate"]["status"] == "analysis_only"
    assert result["gate"]["reportable"] is False
    assert result["gate"]["draft_unlocked"] is False
    checks = {check["id"]: check for check in result["gate"]["checks"]}
    assert checks["provider_opt_in"]["passed"] is True
    assert checks["strict_evidence_bound_claims"]["passed"] is True
    assert checks["strict_evidence_bound_sentences"]["passed"] is True
    assert checks["human_signoff"]["passed"] is False
    assert checks["no_patient_rows_persisted"]["scanned_artifacts"] == 6

    artifact_paths = {Path(item["path"]).name: Path(item["path"]) for item in result["artifacts"]}
    assert set(artifact_paths) == {
        "run_context.json",
        "cohort_summary.json",
        "quality_gate.json",
        "agent_plan.json",
        "manuscript_draft.json",
        "evidence_ledger.json",
    }
    draft = json.loads(artifact_paths["manuscript_draft.json"].read_text(encoding="utf-8"))
    assert draft["status"] == "locked_until_human_signoff"
    assert all(row.get("evidence_ids") for row in draft["claims"])
    assert all(row.get("evidence_ids") for row in draft["sentences"])
    ledger = json.loads(artifact_paths["evidence_ledger.json"].read_text(encoding="utf-8"))
    assert ledger["strict_evidence_audit"]["claims_passed"] is True
    assert ledger["privacy"]["artifact_scan"]["scanned_artifacts"] == 6


def test_full_agent_external_provider_requires_explicit_opt_in(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    monkeypatch.setattr(settings_store, "load_settings", lambda: {"ai_enabled": False})
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    client = TestClient(app)

    response = client.post(
        "/api/jobs/agent-run",
        json={"run_type": "full", "llm_provider": "openai", "external_llm_opt_in": False},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"] == "external_llm_opt_in_required"
    assert detail["provider"] == "openai"
    assert detail["ai_enabled"] is False
    assert detail["per_run_opt_in"] is False
    assert detail["provider_gate"] == "blocked_before_client_construction"
    assert detail["blocked_by"] == "canonical_ai_opt_in"
    assert detail["canonical_opt_in_source"] == provider_gate.CANONICAL_OPT_IN_SOURCE
    assert detail["canonical_opt_in_passed"] is False
    assert detail["credentials_loaded"] is False
    assert detail["credentials_attempted"] is False
    assert detail["client_constructed"] is False


def test_full_agent_external_provider_requires_per_run_opt_in_after_canonical_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    monkeypatch.setattr(settings_store, "load_settings", lambda: {"ai_enabled": True})
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    client = TestClient(app)

    response = client.post(
        "/api/jobs/agent-run",
        json={"run_type": "full", "llm_provider": "openai", "external_llm_opt_in": False},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"] == "external_llm_opt_in_required"
    assert detail["blocked_by"] == "per_run_external_llm_opt_in"
    assert detail["canonical_opt_in_source"] == provider_gate.CANONICAL_OPT_IN_SOURCE
    assert detail["canonical_opt_in_passed"] is True
    assert detail["ai_enabled"] is True
    assert detail["per_run_opt_in"] is False
    assert detail["credentials_loaded"] is False
    assert detail["credentials_attempted"] is False
    assert detail["client_constructed"] is False


def test_agent_provider_status_reports_readiness_without_secret(monkeypatch) -> None:
    monkeypatch.setattr(settings_store, "load_settings", lambda: {"ai_enabled": True})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-provider-status")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9999/v1")
    client = TestClient(app)

    response = client.get("/api/agent-runs/provider-status?provider=openai")

    assert response.status_code == 200
    payload = response.json()
    status = payload["provider_status"]
    assert status["ready"] is True
    assert status["credential_present"] is True
    assert status["credential_source"] == "OPENAI_API_KEY"
    assert status["model_present"] is True
    assert status["model_source"] == "OPENAI_MODEL"
    assert status["base_url_present"] is True
    assert status["base_url_source"] == "OPENAI_BASE_URL"
    assert status["secrets_returned"] is False
    assert status["client_constructed"] is False
    assert status["network_calls"] == 0
    assert status["limits"]["max_external_calls_per_run"] == 1
    assert status["limits"]["max_output_tokens"] == 1200
    serialized = json.dumps(payload)
    assert "sk-test-provider-status" not in serialized
    assert "http://127.0.0.1:9999/v1" not in serialized


def test_agent_provider_status_blocks_when_ai_or_env_missing(monkeypatch) -> None:
    monkeypatch.setattr(settings_store, "load_settings", lambda: {"ai_enabled": False})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EASYICU_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("EASYICU_LLM_MODEL", raising=False)
    client = TestClient(app)

    response = client.get("/api/agent-runs/provider-status?provider=openai")

    assert response.status_code == 200
    status = response.json()["provider_status"]
    assert status["ready"] is False
    assert status["credential_present"] is False
    assert status["model_present"] is False
    assert set(status["missing"]) == {"ai_enabled", "credential", "model"}
    assert status["client_constructed"] is False
    assert status["network_calls"] == 0


def test_agent_provider_status_reads_private_env_file_without_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("EASYICU_DISABLE_PROVIDER_ENV_FILE", raising=False)
    monkeypatch.setattr(settings_store, "load_settings", lambda: {"ai_enabled": True})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    env_path = tmp_path / "provider.env"
    env_path.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=sk-test-private-env",
                "OPENAI_BASE_URL=http://127.0.0.1:8787/v1",
                "OPENAI_MODEL=gpt-private-test",
                "EASYICU_LLM_MAX_TOKENS=640",
            ]
        ),
        encoding="utf-8",
    )
    env_path.chmod(0o600)
    monkeypatch.setenv("EASYICU_LLM_ENV_FILE", str(env_path))
    client = TestClient(app)

    response = client.get("/api/agent-runs/provider-status?provider=openai")

    assert response.status_code == 200
    payload = response.json()
    status = payload["provider_status"]
    assert status["ready"] is True
    assert status["credential_present"] is True
    assert status["credential_source"] == "OPENAI_API_KEY"
    assert status["base_url_present"] is True
    assert status["base_url_source"] == "OPENAI_BASE_URL"
    assert status["model_present"] is True
    assert status["model_source"] == "OPENAI_MODEL"
    assert status["limits"]["max_output_tokens"] == 640
    assert status["env_file"]["status"] == "loaded"
    assert set(status["env_file"]["loaded_keys"]) == {
        "EASYICU_LLM_MAX_TOKENS",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
    }
    serialized = json.dumps(payload)
    assert "sk-test-private-env" not in serialized
    assert "http://127.0.0.1:8787/v1" not in serialized
    assert "gpt-private-test" not in serialized


def test_agent_provider_status_rejects_insecure_private_env_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("EASYICU_DISABLE_PROVIDER_ENV_FILE", raising=False)
    monkeypatch.setattr(settings_store, "load_settings", lambda: {"ai_enabled": True})
    env_path = tmp_path / "provider.env"
    env_path.write_text(
        "OPENAI_API_KEY=sk-test-insecure-env\nOPENAI_MODEL=gpt-insecure\n",
        encoding="utf-8",
    )
    env_path.chmod(0o644)
    monkeypatch.setenv("EASYICU_LLM_ENV_FILE", str(env_path))
    client = TestClient(app)

    response = client.get("/api/agent-runs/provider-status?provider=openai")

    assert response.status_code == 200
    payload = response.json()
    status = payload["provider_status"]
    assert status["ready"] is False
    assert status["credential_present"] is False
    assert status["model_present"] is False
    assert status["env_file"]["status"] == "insecure_permissions"
    assert "env_file_permissions" in status["missing"]
    serialized = json.dumps(payload)
    assert "sk-test-insecure-env" not in serialized
    assert "gpt-insecure" not in serialized


def test_provider_gate_calls_canonical_opt_in_before_per_run_and_credentials(monkeypatch) -> None:
    calls = []

    def fake_check(choice, *, ai_enabled, language):
        calls.append((choice, ai_enabled, language))
        raise provider_gate.AIOptInError("canonical blocked")

    monkeypatch.setattr(provider_gate, "check_external_llm_opt_in", fake_check)

    try:
        agent_runs.validate_agent_run_config(
            run_type="full",
            llm_provider="openai",
            external_llm_opt_in=True,
            ai_enabled=False,
        )
    except agent_runs.AgentRunConfigError as exc:
        detail = exc.detail
    else:
        raise AssertionError("external provider should fail closed")

    assert calls == [("openai", False, "en")]
    assert detail["error"] == "external_llm_opt_in_required"
    assert detail["blocked_by"] == "canonical_ai_opt_in"
    assert detail["credentials_loaded"] is False
    assert detail["credentials_attempted"] is False
    assert detail["client_constructed"] is False


def test_full_agent_external_provider_requires_credentials_after_opt_in(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    monkeypatch.setattr(settings_store, "load_settings", lambda: {"ai_enabled": True})
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    client = TestClient(app)

    response = client.post(
        "/api/jobs/agent-run",
        json={"run_type": "full", "llm_provider": "openai", "external_llm_opt_in": True},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"] == "external_provider_credentials_required"
    assert detail["blocked_by"] == "external_provider_credentials"
    assert detail["provider_gate"] == "credential_lookup_allowed"
    assert detail["canonical_opt_in_passed"] is True
    assert detail["credentials_loaded"] is False
    assert detail["credentials_attempted"] is True
    assert detail["client_constructed"] is False


def test_full_agent_external_provider_uses_adapter_after_opt_in_and_credentials(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    monkeypatch.setattr(settings_store, "load_settings", lambda: {"ai_enabled": True})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-stage9")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("EASYICU_LLM_MAX_TOKENS", "768")
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    captured = {}

    def fake_post(*, url, request, headers, timeout):
        captured["url"] = url
        captured["request"] = request
        captured["headers"] = headers
        captured["timeout"] = timeout
        request_text = json.dumps(request, ensure_ascii=False)
        assert "tableRows" not in request_text
        assert '"stay_id"' not in request_text
        assert headers["Authorization"] == "Bearer sk-test-stage9"
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "agent_plan": {
                                    "steps": [
                                        {
                                            "id": "snapshot",
                                            "title": "Review aggregate snapshot",
                                            "evidence_ids": ["run_context.json"],
                                        }
                                    ]
                                },
                                "manuscript_draft": {
                                    "claims": [
                                        {
                                            "id": "c1",
                                            "text": "The export snapshot contains three stays.",
                                            "evidence_ids": ["run_context.json"],
                                        }
                                    ],
                                    "sentences": [
                                        {
                                            "id": "s1",
                                            "text": "This locked analysis-only draft is bound to local artifacts.",
                                            "evidence_ids": ["run_context.json", "quality_gate.json"],
                                        }
                                    ],
                                },
                            }
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }

    monkeypatch.setattr(provider_adapter, "_post_chat_completion", fake_post)
    client = TestClient(app)

    start = client.post(
        "/api/jobs/agent-run",
        json={
            "run_type": "full",
            "llm_provider": "openai",
            "external_llm_opt_in": True,
            "project_root": str(tmp_path / "projects"),
        },
    )

    assert start.status_code == 200
    snapshot = _wait_for_job(client, start.json()["job_id"])
    assert snapshot["status"] == "done"
    result = snapshot["result"]
    assert result["provider"]["external"] is True
    assert result["provider"]["provider_gate"] == "external_provider_ready"
    assert result["provider"]["credentials_loaded"] is True
    assert result["provider"]["credentials_attempted"] is True
    assert result["provider"]["client_constructed"] is True
    assert result["provider"]["external_calls"] == 1
    assert result["provider"]["max_external_calls_per_run"] == 1
    assert result["provider"]["max_output_tokens"] == 768
    assert result["provider"]["mock_calls"] == 0
    assert result["provider"]["credential_source"] == "OPENAI_API_KEY"
    assert result["provider"]["credential_fingerprint"] != "sk-test-stage9"
    assert result["provider"]["base_url_source"] == "OPENAI_BASE_URL"
    assert result["provider"]["base_url_endpoint"] == "chat_completions"
    assert "base_url" not in result["provider"]
    assert result["provider"]["usage"]["total_tokens"] == 18
    assert result["gate"]["status"] == "analysis_only"
    checks = {check["id"]: check for check in result["gate"]["checks"]}
    assert checks["provider_opt_in"]["passed"] is True
    assert checks["provider_opt_in"]["evidence"] == "external_provider_adapter_after_opt_in"
    assert checks["strict_evidence_bound_claims"]["passed"] is True
    assert checks["strict_evidence_bound_sentences"]["passed"] is True
    assert captured["url"] == "http://127.0.0.1:9999/v1/chat/completions"
    assert captured["request"]["model"] == "test-model"
    assert captured["request"]["max_tokens"] == 768
    assert captured["request"]["easyicu_policy"]["patient_rows_excluded"] is True
    assert captured["request"]["easyicu_policy"]["max_external_calls_per_run"] == 1
    assert captured["request"]["easyicu_policy"]["max_output_tokens"] == 768
    artifact_paths = {Path(item["path"]).name: Path(item["path"]) for item in result["artifacts"]}
    draft = json.loads(artifact_paths["manuscript_draft.json"].read_text(encoding="utf-8"))
    assert draft["status"] == "locked_until_human_signoff"
    assert "http://127.0.0.1:9999/v1" not in json.dumps(result, ensure_ascii=False)


def test_full_agent_external_provider_strict_gate_blocks_unbound_claims(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    monkeypatch.setattr(settings_store, "load_settings", lambda: {"ai_enabled": True})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-stage9")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(export_dir), active=True, crossdb=True)

    def fake_post(*, url, request, headers, timeout):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "agent_plan": {"steps": []},
                                "manuscript_draft": {
                                    "claims": [
                                        {"id": "c1", "text": "Unbound claim", "evidence_ids": []}
                                    ],
                                    "sentences": [
                                        {
                                            "id": "s1",
                                            "text": "Ghost evidence sentence.",
                                            "evidence_ids": ["ghost.json"],
                                        }
                                    ],
                                },
                            }
                        )
                    }
                }
            ],
            "usage": {"total_tokens": 12},
        }

    monkeypatch.setattr(provider_adapter, "_post_chat_completion", fake_post)
    client = TestClient(app)

    start = client.post(
        "/api/jobs/agent-run",
        json={
            "run_type": "full",
            "llm_provider": "openai",
            "external_llm_opt_in": True,
            "project_root": str(tmp_path / "projects"),
        },
    )

    assert start.status_code == 200
    snapshot = _wait_for_job(client, start.json()["job_id"])
    assert snapshot["status"] == "done"
    result = snapshot["result"]
    assert result["provider"]["external_calls"] == 1
    gate = result["gate"]
    assert gate["status"] == "blocked"
    assert gate["reason"] == "strict_evidence_gate_failed"
    audit = result["strict_evidence_audit"]
    assert audit["claims_passed"] is False
    assert audit["sentences_passed"] is False
    assert audit["unbound_claims"] == ["Unbound claim"]
    assert {"owner": "sentence", "evidence_id": "ghost.json"} in audit["missing_evidence"]
    checks = {check["id"]: check for check in gate["checks"]}
    assert checks["provider_opt_in"]["passed"] is True
    assert checks["strict_evidence_bound_claims"]["passed"] is False
    assert checks["strict_evidence_bound_sentences"]["passed"] is False


def test_full_agent_strict_gate_blocks_unbound_mock_claim(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    monkeypatch.setattr(settings_store, "load_settings", lambda: {"ai_enabled": False})
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    original_payload = agent_runs._mock_full_agent_payload

    def unbound_payload(**kwargs):
        payload = original_payload(**kwargs)
        payload["manuscript_draft"]["claims"][0]["evidence_ids"] = []
        payload["manuscript_draft"]["sentences"][0]["evidence_ids"] = ["missing.json"]
        return payload

    monkeypatch.setattr(agent_runs, "_mock_full_agent_payload", unbound_payload)
    client = TestClient(app)

    start = client.post(
        "/api/jobs/agent-run",
        json={
            "run_type": "full",
            "llm_provider": "mock",
            "project_root": str(tmp_path / "projects"),
        },
    )

    assert start.status_code == 200
    snapshot = _wait_for_job(client, start.json()["job_id"])
    assert snapshot["status"] == "done"
    gate = snapshot["result"]["gate"]
    assert gate["status"] == "blocked"
    assert gate["reportable"] is False
    assert gate["draft_unlocked"] is False
    checks = {check["id"]: check for check in gate["checks"]}
    assert checks["strict_evidence_bound_claims"]["passed"] is False
    assert checks["strict_evidence_bound_sentences"]["passed"] is False
    assert checks["strict_evidence_bound_claims"]["unbound_claims"] == ["claim_001"]
    assert checks["strict_evidence_bound_sentences"]["missing_evidence"] == [
        {"owner": "sent_001", "evidence_id": "missing.json"}
    ]


def test_agent_run_signoff_rejects_blocked_strict_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    monkeypatch.setattr(settings_store, "load_settings", lambda: {"ai_enabled": False})
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    original_payload = agent_runs._mock_full_agent_payload

    def unbound_payload(**kwargs):
        payload = original_payload(**kwargs)
        payload["manuscript_draft"]["claims"][0]["evidence_ids"] = []
        return payload

    monkeypatch.setattr(agent_runs, "_mock_full_agent_payload", unbound_payload)
    client = TestClient(app)

    start = client.post(
        "/api/jobs/agent-run",
        json={
            "run_type": "full",
            "llm_provider": "mock",
            "project_root": str(tmp_path / "projects"),
        },
    )
    assert start.status_code == 200
    snapshot = _wait_for_job(client, start.json()["job_id"])
    run_dir = Path(snapshot["result"]["project_dir"])
    assert snapshot["result"]["gate"]["status"] == "blocked"

    response = client.post(
        "/api/agent-runs/signoff",
        json={
            "project_dir": str(run_dir),
            "confirmations": SIGNOFF_CONFIRMATIONS,
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"] == "readiness_gate_not_signable"
    assert detail["readiness"]["status"] == "blocked"
    assert "strict_evidence_bound_claims" in detail["readiness"]["non_human_failures"]
    assert not (run_dir / "human_signoff.json").exists()


def test_agent_run_endpoint_rejects_missing_active_export(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    client = TestClient(app)

    response = client.post("/api/jobs/agent-run", json={"study_id": "sepsis"})

    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "no_active_export"


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 3.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] != "running":
            return payload
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")
