from __future__ import annotations

import io
import json
import threading
import time
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from easyicu import concept_catalog
from easyicu.webserver.app import app
import easyicu.webserver.app as app_module
from easyicu.webserver import agent_runs
from easyicu.webserver import cohort_review
from easyicu.webserver import copilot_sessions
from easyicu.webserver import numeric_evidence_audit
from easyicu.webserver.agent_runs import _scan_artifact_payloads
from easyicu.webserver import dataio
from easyicu.webserver import crossdb_review
from easyicu.webserver import catalog as catalog_module
from easyicu.webserver import guided_sessions
from easyicu.webserver import provider_adapter
from easyicu.webserver import provider_gate
from easyicu.webserver import settings as settings_store
from easyicu.webserver import sources as source_store
from easyicu.webserver.dataio import (
    describe_export_source,
    summarize_crossdb_workspaces,
    summarize_export_workspace,
)
from easyicu.webserver.ideas import mining as idea_mining_web

SIGNOFF_CONFIRMATIONS = [
    "evidence_reviewed",
    "claims_remain_locked",
    "no_patient_rows_persisted",
]

AGENT_PREFLIGHT_ARTIFACTS = {
    "run_context.json",
    "cohort_summary.json",
    "table1_summary.json",
    "missingness_audit.json",
    "roc_curve.json",
    "calibration_curve.json",
    "quality_gate.json",
    "evidence_ledger.json",
}


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
                "los_hosp": [4.0, 3.0, 6.0],
            }
        ),
        "sofa2_score": pd.DataFrame(
            {
                "stay_id": [1, 1, 2],
                "charttime": [
                    "2026-01-01 00:00",
                    "2026-01-01 01:00",
                    "2026-01-01 00:00",
                ],
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
                "charttime": [
                    "2026-01-01 00:00",
                    "2026-01-01 01:00",
                    "2026-01-01 00:00",
                ],
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


def _add_lactate_module(root: Path) -> None:
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "charttime": [
                "2026-01-01 00:00",
                "2026-01-01 00:00",
                "2026-01-01 00:00",
            ],
            "lact": [1.8, 3.2, 2.4],
        }
    ).to_csv(root / "labs.csv", index=False)
    manifest_path = root / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("files", []).append(
        {"file": "labs.csv", "module": "labs", "rows": 3}
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _add_large_ventilator_module(root: Path) -> None:
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "charttime": [
                "2026-01-01 00:00",
                "2026-01-01 00:00",
                "2026-01-01 00:00",
            ],
            "peep": [5.0, 8.0, 10.0],
            "mech_vent": ["invasive", "noninvasive", ""],
        }
    ).to_csv(root / "ventilator.csv", index=False)
    manifest_path = root / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("files", []).append(
        {"file": "ventilator.csv", "module": "ventilator", "rows": 2_000_000}
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_legacy_full_parquet_export(root: Path, database: str = "miiv") -> Path:
    root.mkdir()
    tables = {
        "demographics_age_bmi_height_sex_etc2.parquet": pd.DataFrame(
            {
                "stay_id": [1, 2, 3, 4],
                "age": [50, 70, 60, 65],
                "bmi": [23.1, 31.5, 27.0, 29.2],
                "height": [160, 175, 168, 171],
                "sex": ["F", "M", "F", "M"],
                "weight": [60, 96, 76, 85],
            }
        ),
        "outcome_death_los_icu_los_hosp_persistent_critical_illness_etc5.parquet": pd.DataFrame(
            {
                "stay_id": [1, 2, 3, 4],
                "death": [0, 1, 0, 0],
                "los_icu": [2.0, 5.0, 1.0, 3.0],
                "los_hosp": [4.0, 3.0, 6.0, 7.0],
                "persistent_critical_illness": [0, 0, 0, 1],
            }
        ),
        "sofa2_score_sofa2_sofa2_resp_sofa2_coag_sofa2_liver_etc3.parquet": pd.DataFrame(
            {
                "stay_id": [1, 1, 2, 3, 4],
                "charttime": [
                    "2026-01-01 00:00",
                    "2026-01-01 01:00",
                    "2026-01-01 00:00",
                    "2026-01-01 00:00",
                    "2026-01-01 00:00",
                ],
                "sofa2": [4, 5, 8, 3, 6],
                "sofa2_resp": [1, 1, 2, 0, 1],
                "sofa2_coag": [0, 0, 1, 0, 0],
                "sofa2_liver": [0, 0, 0, 0, 1],
            }
        ),
        "sepsis3_sofa2_sep3_sofa2.parquet": pd.DataFrame(
            {
                "stay_id": [1, 2],
                "sep3_sofa2": [True, False],
            }
        ),
        "vitals_hr_map_sbp_dbp_etc7.parquet": pd.DataFrame(
            {
                "stay_id": [1, 1, 2, 4],
                "charttime": [
                    "2026-01-01 00:00",
                    "2026-01-01 01:00",
                    "2026-01-01 00:00",
                    "2026-01-01 00:00",
                ],
                "hr": [90, 95, 80, 88],
                "map": [70, 72, 75, 76],
                "sbp": [110, 112, 118, 120],
                "dbp": [65, 66, 70, 68],
                "spo2": [97, 98, 96, 95],
                "temp": [37.0, 37.2, 36.8, 37.1],
            }
        ),
        "chemistry_alb_alp_alt_ast_etc26.parquet": pd.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "alb": [3.1, 2.8, 3.4],
                "alp": [80, 120, 95],
                "alt": [22, 34, 18],
                "ast": [30, 45, 20],
            }
        ),
    }
    for file_name, frame in tables.items():
        frame.to_parquet(root / file_name, index=False)

    (root / "easyicu_export_manifest.json").write_text(
        json.dumps(
            {
                "easyicu_version": "unknown",
                "exported_at": "2026-06-22T17:38:52.993877+00:00",
                "database": database,
                "patient_count": 4,
                "entry_mode": "module_grouped_full_export",
                "modules": [
                    {
                        "group": "sepsis3_sofa2",
                        "file": "sepsis3_sofa2_sep3_sofa2.parquet",
                        "rows": 2,
                        "feature_cols": 1,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return root


def test_catalog_active_export_coverage_uses_registered_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = _write_csv_export(tmp_path / "export")
    monkeypatch.setattr(
        source_store, "load_registry", lambda: {"active_path": str(export)}
    )

    catalog = catalog_module.build_catalog()
    active = catalog["activeExportCoverage"]

    assert active["status"] == "ready"
    assert active["denominator"] == 3
    assert active["payload_scope"] == "aggregate_only_no_rows"
    assert active["concepts"]["age"]["coverage_pct"] == 100.0
    assert active["concepts"]["hr"]["coverage_pct"] == 66.7
    assert active["concepts"]["sep3_sofa2"]["kind"] == "active_event"
    assert active["concepts"]["sep3_sofa2"]["coverage_pct"] == 33.3
    assert "hgb" not in active["concepts"]
    assert "stay_id" not in json.dumps(active)


def test_catalog_active_export_coverage_large_export_is_schema_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = _write_legacy_full_parquet_export(tmp_path / "legacy_full")
    desc = describe_export_source(str(export))
    desc["summary"]["total_rows"] = 2_000_001

    def fail_fast_stay_ids(*args, **kwargs):
        raise AssertionError("large catalog coverage must not read stay_id columns")

    def fail_file_coverage(*args, **kwargs):
        raise AssertionError("large catalog coverage must not read concept columns")

    monkeypatch.setattr(
        source_store, "load_registry", lambda: {"active_path": str(export)}
    )
    monkeypatch.setattr(dataio, "describe_export_source", lambda _path: desc)
    monkeypatch.setattr(dataio, "_fast_stay_ids", fail_fast_stay_ids)
    monkeypatch.setattr(catalog_module, "_file_concept_coverage", fail_file_coverage)

    catalog = catalog_module.build_catalog()
    active = catalog["activeExportCoverage"]

    assert active["status"] == "ready"
    assert active["mode"] == "schema_only"
    assert active["denominator"] == 4
    assert active["coverage_basis"] == "column_present_in_export_schema"
    assert active["summary"]["schemaOnly"] >= 10
    assert active["concepts"]["age"]["coverage_pct"] is None
    assert active["concepts"]["age"]["basis"] == "column_present_in_export_schema"
    assert active["concepts"]["hr"]["module"] == "vitals"
    assert "stay_id" not in json.dumps(active)


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
        row
        for row in manifest.get("files", [])
        if row.get("module") != module and row.get("file") != f"{module}.csv"
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _add_sofa1_module(root: Path) -> None:
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 3],
            "charttime": [
                "2026-01-01 00:00",
                "2026-01-01 01:00",
                "2026-01-01 00:00",
                "2026-01-01 00:00",
            ],
            "sofa1": [6, 7, 7, 3],
        }
    )
    frame.to_csv(root / "sofa1_score.csv", index=False)
    manifest_path = root / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("files", []).append(
        {"file": "sofa1_score.csv", "module": "sofa1_score", "rows": len(frame)}
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_native_static_assets_are_served_no_store() -> None:
    client = TestClient(app)

    js_res = client.get("/js/screens-viz.js")
    css_res = client.get("/css/screens.css")

    assert js_res.status_code == 200
    assert css_res.status_code == 200
    assert js_res.headers["cache-control"] == "no-store"
    assert css_res.headers["cache-control"] == "no-store"
    assert js_res.headers["pragma"] == "no-cache"


def test_settings_update_and_reset_are_local_and_whitelisted(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(settings_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(
        settings_store, "_CONFIG_PATH", tmp_path / "cfg" / "settings.json"
    )
    client = TestClient(app)

    updated = client.post(
        "/api/settings",
        json={
            "ai_enabled": True,
            "demo_patients": "50",
            "module_folder_mode": False,
            "token_budget": "42000",
            "working_dir": str(tmp_path / "work"),
            "export_dir": "",
            "unknown": "ignored",
        },
    )
    reset = client.post("/api/settings/reset", json={})

    assert updated.status_code == 200
    body = updated.json()
    assert body["ai_enabled"] is True
    assert body["demo_patients"] == 50
    assert body["module_folder_mode"] is False
    assert body["token_budget"] == 42000
    assert body["working_dir"] == str(tmp_path / "work")
    assert body["export_dir"] is None
    assert "unknown" not in body
    assert "about" in body

    assert reset.status_code == 200
    reset_body = reset.json()
    assert reset_body["ai_enabled"] is False
    assert reset_body["demo_patients"] == 20
    assert reset_body["module_folder_mode"] is True
    assert reset_body["token_budget"] == 120000
    assert reset_body["working_dir"] is None
    assert reset_body["export_dir"] is None
    assert "unknown" not in reset_body
    assert "about" in reset_body

    invalid = client.post(
        "/api/settings",
        json={
            "language": "fr",
            "data_mode": "cloud",
            "density": "microscopic",
            "demo_duration": "999h",
            "agent_model_mode": "surprise-provider",
            "demo_patients": "500",
        },
    )
    invalid_body = invalid.json()
    assert invalid.status_code == 200
    assert invalid_body["language"] == "en"
    assert invalid_body["data_mode"] == "demo"
    assert invalid_body["density"] == "comfortable"
    assert invalid_body["demo_duration"] == "24h"
    assert invalid_body["agent_model_mode"] == "local"
    assert invalid_body["demo_patients"] == 50


def test_settings_store_salvages_tail_corruption_and_rewrites_atomically(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(settings_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(
        settings_store, "_CONFIG_PATH", tmp_path / "cfg" / "settings.json"
    )
    settings_store._CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    settings_store._CONFIG_PATH.write_text(
        '{"language":"zh","data_mode":"real","density":"compact"}e\\n}}',
        encoding="utf-8",
    )
    client = TestClient(app)

    loaded = client.get("/api/settings").json()
    updated = client.post("/api/settings", json={"reduce_motion": True}).json()

    assert loaded["language"] == "zh"
    assert loaded["data_mode"] == "real"
    assert loaded["density"] == "compact"
    assert updated["reduce_motion"] is True
    repaired = json.loads(settings_store._CONFIG_PATH.read_text(encoding="utf-8"))
    assert repaired["language"] == "zh"
    assert repaired["data_mode"] == "real"
    assert repaired["density"] == "compact"
    assert repaired["reduce_motion"] is True


def test_guided_draft_registry_writes_metadata_only_without_row_payload(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(guided_sessions, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(
        guided_sessions, "_CONFIG_PATH", tmp_path / "cfg" / "guided.json"
    )
    monkeypatch.setattr(guided_sessions, "_PROJECTS_ROOT", tmp_path / "projects")
    parent_dir = tmp_path / "user_chosen_parent"
    parent_dir.mkdir()
    client = TestClient(app)

    created = client.post(
        "/api/guided/drafts",
        json={
            "title": "AKI onset draft",
            "folder_slug": "aki-onset-review",
            "parent_dir": str(parent_dir),
            "branch": "quality",
            "depth": "review",
            "data_mode": "real",
            "question": "Audit AKI onset coverage before modelling.",
            "cohort_hint": "adult first ICU",
            "module_hint": "renal + vitals",
            "source": {
                "path": str(tmp_path / "registered_export"),
                "label": "MIIV export",
                "database": "miiv",
            },
            "tableRows": [{"stay_id": 1, "subject_id": 2}],
            "patient": {"hadm_id": 3},
        },
    )
    listed = client.post("/api/guided/drafts/list", json={"limit": 10})

    assert created.status_code == 200
    body = created.json()
    draft = body["draft"]
    assert body["storage"] == "metadata_only"
    assert draft["kind"] == "guided_draft"
    assert draft["status"] == "metadata_only"
    assert draft["agent_run_created"] is False
    assert draft["reportable"] is False
    assert draft["draft_unlocked"] is False
    assert draft["project_kind"] == "guided_draft_folder"
    assert draft["project_artifact"] == "guided_draft.json"
    assert draft["project_parent_dir"] == str(parent_dir)
    assert draft["project_dir"].startswith(str(parent_dir / "guided-aki-onset-review-"))
    assert (Path(draft["project_dir"]) / "guided_draft.json").exists()
    opened = client.post(
        "/api/guided/project/open", json={"project_dir": draft["project_dir"]}
    )
    assert opened.status_code == 200
    assert opened.json()["ok"] is True
    assert opened.json()["session"]["project_dir"] == draft["project_dir"]
    assert draft["local_first"] == {"uploads": 0, "tokens": 0, "external_calls": 0}
    assert draft["privacy"]["no_patient_rows_persisted"] is True
    assert draft["privacy"]["row_level_markers"] == []
    assert draft["source"]["label"] == "MIIV export"
    assert draft["source"]["database"] == "miiv"
    assert "path" not in draft["source"]
    assert "path_hash" in draft["source"]

    dumped = json.dumps(draft)
    assert "tableRows" not in dumped
    assert "stay_id" not in dumped
    assert "subject_id" not in dumped
    assert "hadm_id" not in dumped

    assert listed.status_code == 200
    listed_body = listed.json()
    assert listed_body["storage"] == "metadata_only"
    assert listed_body["drafts"][0]["id"] == draft["id"]
    assert listed_body["drafts"][0]["project_dir"] == draft["project_dir"]

    persisted = json.loads(
        (tmp_path / "cfg" / "guided.json").read_text(encoding="utf-8")
    )
    persisted_dump = json.dumps(persisted)
    assert "tableRows" not in persisted_dump
    assert "stay_id" not in persisted_dump
    assert "subject_id" not in persisted_dump
    assert "hadm_id" not in persisted_dump


def test_guided_draft_registry_preserves_unicode_folder_slug(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(guided_sessions, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(
        guided_sessions, "_CONFIG_PATH", tmp_path / "cfg" / "guided.json"
    )
    monkeypatch.setattr(guided_sessions, "_PROJECTS_ROOT", tmp_path / "projects")
    parent_dir = tmp_path / "中文父目录"
    parent_dir.mkdir()
    client = TestClient(app)

    created = client.post(
        "/api/guided/drafts",
        json={
            "title": "测试1",
            "folder_slug": "测试1",
            "parent_dir": str(parent_dir),
            "data_mode": "real",
        },
    )

    assert created.status_code == 200
    draft = created.json()["draft"]
    assert draft["project_parent_dir"] == str(parent_dir)
    assert draft["project_dir"].startswith(str(parent_dir / "guided-测试1-"))
    assert (Path(draft["project_dir"]) / "guided_draft.json").exists()


def test_guided_draft_remove_unregisters_only_and_preserves_project_folder(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(guided_sessions, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(
        guided_sessions, "_CONFIG_PATH", tmp_path / "cfg" / "guided.json"
    )
    monkeypatch.setattr(guided_sessions, "_PROJECTS_ROOT", tmp_path / "projects")
    client = TestClient(app)

    created = client.post(
        "/api/guided/drafts",
        json={
            "title": "Draft to remove",
            "folder_slug": "remove-me",
            "data_mode": "real",
        },
    )
    assert created.status_code == 200
    draft = created.json()["draft"]
    project_dir = Path(draft["project_dir"])
    artifact = project_dir / "guided_draft.json"
    assert artifact.exists()

    dangerous = client.post(
        "/api/guided/drafts/remove",
        json={"draft_id": draft["id"], "delete_project_folder": True},
    )
    assert dangerous.status_code == 200
    dangerous_body = dangerous.json()
    assert dangerous_body["blocked"] is True
    assert dangerous_body["error"] == "project_folder_delete_not_supported"
    assert dangerous_body["disk_deleted"] is False
    assert artifact.exists()

    removed = client.post(
        "/api/guided/drafts/remove",
        json={
            "draft_id": draft["id"],
            "project_dir": draft["project_dir"],
            "delete_project_folder": False,
        },
    )
    assert removed.status_code == 200
    removed_body = removed.json()
    assert removed_body["ok"] is True
    assert removed_body["removed"] is True
    assert removed_body["draft_id"] == draft["id"]
    assert removed_body["disk_deleted"] is False
    assert artifact.exists()

    listed = client.post("/api/guided/drafts/list", json={"limit": 10})
    assert listed.status_code == 200
    assert listed.json()["drafts"] == []

    created_again = client.post(
        "/api/guided/drafts",
        json={
            "title": "Cached delete caller",
            "folder_slug": "cached-delete",
            "data_mode": "real",
        },
    )
    assert created_again.status_code == 200
    draft_again = created_again.json()["draft"]
    artifact_again = Path(draft_again["project_dir"]) / "guided_draft.json"
    assert artifact_again.exists()

    removed_again = client.request(
        "DELETE",
        "/api/guided/drafts/remove",
        json={"draft_id": draft_again["id"], "project_dir": draft_again["project_dir"]},
    )
    assert removed_again.status_code == 200
    assert removed_again.json()["ok"] is True
    assert removed_again.json()["disk_deleted"] is False
    assert artifact_again.exists()


def test_guided_copilot_session_routes_locally_and_rejects_row_payload(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(guided_sessions, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(
        guided_sessions, "_CONFIG_PATH", tmp_path / "cfg" / "guided.json"
    )
    monkeypatch.setattr(guided_sessions, "_PROJECTS_ROOT", tmp_path / "projects")
    client = TestClient(app)

    created = client.post(
        "/api/guided/session",
        json={
            "mode": "local",
            "context": {
                "route": "guided",
                "data_mode": "real",
                "language": "zh",
                "selected_source": {
                    "label": "MIIV export",
                    "database": "miiv",
                    "path": str(tmp_path / "registered_export"),
                },
                "tableRows": [{"stay_id": 1}],
                "patient": {"subject_id": 2},
            },
        },
    )
    assert created.status_code == 200
    body = created.json()
    session = body["session"]
    assert body["storage"] == "metadata_only"
    assert session["kind"] == "guided_copilot_session"
    assert session["mode"] == "local"
    assert session["step"] == "choose_goal"
    assert session["local_first"] == {"uploads": 0, "tokens": 0, "external_calls": 0}
    assert session["privacy"]["no_patient_rows_persisted"] is True
    assert session["privacy"]["row_level_markers"] == []
    assert session["context"]["selected_source"]["label"] == "MIIV export"
    assert "path_hash" in session["context"]["selected_source"]
    assert "path" not in session["context"]["selected_source"]

    fallback = client.post(
        "/api/guided/message",
        json={
            "session_id": session["id"],
            "message": "随便聊聊一个很模糊的请求",
            "context": {"route": "guided"},
        },
    )
    assert fallback.status_code == 200
    fallback_body = fallback.json()
    assert fallback_body["session"]["step"] == "choose_goal"
    assert fallback_body["session"]["handoff"] is None
    assert fallback_body["local_first"]["external_calls"] == 0
    assert fallback_body["goal_cards"][0]["goal"] == "idea_mining"

    message = client.post(
        "/api/guided/message",
        json={
            "session_id": session["id"],
            "message": "我想找一篇文章里的研究 idea",
            "context": {"route": "guided"},
        },
    )
    assert message.status_code == 200
    message_body = message.json()
    assert message_body["session"]["goal"] == "idea_mining"
    assert message_body["session"]["step"] == "handoff_ready"
    assert message_body["handoff"]["target_route"] == "ideas"
    assert message_body["handoff"]["prefill"]["source"] == "guided_copilot"

    review_message = client.post(
        "/api/guided/message",
        json={
            "session_id": session["id"],
            "message": "review data and patient view",
            "context": {"route": "guided"},
        },
    )
    assert review_message.status_code == 200
    review_body = review_message.json()
    assert review_body["session"]["goal"] == "review_data"
    assert review_body["handoff"]["target_route"] == "patient"

    action = client.post(
        "/api/guided/action",
        json={
            "session_id": session["id"],
            "action": "handoff_to_module",
            "goal": "data_extraction",
        },
    )
    assert action.status_code == 200
    action_body = action.json()
    assert action_body["result"]["target"] == "extraction"
    assert action_body["result"]["prefill"]["goal"] == "data_extraction"
    assert action_body["local_first"] == {
        "uploads": 0,
        "tokens": 0,
        "external_calls": 0,
    }

    blocked = client.post(
        "/api/guided/action",
        json={
            "session_id": session["id"],
            "action": "handoff_to_module",
            "goal": "free_chat_anything",
        },
    )
    assert blocked.status_code == 200
    assert blocked.json()["blocked"] is True
    assert blocked.json()["error"] == "unsupported_guided_goal"

    persisted = json.loads(
        (tmp_path / "cfg" / "guided.json").read_text(encoding="utf-8")
    )
    persisted_dump = json.dumps(persisted)
    assert "tableRows" not in persisted_dump
    assert "stay_id" not in persisted_dump
    assert "subject_id" not in persisted_dump
    assert "hadm_id" not in persisted_dump


def test_guided_project_memory_restores_conversation_per_local_folder(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(guided_sessions, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(
        guided_sessions, "_CONFIG_PATH", tmp_path / "cfg" / "guided.json"
    )
    monkeypatch.setattr(guided_sessions, "_PROJECTS_ROOT", tmp_path / "projects")
    client = TestClient(app)

    created = client.post(
        "/api/guided/drafts",
        json={
            "title": "Folder scoped study",
            "folder_slug": "folder-scoped-study",
            "branch": "predict",
            "depth": "full",
            "data_mode": "real",
            "question": "Evaluate a local export before choosing analysis.",
        },
    )
    draft = created.json()["draft"]

    opened = client.post(
        "/api/guided/project/open",
        json={
            "project_dir": draft["project_dir"],
            "draft_id": draft["id"],
            "title": draft["title"],
            "context": {"route": "guided", "data_mode": "real", "language": "zh"},
        },
    )
    assert opened.status_code == 200
    opened_body = opened.json()
    session = opened_body["session"]
    assert opened_body["opened"] is True
    assert session["project_dir"] == str(Path(draft["project_dir"]).resolve())
    assert session["project_kind"] == "guided_project_memory"
    assert session["project_title"] == "Folder scoped study"
    assert session["draft_id"] == draft["id"]
    assert session["memory_scope"] == "project_folder"
    assert session["messages"] == []
    assert session["local_first"] == {"uploads": 0, "tokens": 0, "external_calls": 0}

    message = client.post(
        "/api/guided/message",
        json={
            "session_id": session["id"],
            "message": "我想审阅已有数据",
            "context": {"route": "guided", "data_mode": "real", "language": "zh"},
        },
    )
    assert message.status_code == 200
    assert message.json()["session"]["goal"] == "review_data"
    assert message.json()["handoff"]["target_route"] == "patient"

    reopened = client.post(
        "/api/guided/project/open",
        json={
            "project_dir": draft["project_dir"],
            "context": {"route": "guided", "language": "zh"},
        },
    )
    restored = reopened.json()["session"]
    restored_messages = restored["messages"]
    assert reopened.json()["messages_restored"] >= 2
    assert any(
        row["role"] == "user" and row.get("text") == "我想审阅已有数据"
        for row in restored_messages
    )
    assert any(
        row["role"] == "assistant" and row.get("goal") == "review_data"
        for row in restored_messages
    )
    assert (Path(draft["project_dir"]) / "guided_copilot_session.json").exists()

    persisted = json.loads(
        (Path(draft["project_dir"]) / "guided_copilot_session.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted["id"] == session["id"]
    persisted_dump = json.dumps(persisted)
    assert "tableRows" not in persisted_dump
    assert "stay_id" not in persisted_dump
    assert "subject_id" not in persisted_dump
    assert "hadm_id" not in persisted_dump

    outside = client.post(
        "/api/guided/project/open", json={"project_dir": str(tmp_path / "outside")}
    )
    assert outside.status_code == 200
    assert outside.json()["blocked"] is True
    assert outside.json()["error"] == "invalid_guided_project_dir"


def test_guided_project_memory_persists_bounded_setup_slots(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(guided_sessions, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(
        guided_sessions, "_CONFIG_PATH", tmp_path / "cfg" / "guided.json"
    )
    monkeypatch.setattr(guided_sessions, "_PROJECTS_ROOT", tmp_path / "projects")
    client = TestClient(app)

    created = client.post(
        "/api/guided/drafts",
        json={
            "title": "Copilot slot study",
            "folder_slug": "copilot-slot-study",
            "data_mode": "real",
        },
    )
    draft = created.json()["draft"]
    opened = client.post(
        "/api/guided/project/open", json={"project_dir": draft["project_dir"]}
    )
    session = opened.json()["session"]

    update = client.post(
        "/api/guided/action",
        json={
            "session_id": session["id"],
            "action": "update_slots",
            "goal": "data_extraction",
            "step": "data_extraction_configuration",
            "context": {"route": "guided", "data_mode": "real", "language": "zh"},
            "slots": {
                "active_flow": "data_extraction",
                "extraction": {
                    "path": str(tmp_path / "raw_miiv"),
                    "cohort": "adult_first",
                    "modules": ["demographics", "vitals", "outcome"],
                    "format": "parquet",
                    "max_patients": 500,
                    "scan": {"ok": True, "ready": True, "db": "MIMIC-IV", "tables": 12},
                    "tableRows": [{"stay_id": 1}],
                },
                "agent": {
                    "question": "Evaluate lactate and mortality.",
                    "patient": {"subject_id": 123},
                },
                "patient": {"hadm_id": 456},
            },
        },
    )

    assert update.status_code == 200
    body = update.json()
    assert body["ok"] is True
    assert body["session"]["memory_scope"] == "project_folder"
    assert body["session"]["goal"] == "data_extraction"
    assert body["session"]["step"] == "data_extraction_configuration"
    slots = body["session"]["slots"]
    assert slots["active_flow"] == "data_extraction"
    assert slots["extraction"]["format"] == "parquet"
    assert slots["extraction"]["modules"] == ["demographics", "vitals", "outcome"]
    assert slots["extraction"]["scan"]["db"] == "MIMIC-IV"
    assert slots["agent"]["question"] == "Evaluate lactate and mortality."
    assert "tableRows" not in slots["extraction"]
    assert "patient" not in slots
    assert "patient" not in slots["agent"]

    reopened = client.post(
        "/api/guided/project/open", json={"project_dir": draft["project_dir"]}
    )
    restored = reopened.json()["session"]["slots"]
    assert restored["active_flow"] == "data_extraction"
    assert restored["extraction"]["path"] == str(tmp_path / "raw_miiv")
    assert restored["agent"]["question"] == "Evaluate lactate and mortality."

    persisted = json.loads(
        (Path(draft["project_dir"]) / "guided_copilot_session.json").read_text(
            encoding="utf-8"
        )
    )
    dumped = json.dumps(persisted)
    assert "tableRows" not in dumped
    assert "stay_id" not in dumped
    assert "subject_id" not in dumped
    assert "hadm_id" not in dumped
    assert persisted["privacy"]["no_patient_rows_persisted"] is True
    assert persisted["privacy"]["row_level_markers"] == []

    missing = client.post(
        "/api/guided/action",
        json={
            "session_id": "missing",
            "action": "update_slots",
            "slots": {"active_flow": "run_agent"},
        },
    )
    assert missing.status_code == 200
    assert missing.json()["blocked"] is True
    assert missing.json()["error"] == "guided_project_session_required"


def test_guided_project_open_accepts_existing_local_folder_without_draft(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(guided_sessions, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(
        guided_sessions, "_CONFIG_PATH", tmp_path / "cfg" / "guided.json"
    )
    monkeypatch.setattr(guided_sessions, "_PROJECTS_ROOT", tmp_path / "projects")
    project_dir = tmp_path / "projects" / "existing-study"
    project_dir.mkdir(parents=True)
    client = TestClient(app)

    opened = client.post(
        "/api/guided/project/open",
        json={
            "project_dir": str(project_dir),
            "title": "Existing study folder",
            "context": {"route": "guided", "language": "zh", "data_mode": "real"},
        },
    )

    assert opened.status_code == 200
    body = opened.json()
    session = body["session"]
    assert body["opened"] is True
    assert body["messages_restored"] == 0
    assert session["project_dir"] == str(project_dir.resolve())
    assert session["project_title"] == "Existing study folder"
    assert session["project_kind"] == "guided_project_memory"
    assert session["memory_scope"] == "project_folder"
    assert session["draft_id"] is None
    assert session["local_first"] == {"uploads": 0, "tokens": 0, "external_calls": 0}
    assert (project_dir / "guided_copilot_session.json").exists()

    outside = tmp_path / "not-projects" / "other-study"
    outside.mkdir(parents=True)
    blocked = client.post(
        "/api/guided/project/open", json={"project_dir": str(outside)}
    )
    assert blocked.status_code == 200
    assert blocked.json()["blocked"] is True
    assert blocked.json()["error"] == "invalid_guided_project_dir"


def test_page_guide_session_backend_is_metadata_only_and_drives_actions(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(copilot_sessions, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(
        copilot_sessions, "_CONFIG_PATH", tmp_path / "cfg" / "page-guide.json"
    )
    monkeypatch.setattr(copilot_sessions, "_PROJECTS_ROOT", tmp_path / "projects")
    client = TestClient(app)

    created = client.post(
        "/api/page-guide/sessions",
        json={
            "scope": "page_guide",
            "context": {
                "route": "extraction",
                "language": "zh",
                "data_mode": "real",
                "selected_source": {
                    "path": str(tmp_path / "registered_export"),
                    "label": "MIIV export",
                    "database": "miiv",
                },
                "tableRows": [{"stay_id": 1, "subject_id": 2}],
                "patient": {"hadm_id": 3},
            },
        },
    )
    assert created.status_code == 200
    body = created.json()
    session = body["session"]
    assert body["storage"] == "metadata_only"
    assert session["scope"] == "page_guide"
    assert session["project_kind"] == "page_guide_session_folder"
    assert session["local_first"] == {"uploads": 0, "tokens": 0, "external_calls": 0}
    assert session["privacy"]["no_patient_rows_persisted"] is True
    assert session["privacy"]["row_level_markers"] == []
    assert session["context"]["selected_source"]["label"] == "MIIV export"
    assert session["context"]["selected_source"]["database"] == "miiv"
    assert "path" not in session["context"]["selected_source"]
    assert "path_hash" in session["context"]["selected_source"]
    assert Path(session["project_dir"]).name.startswith("page-guide-extraction-")
    artifact = Path(session["project_dir"]) / "page_guide_session.json"
    assert artifact.exists()

    message = client.post(
        "/api/page-guide/message",
        json={
            "session_id": session["id"],
            "message": "打开 patient review",
            "context": {"route": "extraction", "language": "zh", "data_mode": "real"},
        },
    )
    assert message.status_code == 200
    message_body = message.json()
    assert message_body["actions"][0] == {
        "type": "navigate",
        "target": "patient",
        "requires_user_confirm": False,
    }
    assert message_body["session"]["local_first"]["external_calls"] == 0

    blocked = client.post(
        "/api/page-guide/action",
        json={"action": "start_external_model", "context": {"route": "agent"}},
    )
    assert blocked.status_code == 200
    assert blocked.json()["blocked"] is True
    assert blocked.json()["error"] == "unsupported_page_guide_action"
    assert blocked.json()["local_first"]["external_calls"] == 0

    listed = client.post("/api/page-guide/sessions/list", json={"limit": 5})
    assert listed.status_code == 200
    assert listed.json()["sessions"][0]["id"] == session["id"]

    compat = client.post(
        "/api/copilot/sessions",
        json={
            "scope": "quick_help",
            "context": {"route": "settings", "language": "en"},
        },
    )
    assert compat.status_code == 200
    assert compat.json()["session"]["scope"] == "page_guide"

    persisted_dump = artifact.read_text(encoding="utf-8") + json.dumps(message_body)
    assert "tableRows" not in persisted_dump
    assert "stay_id" not in persisted_dump
    assert "subject_id" not in persisted_dump
    assert "hadm_id" not in persisted_dump
    assert str(tmp_path / "registered_export") not in persisted_dump


def test_idea_mining_web_run_creates_ledger_preexperiment_and_handoff(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    monkeypatch.setattr(idea_mining_web, "_CONFIG_DIR", tmp_path / "idea_cfg")
    monkeypatch.setattr(idea_mining_web, "_RUN_ROOT", tmp_path / "idea_cfg" / "runs")
    monkeypatch.setattr(
        idea_mining_web, "_HISTORY_PATH", tmp_path / "idea_cfg" / "history.json"
    )
    monkeypatch.setattr(
        idea_mining_web,
        "_AGENT_PROJECTS_ROOT",
        tmp_path / "idea_cfg" / "agent_projects",
    )
    monkeypatch.setattr(
        idea_mining_web,
        "_AGENT_PROJECTS_PATH",
        tmp_path / "idea_cfg" / "agent_projects.json",
    )
    export_dir = _write_csv_export(tmp_path / "idea_export")
    source_store.register_source(
        str(export_dir), label="Idea fixture", active=True, crossdb=True
    )
    client = TestClient(app)

    mined = client.post(
        "/api/ideas/mine",
        json={
            "source_type": "manual",
            "topic": "early lactate clearance and ICU mortality",
            "title": "Lactate clearance review",
            "journal": "Intensive Care Medicine",
            "year": "2026",
            "excerpt": "Lactate clearance may identify high-risk ICU patients.",
        },
    )

    assert mined.status_code == 200
    body = mined.json()
    assert body["ok"] is True
    assert body["privacy"]["network_calls"] == 0
    assert body["privacy"]["external_llm_calls"] == 0
    assert body["source_evidence"][0]["source_text_stored"] is False
    idea = body["idea_ledger"][0]
    concept_ids = {row["concept_id"] for row in idea["mapped_concepts"]}
    assert {"lact", "death"} <= concept_ids
    tiers = {row["concept_id"]: row["tier"] for row in idea["mapped_concepts"]}
    assert tiers["lact"] == "T1_reextract"
    assert tiers["death"] == "executable"
    assert body["pre_experiment"]["status"] == "partial"
    assert "lact" in body["pre_experiment"]["missing_required_concepts"]
    assert body["pre_experiment"]["feature_statistics"][0]["concept_id"] == "death"
    assert body["prior_art"]["status"] == "not_checked_external_search_required"

    dumped = json.dumps(body, ensure_ascii=False)
    for marker in ["stay_id", "subject_id", "hadm_id", "tableRows"]:
        assert marker not in dumped

    handoff = client.post(
        "/api/ideas/handoff",
        json={
            "run_id": body["run_id"],
            "idea_id": body["selected_idea_id"],
            "plan_edits": "Use adult first ICU stay and add a missingness sensitivity check.",
        },
    )
    assert handoff.status_code == 200
    handoff_body = handoff.json()
    assert handoff_body["ok"] is True
    assert handoff_body["agent_seed"]["reportable"] is False
    assert handoff_body["agent_seed"]["draft_unlocked"] is False
    assert handoff_body["handoff_plan"]["human_plan_notes"].startswith("Use adult")

    prior_art = client.post(
        "/api/ideas/prior-art",
        json={"run_id": body["run_id"], "idea_id": body["selected_idea_id"]},
    )
    assert prior_art.status_code == 200
    prior_body = prior_art.json()
    assert prior_body["prior_art"]["status"] == "blocked_network_opt_in_required"
    assert prior_body["privacy"]["network_calls"] == 0
    assert prior_body["privacy"]["external_llm_calls"] == 0
    assert prior_body["prior_art"]["queries_to_run"]

    project = client.post(
        "/api/ideas/create-agent-project",
        json={"run_id": body["run_id"], "idea_id": body["selected_idea_id"]},
    )
    assert project.status_code == 200
    project_body = project.json()
    seed = project_body["project"]
    assert seed["status"] == "seeded_from_idea"
    assert seed["reportable"] is False
    assert seed["draft_unlocked"] is False
    assert seed["question"] == handoff_body["handoff_plan"]["research_question"]
    assert seed["cohort"] == "adult ICU cohort from Idea fixture (n=3)"
    assert seed["active_export_contract"]["status"] == "partial"
    assert seed["active_export_contract"]["label"] == "Idea fixture"
    assert seed["active_export_contract"]["database"] == "miiv"
    assert seed["active_export_contract"]["entities"] == 3
    assert seed["active_export_contract"]["path_hash"]
    assert "lact" in seed["active_export_contract"]["missing_required_concepts"]
    assert seed["prior_art_review"]["status"] == "blocked_network_opt_in_required"
    assert seed["prior_art_review"]["search_performed"] is False
    assert seed["execution_gate"]["project_seed_allowed"] is True
    assert seed["execution_gate"]["agent_run_ready_after_human_confirmation"] is False
    assert "re-extract or confirm missing required concepts" in seed["execution_gate"]["blockers"]
    assert "run prior-art review before Agent execution" in seed["execution_gate"]["blockers"]
    seed_dump = json.dumps(seed, ensure_ascii=False)
    assert str(export_dir) not in seed_dump
    assert "stay_id" not in seed_dump

    blocked_start = client.post(
        "/api/jobs/agent-run",
        json={
            "path": str(export_dir),
            "study_id": seed["study_id"],
            "question": seed["question"],
            "project_seed_dir": seed["project_dir"],
        },
    )
    assert blocked_start.status_code == 400
    assert blocked_start.json()["detail"]["error"] == "agent_project_execution_gate_blocked"
    assert "run prior-art review before Agent execution" in blocked_start.json()["detail"]["blockers"]
    assert (
        tmp_path
        / "idea_cfg"
        / "agent_projects"
        / seed["study_id"]
        / "project_seed.json"
    ).exists()

    projects = client.post("/api/ideas/agent-projects", json={"limit": 5})
    assert projects.status_code == 200
    assert projects.json()["projects"][0]["study_id"] == seed["study_id"]

    history = client.post("/api/ideas/history", json={"limit": 5})
    assert history.status_code == 200
    assert history.json()["runs"][0]["run_id"] == body["run_id"]

    loaded = client.post("/api/ideas/run", json={"run_id": body["run_id"]})
    assert loaded.status_code == 200
    loaded_body = loaded.json()
    assert loaded_body["loaded_from_history"] is True
    assert loaded_body["run_id"] == body["run_id"]
    assert loaded_body["selected_idea_id"] == body["selected_idea_id"]
    assert loaded_body["idea_ledger"][0]["idea_id"] == body["selected_idea_id"]
    assert loaded_body["handoff"]["idea_id"] == body["selected_idea_id"]
    assert (
        loaded_body["prior_art_check"]["prior_art"]["status"]
        == "blocked_network_opt_in_required"
    )
    assert loaded_body["agent_project"]["study_id"] == seed["study_id"]
    assert loaded_body["privacy"]["patient_rows_returned"] is False
    assert loaded_body["privacy"]["external_llm_calls"] == 0


def test_idea_mining_real_export_and_prior_art_unlock_agent_run_gate(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    monkeypatch.setattr(idea_mining_web, "_CONFIG_DIR", tmp_path / "idea_cfg")
    monkeypatch.setattr(idea_mining_web, "_RUN_ROOT", tmp_path / "idea_cfg" / "runs")
    monkeypatch.setattr(
        idea_mining_web, "_HISTORY_PATH", tmp_path / "idea_cfg" / "history.json"
    )
    monkeypatch.setattr(
        idea_mining_web,
        "_AGENT_PROJECTS_ROOT",
        tmp_path / "idea_cfg" / "agent_projects",
    )
    monkeypatch.setattr(
        idea_mining_web,
        "_AGENT_PROJECTS_PATH",
        tmp_path / "idea_cfg" / "agent_projects.json",
    )
    monkeypatch.setattr(
        idea_mining_web,
        "_pubmed_esearch",
        lambda query, limit=5: ["98765"],
    )
    monkeypatch.setattr(
        idea_mining_web,
        "_pubmed_esummary",
        lambda ids: [
            {
                "pmid": "98765",
                "title": "Lactate clearance and mortality in public ICU databases",
                "journal": "Critical Care",
                "year": 2025,
            }
        ],
    )
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: {**settings_store.DEFAULTS, "connector_pubmed_enabled": True},
    )
    export_dir = _write_csv_export(tmp_path / "idea_export")
    _add_lactate_module(export_dir)
    source_store.register_source(
        str(export_dir), label="Real MIIV export", active=True, crossdb=True
    )
    client = TestClient(app)

    mined = client.post(
        "/api/ideas/mine",
        json={
            "source_type": "manual",
            "topic": "lactate clearance and ICU mortality",
            "title": "Lactate clearance review",
            "journal": "Intensive Care Medicine",
            "year": "2026",
            "excerpt": "Lactate clearance may identify high-risk ICU mortality.",
        },
    )
    assert mined.status_code == 200
    body = mined.json()
    idea = body["idea_ledger"][0]
    assert idea["go_no_go"] == "recommend"
    assert body["pre_experiment"]["status"] == "ready"
    assert body["pre_experiment"]["missing_required_concepts"] == []

    prior_art = client.post(
        "/api/ideas/prior-art",
        json={
            "run_id": body["run_id"],
            "idea_id": body["selected_idea_id"],
            "allow_network": True,
        },
    )
    assert prior_art.status_code == 200
    assert prior_art.json()["prior_art"]["status"] == "searched"
    assert prior_art.json()["prior_art"]["search_performed"] is True

    planned = client.post(
        "/api/ideas/plan",
        json={"run_id": body["run_id"], "idea_id": body["selected_idea_id"]},
    )
    assert planned.status_code == 200
    confirmations = planned.json()["plan"]["required_user_confirmations"]
    assert "prepare or register a usable EasyICU export" not in confirmations
    assert "prior-art review opt-in or explicit decision to skip" not in confirmations
    assert (
        planned.json()["plan"]["execution_gate"][
            "agent_run_ready_after_human_confirmation"
        ]
        is True
    )

    handoff = client.post(
        "/api/ideas/handoff",
        json={"run_id": body["run_id"], "idea_id": body["selected_idea_id"]},
    )
    assert handoff.status_code == 200
    assert handoff.json()["handoff_plan"]["prior_art_review"]["search_performed"] is True

    project = client.post(
        "/api/ideas/create-agent-project",
        json={"run_id": body["run_id"], "idea_id": body["selected_idea_id"]},
    )
    assert project.status_code == 200
    seed = project.json()["project"]
    assert seed["active_export_contract"]["status"] == "ready"
    assert seed["active_export_contract"]["label"] == "Real MIIV export"
    assert seed["active_export_contract"]["demo_like"] is False
    assert seed["prior_art_review"]["status"] == "searched"
    assert seed["prior_art_review"]["result_count"] == 1
    assert seed["execution_gate"]["blockers"] == []
    assert seed["execution_gate"]["agent_run_ready_after_human_confirmation"] is True
    assert any(run["label"] == "prior-art review" for run in seed["runs"])
    seed_dump = json.dumps(seed, ensure_ascii=False)
    assert str(export_dir) not in seed_dump
    assert "stay_id" not in seed_dump
    assert "subject_id" not in seed_dump

    started = client.post(
        "/api/jobs/agent-run",
        json={
            "path": str(export_dir),
            "study_id": seed["study_id"],
            "question": seed["question"],
            "project_seed_dir": seed["project_dir"],
        },
    )
    assert started.status_code == 200
    snapshot = _wait_for_job(client, started.json()["job_id"])
    assert snapshot["status"] == "done"
    result = snapshot["result"]
    assert result["summary"]["stays"] == 3
    assert result["gate"]["status"] == "analysis_only"
    assert result["project_dir"].startswith(str(Path(seed["project_dir"]) / "runs"))
    assert result["uploads"] == 0
    assert result["tokens"] == 0


def test_idea_mining_large_module_uses_metadata_only_feature_stats(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    monkeypatch.setattr(idea_mining_web, "_CONFIG_DIR", tmp_path / "idea_cfg")
    monkeypatch.setattr(idea_mining_web, "_RUN_ROOT", tmp_path / "idea_cfg" / "runs")
    monkeypatch.setattr(
        idea_mining_web, "_HISTORY_PATH", tmp_path / "idea_cfg" / "history.json"
    )
    export_dir = _write_csv_export(tmp_path / "large_vent_export")
    _add_large_ventilator_module(export_dir)
    source_store.register_source(
        str(export_dir), label="Large ventilator export", active=True, crossdb=True
    )
    client = TestClient(app)

    mined = client.post(
        "/api/ideas/mine",
        json={
            "source_type": "manual",
            "topic": "PEEP and in-hospital mortality in mechanically ventilated ICU patients",
            "title": "ARDS ventilator review",
            "journal": "Intensive Care Medicine",
            "year": "2026",
            "excerpt": "Reviews highlight uncertainty about PEEP, mechanical ventilation, respiratory failure, ARDS, and mortality.",
        },
    )

    assert mined.status_code == 200
    body = mined.json()
    assert body["pre_experiment"]["status"] == "ready"
    assert body["pre_experiment"]["missing_required_concepts"] == []
    stats = {
        row["concept_id"]: row for row in body["pre_experiment"]["feature_statistics"]
    }
    assert stats["peep"]["status"] == "metadata_only"
    assert stats["peep"]["metric_kind"] == "schema_presence"
    assert stats["peep"]["coverage_basis"] == "manifest_file_inventory"
    assert stats["peep"]["records_declared"] == 2_000_000
    assert stats["death"]["metric_kind"] == "event_rate"
    assert any(
        "manifest/schema only" in note
        for note in body["pre_experiment"]["interpretation"]
    )
    dumped = json.dumps(body, ensure_ascii=False)
    for marker in ["stay_id", "subject_id", "hadm_id", "tableRows"]:
        assert marker not in dumped

    sample = client.post(
        "/api/ideas/bounded-feasibility",
        json={
            "run_id": body["run_id"],
            "idea_id": body["selected_idea_id"],
            "max_records": 500,
        },
    )
    assert sample.status_code == 200
    sample_body = sample.json()
    assert sample_body["schema_version"] == (
        "easyicu.web_idea_bounded_sample_feasibility/1"
    )
    assert sample_body["claim_level"] == "feasibility_sample_not_reportable"
    sample_stats = {
        row["concept_id"]: row
        for row in sample_body["feature_statistics"]
    }
    assert sample_stats["peep"]["status"] == "ready"
    assert sample_stats["peep"]["coverage_basis"] == "bounded_file_head_sample"
    assert sample_stats["peep"]["sample_limit_records"] == 500
    assert sample_stats["peep"]["sample_records"] == 3
    assert sample_stats["peep"]["records_declared"] == 2_000_000
    assert sample_stats["peep"]["coverage_pct"] == 100.0
    assert sample_stats["mech_vent"]["metric_kind"] == "event_rate"
    assert sample_stats["mech_vent"]["records"] == 2
    assert sample_stats["mech_vent"]["event_rate_pct"] == pytest.approx(66.7)
    assert sample_body["privacy"]["patient_rows_returned"] is False
    loaded = client.post("/api/ideas/run", json={"run_id": body["run_id"]})
    assert loaded.status_code == 200
    assert loaded.json()["bounded_sample_feasibility"]["status"] in {
        "ready",
        "needs_review",
    }
    sample_dumped = json.dumps(sample_body, ensure_ascii=False)
    for marker in ["stay_id", "subject_id", "hadm_id", "tableRows"]:
        assert marker not in sample_dumped


def test_idea_mining_repeated_same_source_keeps_distinct_local_records(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    monkeypatch.setattr(idea_mining_web, "_CONFIG_DIR", tmp_path / "idea_cfg")
    monkeypatch.setattr(idea_mining_web, "_RUN_ROOT", tmp_path / "idea_cfg" / "runs")
    monkeypatch.setattr(
        idea_mining_web, "_HISTORY_PATH", tmp_path / "idea_cfg" / "history.json"
    )
    monkeypatch.setattr(
        idea_mining_web,
        "_AGENT_PROJECTS_ROOT",
        tmp_path / "idea_cfg" / "agent_projects",
    )
    monkeypatch.setattr(
        idea_mining_web,
        "_AGENT_PROJECTS_PATH",
        tmp_path / "idea_cfg" / "agent_projects.json",
    )
    export_dir = _write_csv_export(tmp_path / "idea_export")
    source_store.register_source(
        str(export_dir), label="Idea fixture", active=True, crossdb=True
    )
    client = TestClient(app)
    payload = {
        "source_type": "manual",
        "topic": "early lactate clearance and ICU mortality",
        "title": "Lactate clearance review",
        "journal": "Intensive Care Medicine",
        "year": "2026",
        "excerpt": "Lactate clearance may identify high-risk ICU patients.",
    }

    first = client.post("/api/ideas/mine", json=payload)
    second = client.post("/api/ideas/mine", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    first_body = first.json()
    second_body = second.json()
    assert first_body["run_id"] != second_body["run_id"]
    history = client.post("/api/ideas/history", json={"limit": 10})
    assert history.status_code == 200
    history_rows = history.json()["runs"]
    history_ids = [row["run_id"] for row in history_rows[:2]]
    assert history_ids == [second_body["run_id"], first_body["run_id"]]
    history_keys = [row["history_key"] for row in history_rows[:2]]
    assert len(set(history_keys)) == 2
    assert all(
        key.startswith(f"{run_id}::") for key, run_id in zip(history_keys, history_ids)
    )

    first_project = client.post(
        "/api/ideas/create-agent-project",
        json={
            "run_id": first_body["run_id"],
            "idea_id": first_body["selected_idea_id"],
        },
    )
    second_project = client.post(
        "/api/ideas/create-agent-project",
        json={
            "run_id": second_body["run_id"],
            "idea_id": second_body["selected_idea_id"],
        },
    )
    assert first_project.status_code == 200
    assert second_project.status_code == 200
    assert (
        first_project.json()["project"]["study_id"]
        != second_project.json()["project"]["study_id"]
    )


def test_idea_mining_lists_only_existing_local_runs_and_projects(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(idea_mining_web, "_RUN_ROOT", tmp_path / "idea_cfg" / "runs")
    monkeypatch.setattr(
        idea_mining_web, "_HISTORY_PATH", tmp_path / "idea_cfg" / "history.json"
    )
    monkeypatch.setattr(
        idea_mining_web,
        "_AGENT_PROJECTS_ROOT",
        tmp_path / "idea_cfg" / "agent_projects",
    )
    monkeypatch.setattr(
        idea_mining_web,
        "_AGENT_PROJECTS_PATH",
        tmp_path / "idea_cfg" / "agent_projects.json",
    )
    run_dir = tmp_path / "idea_cfg" / "runs" / "real_run"
    run_dir.mkdir(parents=True)
    (run_dir / "idea_mining_run.json").write_text('{"ok": true}', encoding="utf-8")
    project_dir = tmp_path / "idea_cfg" / "agent_projects" / "real_project"
    project_dir.mkdir(parents=True)
    (project_dir / "project_seed.json").write_text('{"ok": true}', encoding="utf-8")
    (tmp_path / "idea_cfg" / "history.json").write_text(
        json.dumps(
            [
                {"run_id": "missing_run", "title": "Stale browser-looking row"},
                {"run_id": "real_run", "title": "Real local row"},
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "idea_cfg" / "agent_projects.json").write_text(
        json.dumps(
            [
                {
                    "study_id": "missing_project",
                    "project_dir": str(tmp_path / "missing_project"),
                },
                {"study_id": "real_project", "project_dir": str(project_dir)},
            ]
        ),
        encoding="utf-8",
    )
    client = TestClient(app)

    history = client.post("/api/ideas/history", json={"limit": 10})
    projects = client.post("/api/ideas/agent-projects", json={"limit": 10})

    assert history.status_code == 200
    assert [row["run_id"] for row in history.json()["runs"]] == ["real_run"]
    assert history.json()["runs"][0]["storage"] == "local_run_dir"
    assert projects.status_code == 200
    assert [row["study_id"] for row in projects.json()["projects"]] == ["real_project"]
    assert projects.json()["projects"][0]["storage"] == "local_project_seed"


def test_idea_mining_web_preserves_vasopressor_fluid_strategy_concept_set(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    monkeypatch.setattr(idea_mining_web, "_CONFIG_DIR", tmp_path / "idea_cfg")
    monkeypatch.setattr(idea_mining_web, "_RUN_ROOT", tmp_path / "idea_cfg" / "runs")
    monkeypatch.setattr(
        idea_mining_web, "_HISTORY_PATH", tmp_path / "idea_cfg" / "history.json"
    )
    export_dir = _write_csv_export(tmp_path / "idea_export")
    source_store.register_source(
        str(export_dir), label="Idea fixture", active=True, crossdb=True
    )
    client = TestClient(app)

    mined = client.post(
        "/api/ideas/mine",
        json={
            "source_type": "url",
            "topic": (
                "Early septic shock resuscitation comparing vasopressor-first or fluid-sparing "
                "strategy against fluid-forward resuscitation, with lactate, blood pressure, "
                "SOFA-2 severity, and mortality outcomes."
            ),
            "title": "Vasopressors or Fluids in Early Septic Shock",
            "journal": "New England Journal of Medicine",
            "year": 2026,
            "doi": "10.1056/NEJMoa2516225",
            "excerpt": (
                "Adult septic shock patients were assigned to restricted intravenous fluid and "
                "earlier vasopressor use or greater fluid volume and later vasopressors."
            ),
        },
    )

    assert mined.status_code == 200
    body = mined.json()
    idea = body["idea_ledger"][0]
    concept_ids = {row["concept_id"] for row in idea["mapped_concepts"]}
    assert {"vaso_ind", "total_input_ml", "lact", "sep3_sofa2", "death"} <= concept_ids
    roles = {row["concept_id"]: row["role"] for row in idea["mapped_concepts"]}
    assert roles["vaso_ind"] == "exposure"
    assert roles["total_input_ml"] == "exposure"
    assert roles["death"] == "outcome"
    assert "Vasopressor-fluid resuscitation strategy" in idea["idea_title"]
    assert idea["go_no_go"] == "hold"
    feature_stats = {
        row["concept_id"]: row
        for row in body["pre_experiment"]["feature_statistics"]
    }
    sep3_stats = feature_stats["sep3_sofa2"]
    assert sep3_stats["metric_kind"] == "event_rate"
    assert sep3_stats["event_entities"] == 1
    assert sep3_stats["non_event_entities"] == 2
    assert sep3_stats["denominator_entities"] == 3
    assert sep3_stats["event_rate_pct"] == 33.3
    assert sep3_stats["missing_pct"] is None
    assert sep3_stats["low_coverage"] is False
    assert "boolean/event indicator(s)" in " ".join(
        body["pre_experiment"]["interpretation"]
    )
    assert {"vaso_ind", "total_input_ml", "lact"} <= set(
        body["pre_experiment"]["missing_required_concepts"]
    )
    assert body["pre_experiment"]["status"] == "partial"

    handoff = client.post(
        "/api/ideas/handoff",
        json={"run_id": body["run_id"], "idea_id": body["selected_idea_id"]},
    )
    assert handoff.status_code == 200
    variables = handoff.json()["handoff_plan"]["variables"]
    assert any(
        row["concept_id"] == "vaso_ind" and row["role"] == "exposure"
        for row in variables
    )
    assert handoff.json()["agent_seed"]["reportable"] is False


def test_idea_mining_source_resolution_is_bounded_and_fail_closed_without_opt_in(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(idea_mining_web, "_CONFIG_DIR", tmp_path / "idea_cfg")
    client = TestClient(app)

    resolved = client.post(
        "/api/ideas/resolve-source",
        json={
            "source_type": "url",
            "url": "https://www.nejm.org/doi/full/10.1056/NEJMoa2516225",
            "title": "Vasopressors or Fluids in Early Septic Shock",
            "excerpt": "Earlier vasopressor use and restricted intravenous fluid may be measurable in ICU data.",
        },
    )

    assert resolved.status_code == 200
    body = resolved.json()
    assert body["ok"] is True
    assert body["source_adapter"]["status"] == "blocked_network_opt_in_required"
    assert body["source_adapter"]["network_calls"] == 0
    assert body["source_adapter"]["external_llm_calls"] == 0
    assert body["privacy"]["full_text_stored"] is False
    dumped = json.dumps(body, ensure_ascii=False)
    for marker in ["stay_id", "subject_id", "hadm_id", "tableRows"]:
        assert marker not in dumped


def test_summarize_export_workspace_builds_bounded_real_snapshot(
    tmp_path: Path,
) -> None:
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
        {
            "stay_id": "1",
            "age": 50.0,
            "sex": "F",
            "sofa2": 5.0,
            "los_icu": 2.0,
            "outcome": "Survived",
        },
        {
            "stay_id": "2",
            "age": 70.0,
            "sex": "M",
            "sofa2": 8.0,
            "los_icu": 5.0,
            "outcome": "Deceased",
        },
        {
            "stay_id": "3",
            "age": 60.0,
            "sex": "F",
            "sofa2": None,
            "los_icu": 1.0,
            "outcome": "Survived",
        },
    ]
    assert result["patient"]["stay_id"] == "1"
    assert result["patient"]["sepsis3"] is True
    assert [series["key"] for series in result["series"]] == [
        "hr",
        "map",
        "spo2",
        "temp",
    ]
    assert result["series"][0]["values"] == [90.0, 95.0]
    quality = {row["module"]: row for row in result["quality"]}
    assert quality["vitals"]["unique_stays"] == 2
    assert quality["vitals"]["coverage_pct"] == 66.7
    assert quality["vitals"]["coverage_basis"] == "unique_stay_id_intersection"
    assert quality["sepsis3_sofa2"]["status"] == "neutral"


def test_cohort_summary_uses_full_loaded_cohort_not_preview_rows(
    tmp_path: Path,
) -> None:
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
    assert result["summary"] == {
        "stays": 3,
        "modules": 5,
        "file_count": 5,
        "total_rows": 14,
    }
    assert result["files"][0]["columns"]


def test_describe_export_source_scans_legacy_full_export_inventory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    export_dir = _write_legacy_full_parquet_export(tmp_path / "legacy_full")

    def fail_read_stay_ids(path: Path):
        raise AssertionError(f"patient_count should avoid stay-id reads: {path}")

    monkeypatch.setattr(dataio, "_read_stay_ids", fail_read_stay_ids)

    result = describe_export_source(str(export_dir))

    assert result["ok"] is True
    assert result["database"] == "miiv"
    assert result["generated"] == "2026-06-22T17:38:52.993877+00:00"
    assert result["summary"] == {
        "stays": 4,
        "modules": 6,
        "file_count": 6,
        "total_rows": 22,
    }
    assert set(result["modules"]) == {
        "chemistry",
        "demographics",
        "outcome",
        "sepsis3_sofa2",
        "sofa2_score",
        "vitals",
    }
    files_by_name = {row["file"]: row for row in result["files"]}
    assert (
        files_by_name["demographics_age_bmi_height_sex_etc2.parquet"]["module"]
        == "demographics"
    )
    assert (
        files_by_name["chemistry_alb_alp_alt_ast_etc26.parquet"]["module"]
        == "chemistry"
    )
    assert (
        files_by_name[
            "sofa2_score_sofa2_sofa2_resp_sofa2_coag_sofa2_liver_etc3.parquet"
        ]["module"]
        == "sofa2_score"
    )
    assert files_by_name["sepsis3_sofa2_sep3_sofa2.parquet"]["rows"] == 2


def test_data_scan_auto_classifies_supported_folder_layouts(tmp_path: Path) -> None:
    module_export = _write_csv_export(tmp_path / "module_export")
    module_result = dataio.scan_path(str(module_export))
    assert module_result["ok"] is True
    assert module_result["source"] == "module"
    assert module_result["ready"] is True
    assert module_result["tables"] >= 5
    assert module_result["privacy"] == {
        "raw_rows_read": False,
        "patient_identifiers_returned": False,
    }

    prepared = tmp_path / "mimiciv_prepared"
    prepared.mkdir()
    for table in ("icustays", "patients", "admissions", "diagnoses_icd"):
        (prepared / f"{table}.parquet").write_bytes(b"placeholder")
    (prepared / "chartevents").mkdir()
    (prepared / "chartevents" / "1.parquet").write_bytes(b"placeholder")
    prepared_result = dataio.scan_path(str(prepared))
    assert prepared_result["ok"] is True
    assert prepared_result["source"] == "prepared"
    assert prepared_result["db_key"] == "miiv"
    assert prepared_result["ready"] is True

    raw = tmp_path / "mimiciv_raw"
    raw.mkdir()
    for table in ("icustays", "patients", "admissions"):
        (raw / f"{table}.csv.gz").write_bytes(b"not-real-gzip-but-sized")
    raw_result = dataio.scan_path(str(raw))
    assert raw_result["ok"] is True
    assert raw_result["source"] == "raw"
    assert raw_result["ready"] is False
    assert raw_result["size_hint"]

    unknown = tmp_path / "notes"
    unknown.mkdir()
    (unknown / "README.txt").write_text("not ICU data", encoding="utf-8")
    unknown_result = dataio.scan_path(str(unknown))
    assert unknown_result["ok"] is False
    assert unknown_result["error"] == "unrecognized_folder"
    assert unknown_result["source"] == "unknown"
    assert unknown_result["privacy"]["raw_rows_read"] is False


def test_workspace_summary_endpoint_returns_snapshot_and_rejects_bad_paths(
    tmp_path: Path,
) -> None:
    export_dir = _write_csv_export(tmp_path / "export")
    client = TestClient(app)

    ok = client.post("/api/workspace/summary", json={"path": str(export_dir)})
    bad = client.post(
        "/api/workspace/summary", json={"path": str(tmp_path / "missing")}
    )

    assert ok.status_code == 200
    assert ok.json()["summary"]["stays"] == 3
    assert ok.json()["series"][0]["key"] == "hr"
    assert bad.status_code == 400
    assert bad.json()["detail"]["error"] == "not_a_directory"


def test_patient_review_drilldown_uses_active_source_with_bounded_table_previews(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    demographics = pd.read_csv(export_dir / "demographics.csv")
    demographics["subject_id"] = [101, 102, 103]
    demographics["hadm_id"] = [201, 202, 203]
    demographics.to_csv(export_dir / "demographics.csv", index=False)
    source_store.register_source(
        str(export_dir), label="Review fixture", active=True, crossdb=True
    )
    client = TestClient(app)

    response = client.post("/api/patient-review/drilldown", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "real"
    assert payload["demo"] is False
    assert payload["source"]["label"] == "Review fixture"
    assert len(payload["source"]["path_hash"]) == 12
    assert "path" not in payload["source"]
    assert payload["summary"]["entities"] == 3
    assert payload["summary"]["modules"] == 5
    assert payload["summary"]["mortality"] == 33.3
    assert payload["summary"]["median_sofa2"] == 6.5
    assert payload["eligibility_flow"]["payload_scope"] == "cohort_attrition_metadata_only"
    assert payload["eligibility_flow"]["has_stepwise_report"] is False
    assert [row["id"] for row in payload["eligibility_flow"]["steps"]] == [
        "final_cohort",
    ]
    assert payload["eligibility_flow"]["steps"][0]["count"] == 3
    assert payload["eligibility_flow"]["steps"][0]["label"] == "Final cohort"
    assert payload["eligibility_flow"]["steps"][-1]["final"] is True
    assert len(payload["entities"]) == 3
    assert payload["entities"][0]["ref"].startswith("ent_")
    assert payload["selected"]["label"] == "Entity 1"
    assert payload["selected"]["demographics"] == {"age": 50.0, "sex": "F"}
    assert payload["selected"]["scores"] == {"sofa2_max": 5.0, "sepsis3_sofa2": True}
    assert payload["selected"]["outcomes"] == {
        "status": "Survived",
        "icu_los_days": 2.0,
    }
    signals = {row["key"]: row for row in payload["selected"]["signals"]}
    assert signals["hr"]["values"] == [90.0, 95.0]
    assert signals["hr"]["bounded"] is True
    assert signals["hr"]["max_points"] == 12
    modules = {row["module"]: row for row in payload["module_profiles"]}
    assert modules["vitals"]["feature_count"] == 4
    assert modules["vitals"]["dynamic_features"] == 4
    assert modules["vitals"]["coverage_pct"] == 66.7
    assert modules["demographics"]["static_features"] == 2
    lanes = {row["lane"]: row for row in payload["time_lanes"]}
    assert lanes["vitals"]["status"] == "ready"
    assert {row["feature"] for row in lanes["vitals"]["signals"]} == {
        "hr",
        "map",
        "spo2",
        "temp",
    }
    # Each lane signal must carry charttime aligned with its values so the
    # front-end renders the real ICU-admission-hour axis (not a 0..N index).
    vitals_hr = next(
        row for row in lanes["vitals"]["signals"] if row["feature"] == "hr"
    )
    assert vitals_hr["values"] == [90.0, 95.0]
    assert vitals_hr["times"] == ["2026-01-01 00:00", "2026-01-01 01:00"]
    assert len(vitals_hr["times"]) == len(vitals_hr["values"])
    assert lanes["scores"]["signals"][0]["feature"] == "sofa2"
    assert not any(
        row["feature"] == "age" for row in lanes.get("other", {}).get("signals", [])
    )
    quality_metrics = payload["quality_metrics"]
    assert (
        quality_metrics["payload_scope"] == "aggregate_quality_metrics_no_row_payload"
    )
    assert quality_metrics["summary"]["concept_count"] >= 8
    assert quality_metrics["summary"]["denominator_entities"] == 3
    quality_features = {row["feature"]: row for row in quality_metrics["features"]}
    assert quality_features["hr"]["coverage_pct"] == 66.7
    assert quality_features["hr"]["missing_pct"] == 33.3
    assert quality_features["hr"]["out_of_physio_pct"] == 0.0
    assert payload["privacy"]["bounded_table_previews"] is True
    assert payload["privacy"]["raw_source_rows_returned"] is False
    assert payload["privacy"]["bounded_pseudonymous_preview_rows_returned"] is True
    assert (
        payload["privacy"]["row_payload_scope"]
        == "bounded_pseudonymous_table_previews"
    )
    assert payload["privacy"]["max_table_preview_rows"] == 24
    assert payload["privacy"]["max_table_page_size"] == 100
    assert (
        payload["data_tables"]["payload_scope"]
        == "old_data_tables_semantics_with_bounded_pseudonymous_table_previews"
    )
    assert (
        payload["data_tables"]["detail_gate"]["title"] == "Bounded local table previews"
    )
    table_modules = {row["module"]: row for row in payload["data_tables"]["modules"]}
    assert table_modules["vitals"]["shape"] == "time_indexed"
    assert table_modules["vitals"]["label_i18n"] == {
        "en": "Vital Signs",
        "zh": "生命体征",
    }
    assert table_modules["vitals"]["preview_features"][0]["feature"] == "hr"
    assert table_modules["vitals"]["preview_features"][0]["name_i18n"] == {
        "en": "Heart Rate",
        "zh": "心率",
    }
    table_previews = {
        row["module"]: row for row in payload["data_tables"]["table_previews"]
    }
    assert table_previews["demographics"]["display_columns"] == ["entity", "age", "sex"]
    demographics_labels = {
        row["column"]: row for row in table_previews["demographics"]["display_column_labels"]
    }
    assert demographics_labels["entity"]["label_zh"] == "伪匿名实体"
    assert demographics_labels["age"]["label_zh"] == "年龄"
    assert (
        table_previews["demographics"]["identifier_policy"]
        == "pseudonymous_entity_token"
    )
    assert table_previews["demographics"]["rows"][0]["entity"].startswith("ent_")
    assert table_previews["demographics"]["rows"][0]["age"] == 50
    assert table_previews["vitals"]["display_columns"] == [
        "entity",
        "charttime",
        "hr",
        "map",
        "spo2",
        "temp",
    ]
    vitals_labels = {
        row["column"]: row for row in table_previews["vitals"]["display_column_labels"]
    }
    assert vitals_labels["charttime"]["label_zh"] == "记录时间"
    assert vitals_labels["hr"]["label_en"] == "Heart Rate"
    assert vitals_labels["hr"]["label_zh"] == "心率"
    assert table_previews["vitals"]["rows"][0]["hr"] == 90
    assert table_previews["vitals"]["pagination"] == {
        "page": 1,
        "page_size": 24,
        "page_count": 1,
        "row_start": 1,
        "row_end": 3,
        "rows_total": 3,
        "has_previous": False,
        "has_next": False,
    }
    assert table_previews["vitals"]["truncated_rows"] is False
    assert (
        payload["trajectory_review"]["payload_scope"]
        == "feature_matrix_semantics_bounded"
    )
    assert {row["id"] for row in payload["trajectory_review"]["modes"]} == {
        "feature_matrix",
        "single_entity",
        "multi_entity_comparison",
    }
    assert payload["trajectory_review"]["contract"][0]["label"] == "Entity scope"
    assert payload["trajectory_review"]["contract"][2]["label"] == "Feature matrices"
    assert (
        payload["patient_overview"]["payload_scope"]
        == "old_patient_overview_semantics_pseudonymous"
    )
    assert payload["patient_overview"]["navigator"]["actions"] == [
        "first",
        "previous",
        "next",
        "last",
        "random",
    ]
    assert (
        payload["patient_overview"]["category_view"]["sections"][0]["title"]
        == "Vital Signs Snapshot"
    )
    assert (
        payload["patient_overview"]["data_table"]["row_preview"]
        == "available_in_data_tables"
    )
    assert (
        payload["quality_review"]["payload_scope"]
        == "old_quality_semantics_aggregate_only"
    )
    assert {row["id"] for row in payload["quality_review"]["panels"]} == {
        "missingness",
        "outliers",
        "temporal",
    }
    assert any(
        item["id"] == "raw_identifier_table" for item in payload["blocked_features"]
    )
    serialized = json.dumps(payload)
    for marker in ["subject_id", "hadm_id", "tableRows", "stay_id", '"series"']:
        assert marker not in serialized


def test_patient_review_multi_entity_traces_keep_times_aligned_after_numeric_filter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    vitals = pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 2, 2],
            "charttime": [
                "2026-01-01 00:00",
                "2026-01-01 01:00",
                "2026-01-01 00:00",
                "2026-01-01 01:00",
                "2026-01-01 02:00",
            ],
            "hr": [90, 95, "", 82, 84],
            "map": [70, 72, 75, 76, 77],
            "spo2": [97, 98, 96, 96, 97],
            "temp": [37.0, 37.2, 36.8, 36.9, 37.0],
        }
    )
    vitals.to_csv(export_dir / "vitals.csv", index=False)
    manifest = json.loads((export_dir / "_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        if item["module"] == "vitals":
            item["rows"] = len(vitals)
    (export_dir / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    source_store.register_source(
        str(export_dir), label="Review fixture", active=True, crossdb=True
    )
    client = TestClient(app)

    response = client.post("/api/patient-review/drilldown", json={})

    assert response.status_code == 200
    comparison = response.json()["trajectory_review"]["multi_entity_comparison"]
    assert comparison["feature"] == "hr"
    traces = {row["label"]: row for row in comparison["traces"]}
    assert traces["Entity 1"]["values"] == [90.0, 95.0]
    assert traces["Entity 1"]["times"] == [
        "2026-01-01 00:00",
        "2026-01-01 01:00",
    ]
    assert traces["Entity 2"]["values"] == [82.0, 84.0]
    assert traces["Entity 2"]["times"] == [
        "2026-01-01 01:00",
        "2026-01-01 02:00",
    ]


def test_patient_review_drilldown_renders_manifest_eligibility_flow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    manifest_path = export_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cohort_contract"] = {
        "preset": "sepsis3",
        "age_min": 18,
        "age_max": 90,
        "min_icu_los_hours": 24,
        "observation_window_hours": 72,
        "exclude_readmissions": True,
        "icd_enabled": True,
        "icd_include": ["N17"],
        "icd_exclude": ["C34"],
    }
    manifest["cohort_report"] = {
        "mode": "sepsis3",
        "source_total": 5,
        "selected_before_concept_prefilter": 4,
        "concept_matches": 4,
        "selected_before_icd": 4,
        "selected_before_cap": 3,
        "selected": 3,
        "max_patients_applied": False,
        "applied_filters": ["demographics", "concept_prefilter", "icd"],
        "icd": {
            "enabled": True,
            "include_tokens": ["N17"],
            "exclude_tokens": ["C34"],
            "include_matches": 3,
            "exclude_matches": 1,
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source_store.register_source(
        str(export_dir), label="Review fixture", active=True, crossdb=True
    )
    client = TestClient(app)

    response = client.post("/api/patient-review/drilldown", json={})

    assert response.status_code == 200
    flow = response.json()["eligibility_flow"]
    assert flow["has_stepwise_report"] is True
    assert flow["privacy"] == {
        "patient_rows_returned": False,
        "direct_identifiers_returned": False,
    }
    steps = flow["steps"]
    assert [row["id"] for row in steps] == [
        "source_total",
        "demographic_stay_filters",
        "concept_prefilter",
        "icd_filters",
        "final_cohort",
    ]
    assert [row["count"] for row in steps] == [5, 4, 4, 3, 3]
    assert steps[1]["excluded"] == 1
    assert steps[2]["label_i18n"] == {
        "en": "Sepsis-3 cohort",
        "zh": "Sepsis-3 脓毒症队列",
    }
    assert steps[2]["note_i18n"] == {
        "en": "suspected infection + SOFA signal · first 72h window",
        "zh": "疑似感染 + SOFA 信号 · 前 72 小时窗口",
    }
    assert steps[3]["excluded"] == 1
    assert steps[4]["final"] is True
    assert steps[1]["label_i18n"]["zh"] == "年龄 18-90 岁 + ICU ≥ 24 小时"


def test_patient_review_drilldown_paginates_module_table_previews(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(
        str(export_dir), label="Review fixture", active=True, crossdb=True
    )
    client = TestClient(app)

    response = client.post(
        "/api/patient-review/drilldown",
        json={"table_module": "vitals", "table_page": 2, "table_page_size": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    previews = {row["module"]: row for row in payload["data_tables"]["table_previews"]}
    vitals = previews["vitals"]
    assert payload["data_tables"]["module_picker"]["default_module"] == "vitals"
    assert vitals["pagination"] == {
        "page": 2,
        "page_size": 1,
        "page_count": 3,
        "row_start": 2,
        "row_end": 2,
        "rows_total": 3,
        "has_previous": True,
        "has_next": True,
    }
    assert vitals["row_count"] == 1
    assert vitals["rows"][0]["hr"] == 95
    assert vitals["rows"][0]["entity"].startswith("ent_")
    assert "stay_id" not in vitals["rows"][0]
    assert "subject_id" not in vitals["rows"][0]
    assert "hadm_id" not in vitals["rows"][0]


def test_patient_review_drilldown_uses_legacy_full_export_counts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_legacy_full_parquet_export(
        tmp_path / "miiv_full", database="miiv"
    )
    source_store.register_source(
        str(export_dir), label="Full fixture", active=True, crossdb=True
    )
    client = TestClient(app)

    response = client.post("/api/patient-review/drilldown", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["entities"] == 4
    assert payload["summary"]["modules"] == 6
    assert payload["summary"]["file_count"] == 6
    assert payload["summary"]["total_rows"] == 22
    assert payload["summary"]["review_scope"] == "full_entity_set"
    modules = {row["module"]: row for row in payload["module_profiles"]}
    assert modules["chemistry"]["rows"] == 3
    assert modules["chemistry"]["feature_count"] == 4
    assert modules["demographics"]["feature_count"] >= 2
    assert payload["data_tables"]["loaded_summary"]["entities"] == 4
    assert payload["source"]["label"] == "Full fixture"


def test_patient_review_sources_lists_registered_exports_without_row_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(
        str(export_dir), label="Review fixture", active=True, crossdb=True
    )
    client = TestClient(app)

    response = client.post("/api/patient-review/sources", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "real"
    assert payload["demo"] is False
    assert payload["source_count"] == 1
    assert payload["can_load"] is True
    assert payload["active_source"]["label"] == "Review fixture"
    assert payload["active_source"]["patient_ready"] is True
    assert payload["active_source"]["summary"]["entities"] == 3
    assert payload["active_source"]["summary"]["modules"] == 5
    assert len(payload["active_source"]["path_hash"]) == 12
    assert payload["privacy"] == {
        "raw_rows_returned": False,
        "direct_identifiers_returned": False,
        "patient_rows_returned": False,
    }
    assert "local_export_source_metadata_only" == payload["provenance"]["payload_scope"]
    serialized = json.dumps(payload)
    for marker in ["subject_id", "hadm_id", "tableRows", "stay_id", '"series"']:
        assert marker not in serialized


def test_patient_review_drilldown_selects_pseudonymous_entity_ref(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    client = TestClient(app)

    first = client.post("/api/patient-review/drilldown", json={})
    assert first.status_code == 200
    second_ref = first.json()["entities"][1]["ref"]
    selected = client.post(
        "/api/patient-review/drilldown", json={"entity_ref": second_ref}
    )

    assert selected.status_code == 200
    payload = selected.json()
    assert payload["selected"]["ref"] == second_ref
    assert payload["selected"]["label"] == "Entity 2"
    assert payload["selected"]["demographics"] == {"age": 70.0, "sex": "M"}
    assert payload["selected"]["outcomes"]["status"] == "Deceased"


def test_patient_review_drilldown_fails_closed_without_registered_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    client = TestClient(app)

    no_active = client.post("/api/patient-review/drilldown", json={})
    unregistered = client.post(
        "/api/patient-review/drilldown",
        json={"source_path": str(tmp_path / "missing")},
    )

    assert no_active.status_code == 400
    assert no_active.json()["detail"]["error"] == "no_active_export"
    assert unregistered.status_code == 400
    assert unregistered.json()["detail"]["error"] == "source_not_registered"
    assert "path_hash" in unregistered.json()["detail"]
    assert "path" not in unregistered.json()["detail"]


def test_patient_review_drilldown_does_not_read_full_export_tables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    original_read_csv = pd.read_csv

    def guarded_read_csv(*args, **kwargs):
        if "nrows" not in kwargs and "usecols" not in kwargs:
            raise AssertionError(f"unexpected full CSV read: {args[0]}")
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    client = TestClient(app)

    response = client.post("/api/patient-review/drilldown", json={})

    assert response.status_code == 200
    assert response.json()["summary"]["entities"] == 3


def test_cohort_review_summary_uses_active_source_without_row_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    demographics = pd.read_csv(export_dir / "demographics.csv")
    demographics["subject_id"] = [101, 102, 103]
    demographics["hadm_id"] = [201, 202, 203]
    demographics.to_csv(export_dir / "demographics.csv", index=False)
    source_store.register_source(
        str(export_dir), label="Cohort fixture", active=True, crossdb=True
    )
    client = TestClient(app)

    response = client.post("/api/cohort-review/summary", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "real"
    assert payload["demo"] is False
    assert payload["source"]["label"] == "Cohort fixture"
    assert len(payload["source"]["path_hash"]) == 12
    assert "path" not in payload["source"]
    summary = payload["summary"]
    assert summary["cohort_size"] == 3
    assert summary["modules"] == 5
    assert summary["mortality"]["deceased_count"] == 1
    assert summary["mortality"]["survived_count"] == 2
    assert summary["mortality_pct"] == 33.3
    assert summary["age"]["count"] == 3
    assert summary["age"]["mean"] == 60.0
    assert summary["age"]["median"] == 60.0
    assert summary["age"]["min"] == 50.0
    assert summary["age"]["max"] == 70.0
    assert [b["label"] for b in summary["age"]["bins"]] == [
        "<40",
        "40-59",
        "60-74",
        ">=75",
    ]
    assert [b["count"] for b in summary["age"]["bins"]] == [0, 1, 2, 0]
    assert "admission" in summary
    assert "count" in summary["admission"]
    assert isinstance(summary["admission"]["bins"], list)
    assert "complexity" not in summary
    clinical_profile = summary["clinical_profile"]
    assert clinical_profile["payload_scope"] == "cohort_aggregate_only_no_patient_rows"
    domains = {row["id"]: row for row in clinical_profile["domains"]}
    assert set(domains) >= {
        "demographics",
        "severity_outcome",
        "treatments",
        "diagnosis",
        "data_completeness",
    }
    severity_items = {
        row["id"]: row for row in domains["severity_outcome"]["items"]
    }
    assert severity_items["sepsis3"]["kind"] == "event_rate"
    assert severity_items["sepsis3"]["pct"] == 33.3
    treatment_items = {row["id"]: row for row in domains["treatments"]["items"]}
    assert treatment_items["vasopressors"]["status"] == "unavailable"
    assert treatment_items["vasopressors"]["reason"] == "module_not_in_current_export"
    assert domains["diagnosis"]["items"][0]["status"] == "unavailable"
    coverage_items = {
        row["id"]: row for row in domains["data_completeness"]["items"]
    }
    assert coverage_items["coverage_vitals"]["pct"] == 66.7
    assert summary["sex"]["female_pct"] == 66.7
    assert summary["sofa2"]["median"] == 6.5
    assert [b["label"] for b in summary["sofa2"]["bins"]] == [
        "0-5",
        "6-8",
        "9-11",
        ">=12",
    ]
    assert summary["los_icu_days"]["median"] == 2.0
    assert [b["label"] for b in summary["los_icu_days"]["bins"]] == [
        "<2d",
        "2-5d",
        "5-10d",
        ">=10d",
    ]
    assert (
        sum(b["count"] for b in summary["los_icu_days"]["bins"])
        == summary["los_icu_days"]["count"]
    )
    assert summary["sepsis3"]["positive_count"] == 1
    assert summary["sepsis_pct"] == 33.3
    modules = {row["module"]: row for row in payload["coverage"]}
    assert modules["demographics"]["coverage_pct"] == 100.0
    assert modules["vitals"]["coverage_pct"] == 66.7
    assert modules["sepsis3_sofa2"]["metric_kind"] == "event_rate"
    assert modules["sepsis3_sofa2"]["quality_status"] == "neutral"
    assert payload["quality"]["watchlist_count"] == 2
    assert payload["groups"]["comparison_mode"] == "descriptive_only"
    assert payload["groups"]["inferential_statistics_allowed"] is False
    assert {row["id"] for row in payload["groups"]["supported"]} >= {
        "survival",
        "age",
        "sex",
        "los",
        "sepsis",
    }
    survival = next(
        row for row in payload["groups"]["supported"] if row["id"] == "survival"
    )
    assert survival["profile"]["status"] == "descriptive_aggregate_only"
    assert survival["profile"]["inferential_statistics_allowed"] is False
    assert survival["profile"]["columns"] == ["Survived", "Deceased", "Unknown"]
    profile_rows = {row["metric"]: row for row in survival["profile"]["rows"]}
    assert profile_rows["N"]["values"] == [2, 1, 0]
    assert profile_rows["Mortality %"]["values"] == [0.0, 100.0, None]
    assert profile_rows["Median age"]["values"] == [55.0, 70.0, None]
    assert profile_rows["Median SOFA-2"]["values"] == [5.0, 8.0, None]
    assert profile_rows["Median ICU LOS"]["values"] == [1.5, 5.0, None]
    assert "p_value" not in json.dumps(survival["profile"])
    assert "smd" not in json.dumps(survival["profile"])
    assert payload["table_one"]["status"] == "blocked"
    survival_analysis = payload["survival_analysis"]
    assert survival_analysis["status"] == "ready"
    assert survival_analysis["mode"] == "kaplan_meier_aggregate"
    assert survival_analysis["scope"] == "exploratory_unadjusted"
    assert survival_analysis["reportable"] is False
    assert survival_analysis["default_outcome"] == "mort_28d"
    hospital = next(
        row for row in survival_analysis["outcomes"] if row["id"] == "hospital_death"
    )
    assert hospital["status"] == "ready"
    assert hospital["event_column"] == "death"
    assert hospital["time_column"] == "los_hosp"
    assert hospital["usable_entities"] == 3
    assert hospital["event_count"] == 1
    assert hospital["display_horizon_days"] == 30.0
    assert hospital["window_label"] == "30-day display window"
    assert hospital["event_summary"] == {
        "status": "available",
        "basis": "event_flag",
        "event_column": "death",
        "time_column": None,
        "time_window_label": None,
        "denominator": 3,
        "event_count": 1,
        "event_rate_pct": 33.3,
    }
    assert (
        next(row for row in survival_analysis["outcomes"] if row["id"] == "icu_death")[
            "status"
        ]
        == "blocked"
    )
    mort_28d = next(row for row in survival_analysis["outcomes"] if row["id"] == "mort_28d")
    assert mort_28d["status"] == "ready"
    assert mort_28d["derived_from"] == "hospital_mortality_time_window"
    assert mort_28d["display_horizon_days"] == 28.0
    assert mort_28d["event_column"] == "death"
    assert mort_28d["time_column"] == "los_hosp"
    assert mort_28d["event_summary"]["basis"] == "derived_time_window"
    assert mort_28d["event_summary"]["event_count"] == 1
    assert mort_28d["event_summary"]["event_rate_pct"] == 33.3
    assert (
        next(row for row in survival_analysis["outcomes"] if row["id"] == "icu_death")[
            "reason"
        ]
        == "ICU mortality is unavailable because this export does not include an ICU-specific event column."
    )
    sepsis_curve = next(
        row
        for row in survival_analysis["curves"]
        if row["outcome_id"] == "hospital_death" and row["group_id"] == "sepsis"
    )
    assert sepsis_curve["logrank"]["status"] == "ready"
    assert sepsis_curve["logrank"]["test"] == "logrank"
    assert sepsis_curve["logrank"]["df"] == 1
    assert sepsis_curve["display_horizon_days"] == 30.0
    assert sepsis_curve["logrank"]["p_value"] > 0
    assert "<" not in sepsis_curve["logrank"]["p_value_label"]
    assert sepsis_curve["number_at_risk"]["times"] == [0.0, 1.0, 3.0, 6]
    risk_rows = {
        row["label"]: row["values"] for row in sepsis_curve["number_at_risk"]["rows"]
    }
    assert risk_rows["Non-sepsis"] == [2, 2, 2, 1]
    assert risk_rows["Sepsis"] == [1, 1, 1, 0]
    assert all("stay_id" not in json.dumps(row) for row in survival_analysis["curves"])
    assert payload["sofa_reclassification"]["status"] == "blocked"
    assert payload["sofa_reclassification"]["missing_modules"] == ["sofa1_score"]
    serialized = json.dumps(payload)
    for marker in ["subject_id", "hadm_id", "tableRows", "stay_id", '"series"']:
        assert marker not in serialized


def test_cohort_review_icu_death_event_rate_does_not_require_km_time(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    cohort_review._SUMMARY_CACHE.clear()
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "death": [0, 1, 0],
            "los_hosp": [4.0, 3.0, 6.0],
            "icu_mortality": [1, 0, 0],
        }
    ).to_csv(export_dir / "outcome.csv", index=False)
    source_store.register_source(str(export_dir), label="ICU event only", active=True)
    client = TestClient(app)

    response = client.post("/api/cohort-review/summary", json={})

    assert response.status_code == 200
    survival = response.json()["survival_analysis"]
    icu = next(row for row in survival["outcomes"] if row["id"] == "icu_death")
    assert icu["status"] == "blocked"
    assert icu["reason"] == (
        "ICU mortality event rate is available, but KM/log-rank needs "
        "ICU-specific time columns."
    )
    assert icu["event_summary"] == {
        "status": "available",
        "basis": "event_flag",
        "event_column": "icu_mortality",
        "time_column": None,
        "time_window_label": None,
        "denominator": 3,
        "event_count": 1,
        "event_rate_pct": 33.3,
    }
    assert all(row["outcome_id"] != "icu_death" for row in survival["curves"])
    serialized = json.dumps(survival)
    for marker in ["subject_id", "hadm_id", "tableRows", "stay_id", '"series"']:
        assert marker not in serialized


def test_cohort_review_presence_rate_modules_are_not_low_coverage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    cohort_review._SUMMARY_CACHE.clear()
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    extra_tables = {
        "sepsis3_sofa1": pd.DataFrame({"stay_id": [1], "sep3_sofa1": [1]}),
        "vasopressors": pd.DataFrame({"stay_id": [2], "vaso_ind": [1]}),
        "ventilator": pd.DataFrame({"stay_id": [3], "vent_ind": [1]}),
    }
    manifest_path = export_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for module, frame in extra_tables.items():
        file_name = f"{module}.csv"
        frame.to_csv(export_dir / file_name, index=False)
        manifest.setdefault("files", []).append(
            {"file": file_name, "module": module, "rows": len(frame)}
        )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    client = TestClient(app)

    response = client.post("/api/cohort-review/summary", json={})

    assert response.status_code == 200
    payload = response.json()
    modules = {row["module"]: row for row in payload["coverage"]}
    assert modules["sepsis3_sofa1"]["coverage_pct"] == 33.3
    assert modules["sepsis3_sofa1"]["metric_kind"] == "event_rate"
    assert modules["sepsis3_sofa1"]["quality_status"] == "neutral"
    assert modules["vasopressors"]["coverage_pct"] == 33.3
    assert modules["vasopressors"]["metric_kind"] == "exposure_rate"
    assert modules["vasopressors"]["quality_status"] == "neutral"
    assert modules["ventilator"]["coverage_pct"] == 33.3
    assert modules["ventilator"]["metric_kind"] == "exposure_rate"
    assert modules["ventilator"]["quality_status"] == "neutral"
    assert payload["quality"]["watchlist_count"] == 2
    assert payload["quality"]["modules_neutral"] == 4
    coverage_domain = {
        row["id"]: row
        for row in payload["summary"]["clinical_profile"]["domains"]
    }["data_completeness"]
    watchlist_modules = {
        item["modules"][0]
        for item in coverage_domain["items"]
        if item["id"].startswith("coverage_")
    }
    assert "sepsis3_sofa1" not in watchlist_modules
    assert "vasopressors" not in watchlist_modules
    assert "ventilator" not in watchlist_modules


def test_cohort_review_feature_catalog_and_selected_feature_profiles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    cohort_review._SUMMARY_CACHE.clear()
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(
        str(export_dir), label="Feature catalog fixture", active=True, crossdb=True
    )
    client = TestClient(app)

    response = client.post(
        "/api/cohort-review/summary",
        json={
            "selected_features": [
                "vitals:hr",
                "vitals:map",
                "outcome:los_hosp",
                "missing:nope",
            ]
        },
    )

    assert response.status_code == 200
    payload = response.json()
    catalog = payload["feature_catalog"]
    assert catalog["total_modules"] == 5
    assert catalog["total_features"] >= 11
    modules = {row["module"]: row for row in catalog["modules"]}
    vitals_features = {row["id"]: row for row in modules["vitals"]["features"]}
    assert {"vitals:hr", "vitals:map", "vitals:spo2", "vitals:temp"}.issubset(
        vitals_features
    )
    assert vitals_features["vitals:hr"]["selected"] is True
    assert payload["feature_selection"]["selected_count"] == 3
    assert payload["feature_selection"]["ignored"] == ["missing:nope"]

    survival = next(
        row for row in payload["groups"]["supported"] if row["id"] == "survival"
    )
    feature_rows = {
        row["feature_id"]: row
        for row in survival["profile"]["rows"]
        if row.get("feature_id")
    }
    assert set(feature_rows) == {"vitals:hr", "vitals:map", "outcome:los_hosp"}
    assert feature_rows["vitals:hr"]["kind"] == "numeric"
    assert feature_rows["vitals:hr"]["values"] == [92.5, 80.0, None]
    assert feature_rows["outcome:los_hosp"]["values"] == [5.0, 3.0, None]
    serialized = json.dumps(payload)
    assert "mapping" not in serialized
    assert "stay_id" not in serialized


def test_cohort_review_sofa_reclassification_uses_paired_aggregate_without_row_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    _add_sofa1_module(export_dir)
    source_store.register_source(
        str(export_dir), label="Cohort fixture", active=True, crossdb=True
    )
    client = TestClient(app)

    response = client.post("/api/cohort-review/summary", json={})

    assert response.status_code == 200
    payload = response.json()
    reclass = payload["sofa_reclassification"]
    assert reclass["status"] == "ready"
    assert reclass["mode"] == "worst_icu"
    assert reclass["paired_backend_ready"] is True
    assert reclass["payload_scope"] == "paired_score_aggregate_only"
    assert reclass["inferential_statistics_allowed"] is False
    assert reclass["paired_count"] == 2
    assert reclass["coverage_pct"] == 66.7
    assert reclass["direction_counts"]["up"] == {"count": 1, "pct": 50.0}
    assert reclass["direction_counts"]["down"] == {"count": 1, "pct": 50.0}
    assert reclass["direction_counts"]["same"] == {"count": 0, "pct": 0.0}
    assert reclass["delta_summary"]["median"] == -0.5
    assert reclass["delta_summary"]["min"] == -2
    assert reclass["delta_summary"]["max"] == 1
    matrix = {row["label"]: row for row in reclass["transition_matrix"]}
    row_6_8 = {cell["label"]: cell for cell in matrix["6-8"]["cells"]}
    assert row_6_8["0-5"]["count"] == 1
    assert row_6_8["6-8"]["count"] == 1
    assert reclass["score_scale"] == {
        "min": 0,
        "max": 24,
        "unit": "SOFA points",
        "aggregation": "nearest_integer_clamped_0_24",
    }
    assert reclass["exact_score_bins"][0] == "0"
    assert reclass["exact_score_bins"][-1] == "24"
    assert len(reclass["exact_score_matrix"]) == 25
    assert len(reclass["exact_score_matrix"][0]["cells"]) == 25
    exact_matrix = {row["label"]: row for row in reclass["exact_score_matrix"]}
    row_7 = {cell["label"]: cell for cell in exact_matrix["7"]["cells"]}
    assert row_7["5"]["count"] == 1
    assert row_7["8"]["count"] == 1
    assert {row["id"]: row["status"] for row in reclass["mode_options"]} == {
        "worst_icu": "ready",
        "first24h": "blocked",
        "time_aligned": "blocked",
    }
    assert "paired_sofa_reclassification" not in {
        row["id"] for row in payload["blocked_features"]
    }
    serialized = json.dumps(payload)
    for marker in ["subject_id", "hadm_id", "tableRows", "stay_id", '"series"']:
        assert marker not in serialized
    reclass_serialized = json.dumps(reclass)
    assert "p_value" not in reclass_serialized
    assert "smd" not in reclass_serialized


def test_cohort_review_summary_fails_closed_without_registered_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    client = TestClient(app)

    no_active = client.post("/api/cohort-review/summary", json={})
    unregistered = client.post(
        "/api/cohort-review/summary",
        json={"source_path": str(tmp_path / "missing")},
    )

    assert no_active.status_code == 400
    assert no_active.json()["detail"]["error"] == "no_active_export"
    assert unregistered.status_code == 400
    assert unregistered.json()["detail"]["error"] == "source_not_registered"
    assert "path_hash" in unregistered.json()["detail"]
    assert "path" not in unregistered.json()["detail"]


def test_cohort_review_summary_rejects_unsupported_filters_and_statistics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    client = TestClient(app)

    row_filter = client.post(
        "/api/cohort-review/summary",
        json={"filters": {"age_at_admission": {"min": 18}}},
    )
    p_value = client.post(
        "/api/cohort-review/summary",
        json={"statistics": ["p_value"]},
    )

    assert row_filter.status_code == 400
    assert row_filter.json()["detail"]["error"] == "unsupported_filter"
    assert row_filter.json()["detail"]["unsupported"][0]["id"] == "age_at_admission"
    assert "summary" not in row_filter.json()["detail"]
    assert p_value.status_code == 400
    assert p_value.json()["detail"]["error"] == "unsupported_statistic"
    assert p_value.json()["detail"]["unsupported"][0]["id"] == "p_value"
    assert "summary" not in p_value.json()["detail"]


def test_cohort_review_summary_does_not_read_full_export_tables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(export_dir), active=True, crossdb=True)

    def fail_read_frame(path: Path):
        raise AssertionError(f"cohort review must not read full frame: {path}")

    original_read_csv = pd.read_csv

    def guarded_read_csv(*args, **kwargs):
        if "nrows" not in kwargs and "usecols" not in kwargs:
            raise AssertionError(f"unexpected full CSV read: {args[0]}")
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(dataio, "_read_export_frame", fail_read_frame)
    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    client = TestClient(app)

    response = client.post("/api/cohort-review/summary", json={})

    assert response.status_code == 200
    assert response.json()["summary"]["cohort_size"] == 3


def test_cohort_review_large_coverage_uses_metadata_without_stay_id_scan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    manifest_path = export_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stays"] = 3
    for row in manifest["files"]:
        row["rows"] = 1_000_001
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source_store.register_source(str(export_dir), active=True, crossdb=True)

    def fail_read_stay_ids(path: Path):
        raise AssertionError(f"large coverage must not scan stay_id columns: {path}")

    monkeypatch.setattr(dataio, "_read_stay_ids", fail_read_stay_ids)
    client = TestClient(app)

    response = client.post("/api/cohort-review/summary", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["cohort_size"] == 3
    modules = {row["module"]: row for row in payload["coverage"]}
    assert modules["demographics"]["coverage_basis"] == "metadata_row_count_only"
    assert modules["demographics"]["coverage_pct"] is None
    assert modules["demographics"]["covered_entities"] is None
    assert (
        modules["demographics"]["skipped_reason"]
        == "unique_stay_scan_skipped_large_module"
    )
    assert payload["quality"]["modules_unknown"] >= 5


def test_cohort_review_large_parquet_export_reuses_active_source_for_km_and_coverage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = tmp_path / "large_parquet"
    export_dir.mkdir()
    n = 25_000
    stay_ids = list(range(1, n + 1))
    pd.DataFrame(
        {
            "stay_id": stay_ids,
            "age": [50 + (i % 30) for i in stay_ids],
            "sex": ["F" if i % 2 else "M" for i in stay_ids],
        }
    ).to_parquet(export_dir / "demographics.parquet", index=False)
    pd.DataFrame(
        {
            "stay_id": stay_ids,
            "death": [i % 10 == 0 for i in stay_ids],
            "los_hosp": [float((i % 60) + 1) for i in stay_ids],
            "los_icu": [float((i % 14) + 1) for i in stay_ids],
        }
    ).to_parquet(export_dir / "outcome.parquet", index=False)
    pd.DataFrame(
        {
            "stay_id": stay_ids,
            "sep3_sofa2": [i % 3 == 0 for i in stay_ids],
        }
    ).to_parquet(export_dir / "sepsis3_sofa2.parquet", index=False)
    (export_dir / "_manifest.json").write_text(
        json.dumps(
            {
                "database": "miiv",
                "generated": "2026-06-26T12:00:00",
                "files": [
                    {
                        "file": "demographics.parquet",
                        "module": "demographics",
                        "rows": 1_000_001,
                    },
                    {"file": "outcome.parquet", "module": "outcome", "rows": 1_000_001},
                    {
                        "file": "sepsis3_sofa2.parquet",
                        "module": "sepsis3_sofa2",
                        "rows": 1_000_001,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    source_store.register_source(
        str(export_dir), label="Large parquet", active=True, crossdb=True
    )
    client = TestClient(app)

    response = client.post("/api/cohort-review/summary", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["cohort_size"] == n
    modules = {row["module"]: row for row in payload["coverage"]}
    assert modules["demographics"]["coverage_basis"] == "unique_entity_intersection"
    assert modules["demographics"]["coverage_pct"] == 100.0
    assert modules["outcome"]["coverage_pct"] == 100.0
    survival = payload["survival_analysis"]
    assert survival["status"] == "ready"
    assert survival["default_outcome"] == "mort_28d"
    assert survival["curves"]
    sepsis_curve = next(
        row
        for row in survival["curves"]
        if row["outcome_id"] == "hospital_death" and row["group_id"] == "sepsis"
    )
    assert sepsis_curve["display_horizon_days"] == 30.0
    assert sepsis_curve["number_at_risk"]["times"][-1] == 30.0
    assert sepsis_curve["logrank"]["p_value_label"]
    assert "<" not in sepsis_curve["logrank"]["p_value_label"]
    assert len(json.dumps(survival)) < 80_000
    assert "stay_id" not in json.dumps(survival)


def test_cohort_review_summary_reuses_cached_payload_for_same_export(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    cohort_review._SUMMARY_CACHE.clear()
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(
        str(export_dir), label="Cached cohort", active=True, crossdb=True
    )
    client = TestClient(app)

    first = client.post("/api/cohort-review/summary", json={})
    assert first.status_code == 200

    def fail_selected_columns(*args, **kwargs):
        raise AssertionError("cached cohort summary should not re-read module files")

    monkeypatch.setattr(cohort_review, "_read_selected_columns", fail_selected_columns)
    second = client.post("/api/cohort-review/summary", json={})

    assert second.status_code == 200
    assert second.json()["summary"] == first.json()["summary"]


def test_cohort_review_skips_large_time_indexed_sofa_for_interactive_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    manifest_path = export_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stays"] = 3
    for row in manifest["files"]:
        if row["module"] == "sofa2_score":
            row["rows"] = 2_000_001
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source_store.register_source(str(export_dir), active=True, crossdb=True)

    original_read_csv = pd.read_csv

    def guarded_read_csv(*args, **kwargs):
        if "sofa2_score" in str(args[0]):
            raise AssertionError(f"large SOFA module should be deferred: {args[0]}")
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    client = TestClient(app)

    response = client.post("/api/cohort-review/summary", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["cohort_size"] == 3
    assert payload["summary"]["sofa2"]["median"] is None
    assert payload["sofa_reclassification"]["status"] == "blocked"
    modules = {row["module"]: row for row in payload["coverage"]}
    assert modules["sofa2_score"]["coverage_basis"] == "metadata_row_count_only"
    assert modules["sofa2_score"]["coverage_pct"] is None


def test_data_scan_recognizes_native_manifest_export_as_module_source(
    tmp_path: Path,
) -> None:
    export_dir = _write_csv_export(tmp_path / "export")

    result = dataio.scan_path(str(export_dir), source_hint="module")

    assert result["ok"] is True
    assert result["source"] == "module"
    assert result["ready"] is True
    assert result["layout"][0] == "EasyICU module export"


def test_extraction_filter_options_use_active_source_without_row_level_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    client = TestClient(app)

    response = client.post("/api/extraction/filter-options", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "real"
    assert payload["demo"] is False
    assert payload["source"]["id"].startswith("src_")
    assert payload["source"]["label"] == "MIIV"
    assert len(payload["source"]["path_hash"]) == 12
    assert "path" not in payload["source"]
    assert payload["summary"]["cohort_size"] == 3
    assert payload["summary"]["modules"] == 5
    assert "source_registry" in payload["provenance"]["computed_from"]
    modules = {row["module"]: row for row in payload["options"]["modules"]}
    assert modules["demographics"]["row_count"] == 3
    assert modules["demographics"]["coverage_pct"] == 100.0
    assert modules["demographics"]["quality_status"] == "ok"
    assert modules["sepsis3_sofa2"]["quality_status"] == "neutral"
    assert "age" in modules["demographics"]["columns"]
    assert modules["demographics"]["hidden_identifier_columns"] == 1
    serialized = json.dumps(payload)
    for marker in ["stay_id", "subject_id", "hadm_id", "tableRows", '"series"']:
        assert marker not in serialized


def test_extraction_filter_preview_applies_supported_metadata_filters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    client = TestClient(app)

    response = client.post(
        "/api/extraction/filter-preview",
        json={"filters": {"min_coverage_pct": 80, "required_columns": ["age"]}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["match_count"] == 1
    assert payload["matched_modules"][0]["module"] == "demographics"
    assert payload["aggregate"]["cohort_size"] == 3
    assert payload["aggregate"]["matched_rows"] == 3
    serialized = json.dumps(payload)
    for marker in ["stay_id", "subject_id", "hadm_id", "tableRows", '"series"']:
        assert marker not in serialized


def test_extraction_advanced_filters_fixture_e2e_register_options_and_preview(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    client = TestClient(app)

    registered = client.post(
        "/api/workspaces/register",
        json={"path": str(export_dir), "label": "Fixture MIIV", "active": True},
    )
    options = client.post("/api/extraction/filter-options", json={})
    preview = client.post(
        "/api/extraction/filter-preview",
        json={"filters": {"quality_statuses": ["warn"], "min_coverage_pct": 50}},
    )

    assert registered.status_code == 200
    assert registered.json()["active_path"] == str(export_dir)
    assert options.status_code == 200
    assert options.json()["source"]["label"] == "Fixture MIIV"
    assert preview.status_code == 200
    payload = preview.json()
    assert payload["source"]["id"] == options.json()["source"]["id"]
    assert payload["match_count"] == 2
    assert {row["module"] for row in payload["matched_modules"]} == {
        "sofa2_score",
        "vitals",
    }
    assert payload["aggregate"]["cohort_size"] == 3
    serialized = json.dumps(payload)
    for marker in ["stay_id", "subject_id", "hadm_id", "tableRows", '"series"']:
        assert marker not in serialized


def test_extraction_filter_preview_rejects_unsupported_filters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    client = TestClient(app)

    response = client.post(
        "/api/extraction/filter-preview",
        json={"filters": {"age_at_admission": {"min": 18}}},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"] == "unsupported_filter"
    assert detail["unsupported"][0]["id"] == "age_at_admission"
    assert "matched_modules" not in detail


def test_extraction_filter_options_fail_closed_without_registered_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    client = TestClient(app)

    no_active = client.post("/api/extraction/filter-options", json={})
    unregistered = client.post(
        "/api/extraction/filter-options",
        json={"source_path": str(tmp_path / "missing")},
    )

    assert no_active.status_code == 400
    assert no_active.json()["detail"]["error"] == "no_active_export"
    assert unregistered.status_code == 400
    assert unregistered.json()["detail"]["error"] == "source_not_registered"
    assert "path_hash" in unregistered.json()["detail"]


def test_local_job_cancel_endpoint_marks_running_job_cancelled() -> None:
    from easyicu.webserver.jobs import MANAGER

    client = TestClient(app)
    started = threading.Event()

    def runner(job):
        started.set()
        deadline = time.time() + 2
        while not job.cancel_requested and time.time() < deadline:
            time.sleep(0.01)
        return {"saw_cancel": job.cancel_requested}

    job = MANAGER.submit("cancel-smoke", runner)
    assert started.wait(timeout=1)

    cancel = client.post(f"/api/jobs/{job.id}/cancel", json={"reason": "test_cancel"})

    assert cancel.status_code == 200
    assert cancel.json()["cancel_request_accepted"] is True
    assert cancel.json()["cancel_requested"] is True

    snap = cancel.json()
    for _ in range(100):
        response = client.get(f"/api/jobs/{job.id}")
        assert response.status_code == 200
        snap = response.json()
        if snap["status"] != "running":
            break
        time.sleep(0.02)

    assert snap["status"] == "cancelled"
    assert snap["result"]["saw_cancel"] is True
    assert any(event.get("type") == "cancel_requested" for event in snap["events"])


def test_extraction_filter_options_do_not_read_full_export_tables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(export_dir), active=True, crossdb=True)

    def fail_read_frame(path: Path):
        raise AssertionError(f"filter metadata must not read full frame: {path}")

    original_read_csv = pd.read_csv

    def guarded_read_csv(*args, **kwargs):
        if "nrows" not in kwargs and "usecols" not in kwargs:
            raise AssertionError(f"unexpected full CSV read: {args[0]}")
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(dataio, "_read_export_frame", fail_read_frame)
    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    client = TestClient(app)

    response = client.post("/api/extraction/filter-options", json={})

    assert response.status_code == 200
    assert response.json()["summary"]["cohort_size"] == 3


class _ExportJob:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def emit(self, payload: dict[str, object]) -> None:
        self.events.append(payload)


def _patch_export_api(
    monkeypatch: pytest.MonkeyPatch, loaded: list[dict[str, object]]
) -> None:
    from contextlib import contextmanager
    import easyicu.api as api_module

    @contextmanager
    def fake_keep_cache(**_: object):
        yield None

    def fake_load_concepts(concepts, **kwargs):
        loaded.append({"concepts": concepts, "kwargs": kwargs})
        ids = (kwargs.get("patient_ids") or {}).get("stay_id", [])
        return pd.DataFrame({"stay_id": ids, "anchor_age": [65] * len(ids)})

    monkeypatch.setattr(api_module, "keep_cache", fake_keep_cache)
    monkeypatch.setattr(api_module, "load_concepts", fake_load_concepts)


def test_export_runner_applies_native_cohort_contract_to_patient_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.patient_filter as patient_filter_module

    class FakePatientFilter:
        def __init__(
            self, database: str, data_path: str, verbose: bool = False
        ) -> None:
            self.database = database
            self.data_path = data_path
            self.verbose = verbose

        def filter(self, **kwargs):
            assert kwargs["age_min"] == 40
            assert kwargs["age_max"] == 80
            assert kwargs["first_icu_stay"] is True
            assert kwargs["los_min"] == 24
            assert kwargs["return_dataframe"] is True
            return pd.DataFrame({"patient_id": [2]})

    loaded: list[dict[str, object]] = []
    _patch_export_api(monkeypatch, loaded)
    monkeypatch.setattr(patient_filter_module, "PatientFilter", FakePatientFilter)

    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["demographics"],
        export_format="csv",
        out_dir=str(tmp_path / "out"),
        max_patients=500,
        cohort={
            "preset": "adult_first",
            "age_min": 40,
            "age_max": 80,
            "min_icu_los_hours": 24,
            "observation_window_hours": 48,
            "exclude_readmissions": True,
        },
    )

    result = runner(_ExportJob())
    manifest = json.loads(
        (tmp_path / "out" / "_manifest.json").read_text(encoding="utf-8")
    )

    assert result["file_count"] == 1
    assert loaded[0]["kwargs"]["patient_ids"] == {"stay_id": [2]}
    assert loaded[0]["kwargs"]["win_length"] == "48h"
    assert manifest["cohort_contract"]["age_min"] == 40
    assert manifest["cohort_report"]["selected"] == 1


def test_export_runner_applies_icd_include_exclude_before_loading_concepts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.patient_filter as patient_filter_module

    pd.DataFrame({"stay_id": [1, 2, 3], "hadm_id": [10, 20, 30]}).to_csv(
        tmp_path / "icustays.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "hadm_id": [10, 20, 30, 30],
            "icd_code": ["A419", "J189", "A410", "R650"],
        }
    ).to_csv(tmp_path / "diagnoses_icd.csv", index=False)

    class FakePatientFilter:
        def __init__(
            self, database: str, data_path: str, verbose: bool = False
        ) -> None:
            pass

        def filter(self, **kwargs):
            return pd.DataFrame({"patient_id": [1, 2, 3]})

    loaded: list[dict[str, object]] = []
    _patch_export_api(monkeypatch, loaded)
    monkeypatch.setattr(patient_filter_module, "PatientFilter", FakePatientFilter)

    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["demographics"],
        export_format="csv",
        out_dir=str(tmp_path / "out"),
        cohort={
            "preset": "icd",
            "icd_enabled": True,
            "icd_include": "A41",
            "icd_exclude": "R65",
        },
    )

    runner(_ExportJob())
    manifest = json.loads(
        (tmp_path / "out" / "_manifest.json").read_text(encoding="utf-8")
    )

    assert loaded[0]["kwargs"]["patient_ids"] == {"stay_id": [1]}
    assert manifest["cohort_report"]["icd"]["include_matches"] == 2
    assert manifest["cohort_report"]["icd"]["exclude_matches"] == 1


def test_export_runner_keeps_legacy_all_icu_default_without_cohort_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.api as api_module

    loaded: list[dict[str, object]] = []
    _patch_export_api(monkeypatch, loaded)
    monkeypatch.setattr(
        api_module, "get_all_patient_ids", lambda *_, **__: ([3, 4], "stay_id")
    )

    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["demographics"],
        export_format="csv",
        out_dir=str(tmp_path / "out"),
        max_patients=2,
    )

    runner(_ExportJob())

    assert loaded[0]["kwargs"]["patient_ids"] == {"stay_id": [3, 4]}
    assert loaded[0]["kwargs"]["win_length"] == "720h"
    manifest = json.loads(
        (tmp_path / "out" / "_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["cohort_contract"]["preset"] == "all_icu"
    assert manifest["cohort_contract"]["observation_window_hours"] == 720


def test_export_runner_honors_module_specific_concept_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.api as api_module

    loaded: list[dict[str, object]] = []
    _patch_export_api(monkeypatch, loaded)
    monkeypatch.setattr(
        api_module, "get_all_patient_ids", lambda *_, **__: ([1, 2], "stay_id")
    )

    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["demographics", "vitals"],
        concepts={"demographics": ["age"], "vitals": ["hr", "map"]},
        export_format="csv",
        out_dir=str(tmp_path / "out"),
        max_patients=2,
    )

    result = runner(_ExportJob())
    manifest = json.loads(
        (tmp_path / "out" / "_manifest.json").read_text(encoding="utf-8")
    )
    readme = (tmp_path / "out" / "README.md").read_text(encoding="utf-8")

    assert result["file_count"] == 2
    assert [row["concepts"] for row in loaded] == [["age"], ["hr", "map"]]
    assert manifest["concept_selection"]["mode"] == "explicit"
    assert manifest["concept_selection"]["modules"] == {
        "demographics": ["age"],
        "vitals": ["hr", "map"],
    }
    assert manifest["files"][0]["concept_ids"] == ["age"]
    assert manifest["files"][1]["concept_ids"] == ["hr", "map"]
    assert "Concepts selected: `3`" in readme


def test_export_runner_rejects_unknown_selected_concepts(
    tmp_path: Path,
) -> None:
    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["vitals"],
        concepts={"vitals": ["hr", "not_a_real_concept"]},
        export_format="csv",
        out_dir=str(tmp_path / "out"),
    )

    with pytest.raises(dataio.ExportCohortError) as exc:
        runner(_ExportJob())

    assert exc.value.detail["error"] == "invalid_selected_concepts"
    assert exc.value.detail["invalid"] == ["vitals:not_a_real_concept"]
    assert not (tmp_path / "out").exists()


def test_export_runner_can_create_timestamped_run_folder_with_readme(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.api as api_module

    root = tmp_path / "exports"
    loaded: list[dict[str, object]] = []
    _patch_export_api(monkeypatch, loaded)
    monkeypatch.setattr(
        api_module, "get_all_patient_ids", lambda *_, **__: ([7, 8], "stay_id")
    )

    runner = dataio.make_export_runner(
        data_path=str(tmp_path / "source"),
        database="miiv",
        modules=["demographics"],
        export_format="parquet",
        out_dir=str(root),
        create_run_subdir=True,
        max_patients=2,
    )

    result = runner(_ExportJob())
    out = Path(result["out_dir"])

    assert out.parent == root
    assert out.name.startswith("easyicu_export_")
    assert out.name.endswith("_miiv_parquet")
    assert (out / "_manifest.json").exists()
    assert (out / "README.md").exists()
    assert result["manifest"] == "_manifest.json"
    assert result["readme"] == "README.md"
    manifest = json.loads((out / "_manifest.json").read_text(encoding="utf-8"))
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert manifest["export_folder"]["run_subdir"] is True
    assert manifest["export_folder"]["label"] == out.name
    assert manifest["cohort_contract"]["observation_window_hours"] == 720
    assert "Observation window: `720 hours`" in readme
    assert "`demographics.parquet`" in readme
    assert "No patient rows are included in this README" in readme


def test_export_runner_ignores_stale_icd_tokens_when_preset_is_not_icd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.patient_filter as patient_filter_module

    class FakePatientFilter:
        def __init__(
            self, database: str, data_path: str, verbose: bool = False
        ) -> None:
            pass

        def filter(self, **kwargs):
            return pd.DataFrame({"patient_id": [1, 2]})

    loaded: list[dict[str, object]] = []
    _patch_export_api(monkeypatch, loaded)
    monkeypatch.setattr(patient_filter_module, "PatientFilter", FakePatientFilter)

    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["demographics"],
        export_format="csv",
        out_dir=str(tmp_path / "out"),
        cohort={
            "preset": "adult_first",
            "icd_enabled": False,
            "icd_include": "A41",
            "icd_exclude": "R65",
        },
    )

    runner(_ExportJob())
    manifest = json.loads(
        (tmp_path / "out" / "_manifest.json").read_text(encoding="utf-8")
    )

    assert loaded[0]["kwargs"]["patient_ids"] == {"stay_id": [1, 2]}
    assert loaded[0]["kwargs"]["win_length"] == "720h"
    assert manifest["cohort_contract"]["icd_include"] == []
    assert manifest["cohort_contract"]["observation_window_hours"] == 720
    assert manifest["cohort_report"]["applied_filters"] == ["demographics"]


@pytest.mark.parametrize(
    ("preset", "concepts", "positive_column"),
    [
        ("sepsis3", ["sep3_sofa2"], "sep3_sofa2"),
        ("aki", ["aki"], "aki"),
        ("ventilation", ["mech_vent", "vent_ind"], "mech_vent"),
        ("vasopressor", ["vaso_ind"], "vaso_ind"),
        (
            "respiratory",
            ["adv_resp", "mech_vent", "vent_ind", "pafi", "safi"],
            "adv_resp",
        ),
    ],
)
def test_export_runner_applies_concept_derived_cohort_prefilter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preset: str,
    concepts: list[str],
    positive_column: str,
) -> None:
    from contextlib import contextmanager
    import easyicu.api as api_module
    import easyicu.patient_filter as patient_filter_module

    class FakePatientFilter:
        def __init__(
            self, database: str, data_path: str, verbose: bool = False
        ) -> None:
            pass

        def filter(self, **kwargs):
            return pd.DataFrame({"patient_id": [1, 2, 3]})

    loaded: list[dict[str, object]] = []

    @contextmanager
    def fake_keep_cache(**_: object):
        yield None

    def fake_load_concepts(concepts, **kwargs):
        loaded.append({"concepts": concepts, "kwargs": kwargs})
        if concepts == loaded_concepts:
            return pd.DataFrame({"stay_id": [1, 2, 3], positive_column: [0, 1, True]})
        ids = (kwargs.get("patient_ids") or {}).get("stay_id", [])
        return pd.DataFrame({"stay_id": ids, "anchor_age": [65] * len(ids)})

    loaded_concepts = list(concepts)
    monkeypatch.setattr(patient_filter_module, "PatientFilter", FakePatientFilter)
    monkeypatch.setattr(api_module, "keep_cache", fake_keep_cache)
    monkeypatch.setattr(api_module, "load_concepts", fake_load_concepts)

    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["demographics"],
        export_format="csv",
        out_dir=str(tmp_path / "out"),
        cohort={"preset": preset, "observation_window_hours": 72},
    )

    job = _ExportJob()
    runner(job)
    manifest = json.loads(
        (tmp_path / "out" / "_manifest.json").read_text(encoding="utf-8")
    )

    assert loaded[0]["concepts"] == concepts
    assert loaded[0]["kwargs"]["patient_ids"] == {"stay_id": [1, 2, 3]}
    assert loaded[0]["kwargs"]["win_length"] == "72h"
    assert loaded[1]["kwargs"]["patient_ids"] == {"stay_id": [2, 3]}
    assert manifest["cohort_report"]["applied_filters"] == [
        "demographics",
        "concept_prefilter",
    ]
    assert manifest["cohort_report"]["concept_matches"] == 2
    stages = [
        event.get("stage") for event in job.events if event.get("phase") == "cohort"
    ]
    assert "concept_prefilter" in stages
    assert "cohort_selected" in stages
    assert all(
        "patient_ids" not in event and "stay_id" not in event for event in job.events
    )


def test_export_runner_fails_closed_for_unsupported_native_cohort_preset(
    tmp_path: Path,
) -> None:
    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["demographics"],
        out_dir=str(tmp_path / "out"),
        cohort={"preset": "obesity"},
    )

    with pytest.raises(dataio.ExportCohortError) as exc:
        runner(_ExportJob())

    assert exc.value.detail["error"] == "unsupported_cohort_preset"


def test_crossdb_summary_requires_two_valid_exports_and_compares_metrics(
    tmp_path: Path,
) -> None:
    miiv = _write_csv_export(tmp_path / "miiv", database="miiv")
    eicu = _write_csv_export(tmp_path / "eicu", database="eicu")

    result = summarize_crossdb_workspaces([str(miiv), str(eicu)])

    assert result["ok"] is True
    assert result["source_count"] == 2
    assert [source["label"] for source in result["sources"]] == ["MIIV", "EICU"]
    assert result["shared_modules"] == [
        "demographics",
        "outcome",
        "sepsis3_sofa2",
        "sofa2_score",
        "vitals",
    ]
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


def test_crossdb_summary_fails_closed_when_core_modules_are_not_comparable(
    tmp_path: Path,
) -> None:
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
    core_check = next(
        check for check in gate["checks"] if check["id"] == "core_modules_shared"
    )
    assert core_check["passed"] is False
    assert core_check["missing_modules"] == ["outcome"]


def test_crossdb_summary_endpoint_is_fail_closed_until_two_exports(
    tmp_path: Path,
) -> None:
    miiv = _write_csv_export(tmp_path / "miiv", database="miiv")
    eicu = _write_csv_export(tmp_path / "eicu", database="eicu")
    no_outcome = _write_csv_export(tmp_path / "no_outcome", database="aumc")
    _drop_export_module(no_outcome, "outcome")
    client = TestClient(app)

    one = client.post("/api/workspaces/crossdb-summary", json={"paths": [str(miiv)]})
    two = client.post(
        "/api/workspaces/crossdb-summary", json={"paths": [str(miiv), str(eicu)]}
    )
    invalid = client.post(
        "/api/workspaces/crossdb-summary",
        json={"paths": [str(miiv), str(tmp_path / "missing")]},
    )
    incompatible = client.post(
        "/api/workspaces/crossdb-summary", json={"paths": [str(miiv), str(no_outcome)]}
    )

    assert one.status_code == 400
    assert one.json()["detail"]["error"] == "need_two_exports"
    assert two.status_code == 200
    assert two.json()["source_count"] == 2
    assert invalid.status_code == 400
    assert invalid.json()["detail"]["error"] == "invalid_export"
    assert incompatible.status_code == 400
    assert incompatible.json()["detail"]["error"] == "crossdb_incompatible"
    assert (
        incompatible.json()["detail"]["compatibility_gate"]["status"] == "incompatible"
    )


def test_crossdb_review_summary_uses_registered_sources_without_row_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    miiv = _write_csv_export(tmp_path / "miiv", database="miiv")
    eicu = _write_csv_export(tmp_path / "eicu", database="eicu")
    for export_dir in (miiv, eicu):
        demographics = pd.read_csv(export_dir / "demographics.csv")
        demographics["subject_id"] = [101, 102, 103]
        demographics["hadm_id"] = [201, 202, 203]
        demographics.to_csv(export_dir / "demographics.csv", index=False)
    source_store.register_source(
        str(miiv), label="Primary MIIV", active=True, crossdb=True
    )
    source_store.register_source(
        str(eicu), label="Comparator eICU", active=False, crossdb=True
    )
    client = TestClient(app)

    response = client.post("/api/crossdb-review/summary", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "real"
    assert payload["demo"] is False
    assert payload["source_count"] == 2
    assert [source["label"] for source in payload["sources"]] == [
        "Primary MIIV",
        "Comparator eICU",
    ]
    assert all("path" not in source for source in payload["sources"])
    assert all(len(source["path_hash"]) == 12 for source in payload["sources"])
    assert payload["shared_modules"] == [
        "demographics",
        "outcome",
        "sepsis3_sofa2",
        "sofa2_score",
        "vitals",
    ]
    gate = payload["compatibility_gate"]
    assert gate["status"] == "compatible"
    assert gate["comparison_mode"] == "descriptive_only"
    assert gate["matched_cohort"] is False
    assert gate["inferential_statistics_allowed"] is False
    rows = {row["key"]: row for row in payload["rows"]}
    assert rows["cohort_size"]["values"] == [3, 3]
    assert rows["mortality_pct"]["values"] == [33.3, 33.3]
    assert rows["age_mean"]["values"] == [60.0, 60.0]
    availability = {row["module"]: row for row in payload["availability"]}
    assert availability["demographics"]["shared"] is True
    assert availability["demographics"]["values"][0]["coverage_pct"] == 100.0
    density = {row["module"]: row for row in payload["feature_density"]}
    assert density["demographics"]["feature_count"] == 2
    assert [row["feature"] for row in density["demographics"]["features"]] == [
        "age",
        "sex",
    ]
    vitals_features = {row["feature"]: row for row in density["vitals"]["features"]}
    assert {"hr", "map", "spo2", "temp"} <= set(vitals_features)
    assert vitals_features["hr"]["values"][0]["density_per_100_entities"] == 100.0
    assert vitals_features["hr"]["values"][0]["coverage_pct"] == 66.7
    distributions = {row["module"]: row for row in payload["feature_distributions"]}
    assert distributions["demographics"]["feature_count"] == 2
    age_dist = next(
        row
        for row in distributions["demographics"]["features"]
        if row["feature"] == "age"
    )
    assert age_dist["values"][0]["kind"] == "numeric"
    assert age_dist["values"][0]["non_null"] == 3
    assert len(age_dist["values"][0]["points"]) >= 3
    sex_dist = next(
        row
        for row in distributions["demographics"]["features"]
        if row["feature"] == "sex"
    )
    assert sex_dist["values"][0]["kind"] == "categorical"
    assert sex_dist["values"][0]["categories"]
    hr_dist = next(
        row for row in distributions["vitals"]["features"] if row["feature"] == "hr"
    )
    assert hr_dist["values"][0]["kind"] == "numeric"
    assert len(hr_dist["values"][0]["points"]) >= 3
    assert payload["provenance"]["payload_scope"] == "cross_database_aggregate_only"
    assert payload["privacy"]["raw_rows_returned"] is False
    assert any(item["id"] == "matched_cohort" for item in payload["blocked_features"])
    serialized = json.dumps(payload)
    for marker in ["subject_id", "hadm_id", "tableRows", "stay_id", '"series"']:
        assert marker not in serialized


def test_crossdb_raw_distribution_uses_real_loader_without_row_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "databases"
    (root / "mimiciv").mkdir(parents=True)
    (root / "eicu").mkdir()

    def fake_loader(
        *,
        data_root: str,
        concepts: list[str],
        databases: list[str],
        max_patients: int,
        sample_size: int,
    ) -> dict[str, pd.DataFrame]:
        assert data_root == str(root)
        assert databases == ["miiv", "eicu"]
        assert concepts == ["hr", "sbp"]
        assert max_patients == 40
        assert sample_size == 100
        return {
            "miiv": pd.DataFrame(
                {
                    "concept": ["hr"] * 20 + ["sbp"] * 20,
                    "value": list(range(70, 90)) + list(range(110, 130)),
                }
            ),
            "eicu": pd.DataFrame(
                {
                    "concept": ["hr"] * 20 + ["sbp"] * 20,
                    "value": list(range(80, 100)) + list(range(120, 140)),
                }
            ),
        }

    monkeypatch.setattr(crossdb_review, "_load_raw_feature_data", fake_loader)
    client = TestClient(app)

    response = client.post(
        "/api/crossdb-review/raw-distribution",
        json={
            "data_root": str(root),
            "databases": ["miiv", "eicu"],
            "features": ["hr", "sbp"],
            "max_patients": 40,
            "sample_size": 100,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["source_type"] == "raw_database_root"
    assert payload["source_count"] == 2
    assert payload["provenance"]["computed_from"] == [
        "raw_icu_data_root",
        "easyicu.load_concepts",
        "MultiDatabaseDistribution",
        "bounded_feature_distribution_aggregates",
    ]
    assert all("path" not in source for source in payload["sources"])
    modules = {row["module"]: row for row in payload["feature_distributions"]}
    assert "vitals" in modules
    hr = next(row for row in modules["vitals"]["features"] if row["feature"] == "hr")
    assert hr["shared"] is True
    assert hr["values"][0]["kind"] == "numeric"
    assert len(hr["values"][0]["points"]) >= 3
    assert payload["privacy"]["raw_rows_returned"] is False
    serialized = json.dumps(payload)
    for marker in ["subject_id", "hadm_id", "tableRows", "stay_id", '"series"']:
        assert marker not in serialized


def test_crossdb_raw_root_scan_reports_detected_missing_and_unrecognized(
    tmp_path: Path,
) -> None:
    root = tmp_path / "databases"
    (root / "mimiciv").mkdir(parents=True)
    (root / "eicu").mkdir()
    (root / "custom_named_icu").mkdir()
    client = TestClient(app)

    response = client.post(
        "/api/crossdb-review/raw-root-scan",
        json={"data_root": str(root), "databases": ["miiv", "eicu", "sic"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["source_type"] == "raw_database_root"
    assert payload["runnable"] is True
    assert payload["detected_selected_count"] == 2
    assert set(payload["detected_databases"]) == {"miiv", "eicu"}
    assert [row["key"] for row in payload["missing_selected"]] == ["sic"]
    assert "custom_named_icu" in payload["unrecognized_folders"]
    assert "miiv" in payload["aliases"]
    assert payload["aliases"]["miiv"]["aliases"][:2] == ["mimiciv", "mimic-iv"]
    serialized = json.dumps(payload)
    assert str(root) not in serialized
    assert payload["privacy"]["raw_rows_returned"] is False


def test_crossdb_raw_root_scan_blocks_until_two_selected_databases(
    tmp_path: Path,
) -> None:
    root = tmp_path / "databases"
    (root / "mimiciv").mkdir(parents=True)
    client = TestClient(app)

    response = client.post(
        "/api/crossdb-review/raw-root-scan",
        json={"data_root": str(root), "databases": ["miiv", "eicu"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["runnable"] is False
    assert payload["detected_selected_count"] == 1
    assert payload["detected_databases"] == ["miiv"]
    assert [row["key"] for row in payload["missing_selected"]] == ["eicu"]


def test_crossdb_raw_distribution_job_streams_progress_and_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "databases"
    (root / "mimiciv").mkdir(parents=True)
    (root / "eicu").mkdir()

    def fake_loader(
        *,
        data_root: str,
        concepts: list[str],
        databases: list[str],
        max_patients: int,
        sample_size: int,
    ) -> dict[str, pd.DataFrame]:
        assert data_root == str(root)
        assert concepts == ["hr", "sbp"]
        assert databases == ["miiv", "eicu"]
        assert max_patients == 40
        assert sample_size == 100
        return {
            "miiv": pd.DataFrame(
                {"concept": ["hr", "hr", "sbp", "sbp"], "value": [70, 90, 110, 130]}
            ),
            "eicu": pd.DataFrame(
                {"concept": ["hr", "hr", "sbp", "sbp"], "value": [80, 100, 120, 140]}
            ),
        }

    monkeypatch.setattr(crossdb_review, "_load_raw_feature_data", fake_loader)
    client = TestClient(app)

    response = client.post(
        "/api/jobs/crossdb-raw-distribution",
        json={
            "data_root": str(root),
            "databases": ["miiv", "eicu"],
            "features": ["hr", "sbp"],
            "max_patients": 40,
            "sample_size": 100,
        },
    )

    assert response.status_code == 200
    assert response.json()["kind"] == "crossdb-raw-distribution"
    job_id = response.json()["job_id"]
    snap = response.json()
    for _ in range(100):
        poll = client.get(f"/api/jobs/{job_id}")
        assert poll.status_code == 200
        snap = poll.json()
        if snap["status"] != "running":
            break
        time.sleep(0.02)

    assert snap["status"] == "done"
    result = snap["result"]
    assert result["ok"] is True
    assert result["source_type"] == "raw_database_root"
    assert result["source_count"] == 2
    phases = [
        event.get("phase")
        for event in snap["events"]
        if event.get("type") == "progress"
    ]
    assert {"resolving", "loading", "finalizing"}.issubset(set(phases))
    loading_events = [
        event
        for event in snap["events"]
        if event.get("type") == "progress" and event.get("phase") == "loading"
    ]
    assert loading_events
    assert loading_events[-1]["max_patients"] == 40
    assert loading_events[-1]["sample_size"] == 100
    assert "max 40 entities/database" in loading_events[-1]["message"]
    assert "max 100 values/feature" in loading_events[-1]["message"]
    serialized = json.dumps(result)
    for marker in ["subject_id", "hadm_id", "tableRows", "stay_id", '"series"']:
        assert marker not in serialized


def test_crossdb_raw_distribution_job_can_be_cancelled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "databases"
    (root / "mimiciv").mkdir(parents=True)
    (root / "eicu").mkdir()

    def slow_loader(**_: object) -> dict[str, pd.DataFrame]:
        time.sleep(0.25)
        return {
            "miiv": pd.DataFrame({"concept": ["hr", "hr"], "value": [70, 90]}),
            "eicu": pd.DataFrame({"concept": ["hr", "hr"], "value": [80, 100]}),
        }

    monkeypatch.setattr(crossdb_review, "_load_raw_feature_data", slow_loader)
    client = TestClient(app)

    started = client.post(
        "/api/jobs/crossdb-raw-distribution",
        json={
            "data_root": str(root),
            "databases": ["miiv", "eicu"],
            "features": ["hr"],
            "max_patients": 40,
            "sample_size": 100,
        },
    )
    assert started.status_code == 200
    job_id = started.json()["job_id"]
    cancel = client.post(f"/api/jobs/{job_id}/cancel", json={"reason": "test_cancel"})

    assert cancel.status_code == 200
    assert cancel.json()["cancel_request_accepted"] is True
    snap = cancel.json()
    for _ in range(100):
        poll = client.get(f"/api/jobs/{job_id}")
        assert poll.status_code == 200
        snap = poll.json()
        if snap["status"] != "running":
            break
        time.sleep(0.02)

    assert snap["status"] == "cancelled"
    assert snap["result"]["cancelled"] is True
    assert snap["result"]["cancelled_at"] in {"resolving", "loading"}
    assert any(event.get("type") == "cancel_requested" for event in snap["events"])


def test_crossdb_demo_distribution_uses_legacy_simulated_frames_without_row_payload() -> (
    None
):
    client = TestClient(app)

    response = client.post(
        "/api/crossdb-review/demo-distribution",
        json={
            "databases": ["miiv", "eicu", "aumc", "hirid", "mimic", "sic"],
            "records_per_feature": 80,
            "features": ["hr", "sbp", "map", "temp", "spo2", "lact", "sofa2"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "demo"
    assert payload["demo"] is True
    assert payload["source_type"] == "legacy_simulated_multidb_feature_frames"
    assert payload["source_count"] == 6
    assert payload["provenance"]["computed_from"] == [
        "legacy_streamlit_generate_mock_multidb_data",
        "seeded_clinical_feature_specs",
        "bounded_feature_distribution_aggregates",
    ]
    assert payload["provenance"]["records_per_feature"] == 80
    modules = {row["module"]: row for row in payload["feature_distributions"]}
    assert "vitals" in modules
    assert "blood_gas" in modules or "sofa2_score" in modules
    hr = next(row for row in modules["vitals"]["features"] if row["feature"] == "hr")
    assert hr["shared"] is True
    assert len(hr["values"]) == 6
    assert all(value["kind"] == "numeric" for value in hr["values"])
    assert all(len(value["points"]) >= 10 for value in hr["values"])
    assert payload["privacy"]["raw_rows_returned"] is False
    serialized = json.dumps(payload)
    for marker in ["subject_id", "hadm_id", "tableRows", "stay_id", '"series"']:
        assert marker not in serialized


def test_crossdb_demo_all_catalog_scope_resolves_every_module_and_feature() -> None:
    features = crossdb_review._resolve_demo_features({"feature_scope": "all_catalog"})
    assert len(features) == len(concept_catalog.CONCEPT_DICTIONARY)
    assert features[:7] == concept_catalog.CONCEPT_GROUPS_INTERNAL["sofa2_score"]

    module_map = {
        feature: module
        for module, module_features in concept_catalog.CONCEPT_GROUPS_INTERNAL.items()
        for feature in module_features
    }
    assert {module_map[feature] for feature in features} == set(
        concept_catalog.CONCEPT_GROUPS_INTERNAL
    )

    frames = crossdb_review._generate_demo_multidb_feature_frames(
        databases=["miiv", "eicu"],
        features=features,
        records_per_feature=24,
    )
    assert int(frames["miiv"]["concept"].nunique()) == len(features)
    assert int(frames["eicu"]["concept"].nunique()) == len(features)


def test_crossdb_raw_distribution_fails_closed_until_two_raw_databases(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "databases"
    (root / "mimiciv").mkdir(parents=True)
    monkeypatch.setattr(crossdb_review, "_COMMON_RAW_ROOT_CANDIDATES", [])
    client = TestClient(app)

    response = client.post(
        "/api/crossdb-review/raw-distribution",
        json={
            "data_root": str(root),
            "databases": ["miiv", "eicu"],
            "features": ["hr"],
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "need_two_raw_databases"


def test_crossdb_review_summary_fails_closed_until_two_registered_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    client = TestClient(app)

    none = client.post("/api/crossdb-review/summary", json={})
    miiv = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(miiv), active=True, crossdb=True)
    one = client.post("/api/crossdb-review/summary", json={})
    unregistered = client.post(
        "/api/crossdb-review/summary",
        json={"paths": [str(miiv), str(tmp_path / "missing")]},
    )

    assert none.status_code == 400
    assert none.json()["detail"]["error"] == "need_two_exports"
    assert one.status_code == 400
    assert one.json()["detail"]["error"] == "need_two_exports"
    assert one.json()["detail"]["source_count"] == 1
    assert unregistered.status_code == 400
    assert unregistered.json()["detail"]["error"] == "source_not_registered"
    assert "path_hash" in unregistered.json()["detail"]
    assert "path" not in unregistered.json()["detail"]


def test_crossdb_review_summary_rejects_incompatible_missing_core_modules(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    miiv = _write_csv_export(tmp_path / "miiv", database="miiv")
    eicu = _write_csv_export(tmp_path / "eicu", database="eicu")
    _drop_export_module(eicu, "outcome")
    source_store.register_source(str(miiv), active=True, crossdb=True)
    source_store.register_source(str(eicu), active=False, crossdb=True)
    client = TestClient(app)

    response = client.post("/api/crossdb-review/summary", json={})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"] == "crossdb_incompatible"
    assert detail["compatibility_gate"]["status"] == "incompatible"
    core_check = next(
        check
        for check in detail["compatibility_gate"]["checks"]
        if check["id"] == "core_modules_shared"
    )
    assert core_check["missing_modules"] == ["outcome"]
    serialized = json.dumps(detail)
    for marker in ["subject_id", "hadm_id", "tableRows", "stay_id", '"series"']:
        assert marker not in serialized


def test_crossdb_review_summary_rejects_unsupported_filters_and_statistics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    miiv = _write_csv_export(tmp_path / "miiv", database="miiv")
    eicu = _write_csv_export(tmp_path / "eicu", database="eicu")
    source_store.register_source(str(miiv), active=True, crossdb=True)
    source_store.register_source(str(eicu), active=False, crossdb=True)
    client = TestClient(app)

    row_filter = client.post(
        "/api/crossdb-review/summary",
        json={"filters": {"row_level_filters": {"age": [18, 80]}}},
    )
    p_value = client.post(
        "/api/crossdb-review/summary",
        json={"statistics": ["p_value"]},
    )
    matched = client.post(
        "/api/crossdb-review/summary",
        json={"matched_cohort": True},
    )

    assert row_filter.status_code == 400
    assert row_filter.json()["detail"]["error"] == "unsupported_filter"
    assert row_filter.json()["detail"]["unsupported"][0]["id"] == "row_level_filters"
    assert "rows" not in row_filter.json()["detail"]
    assert p_value.status_code == 400
    assert p_value.json()["detail"]["error"] == "unsupported_statistic"
    assert p_value.json()["detail"]["unsupported"][0]["id"] == "p_value"
    assert "rows" not in p_value.json()["detail"]
    assert matched.status_code == 400
    assert matched.json()["detail"]["error"] == "unsupported_filter"
    assert matched.json()["detail"]["unsupported"][0]["id"] == "matched_cohort"


def test_crossdb_review_summary_does_not_read_full_export_tables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    miiv = _write_csv_export(tmp_path / "miiv", database="miiv")
    eicu = _write_csv_export(tmp_path / "eicu", database="eicu")
    source_store.register_source(str(miiv), active=True, crossdb=True)
    source_store.register_source(str(eicu), active=False, crossdb=True)

    def fail_read_frame(path: Path):
        raise AssertionError(f"crossdb review must not read full frame: {path}")

    original_read_csv = pd.read_csv

    def guarded_read_csv(*args, **kwargs):
        if "nrows" not in kwargs and "usecols" not in kwargs:
            raise AssertionError(f"unexpected full CSV read: {args[0]}")
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(dataio, "_read_export_frame", fail_read_frame)
    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    client = TestClient(app)

    response = client.post("/api/crossdb-review/summary", json={})

    assert response.status_code == 200
    assert response.json()["source_count"] == 2


def test_export_source_registry_describes_and_persists_sources(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    miiv = _write_csv_export(tmp_path / "miiv", database="miiv")
    eicu = _write_csv_export(tmp_path / "eicu", database="eicu")

    desc = describe_export_source(str(miiv))
    assert desc["ok"] is True
    assert desc["summary"] == {
        "stays": 3,
        "modules": 5,
        "file_count": 5,
        "total_rows": 14,
    }

    registered = source_store.register_source(str(miiv), active=True, crossdb=True)
    saved = source_store.save_registry(
        {
            "sources": [{"path": str(eicu), "label": "External eICU"}],
            "crossdb_paths": [str(miiv), str(eicu)],
        }
    )

    assert registered["ok"] is True
    assert saved["active_path"] == str(miiv)
    assert saved["crossdb_paths"] == [str(miiv), str(eicu)]
    assert [s["label"] for s in saved["sources"]] == ["External eICU", "MIIV"]


def test_export_source_registry_endpoint_registers_and_rejects_invalid_paths(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    miiv = _write_csv_export(tmp_path / "miiv", database="miiv")
    client = TestClient(app)

    ok = client.post("/api/workspaces/register", json={"path": str(miiv)})
    bad = client.post(
        "/api/workspaces/register", json={"path": str(tmp_path / "missing")}
    )
    reg = client.get("/api/workspaces/registry")

    assert ok.status_code == 200
    assert ok.json()["active_path"] == str(miiv)
    assert bad.status_code == 400
    assert bad.json()["detail"]["error"] == "not_a_directory"
    assert reg.status_code == 200
    assert reg.json()["sources"][0]["summary"]["stays"] == 3


def test_export_source_registry_autodiscovery_skips_unreadable_children(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    export_dir = tmp_path / "exports"
    bad = export_dir / "bad-local-folder"
    bad.mkdir(parents=True)
    monkeypatch.setattr(
        source_store.settings_store,
        "load_settings",
        lambda: {"export_dir": str(export_dir)},
    )

    def describe_or_raise(path: str) -> dict:
        if Path(path) == bad:
            raise PermissionError("cannot inspect this folder")
        return {"ok": False}

    monkeypatch.setattr(
        source_store.dataio, "describe_export_source", describe_or_raise
    )

    result = source_store.load_registry()

    assert result["ok"] is True
    assert result["sources"] == []
    assert result["active_path"] is None


def test_export_source_registry_autodiscovery_includes_export_root_and_children(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    export_root = _write_csv_export(tmp_path / "easyicu_export", database="mock")
    child_export = _write_csv_export(
        export_root / "easyicu_export_20260627_miiv_parquet", database="miiv"
    )
    monkeypatch.setattr(
        source_store.settings_store,
        "load_settings",
        lambda: {"export_dir": str(export_root)},
    )

    result = source_store.load_registry()
    paths = {source["path"] for source in result["sources"]}

    assert str(export_root) in paths
    assert str(child_export) in paths
    assert result["active_path"] in paths


def test_export_source_registry_promotes_latest_configured_export_for_old_auto_active(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    export_root = _write_csv_export(tmp_path / "easyicu_export", database="mock")
    old_child = _write_csv_export(export_root / "mock_20260424", database="mock")
    manifest = json.loads((export_root / "_manifest.json").read_text())
    manifest["generated"] = "2026-06-27T08:00:00"
    (export_root / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    old_manifest = json.loads((old_child / "_manifest.json").read_text())
    old_manifest["generated"] = "2026-04-24T19:16:51"
    (old_child / "_manifest.json").write_text(
        json.dumps(old_manifest), encoding="utf-8"
    )
    monkeypatch.setattr(
        source_store.settings_store,
        "load_settings",
        lambda: {"export_dir": str(export_root)},
    )
    source_store._write_raw(
        {
            "sources": [{"path": str(old_child), "label": "MOCK"}],
            "active_path": str(old_child),
            "crossdb_paths": [],
            "removed_paths": [],
        }
    )

    result = source_store.load_registry()

    assert result["active_path"] == str(export_root)
    assert result["active_source"] == "auto_discovered"


def test_export_source_registry_respects_manual_active_inside_configured_export(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    export_root = _write_csv_export(tmp_path / "easyicu_export", database="mock")
    child = _write_csv_export(export_root / "manual_child", database="mock")
    monkeypatch.setattr(
        source_store.settings_store,
        "load_settings",
        lambda: {"export_dir": str(export_root)},
    )
    saved = source_store.save_registry(
        {"sources": [str(export_root), str(child)], "active_path": str(child)}
    )

    result = source_store.load_registry()

    assert saved["active_source"] == "manual"
    assert result["active_path"] == str(child)
    assert result["active_source"] == "manual"


def test_extraction_job_registers_finished_export_as_active(
    tmp_path: Path,
    monkeypatch,
) -> None:
    export_dir = _write_csv_export(tmp_path / "finished_export", database="miiv")
    calls: list[dict[str, object]] = []

    def fake_make_export_runner(**_kwargs):
        def runner(_job):
            return {
                "out_dir": str(export_dir),
                "manifest": "_manifest.json",
                "files": [],
                "file_count": 0,
                "total_rows": 0,
            }

        return runner

    def fake_register_source(path, label=None, active=True, crossdb=True):
        calls.append(
            {"path": path, "label": label, "active": active, "crossdb": crossdb}
        )
        return {"ok": True, "active_path": path, "sources": [{"path": path}]}

    monkeypatch.setattr(app_module.dataio, "make_export_runner", fake_make_export_runner)
    monkeypatch.setattr(app_module.source_store, "register_source", fake_register_source)
    client = TestClient(app)

    submitted = client.post(
        "/api/jobs/extract",
        json={"path": str(tmp_path), "database": "miiv", "label": "Latest export"},
    )
    assert submitted.status_code == 200
    job_id = submitted.json()["job_id"]
    snapshot = None
    for _ in range(50):
        snapshot = client.get(f"/api/jobs/{job_id}").json()
        if snapshot["status"] != "running":
            break
        time.sleep(0.01)

    assert snapshot is not None
    assert snapshot["status"] == "done"
    assert calls == [
        {
            "path": str(export_dir),
            "label": "Latest export",
            "active": True,
            "crossdb": True,
        }
    ]
    assert snapshot["result"]["registered_source"]["active_path"] == str(export_dir)


def test_export_source_registry_rename_and_remove_are_metadata_only(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    miiv = _write_csv_export(tmp_path / "miiv", database="miiv")
    eicu = _write_csv_export(tmp_path / "eicu", database="eicu")
    monkeypatch.setattr(
        source_store, "_autodiscovered_paths", lambda: [str(miiv), str(eicu)]
    )

    renamed = source_store.rename_source(str(miiv), "Primary MIIV")
    removed = source_store.remove_source(str(miiv))
    reloaded = source_store.load_registry()

    assert renamed["ok"] is True
    assert renamed["action"] == "renamed_source_metadata"
    assert renamed["disk_touched"] is False
    assert any(
        s["path"] == str(miiv) and s["label"] == "Primary MIIV"
        for s in renamed["sources"]
    )
    assert removed["ok"] is True
    assert removed["action"] == "unregistered_source_only"
    assert removed["disk_deleted"] is False
    assert removed["removed_path"] == str(miiv)
    assert miiv.exists()
    assert (miiv / "demographics.csv").exists()
    assert all(s["path"] != str(miiv) for s in reloaded["sources"])
    assert reloaded["active_path"] == str(eicu)

    restored = source_store.register_source(
        str(miiv), label="Restored MIIV", active=True, crossdb=True
    )

    assert restored["active_path"] == str(miiv)
    assert any(
        s["path"] == str(miiv) and s["label"] == "Restored MIIV"
        for s in restored["sources"]
    )


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
    rename = client.post(
        "/api/workspaces/rename", json={"path": str(miiv), "label": "Primary source"}
    )
    remove = client.post("/api/workspaces/remove", json={"path": str(miiv)})
    missing = client.post(
        "/api/workspaces/remove", json={"path": str(tmp_path / "missing")}
    )

    assert rename.status_code == 200
    assert any(
        s["path"] == str(miiv) and s["label"] == "Primary source"
        for s in rename.json()["sources"]
    )
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
    assert [
        event["type"] for event in snapshot["events"] if event["type"] != "end"
    ] == [
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
    assert gate_checks["no_patient_rows_persisted"]["scanned_artifacts"] == len(
        AGENT_PREFLIGHT_ARTIFACTS
    )
    assert gate_checks["no_patient_rows_persisted"]["row_level_markers"] == []
    artifact_paths = [Path(item["path"]) for item in result["artifacts"]]
    assert {path.name for path in artifact_paths} == AGENT_PREFLIGHT_ARTIFACTS
    for path in artifact_paths:
        text = path.read_text(encoding="utf-8")
        assert "tableRows" not in text
        assert '"series"' not in text
        assert '"patient"' not in text
        assert '"stay_id"' not in text
    artifact_payloads = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in artifact_paths
    }
    assert artifact_payloads["table1_summary.json"]["status"] == "ok"
    assert {
        row["feature"] for row in artifact_payloads["table1_summary.json"]["variables"]
    } >= {
        "age",
        "sofa2",
        "los_icu",
    }
    missing_rows = artifact_payloads["missingness_audit.json"]["rows"]
    assert artifact_payloads["missingness_audit.json"]["status"] == "ok"
    assert {row["feature"] for row in missing_rows} >= {"age", "death", "sofa2", "hr"}
    assert all(
        row["coverage_basis"] == "entity_non_missing_presence" for row in missing_rows
    )
    roc = artifact_payloads["roc_curve.json"]
    assert roc["kind"] == "roc_curve"
    assert roc["status"] == "ok"
    assert roc["points"]
    calibration = artifact_payloads["calibration_curve.json"]
    assert calibration["kind"] == "calibration_curve"
    assert calibration["status"] in {"ok", "not_available"}
    ledger = json.loads(
        (Path(result["project_dir"]) / "evidence_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    assert ledger["privacy"]["patient_rows_persisted"] is False
    assert ledger["privacy"]["artifact_scan"]["passed"] is True
    assert ledger["privacy"]["artifact_scan"]["scanned_artifacts"] == len(
        AGENT_PREFLIGHT_ARTIFACTS
    )


def test_agent_run_large_export_preflight_uses_registry_metadata_fast_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "large_miiv", database="miiv")
    manifest_path = export_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["patient_count"] = 94_458
    for row in manifest["files"]:
        row["rows"] = 2_000_000
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    client = TestClient(app)

    start = client.post(
        "/api/jobs/agent-run",
        json={
            "study_id": "large-export-preflight",
            "mode": "analysis",
            "question": "metadata-only preflight",
            "project_root": str(tmp_path / "projects"),
        },
    )

    assert start.status_code == 200
    snapshot = _wait_for_job(client, start.json()["job_id"])
    assert snapshot["status"] == "done"
    result = snapshot["result"]
    assert result["summary"]["stays"] == 94_458
    assert result["summary"]["total_rows"] == 10_000_000
    assert result["summary"]["snapshot_basis"] == "registry_metadata"
    assert result["summary"]["artifact_scope"] == "metadata_only_large_export_preflight"
    assert result["cohort"]["status"] == "metadata_only"
    assert result["gate"]["status"] == "analysis_only"

    gate_checks = {check["id"]: check for check in result["gate"]["checks"]}
    assert gate_checks["quality_audited"]["passed"] is True
    assert gate_checks["quality_audited"]["coverage_bases"] == [
        "manifest_file_inventory"
    ]
    assert gate_checks["no_patient_rows_persisted"]["passed"] is True

    artifact_paths = [Path(item["path"]) for item in result["artifacts"]]
    assert {path.name for path in artifact_paths} == AGENT_PREFLIGHT_ARTIFACTS
    for path in artifact_paths:
        text = path.read_text(encoding="utf-8")
        assert "tableRows" not in text
        assert '"series"' not in text
        assert '"patient"' not in text
        assert '"stay_id"' not in text
    artifact_payloads = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in artifact_paths
    }
    table1 = artifact_payloads["table1_summary.json"]
    assert table1["status"] == "metadata_only"
    assert table1["denominator"] == 94_458
    assert {row["feature"] for row in table1["variables"]} >= {
        "age",
        "death",
        "sofa2",
    }
    missingness = artifact_payloads["missingness_audit.json"]
    assert missingness["status"] == "metadata_only"
    assert missingness["denominator"] == 94_458
    assert missingness["rows"]
    assert {
        row["coverage_basis"] for row in missingness["rows"]
    } == {"manifest_file_inventory"}
    assert artifact_payloads["roc_curve.json"]["status"] == "not_available"
    assert "metadata preflight" in artifact_payloads["roc_curve.json"]["reason"]


def test_agent_run_job_blocks_idea_seed_without_execution_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    seed_dir = tmp_path / "old_idea_seed"
    seed_dir.mkdir()
    (seed_dir / "project_seed.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu.agent_project_seed/1",
                "status": "seeded_from_idea",
                "study_id": "old-idea",
                "source_run_id": "idea_old",
                "question": "legacy idea seed",
            }
        ),
        encoding="utf-8",
    )
    client = TestClient(app)

    start = client.post(
        "/api/jobs/agent-run",
        json={
            "path": str(export_dir),
            "study_id": "old-idea",
            "question": "legacy idea seed",
            "project_seed_dir": str(seed_dir),
        },
    )

    assert start.status_code == 400
    detail = start.json()["detail"]
    assert detail["error"] == "agent_project_execution_gate_missing"
    assert "refresh Agent project from Idea Mining" in detail["blockers"][0]


def test_agent_run_job_can_be_cancelled_and_reports_restart_resume(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    export_dir = _write_csv_export(tmp_path / "miiv", database="miiv")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    original_summary = dataio.summarize_export_workspace
    summary_started = threading.Event()

    def slow_summary(path: str) -> dict:
        summary_started.set()
        time.sleep(0.25)
        return original_summary(path)

    monkeypatch.setattr(dataio, "summarize_export_workspace", slow_summary)
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
    assert summary_started.wait(timeout=1)

    cancel = client.post(f"/api/jobs/{job_id}/cancel", json={"reason": "test_cancel"})
    assert cancel.status_code == 200
    assert cancel.json()["cancel_request_accepted"] is True

    snapshot = _wait_for_job(client, job_id, timeout=5)
    assert snapshot["status"] == "cancelled"
    result = snapshot["result"]
    assert result["cancelled"] is True
    assert result["cancelled_at"] == "snapshot"
    assert result["resumable"] is True
    assert result["resume_kind"] == "restart_from_active_export"
    assert result["gate"]["status"] == "cancelled"
    assert result["gate"]["reportable"] is False
    assert result["gate"]["draft_unlocked"] is False
    assert result["artifacts"] == []
    assert result["uploads"] == 0
    assert result["tokens"] == 0
    run_dir = Path(result["project_dir"])
    assert run_dir.exists()
    assert not (run_dir / "evidence_ledger.json").exists()
    assert any(event.get("type") == "cancel_requested" for event in snapshot["events"])


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
    assert "human_signoff.json" not in {
        item["name"] for item in review_payload["artifacts"]
    }

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
    assert "human_signoff.json" in {
        item["name"] for item in signed_payload["artifacts"]
    }

    signoff_path = run_dir / "human_signoff.json"
    assert signoff_path.exists()
    signoff_payload = json.loads(signoff_path.read_text(encoding="utf-8"))
    assert signoff_payload["status"] == "signed_analysis_only"
    assert signoff_payload["reportable"] is False
    assert signoff_payload["draft_unlocked"] is False
    assert signoff_payload["uploads"] == 0
    assert signoff_payload["tokens"] == 0
    assert signoff_payload["external_calls"] == 0
    signed_artifacts = {
        item["name"]: item for item in signoff_payload["signed_artifacts"]
    }
    assert set(signed_artifacts) == AGENT_PREFLIGHT_ARTIFACTS
    for item in signed_artifacts.values():
        assert len(item["sha256"]) == 64
        assert item["bytes"] > 0
    assert (
        _scan_artifact_payloads({"human_signoff.json": signoff_payload})["passed"]
        is True
    )
    text = signoff_path.read_text(encoding="utf-8")
    assert "tableRows" not in text
    assert '"stay_id"' not in text
    assert '"patient"' not in text

    gate = json.loads((run_dir / "quality_gate.json").read_text(encoding="utf-8"))[
        "gate"
    ]
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
        assert set(zf.namelist()) == {*AGENT_PREFLIGHT_ARTIFACTS, "human_signoff.json"}

    cohort_path = run_dir / "cohort_summary.json"
    cohort_payload = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort_payload["summary"]["stays"] = 999
    cohort_path.write_text(json.dumps(cohort_payload, indent=2), encoding="utf-8")

    stale_review = client.post(
        "/api/agent-runs/review", json={"project_dir": str(run_dir)}
    )
    assert stale_review.status_code == 200
    stale_payload = stale_review.json()
    assert stale_payload["signed"] is True
    assert stale_payload["signoff_stale"] is True
    assert stale_payload["readiness"]["status"] == "signoff_stale"
    assert stale_payload["readiness"]["reportable"] is False
    assert stale_payload["readiness"]["draft_unlocked"] is False
    assert stale_payload["signoff_integrity"]["status"] == "stale"
    assert (
        stale_payload["signoff_integrity"]["tampered_artifacts"][0]["name"]
        == "cohort_summary.json"
    )


def test_agent_run_review_exposes_canonical9_import_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "canonical9" / "run_20260613T004906_66dc3b"
    run_dir.mkdir(parents=True)
    payloads = {
        "run_context.json": {
            "run_id": "run_20260613T004906_66dc3b",
            "study_id": "fig2-e1-sepsis3-mortality",
            "mode": "analysis",
            "question": "Sepsis-3 prevalence and mortality",
            "summary": {"stays": 100, "evidence_count": 12},
            "local_first": {"uploads": 0, "tokens": 0, "imported": True},
        },
        "cohort_summary.json": {
            "summary": {"stays": 100, "evidence_count": 12},
            "cohort": {"patient_rows_returned": False},
        },
        "quality_gate.json": {
            "gate": {
                "status": "analysis_only",
                "reportable": False,
                "draft_unlocked": False,
                "checks": [{"id": "human_signoff", "passed": False}],
            },
            "quality": [],
        },
        "agent_plan.json": {"steps": [{"step_id": "s1", "intent": "summarize run"}]},
        "manuscript_draft.json": {
            "run_id": "run_20260613T004906_66dc3b",
            "status": "locked_canonical9_import",
            "claims": [],
            "sentences": [],
        },
        "benchmark_scorecard.json": {
            "kind": "canonical9_benchmark_scorecard",
            "task_id": "E1_sepsis3_mortality",
            "tristate": "gate_reportable",
            "evidence_count": 12,
            "dimensions": [{"id": "plan", "subscore": 1.0, "level": "Pass"}],
        },
        "workflow_graph.json": {
            "kind": "workflow_graph",
            "graph": {"nodes": [{"id": "plan"}], "edges": []},
        },
        "figure_gallery.json": {
            "kind": "figure_gallery",
            "status": "ok",
            "figures": [
                {
                    "label": "Publication figure",
                    "relative_path": "publication_figures/easyicu_publication_figure.png",
                    "data_url": "data:image/png;base64,ZmFrZQ==",
                }
            ],
        },
        "source_run_manifest.json": {
            "kind": "source_run_manifest",
            "run_id": "run_20260613T004906_66dc3b",
            "evidence_count": 12,
        },
        "evidence_ledger.json": {
            "run_id": "run_20260613T004906_66dc3b",
            "run_type": "canonical9_import",
            "status": "analysis_only",
            "artifacts": [],
            "privacy": {"patient_rows_persisted": False},
        },
    }
    for name, payload in payloads.items():
        (run_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    client = TestClient(app)
    review = client.post("/api/agent-runs/review", json={"project_dir": str(run_dir)})

    assert review.status_code == 200
    body = review.json()
    assert body["ok"] is True
    assert body["run_type"] == "canonical9_import"
    assert body["readiness"]["reportable"] is False
    public_payloads = body["artifact_payloads"]
    for name in (
        "benchmark_scorecard.json",
        "workflow_graph.json",
        "figure_gallery.json",
        "source_run_manifest.json",
    ):
        assert name in public_payloads
    assert public_payloads["figure_gallery.json"]["figures"][0][
        "data_url"
    ].startswith("data:image/png;base64,")
    assert {
        "benchmark_scorecard.json",
        "workflow_graph.json",
        "figure_gallery.json",
        "source_run_manifest.json",
    } <= {artifact["name"] for artifact in body["artifacts"]}

    artifact = client.post(
        "/api/agent-runs/artifact",
        json={"project_dir": str(run_dir), "artifact": "figure_gallery.json"},
    )
    assert artifact.status_code == 200
    artifact_body = artifact.json()
    assert artifact_body["payload"]["figures"][0]["label"] == "Publication figure"
    assert artifact_body["privacy_scan"]["passed"] is True


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


def test_numeric_evidence_audit_passes_bound_percent_rounding_range_and_delta() -> None:
    audit = numeric_evidence_audit.audit_numeric_evidence(
        {
            "cohort_summary.json": {
                "summary": {
                    "mortality_pct": 12.2,
                    "mean_age": 62.95,
                    "age": {"min": 50, "max": 70},
                    "mortality_delta_pct": 16.7,
                }
            },
            "manuscript_draft.json": {
                "claims": [
                    {
                        "id": "c1",
                        "text": "Mortality was 12.2%, mean age was 63, age range was 50-70, and mortality delta was 16.7%.",
                        "evidence_ids": ["cohort_summary.json"],
                    }
                ],
                "sentences": [],
            },
        }
    )

    assert audit["passed"] is True
    assert audit["numeric_claim_count"] == 1
    assert audit["numeric_mention_count"] == 5
    assert audit["match_count"] == 5
    assert audit["matches"][1]["number"] == "63"
    assert audit["matches"][1]["evidence_value"] == 62.95
    assert audit["matches"][1]["tolerance"] == 0.5


def test_numeric_evidence_audit_fails_mismatch_ghost_and_missing_evidence() -> None:
    mismatch = numeric_evidence_audit.audit_numeric_evidence(
        {
            "cohort_summary.json": {"summary": {"mortality_pct": 12.2}},
            "manuscript_draft.json": {
                "claims": [
                    {
                        "id": "c1",
                        "text": "Mortality was 13.2%.",
                        "evidence_ids": ["cohort_summary.json"],
                    }
                ],
                "sentences": [],
            },
        }
    )
    ghost = numeric_evidence_audit.audit_numeric_evidence(
        {
            "cohort_summary.json": {"summary": {"mortality_pct": 12.2}},
            "manuscript_draft.json": {
                "claims": [
                    {
                        "id": "c2",
                        "text": "Mortality was 12.2%.",
                        "evidence_ids": ["ghost.json"],
                    }
                ],
                "sentences": [],
            },
        }
    )
    missing = numeric_evidence_audit.audit_numeric_evidence(
        {
            "cohort_summary.json": {"summary": {"mortality_pct": 12.2}},
            "manuscript_draft.json": {
                "claims": [],
                "sentences": [{"id": "s1", "text": "Mortality was 12.2%."}],
            },
        }
    )

    assert mismatch["passed"] is False
    assert mismatch["failures"][0]["reason"] == "numeric_value_not_bound"
    assert ghost["passed"] is False
    assert {failure["reason"] for failure in ghost["failures"]} >= {
        "missing_evidence",
        "numeric_value_not_bound",
    }
    assert missing["passed"] is False
    assert missing["failures"][0]["reason"] == "numeric_claim_missing_evidence_id"
    assert missing["sentences_passed"] is False
    for payload in (mismatch, ghost, missing):
        serialized = json.dumps(payload)
        for marker in ["stay_id", "subject_id", "hadm_id", "tableRows", '"series"']:
            assert marker not in serialized


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
    assert [
        event["type"] for event in snapshot["events"] if event["type"] != "end"
    ] == [
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
    assert (
        result["provider"]["canonical_opt_in_source"]
        == provider_gate.CANONICAL_OPT_IN_SOURCE
    )
    assert result["provider"]["canonical_opt_in_passed"] is True
    assert result["provider"]["mock_calls"] == 1
    assert result["gate"]["status"] == "analysis_only"
    assert result["gate"]["reportable"] is False
    assert result["gate"]["draft_unlocked"] is False
    checks = {check["id"]: check for check in result["gate"]["checks"]}
    assert checks["provider_opt_in"]["passed"] is True
    assert checks["strict_evidence_bound_claims"]["passed"] is True
    assert checks["strict_evidence_bound_sentences"]["passed"] is True
    assert checks["numeric_evidence_value_binding"]["passed"] is True
    assert checks["numeric_evidence_value_binding"]["numeric_mention_count"] == 4
    assert checks["human_signoff"]["passed"] is False
    expected_artifacts = {
        *AGENT_PREFLIGHT_ARTIFACTS,
        "agent_plan.json",
        "manuscript_draft.json",
    }
    assert checks["no_patient_rows_persisted"]["scanned_artifacts"] == len(
        expected_artifacts
    )

    artifact_paths = {
        Path(item["path"]).name: Path(item["path"]) for item in result["artifacts"]
    }
    assert set(artifact_paths) == expected_artifacts
    draft = json.loads(
        artifact_paths["manuscript_draft.json"].read_text(encoding="utf-8")
    )
    assert draft["status"] == "locked_until_human_signoff"
    assert all(row.get("evidence_ids") for row in draft["claims"])
    assert all(row.get("evidence_ids") for row in draft["sentences"])
    ledger = json.loads(
        artifact_paths["evidence_ledger.json"].read_text(encoding="utf-8")
    )
    assert ledger["strict_evidence_audit"]["claims_passed"] is True
    assert ledger["numeric_evidence_audit"]["passed"] is True
    assert ledger["numeric_evidence_audit"]["numeric_mention_count"] == 4
    assert ledger["privacy"]["artifact_scan"]["scanned_artifacts"] == len(
        expected_artifacts
    )


def test_full_agent_numeric_evidence_gate_blocks_mismatched_mock_claim(
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

    def mismatched_payload(**kwargs):
        payload = original_payload(**kwargs)
        payload["manuscript_draft"]["claims"][0][
            "text"
        ] = "Mortality was 13.2% in the active export."
        payload["manuscript_draft"]["claims"][0]["evidence_ids"] = [
            "cohort_summary.json"
        ]
        return payload

    monkeypatch.setattr(agent_runs, "_mock_full_agent_payload", mismatched_payload)
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
    result = snapshot["result"]
    gate = result["gate"]
    assert gate["status"] == "blocked"
    assert gate["reason"] == "numeric_evidence_gate_failed"
    assert gate["reportable"] is False
    assert gate["draft_unlocked"] is False
    checks = {check["id"]: check for check in gate["checks"]}
    assert checks["strict_evidence_bound_claims"]["passed"] is True
    assert checks["strict_evidence_bound_sentences"]["passed"] is True
    assert checks["numeric_evidence_value_binding"]["passed"] is False
    assert checks["numeric_evidence_value_binding"]["failure_count"] == 1
    assert result["numeric_evidence_audit"]["failures"][0]["number"] == "13.2%"
    serialized = json.dumps(
        {
            "numeric_evidence_audit": result["numeric_evidence_audit"],
            "numeric_gate_check": checks["numeric_evidence_value_binding"],
        }
    )
    for marker in ["stay_id", "subject_id", "hadm_id", "tableRows", '"series"']:
        assert marker not in serialized


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
        json={
            "run_type": "full",
            "llm_provider": "openai",
            "external_llm_opt_in": False,
        },
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
        json={
            "run_type": "full",
            "llm_provider": "openai",
            "external_llm_opt_in": False,
        },
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


def test_provider_gate_calls_canonical_opt_in_before_per_run_and_credentials(
    monkeypatch,
) -> None:
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
        json={
            "run_type": "full",
            "llm_provider": "openai",
            "external_llm_opt_in": True,
        },
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
                                            "evidence_ids": [
                                                "run_context.json",
                                                "quality_gate.json",
                                            ],
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
    assert (
        checks["provider_opt_in"]["evidence"]
        == "external_provider_adapter_after_opt_in"
    )
    assert checks["strict_evidence_bound_claims"]["passed"] is True
    assert checks["strict_evidence_bound_sentences"]["passed"] is True
    assert captured["url"] == "http://127.0.0.1:9999/v1/chat/completions"
    assert captured["request"]["model"] == "test-model"
    assert captured["request"]["max_tokens"] == 768
    assert captured["request"]["response_format"] == {"type": "json_object"}
    assert "text" not in captured["request"]
    assert captured["request"]["easyicu_policy"]["patient_rows_excluded"] is True
    assert captured["request"]["easyicu_policy"]["max_external_calls_per_run"] == 1
    assert captured["request"]["easyicu_policy"]["max_output_tokens"] == 768
    assert captured["request"]["easyicu_policy"]["json_format_style"] == "chat"
    artifact_paths = {
        Path(item["path"]).name: Path(item["path"]) for item in result["artifacts"]
    }
    draft = json.loads(
        artifact_paths["manuscript_draft.json"].read_text(encoding="utf-8")
    )
    assert draft["status"] == "locked_until_human_signoff"
    assert "http://127.0.0.1:9999/v1" not in json.dumps(result, ensure_ascii=False)


def test_provider_adapter_can_use_responses_json_format_style() -> None:
    captured = {}

    def fake_transport(request, headers):
        captured["request"] = request
        captured["headers"] = headers
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "agent_plan": [
                                    {
                                        "id": "step_001",
                                        "title": "Review aggregate snapshot",
                                    }
                                ],
                                "manuscript_draft": {
                                    "claims": [
                                        {
                                            "id": "c1",
                                            "text": "Bound aggregate claim.",
                                            "evidence_ids": ["run_context.json"],
                                        }
                                    ],
                                    "sentences": [
                                        {
                                            "id": "s1",
                                            "text": "Bound aggregate sentence.",
                                            "evidence_ids": ["cohort_summary.json"],
                                        }
                                    ],
                                },
                            }
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }

    result = provider_adapter.generate_bound_provider_payload(
        provider_meta={
            "provider": "openai",
            "external": True,
            "external_calls": 0,
            "provider_gate_order": ["credentials_loaded"],
        },
        run_id="run_test",
        study_id="sepsis",
        question="bounded draft",
        summary={"stays": 3},
        cohort={"survived": 2, "deceased": 1},
        quality=[],
        transport=fake_transport,
        environ={
            "OPENAI_API_KEY": "sk-test-responses-format",
            "OPENAI_BASE_URL": "http://127.0.0.1:8787/v1",
            "OPENAI_MODEL": "gpt-test",
            "EASYICU_LLM_JSON_FORMAT_STYLE": "responses",
        },
    )

    assert captured["headers"]["Authorization"] == "Bearer sk-test-responses-format"
    format_spec = captured["request"]["text"]["format"]
    assert format_spec["type"] == "json_schema"
    assert format_spec["name"] == "easyicu_agent_run"
    assert format_spec["strict"] is True
    draft_schema = format_spec["schema"]["properties"]["manuscript_draft"]
    claim_schema = draft_schema["properties"]["claims"]["items"]
    assert claim_schema["required"] == ["id", "text", "evidence_ids"]
    assert claim_schema["properties"]["evidence_ids"]["minItems"] == 1
    assert claim_schema["properties"]["evidence_ids"]["items"]["enum"] == [
        "run_context.json",
        "cohort_summary.json",
        "table1_summary.json",
        "missingness_audit.json",
        "roc_curve.json",
        "calibration_curve.json",
        "quality_gate.json",
    ]
    assert "response_format" not in captured["request"]
    assert (
        "agent_plan must be an object, not an array"
        in captured["request"]["messages"][0]["content"]
    )
    assert captured["request"]["easyicu_policy"]["json_format_style"] == "responses"
    assert result["agent_plan"]["steps"][0]["id"] == "step_001"
    assert result["provider"]["json_format_style"] == "responses"
    assert result["provider"]["external_calls"] == 1


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
                                        {
                                            "id": "c1",
                                            "text": "Unbound claim",
                                            "evidence_ids": [],
                                        }
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
    assert {"owner": "sentence", "evidence_id": "ghost.json"} in audit[
        "missing_evidence"
    ]
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


def test_agent_run_endpoint_rejects_missing_active_export(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    client = TestClient(app)

    response = client.post("/api/jobs/agent-run", json={"study_id": "sepsis"})

    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "no_active_export"


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] != "running":
            return payload
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")
