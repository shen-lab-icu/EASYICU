"""Launch facts cross the existing read boundaries; all records are synthetic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import agent_runs, capabilities, dataio, run_step_trace, sources
from easyicu.webserver import settings as settings_store
from easyicu.webserver.app import app
from easyicu.webserver.pi_copilot.projections import project_artifacts, project_job


@pytest.fixture()
def settings_file(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_store, "_CONFIG_DIR", tmp_path)
    path = tmp_path / "settings.json"
    monkeypatch.setattr(settings_store, "_CONFIG_PATH", path)
    return path


def test_support_contact_defaults_empty_and_uses_the_settings_read_path(settings_file):
    client = TestClient(app)
    assert client.get("/api/settings").json()["support_contact"] == ""
    settings_file.write_text(json.dumps({"support_contact": "Hospital help desk"}))
    assert client.get("/api/settings").json()["support_contact"] == "Hospital help desk"


@pytest.mark.parametrize("value, expected", [
    ("Desk\n\t\x00\x7f\x85 support", "Desk support"),
    ("中" * 210, "中" * 200),
    (None, ""),
    ({"bad": "object"}, ""),
])
def test_support_contact_is_bounded_text_without_controls(settings_file, value, expected):
    settings_file.write_text(json.dumps({"support_contact": value}))
    assert settings_store.load_settings()["support_contact"] == expected


def test_the_page_cannot_write_or_reset_the_administrators_contact(settings_file):
    settings_file.write_text(json.dumps({"support_contact": "Hospital help desk"}))
    client = TestClient(app)
    response = client.post("/api/settings", json={"support_contact": "Override", "language": "zh"})
    assert response.status_code == 400
    assert settings_store.load_settings()["language"] == "en"
    assert client.post("/api/settings/reset").json()["support_contact"] == "Hospital help desk"


def test_the_administrator_can_provision_the_contact_through_the_settings_store(settings_file):
    result = settings_store.update_settings({"support_contact": "Help\n desk"})
    assert result["support_contact"] == "Help desk"
    assert settings_store.load_settings()["support_contact"] == "Help desk"


def test_run_files_keep_their_stat_sizes_through_the_path_free_projection(tmp_path):
    for name, payload in {
        "run_context.json": {"run_id": "run_synthetic"},
        "quality_gate.json": {"gate": {"status": "analysis_only", "checks": []}},
        "evidence_ledger.json": {"run_id": "run_synthetic"},
    }.items():
        (tmp_path / name).write_text(json.dumps(payload))
    (tmp_path / "unlisted.json").write_text("{}")
    review = agent_runs.read_run_review(str(tmp_path))
    assert review["ok"]
    rows = project_artifacts(review["artifacts"])
    assert {row["name"] for row in rows} == {"run_context.json", "quality_gate.json", "evidence_ledger.json"}
    for row in rows:
        assert row["size_bytes"] == (tmp_path / row["name"]).stat().st_size
        assert "path" not in row and "relative_path" not in row


@pytest.mark.parametrize("size", [-1, 2**53, True, "12", None])
def test_run_file_sizes_reject_non_bounded_integers(size):
    assert "size_bytes" not in project_artifacts([{"name": "quality_gate.json", "size_bytes": size}])[0]


def test_a_failed_stat_leaves_the_run_file_size_absent(tmp_path, monkeypatch):
    path = tmp_path / "quality_gate.json"
    original = Path.stat

    def fail_stat(candidate, *args, **kwargs):
        if candidate == path:
            raise OSError("synthetic stat failure")
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_stat)
    row = agent_runs._artifact_from_raw(path, tmp_path, b"{}")
    assert "size_bytes" not in row


@pytest.mark.parametrize("size", [-1, 2**60, True, "12", None])
def test_stat_size_is_bounded_before_it_enters_a_run_record(tmp_path, monkeypatch, size):
    path = tmp_path / "quality_gate.json"
    monkeypatch.setattr(Path, "stat", lambda *a, **kw: SimpleNamespace(st_size=size))
    assert "size_bytes" not in agent_runs._artifact_from_raw(path, tmp_path, b"{}")


def _trace_run(root):
    plan = {"steps": [{"step_id": "grouped_summary", "inputs": ["glu_group_x1", "readmission"]}],
            "display_labels": {"readmission": "ICU readmission"}}
    raw = json.dumps(plan).encode()
    (root / "analysis_plan.json").write_bytes(raw)
    (root / "manifest.json").write_text(json.dumps({"current_plan_authority": {
        "relative_path": "analysis_plan.json", "sha256": hashlib.sha256(raw).hexdigest(),
    }}))
    (root / "research_context.json").write_text(json.dumps({
        "primary_exposure": "glu_group_x1", "target_outcome": "readmission",
        "variables": [{"name": "readmission", "description": "ICU readmission", "role": "outcome"}],
    }))
    (root / "exposure_groupings.json").write_text(json.dumps({
        "schema_version": "easyicu.exposure_groupings_record/1", "compiled": {"groupings": [{
            "variable": "glu_group_x1", "disposition": "applied", "derivation": {
                "concept": "glu", "window": {"start_hours": 0, "end_hours": 24},
                "scale": "nominal", "unmeasured": "own_group",
                "groups": [
                    {"id": "g1", "rule": {"summary": "min", "op": "<", "value": 70, "unit": "mg/dL"}},
                    {"id": "g2", "rule": {"summary": "max", "op": ">", "value": 180, "unit": "mg/dL"}},
                    {"id": "g3", "rule": "otherwise"},
                ],
                "source_columns": {"min": "glu_min", "max": "glu_max"},
            },
            "labels": {"g1": "偏低", "g2": "偏高", "g3": "其余", "gU": "未测量"},
            "compared": {"reference": 1, "contrast": 2, "stated": []},
        }]},
    }))
    (root / "audit_log.jsonl").write_text("\n".join(json.dumps(row) for row in [
        {"timestamp": "2026-01-01T00:00:01+00:00", "phase": "context",
         "event": "Exposure groups staged: glu_group_x1.", "status": "complete"},
        {"timestamp": "2026-01-01T00:00:02+00:00", "phase": "step",
         "event": "Step 1/1 started: grouped_summary.", "step_id": "grouped_summary"},
    ]))


def test_recorded_group_and_outcome_names_reach_the_trace_and_step(tmp_path):
    _trace_run(tmp_path)
    rows = run_step_trace.trace_timings_projection(tmp_path)
    expected_group = {"zh": "血糖分组", "en": "Glucose groups"}
    expected_outcome = {"zh": "ICU readmission", "en": "ICU readmission"}
    assert rows[0]["group_variable_label"] == expected_group
    assert rows[1]["outcome_label"] == expected_outcome
    job = project_job({"id": "synthetic", "result": {"trace_timings": rows}})
    assert job["progress"][1]["group_variable_label"] == expected_group
    assert job["progress"][1]["outcome_label"] == expected_outcome
    inputs = run_step_trace.project_step_detail(tmp_path, "grouped_summary")["inputs"]
    assert inputs["group_variable_label"] == expected_group
    assert inputs["outcome_label"] == expected_outcome


@pytest.mark.parametrize("broken", ["absent", "unreadable", "unbound"])
def test_unrecorded_reader_names_are_never_inferred(tmp_path, broken):
    _trace_run(tmp_path)
    if broken == "absent":
        (tmp_path / "exposure_groupings.json").unlink()
        (tmp_path / "research_context.json").unlink()
    elif broken == "unreadable":
        (tmp_path / "exposure_groupings.json").write_text("{broken")
        (tmp_path / "research_context.json").write_text("{}")
    else:
        (tmp_path / "manifest.json").unlink()
        (tmp_path / "exposure_groupings.json").unlink()
    rows = run_step_trace.trace_timings_projection(tmp_path)
    assert all("group_variable_label" not in row and "outcome_label" not in row for row in rows)


def test_trace_reader_names_are_bounded_path_free_and_rechecked_in_transport():
    row = {"type": "progress", "action": "step_started", "step_id": "synthetic",
           "group_variable_label": {"zh": "中" * 300, "en": "/tmp/private.csv", "path": "/tmp/private"},
           "outcome_label": {"zh": ["not text"], "en": "Readmission"}}
    projected = project_job({"id": "synthetic", "events": [row]})["progress"][0]
    assert projected["group_variable_label"] == {"zh": "中" * 240}
    assert projected["outcome_label"] == {"en": "Readmission"}


def test_a_chinese_outcome_label_keeps_its_recorded_english_description(tmp_path):
    _trace_run(tmp_path)
    path = tmp_path / "analysis_plan.json"
    plan = json.loads(path.read_text())
    plan["display_labels"]["readmission"] = "ICU再入院"
    raw = json.dumps(plan).encode()
    path.write_bytes(raw)
    (tmp_path / "manifest.json").write_text(json.dumps({"current_plan_authority": {
        "relative_path": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
    }}))
    rows = run_step_trace.trace_timings_projection(tmp_path)
    assert rows[1]["outcome_label"] == {"zh": "ICU再入院", "en": "ICU readmission"}


def test_each_step_names_its_own_recorded_grouping(tmp_path):
    _trace_run(tmp_path)
    record_path = tmp_path / "exposure_groupings.json"
    record = json.loads(record_path.read_text())
    secondary = json.loads(json.dumps(record["compiled"]["groupings"][0]))
    secondary["variable"] = "lact_group_x1"
    secondary["derivation"]["concept"] = "lact"
    secondary["derivation"]["source_columns"] = {"min": "lact_min", "max": "lact_max"}
    for group in secondary["derivation"]["groups"][:2]:
        group["rule"]["unit"] = "mmol/L"
    record["compiled"]["groupings"].append(secondary)
    record_path.write_text(json.dumps(record))
    path = tmp_path / "analysis_plan.json"
    plan = json.loads(path.read_text())
    plan["steps"].append({"step_id": "lactate_summary", "inputs": ["lact_group_x1"]})
    raw = json.dumps(plan).encode()
    path.write_bytes(raw)
    (tmp_path / "manifest.json").write_text(json.dumps({"current_plan_authority": {
        "relative_path": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
    }}))
    with (tmp_path / "audit_log.jsonl").open("a") as handle:
        handle.write("\n" + json.dumps({
            "timestamp": "2026-01-01T00:00:03+00:00", "phase": "step",
            "event": "Step 2/2 started: lactate_summary.", "step_id": "lactate_summary",
        }))
    rows = run_step_trace.trace_timings_projection(tmp_path)
    assert rows[-1]["group_variable_label"] == {"zh": "乳酸分组", "en": "Lactate groups"}
    assert run_step_trace.project_step_detail(tmp_path, "lactate_summary")["inputs"]["group_variable_label"] == rows[-1]["group_variable_label"]


def test_an_unconverted_raw_folder_does_not_claim_clean_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(dataio, "_detect_database", lambda path: "miiv")
    monkeypatch.setattr(dataio, "_check_data_status", lambda *a: {
        "parquet_count": 0, "csv_count": 1, "ready": False, "missing_tables": [],
    })
    result = dataio.scan_path(str(tmp_path))
    assert result["ready"] is False
    assert result["conversion_quality"] == {
        "bad_rows_skipped": None, "data_quality_status": "not_converted", "ready_for_analysis": False,
    }


def test_capabilities_count_only_validated_registered_full_sources(monkeypatch):
    from easyicu.webserver import demo_sources

    demo = demo_sources.get_source(demo_sources.allowed_source_ids()[0])
    monkeypatch.setattr(sources, "load_registry", lambda: {"sources": [
        {"ok": True, "id": str(i), "database": "eicu", "path": f"/tmp/private/{i}", "label": "Private"}
        for i in range(43)
    ] + [
        {"ok": True, "id": "miiv", "database": "mimiciv"},
        {"ok": False, "id": "bad", "database": "eicu"},
        {"ok": True, "database": "miiv"},
        {"ok": True, "id": "demo", "database": demo.database, "label": f"{demo.title} v{demo.version}"},
        {"ok": True, "id": "demo-profile", "database": "miiv_demo"},
        {"ok": True, "id": "unknown", "database": "/tmp/private"},
    ]})
    monkeypatch.setattr(capabilities, "zotero_status", lambda *a: {})
    result = TestClient(app).get("/api/capabilities").json()["registered_sources"]
    assert result["by_database"]["eicu"] == 43
    assert result["by_database"]["miiv"] == 1
    assert all(isinstance(value, int) and value >= 0 for value in result["by_database"].values())
    assert set(result) == {"by_database"}
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("value, hosted", [(None, False), ("1", True), ("true", False), ("0", False), (" 1", False)])
def test_capabilities_expose_only_the_explicit_hosted_instance_flag(monkeypatch, value, hosted):
    if value is None:
        monkeypatch.delenv("EASYICU_HOSTED", raising=False)
    else:
        monkeypatch.setenv("EASYICU_HOSTED", value)
    monkeypatch.setattr(sources, "load_registry", lambda: {"sources": []})
    monkeypatch.setattr(capabilities, "zotero_status", lambda *a: {})
    body = TestClient(app).get("/api/capabilities").json()
    assert body["deployment"] == {"hosted": hosted}
