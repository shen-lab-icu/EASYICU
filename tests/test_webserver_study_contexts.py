from __future__ import annotations

import json
import stat
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from easyicu.webserver import capabilities
from easyicu.webserver import sources as source_store
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver.app import app
from easyicu.webserver.routes import agent as agent_routes


@pytest.fixture(autouse=True)
def _isolated_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        context_store, "_CONFIG_PATH", tmp_path / "cfg" / "study-contexts.json"
    )
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "source-cfg")
    monkeypatch.setattr(
        source_store, "_CONFIG_PATH", tmp_path / "source-cfg" / "sources.json"
    )
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])


def _write_export(root: Path, *, database: str = "miiv") -> Path:
    root.mkdir(parents=True)
    demographics = pd.DataFrame({"stay_id": [1, 2], "age": [55, 67], "sex": ["F", "M"]})
    outcome = pd.DataFrame({"stay_id": [1, 2], "death": [0, 1], "los_icu": [2.0, 4.0]})
    demographics.to_csv(root / "demographics.csv", index=False)
    outcome.to_csv(root / "outcome.csv", index=False)
    (root / "_manifest.json").write_text(
        json.dumps(
            {
                "database": database,
                "generated": "2026-07-10T12:00:00Z",
                "patient_count": 2,
                "files": [
                    {"file": "demographics.csv", "module": "demographics", "rows": 2},
                    {"file": "outcome.csv", "module": "outcome", "rows": 2},
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        snapshot = response.json()
        if snapshot["status"] != "running":
            return snapshot
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish")


def test_study_context_api_persists_lists_and_handoffs_metadata(tmp_path: Path) -> None:
    client = TestClient(app)
    source_path = tmp_path / "prepared-export"

    created = client.post(
        "/api/study-contexts",
        json={
            "id": "study_sepsis",
            "title": "Sepsis mortality study",
            "question": "Is admission severity associated with mortality?",
            "purpose": "Prepare an auditable descriptive analysis.",
            "data_source": {
                "path": str(source_path),
                "label": "MIMIC-IV export",
                "database": "miiv",
            },
            "cohort": {"preset": "adult_first", "age_min": 18},
            "modules": ["demographics", "outcome", "sofa2_score"],
            "outcome": "hospital mortality",
            "time_window": {"hours": 24, "anchor": "icu_admission"},
            "comparator": "lower admission severity",
            "export_format": "parquet",
            "analysis_goal": "Estimate an adjusted association.",
            "confirmations": {"cohort_reviewed": True},
        },
    )

    assert created.status_code == 200
    context = created.json()["context"]
    assert created.json()["active_id"] == "study_sepsis"
    assert context["data_source"]["path"] == str(source_path.resolve())
    assert context["current_stage"] == "plan"
    assert context["last_route"] == "entry"
    assert context["revision"] == 1

    active = client.get("/api/study-contexts/active")
    listed = client.get("/api/study-contexts")
    fetched = client.get("/api/study-contexts/study_sepsis")
    assert active.json()["context"]["id"] == "study_sepsis"
    assert listed.json()["contexts"] == [context]
    assert fetched.json()["context"] == context

    handoff = client.post(
        "/api/study-contexts/handoff",
        json={
            "study_context_id": "study_sepsis",
            "current_stage": "review",
            "last_route": "extraction",
            "target_route": "cohort",
        },
    )
    assert handoff.status_code == 200
    assert handoff.json()["handoff"] == {
        "from_stage": "plan",
        "to_stage": "review",
        "from_route": "extraction",
        "target_route": "cohort",
    }

    stored = json.loads(context_store._CONFIG_PATH.read_text(encoding="utf-8"))
    assert stored["active_id"] == "study_sepsis"
    assert stored["contexts"][0]["last_route"] == "cohort"
    assert not context_store._CONFIG_PATH.with_suffix(".json.tmp").exists()


def test_patient_review_scope_metadata_survives_backend_normalization() -> None:
    response = TestClient(app).post(
        "/api/study-contexts",
        json={
            "id": "study_patient_bounded_review",
            "question": "Analyze the bounded Patient Review context.",
            "cohort": {
                "review": "patient",
                "entity_count": 94_458,
                "full_entity_count": 94_458,
                "review_entities": 500,
                "review_entity_cap": 500,
                "review_scope": "browser_bounded_entity_sample",
                "module_count": 19,
            },
            "confirmations": {
                "patient_review_completed": True,
                "patient_review_bounded_sample": True,
                "patient_review_full_entity_set": False,
            },
        },
    )

    assert response.status_code == 200
    context = response.json()["context"]
    assert context["cohort"] == {
        "review": "patient",
        "entity_count": 94_458,
        "full_entity_count": 94_458,
        "review_entities": 500,
        "review_entity_cap": 500,
        "review_scope": "browser_bounded_entity_sample",
        "module_count": 19,
    }
    assert context["confirmations"] == {
        "patient_review_completed": True,
        "patient_review_bounded_sample": True,
        "patient_review_full_entity_set": False,
    }


def test_stale_metadata_save_cannot_overwrite_server_owned_job_lifecycle() -> None:
    client = TestClient(app)
    created = client.post(
        "/api/study-contexts",
        json={"id": "study_revision", "question": "Original question"},
    ).json()["context"]
    assert created["revision"] == 1

    running = client.post(
        "/api/study-contexts/handoff",
        json={
            "study_context_id": created["id"],
            "expected_revision": created["revision"],
            "current_stage": "analyze",
            "target_route": "agent",
            "active_job_id": "job-running",
        },
    ).json()["context"]
    assert running["revision"] == 2
    assert running["active_job_id"] == "job-running"

    unversioned_lifecycle = client.post(
        "/api/study-contexts/handoff",
        json={
            "study_context_id": created["id"],
            "current_stage": "review",
            "target_route": "agent",
            "active_job_id": None,
        },
    )
    assert unversioned_lifecycle.status_code == 409
    assert unversioned_lifecycle.json()["detail"]["error"] == (
        "study_context_revision_required"
    )

    stale = client.post(
        "/api/study-contexts",
        json={
            "id": created["id"],
            "expected_revision": created["revision"],
            "question": "Stale tab question",
            "current_stage": "plan",
            "last_route": "entry",
            "active_job_id": None,
        },
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["error"] == "study_context_revision_conflict"

    missing_revision = client.post(
        "/api/study-contexts",
        json={"id": created["id"], "question": "Unversioned overwrite"},
    )
    assert missing_revision.status_code == 409
    assert missing_revision.json()["detail"]["error"] == (
        "study_context_revision_required"
    )

    updated = client.post(
        "/api/study-contexts",
        json={
            "id": created["id"],
            "expected_revision": running["revision"],
            "question": "Current metadata question",
            # These stale lifecycle values are ignored on the metadata route.
            "current_stage": "plan",
            "last_route": "entry",
            "active_job_id": None,
        },
    )
    assert updated.status_code == 200
    current = updated.json()["context"]
    assert current["revision"] == 3
    assert current["question"] == "Current metadata question"
    assert current["current_stage"] == "analyze"
    assert current["last_route"] == "agent"
    assert current["active_job_id"] == "job-running"

    activated = client.post(
        "/api/study-contexts", json={"id": created["id"]}
    ).json()["context"]
    assert activated["revision"] == current["revision"]
    assert activated["active_job_id"] == "job-running"

    stale_handoff = client.post(
        "/api/study-contexts/handoff",
        json={
            "study_context_id": created["id"],
            "expected_revision": running["revision"],
            "current_stage": "review",
            "target_route": "agent",
        },
    )
    assert stale_handoff.status_code == 409
    assert stale_handoff.json()["detail"]["error"] == (
        "study_context_revision_conflict"
    )


def test_study_context_rejects_bounds_and_row_level_metadata() -> None:
    client = TestClient(app)

    too_long = client.post("/api/study-contexts", json={"title": "x" * 161})
    assert too_long.status_code == 400
    assert too_long.json()["detail"] == {
        "error": "study_context_field_too_long",
        "field": "title",
        "max_length": 160,
    }

    row_level = client.post(
        "/api/study-contexts",
        json={"cohort": {"filters": {"stay_id": [1001, 1002]}}},
    )
    assert row_level.status_code == 400
    assert row_level.json()["detail"]["error"] == "row_level_metadata_forbidden"
    assert row_level.json()["detail"]["markers"] == ["context.cohort.filters.stay_id"]
    assert not context_store._CONFIG_PATH.exists()

    alternate_row_key = client.post(
        "/api/study-contexts",
        json={"cohort": {"records": [{"entity_id": "patient-1"}]}},
    )
    assert alternate_row_key.status_code == 400
    assert alternate_row_key.json()["detail"]["error"] == (
        "row_level_metadata_forbidden"
    )

    too_complex = client.post(
        "/api/study-contexts",
        json={"cohort": {f"group_{idx}": list(range(8)) for idx in range(64)}},
    )
    assert too_complex.status_code == 400
    assert too_complex.json()["detail"] == {
        "error": "study_context_too_complex",
        "max_nodes": 512,
    }

    disguised_members = client.post(
        "/api/study-contexts",
        json={
            "cohort": {
                "members": [
                    {
                        "id": "P001",
                        "measurements": {"heart_rate": [80, 82]},
                    }
                ]
            }
        },
    )
    assert disguised_members.status_code == 400
    assert disguised_members.json()["detail"] == {
        "error": "unknown_study_context_fields",
        "field": "cohort",
        "fields": ["members"],
    }

    structured_text = client.post(
        "/api/study-contexts",
        json={"question": {"prompt": "hidden structure"}},
    )
    assert structured_text.status_code == 400
    assert structured_text.json()["detail"] == {
        "error": "invalid_study_context_field_type",
        "field": "question",
        "expected": "string",
    }

    structured_module = client.post(
        "/api/study-contexts",
        json={"modules": [{"name": "labs"}]},
    )
    assert structured_module.status_code == 400
    assert structured_module.json()["detail"]["expected"] == "list[string]"


def test_extraction_metadata_contract_shape_remains_supported() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/study-contexts",
        json={
            "id": "study_extraction_contract",
            "cohort": {
                "preset": "sepsis3",
                "age_min": 18,
                "age_max": 100,
                "min_icu_los_hours": 0,
                "observation_window_hours": 720,
                "exclude_readmissions": True,
                "icd_enabled": True,
                "icd_include": "A41",
                "icd_exclude": "",
                "include_diagnoses": ["A41"],
                "exclude_diagnoses": [],
                "sepsis_definition": {
                    "record_scope": "metadata_current_runtime_defaults",
                    "runtime_profile": "current",
                    "implementation_profile": "sofa2",
                    "score_family": "SOFA-2",
                    "definition_locked": True,
                    "suspected_infection": {
                        "mode": "first",
                        "abx_win_hours": 24,
                        "samp_win_hours": 72,
                        "abx_count_win_hours": 24,
                        "abx_min_count": 2,
                        "positive_cultures_required": False,
                    },
                    "sofa_increase": {
                        "si_window": "first",
                        "window_before_si_hours": 48,
                        "window_after_si_hours": 24,
                        "delta_function": "cumulative_minimum",
                        "threshold": 2,
                        "keep_components": True,
                    },
                    "review_options": {"si_window": ["first", "any"]},
                    "locked_core": {
                        "suspected_infection_windows": "ABX->sample 24h",
                        "sofa_window": "-48h/+24h",
                        "delta_rule": "cumulative minimum",
                        "sofa_threshold": "delta >= 2",
                    },
                },
            },
            "modules": ["demographics", "sofa2_score", "outcome"],
            "time_window": {
                "preset": "first_24h",
                "label": "First 24 hours",
                "observation_hours": 24,
            },
            "confirmations": {
                "extraction_completed": True,
                "guided_configuration_collected": True,
            },
        },
    )

    assert response.status_code == 200, response.json()
    assert response.json()["context"]["cohort"]["preset"] == "sepsis3"


def test_agent_run_rejects_study_context_source_mismatch(tmp_path: Path) -> None:
    client = TestClient(app)
    active_export = _write_export(tmp_path / "active")
    other_export = _write_export(tmp_path / "other", database="eicu")
    source_store.register_source(str(active_export), active=True, crossdb=True)
    context = client.post(
        "/api/study-contexts",
        json={
            "id": "study_mismatch",
            "question": "Compare mortality.",
            "data_source": {
                "path": str(other_export),
                "label": "Other export",
                "database": "eicu",
            },
        },
    ).json()["context"]

    response = client.post(
        "/api/jobs/agent-run",
        json={
            "path": str(active_export),
            "study_context_id": context["id"],
            "project_root": str(tmp_path / "projects"),
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"] == "study_context_source_mismatch"
    assert detail["study_context_id"] == "study_mismatch"
    assert detail["expected_path"] == str(other_export.resolve())
    assert detail["active_path"] == str(active_export.resolve())


def test_agent_run_blocks_crossdb_plan_until_aggregate_is_bound(
    tmp_path: Path,
) -> None:
    client = TestClient(app)
    export_dir = _write_export(tmp_path / "active")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    context = client.post(
        "/api/study-contexts",
        json={
            "id": "study_crossdb_plan",
            "question": "Plan a cross-database portability analysis.",
            "data_source": {"path": str(export_dir), "database": "miiv"},
            "cohort": {
                "review": "crossdb",
                "source_count": 3,
                "comparison_mode": "descriptive_only",
            },
            "confirmations": {"crossdb_plan_only": True},
        },
    ).json()["context"]

    response = client.post(
        "/api/jobs/agent-run",
        json={"path": str(export_dir), "study_context_id": context["id"]},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "error": "study_context_execution_not_supported",
        "study_context_id": context["id"],
        "reason": "crossdb_aggregate_not_bound_to_agent_runner",
    }

    stage_only = client.post(
        "/api/study-contexts",
        json={
            "id": "study_crossdb_stage_only",
            "question": "Legacy cross-database plan.",
            "data_source": {"path": str(export_dir), "database": "miiv"},
            "current_stage": "crossdb_plan_only",
        },
    ).json()["context"]
    stage_only_response = client.post(
        "/api/jobs/agent-run",
        json={"path": str(export_dir), "study_context_id": stage_only["id"]},
    )
    assert stage_only_response.status_code == 400
    assert stage_only_response.json()["detail"]["reason"] == (
        "crossdb_aggregate_not_bound_to_agent_runner"
    )


def test_agent_run_binds_context_and_uses_context_question_fallback(
    tmp_path: Path,
) -> None:
    client = TestClient(app)
    export_dir = _write_export(tmp_path / "active")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    question = "Is admission severity associated with hospital mortality?"
    context = client.post(
        "/api/study-contexts",
        json={
            "id": "study_bound",
            "title": "Bound study",
            "question": question,
            "purpose": "Use one project context across modules.",
            "data_source": {
                "path": str(export_dir),
                "label": "Active MIIV export",
                "database": "miiv",
            },
            "cohort": {"preset": "adult_first"},
            "modules": ["demographics", "outcome"],
            "outcome": "hospital mortality",
            "time_window": {"hours": 24},
            "analysis_goal": "Descriptive preflight",
        },
    ).json()["context"]

    started = client.post(
        "/api/jobs/agent-run",
        json={
            "path": str(export_dir),
            "study_context_id": context["id"],
            "study_id": "question-slug-must-not-split-history",
            "project_root": str(tmp_path / "projects"),
        },
    )

    assert started.status_code == 200
    assert started.json()["study_context_id"] == "study_bound"
    assert started.json()["study_context_revision"] == context["revision"] + 1
    snapshot = _wait_for_job(client, started.json()["job_id"])
    assert snapshot["status"] == "done", snapshot.get("error")
    result = snapshot["result"]
    assert result["study_id"] == "study_bound"
    assert result["study_context_id"] == "study_bound"
    assert result["study_context_revision"] == started.json()[
        "study_context_revision"
    ] + 1

    run_context = json.loads(
        (Path(result["project_dir"]) / "run_context.json").read_text(encoding="utf-8")
    )
    binding = run_context["context_binding"]
    assert run_context["question"] == question
    assert binding["status"] == "bound"
    assert binding["study_context_id"] == "study_bound"
    assert binding["context_revision"] == context["revision"]
    assert binding["applied"] == {
        "data_source.path": str(export_dir.resolve()),
        "question": question,
    }
    assert binding["applied_from"]["question"] == "study_context"
    assert binding["informational"]["cohort"] == {"preset": "adult_first"}
    assert binding["informational"]["analysis_goal"] == "Descriptive preflight"

    active = client.get("/api/study-contexts/active").json()["context"]
    assert active["active_job_id"] is None
    assert active["current_stage"] == "review"
    assert active["last_route"] == "agent"
    assert active["revision"] == result["study_context_revision"]


def test_agent_run_reports_post_submit_context_sync_failure_without_false_400(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = TestClient(app)
    export_dir = _write_export(tmp_path / "active")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    context = client.post(
        "/api/study-contexts",
        json={
            "id": "study_sync_warning",
            "question": "Audit the active export.",
            "data_source": {"path": str(export_dir), "database": "miiv"},
        },
    ).json()["context"]
    original_handoff = context_store.handoff_context
    calls = 0

    def flaky_handoff(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("write failed")
        return original_handoff(*args, **kwargs)

    monkeypatch.setattr(context_store, "handoff_context", flaky_handoff)

    started = client.post(
        "/api/jobs/agent-run",
        json={
            "path": str(export_dir),
            "study_context_id": context["id"],
            "project_root": str(tmp_path / "projects"),
        },
    )

    assert started.status_code == 200
    warning = started.json()["context_sync_warning"]
    assert warning["error"] == "study_context_active_job_sync_failed"
    assert warning["job_id"] == started.json()["job_id"]
    snapshot = _wait_for_job(client, started.json()["job_id"])
    assert snapshot["status"] == "done"


def test_agent_run_aborts_before_execution_on_context_revision_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = TestClient(app)
    export_dir = _write_export(tmp_path / "active")
    project_root = tmp_path / "projects"
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    context = client.post(
        "/api/study-contexts",
        json={
            "id": "study_submit_race",
            "question": "Old question",
            "data_source": {"path": str(export_dir), "database": "miiv"},
        },
    ).json()["context"]
    original_handoff = context_store.handoff_context
    raced = False

    def racing_handoff(context_id, **kwargs):
        nonlocal raced
        if not raced:
            raced = True
            context_store.upsert_context(
                {"id": context_id, "question": "New authoritative question"},
                expected_revision=context["revision"],
                require_revision=True,
                lifecycle_write=False,
            )
        return original_handoff(context_id, **kwargs)

    monkeypatch.setattr(context_store, "handoff_context", racing_handoff)
    response = client.post(
        "/api/jobs/agent-run",
        json={
            "path": str(export_dir),
            "study_context_id": context["id"],
            "project_root": str(project_root),
        },
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["error"] == "study_context_revision_conflict"
    assert detail["job_started"] is False
    snapshot = _wait_for_job(client, detail["job_id"])
    assert snapshot["status"] == "failed"
    assert "agent_run_start_blocked:study_context_revision_conflict" in snapshot[
        "error"
    ]
    current = client.get(f"/api/study-contexts/{context['id']}").json()["context"]
    assert current["question"] == "New authoritative question"
    assert current["current_stage"] == "plan"
    assert current["active_job_id"] is None
    assert not project_root.exists()


def test_agent_capacity_rejection_does_not_mutate_context_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = TestClient(app)
    export_dir = _write_export(tmp_path / "active")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    context = client.post(
        "/api/study-contexts",
        json={
            "id": "study_capacity",
            "question": "Audit the active export.",
            "data_source": {"path": str(export_dir), "database": "miiv"},
        },
    ).json()["context"]

    def reject_capacity(*_args, **_kwargs):
        raise HTTPException(
            status_code=429,
            detail={"error": "job_capacity_exceeded"},
        )

    monkeypatch.setattr(agent_routes, "submit_job", reject_capacity)
    response = client.post(
        "/api/jobs/agent-run",
        json={"path": str(export_dir), "study_context_id": context["id"]},
    )

    assert response.status_code == 429
    active = client.get("/api/study-contexts/active").json()["context"]
    assert active["current_stage"] == "plan"
    assert active["active_job_id"] is None


@pytest.mark.parametrize("failure", [OSError("read-only audit folder"), RuntimeError("bad settings")])
def test_agent_audit_write_failure_returns_job_id_instead_of_false_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    client = TestClient(app)
    export_dir = _write_export(tmp_path / "active")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    context = client.post(
        "/api/study-contexts",
        json={
            "id": "study_audit_warning",
            "question": "Audit the active export.",
            "data_source": {"path": str(export_dir), "database": "miiv"},
        },
    ).json()["context"]

    def fail_audit(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(capabilities, "record_tool_event", fail_audit)
    started = client.post(
        "/api/jobs/agent-run",
        json={
            "path": str(export_dir),
            "study_context_id": context["id"],
            "project_root": str(tmp_path / "projects"),
        },
    )

    assert started.status_code == 200
    assert started.json()["audit_warning"] == {
        "error": "agent_run_audit_write_failed",
        "job_id": started.json()["job_id"],
    }
    assert _wait_for_job(client, started.json()["job_id"])["status"] == "done"


def test_blocked_agent_gate_persists_review_blocked_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = TestClient(app)
    export_dir = _write_export(tmp_path / "active")
    source_store.register_source(str(export_dir), active=True, crossdb=True)
    context = client.post(
        "/api/study-contexts",
        json={
            "id": "study_blocked_gate",
            "question": "Run a blocked preflight.",
            "data_source": {"path": str(export_dir), "database": "miiv"},
        },
    ).json()["context"]

    monkeypatch.setattr(
        agent_routes.agent_runs,
        "make_agent_run_runner",
        lambda **_kwargs: lambda _job: {"gate": {"status": "blocked"}},
    )
    started = client.post(
        "/api/jobs/agent-run",
        json={"path": str(export_dir), "study_context_id": context["id"]},
    )

    assert started.status_code == 200
    assert _wait_for_job(client, started.json()["job_id"])["status"] == "done"
    active = client.get("/api/study-contexts/active").json()["context"]
    assert active["current_stage"] == "review_blocked"
    assert active["active_job_id"] is None


def test_terminal_job_cleanup_is_compare_and_set(
    tmp_path: Path,
) -> None:
    client = TestClient(app)
    first = client.post(
        "/api/study-contexts",
        json={"id": "study_cas", "question": "Compare concurrent runs."},
    ).json()["context"]
    context_store.handoff_context(first["id"], active_job_id="newer-job")

    stale = context_store.clear_active_job_if(
        first["id"],
        "older-job",
        current_stage="review",
    )

    assert stale["cleared"] is False
    assert stale["context"]["active_job_id"] == "newer-job"


def test_terminal_cleanup_does_not_reactivate_an_older_study() -> None:
    first = context_store.upsert_context({"id": "study_old"}, active=True)
    context_store.handoff_context(first["id"], active_job_id="old-job")
    context_store.upsert_context({"id": "study_new"}, active=True)

    cleared = context_store.clear_active_job_if(
        first["id"],
        "old-job",
        current_stage="review",
    )
    listed = context_store.list_contexts()

    assert cleared["cleared"] is True
    assert listed["active_id"] == "study_new"


def test_active_context_reconciles_job_lost_on_server_restart(tmp_path: Path) -> None:
    client = TestClient(app)
    created = client.post(
        "/api/study-contexts",
        json={"id": "study_restart", "question": "Resume after restart."},
    ).json()["context"]
    context_store.handoff_context(
        created["id"],
        current_stage="analyze",
        last_route="agent",
        active_job_id="missing-job",
    )

    active = client.get("/api/study-contexts/active").json()["context"]

    assert active["active_job_id"] is None
    assert active["current_stage"] == "agent_interrupted"


def test_context_retention_never_drops_the_active_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(context_store, "_MAX_CONTEXTS", 3)
    context_store.upsert_context({"id": "active-study"}, active=True)
    for context_id in ("inactive-1", "inactive-2", "inactive-3"):
        context_store.upsert_context({"id": context_id}, active=False)

    listed = context_store.list_contexts()

    assert listed["active_id"] == "active-study"
    assert len(listed["contexts"]) == 3
    assert "active-study" in {row["id"] for row in listed["contexts"]}


def test_context_store_and_atomic_temp_are_private() -> None:
    context_store.upsert_context(
        {"id": "private-study", "question": "Sensitive local research question."}
    )

    assert stat.S_IMODE(context_store._CONFIG_PATH.stat().st_mode) == 0o600
    assert stat.S_IMODE(context_store._CONFIG_PATH.parent.stat().st_mode) == 0o700
    assert not list(
        context_store._CONFIG_PATH.parent.glob(
            f".{context_store._CONFIG_PATH.name}.*.tmp"
        )
    )
