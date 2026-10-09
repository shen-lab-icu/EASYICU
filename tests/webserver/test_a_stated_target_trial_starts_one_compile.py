"""A stated target trial starts one compile job, or says why it does not.

Copilot states a causal study's trial and population; the host checks the
statement against its menus and names the fields it refuses, refuses a study
of another family, a study without a prepared data package, a study whose own
job runs and a second compile, and otherwise starts one job that holds the
study while it runs; every refusal comes before the conversation spends its
grant.  The job keeps what was asked and what it found: a
database v1 does not emulate trials in stops before any data is read, and an
unexpected error fails with a stable code that keeps its cause.  The conversation sees the
result as codes and numbers, and the tool's arguments are the host's own
menus.  Synthetic studies and manifests only; no patient row is read.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from easyicu.webserver import jobs, target_trial_records
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver import target_trial_setup as setup
from tests.support.target_trial import (
    GRACE,
    STUDY_ID,
    target_trial_population,
    target_trial_spec,
)

_CAUSAL = {
    "analysis_family": "causal_inference",
    "analysis_unit": "icu_stay",
    "variance_estimator": "model_based",
}


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        context_store, "_CONFIG_PATH", tmp_path / "cfg" / "study-contexts.json"
    )
    monkeypatch.setattr(jobs, "MANAGER", jobs.JobManager())


def _statement(**changes: Any) -> setup.TargetTrialStatement:
    return setup.read_target_trial_statement(
        {
            "spec": target_trial_spec(**changes).model_dump(mode="json"),
            "population_spec": target_trial_population().model_dump(mode="json"),
        }
    )


def _export(root: Path, *, database: str) -> Path:
    """A prepared package's manifest: the setup reads nothing else of it."""

    root.mkdir(parents=True)
    (root / "_manifest.json").write_text(
        json.dumps({"database": database, "cohort_contract": {}}), encoding="utf-8"
    )
    return root


def _study(tmp_path: Path, *, database: str = "miiv", **fields: Any) -> dict:
    export = _export(tmp_path / "export", database=database)
    return context_store.upsert_context(
        {
            "id": STUDY_ID,
            "question": "q",
            "analysis_design": dict(_CAUSAL),
            "data_source": {"path": str(export), "database": database},
            **fields,
        }
    )


def _finished(job: Any, seconds: float = 20.0) -> Any:
    deadline = time.monotonic() + seconds
    while job.status == "running":
        if time.monotonic() > deadline:
            raise AssertionError("the compile job did not finish")
        time.sleep(0.01)
    return job


def _gate_free(seconds: float = 20.0) -> bool:
    """Whether the compile gate frees: a refused start's job releases it as it ends."""

    deadline = time.monotonic() + seconds
    while not setup._COMPILE_GATE.acquire(blocking=False):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    setup._COMPILE_GATE.release()
    return True


def _refused(study: dict, statement=None) -> setup.TargetTrialSetupError:
    with pytest.raises(setup.TargetTrialSetupError) as caught:
        setup.submit_target_trial_compile(study, statement or _statement())
    return caught.value


def test_the_statement_names_the_fields_the_host_refuses() -> None:
    spec = target_trial_spec().model_dump(mode="json")
    spec["grace_period"]["hours"] = 0

    with pytest.raises(setup.TargetTrialSetupError) as caught:
        setup.read_target_trial_statement({"spec": spec, "population_spec": None})

    error = caught.value
    assert (error.code, error.status_code) == ("target_trial_statement_invalid", 422)
    assert "spec.grace_period.hours" in error.details["fields"]
    assert any(field.startswith("population_spec") for field in error.details["fields"])


def test_a_statement_names_its_request_by_what_it_states() -> None:
    first, again = _statement().request(), _statement().request()
    other = _statement(grace_period={"hours": GRACE + 1}).request()

    assert first == again
    assert set(first) == {"spec", "population_spec", "request_sha256"}
    assert other["request_sha256"] != first["request_sha256"]


def _statement_params(**changes: Any) -> dict[str, Any]:
    return {
        "spec": target_trial_spec(**changes).model_dump(mode="json"),
        "population_spec": target_trial_population().model_dump(mode="json"),
    }


def test_the_host_refuses_before_the_grant_and_before_any_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_job(*_: Any) -> None:
        raise AssertionError("a refused statement started a job")

    monkeypatch.setattr(jobs.MANAGER, "submit", no_job)
    study = _study(tmp_path)
    invalid = _statement_params()
    invalid["spec"]["grace_period"]["hours"] = 0
    raw_folder = tmp_path / "raw"
    raw_folder.mkdir()
    other_family = context_store.upsert_context(
        {"id": "study_other0001", "question": "q", "data_source": study["data_source"]}
    )
    unbound = context_store.upsert_context(
        {"id": "study_unbound01", "question": "q", "analysis_design": dict(_CAUSAL)}
    )
    not_prepared = context_store.upsert_context(
        {
            "id": "study_raw000001",
            "question": "q",
            "analysis_design": dict(_CAUSAL),
            "data_source": {"path": str(raw_folder), "database": "miiv"},
        }
    )

    def refused(row: dict, params: dict) -> str:
        with pytest.raises(setup.TargetTrialSetupError) as checked:
            setup.check_target_trial_statement(row, params)
        if checked.value.code != "target_trial_statement_invalid":
            assert _refused(row).code == checked.value.code
        return checked.value.code

    # The family comes first: a statement is not read for a study of another.
    assert refused(other_family, invalid) == "target_trial_family_mismatch"
    assert refused(study, invalid) == "target_trial_statement_invalid"
    assert refused(unbound, _statement_params()) == "target_trial_export_required"
    assert refused(not_prepared, _statement_params()) == "target_trial_export_required"
    running = context_store.handoff_context(
        STUDY_ID, active_job_id="job_running", expected_revision=study["revision"]
    )
    assert refused(running, _statement_params()) == "study_job_running"
    assert setup._COMPILE_GATE.acquire(blocking=False)
    try:
        assert refused(study, _statement_params()) == "target_trial_compile_busy"
    finally:
        setup._COMPILE_GATE.release()
    assert setup.latest_target_trial_compile(STUDY_ID) is None
    error = setup.TargetTrialSetupError("target_trial_compile_busy", "busy")
    assert error.tool_result() == {
        "status": "blocked",
        "code": "target_trial_compile_busy",
        "summary": "busy",
        "owner": "easyicu.webserver.target_trial_setup",
        "details": {},
    }


def test_a_statement_the_study_can_compile_is_checked_without_holding_the_gate(
    tmp_path: Path,
) -> None:
    study = _study(tmp_path)

    statement = setup.check_target_trial_statement(study, _statement_params())

    assert statement.request() == _statement().request()
    assert _gate_free(seconds=0.0)


def test_a_compile_in_another_database_stops_before_it_reads_data(
    tmp_path: Path,
) -> None:
    study = _study(
        tmp_path, database="eicu", current_stage="data", last_route="extraction"
    )
    statement = _statement()

    job = _finished(setup.submit_target_trial_compile(study, statement))

    assert job.status == "done"
    result = job.result["target_trial_compile"]
    assert (result["status"], result["reason_code"]) == (
        "stopped",
        "target_trial_database_out_of_scope",
    )
    kept = setup.target_trial_job_record(STUDY_ID, job.id)
    assert kept["request"] == statement.request()
    assert kept["result"] == result
    latest = setup.latest_target_trial_compile(STUDY_ID)
    assert (latest["job_id"], latest["status"], latest["reason_code"]) == (
        job.id,
        "stopped",
        "target_trial_database_out_of_scope",
    )
    assert latest["detail"] == result["detail"]
    shown = setup.project_target_trial_compile(result)
    assert (shown["status"], shown["reason_code"]) == (
        "stopped",
        "target_trial_database_out_of_scope",
    )
    after = context_store.get_context(STUDY_ID)
    assert after["active_job_id"] is None
    assert (after["current_stage"], after["last_route"]) == ("data", "extraction")
    assert after.get("target_trial_design") in (None, {})


def test_a_later_compile_is_the_latest(tmp_path: Path) -> None:
    study = _study(tmp_path, database="eicu")
    first = _finished(setup.submit_target_trial_compile(study, _statement()))
    second = _finished(
        setup.submit_target_trial_compile(
            context_store.get_context(STUDY_ID), _statement(grace_period={"hours": GRACE + 1})
        )
    )

    assert setup.latest_target_trial_compile(STUDY_ID)["job_id"] == second.id
    assert setup.target_trial_job_record(STUDY_ID, first.id)["job_id"] == first.id
    assert setup.target_trial_job_record(STUDY_ID, "../escape") is None


def test_a_failed_compile_keeps_a_stable_code_and_its_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: Any, **__: Any) -> dict:
        raise KeyError("a lower layer broke")

    monkeypatch.setattr(setup, "_compile", broken)
    study = _study(tmp_path)

    job = _finished(setup.submit_target_trial_compile(study, _statement()))

    assert job.status == "failed"
    assert job.error.split(":", 1)[0] == "target_trial_compile_failed"
    result = setup.target_trial_job_record(STUDY_ID, job.id)["result"]
    assert result["status"] == "failed"
    assert result["reason_code"] == "target_trial_compile_failed"
    assert result["cause_code"] == "KeyError"
    latest = setup.latest_target_trial_compile(STUDY_ID)
    # The card and the conversation name the lower layer's cause.
    assert (latest["status"], latest["reason_code"], latest["cause_code"]) == (
        "failed",
        "target_trial_compile_failed",
        "KeyError",
    )
    assert setup.project_target_trial_compile(result)["cause_code"] == "KeyError"
    assert context_store.get_context(STUDY_ID)["active_job_id"] is None
    # The gate is free again.
    again = _finished(
        setup.submit_target_trial_compile(context_store.get_context(STUDY_ID), _statement())
    )
    assert again.status == "failed"


def test_a_compile_that_left_no_result_reads_as_interrupted(tmp_path: Path) -> None:
    study = _study(tmp_path, database="eicu")
    job = _finished(setup.submit_target_trial_compile(study, _statement()))
    # The host stopped while the job ran: its record keeps the request only.
    (path,) = target_trial_records.records_root().rglob(f"jobs/*-{job.id}.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**record, "result": None}), encoding="utf-8")
    assert jobs.MANAGER.get(job.id).status != "running"

    latest = setup.latest_target_trial_compile(STUDY_ID)

    assert (latest["status"], latest["reason_code"]) == (
        "interrupted",
        setup.TARGET_TRIAL_COMPILE_INTERRUPTED,
    )
    assert latest["detail"] and latest["missing_concepts"] == []
    assert latest["cause_code"] is None


def test_a_study_changed_before_the_start_starts_nothing(tmp_path: Path) -> None:
    study = _study(tmp_path)
    context_store.upsert_context({"id": STUDY_ID, "question": "q2"})

    error = _refused(study)

    assert error.code == "study_context_revision_conflict"
    assert _gate_free()
    assert setup.latest_target_trial_compile(STUDY_ID) is None
    assert context_store.get_context(STUDY_ID)["active_job_id"] is None


def test_the_tool_hands_the_conversation_its_job(tmp_path: Path) -> None:
    study = _study(tmp_path, database="eicu")

    job = setup.submit_target_trial_compile(study, _statement())
    submitted = setup.submitted_tool_result(study, job)
    _finished(job)

    assert (submitted["status"], submitted["code"]) == (
        "ok",
        "easyicu_target_trial_compile_submitted",
    )
    assert submitted["details"]["job_id"] == job.id
    assert submitted["details"]["kind"] == "target-trial-compile"
    assert submitted["details"]["study_context_id"] == STUDY_ID


def test_the_conversation_sees_a_compile_as_codes_and_numbers() -> None:
    shown = setup.project_target_trial_compile(
        {
            "status": "stopped",
            "reason_code": "target_trial_data_unavailable",
            "detail": "x" * 1000,
            "missing_concepts": ["vaso_ind", "/Users/someone/export/lact.parquet"],
            "metrics": {"acquisition_seconds": 1.5, "rows": 12},
            "compile_sha256": "not-a-digest",
        }
    )

    assert shown == {
        "status": "stopped",
        "reason_code": "target_trial_data_unavailable",
        "detail": "x" * 400,
        "missing_concepts": ["vaso_ind"],
        "metrics": {"acquisition_seconds": 1.5},
    }
    compiled = setup.project_target_trial_compile(
        {"status": "compiled", "compile_sha256": "a" * 64, "approvable": True}
    )
    assert (compiled["compile_sha256"], compiled["approvable"]) == ("a" * 64, True)
    for raw in (None, {}, {"status": "running"}, ["compiled"]):
        assert setup.project_target_trial_compile(raw) is None


def test_the_tool_arguments_are_the_hosts_own_menus() -> None:
    schema = setup.target_trial_statement_schema()
    validator = jsonschema.Draft202012Validator(schema)

    assert not any(marker in json.dumps(schema) for marker in ("$ref", "$defs", "discriminator"))
    validator.validate(_statement_params())
    for broken in (
        {**_statement_params(), "approve": True},
        {"spec": _statement_params()["spec"]},
        {
            **_statement_params(),
            "population_spec": {"criteria": [{"id": "c1", "kind": "not_a_menu_item"}]},
        },
    ):
        assert not validator.is_valid(broken)
