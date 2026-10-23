"""Synthetic long-run receipts must not take their conversation offline."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.webserver.pi_copilot import projections
from easyicu.webserver.pi_copilot.contracts import PiCopilotError


def _at(tick):
    return (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=tick)).isoformat()


def _snapshot(steps=30, repairs=True):
    trace = []
    for index in range(steps):
        operations = [("step", "step_started", 0), ("coder", "code_generation", 0),
                      ("runner", "run", 0)]
        if repairs:
            operations += [("coder", "repair", 1), ("runner", "run", 1)]
        for offset, (phase, action, attempt) in enumerate(operations):
            for edge, status in ((0, "running"), (1, "complete")):
                trace.append({
                    "phase": phase, "action": action, "attempt": attempt,
                    "event": f"Step {index + 1}/{steps} complete: analysis_{index}.",
                    "step_id": f"analysis_{index}", "status": status,
                    "timestamp": _at(index * 20 + offset * 2 + edge),
                    "group_variable_label": {"zh": "合成分组", "en":
                        "Synthetic grouping used to compare aggregate summaries across the analysis"},
                    "outcome_label": {"zh": "合成结局", "en":
                        "Synthetic outcome reported using the same aggregate definition for each step"},
                })
    return {
        "id": "job-synthetic", "kind": "agent-run", "status": "done",
        "created": 1767225600, "finished": 1767226400,
        "result": {
            "run_id": "run-synthetic", "trace_timings": trace,
            "gate": {"status": "analysis_only", "checks": [
                {"id": "execution_complete", "passed": True},
                {"id": "analysis_validated", "passed": True},
            ]},
            "artifacts": [{"name": name, "sha256": "a" * 64, "size": 128}
                          for name in ["result_tables.json", *[f"artifact_{i}.json" for i in range(17)]]],
        },
        "events": [{"type": "progress", "step": row["phase"],
                    "run_id": "run-synthetic", **row} for row in trace[-12:]],
    }


def _rows(job):
    table = job.get("progress_table")
    if not table:
        return job["progress"]
    return [{**table.get("defaults", {}), **{
        key: table["dictionaries"][key][value] if key in table.get("dictionaries", {}) else value
        for key, value in zip(table["columns"], cells) if value is not None}}
            for batch in table["batches"] for cells in batch]


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))


@pytest.mark.parametrize("steps,repairs", [(30, True), (17, False)])
def test_long_successful_job_keeps_all_operation_identities_and_both_times(steps, repairs):
    snapshot = _snapshot(steps, repairs)
    if steps == 17:
        # Seventy trace rows, eighteen artifact refs and live events.
        trace = [row for index, row in enumerate(snapshot["result"]["trace_timings"])
                 if index % 6 not in (3, 5)]
        trace += [{"phase": "planning", "action": "plan_validation",
                   "planning_unit": "plan", "retry_phase": phase, "current": 1,
                   "timestamp": _at(700 + index)}
                  for index, phase in enumerate(("started", "accepted"))]
        snapshot["result"]["trace_timings"] = trace
        assert len(trace) == 70
    before = copy.deepcopy(snapshot)
    job = projections.project_job(snapshot)
    assert _size(job) <= projections.MAX_PROJECTION_BYTES
    projections.ensure_safe_projection(job)
    assert job["progress_compacted"] is True
    assert not job.get("progress_withheld_reason")
    rows = _rows(job)
    for event in snapshot["result"]["trace_timings"]:
        row = next(row for row in rows if all(row.get(key) == event.get(key) for key in
                   ("step_id", "action", "attempt", "planning_unit", "retry_phase", "current"))
                   and row.get("step") == event["phase"])
        assert row["started_at"] <= event["timestamp"] <= row["ended_at"]
        assert row.get("run_id") == job["run_id"]
        for key in ("group_variable_label", "outcome_label"):
            if key in event:
                assert row[key] == event[key]
    assert len(job["artifact_refs"]) == 18
    assert job["execution_complete"] is True
    assert job["analysis_results_available"] is True
    assert snapshot == before


def test_irreducible_trace_is_explicitly_withheld_instead_of_losing_early_steps():
    job = projections.project_job(_snapshot(300))
    projections.ensure_safe_projection(job)
    assert job["progress"] == []
    assert job["progress_withheld_reason"] == "projection_budget"
    assert job["progress_operation_count"] == 1500
    assert job["run_id"] == "run-synthetic"
    assert job["execution_complete"] is True
    assert len(job["artifact_refs"]) == 18


def test_display_facts_are_removed_only_after_the_lossless_table_exceeds_budget():
    snapshot = _snapshot()
    for index, row in enumerate(snapshot["result"]["trace_timings"]):
        row["queries"] = [f"Synthetic aggregate search {index} term {term} " + "x" * 160
                          for term in range(4)]
    job = projections.project_job(snapshot)
    projections.ensure_safe_projection(job)
    assert job["progress_display_facts_omitted"] is True
    assert not job.get("progress_withheld_reason")
    assert len(_rows(job)) == 150
    assert all("queries" not in row for row in _rows(job))
    assert all(row.get("label") for row in _rows(job) if row["step"] == "step")
    assert not projections.project_job(_snapshot()).get("progress_display_facts_omitted")


def test_artifact_heavy_minimal_receipt_also_respects_the_total_byte_budget():
    snapshot = _snapshot()
    snapshot["result"]["run_id"] = "run_" + "a" * 150
    snapshot["result"]["artifacts"] = [
        {"name": f"artifact_{index}_" + "a" * 130 + ".json", "sha256": "a" * 64}
        for index in range(80)
    ]
    job = projections.project_job(snapshot)
    projections.ensure_safe_projection(job)
    assert _size(job) <= projections.MAX_PROJECTION_BYTES
    assert job["progress_withheld_reason"] == "projection_budget"
    assert job["artifact_refs_withheld_reason"] == "projection_budget"
    assert 0 < len(job["artifact_refs"]) < 80
    assert job["execution_complete"] is True


def test_absent_reason_codes_are_omitted_but_valid_codes_survive():
    job = projections.project_job({"id": "synthetic", "events": [
        {"type": "progress", "step": "planning"},
        {"type": "progress", "step": "writer", "reason": "synthetic_retry"},
    ]})
    assert "reason_code" not in job["progress"][0]
    assert job["progress"][1]["reason_code"] == "synthetic_retry"


def test_dictionary_labels_keep_each_steps_own_facts_without_inheriting_missing_labels():
    snapshot = _snapshot()
    for index, event in enumerate(snapshot["result"]["trace_timings"]):
        if index < 150:
            event["group_variable_label"] = {"zh": "另一合成分组", "en": "Another synthetic group"}
        elif event["phase"] == "coder":
            event.pop("group_variable_label")
    # Do not overwrite the audit's optional facts with a second live copy.
    snapshot["events"] = []
    job = projections.project_job(snapshot)
    assert not job.get("progress_display_facts_omitted")
    for row in _rows(job):
        index = int(row["step_id"].split("_")[-1])
        if index < 15:
            assert row["group_variable_label"]["en"] == "Another synthetic group"
        elif row["step"] == "coder":
            assert "group_variable_label" not in row
        else:
            assert row["group_variable_label"]["en"].startswith("Synthetic grouping")


@pytest.mark.parametrize("audit", [False, True])
def test_already_measured_endpoints_survive_both_live_and_audit_receipts(audit):
    event = {"type": "progress", "step": "planning", "phase": "planning",
             "action": "literature_build", "timestamp": _at(5), "status": "complete",
             "started_at": _at(1), "ended_at": _at(9)}
    job = projections.project_job({
        "id": "synthetic", "events": [] if audit else [event],
        "result": {"trace_timings": [event] if audit else []},
    })
    row = _rows(job)[0]
    assert row["started_at"] == _at(1)
    assert row["ended_at"] == _at(9)
    assert row["timestamp"] == _at(5)


@pytest.mark.parametrize("unsafe", [{"rows": [1]}, "/private/synthetic.csv"])
def test_oversized_progress_still_fails_closed_for_forbidden_fields(unsafe):
    snapshot = _snapshot(300)
    snapshot["events"].append({"type": "progress", "step": unsafe})
    with pytest.raises(PiCopilotError) as error:
        projections.project_job(snapshot)
    assert error.value.code == "pi_projection_blocked"


def test_compaction_keeps_planning_units_attempts_and_provider_attempts_distinct():
    snapshot = _snapshot()
    snapshot["events"] += [
        {"type": "progress", "step": "planning", "action": "plan_validation",
         "planning_unit": unit, "retry_phase": phase, "current": attempt,
         "timestamp": _at(700 + attempt)}
        for unit in ("structure", "rules", "step") for attempt in (1, 2)
        for phase in ("started", "accepted")
    ]
    job = projections.project_job(snapshot)
    actual = {(row["planning_unit"], row["retry_phase"], row["current"])
              for row in _rows(job) if row.get("action") == "plan_validation"}
    assert actual == {(unit, phase, attempt) for unit in ("structure", "rules", "step")
                      for attempt in (1, 2) for phase in ("started", "accepted")}


@pytest.mark.parametrize("forced_budget", [False, True])
def test_session_reload_archive_and_latest_job_survive_a_large_finished_job(
    tmp_path, monkeypatch, forced_budget,
):
    from easyicu.webserver import jobs
    from easyicu.webserver.pi_copilot.contracts import PiSessionRecord, PiSessionResearchJob
    from easyicu.webserver.pi_copilot.service import PiCopilotService

    snapshot = _snapshot()
    service = PiCopilotService(store_path=tmp_path / "sessions.json")
    record = PiSessionRecord(
        session_id="session-synthetic", project_id="project-synthetic",
        latest_research_job=PiSessionResearchJob(
            job_id=snapshot["id"], state="running", started_at=_at(0), at=_at(0),
        ),
    )
    service._write_records([record])
    monkeypatch.setattr(service, "_scoped_record", lambda session_id, **_: service._get_record(session_id))
    monkeypatch.setattr(service, "_ensure_open", lambda *_, **__: {})
    monkeypatch.setattr(service, "_stale_details", lambda _: {})
    monkeypatch.setattr(service, "_cohort_eligibility_selection_projection", lambda _: None)
    monkeypatch.setattr(service, "_backfill_host_action_row", lambda _: None)
    child = SimpleNamespace(id=snapshot["id"], kind="agent-run", status="done", snapshot=lambda: snapshot)
    monkeypatch.setattr(jobs.MANAGER, "get", lambda job_id: child if job_id == child.id else None)
    service.replay_store.record_host_action(
        session_id=record.session_id, project_id=record.project_id,
        action_id="host-synthetic", action_code="execute_plan", action_key="synthetic",
        child_job_id=child.id, status="running",
    )
    if forced_budget:
        def too_large(_):
            raise PiCopilotError("pi_projection_too_large", "Synthetic forced budget", status_code=500)
        monkeypatch.setattr(projections, "project_job", too_large)
    loaded = service.get_session(record.session_id, project_id=record.project_id)
    assert loaded["ok"] is True
    assert service._get_record(record.session_id).latest_research_job.state == "done"
    archived = service.archive_child_job(record.session_id, project_id=record.project_id, job_id=child.id)
    assert archived["ok"] is True
    receipt = archived["job"]
    projections.ensure_safe_projection(receipt)
    assert receipt["execution_complete"] is True
    assert receipt["run_id"] == "run-synthetic"
    assert len(receipt["artifact_refs"]) == 18
    if forced_budget:
        assert receipt["progress"] == []
        assert receipt["progress_withheld_reason"] == "projection_budget"
    assert loaded["session"]["archived_child_jobs"][0]["job_id"] == child.id


def test_replay_fallback_catches_only_size_and_checks_withheld_progress(monkeypatch):
    def fail(_):
        raise PiCopilotError("pi_projection_too_large", "Synthetic forced budget")
    monkeypatch.setattr(projections, "project_job", fail)
    snapshot = _snapshot()
    snapshot["events"].append({"type": "progress", "step": {"rows": [1]}})
    with pytest.raises(PiCopilotError) as error:
        projections.project_job_for_replay(snapshot)
    assert error.value.code == "pi_projection_blocked"

    def blocked(_):
        raise PiCopilotError("pi_projection_blocked", "Synthetic privacy refusal")
    monkeypatch.setattr(projections, "project_job", blocked)
    with pytest.raises(PiCopilotError) as error:
        projections.project_job_for_replay(_snapshot())
    assert error.value.code == "pi_projection_blocked"


def test_real_projection_is_decoded_in_the_browser_without_losing_step_durations():
    from tests.support.node import run_node

    root = Path(__file__).resolve().parents[3]
    script = (root / "tests/js/guided_pi_job_projection.test.js").read_text()
    result = run_node("node", script, json.dumps(projections.project_job(_snapshot())), cwd=root)
    assert result.returncode == 0, result.stderr
