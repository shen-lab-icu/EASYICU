"""A run's trace names the actions it took; a step shows the code it ran.

Synthetic run directories only. The step route reads one executed plan step
by its run and step ids: the code that ran (capped, host paths and secrets
redacted), how it ran, and the files it produced. Runner output and
row-level files never reach the page.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import agent_pipeline_runs, run_step_trace
from easyicu.webserver.app import app
from easyicu.webserver.jobs import Job
from easyicu.webserver.pi_copilot import service as service_module
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.projections import project_job
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.routes import pi_copilot as route_module
from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeGateway

ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "run_20260101T000000_synthetic"
LOG_ROW = "stay_id=30000123, glucose=212"
SECRET = "sk-" + "Q" * 24
BASELINE_CODE = (
    "import pandas as pd\n\n"
    "cohort = pd.read_parquet(INPUT)\n"
    "table = cohort.groupby('glucose_group').size()\n"
    "print(table)\n"
)
FIRST_DRAFT = "rate = cohort['death'].mean()\n"
REPAIRED_CODE = (
    "source = '/Users/someone/private/cohort.parquet'\n"
    f"key = '{SECRET}'\n"
    "rate = cohort.groupby('glucose_group')['death'].mean()\n"
)


def _png() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
        + chunk(b"IEND", b"")
    )


def _at(seconds: int) -> str:
    return f"2026-01-01T00:{seconds // 60:02d}:{seconds % 60:02d}+00:00"


def _audit(seconds: int, phase: str, event: str, **extra: Any) -> Dict[str, Any]:
    return {
        "timestamp": _at(seconds),
        "phase": phase,
        "event": event,
        "status": extra.pop("status", "running"),
        "step_id": extra.pop("step_id", None),
        "detail": extra,
    }


AUDIT = [
    _audit(0, "runtime", "Execution runtime validated before planning.", status="complete"),
    _audit(1, "context", "Research context built.", status="complete"),
    _audit(2, "context", "Exposure groups staged: glucose_group."),
    _audit(3, "hypothesis", "Building pre-plan literature and hypothesis blueprint."),
    _audit(10, "planning", "Study structure validation attempt 1/3 started."),
    _audit(14, "planning", "Study structure attempt 1/3 did not satisfy its contract; retrying.",
           validation_issues=["internal planner wording"]),
    _audit(15, "planning", "Study structure validation attempt 2/3 started."),
    _audit(19, "planning", "Study structure attempt 2/3 passed validation."),
    _audit(20, "plan", "Analysis plan ready with 4 step(s).", status="complete"),
    _audit(21, "step", "Skipped step already completed by pre-execution: cohort_accounting.",
           status="complete", step_id="cohort_accounting"),
    _audit(22, "step", "Step 2/4 started: baseline_table.", step_id="baseline_table"),
    _audit(23, "coder", "Using planner-specified grouped Table 1 executor for baseline_table.",
           step_id="baseline_table"),
    _audit(24, "runner", "Running standard executor script for baseline_table.",
           step_id="baseline_table"),
    _audit(30, "step", "Step 2/4 complete: baseline_table.", status="complete",
           step_id="baseline_table"),
    _audit(31, "step", "Step 3/4 started: outcome_distribution.", step_id="outcome_distribution"),
    _audit(32, "coder", "Generating analysis script for outcome_distribution.",
           step_id="outcome_distribution"),
    _audit(40, "runner", "Running generated script for outcome_distribution.",
           step_id="outcome_distribution"),
    _audit(41, "runner", f"Script failed: {LOG_ROW}", status="error",
           step_id="outcome_distribution", stderr=LOG_ROW),
    _audit(42, "coder", "Repairing failed script for outcome_distribution.",
           step_id="outcome_distribution", repair_attempts=1),
    _audit(50, "runner", "Running generated script for outcome_distribution.",
           step_id="outcome_distribution", repair_attempts=1),
    _audit(55, "step", "Step 3/4 complete: outcome_distribution.", status="complete",
           step_id="outcome_distribution"),
    _audit(56, "step", "Step 4/4 started: running_step.", step_id="running_step"),
    _audit(57, "coder", "Generating analysis script for running_step.", step_id="running_step"),
]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes | str) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = data.encode("utf-8") if isinstance(data, str) else data
    path.write_bytes(raw)
    return raw


def _synthetic_run(run: Path) -> Path:
    """Same shape as an executed pipeline run; every value is invented."""

    _write(run / "audit_log.jsonl", "\n".join(json.dumps(row) for row in AUDIT) + "\n")
    _write(run / "analysis_plan.json", json.dumps({"steps": [
        {"step_id": step} for step in (
            "cohort_accounting", "baseline_table", "outcome_distribution", "running_step",
            "planned_later",
        )
    ]}))
    _write(run / "preplan_literature_bundle.json", json.dumps({
        "search_provenance": {"search_queries": {
            "pubmed": ["sepsis glucose mortality", "glucose /Users/someone/notes.txt"],
        }},
        "citations": [{"pmid": "1"}, {"pmid": "2"}, {"pmid": "3"}],
    }))
    records: List[Dict[str, Any]] = []

    def register(eid: str, kind: str, step: str, name: str, data: bytes, mode: str = "") -> None:
        relative = f"evidence/{eid}__{name}"
        _write(run / relative, data)
        record = {"evidence_id": eid, "kind": kind, "relative_path": relative,
                  "sha256": _sha(data), "produced_by_step": step}
        if mode:
            record["generation_mode"] = mode
        records.append(record)

    _write(run / "steps/cohort_accounting/outputs/cohort.parquet", b"PAR1 synthetic rows PAR1")

    baseline = run / "steps/baseline_table"
    code = _write(baseline / "analysis.py", BASELINE_CODE)
    _write(baseline / "run.log", f"loaded cohort\n{LOG_ROW}\nTraceback: none\n")
    table = _write(baseline / "outputs/table_one.csv", "variable,group,n\nage,high,12\n")
    figure = _write(baseline / "outputs/figure.png", _png())
    _write(baseline / "outputs/figure.svg", "<svg xmlns='http://www.w3.org/2000/svg'/>")
    _write(baseline / "outputs/cohort.parquet", b"PAR1 synthetic rows PAR1")
    _write(baseline / "outputs/step_summary.json", json.dumps({
        "deterministic_standard_analysis": "grouped_table_one",
        "headline": {"en": "Baseline table built for 3 groups.", "zh": "已生成 3 组的基线特征表。"},
    }))
    register("code_analysis_baseline", "code", "baseline_table", "analysis.py", code,
             "deterministic_standard")
    register("table_baseline", "table", "baseline_table", "table_one.csv", table)
    register("figure_baseline", "figure", "baseline_table", "figure.png", figure)

    outcome = run / "steps/outcome_distribution"
    repaired = _write(outcome / "analysis.py", REPAIRED_CODE)
    _write(outcome / "run.log", f"{LOG_ROW}\nKeyError: death\n")
    _write(outcome / "outputs/step_summary.json", json.dumps({"status": "complete"}))
    register("code_analysis_outcome_1", "code", "outcome_distribution", "analysis.py",
             FIRST_DRAFT.encode(), "llm")
    register("code_analysis_outcome_2", "code", "outcome_distribution", "analysis.py",
             repaired, "repaired")

    running = run / "steps/running_step"
    _write(running / "analysis.py", "rows = 1\n")
    _write(running / "outputs/step_summary.json", json.dumps({
        "headline": {"en": "Not finished yet.", "zh": "尚未完成。"},
    }))
    _write(running / "outputs/partial.csv", "a\n1\n")
    _write(run / "evidence/evidence_index.json", json.dumps(records))
    return run


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    return _synthetic_run(tmp_path / "pipeline_run")


def test_finished_run_trace_rows_are_its_actions_with_times(run_dir: Path) -> None:
    rows = run_step_trace.trace_timings_projection(
        run_dir, source={"label": "MIMIC-IV demo", "database": "miiv"}
    )
    encoded = json.dumps(rows, ensure_ascii=False)
    assert len(rows) == len(AUDIT)
    assert [row["timestamp"] for row in rows] == [row["timestamp"] for row in AUDIT]
    for leaked in ("30000123", "stay_id", "/Users", "internal planner wording", "Script failed"):
        assert leaked not in encoded
    actions = [row.get("action") for row in rows]
    assert actions[:9] == [
        "runtime_validated", "context_built", "exposure_groups_staged", "literature_build",
        "plan_validation", "plan_validation", "plan_validation", "plan_validation", "plan_ready",
    ]
    assert rows[1]["source_label"] == "MIMIC-IV demo" and rows[1]["source_database"] == "miiv"
    assert rows[2]["group_variable_count"] == 1
    assert rows[3]["query_count"] == 2 and rows[3]["record_count"] == 3
    assert rows[3]["queries"] == ["sepsis glucose mortality"], "a query naming a host path stays local"
    assert [(row["retry_phase"], row["current"], row["total"]) for row in rows[4:8]] == [
        ("started", 1, 3), ("rejected", 1, 3), ("started", 2, 3), ("accepted", 2, 3),
    ]
    assert rows[8]["plan_step_count"] == 4
    by_step = {(row.get("step_id"), row.get("action"), row.get("attempt")) for row in rows}
    assert ("baseline_table", "standard_executor", None) in by_step
    assert rows[12]["run_kind"] == "standard"
    assert rows[12]["action"] == "run"
    assert rows[11]["executor"] == "grouped_table_one"
    assert ("outcome_distribution", "code_generation", None) in by_step
    assert ("outcome_distribution", "step_error", None) in by_step
    assert rows[18]["repair_kind"] == "runtime" and rows[18]["attempt"] == 1
    assert rows[19]["run_kind"] == "generated" and rows[19]["attempt"] == 1
    # Only the executor's fixed step envelopes keep their text.
    assert [row["event"] for row in rows if row["event"]] == [
        row["event"] for row in AUDIT if row["phase"] == "step"
    ]


def test_trace_action_reads_only_fixed_envelopes() -> None:
    assert run_step_trace.trace_action("coder", "Write anything you like for baseline_table.") == {}
    assert run_step_trace.trace_action(
        "coder", "Generating analysis script for baseline_table.", {"step_id": "other_step"}
    ) == {}, "a step-bound envelope must agree with its typed step id"
    assert run_step_trace.trace_action(
        "coder", "Repairing figure layout for fig_1.", {"visual_repair_attempts": 2, "repair_attempts": 5}
    ) == {"action": "repair", "step_id": "fig_1", "repair_kind": "figure_layout", "attempt": 2}
    assert run_step_trace.trace_action("runner", "Traceback ...", {"status": "error", "step_id": "a_b"}) == {
        "action": "step_error", "step_id": "a_b",
    }
    assert run_step_trace.trace_action("step", "Skipped later_step; required step first_step failed.") == {
        "action": "step_skipped", "step_id": "later_step",
    }


def test_live_events_carry_the_same_facts_and_keep_each_repair(run_dir: Path) -> None:
    job = Job("synthetic", "agent-run")
    for attempt in (1, 2):
        for stage, message in (
            ("coder", "Repairing failed script for outcome_distribution."),
            ("runner", "Running generated script for outcome_distribution."),
        ):
            agent_pipeline_runs._pipeline_progress(job, {
                "stage": stage, "message": message, "status": "running",
                "timestamp": _at(40 + attempt * 5 + (stage == "runner")),
                "step_id": "outcome_distribution", "run_id": RUN_ID, "repair_attempts": attempt,
            })
    agent_pipeline_runs._pipeline_progress(job, {
        "stage": "coder", "message": "Using planner-specified grouped Table 1 executor for baseline_table.",
        "step_id": "baseline_table", "run_id": RUN_ID, "timestamp": _at(60),
    })
    agent_pipeline_runs._pipeline_progress(job, {
        "stage": "runner", "message": f"Script failed: {LOG_ROW}", "status": "error",
        "step_id": "outcome_distribution", "run_id": RUN_ID, "timestamp": _at(61),
    })
    progress = project_job(job.snapshot())["progress"]
    repairs = [row for row in progress if row.get("action") == "repair"]
    assert [row["attempt"] for row in repairs] == [1, 2]
    assert all(row["run_id"] == RUN_ID and row["repair_kind"] == "runtime" for row in repairs)
    assert any(row.get("executor") == "grouped_table_one" for row in progress)
    assert any(row.get("action") == "step_error" for row in progress)
    assert LOG_ROW not in json.dumps(progress) and "30000123" not in json.dumps(progress)

    finished = project_job({"id": "synthetic", "kind": "agent-run", "status": "done", "result": {
        "run_id": RUN_ID, "trace_timings": run_step_trace.trace_timings_projection(run_dir),
    }})["progress"]
    validations = [row for row in finished if row.get("action") == "plan_validation"]
    assert len(validations) == 4, "each validation attempt and verdict keeps its own time"
    assert all(row.get("run_id") == RUN_ID for row in finished)
    assert {row.get("executor") for row in finished} >= {"grouped_table_one"}


def test_a_finished_run_keeps_each_repair_attempt_as_its_own_row(tmp_path: Path) -> None:
    rows = [
        _audit(42 + 8 * attempt + offset, phase, message, step_id="outcome_distribution",
               repair_attempts=attempt)
        for attempt in (1, 2)
        for offset, phase, message in (
            (0, "coder", "Repairing failed script for outcome_distribution."),
            (4, "runner", "Running generated script for outcome_distribution."),
        )
    ]
    _write(tmp_path / "audit_log.jsonl", "\n".join(json.dumps(row) for row in rows) + "\n")
    progress = project_job({"id": "synthetic", "kind": "agent-run", "status": "done", "result": {
        "run_id": RUN_ID, "trace_timings": run_step_trace.trace_timings_projection(tmp_path),
    }})["progress"]
    assert [(row["action"], row["attempt"], row["timestamp"]) for row in progress] == [
        ("repair", 1, _at(50)), ("run", 1, _at(54)), ("repair", 2, _at(58)), ("run", 2, _at(62)),
    ]


def test_step_detail_shows_code_how_it_ran_and_its_files(run_dir: Path) -> None:
    detail = run_step_trace.project_step_detail(run_dir, "baseline_table")
    encoded = json.dumps(detail, ensure_ascii=False)
    assert detail["state"] == "complete"
    assert detail["execution"] == {"mode": "standard", "executor": "grouped_table_one"}
    assert detail["code"]["text"] == BASELINE_CODE
    assert detail["code"]["language"] == "python"
    assert detail["code"]["line_count"] == 5 and detail["code"]["truncated"] is False
    # No owner redacts runner output for disclosure: the log stays local.
    assert detail["log"] == {"shown": False, "reason": "runner_output_stays_local"}
    assert LOG_ROW not in encoded and "30000123" not in encoded and "Traceback" not in encoded
    outputs = {row["name"]: row for row in detail["outputs"]}
    assert sorted(outputs) == ["figure.png", "figure.svg", "step_summary.json", "table_one.csv"]
    assert outputs["figure.png"]["thumbnail"].startswith("data:image/png;base64,iVBORw0KGgo")
    assert "thumbnail" not in outputs["figure.svg"]
    assert outputs["table_one.csv"]["evidence"] == {
        "id": "table_baseline", "sha256": _sha((run_dir / "steps/baseline_table/outputs/table_one.csv").read_bytes()),
    }
    assert outputs["step_summary.json"]["kind"] == "summary"
    assert "cohort.parquet" not in encoded and detail["withheld_output_count"] == 1
    assert detail["headline"] == {"en": "Baseline table built for 3 groups.", "zh": "已生成 3 组的基线特征表。"}
    assert str(run_dir) not in encoded and "/private/" not in encoded


def test_repaired_step_shows_the_code_that_ran_with_paths_and_secrets_redacted(run_dir: Path) -> None:
    detail = run_step_trace.project_step_detail(run_dir, "outcome_distribution")
    text = detail["code"]["text"]
    assert detail["execution"]["mode"] == "generated"
    assert "rate = cohort.groupby" in text, "the last registered script is the one that ran"
    assert "<host path>" in text and "/Users/" not in text
    assert SECRET not in text and "[REDACTED]" in text
    assert detail["code"]["redacted_paths"] == 1


def test_a_changed_script_or_table_loses_its_registry_link(run_dir: Path) -> None:
    (run_dir / "evidence/code_analysis_baseline__analysis.py").write_text("tampered = True\n")
    (run_dir / "steps/baseline_table/outputs/table_one.csv").write_text("variable\nchanged\n")
    detail = run_step_trace.project_step_detail(run_dir, "baseline_table")
    assert detail["code"] is None and detail["code_status"] == "digest_mismatch"
    table = next(row for row in detail["outputs"] if row["name"] == "table_one.csv")
    assert "evidence" not in table


def test_code_is_capped_at_a_line_boundary(run_dir: Path) -> None:
    line = "value = 'x' * 60  # synthetic filler line for the code cap\n"
    big = line * (run_step_trace.MAX_CODE_BYTES // len(line) + 50)
    _write(run_dir / "steps/large_step/analysis.py", big)
    detail = run_step_trace.project_step_detail(run_dir, "large_step")
    code = detail["code"]
    assert code["truncated"] is True and code["bytes"] == len(big.encode())
    assert len(code["text"].encode()) <= run_step_trace.MAX_CODE_BYTES
    assert code["text"].endswith("\n") and code["line_count"] == len(code["text"].splitlines())


def test_running_step_shows_code_and_no_output_until_it_finishes(run_dir: Path) -> None:
    detail = run_step_trace.project_step_detail(run_dir, "running_step")
    assert detail["state"] == "running"
    assert detail["code"]["text"] == "rows = 1\n"
    assert detail["execution"]["mode"] == "generated"
    assert detail["outputs"] == [] and "headline" not in detail
    assert detail["withheld_output_count"] == 0


def test_precompleted_step_names_no_row_level_file(run_dir: Path) -> None:
    detail = run_step_trace.project_step_detail(run_dir, "cohort_accounting")
    assert detail["state"] == "precompleted"
    assert detail["execution"]["mode"] == "host"
    assert detail["code"] is None
    assert detail["outputs"] == [] and detail["withheld_output_count"] == 1
    assert "parquet" not in json.dumps(detail)


@pytest.mark.parametrize("step_id", ["..", ".", "../baseline_table", "a/b", "", ".hidden", "a\\b", "x" * 161])
def test_invalid_and_traversal_step_ids_are_refused(run_dir: Path, step_id: str) -> None:
    with pytest.raises(run_step_trace.RunStepDetailError) as refused:
        run_step_trace.project_step_detail(run_dir, step_id)
    assert (refused.value.code, refused.value.status_code) == ("run_step_id_invalid", 400)


def test_unknown_and_symlinked_steps_are_not_followed(run_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(run_step_trace.RunStepDetailError) as missing:
        run_step_trace.project_step_detail(run_dir, "missing_step")
    assert (missing.value.code, missing.value.status_code) == ("run_step_not_found", 404)
    outside = tmp_path / "outside"
    _write(outside / "analysis.py", "OUTSIDE_SECRET = 1\n")
    _write(outside / "outputs/table.csv", "a\n1\n")
    os.symlink(outside, run_dir / "steps/linked_step")
    with pytest.raises(run_step_trace.RunStepDetailError):
        run_step_trace.project_step_detail(run_dir, "linked_step")
    os.symlink(outside, run_dir / "steps/planned_later")
    planned = run_step_trace.project_step_detail(run_dir, "planned_later")
    assert planned["code"] is None and planned["outputs"] == []
    assert planned["state"] == "not_started"
    (run_dir / "steps/running_step/analysis.py").unlink()
    os.symlink(outside / "analysis.py", run_dir / "steps/running_step/analysis.py")
    assert run_step_trace.project_step_detail(run_dir, "running_step")["code"] is None
    assert "OUTSIDE_SECRET" not in json.dumps(planned)


class _StepService:
    def __init__(self) -> None:
        self.seen: List[Dict[str, str]] = []

    def get_research_step(self, **kwargs: str) -> Dict[str, Any]:
        self.seen.append(kwargs)
        if kwargs["step_id"] == "missing_step":
            raise PiCopilotError("run_step_not_found", "The step does not belong to this run.", status_code=404)
        return {"ok": True, "run_id": kwargs["run_id"], "step": {"step_id": kwargs["step_id"]}}


def test_step_route_is_read_only_and_refuses_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _StepService()
    monkeypatch.setattr(route_module, "get_pi_copilot_service", lambda: fake)
    client = TestClient(app)
    base = f"/api/copilot/pi/projects/project-a/runs/{RUN_ID}/steps"
    response = client.get(f"{base}/baseline_table")
    assert response.status_code == 200
    assert fake.seen == [{"project_id": "project-a", "run_id": RUN_ID, "step_id": "baseline_table"}]
    assert client.get(f"{base}/missing_step").status_code == 404
    for bad in ("..%2Fsecret", ".hidden", "a%5Cb", "%2E%2E"):
        assert client.get(f"{base}/{bad}").status_code in {404, 422}, bad
    assert client.post(f"{base}/baseline_table").status_code == 405
    assert len(fake.seen) == 2


def test_service_reads_a_step_only_through_the_projects_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=FakeGateway())
    service.project_store.bind("project-a", "study-a")
    service.project_store.bind("project-b", "study-b")
    wrapper = tmp_path / "wrapper"
    _synthetic_run(wrapper / "pipeline" / RUN_ID)
    monkeypatch.setattr(
        service_module.agent_runs,
        "list_run_history",
        lambda **kwargs: {"runs": [{"run_id": RUN_ID, "project_dir": str(wrapper)}]}
        if kwargs.get("study_id") == "study-a" else {"runs": []},
    )
    payload = service.get_research_step(project_id="project-a", run_id=RUN_ID, step_id="baseline_table")
    encoded = json.dumps(payload, ensure_ascii=False)
    assert payload["run_id"] == RUN_ID and payload["step"]["code"]["text"] == BASELINE_CODE
    assert str(tmp_path) not in encoded and "project_dir" not in encoded
    with pytest.raises(PiCopilotError) as other:
        service.get_research_step(project_id="project-b", run_id=RUN_ID, step_id="baseline_table")
    assert (other.value.code, other.value.status_code) == ("pi_research_run_not_found", 404)
    with pytest.raises(PiCopilotError) as invalid:
        service.get_research_step(project_id="project-a", run_id=RUN_ID, step_id="../steps")
    assert (invalid.value.code, invalid.value.status_code) == ("run_step_id_invalid", 400)


def test_browser_rows_and_step_panel_contract() -> None:
    js = "src/easyicu/webserver/static/js"
    result = run_node(
        "node",
        (ROOT / "tests/js/guided_pi_step_detail.test.js").read_text(encoding="utf-8"),
        f"{js}/screens-guided-pi-activity.js",
        f"{js}/screens-guided-pi-step-detail.js",
        f"{js}/screens-guided-pi-source-view.js",
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "step detail contracts passed" in result.stdout
