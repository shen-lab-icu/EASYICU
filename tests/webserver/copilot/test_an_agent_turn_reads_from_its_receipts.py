"""An agent turn reads from its receipts: counts, a step's inputs and times.

Synthetic run directories only. The trace carries the research context's own
stay and variable counts; one step's detail names its data source, the stays
and variables it read and its place in the plan, and its own start and end --
each fact present only when a receipt records it, and none of them a path. A
wait for the model service names which bounded continuation it waits for.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from easyicu.webserver import provider_job_recovery, run_step_trace
from easyicu.webserver.jobs import Job
from easyicu.webserver.pi_copilot.projections import project_job

ROOT = Path(__file__).resolve().parents[3]
STATIC = ROOT / "src" / "easyicu" / "webserver" / "static"
INDEX = STATIC / "index.html"


def _audit(second: int, phase: str, event: str, **extra: Any) -> Dict[str, Any]:
    return {
        "timestamp": f"2026-01-01T00:{second // 60:02d}:{second % 60:02d}+00:00",
        "phase": phase,
        "event": event,
        "status": extra.pop("status", "running"),
        "step_id": extra.pop("step_id", None),
        "detail": extra,
    }


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run_20260101T000000_turn"
    run.mkdir()
    events = [
        _audit(1, "context", "Research context built.", status="complete", n_stays=1841, n_variables=37),
        _audit(2, "step", "Skipped step already completed by pre-execution: cohort_accounting.",
               status="complete", step_id="cohort_accounting"),
        _audit(10, "step", "Step 2/3 started: outcome_model.", step_id="outcome_model"),
        # A second attempt of the same step: its start stays the first one.
        _audit(30, "step", "Step 2/3 failed contract checks: outcome_model.", status="error", step_id="outcome_model"),
        _audit(31, "step", "Step 2/3 started: outcome_model.", step_id="outcome_model"),
        _audit(73, "step", "Step 2/3 complete: outcome_model.", status="complete", step_id="outcome_model"),
    ]
    (run / "audit_log.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")
    plan = {"steps": [
        {"step_id": "cohort_accounting", "inputs": []},
        {"step_id": "outcome_model",
         "inputs": ["age", "sofa_total", "age", " ", "artifact:cohort_table", "table:labs", "lactate_max", 7]},
        {"step_id": "robustness"},
    ]}
    (run / "analysis_plan.json").write_text(json.dumps(plan))
    return run


@pytest.mark.parametrize(("detail", "expected"), [
    ({"n_stays": 1841, "n_variables": 37}, {"stay_count": 1841, "variable_count": 37}),
    ({"n_stays": 4_500_000}, {"stay_count": 4_500_000}),
    ({"n_stays": 10_000_001, "n_variables": -1}, {}),
    ({"n_stays": True, "n_variables": "many"}, {}),
    ({}, {}),
])
def test_the_context_event_carries_its_own_bounded_counts(detail, expected):
    facts = run_step_trace.trace_action("context", "Research context built.", detail)
    assert facts.pop("action") == "context_built"
    assert facts == expected


def test_counts_belong_to_the_context_event_only():
    facts = run_step_trace.trace_action("runtime", "Execution runtime validated before planning.",
                                        {"n_stays": 9, "n_variables": 9})
    assert "stay_count" not in facts and "variable_count" not in facts


def test_the_job_projection_keeps_the_counts_within_their_bound():
    job = Job("synthetic", "agent-run")
    job.emit({"type": "progress", "step": "context", "label": "Research context built.",
              "action": "context_built", "stay_count": 1841, "variable_count": 37})
    job.emit({"type": "progress", "step": "context", "label": "Research context built.",
              "action": "context_built", "stay_count": 10_000_001, "variable_count": False})
    first, second = project_job(job.snapshot())["progress"][:2]
    assert (first["stay_count"], first["variable_count"]) == (1841, 37)
    assert "stay_count" not in second and "variable_count" not in second


def test_a_step_detail_names_what_it_read_and_when_it_ran(run_dir):
    detail = run_step_trace.project_step_detail(
        run_dir, "outcome_model", source={"label": "eICU-CRD demo", "database": "eicu", "path": "/Users/x/eicu"},
    )
    assert detail["inputs"] == {
        "source_label": "eICU-CRD demo", "source_database": "eicu", "stay_count": 1841,
        # Distinct untyped plan inputs: products and evidence are not variables.
        "variable_count": 3, "step_index": 2, "step_total": 3,
    }
    assert detail["timing"] == {"started_at": "2026-01-01T00:00:10+00:00", "ended_at": "2026-01-01T00:01:13+00:00"}
    assert not re.search(r"/Users/|/Volumes/|run_20260101", json.dumps(detail))


def test_a_step_detail_states_nothing_no_receipt_records(run_dir):
    (run_dir / "audit_log.jsonl").write_text(json.dumps(
        _audit(5, "step", "Step 3/3 started: robustness.", step_id="robustness")) + "\n")
    detail = run_step_trace.project_step_detail(run_dir, "robustness")
    assert detail["inputs"] == {"step_index": 3, "step_total": 3}
    assert detail["timing"] == {"started_at": "2026-01-01T00:00:05+00:00"}
    # A step that pre-execution completed ends at its receipt and has no start.
    assert run_step_trace.project_step_detail(run_dir, "cohort_accounting")["timing"] == {}
    (run_dir / "audit_log.jsonl").write_text(json.dumps(
        _audit(2, "step", "Skipped step already completed by pre-execution: cohort_accounting.",
               status="complete", step_id="cohort_accounting")) + "\n")
    precompleted = run_step_trace.project_step_detail(run_dir, "cohort_accounting")
    assert precompleted["timing"] == {"ended_at": "2026-01-01T00:00:02+00:00"}
    # Its plan names no variable, so no count is stated.
    assert precompleted["inputs"] == {"step_index": 1, "step_total": 3}


def test_a_source_label_that_is_a_path_does_not_reach_the_page(run_dir):
    detail = run_step_trace.project_step_detail(
        run_dir, "outcome_model", source={"label": "/Users/someone/private/eicu", "database": "eicu"})
    assert "source_label" not in detail["inputs"]
    assert "someone" not in json.dumps(detail)


def test_the_wrapper_names_its_source_only_through_its_run_context(tmp_path, monkeypatch):
    wrapper = tmp_path / "wrapper"
    wrapper.mkdir()
    assert run_step_trace.wrapper_source(str(wrapper)) is None
    (wrapper / "run_context.json").write_text(json.dumps({"source": {"label": "eICU", "database": "eicu"}}))
    # A run without a wrapper never reads the server's working directory.
    monkeypatch.chdir(wrapper)
    assert run_step_trace.wrapper_source("") is None
    (wrapper / "run_context.json").write_text(json.dumps({"source": {"label": "MIMIC-IV", "database": "miiv"}}))
    assert run_step_trace.wrapper_source(str(wrapper)) == {"label": "MIMIC-IV", "database": "miiv"}
    (wrapper / "run_context.json").write_text(json.dumps({"source": "MIMIC-IV"}))
    assert run_step_trace.wrapper_source(str(wrapper)) is None


@pytest.mark.parametrize("spent", [0, 1])
def test_a_wait_for_the_model_service_names_its_continuation(monkeypatch, spent):
    owner = provider_job_recovery
    job = Job("outage", "agent-run")
    state = owner.ProviderJobRecovery(job)
    state.task = SimpleNamespace(assert_active=lambda: 600.0)
    state.client = object()
    state.deadline = owner.time.monotonic() + owner.RECOVERY_WINDOW_SECONDS
    state.resumes = [{"number": index + 1} for index in range(spent)]

    class Probe:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def complete(self, *args: Any, **kwargs: Any) -> str:
            return "OK"

    monkeypatch.setattr(owner, "HardStopClient", Probe)
    monkeypatch.setattr(owner.time, "sleep", lambda seconds: None)
    state.wait_for_provider("provider_http_error")
    waits = [row for row in project_job(job.snapshot())["progress"] if row.get("step") == "provider_recovery_wait"]
    assert len(waits) == 1
    assert (waits[0]["current"], waits[0]["total"]) == (spent + 1, owner.MAX_AUTOMATIC_RESUMES)


def test_the_page_loads_each_turn_owner_before_its_reader():
    index = INDEX.read_text(encoding="utf-8")

    def at(name: str, version: str = "20261021-turn-layout1") -> int:
        position = index.find(f"{name}?v={version}")
        assert position >= 0, name
        return position

    assert 0 <= index.find("css/guided-pi-skin.css?") < at("css/guided-pi-turn-layout.css")
    assert at("js/screens-guided-pi-job-narration.js") < at("js/screens-guided-pi-activity.js", "20261021-named-steps1")
    assert at("js/screens-guided-pi-run-stop.js") < at("js/screens-guided-pi-session-view.js", "20261021-where1")
    assert at("js/screens-guided-pi-session-view.js", "20261021-where1") < at("js/screens-guided-pi.js", "20261021-help1")


def test_the_trace_window_and_the_run_record_link_have_their_owners():
    css = (STATIC / "css" / "guided-pi-turn-layout.css").read_text(encoding="utf-8")
    # About 320px of trace, scrolling; a live window keeps the newest row in view.
    assert ".gpi-workspace .gpi-activity-body.is-clipped>ol{max-height:320px}" in css
    assert ".gpi-trace-live.is-clipped>.gpi-trace-window{display:flex;flex-direction:column-reverse;max-height:320px;overflow-y:auto}" in css
    reduced = css[css.index("@media (prefers-reduced-motion:reduce)"):]
    assert ".gpi-running-spinner{animation:none}" in reduced
    # The running chip shows only where the top bar has room for it.
    assert "@media(min-width:1181px){\n  .gpi-head-title .gpi-live.is-running{display:inline-flex" in css
    events = (STATIC / "js" / "screens-guided-pi-events.js").read_text(encoding="utf-8")
    assert "if (event.target.closest('[data-gpi-open-run-record]')) { if (ASIDE && ASIDE.revealRunRecord) ASIDE.revealRunRecord(); return; }" in events
    aside = (STATIC / "js" / "screens-guided-pi-aside.js").read_text(encoding="utf-8")
    assert "return { syncProjectWorkflowAside, togglePanel, layoutOptions, revealRunRecord };" in aside
    assert "record.scrollIntoView({ block: 'nearest' });" in aside
