"""A fresh plan's run context names the compiled candidate it continues (B10).

The launch records it only when the compiled lineage held then; its review
reads it so the host does not compile the same configuration again. Synthetic
study and acquisition; no benchmark item. A run made before this record
existed (v0.1.0 and earlier) reads exactly as before and is not treated as a
continuation (S1480).
"""

import json

import pytest

from easyicu.webserver import agent_pipeline_runs, agent_runs
from tests.support.v010_run_context import V010_RUN_CONTEXT
from tests.webserver.copilot.research_workflow_fixtures import _acquisition_receipt, complete_study


@pytest.mark.parametrize("recorded", ["run-compiled", ""])
def test_the_run_context_records_the_compiled_candidate_only_when_given(tmp_path, recorded):
    wrapper = tmp_path / "wrapper"
    agent_pipeline_runs._write_projection(
        wrapper_dir=wrapper, study=complete_study(), provider={},
        acquisition=_acquisition_receipt(), run_dir=None,
        blocked_reason="data_foundation_blocked",
        compiled_configuration_source_run_id=recorded,
    )
    context = json.loads((wrapper / "run_context.json").read_text())
    public = agent_runs._public_review_payloads({"run_context.json": context})["run_context.json"]
    if recorded:
        assert context["compiled_configuration_source_run_id"] == public["compiled_configuration_source_run_id"] == recorded
    else:
        assert "compiled_configuration_source_run_id" not in context
        assert "compiled_configuration_source_run_id" not in public


@pytest.mark.parametrize("recorded", [None, 7, ["run-a"], ""])
def test_a_run_context_without_a_string_record_projects_as_before(recorded):
    context = dict(V010_RUN_CONTEXT, **({} if recorded is None else {"compiled_configuration_source_run_id": recorded}))
    public = agent_runs._public_review_payloads({"run_context.json": context})["run_context.json"]
    before = agent_runs._public_review_payloads({"run_context.json": V010_RUN_CONTEXT})["run_context.json"]
    assert public == before and "compiled_configuration_source_run_id" not in public
    assert public["run_id"] == "run-v010" and public["summary"] == V010_RUN_CONTEXT["summary"]


def test_a_v010_review_is_not_read_as_a_continuation():
    from easyicu.webserver.pi_copilot.plan_review_progress import recorded_compiled_source_run_id as recorded

    assert recorded({"artifact_payloads": {"run_context.json": dict(V010_RUN_CONTEXT)}}) == ""
    assert recorded({"artifact_payloads": {"run_context.json": {"compiled_configuration_source_run_id": 7}}}) == ""
    assert recorded({"artifact_payloads": {}}) == "" and recorded({}) == ""
    assert recorded({"artifact_payloads": {"run_context.json": {"compiled_configuration_source_run_id": "run-a"}}}) == "run-a"


@pytest.mark.parametrize("recorded", ["run-compiled", ""])
def test_a_review_resume_keeps_the_record_its_launch_made(tmp_path, monkeypatch, recorded):
    # S1488: a human-review resume continues the same run and rewrites its
    # context; it keeps the launch's record and does not decide it again
    # after the job's own writes. Launches that resume another run's
    # checkpoint or execution are not plan launches and record none.
    from types import SimpleNamespace

    from tests.support.paused_plan_review import _RUN_ID, _FailingPipeline, _paused

    _, wrapper, study = _paused(tmp_path, monkeypatch, _FailingPipeline(ValueError("unused"), resumable=True))
    path = wrapper / "run_context.json"
    launched = json.loads(path.read_text())
    if recorded:
        path.write_text(json.dumps({**launched, "compiled_configuration_source_run_id": recorded}))
    monkeypatch.setattr(agent_pipeline_runs, "resume_pending_review", lambda *a, **k: SimpleNamespace(state="rejected"))
    agent_pipeline_runs.resume_research_pipeline(
        run_id=_RUN_ID, study_context_id=study["id"], decision="rejected", reviewer="local reviewer", note="",
        job=SimpleNamespace(emit=lambda _event: None, cancel_requested=False), current_study_context=study)
    resumed = json.loads(path.read_text())
    assert resumed.get("compiled_configuration_source_run_id", "") == recorded
    public = agent_runs._public_review_payloads({"run_context.json": resumed})["run_context.json"]
    assert public.get("compiled_configuration_source_run_id", "") == recorded
