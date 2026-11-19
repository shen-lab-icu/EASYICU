"""A plan that continues a compiled candidate is not compiled again (B10, S1480).

UX leg 3 en/zh (2026-11-19): after EasyICU compiled a plan's runtime settings
on a raw official-Demo study, confirming the source again (B6) and the fresh
plan's own job writes moved the study's revision, so the next review offered
the same compile again and the plan loop repeated. This drives the real
service chain: a compile through ``apply_agent_plan_configuration``, the
signed reconfirmation, the fresh plan's launch-time lineage, its job writes,
then a second compile request on its review, which is refused with no write.
A write the allowance does not cover breaks the lineage instead, and that
plan may be compiled once. Synthetic official-Demo fixture; no benchmark item.
"""

from types import SimpleNamespace

import pytest

from easyicu.webserver import agent_runs, source_identity_authority, sources, study_contexts
from easyicu.webserver.pi_copilot import plan_review_progress, service as service_module
from easyicu.webserver.pi_copilot.demo_planning_authority import resolve_choice
from easyicu.webserver.pi_copilot.workflow import build_project_workflow_projection
from easyicu.webserver.plan_change_requirements import _compiled_runtime_continuity
from tests.support.official_demo_choice import choice as choice
from tests.support.official_demo_raw import official_raw as official_raw
from tests.support.v010_run_context import V010_RUN_CONTEXT

_PLAN = {"design_selection": {"candidates": [{"design_id": "landmark_adjusted_association", "disposition": "selected"}]},
         "steps": [{"model_requirements": [{
             "analysis_role": "primary", "exposure_source": "aki_stage", "outcome": "death", "covariates": ["age", "sex"],
             "covariate_rationales": {"age": "Baseline age precedes exposure.", "sex": "Baseline sex precedes exposure."},
             "covariate_temporal_roles": {"age": "baseline_static", "sex": "baseline_static"}}]}]}
_REVIEW = {"facts": {"remediation_buckets": {"runtime_capability": ["REPEATED_STAY_IDENTITY_UNAVAILABLE"]}}}


class _Runs:
    """The bound run history and each run's review, as the service reads them."""

    def __init__(self, monkeypatch):
        self.rows, self.contexts = [], {}
        monkeypatch.setattr(service_module, "list_bound_run_history", lambda **_: list(self.rows))
        monkeypatch.setattr(service_module.agent_runs, "read_run_review", self.review)

    def launch(self, study, run_id, *, compiled_source_run_id=""):
        # What the runner records: the launch's lineage, read before its job writes.
        self.rows.insert(0, {"run_id": run_id, "project_dir": f"/synthetic/{run_id}",
                             "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(study)})
        self.contexts[f"/synthetic/{run_id}"] = {"run_id": run_id, **(
            {"compiled_configuration_source_run_id": compiled_source_run_id} if compiled_source_run_id else {})}
        return self.rows[0]

    def review(self, project_dir):
        return {"artifact_payloads": {"agent_plan.json": _PLAN, "scientific_plan_review.json": _REVIEW,
                                      "run_context.json": self.contexts[project_dir]}}


@pytest.fixture
def runs(choice, monkeypatch):
    monkeypatch.setattr(agent_runs, "list_run_history", lambda **_: {"runs": []})
    monkeypatch.setattr(sources, "_autodiscovered_paths", lambda **_: [])
    monkeypatch.setattr(source_identity_authority, "resolve_study_patient_grouping",
                        lambda **_: SimpleNamespace(group_source="opaque_patient"))
    monkeypatch.setattr(source_identity_authority, "resolve_study_first_icu_stay", lambda **_: SimpleNamespace(source="x"))
    choice.service.authorize_data_source(choice.record.session_id, **choice.body)
    return _Runs(monkeypatch)


def _compile(choice, run):
    study = study_contexts.get_context(choice.study["id"])
    choice.record.binding = choice.service._binding_for_context(study, run_id=run["run_id"])
    return choice.service.apply_agent_plan_configuration(
        choice.record.session_id, project_id=choice.body["project_id"],
        expected_revision=study["revision"], run_id=run["run_id"])


def _lineage(study, runs):
    # Preparation's own gate for a fresh plan (compiled_configuration_plan_change).
    row = next((row for row in runs.rows if _compiled_runtime_continuity(study, row)), None)
    return "" if row is None else row["run_id"]


def _job_writes(study_id, job_id):
    # The plan job marks the study at its start and clears it at its end.
    before = study_contexts.get_context(study_id)["revision"]
    study_contexts.upsert_context({"id": study_id, "active_job_id": job_id}, lifecycle_write=True)
    study_contexts.clear_active_job_if(study_id, job_id, current_stage="review", last_route="agent")
    study = study_contexts.get_context(study_id)
    assert study["revision"] == before + 2
    return study


def _reconfirm(choice):
    study = study_contexts.get_context(choice.study["id"])
    choice.record.binding = choice.service._binding_for_context(study)
    workflow = build_project_workflow_projection(study_context_id=study["id"]).workflow
    assert workflow.next_action_code == "demo_planning_source_reconfirm_required"
    choice.service.authorize_data_source(choice.record.session_id, **{
        **choice.body, "expected_revision": study["revision"],
        "offer_sha256": workflow.bound_export_cohort_refusal["planning_source"]["offer_sha256"]})
    reconfirmed = study_contexts.get_context(study["id"])
    assert reconfirmed["revision"] == study["revision"] + 1 and resolve_choice(reconfirmed) is not None
    return reconfirmed


def _assert_refused_without_writes(choice, run):
    study = study_contexts.get_context(choice.study["id"])
    receipt = plan_review_progress._receipt(study)
    with pytest.raises(service_module.PiCopilotError) as raised:
        _compile(choice, run)
    assert raised.value.code == "agent_plan_configuration_already_applied" and raised.value.status_code == 409
    assert study_contexts.get_context(study["id"]) == study
    assert plan_review_progress._receipt(study) == receipt


def test_the_plan_after_one_reconfirmation_is_not_compiled_again(choice, runs):
    first = runs.launch(study_contexts.get_context(choice.study["id"]), "run-a")
    _compile(choice, first)
    reconfirmed = _reconfirm(choice)
    # Its launch reads the lineage before any job write and records it.
    assert _lineage(reconfirmed, runs) == "run-a"
    fresh = runs.launch(reconfirmed, "run-b", compiled_source_run_id="run-a")
    _job_writes(reconfirmed["id"], "job-b")
    _assert_refused_without_writes(choice, fresh)


def test_a_write_outside_the_allowance_lets_the_next_plan_be_compiled_once(choice, runs):
    # Carbon b10 F1: the guard follows the launch record, not the study's revision.
    first = runs.launch(study_contexts.get_context(choice.study["id"]), "run-a")
    _compile(choice, first)
    study_contexts.upsert_context({"id": choice.study["id"], "title": "Renamed project"})
    reconfirmed = _reconfirm(choice)
    assert _lineage(reconfirmed, runs) == ""
    second = runs.launch(reconfirmed, "run-b")
    _job_writes(reconfirmed["id"], "job-b")
    _compile(choice, second)
    compiled = study_contexts.get_context(choice.study["id"])
    assert plan_review_progress._receipt(compiled).run_id == "run-b"
    # Right after that compile the lineage is the exact receipt match.
    assert plan_review_progress.matching_progress(compiled, second) is not None
    assert _lineage(compiled, runs) == "run-b"
    reconfirmed = _reconfirm(choice)
    assert _lineage(reconfirmed, runs) == "run-b"
    third = runs.launch(reconfirmed, "run-c", compiled_source_run_id="run-b")
    _job_writes(reconfirmed["id"], "job-c")
    _assert_refused_without_writes(choice, third)


def test_a_review_from_v010_is_compiled_as_before(choice, runs):
    # S1480: a run made before the record existed carries none; its review
    # offers the compile once, as v0.1.0 did, and the compile writes.
    run = runs.launch(study_contexts.get_context(choice.study["id"]), "run-v010")
    runs.contexts[run["project_dir"]] = {**V010_RUN_CONTEXT, "run_id": "run-v010"}
    before = study_contexts.get_context(choice.study["id"])
    _compile(choice, run)
    after = study_contexts.get_context(choice.study["id"])
    assert after["revision"] == before["revision"] + 1
    assert plan_review_progress._receipt(after).run_id == "run-v010"
