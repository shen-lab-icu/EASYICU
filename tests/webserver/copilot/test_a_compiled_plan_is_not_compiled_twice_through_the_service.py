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
from easyicu.webserver.pi_copilot import workflow as workflow_owner
from easyicu.webserver.plan_change_requirements import compiled_source_run_id_for_launch
from tests.support.official_demo_choice import choice as choice
from tests.support.official_demo_raw import official_raw as official_raw
from tests.support.v010_run_context import V010_RUN_CONTEXT
from tests.support.reviewed_candidate import reviewed_candidate as _candidate

def _plan(*covariates):
    covariates = covariates or ("age", "sex")
    return {"design_selection": {"candidates": [{"design_id": "landmark_adjusted_association", "disposition": "selected"}]},
            "steps": [{"model_requirements": [{
                "analysis_role": "primary", "exposure_source": "aki_stage", "outcome": "death", "covariates": list(covariates),
                "covariate_rationales": {name: "Measured at baseline, before exposure." for name in covariates},
                "covariate_temporal_roles": {name: "baseline_static" for name in covariates}}]}]}


_PLAN = {"design_selection": {"candidates": [{"design_id": "landmark_adjusted_association", "disposition": "selected"}]},
         "steps": [{"model_requirements": [{
             "analysis_role": "primary", "exposure_source": "aki_stage", "outcome": "death", "covariates": ["age", "sex"],
             "covariate_rationales": {"age": "Baseline age precedes exposure.", "sex": "Baseline sex precedes exposure."},
             "covariate_temporal_roles": {"age": "baseline_static", "sex": "baseline_static"}}]}]}
_REVIEW = {"facts": {"remediation_buckets": {"runtime_capability": ["REPEATED_STAY_IDENTITY_UNAVAILABLE"]}}}


class _Runs:
    """The bound run history and each run's review, as the service reads them."""

    def __init__(self, monkeypatch):
        self.rows, self.contexts, self.plans = [], {}, {}
        monkeypatch.setattr(service_module, "list_bound_run_history", lambda **_: list(self.rows))
        monkeypatch.setattr(agent_runs, "list_run_history", lambda **_: {"runs": list(self.rows)})
        monkeypatch.setattr(agent_runs, "read_run_review", self.review)

    def launch(self, study, run_id, *, revision_source_run_id="", plan=None, researcher_change=False):
        # The real launch owner decides the record before the job writes;
        # the runner stores it in the run's context. Preparation never asks
        # it for a researcher's own plan-change request (a user action).
        recorded = "" if researcher_change else compiled_source_run_id_for_launch(
            study=study, project_root="/synthetic", revision_source_run_id=revision_source_run_id)
        self.rows.insert(0, {"run_id": run_id, "project_dir": f"/synthetic/{run_id}",
                             "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(study)})
        self.plans[f"/synthetic/{run_id}"] = plan or _PLAN
        self.contexts[f"/synthetic/{run_id}"] = {"run_id": run_id, **(
            {"compiled_configuration_source_run_id": recorded} if recorded else {})}
        return self.rows[0]

    def recorded(self, run):
        return self.contexts[run["project_dir"]].get("compiled_configuration_source_run_id", "")

    def review(self, project_dir):
        return {"artifact_payloads": {"agent_plan.json": self.plans[project_dir], "scientific_plan_review.json": _REVIEW,
                                      "run_context.json": self.contexts[project_dir]}}


@pytest.fixture
def runs(choice, monkeypatch):
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


def _offers_compile(study, runs, run):
    # The review card's next step for this run, as the workflow projects it.
    # B10's card: the review asks for changes and names a runtime gap.
    candidate = _candidate(study, ["REPEATED_STAY_IDENTITY_UNAVAILABLE"]).model_copy(
        update={"next_action_code": "plan_scientific_changes_required"})
    snapshot = workflow_owner._enrich_plan_review(
        candidate, study=study,
        review=runs.review(run["project_dir"]), run=run)
    return snapshot.next_action_code == "agent_plan_configuration_required"


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
    assert _offers_compile(study_contexts.get_context(choice.study["id"]), runs, first)
    _compile(choice, first)
    reconfirmed = _reconfirm(choice)
    # S1488 (2): after the reconfirmation the receipt's run must still be the
    # plan the compile read; the same run id on another configuration is not.
    runs.rows[0] = {**first, "scientific_configuration_sha256": "e" * 64}
    assert compiled_source_run_id_for_launch(study=reconfirmed, project_root="/synthetic") == ""
    runs.rows[0] = first
    fresh = runs.launch(reconfirmed, "run-b")
    assert runs.recorded(fresh) == "run-a"
    reviewed = _job_writes(reconfirmed["id"], "job-b")
    assert not _offers_compile(reviewed, runs, fresh)
    _assert_refused_without_writes(choice, fresh)
    # Carbon b10r2 F1: the planner's revision of that plan continues it too.
    revised = runs.launch(reviewed, "run-c", revision_source_run_id="run-b")
    assert runs.recorded(revised) == "run-a"
    _job_writes(reviewed["id"], "job-c")
    assert not _offers_compile(study_contexts.get_context(reviewed["id"]), runs, revised)
    _assert_refused_without_writes(choice, revised)


def test_a_write_outside_the_allowance_lets_the_next_plan_be_compiled_once(choice, runs):
    # Carbon b10 F1: the guard follows the launch record, not the study's revision.
    first = runs.launch(study_contexts.get_context(choice.study["id"]), "run-a")
    _compile(choice, first)
    study_contexts.upsert_context({"id": choice.study["id"], "title": "Renamed project"})
    reconfirmed = _reconfirm(choice)
    # Its planner wrote a different plan; with no record, that plan's
    # settings may be compiled once.
    second = runs.launch(reconfirmed, "run-b", plan=_plan("age", "sex", "weight"))
    assert runs.recorded(second) == ""
    reviewed = _job_writes(reconfirmed["id"], "job-b")
    assert _offers_compile(reviewed, runs, second)
    _compile(choice, second)
    compiled = study_contexts.get_context(choice.study["id"])
    assert plan_review_progress._receipt(compiled).run_id == "run-b"
    # Right after that compile the lineage is the exact receipt match.
    assert plan_review_progress.matching_progress(compiled, second) is not None
    reconfirmed = _reconfirm(choice)
    third = runs.launch(reconfirmed, "run-c", plan=_plan("age", "sex", "weight", "sofa"))
    assert runs.recorded(third) == "run-b"
    reviewed = _job_writes(reconfirmed["id"], "job-c")
    assert not _offers_compile(reviewed, runs, third)
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


def test_one_compile_bounds_a_chain_of_planner_revisions(choice, runs):
    # S1488: after one compile, three automatic planner revisions each carry
    # the record and none is compiled; a researcher's plan change in the
    # middle allows exactly one more compile, after which it carries again.
    # Each revised plan differs, so without the record it would be offered.
    compiles = 0

    def review(run):
        nonlocal compiles
        study = study_contexts.get_context(choice.study["id"])
        if not _offers_compile(study, runs, run):
            _assert_refused_without_writes(choice, run)
            return study
        _compile(choice, run)
        compiles += 1
        return _reconfirm(choice)

    study = review(runs.launch(study_contexts.get_context(choice.study["id"]), "run-a"))
    assert compiles == 1
    previous = runs.launch(study, "run-b")
    added = ["weight", "sofa", "lactate"]
    for hop in range(1, len(added) + 1):
        assert runs.recorded(previous) == "run-a"
        _job_writes(study["id"], f"job-{previous['run_id']}")
        review(previous)
        assert compiles == 1
        previous = runs.launch(study_contexts.get_context(study["id"]), f"run-r{hop}",
                               revision_source_run_id=previous["run_id"], plan=_plan("age", "sex", *added[:hop]))
    assert runs.recorded(previous) == "run-a"
    _job_writes(study["id"], f"job-{previous['run_id']}")
    review(previous)
    assert compiles == 1

    changed = runs.launch(study_contexts.get_context(study["id"]), "run-user", researcher_change=True,
                          plan=_plan("age", "sex", "creatinine"))
    assert runs.recorded(changed) == ""
    _job_writes(study["id"], "job-user")
    study = review(changed)
    assert compiles == 2
    fresh = runs.launch(study, "run-after")
    assert runs.recorded(fresh) == "run-user"
    _job_writes(study["id"], "job-after")
    review(fresh)
    revised = runs.launch(study_contexts.get_context(study["id"]), "run-after-r1",
                          revision_source_run_id="run-after", plan=_plan("age", "sex", "creatinine", "weight"))
    assert runs.recorded(revised) == "run-user"
    _job_writes(study["id"], "job-after-r1")
    review(revised)
    assert compiles == 2
