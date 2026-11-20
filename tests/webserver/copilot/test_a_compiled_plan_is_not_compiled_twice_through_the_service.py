"""A plan that continues a compiled candidate is not compiled again (B10, S1480).

UX leg 3 en/zh (2026-11-19): after EasyICU compiled a plan's runtime settings
on a raw official-Demo study, confirming the source again (B6) and the fresh
plan's own job writes moved the study's revision, so the next review offered
the same compile again and the plan loop repeated. This drives the real
service chain: a compile through ``apply_agent_plan_configuration``, the
signed reconfirmation, the fresh plan's launch-time lineage, its job writes,
then a second compile request on its review, which is refused with no write.
A write the allowance does not cover breaks the lineage instead, and that
plan may be compiled once. B11 (UX leg 3 en, 2026-11-19): that round-2
plan's run rebuilt its data profile for the compiled groupings from the raw
folder and stopped (manifest_marker_missing); it reads the registered
official export, as the identity questions do. Synthetic official-Demo
fixture; no benchmark item.
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


def _next_action(study, runs, run):
    # The review card's next step for this run, as the workflow projects it.
    # B10's card: the review asks for changes and names a runtime gap.
    candidate = _candidate(study, ["REPEATED_STAY_IDENTITY_UNAVAILABLE"]).model_copy(
        update={"next_action_code": "plan_scientific_changes_required"})
    snapshot = workflow_owner._enrich_plan_review(
        candidate, study=study,
        review=runs.review(run["project_dir"]), run=run)
    return snapshot.next_action_code


def _offers_compile(study, runs, run):
    return _next_action(study, runs, run) == "agent_plan_configuration_required"


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


def test_a_launch_that_failed_after_the_reconfirmation_costs_exactly_one_more_compile(choice, runs):
    # S1515 known limitation (B11 walk, UX leg 3 en, a414): the round-2 launch
    # failed before it planned, and its job marked and cleared the study. The
    # allowance is one write after the compile (S1471), so the retry carries
    # no record and may be compiled once more; after that compile and its
    # reconfirmation the lineage holds again and the chain stops, and the
    # compiled plan itself is not compiled again (B13).
    first = runs.launch(study_contexts.get_context(choice.study["id"]), "run-a")
    _compile(choice, first)
    reconfirmed = _reconfirm(choice)
    after_failure = _job_writes(reconfirmed["id"], "job-failed")
    retry = runs.launch(after_failure, "run-b")
    assert runs.recorded(retry) == ""
    reviewed = _job_writes(after_failure["id"], "job-b")
    assert _offers_compile(reviewed, runs, retry)
    _compile(choice, retry)
    reconfirmed = _reconfirm(choice)
    # B13 (UX leg 3 en, 80e0455cf): that compile changed only the field the
    # reconfirmation resets, so the compiled plan reads as current again. It
    # is not offered, nor compiled, again: the walk looped compile ->
    # reconfirm -> compile with no way to a run.
    assert retry["scientific_configuration_sha256"] == study_contexts.scientific_configuration_sha256(reconfirmed)
    assert not _offers_compile(reconfirmed, runs, retry)
    _assert_refused_without_writes(choice, retry)
    third = runs.launch(reconfirmed, "run-c")
    assert runs.recorded(third) == "run-b"
    reviewed = _job_writes(reconfirmed["id"], "job-c")
    assert not _offers_compile(reviewed, runs, third)
    _assert_refused_without_writes(choice, third)


def test_the_compiled_plan_stays_continued_after_job_writes_and_in_its_revision(choice, runs):
    # Carbon b13r1: the B13 hold does not depend on the revision (a job's
    # lifecycle writes after the reconfirmation keep it), and a revision of
    # the compiled plan records that plan, though the plan itself carries no
    # record: else the revision is offered the compile again.
    first = runs.launch(study_contexts.get_context(choice.study["id"]), "run-a")
    _compile(choice, first)
    reconfirmed = _reconfirm(choice)
    after_failure = _job_writes(reconfirmed["id"], "job-failed")
    retry = runs.launch(after_failure, "run-b")
    _job_writes(after_failure["id"], "job-b")
    _compile(choice, retry)
    reconfirmed = _reconfirm(choice)
    later = _job_writes(reconfirmed["id"], "job-review")
    assert runs.recorded(retry) == ""
    assert compiled_source_run_id_for_launch(study=later, project_root="/synthetic") == ""
    assert not _offers_compile(later, runs, retry)
    _assert_refused_without_writes(choice, retry)
    revised = runs.launch(later, "run-b-revised", revision_source_run_id="run-b")
    assert runs.recorded(revised) == "run-b"
    reviewed = _job_writes(later["id"], "job-revised")
    assert not _offers_compile(reviewed, runs, revised)
    _assert_refused_without_writes(choice, revised)


def test_another_choice_on_the_compiled_plan_keeps_its_compile(choice, runs):
    # Carbon b14r1: a host choice recorded on the compiled plan after its
    # reconfirmation kept only itself, erasing the compile receipt (B13's
    # hold) and its count (S1542).
    first = runs.launch(study_contexts.get_context(choice.study["id"]), "run-a")
    _compile(choice, first)
    reconfirmed = _reconfirm(choice)
    after_failure = _job_writes(reconfirmed["id"], "job-failed")
    retry = runs.launch(after_failure, "run-b")
    _job_writes(after_failure["id"], "job-b")
    _compile(choice, retry)
    reconfirmed = _reconfirm(choice)
    after = study_contexts.upsert_context({"id": reconfirmed["id"], "title": "Renamed"})
    plan_review_progress.record_choice(before=reconfirmed, after=after, run=retry,
                                       decision_code="plan_timing", option_id="landmark_24h")
    receipt = plan_review_progress._receipt(after)
    assert receipt.run_id == "run-b" and receipt.compiles == 1
    assert receipt.choices[plan_review_progress.AGENT_PLAN_CONFIGURATION_CHOICE[0]] == plan_review_progress.AGENT_PLAN_CONFIGURATION_CHOICE[1]
    assert not _offers_compile(study_contexts.get_context(after["id"]), runs, retry)


def test_a_choice_that_changes_the_compiled_configuration_drops_its_compile_but_keeps_the_count(choice, runs):
    # Carbon b14r2: the receipt carries the compile only while the choice
    # leaves the compiled configuration as it was; the loop count survives.
    first = runs.launch(study_contexts.get_context(choice.study["id"]), "run-a")
    _compile(choice, first)
    reconfirmed = _reconfirm(choice)
    after_failure = _job_writes(reconfirmed["id"], "job-failed")
    retry = runs.launch(after_failure, "run-b")
    _job_writes(after_failure["id"], "job-b")
    _compile(choice, retry)
    reconfirmed = _reconfirm(choice)
    after = study_contexts.upsert_context({"id": reconfirmed["id"], "question": "A changed question?"})
    assert study_contexts.scientific_configuration_sha256(after) != study_contexts.scientific_configuration_sha256(reconfirmed)
    plan_review_progress.record_choice(before=reconfirmed, after=after, run=retry,
                                       decision_code="plan_timing", option_id="landmark_24h")
    receipt = plan_review_progress._receipt(after)
    assert receipt.run_id == "run-b" and receipt.compiles == 1
    assert plan_review_progress.AGENT_PLAN_CONFIGURATION_CHOICE[0] not in receipt.choices


def test_a_third_compile_of_one_plan_stops_the_loop(choice, runs, monkeypatch, caplog):
    # S1542 backstop: whatever lets a plan be offered its compile again (B13
    # was the second retry-path loop), two compiles of one plan with no
    # planning run in between stop the third, with its own card and a
    # diagnosable lineage log. Synthetic: the continuation guard is disabled
    # so the loop B13 closed is forced open.
    first = runs.launch(study_contexts.get_context(choice.study["id"]), "run-a")
    _compile(choice, first)
    reconfirmed = _reconfirm(choice)
    after_failure = _job_writes(reconfirmed["id"], "job-failed")
    retry = runs.launch(after_failure, "run-b")
    _job_writes(after_failure["id"], "job-b")
    monkeypatch.setattr(plan_review_progress, "planned_on_compiled_configuration", lambda *_a, **_k: False)
    _compile(choice, retry)
    assert plan_review_progress._receipt(study_contexts.get_context(choice.study["id"])).compiles == 1
    _reconfirm(choice)
    _compile(choice, retry)
    assert plan_review_progress._receipt(study_contexts.get_context(choice.study["id"])).compiles == 2
    study = _reconfirm(choice)
    with caplog.at_level("WARNING", logger=workflow_owner.__name__):
        assert _next_action(study, runs, retry) == "agent_plan_configuration_loop_stopped"
    assert "agent_plan_configuration_loop_stopped" in caplog.text and "'compiles': 2" in caplog.text
    receipt = plan_review_progress._receipt(study)
    with pytest.raises(service_module.PiCopilotError) as raised:
        _compile(choice, retry)
    assert raised.value.code == "agent_plan_configuration_loop_stopped" and raised.value.status_code == 409
    assert study_contexts.get_context(study["id"]) == study
    assert plan_review_progress._receipt(study) == receipt
    # A planning run in between restarts the count: its plan compiles once.
    fresh = runs.launch(study, "run-c")
    _job_writes(study["id"], "job-c")
    assert _offers_compile(study_contexts.get_context(study["id"]), runs, fresh)


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


def test_the_round_two_plan_reads_the_registered_export_not_the_raw_folder(choice, runs, monkeypatch):
    # B11: raw Demo -> plan -> compile -> signed reconfirmation -> the round-2
    # plan's launch. Its run rebuilds the data profile for the compiled
    # groupings (agent_pipeline_runs, before the first progress step) with the
    # launch study as the source study; the catalog is the registered export's.
    from easyicu.research_agent.acquisition import catalog as catalog_owner
    from easyicu.research_agent.intake.export_package import ExportPackageError
    from easyicu.webserver import research_launch_scientific as launch
    from easyicu.webserver import research_pipeline_run_preparation as preparation

    first = runs.launch(study_contexts.get_context(choice.study["id"]), "run-a")
    _compile(choice, first)
    study = _reconfirm(choice)
    request = preparation.ResearchPipelineLaunchRequest(
        export_path=str(choice.paths.raw), study_context=study, project_root=str(choice.paths.raw.parent / "projects"),
        provider={"provider": "openai"}, provider_environment={}, credential_source="pi_verified",
        literature_search_authorized=False, plan_revision_source_run_id="", execution_resume_source_run_id="",
        development_resume_source_job_id="", budget_mode="planner_canary", runner_image=None)
    scientific = preparation._prepare_scientific_launch(request)
    assert scientific.metadata_only_planning
    # Without the owner the raw folder is opened as a package: the B11 stop.
    with pytest.raises(ExportPackageError) as raised:
        catalog_owner.build_available_catalog(choice.paths.raw)
    assert raised.value.code == "manifest_marker_missing"
    read = []

    def catalog(path):
        read.append(str(path))
        raise _Read()

    monkeypatch.setattr(catalog_owner, "build_available_catalog", catalog)
    with pytest.raises(_Read):
        launch._data_foundation_profile(
            export_path=str(choice.paths.raw), study=scientific.materialization_study,
            target=None, require_target=False, require_primary_exposure=False,
            source_study=scientific.study)
    assert read == [str(choice.paths.export)]
    # The materialization view alone cannot resolve the signed choice; it is
    # the launch study's.
    assert launch.bound_source_authority_path(str(choice.paths.raw), scientific.materialization_study) == str(choice.paths.raw)
    # A study with no resolving choice keeps its bound path, so it refuses as before.
    unsigned = {**scientific.study, "demo_planning_authority_ref": ""}
    assert launch.bound_source_authority_path(str(choice.paths.raw), unsigned) == str(choice.paths.raw)


def _native_export(export):
    # Carbon b12r1: the registered export as the real describer reads it, a
    # native manifest with its column-metadata sidecar (demographics only).
    import json

    import pandas as pd

    from easyicu.concept.export_metadata import build_export_file_metadata_binding
    from easyicu.concept.metadata_sidecar import EXPORT_PHYSICAL_SCOPE, ColumnMetadataSidecar, write_content_addressed_sidecar
    from easyicu.resources import load_dictionary
    from tests.support.published_export import record_member_digests

    frame = pd.DataFrame({"patientunitstayid": [1, 2], "age": [40, 60]})
    frame.to_parquet(export / "demographics.parquet", index=False)
    binding = build_export_file_metadata_binding(
        relative_path="demographics.parquet", module="demographics", frame=frame,
        concept_ids=["age"], database="eicu_demo", database_class_prefixes=("eicu_demo", "eicu"),
        dictionary=load_dictionary(include_sofa2=True))
    reference = write_content_addressed_sidecar(export, ColumnMetadataSidecar(
        source_database="eicu_demo", source_database_class_prefixes=("eicu_demo", "eicu"),
        scope=EXPORT_PHYSICAL_SCOPE, files=(binding,)))
    manifest = json.loads((export / "_manifest.json").read_text())
    manifest.update({
        "concept_selection": {"mode": "explicit", "modules": {"demographics": ["age"]}},
        "files": [{"file": "demographics.parquet", "module": "demographics", "rows": 2, "concepts": 1,
                   "concept_ids": ["age"], "column_metadata_columns": list(binding.columns)}],
        "column_metadata": reference.to_dict()})
    (export / "_manifest.json").write_text(json.dumps(manifest))
    record_member_digests(export)


def test_the_round_two_scope_is_the_registered_exports_modules(choice, runs, monkeypatch):
    # B12 (UX leg 3 en, 80e0455cf): after the reconfirmation the study had no
    # modules, so the launch defaulted the scope from the raw folder's tables
    # and the profile found no stay-level denominator. The default scope is
    # the registered export's modules, read by the real describer.
    from easyicu.webserver import dataio
    from easyicu.webserver import research_launch_scientific as launch
    from easyicu.webserver import research_pipeline_run_preparation as preparation

    _native_export(choice.paths.export)
    first = runs.launch(study_contexts.get_context(choice.study["id"]), "run-a")
    _compile(choice, first)
    # The compile left the study with no modules, as in the walk.
    study_contexts.upsert_context({"id": choice.study["id"], "modules": []})
    study = _reconfirm(choice)
    assert not study.get("modules")
    real = dataio.describe_source_tables
    assert "demographics" not in (real(str(choice.paths.raw), planning_metadata_only=True).get("modules") or [])
    assert real(str(choice.paths.export), planning_metadata_only=True)["modules"] == ["demographics"]
    described = []

    def describe(path, **kwargs):
        described.append(str(path))
        return real(path, **kwargs)

    monkeypatch.setattr(dataio, "describe_source_tables", describe)
    request = preparation.ResearchPipelineLaunchRequest(
        export_path=str(choice.paths.raw), study_context=study, project_root=str(choice.paths.raw.parent / "projects"),
        provider={"provider": "openai"}, provider_environment={}, credential_source="pi_verified",
        literature_search_authorized=False, plan_revision_source_run_id="", execution_resume_source_run_id="",
        development_resume_source_job_id="", budget_mode="planner_canary", runner_image=None)
    scientific = preparation._prepare_scientific_launch(request)
    assert str(choice.paths.export) in described and str(choice.paths.raw) not in described
    assert scientific.materialization_study["modules"] == ["demographics"]
    # Without a resolving choice the bound folder is described as before.
    described.clear()
    unsigned = {**study, "demo_planning_authority_ref": ""}
    assert launch._neutral_materialization_scope(unsigned, export_path=str(choice.paths.raw))["modules"] != ["demographics"]
    assert described == [str(choice.paths.raw)]
    # A scope that still has no demographics stops with its data step.
    from easyicu.webserver.pi_copilot.projections import project_job

    job = project_job({"id": "b12_job", "kind": "agent-run", "status": "failed",
                       "error": "research_pipeline_stay_denominator_unavailable"})
    assert [action["kind"] for action in job["next_actions"]] == ["reexport_data"]


def test_a_launch_on_an_unreadable_package_names_it_and_offers_its_reexport(choice, runs, monkeypatch):
    # B11 card: a bound folder that is not a readable package stops with the
    # launch's typed intake stop, not the bare intake code. The card says what
    # is wrong and offers its re-export.
    from easyicu.webserver import research_launch_scientific as launch
    from easyicu.webserver import research_pipeline_run_preparation as preparation
    from easyicu.webserver.pi_copilot.projections import project_job
    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

    first = runs.launch(study_contexts.get_context(choice.study["id"]), "run-a")
    _compile(choice, first)
    study = _reconfirm(choice)
    request = preparation.ResearchPipelineLaunchRequest(
        export_path=str(choice.paths.raw), study_context=study, project_root=str(choice.paths.raw.parent / "projects"),
        provider={"provider": "openai"}, provider_environment={}, credential_source="pi_verified",
        literature_search_authorized=False, plan_revision_source_run_id="", execution_resume_source_run_id="",
        development_resume_source_job_id="", budget_mode="planner_canary", runner_image=None)
    scientific = preparation._prepare_scientific_launch(request)
    unsigned = {**scientific.study, "demo_planning_authority_ref": ""}

    def profile():
        return launch._data_foundation_profile(
            export_path=str(choice.paths.raw), study=scientific.materialization_study,
            target=None, require_target=False, require_primary_exposure=False, source_study=unsigned)

    # A folder with no package marker needs this study's data extracted first
    # (Carbon b11r1: a converted database is not a corrupt package).
    with pytest.raises(ResearchPipelineRunError) as raised:
        profile()
    assert raised.value.code == "research_pipeline_manifest_required"
    assert raised.value.details["intake_error_code"] == "manifest_marker_missing"
    job = project_job({"id": "b11_job", "kind": "agent-run", "status": "failed", "error": raised.value.code})
    assert [action["kind"] for action in job["next_actions"]] == ["reexport_data"]
    # A package that fails its own checks is named as such.
    from easyicu.research_agent.acquisition import catalog as catalog_owner
    from easyicu.research_agent.intake.export_package import ExportPackageError

    def broken(_path):
        raise ExportPackageError("synthetic", code="manifest_file_missing")

    monkeypatch.setattr(catalog_owner, "build_available_catalog", broken)
    with pytest.raises(ResearchPipelineRunError) as raised:
        profile()
    assert raised.value.code == "research_pipeline_manifest_invalid"
    job = project_job({"id": "b11_job", "kind": "agent-run", "status": "failed", "error": raised.value.code})
    assert [action["kind"] for action in job["next_actions"]] == ["reexport_data"]


class _Read(Exception):
    """The catalog read was reached; the test stops there."""
