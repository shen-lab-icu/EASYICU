"""A raw official-Demo study asks its identity questions of the official export.

UX leg 3 en (2026-11-19, B9): a raw Demo study chose the recommended "adults,
first ICU admission only"; compiling it was refused
(agent_plan_first_stay_coordinate_unavailable) and the card had no way
forward. The first-stay and patient-grouping authorities read the study's
bound path, which for a raw Demo study is the raw folder, not an export
package. The signed choice already proves that folder is the raw source of
the registered official export, so those authorities now read that export.
Without a choice that resolves, the bound path stays and they refuse as before.
Synthetic official-Demo fixture; no benchmark item.
"""

import pytest

from easyicu.webserver import agent_runs, research_launch_scientific as launch, source_identity_authority, sources, study_contexts
from easyicu.webserver.pi_copilot.demo_planning_authority import resolve_choice
from tests.support.official_demo_choice import choice as choice
from tests.support.official_demo_raw import official_raw as official_raw


@pytest.fixture
def chosen(choice, monkeypatch):
    monkeypatch.setattr(agent_runs, "list_run_history", lambda **_: {"runs": []})
    monkeypatch.setattr(sources, "_autodiscovered_paths", lambda **_: [])
    choice.service.authorize_data_source(choice.record.session_id, **choice.body)
    study = study_contexts.get_context(choice.study["id"])
    assert resolve_choice(study) is not None
    assert study["data_source"]["path"] == str(choice.paths.raw)
    return study


def test_a_signed_raw_choice_reads_the_registered_official_export(choice, chosen):
    assert launch.source_authority_export_path(chosen) == str(choice.paths.export)


def test_without_a_choice_that_resolves_the_bound_path_is_kept(choice, chosen):
    unsigned = {**chosen, "demo_planning_authority_ref": ""}
    assert launch.source_authority_export_path(unsigned) == str(choice.paths.raw)
    # A settings change ends the choice; the raw folder stays bound and refuses.
    study_contexts.upsert_context({"id": chosen["id"], "outcome": "hospital_mortality"})
    stale = study_contexts.get_context(chosen["id"])
    assert stale.get("demo_planning_authority_ref") in (None, "") or resolve_choice(stale) is None
    assert launch.source_authority_export_path(stale) == str(choice.paths.raw)


def test_launch_asks_the_official_export_for_the_first_stay_per_hospitalization(choice, chosen, monkeypatch):
    # Carbon b9: the eICU Demo's hospital-level first stay reads the same export.
    asked = []

    def resolve(*, export_path, database):
        asked.append(export_path)
        return None

    monkeypatch.setattr(source_identity_authority, "resolve_study_first_icu_stay_per_hospitalization", resolve)
    assert chosen["cohort"]["first_icu_stay_per_hospitalization"] is True
    with pytest.raises(launch.ResearchPipelineRunError):
        launch._first_icu_stay_per_hospitalization_for_cohort(chosen)
    assert asked == [str(choice.paths.export)]


def test_an_export_study_is_untouched(choice):
    assert launch.source_authority_export_path(choice.study) == str(choice.paths.export)


def _first_stay_reconfirmed(choice, chosen):
    # The researcher's first-stay choice ends the signed choice; the card
    # offers it again (B6) and confirming signs the first-stay settings.
    from easyicu.webserver.pi_copilot.workflow import build_project_workflow_projection

    study_contexts.upsert_context({"id": chosen["id"], "cohort": {
        **(chosen.get("cohort") or {}), "first_icu_stay_per_hospitalization": False, "exclude_readmissions": True}})
    changed = study_contexts.get_context(chosen["id"])
    choice.record.binding = choice.service._binding_for_context(changed)
    offer = build_project_workflow_projection(study_context_id=changed["id"]).workflow.bound_export_cohort_refusal["planning_source"]
    choice.service.authorize_data_source(choice.record.session_id, **{
        **choice.body, "expected_revision": changed["revision"], "offer_sha256": offer["offer_sha256"]})
    study = study_contexts.get_context(chosen["id"])
    assert resolve_choice(study) is not None
    return study


def test_launch_asks_the_official_export_for_first_stay_and_grouping(choice, chosen, monkeypatch):
    study = _first_stay_reconfirmed(choice, chosen)
    asked = []

    def record(name):
        def resolve(*, export_path, database):
            asked.append((name, export_path, database))
            return None
        return resolve

    monkeypatch.setattr(source_identity_authority, "resolve_study_first_icu_stay", record("first_stay"))
    monkeypatch.setattr(source_identity_authority, "resolve_study_patient_grouping", record("grouping"))
    with pytest.raises(launch.ResearchPipelineRunError):
        launch._first_icu_stay_for_cohort(study)  # the stubbed authority proves nothing
    launch.verified_patient_grouping(study)
    database = study["data_source"]["database"]
    assert asked == [("first_stay", str(choice.paths.export), database), ("grouping", str(choice.paths.export), database)]
    # The export answers identity questions only; the study keeps its raw source.
    assert study["data_source"]["path"] == str(choice.paths.raw)
    assert study_contexts.get_context(study["id"])["data_source"] == study["data_source"]


def test_one_reconfirmation_after_a_compile_keeps_the_compiled_lineage(choice, chosen):
    # B10 (UX leg 3 en/zh, 2026-11-19): applying a plan's settings ended the
    # signed choice; confirming it again is one more write. The next plan is
    # still a revision of the compiled candidate, and is not compiled again.
    # Any further write ends that (S1471).
    from easyicu.webserver.pi_copilot import plan_review_progress
    from easyicu.webserver.pi_copilot.workflow import build_project_workflow_projection

    reviewed = {"run_id": "run-compiled", "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(chosen)}
    compiled = study_contexts.upsert_context({"id": chosen["id"], "covariates": ["age"], "confirmations": {
        **(chosen.get("confirmations") or {}), "extraction_completed": True, "agent_plan_configuration_compiled": True}})
    assert resolve_choice(compiled) is None
    plan_review_progress.record_choice(before=chosen, after=compiled, run=reviewed,
                                       decision_code="agent_plan_configuration", option_id="typed_runtime_projection")
    choice.record.binding = choice.service._binding_for_context(compiled)
    offer = build_project_workflow_projection(study_context_id=compiled["id"]).workflow.bound_export_cohort_refusal["planning_source"]
    choice.service.authorize_data_source(choice.record.session_id, **{
        **choice.body, "expected_revision": compiled["revision"], "offer_sha256": offer["offer_sha256"]})
    study = study_contexts.get_context(chosen["id"])
    assert study["revision"] == compiled["revision"] + 1 and resolve_choice(study) is not None
    assert study_contexts.scientific_configuration_sha256(study) != study_contexts.scientific_configuration_sha256(compiled)
    fresh = {"run_id": "run-fresh", "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(study)}
    assert plan_review_progress.compiled_configuration_progress(study, reviewed) is not None
    assert plan_review_progress.planned_on_compiled_configuration(study, fresh)

    renamed = study_contexts.upsert_context({"id": chosen["id"], "title": "Renamed project"})
    assert plan_review_progress.compiled_configuration_progress(renamed, reviewed) is None
    assert not plan_review_progress.planned_on_compiled_configuration(renamed, fresh)
