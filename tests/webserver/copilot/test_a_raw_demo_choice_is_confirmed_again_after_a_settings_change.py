"""A raw official-Demo study whose choice a settings change ended confirms it again.

UX leg 3 (2026-11-19, B6): after the raw Demo choice, applying the reviewed
plan's configuration changed the study's scientific settings, which (rightly)
ended the signed choice while the study kept the raw folder. The automatic
revision then validated that folder as an export package and stopped
(manifest_marker_missing), with no way forward. The launch now refuses that
state with its own reason, and the workflow offers the same choice again on
the current settings; confirming it signs that configuration.
Synthetic official-Demo fixture; no benchmark item.
"""

import pytest

from easyicu.webserver import agent_runs, research_run_submission, sources, study_contexts
from easyicu.webserver.pi_copilot import demo_planning_authority, demo_planning_source
from easyicu.webserver.pi_copilot.demo_planning_authority import resolve_choice
from easyicu.webserver.pi_copilot.workflow import build_project_workflow_projection
from tests.support.official_demo_choice import choice as choice
from tests.support.official_demo_raw import official_raw as official_raw

CODE = "demo_planning_source_reconfirm_required"

# Writes that change the scientific settings after the choice: a projection of
# the reviewed plan (confirmations, as the compile records), and the leg-1 A′
# actions that change the outcome or the grouping.
SETTINGS_CHANGES = {
    "plan_configuration": {"confirmations": {"extraction_completed": False, "agent_plan_configuration_compiled": True,
                                             "plan_adjustment_set_confirmed": True}},
    "outcome": {"outcome": "hospital_mortality"},
    "grouping": {"analysis_design": {"analysis_family": "association_study", "analysis_unit": "icu_stay",
                                     "variance_estimator": "cluster_robust", "cluster_unit": "patient"}},
}


@pytest.fixture
def chosen(choice, monkeypatch):
    monkeypatch.setattr(agent_runs, "list_run_history", lambda **_: {"runs": []})
    monkeypatch.setattr(sources, "_autodiscovered_paths", lambda **_: [])
    choice.service.authorize_data_source(choice.record.session_id, **choice.body)
    study = study_contexts.get_context(choice.study["id"])
    assert resolve_choice(study) is not None
    return study


def _changed(choice, study, change):
    study_contexts.upsert_context({"id": study["id"], **SETTINGS_CHANGES[change]})
    changed = study_contexts.get_context(study["id"])
    choice.record.binding = choice.service._binding_for_context(changed)
    return changed


def _workflow(study):
    return build_project_workflow_projection(study_context_id=study["id"]).workflow


def _launch(study):
    return research_run_submission.submit_research_run(research_run_submission.ResearchRunSubmissionRequest(
        study_context_id=study["id"], provider="mock", credential_source="", external_llm_opt_in=False,
        intent="candidate_plan", planner_start_mode="fresh",
    ))


@pytest.mark.parametrize("change", sorted(SETTINGS_CHANGES))
def test_a_settings_change_lands_on_the_confirmation_card_not_the_export_validator(choice, chosen, change, monkeypatch):
    changed = _changed(choice, chosen, change)
    assert not changed.get("demo_planning_authority_ref")
    assert changed["data_source"]["path"] == str(choice.paths.raw)
    assert demo_planning_source.raw_choice_needs_reconfirmation(changed)

    workflow = _workflow(changed)
    assert workflow.next_action_code == CODE
    offer = workflow.bound_export_cohort_refusal["planning_source"]
    assert offer["expected_revision"] == changed["revision"]
    assert str(choice.paths.root) not in workflow.model_dump_json()
    with pytest.raises(research_run_submission.ResearchRunSubmissionError) as refused:
        _launch(changed)
    assert refused.value.detail["error"] == CODE

    # Confirming signs the current settings; the plan can start again.
    choice.service.authorize_data_source(choice.record.session_id, **{
        **choice.body, "expected_revision": changed["revision"], "offer_sha256": offer["offer_sha256"],
    })
    confirmed = study_contexts.get_context(changed["id"])
    assert resolve_choice(confirmed) is not None
    assert confirmed["revision"] == changed["revision"] + 1
    assert confirmed["data_source"]["path"] == str(choice.paths.raw)
    for key, value in SETTINGS_CHANGES[change].items():
        if key == "confirmations":
            # The applied settings stay; only the extraction flag is reset.
            assert {**value, "extraction_completed": False}.items() <= confirmed["confirmations"].items()
        else:
            assert confirmed[key] == changed[key]
    assert not demo_planning_source.raw_choice_needs_reconfirmation(confirmed)
    assert _workflow(confirmed).next_action_code == "provider_ready_to_generate_plan"
    _assert_launch_passes_the_refusal(confirmed, monkeypatch)


class _PastTheRefusal(Exception):
    pass


def _past_the_refusal(monkeypatch):
    # Every launch reads the package manifest next; reaching it means the
    # source refusal let this launch through.
    def reached(_path):
        raise _PastTheRefusal

    monkeypatch.setattr(research_run_submission.dataio, "prepared_export_manifest_path", reached)
    monkeypatch.setattr(research_run_submission, "_with_stated_first_stay", lambda study: study)
    monkeypatch.setattr(research_run_submission, "_with_repeated_stays_default", lambda study, path: study)


def _assert_launch_passes_the_refusal(study, monkeypatch):
    _past_the_refusal(monkeypatch)
    with pytest.raises(_PastTheRefusal):
        _launch(study)


def test_settings_that_now_match_the_earlier_export_can_still_confirm_the_raw_choice(choice, chosen, monkeypatch):
    """Carbon b6 1a: the earlier export holding the new cohort is no reason for a dead end."""
    study_contexts.upsert_context({"id": chosen["id"], "cohort": {}})
    changed = study_contexts.get_context(chosen["id"])
    choice.record.binding = choice.service._binding_for_context(changed)
    assert demo_planning_source.raw_choice_needs_reconfirmation(changed)
    # A first choice would not be offered on these settings.
    assert demo_planning_source.resolve_demo_planning_source(demo_planning_source._reconfirmation_study(changed)) is None
    offer = _workflow(changed).bound_export_cohort_refusal["planning_source"]
    choice.service.authorize_data_source(choice.record.session_id, **{
        **choice.body, "expected_revision": changed["revision"], "offer_sha256": offer["offer_sha256"],
    })
    confirmed = study_contexts.get_context(changed["id"])
    assert resolve_choice(confirmed) is not None
    _assert_launch_passes_the_refusal(confirmed, monkeypatch)


def test_a_deferred_or_resumed_execution_is_not_refused_here(choice, chosen, monkeypatch):
    """Carbon b6 2: only a launch that reads the study's own source is refused."""
    changed = _changed(choice, chosen, "outcome")
    resumed = research_run_submission.ResearchRunSubmissionRequest(
        study_context_id=changed["id"], provider="mock", credential_source="", external_llm_opt_in=False,
        intent="reviewed_analysis", planner_start_mode="fresh", execution_resume_source_run_id="run-sealed",
    )
    _past_the_refusal(monkeypatch)
    with pytest.raises(_PastTheRefusal):
        research_run_submission.submit_research_run(resumed)


def test_a_researcher_edit_after_the_reconfirmation_still_ends_it(choice, chosen):
    changed = _changed(choice, chosen, "outcome")
    offer = _workflow(changed).bound_export_cohort_refusal["planning_source"]
    choice.service.authorize_data_source(choice.record.session_id, **{
        **choice.body, "expected_revision": changed["revision"], "offer_sha256": offer["offer_sha256"],
    })
    study_contexts.upsert_context({"id": changed["id"], "question": "换一个研究问题"})
    edited = study_contexts.get_context(changed["id"])
    assert resolve_choice(edited) is None
    assert _workflow(edited).next_action_code == CODE


def test_a_displayed_offer_is_used_once_and_only_for_its_revision(choice, chosen):
    from easyicu.webserver.pi_copilot.contracts import PiCopilotError

    changed = _changed(choice, chosen, "plan_configuration")
    offer = _workflow(changed).bound_export_cohort_refusal["planning_source"]
    body = {**choice.body, "expected_revision": changed["revision"], "offer_sha256": offer["offer_sha256"]}
    with pytest.raises(PiCopilotError):
        choice.service.authorize_data_source(choice.record.session_id, **{**body, "offer_sha256": "f" * 64})
    choice.service.authorize_data_source(choice.record.session_id, **body)
    with pytest.raises(PiCopilotError):
        choice.service.authorize_data_source(choice.record.session_id, **body)


def test_an_unreadable_receipt_is_skipped_not_trusted(choice, chosen, monkeypatch):
    changed = _changed(choice, chosen, "outcome")

    def unreadable(_ref):
        raise ValueError("demo_planning_authority_invalid")

    monkeypatch.setattr(demo_planning_authority, "_signed_body", unreadable)
    assert demo_planning_authority.prior_choice_export(changed) is None
    # Only the verified current export's first choice (B32) may remain.
    monkeypatch.setattr(demo_planning_source, "_current_official_export", lambda _source: None)
    assert "planning_source" not in _workflow(changed).bound_export_cohort_refusal


def test_a_prepared_export_study_is_untouched(choice, monkeypatch):
    monkeypatch.setattr(agent_runs, "list_run_history", lambda **_: {"runs": []})
    monkeypatch.setattr(sources, "_autodiscovered_paths", lambda **_: [])
    assert not demo_planning_source.raw_choice_needs_reconfirmation(choice.study)
    assert _workflow(choice.study).next_action_code == "bound_export_cohort_mismatch"
