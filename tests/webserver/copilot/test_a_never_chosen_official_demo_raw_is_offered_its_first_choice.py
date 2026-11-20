"""A study bound to the official Demo raw without any choice is offered the first one (B32, S1741).

UX on rc_4cc: a study bound to the converted official Demo raw folder through
「从它的原数据重定义新队列 → 继续使用已转换数据」 had never signed a raw choice, so
``prior_choice_export`` found no receipt and the reconfirmation card offered
nothing; its only action (an extraction draft) could not start, because the
setup was incomplete.  Dead end.  Such a study is now offered the first choice
from the current verified export of exactly that official folder, through the
full ``resolve_demo_planning_source(again=True)`` checks; nothing is signed
until the researcher confirms.  An unverified folder, a folder whose export is
not ready, or a non-official folder still gets no offer.  A study with an
earlier receipt keeps its path.  Synthetic official-Demo fixture only.
"""

import json

import pytest

from easyicu.export_publication import in_progress_marker_path
from easyicu.webserver import agent_runs, research_run_submission, sources, study_contexts
from easyicu.webserver.pi_copilot import demo_planning_source
from easyicu.webserver.pi_copilot.demo_planning_authority import resolve_choice
from easyicu.webserver.pi_copilot.workflow import build_project_workflow_projection
from tests.support.official_demo_choice import choice as choice
from tests.support.official_demo_raw import official_raw as official_raw

CODE = "demo_planning_source_reconfirm_required"


@pytest.fixture
def unchosen(choice, monkeypatch):
    """A new study bound straight to the official raw folder: no receipt at all."""
    monkeypatch.setattr(agent_runs, "list_run_history", lambda **_: {"runs": []})
    monkeypatch.setattr(sources, "_autodiscovered_paths", lambda **_: [])
    study = study_contexts.upsert_context({
        "question": choice.study["question"],
        "data_source": {"database": choice.study["data_source"]["database"], "path": str(choice.paths.raw)},
        "cohort": choice.study["cohort"], "modules": ["demographics"],
    })
    assert not study.get("demo_planning_authority_ref")
    assert demo_planning_source.raw_choice_needs_reconfirmation(study)
    choice.record.binding = choice.service._binding_for_context(study)
    return study


def _workflow(study):
    return build_project_workflow_projection(study_context_id=study["id"]).workflow


def _launch(study):
    return research_run_submission.submit_research_run(research_run_submission.ResearchRunSubmissionRequest(
        study_context_id=study["id"], provider="mock", credential_source="", external_llm_opt_in=False,
        intent="candidate_plan", planner_start_mode="fresh",
    ))


class _PastTheRefusal(Exception):
    pass


def _assert_launch_passes_the_refusal(study, monkeypatch):
    def reached(_path):
        raise _PastTheRefusal

    monkeypatch.setattr(research_run_submission.dataio, "prepared_export_manifest_path", reached)
    monkeypatch.setattr(research_run_submission, "_with_stated_first_stay", lambda study: study)
    monkeypatch.setattr(research_run_submission, "_with_repeated_stays_default", lambda study, path: study)
    with pytest.raises(_PastTheRefusal):
        _launch(study)


def _assert_no_offer(study):
    assert demo_planning_source.resolve_demo_planning_reconfirmation(study) is None
    workflow = _workflow(study)
    assert workflow.next_action_code == CODE
    assert "planning_source" not in workflow.bound_export_cohort_refusal
    with pytest.raises(research_run_submission.ResearchRunSubmissionError) as refused:
        _launch(study)
    assert refused.value.detail["error"] == CODE


def test_a_verified_official_raw_never_chosen_is_offered_the_first_choice(choice, unchosen, monkeypatch):
    workflow = _workflow(unchosen)
    assert workflow.next_action_code == CODE
    offer = workflow.bound_export_cohort_refusal["planning_source"]
    assert offer["expected_revision"] == unchosen["revision"]
    assert offer["study_id"] == unchosen["id"]
    assert str(choice.paths.root) not in workflow.model_dump_json()
    # Offered, not signed: the launch is still refused until the researcher confirms.
    assert resolve_choice(unchosen) is None
    with pytest.raises(research_run_submission.ResearchRunSubmissionError) as refused:
        _launch(unchosen)
    assert refused.value.detail["error"] == CODE

    choice.service.authorize_data_source(choice.record.session_id, **{
        **choice.body, "expected_revision": unchosen["revision"], "offer_sha256": offer["offer_sha256"],
    })
    confirmed = study_contexts.get_context(unchosen["id"])
    assert resolve_choice(confirmed) is not None
    assert confirmed["data_source"]["path"] == str(choice.paths.raw)
    assert not demo_planning_source.raw_choice_needs_reconfirmation(confirmed)
    assert _workflow(confirmed).next_action_code != CODE
    _assert_launch_passes_the_refusal(confirmed, monkeypatch)


def test_an_export_still_being_written_is_not_offered(choice, unchosen):
    in_progress_marker_path(choice.paths.export).write_text("{}")
    _assert_no_offer(unchosen)


def test_raw_receipts_that_no_longer_verify_are_not_offered(choice, unchosen):
    marker = json.loads(choice.paths.converted_marker.read_text())
    choice.paths.converted_marker.write_text(json.dumps({**marker, "archive_sha256": "0" * 64}))
    _assert_no_offer(unchosen)


def test_a_different_database_on_the_official_folder_is_not_offered(choice, unchosen):
    other = study_contexts.upsert_context({
        "id": unchosen["id"], "data_source": {**unchosen["data_source"], "database": "miiv_demo"},
    })
    assert demo_planning_source.raw_choice_needs_reconfirmation(other)
    _assert_no_offer(other)


def test_a_folder_that_is_not_the_official_raw_is_never_this_offer(choice, unchosen, tmp_path):
    copy = tmp_path / "copied_raw"
    copy.mkdir()
    other = study_contexts.upsert_context({
        "id": unchosen["id"], "data_source": {**unchosen["data_source"], "path": str(copy)},
    })
    assert not demo_planning_source.raw_choice_needs_reconfirmation(other)
    assert demo_planning_source._reconfirmation_study(other) is None  # noqa: SLF001
    assert demo_planning_source.resolve_demo_planning_reconfirmation(other) is None


def test_a_study_with_an_earlier_receipt_keeps_its_path(choice, monkeypatch):
    monkeypatch.setattr(agent_runs, "list_run_history", lambda **_: {"runs": []})
    monkeypatch.setattr(sources, "_autodiscovered_paths", lambda **_: [])
    choice.service.authorize_data_source(choice.record.session_id, **choice.body)
    chosen = study_contexts.get_context(choice.study["id"])
    study_contexts.upsert_context({"id": chosen["id"], "outcome": "hospital_mortality"})
    changed = study_contexts.get_context(chosen["id"])
    assert demo_planning_source.raw_choice_needs_reconfirmation(changed)

    def unused(_source):
        raise AssertionError("a receipt names the export; the fallback is not read")

    monkeypatch.setattr(demo_planning_source, "_current_official_export", unused)
    reconfirm = demo_planning_source._reconfirmation_study(changed)  # noqa: SLF001
    assert reconfirm is not None and reconfirm["data_source"]["path"] == str(choice.paths.export)
    assert "planning_source" in _workflow(changed).bound_export_cohort_refusal
