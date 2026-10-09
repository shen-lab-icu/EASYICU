"""A target trial's card shows the record its study names, and only a click approves it.

The host computes the card of a causal study: the protocol, the lines to
confirm, the limitations and what holds approval come from the record the
host keeps under the digest the study names; whether the section is the
latest statement comes from the study's latest compile job.  The card follows
that job -- compiling, stopped, or the section it wrote, approvable, blocked
or approved -- and the workflow names what the study does next.  A section a
newer statement has not replaced yet, a record the host no longer keeps, a
click on another record, another number of lines, a stale revision or a
running job do not approve; a second click on an approved record returns its
approval.  Synthetic records only.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from easyicu.research_agent.planning.target_trial_compile import (
    CONFOUNDER_NOT_APPLIED_REASONS,
    CONFOUNDER_REQUIRES_EXTRACTION_REASONS,
    NOT_APPLIED_REASONS,
    REQUIRES_EXTRACTION_REASONS,
)
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver import target_trial_card as card_owner
from easyicu.webserver import target_trial_records
from tests.support.target_trial import (
    STUDY_ID,
    compiled_target_trial,
    kept_target_trial_record,
    target_trial_context,
)

_CAUSAL = {
    "analysis_family": "causal_inference",
    "analysis_unit": "icu_stay",
    "variance_estimator": "model_based",
}


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        context_store, "_CONFIG_PATH", tmp_path / "cfg" / "study-contexts.json"
    )
    latest: dict[str, Any] = {"value": None}
    monkeypatch.setattr(
        card_owner, "latest_target_trial_compile", lambda study_id: latest["value"]
    )
    return latest


def _study(**fields) -> dict:
    return context_store.upsert_context(
        {"id": STUDY_ID, "question": "q", "analysis_design": dict(_CAUSAL), **fields}
    )


def _compiled(kept, *, job_id: str = "job_a") -> dict:
    return {
        "job_id": job_id,
        "status": "compiled",
        "reason_code": None,
        "compile_sha256": kept.compile_sha256,
    }


def _stated(_isolated, kept=None) -> tuple[dict, Any]:
    """The study with the section the latest compile job wrote."""

    kept = kept or kept_target_trial_record()
    created = _study()
    target_trial_records.keep_target_trial_record(STUDY_ID, kept)
    stated = context_store.bind_target_trial_design(
        STUDY_ID, kept.design(), expected_revision=created["revision"]
    )
    _isolated["value"] = _compiled(kept)
    return stated, kept


def _refused(study: dict, kept, **changes) -> card_owner.TargetTrialApprovalError:
    arguments = {
        "expected_revision": study["revision"],
        "compile_sha256": kept.compile_sha256,
        "n_lines_confirmed": kept.confirmation_lines,
        **changes,
    }
    with pytest.raises(card_owner.TargetTrialApprovalError) as caught:
        card_owner.approve_target_trial(STUDY_ID, **arguments)
    return caught.value


def test_a_study_of_another_family_has_no_trial_card() -> None:
    study = context_store.upsert_context({"id": STUDY_ID, "question": "q"})
    running = {"job_id": "job_a", "status": "running", "reason_code": None}

    for latest in (None, running):
        assert card_owner.target_trial_card(study, latest) is None
        assert card_owner.target_trial_next_action(study, latest) is None


def test_a_causal_study_without_a_statement_is_asked_for_one() -> None:
    study = _study()

    assert card_owner.target_trial_card(study, None) is None
    assert card_owner.target_trial_next_action(study, None) == "target_trial_statement_needed"


@pytest.mark.parametrize(
    ("latest", "state", "next_action"),
    [
        ({"status": "running", "reason_code": None}, "compiling", "target_trial_review"),
        (
            {"status": "stopped", "reason_code": "target_trial_data_unavailable"},
            "stopped",
            "target_trial_review",
        ),
        ({"status": "failed", "reason_code": "x_failed"}, "stopped", "target_trial_review"),
        ({"status": "interrupted", "reason_code": None}, "stopped", "target_trial_review"),
    ],
    ids=["running", "data_unavailable", "failed", "interrupted"],
)
def test_the_card_follows_the_latest_compile_job(
    _isolated, latest: dict, state: str, next_action: str
) -> None:
    study = _study()
    _isolated["value"] = {"job_id": "job_a", "compile_sha256": None, **latest}

    card = card_owner.target_trial_card(study, _isolated["value"])

    assert card["state"] == state
    assert card["latest_compile"]["reason_code"] == latest["reason_code"]
    assert card["approvable"] is False and card["stale"] is True
    assert card_owner.target_trial_next_action(study, _isolated["value"]) == next_action


def test_the_record_its_section_names_is_shown_for_approval(_isolated) -> None:
    study, kept = _stated(_isolated)

    card = card_owner.target_trial_card(study, _isolated["value"])

    assert card["schema_version"] == "easyicu.target_trial_card/1"
    assert (card["state"], card["approvable"], card["stale"]) == (
        "approvable",
        True,
        False,
    )
    assert card["compile_sha256"] == kept.compile_sha256
    assert [row["item"] for row in card["protocol"]] == [
        "eligibility",
        "treatment_strategies",
        "assignment",
        "time_zero",
        "follow_up",
        "outcome",
        "causal_contrast",
        "analysis_plan",
    ]
    assert len(card["confirmations"]) == card["confirmation_lines"]
    assert card["evidence_ceiling"] == "analysis_only"
    assert card["blocking"] == [] and card["approval"] is None
    assert card_owner.target_trial_next_action(study, _isolated["value"]) == "target_trial_review"


def test_a_record_that_waits_lists_what_holds_approval(_isolated) -> None:
    # Onsets read from a later hour: the treatment waits for an extraction.
    waiting = kept_target_trial_record(
        compiled=compiled_target_trial(
            target_trial_context(onset_window="icu_admission[2,12]h")
        )
    )
    study, _ = _stated(_isolated, waiting)

    card = card_owner.target_trial_card(study, _isolated["value"])

    assert (card["state"], card["approvable"]) == ("blocked", False)
    assert card["blocking"] == waiting.record["approval_blockers"] != []
    assert all(
        set(row) == {"source", "name", "reason", "detail"} for row in card["blocking"]
    )
    assert _refused(study, waiting).code == "target_trial_design_invalid"


def test_the_click_approves_the_record_once(_isolated) -> None:
    study, kept = _stated(_isolated)

    first = card_owner.approve_target_trial(
        STUDY_ID,
        expected_revision=study["revision"],
        compile_sha256=kept.compile_sha256,
        n_lines_confirmed=kept.confirmation_lines,
    )
    again = card_owner.approve_target_trial(
        STUDY_ID,
        expected_revision=study["revision"],
        compile_sha256=kept.compile_sha256,
        n_lines_confirmed=kept.confirmation_lines,
    )

    assert first["repeated"] is False and again["repeated"] is True
    assert again["approval_event_id"] == first["approval_event_id"]
    approved = context_store.get_context(STUDY_ID)
    assert approved["revision"] == first["study"]["revision"]
    card = card_owner.target_trial_card(approved, _isolated["value"])
    assert card["state"] == "approved"
    assert card["approval"]["approval_event_id"] == first["approval_event_id"]
    # The approved trial is planned on the study's data.
    assert card_owner.target_trial_next_action(approved, _isolated["value"]) == (
        "target_trial_plan_ready"
    )


@pytest.mark.parametrize("newer", ["running", "stopped", "other_record"])
def test_a_newer_statement_holds_the_earlier_card(_isolated, newer: str) -> None:
    study, kept = _stated(_isolated)
    _isolated["value"] = {
        "job_id": "job_b",
        "status": "compiled" if newer == "other_record" else newer,
        "reason_code": (
            "target_trial_data_unavailable" if newer == "stopped" else None
        ),
        "compile_sha256": "e" * 64 if newer == "other_record" else None,
    }

    card = card_owner.target_trial_card(study, _isolated["value"])

    assert card["stale"] is True and card["approvable"] is False
    assert card["state"] == ("compiling" if newer == "running" else "stopped")
    assert _refused(study, kept).code == "target_trial_restatement_pending"
    assert card_owner.target_trial_next_action(study, _isolated["value"]) == "target_trial_review"


def test_a_click_on_another_record_lines_or_revision_is_refused(_isolated) -> None:
    study, kept = _stated(_isolated)

    for changes, code, status in (
        ({"compile_sha256": "d" * 64}, "target_trial_approval_record_mismatch", 422),
        (
            {"n_lines_confirmed": kept.confirmation_lines - 1},
            "target_trial_design_invalid",
            422,
        ),
        ({"expected_revision": study["revision"] - 1}, "study_context_revision_conflict", 409),
    ):
        error = _refused(study, kept, **changes)
        assert (error.code, error.status_code) == (code, status)
    running = context_store.handoff_context(
        STUDY_ID, active_job_id="job_running", expected_revision=study["revision"]
    )
    error = _refused(running, kept)
    assert (error.code, error.status_code) == ("study_context_active_job_conflict", 409)
    assert context_store.get_context(STUDY_ID)["target_trial_design"]["approval"] is None


def test_a_record_the_host_no_longer_keeps_is_not_approved(_isolated) -> None:
    study, kept = _stated(_isolated)
    for path in target_trial_records.records_root().rglob("*.json"):
        path.unlink()

    card = card_owner.target_trial_card(study, _isolated["value"])

    assert (card["state"], card["reason_code"]) == (
        "stopped",
        "target_trial_record_missing",
    )
    error = _refused(study, kept)
    assert (error.code, error.status_code) == ("target_trial_record_missing", 409)


def test_an_approval_whose_record_is_gone_does_not_plan(_isolated) -> None:
    study, kept = _stated(_isolated)
    approved = card_owner.approve_target_trial(
        STUDY_ID,
        expected_revision=study["revision"],
        compile_sha256=kept.compile_sha256,
        n_lines_confirmed=kept.confirmation_lines,
    )["study"]
    assert card_owner.target_trial_next_action(approved, _isolated["value"]) == (
        "target_trial_plan_ready"
    )

    for path in target_trial_records.records_root().rglob("*.json"):
        path.unlink()

    assert card_owner.target_trial_next_action(approved, _isolated["value"]) == (
        "target_trial_review"
    )
    card = card_owner.target_trial_card(approved, _isolated["value"])
    assert (card["state"], card["reason_code"]) == (
        "stopped",
        "target_trial_record_missing",
    )


def test_the_ui_codes_name_every_reason_the_card_shows() -> None:
    codes = card_owner.TARGET_TRIAL_UI_CODES

    assert len(codes) == len(set(codes))
    assert all(re.fullmatch(r"[a-z][a-z0-9_]{2,120}", code) for code in codes)
    for reason in (
        *NOT_APPLIED_REASONS,
        *REQUIRES_EXTRACTION_REASONS,
        *CONFOUNDER_NOT_APPLIED_REASONS,
        *CONFOUNDER_REQUIRES_EXTRACTION_REASONS,
        "population_inclusion_requires_extraction",
        "population_inclusion_not_applied",
        "tte_trial_not_confirmed",
        "easyicu_target_trial_compile_submitted",
        "target_trial_approved",
        "target_trial_compile_failed",
        *card_owner.TARGET_TRIAL_WORKFLOW_CODES,
        *card_owner.TARGET_TRIAL_APPROVAL_REFUSALS,
        # The host's own codes the setup and the click also return.
        "study_job_running",
        "job_capacity_exceeded",
        "study_context_revision_conflict",
        "study_context_active_job_conflict",
        "host_action_study_mismatch",
    ):
        assert reason in codes
