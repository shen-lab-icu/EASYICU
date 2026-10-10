"""The approval card of a causal study's target trial, and the click on it.

Owner
-----
The host computes everything the card shows from three owners: the study's
``target_trial_design`` section (the digest it names, the lines to confirm,
the approval), the record the host keeps under that digest
(``target_trial_records``) and the study's latest compile job
(``target_trial_setup``).  The browser renders the card's state; it never
decides whether a record can be approved -- the compile owner's
``approval_blockers`` do -- nor whether the section is the latest statement.

A section is stale when the latest compile job is not the one that wrote it:
the job is still compiling, stopped, failed, or wrote another record.  A stale
section is shown as the previous version and cannot be approved, so a page
left open on an earlier statement cannot approve it.

The click (:func:`approve_target_trial`) is the only approval.  It must be on
the section the study names, at the study's revision, with no job running and
no newer statement pending; a second click on an approved record returns the
approval it already has.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from easyicu.research_agent.planning.population_compile import (
    POPULATION_APPROVAL_STOPS,
)
from easyicu.research_agent.planning.primary_result_contract import (
    TTE_TRIAL_NOT_CONFIRMED,
)
from easyicu.research_agent.planning.target_trial_compile import (
    CONFOUNDER_NOT_APPLIED_REASONS,
    CONFOUNDER_REQUIRES_EXTRACTION_REASONS,
    NOT_APPLIED_REASONS,
    REQUIRES_EXTRACTION_REASONS,
)
from easyicu.research_agent.planning.target_trial_configuration import (
    TargetTrialCompileRecord,
    TargetTrialDesign,
    TargetTrialDesignError,
    load_target_trial_design,
)
from easyicu.webserver import study_contexts
from easyicu.webserver.target_trial_records import (
    TargetTrialRecordError,
    load_target_trial_record,
)
from easyicu.webserver.target_trial_setup import (
    TARGET_TRIAL_COMPILE_FAILED,
    TARGET_TRIAL_COMPILE_INTERRUPTED,
    TARGET_TRIAL_COMPILE_STOPS,
    TARGET_TRIAL_COMPILE_SUBMITTED,
    TARGET_TRIAL_SETUP_REFUSALS,
    latest_target_trial_compile,
    target_trial_family_declared,
)

TARGET_TRIAL_CARD_SCHEMA_VERSION = "easyicu.target_trial_card/1"
#: The host action row the click writes.
TARGET_TRIAL_APPROVED = "target_trial_approved"
#: The workflow's next actions while a causal study's trial is not approved:
#: the study does not plan.
TARGET_TRIAL_HOLDS_PLAN = (
    "target_trial_statement_needed",
    "target_trial_review",
)
#: The study's trial is approved: its plan is generated on the study's data,
#: where the run verifies the approved record, never as a metadata-only
#: candidate (``routes.agent`` keeps candidate authority for other codes).
TARGET_TRIAL_PLAN_READY = "target_trial_plan_ready"
TARGET_TRIAL_WORKFLOW_CODES = (*TARGET_TRIAL_HOLDS_PLAN, TARGET_TRIAL_PLAN_READY)
#: Why a click does not approve.  Stable codes.
TARGET_TRIAL_APPROVAL_REFUSALS = (
    "target_trial_restatement_pending",
    "target_trial_approval_record_mismatch",
    "target_trial_design_invalid",
    "target_trial_record_missing",
)
#: The host's own codes the setup and the click also return.
_HOST_CODES = (
    "study_job_running",
    "job_capacity_exceeded",
    "study_context_revision_conflict",
    "study_context_active_job_conflict",
    "host_action_study_mismatch",
)
#: Every code the card, the conversation and the workflow show for a target
#: trial; the browser's copy covers exactly these.
TARGET_TRIAL_UI_CODES = tuple(
    dict.fromkeys(
        (
            *NOT_APPLIED_REASONS,
            *REQUIRES_EXTRACTION_REASONS,
            *CONFOUNDER_NOT_APPLIED_REASONS,
            *CONFOUNDER_REQUIRES_EXTRACTION_REASONS,
            TTE_TRIAL_NOT_CONFIRMED,
            *POPULATION_APPROVAL_STOPS.values(),
            *TARGET_TRIAL_SETUP_REFUSALS,
            *TARGET_TRIAL_COMPILE_STOPS,
            TARGET_TRIAL_COMPILE_FAILED,
            TARGET_TRIAL_COMPILE_INTERRUPTED,
            TARGET_TRIAL_COMPILE_SUBMITTED,
            TARGET_TRIAL_APPROVED,
            *TARGET_TRIAL_WORKFLOW_CODES,
            *TARGET_TRIAL_APPROVAL_REFUSALS,
            *_HOST_CODES,
        )
    )
)


class TargetTrialApprovalError(RuntimeError):
    """The click does not approve; the card says why and what to do."""

    def __init__(self, code: str, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _section(study: Mapping[str, Any]) -> Optional[TargetTrialDesign]:
    try:
        return load_target_trial_design(
            study.get("target_trial_design"), study_id=str(study.get("id") or "")
        )
    except TargetTrialDesignError:
        return None


def _kept(study_id: str, section: TargetTrialDesign) -> Optional[TargetTrialCompileRecord]:
    try:
        kept = load_target_trial_record(study_id, section.compile_sha256)
        section.check_record(kept)
    except (TargetTrialRecordError, TargetTrialDesignError):
        return None
    return kept


def _stale(
    section: Optional[TargetTrialDesign], latest: Optional[Mapping[str, Any]]
) -> bool:
    if latest is None:
        return False
    return (
        latest.get("status") != "compiled"
        or section is None
        or latest.get("compile_sha256") != section.compile_sha256
    )


def target_trial_card(
    study: Mapping[str, Any], latest: Optional[Mapping[str, Any]]
) -> Optional[dict[str, Any]]:
    """The card of the study's trial, or ``None`` when there is none to show.

    ``latest`` is the study's latest compile job
    (:func:`~easyicu.webserver.target_trial_setup.latest_target_trial_compile`),
    read once by the workflow that shows the card.
    """

    if not target_trial_family_declared(study):
        return None
    study_id = str(study.get("id") or "")
    section = _section(study)
    if section is None and latest is None:
        return None
    kept = _kept(study_id, section) if section is not None else None
    stale = _stale(section, latest)
    record = kept.record if kept is not None else {}
    approvable = bool(kept is not None and kept.approvable and not stale)
    if latest is not None and latest.get("status") == "running":
        state = "compiling"
    elif section is None or kept is None or stale:
        state = "stopped"
    elif section.approval is not None:
        state = "approved"
    else:
        state = "approvable" if kept.approvable else "blocked"
    return {
        "schema_version": TARGET_TRIAL_CARD_SCHEMA_VERSION,
        "state": state,
        "reason_code": (
            "target_trial_record_missing"
            if section is not None and kept is None
            else None
        ),
        "compile_sha256": section.compile_sha256 if section is not None else None,
        "confirmation_lines": (
            section.confirmation_lines if section is not None else None
        ),
        "protocol": list(record.get("protocol") or ()),
        "confirmations": list(record.get("confirmations") or ()),
        "limitations": list(record.get("limitations") or ()),
        "evidence_ceiling": record.get("evidence_ceiling"),
        "approvable": approvable,
        "blocking": list(record.get("approval_blockers") or ()),
        "approval": (
            section.approval.model_dump(mode="json")
            if section is not None and section.approval is not None
            else None
        ),
        "latest_compile": dict(latest) if latest is not None else None,
        "stale": stale,
        # Why the host read the question as a trial, in its words, when it did
        # (``causal_trial_design``); the researcher's own choice states none.
        "causal_trial_reading": study.get("causal_trial_reading") or None,
    }


def _approved_and_current(
    study: Mapping[str, Any], latest: Optional[Mapping[str, Any]]
) -> Optional[bool]:
    """Whether the study's approved trial is the one a run binds now.

    ``None`` when the study approved no trial, or a newer statement holds
    it.  ``False`` when the host no longer keeps the record the approval
    names: the card says what is missing and the study does not plan.
    """

    section = _section(study)
    if section is None or section.approval is None or _stale(section, latest):
        return None
    return _kept(str(study.get("id") or ""), section) is not None


def target_trial_plans_on_data(
    study: Mapping[str, Any], latest: Optional[Mapping[str, Any]]
) -> bool:
    """Whether every plan of the study is generated on its data.

    So for a causal study whose approved trial a run binds now: the run
    compiles the trial again on the context its rows build and requires the
    approved record, which a metadata-only candidate context cannot give
    (``routes.agent`` grants such a study no candidate plan).
    """

    return bool(
        target_trial_family_declared(study) and _approved_and_current(study, latest)
    )


def target_trial_next_action(
    study: Mapping[str, Any], latest: Optional[Mapping[str, Any]]
) -> Optional[str]:
    """What a causal study whose plan step is ready does next.

    ``None`` for a study of another family.  A study whose trial is approved
    plans on its data (:data:`TARGET_TRIAL_PLAN_READY`,
    :func:`target_trial_plans_on_data`).
    """

    if not target_trial_family_declared(study):
        return None
    current = _approved_and_current(study, latest)
    if current is not None:
        return TARGET_TRIAL_PLAN_READY if current else "target_trial_review"
    if _section(study) is None and latest is None:
        return "target_trial_statement_needed"
    return "target_trial_review"


def approve_target_trial(
    study_id: str,
    *,
    expected_revision: int,
    compile_sha256: str,
    n_lines_confirmed: int,
) -> dict[str, Any]:
    """Record the researcher's click on the card showing ``compile_sha256``.

    Returns ``{study, approval_event_id, repeated}``; ``repeated`` when the
    record was approved by an earlier click.
    """

    study = study_contexts.get_context(study_id)
    if study is None:
        raise TargetTrialApprovalError(
            "study_context_not_found", "The study is not known.", status_code=404
        )
    section = _section(study)
    approval = section.approval if section is not None else None
    if (
        approval is not None
        and approval.confirmed_compile_sha256 == compile_sha256
        and approval.n_lines_confirmed == n_lines_confirmed
    ):
        return {
            "study": study,
            "approval_event_id": approval.approval_event_id,
            "repeated": True,
        }
    if str(study.get("active_job_id") or "").strip():
        raise TargetTrialApprovalError(
            "study_context_active_job_conflict",
            "A job is running for this study; approve the trial when it ends.",
            status_code=409,
        )
    if _stale(section, latest_target_trial_compile(study_id)):
        raise TargetTrialApprovalError(
            "target_trial_restatement_pending",
            "A newer statement of the trial has not compiled to this record.",
            status_code=409,
        )
    try:
        approved = study_contexts.record_target_trial_approval(
            study_id,
            confirmed_compile_sha256=compile_sha256,
            n_lines_confirmed=n_lines_confirmed,
            expected_revision=expected_revision,
        )
    except study_contexts.StudyContextError as exc:
        code = str(exc.detail.get("error") or "target_trial_design_invalid")
        status = {
            "study_context_revision_conflict": 409,
            "target_trial_record_missing": 409,
            "target_trial_approval_record_mismatch": 422,
        }.get(code, 422)
        raise TargetTrialApprovalError(
            code, "The click does not approve this trial.", status_code=status
        ) from exc
    return {
        "study": approved,
        "approval_event_id": approved["target_trial_design"]["approval"][
            "approval_event_id"
        ],
        "repeated": False,
    }


__all__ = [
    "TARGET_TRIAL_APPROVAL_REFUSALS",
    "TARGET_TRIAL_APPROVED",
    "TARGET_TRIAL_CARD_SCHEMA_VERSION",
    "TARGET_TRIAL_HOLDS_PLAN",
    "TARGET_TRIAL_PLAN_READY",
    "TARGET_TRIAL_UI_CODES",
    "TARGET_TRIAL_WORKFLOW_CODES",
    "TargetTrialApprovalError",
    "approve_target_trial",
    "target_trial_card",
    "target_trial_next_action",
    "target_trial_plans_on_data",
]
