from __future__ import annotations

from easyicu.research_agent.execution.concept_reaudit import (
    DETERMINISTIC_CONCEPT_REAUDIT_BUDGET_ISSUE_CODE,
    deterministic_concept_reaudit_authority,
    deterministic_concept_reaudit_pending_errors,
)


_DIGEST = "a" * 64
_REPAIR_ID = "availability_fraction_component_denominator_v1"


def _provider_budget_error(*, used: int = 9, limit: int = 9) -> dict[str, object]:
    return {
        "validator": "provider_call_budget",
        "severity": "error",
        "message": "Final concept audit had no provider slot.",
        "detail": {
            "step_id": "08_cluster_profile_figure",
            "category": "concept_audit",
            "used": used,
            "limit": limit,
        },
    }


def _prior_record(*, digest: str = _DIGEST) -> dict[str, object]:
    return {
        "quarantined_repair_succeeded": True,
        "applied_concept_repair_names": [_REPAIR_ID],
        "post_repair_concept_audit_block": {
            "code_sha256": digest,
            "errors": [_provider_budget_error()],
        },
    }


def test_current_exact_deterministic_repair_authorizes_its_own_digest() -> None:
    assert deterministic_concept_reaudit_authority(
        code_sha256=_DIGEST,
        current_repair_count=1,
        current_repair_names=[_REPAIR_ID],
        current_repair_code_sha256=_DIGEST,
        prior_step_record=None,
    ) == (_REPAIR_ID,)


def test_current_repair_does_not_authorize_later_mutated_digest() -> None:
    assert deterministic_concept_reaudit_authority(
        code_sha256="b" * 64,
        current_repair_count=1,
        current_repair_names=[_REPAIR_ID],
        current_repair_code_sha256=_DIGEST,
        prior_step_record=None,
    ) == ()


def test_resume_exact_budget_only_block_authorizes_reaudit() -> None:
    assert deterministic_concept_reaudit_authority(
        code_sha256=_DIGEST,
        current_repair_count=0,
        current_repair_names=[],
        prior_step_record=_prior_record(),
        provider_used=9,
        provider_limit=9,
    ) == (_REPAIR_ID,)


def test_resume_stale_digest_fails_closed() -> None:
    assert deterministic_concept_reaudit_authority(
        code_sha256="b" * 64,
        current_repair_count=0,
        current_repair_names=[],
        prior_step_record=_prior_record(),
        provider_used=9,
        provider_limit=9,
    ) == ()


def test_resume_semantic_error_alongside_budget_error_fails_closed() -> None:
    record = _prior_record()
    record["post_repair_concept_audit_block"]["errors"].append(
        {
            "validator": "llm_concept_audit",
            "severity": "error",
            "message": "Candidate changes the declared method.",
            "detail": {},
        }
    )

    assert deterministic_concept_reaudit_authority(
        code_sha256=_DIGEST,
        current_repair_count=0,
        current_repair_names=[],
        prior_step_record=record,
        provider_used=9,
        provider_limit=9,
    ) == ()


def test_unknown_or_method_substitution_repair_fails_closed() -> None:
    record = _prior_record()
    record["applied_concept_repair_names"] = ["unregistered_scientific_fix_v1"]

    assert deterministic_concept_reaudit_authority(
        code_sha256=_DIGEST,
        current_repair_count=0,
        current_repair_names=[],
        prior_step_record=record,
        provider_used=9,
        provider_limit=9,
    ) == ()


def test_unexhausted_budget_does_not_authorize_resume_extension() -> None:
    record = _prior_record()
    record["post_repair_concept_audit_block"]["errors"] = [
        _provider_budget_error(used=8, limit=9)
    ]

    assert deterministic_concept_reaudit_authority(
        code_sha256=_DIGEST,
        current_repair_count=0,
        current_repair_names=[],
        prior_step_record=record,
        provider_used=8,
        provider_limit=9,
    ) == ()


def test_append_only_history_recovers_latest_exact_budget_only_proof() -> None:
    later_stale_quarantine_projection = {
        "step_id": "08_render",
        "status": "blocked_by_concept_audit",
        "quarantined_draft_sha256": _DIGEST,
        "usage_findings": [
            {
                "validator": "llm_concept_auditor",
                "severity": "error",
                "message": "Historical repaired constraint.",
                "detail": {},
            },
            _provider_budget_error(),
        ],
    }

    assert deterministic_concept_reaudit_authority(
        code_sha256=_DIGEST,
        current_repair_count=0,
        current_repair_names=[],
        prior_step_record=later_stale_quarantine_projection,
        prior_step_records=[_prior_record(), later_stale_quarantine_projection],
        provider_used=9,
        provider_limit=9,
    ) == (_REPAIR_ID,)


def test_newer_exact_semantic_block_supersedes_old_budget_only_proof() -> None:
    newer = _prior_record()
    newer["post_repair_concept_audit_block"]["errors"] = [
        {
            "validator": "llm_concept_auditor",
            "severity": "error",
            "message": "Fresh exact-digest semantic rejection.",
            "detail": {},
        }
    ]

    assert deterministic_concept_reaudit_authority(
        code_sha256=_DIGEST,
        current_repair_count=0,
        current_repair_names=[],
        prior_step_record=newer,
        prior_step_records=[_prior_record(), newer],
        provider_used=9,
        provider_limit=9,
    ) == ()


def test_resume_pending_errors_keep_only_matching_final_budget_failure() -> None:
    findings = [
        {
            "validator": "llm_concept_auditor",
            "severity": "error",
            "message": "Historical repaired constraint.",
            "detail": {},
        },
        _provider_budget_error(),
    ]

    expected = _provider_budget_error()
    expected["detail"]["issue_code"] = (
        DETERMINISTIC_CONCEPT_REAUDIT_BUDGET_ISSUE_CODE
    )
    assert deterministic_concept_reaudit_pending_errors(
        findings,
        provider_used=9,
        provider_limit=9,
    ) == (expected,)
    assert (
        deterministic_concept_reaudit_pending_errors(
            findings,
            provider_used=10,
            provider_limit=10,
        )
        == ()
    )
