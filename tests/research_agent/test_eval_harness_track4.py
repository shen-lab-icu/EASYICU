"""Track 4 skeleton tests: prereg/failure/denominator/recovery/U4 hook.

Structure and口径 only. These tests use synthetic items and assert no real
pass rate, no coverage claim, and no signed independent evaluation.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.evaluation.independent_eval_harness import (
    DENOMINATOR_EXCLUSIONS,
    FAILURE_KINDS,
    RECOVERY_KINDS,
    SCOPE_LABELS_ZH,
    SCOPE_MATRIX,
    SYSTEM_DEFECT_COUNTS_IN_DENOMINATOR,
    U4_STATUS_PENDING,
    FailureLedger,
    FailureRecord,
    HeldOutItem,
    PendingU4Decision,
    RecoveryCounts,
    RecoveryRecord,
    build_report,
    count_recoveries,
    is_excluded_from_denominator,
    lock_preregistration,
    request_journal_quality_adjudication,
)


def _item(item_id="q001", scope="prediction"):
    return HeldOutItem(
        item_id=item_id,
        version="v2026.09.17",
        scope=scope,
        eval_method="engine_adapter_smoke",
        denominator_id="heldout_v1",
        allowed_interventions=("retry_within_budget",),
    )


def test_scope_matrix_covers_six_dimensions_without_thresholds():
    assert tuple(SCOPE_MATRIX) == (
        "descriptive",
        "association",
        "prediction",
        "survival",
        "causal_simulation",
        "subtyping_trajectory",
    )
    assert set(SCOPE_LABELS_ZH) == set(SCOPE_MATRIX)


def test_preregistration_requires_item_version_method_denominator():
    with pytest.raises(ValueError):
        HeldOutItem(
            item_id="",
            version="v1",
            scope="prediction",
            eval_method="m",
            denominator_id="d",
        )
    with pytest.raises(ValueError):
        HeldOutItem(
            item_id="q1",
            version="v1",
            scope="not_a_scope",  # type: ignore[arg-type]
            eval_method="m",
            denominator_id="d",
        )
    with pytest.raises(ValueError):
        lock_preregistration([])


def test_preregistration_lock_binds_items_and_denominator():
    lock = lock_preregistration([_item("q001"), _item("q002", "survival")])
    assert lock.locked is True
    assert lock.denominator_size() == 2
    assert lock.digest.startswith("sha256:")
    assert lock_preregistration([_item("q001")]).digest != lock.digest
    assert lock.to_dict()["denominator_size"] == 2


def test_system_defects_count_in_denominator_and_nothing_excluded():
    assert SYSTEM_DEFECT_COUNTS_IN_DENOMINATOR is True
    assert DENOMINATOR_EXCLUSIONS == ()
    for kind in FAILURE_KINDS:
        assert is_excluded_from_denominator(kind) is False
    assert is_excluded_from_denominator("system_defect") is False
    record = FailureRecord(
        item_id="q001",
        item_version="v2026.09.17",
        kind="system_defect",
        attempt_id="a1",
        detail="synthetic defect",
    )
    assert record.counts_in_denominator() is True


def test_failures_are_always_retained_with_no_removal_api():
    ledger = FailureLedger()
    ledger.add(
        FailureRecord(
            item_id="q001",
            item_version="v2026.09.17",
            kind="system_defect",
            attempt_id="a1",
            detail="synthetic",
        )
    )
    assert len(ledger) == 1
    assert ledger.records()[0].retained is True
    assert not hasattr(ledger, "remove")
    assert not hasattr(ledger, "clear")
    assert not hasattr(ledger, "drop")
    with pytest.raises(ValueError):
        FailureRecord(
            item_id="q001",
            item_version="v1",
            kind="system_defect",
            attempt_id="a2",
            detail="must not be droppable",
            retained=False,
        )


def test_three_recovery_kinds_are_counted_separately():
    assert tuple(RECOVERY_KINDS) == (
        "in_run_recovery",
        "post_fix_recovery",
        "later_reuse",
    )
    counts = count_recoveries(
        [
            RecoveryRecord(item_id="q001", kind="in_run_recovery", human_involved=False),
            RecoveryRecord(item_id="q001", kind="post_fix_recovery", human_involved=True),
            RecoveryRecord(item_id="q002", kind="later_reuse", human_involved=False),
        ]
    )
    assert counts == RecoveryCounts(
        in_run_recovery=1, post_fix_recovery=1, later_reuse=1
    )
    assert set(counts.to_dict()) == set(RECOVERY_KINDS)


def test_report_binds_records_to_prereg_and_claims_nothing():
    lock = lock_preregistration([_item("q001")])
    report = build_report(
        lock,
        [
            FailureRecord(
                item_id="q001",
                item_version="v2026.09.17",
                kind="repairable_failure",
                attempt_id="a1",
                detail="synthetic",
            )
        ],
        [RecoveryRecord(item_id="q001", kind="in_run_recovery", human_involved=False)],
    )
    payload = report.to_dict()
    assert payload["denominator_size"] == 1
    assert len(payload["failures_retained"]) == 1
    assert payload["recovery_counts"]["in_run_recovery"] == 1
    assert payload["recovery_counts"]["post_fix_recovery"] == 0
    assert payload["u4_status"] == U4_STATUS_PENDING == "pending_user_decision"
    assert payload["coverage_claim"] is None
    assert "pass_rate" not in payload
    with pytest.raises(ValueError):
        build_report(lock, [], [RecoveryRecord(item_id="ghost", kind="later_reuse", human_involved=False)])


def test_u4_hook_never_signs_a_judgment():
    lock = lock_preregistration([_item("q001")])
    report = build_report(lock, [], [])
    with pytest.raises(PendingU4Decision):
        request_journal_quality_adjudication(report=report)
    with pytest.raises(PendingU4Decision):
        request_journal_quality_adjudication(report=report, adjudicator="anyone")
