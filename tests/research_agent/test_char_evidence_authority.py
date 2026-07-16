"""Freeze-baseline characterization for current evidence authority.

The scenarios mirror Appendix A (G1) in
``task_logs/20260715_agent_freeze_refactor_safety_plan.md``.  They describe
the observable behavior of the frozen engine; they are not a specification
for a new authority model.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _step_record(
    step_id: str,
    status: str,
    *evidence_ids: str,
) -> dict[str, object]:
    return {
        "step_id": step_id,
        "status": status,
        "evidence_ids": list(evidence_ids),
    }


def test_current_verified_records_follow_latest_outer_checkpoint(ra, tmp_path: Path):
    store = ra.EvidenceStore(tmp_path)
    first = store.register_text(
        kind="statistic",
        description="First sealed result.",
        text='{"estimate": 1}',
        filename="result_v1.json",
        evidence_id="result_v1",
        produced_by_step="01_model",
        publish_aliases=False,
    )
    failed_attempt = store.register_text(
        kind="statistic",
        description="Later failed result.",
        text='{"estimate": 2}',
        filename="result_failed.json",
        evidence_id="result_failed",
        produced_by_step="01_model",
        publish_aliases=False,
    )
    latest_success = store.register_text(
        kind="statistic",
        description="Later successful result.",
        text='{"estimate": 3}',
        filename="result_v2.json",
        evidence_id="result_v2",
        produced_by_step="01_model",
        publish_aliases=False,
    )

    failed_is_current = [
        _step_record("01_model", "ok", first.evidence_id),
        _step_record("01_model", "contract_failed", failed_attempt.evidence_id),
    ]
    assert store.current_verified_records(failed_is_current) == []

    later_success_is_current = [
        *failed_is_current,
        _step_record("01_model", "ok", latest_success.evidence_id),
    ]
    assert [
        record.evidence_id
        for record in store.current_verified_records(later_success_is_current)
    ] == [latest_success.evidence_id]


def test_authoritative_numeric_claims_are_current_and_evidence_scoped(
    ra,
    tmp_path: Path,
):
    store = ra.EvidenceStore(tmp_path)
    accepted = store.register_text(
        kind="statistic",
        description="Accepted summary.",
        text='{"estimate": 1.25}',
        filename="accepted.json",
        evidence_id="accepted_summary",
        produced_by_step="01_model",
        publish_aliases=False,
    )
    rejected = store.register_text(
        kind="statistic",
        description="Rejected summary.",
        text='{"estimate": 9.99}',
        filename="rejected.json",
        evidence_id="rejected_summary",
        produced_by_step="02_failed",
        publish_aliases=False,
    )
    accepted_peer = store.register_text(
        kind="statistic",
        description="Independent accepted summary with the same value.",
        text='{"estimate": 1.25}',
        filename="accepted_peer.json",
        evidence_id="accepted_summary_peer",
        produced_by_step="03_model",
        publish_aliases=False,
    )

    for _ in range(2):
        store.register_numeric_claim(
            value="1.25",
            canonical=1.25,
            evidence_id=accepted.evidence_id,
            step_id="01_model",
            source_field="primary.estimate",
        )
    store.register_numeric_claim(
        value="9.99",
        canonical=9.99,
        evidence_id=rejected.evidence_id,
        step_id="02_failed",
        source_field="primary.estimate",
    )
    store.register_numeric_claim(
        value="1.25",
        canonical=1.25,
        evidence_id=accepted_peer.evidence_id,
        step_id="03_model",
        source_field="primary.estimate",
    )

    claims = store.authoritative_numeric_claims(
        [
            _step_record("01_model", "ok", accepted.evidence_id),
            _step_record("02_failed", "ok", rejected.evidence_id),
            _step_record("02_failed", "contract_failed"),
            _step_record("03_model", "ok", accepted_peer.evidence_id),
        ]
    )

    assert {
        (claim.evidence_id, claim.step_id, claim.source_field, claim.value)
        for claim in claims
    } == {
        (accepted.evidence_id, "01_model", "primary.estimate", "1.25"),
        (accepted_peer.evidence_id, "03_model", "primary.estimate", "1.25"),
    }


def test_bind_manuscript_missing_evidence_soft_and_strict(ra, tmp_path: Path):
    soft = ra.EvidenceStore(tmp_path / "soft")
    assert soft.bind_manuscript("See {evidence:missing_result}.") == (
        "See [evidence missing: missing_result]."
    )

    strict = ra.EvidenceStore(tmp_path / "strict", enforcement_mode="strict")
    with pytest.raises(ra.EvidenceEnforcementError) as exc_info:
        strict.bind_manuscript("See {evidence:missing_result}.")
    assert exc_info.value.detail == {"missing_evidence_ids": ["missing_result"]}


def test_result_like_markdown_bullets_do_not_bypass_scaffold_guard(
    ra,
    tmp_path: Path,
):
    scaffold = (
        "- Mortality was lower in the intervention arm.\n"
        "> Mortality was higher after adjustment.\n"
        "- The prespecified study design is described here.\n"
    )
    soft = ra.EvidenceStore(tmp_path / "soft")
    filtered, removed = soft.enforce_evidence_bound_scaffold(scaffold)
    assert removed == [
        "Mortality was lower in the intervention arm.",
        "Mortality was higher after adjustment.",
    ]
    assert "prespecified study design" in filtered
    assert "Mortality was lower" not in filtered

    strict = ra.EvidenceStore(tmp_path / "strict", enforcement_mode="strict")
    with pytest.raises(ra.EvidenceEnforcementError) as exc_info:
        strict.enforce_evidence_bound_scaffold(scaffold)
    assert exc_info.value.detail["removed_sentences"] == removed


def test_success_promotion_moves_current_alias_only_within_same_step(
    ra,
    tmp_path: Path,
):
    store = ra.EvidenceStore(tmp_path)
    first = store.register_text(
        kind="statistic",
        description="First successful attempt.",
        text='{"estimate": 1}',
        filename="first.json",
        evidence_id="result_attempt_1",
        produced_by_step="01_model",
        publish_aliases=False,
    )
    second = store.register_text(
        kind="statistic",
        description="Second successful attempt.",
        text='{"estimate": 2}',
        filename="second.json",
        evidence_id="result_attempt_2",
        produced_by_step="01_model",
        publish_aliases=False,
    )

    store.publish_success_aliases(first.evidence_id, aliases=["primary_result"])
    assert store.aliases()["primary_result"] == first.evidence_id
    store.publish_success_aliases(second.evidence_id, aliases=["primary_result"])

    assert store.aliases()["primary_result"] == second.evidence_id
    assert store.get("primary_result").evidence_id == second.evidence_id
    assert [
        record.evidence_id
        for record in store.current_verified_records(
            [_step_record("01_model", "ok", second.evidence_id)]
        )
    ] == [second.evidence_id]
