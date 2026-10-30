"""Current parallel results cannot silently compete for a citation alias."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from easyicu.research_agent.authority.evidence_record_resolution import (
    _current_verified_evidence_record,
)
from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementError,
    EvidenceStore,
)
from easyicu.research_agent.authority.registration import StepEvidenceCommit
from easyicu.research_agent.reporting.writer_evidence import _preferred_writer_evidence_names


def _publish(store, step, alias, *, same_basename=False):
    record = store.register_text(
        kind="statistic", description=step, text=step,
        filename="shared.json" if same_basename else f"{step}.json",
        evidence_id=f"result_{step}", produced_by_step=step,
        publish_aliases=False,
    )
    result = StepEvidenceCommit(store).commit_validated_step(
        step_id=step, pending_aliases={record.evidence_id: [alias] if alias else []},
        allowed_evidence_ids=[record.evidence_id],
        register_numeric_claims=lambda: None,
    )
    row = {"step_id": step, "status": "ok", "evidence_ids": [record.evidence_id],
           "retained_cross_step_aliases": result.retained_cross_step_aliases}
    return record, row


@pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
def test_current_conflict_rejects_ambiguous_alias_and_allows_exact_ids(tmp_path, order):
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    first, first_row = _publish(store, order[0], "mortality_result")
    second, second_row = _publish(store, order[1], "mortality_result")
    rows = [first_row, second_row]
    assert second_row["retained_cross_step_aliases"] == {"mortality_result": first.evidence_id}
    assert _current_verified_evidence_record(store, "mortality_result", rows) is None
    assert _current_verified_evidence_record(store, "mortality", rows) is None
    with pytest.raises(EvidenceEnforcementError) as error:
        store.bind_manuscript("See {evidence:mortality_result}.", per_step_records=rows)
    assert error.value.detail["ambiguous_evidence_aliases"] == ["mortality_result"]
    with pytest.raises(EvidenceEnforcementError):
        store.enforce_evidence_bound_scaffold(
            "# Results\n\nSee Table 1 {evidence:mortality_result}.", per_step_records=rows,
        )
    for record in (first, second):
        assert _current_verified_evidence_record(store, record.evidence_id, rows).evidence_id == record.evidence_id
        assert record.relative_path in store.bind_manuscript(
            f"See {{evidence:{record.evidence_id}}}.", per_step_records=rows,
        )
        safe, removed = store.enforce_evidence_bound_scaffold(
            f"# Results\n\nSee Table 1 {{evidence:{record.evidence_id}}}.", per_step_records=rows,
        )
        assert not removed and record.evidence_id in safe


def test_implicit_same_basename_conflict_is_recorded_and_blocks_reference(tmp_path):
    store = EvidenceStore(tmp_path)
    first, first_row = _publish(store, "a", None, same_basename=True)
    _, second_row = _publish(store, "b", None, same_basename=True)
    assert second_row["retained_cross_step_aliases"] == {"shared": first.evidence_id}
    rows = [first_row, second_row]
    assert _current_verified_evidence_record(store, "shared", rows) is None
    assert "[evidence missing: shared]" in store.bind_manuscript(
        "See {evidence:shared}.", per_step_records=rows,
    )


def test_writer_receives_exact_owners_not_the_ambiguous_preferred_alias(tmp_path):
    store = EvidenceStore(tmp_path)
    first, first_row = _publish(store, "a", "primary_association")
    second, second_row = _publish(store, "b", "primary_association")
    rows = [first_row, second_row]
    names = store.current_resolvable_names(rows)
    preferred = _preferred_writer_evidence_names(store, rows)
    assert "primary_association" not in names
    assert "primary_association" not in preferred
    for record in (first, second):
        assert record.evidence_id in names
        assert record.evidence_id in preferred


def test_concurrent_publication_records_conflict_before_report_binding(tmp_path):
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    start = Barrier(2, timeout=5)

    def publish(step):
        start.wait()
        return _publish(store, step, "shared_result")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, ("a", "b")))
    rows = [row for _, row in results]
    assert sum(bool(row["retained_cross_step_aliases"]) for row in rows) == 1
    with pytest.raises(EvidenceEnforcementError) as error:
        store.bind_manuscript("See {evidence:shared_result}.", per_step_records=rows)
    assert error.value.detail["ambiguous_evidence_aliases"] == ["shared_result"]
    for record, _ in results:
        assert record.relative_path in store.bind_manuscript(
            f"See {{evidence:{record.evidence_id}}}.", per_step_records=rows,
        )


def test_conflict_survives_original_owner_retry_with_new_evidence_id(tmp_path):
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    old_owner, first_row = _publish(store, "a", "primary_association")
    other, other_row = _publish(store, "b", "primary_association")
    replacement = store.register_text(
        kind="statistic", description="a retry", text="new a",
        filename="new_a.json", evidence_id="result_a_retry",
        produced_by_step="a", publish_aliases=False,
    )
    StepEvidenceCommit(store).commit_validated_step(
        step_id="a", pending_aliases={replacement.evidence_id: ["primary_association"]},
        allowed_evidence_ids=[replacement.evidence_id], register_numeric_claims=lambda: None,
    )
    rows = [first_row, other_row, {"step_id": "a", "status": "ok",
                                   "evidence_ids": [replacement.evidence_id]}]
    assert store.aliases()["primary_association"] == replacement.evidence_id
    assert other_row["retained_cross_step_aliases"] == {"primary_association": old_owner.evidence_id}
    assert _current_verified_evidence_record(store, "primary_association", rows) is None
    assert "primary_association" not in store.current_resolvable_names(rows)
    preferred = _preferred_writer_evidence_names(store, rows)
    assert "primary_association" not in preferred
    assert {replacement.evidence_id, other.evidence_id} <= set(preferred)
    with pytest.raises(EvidenceEnforcementError) as error:
        store.bind_manuscript("See {evidence:primary_association}.", per_step_records=rows)
    assert error.value.detail["ambiguous_evidence_aliases"] == ["primary_association"]


def test_competing_step_cannot_hide_conflict_by_listing_winner_id(tmp_path):
    store = EvidenceStore(tmp_path)
    winner, winner_row = _publish(store, "a", "shared_result")
    _, claimant_row = _publish(store, "b", "shared_result")
    claimant_row["evidence_ids"].append(winner.evidence_id)
    rows = [winner_row, claimant_row]
    assert "shared_result" not in store.current_resolvable_names(rows)
    assert _current_verified_evidence_record(store, "shared_result", rows) is None


@pytest.mark.parametrize("new_status", ["failed", "ok"])
def test_superseded_conflict_does_not_poison_current_success(tmp_path, new_status):
    store = EvidenceStore(tmp_path)
    first, first_row = _publish(store, "a", "result")
    _, old_row = _publish(store, "b", "result")
    newer = {"step_id": "b", "status": new_status, "evidence_ids": []}
    rows = [first_row, old_row, newer]
    assert "result" in store.current_resolvable_names(rows)
    assert _current_verified_evidence_record(store, "result", rows).evidence_id == first.evidence_id
    assert first.relative_path in store.bind_manuscript("See {evidence:result}.", per_step_records=rows)
