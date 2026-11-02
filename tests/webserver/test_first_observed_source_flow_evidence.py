"""Approved source-flow projection preserves policy and exact local evidence."""

from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.webserver import first_observed_deferred_execution as owner
from tests.support.first_observed_pending import _pending


def _flow():
    return {
        "schema_version": "easyicu.first-observed-private-source-flow/1",
        "scope": "complete_source_before_other_study_filters",
        "not_lifetime_first_icu": True,
        "source_stays": 5,
        "source_patients": 4,
        "eligible_patients": 3,
        "excluded_patients_by_reason": {"first_icu_order_or_type_unverifiable": 1},
        "excluded_stays_by_reason": {"first_icu_order_or_type_unverifiable": 2},
        "public_projection_authorized": False,
    }


def test_small_cells_remain_exact_for_local_report_with_review():
    original = _flow()
    output = owner._public_source_flow(
        original, cohort_rows=3, binding={"actual_capsule_sha256": "a" * 64}
    )
    assert output["counts"]["first_icu_order_or_type_unverifiable_n"] == 1
    assert output["counts"]["source_stays_n"] == 5
    assert output["disclosure"] == "local_report_with_publication_review"
    assert not output["hospitalization_count_available"]
    assert original["public_projection_authorized"] is False
    with pytest.raises(ValueError, match="source_flow_invalid"):
        owner._public_source_flow(original, cohort_rows=2, binding={})


def test_source_flow_registers_exact_claims_against_actual_capsule(
    tmp_path, monkeypatch
):
    from easyicu.research_agent.intake import materialized_metadata
    from easyicu.webserver import agent_pipeline_runs

    scope, checkpoint, _ = _pending()
    source_run = tmp_path / "source"
    source_run.mkdir()
    (source_run / "raw_preparation_checkpoint.json").write_text(
        checkpoint.model_dump_json()
    )
    monkeypatch.setattr(
        agent_pipeline_runs,
        "raw_preparation_review_owner",
        lambda *_: (SimpleNamespace(run_dir=str(source_run)), {}),
    )
    flow = _flow()
    monkeypatch.setattr(
        owner,
        "_load_signed_source_flow",
        lambda _: (SimpleNamespace(sha256="5" * 64), flow, "6" * 64),
    )
    monkeypatch.setattr(
        materialized_metadata,
        "load_verified_materialized_cohort_authority",
        lambda _: SimpleNamespace(
            authority=SimpleNamespace(cohort_rows=3, cohort_sha256="b" * 64)
        ),
    )
    root = tmp_path / "run"
    root.mkdir()
    store = EvidenceStore(root)
    from easyicu.research_agent.authority.run_input import (
        DEFERRED_ACTUAL_CAPSULE_EVIDENCE_ID,
        DEFERRED_RECEIPT_EVIDENCE_ID,
    )

    capsule_record = store.register_text(
        kind="log",
        description="synthetic actual capsule",
        filename="actual.json",
        text="synthetic capsule",
        evidence_id=DEFERRED_ACTUAL_CAPSULE_EVIDENCE_ID,
    )
    receipt_record = store.register_text(
        kind="log",
        description="synthetic signed receipt",
        filename="receipt.json",
        text="synthetic receipt",
        evidence_id=DEFERRED_RECEIPT_EVIDENCE_ID,
    )
    recipe = {
        "source_flow_report": owner._FLOW_DECLARATION,
        "host_binding": {
            "preparation_run_id": "run",
            "study_context_id": "study",
            "project_id": "project",
            "session_id": "session",
        },
    }
    args = dict(
        run_dir=root,
        recipe=recipe,
        approved_plan_sha256="1" * 64,
        decision_set_sha256="2" * 64,
        recipe_sha256="3" * 64,
        capsule_sha256=capsule_record.sha256,
        receipt_sha256=receipt_record.sha256,
        coordinate=SimpleNamespace(sha256="5" * 64),
        cohort_path=root / "cohort.parquet",
        evidence_store=store,
    )
    owner._publish_source_flow(**args)
    record = store.get("first_observed_source_flow")
    assert record is not None and record.kind == "table"
    assert set(record.inputs) == {
        DEFERRED_ACTUAL_CAPSULE_EVIDENCE_ID,
        DEFERRED_RECEIPT_EVIDENCE_ID,
    }
    claims = {claim.source_field: claim.canonical for claim in store.numeric_claims()}
    assert claims["counts.first_icu_order_or_type_unverifiable_n"] == 1
    assert claims["counts.source_stays_n"] == 5
    payload = __import__("json").loads((root / record.relative_path).read_text())
    assert payload["binding"]["actual_capsule_sha256"] == capsule_record.sha256
    assert payload["binding"]["source_attestation_sha256"] == "6" * 64
    from easyicu.research_agent.reporting.publication_disclosure_review import (
        _result_table_cells,
    )

    cells, unread = _result_table_cells([record], root)
    assert not unread
    assert any(
        item["column"] == "first_icu_order_or_type_unverifiable_n"
        and 1 in item["values"]
        for item in cells
    )
    assert flow["public_projection_authorized"] is False
    owner._publish_source_flow(**args)
    assert len(store.numeric_claims()) == len(claims)
