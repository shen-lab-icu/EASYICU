"""Local source-flow values retain field identity and never enter writer claims."""

import json

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.reporting.manuscript_provenance import build_manuscript_provenance
from easyicu.research_agent.reporting.writer_evidence import render_local_first_observed_source_flow


def _store(tmp_path):
    store = EvidenceStore(tmp_path)
    capsule = store.register_json(kind="log", description="Synthetic materialized capsule",
        payload={"cohort_sha256": "c" * 64}, filename="capsule.json",
        evidence_id="deferred_first_observed_actual_capsule")
    receipt = store.register_json(kind="log", description="Synthetic receipt (not a host approval)",
        payload={"body": {"approved_plan_sha256": "a" * 64, "decision_set_sha256": "b" * 64,
                          "recipe_sha256": "d" * 64, "capsule_sha256": capsule.sha256,
                          "cohort_sha256": "c" * 64}, "signature": "synthetic"},
        filename="receipt.json", evidence_id="deferred_first_observed_receipt")
    payload = {
        "schema_version": "easyicu.first-observed-source-flow-evidence/1",
        "not_lifetime_first_icu": True,
        "disclosure": "local_report_with_publication_review",
        "binding": {"approved_plan_sha256": "a" * 64, "decision_set_sha256": "b" * 64,
                    "recipe_sha256": "d" * 64, "coordinate_sha256": "e" * 64,
                    "source_attestation_sha256": "f" * 64, "actual_cohort_sha256": "c" * 64,
                    "actual_capsule_sha256": capsule.sha256, "deferred_receipt_sha256": receipt.sha256},
        "counts": {
            "source_patients_n": 4, "source_stays_n": 5, "eligible_patients_n": 3,
            "first_icu_order_or_type_unverifiable_n": 1,
            "excluded_patients_by_reason": {"first_icu_order_or_type_unverifiable_n": 1},
            "excluded_stays_by_reason": {"first_icu_order_or_type_unverifiable_n": 1, "non_selected_unit_n": 1},
        },
    }
    record = store.register_json(kind="table", description="Synthetic host-approved source flow",
        payload=payload, filename="first_observed_source_flow.json", evidence_id="first_observed_source_flow",
        producer="host", generation_mode="system", inputs=[capsule.evidence_id, receipt.evidence_id])
    for key, value in payload["counts"].items():
        for suffix, number in (value.items() if isinstance(value, dict) else ((None, value),)):
            source_field = "counts." + key + ("." + suffix if suffix else "")
            store.register_numeric_claim(value=str(number), canonical=float(number), evidence_id=record.evidence_id,
                step_id="first_observed_source_flow", source_field=source_field, tolerance=0)
    return store, record


@pytest.mark.parametrize("language", ["en", "zh"])
def test_same_valued_counts_keep_separate_exact_reader_fields(tmp_path, language):
    store, _ = _store(tmp_path)
    text, claims = render_local_first_observed_source_flow(evidence=store,
        approved_plan_sha256="a" * 64, decision_set_sha256="b" * 64, language=language)
    reader = build_manuscript_provenance(manuscript=text, evidence=store, binding_map=claims)
    assert reader["claim_count"] == 7
    same_values = [claim for claim in reader["claims"] if claim["canonical_value"] == 1]
    assert {claim["source_json_pointer"] for claim in same_values} == {
        "/counts/first_icu_order_or_type_unverifiable_n",
        "/counts/excluded_patients_by_reason/first_icu_order_or_type_unverifiable_n",
        "/counts/excluded_stays_by_reason/first_icu_order_or_type_unverifiable_n",
        "/counts/excluded_stays_by_reason/non_selected_unit_n",
    }
    assert all(claim["status"] == "current" for claim in reader["claims"])
    assert not store.authoritative_numeric_claims([])  # Not exposed to the writer's prompt.
    assert not reader["publication_authorized"]
    assert ("终生首次 ICU" if language == "zh" else "lifetime first ICU") in text
    assert ("排除 ICU 单元记录数" if language == "zh" else "Excluded ICU unit-stay records") in text


def test_local_report_rejects_changed_evidence_or_other_decision(tmp_path):
    store, record = _store(tmp_path)
    with pytest.raises(ValueError, match="authority_invalid"):
        render_local_first_observed_source_flow(evidence=store,
            approved_plan_sha256="a" * 64, decision_set_sha256="c" * 64, language="en")
    path = tmp_path / record.relative_path
    payload = json.loads(path.read_text())
    payload["counts"]["eligible_patients_n"] = 100
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="verification|identity|digest"):
        render_local_first_observed_source_flow(evidence=store,
            approved_plan_sha256="a" * 64, decision_set_sha256="b" * 64, language="en")


def test_local_report_requires_the_registered_capsule_and_receipt(tmp_path):
    store, _ = _store(tmp_path)
    dependency = store.get("deferred_first_observed_actual_capsule")
    (tmp_path / dependency.relative_path).unlink()
    with pytest.raises(RuntimeError, match="verification|identity|digest"):
        render_local_first_observed_source_flow(evidence=store,
            approved_plan_sha256="a" * 64, decision_set_sha256="b" * 64, language="en")
