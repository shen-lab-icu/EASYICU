from __future__ import annotations

import copy
import json

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.scientific_claims import (
    derive_scientific_claim_drafts,
    scientific_claim_compilation_requested,
)


def _summary(*, primary: bool = False, outcome: str = "death") -> dict:
    summary = {
        "interpretation_class": "absolute_risk_context",
        "analysis_family": "absolute_risk_context",
        "status": "ok",
        "outcome": outcome,
        "adjusted_effect": None,
        "n_total": 144 if primary else 160,
        "outcome_nonmissing_n": 144,
        "outcome_missing_n": 0 if primary else 16,
        "reportable_descriptive_results": {
            "schema_version": "easyicu.absolute_risk_reporting/1",
            "execution_owner": "absolute_risk_context_executor_v1",
            "interpretation_ceiling": "descriptive_not_causal",
            "overall_outcome": {
                "outcome": outcome,
                "n": 144,
                "event_n": 18,
                "risk_pct": 12.5,
            },
        },
    }
    if primary:
        summary["population_binding"] = {
            "schema_version": "easyicu.primary_population_descriptive/1",
            "scope": "primary_model_complete_cases",
            "owner_ref": "scientific_runtime_contract:" + "a" * 64,
            "runtime_projection_sha256": "b" * 64,
            "source_cohort_n": 600,
            "population_n": 144,
            "event_n": 18,
        }
    return summary


@pytest.mark.parametrize("primary", [False, True])
@pytest.mark.parametrize("outcome", ["death", "icu_readmission", "renal_replacement"])
def test_frequency_preserves_outcome_and_analysis_population(primary, outcome):
    summary = _summary(primary=primary, outcome=outcome)
    original = copy.deepcopy(summary)
    (claim,) = derive_scientific_claim_drafts(summary)
    assert summary == original
    assert claim.outcome == outcome
    assert claim.direction == "descriptive_only" and claim.adjusted_for == []
    assert "18 events among 144 records" in claim.estimand
    assert "12.5%" in claim.estimand
    assert "160" not in claim.estimand and "600" not in claim.estimand
    assert ("primary model complete-case" in claim.population) is primary
    assert claim.point_estimate is None and claim.confidence_level is None


@pytest.mark.parametrize(
    "path,value",
    [
        (("outcome",), "other_endpoint"),
        (("status",), "failed"),
        (("adjusted_effect",), 1.5),
        (("n_total",), 145),
        (("outcome_nonmissing_n",), True),
        (("reportable_descriptive_results", "execution_owner"), "generated_code"),
        (("reportable_descriptive_results", "interpretation_ceiling"), "causal"),
        (("reportable_descriptive_results", "schema_version"), "unknown"),
        (("reportable_descriptive_results", "overall_outcome", "event_n"), 145),
        (("reportable_descriptive_results", "overall_outcome", "n"), True),
        (("reportable_descriptive_results", "overall_outcome", "risk_pct"), 15.5),
        (
            ("reportable_descriptive_results", "overall_outcome", "risk_pct"),
            float("nan"),
        ),
        (("population_binding", "scope"), "all_source_rows"),
        (("population_binding", "population_n"), 600),
        (("population_binding", "event_n"), 20),
        (("population_binding", "owner_ref"), "unbound"),
    ],
)
def test_frequency_rejects_inconsistent_authority_or_counts(path, value):
    summary = _summary(primary=True)
    node = summary
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match="absolute-risk reporting"):
        derive_scientific_claim_drafts(summary)


def test_legacy_summary_cannot_acquire_claim_authority_from_method_name():
    summary = {"interpretation_class": "absolute_risk_context"}
    assert scientific_claim_compilation_requested(summary) is False
    assert derive_scientific_claim_drafts(summary) == []


def test_empty_outcomes_do_not_create_an_absolute_risk():
    summary = _summary()
    summary.update(outcome_missing_n=160, outcome_nonmissing_n=0)
    summary["reportable_descriptive_results"]["overall_outcome"].update(
        n=0, event_n=0, risk_pct=None
    )
    assert derive_scientific_claim_drafts(summary) == []


def test_frequency_is_sealed_idempotent_and_usable_by_strict_writer(tmp_path):
    summary = _summary(primary=True)
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(summary))
    record = store.register_file(
        kind="statistic",
        description="Completed descriptive analysis",
        source_path=source,
        evidence_id="risk_summary",
        produced_by_step="risk",
        generation_mode="deterministic_standard",
    )
    before = (tmp_path / record.relative_path).read_bytes()
    for _ in range(2):
        store.register_step_summary_numerics(
            step_id="risk", evidence_id=record.evidence_id, summary=summary
        )
    (claim,) = store.scientific_claims()
    assert len(store.scientific_claims()) == 1
    assert "18 events among 144 records" in claim.render_reader_text()
    assert "no confidence interval" in claim.render_reader_text()
    store.enforce_evidence_bound_scaffold("## Results\n\n" + claim.placeholder)
    assert (tmp_path / record.relative_path).read_bytes() == before
    altered = copy.deepcopy(summary)
    altered["reportable_descriptive_results"]["overall_outcome"].update(
        event_n=36, risk_pct=25.0
    )
    altered["population_binding"]["event_n"] = 36
    with pytest.raises(ValueError, match="registered summary bytes"):
        store.register_step_summary_numerics(
            step_id="risk", evidence_id=record.evidence_id, summary=altered
        )


def test_generated_summary_cannot_register_a_host_scientific_claim(tmp_path):
    summary = _summary()
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(summary))
    record = store.register_file(
        kind="statistic",
        description="Untrusted generated summary",
        source_path=source,
        evidence_id="generated",
        produced_by_step="risk",
        generation_mode="llm",
    )
    store.register_step_summary_numerics(
        step_id="risk", evidence_id=record.evidence_id, summary=summary
    )
    assert store.scientific_claims() == []
