from copy import deepcopy
from types import SimpleNamespace

import pytest

from easyicu.research_agent.reporting.descriptive_report_facts import (
    compile_counts_only_report_facts,
    place_descriptive_report_facts,
    render_descriptive_report_claims,
)
from easyicu.research_agent.reporting.manuscript_quality import remove_empty_optional_subsections


def _inputs():
    def row(level, count, denominator, *, events=False):
        return {
            "level_index": level, "level": level, "events" if events else "n": count,
            "denominator": denominator, "estimate_pct": 100 * count / denominator,
            "interval_method": "none_counts_only", "covariance": "none_counts_only",
            "ci_low_pct": None, "ci_high_pct": None, "confidence_level": None,
            "standard_error_pct": None, "cluster_count": None,
        }

    estimates = {
        "schema_version": "easyicu.exposure_outcome_descriptive_estimates/1",
        "analysis_role": "primary", "analysis_set": "bound_typed_cohort",
        "interpretation_ceiling": "descriptive_unadjusted_not_causal",
        "exposure_prevalence": [row(0, 60, 100), row(1, 40, 100)],
        "outcome_absolute_risks": [row(0, 6, 60, events=True), row(1, 8, 40, events=True)],
        "risk_difference": None, "dependence": None,
    }
    summary = {
        "status": "ok", "interpretation_class": "exposure_outcome_distribution",
        "analysis_role": "primary", "analysis_set": "bound_typed_cohort",
        "interpretation_ceiling": "descriptive_unadjusted_not_causal",
        "adjusted_effect": None, "interval_method": "none_counts_only",
        "cohort_n": 100, "exposure": "exposure", "outcome": "outcome",
        "descriptive_estimates": estimates,
    }
    records = [{"step_id": "distribution", "step_summary_evidence_id": "summary", "status": "ok", "step_summary": summary}]
    source = SimpleNamespace(evidence_id="summary", produced_by_step="distribution", sha256="a" * 64)
    evidence = SimpleNamespace(get=lambda key: source if key == "summary" else None)
    return records, evidence


def test_counts_and_outcomes_are_host_copied_with_separate_metric_owners():
    records, evidence = _inputs()
    before = deepcopy(records)
    facts = compile_counts_only_report_facts(records, evidence=evidence, reader_display_labels={"exposure=0": "Reference category", "exposure=1": "Other category", "outcome": "Observed endpoint"})
    assert len(facts) == 4
    assert "60 of 100 observations (60.00%)" in facts[0].scaffold
    assert "6 of 60 observations (10.00%)" in facts[2].scaffold
    assert "Reference category" in facts[0].scaffold
    assert "Observed endpoint" in facts[2].scaffold
    assert facts[0].subsection == "Cohort characteristics"
    assert facts[2].subsection == "Primary outcome"
    assert all(fact.evidence_id == "summary" and fact.source_sha256 == "a" * 64 for fact in facts)
    assert facts[0].source_fields == tuple(
        f"descriptive_estimates.exposure_prevalence[0].{key}"
        for key in ("level", "n", "denominator", "estimate_pct")
    )
    assert facts[2].source_fields == tuple(
        f"descriptive_estimates.outcome_absolute_risks[0].{key}"
        for key in ("level", "events", "denominator", "estimate_pct")
    )
    assert records == before
    text = "## Results\n\n### Cohort characteristics\n\n### Primary outcome\n\n### Primary association\n\nExisting comparison boundary.\n\n## Discussion\n\nUnchanged discussion."
    placed = place_descriptive_report_facts(text, facts)
    assert all(fact.scaffold in placed for fact in facts)
    assert place_descriptive_report_facts(placed, facts) == placed
    assert placed.endswith("## Discussion\n\nUnchanged discussion.")
    assert place_descriptive_report_facts("## Methods\n\nNo Results section.", facts) == "## Methods\n\nNo Results section."


@pytest.mark.parametrize("kind", [
    "wrong_owner", "failed", "bad_prevalence", "wrong_denominator", "incomplete_partition",
    "outcome_group_swap", "wrong_events", "nan", "bool_count", "false_interval", "duplicate_level",
])
def test_facts_reject_contradictory_counts_levels_or_owner(kind):
    records, evidence = _inputs()
    record = records[0]
    summary = record["step_summary"]
    prevalence = summary["descriptive_estimates"]["exposure_prevalence"]
    risks = summary["descriptive_estimates"]["outcome_absolute_risks"]
    if kind == "wrong_owner":
        record["step_id"] = "foreign"
    elif kind == "failed":
        record["status"] = "failed"
    elif kind == "bad_prevalence":
        prevalence[0]["estimate_pct"] = 99
    elif kind == "wrong_denominator":
        risks[0]["denominator"] = 100
    elif kind == "incomplete_partition":
        prevalence[0].update(n=50, estimate_pct=50)
    elif kind == "outcome_group_swap":
        risks[0]["level"] = 1
    elif kind == "wrong_events":
        risks[0]["events"] = 7
    elif kind == "nan":
        prevalence[0]["estimate_pct"] = float("nan")
    elif kind == "bool_count":
        prevalence[0]["n"] = True
    elif kind == "false_interval":
        risks[0]["ci_low_pct"] = 1.0
    elif kind == "duplicate_level":
        prevalence[1]["level"] = 0
    with pytest.raises(ValueError):
        compile_counts_only_report_facts(records, evidence=evidence, reader_display_labels={})


def test_numeric_fact_adapter_does_not_claim_other_analysis_families():
    records, evidence = _inputs()
    records[0]["step_summary"]["interpretation_class"] = "logistic_regression"
    assert compile_counts_only_report_facts(records, evidence=evidence, reader_display_labels={}) == ()


def test_legacy_claim_is_replaced_once_by_its_source_fact_not_duplicate_numbers():
    from easyicu.research_agent.authority.scientific_claims import (
        bind_scientific_claim_drafts, derive_scientific_claim_drafts,
    )
    records, evidence = _inputs()
    claims = bind_scientific_claim_drafts(
        [draft.model_dump(mode="json") for draft in derive_scientific_claim_drafts(records[0]["step_summary"])],
        step_id="distribution", evidence_id="summary",
    )
    facts = compile_counts_only_report_facts(
        records, evidence=evidence, reader_display_labels={"exposure=0": "Reference category"},
        scientific_claims=claims,
    )
    fact = facts[2]
    assert fact.replaces_claim_ref == "distribution.observed_absolute_risk_level_0"
    token = "{claim:" + fact.replaces_claim_ref + "}"
    text = (
        f"## Abstract\n\n**Results:**\n{token}\n\n## Results\n\n"
        f"### Cohort characteristics\n\n### Primary outcome\n\n{fact.scaffold}\n\n{token}\n\n"
        "{claim:other.observed_absolute_risk_level_0}\n\n## Conclusion\n\n" + token
    )
    assert token in place_descriptive_report_facts(text, facts)
    rendered = render_descriptive_report_claims(text, facts)
    results = rendered.split("## Results")[1].split("## Conclusion")[0]
    assert results.count(fact.scaffold) == 1
    assert "{claim:other.observed_absolute_risk_level_0}" in rendered
    assert token not in rendered
    assert rendered.count(fact.scaffold) == 3  # abstract, results, conclusion
    assert render_descriptive_report_claims(rendered, facts) == rendered


def test_descriptive_source_contract_is_restored_from_sealed_json_not_mutable_projection(tmp_path):
    import json
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.reporting.registered_report_inputs import (
        ReadOnlyReportEvidence, verified_descriptive_source_records,
    )

    records, _ = _inputs()
    source_path = tmp_path / "original.json"
    source_path.write_text(json.dumps(records[0]["step_summary"]))
    root = tmp_path / "run"
    store = EvidenceStore(root)
    record = store.register_file(
        kind="statistic", source_path=source_path, description="Source summary",
        evidence_id="summary", produced_by_step="distribution",
    )
    projected = deepcopy(records)
    del projected[0]["step_summary"]["interpretation_class"]
    projected[0]["step_summary"]["cohort_n"] = 99999
    restored = verified_descriptive_source_records(projected, ReadOnlyReportEvidence(root))
    assert restored[0]["step_summary"] == records[0]["step_summary"]
    assert len(compile_counts_only_report_facts(restored, evidence=ReadOnlyReportEvidence(root), reader_display_labels={})) == 4
    assert projected[0]["step_summary"]["cohort_n"] == 99999
    (root / record.relative_path).write_text('{"cohort_n":99999}')
    with pytest.raises(RuntimeError, match="digest verification"):
        verified_descriptive_source_records(projected, ReadOnlyReportEvidence(root))


def test_empty_optional_headings_are_not_empty_required_results():
    text = (
        "## Results\n\n### Cohort characteristics\n\n### Primary outcome\n\n"
        "### ICU-specific quality control\n\n## Discussion\n\nA preserved sentence."
    )
    result = remove_empty_optional_subsections(text)
    assert "### Primary outcome" in result
    assert "### Cohort characteristics" in result
    assert "### ICU-specific quality control" not in result
    assert result.endswith("## Discussion\n\nA preserved sentence.")
    nonempty = text.replace("### ICU-specific quality control", "### ICU-specific quality control\n\nExisting evidence.")
    assert remove_empty_optional_subsections(nonempty) == nonempty
