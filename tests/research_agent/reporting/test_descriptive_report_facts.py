from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from easyicu.research_agent.reporting.descriptive_report_facts import (
    compile_counts_only_report_facts,
    place_descriptive_report_facts,
    render_descriptive_report_claims,
    missing_primary_result_facts,
    place_primary_result_summaries,
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
    assert facts[0].cohort_n == 100
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


def test_modern_descriptive_results_contain_all_registered_primary_facts_once():
    records, evidence = _inputs()
    facts = compile_counts_only_report_facts(records, evidence=evidence, reader_display_labels={})
    manuscript = (
        "## Results\n\n### Cohort characteristics\nRecorded cohort.\n\n"
        "### Descriptive results\nSee Figure 1.\n\n"
        "## Discussion\nExisting discussion remains unchanged."
    )
    placed = place_descriptive_report_facts(manuscript, facts)
    primary = placed.split("### Descriptive results", 1)[1].split("## Discussion", 1)[0]
    assert all(primary.count(fact.scaffold) == 1 for fact in facts)
    assert place_descriptive_report_facts(placed, facts) == placed
    assert placed.endswith("## Discussion\nExisting discussion remains unchanged.")


def test_modern_report_keeps_a_source_bound_cohort_count_after_claim_filtering():
    from easyicu.research_agent.authority.manuscript_claim_policy import filter_evidence_bound_scaffold
    from easyicu.research_agent.reporting.manuscript_quality import audit_manuscript_quality, repair_registered_display_callouts
    from .test_plan_driven_result_structure import _plan

    records, evidence = _inputs()
    facts = compile_counts_only_report_facts(records, evidence=evidence, reader_display_labels={})
    draft = "## Results\n\n### Cohort characteristics\n\n### Descriptive results\n"
    canonical = filter_evidence_bound_scaffold(
        draft, resolve_claim=lambda _ref: None, resolve_evidence=lambda _ref: True,
    ).scaffold
    canonical = place_descriptive_report_facts(canonical, facts)
    canonical, _ = repair_registered_display_callouts(canonical, expected_display_labels=("Table 1", "Figure 1"))
    assert "The analysis cohort comprised 100 observations {evidence:summary}." in canonical.split("### Descriptive results")[0]
    assert not [finding for finding in audit_manuscript_quality(canonical, analysis_plan=_plan()).findings
                if finding.section == "Results"]
    assert place_descriptive_report_facts(canonical, facts) == canonical


def test_different_recorded_cohorts_do_not_become_one_cohort_count():
    records, evidence = _inputs()
    facts = compile_counts_only_report_facts(records, evidence=evidence, reader_display_labels={})
    mixed = (replace(facts[0], cohort_n=200), *facts[1:])
    draft = "## Results\n\n### Cohort characteristics\n\n### Descriptive results\n"
    placed = place_descriptive_report_facts(draft, mixed)
    assert "analysis cohort comprised" not in placed


@pytest.mark.parametrize("invalid", (0, True))
def test_invalid_recorded_cohort_count_is_not_rendered(invalid):
    records, evidence = _inputs()
    facts = compile_counts_only_report_facts(records, evidence=evidence, reader_display_labels={})
    invalid_facts = (replace(facts[0], cohort_n=invalid), *facts[1:])
    with pytest.raises(ValueError, match="recorded integer count"):
        place_descriptive_report_facts("## Results\n\n### Descriptive results\n", invalid_facts)


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
    assert "descriptive, unadjusted, noncausal" in rendered.split("## Conclusion")[1]
    assert "descriptive, unadjusted, noncausal" not in results
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


def test_primary_summary_coverage_cannot_borrow_another_metric_or_section():
    from easyicu.research_agent.reporting.manuscript_quality import audit_manuscript_quality
    from easyicu.research_agent.reporting.manuscript_sections import quality_repair_section_errors

    records, evidence = _inputs()
    facts = compile_counts_only_report_facts(records, evidence=evidence, reader_display_labels={})
    risks = "\n\n".join(fact.scaffold for fact in facts[2:])
    text = (
        f"## Abstract\n\n**Results:**\n\n{risks}\n\n**Conclusions:**\nIndependent validation is required.\n\n"
        "## Results\n\n### Cohort characteristics\n\n### Primary outcome\n\n"
        "## Discussion\n\nA preserved limitation, not a result.\n\n## Conclusion\n\n" + risks
    )
    text = place_descriptive_report_facts(text, facts)
    missing = missing_primary_result_facts(text, facts)
    assert missing["Abstract"] == facts[:2]
    assert "Conclusion" not in missing
    assert "Discussion" not in missing
    assert "Results" not in missing
    audit = audit_manuscript_quality(text, expected_primary_result_facts=facts)
    failures = [f for f in audit.findings if f.code == "MANUSCRIPT_PRIMARY_RESULT_COVERAGE_INCOMPLETE"]
    assert {f.section for f in failures} == {"Abstract"}
    assert all(f.severity == "error" for f in failures)
    owners = quality_repair_section_errors(text, expected_primary_result_facts=facts)
    assert "MANUSCRIPT_PRIMARY_RESULT_COVERAGE_INCOMPLETE" in str(owners["abstract"])
    repaired = place_primary_result_summaries(text, facts)
    assert missing_primary_result_facts(repaired, facts) == {}
    assert place_primary_result_summaries(repaired, facts) == repaired
    assert "A preserved limitation, not a result." in repaired
    assert all(repaired.count(f.scaffold) == 2 for f in facts[:2])
    assert all(repaired.count(f.scaffold) == 3 for f in facts[2:])
    assert repaired.split("## Discussion")[1] == text.split("## Discussion")[1]
    assert "confidence interval" not in repaired and "risk difference" not in repaired


def test_numeric_result_projection_does_not_write_the_interpretation_sections():
    records, evidence = _inputs()
    facts = compile_counts_only_report_facts(records, evidence=evidence, reader_display_labels={})
    interpretation = (
        "## Discussion\n\nThe observed distribution does not establish a causal effect.\n\n"
        "## Conclusion\n\nInterpretation is limited to the recorded ICU stays.\n"
    )
    text = (
        "## Abstract\n\n**Results:**\n\n**Conclusions:**\nCaution is needed.\n\n"
        "## Results\n\n### Cohort characteristics\n\n### Primary outcome\n\n"
        + interpretation
    )
    projected = render_descriptive_report_claims(text, facts)
    assert projected[projected.index("## Discussion"):] == interpretation
    assert all(projected.count(fact.scaffold) == 2 for fact in facts)
    assert missing_primary_result_facts(projected, facts) == {}
    assert render_descriptive_report_claims(projected, facts) == projected


def test_hidden_or_wrong_source_metric_text_does_not_satisfy_primary_coverage():
    records, evidence = _inputs()
    facts = compile_counts_only_report_facts(records, evidence=evidence, reader_display_labels={})
    hidden = "\n\n".join(f"<!-- {fact.scaffold} -->" for fact in facts)
    text = f"## Abstract\n\n**Results:**\n\n{hidden}\n\n## Conclusion\n\nUnrelated prose."
    assert missing_primary_result_facts(text, facts)["Abstract"] == facts
    wrong_metric = facts[0].scaffold.replace("Exposure prevalence", "Outcome prevalence")
    assert facts[0] in missing_primary_result_facts(text.replace(hidden, wrong_metric), facts)["Abstract"]
    assert "## Discussion" not in place_primary_result_summaries(text, facts)


def test_primary_facts_remain_covered_after_strict_numeric_binding(tmp_path):
    import json
    from easyicu.research_agent.authority.evidence_store import EvidenceStore, EvidenceEnforcementMode
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    records, _ = _inputs()
    store = EvidenceStore(tmp_path / "run")
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(records[0]["step_summary"]))
    store.register_file(kind="statistic", source_path=source, evidence_id="summary", description="Counts", produced_by_step="distribution")
    store.register_step_summary_numerics(step_id="distribution", evidence_id="summary", summary=records[0]["step_summary"])
    records[0]["evidence_ids"] = ["summary"]
    facts = compile_counts_only_report_facts(records, evidence=store, reader_display_labels={})
    text = "## Abstract\n\n**Results:**\n\n**Conclusions:**\nCaution.\n\n## Results\n\n### Cohort characteristics\n\n### Primary outcome\n\n## Discussion\n\nBoundary.\n\n## Conclusion\n\nCaution."
    projected = render_descriptive_report_claims(text, facts)
    bound, bindings, untraced = bind_numeric_values(projected, evidence=store, enforcement_mode=EvidenceEnforcementMode.STRICT, per_step_records=records)
    assert bindings and not untraced
    assert missing_primary_result_facts(bound, facts) == {}


def test_full_write_boundary_projects_only_after_model_grammar_and_preserves_claim_coverage(tmp_path, monkeypatch):
    import json
    from easyicu.research_agent.authority.evidence_store import EvidenceStore, EvidenceEnforcementMode
    from easyicu.research_agent.reporting import write_phase
    from easyicu.research_agent.schema import CritiqueReport

    records, _ = _inputs()
    root = tmp_path / "run"
    store = EvidenceStore(root, enforcement_mode=EvidenceEnforcementMode.STRICT)
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(records[0]["step_summary"]))
    store.register_file(kind="statistic", source_path=source, evidence_id="summary", description="Counts", produced_by_step="distribution", generation_mode="deterministic_standard")
    store.register_step_summary_numerics(step_id="distribution", evidence_id="summary", summary=records[0]["step_summary"])
    claims = store.register_step_summary_scientific_claims(step_id="distribution", evidence_id="summary", summary=records[0]["step_summary"])
    assert len(claims) == 2
    records[0]["evidence_ids"] = ["summary"]
    labels = {"exposure=0": "未记录暴露", "exposure=1": "记录到暴露", "outcome": "院内死亡"}
    facts = compile_counts_only_report_facts(records, evidence=store, reader_display_labels=labels, scientific_claims=claims)
    store.register_text(
        kind="table", text="group,n\n0,20\n1,10\n", filename="table_one.csv",
        evidence_id="table_one", description="Registered baseline table",
    )
    # Envelope admission has independent contracts and the real frozen-run
    # replay; this test exercises the downstream ordering against a real store.
    monkeypatch.setattr(write_phase, "compile_primary_counts_only_report_facts", lambda *args, **kwargs: facts)
    tokens = "\n\n".join("{claim:" + claim.claim_ref + "}" for claim in claims)
    scaffold = "# Draft\n\n## Abstract\n\n**Background:** Context for the analysis is described here.\n\n**Methods:** The prespecified analysis was performed.\n\n**Results:**\n\n**Conclusions:**\nIndependent validation is required.\n\n## Results\n\n### Cohort characteristics\n\n### Primary outcome\n\n" + tokens + "\n\n## Discussion\n\nIndependent validation is required.\n\n## Conclusion\n\nIndependent validation is required."
    findings = []
    output = write_phase._bind_and_review_manuscript(
        SimpleNamespace(_evidence_enforcement_mode=EvidenceEnforcementMode.STRICT),
        critic=SimpleNamespace(review_manuscript=lambda **kwargs: CritiqueReport(reviewer="test fixture", status="blocked")),
        evidence=store, findings=findings, literature=None, per_step_records=records,
        current_evidence_names=["summary", "table_one"], scaffold=scaffold, writer_error_message=None,
        writer_probe_mode=False, writer_probe_failed_steps=(), run_dir=root,
        reader_display_labels=labels, manuscript_language="en",
    )
    assert missing_primary_result_facts(output.bound, facts) == {}
    assert "See Table 1" in output.bound
    assert "[^claim_" in output.bound
    assert not any(f.validator == "manuscript_result_sufficiency" for f in findings)
    assert not any(f.validator == "manuscript_numeric_auditor" and f.severity == "error" for f in findings)


@pytest.mark.parametrize('heading', ['## Conclusion', '## Discussion', '## Abstract\n\n**Conclusions:**'])
def test_conclusion_compacts_only_same_endpoint_registered_claims(heading):
    from easyicu.research_agent.authority.scientific_claims import bind_scientific_claim_drafts
    from easyicu.research_agent.authority.scientific_claims import derive_scientific_claim_drafts
    records, evidence = _inputs()
    claims = bind_scientific_claim_drafts(
        [draft.model_dump(mode='json') for draft in derive_scientific_claim_drafts(records[0]['step_summary'])],
        step_id='distribution', evidence_id='summary',
    )
    facts = compile_counts_only_report_facts(records, evidence=evidence,
        reader_display_labels={'exposure=0': 'Reference category', 'exposure=1': 'Other category'}, scientific_claims=claims)
    risks = facts[2:]
    text = heading + '\n\n' + '\n\n'.join('{claim:' + fact.replaces_claim_ref + '}' for fact in risks)
    result = render_descriptive_report_claims(text, facts)
    assert '10.00% in the “Reference category” group; 20.00% in the “Other category” group' in result
    assert '6 of 60' not in result and '8 of 40' not in result
    assert 'adjusted or causal effect' in result and '{evidence:summary}' in result
    assert render_descriptive_report_claims(result, facts) == result
    mixed = [*facts[:3], replace(facts[3], outcome_label='different endpoint')]
    assert 'proportions were' not in render_descriptive_report_claims(text, mixed)


def test_prior_revision_binding_cache_is_rebuilt_without_trusting_changed_counts():
    from easyicu.research_agent.reporting.descriptive_report_facts import DescriptiveReportFact, restore_descriptive_revision_claims
    fact = DescriptiveReportFact('Primary outcome', 'Observed endpoint was 6 of 60 (10.00%)', 'summary', 'a' * 64, (), 'distribution.risk')
    text = '## Conclusion\n\nObserved endpoint was 6[^claim_1] of 60 (10.00%) {evidence:summary}. This was a descriptive, unadjusted, noncausal estimate.\n\n[^claim_1]: value=6; evidence=summary'
    restored = restore_descriptive_revision_claims(text, [fact])
    assert '{claim:distribution.risk}' in restored
    assert '[^claim_' not in restored
    changed = restore_descriptive_revision_claims(text.replace('6[^claim_1]', '7[^claim_1]'), [fact])
    assert '{claim:distribution.risk}' not in changed
    assert '7 of 60' in changed  # subject to the normal authority filter and strict numeric binding
