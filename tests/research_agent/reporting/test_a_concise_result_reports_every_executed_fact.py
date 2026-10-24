"""A small, synthetic distribution needs complete reporting, not padding."""

import json

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.reporting.descriptive_report_facts import compile_counts_only_report_facts
from easyicu.research_agent.reporting.scientific_maturity import _manuscript_facts
from easyicu.research_agent.reporting.readiness import write_readiness_artifacts
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, CohortDescriptor, ResearchContext, ValidationFinding


def _row(level, n, denominator, *, events=False):
    return {
        "level_index": level, "level": level, "events" if events else "n": n,
        "denominator": denominator, "estimate_pct": 100 * n / denominator,
        "interval_method": "none_counts_only", "covariance": "none_counts_only",
        "ci_low_pct": None, "ci_high_pct": None, "confidence_level": None,
        "standard_error_pct": None, "cluster_count": None,
    }


@pytest.fixture
def result(tmp_path):
    summary = {
        "status": "ok", "interpretation_class": "exposure_outcome_distribution",
        "analysis_role": "primary", "analysis_set": "bound_typed_cohort",
        "interpretation_ceiling": "descriptive_unadjusted_not_causal",
        "adjusted_effect": None, "interval_method": "none_counts_only",
        "cohort_n": 12, "exposure": "group", "outcome": "event",
        "descriptive_estimates": {
            "schema_version": "easyicu.exposure_outcome_descriptive_estimates/1",
            "analysis_role": "primary", "analysis_set": "bound_typed_cohort",
            "interpretation_ceiling": "descriptive_unadjusted_not_causal",
            "exposure_prevalence": [_row(0, 8, 12), _row(1, 4, 12)],
            "outcome_absolute_risks": [_row(0, 2, 8, events=True), _row(1, 1, 4, events=True)],
            "risk_difference": None, "dependence": None,
        },
    }
    evidence = EvidenceStore(tmp_path, enforcement_mode="strict")
    source = evidence.register_json(
        kind="statistic", description="Synthetic aggregate distribution", payload=summary,
        filename="summary.json", evidence_id="summary", produced_by_step="distribution",
        generation_mode="deterministic_standard",
    )
    evidence.register_step_summary_numerics(step_id="distribution", evidence_id="summary", summary=summary)
    records = [{
        "step_id": "distribution", "status": "ok", "generation_mode": "deterministic_standard",
        "step_summary": summary, "step_summary_evidence_id": "summary", "evidence_ids": ["summary"],
    }]
    facts = compile_counts_only_report_facts(
        records, evidence=evidence, reader_display_labels={},
        scientific_claims=evidence.authoritative_scientific_claims(records),
    )
    sentences = list(dict.fromkeys(fact.text for fact in facts))
    manuscript = "## Results\n\n" + "\n\n".join(sentences) + "\n\n## Discussion\nDescriptive results.\n"
    (tmp_path / "manuscript_scaffold_bound.md").write_text(manuscript)
    return evidence, records, source, sentences, manuscript


def test_a_complete_single_distribution_does_not_need_two_hundred_words(tmp_path, result):
    evidence, records, _, _, _ = result
    facts = _manuscript_facts(tmp_path, evidence=evidence, per_step_records=records)
    assert facts["section_word_counts"]["results"] < 200
    assert "results" not in facts["thin_sections"]
    assert facts["section_word_floors"]["results"] > 0
    assert "methods" in facts["thin_sections"]


def test_the_floor_reads_the_registered_summary_instead_of_a_wrapper_copy(tmp_path, result):
    evidence, records, _, _, _ = result
    records[0]["step_summary"] = {**records[0]["step_summary"], "cohort_n": 13}
    facts = _manuscript_facts(tmp_path, evidence=evidence, per_step_records=records)
    assert "results" not in facts["thin_sections"]


@pytest.mark.parametrize("defect", ["missing_fact", "wrong_number", "hidden", "stale_source", "failed_step", "no_evidence", "extra_result"])
def test_a_short_result_with_missing_or_unverified_substance_keeps_the_floor(tmp_path, result, defect):
    evidence, records, source, sentences, manuscript = result
    if defect == "missing_fact":
        manuscript = manuscript.replace(sentences[-1], "")
    elif defect == "wrong_number":
        manuscript = manuscript.replace("12", "13")
    elif defect == "hidden":
        manuscript = manuscript.replace(sentences[-1], f"<!-- {sentences[-1]} -->")
    elif defect == "stale_source":
        (tmp_path / source.relative_path).write_text("{}")
    elif defect == "failed_step":
        records[0]["status"] = "failed"
    elif defect == "no_evidence":
        evidence = None
    elif defect == "extra_result":
        records.append({"step_id": "model", "status": "ok", "step_summary": {"analysis_role": "secondary"}})
    (tmp_path / "manuscript_scaffold_bound.md").write_text(manuscript)
    facts = _manuscript_facts(tmp_path, evidence=evidence, per_step_records=records)
    assert "results" in facts["thin_sections"]
    assert facts["section_word_floors"]["results"] == 200


def test_readiness_persists_a_failed_draft_beside_the_completed_descriptive_analysis(tmp_path, result):
    evidence, records, _, _, _ = result
    context = ResearchContext(
        research_question="Describe group shares and event proportions.",
        cohort=CohortDescriptor(cohort_name="synthetic", database="synthetic", n_patients=12, n_stays=12),
        variables=[],
    )
    plan = AnalysisPlan(
        research_question=context.research_question, analysis_type="descriptive_epidemiology",
        steps=[AnalysisStep(
            step_id="distribution", intent="Describe the aggregate", method="descriptive",
            planned_analysis_role="primary", inputs=["cohort:synthetic", "group", "event"],
            expected_outputs=["table:exposure_outcome_distribution"],
            descriptive_claim={"unresolved_limitations": ["counts_only_design"]},
            exposure_outcome_distribution_spec={
                "schema_version": "easyicu.exposure_outcome_distribution/3",
                "exposure": "group", "exposure_levels": [0, 1], "outcome": "event", "outcome_levels": [0, 1],
                "outcome_positive_value": 1, "level_match_policy": "exact_typed",
                "denominator_policy": "all_declared_rows", "missing_outcome_policy": "fail_closed",
                "interval_method": "none_counts_only", "repeated_unit_interval_method": None,
                "confidence_level": None,
            },
        )],
    )
    manuscript = tmp_path / "manuscript_scaffold_bound.md"
    manuscript.write_text("## Results\nThe synthetic aggregate is reported.\n")
    gates, _ = write_readiness_artifacts(
        context=context, plan=plan,
        findings=[ValidationFinding(validator="manuscript_quality", severity="error", message="Synthetic draft lacks substance.")],
        per_step_records=records, evidence=evidence, run_dir=tmp_path, manuscript_path=manuscript,
        stop_after_analysis=False,
    )
    persisted = json.loads((tmp_path / "run_status.json").read_text())
    assert gates["analysis_outcome"]["status"] == "completed", gates["analysis_errors"]
    assert gates["manuscript_outcome"] == {"status": "not_passed", "reason_code": "manuscript_checks_not_passed"}
    assert persisted["analysis_outcome"] == gates["analysis_outcome"]
    assert persisted["manuscript_outcome"] == gates["manuscript_outcome"]
    assert persisted["gates"] == gates
    assert gates["manuscript_ready"] is False
    assert gates["publication_ready"] is False
    assert gates["paper_authorized"] is False
