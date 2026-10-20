"""Synthetic aggregate inputs shared by descriptive writer tests."""

import json
from types import SimpleNamespace

from easyicu.research_agent.authority.evidence_store import EvidenceEnforcementMode, EvidenceStore

DRAFT = (
    "## Abstract\n\n**Results:**\n\n**Conclusions:**\n\n"
    "## Results\n\n### Cohort characteristics\n\n### Descriptive results\n\n"
    "### Secondary analyses\n\n## Discussion\n\n## Conclusion\n\n"
)
LABELS = {"exposure=0": "Afebrile", "exposure=1": "Febrile", "outcome": "Readmission",
          "recovery": "Recovery time", "support": "Support duration"}


def _observed():
    def row(level, count, denominator, count_field):
        return {
            "level_index": level, "level": level, count_field: count, "denominator": denominator,
            "estimate_pct": 100 * count / denominator, "interval_method": "none_counts_only",
            "covariance": "none_counts_only", "ci_low_pct": None, "ci_high_pct": None,
            "confidence_level": None, "standard_error_pct": None, "cluster_count": None,
        }

    shared = {"analysis_set": "exposure_observed_rows_within_bound_typed_cohort",
              "analysis_role": "primary", "interpretation_ceiling": "descriptive_unadjusted_not_causal"}
    summary = {
        **shared, "status": "ok", "interpretation_class": "exposure_outcome_distribution",
        "interval_method": "none_counts_only", "cohort_n": 100, "exposure": "exposure",
        "outcome": "outcome", "adjusted_effect": None,
        "descriptive_estimates": {
            **shared, "schema_version": "easyicu.exposure_outcome_descriptive_estimates/1",
            "exposure_prevalence": [row(0, 60, 100, "n"), row(1, 40, 100, "n")],
            "outcome_absolute_risks": [row(0, 6, 60, "events"), row(1, 8, 40, "events")],
            "risk_difference": None, "dependence": None,
        },
        "source_row_count_reconciliation": {
            "source_rows": 107, "analyzed_rows": 100,
            "excluded_missing_exposure_rows": 7, "filtering_performed": True,
        },
    }
    records = [{"step_id": "distribution", "status": "ok", "step_summary": summary,
                "step_summary_evidence_id": "summary"}]
    source = SimpleNamespace(evidence_id="summary", produced_by_step="distribution", sha256="a" * 64)
    evidence = SimpleNamespace(get=lambda key: source if key == "summary" else None)
    return records, evidence


def _store(tmp_path, *, grouping_variable="exposure"):
    records, _ = _observed()
    store = EvidenceStore(tmp_path / "evidence_run", enforcement_mode=EvidenceEnforcementMode.STRICT)
    summary = records[0]["step_summary"]
    store.register_text(kind="statistic", text=json.dumps(summary), filename="summary.json",
                        evidence_id="summary", description="Synthetic distribution",
                        produced_by_step="distribution", generation_mode="deterministic_standard")
    store.register_step_summary_numerics(step_id="distribution", evidence_id="summary", summary=summary)
    store.register_step_summary_scientific_claims(step_id="distribution", evidence_id="summary", summary=summary)
    records[0]["evidence_ids"] = ["summary"]
    continuous = {
        "status": "ok", "interpretation_class": "continuous_outcome_summary",
        "analysis_role": "secondary", "analysis_set": "bound_typed_cohort",
        "interpretation_ceiling": "descriptive_unadjusted_not_causal",
        "adjusted_effect": None, "uncertainty_interval": None, "inferential_tests": [],
        "grouping_variable": grouping_variable, "outcome_variables": ["recovery", "support"],
        "output_files": {"table:continuous_outcome_summary": "continuous.csv"},
    }
    store.register_text(kind="statistic", text=json.dumps(continuous), filename="continuous.json",
                        evidence_id="continuous_summary", description="Synthetic continuous outcomes",
                        produced_by_step="secondary", generation_mode="deterministic_standard")
    store.register_text(
        kind="table", filename="continuous.csv", evidence_id="continuous_table",
        description="Synthetic grouped durations", produced_by_step="secondary",
        text="row_role,variable,group,group_n,n_nonmissing,missing_n,median,q25,q75,unit\n"
             "exposure_level,recovery,0,60,58,2,4.25,2.5,8.125,hours\n"
             "exposure_level,recovery,1,40,40,0,6.5,3.75,9.25,hours\n"
             "exposure_level,support,0,60,60,0,12.5,6.25,18.75,hours\n"
             "exposure_level,support,1,40,39,1,18.25,9.125,24.5,hours\n",
    )
    records.append({
        "step_id": "secondary", "status": "ok", "step_summary": continuous,
        "step_summary_evidence_id": "continuous_summary",
        "evidence_ids": ["continuous_summary", "continuous_table"],
    })
    return store, records


def _plan():
    return SimpleNamespace(analysis_type="descriptive_epidemiology", steps=[
        SimpleNamespace(
            step_id="distribution", expected_outputs=["table:exposure_outcome_distribution"],
            exposure_outcome_distribution_spec=SimpleNamespace(exposure="exposure", outcome="outcome"),
            method="descriptive", planned_analysis_role="primary",
            inputs=["artifact:analysis_cohort", "exposure", "outcome"],
        ),
        SimpleNamespace(
            step_id="secondary", expected_outputs=["table:continuous_outcome_summary"],
            exposure_outcome_distribution_spec=None, method="continuous_outcome_summary",
            planned_analysis_role="secondary",
            inputs=["artifact:analysis_cohort", "exposure", "recovery", "support"],
        ),
    ])

