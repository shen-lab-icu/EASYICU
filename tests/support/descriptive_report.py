"""Synthetic source-bound descriptive report facts."""
from types import SimpleNamespace


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
