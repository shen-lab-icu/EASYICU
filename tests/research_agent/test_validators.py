"""Validator tests — the core of the agent's safety story.

These tests are intentionally adversarial: each one feeds the
validator something that *should* trip an ICU rule, and asserts the
right finding pops out at the right severity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.audits.validators import (
    CrossStepCohortLockValidator,
    CrossStepReconciliationTraceValidator,
    CrossStepRegisteredOutputValidator,
    CrossStepSourceStatusValidator,
    FigureSourceDataValidator,
    StepSummaryFractionValidator,
)
from easyicu.research_agent.schema import AnalysisStep


def _prior_source_status_record(*, valid_n: int = 41_210, total_n: int = 94_458):
    return {
        "step_id": "02_exposure_and_missingness_audit",
        "step_summary": {
            "missingness": {
                "source_status_counts": {
                    "adult_analytic_cohort": {
                        "lab_max": {
                            "valid_observed_level_or_value": valid_n,
                            "no_recorded_source_or_observation": total_n - valid_n,
                            "measured_or_observed_source_with_summary_missing": 0,
                            "contradictory_or_invalid_source_summary": 0,
                        }
                    }
                }
            }
        },
    }


def _current_table_one_status(*, valid_n: int, total_n: int = 94_458):
    return {
        "measurement_status": {
            "source_summary": "lab_max",
            "counts": [
                {"category": "Observed valid", "count": valid_n},
                {"category": "No source", "count": total_n - valid_n},
                {"category": "Measured but summary missing", "count": 0},
                {"category": "Invalid summary", "count": 0},
                {"category": "Contradictory status", "count": 0},
            ],
        }
    }


def _prior_cohort_record(
    *, cohort_n: int = 94_458, step_id: str = "04_absolute_risk_context"
):
    return {
        "step_id": step_id,
        "status": "ok",
        "step_summary": {"status": "completed", "n_total": cohort_n},
    }


def test_cross_step_cohort_lock_blocks_fixed_cohort_drift() -> None:
    findings = CrossStepCohortLockValidator().audit(
        step=AnalysisStep(
            step_id="04_absolute_risk_context_reconciliation",
            intent="Keep the completed cohort, outcome, and window fixed.",
        ),
        step_summary={"n_universe": 94_458, "n_final_cohort": 74_829},
        completed_step_records=[_prior_cohort_record()],
    )

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].validator == "cross_step_cohort_lock"
    assert findings[0].detail["reported_cohort_n"] == 74_829
    assert findings[0].detail["expected_cohort_n"] == 94_458
    assert findings[0].detail["reported_summary_path"] == "n_final_cohort"


def test_cross_step_cohort_lock_accepts_unchanged_cohort() -> None:
    findings = CrossStepCohortLockValidator().audit(
        step=AnalysisStep(
            step_id="04_reconciliation",
            intent="Keep the completed cohort fixed while reconciling the table.",
        ),
        step_summary={"n_final_cohort": 94_458},
        completed_step_records=[_prior_cohort_record()],
    )

    assert findings == []


def test_cross_step_cohort_lock_accepts_nested_final_rows_schema() -> None:
    findings = CrossStepCohortLockValidator().audit(
        step=AnalysisStep(
            step_id="04_reconciliation",
            intent="Keep the completed cohort fixed while reconciling the table.",
        ),
        step_summary={"cohort": {"n_input_rows": 94_458, "n_final_rows": 94_458}},
        completed_step_records=[_prior_cohort_record()],
    )

    assert findings == []


def test_cross_step_cohort_lock_accepts_locked_cohort_output_schema() -> None:
    findings = CrossStepCohortLockValidator().audit(
        step=AnalysisStep(
            step_id="04_reconciliation",
            intent="Keep the completed cohort fixed while reconciling the table.",
        ),
        step_summary={"locked_cohort": {"n_input": 94_458, "n_output": 94_458}},
        completed_step_records=[_prior_cohort_record()],
    )

    assert findings == []


def test_cross_step_cohort_lock_accepts_cohort_count_final_schema() -> None:
    findings = CrossStepCohortLockValidator().audit(
        step=AnalysisStep(
            step_id="04_reconciliation",
            intent="Keep the completed cohort fixed while reconciling the table.",
        ),
        step_summary={"cohort_count_final": 94_458},
        completed_step_records=[_prior_cohort_record()],
    )

    assert findings == []


def test_cross_step_cohort_lock_accepts_final_cohort_n_schema() -> None:
    findings = CrossStepCohortLockValidator().audit(
        step=AnalysisStep(
            step_id="04_reconciliation",
            intent="Keep the completed cohort fixed while reconciling the table.",
        ),
        step_summary={"final_cohort_n": 94_458},
        completed_step_records=[_prior_cohort_record()],
    )

    assert findings == []


def test_cross_step_cohort_lock_requires_explicit_lock_intent() -> None:
    findings = CrossStepCohortLockValidator().audit(
        step=AnalysisStep(
            step_id="04_descriptive",
            intent="Describe absolute risks in the available analysis rows.",
        ),
        step_summary={"n_final_cohort": 74_829},
        completed_step_records=[_prior_cohort_record()],
    )

    assert findings == []


def test_cross_step_cohort_lock_skips_explicit_alternative_cohort() -> None:
    findings = CrossStepCohortLockValidator().audit(
        step=AnalysisStep(
            step_id="06_cohort_definition_sensitivity",
            intent="Compare an alternative cohort definition using LOS eligibility.",
            method="cohort_definition_sensitivity",
        ),
        step_summary={"n_final_cohort": 74_829},
        completed_step_records=[_prior_cohort_record()],
    )

    assert findings == []


def test_cross_step_cohort_lock_uses_latest_successful_analysis_lock() -> None:
    failed = _prior_cohort_record(
        cohort_n=74_829, step_id="04_failed_reconciliation"
    )
    failed["status"] = "contract_failed"
    figure = _prior_cohort_record(cohort_n=3, step_id="04_absolute_risk_figure")
    figure["step_summary"]["rendering_only"] = True

    findings = CrossStepCohortLockValidator().audit(
        step=AnalysisStep(
            step_id="05_reconciliation",
            intent="Preserve the analytic cohort unchanged.",
        ),
        step_summary={"n_final_cohort": 94_457},
        completed_step_records=[_prior_cohort_record(), failed, figure],
    )

    assert len(findings) == 1
    assert findings[0].detail["expected_cohort_n"] == 94_458
    assert findings[0].detail["expected_from_step"] == "04_absolute_risk_context"


def test_cross_step_cohort_lock_requires_current_machine_readable_count() -> None:
    findings = CrossStepCohortLockValidator().audit(
        step=AnalysisStep(
            step_id="04_reconciliation",
            intent="Keep the completed cohort fixed.",
        ),
        step_summary={"n_universe": 94_458},
        completed_step_records=[_prior_cohort_record()],
    )

    assert len(findings) == 1
    assert findings[0].detail["reported_summary_path"] is None
    assert findings[0].detail["expected_cohort_n"] == 94_458


def test_cross_step_cohort_lock_skips_rendering_only_figure_step() -> None:
    findings = CrossStepCohortLockValidator().audit(
        step=AnalysisStep(
            step_id="05_primary_association_figure",
            intent=(
                "Render the registered parent outputs; do not redefine the cohort, "
                "exposure, outcome, or model."
            ),
        ),
        step_summary={
            "rendering_only": True,
            "figure_files": ["publication_figure.png", "publication_figure.svg"],
            "source_step_id": "05_primary_association",
        },
        completed_step_records=[_prior_cohort_record()],
    )

    assert findings == []


def _prior_registered_table_record():
    return {
        "step_id": "04_absolute_risk_context",
        "status": "ok",
        "evidence_ids": ["table_exposure_outcome_summary_8368e5ab"],
        "step_summary": {
            "status": "ok",
            "output_files": {
                "exposure_outcome_summary": "exposure_outcome_summary.csv"
            },
        },
    }


def test_cross_step_registered_output_blocks_false_unavailable_gap() -> None:
    findings = CrossStepRegisteredOutputValidator().audit(
        step=AnalysisStep(
            step_id="04_reconciliation",
            intent="Audit the registered outputs of the prior risk step.",
        ),
        step_summary={
            "registered_output": {
                "upstream_step": "04_absolute_risk_context",
                "source_table_available": False,
                "source_table_path": None,
            }
        },
        completed_step_records=[_prior_registered_table_record()],
    )

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].validator == "cross_step_registered_output"
    assert findings[0].detail["upstream_step"] == "04_absolute_risk_context"
    assert "exposure_outcome_summary.csv" in findings[0].detail[
        "registered_table_artifacts"
    ]


def test_cross_step_registered_output_accepts_readable_parent_table() -> None:
    findings = CrossStepRegisteredOutputValidator().audit(
        step=AnalysisStep(
            step_id="04_reconciliation",
            intent="Audit the registered outputs of the prior risk step.",
        ),
        step_summary={
            "registered_output": {
                "upstream_step": "04_absolute_risk_context",
                "source_table_available": True,
                "source_table_path": "exposure_outcome_summary.csv",
            }
        },
        completed_step_records=[_prior_registered_table_record()],
    )

    assert findings == []


def test_cross_step_registered_output_allows_genuine_missing_parent_table() -> None:
    prior = _prior_registered_table_record()
    prior["evidence_ids"] = ["statistic_step_summary_12345678"]
    prior["step_summary"].pop("output_files")

    findings = CrossStepRegisteredOutputValidator().audit(
        step=AnalysisStep(
            step_id="04_reconciliation",
            intent="Audit the registered outputs of the prior risk step.",
        ),
        step_summary={
            "registered_output": {
                "upstream_step": "04_absolute_risk_context",
                "source_table_available": False,
            }
        },
        completed_step_records=[prior],
    )

    assert findings == []


def test_step_summary_fraction_scale_rejects_percentage_in_fraction_field() -> None:
    findings = StepSummaryFractionValidator().audit(
        step=AnalysisStep(step_id="04_reconciliation", intent="Audit missingness."),
        step_summary={
            "missingness": {
                "missing_fraction": {"lab_max": 56.372144},
                "missing_pct": {"lab_max": 56.372144},
            }
        },
    )

    assert len(findings) == 1
    assert findings[0].validator == "step_summary_fraction_scale"
    assert findings[0].detail["summary_path"] == (
        "missingness.missing_fraction.lab_max"
    )
    assert findings[0].detail["reported_value"] == 56.372144


def test_step_summary_fraction_scale_accepts_fraction_and_percent_units() -> None:
    findings = StepSummaryFractionValidator().audit(
        step=AnalysisStep(step_id="04_reconciliation", intent="Audit missingness."),
        step_summary={
            "missingness": {
                "missing_fraction": {"lab_max": 0.56372144},
                "missing_pct": {"lab_max": 56.372144},
            }
        },
    )

    assert findings == []


def test_step_summary_fraction_scale_does_not_bound_counts_in_metric_record() -> None:
    findings = StepSummaryFractionValidator().audit(
        step=AnalysisStep(step_id="04_incidence", intent="Report absolute risk."),
        step_summary={
            "overall_outcome_prevalence": {
                "outcome_column": "death",
                "n": 94_458,
                "n_eligible": 94_458,
                "event_n": 9_466,
                "non_event_n": 84_992,
                "risk": 0.1002,
                "ci_low": 0.0983,
                "ci_high": 0.1022,
                "ci_alpha": 0.05,
            }
        },
    )

    assert findings == []


def test_step_summary_fraction_scale_does_not_inherit_into_percent_leaf() -> None:
    findings = StepSummaryFractionValidator().audit(
        step=AnalysisStep(step_id="04_incidence", intent="Report absolute risk."),
        step_summary={
            "overall_outcome_prevalence": {
                "n_events": 9_466,
                "mortality_percentage": 10.02,
                "risk": 0.1002,
            }
        },
    )

    assert findings == []


def test_step_summary_fraction_scale_still_bounds_explicit_metric_in_record() -> None:
    findings = StepSummaryFractionValidator().audit(
        step=AnalysisStep(step_id="04_incidence", intent="Report absolute risk."),
        step_summary={
            "overall_outcome_prevalence": {
                "n": 94_458,
                "event_n": 9_466,
                "non_event_n": 84_992,
                "risk": 10.02,
            }
        },
    )

    assert len(findings) == 1
    assert findings[0].detail["summary_path"] == "overall_outcome_prevalence.risk"


def test_step_summary_fraction_scale_ignores_nested_audit_counts() -> None:
    findings = StepSummaryFractionValidator().audit(
        step=AnalysisStep(step_id="04_audit", intent="Audit numeric coercion."),
        step_summary={
            "statistics": {
                "numeric_coercion_audit": {
                    "binary_risk": {
                        "attempted_n": 94_458,
                        "post_coercion_valid_n": 94_458,
                    }
                }
            }
        },
    )

    assert findings == []


def test_step_summary_fraction_scale_stops_at_named_nested_container() -> None:
    validator = StepSummaryFractionValidator()
    step = AnalysisStep(step_id="04_audit", intent="Report absolute risk.")
    summary = {
        "absolute_risk": {
            "bili_distribution": {"q25": 0.8, "q75": 1.4},
            "outcome_risk": 0.1,
        }
    }

    assert validator.audit(step=step, step_summary=summary) == []
    summary["absolute_risk"]["outcome_risk"] = 10.0
    findings = validator.audit(step=step, step_summary=summary)
    assert len(findings) == 1
    assert findings[0].detail["summary_path"] == "absolute_risk.outcome_risk"


def test_step_summary_fraction_scale_does_not_drop_mixed_category_values() -> None:
    findings = StepSummaryFractionValidator().audit(
        step=AnalysisStep(step_id="04_audit", intent="Audit completeness."),
        step_summary={"observed_fraction": {"group_a": 40.0, "risk": 0.2}},
    )

    assert len(findings) == 1
    assert findings[0].detail["summary_path"] == "observed_fraction.group_a"


@pytest.mark.parametrize(
    ("step_summary", "expected_path"),
    (
        (
            {"observed_fraction": {"group_a": {"value": 40.0}}},
            "observed_fraction.group_a.value",
        ),
        (
            {
                "overall_risk": {
                    "interval": {"estimate": 10.0, "ci_low": 9.0, "ci_high": 11.0}
                }
            },
            "overall_risk.interval.estimate",
        ),
        (
            {
                "observed_fraction": {
                    "by_group": [{"group": "a", "value": 40.0}]
                }
            },
            "observed_fraction.by_group.0.value",
        ),
    ),
)
def test_step_summary_fraction_scale_follows_explicit_metric_wrappers(
    step_summary: dict[str, object], expected_path: str
) -> None:
    findings = StepSummaryFractionValidator().audit(
        step=AnalysisStep(step_id="04_audit", intent="Audit bounded metrics."),
        step_summary=step_summary,
    )

    assert findings
    assert findings[0].detail["summary_path"] == expected_path


def test_step_summary_fraction_scale_ignores_structured_record_metadata() -> None:
    findings = StepSummaryFractionValidator().audit(
        step=AnalysisStep(step_id="04_model", intent="Report absolute risk."),
        step_summary={
            "overall_risk": {"risk": 0.2, "bootstrap_replicates": 1_000}
        },
    )

    assert findings == []


@pytest.mark.parametrize(
    "step_summary",
    (
        {"observed_fraction": {"by_group": [0.2, 40.0]}},
        {"observed_fraction": {"point_estimate": 40.0}},
        {"observed_fraction": {"result": 40.0}},
        {"observed_fraction": {"estimates": {"group_a": 0.2, "group_b": 40.0}}},
        {
            "observed_fraction": {
                "n": 100,
                "group_a": 0.2,
                "group_b": 40.0,
            }
        },
        {"observed_fraction": {"metric": "aic", "group_a": 40.0}},
        {"observed_fraction": {"unit": None, "value": 40.0}},
        {"observed_fraction": {"unit": "unknown", "value": 40.0}},
        {"observed_fraction": {"unit": 1, "value": 40.0}},
        {"observed_fraction": {"metric": "availability", "value": 40.0}},
        {
            "observed_fraction": {
                "summary": {"group_a": 0.2, "group_b": 40.0}
            }
        },
        {"observed_fraction": {"data": [0.2, 40.0]}},
        {"observed_fraction": {"payload": {"group_a": 40.0}}},
        {"observed_fraction": {"items": [0.2, 40.0]}},
        {"observed_fraction": {"estimate": 0.2, "group_a": 40.0}},
        {"observed_fraction": {"value": 0.2, "group_a": 40.0}},
        {"observed_fraction": {"fraction": 0.2, "group_a": 40.0}},
        {"observed_fraction": {"point_estimate": 0.2, "group_a": 40.0}},
        {"observed_fraction": {"result": 0.2, "group_a": 40.0}},
        {"observed_fraction": {"ci_low": 0.1, "ci_high": 0.3, "group_a": 40.0}},
        {"overall_risk": {"point": 40.0, "ci_low": 0.1, "ci_high": 0.3}},
        {
            "observed_fraction": {
                "metric": "aic",
                "value": 0.2,
                "group_a": 40.0,
            }
        },
    ),
)
def test_step_summary_fraction_scale_preserves_standard_metric_wrappers(
    step_summary: dict[str, object],
) -> None:
    findings = StepSummaryFractionValidator().audit(
        step=AnalysisStep(step_id="04_audit", intent="Audit bounded metrics."),
        step_summary=step_summary,
    )

    assert findings
    assert any(finding.detail["reported_value"] == 40.0 for finding in findings)


@pytest.mark.parametrize(
    "nested_record",
    (
        {"model_diagnostics": {"metric": "aic", "value": 123.0}},
        {"counts": [{"group": "all", "value": 94_458}]},
        {
            "effect": {
                "measure": "odds_ratio",
                "estimate": 2.0,
                "ci_low": 1.2,
                "ci_high": 3.0,
            }
        },
        {"display": {"unit": "percent", "value": 20.0}},
        {"results": [{"metric": "aic", "value": 123.0}]},
        {"by_group": [{"unit": "percent", "value": 20.0}]},
        {
            "levels": [
                {
                    "level": 4,
                    "count": 1_256,
                    "fraction": 0.013,
                    "risk": 0.378,
                    "ci_low": 0.352,
                    "ci_high": 0.405,
                }
            ]
        },
        {
            "estimates": [
                {
                    "measure": "odds_ratio",
                    "estimate": 2.0,
                    "ci_low": 1.2,
                    "ci_high": 3.0,
                }
            ]
        },
        {"display": {"unit": "%", "value": 20.0}},
        {"display": {"display_unit": "percent", "value": 20.0}},
        {"display": {"value_unit": "percent", "value": 20.0}},
        {"display": {"type": "percent", "value": 20.0}},
        {"display": {"metric_type": "odds_ratio", "estimate": 2.0}},
        {"model_fit": {"value": 123.0}},
        {"sample_sizes": [{"value": 94_458}]},
        {"value": 0.2, "decimal_places": 3},
        {"estimate": 0.2, "model_version": 2},
        {"group_a": {"deaths": 5_117, "prevalence": 0.12}},
        {"estimate": 0.2, "format": {"digits": 3}},
        {"estimate": 0.2, "rounding": {"decimal_places": 3}},
        {"estimate": 0.2, "settings": {"precision": 3}},
        {
            "risk": 0.2,
            "nobs": 94_458,
            "total": 94_458,
            "OR": 2.0,
            "HR": 2.0,
            "aic": 123.0,
            "followup_days": 30,
        },
    ),
)
def test_step_summary_fraction_scale_does_not_cross_domain_boundaries(
    nested_record: dict[str, object],
) -> None:
    findings = StepSummaryFractionValidator().audit(
        step=AnalysisStep(step_id="04_model", intent="Report absolute risk."),
        step_summary={"overall_risk": {"risk": 0.2, **nested_record}},
    )

    assert findings == []


def test_step_summary_fraction_scale_ignores_method_and_count_fields() -> None:
    findings = StepSummaryFractionValidator().audit(
        step=AnalysisStep(step_id="04_model", intent="Fit a flexible model."),
        step_summary={
            "fractional_polynomial_power": 2,
            "sampling_fraction_denominator": 500,
            "sampling_fraction_numerator": 125,
            "attributable_fraction": -0.08,
            "observed_fraction": {
                "value": 0.25,
                "numerator": 125,
                "denominator": 500,
            },
        },
    )

    assert findings == []


def test_step_summary_fraction_scale_still_checks_nested_fraction_map() -> None:
    findings = StepSummaryFractionValidator().audit(
        step=AnalysisStep(step_id="04_audit", intent="Audit completeness."),
        step_summary={"observed_fraction": {"lab": 0.4, "vital": 40.0}},
    )

    assert len(findings) == 1
    assert findings[0].detail["summary_path"] == "observed_fraction.vital"


def test_step_summary_fraction_scale_rejects_fraction_stored_as_percent() -> None:
    findings = StepSummaryFractionValidator().audit(
        step=AnalysisStep(step_id="04_reconciliation", intent="Audit levels."),
        step_summary={
            "valid_level_distribution_percent": {"0": 0.7, "1": 0.3},
            "valid_level_distribution_percent_pct": {"0": 70.0, "1": 30.0},
        },
    )

    assert len(findings) == 1
    assert "Rename the first field to *_fraction" in findings[0].message
    assert findings[0].detail["summary_path"] == (
        "valid_level_distribution_percent"
    )


def _write_reconciliation_trace_fixture(
    tmp_path: Path, *, correct: bool
) -> tuple[dict, Path]:
    parent = pd.DataFrame(
        [
            {
                "exposure": "sofa2_liver_max",
                "group_type": "exposure_level",
                "group_value": 0,
                "estimate_type": "outcome_risk",
                "n": 10,
                "event_n": 2,
                "outcome_risk": 0.2,
            },
            {
                "exposure": "sofa2_liver_max",
                "group_type": "source_state",
                "group_value": "observed",
                "estimate_type": "outcome_risk",
                "n": 10,
                "event_n": 2,
                "outcome_risk": 0.2,
            },
            {
                "exposure": "bili_max",
                "group_type": "continuous_summary",
                "group_value": "observed",
                "estimate_type": "continuous_distribution",
                "n": 10,
                "event_n": np.nan,
                "outcome_risk": np.nan,
                "median": 0.7,
                "q25": 0.4,
                "q75": 1.4,
            },
        ]
    )
    parent_path = tmp_path / "parent.csv"
    parent.to_csv(parent_path, index=False)

    reconciliation = pd.DataFrame(
        [
            {
                "source_variable": "sofa2_liver_max",
                "requested_stratum": "0",
                "requested_role": "required_valid_ordinal_level",
                "registered_output_status": "row_supported",
                "registered_n": 10 if correct else 100,
                "registered_event_n": 2 if correct else 20,
                "registered_risk": 0.2 if correct else 0.1,
                "registered_n_field": "n" if correct else "n_denominator",
                "registered_event_n_field": (
                    "event_n" if correct else "n_positive"
                ),
                "registered_risk_field": "outcome_risk" if correct else "estimate",
            },
            {
                "source_variable": "sofa2_liver_max",
                "requested_stratum": "valid observed",
                "requested_role": "required_source_status",
                "registered_output_status": (
                    "row_supported" if correct else "row_not_supported"
                ),
                "registered_n": 10 if correct else np.nan,
                "registered_event_n": 2 if correct else np.nan,
                "registered_risk": 0.2 if correct else np.nan,
                "registered_n_field": "n" if correct else np.nan,
                "registered_event_n_field": "event_n" if correct else np.nan,
                "registered_risk_field": "outcome_risk" if correct else np.nan,
            },
            {
                "source_variable": "bili_max",
                "requested_stratum": "valid_observed_continuous",
                "requested_role": "required_continuous_representation",
                "registered_output_status": "row_supported",
                "registered_n": 10,
                "registered_risk": np.nan if correct else 0.2,
                "registered_n_field": "n",
                "registered_risk_field": np.nan,
                "registered_median": 0.7,
                "registered_q25": 0.4,
                "registered_q75": 1.4,
                "registered_median_field": "median",
                "registered_q25_field": "q25",
                "registered_q75_field": "q75",
            },
        ]
    )
    reconciliation.to_csv(
        tmp_path / "absolute_risk_representation_reconciliation.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "row_type": "source_status",
                "status": "no source",
                "percentage_of_valid_observed": np.nan if correct else 1.2,
                "percentage_of_valid_observed_pct": np.nan if correct else 120.0,
            }
        ]
    ).to_csv(tmp_path / "reconciled_absolute_risk.csv", index=False)
    summary = {
        "registered_upstream_output": {
            "upstream_step": "04_absolute_risk_context",
            "selected_path": str(parent_path),
        },
        "output_files": ["absolute_risk_representation_reconciliation.csv"],
    }
    return summary, tmp_path


def test_cross_step_reconciliation_trace_rejects_wrong_parent_rows(
    tmp_path: Path,
) -> None:
    summary, out_dir = _write_reconciliation_trace_fixture(tmp_path, correct=False)

    findings = CrossStepReconciliationTraceValidator().audit(
        step=AnalysisStep(step_id="04_reconciliation", intent="Reconcile parent."),
        step_summary=summary,
        out_dir=out_dir,
    )

    assert len(findings) == 1
    issue_types = {item["issue"] for item in findings[0].detail["issues"]}
    assert "registered_n_mismatch" in issue_types
    assert "registered_event_n_mismatch" in issue_types
    assert "registered_risk_mismatch" in issue_types
    assert "supported_parent_row_reported_missing" in issue_types
    assert "continuous_distribution_has_false_risk" in issue_types
    assert "source_status_percentage_field_not_applicable" in issue_types


def test_cross_step_reconciliation_trace_accepts_exact_parent_rows(
    tmp_path: Path,
) -> None:
    summary, out_dir = _write_reconciliation_trace_fixture(tmp_path, correct=True)

    findings = CrossStepReconciliationTraceValidator().audit(
        step=AnalysisStep(step_id="04_reconciliation", intent="Reconcile parent."),
        step_summary=summary,
        out_dir=out_dir,
    )

    assert findings == []


def test_cross_step_reconciliation_trace_normalises_semantic_row_schema(
    tmp_path: Path,
) -> None:
    summary, out_dir = _write_reconciliation_trace_fixture(tmp_path, correct=True)
    pd.DataFrame(
        [
            {
                "variable": "sofa2_liver_max",
                "estimate_type": "outcome_risk",
                "stratum_type": "exposure_level",
                "stratum": "0",
                "source_status": "valid observed",
                "registered_n": np.nan,
                "registered_outcome_risk": np.nan,
                "registered_n_field": np.nan,
                "registered_outcome_risk_field": np.nan,
                "row_supported": False,
            },
            {
                "variable": "sofa2_liver_max",
                "estimate_type": "outcome_risk",
                "stratum_type": "source_status",
                "stratum": "valid observed",
                "source_status": "valid observed",
                "registered_n": np.nan,
                "registered_outcome_risk": np.nan,
                "registered_n_field": np.nan,
                "registered_outcome_risk_field": np.nan,
                "row_supported": False,
            },
            {
                "variable": "bili_max",
                "estimate_type": "distribution",
                "stratum_type": "distribution",
                "stratum": "valid observed",
                "source_status": "valid observed",
                "registered_n": np.nan,
                "registered_outcome_risk": np.nan,
                "registered_n_field": np.nan,
                "registered_outcome_risk_field": np.nan,
                "row_supported": False,
            },
        ]
    ).to_csv(
        out_dir / "absolute_risk_representation_reconciliation.csv", index=False
    )

    findings = CrossStepReconciliationTraceValidator().audit(
        step=AnalysisStep(step_id="04_reconciliation", intent="Reconcile parent."),
        step_summary=summary,
        out_dir=out_dir,
    )

    assert len(findings) == 1
    issues = findings[0].detail["issues"]
    assert sum(
        item["issue"] == "supported_parent_row_reported_missing" for item in issues
    ) == 3


def test_cross_step_reconciliation_trace_normalises_requested_estimate_schema(
    tmp_path: Path,
) -> None:
    summary, out_dir = _write_reconciliation_trace_fixture(tmp_path, correct=True)
    parent_path = summary["registered_upstream_output"]["selected_path"]
    summary = {
        "registered_upstream": {
            "requested_step": "04_absolute_risk_context",
            "path": parent_path,
        },
        "output_files": ["absolute_risk_representation_reconciliation.csv"],
    }
    pd.DataFrame(
        [
            {
                "variable": "sofa2_liver_max",
                "requested_estimate_type": "outcome_risk",
                "requested_stratum": "level_0",
                "requested_source_status": "valid observed",
                "requested_level": 0,
                "registered_supported": False,
                "registered_n": np.nan,
                "registered_outcome_risk": np.nan,
                "registered_selected_fields": np.nan,
            },
            {
                "variable": "sofa2_liver_max",
                "requested_estimate_type": "outcome_risk",
                "requested_stratum": "valid observed",
                "requested_source_status": "valid observed",
                "requested_level": np.nan,
                "registered_supported": False,
                "registered_n": np.nan,
                "registered_outcome_risk": np.nan,
                "registered_selected_fields": np.nan,
            },
            {
                "variable": "bili_max",
                "requested_estimate_type": "distribution",
                "requested_stratum": "valid observed",
                "requested_source_status": "valid observed",
                "requested_level": np.nan,
                "registered_supported": False,
                "registered_n": np.nan,
                "registered_outcome_risk": np.nan,
                "registered_selected_fields": np.nan,
            },
        ]
    ).to_csv(
        out_dir / "absolute_risk_representation_reconciliation.csv", index=False
    )

    findings = CrossStepReconciliationTraceValidator().audit(
        step=AnalysisStep(step_id="04_reconciliation", intent="Reconcile parent."),
        step_summary=summary,
        out_dir=out_dir,
    )

    assert len(findings) == 1
    issues = findings[0].detail["issues"]
    assert sum(
        item["issue"] == "supported_parent_row_reported_missing" for item in issues
    ) == 3


def test_cross_step_reconciliation_trace_accepts_generic_registered_table_path(
    tmp_path: Path,
) -> None:
    summary, out_dir = _write_reconciliation_trace_fixture(tmp_path, correct=True)
    parent_path = summary["registered_upstream_output"]["selected_path"]
    summary = {
        "registered_upstream": {
            "upstream_step": "04_absolute_risk_context",
            "registered_table_path": parent_path,
        },
        "output_files": ["absolute_risk_representation_reconciliation.csv"],
    }
    pd.DataFrame(
        [
            {
                "variable": "sofa2_liver_max",
                "row_type": "level",
                "estimate_type": "outcome_risk",
                "stratum": "level_0",
                "level": 0,
                "source_status": "valid observed",
                "registered_supported": False,
                "row_supported": False,
                "registered_n": np.nan,
                "registered_outcome_risk": np.nan,
            }
        ]
    ).to_csv(
        out_dir / "absolute_risk_representation_reconciliation.csv", index=False
    )

    findings = CrossStepReconciliationTraceValidator().audit(
        step=AnalysisStep(step_id="04_reconciliation", intent="Reconcile parent."),
        step_summary=summary,
        out_dir=out_dir,
    )

    assert len(findings) == 1
    assert findings[0].detail["issues"][0]["issue"] == (
        "supported_parent_row_reported_missing"
    )


def test_cross_step_reconciliation_trace_normalises_row_role_schema(
    tmp_path: Path,
) -> None:
    summary, out_dir = _write_reconciliation_trace_fixture(tmp_path, correct=True)
    pd.DataFrame(
        [
            {
                "variable": "sofa2_liver_max",
                "row_role": "level",
                "estimate_type": "outcome_risk",
                "stratum": "level_0",
                "source_status": "valid observed",
                "registered_row_supported": False,
                "registered_n": np.nan,
                "registered_event_n": np.nan,
                "registered_outcome_risk": np.nan,
            },
            {
                "variable": "sofa2_liver_max",
                "row_role": "source_status",
                "estimate_type": "outcome_risk",
                "stratum": "valid observed",
                "source_status": "valid observed",
                "registered_row_supported": False,
                "registered_n": np.nan,
                "registered_event_n": np.nan,
                "registered_outcome_risk": np.nan,
            },
            {
                "variable": "bili_max",
                "row_role": "distribution",
                "estimate_type": "continuous_distribution",
                "stratum": "valid observed",
                "source_status": "valid observed",
                "registered_row_supported": True,
                "registered_n": 10,
                "registered_event_n": np.nan,
                "registered_outcome_risk": np.nan,
                "registered_median": 0.7,
                "registered_q25": 0.4,
                "registered_q75": 1.4,
                "registered_selected_fields": (
                    "estimate_type;exposure;n;median;q25;q75"
                ),
            },
        ]
    ).to_csv(
        out_dir / "absolute_risk_representation_reconciliation.csv", index=False
    )

    findings = CrossStepReconciliationTraceValidator().audit(
        step=AnalysisStep(step_id="04_reconciliation", intent="Reconcile parent."),
        step_summary=summary,
        out_dir=out_dir,
    )

    assert len(findings) == 1
    issues = findings[0].detail["issues"]
    assert sum(
        item["issue"] == "supported_parent_row_reported_missing"
        for item in issues
    ) == 2


def test_cross_step_reconciliation_trace_normalises_requested_group_schema(
    tmp_path: Path,
) -> None:
    summary, out_dir = _write_reconciliation_trace_fixture(tmp_path, correct=True)
    pd.DataFrame(
        [
            {
                "variable": "sofa2_liver_max",
                "requested_group_type": "exposure_level",
                "requested_group_value": 0,
                "requested_estimate_type": "outcome_risk",
                "registered_supported": False,
                "registered_n": np.nan,
                "registered_event_n": np.nan,
                "registered_outcome_risk": np.nan,
                "selected_parent_row_fields": np.nan,
            },
            {
                "variable": "sofa2_liver_max",
                "requested_group_type": "source_status",
                "requested_group_value": "valid observed",
                "requested_estimate_type": "outcome_risk",
                "registered_supported": True,
                "registered_n": 10,
                "registered_event_n": 2,
                "registered_outcome_risk": 0.2,
                "selected_parent_row_field_names": "n;event_n;outcome_risk",
            },
            {
                "variable": "bili_max",
                "requested_group_type": "distribution",
                "requested_group_value": "valid observed",
                "requested_estimate_type": "continuous_distribution",
                "registered_supported": True,
                "registered_n": 10,
                "registered_event_n": np.nan,
                "registered_outcome_risk": np.nan,
                "registered_median": 0.7,
                "registered_q25": 0.4,
                "registered_q75": 1.4,
                "selected_parent_field_names": "n;median;q25;q75",
            },
        ]
    ).to_csv(
        out_dir / "absolute_risk_representation_reconciliation.csv", index=False
    )

    findings = CrossStepReconciliationTraceValidator().audit(
        step=AnalysisStep(step_id="04_reconciliation", intent="Reconcile parent."),
        step_summary=summary,
        out_dir=out_dir,
    )

    assert len(findings) == 1
    assert findings[0].detail["issues"] == [
        {
            "row": "sofa2_liver_max:0",
            "issue": "supported_parent_row_reported_missing",
        }
    ]


def test_cross_step_reconciliation_trace_requires_declared_range_rows(
    tmp_path: Path,
) -> None:
    summary, out_dir = _write_reconciliation_trace_fixture(tmp_path, correct=True)
    summary["bilirubin"] = {
        "range_flag_counts": {
            "within_declared_range": 8,
            "above_declared_range": 2,
        }
    }
    path = out_dir / "absolute_risk_representation_reconciliation.csv"

    missing = CrossStepReconciliationTraceValidator().audit(
        step=AnalysisStep(step_id="04_reconciliation", intent="Reconcile parent."),
        step_summary=summary,
        out_dir=out_dir,
    )

    assert len(missing) == 1
    assert {
        item["row"]
        for item in missing[0].detail["issues"]
        if item["issue"] == "missing_declared_range_flag_row"
    } == {"within_declared_range", "above_declared_range"}

    frame = pd.read_csv(path)
    frame = pd.concat(
        [
            frame,
            pd.DataFrame(
                [
                    {
                        "source_variable": "bili_max",
                        "requested_stratum": flag,
                        "requested_role": "range_flag",
                        "registered_output_status": "row_not_supported",
                        "registered_n": np.nan,
                    }
                    for flag in ("within_declared_range", "above_declared_range")
                ]
            ),
        ],
        ignore_index=True,
    )
    frame.to_csv(path, index=False)

    present = CrossStepReconciliationTraceValidator().audit(
        step=AnalysisStep(step_id="04_reconciliation", intent="Reconcile parent."),
        step_summary=summary,
        out_dir=out_dir,
    )

    assert present == []


def test_cross_step_reconciliation_trace_fails_closed_on_unknown_schema(
    tmp_path: Path,
) -> None:
    summary, out_dir = _write_reconciliation_trace_fixture(tmp_path, correct=True)
    pd.DataFrame(
        [
            {
                "variable": "sofa2_liver_max",
                "row_role": "level",
                "stratum": "level_0",
                "parent_trace_available": False,
                "registered_n": np.nan,
            }
        ]
    ).to_csv(
        out_dir / "absolute_risk_representation_reconciliation.csv", index=False
    )

    findings = CrossStepReconciliationTraceValidator().audit(
        step=AnalysisStep(step_id="04_reconciliation", intent="Reconcile parent."),
        step_summary=summary,
        out_dir=out_dir,
    )

    assert len(findings) == 1
    assert findings[0].detail["issue"] == "reconciliation_schema_unrecognised"


def test_cross_step_source_status_supports_concept_counts_schema() -> None:
    current = {
        "source_status_summary": {
            "lab_max": {
                "counts": {
                    "valid observed": 41_210,
                    "no source": 53_248,
                    "measured/source present but summary missing": 0,
                    "contradictory/invalid": 0,
                },
                "valid_observed_n": 41_210,
            }
        }
    }

    findings = CrossStepSourceStatusValidator().audit(
        step=AnalysisStep(step_id="04_reconciliation", intent="Reconcile status."),
        step_summary=current,
        completed_step_records=[_prior_source_status_record()],
    )

    assert findings == []


def test_cross_step_source_status_supports_nested_count_detail_schema() -> None:
    current = {
        "source_status_counts": {
            "lab_max": {
                "valid observed": {"count": 41_210},
                "no source": {"count": 53_248},
                "measured/source present but summary missing": {"count": 0},
                "contradictory/invalid": {"count": 0},
            }
        }
    }

    findings = CrossStepSourceStatusValidator().audit(
        step=AnalysisStep(step_id="04_reconciliation", intent="Reconcile status."),
        step_summary=current,
        completed_step_records=[_prior_source_status_record()],
    )

    assert findings == []


def test_cross_step_source_status_supports_count_map_with_n_schema() -> None:
    current = {
        "source_status_count_map": {
            "lab_max": {
                "valid observed": {"n": 41_210},
                "no source": {"n": 53_248},
                "measured/source present but summary missing": {"n": 0},
                "contradictory/invalid": {"n": 0},
            }
        }
    }

    findings = CrossStepSourceStatusValidator().audit(
        step=AnalysisStep(step_id="04_reconciliation", intent="Reconcile status."),
        step_summary=current,
        completed_step_records=[_prior_source_status_record()],
    )

    assert findings == []


def test_cross_step_source_status_requires_explicit_zero_categories() -> None:
    current = {
        "bilirubin_reconciliation": {
            "source_columns": ["lab_max", "lab_n", "lab_measured"],
            "source_status_counts": {
                "valid-observed": 41_210,
                "no-source": 53_248,
            },
        }
    }

    findings = CrossStepSourceStatusValidator().audit(
        step=AnalysisStep(step_id="04_reconciliation", intent="Reconcile status."),
        step_summary=current,
        completed_step_records=[_prior_source_status_record()],
    )

    assert len(findings) == 1
    assert findings[0].validator == "cross_step_source_status"
    assert findings[0].detail["missing_status_roles"] == [
        "contradictory_invalid",
        "measured_summary_missing",
    ]


def test_cross_step_source_status_accepts_explicit_zero_categories() -> None:
    current = {
        "bilirubin_reconciliation": {
            "source_columns": ["lab_max", "lab_n", "lab_measured"],
            "source_status_counts": {
                "valid-observed": 41_210,
                "no-source": 53_248,
                "measured-but-summary-missing": 0,
                "contradictory-or-invalid": 0,
            },
        }
    }

    findings = CrossStepSourceStatusValidator().audit(
        step=AnalysisStep(step_id="04_reconciliation", intent="Reconcile status."),
        step_summary=current,
        completed_step_records=[_prior_source_status_record()],
    )

    assert findings == []


def test_cross_step_source_status_blocks_denominator_drift() -> None:
    findings = CrossStepSourceStatusValidator().audit(
        step=AnalysisStep(step_id="03_table_one", intent="Build Table 1"),
        step_summary=_current_table_one_status(valid_n=2_380),
        completed_step_records=[_prior_source_status_record()],
    )

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].validator == "cross_step_source_status"
    assert findings[0].detail["reported_valid_observed_n"] == 2_380
    assert findings[0].detail["expected_valid_observed_n"] == 41_210


def test_cross_step_source_status_accepts_matching_denominator() -> None:
    findings = CrossStepSourceStatusValidator().audit(
        step=AnalysisStep(step_id="03_table_one", intent="Build Table 1"),
        step_summary=_current_table_one_status(valid_n=41_210),
        completed_step_records=[_prior_source_status_record()],
    )

    assert findings == []


def test_cross_step_source_status_skips_a_different_cohort_total() -> None:
    findings = CrossStepSourceStatusValidator().audit(
        step=AnalysisStep(step_id="03_table_one", intent="Build Table 1"),
        step_summary=_current_table_one_status(valid_n=2_380, total_n=80_000),
        completed_step_records=[_prior_source_status_record()],
    )

    assert findings == []


def test_cross_step_source_status_uses_latest_successful_lock() -> None:
    stale = _prior_source_status_record(valid_n=40_000)
    stale["step_id"] = "01_initial_audit"
    stale["status"] = "ok"
    failed = _prior_source_status_record(valid_n=2_380)
    failed["step_id"] = "02_failed_reconciliation"
    failed["status"] = "contract_failed"
    current_lock = _prior_source_status_record(valid_n=41_210)
    current_lock["step_id"] = "03_locked_reconciliation"
    current_lock["status"] = "ok"

    findings = CrossStepSourceStatusValidator().audit(
        step=AnalysisStep(step_id="04_table_one", intent="Build Table 1"),
        step_summary=_current_table_one_status(valid_n=41_210),
        completed_step_records=[stale, failed, current_lock],
    )

    assert findings == []


def test_cross_step_source_status_supports_scalar_status_summary_schema() -> None:
    current = {
        "bilirubin_definition": {"summary_variable": "lab_max"},
        "missingness_and_measurement_status": {
            "bilirubin": {
                "denominator_n": 94_458,
                "observed_valid_summary_n": 2_380,
                "source_absent_n": 53_248,
                "contradictory_or_invalid_n": 38_830,
            }
        },
    }

    findings = CrossStepSourceStatusValidator().audit(
        step=AnalysisStep(step_id="03_table_one", intent="Build Table 1"),
        step_summary=current,
        completed_step_records=[_prior_source_status_record()],
    )

    assert len(findings) == 1
    assert findings[0].detail["summary_path"] == (
        "missingness_and_measurement_status.bilirubin"
    )
    assert findings[0].detail["reported_valid_observed_n"] == 2_380


def _ctx_with_sofa(ra) -> "ra.ResearchContext":
    """Tiny context with a sofa2 column — the SOFA-aware validators
    fire on this shape."""
    df = pd.DataFrame({
        "stay_id": list(range(1, 11)),
        "age": [60, 70, 50, 80, 65, 75, 90, 40, 55, 60],
        "sofa2": [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        "lact": [1.0, 2.0, 1.5, 3.0, 4.0, 5.0, 2.5, 1.2, 3.3, 7.0],
        "death": [1, 1, 0, 0, 0, 1, 1, 0, 1, 1],
    })
    return ra.build_research_context(
        research_question="sofa2 → death?",
        cohort=df, cohort_name="t", database="synthetic",
        target_outcome="death",
    )


def test_concept_usage_flags_mean_of_sofa(ra):
    # Impartiality contract: mean/SD of an ordinal/composite score is a
    # reporting-practice *preference*, not an objective error, so it is a
    # WARNING (advisory) that never hard-blocks a run. The caution must
    # still be raised (so a reviewer sees it), just not as severity="error".
    ctx = _ctx_with_sofa(ra)
    auditor = ra.ConceptUsageAuditor()
    code = "x = df['sofa2'].mean()  # advisory"
    findings = auditor.audit(context=ctx, script_text=code)
    matched = [
        f for f in findings
        if f.validator == auditor.name
        and ("sofa" in f.message.lower() or "ordinal" in f.message.lower())
    ]
    assert matched, findings
    assert all(f.severity == "warning" for f in matched), matched
    # ...and no forbidden-aggregation finding is escalated to a blocking error.
    assert not any(
        f.severity == "error" and "misleading" in f.message.lower()
        for f in findings
    ), findings


def test_figure_source_data_validator_accepts_source_row_index_trace(tmp_path: Path):
    parent = tmp_path / "steps" / "02_descriptive_results" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "group_label": ["Group A", "Group B"],
            "n": [100, 120],
            "event_n": [10, 24],
            "outcome_risk_pct": [10.0, 20.0],
        }
    ).to_csv(parent / "outcome_by_group.csv", index=False)

    out = tmp_path / "steps" / "02_descriptive_results_figure" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "group_label": ["Group B"],
            "n": [120],
            "event_n": [24],
            "outcome_risk_pct": [20.0],
            "source_table": ["outcome_by_group.csv"],
            "source_row_index": [1],
        }
    ).to_csv(out / "figure_panel_source_data.csv", index=False)

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="02_descriptive_results_figure",
            intent="Render figure for step '02_descriptive_results'.",
        ),
        out_dir=out,
        run_dir=tmp_path,
        step_summary={"rendering_only": True},
    )

    assert findings == []


def test_figure_source_data_validator_handles_shared_boolean_columns(
    tmp_path: Path,
):
    parent = tmp_path / "steps" / "05_sensitivity" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "term": ["primary"],
            "estimate": [1.2],
            "converged": [True],
        }
    ).to_csv(parent / "sensitivity_results.csv", index=False)

    out = tmp_path / "steps" / "05_sensitivity_figure" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "term": ["primary"],
            "estimate": [1.2],
            "converged": [True],
            "source_table": ["sensitivity_results.csv"],
        }
    ).to_csv(out / "figure_panel_source_data.csv", index=False)

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="05_sensitivity_figure",
            intent="Render figure for step '05_sensitivity'.",
        ),
        out_dir=out,
        run_dir=tmp_path,
        step_summary={"rendering_only": True},
    )

    assert findings == []


def test_concept_usage_mean_of_sofa_blocks_under_strict_ablation(ra, monkeypatch):
    # The historical strict fail-closed benchmark stays reproducible behind
    # EASYICU_AUDIT_ORDINAL_STRICT=1, which restores severity="error" for
    # primary-analysis / manuscript stages.
    monkeypatch.setenv("EASYICU_AUDIT_ORDINAL_STRICT", "1")
    ctx = _ctx_with_sofa(ra)
    auditor = ra.ConceptUsageAuditor()
    schema = ra.schema
    step = schema.AnalysisStep(step_id="primary_association", intent="primary")
    findings = auditor.audit(
        context=ctx, script_text="x = df['sofa2'].mean()", step=step
    )
    assert any(
        f.severity == "error" and f.validator == auditor.name for f in findings
    ), findings


def test_concept_usage_flags_mean_of_lact_without_median(ra):
    ctx = _ctx_with_sofa(ra)
    auditor = ra.ConceptUsageAuditor()
    code = "lact_avg = df['lact'].mean()"
    findings = auditor.audit(context=ctx, script_text=code)
    # Lab + mean without median → warning
    assert any(f.severity == "warning" and "lact" in f.message.lower() for f in findings)


def test_concept_usage_silences_lab_mean_when_median_present(ra):
    ctx = _ctx_with_sofa(ra)
    auditor = ra.ConceptUsageAuditor()
    code = "x = df['lact'].mean(); y = df['lact'].median()"
    findings = auditor.audit(context=ctx, script_text=code)
    assert all("lact" not in f.message.lower() for f in findings)


def test_concept_usage_ignores_lab_missingness_fraction_mean(ra):
    ctx = _ctx_with_sofa(ra)
    findings = ra.ConceptUsageAuditor().audit(
        context=ctx,
        script_text='missing_pct = df["lact"].isna().mean() * 100',
    )
    assert all("lact" not in f.message.lower() for f in findings)


def test_concept_usage_silences_generic_helper_with_median_and_mean(ra):
    ctx = _ctx_with_sofa(ra)
    code = """
def add_continuous(series):
    vals = series.dropna().astype(float)
    return {
        "median": vals.median(),
        "q25": vals.quantile(0.25),
        "q75": vals.quantile(0.75),
        "mean": vals.mean(),
    }
summary = add_continuous(df["lact"])
"""
    findings = ra.ConceptUsageAuditor().audit(context=ctx, script_text=code)
    assert all("lact" not in f.message.lower() for f in findings)


def test_concept_usage_flags_fillna_zero(ra):
    ctx = _ctx_with_sofa(ra)
    auditor = ra.ConceptUsageAuditor()
    code = "df['lact'] = df['lact'].fillna(0)"
    findings = auditor.audit(context=ctx, script_text=code)
    assert any("fillna" in f.message.lower() or "imputation" in f.message.lower()
               for f in findings)


def test_concept_usage_allows_boolean_mask_fillna_false(ra):
    ctx = _ctx_with_sofa(ra)
    code = """
mask = pd.to_numeric(df["age"], errors="coerce") >= 18
adult = df.loc[mask.fillna(False)].copy()
"""
    findings = ra.ConceptUsageAuditor().audit(context=ctx, script_text=code)
    assert not any(
        "fillna" in f.message.lower() or "imputation" in f.message.lower()
        for f in findings
    )


def test_concept_usage_fillna_zero_ignores_env_string_subscripts(ra):
    ctx = _ctx_with_sofa(ra)
    findings = ra.ConceptUsageAuditor().audit(
        context=ctx,
        script_text='import os\npath = os.environ["COHORT_PARQUET"]',
    )
    assert not any("fillna" in f.message.lower() or "imputation" in f.message.lower()
                   for f in findings)


def test_concept_usage_flags_agg_mean_of_sofa(ra):
    # Detection still fires across call forms; severity is advisory (warning).
    ctx = _ctx_with_sofa(ra)
    findings = ra.ConceptUsageAuditor().audit(
        context=ctx,
        script_text='x = df["sofa2"].agg("mean")',
    )
    assert any(f.severity == "warning" and "sofa" in f.message.lower() for f in findings)


def test_concept_usage_flags_numpy_mean_of_sofa(ra):
    ctx = _ctx_with_sofa(ra)
    findings = ra.ConceptUsageAuditor().audit(
        context=ctx,
        script_text='import numpy as np\nx = np.mean(df["sofa2"])',
    )
    assert any(f.severity == "warning" and "sofa" in f.message.lower() for f in findings)


def test_concept_usage_flags_rolling_mean_of_sofa(ra):
    ctx = _ctx_with_sofa(ra)
    findings = ra.ConceptUsageAuditor().audit(
        context=ctx,
        script_text='x = df["sofa2"].rolling(3).mean()',
    )
    assert any(f.severity == "warning" and "sofa" in f.message.lower() for f in findings)


def test_statistical_validator_flags_outcome_mismatch(ra, tmp_path: Path):
    ctx = _ctx_with_sofa(ra)
    cohort_path = tmp_path / "cohort.parquet"
    pd.read_parquet  # touch import
    df = pd.DataFrame({
        "stay_id": list(range(1, 11)),
        "age": [60] * 10,
        "sofa2": [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        "lact": [1.0] * 10,
        "death": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 0.2 incidence
    })
    df.to_parquet(cohort_path, index=False)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # write a placeholder so out_dir is non-empty
    (out_dir / "step_summary.json").write_text("{}", encoding="utf-8")

    schema = ra.schema
    step = schema.AnalysisStep(step_id="02_outcome_incidence",
                               intent="incidence", expected_outputs=["statistic:outcome_rate"])
    validator = ra.StatisticalValidator()
    findings = validator.audit(
        context=ctx, cohort_path=cohort_path, step=step,
        out_dir=out_dir,
        # report a clearly wrong outcome rate
        step_summary={"outcome_rate": 0.99},
    )
    msgs = " ".join(f.message for f in findings)
    assert any(f.severity == "error" for f in findings), findings
    assert "outcome rate" in msgs.lower() or "disagrees" in msgs.lower()


def test_statistical_validator_ignores_outcome_blind_component_qc_table(
    ra,
    tmp_path: Path,
):
    ctx = _ctx_with_sofa(ra)
    cohort_path = tmp_path / "cohort.parquet"
    df = pd.DataFrame({
        "stay_id": list(range(1, 21)),
        "age": [60] * 20,
        "sofa2": [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5,
        "death": [1, 1, 1, 1, 0,  # rate at 0 = 0.8
                  0, 0, 0, 0, 0,  # rate at 1 = 0.0
                  0, 0, 0, 1, 0,  # rate at 2 = 0.2
                  1, 1, 1, 1, 1], # rate at 3 = 1.0
    })
    df.to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pd.DataFrame({
        "variable": ["sofa2"],
        "n_rows": [20],
        "n_low_completeness": [5],
        "frac_low_completeness": [0.25],
    }).to_csv(out_dir / "component_completeness_qc.csv", index=False)

    schema = ra.schema
    step = schema.AnalysisStep(
        step_id="05_component_completeness_qc",
        intent="component completeness QC",
    )
    validator = ra.StatisticalValidator()
    findings = validator.audit(
        context=ctx, cohort_path=cohort_path, step=step,
        out_dir=out_dir, step_summary={},
    )
    assert not any(
        "non-monotonic" in f.message.lower() or "exceeds" in f.message.lower()
        for f in findings
    ), findings


def test_statistical_validator_no_artefacts_is_error(ra, tmp_path: Path):
    ctx = _ctx_with_sofa(ra)
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "age": [60], "sofa2": [3], "lact": [1.0], "death": [0]}
                 ).to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()  # deliberately empty
    schema = ra.schema
    step = schema.AnalysisStep(step_id="99_empty", intent="x")
    findings = ra.StatisticalValidator().audit(
        context=ctx, cohort_path=cohort_path, step=step,
        out_dir=out_dir, step_summary={},
    )
    assert any(f.severity == "error" and "no output artefacts" in f.message.lower()
               for f in findings), findings


def test_statistical_validator_blocks_all_unavailable_primary_exposure(
    ra, tmp_path: Path
):
    ctx = _ctx_with_sofa(ra)
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}).to_parquet(
        cohort_path, index=False
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "step_summary.json").write_text("{}", encoding="utf-8")
    step = ra.schema.AnalysisStep(step_id="baseline", intent="Describe the cohort")

    findings = ra.StatisticalValidator().audit(
        context=ctx,
        cohort_path=cohort_path,
        step=step,
        out_dir=out_dir,
        step_summary={
            "cohort_n": 2,
            "primary_exposure": {
                "reconciliation_status": "unavailable",
                "missing_n": 2,
                "counts": {"Unexposed": 0, "Exposed": 0, "Unavailable": 2},
            },
        },
    )

    assert any(
        finding.severity == "error"
        and "no cohort row has a usable reconciled exposure" in finding.message
        for finding in findings
    ), findings


def test_statistical_validator_accepts_reconciled_primary_exposure(
    ra, tmp_path: Path
):
    ctx = _ctx_with_sofa(ra)
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}).to_parquet(
        cohort_path, index=False
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "step_summary.json").write_text("{}", encoding="utf-8")
    step = ra.schema.AnalysisStep(step_id="baseline", intent="Describe the cohort")

    findings = ra.StatisticalValidator().audit(
        context=ctx,
        cohort_path=cohort_path,
        step=step,
        out_dir=out_dir,
        step_summary={
            "cohort_n": 2,
            "primary_exposure": {
                "reconciliation_status": "checked",
                "missing_n": 0,
                "counts": {"Unexposed": 1, "Exposed": 1, "Unavailable": 0},
            },
        },
    )

    assert not any(
        "no cohort row has a usable reconciled exposure" in finding.message
        for finding in findings
    ), findings


def test_statistical_validator_flags_primary_or_mismatch(ra, tmp_path: Path):
    """T1.6 — when the reported OR disagrees with primary_association.csv,
    the validator must surface an error finding."""
    ctx = _ctx_with_sofa(ra)
    cohort_path = tmp_path / "cohort.parquet"
    df = pd.DataFrame({
        "stay_id": list(range(1, 11)),
        "age": [60] * 10,
        "sofa2": [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        "lact": [1.0] * 10,
        "death": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    })
    df.to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pd.DataFrame([
        {"variable": "intercept", "coef": -2.0, "odds_ratio": 0.135},
        {"variable": "sofa2", "coef": 0.4, "odds_ratio": 1.491825},
        {"variable": "age", "coef": 0.01, "odds_ratio": 1.01005},
    ]).to_csv(out_dir / "primary_association.csv", index=False)

    schema = ra.schema
    step = schema.AnalysisStep(step_id="04_primary_association", intent="logit")
    findings = ra.StatisticalValidator().audit(
        context=ctx, cohort_path=cohort_path, step=step,
        out_dir=out_dir,
        # report a wildly wrong OR
        step_summary={"predictor": "sofa2", "primary_or": 5.0,
                      "outcome_rate": float(df["death"].mean())},
    )
    assert any(f.severity == "error" and "primary or" in f.message.lower()
               for f in findings), findings


def _figure_source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    upstream = run_dir / "steps" / "05_sensitivity_comparison" / "outputs"
    figure = run_dir / "steps" / "05_sensitivity_comparison_figure" / "outputs"
    upstream.mkdir(parents=True)
    figure.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "spec_id": "primary_modeled_or",
                "effect_scale": "odds_ratio",
                "point_estimate": 1.07,
                "ci_low": 1.01,
                "ci_high": 1.13,
            },
            {
                "spec_id": "drop_lactate_modeled_or",
                "effect_scale": "odds_ratio",
                "point_estimate": 1.24,
                "ci_low": 1.18,
                "ci_high": 1.30,
            },
            {
                "spec_id": "primary_modeled_rd",
                "effect_scale": "risk_difference",
                "point_estimate": 0.005,
                "ci_low": 0.001,
                "ci_high": 0.009,
            },
        ]
    ).to_csv(upstream / "sensitivity_comparison.csv", index=False)
    return run_dir, figure


def test_figure_source_data_validator_accepts_upstream_subset(ra, tmp_path: Path):
    run_dir, figure_out = _figure_source_fixture(tmp_path)
    pd.DataFrame(
        [
            {
                "spec_id": "primary_modeled_or",
                "effect_scale": "odds_ratio",
                "point_estimate": 1.07,
                "ci_low": 1.01,
                "ci_high": 1.13,
            },
            {
                "spec_id": "primary_modeled_rd",
                "effect_scale": "risk_difference",
                "point_estimate": 0.005,
                "ci_low": 0.001,
                "ci_high": 0.009,
            },
        ]
    ).to_csv(figure_out / "sensitivity_forest_source_data.csv", index=False)

    step = ra.schema.AnalysisStep(
        step_id="05_sensitivity_comparison_figure",
        intent="Render the publication figure declared by step '05_sensitivity_comparison'.",
    )
    findings = ra.FigureSourceDataValidator().audit(
        step=step,
        out_dir=figure_out,
        run_dir=run_dir,
        step_summary={"rendering_only": True},
    )
    assert findings == []


def test_figure_source_data_validator_accepts_definition_id_key(ra, tmp_path: Path):
    run_dir = tmp_path / "run"
    upstream = (
        run_dir
        / "steps"
        / "04_alternative_eligibility_definitions_and_overlap"
        / "outputs"
    )
    figure = (
        run_dir
        / "steps"
        / "04_alternative_eligibility_definitions_and_overlap_figure"
        / "outputs"
    )
    upstream.mkdir(parents=True)
    figure.mkdir(parents=True)
    pd.DataFrame(
        {
            "definition_id": ["primary", "relax_temp"],
            "n_included": [100, 112],
            "moved_in_vs_primary_n": [0, 12],
        }
    ).to_csv(upstream / "alternative_cohort_attrition.csv", index=False)
    pd.DataFrame(
        {
            "definition_id": ["primary", "relax_temp"],
            "n_included": [100, 112],
            "moved_in_vs_primary_n": [0, 12],
        }
    ).to_csv(figure / "publication_figure_definition_source_data.csv", index=False)

    findings = ra.FigureSourceDataValidator().audit(
        step=ra.schema.AnalysisStep(
            step_id="04_alternative_eligibility_definitions_and_overlap_figure",
            intent=(
                "Render the publication figure declared by step "
                "'04_alternative_eligibility_definitions_and_overlap'."
            ),
        ),
        out_dir=figure,
        run_dir=run_dir,
        step_summary={"rendering_only": True},
    )
    assert findings == []


def test_figure_source_data_validator_accepts_pairwise_definition_key(
    ra,
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    upstream = (
        run_dir
        / "steps"
        / "04_alternative_eligibility_definitions_and_overlap"
        / "outputs"
    )
    figure = (
        run_dir
        / "steps"
        / "04_alternative_eligibility_definitions_and_overlap_figure"
        / "outputs"
    )
    upstream.mkdir(parents=True)
    figure.mkdir(parents=True)
    pd.DataFrame(
        {
            "definition_a": ["primary", "primary", "relax_temp"],
            "definition_b": ["primary", "relax_temp", "primary"],
            "intersection_n": [100, 100, 100],
            "jaccard": [1.0, 0.893, 0.893],
        }
    ).to_csv(upstream / "cohort_overlap_matrix.csv", index=False)
    pd.DataFrame(
        {
            "definition_a": ["primary", "relax_temp"],
            "definition_b": ["relax_temp", "primary"],
            "intersection_n": [100, 100],
            "jaccard": [0.893, 0.893],
        }
    ).to_csv(figure / "publication_figure_overlap_source_data.csv", index=False)

    findings = ra.FigureSourceDataValidator().audit(
        step=ra.schema.AnalysisStep(
            step_id="04_alternative_eligibility_definitions_and_overlap_figure",
            intent=(
                "Render the publication figure declared by step "
                "'04_alternative_eligibility_definitions_and_overlap'."
            ),
        ),
        out_dir=figure,
        run_dir=run_dir,
        step_summary={"rendering_only": True},
    )
    assert findings == []


def test_figure_source_data_validator_blocks_resume_evidence_pollution(
    ra,
    tmp_path: Path,
):
    run_dir, figure_out = _figure_source_fixture(tmp_path)
    pd.DataFrame(
        [
            {
                "spec_id": "primary_modeled_or",
                "effect_scale": "odds_ratio",
                "point_estimate": 1.07,
                "ci_low": 1.01,
                "ci_high": 1.13,
            },
            {
                "spec_id": "alt_cohort_from_old_robustness_panel",
                "effect_scale": "odds_ratio",
                "point_estimate": 1.03,
                "ci_low": 0.95,
                "ci_high": 1.11,
            },
        ]
    ).to_csv(figure_out / "sensitivity_forest_source_data.csv", index=False)

    step = ra.schema.AnalysisStep(
        step_id="05_sensitivity_comparison_figure",
        intent="Render the publication figure declared by step '05_sensitivity_comparison'.",
    )
    findings = ra.FigureSourceDataValidator().audit(
        step=step,
        out_dir=figure_out,
        run_dir=run_dir,
        step_summary={"rendering_only": True},
    )
    assert any(f.severity == "error" for f in findings), findings
    assert "absent" in findings[0].message.lower() or (
        findings[0].detail
        and findings[0].detail["best_mismatch"]["reason"]
        == "source_rows_not_in_upstream"
    )


def test_figure_source_data_validator_blocks_numeric_drift(ra, tmp_path: Path):
    run_dir, figure_out = _figure_source_fixture(tmp_path)
    pd.DataFrame(
        [
            {
                "spec_id": "primary_modeled_or",
                "effect_scale": "odds_ratio",
                "point_estimate": 9.99,
                "ci_low": 1.01,
                "ci_high": 1.13,
            }
        ]
    ).to_csv(figure_out / "sensitivity_forest_source_data.csv", index=False)

    step = ra.schema.AnalysisStep(
        step_id="05_sensitivity_comparison_figure",
        intent="Render the publication figure declared by step '05_sensitivity_comparison'.",
    )
    findings = ra.FigureSourceDataValidator().audit(
        step=step,
        out_dir=figure_out,
        run_dir=run_dir,
        step_summary={"rendering_only": True},
    )
    assert any(
        f.severity == "error"
        and f.detail
        and f.detail["best_mismatch"]["reason"] == "source_values_disagree"
        for f in findings
    ), findings


def test_figure_source_data_validator_blocks_inconsistent_percent_counts(
    ra,
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    upstream = (
        run_dir
        / "steps"
        / "02_baseline_characteristics_and_data_quality"
        / "outputs"
    )
    figure = (
        run_dir
        / "steps"
        / "02_baseline_characteristics_and_data_quality_figure"
        / "outputs"
    )
    upstream.mkdir(parents=True)
    figure.mkdir(parents=True)
    pd.DataFrame(
        {
            "variable": ["resp_max"],
            "missing_pct": [0.2512394927],
            "missing_n": [188],
            "total_n": [74829],
        }
    ).to_csv(upstream / "missingness_measurement_panel_source_data.csv", index=False)
    pd.DataFrame(
        {
            "variable": ["resp_max"],
            "missing_pct": [25.12394927],
            "missing_n": [188],
            "total_n": [74829],
        }
    ).to_csv(figure / "missingness_measurement_panel_source_data.csv", index=False)

    findings = ra.FigureSourceDataValidator().audit(
        step=ra.schema.AnalysisStep(
            step_id="02_baseline_characteristics_and_data_quality_figure",
            intent=(
                "Render the publication figure declared by step "
                "'02_baseline_characteristics_and_data_quality'."
            ),
        ),
        out_dir=figure,
        run_dir=run_dir,
        step_summary={"rendering_only": True},
    )

    assert any(
        f.severity == "error"
        and "100*missing_n/total_n" in f.message
        and f.detail
        and f.detail["expected_pct"] == pytest.approx(0.2512394927)
        for f in findings
    ), findings


def test_figure_source_data_validator_accepts_derived_missingness_source_data(
    ra,
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    upstream = (
        run_dir
        / "steps"
        / "02_baseline_characteristics_and_data_quality"
        / "outputs"
    )
    figure = (
        run_dir
        / "steps"
        / "02_baseline_characteristics_and_data_quality_figure"
        / "outputs"
    )
    upstream.mkdir(parents=True)
    figure.mkdir(parents=True)
    pd.DataFrame(
        {
            "concept": ["resp", "lact"],
            "label": ["Respiratory rate", "Lactate"],
            "n_total": [74829, 74829],
            "value_missing_n": [188, 30490],
            "value_missing_pct": [0.2512394927100456, 40.74623474856005],
            "measured_one_n": [74641, 44339],
            "measured_one_pct": [99.74876050728996, 59.25376525143995],
        }
    ).to_csv(upstream / "missingness_measurement_audit.csv", index=False)
    pd.DataFrame(
        {
            "variable": ["resp", "lact"],
            "concept": ["resp", "lact"],
            "label": ["Respiratory rate", "Lactate"],
            "display_label": ["Respiratory rate", "Lactate"],
            "missing_pct": [0.2512394927100456, 40.74623474856005],
            "missing_n": [188, 30490],
            "total_n": [74829, 74829],
            "value_missing_pct": [0.2512394927100456, 40.74623474856005],
            "value_missing_n": [188, 30490],
            "n_total": [74829, 74829],
            "measured_pct": [99.74876050728996, 59.25376525143995],
            "measured_n": [74641, 44339],
            "measured_one_pct": [99.74876050728996, 59.25376525143995],
            "measured_one_n": [74641, 44339],
            "source_table": ["missingness_measurement_audit.csv"] * 2,
            "source_transform": ["missingness_measurement_summary_v1"] * 2,
        }
    ).to_csv(figure / "missingness_measurement_panel_source_data.csv", index=False)

    findings = ra.FigureSourceDataValidator().audit(
        step=ra.schema.AnalysisStep(
            step_id="02_baseline_characteristics_and_data_quality_figure",
            intent=(
                "Render the publication figure declared by step "
                "'02_baseline_characteristics_and_data_quality'."
            ),
        ),
        out_dir=figure,
        run_dir=run_dir,
        step_summary={"rendering_only": True},
    )

    assert findings == []


def test_figure_contract_quality_blocks_rescue_publication_contract(ra, tmp_path: Path):
    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / "04_primary_association_figure" / "outputs"
    out_dir.mkdir(parents=True)
    contract_path = out_dir / "publication_figure.figure_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "figure_id": "publication_figure",
                "core_claim": "Adjusted odds ratios are summarised from source data.",
                "statistics_note": (
                    "Deterministic rescue figure generated when the figure-only "
                    "child step did not emit exports."
                ),
                "panels": [
                    {
                        "panel_id": "A",
                        "title": "Odds-ratio forest plot",
                        "role": "relationship",
                        "claim": "Adjusted odds ratios and 95% intervals are plotted.",
                        "evidence_ids": ["table_association_results"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    step = ra.schema.AnalysisStep(
        step_id="04_primary_association_figure",
        intent="Render the manuscript publication figure.",
        method="figure rendering",
    )

    findings = ra.FigureContractQualityValidator().audit(
        step=step,
        out_dir=out_dir,
        run_dir=run_dir,
        step_summary={"rendering_only": True},
    )

    assert any(
        f.severity == "error" and "fallback/rescue" in f.message.lower()
        for f in findings
    ), findings


def test_figure_contract_quality_requires_contract_for_figure_exports(ra, tmp_path: Path):
    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / "04_primary_association_figure" / "outputs"
    out_dir.mkdir(parents=True)
    (out_dir / "effect_estimate_forest.png").write_bytes(b"fake-png")
    step = ra.schema.AnalysisStep(
        step_id="04_primary_association_figure",
        intent="Render the manuscript publication figure.",
        method="figure rendering",
    )

    findings = ra.FigureContractQualityValidator().audit(
        step=step,
        out_dir=out_dir,
        run_dir=run_dir,
        step_summary={"rendering_only": True},
    )

    assert any(
        f.severity == "error" and "without a .figure_contract.json" in f.message
        for f in findings
    ), findings


def test_figure_contract_quality_blocks_single_panel_result_contract(ra, tmp_path: Path):
    contract_path = tmp_path / "easyicu_publication_figure.figure_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "figure_id": "easyicu_publication_figure",
                "core_claim": "Primary effect and robustness range are shown.",
                "panels": [
                    {
                        "panel_id": "A",
                        "title": "Primary effect and robustness variants",
                        "role": "robustness",
                        "claim": "Primary and robustness estimates are plotted.",
                        "evidence_ids": ["robustness_panel"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    findings = ra.FigureContractQualityValidator().audit_contract_file(
        contract_path,
        manuscript_facing=True,
    )

    assert any(
        f.severity == "error" and "only 1 panel" in f.message
        for f in findings
    ), findings


def test_figure_contract_quality_accepts_multipanel_result_contract(ra, tmp_path: Path):
    contract_path = tmp_path / "easyicu_publication_figure.figure_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "figure_id": "easyicu_publication_figure",
                "core_claim": "Primary effect, robustness, and denominator context are shown.",
                "panels": [
                    {
                        "panel_id": "A",
                        "title": "Primary effect and robustness variants",
                        "role": "robustness",
                        "claim": "Primary and variant estimates are shown with intervals.",
                        "evidence_ids": ["robustness_panel"],
                    },
                    {
                        "panel_id": "B",
                        "title": "Variant convergence by axis",
                        "role": "validation",
                        "claim": "Converged and non-converged variants are counted.",
                        "evidence_ids": ["robustness_panel"],
                    },
                    {
                        "panel_id": "C",
                        "title": "Analytic sample-size range",
                        "role": "audit",
                        "claim": "Sample-size ranges are shown for denominator context.",
                        "evidence_ids": ["robustness_panel"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    findings = ra.FigureContractQualityValidator().audit_contract_file(
        contract_path,
        manuscript_facing=True,
    )

    assert not any(f.severity == "error" for f in findings), findings


def test_cohort_auditor_row_count_mismatch(ra, tmp_path: Path):
    df = pd.DataFrame({
        "stay_id": [1, 2, 3, 4, 5],
        "age": [60.0] * 5,
        "death": [0, 1, 0, 1, 0],
    })
    cohort_path = tmp_path / "cohort.parquet"
    df.to_parquet(cohort_path, index=False)
    ctx = ra.build_research_context(
        research_question="x", cohort=df,
        cohort_name="c", database="synthetic", target_outcome="death",
    )
    # Pretend the descriptor was written when there were 99 rows.
    ctx.cohort.n_stays = 99
    findings = ra.CohortAuditor().audit(context=ctx, cohort_path=cohort_path)
    assert any(f.severity == "error" and "row count mismatch" in f.message.lower()
               for f in findings)


def test_cohort_auditor_allows_correlation_context_without_target_outcome(ra, tmp_path: Path):
    df = pd.DataFrame({
        "stay_id": [1, 2, 3],
        "sofa2_max_24h": [1, 3, 5],
        "sofa2_resp_max_24h": [0, 1, 2],
    })
    cohort_path = tmp_path / "cohort.parquet"
    df.to_parquet(cohort_path, index=False)
    ctx = ra.build_research_context(
        research_question="Correlate SOFA components.",
        cohort=df,
        cohort_name="c",
        database="synthetic",
        target_outcome=None,
    )

    findings = ra.CohortAuditor().audit(context=ctx, cohort_path=cohort_path)

    assert not any("Target outcome" in f.message for f in findings)


# ---------------- cohort-hygiene flags (impartial, advisory) -------------

def _hygiene_ctx(ra, df):
    return ra.build_research_context(
        research_question="Does sepsis predict ICU mortality?",
        cohort=df, cohort_name="c", database="synthetic",
        target_outcome="death",
    )


def test_cohort_hygiene_flags_missing_patient_id_when_outcome(ra):
    from easyicu.research_agent.audits.validators import cohort_hygiene_findings

    df = pd.DataFrame({
        "stay_id": [1, 2, 3],
        "los_icu": [2.0, 3.0, 5.0],
        "death": [0, 1, 0],
    })
    findings = cohort_hygiene_findings(df, _hygiene_ctx(ra, df))
    pid = [f for f in findings
           if f.detail.get("subkind") == "patient_independence_unassessable"]
    assert len(pid) == 1
    assert pid[0].severity == "warning"
    assert pid[0].detail["structural_no_source"] is True
    # Advice, not a mandate: it must not assert independence or demand a filter.
    assert "re-extract" in pid[0].message.lower()


def test_cohort_hygiene_no_patient_flag_with_patient_id(ra):
    from easyicu.research_agent.audits.validators import cohort_hygiene_findings

    df = pd.DataFrame({
        "subject_id": [10, 10, 11],
        "stay_id": [1, 2, 3],
        "los_icu": [2.0, 3.0, 5.0],
        "death": [0, 1, 0],
    })
    findings = cohort_hygiene_findings(df, _hygiene_ctx(ra, df))
    assert not any(
        f.detail.get("subkind") == "patient_independence_unassessable"
        for f in findings
    )


def test_cohort_hygiene_no_patient_flag_without_outcome(ra):
    from easyicu.research_agent.audits.validators import cohort_hygiene_findings

    df = pd.DataFrame({"stay_id": [1, 2, 3], "los_icu": [2.0, 3.0, 5.0]})
    ctx = ra.build_research_context(
        research_question="Describe LoS.", cohort=df,
        cohort_name="c", database="synthetic", target_outcome=None,
    )
    findings = cohort_hygiene_findings(df, ctx)
    assert not any(
        f.detail.get("subkind") == "patient_independence_unassessable"
        for f in findings
    )


def test_cohort_hygiene_short_stay_reported_not_enforced(ra):
    from easyicu.research_agent.audits.validators import cohort_hygiene_findings

    df = pd.DataFrame({
        "stay_id": [1, 2, 3, 4],
        "los_icu": [0.2, 0.5, 3.0, 5.0],  # half are <1 day
        "death": [0, 1, 0, 1],
    })
    findings = cohort_hygiene_findings(df, _hygiene_ctx(ra, df))
    short = [f for f in findings
             if f.detail.get("subkind") == "short_stay_exposure"]
    assert len(short) == 1
    assert short[0].severity == "warning"
    assert short[0].detail["fraction_los_under_1_day"] == 0.5
    assert "no minimum-los filter is imposed" in short[0].message.lower()


def test_cohort_hygiene_findings_never_block(ra):
    """Impartiality: hygiene flags are advisory and must never fail-close."""
    from easyicu.research_agent.audits.validators import cohort_hygiene_findings

    df = pd.DataFrame({
        "stay_id": [1, 2, 3],
        "los_icu": [0.1, 0.2, 5.0],
        "death": [0, 1, 0],
    })
    findings = cohort_hygiene_findings(df, _hygiene_ctx(ra, df))
    assert findings  # both flags fire
    assert all(f.severity == "warning" for f in findings)
    assert all(f.detail.get("impartial") is True for f in findings)


def test_cohort_auditor_surfaces_hygiene_flags(ra, tmp_path: Path):
    """The hygiene flags reach callers through CohortAuditor.audit."""
    df = pd.DataFrame({
        "stay_id": [1, 2, 3],
        "los_icu": [0.2, 3.0, 5.0],
        "age": [60.0, 70.0, 80.0],
        "death": [0, 1, 0],
    })
    cohort_path = tmp_path / "cohort.parquet"
    df.to_parquet(cohort_path, index=False)
    ctx = _hygiene_ctx(ra, df)
    findings = ra.CohortAuditor().audit(context=ctx, cohort_path=cohort_path)
    assert any(f.detail.get("kind") == "cohort_hygiene" for f in findings)


def test_llm_concept_auditor_parses_findings(ra):
    from easyicu.research_agent.audits.validators import parse_llm_concept_audit_response

    raw = """```json
{"findings":[{"severity":"warning","message":"ICU mortality may be confused with hospital mortality.","detail":{"column":"death_hosp"}}]}
```"""
    findings = parse_llm_concept_audit_response(raw, step_id="04_primary")
    assert len(findings) == 1
    assert findings[0].validator == "llm_concept_auditor"
    assert findings[0].severity == "warning"
    assert findings[0].detail["step_id"] == "04_primary"


def test_llm_concept_auditor_prompt_includes_outcome_semantics(ra):
    auditor = ra.LLMConceptAuditor(ra.MockLLMClient())
    ctx = ra.build_research_context(
        research_question="Is age associated with ICU mortality?",
        cohort=pd.DataFrame({
            "stay_id": [1, 2, 3],
            "age": [60, 70, 80],
            "death": [0, 1, 0],
        }),
        cohort_name="c",
        database="synthetic",
        target_outcome="death",
    )
    prompt = auditor._prompt(context=ctx, script_text="print('hello')", step=None)
    assert "icu_mortality" in prompt
    assert "explicitly treated as ICU mortality" in prompt
    assert "named `full_stay` window is an administrative analysis span" in prompt
    assert "does not turn" in prompt
    assert "Named time windows:" in prompt


def test_llm_concept_auditor_checks_summary_source_status_bypasses(ra):
    auditor = ra.LLMConceptAuditor(ra.MockLLMClient())
    ctx = ra.build_research_context(
        research_question="Summarize an early ICU measurement.",
        cohort=pd.DataFrame({
            "stay_id": [1, 2, 3],
            "marker_first": [1.0, None, 2.0],
            "marker_n": [1, 0, 1],
            "marker_measured": [1, 0, 1],
        }),
        cohort_name="c",
        database="synthetic",
    )
    prompt = auditor._prompt(context=ctx, script_text="print('hello')", step=None)

    assert "alternate per-stay summaries (first/max/min/mean)" in prompt
    assert "measured/count/source-status" in prompt
    assert "consistency checks" in prompt
    assert "one narrow sparse-event exception" in prompt
    assert "count-zero/flag-zero rows are the reconciled negative class" in prompt
    assert "keeps only `measured == 1` or `count > 0`" in prompt
    assert "applies this exception to a continuous measurement" in prompt
    assert "structurally missing on reconciled negative rows" in prompt
    assert "do not require an explicit zero there" in prompt
    assert "reject any non-binary value" in prompt
    assert "methods.source_status.reconcile_binary_event_presence" in prompt
    assert "unless the script later mutates" in prompt
    assert "fails closed for the whole completed step" in prompt
    assert "must not change its row-level denominator" in prompt


def test_llm_concept_auditor_downgrades_companion_value_gating_false_positive(ra):
    class _CompanionGatingFalsePositiveLLM:
        def complete(self, messages, *, max_tokens=1024, temperature=0.0):
            return json.dumps(
                {
                    "findings": [
                        {
                            "severity": "error",
                            "message": (
                                "First-value covariates can bypass their "
                                "measured/source-status consistency checks."
                            ),
                            "detail": {
                                "context": (
                                    "The script audits measured/count pairs on the "
                                    "original dataframe, but the flags do not mask or "
                                    "invalidate modeled first-value covariates."
                                )
                            },
                        }
                    ]
                }
            )

    script = """
measurement_provenance_audit = {
    'checks': [{'invalid_pair_n': 0, 'discordant_n': 0, 'role': 'audit_only'}]
}
provenance_failed = any(
    row['invalid_pair_n'] or row['discordant_n']
    for row in measurement_provenance_audit['checks']
)
model.fit(frame[['marker_first']], assignment)
"""
    findings = ra.LLMConceptAuditor(
        _CompanionGatingFalsePositiveLLM()
    ).audit(
        context=ra.build_research_context(
            research_question="Model assignment from early physiology.",
            cohort=pd.DataFrame(
                {
                    "stay_id": [1, 2],
                    "marker_first": [1.0, 2.0],
                    "marker_measured": [1, 1],
                    "marker_n": [1, 1],
                }
            ),
            cohort_name="c",
            database="synthetic",
        ),
        script_text=script,
        step=None,
    )

    assert findings[0].severity == "warning"
    assert findings[0].detail["downgraded_reason"]


def test_llm_concept_auditor_preserves_companion_error_without_global_audit(ra):
    class _MissingAuditLLM:
        def complete(self, messages, *, max_tokens=1024, temperature=0.0):
            return json.dumps(
                {
                    "findings": [
                        {
                            "severity": "error",
                            "message": (
                                "First-value covariates can bypass their "
                                "measured/source-status consistency checks."
                            ),
                            "detail": {
                                "context": (
                                    "The measured flags are not used to mask the "
                                    "first-value summary and no global audit exists."
                                )
                            },
                        }
                    ]
                }
            )

    context = ra.build_research_context(
        research_question="Model assignment from early physiology.",
        cohort=pd.DataFrame(
            {
                "stay_id": [1, 2],
                "marker_first": [1.0, 2.0],
                "marker_measured": [1, 1],
                "marker_n": [1, 1],
            }
        ),
        cohort_name="c",
        database="synthetic",
    )
    findings = ra.LLMConceptAuditor(_MissingAuditLLM()).audit(
        context=context,
        script_text="model.fit(frame[['marker_first']], assignment)",
        step=None,
    )

    assert findings[0].severity == "error"


def test_llm_concept_auditor_prioritizes_late_declared_input_companions(ra):
    from easyicu.research_agent.schema import VariableRole

    variables = [
        ra.ConceptDescriptor(name=f"filler_{index}", dtype="float64")
        for index in range(85)
    ]
    variables.extend(
        [
            ra.ConceptDescriptor(
                name="event_n",
                dtype="int64",
                role=VariableRole.META,
            ),
            ra.ConceptDescriptor(
                name="event_measured",
                dtype="int64",
                role=VariableRole.META,
            ),
            ra.ConceptDescriptor(
                name="event_max",
                dtype="float64",
                role=VariableRole.INTERVENTION,
                clinical_caveats=["Registered sparse event indicator."],
            ),
        ]
    )
    context = ra.ResearchContext(
        research_question="Evaluate a declared binary event exposure.",
        cohort=ra.CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_patients=2,
            n_stays=2,
        ),
        variables=variables,
    )
    step = ra.AnalysisStep(
        step_id="event_protocol",
        intent="Validate the declared event exposure.",
        inputs=["event_n"],
        expected_outputs=["table:protocol"],
        method="target_trial_emulation_protocol",
    )

    prompt = ra.LLMConceptAuditor(ra.MockLLMClient())._prompt(
        context=context,
        script_text="event_max = frame['event_max']",
        step=step,
    )

    assert '"name": "event_n"' in prompt
    assert '"name": "event_measured"' in prompt
    assert '"name": "event_max"' in prompt
    assert '"role": "intervention"' in prompt
    assert "Registered sparse event indicator." in prompt


def test_llm_concept_auditor_sees_late_independent_count_qc(ra):
    auditor = ra.LLMConceptAuditor(ra.MockLLMClient())
    ctx = ra.build_research_context(
        research_question="Audit an ordered early ICU exposure.",
        cohort=pd.DataFrame({
            "stay_id": [1, 2, 3],
            "stage_max": [0, 1, 2],
            "stage_n": [1, 1, 1],
            "stage_measured": [1, 1, 1],
        }),
        cohort_name="c",
        database="synthetic",
    )
    late_qc = "count_consistency_status = 'checked'"
    script = ("# analysis setup\n" * 1000) + late_qc
    prompt = auditor._prompt(context=ctx, script_text=script, step=None)

    assert late_qc in prompt
    assert "independent QC fields" in prompt
    assert "keeps that comparison audit-only" in prompt


def test_llm_concept_auditor_samples_head_middle_and_tail_of_long_script(ra):
    auditor = ra.LLMConceptAuditor(ra.MockLLMClient())
    ctx = ra.build_research_context(
        research_question="Audit a long generated ICU analysis script.",
        cohort=pd.DataFrame({"stay_id": [1], "death": [0]}),
        cohort_name="c",
        database="synthetic",
    )
    head = "HEAD_SENTINEL\n" + ("h" * 50_000)
    middle = "MIDDLE_MODEL_SENTINEL\n"
    tail = ("t" * 50_000) + "\nTAIL_CONTRACT_SENTINEL"

    prompt = auditor._prompt(
        context=ctx,
        script_text=head + middle + tail,
        step=None,
    )

    assert "HEAD_SENTINEL" in prompt
    assert "MIDDLE_MODEL_SENTINEL" in prompt
    assert "TAIL_CONTRACT_SENTINEL" in prompt
    assert "concept-audit excerpt omitted" in prompt


def test_llm_concept_auditor_makes_result_changing_figure_semantics_errors(ra):
    auditor = ra.LLMConceptAuditor(ra.MockLLMClient())
    ctx = ra.build_research_context(
        research_question="Render an ordered exposure distribution.",
        cohort=pd.DataFrame({"stay_id": [1, 2], "stage_max": [0, 1]}),
        cohort_name="c",
        database="synthetic",
    )
    prompt = auditor._prompt(context=ctx, script_text="plot(stage_max)", step=None)

    assert "severity='error'" in prompt
    assert "silently invent numeric zeros" in prompt
    assert "reconciling them to counts and denominators" in prompt
    assert "alternate per-stay summary" in prompt


def test_llm_concept_auditor_downgrades_nonblocking_outcome_confusion(ra):
    from easyicu.research_agent.audits.validators import parse_llm_concept_audit_response

    raw = """
    {
      "findings": [
        {
          "severity": "error",
          "message": "ICU vs hospital mortality confusion",
          "detail": {
            "issue": "Explicitly noted that 'death' is ICU mortality, but the script does not verify or enforce consistent usage across all downstream analyses or reporting."
          }
        }
      ]
    }
    """
    findings = parse_llm_concept_audit_response(raw, step_id="02_model")
    assert len(findings) == 1
    assert findings[0].severity == "warning"


def test_llm_concept_auditor_uses_context_to_downgrade_outcome_ambiguity(ra):
    class _FalsePositiveLLM:
        def complete(self, messages, *, max_tokens=1024, temperature=0.0):
            return """
            {
              "findings": [
                {
                  "severity": "error",
                  "message": "ICU vs hospital mortality confusion",
                  "detail": {
                    "context": "The script uses death without clarifying whether it is ICU, hospital, or 28-day mortality."
                  }
                }
              ]
            }
            """

    ctx = ra.build_research_context(
        research_question="Is early lactate associated with ICU mortality?",
        cohort=pd.DataFrame({
            "stay_id": [1, 2, 3],
            "lactate_max_24h": [1.0, 2.0, 3.0],
            "death": [0, 1, 0],
        }),
        cohort_name="c",
        database="synthetic",
        target_outcome="death",
    )
    findings = ra.LLMConceptAuditor(_FalsePositiveLLM()).audit(
        context=ctx,
        script_text="model.fit(df[['lactate_max_24h']], df['death'])",
        step=None,
    )

    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert findings[0].detail["downgraded_reason"]


def test_llm_concept_auditor_preserves_error_for_conflicting_outcome_label(ra):
    class _ConfusionLLM:
        def complete(self, messages, *, max_tokens=1024, temperature=0.0):
            return """
            {
              "findings": [
                {
                  "severity": "error",
                  "message": "ICU vs hospital mortality confusion",
                  "detail": {"context": "The plot labels ICU death as hospital mortality."}
                }
              ]
            }
            """

    ctx = ra.build_research_context(
        research_question="Is early lactate associated with ICU mortality?",
        cohort=pd.DataFrame({
            "stay_id": [1, 2, 3],
            "lactate_max_24h": [1.0, 2.0, 3.0],
            "death": [0, 1, 0],
        }),
        cohort_name="c",
        database="synthetic",
        target_outcome="death",
    )
    findings = ra.LLMConceptAuditor(_ConfusionLLM()).audit(
        context=ctx,
        script_text="ax.set_title('Adjusted association with hospital mortality')",
        step=None,
    )

    assert len(findings) == 1
    assert findings[0].severity == "error"


class _HorizonMismatchLLM:
    def complete(self, messages, *, max_tokens=1024, temperature=0.0):
        return """
        {
          "findings": [
            {
              "severity": "error",
              "message": "The fixed-window death alternative is incompatible with the bound hospital-mortality outcome.",
              "detail": {
                "context": "The script copies a 0–720 hour window but consumes the hospital mortality flag without deriving 30-day mortality from event time."
              }
            }
          ]
        }
        """


def _hospital_mortality_context(ra):
    return ra.build_research_context(
        research_question="Is an early exposure associated with in-hospital mortality?",
        cohort=pd.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "exposure": [0.0, 1.0, 2.0],
                "death": [0, 1, 0],
            }
        ),
        cohort_name="c",
        database="synthetic",
        target_outcome="death",
    )


def _full_stay_hospital_mortality_script(extra: str = "") -> str:
    return (
        "OUTCOME_OVERRIDE = {\n"
        "    'concept_id': 'death',\n"
        "    'time_window': {\n"
        "        'anchor': 'icu_admit',\n"
        "        'start_offset_hours': 0.0,\n"
        "        'end_offset_hours': 720.0,\n"
        "    },\n"
        "    'aggregation': 'first',\n"
        "    'op': '==',\n"
        "    'value': 1,\n"
        "}\n"
        "y = df['death']\n"
        "model.fit(x, y)\n"
        + extra
    )


def test_llm_concept_auditor_downgrades_named_full_stay_horizon_false_positive(
    ra,
) -> None:
    findings = ra.LLMConceptAuditor(_HorizonMismatchLLM()).audit(
        context=_hospital_mortality_context(ra),
        script_text=_full_stay_hospital_mortality_script(),
        step=None,
    )

    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert "full_stay administrative window" in findings[0].detail[
        "downgraded_reason"
    ]


@pytest.mark.parametrize(
    "conflicting_code",
    [
        "label = 'ICU mortality'\n",
        "label = '28-day mortality'\n",
        "label = '30-day mortality'\n",
        "label = 'fixed-horizon mortality'\n",
        "alternate = df['death_30d']\n",
        "columns = ['death_30d']\nalternate = df[columns]\n",
        "derived = df['death_time'].le(720)\n",
        "derived = (df['death'] == 1) & (df['los_icu'] <= 30)\n",
    ],
)
def test_llm_concept_auditor_preserves_horizon_error_for_real_conflicts(
    ra, conflicting_code
) -> None:
    findings = ra.LLMConceptAuditor(_HorizonMismatchLLM()).audit(
        context=_hospital_mortality_context(ra),
        script_text=_full_stay_hospital_mortality_script(conflicting_code),
        step=None,
    )

    assert len(findings) == 1
    assert findings[0].severity == "error"


def test_clinical_constraint_validator_warns_on_missing_time_zero(ra, tmp_path: Path):
    ctx = _ctx_with_sofa(ra).model_copy(
        update={
            "research_question": "Estimate the effect of early vasopressor treatment on death.",
            "user_preferences": ra.schema.UserPreferences(
                inferred_analysis_family="causal_inference"
            ),
        }
    )
    step = ra.schema.AnalysisStep(
        step_id="04_causal_protocol",
        intent="Target-trial style causal analysis",
        method="target_trial_emulation",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    findings = ra.ClinicalConstraintValidator().audit(
        context=ctx,
        step=step,
        out_dir=out_dir,
        step_summary={},
    )
    assert any("immortal time bias" in f.message.lower() for f in findings), findings


def test_clinical_constraint_validator_does_not_flag_prediction_feature_list_as_treatment_effect(
    ra, tmp_path: Path
):
    ctx = _ctx_with_sofa(ra).model_copy(
        update={
            "research_question": (
                "Build a mortality prediction workflow using age, sex, SOFA-2, "
                "lactate, MAP, and vasopressor exposure."
            ),
            "user_preferences": ra.schema.UserPreferences(
                inferred_analysis_family="prediction_model"
            ),
        }
    )
    step = ra.schema.AnalysisStep(
        step_id="01_model_training",
        intent="Train and validate the mortality prediction model with AUROC and calibration.",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    findings = ra.ClinicalConstraintValidator().audit(
        context=ctx,
        step=step,
        out_dir=out_dir,
        step_summary={"statistic:auroc": 0.8, "statistic:brier_score": 0.18},
    )
    assert not any("immortal time bias" in f.message.lower() for f in findings), findings


def test_clinical_constraint_validator_does_not_flag_association_named_exposure(
    ra, tmp_path: Path
):
    ctx = _ctx_with_sofa(ra).model_copy(
        update={
            "research_question": "Is Sepsis-3 status associated with ICU mortality?",
            "user_preferences": ra.schema.UserPreferences(
                inferred_analysis_family="association"
            ),
        }
    )
    step = ra.schema.AnalysisStep(
        step_id="03b_dataset_validation",
        intent="Validate the modeling dataset and named exposure before regression.",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    findings = ra.ClinicalConstraintValidator().audit(
        context=ctx,
        step=step,
        out_dir=out_dir,
        step_summary={
            "named_exposure": "sepsis3",
            "method": "post_audit_modeling_dataset_validation_and_repair",
        },
    )
    assert not any("immortal time bias" in f.message.lower() for f in findings), findings


def test_clinical_constraint_validator_does_not_treat_ordinal_dose_response_as_treatment(
    ra, tmp_path: Path
):
    ctx = _ctx_with_sofa(ra).model_copy(
        update={
            "research_question": (
                "Characterise the dose-response gradient of an ordered organ-"
                "dysfunction stage against mortality."
            ),
            "user_preferences": ra.schema.UserPreferences(),
        }
    )
    step = ra.schema.AnalysisStep(
        step_id="02_data_quality",
        intent="Audit missingness and measurement availability before modelling.",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    findings = ra.ClinicalConstraintValidator().audit(
        context=ctx,
        step=step,
        out_dir=out_dir,
        step_summary={"analysis_family": "data_quality"},
    )
    assert not any("immortal time bias" in f.message.lower() for f in findings), findings


def test_clinical_constraint_validator_does_not_flag_negated_noncausal_association(
    ra, tmp_path: Path
):
    ctx = _ctx_with_sofa(ra).model_copy(
        update={
            "research_question": "Is an ordered exposure associated with mortality?",
            "user_preferences": None,
        }
    )
    step = ra.schema.AnalysisStep(
        step_id="06_adjusted_association",
        intent="Estimate supportive adjusted associations.",
        method="adjusted_association_models",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    findings = ra.ClinicalConstraintValidator().audit(
        context=ctx,
        step=step,
        out_dir=out_dir,
        step_summary={
            "analysis_family": "association_study",
            "notes": [
                "The analysis is observational and supportive, not a causal treatment effect."
            ],
        },
    )

    assert not any("immortal time bias" in f.message.lower() for f in findings)


def test_clinical_constraint_validator_does_not_flag_support_step_in_causal_study(
    ra, tmp_path: Path
):
    ctx = _ctx_with_sofa(ra).model_copy(
        update={
            "research_question": "Estimate a treatment effect on mortality.",
            "user_preferences": ra.schema.UserPreferences(
                inferred_analysis_family="causal_inference"
            ),
        }
    )
    step = ra.schema.AnalysisStep(
        step_id="02_baseline_table",
        intent="Describe baseline characteristics before the causal model.",
        method="table_one",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    findings = ra.ClinicalConstraintValidator().audit(
        context=ctx,
        step=step,
        out_dir=out_dir,
        step_summary={"analysis_family": "descriptive"},
    )

    assert not any("immortal time bias" in f.message.lower() for f in findings)


def test_clinical_constraint_validator_accepts_causal_step_with_explicit_time_zero(
    ra, tmp_path: Path
):
    ctx = _ctx_with_sofa(ra).model_copy(
        update={
            "user_preferences": ra.schema.UserPreferences(
                inferred_analysis_family="causal_inference",
                timing_and_design=(
                    "Eligibility and treatment assignment are aligned at ICU admission time zero."
                ),
            )
        }
    )
    step = ra.schema.AnalysisStep(
        step_id="04_causal_protocol",
        intent="Estimate the target-trial effect.",
        method="target_trial_emulation",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    findings = ra.ClinicalConstraintValidator().audit(
        context=ctx,
        step=step,
        out_dir=out_dir,
        step_summary={"analysis_family": "causal_inference"},
    )

    assert not any("immortal time bias" in f.message.lower() for f in findings)


def test_statistical_guard_warns_when_prediction_outputs_lack_split_metadata(ra, tmp_path: Path):
    ctx = _ctx_with_sofa(ra).model_copy(
        update={
            "user_preferences": ra.schema.UserPreferences(
                inferred_analysis_family="prediction_model",
                covariates=["age", "sex", "sofa2"],
            ),
        }
    )
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({
        "stay_id": list(range(1, 11)),
        "age": [60] * 10,
        "sofa2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "death": [0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    }).to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pd.DataFrame({
        "model": ["logit"],
        "auc": [0.76],
        "brier": [0.18],
    }).to_csv(out_dir / "model_performance_train_test.csv", index=False)
    step = ra.schema.AnalysisStep(step_id="04_prediction", intent="prediction model analysis")
    findings = ra.StatisticalGuard().audit(
        context=ctx,
        cohort_path=cohort_path,
        step=step,
        out_dir=out_dir,
        step_summary={},
    )
    messages = " ".join(f.message.lower() for f in findings)
    assert "train/test split" in messages
    assert "calibration_slope" in messages


def test_statistical_guard_accepts_v14_cv_prediction_summary(ra, tmp_path: Path):
    ctx = _ctx_with_sofa(ra).model_copy(
        update={
            "user_preferences": ra.schema.UserPreferences(
                inferred_analysis_family="prediction_model",
                covariates=["age", "sex", "sofa2"],
            ),
        }
    )
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({
        "stay_id": list(range(1, 41)),
        "age": [60] * 40,
        "sofa2": list(range(10)) * 4,
        "death": [0, 1] * 20,
    }).to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    step = ra.schema.AnalysisStep(step_id="01_model_training", intent="prediction model analysis")

    findings = ra.StatisticalGuard().audit(
        context=ctx,
        cohort_path=cohort_path,
        step=step,
        out_dir=out_dir,
        step_summary={
            "statistic:cv_auroc_mean": 0.74,
            "statistic:cv_brier_mean": 0.18,
            "cv_folds": 5,
            "split_strategy": "5-fold cross-validation",
        },
    )

    messages = " ".join(f.message.lower() for f in findings)
    assert "held-out performance" not in messages
    assert "train/test split" not in messages
    assert "calibration_slope" not in messages


def test_statistical_guard_ignores_empty_p_value_placeholder_column(
    ra, tmp_path: Path
):
    ctx = _ctx_with_sofa(ra).model_copy(
        update={
            "user_preferences": ra.schema.UserPreferences(
                inferred_analysis_family="association"
            )
        }
    )
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"death": [0, 1, 0]}).to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pd.DataFrame(
        {
            "term": ["exposure", "age", "sex"],
            "p_value": [None, None, None],
        }
    ).to_csv(out_dir / "coefficients.csv", index=False)

    findings = ra.StatisticalGuard().audit(
        context=ctx,
        cohort_path=cohort_path,
        step=ra.schema.AnalysisStep(
            step_id="05_association", intent="association analysis"
        ),
        out_dir=out_dir,
        step_summary={},
    )

    assert not any("multiple p-values" in finding.message for finding in findings)


def test_statistical_guard_warns_for_multiple_finite_unadjusted_p_values(
    ra, tmp_path: Path
):
    ctx = _ctx_with_sofa(ra).model_copy(
        update={
            "user_preferences": ra.schema.UserPreferences(
                inferred_analysis_family="association"
            )
        }
    )
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"death": [0, 1, 0]}).to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pd.DataFrame(
        {
            "term": ["level_1", "level_2", "level_3"],
            "term_role": ["exposure", "exposure", "exposure"],
            "analysis_role": ["primary", "primary", "primary"],
            "hypothesis_family_id": ["prespecified_contrasts"] * 3,
            "p_value": [0.01, 0.03, None],
        }
    ).to_csv(out_dir / "coefficients.csv", index=False)

    findings = ra.StatisticalGuard().audit(
        context=ctx,
        cohort_path=cohort_path,
        step=ra.schema.AnalysisStep(
            step_id="05_association", intent="association analysis"
        ),
        out_dir=out_dir,
        step_summary={},
    )

    warning = next(
        finding for finding in findings if "multiple p-values" in finding.message
    )
    assert warning.detail["finite_p_value_count"] == 2
    assert warning.detail["hypothesis_family_id"] == "prespecified_contrasts"


def test_statistical_guard_does_not_warn_for_untyped_coefficient_dump(
    ra, tmp_path: Path
):
    ctx = _ctx_with_sofa(ra)
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"death": [0, 1, 0]}).to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pd.DataFrame(
        {
            "term": ["intercept", "age", "exposure_1", "exposure_2"],
            "term_role": ["intercept", "adjustment", "exposure", "exposure"],
            "analysis_role": ["primary", "primary", "primary", "sensitivity"],
            "p_value": [0.2, 0.03, 0.01, 0.02],
        }
    ).to_csv(out_dir / "coefficients.csv", index=False)

    findings = ra.StatisticalGuard().audit(
        context=ctx,
        cohort_path=cohort_path,
        step=ra.schema.AnalysisStep(
            step_id="05_association", intent="association analysis"
        ),
        out_dir=out_dir,
        step_summary={},
    )

    assert not any("multiple p-values" in finding.message for finding in findings)


def test_statistical_guard_scopes_typed_family_to_primary_result_terms(
    ra, tmp_path: Path
):
    ctx = _ctx_with_sofa(ra)
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"death": [0, 1, 0]}).to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pd.DataFrame(
        {
            "term": ["intercept", "age", "exposure_primary", "exposure_sensitivity"],
            "term_role": ["intercept", "adjustment", "exposure", "exposure"],
            "analysis_role": ["primary", "primary", "primary", "sensitivity"],
            "hypothesis_family_id": ["family_a"] * 4,
            "p_value": [0.2, 0.03, 0.01, 0.02],
        }
    ).to_csv(out_dir / "coefficients.csv", index=False)

    findings = ra.StatisticalGuard().audit(
        context=ctx,
        cohort_path=cohort_path,
        step=ra.schema.AnalysisStep(
            step_id="05_association", intent="association analysis"
        ),
        out_dir=out_dir,
        step_summary={},
    )

    assert not any("multiple p-values" in finding.message for finding in findings)


# ---------------------------------------------------------------------------
# Degenerate-partition disclosure caution (clustering / trajectory)
# ---------------------------------------------------------------------------


def _cluster_sizes_dir(tmp_path: Path, sizes) -> Path:
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    total = float(sum(sizes))
    pd.DataFrame({
        "cluster": list(range(len(sizes))),
        "n": sizes,
        "pct": [s / total * 100.0 for s in sizes],
    }).to_csv(out / "cluster_sizes.csv", index=False)
    return out


def test_statistical_validator_flags_degenerate_partition(ra, tmp_path: Path):
    # The M3 scenario: a "2-cluster solution" that is really 99.5% / 0.5%.
    # silhouette/ARI on such a split are inflated by outlier isolation, so the
    # agent must be cautioned to disclose the size imbalance.
    ctx = _ctx_with_sofa(ra)
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}).to_parquet(cohort, index=False)
    out_dir = _cluster_sizes_dir(tmp_path, [38584, 203])
    step = ra.schema.AnalysisStep(
        step_id="01_phenotype_trajectory_clustering", intent="subphenotype clustering"
    )
    findings = ra.StatisticalValidator().audit(
        context=ctx, cohort_path=cohort, step=step, out_dir=out_dir,
        step_summary={"silhouette": 0.808, "mean_ari": 1.0},
    )
    deg = [f for f in findings if "degenerate" in f.message.lower()]
    assert deg, findings
    assert all(f.severity == "warning" for f in deg)  # never blocks honest reporting
    assert deg[0].detail["min_cluster_fraction"] < 0.01


def test_statistical_validator_silent_on_balanced_partition(ra, tmp_path: Path):
    # A genuinely separated partition must NOT be flagged — the rule layer only
    # surfaces objective degeneracy, never imposes a "good enough" threshold.
    ctx = _ctx_with_sofa(ra)
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}).to_parquet(cohort, index=False)
    out_dir = _cluster_sizes_dir(tmp_path, [12000, 9000, 7500, 6000])
    step = ra.schema.AnalysisStep(
        step_id="01_phenotype_trajectory_clustering", intent="subphenotype clustering"
    )
    findings = ra.StatisticalValidator().audit(
        context=ctx, cohort_path=cohort, step=step, out_dir=out_dir, step_summary={},
    )
    assert not [f for f in findings if "degenerate" in f.message.lower()]


def test_statistical_validator_degeneracy_silent_without_cluster_evidence(ra, tmp_path: Path):
    # Absence of a cluster-size distribution is not degeneracy: a non-clustering
    # step must never trip this caution.
    ctx = _ctx_with_sofa(ra)
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}).to_parquet(cohort, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "primary_association.csv").write_text("variable,odds_ratio\nage,1.1\n", encoding="utf-8")
    step = ra.schema.AnalysisStep(step_id="04_primary_association", intent="association")
    findings = ra.StatisticalValidator().audit(
        context=ctx, cohort_path=cohort, step=step, out_dir=out_dir, step_summary={},
    )
    assert not [f for f in findings if "degenerate" in f.message.lower()]


def test_statistical_validator_flags_single_group_partition(ra, tmp_path: Path):
    ctx = _ctx_with_sofa(ra)
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}).to_parquet(cohort, index=False)
    out_dir = _cluster_sizes_dir(tmp_path, [38787])
    step = ra.schema.AnalysisStep(
        step_id="01_phenotype_trajectory_clustering", intent="subphenotype clustering"
    )
    findings = ra.StatisticalValidator().audit(
        context=ctx, cohort_path=cohort, step=step, out_dir=out_dir, step_summary={},
    )
    deg = [f for f in findings if "single-group" in f.message.lower() or "degenerate" in f.message.lower()]
    assert deg and all(f.severity == "warning" for f in deg)
