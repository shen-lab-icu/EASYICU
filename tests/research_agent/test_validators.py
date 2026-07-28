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
from easyicu.research_agent.schema import AnalysisStep, ValidationFinding
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient


def _offline_concept_auditor(ra, fixed_client):  # noqa: ANN001
    """Run legacy fixed-response fixtures through the reviewed mock type."""

    try:
        response = fixed_client.complete([], max_tokens=1024, temperature=0.0)
    except Exception as exc:  # fixed provider-failure fixture
        response = exc
    return ra.LLMConceptAuditor(ScriptedMockLLMClient([response]))


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
    failed = _prior_cohort_record(cohort_n=74_829, step_id="04_failed_reconciliation")
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
    assert (
        "exposure_outcome_summary.csv"
        in findings[0].detail["registered_table_artifacts"]
    )


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


def test_cross_step_registered_output_does_not_promote_unregistered_outputs() -> None:
    prior = _prior_registered_table_record()
    prior["evidence_ids"] = ["statistic_step_summary_12345678"]
    prior["step_summary"].pop("output_files")
    prior["step_summary"]["outputs"] = {
        "guessed_table": {
            "file": "not_a_registered_product.csv",
        }
    }

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
            {"observed_fraction": {"by_group": [{"group": "a", "value": 40.0}]}},
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
        step_summary={"overall_risk": {"risk": 0.2, "bootstrap_replicates": 1_000}},
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
        {"observed_fraction": {"summary": {"group_a": 0.2, "group_b": 40.0}}},
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
    assert findings[0].detail["summary_path"] == ("valid_level_distribution_percent")


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
                "registered_event_n_field": ("event_n" if correct else "n_positive"),
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
    ).to_csv(out_dir / "absolute_risk_representation_reconciliation.csv", index=False)

    findings = CrossStepReconciliationTraceValidator().audit(
        step=AnalysisStep(step_id="04_reconciliation", intent="Reconcile parent."),
        step_summary=summary,
        out_dir=out_dir,
    )

    assert len(findings) == 1
    issues = findings[0].detail["issues"]
    assert (
        sum(item["issue"] == "supported_parent_row_reported_missing" for item in issues)
        == 3
    )


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
    ).to_csv(out_dir / "absolute_risk_representation_reconciliation.csv", index=False)

    findings = CrossStepReconciliationTraceValidator().audit(
        step=AnalysisStep(step_id="04_reconciliation", intent="Reconcile parent."),
        step_summary=summary,
        out_dir=out_dir,
    )

    assert len(findings) == 1
    issues = findings[0].detail["issues"]
    assert (
        sum(item["issue"] == "supported_parent_row_reported_missing" for item in issues)
        == 3
    )


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
    ).to_csv(out_dir / "absolute_risk_representation_reconciliation.csv", index=False)

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
    ).to_csv(out_dir / "absolute_risk_representation_reconciliation.csv", index=False)

    findings = CrossStepReconciliationTraceValidator().audit(
        step=AnalysisStep(step_id="04_reconciliation", intent="Reconcile parent."),
        step_summary=summary,
        out_dir=out_dir,
    )

    assert len(findings) == 1
    issues = findings[0].detail["issues"]
    assert (
        sum(item["issue"] == "supported_parent_row_reported_missing" for item in issues)
        == 2
    )


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
    ).to_csv(out_dir / "absolute_risk_representation_reconciliation.csv", index=False)

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
    ).to_csv(out_dir / "absolute_risk_representation_reconciliation.csv", index=False)

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


def test_cross_step_source_status_locks_top_level_quality_summary() -> None:
    prior = {
        "step_id": "02_value_quality",
        "status": "ok",
        "step_summary": {
            "primary_exposure": {"column": "lab_max"},
            "source_status_counts": {
                "valid observed": 41_210,
                "no source": 53_246,
                "measured/source present but summary missing": 0,
                "contradictory/invalid": 2,
            },
        },
    }
    current = {
        "primary_exposure": "lab_max",
        "source_status_schema": {
            "valid observed": 41_210,
            "no source": 0,
            "measured/source present but summary missing": 53_246,
            "contradictory/invalid": 2,
        },
    }

    findings = CrossStepSourceStatusValidator().audit(
        step=AnalysisStep(step_id="03_distribution", intent="Describe values."),
        step_summary=current,
        completed_step_records=[prior],
    )

    # Matching valid-observed totals alone are insufficient: the complete
    # four-role mapping must remain stable, including no-source vs measured-
    # source-present-but-summary-missing.
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].detail["reported_status_counts"]["no_source"] == 0
    assert findings[0].detail["expected_status_counts"]["no_source"] == 53_246


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
    df = pd.DataFrame(
        {
            "stay_id": list(range(1, 11)),
            "age": [60, 70, 50, 80, 65, 75, 90, 40, 55, 60],
            "sofa2": [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            "lact": [1.0, 2.0, 1.5, 3.0, 4.0, 5.0, 2.5, 1.2, 3.3, 7.0],
            "death": [1, 1, 0, 0, 0, 1, 1, 0, 1, 1],
        }
    )
    return ra.build_research_context(
        research_question="sofa2 → death?",
        cohort=df,
        cohort_name="t",
        database="synthetic",
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
        f
        for f in findings
        if f.validator == auditor.name
        and ("sofa" in f.message.lower() or "ordinal" in f.message.lower())
    ]
    assert matched, findings
    assert all(f.severity == "warning" for f in matched), matched
    # ...and no forbidden-aggregation finding is escalated to a blocking error.
    assert not any(
        f.severity == "error" and "misleading" in f.message.lower() for f in findings
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
    assert any(
        f.severity == "warning" and "lact" in f.message.lower() for f in findings
    )


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
    assert any(
        "fillna" in f.message.lower() or "imputation" in f.message.lower()
        for f in findings
    )


def test_primary_result_rejects_exposure_or_outcome_imputation(ra):
    ctx = _ctx_with_sofa(ra).model_copy(update={"primary_exposure": "lact"})
    step = ra.schema.AnalysisStep(
        step_id="04_primary_association",
        planned_analysis_role="primary",
        intent="Fit the primary adjusted association.",
    )
    code = """
primary_predictor = "lact"
outcome_col = "death"
df[primary_predictor] = df[primary_predictor].fillna(df[primary_predictor].median())
df[outcome_col] = df[outcome_col].ffill()
"""
    findings = ra.AnalysisPatternAuditor().audit(
        context=ctx,
        script_text=code,
        step=step,
    )

    protected = [
        finding
        for finding in findings
        if finding.detail.get("kind") == "primary_estimand_imputation"
    ]
    assert len(protected) == 1
    assert protected[0].severity == "error"
    assert set(protected[0].detail["columns"]) == {"lact", "death"}


def test_sensitivity_result_may_explicitly_handle_exposure_missingness(ra):
    ctx = _ctx_with_sofa(ra).model_copy(update={"primary_exposure": "lact"})
    step = ra.schema.AnalysisStep(
        step_id="05_missingness_sensitivity",
        planned_analysis_role="sensitivity",
        intent="Run an explicitly declared missingness sensitivity analysis.",
    )
    findings = ra.AnalysisPatternAuditor().audit(
        context=ctx,
        script_text='df["lact"] = df["lact"].fillna(df["lact"].median())',
        step=step,
    )

    assert not any(
        finding.detail.get("kind") == "primary_estimand_imputation"
        for finding in findings
    )


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
    assert not any(
        "fillna" in f.message.lower() or "imputation" in f.message.lower()
        for f in findings
    )


def test_concept_usage_flags_agg_mean_of_sofa(ra):
    # Detection still fires across call forms; severity is advisory (warning).
    ctx = _ctx_with_sofa(ra)
    findings = ra.ConceptUsageAuditor().audit(
        context=ctx,
        script_text='x = df["sofa2"].agg("mean")',
    )
    assert any(
        f.severity == "warning" and "sofa" in f.message.lower() for f in findings
    )


def test_concept_usage_flags_numpy_mean_of_sofa(ra):
    ctx = _ctx_with_sofa(ra)
    findings = ra.ConceptUsageAuditor().audit(
        context=ctx,
        script_text='import numpy as np\nx = np.mean(df["sofa2"])',
    )
    assert any(
        f.severity == "warning" and "sofa" in f.message.lower() for f in findings
    )


def test_concept_usage_flags_rolling_mean_of_sofa(ra):
    ctx = _ctx_with_sofa(ra)
    findings = ra.ConceptUsageAuditor().audit(
        context=ctx,
        script_text='x = df["sofa2"].rolling(3).mean()',
    )
    assert any(
        f.severity == "warning" and "sofa" in f.message.lower() for f in findings
    )


def test_statistical_validator_flags_outcome_mismatch(ra, tmp_path: Path):
    ctx = _ctx_with_sofa(ra)
    cohort_path = tmp_path / "cohort.parquet"
    pd.read_parquet  # touch import
    df = pd.DataFrame(
        {
            "stay_id": list(range(1, 11)),
            "age": [60] * 10,
            "sofa2": [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            "lact": [1.0] * 10,
            "death": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 0.2 incidence
        }
    )
    df.to_parquet(cohort_path, index=False)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # write a placeholder so out_dir is non-empty
    (out_dir / "step_summary.json").write_text("{}", encoding="utf-8")

    schema = ra.schema
    step = schema.AnalysisStep(
        step_id="02_outcome_incidence",
        intent="incidence",
        expected_outputs=["statistic:outcome_rate"],
    )
    validator = ra.StatisticalValidator()
    findings = validator.audit(
        context=ctx,
        cohort_path=cohort_path,
        step=step,
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
    df = pd.DataFrame(
        {
            "stay_id": list(range(1, 21)),
            "age": [60] * 20,
            "sofa2": [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5,
            "death": [
                1,
                1,
                1,
                1,
                0,  # rate at 0 = 0.8
                0,
                0,
                0,
                0,
                0,  # rate at 1 = 0.0
                0,
                0,
                0,
                1,
                0,  # rate at 2 = 0.2
                1,
                1,
                1,
                1,
                1,
            ],  # rate at 3 = 1.0
        }
    )
    df.to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pd.DataFrame(
        {
            "variable": ["sofa2"],
            "n_rows": [20],
            "n_low_completeness": [5],
            "frac_low_completeness": [0.25],
        }
    ).to_csv(out_dir / "component_completeness_qc.csv", index=False)

    schema = ra.schema
    step = schema.AnalysisStep(
        step_id="05_component_completeness_qc",
        intent="component completeness QC",
    )
    validator = ra.StatisticalValidator()
    findings = validator.audit(
        context=ctx,
        cohort_path=cohort_path,
        step=step,
        out_dir=out_dir,
        step_summary={},
    )
    assert not any(
        "non-monotonic" in f.message.lower() or "exceeds" in f.message.lower()
        for f in findings
    ), findings


def test_statistical_validator_no_artefacts_is_error(ra, tmp_path: Path):
    ctx = _ctx_with_sofa(ra)
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {"stay_id": [1], "age": [60], "sofa2": [3], "lact": [1.0], "death": [0]}
    ).to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()  # deliberately empty
    schema = ra.schema
    step = schema.AnalysisStep(step_id="99_empty", intent="x")
    findings = ra.StatisticalValidator().audit(
        context=ctx,
        cohort_path=cohort_path,
        step=step,
        out_dir=out_dir,
        step_summary={},
    )
    assert any(
        f.severity == "error" and "no output artefacts" in f.message.lower()
        for f in findings
    ), findings


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


def test_statistical_validator_accepts_reconciled_primary_exposure(ra, tmp_path: Path):
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


def test_statistical_validator_blocks_fail_closed_exposure_without_cohort_n(
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
            "primary_exposure": {
                "reconciliation_status": "failed_closed",
                "available_n": 0,
            }
        },
    )

    assert any(
        finding.severity == "error"
        and "no cohort row has a usable reconciled exposure" in finding.message
        for finding in findings
    ), findings


def test_statistical_validator_blocks_all_unavailable_counts_without_cohort_n(
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
            "primary_exposure": {
                "reconciliation_status": "checked",
                "missing_n": 2,
                "counts": {"Unexposed": 0, "Exposed": 0, "Unavailable": 2},
            }
        },
    )

    assert any(
        finding.severity == "error"
        and "no cohort row has a usable reconciled exposure" in finding.message
        for finding in findings
    ), findings


def test_statistical_validator_flags_primary_or_mismatch(ra, tmp_path: Path):
    """T1.6 — when the reported OR disagrees with primary_association.csv,
    the validator must surface an error finding."""
    ctx = _ctx_with_sofa(ra)
    cohort_path = tmp_path / "cohort.parquet"
    df = pd.DataFrame(
        {
            "stay_id": list(range(1, 11)),
            "age": [60] * 10,
            "sofa2": [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            "lact": [1.0] * 10,
            "death": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )
    df.to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pd.DataFrame(
        [
            {"variable": "intercept", "coef": -2.0, "odds_ratio": 0.135},
            {"variable": "sofa2", "coef": 0.4, "odds_ratio": 1.491825},
            {"variable": "age", "coef": 0.01, "odds_ratio": 1.01005},
        ]
    ).to_csv(out_dir / "primary_association.csv", index=False)

    schema = ra.schema
    step = schema.AnalysisStep(step_id="04_primary_association", intent="logit")
    findings = ra.StatisticalValidator().audit(
        context=ctx,
        cohort_path=cohort_path,
        step=step,
        out_dir=out_dir,
        # report a wildly wrong OR
        step_summary={
            "predictor": "sofa2",
            "primary_or": 5.0,
            "outcome_rate": float(df["death"].mean()),
        },
    )
    assert any(
        f.severity == "error" and "primary or" in f.message.lower() for f in findings
    ), findings


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
        run_dir / "steps" / "02_baseline_characteristics_and_data_quality" / "outputs"
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
        run_dir / "steps" / "02_baseline_characteristics_and_data_quality" / "outputs"
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


def test_figure_contract_quality_requires_contract_for_figure_exports(
    ra, tmp_path: Path
):
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


def test_figure_contract_quality_blocks_single_panel_result_contract(
    ra, tmp_path: Path
):
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
        f.severity == "error" and "only 1 panel" in f.message for f in findings
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
    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "age": [60.0] * 5,
            "death": [0, 1, 0, 1, 0],
        }
    )
    cohort_path = tmp_path / "cohort.parquet"
    df.to_parquet(cohort_path, index=False)
    ctx = ra.build_research_context(
        research_question="x",
        cohort=df,
        cohort_name="c",
        database="synthetic",
        target_outcome="death",
    )
    # Pretend the descriptor was written when there were 99 rows.
    ctx.cohort.n_stays = 99
    findings = ra.CohortAuditor().audit(context=ctx, cohort_path=cohort_path)
    assert any(
        f.severity == "error" and "row count mismatch" in f.message.lower()
        for f in findings
    )


def test_cohort_auditor_allows_correlation_context_without_target_outcome(
    ra, tmp_path: Path
):
    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "sofa2_max_24h": [1, 3, 5],
            "sofa2_resp_max_24h": [0, 1, 2],
        }
    )
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
        cohort=df,
        cohort_name="c",
        database="synthetic",
        target_outcome="death",
    )


def test_cohort_hygiene_flags_missing_patient_id_when_outcome(ra):
    from easyicu.research_agent.audits.validators import cohort_hygiene_findings

    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "los_icu": [2.0, 3.0, 5.0],
            "death": [0, 1, 0],
        }
    )
    findings = cohort_hygiene_findings(df, _hygiene_ctx(ra, df))
    pid = [
        f
        for f in findings
        if f.detail.get("subkind") == "patient_independence_unassessable"
    ]
    assert len(pid) == 1
    assert pid[0].severity == "warning"
    assert pid[0].detail["structural_no_source"] is True
    # Advice, not a mandate: it must not assert independence or demand a filter.
    assert "re-extract" in pid[0].message.lower()


def test_cohort_hygiene_no_patient_flag_with_patient_id(ra):
    from easyicu.research_agent.audits.validators import cohort_hygiene_findings

    df = pd.DataFrame(
        {
            "subject_id": [10, 10, 11],
            "stay_id": [1, 2, 3],
            "los_icu": [2.0, 3.0, 5.0],
            "death": [0, 1, 0],
        }
    )
    findings = cohort_hygiene_findings(df, _hygiene_ctx(ra, df))
    assert not any(
        f.detail.get("subkind") == "patient_independence_unassessable" for f in findings
    )


def test_cohort_hygiene_no_patient_flag_without_outcome(ra):
    from easyicu.research_agent.audits.validators import cohort_hygiene_findings

    df = pd.DataFrame({"stay_id": [1, 2, 3], "los_icu": [2.0, 3.0, 5.0]})
    ctx = ra.build_research_context(
        research_question="Describe LoS.",
        cohort=df,
        cohort_name="c",
        database="synthetic",
        target_outcome=None,
    )
    findings = cohort_hygiene_findings(df, ctx)
    assert not any(
        f.detail.get("subkind") == "patient_independence_unassessable" for f in findings
    )


def test_cohort_hygiene_short_stay_reported_not_enforced(ra):
    from easyicu.research_agent.audits.validators import cohort_hygiene_findings

    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "los_icu": [0.2, 0.5, 3.0, 5.0],  # half are <1 day
            "death": [0, 1, 0, 1],
        }
    )
    findings = cohort_hygiene_findings(df, _hygiene_ctx(ra, df))
    short = [f for f in findings if f.detail.get("subkind") == "short_stay_exposure"]
    assert len(short) == 1
    assert short[0].severity == "warning"
    assert short[0].detail["fraction_los_under_1_day"] == 0.5
    assert "no minimum-los filter is imposed" in short[0].message.lower()


def test_cohort_hygiene_findings_never_block(ra):
    """Impartiality: hygiene flags are advisory and must never fail-close."""
    from easyicu.research_agent.audits.validators import cohort_hygiene_findings

    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "los_icu": [0.1, 0.2, 5.0],
            "death": [0, 1, 0],
        }
    )
    findings = cohort_hygiene_findings(df, _hygiene_ctx(ra, df))
    assert findings  # both flags fire
    assert all(f.severity == "warning" for f in findings)
    assert all(f.detail.get("impartial") is True for f in findings)


def test_cohort_auditor_surfaces_hygiene_flags(ra, tmp_path: Path):
    """The hygiene flags reach callers through CohortAuditor.audit."""
    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "los_icu": [0.2, 3.0, 5.0],
            "age": [60.0, 70.0, 80.0],
            "death": [0, 1, 0],
        }
    )
    cohort_path = tmp_path / "cohort.parquet"
    df.to_parquet(cohort_path, index=False)
    ctx = _hygiene_ctx(ra, df)
    findings = ra.CohortAuditor().audit(context=ctx, cohort_path=cohort_path)
    assert any(f.detail.get("kind") == "cohort_hygiene" for f in findings)


def test_llm_concept_auditor_parses_findings(ra):
    from easyicu.research_agent.audits.validators import (
        parse_llm_concept_audit_response,
    )

    raw = """```json
{"findings":[{"severity":"warning","message":"ICU mortality may be confused with hospital mortality.","detail":{"issue_code":"other","column":"death_hosp"}}]}
```"""
    findings = parse_llm_concept_audit_response(raw, step_id="04_primary")
    assert len(findings) == 1
    assert findings[0].validator == "llm_concept_auditor"
    assert findings[0].severity == "warning"
    assert findings[0].detail["step_id"] == "04_primary"


@pytest.mark.parametrize(
    ("raw", "response_issue"),
    [
        ("not json", "invalid_json"),
        ('{"findings":{}}', "findings_must_be_a_list"),
        (
            '{"findings":[{"severity":"urgent","message":"x",'
            '"detail":{"issue_code":"other"}}]}',
            "finding_0_severity_is_invalid",
        ),
        (
            '{"findings":[{"severity":"error","message":"x","detail":{}}]}',
            "finding_0_issue_code_is_invalid",
        ),
    ],
)
def test_llm_concept_auditor_invalid_schema_fails_closed(
    raw: str,
    response_issue: str,
) -> None:
    from easyicu.research_agent.audits.validators import (
        parse_llm_concept_audit_response,
    )

    findings = parse_llm_concept_audit_response(raw, step_id="04_primary")

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].detail["issue_code"] == ("llm_concept_audit_response_invalid")
    assert findings[0].detail["response_issue"] == response_issue
    assert findings[0].detail["step_id"] == "04_primary"


def test_llm_concept_auditor_does_not_downgrade_invalid_schema(ra) -> None:
    class _MalformedAuditLLM:
        def complete(self, messages, *, max_tokens=1024, temperature=0.0):
            return '{"findings":[{"severity":"error","message":"x"}]}'

    context = ra.build_research_context(
        research_question="Describe the cohort.",
        cohort=pd.DataFrame({"stay_id": [1, 2]}),
        cohort_name="c",
        database="synthetic",
    )

    findings = _offline_concept_auditor(ra, _MalformedAuditLLM()).audit(
        context=context,
        script_text="print('complete')",
    )

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].detail["issue_code"] == ("llm_concept_audit_response_invalid")


def test_llm_concept_auditor_provider_failure_fails_closed(ra) -> None:
    class _UnavailableAuditLLM:
        def complete(self, messages, *, max_tokens=1024, temperature=0.0):
            raise ConnectionError("provider transport unavailable")

    context = ra.build_research_context(
        research_question="Describe the cohort.",
        cohort=pd.DataFrame({"stay_id": [1, 2]}),
        cohort_name="c",
        database="synthetic",
    )
    step = ra.AnalysisStep(
        step_id="04_primary",
        intent="Fit the planner-selected primary analysis.",
    )

    findings = _offline_concept_auditor(ra, _UnavailableAuditLLM()).audit(
        context=context,
        script_text="print('candidate')",
        step=step,
    )

    assert len(findings) == 1
    assert findings[0].validator == "llm_concept_auditor"
    assert findings[0].severity == "error"
    assert findings[0].detail == {
        "issue_code": "llm_concept_audit_provider_failure",
        "error_type": "ConnectionError",
        "step_id": "04_primary",
    }


def test_llm_concept_auditor_prompt_includes_outcome_semantics(ra):
    auditor = ra.LLMConceptAuditor(ra.MockLLMClient())
    ctx = ra.build_research_context(
        research_question="Is age associated with ICU mortality?",
        cohort=pd.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "age": [60, 70, 80],
                "death": [0, 1, 0],
            }
        ),
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


def _plausibility_range_context(ra, *, binary: bool = False):
    return ra.ResearchContext(
        research_question="Assess a continuous ICU marker.",
        cohort=ra.CohortDescriptor(
            cohort_name="c",
            database="synthetic",
            n_stays=3,
            n_patients=3,
        ),
        variables=[
            ra.ConceptDescriptor(
                name="marker",
                description="Continuous ICU marker",
                dtype="float64",
                unit="units",
                valid_range=[0.0, 10.0],
                observed_domain={
                    "min": 1.0,
                    "max": 12.0,
                    "is_binary": binary,
                },
            )
        ],
        primary_exposure="marker",
    )


def test_llm_concept_auditor_prompt_declares_flag_only_plausibility_policy(ra):
    auditor = ra.LLMConceptAuditor(ra.MockLLMClient())
    prompt = auditor._prompt(
        context=_plausibility_range_context(ra),
        script_text="model.fit(frame[['marker']])",
        step=None,
    )

    assert '"plausibility_range":[0.0,10.0]' in prompt
    assert '"out_of_range_action":"retain_and_flag"' in prompt
    assert '"is_binary":false' in prompt
    assert "not a locked eligibility or exclusion rule" in prompt
    assert "finite continuous value outside that range is not an ERROR" in prompt
    assert "plausibility_range_exclusion_required" in prompt


def test_llm_concept_auditor_prompt_declares_strict_numeric_host_boundary(ra):
    auditor = ra.LLMConceptAuditor(ra.MockLLMClient())
    prompt = auditor._prompt(
        context=_plausibility_range_context(ra),
        script_text="strict_numeric_input(frame['marker']).values",
        step=None,
    )

    assert "unconvertible" in prompt
    assert "semantically nonnumeric" in prompt
    assert "non-finite" in prompt
    assert "strict_numeric_nonfinite_guard_required" in prompt
    assert "do not demand a second `isfinite` guard" in prompt


def test_llm_concept_auditor_reclassifies_typed_flag_only_range_demand(ra):
    class _RangeDemandLLM:
        def complete(self, messages, *, max_tokens=1024, temperature=0.0):
            return json.dumps(
                {
                    "findings": [
                        {
                            "severity": "error",
                            "message": "Finite values outside the plausible range should be excluded.",
                            "detail": {
                                "issue_code": "plausibility_range_exclusion_required",
                                "variable": "marker",
                                "requested_action": "exclude",
                                "value_class": "finite_outside_plausibility_range",
                            },
                        }
                    ]
                }
            )

    findings = _offline_concept_auditor(ra, _RangeDemandLLM()).audit(
        context=_plausibility_range_context(ra),
        script_text="model.fit(frame[['marker']])",
    )

    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert findings[0].detail["range_policy_authority"] == (
        "concept_descriptor_flag_only"
    )
    assert "deterministic gate" in findings[0].detail[
        "flag_obligation"
    ]
    assert findings[0].detail["retain_and_flag_half_satisfied"] == "retain"


def test_llm_concept_auditor_reclassifies_scoped_flag_only_range_demand(ra):
    class _ScopedRangeDemandLLM:
        def complete(self, messages, *, max_tokens=1024, temperature=0.0):
            return json.dumps(
                {
                    "findings": [
                        {
                            "severity": "error",
                            "message": "The finite value should not remain valid.",
                            "detail": {
                                "issue_code": "plausibility_range_exclusion_required",
                                "variable": "marker",
                                "requested_action": (
                                    "exclude from valid_observed and classify as "
                                    "contradictory/invalid"
                                ),
                                "value_class": "finite_outside_plausibility_range",
                            },
                        }
                    ]
                }
            )

    findings = _offline_concept_auditor(ra, _ScopedRangeDemandLLM()).audit(
        context=_plausibility_range_context(ra),
        script_text="model.fit(frame[['marker']])",
    )

    assert len(findings) == 1
    assert findings[0].detail["range_policy_authority"] == (
        "concept_descriptor_flag_only"
    )
    assert findings[0].severity == "warning"
    assert findings[0].detail["retain_and_flag_half_satisfied"] == "retain"


@pytest.mark.parametrize(
    ("binary", "variable", "value_class"),
    [
        (True, "marker", "finite_outside_plausibility_range"),
        (False, "marker", "nonfinite_value"),
        (False, "unknown_marker", "finite_outside_plausibility_range"),
    ],
)
def test_flag_only_range_reclassifier_preserves_strict_or_unbound_errors(
    ra, binary, variable, value_class
):
    from easyicu.research_agent.audits.validators import (
        _reclassify_flag_only_plausibility_range_findings,
    )

    finding = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="The candidate value should be excluded.",
        detail={
            "issue_code": "plausibility_range_exclusion_required",
            "variable": variable,
            "requested_action": "exclude",
            "value_class": value_class,
        },
    )

    findings = _reclassify_flag_only_plausibility_range_findings(
        findings=[finding],
        context=_plausibility_range_context(ra, binary=binary),
    )

    assert findings[0].severity == "error"


def test_flag_only_llm_reclassifier_does_not_duplicate_deterministic_gate(ra):
    """The LLM adapter owns retention; the deterministic gate owns flagging."""

    from easyicu.research_agent.audits.validators import (
        _reclassify_flag_only_plausibility_range_findings,
    )

    finding = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="The candidate value should be excluded.",
        detail={
            "issue_code": "plausibility_range_exclusion_required",
            "variable": "marker",
            "requested_action": "exclude",
            "value_class": "finite_outside_plausibility_range",
        },
    )
    result = _reclassify_flag_only_plausibility_range_findings(
        findings=[finding],
        context=_plausibility_range_context(ra),
    )[0]

    assert result.severity == "warning"
    assert result.detail["retain_and_flag_half_satisfied"] == "retain"
    assert "deterministic gate" in result.detail[
        "flag_obligation"
    ]
    assert "flag_evidence" not in result.detail
    # The message remains stable for quarantine replay identity.
    assert result.message == "The candidate value should be excluded."


def test_llm_concept_auditor_checks_summary_source_status_bypasses(ra):
    auditor = ra.LLMConceptAuditor(ra.MockLLMClient())
    ctx = ra.build_research_context(
        research_question="Summarize an early ICU measurement.",
        cohort=pd.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "marker_first": [1.0, None, 2.0],
                "marker_n": [1, 0, 1],
                "marker_measured": [1, 0, 1],
            }
        ),
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
    assert "row-aligned finalized table" in prompt
    assert "use it directly" in prompt
    assert "must neither redefine nor overwrite the finalized exposure" in prompt
    assert "Do not demand a second sparse-event reconciliation" in prompt
    assert "detail.issue_code" in prompt
    assert "Message text is explanatory only, never routing" in prompt


def test_llm_concept_auditor_trusts_exact_step_input_metadata_binding(ra):
    context = ra.ResearchContext(
        research_question="Assess one planned early summary.",
        cohort=ra.CohortDescriptor(
            cohort_name="c",
            database="synthetic",
            n_stays=2,
            n_patients=2,
        ),
        variables=[
            ra.ConceptDescriptor(
                name="marker_max",
                dtype="float64",
                source_concept="marker",
                analysis_window="icu_admit_0_24h",
            )
        ],
        primary_exposure="marker_max",
    )
    step = ra.AnalysisStep(
        step_id="early_summary",
        intent="Use the exact planned early summary.",
        inputs=["marker_max"],
        expected_outputs=["table:summary"],
        method="descriptive_statistics",
    )

    prompt = ra.LLMConceptAuditor(ra.MockLLMClient())._prompt(
        context=context,
        script_text="values = frame['marker_max']",
        step=step,
    )

    assert "host-owned binding" in prompt
    assert "do not require generated code to re-prove the host metadata" in prompt
    assert (
        "flag it merely because its column name contains first/max/min/mean" in prompt
    )


def test_llm_concept_auditor_binds_planner_science_and_derived_concepts(ra):
    context = ra.ResearchContext(
        research_question="Estimate one adjusted ICU association.",
        cohort=ra.CohortDescriptor(
            cohort_name="c",
            database="synthetic",
            n_stays=3,
            n_patients=3,
        ),
        variables=[
            ra.ConceptDescriptor(
                name="derived_exposure",
                dtype="int64",
                source_concept="derived_exposure",
                derived_from_concepts=["component_a", "component_b"],
            ),
            ra.ConceptDescriptor(
                name="derived_exposure_measured",
                dtype="int64",
                role="meta",
            ),
            ra.ConceptDescriptor(name="component_a", dtype="float64"),
            ra.ConceptDescriptor(name="component_b", dtype="float64"),
            ra.ConceptDescriptor(name="death", dtype="int64", role="outcome"),
        ],
        primary_exposure="derived_exposure",
        target_outcome="death",
    )
    step = ra.AnalysisStep(
        step_id="adjusted_model",
        planned_analysis_role="primary",
        intent="Fit the Planner-owned adjusted model.",
        inputs=["derived_exposure", "death"],
        expected_outputs=["table:adjusted_association_estimates"],
        method="adjusted_association_models",
        icu_rule_refs=[
            "Do not use measurement or source-status flags as adjustment covariates."
        ],
        model_requirements=[
            {
                "requirement_id": "primary_model",
                "outcome": "death",
                "outcome_type": "binary",
                "method_family": "logistic_regression",
                "exposure_source": "derived_exposure",
                "analysis_role": "primary",
                "analysis_set": "source_aware",
                "required_for_step_success": True,
            }
        ],
    )

    prompt = ra.LLMConceptAuditor(ra.MockLLMClient())._prompt(
        context=context,
        script_text=(
            "X = frame[['derived_exposure', 'derived_exposure_measured']]\n"
            "model.fit(X, frame['death'])"
        ),
        step=step,
    )

    assert "Planner-declared step contract below is binding scientific authority" in prompt
    assert '"method":"adjusted_association_models"' in prompt
    assert '"requirement_id":"primary_model"' in prompt
    assert "Do not use measurement or source-status flags" in prompt
    assert '"derived_from_concepts":["component_a","component_b"]' in prompt
    assert "do not require the generated script to load or filter on each raw component" in prompt
    assert "not automatic adjustment covariates" in prompt
    assert "deterministic missing-indicator complement" in prompt
    assert "`>= 0` retains both zero and one" in prompt


def test_llm_concept_auditor_downgrades_finalized_exposure_rederivation_demand(ra):
    class _FalseReconciliationDemandLLM:
        def complete(self, messages, *, max_tokens=1024, temperature=0.0):
            return json.dumps(
                {
                    "findings": [
                        {
                            "severity": "error",
                            "message": (
                                "A finalized DataFrame exposure path bypasses the "
                                "required binary-event triad reconciliation."
                            ),
                            "detail": {
                                "issue_code": "finalized_exposure_missing_reconciliation",
                                "context": (
                                    "The row-aligned values are validated but the "
                                    "script does not call "
                                    "reconcile_binary_event_presence."
                                ),
                            },
                        }
                    ]
                }
            )

    script = """
exposure_binding = resolved_inputs['artifact:primary_exposure_definition']
exposure_definition = load_typed_table(exposure_binding)
product_contract = exposure_binding['product_contract']
if isinstance(exposure_definition, pd.DataFrame):
    finalized = pd.to_numeric(exposure_definition['vasopressor'], errors='coerce')
    if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
        raise RuntimeError('invalid finalized exposure')
    treatment = finalized.astype(int)
    frame['vasopressor'] = treatment
    model = sm.Logit(outcome, pd.DataFrame({'vasopressor': treatment}))
"""
    findings = _offline_concept_auditor(ra, _FalseReconciliationDemandLLM()).audit(
        context=ra.build_research_context(
            research_question="Assess balance by vasopressor exposure.",
            cohort=pd.DataFrame({"stay_id": [1, 2], "vasopressor": [0, 1]}),
            cohort_name="c",
            database="synthetic",
            primary_exposure="vasopressor",
        ),
        script_text=script,
        step=None,
    )

    assert findings[0].severity == "warning"
    assert findings[0].detail["downgraded_reason"]


def test_llm_concept_auditor_keeps_finalized_exposure_override_blocking(ra):
    from easyicu.research_agent.audits.validators import (
        _downgrade_finalized_exposure_reconciliation_findings,
    )
    from easyicu.research_agent.contracts.runtime import ValidationFinding

    script = """
exposure_binding = resolved_inputs['artifact:primary_exposure_definition']
exposure_definition = load_typed_table(exposure_binding)
product_contract = exposure_binding['product_contract']
if isinstance(exposure_definition, pd.DataFrame):
    finalized = pd.to_numeric(exposure_definition['vasopressor'], errors='coerce')
    if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
        raise RuntimeError('invalid finalized exposure')
    treatment = helper_result.values
"""
    context = ra.build_research_context(
        research_question="Assess balance by vasopressor exposure.",
        cohort=pd.DataFrame({"stay_id": [1, 2], "vasopressor": [0, 1]}),
        cohort_name="c",
        database="synthetic",
        primary_exposure="vasopressor",
    )
    findings = _downgrade_finalized_exposure_reconciliation_findings(
        findings=[
            ValidationFinding(
                validator="llm_concept_auditor",
                severity="error",
                message="The finalized exposure artifact is bypassed.",
                detail={
                    "issue_code": "finalized_exposure_overridden",
                    "context": (
                        "The script ignores its values and overwrites them with "
                        "raw companion reconciliation."
                    ),
                },
            )
        ],
        context=context,
        script_text=script,
    )

    assert findings[0].severity == "error"
    assert "downgraded_reason" not in findings[0].detail


def test_llm_concept_auditor_downgrades_raw_resolver_branch_false_override(ra):
    from easyicu.research_agent.audits.validators import (
        _downgrade_finalized_exposure_reconciliation_findings,
    )
    from easyicu.research_agent.contracts.runtime import ValidationFinding

    script = """
exposure_binding = resolved_inputs['artifact:primary_exposure_definition']
exposure_definition = load_typed_table(exposure_binding)
product_contract = exposure_binding['product_contract']
def resolve_raw_exposure(definition, frame):
    return reconcile_binary_event_presence(frame)

if isinstance(exposure_definition, pd.DataFrame):
    finalized = pd.to_numeric(exposure_definition['vasopressor'], errors='coerce')
    if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
        raise RuntimeError('invalid finalized exposure')
    treatment = finalized.astype(int)
else:
    treatment = resolve_raw_exposure(exposure_definition, frame).values
frame['vasopressor'] = treatment
model = sm.Logit(outcome, pd.DataFrame({'vasopressor': treatment}))
"""
    context = ra.build_research_context(
        research_question="Assess balance by vasopressor exposure.",
        cohort=pd.DataFrame({"stay_id": [1, 2], "vasopressor": [0, 1]}),
        cohort_name="c",
        database="synthetic",
        primary_exposure="vasopressor",
    )
    findings = _downgrade_finalized_exposure_reconciliation_findings(
        findings=[
            ValidationFinding(
                validator="llm_concept_auditor",
                severity="error",
                message=(
                    "Authoritative exposure is overwritten by a hardcoded "
                    "reconciliation."
                ),
                detail={
                    "issue_code": "finalized_exposure_overridden",
                    "context": (
                        "After resolving the finalized artifact, the script "
                        "unconditionally replaces treatment with "
                        "reconcile_binary_event_presence."
                    ),
                },
            ),
            ValidationFinding(
                validator="llm_concept_auditor",
                severity="error",
                message="The script overwrites the finalized exposure.",
                detail={"context": "English prose is not a routing contract."},
            ),
        ],
        context=context,
        script_text=script,
    )

    assert [finding.severity for finding in findings] == ["warning", "error"]
    assert "AST control-flow verification" in findings[0].detail["downgraded_reason"]
    assert "downgraded_reason" not in findings[1].detail

    decoy_findings = _downgrade_finalized_exposure_reconciliation_findings(
        findings=[
            findings[0].model_copy(
                update={
                    "severity": "error",
                    "detail": {"issue_code": "finalized_exposure_overridden"},
                }
            )
        ],
        context=context,
        script_text=script.replace(
            "isinstance(exposure_definition, pd.DataFrame)",
            "isinstance(decoy, pd.DataFrame)",
        ),
    )
    assert decoy_findings[0].severity == "error"
    assert "downgraded_reason" not in decoy_findings[0].detail

    unbound_findings = _downgrade_finalized_exposure_reconciliation_findings(
        findings=[
            findings[0].model_copy(
                update={
                    "severity": "error",
                    "detail": {"issue_code": "finalized_exposure_overridden"},
                }
            )
        ],
        context=context,
        script_text=script.replace(
            "exposure_binding = resolved_inputs['artifact:primary_exposure_definition']\n"
            "exposure_definition = load_typed_table(exposure_binding)\n"
            "product_contract = exposure_binding['product_contract']",
            "REQUESTED_INPUTS = ['artifact:primary_exposure_definition']",
        ),
    )
    assert unbound_findings[0].severity == "error"
    assert "downgraded_reason" not in unbound_findings[0].detail


def test_llm_concept_auditor_keeps_post_branch_reconciliation_blocking(ra):
    from easyicu.research_agent.audits.validators import (
        _downgrade_finalized_exposure_reconciliation_findings,
    )
    from easyicu.research_agent.contracts.runtime import ValidationFinding

    script = """
exposure_binding = resolved_inputs['artifact:primary_exposure_definition']
exposure_definition = load_typed_table(exposure_binding)
product_contract = exposure_binding['product_contract']
def resolve_raw_exposure(definition, frame):
    return reconcile_binary_event_presence(frame)

if isinstance(exposure_definition, pd.DataFrame):
    finalized = pd.to_numeric(exposure_definition['vasopressor'], errors='coerce')
    if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
        raise RuntimeError('invalid finalized exposure')
    treatment = finalized.astype(int)
else:
    treatment = resolve_raw_exposure(exposure_definition, frame).values
treatment = resolve_raw_exposure(exposure_definition, frame).values
"""
    context = ra.build_research_context(
        research_question="Assess balance by vasopressor exposure.",
        cohort=pd.DataFrame({"stay_id": [1, 2], "vasopressor": [0, 1]}),
        cohort_name="c",
        database="synthetic",
        primary_exposure="vasopressor",
    )
    findings = _downgrade_finalized_exposure_reconciliation_findings(
        findings=[
            ValidationFinding(
                validator="llm_concept_auditor",
                severity="error",
                message="The finalized exposure is overwritten.",
                detail={
                    "issue_code": "finalized_exposure_overridden",
                    "context": (
                        "The script replaces the finalized values with "
                        "reconcile_binary_event_presence."
                    ),
                },
            )
        ],
        context=context,
        script_text=script,
    )

    assert findings[0].severity == "error"
    assert "downgraded_reason" not in findings[0].detail


def test_llm_concept_auditor_accepts_contract_bound_finalized_branch(ra):
    from easyicu.research_agent.audits.validators import (
        _downgrade_finalized_exposure_reconciliation_findings,
    )
    from easyicu.research_agent.contracts.runtime import ValidationFinding

    script = """
exposure_binding = resolved_inputs['artifact:primary_exposure_definition']
exposure_definition = load_typed_table(exposure_binding)
product_contract = exposure_binding['product_contract']
def resolve_finalized_exposure(definition, product_contract, frame):
    executable = product_contract['executable_column']
    finalized = pd.to_numeric(definition[executable], errors='coerce')
    if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
        raise RuntimeError('invalid finalized exposure')
    return finalized.astype(int)

def resolve_raw_exposure(definition, frame):
    return reconcile_binary_event_presence(frame)

if isinstance(exposure_definition, pd.DataFrame):
    treatment = resolve_finalized_exposure(
        exposure_definition, product_contract, frame
    )
else:
    treatment = resolve_raw_exposure(exposure_definition, frame).values
frame[product_contract['executable_column']] = treatment
model = sm.Logit(outcome, pd.DataFrame({'treatment': treatment}))
"""
    context = ra.build_research_context(
        research_question="Assess balance by the registered exposure.",
        cohort=pd.DataFrame({"stay_id": [1, 2], "treatment": [0, 1]}),
        cohort_name="c",
        database="synthetic",
        primary_exposure="treatment",
    )
    findings = _downgrade_finalized_exposure_reconciliation_findings(
        findings=[
            ValidationFinding(
                validator="llm_concept_auditor",
                severity="error",
                message=(
                    "The script overwrites the authoritative finalized exposure "
                    "with a second helper result."
                ),
                detail={
                    "issue_code": "finalized_exposure_overridden",
                    "context": (
                        "It unconditionally runs "
                        "reconcile_binary_event_presence after resolving the "
                        "artifact."
                    ),
                },
            )
        ],
        context=context,
        script_text=script,
    )

    assert findings[0].severity == "warning"
    assert "AST control-flow verification" in findings[0].detail["downgraded_reason"]


def test_authoritative_exposure_flow_accepts_exact_host_manifest_binding() -> None:
    from easyicu.research_agent.audits.validators import (
        _verified_authoritative_exposure_flow,
    )

    script = """
manifest_path = Path(os.environ['EASYICU_RESOLVED_INPUTS_JSON'])
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
exposure_binding = manifest['inputs']['artifact:primary_exposure_definition']
exposure_definition = load_typed_table(exposure_binding)
product_contract = exposure_binding['product_contract']
def resolve_exposure(definition, product_contract):
    executable = product_contract['executable_column']
    finalized = pd.to_numeric(definition[executable], errors='coerce')
    if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
        raise RuntimeError('invalid finalized exposure')
    return finalized.astype(int)
treatment = resolve_exposure(exposure_definition, product_contract)
model = sm.Logit(outcome, pd.DataFrame({'treatment': treatment}))
"""

    assert _verified_authoritative_exposure_flow(
        script,
        primary_exposure="treatment",
    )


def test_authoritative_exposure_flow_rejects_shadowed_resolved_inputs() -> None:
    from easyicu.research_agent.audits.validators import (
        _verified_authoritative_exposure_flow,
    )

    script = """
decoy = pd.DataFrame({'treatment': [0, 1]})
resolved_inputs = {
    'artifact:primary_exposure_definition': {'value': decoy},
}
exposure_binding = resolved_inputs['artifact:primary_exposure_definition']
exposure_definition = exposure_binding['value']
finalized = pd.to_numeric(exposure_definition['treatment'], errors='coerce')
if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
    raise RuntimeError('invalid exposure')
model = sm.Logit(outcome, pd.DataFrame({'treatment': finalized}))
"""

    assert not _verified_authoritative_exposure_flow(
        script,
        primary_exposure="treatment",
    )


def test_authoritative_exposure_flow_rejects_decoy_result_sink() -> None:
    from easyicu.research_agent.audits.validators import (
        _verified_authoritative_exposure_flow,
    )

    script = """
exposure_binding = resolved_inputs['artifact:primary_exposure_definition']
exposure_definition = exposure_binding['value']
product_contract = exposure_binding['product_contract']
executable = product_contract['executable_column']
finalized = pd.to_numeric(exposure_definition[executable], errors='coerce')
if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
    raise RuntimeError('invalid finalized exposure')
decoy = sm.Logit(outcome, pd.DataFrame({'treatment': finalized})).fit()
raw_wrong = pd.to_numeric(frame['raw_wrong'], errors='coerce')
primary = sm.Logit(outcome, pd.DataFrame({'treatment': raw_wrong})).fit()
"""

    assert not _verified_authoritative_exposure_flow(
        script,
        primary_exposure="treatment",
    )


def test_authoritative_exposure_flow_rejects_checks_without_fail_closed_guard() -> None:
    from easyicu.research_agent.audits.validators import (
        _verified_authoritative_exposure_flow,
    )

    script = """
exposure_binding = resolved_inputs['artifact:primary_exposure_definition']
exposure_definition = exposure_binding['value']
product_contract = exposure_binding['product_contract']
executable = product_contract['executable_column']
finalized = pd.to_numeric(exposure_definition[executable], errors='coerce')
finalized.isna().any()
np.isfinite(finalized).all()
finalized.isin([0, 1]).all()
model = sm.Logit(outcome, pd.DataFrame({'treatment': finalized}))
"""

    assert not _verified_authoritative_exposure_flow(
        script,
        primary_exposure="treatment",
    )


def test_authoritative_exposure_flow_rejects_conditional_helper_return() -> None:
    from easyicu.research_agent.audits.validators import (
        _verified_authoritative_exposure_flow,
    )

    script = """
exposure_binding = resolved_inputs['artifact:primary_exposure_definition']
exposure_definition = exposure_binding['value']
product_contract = exposure_binding['product_contract']
executable = product_contract['executable_column']
finalized = pd.to_numeric(exposure_definition[executable], errors='coerce')
if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
    raise RuntimeError('invalid finalized exposure')
def choose_exposure():
    if use_registered_exposure:
        return finalized
    return frame['raw_wrong']
treatment = choose_exposure()
model = sm.Logit(outcome, pd.DataFrame({'treatment': treatment}))
"""

    assert not _verified_authoritative_exposure_flow(
        script,
        primary_exposure="treatment",
    )


def test_authoritative_exposure_flow_rejects_conditional_expression_helper_return() -> (
    None
):
    from easyicu.research_agent.audits.validators import (
        _verified_authoritative_exposure_flow,
    )

    script = """
exposure_binding = resolved_inputs['artifact:primary_exposure_definition']
exposure_definition = exposure_binding['value']
product_contract = exposure_binding['product_contract']
executable = product_contract['executable_column']
finalized = pd.to_numeric(exposure_definition[executable], errors='coerce')
if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
    raise RuntimeError('invalid finalized exposure')
def choose_exposure():
    return finalized if use_registered_exposure else frame['raw_wrong']
treatment = choose_exposure()
model = sm.Logit(outcome, pd.DataFrame({'treatment': treatment}))
"""

    assert not _verified_authoritative_exposure_flow(
        script,
        primary_exposure="treatment",
    )


def test_authoritative_exposure_flow_rejects_scope_shadowed_selected_name() -> None:
    from easyicu.research_agent.audits.validators import (
        _verified_authoritative_exposure_flow,
    )

    script = """
exposure_binding = resolved_inputs['artifact:primary_exposure_definition']
exposure_definition = exposure_binding['value']
product_contract = exposure_binding['product_contract']
executable = product_contract['executable_column']
finalized = pd.to_numeric(exposure_definition[executable], errors='coerce')
if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
    raise RuntimeError('invalid finalized exposure')
treatment = finalized
def primary_model():
    treatment = frame['raw_wrong']
    return sm.Logit(outcome, pd.DataFrame({'treatment': treatment}))
primary_model()
"""

    assert not _verified_authoritative_exposure_flow(
        script,
        primary_exposure="treatment",
    )


def test_authoritative_exposure_flow_rejects_mixedlm_after_decoy_model() -> None:
    from easyicu.research_agent.audits.validators import (
        _verified_authoritative_exposure_flow,
    )

    script = """
exposure_binding = resolved_inputs['artifact:primary_exposure_definition']
exposure_definition = exposure_binding['value']
product_contract = exposure_binding['product_contract']
executable = product_contract['executable_column']
finalized = pd.to_numeric(exposure_definition[executable], errors='coerce')
if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
    raise RuntimeError('invalid finalized exposure')
decoy = sm.Logit(outcome, pd.DataFrame({'treatment': finalized}))
raw_design = pd.DataFrame({'treatment': frame['raw_wrong']})
primary = sm.MixedLM(outcome, raw_design, groups=frame['hospital_id']).fit()
"""

    assert not _verified_authoritative_exposure_flow(
        script,
        primary_exposure="treatment",
    )


def test_authoritative_exposure_flow_rejects_unlisted_estimator_after_decoy_model() -> (
    None
):
    from easyicu.research_agent.audits.validators import (
        _verified_authoritative_exposure_flow,
    )

    script = """
exposure_binding = resolved_inputs['artifact:primary_exposure_definition']
exposure_definition = exposure_binding['value']
product_contract = exposure_binding['product_contract']
executable = product_contract['executable_column']
finalized = pd.to_numeric(exposure_definition[executable], errors='coerce')
if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
    raise RuntimeError('invalid finalized exposure')
decoy = sm.Logit(outcome, pd.DataFrame({'treatment': finalized})).fit()
primary = sm.Probit(
    outcome,
    pd.DataFrame({'treatment': frame['raw_wrong']}),
).fit()
"""

    assert not _verified_authoritative_exposure_flow(
        script,
        primary_exposure="treatment",
    )


def test_authoritative_exposure_flow_rejects_generic_train_sink_after_decoy_model() -> (
    None
):
    from easyicu.research_agent.audits.validators import (
        _verified_authoritative_exposure_flow,
    )

    script = """
exposure_binding = resolved_inputs['artifact:primary_exposure_definition']
exposure_definition = exposure_binding['value']
product_contract = exposure_binding['product_contract']
executable = product_contract['executable_column']
finalized = pd.to_numeric(exposure_definition[executable], errors='coerce')
if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
    raise RuntimeError('invalid finalized exposure')
decoy = sm.Logit(outcome, pd.DataFrame({'treatment': finalized})).fit()
raw_design = xgb.DMatrix(
    pd.DataFrame({'treatment': frame['raw_wrong']}),
    label=outcome,
)
primary = xgb.train({}, raw_design)
"""

    assert not _verified_authoritative_exposure_flow(
        script,
        primary_exposure="treatment",
    )


def test_authoritative_exposure_flow_accepts_generic_train_sink_with_authority() -> (
    None
):
    from easyicu.research_agent.audits.validators import (
        _verified_authoritative_exposure_flow,
    )

    script = """
exposure_binding = resolved_inputs['artifact:primary_exposure_definition']
exposure_definition = exposure_binding['value']
product_contract = exposure_binding['product_contract']
executable = product_contract['executable_column']
finalized = pd.to_numeric(exposure_definition[executable], errors='coerce')
if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
    raise RuntimeError('invalid finalized exposure')
design = xgb.DMatrix(
    pd.DataFrame({'treatment': finalized}),
    label=outcome,
)
primary = xgb.train({}, design)
"""

    assert _verified_authoritative_exposure_flow(
        script,
        primary_exposure="treatment",
    )


def test_authoritative_exposure_flow_allows_unrelated_diagnostic_plot() -> None:
    from easyicu.research_agent.audits.validators import (
        _verified_authoritative_exposure_flow,
    )

    script = """
exposure_binding = resolved_inputs['artifact:primary_exposure_definition']
exposure_definition = exposure_binding['value']
product_contract = exposure_binding['product_contract']
executable = product_contract['executable_column']
finalized = pd.to_numeric(exposure_definition[executable], errors='coerce')
if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
    raise RuntimeError('invalid finalized exposure')
model = sm.Logit(outcome, pd.DataFrame({'treatment': finalized}))
plt.plot(calibration_x, calibration_y)
"""

    assert _verified_authoritative_exposure_flow(
        script,
        primary_exposure="treatment",
    )


@pytest.mark.parametrize(
    "resolver_name", ["resolve_finalized_exposure", "resolve_exposure"]
)
def test_llm_concept_auditor_accepts_finalized_only_consumer_without_helper_call(
    ra, resolver_name
):
    from easyicu.research_agent.audits.validators import (
        _downgrade_finalized_exposure_reconciliation_findings,
    )
    from easyicu.research_agent.contracts.runtime import ValidationFinding

    script = f"""
exposure_binding = resolved_inputs['artifact:primary_exposure_definition']
exposure_definition = load_typed_table(exposure_binding)
product_contract = exposure_binding['product_contract']
def {resolver_name}(definition, product_contract, frame):
    if not isinstance(definition, pd.DataFrame):
        raise RuntimeError('finalized table required')
    executable = product_contract['executable_column']
    finalized = pd.to_numeric(definition[executable], errors='coerce')
    if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
        raise RuntimeError('invalid finalized exposure')
    return finalized.astype(int)
treatment = {resolver_name}(
    exposure_definition, product_contract, frame
)
model = sm.Logit(outcome, pd.DataFrame({{'treatment': treatment}}))
"""
    context = ra.build_research_context(
        research_question="Assess balance by the registered exposure.",
        cohort=pd.DataFrame({"stay_id": [1, 2], "treatment": [0, 1]}),
        cohort_name="c",
        database="synthetic",
        primary_exposure="treatment",
    )
    findings = _downgrade_finalized_exposure_reconciliation_findings(
        findings=[
            ValidationFinding(
                validator="llm_concept_auditor",
                severity="error",
                message="The script overwrites the finalized exposure.",
                detail={
                    "issue_code": "finalized_exposure_overridden",
                    "context": (
                        "It assigns reconcile_binary_event_presence values after "
                        "the finalized binding."
                    ),
                },
            ),
            ValidationFinding(
                validator="llm_concept_auditor",
                severity="error",
                message=(
                    "The finalized-table exposure branch is incorrectly forced "
                    "through sparse-event triad validation."
                ),
                detail={
                    "issue_code": "finalized_exposure_forced_raw_reconciliation",
                    "context": (
                        "It requires source_count_column and raw companion fields."
                    ),
                },
            ),
        ],
        context=context,
        script_text=script,
    )

    assert [finding.severity for finding in findings] == ["error", "warning"]
    assert "downgraded_reason" not in findings[0].detail
    assert "AST control-flow verification" in findings[1].detail["downgraded_reason"]


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
                                "issue_code": "audit_only_companion_row_gating_required",
                                "context": (
                                    "The script audits measured/count pairs on the "
                                    "original dataframe, but the flags do not mask or "
                                    "invalidate modeled first-value covariates."
                                ),
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
if provenance_failed:
    raise RuntimeError('invalid measurement provenance')
model.fit(frame[['marker_first']], assignment)
"""
    findings = _offline_concept_auditor(ra, _CompanionGatingFalsePositiveLLM()).audit(
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

    conditional_findings = _offline_concept_auditor(
        ra, _CompanionGatingFalsePositiveLLM()
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
        script_text=script.replace(
            "raise RuntimeError('invalid measurement provenance')",
            "if strict_mode:\n        raise RuntimeError('invalid measurement provenance')",
        ),
        step=None,
    )
    assert conditional_findings[0].severity == "error"
    assert "downgraded_reason" not in conditional_findings[0].detail


@pytest.mark.parametrize(
    ("selector", "expected_severity"),
    [("in_range_mask", "warning"), ("support_mask", "error")],
)
def test_llm_concept_auditor_checks_named_value_selector_ownership(
    ra, selector, expected_severity
):
    class _FindingLLM:
        def complete(self, messages, *, max_tokens=1024, temperature=0.0):
            return json.dumps(
                {
                    "findings": [
                        {
                            "severity": "error",
                            "message": "Companions gate the value distribution.",
                            "detail": {
                                "issue_code": "audit_only_companion_row_gating_required",
                                "variables": ["support_mask", "valid_distribution"],
                            },
                        }
                    ]
                }
            )

    script = f"""
measurement_provenance_audit = {{
    'checks': [{{'invalid_pair_n': 0, 'discordant_n': 0, 'role': 'audit_only'}}]
}}
invalid_pair_n = 0
discordant_n = 0
if invalid_pair_n or discordant_n:
    raise RuntimeError('invalid measurement provenance')
support_mask = measured_series.eq(1) & count_series.gt(0)
in_range_mask = value_series.notna() & value_series.between(0.0, 30.0)
valid_distribution = value_series.loc[{selector}]
"""
    findings = _offline_concept_auditor(ra, _FindingLLM()).audit(
        context=ra.build_research_context(
            research_question="Audit one continuous measurement.",
            cohort=pd.DataFrame({"stay_id": [1, 2], "value": [1.0, 2.0]}),
            cohort_name="c",
            database="synthetic",
        ),
        script_text=script,
        step=None,
    )

    assert findings[0].severity == expected_severity
    assert ("downgraded_reason" in findings[0].detail) is (
        expected_severity == "warning"
    )


def test_llm_concept_auditor_accepts_direct_self_raising_host_receipts(ra):
    class _FalsePositiveLLM:
        def complete(self, messages, *, max_tokens=1024, temperature=0.0):
            return json.dumps(
                {
                    "findings": [
                        {
                            "severity": "error",
                            "message": "The receipt status is not inspected.",
                            "detail": {
                                "issue_code": "audit_only_companion_row_gating_required",
                                "variables": ["marker_first"],
                            },
                        }
                    ]
                }
            )

    script = """
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt

def main():
    receipts = [measurement_provenance_receipt(
        frame,
        measured_column="marker_measured",
        count_column="marker_n",
    )]
    model.fit(frame[["marker_first"]], assignment)

if __name__ == "__main__":
    main()
"""
    findings = _offline_concept_auditor(ra, _FalsePositiveLLM()).audit(
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
    assert "self-raising" in findings[0].detail["downgraded_reason"]


def test_llm_concept_auditor_does_not_accept_unused_host_receipt_decoy(ra):
    class _FindingLLM:
        def complete(self, messages, *, max_tokens=1024, temperature=0.0):
            return json.dumps(
                {
                    "findings": [
                        {
                            "severity": "error",
                            "message": "Provenance is not fail-closed.",
                            "detail": {
                                "issue_code": "audit_only_companion_row_gating_required",
                                "variables": ["marker_first"],
                            },
                        }
                    ]
                }
            )

    script = """
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt

def unused():
    measurement_provenance_receipt(
        frame, measured_column="marker_measured", count_column="marker_n"
    )

model.fit(frame[["marker_first"]], assignment)
"""
    findings = _offline_concept_auditor(ra, _FindingLLM()).audit(
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

    assert findings[0].severity == "error"
    assert "downgraded_reason" not in findings[0].detail


def test_llm_concept_auditor_does_not_downgrade_unused_provenance_flag(ra):
    class _CompanionGatingFindingLLM:
        def complete(self, messages, *, max_tokens=1024, temperature=0.0):
            return json.dumps(
                {
                    "findings": [
                        {
                            "severity": "error",
                            "message": (
                                "First-value covariates can bypass their measured/"
                                "source-status consistency checks."
                            ),
                            "detail": {
                                "issue_code": "audit_only_companion_row_gating_required",
                                "context": (
                                    "The measured flags do not mask or invalidate "
                                    "modeled first-value covariates."
                                ),
                            },
                        }
                    ]
                }
            )

    script = """
measurement_provenance_audit = {
    'checks': [{'invalid_pair_n': 1, 'discordant_n': 0, 'role': 'audit_only'}]
}
provenance_valid = False
model.fit(frame[['marker_first']], assignment)
"""
    findings = _offline_concept_auditor(ra, _CompanionGatingFindingLLM()).audit(
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

    assert findings[0].severity == "error"
    assert "downgraded_reason" not in findings[0].detail


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
    findings = _offline_concept_auditor(ra, _MissingAuditLLM()).audit(
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
                missingness_semantics="Registered sparse event indicator.",
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

    assert '"name":"event_n"' in prompt
    assert '"name":"event_measured"' in prompt
    assert '"name":"event_max"' in prompt
    assert '"role":"intervention"' in prompt
    assert "Registered sparse event indicator." in prompt


def test_llm_concept_auditor_sees_late_independent_count_qc(ra):
    auditor = ra.LLMConceptAuditor(ra.MockLLMClient())
    ctx = ra.build_research_context(
        research_question="Audit an ordered early ICU exposure.",
        cohort=pd.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "stage_max": [0, 1, 2],
                "stage_n": [1, 1, 1],
                "stage_measured": [1, 1, 1],
            }
        ),
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
    from easyicu.research_agent.audits.validators import (
        parse_llm_concept_audit_response,
    )

    raw = """
    {
      "findings": [
        {
          "severity": "error",
          "message": "ICU vs hospital mortality confusion",
          "detail": {
            "issue_code": "other",
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
                    "issue_code": "other",
                    "context": "The script uses death without clarifying whether it is ICU, hospital, or 28-day mortality."
                  }
                }
              ]
            }
            """

    ctx = ra.build_research_context(
        research_question="Is early lactate associated with ICU mortality?",
        cohort=pd.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "lactate_max_24h": [1.0, 2.0, 3.0],
                "death": [0, 1, 0],
            }
        ),
        cohort_name="c",
        database="synthetic",
        target_outcome="death",
    )
    findings = _offline_concept_auditor(ra, _FalsePositiveLLM()).audit(
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
                  "detail": {"issue_code": "other", "context": "The plot labels ICU death as hospital mortality."}
                }
              ]
            }
            """

    ctx = ra.build_research_context(
        research_question="Is early lactate associated with ICU mortality?",
        cohort=pd.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "lactate_max_24h": [1.0, 2.0, 3.0],
                "death": [0, 1, 0],
            }
        ),
        cohort_name="c",
        database="synthetic",
        target_outcome="death",
    )
    findings = _offline_concept_auditor(ra, _ConfusionLLM()).audit(
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
                "issue_code": "other",
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
        "model.fit(x, y)\n" + extra
    )


def test_llm_concept_auditor_downgrades_named_full_stay_horizon_false_positive(
    ra,
) -> None:
    findings = _offline_concept_auditor(ra, _HorizonMismatchLLM()).audit(
        context=_hospital_mortality_context(ra),
        script_text=_full_stay_hospital_mortality_script(),
        step=None,
    )

    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert "full_stay administrative window" in findings[0].detail["downgraded_reason"]


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
    findings = _offline_concept_auditor(ra, _HorizonMismatchLLM()).audit(
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
    assert not any(
        "immortal time bias" in f.message.lower() for f in findings
    ), findings


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
    assert not any(
        "immortal time bias" in f.message.lower() for f in findings
    ), findings


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
    assert not any(
        "immortal time bias" in f.message.lower() for f in findings
    ), findings


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


def test_statistical_guard_warns_when_prediction_outputs_lack_split_metadata(
    ra, tmp_path: Path
):
    ctx = _ctx_with_sofa(ra).model_copy(
        update={
            "user_preferences": ra.schema.UserPreferences(
                inferred_analysis_family="prediction_model",
                covariates=["age", "sex", "sofa2"],
            ),
        }
    )
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "stay_id": list(range(1, 11)),
            "age": [60] * 10,
            "sofa2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "death": [0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
        }
    ).to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pd.DataFrame(
        {
            "model": ["logit"],
            "auc": [0.76],
            "brier": [0.18],
        }
    ).to_csv(out_dir / "model_performance_train_test.csv", index=False)
    step = ra.schema.AnalysisStep(
        step_id="04_prediction", intent="prediction model analysis"
    )
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
    pd.DataFrame(
        {
            "stay_id": list(range(1, 41)),
            "age": [60] * 40,
            "sofa2": list(range(10)) * 4,
            "death": [0, 1] * 20,
        }
    ).to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    step = ra.schema.AnalysisStep(
        step_id="01_model_training", intent="prediction model analysis"
    )

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


def test_statistical_guard_ignores_empty_p_value_placeholder_column(ra, tmp_path: Path):
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
    pd.DataFrame(
        {
            "cluster": list(range(len(sizes))),
            "n": sizes,
            "pct": [s / total * 100.0 for s in sizes],
        }
    ).to_csv(out / "cluster_sizes.csv", index=False)
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
        context=ctx,
        cohort_path=cohort,
        step=step,
        out_dir=out_dir,
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
        context=ctx,
        cohort_path=cohort,
        step=step,
        out_dir=out_dir,
        step_summary={},
    )
    assert not [f for f in findings if "degenerate" in f.message.lower()]


def test_statistical_validator_degeneracy_silent_without_cluster_evidence(
    ra, tmp_path: Path
):
    # Absence of a cluster-size distribution is not degeneracy: a non-clustering
    # step must never trip this caution.
    ctx = _ctx_with_sofa(ra)
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}).to_parquet(cohort, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "primary_association.csv").write_text(
        "variable,odds_ratio\nage,1.1\n", encoding="utf-8"
    )
    step = ra.schema.AnalysisStep(
        step_id="04_primary_association", intent="association"
    )
    findings = ra.StatisticalValidator().audit(
        context=ctx,
        cohort_path=cohort,
        step=step,
        out_dir=out_dir,
        step_summary={},
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
        context=ctx,
        cohort_path=cohort,
        step=step,
        out_dir=out_dir,
        step_summary={},
    )
    deg = [
        f
        for f in findings
        if "single-group" in f.message.lower() or "degenerate" in f.message.lower()
    ]
    assert deg and all(f.severity == "warning" for f in deg)
