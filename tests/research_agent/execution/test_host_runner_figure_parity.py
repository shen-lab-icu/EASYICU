"""Specialized typed renderers preserve their source data and image bytes."""

import json

import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS,
)
from easyicu.research_agent.execution.runners.deterministic_robustness import (
    _MATRIX_COLUMNS,
)
from tests.support.host_runner_parity import (
    select_for_parity as select_standard_executor,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from tests.support import host_runner_figure_cases as cases
from tests.support.host_runner_parity import assert_parity, table_binding


def _frames(owner):
    if owner == "cohort_flow_figure":
        return {"table:cohort_flow": cases.cohort_flow_frames()}
    if owner == "prevalence_outcome_figure":
        return {"table:absolute_risk_context": cases.prevalence_outcome_frames()}
    if owner == "prevalence_mortality_figure":
        return {
            "table:cohort_summary": cases.prevalence_cohort_frames(),
            "table:outcome_incidence": cases.prevalence_incidence_frames(),
        }
    if owner == "composite_descriptive_figure":
        return cases.composite_frames()
    if owner == "landmark_association_composite_figure":
        return cases.landmark_frames()
    if owner == "adjusted_association_figure":
        row = {
            "fit_status": "fitted",
            "estimate": 1.5,
            "ci_low": 1.2,
            "ci_high": 1.9,
            "effect_scale": "odds_ratio",
            "exposure": "exposure",
            "requirement_id": "primary",
            "outcome": "response",
            "covariates": "age",
            "estimator_kind": "logistic",
            "analysis_set": "complete_case",
            "n": 100,
            "n_events": 30,
            "standard_error": 0.12,
            "notes": "Synthetic fit",
        }
        return {
            "table:adjusted_association_estimates": pd.DataFrame(
                [row], columns=ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS
            )
        }
    if owner == "robustness_figure":
        row = {
            "spec_id": "primary",
            "effect_scale": "OR",
            "point_estimate": 1.5,
            "ci_low": 1.2,
            "ci_high": 1.9,
            "modeled_analytic_n": 100,
            "axis": "primary",
            "converged": True,
            "estimability_status": "estimated",
            "membership_n": 100,
            "membership_executable": True,
            "notes": "Synthetic fit",
        }
        return {"table:robustness_matrix": pd.DataFrame([row], columns=_MATRIX_COLUMNS)}
    if owner in {"measurement_missingness_figure", "missingness_measurement_figure"}:
        row = {
            "concept": "measurement",
            "variable": "measurement",
            "label": "Measurement",
            "value_column": "measurement",
            "n_total": 10,
            "measured_one_n": 8,
            "measured_one_pct": 80.0,
            "value_missing_n": 2,
            "value_missing_pct": 20.0,
            "eligible_n": 10,
            "not_applicable_n": 0,
            "indicator_semantics": "measurement_availability",
            "missingness_kind": "ordinary",
        }
        frames = {"table:missingness_measurement_audit": pd.DataFrame([row])}
        if owner == "missingness_measurement_figure":
            frames["table:measurement_process_audit"] = pd.DataFrame(
                [
                    {
                        **row,
                        "measurement_total_n": 16,
                        "measurement_count_max": 2,
                        "repeat_measured_n": 8,
                    }
                ]
            )
        return frames
    raise AssertionError(owner)


@pytest.mark.parametrize(
    "owner",
    [
        "cohort_flow_figure",
        "prevalence_outcome_figure",
        "prevalence_mortality_figure",
        "composite_descriptive_figure",
        "landmark_association_composite_figure",
        "adjusted_association_figure",
        "robustness_figure",
        "measurement_missingness_figure",
        "missingness_measurement_figure",
    ],
)
def test_specialized_figure_products_match(tmp_path, owner):
    frames = _frames(owner)
    step = AnalysisStep(
        step_id="render",
        planned_analysis_role="auxiliary",
        intent="Render invented result tables",
        method="visualization",
        inputs=list(frames),
        expected_outputs=[
            "figure:prevalence_mortality"
            if owner.startswith("prevalence_")
            else "figure:synthetic_results"
        ],
        input_consumption_contracts=[
            {"input_key": key, "mode": "all_rows"} for key in frames
        ],
    )
    bindings = {
        key: table_binding(tmp_path, key, frame) for key, frame in frames.items()
    }
    plan = AnalysisPlan(research_question="Synthetic observations", steps=[step])
    selected = select_standard_executor(step, plan=plan, resolved_bindings=bindings)
    assert selected is not None, owner
    assert selected.analysis_kind == owner
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"measurement": [1.0, 2.0]}).to_parquet(cohort, index=False)
    manifest = tmp_path / "resolved_inputs.json"
    manifest.write_text(json.dumps({"step_id": step.step_id, "inputs": bindings}))
    assert_parity(
        tmp_path,
        cohort,
        step,
        selected.code,
        manifest,
        {"SOURCE_DATE_EPOCH": "1700000000"},
    )
