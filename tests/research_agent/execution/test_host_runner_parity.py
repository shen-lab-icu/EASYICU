"""Run the same rendered typed scripts through both local execution paths.

All cohorts below are synthetic. Both backends use the same workspace so path
receipts are identical; products are compared as bytes, without normalization.
Docker is deliberately not needed by this desktop regression.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from tests.support.host_runner_parity import (
    select_for_parity as select_standard_executor,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from tests.support import host_runner_cases
from tests.support.host_runner_parity import assert_parity, table_binding


def _case(family):
    if family == "cohort_root":
        from easyicu.research_agent.contracts.primary_cohort import (
            HOST_BOUND_COHORT_METHOD,
            HOST_BOUND_COHORT_PRODUCT,
        )

        step = AnalysisStep(
            step_id="cohort",
            intent="Publish the synthetic cohort",
            method=HOST_BOUND_COHORT_METHOD,
            planned_analysis_role="auxiliary",
            expected_outputs=[HOST_BOUND_COHORT_PRODUCT],
        )
        return (
            pd.DataFrame({"measurement": [1.0, 2.0, 3.0]}),
            step,
            AnalysisPlan(research_question="Synthetic cohort", steps=[step]),
            {},
        )
    if family == "continuous_outcomes":
        step = AnalysisStep(
            step_id="durations",
            intent="Summarize synthetic durations by group",
            method="continuous_outcome_summary",
            planned_analysis_role="secondary",
            inputs=["artifact:analysis_cohort", "group", "duration"],
            expected_outputs=["table:continuous_outcome_summary"],
        )
        context = {
            "variables": [
                {
                    "name": "group",
                    "dtype": "int64",
                    "observed_domain": {"levels": [0, 1], "n_unique": 2},
                },
                {
                    "name": "duration",
                    "role": "outcome",
                    "dtype": "float64",
                    "unit": "hours",
                },
            ]
        }
        return (
            pd.DataFrame(
                {
                    "group": [0, 0, 0, 1, 1, 1],
                    "duration": [1.0, 3.0, None, 2.0, 4.0, 8.0],
                }
            ),
            step,
            AnalysisPlan(research_question="Synthetic durations", steps=[step]),
            context,
        )
    if family == "exposure_distribution":
        step = AnalysisStep(
            step_id="distribution",
            intent="Summarize synthetic exposure and response",
            method="descriptive",
            planned_analysis_role="auxiliary",
            inputs=["artifact:analysis_cohort", "exposure", "response"],
            expected_outputs=["table:exposure_outcome_distribution"],
            exposure_outcome_distribution_spec={
                "schema_version": "easyicu.exposure_outcome_distribution/3",
                "exposure": "exposure",
                "exposure_levels": [0, 1],
                "outcome": "response",
                "outcome_levels": [0, 1],
                "outcome_positive_value": 1,
                "level_match_policy": "exact_typed",
                "denominator_policy": "all_declared_rows",
                "missing_outcome_policy": "structural_absence_is_non_event",
                "interval_method": "none_counts_only",
                "repeated_unit_interval_method": None,
                "confidence_level": None,
            },
        )
        return (
            pd.DataFrame({"exposure": [0, 0, 1, 1], "response": [0, 1, 0, 1]}),
            step,
            AnalysisPlan(research_question="Synthetic distribution", steps=[step]),
            {},
        )
    if family == "missingness":
        step = AnalysisStep(
            step_id="audit",
            intent="Audit synthetic measurement availability",
            method="missingness_and_source_availability_audit",
            inputs=[
                "artifact:analysis_cohort",
                "lactate",
                "lactate_measured",
                "lactate_n",
            ],
            expected_outputs=[
                "table:missingness_audit",
                "table:measurement_source_audit",
            ],
        )
        return (
            host_runner_cases.deterministic_missingness_cohort(n=20),
            step,
            AnalysisPlan(research_question="Synthetic availability", steps=[step]),
            {},
        )
    modules = {
        "summary": "cohort_summary",
        "table_one": "table_one",
        "ordered": "ordered_stratified",
        "prediction": "prediction_model",
        "phenotyping": "cross_sectional_phenotyping",
        "survival": "survival_primary",
        "distribution": "descriptive_distribution",
        "association": "descriptive_association",
    }
    if family == "adjusted":
        rng = np.random.default_rng(817)
        frame = pd.DataFrame(
            {
                "exposure": rng.integers(0, 2, 160),
                "age": rng.normal(50, 8, 160),
                "response": rng.binomial(1, 0.4, 160),
            }
        )
        step = AnalysisStep(
            step_id="adjusted",
            intent="Estimate the synthetic adjusted association",
            method="adjusted_association_models",
            planned_analysis_role="primary",
            inputs=["artifact:analysis_cohort", "exposure", "age", "response"],
            expected_outputs=["table:adjusted_association_estimates"],
            model_requirements=[
                {
                    "requirement_id": "primary",
                    "outcome": "response",
                    "outcome_type": "binary",
                    "method_family": "statsmodels_logit_mle",
                    "exposure_source": "exposure",
                    "analysis_role": "primary",
                    "analysis_set": "source_aware",
                    "covariates": ["age"],
                    "model_terms": [
                        {
                            "name": "exposure",
                            "role": "exposure",
                            "coding": "binary",
                            "levels": ["0", "1"],
                            "reference_level": "0",
                            "transform": "treatment_contrast",
                        },
                        {
                            "name": "age",
                            "role": "covariate",
                            "coding": "continuous",
                            "transform": "identity",
                        },
                    ],
                }
            ],
        )
        context = {}
        plan = AnalysisPlan(research_question="Synthetic association", steps=[step])
        return frame, step, plan, context
    fixture = getattr(host_runner_cases, modules[family])
    step = (
        fixture._primary_step()
        if family in {"prediction", "phenotyping"}
        else fixture._step()
    )
    if family in {"distribution", "association"}:
        frame = pd.DataFrame(
            {
                "sex": ["F", "M"] * 4,
                "age": np.arange(8) + 30.0,
                "los_icu": [1.0, 3.0, 2.0, 6.0, 7.0, 4.0, 8.0, 5.0],
            }
        )
    elif family == "survival":
        rng = np.random.default_rng(72)
        frame = pd.DataFrame(
            {
                "treatment": rng.integers(0, 2, 160),
                "age": rng.normal(50, 10, 160),
                "death": rng.binomial(1, 0.6, 160),
                "follow_up_days": rng.uniform(1, 28, 160),
            }
        )
    else:
        frame = fixture._frame()
    plan = (
        fixture._plan(step)
        if family == "ordered"
        else AnalysisPlan(research_question="Synthetic observations", steps=[step])
    )
    context = (
        fixture._context(len(frame))
        if family in {"prediction", "phenotyping"}
        else fixture._context()
        if family == "summary"
        else {}
    )
    if hasattr(context, "model_dump"):
        context = context.model_dump(mode="json")
    if family == "phenotyping":
        next(
            variable for variable in context["variables"] if variable["name"] == "death"
        )["observed_domain"] = {"is_binary": True, "levels": [0, 1]}
    if family == "distribution":
        context = {
            "variables": [
                {
                    "name": "sex",
                    "dtype": "object",
                    "observed_domain": {"levels": ["F", "M"], "n_unique": 2},
                },
                {"name": "los_icu", "dtype": "float64"},
            ]
        }
    return frame, step, plan, context


@pytest.mark.parametrize(
    "family",
    [
        "summary",
        "distribution",
        "association",
        "table_one",
        "ordered",
        "adjusted",
        "prediction",
        "phenotyping",
        "survival",
        "missingness",
        "cohort_root",
        "continuous_outcomes",
        "exposure_distribution",
    ],
)
def test_typed_executor_products_match_code_runner(tmp_path, family):
    frame, step, plan, context = _case(family)
    cohort = tmp_path / "cohort.parquet"
    frame.to_parquet(cohort, index=False)
    (tmp_path / "research_context.json").write_text(json.dumps(context))
    selected = select_standard_executor(step, plan=plan)
    assert selected is not None, family
    sha = hashlib.sha256(cohort.read_bytes()).hexdigest()
    inputs = {}
    for key in selected.consumed_input_keys:
        if family == "ordered" and key == "table:adjusted_association_estimates":
            # The trend refit is checked against the primary's own estimates.
            inputs[key] = host_runner_cases.ordered_stratified._primary_binding(
                plan, frame, tmp_path,
            )
            continue
        inputs[key] = {
            "relative_path": cohort.name,
            "sha256": sha,
            "declared_kind": key.split(":")[0],
            "product": key.split(":")[1],
            "evidence_kind": key.split(":")[0],
            "evidence_id": "synthetic-cohort",
            "produced_by_step": "source",
            "product_contract": {
                "columns": list(frame.columns),
                "row_count": len(frame),
            },
        }
    manifest = tmp_path / "resolved_inputs.json"
    manifest.write_text(json.dumps({"step_id": step.step_id, "inputs": inputs}))
    extra_env = {
        "EASYICU_RESEARCH_CONTEXT": str(tmp_path / "research_context.json"),
        "OUTCOME_COL": "death",
        "SOURCE_DATE_EPOCH": "1700000000",
    }
    assert_parity(tmp_path, cohort, step, selected.code, manifest, extra_env)
    assert hashlib.sha256(cohort.read_bytes()).hexdigest() == sha
    if family in {"prediction", "phenotyping"}:
        _check_downstream(tmp_path, cohort, step, family, extra_env)
    if family == "exposure_distribution":
        output = tmp_path / "steps" / step.step_id / "outputs"
        key = "table:exposure_outcome_distribution"
        binding = table_binding(
            tmp_path, key, pd.read_csv(output / "exposure_outcome_distribution.csv")
        )
        figure = AnalysisStep(
            step_id="distribution_figure",
            intent="Render generated counts",
            method="visualization",
            planned_analysis_role="auxiliary",
            inputs=[key],
            expected_outputs=["figure:distribution"],
            input_consumption_contracts=[{"input_key": key, "mode": "all_rows"}],
        )
        selected = select_standard_executor(
            figure,
            plan=AnalysisPlan(research_question="Synthetic counts", steps=[figure]),
            resolved_bindings={key: binding},
        )
        assert selected.analysis_kind == "exposure_outcome_distribution_figure"
        manifest.write_text(
            json.dumps({"step_id": figure.step_id, "inputs": {key: binding}})
        )
        assert_parity(tmp_path, cohort, figure, selected.code, manifest, extra_env)
        from easyicu.research_agent.execution.runners.adjusted_association_executor import (
            ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS,
        )

        result_key = "table:adjusted_association_estimates"
        result = pd.DataFrame(
            [
                {
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
                    "notes": "Invented upstream estimate",
                }
            ],
            columns=ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS,
        )
        # The overview's interval grammar consumes an upstream interval table.
        # This separate invented table carries explicit conservative bounds.
        interval_frame = pd.read_csv(output / "exposure_outcome_distribution.csv")
        for column in ("exposure_ci_low_pct", "ci_low_pct"):
            interval_frame[column] = 0.0
        for column in ("exposure_ci_high_pct", "ci_high_pct"):
            interval_frame[column] = 100.0
        binding = table_binding(tmp_path, key, interval_frame)
        bindings = {
            key: binding,
            result_key: table_binding(tmp_path, result_key, result),
        }
        figure = figure.model_copy(
            update={
                "step_id": "overview",
                "inputs": list(bindings),
                "input_consumption_contracts": [],
            }
        )
        figure = AnalysisStep.model_validate(
            {
                **figure.model_dump(mode="json"),
                "input_consumption_contracts": [
                    {"input_key": input_key, "mode": "all_rows"}
                    for input_key in bindings
                ],
            }
        )
        selected = select_standard_executor(
            figure,
            plan=AnalysisPlan(research_question="Synthetic overview", steps=[figure]),
            resolved_bindings=bindings,
        )
        assert selected.analysis_kind == "association_overview_figure"
        manifest.write_text(json.dumps({"step_id": figure.step_id, "inputs": bindings}))
        assert_parity(tmp_path, cohort, figure, selected.code, manifest, extra_env)


def _check_downstream(workdir, cohort, primary, family, extra_env):
    """Replay the diagnostic actions and their figure from real primary products."""
    from easyicu.research_agent.contracts.prediction_execution import (
        STATIC_PREDICTION_ACTION_OUTPUTS,
    )
    from easyicu.research_agent.contracts.phenotyping_execution import (
        CROSS_SECTIONAL_PHENOTYPING_ACTION_OUTPUTS,
    )
    from easyicu.research_agent.execution.runners.prediction_figure_executor import (
        PREDICTION_COMPOSITE_FIGURE_INPUTS,
    )
    from easyicu.research_agent.execution.runners.cross_sectional_phenotyping_figure_executor import (
        PHENOTYPING_FIGURE_INPUTS,
    )

    products = {}

    def collect(step):
        output = workdir / "steps" / step.step_id / "outputs"
        summary = json.loads((output / "step_summary.json").read_text())
        for key, filename in summary["output_files"].items():
            if key.startswith("table:"):
                products[key] = table_binding(
                    workdir, key, pd.read_csv(output / filename)
                )

    def run(step):
        bindings = {key: products[key] for key in step.inputs if ":" in key}
        selected = select_standard_executor(
            step,
            plan=AnalysisPlan(research_question="Synthetic diagnostics", steps=[step]),
            resolved_bindings=bindings,
        )
        assert selected is not None, step
        manifest = workdir / "resolved_inputs.json"
        manifest.write_text(json.dumps({"step_id": step.step_id, "inputs": bindings}))
        assert_parity(workdir, cohort, step, selected.code, manifest, extra_env)
        return selected

    collect(primary)
    actions = (
        [
            "prediction.internal_validation",
            "prediction.calibration_metrics",
            "prediction.decision_curve",
        ]
        if family == "prediction"
        else ["phenotyping.k_selection", "phenotyping.cluster_stability"]
    )
    roster = (
        STATIC_PREDICTION_ACTION_OUTPUTS
        if family == "prediction"
        else CROSS_SECTIONAL_PHENOTYPING_ACTION_OUTPUTS
    )
    input_key = (
        "table:prediction_scores"
        if family == "prediction"
        else "table:phenotype_assignments"
    )
    for index, action in enumerate(actions):
        step = AnalysisStep(
            step_id=f"diagnostic_{index}",
            intent="Validate the frozen synthetic fit",
            planned_analysis_role="secondary",
            method="typed_diagnostic",
            scientific_action_id=action,
            inputs=[input_key],
            expected_outputs=list(roster[action]),
        )
        run(step)
        collect(step)
    keys = (
        PREDICTION_COMPOSITE_FIGURE_INPUTS
        if family == "prediction"
        else PHENOTYPING_FIGURE_INPUTS
    )
    step = AnalysisStep(
        step_id="diagnostic_figure",
        intent="Render frozen synthetic diagnostics",
        planned_analysis_role="auxiliary",
        method="visualization",
        inputs=list(keys),
        expected_outputs=["figure:diagnostic_figure"],
        input_consumption_contracts=[
            {"input_key": key, "mode": "all_rows"} for key in keys
        ],
    )
    selected = run(step)
    assert selected.host_sealed_renderer
    if family == "phenotyping":
        frame = pd.read_parquet(cohort)
        products["artifact:analysis_cohort"] = {
            "relative_path": cohort.name,
            "sha256": hashlib.sha256(cohort.read_bytes()).hexdigest(),
            "declared_kind": "artifact",
            "evidence_kind": "table",
            "product": "analysis_cohort",
            "evidence_id": "synthetic-cohort",
            "produced_by_step": "source",
            "product_contract": {
                "columns": list(frame.columns),
                "row_count": len(frame),
            },
            "consumption_contract": {
                "input_key": "artifact:analysis_cohort",
                "mode": "all_rows",
                "verified_row_count": len(frame),
                "artifact_sha256": hashlib.sha256(cohort.read_bytes()).hexdigest(),
            },
        }
        comparison = AnalysisStep(
            step_id="comparison",
            intent="Describe the frozen synthetic classes",
            method="descriptive_profile_by_frozen_cluster",
            scientific_action_id="phenotyping.outcome_by_cluster",
            planned_analysis_role="secondary",
            inputs=[
                "stay_id",
                "marker_a",
                "death",
                "artifact:analysis_cohort",
                "table:phenotype_assignments",
            ],
            expected_outputs=["table:outcome_by_cluster"],
            phenotype_comparison_spec={
                "identity_column": "stay_id",
                "outcome_columns": ["death"],
                "variables": [
                    {
                        "name": "marker_a",
                        "variable_kind": "continuous",
                        "summary": "median_iqr",
                        "test": "none_descriptive_smd_only",
                    },
                    {
                        "name": "death",
                        "variable_kind": "categorical",
                        "summary": "count_percent",
                        "test": "none_descriptive_smd_only",
                        "levels": [0, 1],
                    },
                ],
            },
        )
        assert run(comparison).analysis_kind == "phenotype_comparison"


@pytest.mark.parametrize("kind", ["table", "statistic", "audit_panel"])
def test_typed_figure_products_match_code_runner(tmp_path, kind):
    fixture = host_runner_cases.descriptive_result_figure

    if kind == "audit_panel":
        step = host_runner_cases.audit_panel_step()
        run_dir = tmp_path / "run"
        prior = run_dir / "steps" / "prior" / "outputs"
        prior.mkdir(parents=True)
        (prior / "step_summary.json").write_text(
            json.dumps({"status": "ok", "missingness_audit": {"missing_n": 2}})
        )
        manifest, bindings = {"inputs": {}}, {}
    else:
        step = (
            fixture._distribution_step()
            if kind == "table"
            else fixture._statistic_step()
        )
        run_dir, manifest, binding = fixture._binding(tmp_path, step=step, kind=kind)
        bindings = {step.inputs[0]: binding}
    cohort = run_dir / "cohort.parquet"
    pd.DataFrame({"measurement": [1.0, 2.0, 3.0]}).to_parquet(cohort, index=False)
    selected = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Synthetic figure", steps=[step]),
        resolved_bindings=bindings,
    )
    assert selected is not None
    manifest_path = run_dir / "resolved_inputs.json"
    manifest_path.write_text(json.dumps(manifest))
    assert_parity(
        run_dir,
        cohort,
        step,
        selected.code,
        manifest_path,
        {"SOURCE_DATE_EPOCH": "1700000000"},
    )


@pytest.mark.parametrize("method", ["scientific_reporting", "feasibility_protocol"])
def test_typed_report_products_match_code_runner(tmp_path, method):
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"measurement": [1.0, 2.0, 3.0]}).to_parquet(cohort, index=False)
    result = tmp_path / "primary.csv"
    result.write_text("estimate,lower,upper\n1.5,1.1,2.0\n")
    performance = tmp_path / "performance.json"
    performance.write_text('{"auroc": 0.81}\n')
    paths = {"table:primary_results": result}
    paths.update(
        {"statistic:model_performance": performance}
        if method == "scientific_reporting"
        else {"artifact:analysis_cohort": cohort}
    )
    step = AnalysisStep(
        step_id="report",
        intent="Index synthetic estimates and their source evidence",
        method=method,
        planned_analysis_role="auxiliary"
        if method == "scientific_reporting"
        else "sensitivity",
        inputs=list(paths),
        expected_outputs=["report:analysis_results"],
    )
    selected = select_standard_executor(
        step, plan=AnalysisPlan(research_question="Synthetic results", steps=[step])
    )
    assert selected is not None
    inputs = {}
    for key, path in paths.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        inputs[key] = {
            "relative_path": path.name,
            "declared_kind": key.split(":")[0],
            "sha256": digest,
            "evidence_id": key,
            "produced_by_step": "primary",
            "identity_row": {"input_key": key, "sha256": digest},
        }
    manifest = tmp_path / "resolved_inputs.json"
    manifest.write_text(json.dumps({"step_id": step.step_id, "inputs": inputs}))
    assert_parity(
        tmp_path,
        cohort,
        step,
        selected.code,
        manifest,
        {"SOURCE_DATE_EPOCH": "1700000000"},
    )
