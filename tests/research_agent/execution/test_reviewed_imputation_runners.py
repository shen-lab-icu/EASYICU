"""Signed MI reaches host fits; failures never turn into complete-case primaries."""

import json

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.contracts.model_terms import (
    DataPreparationPlan,
    preparation_method,
)
from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    adjusted_association_executor_code,
    run_adjusted_association_from_env,
)
from easyicu.research_agent.methods.missing_data import ImputationRefusal
from easyicu.research_agent.schema import AnalysisStep


def _choice(strategy="multiple_imputation"):
    return DataPreparationPlan(
        missing_data={
            "strategy": strategy,
            "m": 4,
            "iterations": 3,
            "variables": [{"name": "x", "method": "pmm"}],
            "auxiliaries": ["aux"],
        }
    )


def _frame():
    rng = np.random.default_rng(512)
    x, z = rng.normal(size=(2, 220))
    result = pd.DataFrame(
        {
            "x": x,
            "z": z,
            "aux": x + rng.normal(size=220),
            "y": x + z + rng.normal(size=220),
        }
    )
    result.loc[::3, "x"] = np.nan
    return result


def _arguments():
    return dict(
        requirement_id="fit",
        exposure="z",
        outcome="y",
        covariates=["x"],
        model_terms=[
            {
                "name": "z",
                "role": "exposure",
                "coding": "continuous",
                "transform": "identity",
            },
            {
                "name": "x",
                "role": "covariate",
                "coding": "continuous",
                "transform": "identity",
            },
        ],
        estimator_kind="linear",
        analysis_set="source_aware",
        analysis_role="primary",
        method_family="statsmodels_ols",
    )


def test_association_pools_m_imputations_and_keeps_complete_case_sensitivity(tmp_path):
    summary = run_adjusted_association_from_env(
        frame=_frame(),
        output_dir=tmp_path,
        emit_step_summary=False,
        data_preparation=_choice(),
        **_arguments(),
    )
    assert summary["n_total"] == 220
    assert (
        summary["model_contracts"][0]["baseline_missing_policy"]
        == "multiple_imputation"
    )
    receipt = json.loads((tmp_path / "data_preparation_provenance.json").read_text())
    assert receipt["missing_data"]["m"] == 4
    assert receipt["missing_data"]["pooling"]["z"]["between_variance"] > 0
    assert receipt["sensitivity"]["strategy"] == "complete_case"
    assert receipt["sensitivity"]["n"] < 220
    assert "Rubin" in receipt["methods_text"]


def test_complete_case_primary_reports_mi_as_sensitivity(tmp_path):
    summary = run_adjusted_association_from_env(
        frame=_frame(),
        output_dir=tmp_path,
        emit_step_summary=False,
        data_preparation=_choice("complete_case"),
        **_arguments(),
    )
    assert summary["n_total"] < 220
    receipt = json.loads((tmp_path / "data_preparation_provenance.json").read_text())
    assert receipt["sensitivity"]["strategy"] == "multiple_imputation"
    assert receipt["sensitivity"]["n"] == 220


def test_an_unsupported_mi_primary_refuses_without_any_primary_table(tmp_path):
    frame = _frame()
    frame.x = np.nan
    with pytest.raises(ImputationRefusal) as caught:
        run_adjusted_association_from_env(
            frame=frame,
            output_dir=tmp_path,
            emit_step_summary=False,
            data_preparation=_choice(),
            **_arguments(),
        )
    assert caught.value.reason_code == "mi_insufficient_observations"
    assert not (tmp_path / "adjusted_association_estimates.csv").exists()


def test_signed_mi_choice_is_in_the_sealed_association_script():
    args = _arguments()
    requirement = {
        key: args[key]
        for key in (
            "requirement_id",
            "outcome",
            "covariates",
            "model_terms",
            "method_family",
            "analysis_set",
            "analysis_role",
        )
    }
    requirement.update(outcome_type="continuous", exposure_source="z")
    step = AnalysisStep(
        step_id="fit",
        intent="Fit the signed adjusted association",
        planned_analysis_role="primary",
        method=preparation_method("adjusted_association_models", _choice()),
        inputs=["artifact:analysis_cohort"],
        expected_outputs=["table:adjusted_association_estimates"],
        model_requirements=[requirement],
    )
    code = adjusted_association_executor_code(step)
    assert "data_preparation" in code
    assert "multiple_imputation" in code
    assert "'m': 4" in code


def test_the_retention_probe_counts_the_signed_mi_policy(tmp_path):
    from easyicu.research_agent.execution.primary_model_retention import (
        measure_primary_model_retention,
    )
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        CohortDescriptor,
        ResearchContext,
    )

    args = _arguments()
    requirement = {
        key: args[key]
        for key in (
            "requirement_id",
            "outcome",
            "covariates",
            "model_terms",
            "method_family",
            "analysis_set",
            "analysis_role",
        )
    }
    requirement.update(outcome_type="continuous", exposure_source="z")
    step = AnalysisStep(
        step_id="fit",
        intent="Fit the signed adjusted association",
        planned_analysis_role="primary",
        method=preparation_method("adjusted_association_models", _choice()),
        inputs=["artifact:analysis_cohort"],
        expected_outputs=["table:adjusted_association_estimates"],
        model_requirements=[requirement],
    )
    frame = _frame()
    path = tmp_path / "synthetic.parquet"
    frame.to_parquet(path)
    context = ResearchContext(
        research_question="Synthetic association",
        variables=[],
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_stays=len(frame),
            n_patients=len(frame),
        ),
    )
    evidence = measure_primary_model_retention(
        context=context,
        plan=AnalysisPlan(research_question="Synthetic association", steps=[step]),
        cohort_path=path,
    )
    assert evidence.status == "measured"
    result = evidence.requirements[0]
    assert result.policy == "multiple_imputation"
    assert result.model_n == len(frame)
    assert result.complete_case_n < result.model_n


def test_pool_failure_cannot_turn_mi_into_a_complete_case_primary(
    tmp_path, monkeypatch
):
    from easyicu.research_agent.execution import model_matrix as owner

    def fail(*args, **kwargs):
        raise ImputationRefusal("mi_analysis_fit_failed", "Synthetic pooling failure")

    monkeypatch.setattr(owner, "_pool_association", fail)
    with pytest.raises(ImputationRefusal, match="mi_analysis_fit_failed"):
        run_adjusted_association_from_env(
            frame=_frame(),
            output_dir=tmp_path,
            emit_step_summary=False,
            data_preparation=_choice(),
            **_arguments(),
        )
    assert not (tmp_path / "adjusted_association_estimates.csv").exists()


def test_association_ratios_are_pooled_on_the_log_coefficient_scale():
    from easyicu.research_agent.execution.model_matrix import _pool_association
    from easyicu.research_agent.robustness.estimators import (
        EstimatorResult,
        EstimatorTerm,
    )

    fits = [
        EstimatorResult(
            float(np.exp(value)),
            0.5,
            2.0,
            0.2,
            100,
            True,
            terms=(EstimatorTerm("x", "x", float(np.exp(value)), 0.5, 2.0, 0.2),),
        )
        for value in (0.1, 0.2, 0.3)
    ]
    result, diagnostics = _pool_association(fits, focal_term="x", kind="logistic")
    assert result.point_estimate == pytest.approx(np.exp(0.2))
    assert diagnostics["x"]["within_variance"] == pytest.approx(0.04)
    assert diagnostics["x"]["between_variance"] == pytest.approx(0.01)


def test_pooled_primary_passes_the_existing_model_contract_audit(tmp_path):
    from easyicu.research_agent.audits.validators import PrimaryModelContractValidator
    from easyicu.research_agent.schema import (
        CohortDescriptor,
        ConceptDescriptor,
        ResearchContext,
        VariableRole,
    )

    arguments = _arguments()
    requirement = {
        key: arguments[key]
        for key in (
            "requirement_id",
            "outcome",
            "covariates",
            "model_terms",
            "method_family",
            "analysis_set",
            "analysis_role",
        )
    }
    requirement.update(outcome_type="continuous", exposure_source="z")
    step = AnalysisStep(
        step_id="fit",
        intent="Fit the synthetic association",
        planned_analysis_role="primary",
        method=preparation_method("adjusted_association_models", _choice()),
        expected_outputs=["table:adjusted_association_estimates"],
        inputs=["artifact:analysis_cohort"],
        model_requirements=[requirement],
    )
    frame = _frame()
    source = tmp_path / "synthetic.parquet"
    frame.to_parquet(source, index=False)
    output = tmp_path / "outputs"
    summary = run_adjusted_association_from_env(
        frame=frame,
        output_dir=output,
        emit_step_summary=False,
        data_preparation=_choice(),
        **arguments,
    )
    context = ResearchContext(
        research_question="Synthetic association",
        primary_exposure="z",
        target_outcome="y",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_stays=len(frame),
            n_patients=len(frame),
        ),
        variables=[
            ConceptDescriptor(
                name=name,
                dtype="float64",
                role=VariableRole.OUTCOME if name == "y" else VariableRole.OTHER,
            )
            for name in frame
        ],
    )
    findings = PrimaryModelContractValidator().audit(
        step=step,
        step_summary=summary,
        context=context,
        completed_step_records=[],
        out_dir=output,
        cohort_path=source,
    )
    assert findings == [], json.dumps([finding.model_dump(mode="json") for finding in findings])
