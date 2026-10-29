"""Validation observations cannot train any imputation conditional or donor set."""

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.contracts.model_terms import DataPreparationPlan
from easyicu.research_agent.execution.runners import prediction_model_executor as owner


def _input():
    rng = np.random.default_rng(34)
    n = 180
    x, z = rng.normal(size=(2, n))
    frame = pd.DataFrame({"x": x, "z": z})
    outcome = pd.Series((rng.random(n) < 1 / (1 + np.exp(-x))).astype(int), name="y")
    frame.loc[::4, "x"] = np.nan
    plan = DataPreparationPlan(
        missing_data={
            "strategy": "multiple_imputation",
            "m": 3,
            "iterations": 3,
            "variables": [{"name": "x", "method": "pmm"}],
        }
    )
    return frame, outcome, plan


def test_validation_labels_never_enter_imputation_or_change_predictions():
    frame, outcome, plan = _input()
    training = np.arange(len(frame)) < 140
    rows = ~training
    first = owner._fit_probabilities(
        frame=frame,
        outcome=outcome,
        features=("x", "z"),
        development=training,
        prediction_rows=rows,
        data_preparation=plan,
    )
    changed = outcome.copy()
    changed.loc[rows] = 1 - changed.loc[rows]
    second = owner._fit_probabilities(
        frame=frame,
        outcome=changed,
        features=("x", "z"),
        development=training,
        prediction_rows=rows,
        data_preparation=plan,
    )
    np.testing.assert_array_equal(first, second)


def test_only_training_rows_fit_the_imputer_and_each_split_refits(monkeypatch):
    from easyicu.research_agent.execution.runners import (
        prediction_model_executor as mi_owner,
    )

    frame, outcome, plan = _input()
    seen = []
    original = mi_owner.multiple_impute

    def capture(data, **kwargs):
        seen.append(set(data.index))
        return original(data, **kwargs)

    monkeypatch.setattr(mi_owner, "multiple_impute", capture)
    for cutoff in (130, 140):
        training = np.arange(len(frame)) < cutoff
        owner._fit_probabilities(
            frame=frame,
            outcome=outcome,
            features=("x", "z"),
            development=training,
            prediction_rows=~training,
            data_preparation=plan,
        )
    assert seen == [set(range(130)), set(range(140))]


def test_validation_values_cannot_rescue_an_all_missing_training_predictor():
    from easyicu.research_agent.methods.missing_data import ImputationRefusal

    frame, outcome, plan = _input()
    training = np.arange(len(frame)) < 140
    frame.loc[training, "x"] = np.nan
    frame.loc[~training, "x"] = 1000
    with pytest.raises(ImputationRefusal) as caught:
        owner._fit_probabilities(
            frame=frame,
            outcome=outcome,
            features=("x", "z"),
            development=training,
            prediction_rows=~training,
            data_preparation=plan,
        )
    assert caught.value.reason_code == "mi_insufficient_observations"


def test_validation_uncertainty_is_pooled_across_imputations():
    frame, outcome, plan = _input()
    training = np.arange(len(frame)) < 140
    diagnostics = {}
    owner._fit_probabilities(
        frame=frame,
        outcome=outcome,
        features=("x", "z"),
        development=training,
        prediction_rows=~training,
        data_preparation=plan,
        imputation_diagnostics=diagnostics,
    )
    pool = diagnostics["pooled_auroc_logit"]
    assert pool["within_variance"] > 0
    assert pool["between_variance"] >= 0
    assert pool["total_variance"] >= pool["within_variance"]
    assert 0 < diagnostics["pooled_auroc"]["estimate"] < 1
    aucs = np.asarray([item["auroc"] for item in diagnostics["per_imputation_auroc"]])
    errors = np.asarray(
        [item["standard_error"] for item in diagnostics["per_imputation_auroc"]]
    )
    assert pool["estimate"] == pytest.approx(np.log(aucs / (1 - aucs)).mean())
    assert pool["within_variance"] == pytest.approx(
        np.square(errors / (aucs * (1 - aucs))).mean()
    )


def test_primary_prediction_refuses_missing_within_imputation_uncertainty(monkeypatch):
    from types import SimpleNamespace
    from easyicu.research_agent.contracts.model_terms import PreparationRefusal

    frame, outcome, plan = _input()
    training = np.arange(len(frame)) < 140
    monkeypatch.setattr(
        owner,
        "auc_interval",
        lambda *args: SimpleNamespace(auc=0.7, se=0.0, method="synthetic"),
    )
    with pytest.raises(PreparationRefusal, match="mi_prediction_pooling_not_estimable"):
        owner._fit_probabilities(
            frame=frame,
            outcome=outcome,
            features=("x", "z"),
            development=training,
            prediction_rows=~training,
            data_preparation=plan,
            imputation_diagnostics={},
        )


def test_prediction_runner_publishes_the_signed_policy_and_comparison(
    tmp_path, monkeypatch
):
    from easyicu.research_agent.schema import (
        CohortDescriptor,
        ConceptDescriptor,
        ResearchContext,
        VariableRole,
    )

    frame, outcome, plan = _input()
    frame["y"] = outcome
    frame["patient_stay_id"] = [
        f"p{index // 2}:s{index % 2}" for index in range(len(frame))
    ]
    context = ResearchContext(
        research_question="Predict the synthetic outcome",
        target_outcome="y",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_stays=len(frame),
            id_columns=["patient_stay_id"],
            provenance={
                "replacement_row_identity": {
                    "output_identity_column": "patient_stay_id",
                    "mapping_file_sha256": "a" * 64,
                    "patient_group_derivation": {
                        "algorithm": "prefix_before_:s",
                        "delimiter": ":s",
                    },
                }
            },
        ),
        variables=[
            ConceptDescriptor(name="x", dtype="float64"),
            ConceptDescriptor(name="z", dtype="float64"),
            ConceptDescriptor(name="y", dtype="int64", role=VariableRole.OUTCOME),
        ],
    )
    source = tmp_path / "synthetic.parquet"
    frame.to_parquet(source, index=False)
    (tmp_path / "research_context.json").write_text(context.model_dump_json())
    monkeypatch.setattr(owner, "_REPEATED_SPLIT_SEEDS", (1730, 1731))
    summary = owner.run_prediction_model(
        frame=frame,
        declared_columns=("x", "z", "y"),
        typed_cohort_input="artifact:analysis_cohort",
        source_cohort=source,
        out_dir=tmp_path / "out",
        run_dir=tmp_path,
        step_id="fit",
        data_preparation=plan,
    )
    receipt = summary["data_preparation"]
    assert receipt["plan"]["missing_data"]["strategy"] == "multiple_imputation"
    assert receipt["missing_data"]["fit_scope"] == "training_partition_only"
    assert receipt["sensitivity"]["strategy"] == "complete_case"
    assert summary["resampling_validation"]["n_repeats"] == 2
    performance = pd.read_csv(tmp_path / "out" / "prediction_performance.csv")
    assert "Rubin" in performance.auroc_ci_method.iloc[0]
