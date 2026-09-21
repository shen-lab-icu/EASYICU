from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.methods.subtype_assignment import (
    SubtypeAssignmentError,
    fit_and_evaluate_early_subtype_assignment,
)


def _dataset(seed: int = 17):
    rng = np.random.default_rng(seed)
    centers = {
        "resolving": (-1.8, -0.8),
        "persistent": (0.1, 1.7),
        "worsening": (1.9, -0.4),
    }
    development_rows = []
    validation_rows = []
    development_labels = []
    validation_labels = []
    for label, center in centers.items():
        development_rows.extend(rng.normal(center, 0.35, size=(20, 2)))
        validation_rows.extend(rng.normal(center, 0.35, size=(6, 2)))
        development_labels.extend([label] * 20)
        validation_labels.extend([label] * 6)
    development = pd.DataFrame(development_rows, columns=["early_lactate", "early_sofa"])
    validation = pd.DataFrame(validation_rows, columns=development.columns)
    development.loc[2, "early_lactate"] = np.nan
    validation.loc[4, "early_sofa"] = np.nan
    return development, development_labels, validation, validation_labels


def test_early_subtype_assignment_uses_disjoint_patients_and_reports_validation_products():
    development, development_labels, validation, validation_labels = _dataset()

    result = fit_and_evaluate_early_subtype_assignment(
        development,
        development_labels,
        validation,
        validation_labels,
        development_patient_ids=[f"dev-{index}" for index in range(len(development))],
        validation_patient_ids=[f"val-{index}" for index in range(len(validation))],
        feature_window_end=6.0,
        phenotype_window_start=12.0,
        forbidden_feature_names=("trajectory_subtype", "death_28d"),
        calibration_bins=3,
        random_state=29,
    )

    metrics = result.metrics.iloc[0]
    assert metrics["n_development"] == 60
    assert metrics["n_validation"] == 18
    assert metrics["n_subtypes"] == 3
    assert metrics["balanced_accuracy"] > 0.9
    assert metrics["macro_f1"] > 0.9
    assert result.claim_ceiling == "analysis_only"
    assert len(result.predictions) == 18
    assert set(result.confusion["observed_subtype"]) == {
        "resolving",
        "persistent",
        "worsening",
    }
    assert set(result.coefficients["feature"]) == {"early_lactate", "early_sofa"}
    assert set(result.calibration.columns) == {
        "subtype",
        "bin",
        "n",
        "mean_predicted_probability",
        "observed_fraction",
    }
    assert result.to_json()["claim_ceiling"] == "analysis_only"


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"validation_patient_ids": ["same"] + [f"val-{index}" for index in range(1, 18)]}, "overlap"),
        ({"feature_window_end": 12.0}, "must end before"),
        ({"forbidden_feature_names": ("early_sofa",)}, "forbidden"),
    ],
)
def test_early_subtype_assignment_fails_closed_on_leakage_boundaries(patch, message):
    development, development_labels, validation, validation_labels = _dataset()
    kwargs = {
        "development_patient_ids": ["same"] + [f"dev-{index}" for index in range(1, 60)],
        "validation_patient_ids": [f"val-{index}" for index in range(18)],
        "feature_window_end": 6.0,
        "phenotype_window_start": 12.0,
        "forbidden_feature_names": (),
    }
    kwargs.update(patch)

    with pytest.raises(SubtypeAssignmentError, match=message):
        fit_and_evaluate_early_subtype_assignment(
            development,
            development_labels,
            validation,
            validation_labels,
            **kwargs,
        )
