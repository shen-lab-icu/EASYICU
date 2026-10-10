"""Lasso selection keeps a patient's stays in one fold when given groups.

A shuffled K-fold puts copies of one patient's stays on both sides of a fold,
so its cross-validated error is optimistic.  ``groups`` folds by patient
instead; without it the result, and its digest, are unchanged.  Synthetic
data only.
"""

from __future__ import annotations

import numpy as np
import pytest

from easyicu.research_agent.methods.lasso_selection import (
    GROUPED_CV_STRATEGY,
    LassoSelectionError,
    lasso_select,
    result_sha256,
)


def _repeated_patients(patients: int = 60, stays: int = 3, features: int = 30):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(patients, features))
    y = 0.5 * x[:, 0] + rng.normal(size=patients)
    # Each patient's stays repeat the patient's values exactly.
    return (
        np.repeat(x, stays, axis=0),
        np.repeat(y, stays),
        np.repeat([f"p{index}" for index in range(patients)], stays),
    )


def test_an_ungrouped_result_states_no_fold_strategy() -> None:
    x, y, _ = _repeated_patients()
    result = lasso_select(x, y, alpha=0.001)

    assert "cv_strategy" not in result.to_json()
    assert "cv_group_n" not in result.to_json()


def test_grouped_folds_do_not_learn_a_patient_from_their_other_stays() -> None:
    x, y, patients = _repeated_patients()

    shuffled = lasso_select(x, y, alpha=0.001)
    grouped = lasso_select(x, y, alpha=0.001, groups=list(patients))

    assert grouped.to_json()["cv_strategy"] == GROUPED_CV_STRATEGY
    assert grouped.to_json()["cv_group_n"] == 60
    # The same fit, so the same selection and coefficients ...
    assert grouped.selected_vars == shuffled.selected_vars
    assert grouped.coefs == shuffled.coefs
    # ... but the shuffled folds score each held-out stay against a copy of
    # itself in training; patient folds do not.
    assert grouped.cv_mean_mse[0] > 3 * shuffled.cv_mean_mse[0]
    assert result_sha256(grouped) != result_sha256(shuffled)
    assert result_sha256(lasso_select(x, y, alpha=0.001, groups=list(patients))) == (
        result_sha256(grouped)
    )


def test_cross_validated_alpha_uses_the_patient_folds() -> None:
    x, y, patients = _repeated_patients()

    grouped = lasso_select(
        x, y, method="lassocv", alphas=(0.001, 0.01, 0.1, 0.5), groups=list(patients)
    )
    shuffled = lasso_select(x, y, method="lassocv", alphas=(0.001, 0.01, 0.1, 0.5))

    # Leaked copies reward a small penalty; patient folds do not.
    assert grouped.alpha > shuffled.alpha


@pytest.mark.parametrize(
    "groups, message",
    [
        (["p1"] * 10, "entries"),
        ([None] + [f"p{index}" for index in range(179)], "every row"),
        ([" "] + [f"p{index}" for index in range(179)], "every row"),
        (["a", "b"] * 90, "as many groups"),
    ],
)
def test_unusable_groups_are_refused(groups: list[object], message: str) -> None:
    x, y, _ = _repeated_patients()
    with pytest.raises(LassoSelectionError, match=message):
        lasso_select(x, y, alpha=0.001, groups=groups)
