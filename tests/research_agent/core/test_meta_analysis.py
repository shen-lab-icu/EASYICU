"""Synthetic analytic checks on the aggregate inverse-variance kernel."""

import math

import pytest

from easyicu.research_agent.methods.sensitivity import compute_meta_analysis


def test_fixed_effect_uses_inverse_variance_and_cochran_q():
    result = compute_meta_analysis([0, 1, 2], [.5, 1, .5], model="fixed")
    assert result["estimate"] == pytest.approx(1)
    assert result["weights"] == pytest.approx([4 / 9, 1 / 9, 4 / 9])
    assert result["standard_error"] == pytest.approx(1 / 3)
    assert result["ci_low"] == pytest.approx(1 - 1.959963984540054 / 3)
    assert result["q"] == pytest.approx(8)
    assert result["q_df"] == 2
    assert result["q_p"] == pytest.approx(math.exp(-4))
    assert result["i_squared"] == pytest.approx(75)
    assert result["tau_squared"] == 0
    assert result["prediction_low"] is None


def test_reml_and_prediction_interval_have_hand_computed_equal_variance_solution():
    # sample variance = 1; within-database variance = .25; REML tau² = .75.
    result = compute_meta_analysis([0, 1, 2], [.5, .5, .5])
    assert result["tau_method"] == "REML"
    assert result["tau_squared"] == pytest.approx(.75, abs=1e-7)
    assert result["weights"] == pytest.approx([1 / 3] * 3)
    assert result["estimate"] == pytest.approx(1)
    assert result["standard_error"] == pytest.approx(math.sqrt(1 / 3))
    # t(.975, k-2=1) = 12.7062047364; Var(mean) = 1/3.
    width = 12.706204736432095 * math.sqrt(.75 + 1 / 3)
    assert result["prediction_low"] == pytest.approx(1 - width)
    assert result["prediction_high"] == pytest.approx(1 + width)
    assert result["prediction_df"] == 1


def test_dl_is_explicitly_labelled_sensitivity_and_uses_unequal_weights():
    result = compute_meta_analysis([0, 1, 2], [.5, 1, .5], tau_method="dl")
    assert result["tau_squared"] == pytest.approx(1.125)
    assert result["tau_method"] == "DerSimonian-Laird (sensitivity)"
    assert result["weights"] == pytest.approx([17 / 45, 11 / 45, 17 / 45])


def test_hartung_knapp_uses_residual_scale_and_student_t():
    # Fixed within-variance weights: Q/(k-1)=4, SE=sqrt(4/12).
    result = compute_meta_analysis([0, 1, 2], [.5] * 3, model="fixed", hartung_knapp=True)
    assert result["interval_method"] == "Hartung-Knapp"
    assert result["hartung_knapp_scale"] == pytest.approx(4)
    assert result["standard_error"] == pytest.approx(math.sqrt(1 / 3))
    assert result["ci_high"] == pytest.approx(1 + 4.302652729696142 * math.sqrt(1 / 3))
    assert result["ci_df"] == 2


def test_reml_boundary_and_two_database_prediction_are_explicit():
    result = compute_meta_analysis([1, 1], [.2, .3])
    assert result["tau_squared"] == 0
    assert result["q"] == result["i_squared"] == 0
    assert result["q_p"] == 1
    assert result["prediction_low"] is None
    assert result["prediction_reason"] == "at_least_three_databases_required"


def test_i_squared_truncates_negative_moment_estimate_and_reml_is_scale_equivariant():
    quiet = compute_meta_analysis([0, .01, .02], [.5] * 3)
    assert quiet["i_squared"] == quiet["tau_squared"] == 0
    original = compute_meta_analysis([0, 1, 2], [.5] * 3)
    scaled = compute_meta_analysis([10, 12, 14], [1] * 3)
    assert scaled["estimate"] == pytest.approx(10 + 2 * original["estimate"])
    assert scaled["tau_squared"] == pytest.approx(4 * original["tau_squared"])
    assert scaled["standard_error"] == pytest.approx(2 * original["standard_error"])


def test_reml_unequal_variances_solves_restricted_likelihood_score():
    result = compute_meta_analysis([-.3, .6, 1.8, 2.1], [.2, .4, .6, .3])
    weights = [1 / (se ** 2 + result["tau_squared"]) for se in [.2, .4, .6, .3]]
    score = sum((w * (y - result["estimate"])) ** 2 for w, y in zip(weights, [-.3, .6, 1.8, 2.1]))
    score -= sum(weights) - sum(w ** 2 for w in weights) / sum(weights)
    assert score == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("values,errors,options", [
    ([1], [.1], {}), ([1, 2], [.1], {}), ([1, float("nan")], [.1, .2], {}),
    ([1, 2], [0, .2], {}), ([1, 2], [-.1, .2], {}), ([1, 2], [.1, float("inf")], {}),
    ([1, 2], [.1, .2], {"model": "unknown"}), ([1, 2], [.1, .2], {"tau_method": "unknown"}),
    ([1, 2], [.1, .2], {"confidence": 1}),
])
def test_invalid_inputs_never_produce_an_estimate(values, errors, options):
    with pytest.raises(ValueError):
        compute_meta_analysis(values, errors, **options)
