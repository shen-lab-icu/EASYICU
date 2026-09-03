"""Regression tests for reusable ordered-group statistical primitives."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.contingency_tables import Table
from statsmodels.stats.proportion import proportion_confint

from easyicu.research_agent.methods import ordered_trends
from easyicu.research_agent.methods.ordered_trends import (
    cochran_armitage_trend,
    jonckheere_terpstra_trend,
    wilson_interval,
)


def test_wilson_interval_matches_statsmodels() -> None:
    result = wilson_interval(7, 23)
    expected_low, expected_high = proportion_confint(7, 23, method="wilson")

    assert result.estimate == pytest.approx(7 / 23)
    assert result.ci_low == pytest.approx(expected_low)
    assert result.ci_high == pytest.approx(expected_high)
    assert result.method == "Wilson score interval"


@pytest.mark.parametrize("event_n,n", [(-1, 10), (11, 10), (0, 0)])
def test_wilson_interval_rejects_invalid_counts(event_n: int, n: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(event_n, n)


def test_cochran_armitage_matches_statsmodels_fixed_margin_test() -> None:
    # Rows are the ordered groups; columns are non-event/event.
    table = np.asarray([[9, 1], [6, 4], [2, 8]])
    result = cochran_armitage_trend(
        event_counts=[1, 4, 8],
        totals=[10, 10, 10],
        scores=[0, 1, 2],
        group_order=["lower", "middle", "higher"],
    )
    oracle = Table(table, shift_zeros=False).test_ordinal_association(
        row_scores=np.asarray([0, 1, 2]),
        col_scores=np.asarray([0, 1]),
    )

    assert result.test_name == "Cochran-Armitage trend test"
    assert result.statistic_type == "z"
    assert result.z_statistic == pytest.approx(oracle.zscore)
    assert result.p_value == pytest.approx(oracle.pvalue)
    assert result.chi_square == pytest.approx(oracle.zscore**2)
    assert result.group_order == ("lower", "middle", "higher")
    assert result.scores == (0.0, 1.0, 2.0)
    assert result.score_scheme == "consecutive_ordinal_ranks"


def test_cochran_armitage_declares_consecutive_score_assumption() -> None:
    result = cochran_armitage_trend([2, 3, 5], [10, 10, 10])

    assert result.score_scheme == "consecutive_ordinal_ranks"
    assert result.scores == (0.0, 1.0, 2.0)


def test_cochran_armitage_never_serializes_extreme_tail_as_zero() -> None:
    result = cochran_armitage_trend(
        event_counts=[0, 1_000_000],
        totals=[1_000_000, 1_000_000],
    )

    assert result.p_value == 1e-300
    assert result.p_value_bounded is True
    assert result.p_value_reporting == "<1e-300"
    assert result.log_p_value < np.log(result.p_value)
    assert result.negative_log10_p > 300


def test_jonckheere_terpstra_matches_tie_corrected_kendall_oracle() -> None:
    groups = [1, 1, 2, 2, 2, 3, 3]
    values = [0, 0, 1, 1, 1, 2, 3]
    result = jonckheere_terpstra_trend(
        values,
        groups,
        group_order=[1, 2, 3],
    )
    oracle = stats.kendalltau(groups, values, method="asymptotic")

    assert result.test_name == "Jonckheere-Terpstra trend test"
    assert result.statistic_type == "J"
    assert result.statistic == pytest.approx(16.0)
    assert result.concordance_minus_discordance == pytest.approx(16.0)
    assert result.expected_statistic == pytest.approx(8.0)
    assert result.z_statistic > 0
    assert result.p_value == pytest.approx(oracle.pvalue)
    assert result.tie_correction is True
    assert result.continuity_correction is False
    assert result.group_sizes == (2, 3, 2)


def test_spearman_is_not_accepted_as_a_jonckheere_terpstra_equivalent() -> None:
    groups = [1, 1, 2, 2, 2, 3, 3]
    values = [0, 0, 1, 1, 1, 2, 3]
    jt = jonckheere_terpstra_trend(values, groups, group_order=[1, 2, 3])
    spearman = stats.spearmanr(groups, values)

    assert jt.p_value == pytest.approx(0.006818149312489839)
    assert spearman.pvalue == pytest.approx(1.8408529611022173e-05)
    assert jt.p_value != pytest.approx(spearman.pvalue)


def test_jonckheere_terpstra_requires_explicit_order_for_text_labels() -> None:
    with pytest.raises(ValueError, match="alphabetical order"):
        jonckheere_terpstra_trend(
            values=[1, 2, 3, 4],
            groups=["lower", "lower", "higher", "higher"],
        )


def test_jonckheere_terpstra_omits_nonfinite_values_pairwise() -> None:
    result = jonckheere_terpstra_trend(
        values=[0.0, np.nan, 1.0, 2.0, np.inf],
        groups=["a", "a", "b", "c", "c"],
        group_order=["a", "b", "c"],
    )

    assert result.input_n == 5
    assert result.n == 3
    assert result.excluded_n == 2
    assert result.group_sizes == (1, 1, 1)


def test_jonckheere_terpstra_fails_when_null_variance_is_zero() -> None:
    with pytest.raises(ValueError, match="not estimable"):
        jonckheere_terpstra_trend(
            values=[1, 1, 1, 1],
            groups=[0, 0, 1, 1],
            group_order=[0, 1],
        )


def test_ordered_trend_tool_is_case_neutral() -> None:
    source = inspect.getsource(ordered_trends).lower()
    forbidden = (
        "kdigo",
        "aki_stage",
        "los_icu",
        "e3_kdigo_gradient",
    )
    assert not any(token in source for token in forbidden)
