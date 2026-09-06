"""A table's confidence declaration survives figure normalization and export."""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.research_agent.figures.exposure_outcome_distribution import (
    normalise_distribution_outcome_rates,
    normalise_distribution_risk_difference,
)


def _table(confidence: float) -> pd.DataFrame:
    # Projection fixture only; the real executor/renderer suite separately
    # verifies confidence-interval arithmetic against the underlying counts.
    return pd.DataFrame(
        {
            "row_role": ["exposure_level", "exposure_level"],
            "exposure_level_index": [0, 1],
            "exposure_level": [0, 1],
            "outcome_denominator": [100, 100],
            "outcome_rate_pct": [10.0, 30.0],
            "ci_low_pct": [6.0, 26.0],
            "ci_high_pct": [14.0, 34.0],
            "risk_difference_pct": [20.0, 20.0],
            "risk_difference_ci_low_pct": [10.0, 10.0],
            "risk_difference_ci_high_pct": [30.0, 30.0],
            "risk_difference_reference_index": [0, 0],
            "risk_difference_comparison_index": [1, 1],
            "risk_difference_effect_measure": ["risk_difference"] * 2,
            "confidence_level": [confidence] * 2,
        }
    )


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
def test_both_figure_projections_preserve_the_declared_confidence(confidence) -> None:
    frame = _table(confidence)
    contrast = normalise_distribution_risk_difference(frame)
    rates = normalise_distribution_outcome_rates(frame)
    assert contrast.attrs["header"] == f"Risk difference, pp ({100 * confidence:g}% CI)"
    assert contrast["confidence_level"].tolist() == [confidence]
    assert rates["confidence_level"].tolist() == [confidence, confidence]
    assert (
        contrast.attrs["confidence_level"]
        == rates.attrs["confidence_level"]
        == confidence
    )
    assert contrast["estimate"].tolist() == [20.0]
    assert rates["rate"].tolist() == [0.10, 0.30]


@pytest.mark.parametrize(
    "values", [None, [0.90, 0.95], [0.90, None], [0.90, float("inf")], [0.5, 0.5]]
)
def test_unknown_or_inconsistent_coverage_cannot_be_relabelled_95_percent(
    values,
) -> None:
    frame = _table(0.90)
    if values is None:
        frame = frame.drop(columns="confidence_level")
    else:
        frame["confidence_level"] = values
    assert normalise_distribution_risk_difference(frame).empty
    assert normalise_distribution_outcome_rates(frame).empty
