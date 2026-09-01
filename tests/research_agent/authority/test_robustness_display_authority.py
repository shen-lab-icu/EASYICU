from __future__ import annotations

import pandas as pd
import pytest

from easyicu.research_agent.figures.robustness import (
    ROBUSTNESS_EFFECT_COMPARABILITY_UNRESOLVED,
    ROBUSTNESS_EFFECT_COMPARABLE,
    assess_robustness_effect_comparability,
    prepare_robustness_coverage,
)


def test_summary_envelopes_do_not_authorize_a_common_effect_axis() -> None:
    summary = pd.DataFrame(
        {
            "axis": ["primary", "functional_form", "missing"],
            "total_specs": [1, 1, 1],
            "converged_specs": [1, 1, 1],
            "non_independent_specs": [0, 0, 1],
            "range_low": [1.89, 1.24, 1.89],
            "range_high": [2.03, 1.27, 2.03],
        }
    )

    assessment = assess_robustness_effect_comparability(summary)

    assert assessment.authorized is False
    assert assessment.reason_code == ROBUSTNESS_EFFECT_COMPARABILITY_UNRESOLVED
    assert "estimand_id" in assessment.missing_columns
    assert "contrast_id" in assessment.missing_columns
    assert "effect_unit" in assessment.missing_columns


def test_non_independent_row_blocks_a_quantitative_robustness_forest() -> None:
    rows = pd.DataFrame(
        {
            "point_estimate": [1.9, 1.8],
            "ci_low": [1.7, 1.6],
            "ci_high": [2.1, 2.0],
            "effect_scale": ["OR", "OR"],
            "estimand_id": ["mortality_association", "mortality_association"],
            "contrast_id": ["5_vs_2_1", "5_vs_2_1"],
            "effect_unit": ["mmol/L", "mmol/L"],
            "converged": [True, True],
            "independent_variant": [True, False],
        }
    )

    assessment = assess_robustness_effect_comparability(rows)

    assert assessment.authorized is False
    assert assessment.reason_code == ROBUSTNESS_EFFECT_COMPARABILITY_UNRESOLVED
    assert "not an independent estimate" in assessment.message


def test_one_explicit_shared_estimand_authorizes_a_robustness_forest() -> None:
    rows = pd.DataFrame(
        {
            "point_estimate": [1.9, 1.8],
            "ci_low": [1.7, 1.6],
            "ci_high": [2.1, 2.0],
            "effect_scale": ["OR", "OR"],
            "estimand_id": ["mortality_association", "mortality_association"],
            "contrast_id": ["5_vs_2_1", "5_vs_2_1"],
            "effect_unit": ["mmol/L", "mmol/L"],
            "converged": [True, True],
            "independent_variant": [True, True],
        }
    )

    assessment = assess_robustness_effect_comparability(rows)

    assert assessment.authorized is True
    assert assessment.reason_code == ROBUSTNESS_EFFECT_COMPARABLE


def test_coverage_preserves_non_independence_as_an_audit_dimension() -> None:
    summary = pd.DataFrame(
        {
            "axis": ["primary", "missing"],
            "total_specs": [1, 1],
            "converged_specs": [1, 1],
            "non_independent_specs": [0, 1],
        }
    )

    display = prepare_robustness_coverage(summary)

    assert display["independent_specs"].tolist() == [1, 0]


def test_coverage_rejects_counts_outside_the_registered_total() -> None:
    with pytest.raises(ValueError, match="do not nest"):
        prepare_robustness_coverage(
            pd.DataFrame(
                {
                    "axis": ["primary"],
                    "total_specs": [1],
                    "converged_specs": [2],
                }
            )
        )
