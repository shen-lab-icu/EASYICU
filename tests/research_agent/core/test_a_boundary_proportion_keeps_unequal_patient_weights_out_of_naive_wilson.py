"""Distinct patients are not the effective trials of an unequal-weight stay mean.

Keep the current boundary refusal until a source-bound method handles those
weights. This protects against the archived patient-count Wilson candidate.
All rows are synthetic.
"""

import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
    _proportion_interval,
)
from easyicu.research_agent.schema import ExposureOutcomeDistributionSpec


@pytest.mark.parametrize("outcome", [0, 1])
def test_unequal_patient_weights_do_not_get_a_naive_patient_count_interval(outcome):
    frame = pd.DataFrame({"patient_id": ["a"] * 99 + ["b"], "death": [outcome] * 100})
    spec = ExposureOutcomeDistributionSpec.model_validate(
        {
            "exposure": "glucose_group",
            "exposure_levels": [1, 2],
            "outcome": "death",
            "outcome_levels": [0, 1],
            "outcome_positive_value": 1,
            "level_match_policy": "exact_typed",
            "denominator_policy": "all_declared_rows",
            "missing_outcome_policy": "structural_absence_is_non_event",
            "confidence_level": 0.95,
            "dependence": {
                "group_source": "patient_id",
                "group_derivation": "identity",
            },
        }
    )
    with pytest.raises(RuntimeError, match="degenerate"):
        _proportion_interval(
            frame,
            spec=spec,
            denominator_mask=pd.Series(True, index=frame.index),
            numerator_mask=frame["death"].eq(1),
        )
