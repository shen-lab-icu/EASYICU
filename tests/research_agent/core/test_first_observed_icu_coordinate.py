"""A source-visible coordinate cannot silently broaden the requested subset."""
import hashlib

import pandas as pd
import pytest

from easyicu.research_agent.acquisition.eicu_first_observed_candidate import FLAG_COLUMN, REASON_COLUMN
from easyicu.research_agent.acquisition.first_observed_icu_coordinate import (
    FirstObservedCoordinateError, FirstObservedIcuCoordinate, restrict_to_first_observed_icu,
)


def _binding(tmp_path, rows):
    path = tmp_path / "private.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return FirstObservedIcuCoordinate(path, digest, *(["1" * 64] * 5))


def test_filters_exactly_and_does_not_disclose_flow_or_private_columns(tmp_path):
    binding = _binding(tmp_path, {"stay_id": [1, 2, 3], FLAG_COLUMN: [True, False, False],
        REASON_COLUMN: ["kept", "multiple_observed_hospitalizations", "age_unknown_or_invalid"]})
    cohort = pd.DataFrame({"stay_id": [3, 1, 2], "value": [30, 10, 20]})
    result = restrict_to_first_observed_icu(cohort, coordinate=binding)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"stay_id": [1], "value": [10]}))


@pytest.mark.parametrize("rows", [
    {"stay_id": [1, 1], FLAG_COLUMN: [True, False], REASON_COLUMN: ["kept", "non_selected_unit"]},
    {"stay_id": [1, 2], FLAG_COLUMN: [True, False], REASON_COLUMN: ["kept", "bogus"]},
    {"stay_id": [1, 2], FLAG_COLUMN: [True, True], REASON_COLUMN: ["kept", "under_18"]},
    {"stay_id": [1, 2], FLAG_COLUMN: [True, None], REASON_COLUMN: ["kept", "under_18"]},
    {"stay_id": [1], FLAG_COLUMN: [True], REASON_COLUMN: ["kept"]},
])
def test_invalid_coordinates_or_missing_stays_fail_closed(tmp_path, rows):
    binding = _binding(tmp_path, rows)
    with pytest.raises(FirstObservedCoordinateError):
        restrict_to_first_observed_icu(pd.DataFrame({"stay_id": [1, 2]}), coordinate=binding)


def test_tampered_bytes_fail_closed(tmp_path):
    binding = _binding(tmp_path, {"stay_id": [1], FLAG_COLUMN: [True], REASON_COLUMN: ["kept"]})
    binding.path.write_bytes(binding.path.read_bytes() + b"tampered")
    with pytest.raises(FirstObservedCoordinateError):
        restrict_to_first_observed_icu(pd.DataFrame({"stay_id": [1]}), coordinate=binding)
