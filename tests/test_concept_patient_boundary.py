import pandas as pd

from easyicu.api.concepts import _restrict_result_to_requested_ids


def test_requested_stay_partition_drops_sibling_stays_after_outer_merge():
    frame = pd.DataFrame(
        {
            "icustay_id": [10, 11, 12, None],
            "death": [False, True, False, True],
        }
    )

    result = _restrict_result_to_requested_ids(
        frame,
        {"icustay_id": [10, 12]},
    )

    assert result["icustay_id"].tolist() == [10.0, 12.0]


def test_requested_partition_filters_each_nonmerged_concept_frame():
    result = _restrict_result_to_requested_ids(
        {
            "death": pd.DataFrame({"stay_id": [1, 2], "death": [False, True]}),
            "subject_only": pd.DataFrame({"subject_id": [7, 8], "x": [1, 2]}),
        },
        {"stay_id": [2]},
    )

    assert result["death"]["stay_id"].tolist() == [2]
    assert result["subject_only"]["subject_id"].tolist() == [7, 8]


def test_non_dict_patient_request_keeps_existing_result_contract():
    frame = pd.DataFrame({"stay_id": [1, 2], "value": [3, 4]})

    result = _restrict_result_to_requested_ids(frame, [1])

    assert result.equals(frame)
