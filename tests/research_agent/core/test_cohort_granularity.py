from __future__ import annotations

import pandas as pd

from easyicu.research_agent.research_context.cohort_granularity import (
    resolve_cohort_granularity,
)


def test_stay_identifier_establishes_unit_but_not_patient_count() -> None:
    frame = pd.DataFrame({"stay_id": [1, 2, 3]})

    result = resolve_cohort_granularity(
        frame=frame,
        id_columns=["stay_id"],
    )

    assert result.analysis_unit == "icu_stay"
    assert result.stay_id_columns == ("stay_id",)
    assert result.patient_id_columns == ()
    assert result.n_patients is None
    assert result.patient_identity_available is False


def test_known_patient_identifier_is_counted_independently_of_stays() -> None:
    frame = pd.DataFrame(
        {
            "subject_id": [10, 10, 20],
            "stay_id": [1, 2, 3],
        }
    )

    result = resolve_cohort_granularity(
        frame=frame,
        id_columns=["subject_id", "stay_id"],
    )

    assert result.n_patients == 2
    assert result.patient_id_columns == ("subject_id",)
    assert result.provenance()["n_patients_source"] == "subject_id"


def test_unknown_identifier_is_not_guessed_to_be_patient_identity() -> None:
    frame = pd.DataFrame({"custom_id": [1, 1, 2]})

    result = resolve_cohort_granularity(
        frame=frame,
        id_columns=["custom_id"],
    )

    assert result.analysis_unit == "row"
    assert result.n_patients is None
