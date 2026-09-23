"""The host owner that proves each patient's first ICU stay."""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from easyicu.research_agent.acquisition.first_icu_stay import (
    COORDINATE_FLAG_COLUMN,
    COORDINATE_STAY_COLUMN,
    FirstIcuStayBinding,
    FirstIcuStayError,
    derive_first_icu_stay,
    load_verified_first_icu_stay,
)


def _stays(rows):
    return pd.DataFrame(rows, columns=["stay_id", "subject_id", "intime"])


def _derive(table):
    return derive_first_icu_stay(
        table,
        stay_column="stay_id",
        patient_column="subject_id",
        order_column="intime",
        identity_table_name="icustays",
    )


def test_the_earliest_admission_of_each_patient_is_the_first_stay():
    derived = _derive(
        _stays(
            [
                (12, 5, "2150-03-01 08:00:00"),
                (11, 5, "2150-01-01 08:00:00"),  # listed second, admitted first
                (13, 5, "2151-01-01 08:00:00"),
                (21, 7, None),  # a single stay needs no order
            ]
        )
    )

    flags = derived.frame.set_index(COORDINATE_STAY_COLUMN)[COORDINATE_FLAG_COLUMN]
    assert flags.to_dict() == {11: True, 12: False, 13: False, 21: True}
    # The coordinate names stays only, never a patient.
    assert list(derived.frame.columns) == [COORDINATE_STAY_COLUMN, COORDINATE_FLAG_COLUMN]
    receipt = dict(derived.receipt)
    assert receipt["stays"] == 4
    assert receipt["patients"] == receipt["first_icu_stays"] == 2
    assert receipt["non_first_icu_stays"] == 2
    assert receipt["scope"] == "source_global"
    assert receipt["tie_policy"] == "fail_closed"
    assert receipt["identifier_values_returned"] is False


@pytest.mark.parametrize(
    ("rows", "code"),
    [
        (
            [(11, 5, "2150-01-01 08:00:00"), (12, 5, "2150-01-01 08:00:00")],
            "first_icu_stay_order_tied",
        ),
        (
            [(11, 5, "2150-01-01 08:00:00"), (12, 5, "not a time")],
            "first_icu_stay_order_time_missing",
        ),
        ([(11, 5, "2150-01-01"), (12, None, "2150-02-01")], "first_icu_stay_patient_identifier_missing"),
        ([(11, 5, "2150-01-01"), (11, 6, "2150-02-01")], "first_icu_stay_stay_identity_duplicate"),
    ],
)
def test_an_unprovable_order_fails_the_whole_coordinate(rows, code):
    """No stay-number tie-break and no silent drop: either would change the population."""

    with pytest.raises(FirstIcuStayError) as exc:
        _derive(_stays(rows))

    assert exc.value.code == code


def test_a_later_stay_tie_does_not_block_the_first():
    derived = _derive(
        _stays(
            [
                (11, 5, "2150-01-01 08:00:00"),
                (12, 5, "2150-06-01 08:00:00"),
                (13, 5, "2150-06-01 08:00:00"),
            ]
        )
    )

    assert derived.frame.set_index(COORDINATE_STAY_COLUMN)[COORDINATE_FLAG_COLUMN].to_dict() == {
        11: True,
        12: False,
        13: False,
    }


def test_a_missing_column_is_named():
    with pytest.raises(FirstIcuStayError) as exc:
        _derive(pd.DataFrame({"stay_id": [1], "subject_id": [2]}))

    assert exc.value.code == "first_icu_stay_identity_columns_missing"


def _written(tmp_path, frame):
    path = tmp_path / "first.parquet"
    frame.to_parquet(path, index=False)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_coordinate_is_read_only_through_its_digest(tmp_path):
    derived = _derive(_stays([(11, 5, "2150-01-01"), (12, 5, "2150-02-01")]))
    path, digest = _written(tmp_path, derived.frame)

    loaded = load_verified_first_icu_stay(path, expected_sha256=digest)
    assert loaded.equals(derived.frame)
    with pytest.raises(FirstIcuStayError, match="digest mismatch"):
        load_verified_first_icu_stay(path, expected_sha256="0" * 64)


def test_a_coordinate_without_boolean_flags_is_refused(tmp_path):
    path, digest = _written(
        tmp_path, pd.DataFrame({COORDINATE_STAY_COLUMN: [1, 2], COORDINATE_FLAG_COLUMN: [1, 0]})
    )

    with pytest.raises(FirstIcuStayError, match="boolean"):
        load_verified_first_icu_stay(path, expected_sha256=digest)


def test_the_binding_hands_the_materializer_exact_coordinates(tmp_path):
    binding = FirstIcuStayBinding(
        coordinate_path=tmp_path / "first.parquet",
        coordinate_sha256="b" * 64,
        authority_coordinates={"coordinate_sha256": "b" * 64},
    )

    assert binding.materializer_kwargs() == {
        "first_icu_stay_path": tmp_path / "first.parquet",
        "first_icu_stay_sha256": "b" * 64,
        "first_icu_stay_authority_coordinates": {"coordinate_sha256": "b" * 64},
    }
    with pytest.raises(ValueError):
        FirstIcuStayBinding(coordinate_path="relative.parquet", coordinate_sha256="b" * 64)
    with pytest.raises(ValueError):
        FirstIcuStayBinding(coordinate_path=tmp_path / "x.parquet", coordinate_sha256="nope")
