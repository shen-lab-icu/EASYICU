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


def _declared(rows, *, first_order_value=0):
    """A table whose order column is an offset from each patient's first admission."""

    return derive_first_icu_stay(
        pd.DataFrame(rows, columns=["CaseID", "PatientID", "offset"]),
        stay_column="CaseID",
        patient_column="PatientID",
        order_column="offset",
        identity_table_name="cases",
        first_order_value=first_order_value,
    )


def test_a_declared_first_value_marks_the_first_stay_and_never_promotes_a_readmission():
    derived = _declared(
        [
            (32, 70, 86_400),
            (31, 70, 0),  # listed second, the patient's first admission
            (41, 71, 0),
            (51, 72, 3_600),  # a readmission whose first admission is not in the table
        ]
    )

    flags = derived.frame.set_index(COORDINATE_STAY_COLUMN)[COORDINATE_FLAG_COLUMN]
    assert flags.to_dict() == {31: True, 32: False, 41: True, 51: False}
    receipt = dict(derived.receipt)
    assert receipt["order_rule"] == "declared_first_admission_value_per_patient"
    assert receipt["first_order_value"] == 0
    assert (receipt["stays"], receipt["patients"], receipt["first_icu_stays"]) == (4, 3, 2)
    assert receipt["patients_without_first_icu_stay_in_source"] == 1
    assert receipt["tie_policy"] == receipt["missing_order_policy"] == "fail_closed"
    assert receipt["identifier_values_returned"] is False


def test_an_admission_number_declares_its_first_value_as_one():
    derived = _declared([(2, 1, 2), (1, 1, 1), (3, 2, 1)], first_order_value=1)

    assert derived.frame.set_index(COORDINATE_STAY_COLUMN)[COORDINATE_FLAG_COLUMN].to_dict() == {
        1: True,
        2: False,
        3: True,
    }
    assert derived.receipt["patients_without_first_icu_stay_in_source"] == 0


def test_the_earliest_time_rule_keeps_its_receipt_shape():
    receipt = dict(_derive(_stays([(11, 5, "2150-01-01")])).receipt)

    assert receipt["order_rule"] == "earliest_order_time_per_patient"
    assert "first_order_value" not in receipt
    assert "patients_without_first_icu_stay_in_source" not in receipt


@pytest.mark.parametrize(
    ("rows", "code"),
    [
        # Two stays of one patient both claim the first admission.
        ([(31, 70, 0), (32, 70, 0)], "first_icu_stay_order_tied"),
        # A single stay still needs its value: absolute order cannot be guessed.
        ([(31, 70, 0), (41, 71, None)], "first_icu_stay_order_time_missing"),
        ([(31, 70, 0), (41, 71, -5)], "first_icu_stay_order_value_invalid"),
        ([(31, 70, 0), (41, 71, 1.5)], "first_icu_stay_order_value_invalid"),
        ([(31, 70, 0), (31, 71, 0)], "first_icu_stay_stay_identity_duplicate"),
    ],
)
def test_a_declared_first_value_fails_the_whole_coordinate_when_unprovable(rows, code):
    with pytest.raises(FirstIcuStayError) as exc:
        _declared(rows)

    assert exc.value.code == code


def test_a_declared_first_value_must_be_an_integer():
    with pytest.raises(FirstIcuStayError) as exc:
        _declared([(31, 70, 0)], first_order_value=True)

    assert exc.value.code == "first_icu_stay_order_value_invalid"


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


def test_a_sealed_restriction_is_found_through_the_cohort_lineage(tmp_path, monkeypatch):
    """A follow-up extension or run stage inherits its universe's restriction."""

    from types import SimpleNamespace

    from easyicu.research_agent.intake import materialized_metadata as metadata

    restriction = {
        "schema_version": "easyicu.first_icu_stay_restriction/1",
        "coordinate_sha256": "f" * 64,
        "stays_after": 2,
        "authority_coordinates": {"authority_ref": "private detail stays behind"},
    }
    universe = SimpleNamespace(
        producer_parameters={"first_icu_stay_restriction": restriction},
        parent_authority_sha256=None,
    )
    child = SimpleNamespace(producer_parameters={}, parent_authority_sha256="1" * 64)
    monkeypatch.setattr(
        metadata,
        "_local_authority_reference",
        lambda root, *, authority_sha256: (None, universe),
    )

    found = metadata.sealed_first_icu_stay_restriction(
        SimpleNamespace(authority=child), cohort_path=tmp_path / "cohort.parquet"
    )
    assert found == {
        "schema_version": "easyicu.first_icu_stay_restriction/1",
        "coordinate_sha256": "f" * 64,
        "stays_after": 2,
    }

    unrestricted = SimpleNamespace(producer_parameters={}, parent_authority_sha256=None)
    assert (
        metadata.sealed_first_icu_stay_restriction(
            SimpleNamespace(authority=unrestricted), cohort_path=tmp_path / "cohort.parquet"
        )
        is None
    )

    forged = SimpleNamespace(
        producer_parameters={"first_icu_stay_restriction": {**restriction, "schema_version": "x"}},
        parent_authority_sha256=None,
    )
    with pytest.raises(metadata.MaterializedMetadataError):
        metadata.sealed_first_icu_stay_restriction(
            SimpleNamespace(authority=forged), cohort_path=tmp_path / "cohort.parquet"
        )
