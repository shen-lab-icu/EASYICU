import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.acquisition.eicu_first_observed_candidate import (
    FLAG_COLUMN,
    REASON_COLUMN,
    REQUIRED_COLUMNS,
    FirstObservedCandidateError,
    derive_first_observed_candidate,
)
from easyicu.webserver import eicu_first_observed_source


def table(rows):
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def test_conservative_selection_and_patient_level_exclusions():
    frame = table([
        (1, "adult", 101, 1, "SDU/Step down", "60"),
        (2, "adult", 101, 2, "MICU", "60"),
        (3, "adult", 101, 3, "SICU", "60"),
        (4, "repeat", 102, 1, "MICU", "70"),
        (5, "repeat", 103, 1, "MICU", "70"),
        (6, "unknown", 104, 1, "MICU", None),
        (7, "unknown", 104, 2, "MICU", "30"),
        (8, "minor", 105, 1, "MICU", "17"),
        (9, "censored", 106, 1, "MICU", "> 89"),
        (10, "gap", 107, 2, "MICU", "50"),
        (11, "bad_type", 108, 1, "mystery", "50"),
        (12, "missing_hospital", 109, 1, "MICU", "50"),
        (13, "missing_hospital", None, 1, "MICU", "50"),
        (14, "shared_a", 110, 1, "MICU", "50"),
        (15, "shared_b", 110, 2, "MICU", "50"),
    ])
    result = derive_first_observed_candidate(frame)
    indexed = result.frame.set_index("stay_id")
    assert indexed.index[indexed[FLAG_COLUMN]].tolist() == [2, 9]
    reasons = indexed[REASON_COLUMN].to_dict()
    assert reasons[4] == reasons[5] == "multiple_observed_hospitalizations"
    assert reasons[6] == reasons[7] == "age_unknown_or_invalid"
    assert reasons[8] == "under_18"
    assert reasons[10] == reasons[11] == "first_icu_order_or_type_unverifiable"
    assert reasons[12] == reasons[13] == "hospitalization_identifier_missing_or_invalid"
    assert reasons[14] == reasons[15] == "hospitalization_identifier_shared_across_patients"
    assert result.receipt["source_patients"] == 10
    assert result.receipt["eligible_patients"] == 2
    assert result.receipt["excluded_patients_by_reason"]["multiple_observed_hospitalizations"] == 1
    assert result.receipt["not_lifetime_first_icu"] is True
    assert result.receipt["research_execution_authorized"] is False
    assert result.receipt["receipt_contains_identifier_values"] is False
    assert result.receipt["private_coordinate_contains_stay_ids"] is True
    assert result.receipt["private_coordinate_contains_patient_ids"] is False
    assert "identifier_values_returned" not in result.receipt
    assert "uniquepid" not in result.frame
    assert '"adult"' not in json.dumps(result.receipt)


@pytest.mark.parametrize("missing", [None, "", " ", " a "])
def test_missing_patient_identity_refuses_entire_source(missing):
    frame = table([(1, "a", 1, 1, "MICU", "50"), (2, missing, 2, 1, "MICU", "50")])
    with pytest.raises(FirstObservedCandidateError, match="all source patient identifiers"):
        derive_first_observed_candidate(frame)


@pytest.mark.parametrize("age", [None, "", "Unknown", "90", "-1", "19.0", True])
def test_invalid_age_does_not_promote_a_later_unit(age):
    frame = table([(1, "a", 1, 1, "MICU", age), (2, "a", 1, 2, "MICU", "50")])
    result = derive_first_observed_candidate(frame)
    assert not result.frame[FLAG_COLUMN].any()
    assert set(result.frame[REASON_COLUMN]) == {"age_unknown_or_invalid"}


def test_duplicate_stay_identity_refuses_entire_source():
    frame = table([(1, "a", 1, 1, "MICU", "50"), (1, "b", 2, 1, "MICU", "50")])
    with pytest.raises(FirstObservedCandidateError, match="stay identities"):
        derive_first_observed_candidate(frame)


def test_bound_source_reads_only_needed_columns_and_rejects_changed_bytes(tmp_path, monkeypatch):
    path = tmp_path / "patient.parquet"
    frame = table([(1, "a", 1, 1, "MICU", "50")])
    frame["hospitaldischargestatus"] = ["Expired"]
    frame.to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    binding = SimpleNamespace(
        family="eicu", database="eicu_demo", table_paths={"patient": path},
        table_sha256={"patient": digest}, export_manifest_sha256="a" * 64,
    )
    monkeypatch.setattr(eicu_first_observed_source, "resolve_raw_source_identity_binding", lambda **_: binding)
    real_read = eicu_first_observed_source.read_digest_bound_parquet
    read_columns = []

    def record_read(*args, **kwargs):
        read_columns.extend(kwargs["columns"])
        return real_read(*args, **kwargs)

    monkeypatch.setattr(eicu_first_observed_source, "read_digest_bound_parquet", record_read)
    result = eicu_first_observed_source.derive_from_registered_eicu_export(tmp_path)
    assert set(read_columns) == set(REQUIRED_COLUMNS)
    assert "hospitaldischargestatus" not in read_columns
    assert result.receipt["source_binding"]["patient_table_sha256"] == digest
    assert result.receipt["source_binding"]["official_release_independently_verified"] is False
    frame.loc[0, "age"] = "30"
    frame.to_parquet(path, index=False)
    with pytest.raises(FirstObservedCandidateError, match="digest"):
        eicu_first_observed_source.derive_from_registered_eicu_export(tmp_path)
