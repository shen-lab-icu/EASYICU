"""Synthetic source coordinates preserve the hospital/patient distinction."""

import hashlib

import pandas as pd
import pytest

from easyicu.research_agent.acquisition.first_icu_stay import (
    FLAG, REASON, FirstIcuStayPerHospitalizationError,
    derive_first_icu_stay_per_hospitalization,
    load_verified_first_icu_stay_per_hospitalization,
    restrict_to_first_icu_stay_per_hospitalization,
)


def _table(hospitals, ordinals, *, stays=None):
    return pd.DataFrame({
        "patientunitstayid": stays or list(range(10, 10 + len(hospitals))),
        "patienthealthsystemstayid": hospitals, "unitvisitnumber": ordinals,
        "uniquepid": ["same-patient"] * len(hospitals),
        "unittype": ["MICU"] * len(hospitals),
    })


def test_each_hospitalization_retains_its_ordinal_one_not_the_smallest_stay_id():
    result = derive_first_icu_stay_per_hospitalization(_table([1, 1, 2], [2, 1, 1], stays=[10, 90, 20]))
    assert result.frame.set_index("stay_id")[FLAG].to_dict() == {10: False, 20: True, 90: True}
    assert set(result.frame) == {"stay_id", FLAG, REASON}
    assert result.receipt["first_icu_stays"] == 2
    assert result.receipt["establishes_patient_independence"] is False


@pytest.mark.parametrize(("hospitals", "ordinals", "reason"), [
    ([None], [1], "hospitalization_identifier_missing_or_invalid"),
    ([0], [1], "hospitalization_identifier_missing_or_invalid"),
    ([True], [1], "hospitalization_identifier_missing_or_invalid"),
    ([2**53], [1], "hospitalization_identifier_missing_or_invalid"),
    ([1, 1], [1, None], "hospitalization_unit_ordinal_missing_or_invalid"),
    ([1, 1], [1, 1.5], "hospitalization_unit_ordinal_missing_or_invalid"),
    ([1, 1], [1, 0], "hospitalization_unit_ordinal_missing_or_invalid"),
    ([1, 1], [1, True], "hospitalization_unit_ordinal_missing_or_invalid"),
    ([1, 1], [1, 1], "hospitalization_unit_ordinal_duplicate"),
    ([1, 1, 1], [1, 2, 2], "hospitalization_unit_ordinal_duplicate"),
    ([1, 1], [2, 3], "hospitalization_prior_unit_coverage_incomplete"),
])
def test_unknown_or_ambiguous_hospitalization_is_explicitly_excluded(hospitals, ordinals, reason):
    result = derive_first_icu_stay_per_hospitalization(_table(hospitals, ordinals))
    assert not result.frame[FLAG].any()
    assert result.frame[REASON].tolist() == [reason] * len(hospitals)
    assert result.receipt["unknown_eligibility_stays"] == len(hospitals)
    assert result.receipt["excluded_reason_counts"] == {reason: len(hospitals)}


@pytest.mark.parametrize("stays", [[10, 10], [10, None], [10, 1.5]])
def test_ambiguous_stay_identity_refuses_coordinate(stays):
    with pytest.raises(FirstIcuStayPerHospitalizationError):
        derive_first_icu_stay_per_hospitalization(_table([1, 1], [1, 2], stays=stays))


def test_verified_coordinate_covers_export_and_retains_exclusion_ledger(tmp_path):
    derived = derive_first_icu_stay_per_hospitalization(_table([1, 1, 2, 3], [1, 2, 2, 1]))
    path = tmp_path / "coordinate.parquet"
    derived.frame.to_parquet(path, index=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    cohort = pd.DataFrame({"stay_id": [10, 11, 12, 13], "age": [50, 51, 52, 53]})
    result, receipt = restrict_to_first_icu_stay_per_hospitalization(
        cohort, coordinate_path=path, coordinate_sha256=sha, authority_coordinates=derived.receipt,
    )
    assert result["stay_id"].tolist() == [10, 13]
    assert (receipt["stays_before"], receipt["stays_after"]) == (4, 2)
    assert receipt["non_first_icu_stays_removed"] == receipt["unknown_eligibility_stays_removed"] == 1
    assert sum(receipt["excluded_reason_counts"].values()) == 2
    with pytest.raises(FirstIcuStayPerHospitalizationError, match="cover"):
        restrict_to_first_icu_stay_per_hospitalization(
            pd.DataFrame({"stay_id": [999]}), coordinate_path=path, coordinate_sha256=sha,
            authority_coordinates={},
        )
    with pytest.raises(FirstIcuStayPerHospitalizationError, match="digest"):
        load_verified_first_icu_stay_per_hospitalization(path, expected_sha256="0" * 64)


def test_later_only_export_does_not_promote_the_first_observed_row(tmp_path):
    derived = derive_first_icu_stay_per_hospitalization(_table([1, 1], [1, 2]))
    path = tmp_path / "coordinate.parquet"
    derived.frame.to_parquet(path, index=False)
    result, receipt = restrict_to_first_icu_stay_per_hospitalization(
        pd.DataFrame({"stay_id": [11]}), coordinate_path=path,
        coordinate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), authority_coordinates={},
    )
    assert result.empty and receipt["non_first_icu_stays_removed"] == 1


@pytest.mark.parametrize(("ordinals", "types", "expected", "reason"), [
    ([1, 2], ["SDU/Step down", "MICU"], [False, True], None),
    ([2], ["MICU"], [False], "hospitalization_prior_unit_coverage_incomplete"),
    ([1, 3], ["SDU/Step down", "MICU"], [False, False], "hospitalization_prior_unit_coverage_incomplete"),
    ([1, 3], ["MICU", "MICU"], [True, False], None),
    ([1, 2], ["MICU", None], [False, False], "hospitalization_unit_type_missing_or_unknown"),
    ([1, 2], ["Test ICU", "MICU"], [False, False], "hospitalization_unit_type_missing_or_unknown"),
    ([1], ["SDU/Step down"], [False], "non_icu_unit"),
])
def test_first_icu_requires_explicit_type_and_complete_prior_units(ordinals, types, expected, reason):
    frame = _table([1] * len(ordinals), ordinals)
    frame["unittype"] = types
    result = derive_first_icu_stay_per_hospitalization(frame)
    assert result.frame[FLAG].tolist() == expected
    if reason:
        assert result.frame[REASON].tolist() == [reason] * len(ordinals)
    if types == ["SDU/Step down", "MICU"] and ordinals == [1, 2]:
        assert result.receipt["non_icu_stays"] == 1
        assert result.receipt["unknown_eligibility_stays"] == 0


def test_materialized_typed_lineage_preserves_distinct_hospital_restriction(tmp_path):
    from easyicu.research_agent.cohort.materializer import materialize_to_parquet
    from easyicu.research_agent.research_context.builder import build_research_context
    from tests.support.typed_export import typed_export

    # The shared synthetic native-export fixture exercises the database-neutral
    # materializer. eICU-only source eligibility is tested at its resolver.
    source = typed_export(tmp_path / "export")
    derived = derive_first_icu_stay_per_hospitalization(_table([1, 1], [2, 1], stays=[1, 2]))
    coordinate = tmp_path / "coordinate.parquet"
    derived.frame.to_parquet(coordinate, index=False)
    digest = hashlib.sha256(coordinate.read_bytes()).hexdigest()
    paths = materialize_to_parquet(
        tmp_path / "materialized", data_path=source, database="miiv",
        static_concepts=("age",), feature_concepts=("lact",), outcome_concepts=("death",),
        first_icu_stay_per_hospitalization_path=coordinate,
        first_icu_stay_per_hospitalization_sha256=digest,
        first_icu_stay_per_hospitalization_authority_coordinates=derived.receipt,
    )
    assert pd.read_parquet(paths["parquet"])["stay_id"].tolist() == [2]
    context = build_research_context(
        research_question="Synthetic hospital first-stay contract", cohort=paths["parquet"],
        cohort_name="synthetic", database="miiv", id_columns=("stay_id",), outcome_columns=("death",),
    )
    summary = context.cohort.provenance
    receipt = summary["first_icu_stay_per_hospitalization_restriction"]
    assert receipt["coordinate_sha256"] == digest
    assert receipt["stays_before"] == 2 and receipt["stays_after"] == 1
    assert receipt["establishes_patient_independence"] is False
    assert "first_icu_stay_restriction" not in summary


@pytest.mark.parametrize("corruption", ["duplicate", "nonpositive", "fractional", "flag_dtype", "null_flag", "unknown_reason", "disagreement"])
def test_verified_coordinate_rejects_internal_identity_or_reason_corruption(tmp_path, corruption):
    frame = derive_first_icu_stay_per_hospitalization(_table([1, 1], [1, 2])).frame
    if corruption == "duplicate":
        frame["stay_id"] = [10, 10]
    elif corruption == "nonpositive":
        frame.loc[0, "stay_id"] = 0
    elif corruption == "fractional":
        frame["stay_id"] = [10.5, 11.0]
    elif corruption == "flag_dtype":
        frame[FLAG] = [1, 0]
    elif corruption == "null_flag":
        frame[FLAG] = pd.Series([True, pd.NA], dtype="boolean")
    elif corruption == "unknown_reason":
        frame.loc[1, REASON] = "undeclared exclusion"
    elif corruption == "disagreement":
        frame.loc[1, FLAG] = True
    path = tmp_path / "coordinate.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(FirstIcuStayPerHospitalizationError):
        load_verified_first_icu_stay_per_hospitalization(
            path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
