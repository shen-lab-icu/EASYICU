import pandas as pd
import pytest

from easyicu.scores.aki_profiles import (
    AKIProfileError,
    AKIProfilePrerequisiteError,
    apply_aki_profile,
    apply_source_native_aki,
    compare_aki_profiles,
    default_source_native_profile,
    get_aki_profile,
    list_aki_profiles,
    load_aki_profile_registry,
)


MIMIC_IV = "MIMIC_IV_MIT_LCP_KDIGO_D20B49A7"
MIMIC_III = "MIMIC_III_MIT_LCP_KDIGO_D20B49A7"
HIRID = "HIRID_AKI_EWS_2024_BTAE212"
SICDB = "SICDB_NATIVE_KDIGO_AKI_168_44A27CC8"
AUMC = "AUMC_LEGACY_ACUTE_RENAL_FAILURE_8906394D"
EICU = "EICU_OFFICIAL_RENAL_COMPONENTS_34CECE8C"


def test_registry_has_harmonized_and_all_six_database_profiles():
    registry = load_aki_profile_registry()

    assert registry["contract_status"] == "FROZEN_BEFORE_SOURCE_NATIVE_COMPARISON"
    assert len(registry["profiles"]) == 7
    assert {profile.database for profile in list_aki_profiles()} == {
        "all",
        "aumc",
        "eicu",
        "hirid",
        "mimic",
        "miiv",
        "sic",
    }


@pytest.mark.parametrize(
    ("profile_id", "commit", "grade"),
    [
        (MIMIC_IV, "d20b49a71ebb8cafc6febb0821432778592192d5", "A"),
        (MIMIC_III, "d20b49a71ebb8cafc6febb0821432778592192d5", "B"),
        (HIRID, "4bcba852f4007f72aba17b8f3576ec43d268c2ff", "B"),
        (SICDB, "44a27cc8dc7923917a7b07c01717a4e63e464ee4", "B"),
        (AUMC, "8906394d5642ea8b359e32f4a4f3c3012ca4a99a", "C"),
        (EICU, "34cece8c70771a3fab48da84d4c47f0e133ca021", "C"),
    ],
)
def test_source_native_profiles_pin_upstream_and_reliability(profile_id, commit, grade):
    profile = get_aki_profile(profile_id)

    assert profile.payload["upstream_commit"] == commit
    assert len(profile.payload["source_bundle_sha256"]) == 64
    assert profile.reliability_grade == grade
    assert profile.payload["sources"]
    assert profile.payload["limitations"]


def test_database_aliases_resolve_one_source_native_profile():
    assert default_source_native_profile("MIMIC-IV").profile_id == MIMIC_IV
    assert default_source_native_profile("mimic_iii").profile_id == MIMIC_III
    assert default_source_native_profile("sicdb").profile_id == SICDB


def test_database_convenience_entry_point_selects_but_never_falls_back():
    creatinine = pd.DataFrame(
        {"stay_id": [1, 1], "charttime": [0, 60], "crea": [1.0, 1.4]}
    )

    result = apply_source_native_aki(
        "MIMIC-IV",
        crea_df=creatinine,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
    )

    assert result["aki_source_native_profile"].eq(MIMIC_IV).all()
    assert "aki_stage_canonical" not in result

    with pytest.raises(AKIProfileError):
        apply_source_native_aki("unsupported_database", crea_df=creatinine)


def test_profile_entry_points_are_stable_top_level_exports():
    import easyicu
    from easyicu._public_api import STABLE_EXPORTS

    expected = {
        "apply_aki_profile",
        "apply_source_native_aki",
        "compare_aki_profiles",
        "get_aki_profile",
        "list_aki_profiles",
    }

    assert expected <= set(STABLE_EXPORTS)
    assert all(hasattr(easyicu, name) for name in expected)


def test_canonical_profile_adds_namespaced_columns_without_losing_original():
    creatinine = pd.DataFrame(
        {"stay_id": [1, 1], "charttime": [0, 60], "crea": [1.0, 1.4]}
    )

    result = apply_aki_profile(
        "EASYICU_KDIGO_STRICT_PRIOR_V1",
        crea_df=creatinine,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
    )

    assert "aki_stage" in result
    assert "aki_stage_canonical" in result
    assert "aki_stage_source_native" not in result
    assert result["aki_stage"].equals(result["aki_stage_canonical"])
    assert result["aki_canonical_profile"].nunique() == 1


def test_mimic_iv_profile_retains_coalesced_missing_component_semantics():
    creatinine = pd.DataFrame(
        {"stay_id": [1, 1], "charttime": [0, 60], "crea": [1.0, 1.1]}
    )

    result = apply_aki_profile(
        MIMIC_IV,
        crea_df=creatinine,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
    )

    assert result["aki_stage_source_native"].tolist() == [0, 0]
    assert result["aki_source_native_ascertainment"].tolist() == [
        "component_negative_or_unobserved_coalesced",
        "component_negative_or_unobserved_coalesced",
    ]
    assert "aki_stage" not in result


def test_mimic_iv_profile_has_six_hour_smoothed_stage():
    creatinine = pd.DataFrame(
        {
            "stay_id": [1, 1, 1],
            "charttime": [0, 60, 120],
            "crea": [1.0, 1.4, 1.1],
        }
    )

    result = apply_aki_profile(
        MIMIC_IV,
        crea_df=creatinine,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
    )

    assert result["aki_stage_source_native"].tolist() == [0, 1, 0]
    assert result["aki_stage_source_native_smoothed"].tolist() == [0, 1, 1]


def test_mimic_iii_legacy_urine_coverage_is_distinct_from_mimic_iv():
    urine = pd.DataFrame(
        {
            "stay_id": [1, 1, 1],
            "charttime": [0, 60, 120],
            "urine": [10.0, 10.0, 10.0],
        }
    )
    weight = pd.DataFrame({"stay_id": [1], "weight": [100.0]})

    legacy = apply_aki_profile(
        MIMIC_III,
        urine_df=urine,
        weight_df=weight,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
    )
    current = apply_aki_profile(
        MIMIC_IV,
        urine_df=urine,
        weight_df=weight,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
    )

    assert legacy["aki_stage_source_native"].iloc[-1] == 1
    assert current["aki_stage_source_native"].iloc[-1] == 0


def test_sicdb_profile_reproduces_fixed_70kg_and_whole_window_summary():
    urine = pd.DataFrame(
        {
            "CaseID": [1] * 8,
            "Offset": [hour * 3600 for hour in range(8)],
            "urine": [0.0] * 8,
        }
    )

    result = apply_aki_profile(
        SICDB,
        urine_df=urine,
        id_col="CaseID",
        time_col="Offset",
        time_unit="seconds",
    )

    assert result["aki_stage_source_native"].item() == 1
    assert result["aki_stage_uo_assessable_source_native"].item()
    assert result["aki_source_native"].item()


def test_sicdb_missing_component_can_remain_stage_zero_by_native_contract():
    creatinine = pd.DataFrame({"CaseID": [1], "Offset": [0], "crea": [1.0]})

    result = apply_aki_profile(
        SICDB,
        crea_df=creatinine,
        id_col="CaseID",
        time_col="Offset",
        time_unit="seconds",
    )

    assert result["aki_stage_source_native"].item() == 0
    assert result["aki_source_native_ascertainment"].item() == (
        "stage_zero_including_unobserved_components"
    )


def test_aumc_profile_is_stage3_like_binary_not_complete_kdigo():
    creatinine = pd.DataFrame(
        {
            "admissionid": [1, 1],
            "time": [-60, 60],
            "crea": [1.0, 4.1],
        }
    )

    result = apply_aki_profile(
        AUMC,
        crea_df=creatinine,
        id_col="admissionid",
        time_col="time",
        time_unit="minutes",
    )

    assert result["acute_renal_failure_source_native"].item()
    assert result["aki_stage_source_native"].item() == 3
    assert get_aki_profile(AUMC).output_kind == "CASE_LEVEL_STAGE3_LIKE_BINARY"


def test_eicu_profile_emits_official_component_but_no_official_stage():
    urine = pd.DataFrame(
        {
            "patientunitstayid": [1, 1],
            "observationoffset": [60, 120],
            "cellvaluenumeric": [25.0, 99.0],
            "cellpath": ["I&O|Output (ml)|Urine", "I&O|Intake (ml)|Oral"],
        }
    )

    result = apply_aki_profile(
        EICU,
        urine_df=urine,
        id_col="patientunitstayid",
        time_col="observationoffset",
        urine_col="cellvaluenumeric",
    )

    assert result["urine_output_source_native"].tolist() == [25.0]
    assert result["aki_stage_source_native"].isna().all()
    assert result["aki_source_native_status"].eq("components_only").all()


def test_hirid_profile_fails_closed_without_publication_only_endpoint():
    urine = pd.DataFrame(
        {
            "patientid": [1],
            "datetime": [pd.Timestamp("2020-01-01")],
            "urine": [20.0],
        }
    )

    result = apply_aki_profile(
        HIRID,
        urine_df=urine,
        id_col="patientid",
        time_col="datetime",
    )

    assert result["aki_stage_source_native"].isna().all()
    assert (
        result["aki_source_native_status"]
        .eq("not_evaluable_required_source_missing")
        .all()
    )
    with pytest.raises(AKIProfilePrerequisiteError):
        apply_aki_profile(
            HIRID,
            urine_df=urine,
            id_col="patientid",
            time_col="datetime",
            strict_prerequisites=True,
        )


def test_hirid_profile_normalizes_author_endpoint_without_unknown_to_zero():
    endpoint = pd.DataFrame(
        {
            "PatientID": [1, 1, 1],
            "AbsDatetime": pd.date_range("2020-01-01", periods=3, freq="5min"),
            "endpoint_status": ["unknown", "0", "2"],
        }
    )

    result = apply_aki_profile(
        HIRID,
        native_endpoint_df=endpoint,
        id_col="PatientID",
        time_col="AbsDatetime",
    )

    assert result["aki_stage_source_native"].tolist() == [pd.NA, 0, 2]
    assert result["aki_source_native_ascertainment"].tolist() == [
        "indeterminate",
        "observed",
        "observed",
    ]


def test_profile_comparison_preserves_unknown_and_reports_only_comparable_rows():
    canonical = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "charttime": [0, 0],
            "aki_stage_canonical": pd.Series([1, pd.NA], dtype="Int64"),
        }
    )
    native = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "charttime": [0, 0],
            "aki_stage_source_native": pd.Series([0, 0], dtype="Int64"),
            "aki_source_native_profile": [SICDB, SICDB],
        }
    )

    result = compare_aki_profiles(
        canonical, native, id_col="stay_id", time_col="charttime"
    )

    assert result["stage_comparable"].tolist() == [True, False]
    assert result["stage_agreement"].tolist() == [False, pd.NA]
    assert result["stage_difference_native_minus_canonical"].tolist() == [-1, pd.NA]


def test_source_native_profile_never_overwrites_canonical_column_name():
    creatinine = pd.DataFrame(
        {"stay_id": [1, 1], "charttime": [0, 60], "crea": [1.0, 1.4]}
    )

    result = apply_aki_profile(
        MIMIC_IV,
        crea_df=creatinine,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
    )

    assert "aki_stage_source_native" in result
    assert "aki_stage_canonical" not in result
    assert "aki_stage" not in result
