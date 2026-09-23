import pandas as pd
import pytest

from easyicu.scores import aki_profiles
from easyicu.scores.aki_profiles import (
    AKIProfileError,
    AKIProfilePrerequisiteError,
    RENAL_AKI_BUNDLE_OUTPUTS,
    SOURCE_NATIVE_BUNDLE_OUTPUTS,
    apply_aki_profile,
    apply_reference_aki,
    apply_source_native_aki,
    build_renal_aki_bundle,
    compare_aki_profiles,
    default_source_native_profile,
    get_aki_profile,
    list_aki_profiles,
    load_aki_profile_registry,
    published_source_native_outputs,
    renal_bundle_column_unavailability,
)


MIMIC_IV = "MIMIC_IV_MIT_LCP_KDIGO_D20B49A7"
MIMIC_III = "MIMIC_III_MIT_LCP_KDIGO_D20B49A7"
HIRID = "HIRID_AKI_EWS_2024_BTAE212"
SICDB = "SICDB_NATIVE_KDIGO_AKI_168_44A27CC8"
AUMC = "AUMC_LEGACY_ACUTE_RENAL_FAILURE_8906394D"
EICU = "EICU_OFFICIAL_RENAL_COMPONENTS_34CECE8C"
REFERENCE = "MIT_LCP_KDIGO_REFERENCE_PORT_V1"


def test_a_demo_release_resolves_its_parent_database_profile():
    """An official ``*_demo`` release is a row subset of the same schema.

    The data-source owner already declares that rule; without it here the renal
    bundle fails closed on every demo database, which is exactly the source a
    new user starts from.
    """

    for demo, parent in (
        ("eicu_demo", EICU),
        ("miiv_demo", MIMIC_IV),
        ("mimic_demo", MIMIC_III),
    ):
        assert default_source_native_profile(demo).profile_id == parent
        assert {profile.profile_id for profile in list_aki_profiles(demo)} == {
            profile.profile_id for profile in list_aki_profiles(parent_database(demo))
        }
    with pytest.raises(AKIProfileError):
        default_source_native_profile("not_a_database_demo")


def parent_database(demo: str) -> str:
    return demo.removesuffix("_demo")


def test_registry_defaults_to_public_reference_and_has_six_native_profiles():
    registry = load_aki_profile_registry()

    assert registry["contract_status"] == "REFERENCE_AND_SOURCE_NATIVE_V2"
    assert registry["default_profile"] == REFERENCE
    assert len(registry["profiles"]) == 8
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
        "apply_reference_aki",
        "apply_source_native_aki",
        "compare_aki_profiles",
        "get_aki_profile",
        "list_aki_profiles",
    }

    assert expected <= set(STABLE_EXPORTS)
    assert all(hasattr(easyicu, name) for name in expected)


def test_reference_profile_is_not_the_historical_strict_phenotype():
    creatinine = pd.DataFrame(
        {"stay_id": [1, 1], "charttime": [0, 60], "crea": [1.0, 1.1]}
    )

    result = apply_reference_aki(
        crea_df=creatinine,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
    )

    assert result["aki_stage_reference"].tolist() == [0, 0]
    assert result["aki_reference"].tolist() == [False, False]
    assert result["aki_reference_profile"].eq(REFERENCE).all()
    assert result["aki_reference_rrt_scope"].eq(
        "all_active_rrt_cross_database_port"
    ).all()
    assert not any("strict" in column for column in result)
    assert not any("harmonized" in column for column in result)


def test_renal_bundle_separates_cross_database_rrt_from_miiv_native_crrt():
    creatinine = pd.DataFrame(
        {"stay_id": [1, 1], "charttime": [0, 60], "crea": [1.0, 1.1]}
    )
    active_rrt = pd.DataFrame(
        {"stay_id": [1], "charttime": [60], "rrt": [True]}
    )
    no_crrt_mode = pd.DataFrame(columns=["stay_id", "charttime", "rrt"])

    result = build_renal_aki_bundle(
        "miiv",
        crea_df=creatinine,
        rrt_df=active_rrt,
        crrt_df=no_crrt_mode,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
        rrt_source_complete=True,
    )

    at_rrt = result.loc[result["charttime"].eq(60)].iloc[0]
    assert at_rrt["aki_stage_reference"] == 3
    assert at_rrt["aki_stage_source_native"] == 0
    assert at_rrt["aki_reference_rrt_scope"] == (
        "all_active_rrt_cross_database_port"
    )
    assert at_rrt["aki_source_native_status"] == "evaluated"
    assert "aki_assessable" not in result
    assert "aki_ascertainment" not in result
    assert "creatinine_evidence_status" in result


def test_empty_rrt_source_requires_explicit_completed_search_receipt():
    creatinine = pd.DataFrame(
        {"stay_id": [1, 1], "charttime": [0, 60], "crea": [1.0, 1.1]}
    )
    empty_rrt = pd.DataFrame(columns=["stay_id", "charttime", "rrt"])

    unresolved = build_renal_aki_bundle(
        "miiv",
        crea_df=creatinine,
        rrt_df=empty_rrt,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
    )
    completed = build_renal_aki_bundle(
        "miiv",
        crea_df=creatinine,
        rrt_df=empty_rrt,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
        rrt_source_complete=True,
    )

    assert unresolved["rrt_evidence_status"].eq("indeterminate").all()
    assert unresolved["rrt_evidence_reason"].eq("source_absent").all()
    assert completed["rrt_evidence_status"].eq("negative").all()
    assert completed["rrt_evidence_reason"].eq(
        "source_searched_no_active_rrt"
    ).all()


@pytest.mark.parametrize("database", ["aumc", "sic"])
def test_future_case_native_profile_is_not_broadcast_into_dynamic_renal(database):
    creatinine = pd.DataFrame(
        {"stay_id": [1, 1], "charttime": [0, 1], "crea": [1.0, 4.5]}
    )

    result = build_renal_aki_bundle(
        database,
        crea_df=creatinine,
        id_col="stay_id",
        time_col="charttime",
        time_unit="hours",
    )

    assert result["aki_stage_source_native"].isna().all()
    assert result["aki_source_native_status"].eq(
        "case_level_future_endpoint_not_embedded_in_dynamic_renal"
    ).all()
    assert result["aki_source_native_uses_future"].all()


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


def test_every_source_native_profile_declares_the_columns_it_publishes():
    """The declaration is what the export shows a user; it must be complete."""

    assert set(SOURCE_NATIVE_BUNDLE_OUTPUTS) <= set(RENAL_AKI_BUNDLE_OUTPUTS)
    for profile in list_aki_profiles():
        if profile.database == "all":
            continue
        published = published_source_native_outputs(profile.profile_id)
        assert published <= set(SOURCE_NATIVE_BUNDLE_OUTPUTS)
        # Every profile publishes the stage spine and its provenance stamp, so
        # a database with no official stage still reports why.
        assert {
            "aki_stage_source_native",
            "aki_source_native_profile",
            "aki_source_native_status",
        } <= published
    with pytest.raises(KeyError):
        published_source_native_outputs("NOT_A_REGISTERED_PROFILE")


def test_a_source_without_a_component_explains_the_absent_column():
    """eICU has no official AKI stage, so its component columns cannot exist.

    Without this receipt the renal module cannot be exported for eICU at all:
    the export demands a typed primary binding for every planned concept.
    """

    receipt = renal_bundle_column_unavailability(
        "aki_stage_creat_source_native", "eicu_demo"
    )

    assert receipt is not None
    assert receipt.profile_id == EICU
    assert receipt.output_kind == "URINE_COMPONENT_ONLY_NO_OFFICIAL_AKI_STAGE"
    assert receipt.reason_code == "source_native_profile_publishes_no_such_component"
    assert receipt.supported_databases == ("miiv", "mimic")
    # MIMIC-III's pinned implementation carries no RRT component either.
    assert (
        renal_bundle_column_unavailability("aki_stage_crrt_source_native", "mimic")
        is not None
    )


def test_a_published_column_is_never_explained_away_as_structural():
    """A gap the profile should have filled must stay a loud failure."""

    assert (
        renal_bundle_column_unavailability("aki_stage_crrt_source_native", "miiv")
        is None
    )
    assert renal_bundle_column_unavailability("aki_stage_source_native", "eicu") is None
    # Reference-layer and evidence columns are not this owner's to explain.
    assert renal_bundle_column_unavailability("aki_stage_reference", "eicu") is None
    assert (
        renal_bundle_column_unavailability("creatinine_evidence_status", "eicu") is None
    )
    assert renal_bundle_column_unavailability("aki_source_native", "not_a_db") is None


MIMIC_IV_COMPONENTS = (
    "aki_source_native",
    "aki_stage_creat_source_native",
    "aki_stage_uo_source_native",
    "aki_stage_crrt_source_native",
    "aki_stage_source_native_smoothed",
)


def _mimic_iv_bundle(crrt_df):
    creatinine = pd.DataFrame(
        {"stay_id": [1, 1], "charttime": [0, 60], "crea": [1.0, 1.1]}
    )
    return build_renal_aki_bundle(
        "miiv",
        crea_df=creatinine,
        rrt_df=pd.DataFrame(columns=["stay_id", "charttime", "rrt"]),
        crrt_df=crrt_df,
        id_col="stay_id",
        time_col="charttime",
        time_unit="minutes",
        rrt_source_complete=True,
    )


def test_the_bundle_runs_exactly_one_declared_mode_of_its_profile():
    """MIMIC-IV emits the components, or -- with no CRRT-mode source -- a reason."""

    components = _mimic_iv_bundle(pd.DataFrame(columns=["stay_id", "charttime", "rrt"]))
    unavailable = _mimic_iv_bundle(None)

    assert aki_profiles.source_native_mode_emitted(MIMIC_IV, components) == "components"
    assert "aki_source_native_reason" not in components
    assert (
        aki_profiles.source_native_mode_emitted(MIMIC_IV, unavailable)
        == "crrt_source_unavailable"
    )
    assert not set(MIMIC_IV_COMPONENTS) & set(unavailable)
    # Every profile's modes together are exactly what it may publish.
    for profile in list_aki_profiles():
        if profile.database == "all":
            continue
        modes = aki_profiles.source_native_output_modes(profile.profile_id)
        assert modes
        assert frozenset().union(*modes.values()) <= published_source_native_outputs(
            profile.profile_id
        )


def test_a_column_of_the_other_mode_is_explained_by_the_mode_that_ran():
    """The official MIMIC-IV demo publishes the components, so no reason column.

    Without the mode the export demanded both branches' columns at once, and
    every current MIMIC-IV renal export failed on the reason column.
    """

    ran_components = [*MIMIC_IV_COMPONENTS, "aki_stage_source_native"]
    receipt = renal_bundle_column_unavailability(
        "aki_source_native_reason", "miiv", published_columns=ran_components
    )

    assert receipt is not None
    assert receipt.reason_code == "source_native_profile_mode_emits_no_such_column"
    assert receipt.source_native_mode == "components"
    assert receipt.profile_id == MIMIC_IV
    # The other direction: the reason proves the components were not computed.
    reason_only = ["aki_source_native_reason", "aki_stage_source_native"]
    for component in MIMIC_IV_COMPONENTS:
        other = renal_bundle_column_unavailability(
            component, "miiv", published_columns=reason_only
        )
        assert other is not None
        assert other.source_native_mode == "crrt_source_unavailable"


def test_a_column_of_the_mode_that_ran_stays_a_loud_gap():
    """Only the branch the columns prove may explain an absence."""

    without_crrt = [
        column for column in MIMIC_IV_COMPONENTS if column != "aki_stage_crrt_source_native"
    ]
    assert (
        renal_bundle_column_unavailability(
            "aki_stage_crrt_source_native", "miiv", published_columns=without_crrt
        )
        is None
    )
    # No optional column at all: which branch ran cannot be told.
    assert (
        renal_bundle_column_unavailability(
            "aki_source_native_reason",
            "miiv",
            published_columns=["aki_stage_source_native"],
        )
        is None
    )
    # Columns of both branches fit no single mode.
    assert (
        renal_bundle_column_unavailability(
            "aki_stage_crrt_source_native",
            "miiv",
            published_columns=["aki_source_native", "aki_source_native_reason"],
        )
        is None
    )
    # The universal spine is never explained by a mode.
    assert (
        renal_bundle_column_unavailability(
            "aki_source_native_status", "miiv", published_columns=MIMIC_IV_COMPONENTS
        )
        is None
    )


def test_the_bundle_refuses_to_mix_two_declared_modes(monkeypatch):
    """The export reads the branch from the columns, so one bundle, one mode."""

    split = dict(aki_profiles._PROFILE_SOURCE_NATIVE_MODES)
    split[MIMIC_IV] = {
        "creatinine_half": frozenset(MIMIC_IV_COMPONENTS[:2]),
        "rest": frozenset(
            {*MIMIC_IV_COMPONENTS[2:], "aki_source_native_reason"}
        ),
    }
    monkeypatch.setattr(aki_profiles, "_PROFILE_SOURCE_NATIVE_MODES", split)

    with pytest.raises(AKIProfileError, match="fit no single declared mode"):
        _mimic_iv_bundle(pd.DataFrame(columns=["stay_id", "charttime", "rrt"]))


def test_the_bundle_refuses_to_publish_an_undeclared_source_native_column(monkeypatch):
    """An adapter must not outgrow the declaration the export explains from."""

    shrunk = dict(aki_profiles._PROFILE_SOURCE_NATIVE_MODES)
    shrunk[MIMIC_IV] = {
        **shrunk[MIMIC_IV],
        "components": frozenset(
            shrunk[MIMIC_IV]["components"] - {"aki_stage_crrt_source_native"}
        ),
    }
    monkeypatch.setattr(aki_profiles, "_PROFILE_SOURCE_NATIVE_MODES", shrunk)
    creatinine = pd.DataFrame(
        {"stay_id": [1, 1], "charttime": [0, 60], "crea": [1.0, 1.1]}
    )

    with pytest.raises(AKIProfileError, match="undeclared source-native"):
        build_renal_aki_bundle(
            "miiv",
            crea_df=creatinine,
            rrt_df=pd.DataFrame(columns=["stay_id", "charttime", "rrt"]),
            crrt_df=pd.DataFrame(columns=["stay_id", "charttime", "rrt"]),
            id_col="stay_id",
            time_col="charttime",
            time_unit="minutes",
            rrt_source_complete=True,
        )


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
