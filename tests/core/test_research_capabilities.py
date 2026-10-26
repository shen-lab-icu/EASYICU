"""Registry/dictionary capability facts, with no source data access."""
from __future__ import annotations

import pytest

from easyicu.databases.research_capabilities import research_capability_profile

KEYS = ("mimic", "mimic_demo", "miiv", "miiv_demo", "eicu", "eicu_demo", "hirid", "aumc", "sic")


@pytest.mark.parametrize("database", KEYS)
def test_every_database_has_evidenced_units_and_concepts(database):
    profile = research_capability_profile(database)
    assert profile.database == database
    assert "icu_stay" in profile.analysis_units
    assert profile.icu_admission.source
    assert profile.concept("hr").mapped
    assert profile.concept("hr").defined
    assert profile.concept("weight").mapped
    assert profile.concept("sofa2").source
    assert not profile.concept("synthetic_unmapped_measurement").mapped
    assert not profile.concept("synthetic_unmapped_measurement").defined
    assert profile.concept("apixaban").defined
    assert all(unit.source for unit in profile.units)


@pytest.mark.parametrize("database,units", [
    ("miiv", {"patient", "hospitalization", "icu_stay"}),
    ("mimic", {"patient", "hospitalization", "icu_stay"}),
    ("eicu", {"hospitalization", "icu_stay"}),
    ("hirid", {"icu_stay"}),
    ("aumc", {"patient", "icu_stay"}),
    ("sic", {"patient", "icu_stay"}),
])
def test_units_follow_identifier_config_not_column_spelling(database, units):
    assert set(research_capability_profile(database).analysis_units) == units


def test_mortality_meanings_and_clocks_do_not_inherit_the_dictionary_label():
    profiles = {key: research_capability_profile(key) for key in KEYS}
    assert profiles["miiv"].mortality.meaning == "hospital_death"
    assert profiles["mimic"].mortality.event_time.column == "deathtime"
    assert profiles["eicu"].mortality.meaning == "hospital_death"
    assert profiles["eicu"].mortality.event_time is None
    assert profiles["eicu"].hospital_discharge.column == "hospitaldischargeoffset"
    assert profiles["hirid"].mortality.meaning == "icu_death"
    assert profiles["hirid"].mortality.status.column == "discharge_status"
    assert profiles["hirid"].mortality.time_kind == "last_observation_proxy"
    assert profiles["hirid"].icu_discharge is None
    assert profiles["aumc"].mortality.meaning == "death_within_72h_after_icu_discharge"
    assert profiles["aumc"].mortality.time_kind == "date"
    assert profiles["aumc"].hospital_discharge is None
    assert profiles["sic"].mortality.status.column == "HospitalDischargeType"
    assert profiles["sic"].mortality.event_time.column == "OffsetOfDeath"
    assert profiles["sic"].icu_admission.column == "ICUOffset"
    assert profiles["sic"].hospital_discharge is None


def test_hirid_timing_items_are_monitoring_measurements_not_death_status():
    from easyicu.databases.research_capabilities import HIRID_TIMING_ITEMS

    assert {item.item_id: item.concept for item in HIRID_TIMING_ITEMS} == {110: "map", 200: "hr"}
    assert all("concept-dict.json" in item.source for item in HIRID_TIMING_ITEMS)
    from easyicu.resources import load_dictionary

    dictionary = load_dictionary()
    for item in HIRID_TIMING_ITEMS:
        assert item.item_id in dictionary[item.concept].sources["hirid"][0].ids


def test_binned_and_capped_measurements_remain_distinct_from_exact_values():
    aumc = research_capability_profile("aumc")
    assert aumc.age.kind == aumc.weight.kind == "binned"
    assert aumc.age.bin_edges == (18, 40, 50, 60, 70, 80)
    assert aumc.weight.bin_edges == (60, 70, 80, 90, 100, 110)
    for key in ("miiv", "mimic", "eicu"):
        assert research_capability_profile(key).age.kind == "capped_or_shifted"
        assert research_capability_profile(key).age.exact_through == 89
    assert research_capability_profile("sic").weight.kind == "exact"


def test_los_and_admission_type_require_their_actual_source():
    for key in ("miiv", "mimic", "eicu"):
        assert research_capability_profile(key).hospital_los
    for key in ("hirid", "aumc", "sic"):
        assert not research_capability_profile(key).hospital_los
    assert not research_capability_profile("hirid").icu_los
    assert not research_capability_profile("hirid").admission_type
    assert not research_capability_profile("sic").admission_type
    assert research_capability_profile("aumc").icu_los


def test_unregistered_demo_does_not_claim_source_registration():
    assert research_capability_profile("mimic_demo").source_registered
    assert not research_capability_profile("miiv_demo").source_registered
    assert research_capability_profile("miiv_demo").schema_database == "miiv"
    with pytest.raises(KeyError):
        research_capability_profile("synthetic_unknown_database")


def test_registered_loader_outputs_are_not_mistaken_for_missing_raw_concepts():
    profile = research_capability_profile("miiv")
    assert profile.concept("charlson").mapped
    assert "concept_output_sources" in profile.concept("charlson").source
    assert not research_capability_profile("hirid").concept("charlson").mapped
    assert research_capability_profile("hirid").concept("charlson").defined
    assert research_capability_profile("eicu").concept("mort_28d").defined


def test_dictionary_recipes_use_the_existing_dependency_availability_owner():
    from easyicu.research_agent.concept_availability import explain_concept_availability

    profile = research_capability_profile("miiv")
    assert profile.concept("sofa_coag").mapped
    for name in ("sofa_coag", "sofa2"):
        item = profile.concept(name)
        expected = explain_concept_availability(concept=name, database="miiv")
        assert item.derivation_status == expected.status
        assert item.derivation_reason_code == expected.reason
        assert item.mapped == (expected.status == "full")
    assert profile.concept("sofa2").derivation_status == "degraded"


def test_profile_construction_does_not_assess_unrequested_recipes(monkeypatch):
    from easyicu.research_agent import concept_availability

    def unexpected_assessment(**kwargs):
        pytest.fail("an unrequested recipe was assessed")

    monkeypatch.setattr(concept_availability, "explain_concept_availability", unexpected_assessment)
    profile = research_capability_profile.__wrapped__("miiv")
    assert profile.concept("hr").mapped


def test_eicu_has_separate_hospital_and_icu_mortality_statuses():
    profile = research_capability_profile("eicu")
    assert {item.meaning: item.status.column for item in profile.mortality_outcomes} == {
        "hospital_death": "hospitaldischargestatus", "icu_death": "unitdischargestatus",
    }
    assert all(item.event_time is None for item in profile.mortality_outcomes)


@pytest.mark.parametrize("database,clock,anchor", [
    ("miiv", "shifted_timestamp", "intime"), ("mimic", "shifted_timestamp", "intime"),
    ("eicu", "icu_offset_minutes", "unitadmitoffset"), ("hirid", "shifted_timestamp", "admissiontime"),
    ("aumc", "global_offset_milliseconds", "admittedat"), ("sic", "admission_offset_seconds", "ICUOffset"),
])
def test_source_clocks_and_icu_origins_remain_distinct(database, clock, anchor):
    profile = research_capability_profile(database)
    assert (profile.time_representation, profile.icu_admission.column) == (clock, anchor)
