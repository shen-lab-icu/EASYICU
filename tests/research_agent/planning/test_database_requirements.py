"""Concrete source decisions use synthetic requirements only."""
from __future__ import annotations

import pytest

from easyicu.databases.research_capabilities import research_capability_profile
from easyicu.databases.research_requirements import (
    StudyRequirements, ThresholdRequirement, assess_study, family_requirements,
)

KEYS = ("mimic", "mimic_demo", "miiv", "miiv_demo", "eicu", "eicu_demo", "hirid", "aumc", "sic")


def _synthetic_sic_mortality_mappings_with_explicit_icu_bounds():
    """Exercise mortality selection with invented bounds, not native SICdb facts."""
    from easyicu.databases.research_capabilities import SourceField

    return research_capability_profile("sic").model_copy(update={
        "database": "synthetic_icu_source",
        "schema_database": "synthetic",
        "source_registered": False,
        "icu_admission": SourceField(source="synthetic_registry", table="stays", column="icu_start"),
        "icu_discharge": SourceField(source="synthetic_registry", table="stays", column="icu_end"),
    })


@pytest.mark.parametrize("database", ["hirid"])
@pytest.mark.parametrize("concept,value,aligned", [("age", 18, 20), ("age", 63, 65), ("age", 95, 90), ("weight", 73, 75)])
def test_published_five_unit_bands_require_an_explicit_threshold_proposal(database, concept, value, aligned):
    threshold = ThresholdRequirement(concept=concept, operator=">=", value=value)
    decision = assess_study(research_capability_profile(database), StudyRequirements(
        family="association_study", thresholds=(threshold,)))
    assert decision.status == "substitution"
    change, = decision.substitutions
    assert change.original == threshold
    assert change.reason_code == "numeric_threshold_aligned_to_band"
    assert (change.substitute.operator, change.substitute.value) == (">=", aligned)


@pytest.mark.parametrize("concept,value", [("age", 18), ("age", 20), ("age", 65), ("age", 90), ("weight", 75)])
@pytest.mark.parametrize("operator", [">", ">=", "<", "<=", "==", "!="])
def test_sic_rounded_measurement_has_no_proven_exact_threshold(concept, value, operator):
    threshold = ThresholdRequirement(concept=concept, operator=operator, value=value)
    decision = assess_study(research_capability_profile("sic"), StudyRequirements(
        family="association_study", thresholds=(threshold,)))
    assert decision.status == "refusal"
    assert decision.substitutions == ()
    assert any(item.capability == f"threshold:{concept}" and item.code == "numeric_threshold_not_resolvable"
               for item in decision.missing)


@pytest.mark.parametrize("database", ["mimic", "miiv", "sic"])
@pytest.mark.parametrize("unit", ["icu_stay", None])
def test_hospital_death_never_uses_icu_censoring_as_exact(database, unit):
    profile = research_capability_profile(database)
    requirements = StudyRequirements(family="survival", outcome="hospital_death", censoring_unit=unit)
    decision = assess_study(profile, requirements)
    if database == "sic":
        assert decision.status == "refusal"
        assert decision.requirements == requirements
        assert {item.capability for item in decision.missing} == {"time_zero", "censor_time"}
        assert decision.substitutions == ()
        native = assess_study(profile, requirements.model_copy(update={"outcome": "icu_death"}))
        assert native.status == "refusal"
        assert "time_zero" in {item.capability for item in native.missing}
        return
    assert decision.status == "substitution"
    assert decision.requirements == requirements
    changes = {item.capability: item for item in decision.substitutions}
    censor = changes["censor_time"]
    assert censor.original == "icu_discharge"
    assert censor.reason_code == "mortality_censoring_differs"
    assert set(changes) == {"censor_time"}
    assert censor.substitute == "hospital_discharge"
    assert censor.source_field == profile.hospital_discharge
    native = assess_study(profile, requirements.model_copy(update={"outcome": "icu_death"}))
    assert native.status == "exact"
    assert native.substitutions == ()


def test_censoring_mismatch_uses_source_facts_and_refuses_when_no_complete_alternative_exists():
    profile = research_capability_profile("sic").model_copy(update={
        "database": "invented_source", "mortality_outcomes": (),
        "concepts": tuple(item.model_copy(update={"mapped": False})
                          for item in research_capability_profile("sic").concepts),
    })
    decision = assess_study(profile, StudyRequirements(
        family="survival", outcome="hospital_death", censoring_unit="icu_stay"))
    assert decision.status == "refusal"
    assert {(item.capability, item.code) for item in decision.missing} == {
        ("time_zero", "time_zero_unavailable"), ("censor_time", "censor_time_unavailable")}


@pytest.mark.parametrize("database,status,endpoint,table,status_column,time_column", [
    ("mimic", "exact", None, None, None, None),
    ("miiv", "exact", None, None, None, None),
    ("eicu", "substitution", "icu_death", "patient", "unitdischargestatus", "unitdischargeoffset"),
    ("hirid", "refusal", None, None, None, None),
    ("aumc", "substitution", "icu_death", "admissions", "destination", "dischargedat"),
    ("sic", "refusal", None, None, None, None),
])
def test_default_survival_offers_only_supported_mortality_with_explicit_coordinates(
    database, status, endpoint, table, status_column, time_column,
):
    requirements = StudyRequirements(family="survival", outcome="hospital_death",
                                    concepts=("death",), censoring_unit="hospitalization")
    profile = research_capability_profile(database)
    decision = assess_study(profile, requirements)
    assert decision.status == status
    assert decision.requirements == requirements
    if status == "exact":
        assert decision.substitutions == decision.missing == ()
    elif status == "refusal":
        expected = {("censor_time", "censor_time_unavailable")}
        if database == "sic":
            expected.add(("time_zero", "time_zero_unavailable"))
            assert decision.substitutions == ()
        assert {(item.capability, item.code) for item in decision.missing} == expected
    else:
        assert not decision.missing
        changes = {item.capability: item for item in decision.substitutions}
        assert set(changes) == {"outcome_meaning", "event_time", "censor_time"}
        assert changes["outcome_meaning"].original == "hospital_death"
        assert changes["outcome_meaning"].substitute == endpoint
        assert changes["outcome_meaning"].source_field.table == table
        assert changes["outcome_meaning"].source_field.column == status_column
        for capability in ("event_time", "censor_time"):
            assert changes[capability].source_field.table == table
            assert changes[capability].source_field.column == time_column
        assert changes["event_time"].substitute == ("icu_discharge_proxy" if endpoint == "icu_death" else time_column)
        assert changes["censor_time"].substitute == ("icu_discharge" if endpoint == "icu_death" else time_column)
        assert all(item.source_field.source for item in changes.values())
        if endpoint == "icu_death":
            native = next(item for item in profile.mortality_outcomes if item.meaning == endpoint)
            assert changes["outcome_meaning"].source_field == native.status
            assert changes["event_time"].source_field == changes["censor_time"].source_field == profile.icu_discharge
        else:
            assert changes["outcome_meaning"].source_field.source == profile.concept(endpoint).source
            assert changes["event_time"].source_field.source == profile.concept(time_column).source
            assert changes["censor_time"].source_field == changes["event_time"].source_field
            assert changes["event_time"].source_field.interpretation == "easyicu.scores.outcomes.load_outcomes"


def test_default_survival_order_uses_facts_and_shortest_complete_followup():
    profile = _synthetic_sic_mortality_mappings_with_explicit_icu_bounds()
    requirements = StudyRequirements(family="survival", outcome="hospital_death", censoring_unit="hospitalization")
    assert assess_study(profile, requirements).substitutions[0].substitute == "mort_28d"
    concepts = tuple(item.model_copy(update={"mapped": False}) if item.concept == "followup_days_28d" else item
                     for item in reversed(profile.concepts))
    changed = profile.model_copy(update={"concepts": concepts})
    assert assess_study(changed, requirements).substitutions[0].substitute == "mort_90d"
    concepts = tuple(item.model_copy(update={"mapped": False}) if item.concept.startswith("mort_") else item
                     for item in concepts)
    changed = profile.model_copy(update={"concepts": concepts})
    assert assess_study(changed, requirements).substitutions[0].substitute == "icu_death"
    changed = changed.model_copy(update={"icu_discharge": None})
    assert assess_study(changed, requirements).status == "refusal"


def test_mortality_offer_does_not_hide_other_missing_requirements_or_change_a_named_endpoint():
    profile = research_capability_profile("sic")
    requirements = StudyRequirements(family="survival", outcome="hospital_death", censoring_unit="hospitalization")
    decision = assess_study(profile.model_copy(update={"icu_admission": None}), requirements)
    assert decision.status == "refusal"
    assert "time_zero" in {item.capability for item in decision.missing}
    native = next(item for item in profile.mortality_outcomes if item.meaning == "icu_death")
    profile = profile.model_copy(update={"mortality_outcomes": (native.model_copy(update={"event_time": None}),)})
    decision = assess_study(profile, requirements.model_copy(update={"outcome": "icu_death", "censoring_unit": None}))
    assert decision.status == "refusal"
    assert decision.substitutions == ()
    assert {(item.capability, item.code) for item in decision.missing} == {
        ("time_zero", "time_zero_unavailable"), ("event_time", "event_time_unavailable"),
        ("censor_time", "censor_time_unavailable"),
    }


def test_source_preparation_preserves_the_assessed_time_to_event_proposal():
    from easyicu.databases.research_capabilities import SourceField
    from easyicu.databases.research_substitutions import source_decision

    profile = research_capability_profile("aumc").model_copy(update={
        "database": "synthetic_source", "hospital_discharge": SourceField(
            source="synthetic_registry", table="synthetic_admissions", column="hospital_end"),
    })
    requirements = StudyRequirements(family="survival", outcome="hospital_death", censoring_unit="hospitalization")
    decision = assess_study(profile, requirements)
    assert decision.status == "substitution"
    assert decision.substitutions[0].substitute == "death_within_72h_after_icu_discharge"
    assert source_decision(profile, requirements) == decision


@pytest.mark.parametrize("database", KEYS)
def test_every_source_allows_an_exact_study_and_refuses_a_known_unmapped_concept(database):
    profile = research_capability_profile(database)
    exact = assess_study(profile, StudyRequirements(family="descriptive_epidemiology", concepts=("hr",)))
    assert exact.status == "exact"
    concept = "d_dimer" if database == "sic" else "apixaban"
    refused = assess_study(profile, StudyRequirements(family="descriptive_epidemiology", concepts=(concept,)))
    assert refused.status == "refusal"
    assert refused.missing[0].code == "required_concept_unmapped"
    assert refused.missing[0].capability == f"concept:{concept}"


@pytest.mark.parametrize("database", KEYS)
def test_unknown_concepts_are_unassessed_without_refusing_export_columns(database):
    decision = assess_study(research_capability_profile(database), StudyRequirements(
        family="association_study", outcome="synthetic_endpoint",
        concepts=("hr", "synthetic_assay", "synthetic_assay"),
    ))
    assert decision.status == "exact"
    assert decision.missing == decision.substitutions == ()
    assert decision.unassessed_concepts == ("synthetic_assay", "synthetic_endpoint")
    assert decision.model_dump(mode="json")["unassessed_concepts"] == ["synthetic_assay", "synthetic_endpoint"]
    assert all(not item.defined for item in decision.unassessed_concept_details)


def test_an_unknown_concept_does_not_hide_a_known_unmapped_concept():
    decision = assess_study(research_capability_profile("eicu"), StudyRequirements(
        family="association_study", concepts=("synthetic_assay", "d_dimer"),
    ))
    assert decision.status == "refusal"
    assert [(item.capability, item.code) for item in decision.missing] == [
        ("concept:d_dimer", "required_concept_unmapped"),
    ]
    assert decision.unassessed_concepts == ("synthetic_assay",)


def test_an_unknown_concept_does_not_remove_a_required_substitution():
    decision = assess_study(research_capability_profile("hirid"), StudyRequirements(
        family="association_study", outcome="hospital_death", concepts=("synthetic_assay",),
    ))
    assert decision.status == "substitution"
    assert decision.substitutions[0].reason_code == "mortality_meaning_differs"
    assert decision.unassessed_concepts == ("synthetic_assay",)


def test_partial_derivation_is_unassessed_instead_of_claimed_exact_or_absent():
    decision = assess_study(research_capability_profile("miiv"), StudyRequirements(
        family="association_study", concepts=("sofa2", "sofa_coag"),
    ))
    assert decision.status == "exact"
    assert decision.missing == decision.substitutions == ()
    assert decision.unassessed_concepts == ("sofa2",)
    (detail,) = decision.unassessed_concept_details
    assert detail.concept == "sofa2"
    assert detail.derivation_reason_code == "partial_dependency_availability"
    assert detail.source.endswith("#/sofa2")


@pytest.mark.parametrize("database", KEYS)
def test_each_source_can_propose_an_explicit_substitution(database):
    profile = research_capability_profile(database)
    if database in {"hirid", "sic"}:
        requirements = StudyRequirements(family="descriptive_epidemiology", analysis_unit="hospitalization")
    else:
        requirements = StudyRequirements(family="descriptive_epidemiology", thresholds=(
            ThresholdRequirement(concept="age", operator=">=", value=95 if database != "aumc" else 65),
        ))
    decision = assess_study(profile, requirements)
    assert decision.status == "substitution"
    assert len(decision.substitutions) == 1
    record = decision.substitutions[0]
    assert record.original != record.substitute
    assert record.reason_code
    assert record.source_field.source


def test_mortality_substitution_names_actual_meaning():
    for database, replacement in (("hirid", "icu_death"), ("aumc", "death_within_72h_after_icu_discharge")):
        decision = assess_study(research_capability_profile(database), StudyRequirements(
            family="association_study", outcome="hospital_death",
        ))
        assert decision.status == "substitution"
        assert decision.substitutions[0].original == "hospital_death"
        assert decision.substitutions[0].substitute == replacement


def test_survival_refuses_missing_event_or_matching_censor_time_without_an_alternative():
    for database, missing in (("eicu", "event_time"), ("hirid", "censor_time"), ("sic", "censor_time")):
        profile = research_capability_profile(database)
        profile = profile.model_copy(update={"mortality_outcomes": (), "concepts": tuple(
            item.model_copy(update={"mapped": False}) for item in profile.concepts)})
        decision = assess_study(profile, StudyRequirements(
            family="survival", outcome="hospital_death", censoring_unit="hospitalization",
        ))
        assert decision.status == "refusal"
        assert missing in {item.capability for item in decision.missing}
    assert assess_study(research_capability_profile("miiv"), StudyRequirements(
        family="survival", outcome="hospital_death", censoring_unit="hospitalization",
    )).status == "exact"


def test_hospital_los_is_not_replaced_by_icu_los():
    decision = assess_study(research_capability_profile("hirid"), StudyRequirements(
        family="association_study", concepts=("los_hosp",),
    ))
    assert decision.status == "refusal"
    assert decision.missing[0].capability == "hospital_los"


def test_bin_alignment_respects_operator_and_never_treats_midpoints_as_exact():
    profile = research_capability_profile("aumc")
    for operator, value, expected in ((">=", 60, "exact"), ("<", 60, "exact"),
                                      (">=", 65, "substitution"), ("<=", 60, "substitution"),
                                      ("==", 65, "refusal")):
        decision = assess_study(profile, StudyRequirements(family="descriptive_epidemiology", thresholds=(
            ThresholdRequirement(concept="age", operator=operator, value=value),
        )))
        assert decision.status == expected
        if expected == "substitution":
            assert decision.substitutions[0].substitute.value == 60


def test_family_requirements_come_from_existing_family_and_protocol_owners():
    assert "event_time" in family_requirements("survival")
    assert "censor_time" in family_requirements("survival")
    assert {"time_zero", "treatment_timing"} <= set(family_requirements("causal_inference"))
    for key in KEYS:
        profile = research_capability_profile(key)
        for req in (StudyRequirements(family="trajectory_clustering"),
                    StudyRequirements(family="causal_inference", treatment_concepts=("abx",))):
            decision = assess_study(profile, req)
            if key == "sic":
                assert decision.status == "refusal"
                assert [(item.capability, item.code) for item in decision.missing] == [("time_zero", "time_zero_unavailable")]
            else:
                assert decision.status == "exact"


def test_capability_loss_not_database_name_controls_decision():
    profile = research_capability_profile("miiv")
    changed = profile.model_copy(update={"hospital_discharge": None, "mortality_outcomes": (), "concepts": tuple(
        item.model_copy(update={"mapped": False}) for item in profile.concepts)})
    requirements = StudyRequirements(family="survival", outcome="hospital_death", censoring_unit="hospitalization")
    assert assess_study(profile, requirements).status == "exact"
    assert assess_study(changed, requirements).status == "refusal"


def test_refusal_dominates_a_proposed_substitution():
    decision = assess_study(research_capability_profile("hirid"), StudyRequirements(
        family="association_study", outcome="hospital_death", concepts=("los_hosp",),
    ))
    assert decision.status == "refusal"


def test_unknown_family_is_not_silently_offered():
    decision = assess_study(research_capability_profile("miiv"), StudyRequirements(family="synthetic_unknown_family"))
    assert decision.status == "refusal"
    assert decision.missing[0].capability == "family_requirements"


def test_icu_death_is_exact_where_its_own_status_is_recorded():
    assert assess_study(research_capability_profile("eicu"), StudyRequirements(
        family="descriptive_epidemiology", outcome="icu_death",
    )).status == "exact"


def test_missing_time_zero_treatment_and_observation_timing_are_named():
    profile = research_capability_profile("miiv")
    for family, changes, missing in (
        ("causal_inference", {"icu_admission": None}, "time_zero"),
        ("causal_inference", {"concepts": ()}, "treatment_timing"),
        ("trajectory_clustering", {"concepts": ()}, "observation_timing"),
    ):
        decision = assess_study(profile.model_copy(update=changes), StudyRequirements(family=family))
        assert decision.status == "refusal"
        assert missing in {item.capability for item in decision.missing}


@pytest.mark.parametrize("database,expected", [("miiv", "exact"), ("sic", "refusal"), ("mimic", "refusal"), ("eicu", "refusal")])
def test_fixed_horizon_followup_reads_its_existing_registry(database, expected):
    decision = assess_study(research_capability_profile(database), StudyRequirements(
        family="survival", outcome="mort_28d", concepts=("mort_28d", "followup_days_28d"),
    ))
    assert decision.status == expected
    if database == "sic":
        assert [(item.capability, item.code) for item in decision.missing] == [("time_zero", "time_zero_unavailable")]
        assert decision.substitutions == ()


def test_an_outcome_is_a_required_concept_even_when_the_caller_does_not_repeat_it():
    decision = assess_study(research_capability_profile("eicu"), StudyRequirements(
        family="association_study", outcome="mort_28d",
    ))
    assert decision.status == "refusal"
    assert decision.missing[0].capability == "concept:mort_28d"


def test_absent_units_and_time_zero_are_reported_together_without_an_attribute_error():
    profile = research_capability_profile("miiv").model_copy(update={"units": (), "icu_admission": None})
    decision = assess_study(profile, StudyRequirements(family="causal_inference"))
    assert {item.capability for item in decision.missing} == {"analysis_unit", "time_zero"}


def test_fixed_horizon_followup_supplies_its_own_event_and_censor_coordinates():
    native_sic = research_capability_profile("sic")
    decision = assess_study(native_sic, StudyRequirements(family="survival", outcome="mort_28d"))
    assert decision.status == "refusal"
    assert [(item.capability, item.code) for item in decision.missing] == [("time_zero", "time_zero_unavailable")]
    # A distinct, explicitly synthetic profile isolates event/censor behavior
    # once exact ICU bounds are provided; it does not upgrade native SICdb.
    profile = _synthetic_sic_mortality_mappings_with_explicit_icu_bounds().model_copy(update={
        "mortality": None, "mortality_outcomes": (), "icu_discharge": None, "hospital_discharge": None,
    })
    assert assess_study(profile, StudyRequirements(family="survival", outcome="mort_28d")).status == "exact"


def test_sic_time_origin_proxy_is_review_only_and_does_not_grant_survival_execution():
    from easyicu.databases.research_requirements import review_only_time_origin

    profile = research_capability_profile("sic")
    candidate = review_only_time_origin(profile)
    assert profile.icu_admission is None and profile.icu_discharge is None
    assert candidate is not None
    assert candidate.requested_origin == "icu_admission"
    assert candidate.proposed_origin == "first_intensive_or_intermediate_care_transfer"
    assert candidate.source_field.table == "cases" and candidate.source_field.column == "ICUOffset"
    assert candidate.source_identity_scope == "packaged_registry_only"
    assert candidate.review_required is True and candidate.executable is False
    assert candidate.execution_contract_available is False
    decision = assess_study(profile, StudyRequirements(family="survival", outcome="mort_28d"))
    assert decision.status == "refusal"
    assert decision.substitutions == ()
    assert [(item.capability, item.code) for item in decision.missing] == [("time_zero", "time_zero_unavailable")]
