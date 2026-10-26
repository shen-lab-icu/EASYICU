"""Concrete source decisions use synthetic requirements only."""
from __future__ import annotations

import pytest

from easyicu.databases.research_capabilities import research_capability_profile
from easyicu.databases.research_requirements import (
    StudyRequirements, ThresholdRequirement, assess_study, family_requirements,
)

KEYS = ("mimic", "mimic_demo", "miiv", "miiv_demo", "eicu", "eicu_demo", "hirid", "aumc", "sic")


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


def test_survival_refuses_missing_event_or_matching_censor_time():
    for database, missing in (("eicu", "event_time"), ("hirid", "censor_time"), ("sic", "censor_time")):
        decision = assess_study(research_capability_profile(database), StudyRequirements(
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
        assert assess_study(profile, StudyRequirements(family="trajectory_clustering")).status == "exact"
        assert assess_study(profile, StudyRequirements(family="causal_inference", treatment_concepts=("abx",))).status == "exact"


def test_capability_loss_not_database_name_controls_decision():
    profile = research_capability_profile("miiv")
    changed = profile.model_copy(update={"hospital_discharge": None})
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


@pytest.mark.parametrize("database,expected", [("miiv", "exact"), ("sic", "exact"), ("mimic", "refusal"), ("eicu", "refusal")])
def test_fixed_horizon_followup_reads_its_existing_registry(database, expected):
    decision = assess_study(research_capability_profile(database), StudyRequirements(
        family="survival", outcome="mort_28d", concepts=("mort_28d", "followup_days_28d"),
    ))
    assert decision.status == expected


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
    profile = research_capability_profile("sic").model_copy(update={
        "mortality": None, "mortality_outcomes": (), "icu_discharge": None, "hospital_discharge": None,
    })
    assert assess_study(profile, StudyRequirements(family="survival", outcome="mort_28d")).status == "exact"
