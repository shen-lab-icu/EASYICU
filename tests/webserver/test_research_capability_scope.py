"""Launch/workflow gates compare capabilities, without HTTP or source rows."""
import pytest

from easyicu.webserver.study_family_scope import study_family_database_gap
from easyicu.webserver.target_trial_setup import target_trial_database_gap

KEYS = ("mimic", "mimic_demo", "miiv", "miiv_demo", "eicu", "eicu_demo", "hirid", "aumc", "sic")


@pytest.mark.parametrize("database,outcome,censor", [
    ("eicu", "icu_death", "icu_discharge"),
    ("aumc", "icu_death", "icu_discharge"),
])
def test_generic_survival_scope_keeps_databases_with_a_named_substitution(database, outcome, censor):
    from easyicu.databases.research_capabilities import research_capability_profile
    from easyicu.databases.research_requirements import StudyRequirements, assess_study
    from easyicu.webserver.research_capability_scope import eligible_family_databases

    requirements = StudyRequirements(family="survival", outcome="hospital_death")
    decision = assess_study(research_capability_profile(database), requirements)
    assert decision.status == "substitution"
    assert decision.requirements == requirements
    changes = {item.capability: item.substitute for item in decision.substitutions}
    assert changes["outcome_meaning"] == outcome
    assert changes["censor_time"] == censor
    assert database in eligible_family_databases("survival")
    assert database in eligible_family_databases("survival", requirements=requirements)


def test_sic_survival_scope_refuses_unverified_icu_origin_without_offering_outcome_substitution():
    from easyicu.databases.research_capabilities import research_capability_profile
    from easyicu.databases.research_requirements import StudyRequirements, assess_study, review_only_time_origin
    from easyicu.webserver.research_capability_scope import eligible_family_databases

    requirements = StudyRequirements(family="survival", outcome="hospital_death")
    profile = research_capability_profile("sic")
    decision = assess_study(profile, requirements)
    assert decision.status == "refusal"
    assert "time_zero" in {item.capability for item in decision.missing}
    assert decision.substitutions == ()
    candidate = review_only_time_origin(profile)
    assert candidate is not None and candidate.executable is False and candidate.review_required is True
    assert "sic" not in eligible_family_databases("survival")
    assert "sic" not in eligible_family_databases("survival", requirements=requirements)


@pytest.mark.parametrize("database", KEYS)
@pytest.mark.parametrize("family", ["descriptive_epidemiology", "association_study", "prediction_model", "trajectory_clustering"])
def test_family_scope_requires_a_verified_time_origin_when_trajectory_needs_it(database, family):
    gap = study_family_database_gap({"analysis_design": {"analysis_family": family}},
                                    question="Summarize the recorded heart rate.", database=database)
    if database == "sic" and family == "trajectory_clustering":
        assert gap["code"] == "study_family_database_out_of_scope"
        assert [(item["capability"], item["code"]) for item in gap["missing_capabilities"]] == [("time_zero", "time_zero_unavailable")]
    else:
        assert gap is None


@pytest.mark.parametrize("database", KEYS)
def test_target_trial_setup_checks_time_zero_and_treatment_timing(database):
    gap = target_trial_database_gap({"data_source": {"database": database}})
    if database == "sic":
        assert gap["code"] == "target_trial_database_out_of_scope"
        assert [(item["capability"], item["code"]) for item in gap["missing_capabilities"]] == [("time_zero", "time_zero_unavailable")]
    else:
        assert gap is None


def test_a_hospital_los_request_retains_the_legacy_refusal_code_with_a_capability():
    gap = study_family_database_gap({"analysis_design": {"analysis_family": "association_study"},
                                     "execution_concepts": {"outcome": "los_hosp"}},
                                    question="Estimate the association with hospital length of stay.", database="hirid")
    assert gap["code"] == "study_family_database_out_of_scope"
    assert gap["missing_capabilities"][0]["capability"] == "hospital_los"


def test_a_missing_censor_time_without_a_supported_alternative_is_a_capability_refusal():
    gap = study_family_database_gap({"analysis_design": {"analysis_family": "survival"},
                                     "execution_concepts": {"outcome": "death"}},
                                    question="Estimate time to death.", database="hirid")
    assert gap["code"] == "study_family_database_out_of_scope"
    assert "censor_time" in {item["capability"] for item in gap["missing_capabilities"]}


def test_an_unknown_export_exposure_does_not_refuse_planning_but_a_known_unmapped_one_does():
    study = {"analysis_design": {"analysis_family": "association_study"},
             "execution_concepts": {"outcome": "death", "primary_exposure": "synthetic_assay"}}
    assert study_family_database_gap(study, question="", database="eicu") is None
    study["execution_concepts"]["primary_exposure"] = "d_dimer"
    gap = study_family_database_gap(study, question="", database="eicu")
    assert gap["code"] == "study_family_database_out_of_scope"
    assert [(item["capability"], item["code"]) for item in gap["missing_capabilities"]] == [
        ("concept:d_dimer", "required_concept_unmapped"),
    ]
    assert set(gap["supported_databases"]) == {"mimic", "miiv"}


def test_an_unregistered_source_keeps_its_existing_preplanning_path():
    study = {"data_source": {"database": "synthetic"}, "analysis_design": {"analysis_family": "association_study"}}
    assert target_trial_database_gap(study) is None
    assert study_family_database_gap(study, question="", database="synthetic") is None


def test_a_bound_public_source_without_capability_metadata_reports_the_missing_profile(monkeypatch):
    from easyicu.webserver import research_capability_scope as scope

    def unavailable(database):
        raise KeyError(database)

    monkeypatch.setattr(scope, "research_capability_profile", unavailable)
    gap = target_trial_database_gap({"data_source": {"database": "eicu"}})
    assert gap["code"] == "target_trial_database_out_of_scope"
    assert gap["missing_capabilities"][0]["capability"] == "source_profile"
    assert gap["supported_databases"] == []


def test_source_suggestions_keep_assessable_sources_when_one_profile_is_unavailable(monkeypatch):
    from easyicu.databases.profiles import public_database_keys
    from easyicu.webserver import research_capability_scope as scope

    profile = scope.research_capability_profile

    def without_eicu(database):
        if database == "eicu":
            raise KeyError(database)
        return profile(database)

    monkeypatch.setattr(scope, "research_capability_profile", without_eicu)
    gap = target_trial_database_gap({"data_source": {"database": "eicu"}})
    assert gap["missing_capabilities"][0]["code"] == "source_profile_unavailable"
    assert tuple(gap["supported_databases"]) == tuple(key for key in public_database_keys() if key not in {"eicu", "sic"})
    assert "sic" not in gap["supported_databases"]  # Its ICU time zero is not established.


def test_invalid_scientific_bindings_are_left_to_the_existing_validation_order():
    from easyicu.webserver.research_capability_scope import study_requirements
    from easyicu.webserver.study_scientific_configuration import ScientificConfigurationError

    study = {"analysis_design": {"analysis_family": "association_study"},
             "covariate_selection": "exact", "covariates": ["age"],
             "execution_concepts": {"outcome": "death", "covariates": []}}
    with pytest.raises(ScientificConfigurationError) as error:
        study_requirements(study, family="association_study")
    assert error.value.code == "research_pipeline_covariate_execution_binding_mismatch"
    assert study_family_database_gap(study, question="", database="miiv") is None


def test_original_study_thresholds_travel_to_plan_review_without_rewriting_the_study():
    import json
    from copy import deepcopy
    from easyicu.webserver.agent_pipeline_runs import _research_user_preferences

    study = {"data_source": {"database": "aumc"}, "question": "Summarize the selected adult stays.",
             "analysis_design": {"analysis_family": "descriptive_epidemiology", "analysis_unit": "icu_stay",
                                 "variance_estimator": "none_counts_only"}, "cohort": {"age_min": 65}}
    original = deepcopy(study)
    constraints = json.loads(_research_user_preferences(study)["data_constraints"])
    assert constraints["database_study_requirements"]["thresholds"] == [{"concept": "age", "operator": ">=", "value": 65.0}]
    assert study == original


def test_a_refusal_only_suggests_sources_that_can_answer_the_same_request():
    from easyicu.webserver.study_family_scope import v1_scope_description

    gap = study_family_database_gap({"analysis_design": {"analysis_family": "association_study"},
                                    "execution_concepts": {"outcome": "los_hosp"}},
                                   question="Estimate hospital length of stay.", database="hirid")
    assert set(gap["supported_databases"]) == {"mimic", "miiv", "eicu"}
    detail = v1_scope_description(reason_code=gap["code"], detail=gap)
    assert set(detail["supported_databases"]) == {"mimic", "miiv", "eicu"}
    assert detail["missing_capabilities"] == gap["missing_capabilities"]


def test_population_measurements_and_extreme_age_bounds_are_retained():
    from easyicu.webserver.research_capability_scope import study_requirements

    requirements = study_requirements({"cohort": {"age_min": 100}, "population_spec": {"criteria": [
        {"kind": "measurement", "concept": "weight", "op": ">=", "value": 75},
    ]}}, family="descriptive_epidemiology")
    assert {(item.concept, item.value) for item in requirements.thresholds} == {("age", 100), ("weight", 75)}


@pytest.mark.parametrize("preset", ["adult_all", "adult_first"])
@pytest.mark.parametrize("provided,expected", [(None, 18), (16, 18), (21, 21)])
def test_sic_adult_preset_requires_unavailable_exact_age_boundary(preset, provided, expected):
    from easyicu.databases.research_capabilities import research_capability_profile
    from easyicu.databases.research_requirements import assess_study
    from easyicu.webserver.research_capability_scope import study_requirements

    cohort = {"preset": preset}
    if provided is not None:
        cohort["age_min"] = provided
    requirements = study_requirements({"cohort": cohort}, family="association_study")
    assert [(item.concept, item.operator, item.value) for item in requirements.thresholds] == [("age", ">=", expected)]
    decision = assess_study(research_capability_profile("sic"), requirements)
    assert decision.status == "refusal"
    assert any(item.code == "numeric_threshold_not_resolvable" for item in decision.missing)


@pytest.mark.parametrize("code", ["study_capability_refused", "capability_substitution_not_applied",
                                  "capability_substitution_approval_required", "capability_substitution_review_mismatch"])
def test_execution_refusal_retains_its_typed_code_and_decision_at_the_web_boundary(code):
    from easyicu.databases.research_approval import CapabilityApprovalError
    from easyicu.databases.research_capabilities import research_capability_profile
    from easyicu.databases.research_requirements import StudyRequirements, assess_study
    from easyicu.webserver.agent_pipeline_runs import _pipeline_failure_code, _safe_pipeline_typed_failure

    decision = assess_study(research_capability_profile("hirid"), StudyRequirements(
        family="descriptive_epidemiology", outcome="hospital_death"))
    error = CapabilityApprovalError(code, decision)
    assert _pipeline_failure_code(error) == code
    diagnostic = _safe_pipeline_typed_failure(error)
    assert diagnostic["reason_code"] == code
    assert diagnostic["database_capability_decision"] == decision.model_dump(mode="json")
    error.code = "synthetic_unregistered_failure"
    assert _pipeline_failure_code(error) == "study_capability_refused"
