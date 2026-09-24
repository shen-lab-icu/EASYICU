"""Shared family-spec planner fixtures: synthetic, case-neutral contexts.

Test modules import these instead of each other (tests/governance E-P2-12).
"""

from __future__ import annotations

import json

from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
)
from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    TimeWindow,
    UserPreferences,
    VariableRole,
)


ALLOWED_CITATIONS = (
    "strobe_2007",
    "record_2015",
    "suissa_immortal_time_2008",
    "anderson_landmark_1983",
    "durrleman_splines_1989",
    "sterne_missing_data_2009",
    "comparator_alpha_2020_1",
)

DIRECT_COMPARATORS = ("comparator_alpha_2020_1",)

def _run(context: ResearchContext, responses: list[str], **kwargs):
    llm = ScriptedMockLLMClient(responses)
    agent = ProgressivePlannerAgent(llm)
    kwargs.setdefault("required_primary_cohort_selection_mode", "predicate_filtered")
    result = agent.run_attempt(
        context,
        planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=ALLOWED_CITATIONS,
        direct_comparator_literature_keys=DIRECT_COMPARATORS,
        comparison_literature_keys=DIRECT_COMPARATORS,
        enforce_article_contract=True,
        article_contract_context=context,
        planning_contract_context="",
        **kwargs,
    )
    return llm, result

def _descriptive_context() -> ResearchContext:
    data_constraints = json.dumps(
        {"materialization_window": {"anchor": "icu_admission", "hours": 24.0, "role": "outer_observation_window"}}
    )
    return ResearchContext(
        research_question=(
            "Among adult ICU stays, what proportion meet the phenotype in the first 24 h, "
            "and what is the in-hospital mortality of stays with and without it?"
        ),
        cohort=CohortDescriptor(
            cohort_name="synthetic_descriptive",
            database="synthetic",
            n_stays=0,
            id_columns=["stay_id"],
            outcome_columns=["death"],
            requested_outcome_columns=["death"],
            provenance={
                "database": "synthetic",
                "analysis_unit": "icu_stay",
                "stay_id_columns": ["stay_id"],
                "patient_id_columns": [],
                "patient_identity_available": False,
                "evidence_stage": "metadata_only_planning",
                "patient_rows_read": False,
            },
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            ConceptDescriptor(name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64", unit="years", source_concept="age"),
            ConceptDescriptor(
                name="sex", role=VariableRole.DEMOGRAPHIC, dtype="float64", source_concept="sex",
                observed_domain={"n_unique": 2, "is_binary": True, "levels": [0, 1]},
            ),
            ConceptDescriptor(
                name="phenotype_flag", description="phenotype status", role=VariableRole.OTHER, dtype="int64",
                source_concept="phenotype", analysis_window="icu_admission[0,24]h",
                observed_domain={"n_unique": 2, "is_binary": True, "levels": [0, 1]},
            ),
            ConceptDescriptor(
                name="score_first", description="chronic disease score", role=VariableRole.OTHER,
                dtype="float64", source_concept="score", analysis_window="icu_admission[0,24]h",
            ),
            ConceptDescriptor(name="readmit_flag", role=VariableRole.OTHER, dtype="float64"),
            ConceptDescriptor(name="phenotype_n", role=VariableRole.META, dtype="float64", source_concept="phenotype"),
            ConceptDescriptor(
                name="death", description="in hospital mortality", role=VariableRole.OUTCOME, dtype="bool",
                source_concept="death", observed_domain={"n_unique": 2, "is_binary": True, "levels": [False, True]},
            ),
        ],
        time_windows=[
            TimeWindow(name="first_24h", anchor="icu_admission", start_hours=0.0, end_hours=24.0,
                       rationale="Outer feature-materialization window bound by the host."),
        ],
        target_outcome="death",
        endpoint=EndpointSpec(name="death", kind="binary", absence_semantics="no_absent_rows", levels=[False, True]),
        primary_exposure="phenotype_flag",
        user_preferences=UserPreferences(
            data_constraints=data_constraints,
            covariate_selection="planner_selectable",
            inferred_analysis_family="descriptive_epidemiology",
        ),
    )

PHENOTYPING_LABELS = {
    "phenotype_flag": "Phenotype present in the first 24 h",
    "death": "In-hospital death",
    "age": "Age at ICU admission (years)",
    "sex": "Patient sex",
    "hr_max": "Peak heart rate in the first 24 h",
    "lactate_max": "Peak lactate in the first 24 h (mmol/L)",
    "map_min": "Lowest mean arterial pressure in the first 24 h",
    "score_first": "Chronic disease score (first value)",
}

def _phenotyping_context() -> ResearchContext:
    base = _descriptive_context()
    features = [
        ConceptDescriptor(name="hr_max", description="heart rate", role=VariableRole.VITAL, dtype="float64",
                          source_concept="hr", analysis_window="icu_admission[0,24]h"),
        ConceptDescriptor(name="lactate_max", description="lactate", role=VariableRole.LAB, dtype="float32",
                          source_concept="lact", analysis_window="icu_admission[0,24]h"),
        ConceptDescriptor(name="map_min", description="mean arterial pressure", role=VariableRole.VITAL,
                          dtype="float64", source_concept="map", analysis_window="icu_admission[0,24]h"),
        ConceptDescriptor(name="los_flag", description="long stay", role=VariableRole.OUTCOME, dtype="float64",
                          source_concept="los_icu"),
    ]
    return base.model_copy(
        update={
            "research_question": (
                "Among ICU stays with the phenotype in the first 24 h, which candidate subphenotypes "
                "emerge from first-24-hour vitals and labs by unsupervised clustering, and how do "
                "their clinical characteristics and in-hospital mortality differ?"
            ),
            "variables": [*base.variables, *features],
            "user_preferences": base.user_preferences.model_copy(
                update={"inferred_analysis_family": "trajectory_clustering"}
            ),
        }
    )

def _phenotyping_payload(request, *, features, baseline, membership):
    return {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": [],
        "baseline_variables": baseline,
        "feature_variables": features,
        "cohort_membership_column": membership,
        "reader_display_labels": [
            {"key": key, "value": PHENOTYPING_LABELS.get(key, key.replace("_", " ") + " (label)")}
            for key in dict.fromkeys([*request.required_reader_label_keys, *features, *baseline])
        ],
        "comparator_applications": [
            {
                "citation_key": key,
                "application": (
                    f"Compare these candidate phenotypes with {key} on population, features, "
                    "time zero, and estimand without copying its design or claiming novelty."
                ),
            }
            for key in request.direct_comparator_literature_keys
        ],
        "roster_decision_note": "Phenotyping family: window-bound vitals and labs as fit features.",
    }

def _prediction_context() -> ResearchContext:
    base = _phenotyping_context()
    return base.model_copy(
        update={
            "research_question": (
                "Among adult ICU stays, how well do first-24-hour vitals, labs, and demographics "
                "predict in-hospital mortality?"
            ),
            "primary_exposure": None,
            "user_preferences": base.user_preferences.model_copy(
                update={"inferred_analysis_family": "prediction_model"}
            ),
        }
    )

def _prediction_payload(request, *, features):
    return {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": [],
        "feature_variables": features,
        "reader_display_labels": [
            {"key": key, "value": PHENOTYPING_LABELS.get(key, key.replace("_", " ") + " (label)")}
            for key in dict.fromkeys([*request.required_reader_label_keys, *features])
        ],
        "comparator_applications": [
            {
                "citation_key": key,
                "application": (
                    f"Compare this model with {key} on population, predictors, time zero, "
                    "and outcome without copying its design or claiming novelty."
                ),
            }
            for key in request.direct_comparator_literature_keys
        ],
        "roster_decision_note": "Prediction family: window-bound vitals, labs, and demographics as predictors.",
    }
