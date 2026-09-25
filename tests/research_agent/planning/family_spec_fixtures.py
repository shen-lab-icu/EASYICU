"""Shared family-spec planner fixtures: synthetic, case-neutral contexts.

Test modules import these instead of each other (tests/governance E-P2-12).
"""

from __future__ import annotations

import json

from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
    candidate_analysis_types,
    select_progressive_variables,
)
from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.planning.family_spec import build_family_spec_request
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ObservationSemantics,
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


# The landmark association family's synthetic context and spec builders.

LABELS = {
    "injury_stage": "Injury stage in the first 24 h",
    "death": "In-hospital death",
    "death_time_hours": "Time of in-hospital death (hours)",
    "followup_time_hours": "In-hospital follow-up time (hours)",
    "los_icu": "ICU length of stay (days)",
    "injury_stage_alt_a": "Injury stage, alternate definition A",
    "injury_stage_alt_b": "Injury stage, alternate definition B",
    "first_stay_flag": "First ICU stay indicator",
    "stage_source_flag": "Stage ascertainment source",
    "window_row_count": "Observation rows in the window",
    "age": "Age at ICU admission (years)",
    "sex": "Patient sex",
    "comorbidity_index": "Comorbidity burden index",
    "severity_score_24h": "Illness severity score in the first 24 h",
}


def _ordinal(name: str, **extra: object) -> ConceptDescriptor:
    return ConceptDescriptor(
        name=name,
        role=VariableRole.ORDINAL_SCORE,
        dtype="float64",
        valid_range=[0.0, 3.0],
        is_ordinal=True,
        ordinal_levels=[0, 1, 2, 3],
        **extra,
    )


def _specs(*, exact: bool) -> list[PrespecifiedSensitivitySpec]:
    specs = [
        PrespecifiedSensitivitySpec(
            spec_id="landmark_24h_primary",
            axis="timing",
            strategy="landmark",
            execution_variables=("death_time_hours", "followup_time_hours"),
            landmark_hours=24.0,
            require_alive_at_landmark=True,
            exclude_negative_event_times=True,
            event_time_variable="death_time_hours",
            observation_duration_variable="followup_time_hours",
            observation_duration_unit="hours",
        ),
        PrespecifiedSensitivitySpec(
            spec_id="stage_alt_a",
            axis="exposure_definition",
            strategy="alternate_exposure",
            execution_variables=("injury_stage_alt_a",),
        ),
        PrespecifiedSensitivitySpec(
            spec_id="stage_alt_b",
            axis="exposure_definition",
            strategy="alternate_exposure",
            execution_variables=("injury_stage_alt_b",),
        ),
        PrespecifiedSensitivitySpec(
            spec_id="first_stay_only",
            axis="repeated_stays",
            strategy="first_stay",
            execution_variables=("first_stay_flag",),
        ),
    ]
    if exact:
        specs.extend(
            [
                PrespecifiedSensitivitySpec(
                    spec_id="complete_case_primary_covariates",
                    axis="missing_data",
                    strategy="complete_case",
                    execution_variables=("injury_stage", "age", "sex", "comorbidity_index", "death"),
                ),
                PrespecifiedSensitivitySpec(
                    spec_id="age_restricted_cubic_spline",
                    axis="functional_form",
                    strategy="restricted_cubic_spline",
                    execution_variables=("age",),
                ),
                PrespecifiedSensitivitySpec(
                    spec_id="comorbidity_restricted_cubic_spline",
                    axis="functional_form",
                    strategy="restricted_cubic_spline",
                    execution_variables=("comorbidity_index",),
                ),
            ]
        )
    return specs


def _context(*, exact: bool = True) -> ResearchContext:
    data_constraints = json.dumps(
        {
            "analysis_design": {
                "analysis_family": "association_study",
                "analysis_unit": "icu_stay",
                "cluster_unit": "patient",
                "variance_estimator": "cluster_robust",
            },
            "cohort": {"age_min": 18, "exclude_readmissions": False},
        }
    )
    exact_fields = (
        {
            "covariates": ["age", "sex", "Comorbidity"],
            "covariate_selection": "exact",
            "covariate_authority": "user",
            "covariate_rationales": {
                "age": "Prespecified baseline demographic confounder fixed before the landmark.",
                "sex": "Prespecified baseline demographic confounder fixed before the landmark.",
                "Comorbidity": "Prespecified chronic comorbidity burden recorded before the landmark.",
            },
            "covariate_temporal_roles": {
                "age": "baseline_static",
                "sex": "baseline_static",
                "Comorbidity": "baseline_static",
            },
            "covariate_operationalizations": {
                "age": "age",
                "sex": "sex",
                "Comorbidity": "comorbidity_index",
            },
        }
        if exact
        else {"covariate_selection": "planner_selectable"}
    )
    return ResearchContext(
        research_question=(
            "Among adult ICU stays, how is the injury stage in the first 24 h "
            "associated with in-hospital death after a 24 h landmark, with a graded "
            "trend, and how does ICU length of stay differ by stage?"
        ),
        cohort=CohortDescriptor(
            cohort_name="synthetic_landmark",
            database="synthetic",
            n_stays=0,
            inclusion_criteria=["age range: 18 to *"],
            id_columns=["patient_stay_id"],
            outcome_columns=["death", "los_icu"],
            requested_outcome_columns=["death", "los_icu"],
            provenance={
                "database": "synthetic",
                "analysis_unit": "icu_stay",
                "stay_id_columns": ["patient_stay_id"],
                "patient_id_columns": [],
                "patient_identity_available": False,
                "evidence_stage": "metadata_only_planning",
                "patient_rows_read": False,
                "replacement_row_identity": {
                    "output_identity_column": "patient_stay_id",
                    "mapping_file_sha256": "c" * 64,
                    "mapped_cohort_rows": 0,
                    "patient_group_derivation": {
                        "algorithm": "prefix_before_:s",
                        "delimiter": ":s",
                    },
                },
            },
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            ConceptDescriptor(
                name="patient_stay_id",
                description="Host-verified unique ICU-stay identity.",
                role=VariableRole.ID,
                dtype="string",
            ),
            ConceptDescriptor(
                name="age",
                description="patient age",
                role=VariableRole.DEMOGRAPHIC,
                dtype="float64",
                unit="years",
                source_concept="age",
            ),
            ConceptDescriptor(
                name="sex",
                description="patient sex",
                role=VariableRole.DEMOGRAPHIC,
                dtype="float64",
                source_concept="sex",
                observed_domain={"n_unique": 2, "is_binary": True, "levels": [0, 1]},
            ),
            ConceptDescriptor(
                name="comorbidity_index",
                description="Chronic comorbidity burden index",
                role=VariableRole.OTHER,
                dtype="float64",
                source_concept="comorbidity_index",
            ),
            ConceptDescriptor(
                name="severity_score_24h",
                description="Illness severity score aggregated over the first 24 h",
                role=VariableRole.COMPOSITE_SCORE,
                dtype="float64",
                source_concept="severity_score",
            ),
            ConceptDescriptor(
                name="death_time_hours",
                role=VariableRole.TIME,
                dtype="float64",
                observation_semantics=ObservationSemantics(
                    kind="conditional_event_time",
                    event_status_column="death",
                    representative_column="death_time_hours",
                ),
            ),
            ConceptDescriptor(
                name="followup_time_hours", role=VariableRole.OTHER, dtype="float64"
            ),
            _ordinal("injury_stage_alt_a"),
            _ordinal("injury_stage_alt_b"),
            ConceptDescriptor(name="first_stay_flag", role=VariableRole.OTHER, dtype="float64"),
            _ordinal("injury_stage"),
            ConceptDescriptor(
                name="death",
                description="in hospital mortality",
                role=VariableRole.OUTCOME,
                dtype="float64",
                source_concept="death",
            ),
            ConceptDescriptor(name="stage_source_flag", role=VariableRole.OTHER, dtype="float64"),
            ConceptDescriptor(name="window_row_count", role=VariableRole.OTHER, dtype="float64"),
            ConceptDescriptor(
                name="los_icu",
                description="ICU length of stay",
                role=VariableRole.OUTCOME,
                dtype="float64",
                unit="days",
                source_concept="los_icu",
            ),
        ],
        time_windows=[
            TimeWindow(
                name="first_24h",
                anchor="icu_admission",
                start_hours=0.0,
                end_hours=24.0,
                rationale="Outer feature-materialization window bound by the host.",
            )
        ],
        target_outcome="death",
        endpoint=EndpointSpec(
            name="death", kind="binary", absence_semantics="no_absent_rows", levels=[0, 1]
        ),
        primary_exposure="injury_stage",
        user_preferences=UserPreferences(
            data_constraints=data_constraints,
            must_have_outputs=(
                "Use the typed association.ordinal_trend action for the secondary ICU "
                "length-of-stay analysis. Generate figures for the primary result."
            ),
            landmark_hours=24.0,
            sensitivity_specs=_specs(exact=exact),
            **exact_fields,
        ),
    )


def _request(context: ResearchContext, *, cohort_mode: str = "predicate_filtered"):
    types = candidate_analysis_types(context)
    variables = select_progressive_variables(context)
    return build_family_spec_request(
        context,
        analysis_types=types,
        variable_roster=variables,
        allowed_literature_citation_keys=ALLOWED_CITATIONS,
        direct_comparator_literature_keys=DIRECT_COMPARATORS,
        comparison_literature_keys=DIRECT_COMPARATORS,
        required_primary_cohort_selection_mode=cohort_mode,
    )


def _spec_payload(request, *, adjustment_set: list[dict] | None = None, **overrides):
    payload = {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": adjustment_set or [],
        "reader_display_labels": [
            {"key": key, "value": LABELS[key]} for key in request.required_reader_label_keys
        ],
        "comparator_applications": [
            {
                "citation_key": key,
                "application": (
                    "Compare population, exposure definition, time zero, and estimand "
                    f"with {key} without copying its design or claiming novelty."
                ),
            }
            for key in request.direct_comparator_literature_keys
        ],
        "roster_decision_note": "Roster follows the host candidates and their timing authority.",
    }
    payload.update(overrides)
    return payload


PLANNER_ROSTER = [
    {
        "name": "age",
        "coding": "continuous",
        "reference_level_index": None,
        "clinical_rationale": "Age is fixed before admission and relates to both stage and death.",
    },
    {
        "name": "sex",
        "coding": "binary",
        "reference_level_index": 0,
        "clinical_rationale": "Sex is fixed before admission and relates to presentation and mortality.",
    },
    {
        "name": "severity_score_24h",
        "coding": "continuous",
        "reference_level_index": None,
        "clinical_rationale": "Severity measured within the 24 h window is available at the landmark and predicts death.",
    },
]
