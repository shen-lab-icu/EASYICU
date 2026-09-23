"""Contract tests for the ``family_spec_v1`` planner strategy.

The Planner returns one small typed spec; the host projects and compiles the
whole landmark categorical plan through the unchanged Progressive validators.
Every test runs offline with the reviewed scripted mock and a synthetic,
case-neutral landmark context.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.family_spec_planner import (
    family_spec_response_shape,
    family_spec_structured_output_request,
    family_spec_user_prompt,
    parse_family_plan_spec,
)
from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
    candidate_analysis_types,
    select_progressive_variables,
)
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.progressive_planning import (
    run_progressive_planner,
)
from easyicu.research_agent.planning.adjustment_authority import (
    host_proven_temporal_roles,
)
from easyicu.research_agent.planning.family_spec import (
    FamilySpecError,
    build_family_spec_request,
    family_template_id_for_context,
)
from easyicu.research_agent.planning.preplan_know_how import PlannerKnowHowBinding
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.contracts.endpoint import EndpointSpec
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


def test_template_matches_the_landmark_categorical_family_only() -> None:
    context = _context()
    assert (
        family_template_id_for_context(context, analysis_types=("association_study",))
        == "landmark_categorical_association"
    )
    no_landmark = context.model_copy(
        update={
            "user_preferences": context.user_preferences.model_copy(
                update={"landmark_hours": None, "sensitivity_specs": []}
            )
        }
    )
    assert family_template_id_for_context(no_landmark, analysis_types=("association_study",)) is None
    # A descriptive headline routes the same closed-level exposure to the
    # descriptive family; an unsupported headline routes nowhere.
    assert (
        family_template_id_for_context(context, analysis_types=("descriptive_epidemiology",))
        == "descriptive_exposure_outcome"
    )
    # A prediction headline routes to the static prediction family (no
    # exposure needed); an unsupported headline routes nowhere.
    assert (
        family_template_id_for_context(context, analysis_types=("prediction_model",))
        == "static_prediction_model"
    )
    assert family_template_id_for_context(context, analysis_types=("causal_inference",)) is None


def test_host_proven_timing_admits_window_bound_scores_and_rejects_unproven_columns() -> None:
    roles = host_proven_temporal_roles(_context(exact=False))
    assert roles["age"] == "baseline_static"
    assert roles["sex"] == "baseline_static"
    assert roles["severity_score_24h"] == "at_or_before_time_zero"
    assert "comorbidity_index" not in roles
    assert "followup_time_hours" not in roles
    late_landmark = _context(exact=False).model_copy(
        update={"time_windows": [TimeWindow(name="first_48h", end_hours=48.0)]}
    )
    assert "severity_score_24h" not in host_proven_temporal_roles(late_landmark)


def test_request_exposes_candidates_with_the_host_timing_verdict() -> None:
    request = _request(_context(exact=False))
    by_name = {item.name: item for item in request.adjustment_candidates}
    assert by_name["age"].selectable and by_name["age"].host_temporal_role == "baseline_static"
    assert by_name["severity_score_24h"].selectable
    assert by_name["severity_score_24h"].host_temporal_role == "at_or_before_time_zero"
    assert not by_name["comorbidity_index"].selectable
    assert "exact user-reviewed roster" in by_name["comorbidity_index"].boundary
    assert request.secondary_continuous_outcome == "los_icu"
    assert [item.execution_variables[0] for item in request.alternate_exposures] == [
        "injury_stage_alt_a",
        "injury_stage_alt_b",
    ]
    assert request.first_stay is not None and request.first_stay.execution_variables == ["first_stay_flag"]
    # An untimed ``other`` column is audit context under Planner selection; only
    # an exact roster turns it into a covariate.
    assert request.measurement_audit_columns == [
        "comorbidity_index",
        "stage_source_flag",
        "window_row_count",
    ]
    assert _request(_context(exact=True)).measurement_audit_columns == [
        "stage_source_flag",
        "window_row_count",
    ]
    assert request.cluster_unit == "patient"
    assert request.age_min == 18.0


def test_exact_roster_request_carries_user_authority() -> None:
    request = _request(_context(exact=True))
    assert request.adjustment_selection == "exact"
    assert request.exact_roster == ["age", "sex", "comorbidity_index"]
    assert request.functional_form_spec_ids == {
        "age": "age_restricted_cubic_spline",
        "comorbidity_index": "comorbidity_restricted_cubic_spline",
    }
    assert request.complete_case_spec_id == "complete_case_primary_covariates"


def test_one_call_compiles_the_exact_roster_plan() -> None:
    context = _context(exact=True)
    request = _request(context)
    llm, result = _run(context, [json.dumps(_spec_payload(request))])
    plan = result.output
    assert len(llm.calls) == 1
    assert [step.step_id for step in plan.steps] == [
        "cohort_definition",
        "table_one",
        "measurement_audit",
        "adjusted_association",
        "absolute_risk_context",
        "robustness_replay",
        "age_functional_form",
        "comorbidity_index_functional_form",
        "ordinal_trend",
        "visualization",
        "report",
    ]
    primary = plan.steps[3].model_requirements[0]
    assert primary.covariates == ["age", "sex", "comorbidity_index"]
    assert primary.covariate_temporal_roles == {
        "age": "baseline_static",
        "sex": "baseline_static",
        "comorbidity_index": "baseline_static",
    }
    assert primary.primary_contrast_level == "3" and primary.exposure_reference_level == "0"
    metrics = result.facts.prompt_metrics
    assert metrics["planner_strategy"] == FAMILY_SPEC_STRATEGY
    assert metrics["host_step_materialization_count"] == 11
    assert metrics["family_spec_request_sha256"] == request.request_sha256
    assert metrics["compiled_plan_sha256"] == result.facts.compile_receipt.analysis_plan_sha256
    assert result.facts.complete_for_persistence
    assert [item.spec_id for item in plan.robustness_specs] == ["complete_case_primary_covariates"]
    replay = plan.steps[5]
    assert set(replay.sensitivity_spec_ids) >= {
        "stage_alt_a",
        "stage_alt_b",
        "first_stay_only",
        "complete_case_primary_covariates",
        "landmark_24h_primary",
    }


def _host_contract_binding(input_key: str, columns: list[str]) -> dict:
    """A resolved typed binding carrying the *producer's* own column contract."""

    kind, _, product = input_key.partition(":")
    return {
        "relative_path": f"steps/{product}.csv",
        "sha256": "0" * 64,
        "declared_kind": kind,
        "evidence_kind": "table",
        "product": product,
        "evidence_id": f"ev_{product}",
        "identity_row": {},
        "product_contract": {"columns": list(columns), "row_count": 4},
        "consumption_contract": {
            "input_key": input_key,
            "mode": "all_rows",
            "artifact_sha256": "0" * 64,
        },
    }


def test_landmark_family_figure_and_report_steps_are_host_owned() -> None:
    """The template's figure and report steps never fall to the Coder.

    The reference workflow's last two steps are rendering and reporting.  Their
    typed inputs are exactly the four-table profile the deterministic composite
    association renderer owns, and the report step is the evidence-bound
    reporting executor's.  Ownership is resolved with bindings that carry each
    *producer's* real column contract, so a drift on either side -- the
    template declaring a different input, a producer dropping a column the
    renderer reads -- surfaces here as a lost owner instead of as a Coder
    writing plotting code in a formal run.
    """
    import pandas as pd

    from easyicu.research_agent.contracts.figure_plan import (
        ABSOLUTE_RISK_ASSOCIATION_COMPOSITE_INPUTS,
    )
    from easyicu.research_agent.execution.runners.adjusted_association_executor import (
        ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS,
    )
    from easyicu.research_agent.execution.runners.deterministic_descriptive import (
        _group_rows,
    )
    from easyicu.research_agent.execution.runners.deterministic_robustness import (
        _MATRIX_COLUMNS,
        _robustness_summary,
    )
    from easyicu.research_agent.execution.runners.selection import (
        select_standard_executor,
    )
    from easyicu.research_agent.planning.figure_plan_shaping import (
        bind_deterministic_figure_panels,
    )

    context = _context(exact=True)
    request = _request(context)
    _, result = _run(context, [json.dumps(_spec_payload(request))])
    # The same panel binding the pipeline applies before plan digest and review.
    plan, _findings = bind_deterministic_figure_panels(plan=result.output)
    steps = {step.step_id: step for step in plan.steps}

    visualization = steps["visualization"]
    assert tuple(visualization.inputs) == ABSOLUTE_RISK_ASSOCIATION_COMPOSITE_INPUTS
    assert [panel.article_role for panel in visualization.figure_panels] == [
        "descriptive_result",
        "primary_estimand",
        "robustness",
        "robustness",
    ]
    absolute_risk_columns = list(
        _group_rows(
            exposure="stage",
            group_type="exposure_level",
            group_value="1",
            label="stage = 1",
            mask=pd.Series([True, False]),
            outcome=pd.Series([1.0, 0.0]),
            n_total=2,
            measured_col=None,
            count_col=None,
        )[0]
    )
    producer_columns = {
        "table:absolute_risk_context": absolute_risk_columns,
        "table:adjusted_association_estimates": list(ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS),
        "table:robustness_matrix": [*_MATRIX_COLUMNS, "se", "evidence_id"],
        "table:robustness_summary": list(_robustness_summary(pd.DataFrame()).columns),
    }
    bindings = {
        key: _host_contract_binding(key, producer_columns[key]) for key in visualization.inputs
    }
    figure_owner = select_standard_executor(
        visualization, plan=plan, resolved_bindings=bindings
    )
    assert figure_owner is not None
    assert figure_owner.analysis_kind == "composite_descriptive_figure"

    report_owner = select_standard_executor(steps["report"], plan=plan)
    assert report_owner is not None
    assert report_owner.analysis_kind == "evidence_bound_scientific_reporting"


def test_planner_selected_roster_uses_host_timing_not_the_rationale() -> None:
    context = _context(exact=False)
    request = _request(context)
    llm, result = _run(context, [json.dumps(_spec_payload(request, adjustment_set=PLANNER_ROSTER))])
    primary = result.output.steps[3].model_requirements[0]
    assert primary.covariates == ["age", "sex", "severity_score_24h"]
    assert primary.covariate_temporal_roles == {
        "age": "baseline_static",
        "sex": "baseline_static",
        "severity_score_24h": "at_or_before_time_zero",
    }
    assert primary.covariate_rationales["severity_score_24h"].startswith("Severity measured")
    step_ids = [step.step_id for step in result.output.steps]
    assert "age_functional_form" in step_ids
    assert "severity_score_24h_functional_form" in step_ids
    assert "comorbidity_index_functional_form" not in step_ids
    assert result.facts.prompt_metrics["family_spec_adjustment_set"] == [
        "age",
        "sex",
        "severity_score_24h",
    ]


def test_unproven_covariate_is_refused_even_with_a_rationale() -> None:
    context = _context(exact=False)
    request = _request(context)
    bad = [
        *PLANNER_ROSTER[:2],
        {
            "name": "comorbidity_index",
            "coding": "continuous",
            "reference_level_index": None,
            "clinical_rationale": "Chronic comorbidity is recorded before admission and predicts death.",
        },
    ]
    with pytest.raises(FamilySpecError) as caught:
        parse_family_plan_spec(
            json.dumps(_spec_payload(request, adjustment_set=bad)), request
        )
    assert caught.value.reason_code == "family_spec_covariate_timing_unproven"
    # The retry loop feeds the refusal back to the provider; a provider that
    # keeps answering the same way fails closed with the same reason code.
    with pytest.raises(Exception, match="family_spec_covariate_timing_unproven"):
        _run(context, [json.dumps(_spec_payload(request, adjustment_set=bad))] * 3)


def test_exact_roster_cannot_be_edited_by_the_spec() -> None:
    context = _context(exact=True)
    request = _request(context)
    with pytest.raises(FamilySpecError) as caught:
        parse_family_plan_spec(
            json.dumps(_spec_payload(request, adjustment_set=PLANNER_ROSTER[:2])), request
        )
    assert caught.value.reason_code == "family_spec_exact_roster_mismatch"


@pytest.mark.parametrize(
    ("override", "reason_code"),
    [
        ({"request_sha256": "0" * 64}, "family_spec_request_digest_mismatch"),
        (
            {"reader_display_labels": [{"key": "sex", "value": "sex"}]},
            "family_spec_reader_label_missing",
        ),
        ({"comparator_applications": []}, "family_spec_comparator_application_missing"),
    ],
)
def test_spec_validation_fails_closed(override, reason_code) -> None:
    context = _context(exact=True)
    request = _request(context)
    with pytest.raises(FamilySpecError) as caught:
        parse_family_plan_spec(json.dumps(_spec_payload(request, **override)), request)
    assert caught.value.reason_code == reason_code


def test_strict_schema_is_closed_and_run_bound() -> None:
    request = _request(_context(exact=False))
    schema = json.loads(family_spec_structured_output_request(request).schema_json)
    assert schema["additionalProperties"] is False
    names = schema["properties"]["adjustment_set"]["items"]["properties"]["name"]["enum"]
    assert set(names) == {"age", "sex", "severity_score_24h"}
    assert schema["properties"]["request_sha256"]["enum"] == [request.request_sha256]
    prompt = family_spec_user_prompt(request, variable_descriptions={"age": "patient age"})
    assert '"selectable": false' in prompt and "comorbidity_index" in prompt
    assert request.request_sha256 in prompt


def test_route_without_strict_schema_gets_the_written_shape_on_the_first_request() -> None:
    # A live eICU demo route without strict-schema support failed all three
    # attempts: the first answer guessed the item keys, the second the schema
    # version, the third the reference-index rule. The first request now
    # carries every field, closed value, and the coding rule.
    context = _context(exact=False)
    request = _request(context)
    llm, result = _run(
        context, [json.dumps(_spec_payload(request, adjustment_set=PLANNER_ROSTER))]
    )
    assert len(llm.calls) == 1 and result.output.steps
    messages, kwargs = llm.calls[0]
    assert kwargs.get("structured_output") is None
    prompt = messages[-1].content
    assert "Response shape (no schema is attached on this route)" in prompt
    shape = family_spec_response_shape(request)
    assert shape in prompt
    for fragment in (
        '"schema_version": "easyicu.family_plan_spec/1"',
        f'"family_id": "{request.family_id}"',
        f'"request_sha256": "{request.request_sha256}"',
        '"name" (one of ["age", "sex", "severity_score_24h"])',
        "null for continuous coding",
        '["comparator_alpha_2020_1"]',
        # The rerun's first answer labelled icu_unit_type "ICU unit type" and
        # was refused as a mechanical label; the rule now ships with a
        # concrete example drawn from the request's own keys.
        'restates its key (such as "injury stage" for injury_stage) is rejected',
    ):
        assert fragment in shape, fragment
    assert "reference_level_index" in messages[0].content


def test_strict_route_gets_the_schema_and_not_the_written_shape() -> None:
    context = _context(exact=False)
    request = _request(context)
    llm = ScriptedMockLLMClient([json.dumps(_spec_payload(request, adjustment_set=PLANNER_ROSTER))])
    llm.supports_strict_json_schema = True
    ProgressivePlannerAgent(llm).run_attempt(
        context,
        planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=ALLOWED_CITATIONS,
        direct_comparator_literature_keys=DIRECT_COMPARATORS,
        comparison_literature_keys=DIRECT_COMPARATORS,
        enforce_article_contract=True,
        article_contract_context=context,
        planning_contract_context="",
        required_primary_cohort_selection_mode="predicate_filtered",
    )
    messages, kwargs = llm.calls[0]
    assert kwargs["structured_output"].authority_sha256 == (
        family_spec_structured_output_request(request).authority_sha256
    )
    assert "Response shape" not in messages[-1].content


def test_written_shape_and_strict_schema_name_the_same_keys() -> None:
    requests = [
        _request(_context(exact=True)),
        _request(_context(exact=False)),
        _request(_descriptive_context(), cohort_mode="all_input_rows"),
        _request(_phenotyping_context(), cohort_mode=None),
        _request(_prediction_context(), cohort_mode=None),
    ]
    assert len({request.family_id for request in requests}) == 4
    for request in requests:
        schema = json.loads(family_spec_structured_output_request(request).schema_json)
        shape = family_spec_response_shape(request)
        written = re.findall(r'^- "([a-z_0-9]+)"', shape, flags=re.MULTILINE)
        assert len(written) == len(set(written)), request.family_id
        assert set(written) == set(schema["properties"]), request.family_id


def test_host_writes_the_envelope_and_the_request_digest_still_binds() -> None:
    request = _request(_context(exact=True))
    for envelope in (
        {"schema_version": "1.0", "family_id": "landmark association"},
        {"schema_version": None},
    ):
        payload = {**_spec_payload(request), **envelope}
        spec = parse_family_plan_spec(json.dumps(payload), request)
        assert spec.schema_version == "easyicu.family_plan_spec/1"
        assert spec.family_id == request.family_id
    omitted = _spec_payload(request)
    del omitted["schema_version"], omitted["family_id"]
    assert parse_family_plan_spec(json.dumps(omitted), request).family_id == request.family_id
    unbound = _spec_payload(request)
    del unbound["request_sha256"]
    with pytest.raises(ValidationError):
        parse_family_plan_spec(json.dumps(unbound), request)
    with pytest.raises(FamilySpecError) as caught:
        parse_family_plan_spec(
            json.dumps(_spec_payload(request, request_sha256="f" * 64)), request
        )
    assert caught.value.reason_code == "family_spec_request_digest_mismatch"


def test_retry_shows_the_rejected_answer_beside_the_reason_and_the_shape() -> None:
    context = _context(exact=False)
    request = _request(context)
    wrong_reference = [
        {**item, "reference_level_index": 0} if item["name"] == "age" else item
        for item in PLANNER_ROSTER
    ]
    rejected = json.dumps(_spec_payload(request, adjustment_set=wrong_reference))
    llm, result = _run(
        context,
        [rejected, json.dumps(_spec_payload(request, adjustment_set=PLANNER_ROSTER))],
    )
    assert len(llm.calls) == 2 and result.output.steps
    retry, _ = llm.calls[1]
    assert [message.role for message in retry[-2:]] == ["assistant", "user"]
    assert retry[-2].content == rejected
    assert "family_spec_reference_index_shape" in retry[-1].content
    assert family_spec_response_shape(request) in retry[-1].content


def _reviewable_plan(plan) -> list[str]:
    return list(plan.design_selection.selected.reviewable_plan)


def test_spline_recommendation_names_only_the_covariates_that_get_a_spline() -> None:
    # A live eICU demo summary promised a restricted cubic spline for sex and
    # admission type; the plan itself only checks continuous covariates.
    context = _context(exact=False)
    request = _request(context)
    _llm, result = _run(
        context, [json.dumps(_spec_payload(request, adjustment_set=PLANNER_ROSTER))]
    )
    plan = result.output
    spline_steps = sorted(
        step.step_id for step in plan.steps if step.step_id.endswith("_functional_form")
    )
    assert spline_steps == ["age_functional_form", "severity_score_24h_functional_form"]
    sensitivity = _reviewable_plan(plan)[5]
    assert f"restricted cubic spline for {LABELS['age']}" in sensitivity
    assert f"restricted cubic spline for {LABELS['severity_score_24h']}" in sensitivity
    assert LABELS["sex"] not in sensitivity


def test_recommendation_follows_the_question_language_without_internal_identifiers() -> None:
    context = _context(exact=False)
    request = _request(context)
    payload = [json.dumps(_spec_payload(request, adjustment_set=PLANNER_ROSTER))]
    english_result = _run(context, payload)[1]
    english = english_result.output
    chinese_context = context.model_copy(
        update={"research_question": "24 小时 landmark 时的损伤分期与院内死亡有什么关联？"}
    )
    chinese_request = _request(chinese_context)
    chinese_result = _run(
        chinese_context,
        [json.dumps(_spec_payload(chinese_request, adjustment_set=PLANNER_ROSTER))],
    )[1]
    chinese = chinese_result.output
    han = re.compile(r"[一-鿿]")
    assert len(_reviewable_plan(english)) == len(_reviewable_plan(chinese)) == 6
    assert all(not han.search(item) for item in _reviewable_plan(english))
    assert all(han.search(item) for item in _reviewable_plan(chinese))
    # No label prefix in either language: every reader labels the six
    # positions itself, so a prefix would only repeat it in one language.
    assert _reviewable_plan(chinese)[0].startswith("研究队列")
    assert _reviewable_plan(english)[0].startswith("Analysis rows of the study cohort")
    for item in [*_reviewable_plan(english), *_reviewable_plan(chinese)]:
        assert not re.match(r"^[A-Za-z /-]{3,40}:\s", item), item
        assert not re.match(r"^[\u4e00-\u9fff]{2,12}：", item), item
    assert f"{LABELS['sex']} 的限制性立方样条" not in _reviewable_plan(chinese)[5]
    assert f"{LABELS['age']} 的限制性立方样条" in _reviewable_plan(chinese)[5]
    # Only the recommendation follows the question: the executable plan is
    # the same science, so the step roster does not change.
    assert [step.step_id for step in english.steps] == [step.step_id for step in chinese.steps]
    for plan, result in ((english, english_result), (chinese, chinese_result)):
        prose = [*_reviewable_plan(plan), *(step.objective for step in result.facts.outline.steps)]
        for text in prose:
            assert request.cohort_name not in text
            assert f"row identity {request.identity_column}" not in text


def test_every_family_recommendation_keeps_internal_identifiers_out() -> None:
    from easyicu.research_agent.planning.family_spec import (
        build_descriptive_skeleton,
        build_phenotyping_skeleton,
        build_prediction_skeleton,
    )

    cases = [
        (
            _request(_descriptive_context(), cohort_mode="all_input_rows"),
            build_descriptive_skeleton,
            lambda request: _descriptive_payload(
                request, baseline_variables=["age", "sex"]
            ),
        ),
        (
            _request(_phenotyping_context(), cohort_mode=None),
            build_phenotyping_skeleton,
            lambda request: _phenotyping_payload(
                request,
                features=["hr_max", "lactate_max", "map_min"],
                baseline=["age"],
                membership=None,
            ),
        ),
        (
            _request(_prediction_context(), cohort_mode=None),
            build_prediction_skeleton,
            lambda request: _prediction_payload(
                request, features=["age", "sex", "hr_max", "lactate_max", "map_min"]
            ),
        ),
    ]
    for request, builder, payload in cases:
        spec = parse_family_plan_spec(json.dumps(payload(request)), request)
        outline = builder(request, spec).outline
        texts = [
            *outline.design_selection.selected.reviewable_plan,
            *(step.objective for step in outline.steps),
        ]
        for text in texts:
            assert request.cohort_name not in text, (request.family_id, text)
            assert f"row identity {request.identity_column}" not in text, request.family_id


def test_same_spec_compiles_to_the_same_plan_digest() -> None:
    context = _context(exact=True)
    request = _request(context)
    first = _run(context, [json.dumps(_spec_payload(request))])[1]
    second = _run(context, [json.dumps(_spec_payload(request))])[1]
    assert (
        first.facts.compile_receipt.analysis_plan_sha256
        == second.facts.compile_receipt.analysis_plan_sha256
    )
    assert canonical_sha256(first.facts.outline.model_dump(mode="json")) == canonical_sha256(
        second.facts.outline.model_dump(mode="json")
    )


def test_strategy_falls_back_to_progressive_when_no_template_fits() -> None:
    context = _context(exact=True).model_copy(
        update={
            "user_preferences": _context().user_preferences.model_copy(
                update={"landmark_hours": None, "sensitivity_specs": []}
            )
        }
    )
    llm = ScriptedMockLLMClient([])
    agent = ProgressivePlannerAgent(llm)
    with pytest.raises(Exception, match="exhausted|attempts failed"):
        agent.run_attempt(
            context,
            planner_strategy=FAMILY_SPEC_STRATEGY,
            allowed_literature_citation_keys=ALLOWED_CITATIONS,
            enforce_article_contract=True,
            article_contract_context=context,
        )
    metrics = agent.last_result.facts.prompt_metrics if agent.last_result else {}
    # The attempt failed inside the ordinary outline call; the fallback reason
    # is recorded on the ordinary Progressive metrics before that call.
    assert metrics == {} or metrics.get("family_spec_fallback_reason") == "no_family_template_for_context"


def test_unknown_strategy_is_refused() -> None:
    context = _context()
    with pytest.raises(ValueError, match="unsupported planner_strategy"):
        ProgressivePlannerAgent(ScriptedMockLLMClient([])).run_attempt(
            context, planner_strategy="monolithic_v9"
        )


class _RecordingEvidence:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}

    def get(self, evidence_id_or_alias: str) -> object | None:
        return self.records.get(evidence_id_or_alias)

    def register_file(self, **kwargs: object) -> object:
        evidence_id = str(kwargs["evidence_id"])
        source_path = Path(str(kwargs["source_path"]))
        self.records[evidence_id] = {
            **dict(kwargs),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }
        return self.records[evidence_id]


def test_orchestration_persists_the_family_spec_chain(tmp_path: Path) -> None:
    context = _context(exact=True)
    request = _request(context)
    llm = ScriptedMockLLMClient([json.dumps(_spec_payload(request))])
    evidence = _RecordingEvidence()
    cohort_path = tmp_path / "cohort.parquet"
    cohort_path.write_bytes(b"synthetic zero-row cohort")
    result = run_progressive_planner(
        planner=ProgressivePlannerAgent(llm),
        context=context,
        run_dir=tmp_path,
        evidence=evidence,
        prompt_pack_version="test-v1",
        resume_checkpoint_path=None,
        resume_checkpoint_sha256=None,
        cohort_path=cohort_path,
        llm_signature="mock:test",
        planner_kwargs={
            "planner_strategy": FAMILY_SPEC_STRATEGY,
            "allowed_literature_citation_keys": ALLOWED_CITATIONS,
            "direct_comparator_literature_keys": DIRECT_COMPARATORS,
            "comparison_literature_keys": DIRECT_COMPARATORS,
            "enforce_article_contract": True,
            "article_contract_context": context,
            "required_primary_cohort_selection_mode": "predicate_filtered",
        },
        know_how_binding=PlannerKnowHowBinding(),
        planning_contract_context="",
        finding_sink=lambda _finding: None,
    )
    assert result.generation_mode == "llm_progressive_v2"
    assert result.prompt_metrics["planner_strategy"] == FAMILY_SPEC_STRATEGY
    assert (tmp_path / "progressive_plan_outline.json").exists()
    assert (tmp_path / "progressive_plan_skeleton.json").exists()
    assert (tmp_path / "progressive_step_materializations.json").exists()
    checkpoints = sorted(tmp_path.glob("progressive_planner_checkpoint_*.json"))
    # outline + foundation + one per host-materialized step
    assert len(checkpoints) == 2 + 11
    assert len(llm.calls) == 1


def test_template_failure_keeps_the_compiler_reason_code() -> None:
    context = _context(exact=True)
    request = _request(context)
    # A label roster missing a required design variable is caught by the
    # ordinary foundation validator, not by a template-specific shortcut.
    payload = _spec_payload(request)
    payload["reader_display_labels"] = [
        item for item in payload["reader_display_labels"] if item["key"] != "los_icu"
    ]
    with pytest.raises(FamilySpecError) as caught:
        parse_family_plan_spec(json.dumps(payload), request)
    assert caught.value.reason_code == "family_spec_reader_label_missing"


def test_planner_selected_roster_seals_into_the_plan_bound_runtime_contract() -> None:
    """End to end: Planner spec → host plan → sealed runtime authority.

    The Web projection for a ``planner_selectable`` landmark study seals only
    the executable column domain; the same digest-bound owner then seals the
    Planner's roster from the compiled primary model, so the reviewed plan and
    the deterministic executor agree on one adjustment set without a second
    run or a user-filled roster.
    """
    from easyicu.research_agent.authority.current_case_scientific_runtime import (
        build_current_case_scientific_runtime_authority,
    )
    from easyicu.research_agent.orchestration.scientific_runtime import (
        ScientificRuntimeAuthorities,
    )

    context = _context(exact=False)
    request = _request(context)
    _, result = _run(context, [json.dumps(_spec_payload(request, adjustment_set=PLANNER_ROSTER))])
    plan = result.output
    unsealed = build_current_case_scientific_runtime_authority(
        {
            "schema_version": "easyicu.landmark_categorical_association_runtime_authority/3",
            "authority_kind": "landmark_categorical_association",
            "protocol_content_sha256": "c" * 64,
            "cohort_method": "signed_landmark_analysis_cohort",
            "primary_method": "signed_landmark_categorical_association",
            "plan_intent": "Estimate the adjusted categorical association at the 24-hour landmark.",
            "landmark_spec_id": "landmark_24h_primary",
            "cohort_product": "artifact:analysis_cohort",
            "cohort_flow_product": "table:cohort_flow",
            "primary_product": "table:adjusted_association_estimates",
            "exposure_column": "injury_stage",
            "exposure_kind": "ordinal",
            "exposure_levels": ["0", "1", "2", "3"],
            "exposure_reference_level": "0",
            "primary_contrast_level": "3",
            "outcome_column": "death",
            "event_time_column": "death_time_hours",
            "observation_duration_column": "followup_time_hours",
            "observation_duration_unit": "hours",
            "landmark_hours": 24,
            "exclude_negative_event_times": True,
            "require_alive_at_landmark": True,
            "required_adjustment_columns": [],
            "categorical_adjustment_columns": [],
            "dependence": None,
            "interpretation": "descriptive_prognostic_association_not_causal",
            "association_model_grid": None,
            "plan_bound_adjustment_roster": {
                "authority": "plan_primary_model",
                "admissible_columns": ["age", "sex", "comorbidity_index", "severity_score_24h"],
                "admissible_categorical_columns": [],
                "sealed": False,
            },
        }
    )
    runtime = ScientificRuntimeAuthorities(trajectory=None, current_case=unsealed)
    bound, findings = runtime.bind_plan(plan)
    sealed_runtime = runtime.seal_for_plan(plan)
    sealed = sealed_runtime.current_case
    assert sealed.required_adjustment_columns == ("age", "sex", "severity_score_24h")
    assert findings[0].detail["adjustment_roster"] == ["age", "sex", "severity_score_24h"]
    assert findings[0].detail["unsealed_execution_contract_sha256"] == unsealed.execution_contract_sha256
    sealed_runtime.validate_plan(bound)
    primary = sealed.governed_primary_step(bound)
    assert primary.step_id == "adjusted_association"
    assert primary.model_requirements[0].covariates == ["age", "sex", "severity_score_24h"]
    assert sealed.plan_rule_ref in primary.icu_rule_refs
    assert [step.step_id for step in bound.steps] == [step.step_id for step in plan.steps]


# ---------------------------------------------------------------------------
# Continuous-exposure (restricted cubic spline) landmark family
# ---------------------------------------------------------------------------

SPLINE_LABELS = {
    "peak_lactate": "Peak lactate in the first 24 h (mmol/L)",
    "death": "In-hospital death",
    "death_time_hours": "Time of in-hospital death (hours)",
    "followup_time_hours": "In-hospital follow-up time (hours)",
    "first_stay_flag": "First ICU stay indicator",
    "age": "Age at ICU admission (years)",
    "sex": "Patient sex",
    "comorbidity_index": "Comorbidity burden index",
    "severity_score_24h": "Illness severity score in the first 24 h",
}


def _spline_context() -> ResearchContext:
    base = _context(exact=False)
    lactate = [
        ConceptDescriptor(
            name="peak_lactate",
            description="lactate",
            role=VariableRole.LAB,
            dtype="float32",
            unit="mmol/L",
            source_concept="lact",
            analysis_window="icu_admission[0,24]h",
        ),
        ConceptDescriptor(
            name="lactate_min",
            description="lactate",
            role=VariableRole.LAB,
            dtype="float32",
            unit="mmol/L",
            source_concept="lact",
            analysis_window="icu_admission[0,24]h",
        ),
        ConceptDescriptor(
            name="lactate_n",
            description="Non-null observation count: lactate",
            role=VariableRole.META,
            dtype="float64",
            source_concept="lact",
            analysis_window="icu_admission[0,24]h",
        ),
        ConceptDescriptor(
            name="lactate_measured",
            description="Measurement availability: lactate",
            role=VariableRole.META,
            dtype="int64",
            source_concept="lact",
            analysis_window="icu_admission[0,24]h",
        ),
    ]
    keep = {
        "stay_id", "patient_stay_id", "age", "sex", "comorbidity_index", "severity_score_24h",
        "death_time_hours", "followup_time_hours", "first_stay_flag", "death",
    }
    variables = [item for item in base.variables if item.name in keep] + lactate
    specs = [spec for spec in base.user_preferences.sensitivity_specs if spec.axis != "exposure_definition"]
    preferences = base.user_preferences.model_copy(update={"sensitivity_specs": specs})
    return base.model_copy(
        update={
            "research_question": (
                "Among adult ICU stays, how is the peak lactate in the first 24 h associated with "
                "in-hospital death after a 24 h landmark?"
            ),
            "primary_exposure": "peak_lactate",
            "variables": variables,
            "user_preferences": preferences,
            "cohort": base.cohort.model_copy(
                update={"outcome_columns": ["death"], "requested_outcome_columns": ["death"]}
            ),
        }
    )


def _spline_spec_payload(request, *, adjustment_set):
    return {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": adjustment_set,
        "reader_display_labels": [
            {"key": key, "value": SPLINE_LABELS[key]} for key in request.required_reader_label_keys
        ],
        "comparator_applications": [
            {
                "citation_key": key,
                "application": (
                    f"Compare this landmark analysis with {key} on population, exposure definition, "
                    "time zero, and estimand without copying its design or claiming novelty."
                ),
            }
            for key in request.direct_comparator_literature_keys
        ],
        "roster_decision_note": "Planner-selected baseline covariates within the host timing authority.",
    }


def test_spline_family_request_routes_companions_to_the_audit_and_labels_only_the_design() -> None:
    from easyicu.research_agent.planning.family_spec import LANDMARK_SPLINE_FAMILY_ID

    context = _spline_context()
    assert family_template_id_for_context(
        context, analysis_types=candidate_analysis_types(context)
    ) == LANDMARK_SPLINE_FAMILY_ID
    request = _request(context)
    assert request.family_id == LANDMARK_SPLINE_FAMILY_ID
    assert request.exposure_kind == "continuous"
    assert request.exposure_levels == [] and request.exposure_is_ordered is False
    assert request.exposure_companion_columns == ["lactate_min", "lactate_n", "lactate_measured"]
    selectable = {item.name for item in request.selectable_candidates}
    # Companion aggregates of the exposure are audit context, never covariates.
    assert selectable == {"age", "sex", "severity_score_24h"}
    assert "lactate_min" not in {item.name for item in request.adjustment_candidates}
    assert "lactate_min" not in request.required_reader_label_keys
    assert request.alternate_exposures == []
    assert request.first_stay is not None


def test_spline_family_compiles_and_seals_into_the_plan_bound_spline_runtime() -> None:
    from easyicu.research_agent.authority.current_case_scientific_runtime import (
        build_current_case_scientific_runtime_authority,
    )
    from easyicu.research_agent.orchestration.scientific_runtime import (
        ScientificRuntimeAuthorities,
    )

    context = _spline_context()
    request = _request(context)
    roster = [item for item in PLANNER_ROSTER if item["name"] in {"age", "sex"}]
    llm, result = _run(context, [json.dumps(_spline_spec_payload(request, adjustment_set=roster))])
    plan = result.output
    assert len(llm.calls) == 1
    assert result.facts.prompt_metrics["planner_strategy"] == FAMILY_SPEC_STRATEGY
    step_ids = [step.step_id for step in plan.steps]
    assert "ordinal_trend" not in step_ids
    assert "age_functional_form" in step_ids
    primary = next(step for step in plan.steps if step.planned_analysis_role == "primary")
    requirement = primary.model_requirements[0]
    assert requirement.exposure_source == "peak_lactate"
    assert requirement.covariates == ["age", "sex"]
    assert [term.coding for term in requirement.model_terms if term.role == "exposure"] == ["continuous"]
    assert primary.sensitivity_spec_ids == ["landmark_24h_primary"]
    audit = next(step for step in plan.steps if step.step_id == "measurement_audit")
    assert {"lactate_min", "lactate_n", "lactate_measured"} <= set(audit.inputs)
    table_one = next(step for step in plan.steps if step.step_id == "table_one")
    assert table_one.table_one_spec is not None and table_one.table_one_spec.group_by == "death"

    unsealed = build_current_case_scientific_runtime_authority(
        {
            "schema_version": "easyicu.landmark_spline_runtime_authority/5",
            "authority_kind": "landmark_spline_association",
            "protocol_content_sha256": "e" * 64,
            "plan_method": "signed_landmark_restricted_cubic_spline",
            "plan_intent": "Execute the reviewed 24-hour landmark restricted-cubic-spline association.",
            "plan_outputs": [
                "table:landmark_rcs_curve",
                "table:landmark_rcs_contrasts",
                "table:landmark_linear_sensitivity",
                "table:landmark_adjusted_absolute_risk",
                "table:landmark_population_flow",
                "table:landmark_variable_opportunity_sensitivity",
                "log:landmark_scientific_runtime_receipt",
            ],
            "exposure_column": "peak_lactate",
            "outcome_column": "death",
            "outcome_time_column": "death_time_hours",
            "observation_duration_column": "followup_time_hours",
            "observation_duration_unit": "hours",
            "landmark_hours": 24,
            "required_adjustment_columns": [],
            "categorical_adjustment_columns": [],
            "alternative_exposure_columns": [],
            "dependence": None,
            "adjusted_absolute_risk_product": "table:landmark_adjusted_absolute_risk",
            "population_flow_product": "table:landmark_population_flow",
            "variable_opportunity_sensitivity_product": "table:landmark_variable_opportunity_sensitivity",
            "spline_knot_quantiles": [0.10, 0.50, 0.90],
            "spline_reference": "median_in_primary_population",
            "curve_quantile_range": [0.10, 0.90],
            "curve_points": 41,
            "linear_sensitivity_per_unit": 1.0,
            "interpretation": "descriptive_prognostic_association_not_causal",
            "plan_bound_adjustment_roster": {
                "authority": "plan_primary_model",
                "admissible_columns": ["age", "sex", "comorbidity_index", "severity_score_24h"],
                "admissible_categorical_columns": [],
                "sealed": False,
            },
        }
    )
    runtime = ScientificRuntimeAuthorities(trajectory=None, current_case=unsealed)
    sealed = runtime.seal_for_plan(plan).current_case
    assert sealed.required_adjustment_columns == ("age", "sex")
    bound, findings = runtime.bind_plan(plan)
    assert findings[0].detail["reason_code"] == "landmark_spline_host_compiled"
    assert findings[0].detail["adjustment_roster"] == ["age", "sex"]
    runtime.seal_for_plan(plan).validate_plan(bound)
    signed = sealed.governed_step(bound)
    assert signed.step_id == "adjusted_association"
    assert list(signed.expected_outputs) == list(sealed.plan_outputs)
    assert list(signed.inputs) == ["artifact:analysis_cohort", *sealed.required_columns]


# ---------------------------------------------------------------------------
# Descriptive exposure–outcome family (counts-only when repeated units are unresolved)
# ---------------------------------------------------------------------------

DESCRIPTIVE_LABELS = {
    "phenotype_flag": "Phenotype present in the first 24 h",
    "phenotype_flag=0": "Phenotype absent",
    "phenotype_flag=1": "Phenotype present",
    "death": "In-hospital death",
    "age": "Age at ICU admission (years)",
    "sex": "Patient sex",
    "score_first": "Chronic disease score (first value)",
    "readmit_flag": "ICU readmission indicator",
}


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


def _descriptive_payload(request, *, baseline_variables, adjustment_set=None):
    return {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": adjustment_set or [],
        "baseline_variables": baseline_variables,
        "reader_display_labels": [
            {"key": key, "value": DESCRIPTIVE_LABELS[key]}
            for key in [*request.required_reader_label_keys, *request.level_label_keys]
        ],
        "comparator_applications": [
            {
                "citation_key": key,
                "application": (
                    f"Compare this description with {key} on population, exposure definition, "
                    "time zero, and estimand without copying its design or claiming novelty."
                ),
            }
            for key in request.direct_comparator_literature_keys
        ],
        "roster_decision_note": "Descriptive family: no adjustment; baseline variables chosen from host-timed candidates.",
    }


def test_descriptive_family_request_offers_window_bound_baseline_variables_and_level_labels() -> None:
    from easyicu.research_agent.planning.family_spec import DESCRIPTIVE_FAMILY_ID

    context = _descriptive_context()
    types = candidate_analysis_types(context)
    assert types[0] == "descriptive_epidemiology"
    assert family_template_id_for_context(context, analysis_types=types) == DESCRIPTIVE_FAMILY_ID
    request = _request(context, cohort_mode="all_input_rows")
    assert request.family_id == DESCRIPTIVE_FAMILY_ID
    assert request.analysis_type == "descriptive_epidemiology"
    assert request.landmark_hours is None and request.observation_window_hours == 24.0
    assert request.counts_only is True
    assert request.level_label_keys == ["phenotype_flag=0", "phenotype_flag=1"]
    candidates = {item.name: item for item in request.adjustment_candidates}
    assert candidates["age"].selectable and candidates["age"].host_temporal_role == "baseline_static"
    assert candidates["score_first"].selectable
    assert candidates["score_first"].host_temporal_role == "at_or_before_time_zero"
    # A column with no window the host can place is offered only as audit context.
    assert candidates["readmit_flag"].selectable is False
    assert "phenotype_n" not in candidates and "death" not in candidates
    assert request.measurement_audit_columns == ["readmit_flag"]


def test_descriptive_family_compiles_the_reference_layout_and_fails_closed_on_models() -> None:
    from easyicu.research_agent.planning.family_spec import validate_family_plan_spec
    from easyicu.research_agent.planning.family_spec.contract import spec_from_mapping

    context = _descriptive_context()
    request = _request(context, cohort_mode="all_input_rows")
    llm, result = _run(
        context,
        [json.dumps(_descriptive_payload(request, baseline_variables=["age", "sex", "score_first"]))],
        required_primary_cohort_selection_mode="all_input_rows",
    )
    plan = result.output
    assert len(llm.calls) == 1
    assert plan.analysis_type == "descriptive_epidemiology"
    assert [step.step_id for step in plan.steps] == [
        "cohort_accounting",
        "baseline_context",
        "exposure_outcome_distribution",
        "measurement_audit",
    ]
    primary = plan.steps[2]
    assert primary.planned_analysis_role == "primary"
    assert primary.scientific_capability == "descriptive_exposure_outcome_distribution_v1"
    assert primary.model_requirements == []
    table_one = plan.steps[1].table_one_spec
    assert table_one is not None and table_one.group_by == "phenotype_flag"
    assert [item.name for item in table_one.variables] == ["age", "sex", "score_first"]
    assert plan.display_labels["phenotype_flag=1"] == "Phenotype present"
    selected = plan.design_selection.selected
    assert selected.design_id == "stay_level_descriptive"
    assert "confidence interval" not in selected.estimand.casefold()
    assert result.facts.prompt_metrics["planner_strategy"] == FAMILY_SPEC_STRATEGY

    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(
            spec_from_mapping(
                _descriptive_payload(
                    request,
                    baseline_variables=["age"],
                    adjustment_set=[
                        {
                            "name": "age",
                            "coding": "continuous",
                            "reference_level_index": None,
                            "clinical_rationale": "Age is a confounder fixed before admission.",
                        }
                    ],
                )
            ),
            request,
        )
    assert caught.value.reason_code == "family_spec_adjustment_not_applicable"
    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(
            spec_from_mapping(_descriptive_payload(request, baseline_variables=["readmit_flag"])), request
        )
    assert caught.value.reason_code == "family_spec_baseline_variable_unavailable"


# ---------------------------------------------------------------------------
# Cross-sectional phenotyping family
# ---------------------------------------------------------------------------

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


def test_phenotyping_family_offers_window_bound_features_and_compiles_the_reference_layout() -> None:
    from easyicu.research_agent.planning.family_spec import PHENOTYPING_FAMILY_ID, validate_family_plan_spec
    from easyicu.research_agent.planning.family_spec.contract import spec_from_mapping

    context = _phenotyping_context()
    types = candidate_analysis_types(context)
    assert types[0] == "trajectory_clustering"
    assert family_template_id_for_context(context, analysis_types=types) == PHENOTYPING_FAMILY_ID
    request = _request(context, cohort_mode=None)
    features = {item.name: item for item in request.feature_candidates}
    assert {name for name, item in features.items() if item.selectable} == {"hr_max", "lactate_max", "map_min", "score_first"}
    # Outcome lineage never becomes a fit feature; demographics are characterization, not features.
    assert "los_flag" not in features and "death" not in features and "age" not in features
    assert request.membership_candidates == ["phenotype_flag"]
    assert {item.name for item in request.adjustment_candidates if item.selectable} >= {"age", "sex"}

    llm, result = _run(
        context,
        [json.dumps(_phenotyping_payload(
            request, features=["hr_max", "lactate_max", "map_min"], baseline=["age", "sex"],
            membership="phenotype_flag",
        ))],
        required_primary_cohort_selection_mode=None,
    )
    plan = result.output
    assert len(llm.calls) == 1
    assert plan.analysis_type == "trajectory_clustering"
    assert [step.step_id for step in plan.steps] == [
        "cohort_accounting", "feature_quality_audit", "primary_cluster_solution",
        "cluster_number_selection", "cluster_stability", "cluster_characterization",
    ]
    primary = plan.steps[2]
    assert primary.scientific_action_id == "phenotyping.cluster_solution"
    assert primary.phenotyping_feature_columns == ["hr_max", "lactate_max", "map_min"]
    assert "artifact:analysis_cohort" in primary.inputs
    characterization = plan.steps[5]
    assert characterization.phenotype_comparison_spec is not None
    assert [v.name for v in characterization.phenotype_comparison_spec.variables][:3] == ["death", "age", "sex"]
    assert plan.cohort.selection_mode == "predicate_filtered"
    assert any(item.concept_id == "phenotype_flag" for item in plan.cohort.inclusion)

    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(
            spec_from_mapping(_phenotyping_payload(request, features=["hr_max"], baseline=[], membership=None)),
            request,
        )
    assert caught.value.reason_code == "family_spec_feature_roster_invalid"
    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(
            spec_from_mapping(
                _phenotyping_payload(request, features=["hr_max", "los_flag"], baseline=[], membership=None)
            ),
            request,
        )
    assert caught.value.reason_code == "family_spec_feature_unavailable"
    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(
            spec_from_mapping(
                _phenotyping_payload(request, features=["hr_max", "map_min"], baseline=[], membership="death")
            ),
            request,
        )
    assert caught.value.reason_code == "family_spec_membership_unavailable"


def test_phenotyping_template_robustness_is_the_host_owned_grid_and_stability() -> None:
    """The candidate-k grid and resampling stability are the playbook's two axes.

    They run under the published host phenotyping policy, so the reviewer sees
    prespecified robustness only while the host owner claims both steps.
    """

    from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review

    context = _phenotyping_context()
    request = _request(context, cohort_mode=None)
    _llm, result = _run(
        context,
        [json.dumps(_phenotyping_payload(
            request, features=["hr_max", "lactate_max", "map_min"], baseline=["age", "sex"],
            membership="phenotype_flag",
        ))],
        required_primary_cohort_selection_mode=None,
    )
    plan = result.output

    def codes(candidate):
        return {
            finding.code
            for finding in build_plan_scientific_review(
                context=context, plan=candidate, literature=None,
                figure_strategy=build_article_figure_strategy(context), runtime_authority=None,
            ).findings
        }

    assert "ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED" not in codes(plan)
    assert "ROBUSTNESS_AXES_TOO_NARROW" not in codes(plan)
    # The same step ids without the host action are agent-coded prose, not the
    # published design, and the reviewer is owed the finding again.
    unowned = plan.model_copy(
        update={
            "steps": [
                step.model_copy(update={"scientific_action_id": None})
                if step.step_id in {"cluster_number_selection", "cluster_stability"}
                else step
                for step in plan.steps
            ]
        }
    )
    assert "ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED" in codes(unowned)


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


def test_prediction_family_compiles_the_host_owned_reference_layout() -> None:
    from easyicu.research_agent.contracts.figure_plan import (
        STATIC_PREDICTION_FIGURE_PANELS,
        STATIC_PREDICTION_VALIDATION_FIGURE_PANELS,
    )
    from easyicu.research_agent.contracts.prediction_execution import static_prediction_model_columns
    from easyicu.research_agent.planning.family_spec import PREDICTION_FAMILY_ID, validate_family_plan_spec
    from easyicu.research_agent.planning.family_spec.contract import spec_from_mapping
    from easyicu.research_agent.planning.figure_plan_shaping import bind_deterministic_figure_panels
    from easyicu.research_agent.planning.scientific_review import planned_model_outcomes

    context = _prediction_context()
    types = candidate_analysis_types(context)
    assert types[0] == "prediction_model"
    assert family_template_id_for_context(context, analysis_types=types) == PREDICTION_FAMILY_ID
    request = _request(context, cohort_mode=None)
    assert request.exposure_kind == "none" and request.primary_exposure == ""
    selectable = {item.name for item in request.feature_candidates if item.selectable}
    # Demographics are legitimate predictors here; outcome lineage and identity never are.
    assert selectable >= {"age", "sex", "hr_max", "lactate_max", "map_min", "score_first"}
    assert not {"death", "los_flag", "stay_id"} & {item.name for item in request.feature_candidates}
    assert request.adjustment_candidates == []

    features = ["age", "sex", "hr_max", "lactate_max", "map_min"]
    llm, result = _run(
        context,
        [json.dumps(_prediction_payload(request, features=features))],
        required_primary_cohort_selection_mode=None,
    )
    plan = result.output
    assert len(llm.calls) == 1
    assert plan.analysis_type == "prediction_model"
    assert [step.step_id for step in plan.steps] == [
        "cohort_accounting", "baseline_context", "measurement_audit", "primary_performance",
        "calibration_metrics", "internal_validation", "clinical_utility", "visualization", "report",
    ]
    primary = plan.steps[3]
    assert primary.scientific_action_id == "prediction.discrimination_calibration"
    # The model-column prefix is outcome + predictors only: the host owner would
    # otherwise fit the row identity as a feature.
    assert static_prediction_model_columns(primary) == ("death", *features)
    assert primary.expected_outputs == ["table:prediction_scores", "table:model_performance"]
    assert not primary.model_requirements
    assert planned_model_outcomes(plan, context) == ("death",)
    for step, action in (
        (plan.steps[4], "prediction.calibration_metrics"),
        (plan.steps[5], "prediction.internal_validation"),
        (plan.steps[6], "prediction.decision_curve"),
    ):
        assert step.scientific_action_id == action
        assert "table:prediction_scores" in step.inputs

    shaped, findings = bind_deterministic_figure_panels(plan=plan)
    figure = next(step for step in shaped.steps if step.step_id == "visualization")
    assert set(figure.inputs) == {
        "table:prediction_scores", "table:model_performance", "table:validation",
        "table:calibration", "table:clinical_utility",
    }
    # The renderer exports two main surfaces, so the step declares two product
    # slots and the plan promises each surface's own roles.
    assert figure.expected_outputs == [
        "figure:visualization",
        "figure:visualization_validation_stability",
    ]
    assert [(panel.panel_id, panel.article_role, panel.chart_type) for panel in figure.figure_panels] == [
        (panel.panel_id, panel.article_role, panel.chart_type)
        for panel in (
            *STATIC_PREDICTION_FIGURE_PANELS,
            *STATIC_PREDICTION_VALIDATION_FIGURE_PANELS,
        )
    ]
    assert {finding.detail["reason"] for finding in findings} >= {
        "prediction_figure_clinical_utility_bound", "deterministic_figure_panels_bound",
    }

    # The playbook's "complete-case versus imputed workflow" is the one variant
    # the host owner executes: a complete-case refit of this exact roster.
    assert [
        (spec.spec_id, spec.axis, spec.missing_override["strategy"], set(spec.missing_override["variables"]))
        for spec in plan.robustness_specs
    ] == [("complete_case_model_roster", "missing", "complete_case", {"death", *features})]

    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(
            spec_from_mapping(_prediction_payload(request, features=["age", "death"])), request,
        )
    assert caught.value.reason_code == "family_spec_feature_unavailable"
    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(spec_from_mapping(_prediction_payload(request, features=["age"])), request)
    assert caught.value.reason_code == "family_spec_feature_roster_invalid"
    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(
            spec_from_mapping({
                **_prediction_payload(request, features=features),
                "adjustment_set": PLANNER_ROSTER[:1],
            }),
            request,
        )
    assert caught.value.reason_code == "family_spec_adjustment_not_applicable"


def _with_routine_measurements(context: ResearchContext, count: int) -> ResearchContext:
    extra = [
        ConceptDescriptor(
            name=f"lab_{index:02d}_max", description=f"routine laboratory value {index}",
            role=VariableRole.LAB, dtype="float64", source_concept=f"lab_{index:02d}",
            analysis_window="icu_admission[0,24]h",
        )
        for index in range(count)
    ]
    return context.model_copy(update={"variables": [*context.variables, *extra]})


def test_a_first_day_roster_at_the_design_bound_compiles_with_a_bounded_estimand() -> None:
    """A realistic first-day roster reaches the design, and the estimand stays one sentence.

    The design names at most its own bound of variables and the estimand once
    listed every label, so a first-day model with a realistic roster failed
    after the Planner had already been paid for its spec.
    """

    from easyicu.research_agent.planning.family_spec.contract import (
        MAX_FIT_FEATURES,
        design_field_max_length,
    )

    context = _with_routine_measurements(_prediction_context(), 30)
    request = _request(context, cohort_mode=None)
    features = [
        "age", "sex", "hr_max", "lactate_max", "map_min", *[f"lab_{i:02d}_max" for i in range(30)]
    ][:MAX_FIT_FEATURES]
    _llm, result = _run(
        context,
        [json.dumps(_prediction_payload(request, features=features))],
        required_primary_cohort_selection_mode=None,
    )

    selected = next(
        item for item in result.output.design_selection.candidates if item.disposition == "selected"
    )
    assert set(features) <= set(selected.required_variables)
    assert len(selected.estimand) <= design_field_max_length("estimand")
    assert f"{MAX_FIT_FEATURES} prespecified predictors" in selected.estimand
    assert "named in the plan" in selected.estimand


def test_a_roster_the_design_cannot_name_is_refused_before_it_compiles() -> None:
    """The bound is part of the spec contract, so the Planner is told before it answers."""

    from easyicu.research_agent.planning.family_spec.contract import MAX_FIT_FEATURES

    prediction = _with_routine_measurements(_prediction_context(), 30)
    request = _request(prediction, cohort_mode=None)
    too_many = [f"lab_{i:02d}_max" for i in range(MAX_FIT_FEATURES + 1)]
    # A schema violation is a ValueError the structured retry returns to the Planner.
    with pytest.raises(ValueError, match="at most 22 items"):
        parse_family_plan_spec(json.dumps(_prediction_payload(request, features=too_many)), request)

    phenotyping = _with_routine_measurements(_phenotyping_context(), 30)
    request = _request(phenotyping, cohort_mode=None)
    features = [f"lab_{i:02d}_max" for i in range(18)]
    baseline = ["age", "sex", "hr_max", "map_min"]
    with pytest.raises(FamilySpecError) as caught:
        parse_family_plan_spec(
            json.dumps(_phenotyping_payload(
                request, features=features, baseline=baseline, membership="phenotype_flag",
            )),
            request,
        )
    assert caught.value.reason_code == "family_spec_roster_exceeds_design"


def test_prediction_template_robustness_is_the_owner_executed_refit_and_decision_curve() -> None:
    """Two playbook axes, both executed by the host prediction owner.

    The complete-case refit of the exact roster and the decision curve over the
    owner's fixed threshold grid; narrow the refit's variable set and it is
    another analysis the owner would not run, so the reviewer is owed the
    finding again.
    """

    from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy
    from easyicu.research_agent.planning.scientific_review import (
        build_plan_scientific_review,
        remediation_route_for_finding,
    )

    context = _prediction_context()
    request = _request(context, cohort_mode=None)
    features = ["age", "sex", "hr_max", "lactate_max", "map_min"]
    _llm, result = _run(
        context,
        [json.dumps(_prediction_payload(request, features=features))],
        required_primary_cohort_selection_mode=None,
    )
    plan = result.output

    def review(candidate):
        return build_plan_scientific_review(
            context=context, plan=candidate, literature=None,
            figure_strategy=build_article_figure_strategy(context), runtime_authority=None,
        )

    robustness_codes = {"ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED", "ROBUSTNESS_AXES_TOO_NARROW"}
    reviewed = review(plan)
    assert not {finding.code for finding in reviewed.findings} & robustness_codes
    readiness = reviewed.facts["robustness_readiness"]
    assert readiness["status"] == "satisfied"
    assert readiness["executable_axes"] == ["decision_threshold", "missing"]
    assert reviewed.facts["sensitivity"]["owner_executed_plan_spec_ids"] == ["complete_case_model_roster"]
    narrowed = plan.model_copy(
        update={
            "robustness_specs": [
                type(spec).from_dict(
                    {
                        **spec.to_dict(),
                        "missing_override": {
                            **spec.missing_override,
                            "variables": ["death", "age"],
                        },
                    }
                )
                for spec in plan.robustness_specs
            ]
        }
    )
    narrowed_review = review(narrowed)
    narrowed_readiness = narrowed_review.facts["robustness_readiness"]
    assert narrowed_readiness["status"] == "blocked"
    assert narrowed_readiness["reason"] == "typed_sensitivity_authority_not_executable"
    assert narrowed_readiness["executable_axes"] == ["decision_threshold"]
    # The reviewer is owed a finding before execution, not only the run-level
    # panel's fail-closed verdict after it.
    assert narrowed_review.facts["sensitivity"]["protocol_only_plan_spec_ids"] == [
        "complete_case_model_roster"
    ]
    not_executable = [
        finding
        for finding in narrowed_review.findings
        if finding.code == "ROBUSTNESS_SPECS_NOT_EXECUTABLE"
    ]
    assert len(not_executable) == 1
    assert "complete_case_model_roster" in not_executable[0].message
    assert not_executable[0].severity == "major"
    assert remediation_route_for_finding(not_executable[0]) == "agent_plan_revision"
    assert narrowed_review.score < reviewed.score


SURVIVAL_CARD_SOURCES = ("grambsch_therneau_ph_1994", "royston_parmar_rmst_2011")


def test_logistic_landmark_plan_never_cites_a_survival_method_card() -> None:
    # A live KDIGO plan (landmark logistic) bound the proportional-hazards
    # diagnostics and the restricted mean survival time: their generic design
    # elements matched, but the plan fits no hazard model.
    context = _context(exact=False)
    allowed = (*ALLOWED_CITATIONS, *SURVIVAL_CARD_SOURCES)
    request = build_family_spec_request(
        context,
        analysis_types=candidate_analysis_types(context),
        variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=allowed,
        direct_comparator_literature_keys=DIRECT_COMPARATORS,
        comparison_literature_keys=DIRECT_COMPARATORS,
        required_primary_cohort_selection_mode="predicate_filtered",
    )
    llm = ScriptedMockLLMClient(
        [json.dumps(_spec_payload(request, adjustment_set=PLANNER_ROSTER))]
    )
    result = ProgressivePlannerAgent(llm).run_attempt(
        context,
        planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=allowed,
        direct_comparator_literature_keys=DIRECT_COMPARATORS,
        comparison_literature_keys=DIRECT_COMPARATORS,
        enforce_article_contract=True,
        article_contract_context=context,
        planning_contract_context="",
        required_primary_cohort_selection_mode="predicate_filtered",
    )
    cited = {
        key for step in result.facts.outline.steps for key in step.literature_citation_keys
    }
    assert "anderson_landmark_1983" in cited
    assert not cited & set(SURVIVAL_CARD_SOURCES)


def test_follow_up_unit_is_not_repeated_after_a_label_that_carries_it() -> None:
    context = _context(exact=False)
    request = _request(context)
    with_unit = _run(
        context, [json.dumps(_spec_payload(request, adjustment_set=PLANNER_ROSTER))]
    )[1].output
    window = with_unit.design_selection.selected.observation_window
    assert f"{LABELS['followup_time_hours']};" in window
    assert "(hours) (hours)" not in window
    payload = _spec_payload(request, adjustment_set=PLANNER_ROSTER)
    payload["reader_display_labels"] = [
        {**item, "value": "In-hospital follow-up time"}
        if item["key"] == "followup_time_hours"
        else item
        for item in payload["reader_display_labels"]
    ]
    without_unit = _run(context, [json.dumps(payload)])[1].output
    assert "In-hospital follow-up time (hours);" in (
        without_unit.design_selection.selected.observation_window
    )
