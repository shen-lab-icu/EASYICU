"""Boundary and owner-contract tests for deterministic requirement coverage."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from easyicu.clinical_contracts import load_clinical_contracts
from easyicu.concept.metadata_sidecar import TimeCoordinate
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.contracts.exposure_outcome_distribution import ExposureOutcomeDistributionSpec
from easyicu.research_agent.contracts.model_terms import (
    DataPreparationPlan,
    ImputationVariable,
    MissingDataPlan,
    PlannedModelRequirement,
    WindowAggregation,
    preparation_method,
)
from easyicu.research_agent.contracts.prediction_validation import (
    EXTERNAL_ACTION,
    EXTERNAL_OUTPUTS,
    FROZEN_MODEL_INPUT,
    ExternalValidationSpec,
    external_cohort_input,
    external_validation_method,
)
from easyicu.research_agent.planning.cohort_contract import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
)
from easyicu.research_agent.planning.requirement_typing import matcher
from easyicu.research_agent.planning.requirement_typing.contract import (
    AtomicPart,
    Binding,
    CohortPredicate,
    CoverageRecord,
    Delivery,
    Grading,
    Method,
    NotApplicable,
    Parameter,
    PartResult,
    RepeatStayPolicy,
    RepeatStayReporting,
    RequirementKey,
    SourceSpan,
    TimeZero,
    Unit,
    VerifiedGap,
    Window,
)
from easyicu.research_agent.planning.requirement_typing.facts import project_facts
from easyicu.research_agent.planning.requirement_typing.registry import (
    STUDY_POPULATION,
    registered_dictionary,
)
from easyicu.research_agent.planning.sensitivity_authority import PrespecifiedSensitivitySpec
from easyicu.research_agent.research_context.clinical_definition import ClinicalDefinitionReference
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
)

from .test_typed_question_requirement_regressions import (
    AUTHORITY,
    CLUSTER,
    DICTIONARY,
    DIALYSIS16,
    EXTERNAL,
    IMPUTATION,
    KDIGO_48,
    ICU,
    MAX_POTASSIUM,
    bind,
    fact,
    interval,
    key,
    point,
)


def test_matcher_can_only_read_the_typed_judgment_boundary() -> None:
    signature = inspect.signature(matcher.match)
    assert tuple(signature.parameters) == ("key", "authority", "dictionary", "facts", "gaps")
    assert [item.annotation for item in signature.parameters.values()] == [
        "RequirementKey | None", "FactAuthority", "DictionarySnapshot",
        "tuple[HostFact, ...]", "tuple[VerifiedGap, ...]",
    ]
    tree = ast.parse(Path(matcher.__file__).read_text(encoding="utf-8"))
    assert all(not isinstance(node, ast.Import) for node in ast.walk(tree))
    assert {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)} == {
        "__future__", "decimal", "fractions", "math", "contract",
    }
    assert not {"quote", "display_labels", "intent", "plan", "prior_verdict"} & {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }


def test_schema_rejects_missing_roles_mismatched_anchors_and_prose() -> None:
    payload = DIALYSIS16.model_dump(mode="json")
    for bad in (
        {**payload, "bindings": []},
        {**payload, "bindings": [payload["bindings"][-1]]},
        {**payload, "bindings": [payload["bindings"][0]]},
        {**payload, "bindings": [*payload["bindings"], payload["bindings"][0]]},
        {**payload, "time": {**payload["time"], "anchor": "hospital_admission"}},
        {**payload, "quote": "dialysis time zero"},
        {**payload, "facet": {"kind": "free_text", "description": "covered"}},
    ):
        with pytest.raises(ValidationError):
            RequirementKey.model_validate(bad)
    with pytest.raises(ValidationError):
        SourceSpan(start=1, end=1)
    with pytest.raises(ValidationError):
        key(Delivery(output_id="figures"), bind("k"))
    with pytest.raises(ValidationError):
        key(TimeZero(), STUDY_POPULATION)


def test_all_four_states_have_action_hints_and_bind_to_current_evidence() -> None:
    proof = fact(DIALYSIS16)
    gap = VerifiedGap(key=DIALYSIS16, provenance=proof.provenance, reason="concept_unavailable")
    expected = (
        ("covered", "none", DIALYSIS16, (proof,), ()),
        ("not_covered", "edit_plan", DIALYSIS16, (), ()),
        ("unknown", "confirm_requirements", None, (proof,), ()),
        ("capability_gap", "verified_alternative_or_question_revision", DIALYSIS16, (), (gap,)),
        ("unknown", "confirm_requirements", DIALYSIS16, (proof,), (gap,)),
    )
    results = []
    for index, (state, hint, asked, facts, gaps) in enumerate(expected):
        decision = matcher.match(asked, authority=AUTHORITY, dictionary=DICTIONARY, facts=facts, gaps=gaps)
        assert (decision.state, decision.next_action) == (state, hint)
        results.append(PartResult(
            part=AtomicPart(part_id=f"r{index}", source_span=SourceSpan(start=0, end=1), key=asked),
            decision=decision,
        ))
    record = CoverageRecord(authority=AUTHORITY, dictionary=DICTIONARY, parts=tuple(results))
    assert not record.complete
    assert CoverageRecord.model_validate_json(record.model_dump_json()) == record
    assert record.schema_version == "easyicu.question_requirement_coverage/2"
    assert CoverageRecord(authority=AUTHORITY, dictionary=DICTIONARY, parts=(results[0],)).complete
    with pytest.raises(ValidationError):
        CoverageRecord(authority=AUTHORITY, dictionary=DICTIONARY, parts=(results[0],)*2)

    for field, value in (
        ("plan_sha256", "1" * 63 + "0"),
        ("context_sha256", None),
        ("dictionary_version", "old/1"),
        ("coordinate_sha256", "3" * 64),
    ):
        expired = proof.provenance.model_copy(update={
            "authority": AUTHORITY.model_copy(update={field: value}),
        })
        assert matcher.match(
            DIALYSIS16, authority=AUTHORITY, dictionary=DICTIONARY,
            facts=(proof.model_copy(update={"provenance": expired}),),
            gaps=(gap.model_copy(update={"provenance": expired}),),
        ).state == "not_covered"


def test_matching_is_exact_across_units_sign_boundary_and_scope() -> None:
    asked = key(
        Window(purpose="aggregation", operation="max"), bind("k"),
        time=interval(-6, 0),
    )
    same = key(asked.facet, bind("k"), time=interval(-360, 0, unit="min"))
    nearly = key(asked.facet, bind("k"), time=interval("-360.000000000000000001", 0, unit="min"))
    inclusive_end = key(asked.facet, bind("k"), time=interval(-6, 0, end_inclusive=True))
    wrong_role = key(asked.facet, bind("k", "outcome"), time=asked.time)
    wider = key(asked.facet, bind("k"), time=interval(-12, 0))
    assert matcher.match(asked, authority=AUTHORITY, dictionary=DICTIONARY, facts=(fact(same),)).state == "covered"
    assert matcher.match(
        asked, authority=AUTHORITY, dictionary=DICTIONARY,
        facts=tuple(fact(actual) for actual in (nearly, inclusive_end, wrong_role, wider)),
    ).state == "not_covered"


def test_only_observation_windows_allow_same_anchor_containment() -> None:
    asked = key(Window(purpose="observation"), bind("k"), time=interval(0, 6, end_inclusive=True))
    boundary_open = key(asked.facet, bind("k"), time=interval(-1, 6))
    wider = key(asked.facet, bind("k"), time=interval(-1, 7))
    other_anchor = key(asked.facet, bind("k"), time=interval(-1, 7, "hospital_admission"))
    other_facet = key(Window(), bind("k"), time=interval(-1, 7))
    assert matcher.match(
        asked, authority=AUTHORITY, dictionary=DICTIONARY,
        facts=(fact(boundary_open), fact(other_anchor), fact(other_facet)),
    ).state == "not_covered"
    assert matcher.match(asked, authority=AUTHORITY, dictionary=DICTIONARY, facts=(fact(wider),)).state == "covered"


def test_event_relative_is_directional_and_cannot_substitute_non_event_roles() -> None:
    narrower = point(16, bind("acute_rrt_input"))
    assert matcher.match(narrower, authority=AUTHORITY, dictionary=DICTIONARY, facts=(fact(DIALYSIS16),)).state == "not_covered"
    for facet, time in (
        (Window(), interval(0, 16)),
        (Unit(code="mg"), NotApplicable()),
    ):
        asked = key(facet, bind("rrt"), time=time)
        actual = key(facet, bind("acute_rrt_input"), time=time)
        assert matcher.match(
            asked, authority=AUTHORITY, dictionary=DICTIONARY, facts=(fact(actual),),
        ).state == "not_covered"
    asked = point(16, bind("rrt", "outcome"))
    actual = point(16, bind("acute_rrt_input", "outcome"))
    assert matcher.match(
        asked, authority=AUTHORITY, dictionary=DICTIONARY, facts=(fact(actual),),
    ).state == "not_covered"


def test_method_parameters_bind_by_name_with_strict_values() -> None:
    asked = key(Method(action_id="multiple_imputation", parameters=(
        Parameter(name="m", value=20), Parameter(name="include_outcome", value=True),
    )), bind("k"))
    reordered = key(Method(
        action_id="multiple_imputation", parameters=asked.facet.parameters[::-1],
    ), bind("k"))
    bool_as_int = key(Method(action_id="multiple_imputation", parameters=(
        Parameter(name="m", value=20), Parameter(name="include_outcome", value=1),
    )), bind("k"))
    assert matcher.match(
        asked, authority=AUTHORITY, dictionary=DICTIONARY, facts=(fact(reordered),),
    ).state == "covered"
    assert matcher.match(
        asked, authority=AUTHORITY, dictionary=DICTIONARY, facts=(fact(bool_as_int),),
    ).state == "not_covered"


def _context(*, preferences=None, first=False, variables=None) -> ResearchContext:
    provenance = {}
    if first:
        provenance["first_icu_stay_restriction"] = {
            "schema_version": "easyicu.first_icu_stay_restriction/1", "coordinate_sha256": "b" * 64,
        }
    if preferences and preferences.data_constraints:
        provenance["patient_id_columns"] = ["patient_id"]
    return ResearchContext(
        research_question="Use peak potassium in the first six hours after ICU entry.",
        cohort=CohortDescriptor(
            cohort_name="test_cohort", database="miiv", n_stays=4,
            time_columns=["clock"], id_columns=["patient_id", "stay_id"], provenance=provenance,
        ),
        primary_exposure="k", target_outcome="death",
        variables=variables if variables is not None else [
            ConceptDescriptor(name="k", dtype="float64", unit="mmol/L"),
            ConceptDescriptor(name="death", dtype="bool"),
        ],
        user_preferences=preferences,
    )


def _model(*, exposure="k", dependence=None) -> PlannedModelRequirement:
    return PlannedModelRequirement(
        requirement_id="m1", outcome="death", outcome_type="binary",
        method_family="logistic_regression", exposure_source=exposure,
        analysis_role="primary", analysis_set="complete_case", covariates=[],
        dependence=dependence,
    )


def _model_plan(preparation=None, *, model=True) -> AnalysisPlan:
    method = "adjusted_association_models"
    return AnalysisPlan(
        research_question="Use peak potassium in the first six hours after ICU entry.",
        steps=[AnalysisStep(
            step_id="model", intent="fit a model", inputs=["k", "death", "clock", "stay_id"],
            method=preparation_method(method, preparation) if preparation is not None else method,
            expected_outputs=["table:adjusted_association_estimates"],
            planned_analysis_role="primary", model_requirements=[_model()] if model else [],
        )],
    )


def _evaluate(key, plan, *, context=None, clocks=()):
    dictionary = registered_dictionary(context.cohort.database if context else None)
    projected = project_facts(plan, context=context, dictionary=dictionary, time_coordinates=clocks)
    return matcher.match(
        key, authority=projected.authority, dictionary=dictionary, facts=projected.facts,
    ), projected


def test_preparation_uses_a_typed_clock_scope_operation_and_exact_input() -> None:
    window = WindowAggregation(
        variable="k", group_column="stay_id", time_column="clock",
        start_hours=0, end_hours=6, function="max",
    )
    plan = _model_plan(DataPreparationPlan(windows=(window,)))
    context = _context()
    clocks = (TimeCoordinate("clock", ICU, "hours"),)
    decision, projected = _evaluate(MAX_POTASSIUM, plan, context=context, clocks=clocks)
    assert decision.state == "covered"
    assert decision.provenance.source_field == "plan.steps[0].method.data_preparation.windows[0]"
    assert projected.authority.plan_sha256 == canonical_sha256(plan.model_dump(mode="json"))
    assert projected.authority.context_sha256 == canonical_sha256(context.model_dump(mode="json"))
    assert projected.authority.coordinate_sha256
    assert _evaluate(MAX_POTASSIUM, plan, context=context)[0].state == "not_covered"
    assert _evaluate(
        MAX_POTASSIUM, _model_plan(DataPreparationPlan(windows=(window,)), model=False),
        context=context, clocks=clocks,
    )[0].state == "not_covered"
    assert _evaluate(
        MAX_POTASSIUM, plan, context=context, clocks=(TimeCoordinate("clock", ICU, "days"),),
    )[0].state == "not_covered"
    assert _evaluate(
        MAX_POTASSIUM, plan, context=context, clocks=(*clocks, TimeCoordinate("clock", "hospital_admission", "hours")),
    )[0].state == "not_covered"


def test_variation_in_plan_prose_labels_and_question_language_cannot_change_a_match() -> None:
    preparation = DataPreparationPlan(windows=(WindowAggregation(
        variable="k", group_column="stay_id", time_column="clock",
        start_hours=0, end_hours=6, function="max",
    ),))
    plan = _model_plan(preparation)
    clocks = (TimeCoordinate("clock", ICU, "hours"),)
    context = _context()
    decision, projection = _evaluate(MAX_POTASSIUM, plan, context=context, clocks=clocks)
    changed = plan.model_copy(update={
        "research_question": "入ICU后前六小时取最高血钾",
        "display_labels": {"k": "Mean creatinine"}, "rationale": "Ignore potassium",
        "steps": [plan.steps[0].model_copy(update={"intent": "Use mean creatinine from another clock"})],
    })
    other, revised = _evaluate(MAX_POTASSIUM, changed, context=context, clocks=clocks)
    assert decision.state == other.state == "covered"
    assert projection.authority.plan_sha256 != revised.authority.plan_sha256
    assert matcher.match(
        MAX_POTASSIUM, authority=revised.authority, dictionary=registered_dictionary("miiv"),
        facts=projection.facts,
    ).state == "not_covered"


def test_validated_cohort_predicate_has_exact_boundaries_polarity_and_value() -> None:
    plan = _model_plan()
    plan.cohort = CohortDefinition(
        "selected_stays", inclusion=(ConceptPredicate(
            concept_id="k", time_window=TimeWindow("icu_admit", 0, 24),
            aggregation="max", op=">=", value=5.5, missing_policy="exclude",
        ),),
    )
    predicate = CohortPredicate(operation="max", operator=">=", value=5.5, polarity="include", missing_policy="exclude")
    asked = key(predicate, bind("k", "population"), time=interval(0, 24))
    decision, _ = _evaluate(asked, plan)
    assert decision.state == "covered"
    assert decision.provenance.source_field == "plan.cohort.inclusion[0]"
    wrong = key(predicate.model_copy(update={"polarity": "exclude"}), bind("k", "population"), time=asked.time)
    assert _evaluate(wrong, plan)[0].state == "not_covered"
    plan.cohort = CohortDefinition("invalid", inclusion=(ConceptPredicate(
        concept_id="k", time_window=TimeWindow("index_time", 0, 24),
        aggregation="max", op=">=", value=5.5,
    ),))
    assert _evaluate(asked, plan)[0].state == "not_covered"


def test_unit_and_clinical_definition_require_the_used_role_and_registry_version() -> None:
    owner = load_clinical_contracts()["kdigo_aki_2012"]
    context = _context(variables=[ConceptDescriptor(
        name="kdigo_creat", dtype="int64", clinical_definition=ClinicalDefinitionReference(
            contract_id=owner.contract_id, version=owner.version, source_id=owner.source_id,
            status=owner.status, definition=owner.definition, validation_status=owner.validation_status,
            canonical_definition=owner.canonical_definition,
        ),
    )])
    plan = _model_plan()
    plan.steps[0].model_requirements = [_model(exposure="kdigo_creat")]
    graded = key(Grading(contract_id=owner.contract_id, version=owner.version), bind("kdigo_creat"))
    assert _evaluate(graded, plan, context=context)[0].state == "not_covered"
    plan.steps[0].inputs[0] = "kdigo_creat"
    assert _evaluate(graded, plan, context=context)[0].state == "covered"
    assert _evaluate(
        key(graded.facet, bind("kdigo_creat", "outcome")), plan, context=context,
    )[0].state == "not_covered"
    assert _evaluate(
        key(graded.facet, bind("kdigo_creat"), time=interval(0, 48)), plan, context=context,
    )[0].state == "not_covered"
    context = _context()
    unit = key(Unit(code="mmol/L"), bind("k"))
    assert _evaluate(unit, _model_plan(), context=context)[0].state == "covered"
    assert _evaluate(unit, _model_plan(model=False), context=context)[0].state == "not_covered"


def test_sensitivity_landmark_needs_its_step_and_bound_outcome_clock() -> None:
    spec = PrespecifiedSensitivitySpec(
        spec_id="landmark-16", axis="timing", strategy="landmark", landmark_hours=16,
        event_time_variable="clock", execution_variables=("clock",),
    )
    context = _context(preferences=UserPreferences(sensitivity_specs=[spec]))
    plan = _model_plan()
    plan.steps.append(AnalysisStep(
        step_id="landmark", intent="prespecified landmark",
        method="landmark_analysis", inputs=["clock", "death"],
        planned_analysis_role="sensitivity", sensitivity_spec_ids=[spec.spec_id],
    ))
    asked = point(16, bind("death", "outcome"))
    typed = key(TimeZero(purpose="sensitivity_landmark"), bind("death", "outcome"), time=asked.time)
    clocks = (TimeCoordinate("clock", ICU, "hours"),)
    assert _evaluate(typed, plan, context=context, clocks=clocks)[0].state == "covered"
    assert _evaluate(asked, plan, context=context, clocks=clocks)[0].state == "not_covered"
    assert _evaluate(typed, plan, context=context)[0].state == "not_covered"
    plan.steps[1].sensitivity_spec_ids = []
    assert _evaluate(typed, plan, context=context, clocks=clocks)[0].state == "not_covered"


def test_imputation_external_validation_and_delivery_project_from_contracts() -> None:
    plan = _model_plan(DataPreparationPlan(missing_data=MissingDataPlan(
        strategy="multiple_imputation", variables=(ImputationVariable(name="k", method="pmm"),),
    )))
    spec = ExternalValidationSpec(
        development_database="miiv", validation_databases=["eicu"], frozen_model_sha256="a" * 64,
        predictor_concepts=["k"], outcome_concept="death",
    )
    plan.steps.append(AnalysisStep(
        step_id="validate", intent="validation", method=external_validation_method(spec),
        scientific_action_id=EXTERNAL_ACTION, planned_analysis_role="secondary",
        inputs=[FROZEN_MODEL_INPUT, external_cohort_input("eicu")], expected_outputs=list(EXTERNAL_OUTPUTS),
    ))
    context = _context()
    _, projected = _evaluate(MAX_POTASSIUM, plan, context=context)
    methods = {
        item.key.facet.action_id: item.key for item in projected.facts
        if isinstance(item.key.facet, Method) and len(item.key.facet.parameters) > 2
    }
    imputation, validation = methods["multiple_imputation"], methods[EXTERNAL_ACTION]
    assert imputation.bindings == (bind("k"),)
    assert {item.name for item in imputation.facet.parameters} == {
        "m", "iterations", "donors", "seed", "auxiliaries", "impute/k",
    }
    assert set(validation.bindings) == {bind("death", "outcome"), bind("k", "covariate")}
    assert _evaluate(key(Delivery(output_id="figures"), STUDY_POPULATION), plan, context=context)[0].state == "covered"
    plan.steps[1].expected_outputs = ["table:external_validation"]
    assert _evaluate(validation, plan, context=context)[0].state == "not_covered"


def test_first_stay_and_clustering_have_separate_policy_and_reporting_facts() -> None:
    from easyicu.research_agent.planning.dependence_authority import context_dependence_authority

    first = key(RepeatStayPolicy(policy="first_icu_stay_per_patient"), STUDY_POPULATION)
    clustered = key(RepeatStayPolicy(policy="retain_each_stay_cluster_by_patient"), STUDY_POPULATION)
    report = key(RepeatStayReporting(), STUDY_POPULATION)
    plan = _model_plan()
    assert _evaluate(first, plan, context=_context(first=True))[0].state == "covered"
    assert _evaluate(report, plan, context=_context(first=True))[0].state == "covered"
    assert _evaluate(clustered, plan, context=_context(first=True))[0].state == "not_covered"
    context = _context(preferences=UserPreferences(data_constraints=json.dumps({"analysis_design": {
        "analysis_unit": "icu_stay", "cluster_unit": "patient", "variance_estimator": "cluster_robust",
    }})))
    assert _evaluate(report, plan, context=context)[0].state == "not_covered"
    plan.steps[0].model_requirements = [_model(dependence=context_dependence_authority(context))]
    assert _evaluate(clustered, plan, context=context)[0].state == "covered"
    assert _evaluate(report, plan, context=context)[0].state == "covered"
    assert _evaluate(first, plan, context=context)[0].state == "not_covered"


def test_grouping_needs_the_exact_typed_output_and_scopes() -> None:
    spec = ExposureOutcomeDistributionSpec(
        exposure="k", exposure_levels=[0, 1], outcome="death", outcome_levels=[0, 1],
        outcome_positive_value=1, level_match_policy="exact_typed",
        denominator_policy="all_declared_rows", missing_outcome_policy="fail_closed", confidence_level=0.95,
    )
    plan = _model_plan()
    plan.steps.append(AnalysisStep(
        step_id="grouped_distribution", intent="Report counts by group",
        inputs=["table:analysis_cohort", "k", "death"], method="distribution",
        expected_outputs=["table:exposure_outcome_distribution"],
        exposure_outcome_distribution_spec=spec,
    ))
    facet = Method(action_id="descriptive.exposure_outcome_distribution", parameters=(
        Parameter(name="contract_sha256", value=canonical_sha256(spec.model_dump(mode="json"))),
    ))
    asked = key(facet, bind("k"), bind("death", "outcome"))
    assert _evaluate(asked, plan)[0].state == "covered"
    assert _evaluate(key(facet, bind("k", "covariate"), bind("death", "outcome")), plan)[0].state == "not_covered"
    plan.steps[-1].expected_outputs = ["table:generic"]
    assert _evaluate(asked, plan)[0].state == "not_covered"


def test_derived_lineage_does_not_supply_a_concept_binding() -> None:
    plan = _model_plan()
    plan.steps[0].model_requirements = [_model(exposure="calculated_ratio")]
    context = _context(variables=[ConceptDescriptor(
        name="calculated_ratio", dtype="float64", unit="mmol/L",
        source_concept="k", derived_from_concepts=["k", "crea"],
    )])
    assert _evaluate(key(Unit(code="mmol/L"), bind("k")), plan, context=context)[0].state == "not_covered"


@pytest.mark.parametrize("asked", ["imputation", "external"])
def test_question_method_keys_match_applied_plan_contracts(asked) -> None:
    plan = _model_plan(DataPreparationPlan(missing_data=MissingDataPlan(
        strategy="multiple_imputation", m=20,
        variables=(ImputationVariable(name="lact", method="pmm"), ImputationVariable(name="k", method="pmm")),
    )))
    plan.steps[0].inputs.append("lact")
    plan.steps[0].model_requirements = [_model().model_copy(update={"covariates": ["lact"]})]
    spec = ExternalValidationSpec(
        development_database="miiv", validation_databases=["eicu"], frozen_model_sha256="a" * 64,
        predictor_concepts=["lact"], outcome_concept="death", recalibration="none",
    )
    plan.steps.append(AnalysisStep(
        step_id="validate", intent="validation", method=external_validation_method(spec),
        scientific_action_id=EXTERNAL_ACTION, planned_analysis_role="secondary",
        inputs=[FROZEN_MODEL_INPUT, external_cohort_input("eicu")], expected_outputs=list(EXTERNAL_OUTPUTS),
    ))
    request = {"imputation": IMPUTATION, "external": EXTERNAL}[asked]
    decision, _ = _evaluate(request, plan, context=_context())
    assert decision.state == "covered"
    if asked == "imputation":
        assert decision.provenance.source_field == "plan.steps[0].method.data_preparation.missing_data"
        incorrect = key(Method(action_id="multiple_imputation", parameters=(Parameter(name="m", value=5),)),
                        bind("lact", "covariate"))
        assert _evaluate(incorrect, plan, context=_context())[0].state == "not_covered"
        plan.steps[0].method = "adjusted_association_models"
    else:
        assert decision.provenance.source_field == "plan.steps[1].method.external_validation_spec"
        spec = spec.model_copy(update={"predictor_concepts": ["lact", "k"]})
        plan.steps[1].method = external_validation_method(spec)
        assert _evaluate(request, plan, context=_context())[0].state == "not_covered"
        plan.steps.pop()
    assert _evaluate(request, plan, context=_context())[0].state == "not_covered"


def test_question_kdigo_48_key_requires_applied_staging_and_exact_window() -> None:
    owner = load_clinical_contracts()["kdigo_aki_2012"]
    context = _context(variables=[ConceptDescriptor(
        name="kdigo_creat", dtype="int64", clinical_definition=ClinicalDefinitionReference(
            contract_id=owner.contract_id, version=owner.version, source_id=owner.source_id,
            status=owner.status, definition=owner.definition, validation_status=owner.validation_status,
            canonical_definition=owner.canonical_definition,
        ),
    )])
    window = WindowAggregation(
        variable="kdigo_creat", group_column="stay_id", time_column="clock",
        start_hours=0, end_hours=48, function="max",
    )
    clocks = (TimeCoordinate("clock", ICU, "hours"),)

    def evaluate(applied_window, context=context, clocks=clocks):
        plan = _model_plan(DataPreparationPlan(windows=(applied_window,)) if applied_window else None)
        plan.steps[0].inputs[0] = "kdigo_creat"
        plan.steps[0].model_requirements = [_model(exposure="kdigo_creat")]
        return _evaluate(KDIGO_48, plan, context=context, clocks=clocks)[0].state

    assert evaluate(window) == "covered"
    assert evaluate(None) == "not_covered"
    assert evaluate(window, context=_context()) == "not_covered"
    assert evaluate(window, clocks=(TimeCoordinate("clock", "hospital_admission", "hours"),)) == "not_covered"
    assert evaluate(window.model_copy(update={"end_hours": 168})) == "not_covered"
    assert evaluate(window.model_copy(update={"function": "mean"})) == "not_covered"


def test_imputation_only_auxiliaries_resolve_to_sorted_concepts() -> None:
    plan = _model_plan(DataPreparationPlan(missing_data=MissingDataPlan(
        strategy="multiple_imputation", variables=(ImputationVariable(name="k", method="pmm"),),
        auxiliaries=("sex", "age_years"),
    )))
    plan.steps[0].inputs.extend(("sex", "age_years"))
    context = _context()
    context.variables.append(ConceptDescriptor(name="age_years", dtype="float64", source_concept="age"))
    _, projected = _evaluate(MAX_POTASSIUM, plan, context=context)
    [detail] = [
        item.key for item in projected.facts if isinstance(item.key.facet, Method)
        and item.key.facet.action_id == "multiple_imputation"
        and any(param.name == "auxiliaries" for param in item.key.facet.parameters)
    ]
    assert detail.bindings == (bind("k"),)
    assert Parameter(name="auxiliaries", value=("age", "sex")) in detail.facet.parameters


def test_repeated_stay_policy_does_not_credit_only_the_last_primary() -> None:
    from easyicu.research_agent.planning.dependence_authority import context_dependence_authority

    context = _context(preferences=UserPreferences(data_constraints=json.dumps({"analysis_design": {
        "analysis_unit": "icu_stay", "cluster_unit": "patient", "variance_estimator": "cluster_robust",
    }})))
    plan = _model_plan()
    second = plan.steps[0].model_copy(update={
        "step_id": "second",
        "model_requirements": [_model(dependence=context_dependence_authority(context))],
    })
    plan.steps.append(second)
    for requirements in (second.model_requirements, []):
        second.model_requirements = requirements
        assert _evaluate(CLUSTER, plan, context=context)[0].state == "not_covered"


def test_separate_binary_model_requirements_project_their_own_pairs() -> None:
    plan = _model_plan()
    plan.steps[0].inputs.extend(("mg", "kdigo_aki"))
    plan.steps[0].model_requirements.append(_model(exposure="mg").model_copy(update={
        "requirement_id": "m2", "outcome": "kdigo_aki",
    }))
    facet = Method(action_id="binary_effect_measure", parameters=(Parameter(name="measure", value="odds_ratio"),))
    for index, (exposure, outcome) in enumerate((("k", "death"), ("mg", "kdigo_aki"))):
        decision, _ = _evaluate(key(facet, bind(exposure), bind(outcome, "outcome")), plan, context=_context())
        assert decision.state == "covered"
        assert decision.provenance.source_field == f"plan.steps[0].model_requirements[{index}].method_family"
    assert _evaluate(key(facet, bind("mg"), bind("death", "outcome")), plan)[0].state == "not_covered"


@pytest.mark.parametrize("variable", [
    ConceptDescriptor(name="k", dtype="float64", unit="mmol/L", source_concept="k",
                      derived_from_concepts=["k", "crea"]),
    ConceptDescriptor(name="k", dtype="float64", unit="mmol/L", source_concept="na"),
], ids=["derived", "renamed"])
def test_registered_column_name_cannot_mask_conflicting_source_identity(variable) -> None:
    asked = key(Unit(code="mmol/L"), bind("k"))
    assert _evaluate(asked, _model_plan(), context=_context(variables=[variable]))[0].state == "not_covered"


def test_aliases_that_repeat_one_imputed_concept_do_not_abort_projection() -> None:
    context = _context()
    context.variables.append(ConceptDescriptor(name="k_alias", dtype="float64", source_concept="k"))
    plan = _model_plan(DataPreparationPlan(missing_data=MissingDataPlan(
        strategy="multiple_imputation",
        variables=(ImputationVariable(name="k", method="pmm"), ImputationVariable(name="k_alias", method="pmm")),
    )))
    plan.steps[0].inputs.append("k_alias")
    plan.steps[0].model_requirements = [_model().model_copy(update={"covariates": ["k_alias"]})]
    delivery = key(Delivery(output_id="figures"), STUDY_POPULATION)
    decision, projected = _evaluate(delivery, plan, context=context)
    assert decision.state == "covered"
    assert not any(
        isinstance(item.key.facet, Method) and item.key.facet.action_id == "multiple_imputation"
        for item in projected.facts
    )


@pytest.mark.parametrize("change", [{"function": "unsupported"}, {"end_hours": 0}])
def test_invalid_aggregation_window_is_refused_before_typed_fact_construction(change) -> None:
    window = WindowAggregation(
        variable="k", group_column="stay_id", time_column="clock",
        start_hours=0, end_hours=6, function="max",
    )
    plan = _model_plan()
    plan.steps[0].method = plan.steps[0].method + " with easyicu.data_preparation/1:" + json.dumps({
        "windows": [{**window.model_dump(), **change}],
    })
    delivery = key(Delivery(output_id="figures"), STUDY_POPULATION)
    decision, projected = _evaluate(
        delivery, plan, context=_context(), clocks=(TimeCoordinate("clock", ICU, "hours"),),
    )
    assert decision.state == "covered"
    assert not any(isinstance(item.key.facet, Window) for item in projected.facts)


@pytest.mark.parametrize("hours", [float("inf"), float("-inf"), float("nan")])
def test_sensitivity_schema_rejects_nonfinite_landmark_before_projection(hours) -> None:
    with pytest.raises(ValidationError, match="landmark_hours"):
        PrespecifiedSensitivitySpec(
            spec_id="invalid", axis="timing", strategy="landmark", landmark_hours=hours,
        )
