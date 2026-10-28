"""Family drafts promise a count-only cohort figure, with a traceable producer."""

import json
from dataclasses import replace

import pytest

from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
    candidate_analysis_types,
    select_progressive_variables,
)
from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.contracts.trajectory_design import (
    load_trajectory_design,
    sealed_trajectory_authority_body,
)
from easyicu.research_agent.execution.runners.cohort_flow_figure_executor import (
    cohort_flow_figure_executor_declares_step,
    cohort_flow_figure_executor_owns_step,
)
from easyicu.research_agent.execution.owner_declaration import _declared_host_figure, require_host_rendered_plan
from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
from easyicu.research_agent.planning.family_spec import (
    build_baseline_association_skeleton,
    build_descriptive_skeleton,
    build_family_spec_request,
    build_fixed_window_trajectory_skeleton,
    build_landmark_association_skeleton,
    build_landmark_continuous_survival_skeleton,
    build_landmark_survival_skeleton,
    build_phenotyping_skeleton,
    build_prediction_skeleton,
    build_source_feasibility_skeleton,
)
from easyicu.research_agent.planning.family_spec.contract import (
    SOURCE_FEASIBILITY_FAMILY_ID,
    SealedFeasibilityCoordinates,
    spec_from_mapping,
)
from easyicu.research_agent.contracts.claim_ceiling import first_result_instructions
from easyicu.research_agent.contracts.figure_plan import DATA_QUALITY_FIGURE_PANELS
from easyicu.research_agent.planning.figure_plan_shaping import (
    apply_runtime_bound_figure_contracts,
    ensure_cohort_accounting_figure_step,
)
from easyicu.research_agent.planning.progressive_contract import ProgressiveOutputIntent, ProgressiveProductRef
from easyicu.research_agent.planning.progressive_module_ids import progressive_module_ids_for_analysis_types
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ArtifactConsumptionContract,
    ResearchContext,
    UserPreferences,
)
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    build_trajectory_scientific_runtime_authority,
)
from tests.research_agent.planning.family_spec_fixtures import (
    PLANNER_ROSTER,
    _context,
    _descriptive_context,
    _descriptive_payload,
    _phenotyping_context,
    _phenotyping_payload,
    _prediction_context,
    _prediction_payload,
    _request,
    _run,
    _spec_payload,
)
from tests.support.baseline_association import baseline_context, baseline_plan_attempt, baseline_runtime
from tests.support.continuous_survival import (
    continuous_authority_body,
    continuous_survival_context,
    continuous_survival_request,
    continuous_survival_spec,
    proposed_continuous_survival_plan,
)
from tests.support.survival_proposal import AGE, survival_context, survival_request, survival_spec
from tests.support.survival_sealed import sealed_draft, sealed_survival


def _baseline():
    context = baseline_context("sex", covariates=("age",))
    request = build_family_spec_request(
        context, analysis_types=("association_study",),
        variable_roster=[v.name for v in context.variables],
        allowed_literature_citation_keys=(), required_primary_cohort_selection_mode="all_input_rows",
    )
    payload = {
        "schema_version": "easyicu.family_plan_spec/1", "family_id": request.family_id,
        "request_sha256": request.request_sha256, "adjustment_set": [],
        "reader_display_labels": [
            {"key": key, "value": "Synthetic " + key} for key in request.required_reader_label_keys
        ],
        "comparator_applications": [], "roster_decision_note": "Use the stated baseline roster.",
    }
    return build_baseline_association_skeleton(request, spec_from_mapping(payload))


def _landmark():
    context = _context(exact=False).model_copy(update={
        "research_question": "For synthetic stays, describe the landmark association and outcome."
    })
    request = _request(context)
    return build_landmark_association_skeleton(
        request, spec_from_mapping(_spec_payload(request, adjustment_set=PLANNER_ROSTER))
    )


def _survival(continuous=False):
    if continuous:
        context = continuous_survival_context()
        request = continuous_survival_request(context)
        return build_landmark_continuous_survival_skeleton(
            request, spec_from_mapping(continuous_survival_spec(request, []))
        )
    context = survival_context()
    request = survival_request(context)
    return build_landmark_survival_skeleton(request, spec_from_mapping(survival_spec(request, [AGE])))


def _phenotyping():
    request = _request(_phenotyping_context(), cohort_mode=None)
    payload = _phenotyping_payload(request, features=["hr_max", "lactate_max", "map_min"],
                                   baseline=["age"], membership=None)
    return build_phenotyping_skeleton(request, spec_from_mapping(payload))


def _prediction():
    request = _request(_prediction_context(), cohort_mode=None)
    payload = _prediction_payload(request, features=["age", "sex", "hr_max", "lactate_max", "map_min"])
    return build_prediction_skeleton(request, spec_from_mapping(payload))


def _descriptive():
    request = _request(_descriptive_context(), cohort_mode="all_input_rows")
    payload = _descriptive_payload(request, baseline_variables=["age", "sex"])
    return build_descriptive_skeleton(request, spec_from_mapping(payload))


def _trajectory(*, parts=False):
    coordinates = ("hr", "map", "lact")
    design = load_trajectory_design({"coordinate_concepts": list(coordinates),
                                     "descriptive_only_concepts": [], "window_end_hours": 24,
                                     "grid_width_hours": 4})
    authority = build_trajectory_scientific_runtime_authority(
        sealed_trajectory_authority_body(design, protocol_content_sha256="f" * 64)
    )
    authorities = ScientificRuntimeAuthorities(trajectory=authority, current_case=None)
    disclosure = authorities.planning_contract_context()
    context = ResearchContext(
        research_question="Do synthetic measurements over the first day form trajectory classes?",
        cohort=CohortDescriptor(cohort_name="synthetic", database="synthetic", n_stays=0,
                                id_columns=["stay_id"]),
        variables=[
            ConceptDescriptor(name="stay_id", role="id", dtype="int64"),
            *(ConceptDescriptor(name=name, description=f"Synthetic {name}", role="vital", dtype="float64",
                                analysis_window="icu_admission[0,24]h") for name in coordinates),
        ],
        user_preferences=UserPreferences(inferred_analysis_family="trajectory_clustering"),
    )
    request = build_family_spec_request(
        context, analysis_types=candidate_analysis_types(context),
        variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=("strobe_2007", "record_2015"),
        required_primary_cohort_selection_mode="all_input_rows", planning_contract_context=disclosure,
    )
    payload = {
        "schema_version": "easyicu.family_plan_spec/1", "family_id": request.family_id,
        "request_sha256": request.request_sha256, "adjustment_set": [],
        "reader_display_labels": [{"key": key, "value": "Synthetic " + key}
                                  for key in request.required_reader_label_keys],
        "comparator_applications": [], "roster_decision_note": "Use the stated trajectory coordinates.",
    }
    if parts:
        return context, payload, disclosure, authorities
    return build_fixed_window_trajectory_skeleton(request, spec_from_mapping(payload))


@pytest.mark.parametrize("build", [
    _baseline, _landmark, _survival, lambda: _survival(True),
    _prediction, _phenotyping, _trajectory, _descriptive,
], ids=["baseline", "landmark", "survival", "continuous_survival", "prediction",
        "phenotyping", "trajectory", "descriptive"])
def test_every_cohort_family_declares_a_flow_figure_with_its_aggregate_source(build):
    draft = build()
    cohort = next(s for s in draft.outline.steps
                  if s.module_id == "cohort_definition" and s.population_scope != "study_cohort")
    matches = [item for item in draft.materializations
               if any(o.product_id == "figure:cohort_flow" for o in item.step.outputs)]
    assert len(matches) == 1
    materialization = matches[0]
    step = materialization.step
    outline = next(s for s in draft.outline.steps if s.step_id == step.step_id)
    assert step.module_id == outline.module_id == "visualization"
    assert step.depends_on == outline.depends_on
    assert cohort.step_id in step.depends_on
    assert not step.raw_inputs
    assert [(r.producer_step_id, r.product_id) for r in step.product_inputs] == [
        (cohort.step_id, "table:cohort_flow")
    ]
    assert [(o.product_id, o.semantic_role) for o in step.outputs] == [("figure:cohort_flow", "figure")]
    assert materialization.outline_step_sha256 == canonical_sha256(outline.model_dump(mode="json"))
    assert draft.foundation.outline_sha256 == canonical_sha256(draft.outline.model_dump(mode="json"))


def test_draft_without_cohort_is_unchanged():
    draft = _descriptive()
    outline = draft.outline.model_copy(update={"steps": [
        step for step in draft.outline.steps if step.module_id not in {"cohort_definition", "visualization"}
    ]})
    foundation = draft.foundation.model_copy(update={"outline_sha256": canonical_sha256(outline.model_dump(mode="json"))})
    materials = tuple(item for item in draft.materializations
                      if item.step.module_id not in {"cohort_definition", "visualization"})
    other_family = replace(draft, outline=outline, foundation=foundation, materializations=materials)
    assert other_family.outline == outline
    assert other_family.foundation == foundation
    assert other_family.materializations == materials


def test_fail_closed_source_decision_does_not_promise_an_unproduced_cohort_figure():
    # This family projects to one signed decision step, which emits no cohort ledger.
    request = _request(_context()).model_copy(update={
        "family_id": SOURCE_FEASIBILITY_FAMILY_ID,
        "sealed_feasibility": SealedFeasibilityCoordinates(
            sealed_owner="signed_source_feasibility_fail_closed", plan_intent="Record the synthetic source decision.",
            plan_outputs=["table:decision", "log:decision"], source="synthetic source",
            audited_window_hours=(0, 24), decision="fail_closed", reason_code="SOURCE_UNKNOWN",
            forbidden_plan_tokens=["effect_estimate"],
        ),
    })
    payload = _spec_payload(request)
    draft = build_source_feasibility_skeleton(request, spec_from_mapping(payload))
    assert draft.cohort_flow_product_unavailable_reason == "signed_source_decision"
    assert [step.step_id for step in draft.outline.steps] == ["cohort_accounting", "source_feasibility", "report"]
    assert all("figure:cohort_flow" not in [out.product_id for out in item.step.outputs]
               for item in draft.materializations)


def test_study_population_root_does_not_replace_the_analysis_cohort_producer():
    draft = _landmark()
    cohort = next(s for s in draft.outline.steps if s.step_id == "cohort_definition")
    material = next(item for item in draft.materializations if item.step.step_id == cohort.step_id)
    root = cohort.model_copy(update={"step_id": "study_population", "population_scope": "study_cohort"})
    root_step = material.step.model_copy(update={"step_id": "study_population", "population_scope": "study_cohort"})
    root_material = material.model_copy(update={"step": root_step})
    outline = draft.outline.model_copy(update={"steps": [root, *[
        s for s in draft.outline.steps if s.step_id != "family_cohort_flow_figure"
    ]]})
    materials = (root_material, *(item for item in draft.materializations
                                  if item.step.step_id != "family_cohort_flow_figure"))
    revised = replace(draft, outline=outline, materializations=materials)
    figure = next(item.step for item in revised.materializations if item.step.step_id == "family_cohort_flow_figure")
    assert figure.depends_on == ["cohort_definition"]
    assert figure.product_inputs[0].producer_step_id == "cohort_definition"


def test_adding_the_figure_is_idempotent():
    draft = _landmark()
    assert replace(draft) == draft
    assert sum(item.step.step_id == "family_cohort_flow_figure" for item in draft.materializations) == 1


def test_ambiguous_analysis_cohort_owners_are_refused():
    draft = _descriptive()
    owner = next(step for step in draft.outline.steps if step.module_id == "cohort_definition")
    duplicate = owner.model_copy(update={"step_id": "second_cohort"})
    outline = draft.outline.model_copy(update={"steps": [duplicate, *[
        s for s in draft.outline.steps if s.step_id != "family_cohort_flow_figure"
    ]]})
    materials = tuple(item for item in draft.materializations if item.step.step_id != "family_cohort_flow_figure")
    with pytest.raises(ValueError, match="exactly one analysis cohort"):
        replace(draft, outline=outline, materializations=materials)


def test_a_reserved_figure_step_id_cannot_be_reused():
    draft = _descriptive()
    materials = tuple(item for item in draft.materializations if item.step.step_id != "family_cohort_flow_figure")
    with pytest.raises(ValueError, match="step id is already in use"):
        replace(draft, materializations=materials)


@pytest.mark.parametrize("change", ["id", "module", "role", "action", "parents", "raw", "producer", "source", "output"])
def test_descriptive_exception_accepts_only_the_exact_family_renderer(change):
    draft = _descriptive()
    step = next(item.step for item in draft.materializations if item.step.step_id == "family_cohort_flow_figure")
    def modules(candidate):
        return progressive_module_ids_for_analysis_types(("descriptive_epidemiology",), step=candidate)
    assert "visualization" not in progressive_module_ids_for_analysis_types(("descriptive_epidemiology",))
    assert "visualization" in modules(step)
    updates = {
        "id": {"step_id": "unrelated"},
        "module": {"module_id": "report"},
        "role": {"planned_analysis_role": "primary"},
        "action": {"scientific_action_id": "descriptive.exposure_outcome_distribution"},
        "parents": {"depends_on": [*step.depends_on, "other"]},
        "raw": {"raw_inputs": ["stay_id"]},
        "producer": {"product_inputs": [ProgressiveProductRef(producer_step_id="other", product_id="table:cohort_flow")]},
        "source": {"product_inputs": [ProgressiveProductRef(producer_step_id=step.depends_on[0], product_id="table:other")]},
        "output": {"outputs": [ProgressiveOutputIntent(product_id="figure:other", semantic_role="figure")]},
    }
    assert "visualization" not in modules(step.model_copy(update=updates[change]))


def test_compiled_flow_figure_is_owned_by_the_count_only_renderer():
    result, _ = baseline_plan_attempt(baseline_context("sex", covariates=("age",)))
    step = next(s for s in result.output.steps if "figure:cohort_flow" in s.expected_outputs)
    assert step.inputs == ["table:cohort_flow"]
    assert cohort_flow_figure_executor_declares_step(step)
    assert [(p.chart_type, p.source_products, p.figure_output) for p in step.figure_panels] == [
        ("cohort_flow", ["table:cohort_flow"], "figure:cohort_flow")
    ]
    binding = {
        "declared_kind": "table", "evidence_kind": "table", "product": "cohort_flow",
        "product_contract": {"columns": ["step_order", "predicate_kind", "n_before", "n_excluded", "n_remaining"]},
        "consumption_contract": {"mode": "all_rows"},
    }
    assert cohort_flow_figure_executor_owns_step(step, resolved_bindings={"table:cohort_flow": binding})


def test_descriptive_family_compiles_the_figure():
    context = _descriptive_context()
    request = _request(context, cohort_mode="all_input_rows")
    _, result = _run(context, [json.dumps(_descriptive_payload(request, baseline_variables=["age", "sex"]))],
                     required_primary_cohort_selection_mode="all_input_rows")
    step = next(s for s in result.output.steps if "figure:cohort_flow" in s.expected_outputs)
    assert step.inputs == ["table:cohort_flow"]
    assert cohort_flow_figure_executor_declares_step(step)


@pytest.mark.parametrize("first_result", [False, True])
def test_signed_baseline_accepts_only_the_count_figure_as_a_companion(first_result):
    context = baseline_context("sex", covariates=("age",))
    attempt, _ = baseline_plan_attempt(
        context, planning_contract_context=first_result_instructions() if first_result else "",
    )
    runtime = baseline_runtime(context, attempt.output).seal_for_plan(attempt.output)
    bound, _ = runtime.bind_plan(attempt.output)
    figure = next(s for s in bound.steps if "figure:cohort_flow" in s.expected_outputs)
    assert cohort_flow_figure_executor_declares_step(figure)
    tampered = figure.model_copy(update={"inputs": ["table:cohort_flow", "stay_id"]})
    corrupted = bound.model_copy(update={"steps": [tampered if s is figure else s for s in bound.steps]})
    with pytest.raises(ValueError, match="step roster"):
        runtime.validate_plan(corrupted)
    if first_result:
        data_sources = ("table:measurement_audit_missingness", "table:measurement_audit_process")
        extra = figure.model_copy(update={
            "step_id": "data_quality_figure", "inputs": list(data_sources),
            "expected_outputs": ["figure:data_quality"],
            "input_consumption_contracts": [ArtifactConsumptionContract(input_key=key, mode="all_rows")
                                            for key in data_sources],
            "figure_panels": [panel.model_copy(update={"source_products": (source,)}).bind(
                figure_output="figure:data_quality"
            ) for panel, source in zip(DATA_QUALITY_FIGURE_PANELS, data_sources, strict=True)],
        })
        with pytest.raises(ValueError, match="step roster"):
            runtime.validate_plan(bound.model_copy(update={"steps": [*bound.steps, extra]}))


def test_shaping_keeps_the_family_figure_bound_to_cohort_accounting():
    attempt, _ = baseline_plan_attempt(baseline_context("sex", covariates=("age",)))
    steps = list(attempt.output.steps)
    primary = next(s for s in steps if s.step_id == "adjusted_association")
    primary = primary.model_copy(update={"expected_outputs": [*primary.expected_outputs, "table:model_population_flow"]})
    plan = attempt.output.model_copy(update={"steps": [primary if s.step_id == primary.step_id else s for s in steps]})
    shaped, _ = ensure_cohort_accounting_figure_step(plan=plan)
    figure = next(s for s in shaped.steps if "figure:cohort_flow" in s.expected_outputs)
    assert figure.inputs == ["table:cohort_flow"]


def test_signed_trajectory_keeps_the_family_count_figure():
    context, payload, disclosure, authorities = _trajectory(parts=True)
    llm = ScriptedMockLLMClient([json.dumps(payload)])
    draft = ProgressivePlannerAgent(llm).run_attempt(
        context, planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=("strobe_2007", "record_2015"),
        enforce_article_contract=True, article_contract_context=context,
        planning_contract_context=disclosure, required_primary_cohort_selection_mode="all_input_rows",
    ).output
    bound, _ = authorities.bind_plan(draft, context=context)
    bound = apply_runtime_bound_figure_contracts(bound, [])
    figure = next(s for s in bound.steps if "figure:cohort_flow" in s.expected_outputs)
    assert figure.inputs == ["table:cohort_flow"]
    assert cohort_flow_figure_executor_declares_step(figure)
    producer = next(s for s in bound.steps if "table:cohort_flow" in s.expected_outputs)
    assert bound.steps.index(producer) < bound.steps.index(figure)
    authorities.validate_plan(bound)
    require_host_rendered_plan(plan=bound, trajectory_scientific_runtime_authority=authorities.trajectory,
                               scientific_runtime_projection_sha256="f" * 64)
    assert not _declared_host_figure(figure, plan=bound, host_cohort_step=None)
    unsigned = producer.model_copy(update={"icu_rule_refs": []})
    invalid = bound.model_copy(update={"steps": [unsigned if s is producer else s for s in bound.steps]})
    assert not _declared_host_figure(figure, plan=invalid, host_cohort_step=None,
                                     trajectory_scientific_runtime_authority=authorities.trajectory,
                                     scientific_runtime_projection_sha256="f" * 64)
    draft_figure = next(s for s in draft.steps if s.step_id == "family_cohort_flow_figure")
    tampered = draft_figure.model_copy(update={"inputs": ["table:cohort_flow", "stay_id"]})
    corrupt = draft.model_copy(update={"steps": [tampered if s is draft_figure else s for s in draft.steps]})
    with pytest.raises(ValueError, match="trajectory cohort-flow figure"):
        authorities.bind_plan(corrupt, context=context)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="Signed survival replaces the cohort accounting owner and its figure.")
def test_signed_landmark_survival_keeps_the_family_count_figure(tmp_path):
    context, authority = sealed_survival(tmp_path)
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)
    draft = sealed_draft(context, authorities)
    assert any("figure:cohort_flow" in step.expected_outputs for step in draft.steps)
    bound, _ = authorities.bind_plan(draft)
    assert any("figure:cohort_flow" in step.expected_outputs for step in bound.steps)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="Signed continuous survival replaces the cohort accounting owner and its figure.")
def test_signed_continuous_survival_keeps_the_family_count_figure():
    context = continuous_survival_context()
    request = continuous_survival_request(context)
    draft, _ = proposed_continuous_survival_plan(context, continuous_survival_spec(request, [AGE]))
    assert any("figure:cohort_flow" in step.expected_outputs for step in draft.steps)
    authority = build_current_case_scientific_runtime_authority(continuous_authority_body())
    bound = authority.bind_plan(draft)
    assert any("figure:cohort_flow" in step.expected_outputs for step in bound.steps)
