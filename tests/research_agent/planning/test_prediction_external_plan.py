"""A reviewed external-validation action has a closed executable shape."""

import pytest
import json

from easyicu.research_agent.contracts.prediction_validation import (
    EXTERNAL_ACTION, EXTERNAL_OUTPUTS, ExternalValidationSpec, FROZEN_MODEL_INPUT,
    external_cohort_input, external_validation_method, external_validation_spec_for_step, external_validation_step_matches,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.planning.scientific_action_catalog import (
    scientific_action_for_id, validate_plan_scientific_action_selections,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from easyicu.research_agent.authority.plan_scope import _step_scientific_signature


def _step():
    spec = ExternalValidationSpec(
        development_database="miiv", validation_databases=("eicu", "hirid"),
        frozen_model_sha256="a" * 64, predictor_concepts=("hr",), outcome_concept="death",
    )
    return AnalysisStep(
        step_id="external_validation", planned_analysis_role="secondary",
        intent="Evaluate the unchanged fitted model in the declared independent databases.",
        scientific_action_id=EXTERNAL_ACTION, method=external_validation_method(spec),
        inputs=[FROZEN_MODEL_INPUT, *(external_cohort_input(db) for db in spec.validation_databases)],
        expected_outputs=list(EXTERNAL_OUTPUTS),
    )


def test_external_action_is_host_owned_and_reaches_the_exact_runner():
    from easyicu.research_agent.planning.analysis_method_suite import get_suite

    method = next(m for m in get_suite("prediction").methods if m.key == "external_validation")
    assert method.implementation == "deterministic"
    assert method.runner == "prediction_external_validation"
    action = scientific_action_for_id(analysis_type="prediction_model", action_id=EXTERNAL_ACTION)
    assert action.execution_mode == "host_owned"
    assert action.kernel_imports == ("easyicu.research_agent.methods.prediction_validation",)
    assert action.method_adapter.claim_ceiling == "analysis_only"
    step = _step()
    plan = AnalysisPlan(research_question="Transport a prespecified clinical risk model.",
                        analysis_type="prediction_model", steps=[step])
    validate_plan_scientific_action_selections(plan=plan, inferred_analysis_type="prediction_model")
    selection = select_standard_executor(step, plan=plan)
    assert selection.analysis_kind == "prediction_external_validation"
    assert selection.consumed_input_keys == tuple(step.inputs)
    assert "run_external_validation_step" in selection.code
    assert "refit" not in selection.code
    assert external_validation_spec_for_step(step).validation_databases == ("eicu", "hirid")


def test_underdeclared_external_action_is_refused_in_plan_validation():
    step = _step().model_copy(update={"method": "external_validation"})
    plan = AnalysisPlan(research_question="Validate a model.", analysis_type="prediction_model", steps=[step])
    with pytest.raises(ValueError, match="external_validation_declaration_required"):
        validate_plan_scientific_action_selections(plan=plan, inferred_analysis_type="prediction_model")


def test_compact_wire_spec_is_restored_and_changes_invalidate_plan_identity():
    from easyicu.research_agent.planning.progressive_contract import ProgressiveSkeletonStep, ProgressiveOutputIntent

    step = _step()
    spec = external_validation_spec_for_step(step)
    progressive = ProgressiveSkeletonStep(
        step_id="external", planned_analysis_role="secondary", module_id="custom_analysis",
        objective=step.intent, scientific_action_id=EXTERNAL_ACTION, custom_method="external_validation",
        external_validation_spec=json.dumps(spec.model_dump(mode="json")),
        outputs=[ProgressiveOutputIntent(product_id=p, semantic_role="custom") for p in EXTERNAL_OUTPUTS],
    )
    assert progressive.external_validation_spec == spec
    restored = AnalysisStep.model_validate(step.model_dump(mode="json"))
    assert external_validation_spec_for_step(restored) == spec
    changed = step.model_copy(update={"method": external_validation_method(spec.model_copy(
        update={"frozen_model_sha256": "b" * 64},
    ))})
    assert _step_scientific_signature(step) != _step_scientific_signature(changed)


def test_an_external_spec_cannot_be_silently_ignored_on_another_action():
    step = _step().model_copy(update={"scientific_action_id": "prediction.internal_validation"})
    plan = AnalysisPlan(research_question="Validate a model.", analysis_type="prediction_model", steps=[step])
    with pytest.raises(ValueError, match="external_validation_spec belongs only"):
        validate_plan_scientific_action_selections(plan=plan, inferred_analysis_type="prediction_model")


@pytest.mark.parametrize("method", [
    "prediction_external_validation({not-json})", "prediction_external_validation({})",
    external_validation_method(ExternalValidationSpec(
        development_database="miiv", validation_databases=("eicu",), frozen_model_sha256="a" * 64,
        predictor_concepts=("hr",), outcome_concept="death",
    ))[:-1] + " ",
])
def test_malformed_external_method_is_refused(method):
    with pytest.raises(ValueError, match="external_validation_declaration_required"):
        external_validation_spec_for_step(_step().model_copy(update={"method": method}))


@pytest.mark.parametrize("change", [
    {"database": "miiv"}, {"concept": "unknown"},
    {"development_meaning": "icu_death", "validation_meaning": "icu_death"},
])
def test_confirmation_coordinates_must_belong_to_a_changed_validation_concept(change):
    spec = external_validation_spec_for_step(_step())
    confirmation = {"database": "eicu", "concept": "death", "development_meaning": "hospital_death",
                    "validation_meaning": "icu_death", "confirmation": "Accept a different endpoint in the target database."}
    with pytest.raises(ValueError):
        ExternalValidationSpec.model_validate({**spec.model_dump(), "confirmed_changes": [{**confirmation, **change}]})


def test_progressive_external_action_requires_its_exact_reviewed_source_roster():
    from easyicu.research_agent.planning.progressive_contract import (
        ProgressiveSkeletonStep, ProgressiveProductRef, ProgressiveOutputIntent,
    )
    from easyicu.research_agent.planning.progressive_compiler import _validate_scientific_action_runtime_contract

    spec = external_validation_spec_for_step(_step())
    action = scientific_action_for_id(analysis_type="prediction_model", action_id=EXTERNAL_ACTION)
    inputs = [FROZEN_MODEL_INPUT, *(external_cohort_input(db) for db in spec.validation_databases)]
    step = ProgressiveSkeletonStep(
        step_id="external", module_id="custom_analysis", planned_analysis_role="secondary",
        custom_method="external_validation", scientific_action_id=EXTERNAL_ACTION,
        objective="Validate the frozen model in the declared databases.", depends_on=["intake"],
        external_validation_spec=spec,
        product_inputs=[ProgressiveProductRef(product_id=p, producer_step_id="intake") for p in inputs],
        outputs=[ProgressiveOutputIntent(product_id=p, semantic_role=role) for p, role in action.runtime_contract.outputs],
    )
    _validate_scientific_action_runtime_contract(action=action, step=step, step_index=0, outputs=action.runtime_contract.outputs)
    with pytest.raises(ValueError, match="progressive_scientific_action_inputs_mismatch"):
        _validate_scientific_action_runtime_contract(action=action, step=step.model_copy(update={"product_inputs": []}),
                                                     step_index=0, outputs=action.runtime_contract.outputs)


def test_progressive_wire_offers_the_compact_spec_only_to_its_external_step():
    from easyicu.research_agent.agents.progressive_payload import progressive_step_materialization_request
    from easyicu.research_agent.planning.progressive_contract import ProgressiveOutlineStep

    outline = ProgressiveOutlineStep(
        step_id="external", planned_analysis_role="secondary", module_id="custom_analysis",
        objective="Apply the frozen model in independent validation databases.",
        scientific_action_id=EXTERNAL_ACTION, depends_on=["intake"], variable_names=["hr", "death"],
    )
    request = progressive_step_materialization_request(
        outline_step=outline, outline_step_sha256="a" * 64, variable_names=["hr", "death"],
        scientific_action_ids=[EXTERNAL_ACTION], available_product_refs=[("intake", FROZEN_MODEL_INPUT)],
    )
    assert '"external_validation_spec"' in request.schema_json
    assert "ExternalValidationSpec" not in request.schema_json
    assert "ConfirmedMeaningChange" not in request.schema_json


def test_progressive_compiler_carries_the_spec_through_the_sealed_method():
    from types import SimpleNamespace
    from easyicu.research_agent.planning.progressive_compiler import _compile_one_step
    from easyicu.research_agent.planning.progressive_contract import (
        ProgressiveSkeletonStep, ProgressiveOutputIntent, ProgressiveProductRef,
    )
    from easyicu.research_agent.schema import ResearchContext, CohortDescriptor

    spec = external_validation_spec_for_step(_step())
    inputs = [FROZEN_MODEL_INPUT, *(external_cohort_input(db) for db in spec.validation_databases)]
    source = ProgressiveSkeletonStep(
        step_id="intake", module_id="custom_analysis", planned_analysis_role="auxiliary",
        custom_method="reviewed_external_intake", objective="Supply reviewed model and cohort inputs.",
        outputs=[ProgressiveOutputIntent(product_id=p, semantic_role="custom") for p in inputs],
    )
    step = ProgressiveSkeletonStep(
        step_id="external", module_id="custom_analysis", planned_analysis_role="secondary",
        custom_method="external_validation", objective="Apply the frozen model in the declared databases.",
        scientific_action_id=EXTERNAL_ACTION, external_validation_spec=spec, depends_on=["intake"],
        product_inputs=[ProgressiveProductRef(product_id=p, producer_step_id="intake") for p in inputs],
        outputs=[ProgressiveOutputIntent(product_id=p, semantic_role="custom") for p in EXTERNAL_OUTPUTS],
    )
    compiled = _compile_one_step(
        context=ResearchContext(research_question="Validate a frozen clinical risk model.",
                                cohort=CohortDescriptor(cohort_name="development", database="miiv", n_stays=0), variables=[]),
        skeleton=SimpleNamespace(analysis_type="prediction_model", steps=[source, step]), step=step, step_index=1,
        variables={}, allowed_citations=frozenset(), producers={p: "intake" for p in inputs},
        outputs_by_step={"intake": inputs},
    ).step
    assert external_validation_spec_for_step(compiled) == spec
    assert external_validation_step_matches(compiled)


def test_workflow_graph_shows_the_optional_frozen_model_and_external_cohort():
    from easyicu.webserver.research_workflows import research_workflow_catalog

    workflow = next(r for r in research_workflow_catalog(enabled=True)["items"] if r["id"] == "clinical-risk-prediction")
    nodes = {r["id"]: r for r in workflow["steps"]}
    assert nodes["external_validation"]["action_id"] == EXTERNAL_ACTION
    assert nodes["external_validation"]["optional"] is True
    assert FROZEN_MODEL_INPUT in nodes["external_validation_intake"]["outputs"]
    assert any(e["source"] == "external_validation_intake" and e["target"] == "external_validation"
               for e in workflow["edges"])


@pytest.mark.parametrize("change", [
    {"inputs": [FROZEN_MODEL_INPUT]}, {"expected_outputs": ["table:external_validation"]},
    {"planned_analysis_role": "primary"}, {"model_requirements": [object()]},
])
def test_external_owner_does_not_widen_or_approximate_its_contract(change):
    assert not external_validation_step_matches(_step().model_copy(update=change))


@pytest.mark.parametrize("update", [
    {"validation_databases": ("miiv",)}, {"validation_databases": ("eicu", "eicu")},
    {"development_database": "mimiciv"}, {"predictor_concepts": ("hr", "hr")},
    {"predictor_concepts": ("death",)}, {"frozen_model_sha256": "unknown"},
])
def test_database_roster_and_frozen_model_coordinates_are_closed(update):
    with pytest.raises(ValueError):
        ExternalValidationSpec.model_validate({**external_validation_spec_for_step(_step()).model_dump(), **update})
