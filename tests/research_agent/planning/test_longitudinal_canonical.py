"""Canonical workflow metadata builds graphs without granting run authority."""

import ast
import inspect

import pytest
from pydantic import ValidationError

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.planning import family_spec
from easyicu.research_agent.planning.family_spec import longitudinal_canonical
from easyicu.research_agent.planning.family_spec.contract import FamilySpecError
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_resume import assemble_progressive_skeleton
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
)


FAMILIES = {
    "target_trial_emulation": (
        family_spec.build_target_trial_skeleton,
        ("cohort_accounting", "measurement_audit", "target_trial_suite", "visualization", "report"),
    ),
    "time_varying_exposure_survival": (
        family_spec.build_time_varying_survival_skeleton,
        ("cohort_accounting", "baseline_context", "measurement_audit", "time_varying_cox", "report"),
    ),
    "dynamic_landmark_prediction": (
        family_spec.build_dynamic_landmark_skeleton,
        (
            "cohort_accounting", "measurement_audit", "landmark_features", "horizon_labels",
            "patient_split", "model_fit", "landmark_performance", "visualization", "report",
        ),
    ),
}
SIGNED_FAMILIES = ("target_trial_emulation", "time_varying_exposure_survival")


def _compiled(family_id):
    request, spec = longitudinal_canonical.canonical_longitudinal_request(family_id)
    builder, _expected_steps = FAMILIES[family_id]
    built = builder(request, spec)
    skeleton = assemble_progressive_skeleton(
        outline=built.outline, foundation=built.foundation.foundation,
        steps=[item.step for item in built.materializations],
    )
    patient = (
        request.dynamic_landmark.patient_column if request.dynamic_landmark
        else request.sealed_target_trial.patient_group_column if request.sealed_target_trial
        else request.identity_column
    )
    context = ResearchContext(
        research_question=request.research_question,
        cohort=CohortDescriptor(
            cohort_name=request.cohort_name, database="miiv", n_stays=0,
            id_columns=[request.identity_column], outcome_columns=[request.outcome],
        ),
        variables=[ConceptDescriptor(
            name=column, source_concept=column, description="Workflow " + column.replace("_", " "),
            dtype="int64" if column == request.outcome else "float64",
            role="id" if column in {request.identity_column, patient}
            else "outcome" if column == request.outcome else "demographic",
            observed_domain={"n_unique": 2, "is_binary": True, "levels": [0, 1]}
            if column == request.outcome else None,
        ) for column in request.variable_roster],
        target_outcome=request.outcome, primary_exposure=request.primary_exposure or None,
        user_preferences=UserPreferences(inferred_analysis_family=request.analysis_type),
    )
    plan, receipt = compile_progressive_plan(skeleton=skeleton, context=context)
    return request, spec, skeleton, plan, receipt


@pytest.mark.parametrize("family_id", FAMILIES)
def test_canonical_requests_build_full_validated_step_graphs(family_id):
    request, spec, skeleton, plan, receipt = _compiled(family_id)
    assert request.family_id == spec.family_id == family_id
    assert spec.request_sha256 == request.request_sha256
    assert tuple(step.step_id for step in skeleton.steps) == FAMILIES[family_id][1]
    assert len(plan.steps) == len(skeleton.steps) == len(receipt.compiled_steps)
    assert plan.design_selection.claim_ceiling == "analysis_only"
    assert plan.steps[-1].expected_outputs == ["report:report"]
    producers = {output: step.step_id for step in plan.steps for output in step.expected_outputs}
    assert len(producers) == sum(len(step.expected_outputs) for step in plan.steps)
    assert any(step.depends_on for step in skeleton.steps)
    assert any(step.product_inputs for step in skeleton.steps)
    for step in skeleton.steps:
        for product in step.product_inputs:
            assert producers[product.product_id] == product.producer_step_id
            assert product.producer_step_id in step.depends_on


def test_canonical_module_has_no_test_imports():
    tree = ast.parse(inspect.getsource(longitudinal_canonical))
    imports = [
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    ] + [
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    ]
    assert all(name != "tests" and not name.startswith("tests.") for name in imports)
    assert "read-only workflow metadata" in longitudinal_canonical.__doc__.lower()


@pytest.mark.parametrize("family_id", FAMILIES)
def test_canonical_requests_are_repeatable_and_do_not_share_mutable_coordinates(family_id):
    request, spec = longitudinal_canonical.canonical_longitudinal_request(family_id)
    expected_request, expected_spec = request.model_dump(mode="json"), spec.model_dump(mode="json")
    request.variable_roster.append("unbound_column")
    spec.reader_display_labels.clear()
    fresh, fresh_spec = longitudinal_canonical.canonical_longitudinal_request(family_id)
    assert fresh.model_dump(mode="json") == expected_request
    assert fresh_spec.model_dump(mode="json") == expected_spec
    assert fresh.cohort_selection_mode == "all_input_rows"
    assert all(bound is None for bound in fresh.typed_cohort_bounds)


@pytest.mark.parametrize("family_id", ["baseline_adjusted_association", "unknown", ""])
def test_unknown_families_fail_closed(family_id):
    with pytest.raises(FamilySpecError) as error:
        longitudinal_canonical.canonical_longitudinal_request(family_id)
    assert error.value.reason_code == "family_spec_template_mismatch"


@pytest.mark.parametrize("family_id", FAMILIES)
@pytest.mark.parametrize("coordinate", ["request", "spec"])
def test_workflow_coordinates_are_not_runtime_authorities(family_id, coordinate):
    request, spec = longitudinal_canonical.canonical_longitudinal_request(family_id)
    value = request if coordinate == "request" else spec
    with pytest.raises(ValidationError):
        load_current_case_scientific_runtime_authority(value.model_dump(mode="json"))


@pytest.mark.parametrize("family_id", SIGNED_FAMILIES)
def test_request_digest_cannot_replace_an_execution_digest(family_id):
    request, _spec = longitudinal_canonical.canonical_longitudinal_request(family_id)
    owner = request.sealed_target_trial or request.sealed_time_varying
    payload = owner.model_dump(mode="json")
    payload["execution_contract_sha256"] = request.request_sha256
    with pytest.raises(ValidationError, match="authority digest mismatch"):
        load_current_case_scientific_runtime_authority(payload)


@pytest.mark.parametrize("family_id", SIGNED_FAMILIES)
def test_metadata_authority_is_rejected_by_runtime_loading(family_id):
    request, _spec = longitudinal_canonical.canonical_longitudinal_request(family_id)
    owner = request.sealed_target_trial or request.sealed_time_varying
    with pytest.raises(ValidationError):
        load_current_case_scientific_runtime_authority(owner.model_dump(mode="json"))


@pytest.mark.parametrize("family_id", SIGNED_FAMILIES)
def test_metadata_authority_cannot_bind_execution(family_id):
    request, _spec, _skeleton, plan, _receipt = _compiled(family_id)
    owner = request.sealed_target_trial or request.sealed_time_varying
    with pytest.raises(ValueError):
        bound = owner.bind_plan(plan)
        owner.validate_plan(bound)
