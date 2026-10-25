"""Custom products retain one producer across full and incremental compilation."""

from collections import Counter
from copy import deepcopy
import json

import pytest

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.planning.progressive_compiler import (
    _compile_one_step,
    compile_progressive_plan,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlanFoundation,
    ProgressivePlanOutline,
    ProgressivePlanSkeleton,
    ProgressiveSkeletonStep,
    ProgressiveStepMaterialization,
)
from easyicu.research_agent.planning.progressive_resume import (
    ProgressivePrefixState,
    compile_progressive_prefix,
)
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
)


from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from .progressive_planner_fixtures import (
    _foundation_payload, _materialization_payloads, _outline_payload,
)

def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Describe a synthetic measurement and its baseline context.",
        cohort=CohortDescriptor(
            cohort_name="synthetic", database="synthetic", n_stays=0,
            id_columns=["stay_id"],
        ),
        variables=[
            ConceptDescriptor(name="measurement", role="outcome", dtype="float64"),
            ConceptDescriptor(
                name="category", role="demographic", dtype="object",
                observed_domain={"levels": ["A", "B"]},
            ),
            ConceptDescriptor(name="age", role="demographic", dtype="float64"),
        ],
        user_preferences=UserPreferences(
            inferred_analysis_family="descriptive_epidemiology",
        ),
    )


def _skeleton() -> ProgressivePlanSkeleton:
    """Keep the replay's five-step shape, using only synthetic coordinates."""
    cohort_ref = {"producer_step_id": "cohort", "product_id": "artifact:analysis_cohort"}
    return ProgressivePlanSkeleton(
        analysis_type="descriptive_epidemiology",
        cohort={"name": "primary", "selection_mode": "all_input_rows"},
        display_labels=[
            {"key": "measurement", "value": "Recorded measurement"},
            {"key": "category", "value": "Recorded category"},
            {"key": "age", "value": "Age at admission"},
        ],
        steps=[
            ProgressiveSkeletonStep(
                step_id="cohort", planned_analysis_role="auxiliary",
                module_id="cohort_definition", objective="Account for the declared cohort.",
            ),
            ProgressiveSkeletonStep(
                step_id="primary_summary", planned_analysis_role="primary",
                module_id="custom_analysis", objective="Describe the empirical measurement distribution.",
                depends_on=["cohort"], raw_inputs=["measurement"],
                product_inputs=[cohort_ref],
                outputs=[
                    {"product_id": "table:cohort_summary", "semantic_role": "custom"},
                    {"product_id": "table:distribution_summary", "semantic_role": "custom"},
                ],
                scientific_action_id="descriptive.descriptive_summary",
                custom_method="empirical_measurement_summary",
            ),
            ProgressiveSkeletonStep(
                step_id="group_summary", planned_analysis_role="secondary",
                module_id="continuous_outcome_summary", objective="Describe the measurement within declared categories.",
                depends_on=["primary_summary"], raw_inputs=["category", "measurement"],
                primary_exposure="category",
            ),
            ProgressiveSkeletonStep(
                step_id="baseline", planned_analysis_role="auxiliary",
                module_id="custom_analysis", objective="Describe ungrouped baseline characteristics.",
                depends_on=["cohort"], raw_inputs=["age", "category"],
                product_inputs=[cohort_ref],
                outputs=[{"product_id": "table:table_one", "semantic_role": "custom"}],
                custom_method="descriptive_baseline_context",
            ),
            ProgressiveSkeletonStep(
                step_id="audit", planned_analysis_role="auxiliary",
                module_id="measurement_audit", objective="Audit the measurement availability and denominator.",
                depends_on=["cohort"], raw_inputs=["measurement"],
                outputs=[{
                    "product_id": "table:measurement_missingness",
                    "semantic_role": "measurement_missingness",
                }],
            ),
        ],
        rationale="Separate the primary measurement distribution from baseline characteristics.",
    )


def _with_consumers(skeleton: ProgressivePlanSkeleton) -> ProgressivePlanSkeleton:
    references = [
        {"producer_step_id": "primary_summary", "product_id": "table:cohort_summary"},
        {"producer_step_id": "baseline", "product_id": "table:table_one"},
    ]
    return skeleton.model_copy(update={"steps": [
        *skeleton.steps,
        *[
            ProgressiveSkeletonStep(
                step_id=module, planned_analysis_role="auxiliary", module_id=module,
                objective="Consume the primary and auxiliary tables with exact provenance.",
                depends_on=["primary_summary", "baseline"], product_inputs=references,
                outputs=[{"product_id": product, "semantic_role": role}],
            )
            for module, product, role in (
                ("visualization", "figure:summary", "figure"),
                ("report", "report:summary", "report"),
            )
        ],
    ]})


def test_distinct_custom_products_compile_and_consumers_bind_to_the_declaring_step():
    context = _context()
    skeleton = _skeleton()
    plan, _ = compile_progressive_plan(
        skeleton=skeleton, context=context,
    )
    steps = {step.step_id: step for step in plan.steps}
    products = Counter(product for step in plan.steps for product in step.expected_outputs)
    assert set(products.values()) == {1}
    assert steps["primary_summary"].expected_outputs == [
        "table:cohort_summary", "table:distribution_summary",
    ]
    assert steps["baseline"].expected_outputs == ["table:cohort_summary_baseline"]
    # Rendering modules are not offered by this descriptive family's outline;
    # exercise their shared product-binding owner against the compiled registry.
    with_consumers = _with_consumers(skeleton)
    for index in range(len(skeleton.steps), len(with_consumers.steps)):
        compiled = _compile_one_step(
            context=context, skeleton=with_consumers, step=with_consumers.steps[index],
            step_index=index, variables={item.name: item for item in context.variables},
            allowed_citations=frozenset(),
            producers={product: step.step_id for step in plan.steps for product in step.expected_outputs},
            outputs_by_step={step.step_id: step.expected_outputs for step in plan.steps},
        )
        steps[compiled.step.step_id] = compiled.step
    for consumer in ("visualization", "report"):
        assert "table:cohort_summary" in steps[consumer].inputs
        assert "table:cohort_summary_baseline" in steps[consumer].inputs
        assert "table:table_one" not in steps[consumer].inputs
    assert [item.input_key for item in steps["visualization"].input_consumption_contracts] == [
        "table:cohort_summary", "table:cohort_summary_baseline",
    ]


@pytest.mark.parametrize("product", ["table:cohort_summary", "table:table_one"])
def test_two_steps_declaring_the_same_product_still_fail_closed(product):
    skeleton = _skeleton()
    if product == "table:cohort_summary":
        skeleton.steps[3].outputs[0] = skeleton.steps[1].outputs[0]
    else:
        duplicate = skeleton.steps[3].model_copy(update={"step_id": "other_baseline"})
        skeleton = skeleton.model_copy(update={"steps": [*skeleton.steps, duplicate]})
    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(skeleton=skeleton, context=_context())
    assert caught.value.reason_code == "progressive_product_has_multiple_owners"
    assert caught.value.path == "outputs"
    assert caught.value.step_id == ("baseline" if product == "table:cohort_summary" else "other_baseline")


def test_different_baseline_aliases_receive_different_step_local_defaults():
    skeleton = _skeleton()
    other = skeleton.steps[3].model_dump(mode="json")
    other.update(
        step_id="another_baseline", custom_method="another_descriptive_method",
        outputs=[{"product_id": "artifact:baseline_context", "semantic_role": "custom"}],
    )
    skeleton = skeleton.model_copy(update={"steps": [
        *skeleton.steps, ProgressiveSkeletonStep.model_validate(other),
    ]})
    plan, _ = compile_progressive_plan(skeleton=skeleton, context=_context())
    assert plan.steps[3].expected_outputs == ["table:cohort_summary_baseline"]
    assert plan.steps[-1].expected_outputs == ["table:cohort_summary_another_baseline"]
    assert plan.steps[1].expected_outputs == [
        "table:cohort_summary", "table:distribution_summary",
    ]


def test_an_explicit_step_local_product_retains_the_same_closed_summary_method():
    skeleton = _skeleton()
    skeleton.steps[3].outputs[0] = skeleton.steps[3].outputs[0].model_copy(update={
        "product_id": "table:cohort_summary_baseline",
    })
    plan, _ = compile_progressive_plan(skeleton=skeleton, context=_context())
    assert plan.steps[3].method == "descriptive_cohort_summary"
    assert plan.steps[3].expected_outputs == ["table:cohort_summary_baseline"]


def test_incremental_replay_preserves_the_accepted_prefix_and_distinct_custom_outputs():
    skeleton = _skeleton()
    outline = ProgressivePlanOutline(
        analysis_type=skeleton.analysis_type,
        cohort_objective="Account for the sealed synthetic cohort.",
        steps=[{
            **step.model_dump(include={
                "step_id", "module_id", "objective", "planned_analysis_role",
                "depends_on", "scientific_action_id",
            }),
            "variable_names": step.raw_inputs or ["measurement"],
        } for step in skeleton.steps],
        rationale=skeleton.rationale,
    )
    foundation = ProgressivePlanFoundation(
        cohort=skeleton.cohort, display_labels=skeleton.display_labels,
    )
    state = ProgressivePrefixState()
    for index, step in enumerate(skeleton.steps):
        previous_refs = state.available_product_refs
        previous_receipt = state.receipt
        state = compile_progressive_prefix(
            state,
            ProgressiveStepMaterialization(
                outline_step_sha256=canonical_sha256(outline.steps[index].model_dump(mode="json")),
                foundation=None, step=step,
            ),
            outline=outline, foundation=foundation, context=_context(),
            allowed_literature_citation_keys=(), allowed_know_how_decisions=None,
            reporting_method_source_keys=(),
        )
        assert set(previous_refs) <= set(state.available_product_refs)
        if previous_receipt is not None:
            assert state.receipt.compiled_steps[:index] == previous_receipt.compiled_steps
        if index == 3:
            assert ("primary_summary", "table:cohort_summary") in state.available_product_refs
            assert ("baseline", "table:cohort_summary_baseline") in state.available_product_refs
            assert ("baseline", "table:cohort_summary") not in state.available_product_refs
    assert state.plan is not None
    assert len(state.materializations) == 5


def _context_with_patients():
    context = _context()
    context.cohort.provenance.update({
        "analysis_unit": "icu_stay", "patient_identity_available": True,
        "evidence_stage": "metadata_only_planning", "patient_rows_read": False,
        "replacement_row_identity": {
            "output_identity_column": "stay_id", "mapping_file_sha256": "a" * 64,
            "patient_group_derivation": {"algorithm": "prefix_before_:s", "delimiter": ":s"},
        },
    })
    return context


@pytest.mark.parametrize("products", [
    ["table:invented_distribution"],
    ["table:cohort_summary"],
    ["table:distribution_summary", "table:cohort_summary"],
    ["table:cohort_summary", "table:distribution_summary", "table:extra"],
])
def test_primary_product_mismatch_names_the_step_and_exact_repair(products):
    payload = _skeleton().model_dump(mode="json")
    payload["steps"][1]["outputs"] = [
        {"product_id": product, "semantic_role": "custom"} for product in products
    ]
    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(payload), context=_context(),
        )
    error = caught.value
    assert error.reason_code == "progressive_scientific_action_outputs_mismatch"
    assert error.step_id == "primary_summary" and error.step_index == 1
    assert error.path == "outputs"
    assert "table:cohort_summary" in str(error)
    assert "table:distribution_summary" in str(error)


@pytest.mark.parametrize("invalid_attempts", [1, 3])
def test_primary_product_repair_preserves_the_question_and_locked_prefix(invalid_attempts):
    payload = _skeleton().model_dump(mode="json")
    materializations = _materialization_payloads(payload)
    outline = _outline_payload(payload)
    for candidate in outline["design_selection"]["candidates"]:
        candidate["required_variables"] = ["measurement", "category"]
        candidate["estimand"] = ("The pooled empirical distribution of the recorded measurement."
                                 if candidate["disposition"] == "selected"
                                 else "The empirical measurement distribution within each category.")
        candidate["primary_method"] = "Empirical summary with observed denominators."
        candidate["figure_role"] = "Display the empirical measurement distribution."
        candidate["supports"] = "Stay-level empirical descriptions."
        candidate["cannot_prove"] = "Patient-level inference or causal effects."
    for step, source in zip(outline["steps"], payload["steps"], strict=True):
        step["variable_names"] = source["raw_inputs"] or ["measurement"]
    validated = ProgressivePlanOutline.model_validate(outline)
    for response, step in zip(materializations, validated.steps, strict=True):
        response["outline_step_sha256"] = canonical_sha256(step.model_dump(mode="json"))
    foundation = _foundation_payload(payload)
    foundation["outline_sha256"] = canonical_sha256(validated.model_dump(mode="json"))
    invalid = deepcopy(materializations[1])
    invalid["step"]["outputs"] = [
        {"product_id": "table:invented_distribution", "semantic_role": "custom"},
    ]
    responses = [
        outline, foundation, materializations[0],
        *[deepcopy(invalid) for _ in range(invalid_attempts)], *materializations[1:],
    ]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)
    context = _context_with_patients()
    context.cohort.provenance.update({
        "analysis_unit": "icu_stay", "patient_identity_available": True,
        "patient_id_columns": ["person_id"],
        "evidence_stage": "metadata_only_planning", "patient_rows_read": False,
    })
    original = context.model_dump_json()
    checkpoints = []
    if invalid_attempts == 1:
        plan = agent.run(context, checkpoint_callback=checkpoints.append)
        assert len(llm.calls) == 8
        assert agent.last_result.facts.prompt_metrics["compile_revision_count"] == 1
        assert agent.last_result.facts.prompt_metrics["full_revision_count"] == 0
        assert plan.steps[1].expected_outputs == ["table:cohort_summary", "table:distribution_summary"]
        review = build_plan_scientific_review(context=context, plan=plan, require_reportable_capability=True)
        assert review.facts["repeat_units_possible"] and review.facts["patient_identity_available"]
        assert not {"SCIENTIFIC_CAPABILITY_NOT_REPORTABLE", "REPEATED_STAY_METHOD_NOT_DECLARED"} & {
            finding.code for finding in review.findings
        }
        prefixes = [cp.materializations[:1] for cp in checkpoints if cp.materializations]
        assert prefixes and all(prefix == prefixes[0] for prefix in prefixes)
    else:
        with pytest.raises(ProgressivePlanCompileError) as caught:
            agent.run(context, checkpoint_callback=checkpoints.append)
        assert caught.value.reason_code == "progressive_scientific_action_outputs_mismatch"
        assert len(llm.calls) == 6  # Three attempts, never an unbounded retry.
    assert context.model_dump_json() == original
    prompt = llm.calls[4][0][-1].content
    assert "HOST COMPILER OBSERVATION FOR THIS CURRENT STEP" in prompt
    assert "progressive_scientific_action_outputs_mismatch" in prompt
    assert "table:cohort_summary" in prompt and "table:distribution_summary" in prompt


def test_an_unrepaired_archived_plan_still_fails_closed_at_review():
    context = _context_with_patients()
    plan, _ = compile_progressive_plan(skeleton=_skeleton(), context=context)
    plan.steps[1] = plan.steps[1].model_copy(update={"expected_outputs": ["table:invented_distribution"]})
    review = build_plan_scientific_review(context=context, plan=plan, require_reportable_capability=True)
    assert {"SCIENTIFIC_CAPABILITY_NOT_REPORTABLE", "REPEATED_STAY_METHOD_NOT_DECLARED"} <= {
        finding.code for finding in review.findings if finding.severity == "blocker"
    }
    assert review.approval_allowed is False
