"""Custom products retain one producer across full and incremental compilation."""

from collections import Counter

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
