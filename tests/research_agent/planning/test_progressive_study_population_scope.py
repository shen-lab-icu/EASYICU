"""The study-cohort scope belongs to the host, and binds one reader to one root.

``population_scope="study_cohort"`` marks the host root that republishes the
run cohort and the one distribution that reads it.  A Planner never sees the
value in its schemas and cannot return it; the contract fixes the root's
shape; and the compiler binds each study-cohort reader to exactly one
preceding root it depends on, while refusing the study product to every other
step.  Contexts are synthetic.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.progressive_payload import (
    parse_progressive_model,
    parse_progressive_step_materialization,
    progressive_outline_structured_output_request,
    progressive_step_materialization_request,
    progressive_structured_output_request,
)
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.planning.progressive_compiler import (
    ProgressivePlanCompileError,
    compile_progressive_plan,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveOutlineStep,
    ProgressiveOutputIntent,
    ProgressivePlanOutline,
    ProgressiveProductRef,
    ProgressiveSkeletonStep,
    duplicated_host_singletons,
)

from .family_spec_fixtures import ALLOWED_CITATIONS, _context, _request, _run, _spec_payload

ROOT_OUTPUT = ProgressiveOutputIntent(
    product_id="cohort:study_population", semantic_role="analysis_cohort"
)


def _root(**updates) -> ProgressiveSkeletonStep:
    payload = {
        "step_id": "study_population",
        "planned_analysis_role": "auxiliary",
        "module_id": "cohort_definition",
        "objective": "Publish the study cohort as the population of the exposure occurrence.",
        "depends_on": [],
        "raw_inputs": [],
        "outputs": [ROOT_OUTPUT],
        "population_scope": "study_cohort",
        "literature_bindings": [],
    }
    payload.update(updates)
    return ProgressiveSkeletonStep.model_validate(payload)


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def test_the_root_reads_nothing_and_publishes_one_study_product() -> None:
    assert _root().population_scope == "study_cohort"


@pytest.mark.parametrize(
    "updates",
    [
        {"planned_analysis_role": "secondary"},
        {"depends_on": ["cohort_definition"]},
        {"raw_inputs": ["patient_stay_id"]},
        {
            "product_inputs": [
                ProgressiveProductRef(
                    producer_step_id="cohort_definition", product_id="artifact:analysis_cohort"
                )
            ]
        },
        {
            "outputs": [
                ROOT_OUTPUT,
                ProgressiveOutputIntent(product_id="table:cohort_flow", semantic_role="cohort_flow"),
            ]
        },
        {
            "outputs": [
                ProgressiveOutputIntent(
                    product_id="cohort:any_population", semantic_role="analysis_cohort"
                )
            ]
        },
        {"population_scope_change_reason": "An intentional amendment of the population."},
    ],
)
def test_any_other_root_shape_is_refused(updates) -> None:
    with pytest.raises(ValidationError):
        _root(**updates)


@pytest.mark.parametrize("module_id", ["table_one", "adjusted_association", "absolute_risk_context"])
def test_the_scope_belongs_to_the_root_and_its_distribution_only(module_id) -> None:
    with pytest.raises(ValidationError, match="study-cohort population"):
        ProgressiveOutlineStep(
            step_id="elsewhere",
            planned_analysis_role="secondary",
            module_id=module_id,
            objective="Describe something on the study cohort.",
            depends_on=[],
            variable_names=["injury_stage"],
            literature_citation_keys=[],
            population_scope="study_cohort",
        )


def test_the_root_is_a_second_cohort_owner_but_each_stays_singular() -> None:
    primary = ProgressiveOutlineStep(
        step_id="cohort_definition", planned_analysis_role="auxiliary",
        module_id="cohort_definition", objective="Build the landmark analysis cohort.",
        depends_on=[], variable_names=["patient_stay_id"], literature_citation_keys=[],
    )
    root = ProgressiveOutlineStep(
        step_id="study_population", planned_analysis_role="auxiliary",
        module_id="cohort_definition", objective="Publish the study cohort.",
        depends_on=[], variable_names=["patient_stay_id"], literature_citation_keys=[],
        population_scope="study_cohort",
    )

    assert duplicated_host_singletons([root, primary]) == {}
    assert duplicated_host_singletons(
        [root, root.model_copy(update={"step_id": "study_population_again"}), primary]
    ) == {"cohort_definition:study_cohort": ["study_population", "study_population_again"]}
    assert duplicated_host_singletons(
        [primary, primary.model_copy(update={"step_id": "cohort_again"})]
    ) == {"cohort_definition": ["cohort_definition", "cohort_again"]}


# ---------------------------------------------------------------------------
# The Planner's transport
# ---------------------------------------------------------------------------


def test_no_planner_schema_offers_the_host_scope() -> None:
    variables = ["injury_stage", "death", "age"]
    actions = ["association.adjusted_association"]
    outline = progressive_outline_structured_output_request(
        analysis_types=["association_study"], variable_names=variables,
        scientific_action_ids=actions,
    )
    initial = progressive_structured_output_request(
        analysis_types=["association_study"], variable_names=variables,
        scientific_action_ids=actions,
    )
    suffix = progressive_structured_output_request(
        analysis_types=["association_study"], variable_names=variables,
        scientific_action_ids=actions, suffix=True,
    )
    absolute_risk = ProgressiveOutlineStep(
        step_id="absolute_risk_context", planned_analysis_role="secondary",
        module_id="absolute_risk_context", objective="Report absolute risk by level.",
        depends_on=[], variable_names=["injury_stage", "death"], literature_citation_keys=[],
        population_scope="primary_model",
    )
    step = progressive_step_materialization_request(
        outline_step=absolute_risk,
        outline_step_sha256=canonical_sha256(absolute_risk.model_dump(mode="json")),
        variable_names=variables, scientific_action_ids=actions,
    )

    for request in (outline, initial, suffix, step):
        assert "primary_model" in request.schema_json
        assert "study_cohort" not in request.schema_json


def _asking_context():
    context = _context()
    return context.model_copy(
        update={
            "research_question": context.research_question
            + " What proportion of stays reach each injury stage?",
            "variables": [
                item.model_copy(
                    update={
                        "analysis_window": "icu_admission[0,24]h",
                        "analysis_window_role": "exposure_definition",
                    }
                )
                if item.name == "injury_stage"
                else item
                for item in context.variables
            ],
        }
    )


def _planned():
    context = _asking_context()
    request = _request(context)
    _llm, result = _run(context, [json.dumps(_spec_payload(request))])
    return context, result.facts


def test_a_planner_response_cannot_claim_the_host_scope() -> None:
    _context_used, facts = _planned()
    outline_payload = facts.outline.model_dump(mode="json")
    assert any(step.get("population_scope") == "study_cohort" for step in outline_payload["steps"])

    with pytest.raises(ValueError, match="host template"):
        parse_progressive_model(json.dumps(outline_payload), ProgressivePlanOutline)
    materialization = next(
        item for item in facts.materializations if item.step.step_id == "exposure_occurrence"
    )
    with pytest.raises(ValueError, match="host template"):
        parse_progressive_step_materialization(
            json.dumps(materialization.model_dump(mode="json"))
        )


# ---------------------------------------------------------------------------
# The compiler binds one reader to one preceding root
# ---------------------------------------------------------------------------


def _compile(context, skeleton):
    return compile_progressive_plan(
        skeleton=skeleton, context=context,
        allowed_literature_citation_keys=ALLOWED_CITATIONS,
    )


def _replace_step(skeleton, step_id: str, **updates):
    return skeleton.model_copy(
        update={
            "steps": [
                step.model_copy(update=updates) if step.step_id == step_id else step
                for step in skeleton.steps
            ]
        }
    )


def test_the_template_skeleton_recompiles() -> None:
    context, facts = _planned()
    plan, _receipt = _compile(context, facts.skeleton)

    root = next(step for step in plan.steps if step.step_id == "study_population")
    assert root.method == "host_materialized_locked_cohort"
    assert root.population_scope is None


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"depends_on": []}, "progressive_study_population_binding_invalid"),
        (
            {
                "product_inputs": [
                    ProgressiveProductRef(
                        producer_step_id="study_population", product_id="cohort:study_population"
                    ),
                    ProgressiveProductRef(
                        producer_step_id="cohort_definition", product_id="artifact:analysis_cohort"
                    ),
                ],
                "depends_on": ["study_population", "cohort_definition"],
            },
            "progressive_study_population_binding_invalid",
        ),
        (
            {
                "product_inputs": [
                    ProgressiveProductRef(
                        producer_step_id="cohort_definition", product_id="artifact:analysis_cohort"
                    )
                ],
                "depends_on": ["cohort_definition"],
            },
            "progressive_study_population_binding_invalid",
        ),
    ],
    ids=["no-dependency-on-the-root", "a-second-cohort", "the-landmark-cohort-instead"],
)
def test_a_study_cohort_reader_binds_exactly_one_preceding_root(updates, reason) -> None:
    context, facts = _planned()
    skeleton = _replace_step(facts.skeleton, "exposure_occurrence", **updates)

    with pytest.raises(ProgressivePlanCompileError) as caught:
        _compile(context, skeleton)
    assert caught.value.reason_code == reason


def test_a_reader_after_the_landmark_cohort_still_reads_only_the_study_cohort() -> None:
    context, facts = _planned()
    by_id = {step.step_id: step for step in facts.skeleton.steps}
    order = [
        "cohort_definition",
        "study_population",
        "exposure_occurrence",
        *(
            step.step_id
            for step in facts.skeleton.steps
            if step.step_id not in {"cohort_definition", "study_population", "exposure_occurrence"}
        ),
    ]
    skeleton = facts.skeleton.model_copy(update={"steps": [by_id[step_id] for step_id in order]})

    plan, _receipt = _compile(context, skeleton)

    reader = next(step for step in plan.steps if step.step_id == "exposure_occurrence")
    assert [value for value in reader.inputs if ":" in value] == ["cohort:study_population"]


def test_no_other_step_may_read_the_study_product() -> None:
    context, facts = _planned()
    table_one = next(step for step in facts.skeleton.steps if step.step_id == "table_one")
    skeleton = _replace_step(
        facts.skeleton,
        "table_one",
        product_inputs=[
            *table_one.product_inputs,
            ProgressiveProductRef(
                producer_step_id="study_population", product_id="cohort:study_population"
            ),
        ],
        depends_on=[*table_one.depends_on, "study_population"],
    )

    with pytest.raises(ProgressivePlanCompileError) as caught:
        _compile(context, skeleton)
    assert caught.value.reason_code == "progressive_study_population_scope_mismatch"


def test_the_root_never_publishes_under_the_locked_cohort_name() -> None:
    context, facts = _planned()
    skeleton = facts.skeleton.model_copy(
        update={
            "cohort": facts.skeleton.cohort.model_copy(update={"name": "study_population"})
        }
    )

    with pytest.raises(ProgressivePlanCompileError) as caught:
        _compile(context, skeleton)
    assert caught.value.reason_code == "progressive_study_population_product_is_locked_cohort"
