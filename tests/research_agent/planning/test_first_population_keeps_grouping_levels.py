"""First-plan population/grouping reconciliation uses typed readings, never rows."""

import json

from tests.support.distribution_policies import EXCLUDE_MISSING

import pytest

from easyicu.research_agent.agents.progressive_planner import select_progressive_variables
from easyicu.research_agent.contracts.exposure_group_rules import EXPOSURE_GROUP_LABELS_KEY
from easyicu.research_agent.planning.cohort_predicate_domain import population_emptied_exposure_levels
from easyicu.research_agent.planning.family_spec import (
    build_descriptive_skeleton,
    build_family_spec_request,
)
from easyicu.research_agent.planning.family_spec.contract import spec_from_mapping
from easyicu.research_agent.planning.progressive_compiler import (
    compile_progressive_plan,
    progressive_cohort_concept_ids,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlanSkeleton,
)
from easyicu.research_agent.planning.progressive_resume import assemble_progressive_skeleton
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole
from tests.research_agent.planning.family_spec_fixtures import (
    ALLOWED_CITATIONS,
    DIRECT_COMPARATORS,
    _descriptive_context,
    _run,
)


def _study(*, grouping=True, typed_bound=False):
    context = _descriptive_context()
    concept = "age" if typed_bound else "entry_measure"
    question = (
        "Among eligible stays, what share has an entry value at least 47 and what is its outcome?"
        if grouping else
        "Among adults with an entry value at least 47, describe outcomes within that population."
    )
    parameters = {
        "concept": concept, "window": None, "scale": "ordinal",
        "groups": [
            {"id": "g1", "rule": {"summary": "value", "op": "<", "value": 47.0}},
            {"id": "g2", "rule": {"summary": "value", "op": ">=", "value": 47.0}},
        ],
        "unmeasured": "exclude", "source_columns": {"value": concept},
    }
    provenance = dict(context.cohort.provenance)
    if grouping:
        provenance[EXPOSURE_GROUP_LABELS_KEY] = [{
            "variable": "entry_band", "concept": concept, "scale": "ordinal",
            "parameters": parameters,
            "levels": [{"code": 1, "group": "g1", "label": "Lower"},
                       {"code": 2, "group": "g2", "label": "Upper"}],
        }]
    constraints = json.loads(context.user_preferences.data_constraints)
    constraints["cohort"] = {"age_min": 47 if typed_bound else 18}
    return context.model_copy(update={
        "research_question": question,
        "primary_exposure": "entry_band" if grouping else "phenotype_flag",
        "variables": [*context.variables,
            ConceptDescriptor(name="entry_measure", role=VariableRole.DEMOGRAPHIC, dtype="float64"),
            ConceptDescriptor(name="entry_band", role=VariableRole.DEMOGRAPHIC, dtype="int64",
                              unit_normalization="exposure_group_ordinal", valid_range=(1, 2)),
        ],
        "cohort": context.cohort.model_copy(update={"provenance": provenance}),
        "user_preferences": context.user_preferences.model_copy(update={
            "data_constraints": json.dumps(constraints),
        }),
    })


def _predicate(concept="entry_measure", *, op=">=", value=47):
    return {
        "concept_id": concept, "anchor": "icu_admission", "start_offset_hours": 0,
        "end_offset_hours": 24, "aggregation": "first", "op": op,
        "value": {"mode": "number", "number_value": value},
    }


def _request(context):
    variables = select_progressive_variables(context)
    return build_family_spec_request(
        context, analysis_types=["descriptive_epidemiology"], variable_roster=variables,
        allowed_literature_citation_keys=ALLOWED_CITATIONS,
        direct_comparator_literature_keys=DIRECT_COMPARATORS,
        comparison_literature_keys=DIRECT_COMPARATORS,
        required_primary_cohort_selection_mode="predicate_filtered",
        cohort_concept_ids=progressive_cohort_concept_ids(context, variables),
    )


def _payload(context, *, side="inclusion", reading=True, typed_bound=False):
    request = _request(context)
    return {
        "schema_version": "easyicu.family_plan_spec/1", "family_id": request.family_id,
        "request_sha256": request.request_sha256, "adjustment_set": [],
        "baseline_variables": ["age"],
        "distribution_missing_policies": dict(EXCLUDE_MISSING),
        "reader_display_labels": [
            {"key": key, "value": "Clinical " + key.replace("_", " ")}
            for key in dict.fromkeys([*request.required_reader_label_keys, *request.level_label_keys])
        ],
        "comparator_applications": [
            {"citation_key": key, "application": "Compare population and outcome definitions."}
            for key in DIRECT_COMPARATORS
        ],
        "roster_decision_note": "No additional baseline variables requested.",
        "question_requirements": [{
            "id": "r1", "kind": "estimand", "quote": context.research_question,
            "concepts": ["age" if typed_bound else "entry_measure", "entry_band", "death"]
            if reading else ["death"],
            "coverage": "plan",
        }],
        "population": None if typed_bound else {
            "criteria": [{"criterion": "entry value at least 47", "concept_ids": ["entry_measure"]}],
            "inclusion": [_predicate()] if side == "inclusion" else [],
            "exclusion": [_predicate(op="<")] if side == "exclusion" else [],
        },
    }


@pytest.mark.parametrize("side", ["inclusion", "exclusion"])
def test_share_keeps_both_levels_and_independent_adult_eligibility(side):
    context = _study()
    _, result = _run(context, [json.dumps(_payload(context, side=side))])
    plan = result.output
    assert [(p.concept_id, p.value) for p in plan.cohort.inclusion] == [("age", 18)]
    assert not plan.cohort.exclusion
    assert not population_emptied_exposure_levels(context, plan)
    review = build_plan_scientific_review(context=context, plan=plan)
    assert "POPULATION_EMPTIES_EXPOSURE_LEVEL" not in {f.code for f in review.findings}
    note = plan.design_selection.selected.reviewable_plan[0]
    assert all(token in note for token in ("entry_measure", "47", "entry_band", "g1"))
    assert "removed" in note.lower()
    assert plan.steps[0].cohort_definition_spec.eligibility_criteria
    assert all("entry_measure" not in c.description
               for c in plan.steps[0].cohort_definition_spec.eligibility_criteria)


def test_restricted_population_with_no_grouping_on_its_concept_is_kept():
    context = _study(grouping=False)
    _, result = _run(context, [json.dumps(_payload(context, reading=False))])
    assert [(p.concept_id, p.value) for p in result.output.cohort.inclusion] == [
        ("age", 18), ("entry_measure", 47),
    ]
    assert not population_emptied_exposure_levels(context, result.output)


def test_outcome_only_within_a_restricted_population_declares_no_exposure_levels():
    context = _study(grouping=False).model_copy(update={"primary_exposure": None})
    skeleton = ProgressivePlanSkeleton(
        analysis_type="descriptive_epidemiology",
        rationale="Describe outcomes within the stated population without an exposure contrast.",
        cohort={"name": "restricted", "selection_mode": "predicate_filtered",
                "inclusion": [_predicate("age", value=18), _predicate()]},
        display_labels=[{"key": "death", "value": "In-hospital outcome"}],
        steps=[
            {"step_id": "cohort", "planned_analysis_role": "auxiliary",
             "module_id": "cohort_definition", "objective": "Apply independent adult and entry eligibility.",
             "raw_inputs": ["age", "entry_measure"]},
            {"step_id": "outcome", "planned_analysis_role": "primary",
             "module_id": "custom_analysis", "scientific_action_id": "descriptive.descriptive_summary",
             "objective": "Describe the outcome only within eligible stays.", "depends_on": ["cohort"],
             "raw_inputs": ["death"], "custom_method": "univariate_descriptive_summary",
             "product_inputs": [{"producer_step_id": "cohort", "product_id": "artifact:analysis_cohort"}],
             "outputs": [{"product_id": "table:cohort_summary", "semantic_role": "custom"},
                         {"product_id": "table:distribution_summary", "semantic_role": "custom"}]},
        ],
    )
    plan, _ = compile_progressive_plan(skeleton=skeleton, context=context)
    assert [(p.concept_id, p.value) for p in plan.cohort.inclusion] == [
        ("age", 18), ("entry_measure", 47),
    ]
    assert all(step.exposure_outcome_distribution_spec is None for step in plan.steps)
    assert not population_emptied_exposure_levels(context, plan)


def test_an_unresolved_conflict_stops_before_review_without_dropping_eligibility():
    context = _study()
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _run(context, [json.dumps(_payload(context, reading=False))])
    assert caught.value.reason_code == "declared_exposure_level_empty"


@pytest.mark.parametrize("concepts", [["entry_band", "death"], ["entry_measure", "death"]])
def test_a_reading_must_bind_both_the_source_concept_and_its_grouping(concepts):
    context = _study()
    payload = _payload(context)
    payload["question_requirements"][0]["concepts"] = concepts
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _run(context, [json.dumps(payload)])
    assert caught.value.reason_code == "declared_exposure_level_empty"


@pytest.mark.parametrize("kind,coverage", [("definition", "definition_only"), ("analysis", "plan")])
def test_reading_a_concept_without_requesting_its_grouping_cannot_remove_eligibility(kind, coverage):
    context = _study()
    payload = _payload(context)
    payload["question_requirements"][0].update(kind=kind, coverage=coverage)
    if kind == "definition":
        payload["question_requirements"][0]["note"] = "Defines the eligible population only."
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _run(context, [json.dumps(payload)])
    assert caught.value.reason_code == "declared_exposure_level_empty"


def test_an_independent_compatible_bound_on_the_same_concept_survives():
    context = _study()
    payload = _payload(context)
    payload["population"]["inclusion"].append(_predicate(value=19))
    _, result = _run(context, [json.dumps(payload)])
    assert [(p.concept_id, p.value) for p in result.output.cohort.inclusion] == [
        ("age", 18), ("entry_measure", 19),
    ]
    assert not population_emptied_exposure_levels(context, result.output)


def test_compatible_eligibility_does_not_need_a_grouping_reading():
    context = _study()
    payload = _payload(context, reading=False)
    payload["population"]["inclusion"] = [_predicate(value=19)]
    payload["population"]["criteria"][0]["criterion"] = "entry value at least 19"
    _, result = _run(context, [json.dumps(payload)])
    assert [(p.concept_id, p.value) for p in result.output.cohort.inclusion] == [
        ("age", 18), ("entry_measure", 19),
    ]
    assert "removed" not in result.output.design_selection.selected.reviewable_plan[0].lower()


def test_replay_of_host_bound_foundation_removes_only_the_conflicting_bound():
    # Real JSON shape: no population spec; the family request's typed bound
    # becomes a foundation predicate beside an ordinal grouping of that concept.
    # Names, threshold and question are synthetic; there are no patient rows.
    context = _study(typed_bound=True)
    _, result = _run(context, [json.dumps(_payload(context, typed_bound=True))])
    assert result.output.cohort.selection_mode == "all_input_rows"
    assert not result.output.cohort.inclusion
    assert "removed" in result.output.design_selection.selected.reviewable_plan[0].lower()


@pytest.mark.parametrize("population_spec", [False, True])
def test_compiler_refuses_a_conflict_even_when_family_reconciliation_is_bypassed(population_spec):
    context = _study(typed_bound=True)
    draft = build_descriptive_skeleton(
        _request(context), spec_from_mapping(_payload(context, typed_bound=True)),
    )
    skeleton = assemble_progressive_skeleton(
        outline=draft.outline, foundation=draft.foundation.foundation,
        steps=[item.step for item in draft.materializations],
    )
    if population_spec:
        skeleton = skeleton.model_copy(update={"cohort": skeleton.cohort.model_copy(update={
            "population_spec": {
                "criteria": [{"id": "c1", "kind": "age_years", "min_years": 47, "role": "include",
                              "quote": "entry value at least 47", "source": "question"}],
            },
        })})
    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=skeleton, context=context, allowed_literature_citation_keys=ALLOWED_CITATIONS,
        )
    assert caught.value.reason_code == "declared_exposure_level_empty"
