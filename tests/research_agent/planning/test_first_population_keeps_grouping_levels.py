"""First-plan population/grouping reconciliation uses typed readings, never rows."""

import json

from tests.support.distribution_policies import EXCLUDE_MISSING

import pytest

from easyicu.research_agent.agents.progressive_planner import select_progressive_variables
from easyicu.research_agent.agents.progressive_planner import _MAX_STEP_PARSE_RETRIES
from easyicu.research_agent.providers.structured_retry import StructuredResponseFailure
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
    ProgressiveCohortPredicate,
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


# Wording without a population share: only a typed reading can decide it.
_NOT_A_SHARE = "Among eligible stays, how do stays with an entry value at least 47 fare, and what is their outcome?"


def _study(*, grouping=True, typed_bound=False, question=None, measure="entry_measure", cut=47):
    context = _descriptive_context()
    concept = "age" if typed_bound else measure
    question = question or (
        "Among eligible stays, what share has an entry value at least 47 and what is its outcome?"
        if grouping else
        "Among adults with an entry value at least 47, describe outcomes within that population."
    )
    parameters = {
        "concept": concept, "window": None, "scale": "ordinal",
        "groups": [
            {"id": "g1", "rule": {"summary": "value", "op": "<", "value": float(cut)}},
            {"id": "g2", "rule": {"summary": "value", "op": ">=", "value": float(cut)}},
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
            ConceptDescriptor(name=measure, role=VariableRole.DEMOGRAPHIC, dtype="float64"),
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


def _payload(context, *, side="inclusion", reading=True, typed_bound=False, measure="entry_measure", cut=47):
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
            "concepts": ["age" if typed_bound else measure, "entry_band", "death"]
            if reading else ["death"],
            "coverage": "plan",
        }],
        "population": None if typed_bound else {
            "criteria": [{"criterion": f"entry value at least {cut}", "concept_ids": [measure]}],
            "inclusion": [_predicate(measure, value=cut)] if side == "inclusion" else [],
            "exclusion": [_predicate(measure, op="<", value=cut)] if side == "exclusion" else [],
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
    context = _study(question=_NOT_A_SHARE)
    _expect_exhausted_empty_level(context, _payload(context, reading=False))


def _expect_exhausted_empty_level(context, payload):
    with pytest.raises(StructuredResponseFailure) as caught:
        _run(context, [json.dumps(payload)] * (_MAX_STEP_PARSE_RETRIES + 1))
    assert caught.value.__cause__.reason_code == "declared_exposure_level_empty"
    assert len(caught.value.easyicu_structured_attempt_metadata) == _MAX_STEP_PARSE_RETRIES + 1


def test_a_real_empty_level_refusal_can_be_repaired_in_the_same_family_attempt():
    context = _study(question=_NOT_A_SHARE)
    llm, result = _run(context, [
        json.dumps(_payload(context, reading=False)), json.dumps(_payload(context)),
    ])
    assert len(llm.calls) == 2
    assert not population_emptied_exposure_levels(context, result.output)
    feedback = llm.calls[1][0][-1].content
    assert "declared_exposure_level_empty" in feedback
    assert "g1" in feedback and "code=1" in feedback and "Lower" in feedback
    assert "Observed level counts: unavailable" in feedback
    # The feedback names the one move that resolves it: the host's own
    # reconciliation, triggered by a reading that requests the grouping.
    # Both sides and both options are named; only (a) is the Planner's to take.
    assert "Conflicting pair: population restriction on 'entry_measure' versus the declared grouping 'entry_band'" in feedback
    assert "(a) drop the population restriction and keep the grouping" in feedback
    assert "list both 'entry_measure' and 'entry_band'" in feedback
    assert "(b) keep the restriction and drop the grouping" in feedback
    assert "the Planner must not choose (b)" in feedback


@pytest.mark.parametrize("concepts", [["entry_band", "death"], ["entry_measure", "death"]])
def test_a_reading_must_bind_both_the_source_concept_and_its_grouping(concepts):
    context = _study(question=_NOT_A_SHARE)
    payload = _payload(context)
    payload["question_requirements"][0]["concepts"] = concepts
    _expect_exhausted_empty_level(context, payload)


@pytest.mark.parametrize("kind,coverage", [("definition", "definition_only"), ("analysis", "plan")])
def test_reading_a_concept_without_requesting_its_grouping_cannot_remove_eligibility(kind, coverage):
    context = _study(question=_NOT_A_SHARE)
    payload = _payload(context)
    payload["question_requirements"][0].update(kind=kind, coverage=coverage)
    if kind == "definition":
        payload["question_requirements"][0]["note"] = "Defines the eligible population only."
    _expect_exhausted_empty_level(context, payload)


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
    else:
        # The request now resolves the saved bound before projecting it. Inject
        # the conflict explicitly to exercise bypassing reconciliation.
        skeleton = skeleton.model_copy(update={"cohort": skeleton.cohort.model_copy(update={
            "selection_mode": "predicate_filtered", "inclusion": [
                ProgressiveCohortPredicate.model_validate(_predicate("age"))
            ],
        })})
    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=skeleton, context=context, allowed_literature_citation_keys=ALLOWED_CITATIONS,
        )
    assert caught.value.reason_code == "declared_exposure_level_empty"


@pytest.mark.parametrize("measure,cut,question", [
    # The dH4 shape (a share of the whole population, then the level's outcome),
    # in Chinese, and two other variables with their own declared groupings.
    ("entry_measure", 47, "入 ICU 时 entry value at least 47 的住院占多大比例？他们的院内死亡率是多少？"),
    ("sofa_score_x", 10, "What proportion of ICU stays have an entry value at least 10, and what is their mortality?"),
    ("lactate_x", 4, "乳酸 entry value at least 4 的患者占比是多少，死亡率如何？"),
])
def test_a_share_question_keeps_every_declared_level_without_a_grouping_reading(measure, cut, question):
    from easyicu.research_agent.agents.progressive_planner import FAMILY_SPEC_STRATEGY, ProgressivePlannerAgent
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    context = _study(question=question, measure=measure, cut=cut)
    # These zero-row synthetic catalogs declare no patient clustering. Their
    # requested counts and proportions therefore carry no interval inference.
    constraints = json.loads(context.user_preferences.data_constraints)
    constraints["analysis_design"] = {
        "analysis_family": "descriptive_epidemiology", "analysis_unit": "icu_stay",
        "variance_estimator": "none_counts_only",
    }
    context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(update={
        "data_constraints": json.dumps(constraints),
    })})
    payload = _payload(context, reading=False, measure=measure, cut=cut)
    payload["comparator_applications"] = []
    payload["baseline_variables"] = []  # Only the requested share and outcome rates.
    llm = ScriptedMockLLMClient([json.dumps(payload)])
    result = ProgressivePlannerAgent(llm).run_attempt(
        context, planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=[], direct_comparator_literature_keys=[],
        comparison_literature_keys=[], required_primary_cohort_selection_mode="predicate_filtered",
        enforce_article_contract=False,
    )
    assert len(llm.calls) == 1  # decided by the host, not by another Planner attempt
    plan = result.output
    assert [(p.concept_id, p.value) for p in plan.cohort.inclusion] == [("age", 18)]
    assert not population_emptied_exposure_levels(context, plan)
    note = plan.design_selection.selected.reviewable_plan[0]
    assert "share of the whole population" in note and "removed" in note.lower()
    review = build_plan_scientific_review(context=context, plan=plan)
    assert review.approval_allowed, [(f.code, f.message) for f in review.findings if f.severity == "blocker"]


def test_a_share_question_keeps_a_compatible_restriction_on_the_same_concept():
    # The share wording removes only a restriction that empties a declared level.
    context = _study(question=_NOT_A_SHARE.replace("how do stays", "what share of stays"))
    payload = _payload(context, reading=False)
    payload["population"]["inclusion"] = [_predicate(value=19)]
    payload["population"]["criteria"][0]["criterion"] = "entry value at least 19"
    _, result = _run(context, [json.dumps(payload)])
    assert ("entry_measure", 19) in [(p.concept_id, p.value) for p in result.output.cohort.inclusion]


def test_share_request_absorbs_saved_bound_before_projecting_typed_bounds():
    context = _study(typed_bound=True)
    request = _request(context)
    assert request.age_min is None
    assert request.cohort_selection_mode == "all_input_rows"
    assert request.population_grouping_note
    assert "entry_band" in request.population_grouping_note


@pytest.mark.parametrize("remaining", [False, True])
def test_caller_bound_mode_uses_the_same_absorbed_saved_restrictions(remaining):
    from easyicu.research_agent.planning.family_spec import caller_bound_population_conflict
    from easyicu.research_agent.cohort.schema import CohortDefinition
    from types import SimpleNamespace

    context = _study(typed_bound=True)
    if remaining:
        constraints = json.loads(context.user_preferences.data_constraints)
        constraints["cohort"]["min_icu_los_hours"] = 6
        context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(
            update={"data_constraints": json.dumps(constraints)},
        )})
    plan = SimpleNamespace(cohort=CohortDefinition(name="all", selection_mode="all_input_rows"))
    conflict = caller_bound_population_conflict(plan, context=context, required_selection_mode="predicate_filtered")
    assert (conflict is not None) == remaining


def test_share_reconciles_a_typed_spec_and_keeps_the_other_criterion_and_quote():
    context = _study(typed_bound=True)
    context = context.model_copy(update={"cohort": context.cohort.model_copy(update={
        "inclusion_criteria": ["age at least 19"],
    })})
    payload = _payload(context, typed_bound=True, reading=False)
    payload["population"] = {"population_spec": {"criteria": [
        {"id": "c1", "kind": "age_years", "role": "include", "source": "question",
         "quote": "entry value at least 47", "min_years": 47},
        {"id": "c2", "kind": "age_years", "role": "include", "source": "study_wording",
         "quote": "age at least 19", "min_years": 19},
    ]}}
    _, result = _run(context, [json.dumps(payload)])
    assert [(p.concept_id, p.value) for p in result.output.cohort.inclusion] == [("age", 19)]
    kept = result.facts.foundation.foundation.cohort.population_spec["criteria"]
    assert [(c["id"], c["quote"]) for c in kept] == [("c2", "age at least 19")]
    assert "47" in result.output.design_selection.selected.reviewable_plan[0]


@pytest.mark.parametrize("sequence", [f"v24{letter}" for letter in "abcdefgh"])
@pytest.mark.parametrize("caller_mode", ["predicate_filtered", "all_input_rows"])
def test_recorded_public_demo_share_answers_reach_an_acceptable_plan(sequence, caller_mode, tmp_path):
    from pathlib import Path

    from easyicu.research_agent.agents.progressive_planner import FAMILY_SPEC_STRATEGY, ProgressivePlannerAgent
    from easyicu.research_agent.planning.family_spec import caller_bound_population_conflict
    from easyicu.research_agent.planning.population_shadow import population_cohort_audit
    from easyicu.research_agent.planning.progressive_compiler import stated_population
    from easyicu.research_agent.orchestration.progressive_planning import question_requirement_outcome
    from easyicu.research_agent.planning.question_requirements import question_requirements_on_plan_under_review
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from easyicu.research_agent.research_context.typed import parse_research_context_json

    fixture = json.loads((Path(__file__).parent / "fixtures/population_share_demo.json").read_text())
    context = parse_research_context_json(json.dumps(fixture["context"]))
    llm = ScriptedMockLLMClient(fixture["responses"][sequence])
    result = ProgressivePlannerAgent(llm).run_attempt(
        context, planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=[], direct_comparator_literature_keys=[],
        comparison_literature_keys=[], required_primary_cohort_selection_mode=caller_mode,
        enforce_article_contract=False, planning_contract_context="execution_tier=first_result",
    )
    plan = result.output
    assert result.facts.compile_receipt is not None
    prompt = "\n".join(message.content for message in llm.calls[0][0])
    assert "The population is all input stays" in prompt
    assert "grouping retains the level of interest" in prompt
    assert plan.cohort.selection_mode == "all_input_rows"
    assert not plan.cohort.inclusion and not plan.cohort.exclusion
    assert caller_bound_population_conflict(
        plan, context=context, required_selection_mode="predicate_filtered",
    ) is None
    assert not population_emptied_exposure_levels(context, plan)
    distribution = next(s.exposure_outcome_distribution_spec for s in plan.steps
                        if s.exposure_outcome_distribution_spec is not None)
    assert distribution.exposure == "age_group_x1" and list(distribution.exposure_levels) == [1, 2]
    note = plan.design_selection.selected.reviewable_plan[0]
    assert "Host-decided" in note and "80" in note and "age_group_x1" in note
    cohort = result.facts.foundation.foundation.cohort
    compiled = stated_population(cohort, context=context, plan=plan)
    assert compiled is None or not compiled.blocking
    audit = population_cohort_audit(context=context, plan=plan, cohort=cohort)
    assert audit["host_population_grouping_resolution"]["decision"] == "host_decided"
    assert audit["host_population_grouping_resolution"]["absorbed_study_fields"] == ["age_min"]
    review = build_plan_scientific_review(context=context, plan=plan)
    assert review.approval_allowed
    assert not {"POPULATION_EMPTIES_EXPOSURE_LEVEL", "POPULATION_CRITERION_NOT_APPLIED"}.intersection(
        f.code for f in review.findings
    )
    assert review.facts["host_population_grouping_resolution"]["decision"] == "host_decided"
    findings = question_requirement_outcome(context=context, plan=plan, facts=result.facts, run_dir=tmp_path)
    reviewed = question_requirements_on_plan_under_review(findings, plan=plan, run_dir=tmp_path)
    assert not [finding for finding in reviewed if finding.severity == "error"], reviewed


@pytest.mark.parametrize("change", ["no_grouping", "compatible", "not_share"])
def test_a_saved_bound_only_yields_to_a_share_and_a_conflicting_grouping(change):
    from easyicu.research_agent.planning.cohort_predicate_domain import population_grouping_resolution

    context = _study(typed_bound=True)
    if change == "no_grouping":
        context = context.model_copy(update={"cohort": context.cohort.model_copy(update={"provenance": {}})})
    elif change == "compatible":
        constraints = json.loads(context.user_preferences.data_constraints)
        constraints["cohort"]["age_min"] = 19
        context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(update={
            "data_constraints": json.dumps(constraints),
        })})
    else:
        context = context.model_copy(update={"research_question": _NOT_A_SHARE})
    decision = population_grouping_resolution(context)
    assert not decision.absorbed_fields
    assert decision.selection_mode("predicate_filtered") == "predicate_filtered"
    assert _request(context).age_min == (19 if change == "compatible" else 47)


def test_a_nonshare_mode_conflict_reaches_the_failure_receipt_with_a_typed_code(tmp_path):
    from types import SimpleNamespace

    from easyicu.research_agent.cohort.schema import CohortAuthorityError, CohortDefinition
    from easyicu.research_agent.planning.family_spec import caller_bound_population_conflict
    from easyicu.webserver.agent_pipeline_runs import (
        _pipeline_failure_code, _record_pipeline_failure, _safe_pipeline_typed_failure,
    )

    context = _study(typed_bound=True, question=_NOT_A_SHARE)
    plan = SimpleNamespace(cohort=CohortDefinition(name="all", selection_mode="all_input_rows"))
    conflict = caller_bound_population_conflict(plan, context=context, required_selection_mode="predicate_filtered")
    assert conflict
    error = CohortAuthorityError(conflict, reason_code="caller_bound_population_mode_conflict")
    wrapper = RuntimeError("private exception text")
    wrapper.__cause__ = error
    code = _pipeline_failure_code(wrapper)
    assert code == "caller_bound_population_mode_conflict"
    _record_pipeline_failure(wrapper_dir=tmp_path, study={"id": "synthetic"}, provider={},
                             exc=wrapper, code=code, execution_retry_id=None)
    receipt = json.loads((tmp_path / "diagnostics/research_pipeline_failure.json").read_text())
    assert receipt["typed_failure"] == _safe_pipeline_typed_failure(wrapper) == {
        "owner": "easyicu.planning.cohort_authority_v1", "reason_code": code,
    }
    assert "private exception text" not in json.dumps(receipt["typed_failure"])


def test_a_compatible_saved_upper_age_bound_survives_the_absorbed_lower_bound():
    context = _study(typed_bound=True)
    constraints = json.loads(context.user_preferences.data_constraints)
    constraints["cohort"]["age_max"] = 120
    context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(update={
        "data_constraints": json.dumps(constraints),
    })})
    request = _request(context)
    assert request.age_min is None and request.age_max == 120
    assert request.cohort_selection_mode == "predicate_filtered"
    payload = _payload(context, typed_bound=True)
    _, result = _run(context, [json.dumps(payload)])
    assert [(p.concept_id, p.op, p.value) for p in result.output.cohort.inclusion] == [("age", "<=", 120)]


def test_an_additional_saved_population_setting_cannot_be_assumed_absorbed():
    from easyicu.research_agent.planning.cohort_predicate_domain import population_grouping_resolution

    context = _study(typed_bound=True)
    constraints = json.loads(context.user_preferences.data_constraints)
    constraints["cohort"]["preset"] = "adult_first"
    context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(update={
        "data_constraints": json.dumps(constraints),
    })})
    decision = population_grouping_resolution(context)
    assert decision.absorbed_fields == ("age_min",)
    assert decision.remaining_fields == ("preset",)
    assert decision.selection_mode("predicate_filtered") == "predicate_filtered"


def test_absorption_preserves_a_compatible_bound_in_the_same_typed_criterion():
    context = _study(typed_bound=True)
    constraints = json.loads(context.user_preferences.data_constraints)
    constraints["cohort"]["age_max"] = 90
    context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(update={
        "data_constraints": json.dumps(constraints),
    })})
    payload = _payload(context, typed_bound=True)
    payload["population"] = {"population_spec": {"criteria": [{
        "id": "c1", "kind": "age_years", "role": "include", "source": "question",
        "quote": "entry value at least 47", "min_years": 47, "max_years": 90,
    }]}}
    _, result = _run(context, [json.dumps(payload)])
    criterion, = result.facts.foundation.foundation.cohort.population_spec["criteria"]
    assert criterion["id"] == "c1" and criterion["quote"] == "entry value at least 47"
    assert criterion["min_years"] is None and criterion["max_years"] == 90
    assert [(p.concept_id, p.op, p.value) for p in result.output.cohort.inclusion] == [("age", "<=", 90)]


def test_a_compound_typed_condition_keeps_its_other_concept_and_original_quote():
    question = "What share of eligible stays has the phenotype and auxiliary condition, and what is their outcome?"
    context = _study(measure="phenotype_flag", cut=0.5, question=question)
    variables = [v for v in context.variables if v.name != "phenotype_flag"]
    variables.extend(ConceptDescriptor(
        name=name, source_concept=name, role=VariableRole.OTHER, dtype="int64",
        observed_domain={"n_unique": 2, "is_binary": True, "levels": [0, 1]},
        analysis_window="icu_admission[0,24]h", unit_normalization="window_presence_max",
    ) for name in ("phenotype_flag", "auxiliary_flag"))
    provenance = dict(context.cohort.provenance)
    grouping = json.loads(json.dumps(provenance[EXPOSURE_GROUP_LABELS_KEY][0]))
    grouping["parameters"]["window"] = {"start_hours": 0, "end_hours": 24}
    grouping["parameters"]["source_columns"] = {"max": "phenotype_flag"}
    for group in grouping["parameters"]["groups"]:
        group["rule"]["summary"] = "max"
    provenance[EXPOSURE_GROUP_LABELS_KEY] = [grouping]
    context = context.model_copy(update={"variables": variables,
        "cohort": context.cohort.model_copy(update={"provenance": provenance})})
    payload = _payload(context, reading=False, measure="phenotype_flag", cut=0.5)
    payload["population"] = {"population_spec": {"criteria": [
        {"id": "c1", "kind": "age_years", "role": "include", "source": "preset",
         "quote": "independent adult eligibility", "min_years": 18},
        {"id": "c2", "kind": "condition_present", "role": "include", "source": "question",
         "quote": "phenotype and auxiliary condition", "concepts_all_of": ["phenotype_flag", "auxiliary_flag"],
         "window": {"start_hours": 0, "end_hours": 24}},
    ]}}
    _, result = _run(context, [json.dumps(payload)])
    kept = result.facts.foundation.foundation.cohort.population_spec["criteria"]
    assert kept[1]["concepts_all_of"] == ["auxiliary_flag"]
    assert kept[1]["quote"] == "phenotype and auxiliary condition"
    assert [(p.concept_id, p.value) for p in result.output.cohort.inclusion] == [("age", 18), ("auxiliary_flag", 1)]
    assert not population_emptied_exposure_levels(context, result.output)


@pytest.mark.parametrize("unit,cut", [("hours", 6), ("days", 0.25)])
def test_a_saved_icu_duration_bound_is_absorbed_only_by_its_own_grouping(unit, cut):
    from easyicu.research_agent.planning.family_spec import caller_bound_population_conflict

    context = _study(measure="los_icu", cut=cut,
        question="What proportion of stays last at least six ICU hours, and what is their mortality?")
    constraints = json.loads(context.user_preferences.data_constraints)
    constraints["cohort"] = {"min_icu_los_hours": 6}
    context = context.model_copy(update={
        "variables": [v.model_copy(update={"unit": unit}) if v.name == "los_icu" else v for v in context.variables],
        "user_preferences": context.user_preferences.model_copy(update={"data_constraints": json.dumps(constraints)}),
    })
    request = _request(context)
    assert request.minimum_icu_hours is None and request.cohort_selection_mode == "all_input_rows"
    _, result = _run(context, [json.dumps(_payload(context, reading=False, measure="los_icu", cut=cut))])
    assert result.output.cohort.selection_mode == "all_input_rows"
    assert not result.output.cohort.inclusion
    assert not population_emptied_exposure_levels(context, result.output)
    assert caller_bound_population_conflict(
        result.output, context=context, required_selection_mode="predicate_filtered",
    ) is None
