"""The question's adult population is not just a cohort display label."""

import pytest

from easyicu.research_agent.planning.population_question_completeness import explicit_adult_population
from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.orchestration.progressive_planning import population_approval_findings
from easyicu.research_agent.planning.progressive_compiler import validate_progressive_foundation
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from tests.research_agent.planning.test_a_population_spec_compiles_to_the_cohort_it_states import _recorded_ctx
from tests.research_agent.planning.test_a_stated_population_is_applied_by_its_predicates import (
    _context, _foundation, _predicate,
)


def _cohort(spec=None, inclusion=()):
    value = {"name": "Adults", "selection_mode": "predicate_filtered" if inclusion else "all_input_rows",
             "inclusion": list(inclusion), "exclusion": []}
    if spec is not None:
        value["population_spec"] = {"criteria": spec}
    return value


def _age(minimum=18):
    return {"id": "c1", "kind": "age_years", "min_years": minimum,
            "max_years": None, "quote": "成人患者", "source": "question", "role": "include"}


def _validate(context, cohort):
    validate_progressive_foundation(_foundation(cohort), context=context, analysis_type="descriptive_epidemiology")


def _question_context():
    return _context().model_copy(update={"research_question": "研究成人患者 ICU 首日乳酸与住院时长的关系"})


@pytest.mark.parametrize("question", [
    "我想用 SICdb 研究成人患者入住 ICU 后第一天的乳酸水平",
    "Among adult patients admitted to ICU, assess lactate.",
    "Among adults with shock, assess lactate.",
])
def test_recognizes_only_the_explicit_adult_words(question):
    assert explicit_adult_population(question)


@pytest.mark.parametrize("question", [
    "比较成人患者与儿童患者的乳酸水平", "排除成人患者后研究儿童患者", "研究非成人患者",
    "Compare adult patients versus children", "Exclude adults from the cohort",
    "Not only adult patients but also older children", "Adult patients compared with pediatric patients",
])
def test_comparisons_and_negations_do_not_make_all_rows_adults(question):
    assert explicit_adult_population(question) is None


@pytest.mark.parametrize("cohort", [_cohort(), _cohort([]), _cohort([_age(1)])])
def test_missing_or_inadequate_age_condition_fails_closed(cohort):
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _validate(_question_context(), cohort)
    assert caught.value.reason_code == "progressive_stated_adult_population_missing"
    assert "成人患者" in str(caught.value)


def test_proposed_typed_age_and_legacy_predicate_are_accounted_for():
    context = _question_context()
    _validate(context, _cohort([_age()]))
    _validate(context, _cohort(inclusion=[_predicate("age_years", ">=", 18, "first")]))


def test_verified_source_age_scope_can_satisfy_the_question():
    context = _recorded_ctx(executed={"age_min": 18, "age_max": 100}).model_copy(
        update={"research_question": "研究成人患者的 ICU 结局"})
    _validate(context, _cohort())


def test_an_unbound_source_claim_cannot_satisfy_the_question():
    context = _recorded_ctx(executed={"age_min": 18, "age_max": 100}, basis="unrecorded").model_copy(
        update={"research_question": "研究成人患者的 ICU 结局"})
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _validate(context, _cohort())
    assert caught.value.reason_code == "progressive_stated_adult_population_missing"


def test_untyped_adult_criterion_remains_visible_and_cannot_be_approved():
    context = _question_context()
    raw = [{"id": "c1", "kind": "not_typed", "quote": "成人患者", "source": "question",
            "role": "include", "why": "Source age is unavailable for this population."}]
    _validate(context, _cohort(raw))
    findings = population_approval_findings(compile_population(PopulationSpec.model_validate({"criteria": raw}), context))
    assert findings and all(item.detail["approval_allowed"] is False for item in findings)


def test_unverified_sicdb_age_does_not_become_an_exact_adult_boundary():
    context = _question_context()
    context = context.model_copy(update={"cohort": context.cohort.model_copy(update={"provenance": {
        "database_capability_binding": {"approximate_covariates": {"age": {
            "coding": "unverified", "interpretation": "rounded to +/-5 years"}}}}})})
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _validate(context, _cohort([_age()]))
    assert caught.value.reason_code == "progressive_adult_age_source_unverified"
    assert "approximate" in str(caught.value)


def test_approximate_age_binding_uses_the_typed_source_concept_for_aliases():
    context = _question_context()
    variables = [variable.model_copy(update={"name": "admission_age", "source_concept": "age"}) if variable.name == "age_years" else variable
                 for variable in context.variables]
    context = context.model_copy(update={"variables": variables, "cohort": context.cohort.model_copy(update={"provenance": {
        "database_capability_binding": {"approximate_covariates": {"admission_age": {
            "coding": "unverified", "interpretation": "rounded age"}}}}})})
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _validate(context, _cohort([_age()]))
    assert caught.value.reason_code == "progressive_adult_age_source_unverified"


def test_approximate_unrelated_weight_does_not_become_an_age_restriction():
    context = _question_context()
    context = context.model_copy(update={"cohort": context.cohort.model_copy(update={"provenance": {
        "database_capability_binding": {"approximate_covariates": {"weight": {"coding": "unverified"}}}}})})
    _validate(context, _cohort([_age()]))


def test_sicdb_profile_blocks_verified_prefilter_when_no_age_column_remains():
    context = _recorded_ctx(executed={"age_min": 18, "age_max": 100}).model_copy(
        update={"research_question": "研究成人患者的 ICU 结局"})
    context = context.model_copy(update={
        "cohort": context.cohort.model_copy(update={"database": "sic"}),
        "variables": [variable for variable in context.variables
                      if variable.name not in {"age", "age_years"} and variable.source_concept != "age"],
    })
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _validate(context, _cohort())
    assert caught.value.reason_code == "progressive_adult_age_source_unverified"
