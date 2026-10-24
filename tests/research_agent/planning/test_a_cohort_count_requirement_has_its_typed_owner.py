"""Only the typed cohort owner answers a synthetic attrition requirement."""
from __future__ import annotations

import json

import pytest

from easyicu.research_agent.planning.question_requirements import (
    QuestionRequirement,
    judge_question_requirements,
)
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole
from tests.support.descriptive_plan import _payload

from .family_spec_fixtures import _descriptive_context, _request, _run


def _synthetic_plan():
    from tests.support.descriptive_plan import _plan

    return _plan()


def _requirement(kind='cohort_accounting'):
    return QuestionRequirement(id='r3', kind=kind, quote='按原因报告排除人数',
        concepts=['stay_id', 'icu_unit_type'], coverage='plan')


def _judge(plan, requirement=None):
    return judge_question_requirements([requirement or _requirement()], plan=plan,
        family_template=True, relatives=lambda name: frozenset([name]),
        check_gap=lambda gap: ('unverified', 'No capability gap asserted'))[0]


def _accounting_plan():
    plan = _synthetic_plan()
    return plan.model_copy(update={'steps': [
        step.model_copy(update={'inputs': [*step.inputs, 'icu_unit_type']})
        if step.step_id == 'cohort_accounting' else step for step in plan.steps]})


def test_generic_analysis_requirement_remains_unanswered():
    assert _judge(_synthetic_plan(), _requirement('analysis')).disposition == 'not_covered'


def test_only_the_typed_cohort_and_flow_owner_answers_attrition():
    result = _judge(_accounting_plan())
    assert result.disposition == 'covered'
    assert result.owner_step_ids == ('cohort_accounting',)
    assert result.host_judged is True


def test_the_declared_row_identity_is_read_for_cohort_accounting():
    plan = _accounting_plan()
    owner = next(step for step in plan.steps if step.step_id == 'cohort_accounting')
    identity = owner.cohort_definition_spec.identity_column
    owner.inputs = [value for value in owner.inputs if value != identity]
    result = _judge(plan, QuestionRequirement(
        id='r2', kind='cohort_accounting', quote='Report excluded stay counts',
        concepts=[identity, 'icu_unit_type'], coverage='plan',
    ))
    assert result.disposition == 'covered'
    assert result.owner_step_ids == ('cohort_accounting',)
    assert identity not in owner.inputs


def test_a_different_row_identity_cannot_cover_the_requested_identifier():
    plan = _accounting_plan()
    owner = next(step for step in plan.steps if step.step_id == 'cohort_accounting')
    identity = owner.cohort_definition_spec.identity_column
    owner.inputs = [value for value in owner.inputs if value != identity]
    owner.cohort_definition_spec.identity_column = 'different_row_key'
    result = _judge(plan, QuestionRequirement(
        id='r2', kind='cohort_accounting', quote='Report excluded stay counts',
        concepts=[identity, 'icu_unit_type'], coverage='plan',
    ))
    assert result.disposition == 'not_covered'


@pytest.mark.parametrize('change', [
    {'inputs': ['stay_id', 'sex', 'death']},
    {'cohort_definition_spec': None},
    {'method': 'descriptive'},
    {'expected_outputs': ['artifact:analysis_cohort']},
    {'expected_outputs': ['table:cohort_flow']},
])
def test_prose_or_a_column_without_the_complete_typed_owner_is_not_enough(change):
    plan = _accounting_plan()
    plan = plan.model_copy(update={'steps': [
        step.model_copy(update=change) if step.step_id == 'cohort_accounting'
        else step for step in plan.steps]})
    assert _judge(plan).disposition == 'not_covered'


def test_family_compiler_binds_count_inputs_without_new_primary_or_baseline_analysis():
    context = _descriptive_context()
    context = context.model_copy(update={
        'research_question': 'Among ICU stays, describe sex and in-hospital mortality. '
            'Report excluded stay counts by ICU unit type. Counts and proportions only, '
            'no confidence intervals, tests, adjustment, prediction or causal analysis.',
        'primary_exposure': 'sex',
        'variables': [*context.variables, ConceptDescriptor(name='icu_unit_type',
            role=VariableRole.DEMOGRAPHIC, dtype='category', source_concept='icu_unit_type',
            observed_domain={'n_unique': 2, 'is_binary': False, 'levels': ['MICU', 'SICU']})],
    })
    request = _request(context, cohort_mode='all_input_rows')
    payload = _payload(request)
    payload['question_requirements'] = [{
        'id': 'r1', 'kind': 'cohort_accounting',
        'quote': 'Report excluded stay counts by ICU unit type',
        'concepts': ['stay_id', 'icu_unit_type'], 'coverage': 'plan', 'gap': None, 'note': None,
    }]
    _, attempt = _run(context, [json.dumps(payload)],
        required_primary_cohort_selection_mode='all_input_rows')
    plan = attempt.output
    assert plan is not None, attempt
    accounting = next(step for step in plan.steps if step.step_id == 'cohort_accounting')
    assert {'stay_id', 'icu_unit_type'} <= set(accounting.inputs)
    assert accounting.planned_analysis_role == 'auxiliary'
    assert not any(step.table_one_spec is not None for step in plan.steps)
    primary, = [step for step in plan.steps if step.planned_analysis_role == 'primary']
    assert primary.exposure_outcome_distribution_spec.interval_method == 'none_counts_only'
    assert 'icu_unit_type' not in primary.inputs
