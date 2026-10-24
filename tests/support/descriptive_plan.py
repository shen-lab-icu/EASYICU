"""Shared synthetic fixtures; no recorded research inputs."""
from __future__ import annotations

import json
from easyicu.research_agent.schema import MissingnessProfile
from tests.research_agent.planning.family_spec_fixtures import _descriptive_context, _request, _run

from tests.support.distribution_policies import EXCLUDE_MISSING


def _context():
    context = _descriptive_context()
    return context.model_copy(update={
        'research_question': '描述成年 ICU 住院的性别及院内死亡计数和比例；不计算置信区间，不做检验、调整、预测或因果分析。',
        'primary_exposure': 'sex',
        'variables': [
            variable.model_copy(update={
                'dtype': 'category' if variable.name == 'sex' else 'Int8',
                'observed_domain': {
                    'n_unique': 2, 'is_binary': True,
                    'levels': ['Female', 'Male'] if variable.name == 'sex' else [0, 1],
                },
                'missingness': MissingnessProfile(fraction_missing=0.0, n_missing=0, n_total=0),
            }) if variable.name in {'sex', 'death'} else variable
            for variable in context.variables
        ],
        'endpoint': context.endpoint.model_copy(update={'levels': [0, 1]}),
    })


def _payload(request, *, baseline=(), policies=None):
    return {
        'schema_version': 'easyicu.family_plan_spec/1',
        'family_id': request.family_id,
        'request_sha256': request.request_sha256,
        'baseline_variables': list(baseline),
        'distribution_missing_policies': dict(EXCLUDE_MISSING if policies is None else policies),
        'reader_display_labels': [
            {'key': key, 'value': f'Clinical reader label {index}'}
            for index, key in enumerate([*request.required_reader_label_keys, *request.level_label_keys])
        ],
        'comparator_applications': [
            {'citation_key': key, 'application': f'Compare population and time zero with {key} without claiming novelty.'}
            for key in request.direct_comparator_literature_keys
        ],
        'roster_decision_note': 'Age is used only for eligibility. No baseline summaries are requested.',
    }


def _plan(*, policies=None):
    context = _context()
    request = _request(context, cohort_mode='all_input_rows')
    _, result = _run(context, [json.dumps(_payload(request, policies=policies))],
                     required_primary_cohort_selection_mode='all_input_rows')
    return result.output
