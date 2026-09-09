"""Display metadata must come from exact host-bound bytes, never column spelling."""
import hashlib
import json
from pathlib import Path

import pytest

from easyicu.research_agent.schema import ResearchContext
from easyicu.research_agent.execution.runners.bound_variable_display import load_bound_variable_display


def _binding(root, name, payload):
    path = root / name
    path.write_text(json.dumps(payload))
    return {'relative_path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _manifest(root, *, unit='mg/dL', labels=None):
    context = ResearchContext(
        research_question='Describe marker X.',
        cohort={'cohort_name':'cohort','database':'synthetic','n_stays':10},
        variables=[{'name':'unfamiliar_column','description':'Source marker',
                    'dtype':'float64','unit':unit}],
    )
    return {'step_id':'display_step',
            'context':_binding(root,'context.json',context.model_dump(mode='json')),
            'plan':_binding(root,'plan.json',{'display_labels': labels or {}})}


def _load(root, manifest, **kwargs):
    return load_bound_variable_display(run_dir=root, manifest=manifest,
                                      step_id='display_step', column=kwargs.get('column','unfamiliar_column'))


@pytest.mark.parametrize('unit', ['mg/dL', 'mmol/L', None])
def test_uses_exact_units_and_planner_label_without_conversion(tmp_path, unit):
    manifest = _manifest(tmp_path, unit=unit, labels={'unfamiliar_column':'Reviewed marker'})
    value = _load(tmp_path, manifest)
    assert value.label == 'Reviewed marker' and value.unit == unit
    assert value.context_sha256 == manifest['context']['sha256']
    assert value.plan_sha256 == manifest['plan']['sha256']


def test_source_description_supplies_an_unlabelled_variable(tmp_path):
    assert _load(tmp_path, _manifest(tmp_path)).label == 'Source marker'


@pytest.mark.parametrize('key', ['context', 'plan'])
def test_changed_metadata_bytes_are_rejected(tmp_path, key):
    manifest = _manifest(tmp_path)
    with (tmp_path / manifest[key]['relative_path']).open('a') as handle:
        handle.write(' ')
    with pytest.raises(ValueError, match='digest mismatch'):
        _load(tmp_path, manifest)


@pytest.mark.parametrize('mutation', ['other_step','missing_column','duplicate_column','escape','symlink'])
def test_ambiguous_or_unbound_metadata_is_not_used(tmp_path, mutation):
    manifest = _manifest(tmp_path)
    kwargs = {}
    if mutation == 'other_step':
        manifest['step_id'] = 'different_step'
    elif mutation == 'missing_column':
        kwargs['column'] = 'not_bound'
    elif mutation == 'duplicate_column':
        path = tmp_path / 'context.json'
        body = json.loads(path.read_text())
        body['variables'] *= 2
        manifest['context'] = _binding(tmp_path,'context.json',body)
    elif mutation == 'escape':
        manifest['context']['relative_path'] = '../context.json'
    else:
        (tmp_path / 'alias.json').symlink_to(tmp_path / 'context.json')
        manifest['context']['relative_path'] = 'alias.json'
    with pytest.raises(ValueError):
        _load(tmp_path, manifest, **kwargs)
