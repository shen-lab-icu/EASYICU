"""Registered export to raw-root binding for partial review scopes."""
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from easyicu.webserver import reviewed_multitable_source as bound
pytest_plugins = ("tests.webserver.test_source_partition_candidates",)


def _scope(identity):
    return bound.build_source_bound_multitable_scope(identity=identity,
        requested_columns={'laboratory': ['LaboratoryValue', 'CaseID'], 'cases': ['PatientID', 'CaseID']},
        plan_sha256='c' * 64)


def test_exact_scope_resolves_to_raw_not_registered_export(source):
    identity, raw_root, _ = source
    scope, candidate = _scope(identity)
    root, checked = bound.resolve_source_bound_multitable_scope(scope)
    assert root == raw_root and root.name == 'raw'
    assert checked == candidate
    assert [entry['path'] for entry in scope['inputs']] == [
        'cases.parquet', 'laboratory/1.parquet', 'laboratory/2.parquet']
    assert scope['inputs'][1]['columns'] == ['CaseID', 'LaboratoryValue']
    assert scope['dependencies_sha256'] == candidate['candidate_sha256']
    assert not candidate['files_content_verified'] and not candidate['analysis_authorized']
    assert not candidate['full_source_roster_verified'] and not candidate['cohort_complete']


def test_rejects_widened_dropped_reordered_or_rebound_scope(source):
    identity, _, _ = source
    scope, _ = _scope(identity)
    for edit in ('widen', 'drop', 'reorder', 'changed_digest'):
        changed = json.loads(json.dumps(scope))
        if edit == 'widen':
            changed['inputs'][0]['columns'].append('ICUOffset')
        elif edit == 'drop':
            changed['inputs'].pop()
        elif edit == 'reorder':
            changed['inputs'].reverse()
        else:
            changed['dependencies_sha256'] = 'd' * 64
        with pytest.raises(ValueError):
            bound.resolve_source_bound_multitable_scope(changed)
    with pytest.raises(ValueError):
        bound.resolve_source_bound_multitable_scope({**scope, 'source_identity': {'database': 'sic', 'registration_id': 'other'}})
    with pytest.raises(ValueError):
        bound.resolve_source_bound_multitable_scope({**scope, 'plan_sha256': 'invalid'})


def test_conversion_roster_or_source_binding_drift_refuse(source, monkeypatch):
    identity, root, manifest = source
    scope, _ = _scope(identity)
    (root / 'laboratory' / 'extra.parquet').write_bytes(b'extra')
    with pytest.raises(ValueError, match='directory_roster_mismatch'):
        bound.resolve_source_bound_multitable_scope(scope)
    (root / 'laboratory' / 'extra.parquet').unlink()
    manifest['tables'][1]['outputs'][1]['relative_path'] = 'laboratory/3.parquet'
    (root / 'conversion_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        bound.resolve_source_bound_multitable_scope(scope)
    monkeypatch.setattr(bound.discovery, 'resolve_raw_source_identity_binding', lambda **_: SimpleNamespace(
        source_root=root, export_manifest_sha256='f' * 64, table_sha256={'cases': 'b' * 64}))
    with pytest.raises(ValueError):
        bound.resolve_source_bound_multitable_scope(scope)


def test_candidate_scope_does_not_claim_parquet_content_is_frozen(source):
    identity, root, _ = source
    scope, before = _scope(identity)
    pq.write_table(pa.table({'CaseID': [8], 'LaboratoryID': [2], 'Offset': [0], 'LaboratoryValue': [9.]}),
                   root / 'laboratory' / '1.parquet')
    _, after = bound.resolve_source_bound_multitable_scope(scope)
    assert before['candidate_sha256'] == after['candidate_sha256']
    assert after['files_content_verified'] is False
