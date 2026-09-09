import hashlib
import json
from types import SimpleNamespace

import pytest

from easyicu.webserver.report_revision_replay import load_failed_writer_replay, WriterOnlyMigrationError


def _saved(tmp_path):
    root = tmp_path / 'report_revisions' / 'failed'
    (root / 'runtime').mkdir(parents=True)
    prepared = SimpleNamespace(source_run_dir=tmp_path / 'source', source_hashes={'plan':'a'},
        migration_draft_sha256='draft-hash', migration_draft_path=None, evidence_digest='verified evidence')
    (root / 'writer_only_migration_receipt.json').write_text(json.dumps({
        'status':'failed','reason_code':'WRITER_ONLY_AUTHORITY_REPAIR_FAILED_PRIOR_PRESERVED',
        'source_run_modified':False,'source_hashes':prepared.source_hashes,
        'source_run_dir':str(prepared.source_run_dir),'provider_summary':{'n_calls':1}}))
    (root / 'preflight.json').write_text(json.dumps({'migration_draft_sha256':'draft-hash',
        'migration_draft_path':None,'writer_evidence_digest_sha256':hashlib.sha256(prepared.evidence_digest.encode()).hexdigest()}))
    candidate=root/'runtime/writer_candidate_01.json'
    candidate.write_text(json.dumps({'section':'Discussion','instruction':'Fix recorded status.','text':'Recorded status is not confirmed diagnosis.'}))
    return prepared,candidate


def test_replay_reuses_exact_section_without_allowing_additional_calls(tmp_path):
    prepared,_=_saved(tmp_path)
    replay=load_failed_writer_replay(tmp_path,prepared)
    assert replay.revision_id=='failed'
    assert replay.section(section_name='Discussion',instruction='Fix recorded status.')=='Recorded status is not confirmed diagnosis.'
    with pytest.raises(WriterOnlyMigrationError,match='REPLAY_EXHAUSTED'):
        replay.section(section_name='Discussion',instruction='Fix recorded status.')


@pytest.mark.parametrize('change',['source','draft','digest'])
def test_replay_never_reuses_another_study_or_report(tmp_path,change):
    prepared,_=_saved(tmp_path)
    if change == 'source':
        prepared.source_hashes = {'plan': 'different'}
    if change == 'draft':
        prepared.migration_draft_sha256 = 'other'
    if change == 'digest':
        prepared.evidence_digest = 'other evidence'
    assert load_failed_writer_replay(tmp_path,prepared) is None


@pytest.mark.parametrize('change',['file','instruction','section'])
def test_replay_rejects_drift_before_returning_saved_text(tmp_path,change):
    prepared,path=_saved(tmp_path)
    replay=load_failed_writer_replay(tmp_path,prepared)
    if change == 'file':
        path.write_text('{}')
    with pytest.raises(WriterOnlyMigrationError,match='REPLAY_MISMATCH'):
        replay.section(section_name='Methods' if change=='section' else 'Discussion',
                       instruction='Other request' if change=='instruction' else 'Fix recorded status.')


def test_governed_runner_continues_only_after_exact_prefix_is_exhausted(tmp_path):
    from easyicu.webserver.manuscript_repair import _replay_or_generate_section

    prepared, _ = _saved(tmp_path)
    replay = load_failed_writer_replay(tmp_path, prepared)
    events = []

    def generate():
        events.append('provider')
        return 'New section'

    kwargs = dict(replay=replay, generate=generate,
                  on_continue=lambda: events.append('continuing'))
    assert _replay_or_generate_section(
        section='Discussion', instruction='Fix recorded status.', **kwargs
    ) == ('Recorded status is not confirmed diagnosis.', True)
    assert events == []
    assert _replay_or_generate_section(
        section='Results', instruction='Fix missing cohort.', **kwargs
    ) == ('New section', False)
    assert events == ['continuing', 'provider']


@pytest.mark.parametrize('change', ['file', 'instruction'])
def test_governed_runner_never_generates_after_replay_drift(tmp_path, change):
    from easyicu.webserver.manuscript_repair import _replay_or_generate_section

    prepared, path = _saved(tmp_path)
    replay = load_failed_writer_replay(tmp_path, prepared)
    if change == 'file':
        path.write_text('{}')
    events = []
    with pytest.raises(WriterOnlyMigrationError, match='REPLAY_MISMATCH'):
        _replay_or_generate_section(
            replay, section='Discussion',
            instruction='Changed instruction' if change == 'instruction' else 'Fix recorded status.',
            generate=lambda: events.append('provider'),
            on_continue=lambda: events.append('continuing'),
        )
    assert events == []


def test_governed_continuation_propagates_provider_stop(tmp_path):
    from easyicu.webserver.manuscript_repair import _replay_or_generate_section

    prepared, _ = _saved(tmp_path)
    replay = load_failed_writer_replay(tmp_path, prepared)
    replay.section(section_name='Discussion', instruction='Fix recorded status.')

    def exhausted_provider():
        raise RuntimeError('provider budget exhausted')

    with pytest.raises(RuntimeError, match='provider budget exhausted'):
        _replay_or_generate_section(
            replay, section='Results', instruction='Fix cohort.',
            generate=exhausted_provider, on_continue=lambda: None,
        )


@pytest.mark.parametrize('recorded_replayed', [0, 1, 2])
def test_mixed_replay_receipt_counts_generated_and_reused_sections(tmp_path, recorded_replayed):
    prepared, path = _saved(tmp_path)
    first = json.loads(path.read_text())
    first['replayed_from_revision'] = 'earlier-failure'
    path.write_text(json.dumps(first))
    (path.parent / 'writer_candidate_02.json').write_text(json.dumps({
        'section': 'Results', 'instruction': 'Fix cohort.', 'text': 'Cohort summary.',
        'replayed_from_revision': None,
    }))
    receipt_path = path.parent.parent / 'writer_only_migration_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['provider_summary'] = {'n_calls': 1, 'replayed_sections': recorded_replayed}
    receipt_path.write_text(json.dumps(receipt))
    replay = load_failed_writer_replay(tmp_path, prepared)
    assert (replay is not None) == (recorded_replayed == 1)
