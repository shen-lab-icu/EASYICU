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
    if change=='source':prepared.source_hashes={'plan':'different'}
    if change=='draft':prepared.migration_draft_sha256='other'
    if change=='digest':prepared.evidence_digest='other evidence'
    assert load_failed_writer_replay(tmp_path,prepared) is None


@pytest.mark.parametrize('change',['file','instruction','section'])
def test_replay_rejects_drift_before_returning_saved_text(tmp_path,change):
    prepared,path=_saved(tmp_path)
    replay=load_failed_writer_replay(tmp_path,prepared)
    if change=='file':path.write_text('{}')
    with pytest.raises(WriterOnlyMigrationError,match='REPLAY_MISMATCH'):
        replay.section(section_name='Methods' if change=='section' else 'Discussion',
                       instruction='Other request' if change=='instruction' else 'Fix recorded status.')
