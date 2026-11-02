"""Synthetic signed raw planning identity; no real patient or Provider access."""
import hashlib
from dataclasses import replace
import json
import shutil
from pathlib import Path

import pytest
import pyarrow.parquet as pq
pytest_plugins = ("tests.support.official_demo_raw", "tests.support.official_demo_choice")
from tests.support.typed_export import typed_export
from easyicu.concept.metadata_sidecar import ColumnMetadataSidecar, write_content_addressed_sidecar
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.authority.filesystem import AuthorityFilesystemError
from easyicu.webserver import agent_pipeline_runs as runs, research_launch_resume, study_contexts
from easyicu.webserver.pi_copilot.demo_planning_authority import resolve_choice
from easyicu.webserver.pi_copilot.demo_planning_source import resolve_demo_planning_source


def _study_with_typed_export(choice, tmp_path):
    previous = json.loads((choice.paths.export / '_manifest.json').read_text())
    source = typed_export(tmp_path / 'typed', database='eicu_demo')
    for item in source.iterdir():
        shutil.copyfile(item, choice.paths.export / item.name)
    manifest_path = choice.paths.export / '_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    sidecar_path = choice.paths.export / manifest['column_metadata']['file']
    sidecar = ColumnMetadataSidecar.from_dict(json.loads(sidecar_path.read_text()))
    sidecar = replace(sidecar, files=tuple(replace(item, identity_column='patientunitstayid') for item in sidecar.files))
    manifest['column_metadata'] = write_content_addressed_sidecar(choice.paths.export, sidecar).to_dict()
    manifest.update({key: previous[key] for key in ('data_path', 'cohort_contract', 'cohort_execution')})
    manifest_path.write_text(json.dumps(manifest))
    resolved = resolve_demo_planning_source(choice.study)
    assert resolved is not None
    body = {**choice.body, 'offer_sha256': resolved.offer['offer_sha256']}
    choice.service.authorize_data_source(choice.record.session_id, **body)
    study = study_contexts.get_context(choice.study['id'])
    return study, resolve_choice(study)


def _acquire(tmp_path, study, identity=None):
    kwargs = {}
    if identity is not None:
        kwargs['_demo_output_identity'] = runs._verified_demo_planning_output_identity(
            study=study, source_identity=identity, database='eicu_demo',
            export_path=study['data_source']['path'])
    return runs._metadata_only_planning_acquisition(database='eicu_demo',
        export_path=study['data_source']['path'], question='Describe sex and death.',
        llm=ScriptedMockLLMClient([json.dumps({'selected_concepts':['sex','death'],
            'rationale':'synthetic','inclusion_exclusion':[]})]), output_dir=tmp_path,
        target_outcome='death', required_coordinates=('sex','death'), **kwargs)


def _profile(result):
    return research_launch_resume._DevelopmentResumeAcquisition(
        kind='metadata_only_planning_catalog', selected_concepts=tuple(result.selection.selected_concepts),
        universe_path=result.universe_path, provenance_path=result.provenance_path,
        universe_sha256=hashlib.sha256(result.universe_path.read_bytes()).hexdigest(),
        provenance_sha256=hashlib.sha256(result.provenance_path.read_bytes()).hexdigest())


def test_signed_raw_identity_and_restore(choice, tmp_path, monkeypatch):
    study, identity = _study_with_typed_export(choice, tmp_path)
    real_parquet_file = pq.ParquetFile
    def guard(path, *args, **kwargs):
        path = Path(path)
        assert path.name not in {'labs.parquet', 'outcomes.parquet'}
        return real_parquet_file(path, *args, **kwargs)
    monkeypatch.setattr(pq, 'ParquetFile', guard)
    result = _acquire(tmp_path/'fresh', study, identity)
    assert not result.blocked
    receipt=json.loads(result.provenance_path.read_text())
    assert receipt['row_identity_column']=='stay_id'
    assert 'demo_output_identity' in receipt
    assert pq.read_metadata(result.universe_path).num_rows == 0
    restored = runs._restore_metadata_only_planning_acquisition(database='eicu_demo',
        export_path=study['data_source']['path'], profile=_profile(result), output_dir=tmp_path/'restore',
        _demo_identity_study=study, _demo_identity_source=identity)
    assert restored.universe_path.read_bytes()==result.universe_path.read_bytes()
    assert restored.provenance_path.read_bytes()==result.provenance_path.read_bytes()
    with pytest.raises((ValueError, runs.ResearchPipelineRunError, AuthorityFilesystemError)):
        runs._restore_metadata_only_planning_acquisition(database='eicu_demo',
            export_path=study['data_source']['path'], profile=_profile(result), output_dir=tmp_path/'bad',
            _demo_identity_study={**study, 'demo_planning_authority_ref': '0'*64}, _demo_identity_source=identity)


def test_old_signed_resume_keeps_original_contract(choice, tmp_path, monkeypatch):
    study, identity = _study_with_typed_export(choice, tmp_path)
    result = _acquire(tmp_path/'old', study)
    assert json.loads(result.provenance_path.read_text())['row_identity_column']=='patientunitstayid'
    monkeypatch.setattr(runs, '_verified_demo_planning_output_identity', lambda **kwargs: pytest.fail('old resume must not use new authority'))
    restored = runs._restore_metadata_only_planning_acquisition(database='eicu_demo',
        export_path=study['data_source']['path'], profile=_profile(result), output_dir=tmp_path/'old-restored',
        _demo_identity_study=study, _demo_identity_source=identity)
    assert restored.provenance_path.read_bytes()==result.provenance_path.read_bytes()


def test_signed_source_sidecar_tampering_rejected(choice, tmp_path):
    study, identity = _study_with_typed_export(choice, tmp_path)
    output = runs._verified_demo_planning_output_identity(study=study, source_identity=identity,
        database='eicu_demo', export_path=study['data_source']['path'])
    assert output['output_identity_column'] == 'stay_id'
    manifest = json.loads((choice.paths.export / '_manifest.json').read_text())
    sidecar = choice.paths.export / manifest['column_metadata']['file']
    sidecar.write_bytes(sidecar.read_bytes() + b' ')
    with pytest.raises((ValueError, runs.ResearchPipelineRunError, AuthorityFilesystemError)):
        runs._verified_demo_planning_output_identity(study=study, source_identity=identity,
            database='eicu_demo', export_path=study['data_source']['path'])
