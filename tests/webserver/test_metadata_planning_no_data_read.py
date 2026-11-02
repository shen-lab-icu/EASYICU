"""Native planning-coordinate resolution reads metadata, not export rows."""
import builtins
import json
import os
from pathlib import Path

import pytest

from easyicu.webserver import research_launch_scientific as owner
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from tests.support.typed_export import typed_export


def test_native_planning_coordinates_resolve_sidecar_without_opening_data(tmp_path, monkeypatch):
    from easyicu.research_agent.acquisition import catalog
    import pyarrow.parquet as pq

    source = typed_export(tmp_path / 'synthetic-export', database='eicu_demo')
    real_open = builtins.open
    real_path_open = Path.open
    real_parquet = pq.ParquetFile
    def is_source_data(path):
        return isinstance(path, (str, bytes, os.PathLike)) and str(path).startswith(str(source)) and str(path).endswith('.parquet')
    def guarded_open(path, *args, **kwargs):
        if is_source_data(path):
            pytest.fail('native source patient-bearing bytes opened')
        return real_open(path, *args, **kwargs)
    def guarded_path_open(path, *args, **kwargs):
        if is_source_data(path):
            pytest.fail('native source patient-bearing bytes opened')
        return real_path_open(path, *args, **kwargs)
    def guarded_parquet(path, *args, **kwargs):
        if is_source_data(path):
            pytest.fail('native source patient-bearing parquet decoded')
        return real_parquet(path, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', guarded_open)
    monkeypatch.setattr(Path, 'open', guarded_path_open)
    monkeypatch.setattr(pq, 'ParquetFile', guarded_parquet)
    monkeypatch.setattr(catalog, 'build_available_catalog', lambda *_: pytest.fail('full package snapshot entered'))
    coordinates = owner._metadata_only_planning_coordinates(question='Describe hospital mortality among ICU patients.',
        database='eicu_demo', export_path=str(source), configured_outcome='death')
    assert coordinates['target_outcome'] == 'death'


@pytest.mark.parametrize('defect', ['missing', 'tampered', 'identity', 'wrong_database'])
def test_declared_native_metadata_drift_fails_before_any_fallback(tmp_path, monkeypatch, defect):
    from easyicu.research_agent.acquisition import catalog
    from easyicu.webserver import raw_preparation_dependencies

    source = typed_export(tmp_path / 'synthetic-export', database='eicu_demo')
    manifest = json.loads((source / '_manifest.json').read_text())
    sidecar = source / manifest['column_metadata']['file']
    if defect == 'missing':
        sidecar.unlink()
    elif defect == 'tampered':
        sidecar.write_bytes(sidecar.read_bytes() + b' ')
    elif defect == 'identity':
        real = raw_preparation_dependencies._declared_typed_export_columns
        def inconsistent(**kwargs):
            result = real(**kwargs)
            result['files'][0]['identity_column'] = 'patientunitstayid'
            return result
        monkeypatch.setattr(raw_preparation_dependencies, '_declared_typed_export_columns', inconsistent)
    monkeypatch.setattr(catalog, 'build_available_catalog', lambda *_: pytest.fail('must not fall back to data snapshot'))
    with pytest.raises(ResearchPipelineRunError) as error:
        owner._bound_source_column(str(source), database='miiv' if defect == 'wrong_database' else 'eicu_demo')
    assert error.value.code == 'research_pipeline_planning_native_metadata_invalid'


def test_non_native_column_resolution_preserves_existing_catalog_contract(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from easyicu.research_agent.acquisition import catalog

    seen = []
    monkeypatch.setattr(catalog, 'build_available_catalog',
        lambda path: seen.append(path) or SimpleNamespace(concepts=[SimpleNamespace(concept_id='death')]))
    resolver = owner._bound_source_column(str(tmp_path), database='eicu_demo')
    assert resolver('death') == 'death'
    assert seen == [tmp_path]


def test_native_coordinate_availability_excludes_manifest_declared_all_null(tmp_path, monkeypatch):
    from easyicu import concept_output_sources

    source = typed_export(tmp_path / 'synthetic-export', database='eicu_demo')
    manifest_path = source / '_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for item in manifest['files']:
        if item['module'] == 'outcomes':
            item['concept_status'] = {'death': {'availability': 'produced_all_null'}}
    manifest_path.write_text(json.dumps(manifest))
    seen = []
    monkeypatch.setattr(concept_output_sources, 'resolve_composite_concept_output',
        lambda concept, available: seen.append(available) or None)
    assert owner._bound_source_column(str(source), database='eicu_demo')('death') == 'death'
    assert len(seen) == 1 and 'death' not in seen[0]


@pytest.mark.parametrize('kind', ['adjustment', 'subgroup', 'exposure'])
def test_native_stated_variables_use_same_metadata_only_catalog(tmp_path, monkeypatch, kind):
    from easyicu.research_agent.acquisition import catalog

    source = typed_export(tmp_path / 'synthetic-export', database='eicu_demo')
    monkeypatch.setattr(catalog, 'build_available_catalog', lambda *_: pytest.fail('full snapshot entered'))
    reading = {'variable':'age', 'words':'age', 'concept':'age', 'source_status':'available'}
    kwargs = {'question':'Describe age and mortality in ICU patients.', 'database':'eicu_demo',
              'export_path':str(source), 'configured_outcome':'death'}
    if kind == 'adjustment':
        kwargs['stated_adjustments'] = [reading]
    elif kind == 'subgroup':
        kwargs['stated_subgroups'] = [reading]
    else:
        kwargs.update(stated_exposure='age', stated_exposure_concept='age',
                      analysis_family='descriptive_epidemiology')
    result = owner._metadata_only_planning_coordinates(**kwargs)
    assert result['target_outcome'] == 'death'
    if kind == 'adjustment':
        assert 'age' in result['required_adjustment_concepts']
    elif kind == 'subgroup':
        assert 'age' in result['required_subgroup_concepts']


def test_native_describe_source_uses_metadata_not_parquet_footer(tmp_path, monkeypatch):
    from easyicu.webserver import dataio
    import pyarrow.parquet as pq

    source = typed_export(tmp_path / 'synthetic-export', database='eicu_demo')
    monkeypatch.setattr(pq, 'ParquetFile', lambda *_args, **_kwargs: pytest.fail('source parquet footer opened'))
    monkeypatch.setattr(dataio, '_read_columns', lambda *_args, **_kwargs: pytest.fail('generic column scanner entered'))
    result = dataio.describe_source_tables(str(source))
    assert result['ok'] is True
    assert result['database'] == 'eicu_demo'
    assert set(result['modules']) == {'labs', 'outcomes'}
    assert all('stay_id' in item['columns'] for item in result['files'])


@pytest.mark.parametrize('defect', ['missing', 'tampered', 'missing_member'])
def test_native_description_rejects_sidecar_drift_without_scan(tmp_path, monkeypatch, defect):
    from easyicu.webserver import dataio
    source = typed_export(tmp_path / 'synthetic-export', database='eicu_demo')
    manifest = json.loads((source / '_manifest.json').read_text())
    sidecar = source / manifest['column_metadata']['file']
    if defect == 'missing':
        sidecar.unlink()
    elif defect == 'tampered':
        sidecar.write_bytes(sidecar.read_bytes() + b' ')
    else:
        (source / 'labs.parquet').unlink()
    monkeypatch.setattr(dataio, '_read_columns', lambda *_args, **_kwargs: pytest.fail('fallback scan entered'))
    result = dataio.describe_source_tables(str(source))
    assert result['ok'] is False
    assert result['error'] == 'research_pipeline_planning_native_metadata_invalid'
