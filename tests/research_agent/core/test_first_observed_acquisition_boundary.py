"""Exercise the foundation-to-materializer bridge with a synthetic export.

This test supplies a coordinate directly and grants no research authority.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

import pandas as pd
import pytest

from easyicu.research_agent.acquisition.eicu_first_observed_candidate import FLAG_COLUMN, REASON_COLUMN
from easyicu.research_agent.acquisition.first_observed_icu_coordinate import FirstObservedIcuCoordinate
from easyicu.research_agent.acquisition.foundation import acquire_universe_for_question
from easyicu.research_agent.intake.export_package import open_export_package
from easyicu.research_agent.intake.materialized_metadata import MaterializedMetadataError, load_verified_materialized_cohort_authority
from tests.support.typed_export import typed_export


def _inputs(tmp_path):
    source = typed_export(tmp_path / 'export', database='eicu')
    with open_export_package(source) as package:
        digest = package.manifest_sha256
    path = tmp_path / 'coordinate.parquet'
    pd.DataFrame({'stay_id': [1, 2], FLAG_COLUMN: [False, True],
        REASON_COLUMN: ['multiple_observed_hospitalizations', 'kept']}).to_parquet(path, index=False)
    coordinate = FirstObservedIcuCoordinate(path=path.absolute(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        archive_sha256='a' * 64, patient_table_sha256='b' * 64, export_manifest_sha256=digest,
        plan_sha256='c' * 64, preparation_checkpoint_sha256='d' * 64)
    return source, coordinate


def _acquire(tmp_path, source, coordinate):
    return acquire_universe_for_question(export_dir=source, question='Synthetic first observed ICU', llm=None,
        output_dir=tmp_path / 'out', target_outcome='death', outcome_concepts=('death',),
        required_feature_concepts=('lact',), static_concepts=('age',), database='eicu',
        concept_selection_authority='host_exact', first_observed_icu_coordinate=coordinate)


def test_foundation_restricts_universe_and_trajectory_without_losing_outcome(tmp_path):
    source, coordinate = _inputs(tmp_path)
    result = _acquire(tmp_path, source, coordinate)
    assert not result.blocked
    cohort = pd.read_parquet(result.universe_path)
    trajectory = pd.read_parquet(result.trajectory_path)
    assert cohort['stay_id'].tolist() == [2]
    assert cohort['death'].tolist() == [True]
    assert set(trajectory['stay_id']) == {2}
    owner = load_verified_materialized_cohort_authority(result.universe_path)
    assert owner.authority.producer_parameters['first_observed_icu_restriction']['coordinate_sha256'] == coordinate.sha256


def test_foundation_refuses_another_source_before_materialization(tmp_path, monkeypatch):
    from easyicu.research_agent.cohort import materializer
    source, coordinate = _inputs(tmp_path)
    coordinate = replace(coordinate, export_manifest_sha256='e' * 64)
    monkeypatch.setattr(materializer, '_materialize_with_open_export_package',
                        lambda *a, **kw: pytest.fail('patient rows were read'))
    with pytest.raises(MaterializedMetadataError, match='source mismatch'):
        _acquire(tmp_path, source, coordinate)


def test_foundation_refuses_unreviewed_concept_selection_before_provider(tmp_path):
    source, coordinate = _inputs(tmp_path)
    with pytest.raises(MaterializedMetadataError, match='exact host concept selection'):
        acquire_universe_for_question(export_dir=source, question='first ICU', llm=None,
            output_dir=tmp_path / 'out', target_outcome='death', outcome_concepts=('death',),
            database='eicu', first_observed_icu_coordinate=coordinate)
    assert not (tmp_path / 'out').exists()
