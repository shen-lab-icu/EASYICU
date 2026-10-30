"""Source-visible first ICU coordinates can restrict the common data boundary.

These synthetic contracts do not constitute host or scientific authorization.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

import pandas as pd
import pytest

from easyicu.research_agent.acquisition.eicu_first_observed_candidate import FLAG_COLUMN, REASON_COLUMN
from easyicu.research_agent.acquisition.first_observed_icu_coordinate import FirstObservedIcuCoordinate
from easyicu.research_agent.cohort.materializer import MaterializedMetadataError, materialize_to_parquet
from easyicu.research_agent.cohort.schema import CohortDefinition, ConceptPredicate, TimeWindow
from easyicu.research_agent.intake.export_package import open_export_package
from easyicu.research_agent.intake.materialized_metadata import load_verified_materialized_cohort_authority
from easyicu.research_agent.intake.materialized_trajectory import load_verified_materialized_trajectory_authority
from tests.support.typed_export import typed_export


def _coordinate(tmp_path, manifest):
    path = tmp_path / 'coordinate.parquet'
    pd.DataFrame({'stay_id': [1, 2, 3], FLAG_COLUMN: [False, True, True],
                  REASON_COLUMN: ['multiple_observed_hospitalizations', 'kept', 'kept']}).to_parquet(path, index=False)
    return FirstObservedIcuCoordinate(path=path.absolute(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        archive_sha256='a' * 64, patient_table_sha256='b' * 64, export_manifest_sha256=manifest,
        plan_sha256='c' * 64, preparation_checkpoint_sha256='d' * 64)


def test_typed_materializer_restricts_cohort_and_all_trajectory_rows(tmp_path):
    labs = pd.DataFrame({'stay_id': [1, 1, 2, 3], 'charttime': [1., 2., 1., 1.],
                         'age': [65, 65, 60, 40], 'lact': [1., 2., 3., 4.],
                         'mech_vent': [False, True, False, False]})
    outcomes = pd.DataFrame({'stay_id': [1, 2, 3], 'death': [False, True, False]})
    source = typed_export(tmp_path / 'export', database='eicu', labs=labs, outcomes=outcomes)
    definition = CohortDefinition(name='older_adults', inclusion=(ConceptPredicate(
        concept_id='age', time_window=TimeWindow(anchor='icu_admit', start_offset_hours=0,
        end_offset_hours=24), aggregation='first', op='>=', value=55),))
    with open_export_package(source) as package:
        coordinate = _coordinate(tmp_path, package.manifest_sha256)
        paths = materialize_to_parquet(tmp_path / 'out', database='eicu', data_path=source,
            source_package=package, feature_concepts=('lact',), static_concepts=('age',),
            outcome_concepts=('death',), cohort_definition=definition,
            emit_trajectory=True, trajectory_concepts=('lact',),
            first_observed_icu_coordinate=coordinate)
    cohort = pd.read_parquet(paths['parquet'])
    trajectory = pd.read_parquet(paths['trajectory'])
    assert cohort['stay_id'].tolist() == [2]
    assert cohort['death'].tolist() == [True]
    assert trajectory['stay_id'].tolist() == [2]
    assert trajectory['concept'].tolist() == ['lact']
    assert trajectory['value_num'].tolist() == [3.0]
    owner = load_verified_materialized_cohort_authority(paths['parquet'])
    assert owner is not None
    receipt = owner.authority.producer_parameters['first_observed_icu_restriction']
    assert receipt['coordinate_sha256'] == coordinate.sha256
    assert receipt['export_manifest_sha256'] == coordinate.export_manifest_sha256
    assert receipt['plan_sha256'] == coordinate.plan_sha256
    assert receipt['not_lifetime_first_icu'] is True
    assert load_verified_materialized_trajectory_authority(paths['trajectory'], expected_universe_authority=owner.reference)


@pytest.mark.parametrize('change', ['source', 'coordinate', 'missing_package', 'conflicting_restriction'])
def test_invalid_coordinate_binding_refuses_before_cohort_rows(tmp_path, monkeypatch, change):
    from easyicu.research_agent.cohort import materializer
    source = typed_export(tmp_path / 'export', database='eicu')
    with open_export_package(source) as package:
        coordinate = _coordinate(tmp_path, package.manifest_sha256)
        options = {'source_package': package}
        if change == 'source':
            coordinate = replace(coordinate, export_manifest_sha256='e' * 64)
        elif change == 'coordinate':
            coordinate = replace(coordinate, sha256='e' * 64)
        elif change == 'missing_package':
            options['source_package'] = None
        else:
            options['first_icu_stay_path'] = coordinate.path
        monkeypatch.setattr(materializer, '_materialize_with_open_export_package',
                            lambda *a, **kw: pytest.fail('patient rows were read'))
        with pytest.raises((MaterializedMetadataError, ValueError)):
            materialize_to_parquet(tmp_path / 'out', database='eicu', data_path=source,
                feature_concepts=('lact',), first_observed_icu_coordinate=coordinate, **options)


def test_changed_coordinate_during_extraction_blocks_publication(tmp_path, monkeypatch):
    from easyicu.research_agent.cohort import materializer
    source = typed_export(tmp_path / 'export', database='eicu')
    with open_export_package(source) as package:
        coordinate = _coordinate(tmp_path, package.manifest_sha256)
        extract = materializer._materialize_with_open_export_package
        def changing(*args, **kwargs):
            result = extract(*args, **kwargs)
            coordinate.path.write_bytes(coordinate.path.read_bytes() + b'changed')
            return result
        monkeypatch.setattr(materializer, '_materialize_with_open_export_package', changing)
        with pytest.raises(ValueError):
            materialize_to_parquet(tmp_path / 'out', database='eicu', data_path=source,
                source_package=package, feature_concepts=('lact',),
                first_observed_icu_coordinate=coordinate)
    assert not (tmp_path / 'out' / 'cohort.parquet').exists()
