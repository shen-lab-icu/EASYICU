"""Regression tests for the cohort workspace ConceptBundle refactor and
the Cohort Analysis gate / state-seeding contract.

Locked behaviors:
- Bundle factories are pure: same input → same output, no session_state.
- Seeded state shape matches what the 5 Cohort Analysis panels read.
- Per-database id_col detection (stay_id / icustay_id / patientunitstayid /
  admissionid / patientid / CaseID).
- sex↔gender alias + los_hours derivation are stable.
- Gate logic admits BOTH a sidebar-validated raw path AND a loaded
  exports cohort.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.webapp.cohort_workspace import (
    ConceptBundle,
    _bundle_from_loaded_concepts,
    _cohort_real_workspace_matches_sidebar,
    _cohort_real_workspace_ready,
    _ensure_cohort_real_workspace_from_loaded_concepts,
    _seed_workspace_state,
)


def _mimic_iv_concepts() -> dict[str, pd.DataFrame]:
    return {
        'age': pd.DataFrame({'stay_id': [1, 2, 3], 'age': [65, 70, 80]}),
        'sex': pd.DataFrame({'stay_id': [1, 2, 3], 'sex': ['M', 'F', 'M']}),
        'death': pd.DataFrame({'stay_id': [1, 2, 3], 'death': [0, 1, 0]}),
        'los_icu': pd.DataFrame({'stay_id': [1, 2, 3], 'los_icu': [2.0, 4.5, 7.0]}),
        'sofa': pd.DataFrame({
            'stay_id': [1, 2, 3],
            'charttime': [0, 0, 0],
            'sofa': [3, 5, 7],
        }),
    }


def test_bundle_empty_loaded_concepts_returns_failure():
    ok, msg, bundle = _bundle_from_loaded_concepts({}, lang='en')
    assert ok is False
    assert bundle is None
    assert msg  # has an explanatory message


def test_bundle_mimic_iv_detects_stay_id_and_recovers_demographics():
    loaded = _mimic_iv_concepts()
    ok, msg, bundle = _bundle_from_loaded_concepts(loaded, lang='en')
    assert ok is True
    assert bundle is not None
    assert bundle.id_col == 'stay_id'
    assert bundle.origin == 'loaded_exports'
    assert bundle.configure_multidb is False  # exports must NOT seed Cross-DB
    assert bundle.loaded_data_origin == 'loaded_exports'
    assert len(bundle.patient_ids) == 3
    cols = set(bundle.demographics.columns)
    assert {'stay_id', 'age', 'sex', 'gender', 'death', 'los_icu', 'los_hours'} <= cols


@pytest.mark.parametrize(
    'id_col,sample_df_factory',
    [
        ('icustay_id',
         lambda: pd.DataFrame({'icustay_id': [100, 200], 'age': [60, 75]})),
        ('patientunitstayid',
         lambda: pd.DataFrame({'patientunitstayid': [1, 2], 'age': [55, 80]})),
        ('admissionid',
         lambda: pd.DataFrame({'admissionid': [10, 20], 'age': [50, 70]})),
        ('patientid',
         lambda: pd.DataFrame({'patientid': [11, 22], 'age': [65, 72]})),
        ('CaseID',
         lambda: pd.DataFrame({'CaseID': [1, 2], 'age': [68, 78]})),
    ],
)
def test_bundle_detects_per_database_id_column(id_col, sample_df_factory):
    loaded = {'age': sample_df_factory()}
    ok, msg, bundle = _bundle_from_loaded_concepts(loaded, lang='en')
    assert ok is True
    assert bundle is not None
    assert bundle.id_col == id_col, f'expected {id_col!r}, got {bundle.id_col!r}'


def test_bundle_los_icu_derives_los_hours():
    loaded = {
        'los_icu': pd.DataFrame({'stay_id': [1, 2], 'los_icu': [1.5, 3.0]}),
    }
    _, _, bundle = _bundle_from_loaded_concepts(loaded, lang='en')
    assert bundle is not None
    assert 'los_hours' in bundle.demographics.columns
    assert bundle.demographics['los_hours'].tolist() == [36.0, 72.0]


def test_bundle_sex_aliases_to_gender_and_vice_versa():
    # sex → gender
    _, _, b1 = _bundle_from_loaded_concepts(
        {'sex': pd.DataFrame({'stay_id': [1], 'sex': ['M']})}, lang='en',
    )
    assert b1 is not None
    assert b1.demographics['gender'].iloc[0] == 'M'

    # Note: bridge function pulls 'sex' and 'gender' as separate concepts.
    # When only 'gender' is loaded, the demographics frame copies it to 'sex'.
    _, _, b2 = _bundle_from_loaded_concepts(
        {'gender': pd.DataFrame({'stay_id': [1], 'gender': ['F']})}, lang='en',
    )
    assert b2 is not None
    assert b2.demographics['sex'].iloc[0] == 'F'


def test_seed_workspace_state_writes_all_cohort_keys():
    loaded = _mimic_iv_concepts()
    _, _, bundle = _bundle_from_loaded_concepts(loaded, lang='en')
    assert bundle is not None
    state: dict = {'loaded_concepts': loaded}
    _seed_workspace_state(state, bundle)

    assert state['_cohort_real_ws_ready'] is True
    assert state['_cohort_real_ws_origin'] == 'loaded_exports'
    assert state['_cohort_real_ws_path'] == '__loaded_exports__'
    assert state['_cohort_real_ws_db']  # any of 6 supported
    assert isinstance(state['_cohort_real_ws_demographics'], pd.DataFrame)
    assert len(state['_cohort_real_ws_patient_ids']) == 3
    assert state['id_col'] == 'stay_id'
    assert state['time_col'] == 'charttime'
    # Per-panel seeds present
    for prefix in ('grp', 'dash'):
        assert isinstance(state[f'{prefix}_demographics'], pd.DataFrame)
        assert state[f'{prefix}_is_demo'] is False
    # Cross-DB must NOT be auto-configured from a single-DB exports bundle
    assert 'multidb_data_root' not in state
    assert 'multidb_selected' not in state


def test_ensure_workspace_from_exports_preserves_existing_selected_patient():
    loaded = _mimic_iv_concepts()
    state: dict = {'loaded_concepts': loaded, 'selected_patient': 2}
    ok, _ = _ensure_cohort_real_workspace_from_loaded_concepts(state, lang='en')
    assert ok is True
    # Patient 2 is in the cohort, so it should be preserved.
    assert state['selected_patient'] == 2


def test_ensure_workspace_resets_invalid_selected_patient():
    loaded = _mimic_iv_concepts()
    state: dict = {'loaded_concepts': loaded, 'selected_patient': 999}
    ok, _ = _ensure_cohort_real_workspace_from_loaded_concepts(state, lang='en')
    assert ok is True
    # 999 not in cohort → reset to first patient_id.
    assert state['selected_patient'] == 1


def test_workspace_ready_check():
    state: dict = {}
    assert _cohort_real_workspace_ready(state) is False

    loaded = _mimic_iv_concepts()
    state = {'loaded_concepts': loaded}
    _ensure_cohort_real_workspace_from_loaded_concepts(state, lang='en')
    assert _cohort_real_workspace_ready(state) is True


def test_matches_sidebar_skips_check_for_loaded_exports():
    """The exports-origin workspace has no real sidebar path; the
    matches-sidebar check should early-return True so the panels don't
    show a stale 'sidebar changed' warning."""
    state: dict = {
        '_cohort_real_ws_origin': 'loaded_exports',
        '_cohort_real_ws_path': '__loaded_exports__',
        '_cohort_real_ws_db': 'miiv',
    }
    assert _cohort_real_workspace_matches_sidebar(state) is True


def test_conceptbundle_dataclass_defaults():
    """ConceptBundle should have safe defaults so partial construction
    (e.g. by future loaders) doesn't crash _seed_workspace_state."""
    b = ConceptBundle(
        demographics=pd.DataFrame({'stay_id': [1]}),
        concepts={},
        patient_ids=[1],
        database='miiv',
        origin='raw_schema',
        data_path='/tmp/test',
        resolved_path='/tmp/test/mimiciv',
        id_col='stay_id',
    )
    assert b.time_col == 'charttime'
    assert b.max_patients == 0
    assert b.errors == []
    assert b.loaded_data_origin == ''
    assert b.configure_multidb is False
