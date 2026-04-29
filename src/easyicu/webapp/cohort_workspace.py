"""Shared cohort workspace state helpers for the EasyICU webapp."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import os

import pandas as pd

from easyicu.webapp.cohort_filters import _get_sepsis_runtime_options
from easyicu.webapp.data_paths import _default_real_database, _default_real_data_root, find_database_path
from easyicu.webapp.demo_data import (
    _generate_mock_cohort_dashboard_data,
    _generate_mock_demographics,
    _generate_mock_multidb_data,
)


COHORT_DEMO_MULTIDB_CONCEPTS = [
    'hr', 'sbp', 'dbp', 'map', 'temp', 'resp', 'spo2',
    'glu', 'na', 'k', 'crea', 'bili', 'lact',
    'hgb', 'plt', 'wbc',
    'ph', 'po2', 'pco2', 'fio2',
    'sofa2', 'sofa2_resp', 'sofa2_coag', 'sofa2_liver',
    'sofa2_cardio', 'sofa2_cns', 'sofa2_renal',
]


def _cohort_demo_workspace_ready(state: dict[str, Any]) -> bool:
    """Return whether all shared Cohort Analysis demo panels have data."""
    return bool(
        state.get('cohort_is_demo')
        and state.get('grp_is_demo')
        and state.get('multidb_is_demo')
        and state.get('dash_is_demo')
        and isinstance(state.get('grp_demographics'), pd.DataFrame)
        and not state.get('grp_demographics').empty
        and bool(state.get('multidb_data'))
        and isinstance(state.get('dash_demographics'), pd.DataFrame)
        and not state.get('dash_demographics').empty
    )


def _ensure_cohort_demo_workspace(
    state: dict[str, Any],
    *,
    lang: str = 'en',
    n_patients: Optional[int] = None,
    force: bool = False,
) -> None:
    """Prepare all Cohort Analysis demo panels once and share the same state."""
    mock_params = state.get('mock_params') if isinstance(state.get('mock_params'), dict) else {}
    state['mock_params'] = mock_params

    patient_count = n_patients if n_patients is not None else mock_params.get('n_patients', 100)
    try:
        patient_count = int(patient_count)
    except (TypeError, ValueError):
        patient_count = 100
    patient_count = max(1, patient_count)
    mock_params['n_patients'] = patient_count

    if force or not (state.get('grp_is_demo') and isinstance(state.get('grp_demographics'), pd.DataFrame)):
        state['grp_demographics'] = _generate_mock_demographics(patient_count, lang)
        state['grp_loaded_db'] = 'demo'
        state['grp_is_demo'] = True
        state.pop('grp_feature_data', None)

    if force or not (state.get('multidb_is_demo') and state.get('multidb_data')):
        state['multidb_data'] = _generate_mock_multidb_data(lang)
        state['multidb_concepts'] = list(COHORT_DEMO_MULTIDB_CONCEPTS)
        state['multidb_is_demo'] = True

    if force or not (state.get('dash_is_demo') and isinstance(state.get('dash_demographics'), pd.DataFrame)):
        state['dash_demographics'] = _generate_mock_cohort_dashboard_data(lang)
        state['dash_loaded_db'] = 'Demo'
        state['dash_is_demo'] = True

    if force or not isinstance(state.get('reclass_demo_df'), pd.DataFrame):
        dash_df = state.get('dash_demographics')
        if isinstance(dash_df, pd.DataFrame) and not dash_df.empty:
            state['reclass_demo_df'] = dash_df.copy()

    state['cohort_is_demo'] = True


def _ensure_cohort_figure_demo_data(state: dict[str, Any], panel: str, *, lang: str) -> None:
    """Preload the cohort demo data needed by the requested paper-style panel."""
    if panel in {'Group Contrast', 'Coverage Audit', 'Cross-DB Benchmark', 'Cohort Snapshot', 'SOFA-1 vs SOFA-2'}:
        _ensure_cohort_demo_workspace(state, lang=lang)


_REAL_WORKSPACE_DEFAULT_MAX_PATIENTS = 100


_REAL_WORKSPACE_MAX_PATIENTS = 5000


_REAL_WORKSPACE_PREVIEW_CONCEPTS = [
    'hr', 'map', 'resp', 'temp', 'spo2', 'crea', 'bili', 'lact', 'glu', 'plt',
]


_REAL_WORKSPACE_SOFA_CONCEPTS = [
    'sofa', 'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal',
    'sofa2', 'sofa2_resp', 'sofa2_coag', 'sofa2_liver', 'sofa2_cardio', 'sofa2_cns', 'sofa2_renal',
]


def _cohort_real_workspace_ready(state: dict[str, Any]) -> bool:
    """Return whether the shared real-data workspace is prepared for all panels."""
    return bool(
        state.get('_cohort_real_ws_ready')
        and isinstance(state.get('_cohort_real_ws_demographics'), pd.DataFrame)
        and not state.get('_cohort_real_ws_demographics').empty
    )


def _cohort_real_workspace_matches_sidebar(state: dict[str, Any]) -> bool:
    """Check if the loaded workspace still matches the sidebar-validated path."""
    ws_path = state.get('_cohort_real_ws_path', '')
    ws_db = state.get('_cohort_real_ws_db', '')
    sidebar_path = _default_real_data_root()
    sidebar_db = _default_real_database()
    return bool(ws_path and ws_path == sidebar_path and ws_db == sidebar_db)


def _ensure_cohort_real_workspace(
    state: dict[str, Any],
    *,
    lang: str = 'en',
    max_patients: int = _REAL_WORKSPACE_DEFAULT_MAX_PATIENTS,
    load_concepts: bool = True,
    force: bool = False,
) -> tuple[bool, str]:
    """Load shared real-data workspace for all Cohort Analysis panels.

    Returns (success, message).
    """
    import streamlit as st

    database = _default_real_database()
    data_path = _default_real_data_root()
    if not data_path or not Path(data_path).exists():
        return False, ("Please validate a real data path in the sidebar first."
                       if lang == 'en' else "请先在侧边栏验证真实数据路径。")

    resolved_path = find_database_path(data_path, database)
    if not os.path.isdir(resolved_path):
        return False, (f"Database path not found: {resolved_path}"
                       if lang == 'en' else f"数据库路径不存在: {resolved_path}")

    # Skip if already loaded for same path+db and not forced
    if (not force
        and _cohort_real_workspace_ready(state)
        and state.get('_cohort_real_ws_path') == data_path
        and state.get('_cohort_real_ws_db') == database):
        return True, ""

    errors: list[str] = []
    loaded_concepts_dict: dict[str, Any] = {}

    # 1) Demographics
    try:
        from easyicu.patient_filter import PatientFilter
        pf = PatientFilter(database=database, data_path=resolved_path, verbose=False)
        demographics_df = pf._load_demographics()
        if len(demographics_df) > max_patients:
            demographics_df = demographics_df.head(max_patients)
        id_col = 'stay_id' if 'stay_id' in demographics_df.columns else 'patient_id'
        patient_ids = demographics_df[id_col].dropna().astype(int).tolist()
    except Exception as e:
        return False, f"Failed to load demographics: {e}"

    # 2) Preview concepts + SOFA (best-effort)
    if load_concepts:
        try:
            from easyicu import load_concepts as lc
            all_concepts = _REAL_WORKSPACE_PREVIEW_CONCEPTS + _REAL_WORKSPACE_SOFA_CONCEPTS
            concept_df = lc(
                concepts=all_concepts,
                database=database,
                data_path=resolved_path,
                patient_ids=patient_ids,
                verbose=False,
                **_get_sepsis_runtime_options(),
            )
            if concept_df is not None and not concept_df.empty:
                detected_id_col = next(
                    (col for col in ['stay_id', 'patient_id', 'patientunitstayid', 'admissionid', 'patientid']
                     if col in concept_df.columns), None)
                time_cols = [col for col in ['charttime', 'time'] if col in concept_df.columns]
                base_cols = ([detected_id_col] if detected_id_col else []) + time_cols
                for concept in all_concepts:
                    if concept in concept_df.columns:
                        keep_cols = base_cols + [concept]
                        loaded_concepts_dict[concept] = concept_df[keep_cols].dropna(subset=[concept]).copy()
        except Exception as e:
            errors.append(f"Concept loading partial failure: {e}")

    # ---- Populate shared state ----
    state['_cohort_real_ws_ready'] = True
    state['_cohort_real_ws_path'] = data_path
    state['_cohort_real_ws_db'] = database
    state['_cohort_real_ws_resolved_path'] = resolved_path
    state['_cohort_real_ws_demographics'] = demographics_df
    state['_cohort_real_ws_patient_ids'] = patient_ids
    state['_cohort_real_ws_max_patients'] = max_patients
    state['_cohort_real_ws_concepts'] = loaded_concepts_dict
    state['_cohort_real_ws_errors'] = errors

    # Keep the global review footer and patient selectors aligned with the
    # newly loaded real workspace instead of leaving stale demo IDs behind.
    state['patient_ids'] = patient_ids
    state['available_patient_ids'] = patient_ids
    state['all_patient_count'] = len(patient_ids)
    state['id_col'] = id_col
    state['time_col'] = 'charttime'
    state['selected_patient'] = patient_ids[0] if patient_ids else None
    state['selected_concepts'] = list(loaded_concepts_dict.keys())

    # Seed individual panel keys so subpanels see data without re-loading
    state['grp_demographics'] = demographics_df.copy()
    state['grp_loaded_db'] = database
    state['grp_loaded_path'] = resolved_path
    state['grp_is_demo'] = False
    state['grp_data_root'] = data_path
    state['grp_db_select'] = database

    state['dash_demographics'] = demographics_df.copy()
    state['dash_loaded_db'] = database
    state['dash_loaded_path'] = resolved_path
    state['dash_is_demo'] = False
    state['dash_data_root'] = data_path
    state['dash_db_select'] = database

    state['multidb_data_root'] = data_path
    state['multidb_selected'] = [database]

    # SOFA concepts → loaded_concepts so reclassification panel picks them up
    if loaded_concepts_dict:
        state['loaded_concepts'] = dict(loaded_concepts_dict)
        state['loaded_data_origin'] = 'real_workspace'
    else:
        state['loaded_concepts'] = {}
        state['loaded_data_origin'] = 'real_workspace_demographics_only'

    msg_parts = [f"Loaded {len(demographics_df):,} patients"]
    if loaded_concepts_dict:
        msg_parts.append(f"{len(loaded_concepts_dict)} concepts")
    if errors:
        msg_parts.append(f"({len(errors)} warnings)")
    return True, "; ".join(msg_parts)
