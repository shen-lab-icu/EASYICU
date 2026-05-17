"""Shared cohort workspace state helpers for the EasyICU webapp.

Cohort Analysis accepts two real-data input sources:

* **Raw ICU schema** — sidebar-validated path with ``icustays`` /
  ``patients`` / ``admissions`` etc. Loaded via
  :class:`easyicu.patient_filter.PatientFilter`.
* **Module exports** — already-loaded concept DataFrames sitting in
  ``state['loaded_concepts']`` (typically populated by Quick
  Visualization's "Previously Exported Data" path).

Both paths converge on a single :class:`ConceptBundle` dataclass.
:func:`_seed_workspace_state` is the sole writer of cohort workspace
``st.session_state`` keys, so panel code stays oblivious to the
ingestion branch.

To add a new source: write ``_bundle_from_<source>(...) -> (ok, msg,
ConceptBundle | None)`` and call :func:`_seed_workspace_state`. Do not
write panel state outside the seeder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class ConceptBundle:
    """Canonical input shape for the shared Cohort Analysis workspace.

    Every code path that loads real data for Cohort Analysis must
    produce a ``ConceptBundle``; :func:`_seed_workspace_state` then
    writes ``st.session_state``. Panel state stays consistent regardless
    of source.
    """

    demographics: pd.DataFrame
    concepts: dict[str, pd.DataFrame]
    patient_ids: list[Any]
    database: str
    origin: str  # 'raw_schema' | 'loaded_exports'
    data_path: str  # user-visible path or sentinel '__loaded_exports__'
    resolved_path: str  # filesystem-resolved DB folder for raw; '' for exports
    id_col: str
    time_col: str = 'charttime'
    max_patients: int = 0
    errors: list[str] = field(default_factory=list)
    # Distinct from `origin` because existing app code reads several
    # markers ('real_workspace' / 'real_workspace_demographics_only' /
    # 'loaded_exports') for analytics and panel availability.
    loaded_data_origin: str = ''
    # When False, do not seed Cross-DB's multidb_* keys — exports of one
    # database can't power a cross-database comparison.
    configure_multidb: bool = False


_LOADED_EXPORTS_SENTINEL = '__loaded_exports__'


def _seed_workspace_state(state: dict[str, Any], bundle: ConceptBundle) -> None:
    """Write all cohort workspace state keys from a :class:`ConceptBundle`.

    Single source of truth for the ``_cohort_real_ws_*`` keys plus the
    panel-specific seed keys (``grp_*``, ``dash_*``, ``multidb_*``) and
    the global review-footer keys (``patient_ids``, ``selected_patient``
    etc.). Both raw-schema and module-exports ingest paths funnel here.
    """
    demographics_df = bundle.demographics
    patient_ids = bundle.patient_ids
    database = bundle.database
    data_path = bundle.data_path
    resolved_path = bundle.resolved_path

    # ---- Cohort Analysis shared workspace ----
    state['_cohort_real_ws_ready'] = True
    state['_cohort_real_ws_origin'] = bundle.origin
    state['_cohort_real_ws_path'] = data_path
    state['_cohort_real_ws_db'] = database
    state['_cohort_real_ws_resolved_path'] = resolved_path
    state['_cohort_real_ws_demographics'] = demographics_df
    state['_cohort_real_ws_patient_ids'] = patient_ids
    state['_cohort_real_ws_max_patients'] = (
        bundle.max_patients if bundle.max_patients else len(patient_ids)
    )
    state['_cohort_real_ws_concepts'] = dict(bundle.concepts)
    state['_cohort_real_ws_errors'] = list(bundle.errors)

    # ---- Global review footer / patient pickers ----
    state['patient_ids'] = patient_ids
    state['available_patient_ids'] = patient_ids
    state['all_patient_count'] = len(patient_ids)
    state['id_col'] = bundle.id_col
    state['time_col'] = bundle.time_col
    # Preserve a user-chosen patient if it's still in the cohort.
    if bundle.origin == 'loaded_exports' and state.get('selected_patient') in patient_ids:
        pass  # keep existing
    else:
        state['selected_patient'] = patient_ids[0] if patient_ids else None
    # Raw path historically (over)wrote selected_concepts; preserve that.
    if bundle.origin == 'raw_schema':
        state['selected_concepts'] = list(bundle.concepts.keys())

    # ---- Per-panel seed keys (Group Contrast + Cohort Snapshot) ----
    panel_path = data_path  # exports use the sentinel; raw uses the real path
    for prefix in ('grp', 'dash'):
        state[f'{prefix}_demographics'] = demographics_df.copy()
        state[f'{prefix}_loaded_db'] = database
        state[f'{prefix}_loaded_path'] = resolved_path if bundle.origin == 'raw_schema' else panel_path
        state[f'{prefix}_is_demo'] = False
        state[f'{prefix}_data_root'] = panel_path
        state[f'{prefix}_db_select'] = database

    # ---- Cross-DB seed (only when the source can support it) ----
    if bundle.configure_multidb:
        state['multidb_data_root'] = data_path
        state['multidb_selected'] = [database]
    # Otherwise: leave multidb_* untouched so the Cross-DB panel falls
    # back to its in-panel ICU Data Root input.

    # ---- SOFA reclassification / global loaded concepts ----
    if bundle.concepts:
        state['loaded_concepts'] = dict(bundle.concepts)
    elif bundle.origin == 'raw_schema':
        # Raw-schema bundle with zero concepts → clear stale demo state.
        state['loaded_concepts'] = {}
    # Exports bundle: state['loaded_concepts'] was already populated by
    # Quick Visualization's loader before this function runs; don't
    # overwrite it (the bundle copy may be identical anyway).
    state['loaded_data_origin'] = bundle.loaded_data_origin or bundle.origin


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
        state['dash_demographics'] = _generate_mock_cohort_dashboard_data(lang, n_patients=patient_count)
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
    'death', 'los_icu', 'aki', 'rrt', 'vent_ind', 'mech_vent', 'vaso_ind', 'abx',
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
    """Check if the loaded workspace still matches the sidebar-validated path.

    Workspaces sourced from already-loaded module exports
    (``_cohort_real_ws_origin == 'loaded_exports'``) intentionally do not
    track the sidebar Data Path — they were built from
    ``state['loaded_concepts']``, not by reading raw DB schema. Treat them
    as "in sync" so the panels don't show a stale "sidebar changed" warning.
    """
    if state.get('_cohort_real_ws_origin') == 'loaded_exports':
        return True
    ws_path = state.get('_cohort_real_ws_path', '')
    ws_db = state.get('_cohort_real_ws_db', '')
    sidebar_path = _default_real_data_root()
    sidebar_db = _default_real_database()
    return bool(ws_path and ws_path == sidebar_path and ws_db == sidebar_db)


# Demographic concepts that map 1:1 from prepared module exports
# (`demographics_*.parquet`, `outcome_*.parquet`) into the columns the
# Cohort Analysis panels expect from ``_load_demographics()``.
_DEMOGRAPHIC_CONCEPT_COLUMNS = (
    'age', 'sex', 'gender', 'height', 'weight', 'bmi', 'adm',
    'los_icu', 'los_hosp', 'death',
)


def _bundle_from_loaded_concepts(
    loaded: dict[str, Any],
    *,
    lang: str = 'en',
    database_hint: str = '',
    id_col_hint: str = '',
) -> tuple[bool, str, Optional[ConceptBundle]]:
    """Build a :class:`ConceptBundle` from already-loaded module exports.

    Pure factory — does not touch ``st.session_state``. Callers wrap this
    with state validation and pass the result to
    :func:`_seed_workspace_state`.
    """
    if not loaded:
        return False, (
            "No loaded concepts to bridge. Load module exports via Quick "
            "Visualization first." if lang == 'en'
            else "未检测到已加载的概念。请先在 Quick Visualization 加载模块导出文件。"
        ), None

    # Each supported database uses a different id column. Prefer the
    # caller-supplied hint, then fall back to a ranked candidate list and
    # pick whichever actually appears in the loaded DataFrames.
    id_candidates: list[str] = []
    if id_col_hint:
        id_candidates.append(id_col_hint)
    # MIMIC-IV (stay_id) / MIMIC-III (icustay_id) / eICU (patientunitstayid)
    # / AmsterdamUMCdb (admissionid) / HiRID (patientid) / SICdb (CaseID).
    for cand in ('stay_id', 'icustay_id', 'patientunitstayid',
                 'admissionid', 'patientid', 'CaseID'):
        if cand not in id_candidates:
            id_candidates.append(cand)

    scaffold = None
    id_col: Optional[str] = None
    preferred_keys = ('age', 'sex', 'gender', 'los_icu', 'los_hosp', 'death')
    for candidate in id_candidates:
        for key in preferred_keys:
            df = loaded.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty and candidate in df.columns:
                scaffold = df[[candidate]].drop_duplicates()
                id_col = candidate
                break
        if scaffold is not None:
            break
    if scaffold is None:
        for candidate in id_candidates:
            for df in loaded.values():
                if isinstance(df, pd.DataFrame) and not df.empty and candidate in df.columns:
                    scaffold = df[[candidate]].drop_duplicates()
                    id_col = candidate
                    break
            if scaffold is not None:
                break
    if scaffold is None or scaffold.empty or id_col is None:
        return False, (
            "Loaded concepts don't include a known stay/patient id column "
            "(stay_id / icustay_id / patientunitstayid / admissionid / "
            "patientid / CaseID)."
            if lang == 'en' else
            "已加载概念中缺少识别的患者/住院 ID 列 "
            "（stay_id / icustay_id / patientunitstayid / admissionid / "
            "patientid / CaseID）。"
        ), None

    demographics_df = scaffold.copy()
    for concept_key in _DEMOGRAPHIC_CONCEPT_COLUMNS:
        df = loaded.get(concept_key)
        if not isinstance(df, pd.DataFrame) or df.empty or concept_key not in df.columns:
            continue
        d = df[[id_col, concept_key]].drop_duplicates(subset=[id_col], keep='first')
        demographics_df = demographics_df.merge(d, on=id_col, how='left')

    # Downstream panels read both ``sex`` and ``gender`` interchangeably;
    # ``_load_miiv_demographics`` also exposes ``los_hours`` (days*24).
    if 'sex' in demographics_df.columns and 'gender' not in demographics_df.columns:
        demographics_df['gender'] = demographics_df['sex']
    if 'gender' in demographics_df.columns and 'sex' not in demographics_df.columns:
        demographics_df['sex'] = demographics_df['gender']
    if 'los_icu' in demographics_df.columns and 'los_hours' not in demographics_df.columns:
        demographics_df['los_hours'] = pd.to_numeric(
            demographics_df['los_icu'], errors='coerce',
        ) * 24

    try:
        patient_ids = demographics_df[id_col].dropna().astype(int).tolist()
    except (ValueError, TypeError):
        patient_ids = demographics_df[id_col].dropna().tolist()
    if not patient_ids:
        return False, (
            "No patient IDs derived from loaded concepts."
            if lang == 'en' else "已加载概念无法解析出患者 ID。"
        ), None

    database = database_hint or _default_real_database()

    bundle = ConceptBundle(
        demographics=demographics_df,
        concepts=dict(loaded),
        patient_ids=patient_ids,
        database=database,
        origin='loaded_exports',
        data_path=_LOADED_EXPORTS_SENTINEL,
        resolved_path='',
        id_col=id_col,
        max_patients=len(patient_ids),
        errors=[],
        loaded_data_origin='loaded_exports',
        configure_multidb=False,  # Cross-DB needs ≥2 DBs' raw schema
    )

    n_demo_cols = sum(
        1 for c in _DEMOGRAPHIC_CONCEPT_COLUMNS if c in demographics_df.columns
    )
    msg = (
        f"Bridged {len(demographics_df):,} patients × {len(loaded)} concepts "
        f"({n_demo_cols} demographic columns recovered)"
        if lang == 'en' else
        f"已桥接 {len(demographics_df):,} 名患者 × {len(loaded)} 个概念"
        f"（恢复 {n_demo_cols} 个人口统计学列）"
    )
    return True, msg, bundle


def _ensure_cohort_real_workspace_from_loaded_concepts(
    state: dict[str, Any], *, lang: str = 'en',
) -> tuple[bool, str]:
    """Build the shared Cohort Analysis workspace from already-loaded
    module exports, bypassing the raw-schema ``_load_demographics()`` path.

    Thin wrapper: read ``loaded_concepts`` from state, call
    :func:`_bundle_from_loaded_concepts`, then :func:`_seed_workspace_state`.
    Supports Group Contrast, Coverage, Cohort Snapshot, SOFA Δ. Cross-DB
    Benchmark stays gated.
    """
    loaded = state.get('loaded_concepts') or {}
    ok, msg, bundle = _bundle_from_loaded_concepts(
        loaded,
        lang=lang,
        database_hint=state.get('database') or _default_real_database(),
        id_col_hint=state.get('id_col') or '',
    )
    if not ok or bundle is None:
        return False, msg
    _seed_workspace_state(state, bundle)
    return True, msg


def _bundle_from_raw_schema(
    database: str,
    data_path: str,
    *,
    lang: str = 'en',
    max_patients: int = 100,
    load_concepts: bool = True,
) -> tuple[bool, str, Optional[ConceptBundle]]:
    """Build a :class:`ConceptBundle` from a sidebar-validated raw ICU
    database root (icustays / patients / admissions schema).

    Pure factory — does not touch ``st.session_state``. Resolution of
    the DB subfolder under ``data_path`` happens here so callers stay
    simple.
    """
    resolved_path = find_database_path(data_path, database)
    if not os.path.isdir(resolved_path):
        return False, (
            f"Database path not found: {resolved_path}"
            if lang == 'en' else f"数据库路径不存在: {resolved_path}"
        ), None

    errors: list[str] = []

    # 1) Demographics via PatientFilter.
    try:
        from easyicu.patient_filter import PatientFilter
        pf = PatientFilter(database=database, data_path=resolved_path, verbose=False)
        demographics_df = pf._load_demographics()
        if len(demographics_df) > max_patients:
            demographics_df = demographics_df.head(max_patients)
        id_col = 'stay_id' if 'stay_id' in demographics_df.columns else 'patient_id'
        patient_ids = demographics_df[id_col].dropna().astype(int).tolist()
    except Exception as exc:
        return False, f"Failed to load demographics: {exc}", None

    # 2) Preview concepts + SOFA (best-effort).
    loaded_concepts_dict: dict[str, Any] = {}
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
                    (col for col in ['stay_id', 'patient_id', 'patientunitstayid',
                                     'admissionid', 'patientid']
                     if col in concept_df.columns), None)
                time_cols = [col for col in ['charttime', 'time'] if col in concept_df.columns]
                base_cols = ([detected_id_col] if detected_id_col else []) + time_cols
                for concept in all_concepts:
                    if concept in concept_df.columns:
                        keep_cols = base_cols + [concept]
                        loaded_concepts_dict[concept] = (
                            concept_df[keep_cols].dropna(subset=[concept]).copy()
                        )
        except Exception as exc:
            errors.append(f"Concept loading partial failure: {exc}")

    loaded_data_origin = (
        'real_workspace' if loaded_concepts_dict
        else 'real_workspace_demographics_only'
    )

    bundle = ConceptBundle(
        demographics=demographics_df,
        concepts=loaded_concepts_dict,
        patient_ids=patient_ids,
        database=database,
        origin='raw_schema',
        data_path=data_path,
        resolved_path=resolved_path,
        id_col=id_col,
        max_patients=max_patients,
        errors=errors,
        loaded_data_origin=loaded_data_origin,
        configure_multidb=True,  # Raw root can seed Cross-DB defaults
    )

    msg_parts = [f"Loaded {len(demographics_df):,} patients"]
    if loaded_concepts_dict:
        msg_parts.append(f"{len(loaded_concepts_dict)} concepts")
    if errors:
        msg_parts.append(f"({len(errors)} warnings)")
    return True, "; ".join(msg_parts), bundle


def _ensure_cohort_real_workspace(
    state: dict[str, Any],
    *,
    lang: str = 'en',
    max_patients: int = _REAL_WORKSPACE_DEFAULT_MAX_PATIENTS,
    load_concepts: bool = True,
    force: bool = False,
) -> tuple[bool, str]:
    """Load shared real-data workspace for all Cohort Analysis panels.

    Thin wrapper: validate sidebar path → call
    :func:`_bundle_from_raw_schema` → :func:`_seed_workspace_state`.
    """
    database = _default_real_database()
    data_path = _default_real_data_root()
    if not data_path or not Path(data_path).exists():
        return False, (
            "Please validate a real data path in the sidebar first."
            if lang == 'en' else "请先在侧边栏验证真实数据路径。"
        )

    # Skip if already loaded for same path+db and not forced
    if (not force
        and _cohort_real_workspace_ready(state)
        and state.get('_cohort_real_ws_path') == data_path
        and state.get('_cohort_real_ws_db') == database):
        return True, ""

    ok, msg, bundle = _bundle_from_raw_schema(
        database, data_path,
        lang=lang, max_patients=max_patients, load_concepts=load_concepts,
    )
    if not ok or bundle is None:
        return False, msg
    _seed_workspace_state(state, bundle)
    return True, msg
