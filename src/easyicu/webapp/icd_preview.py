"""ICD match preview helpers for the EasyICU Streamlit app."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from easyicu.webapp.cohort_filters import _split_query_tokens, _supports_icd_filter


def _clear_icd_preview_state() -> None:
    """Remove ICD preview caches and temporary UI state."""
    for key in (
        '_icd_preview_cache_include',
        '_icd_preview_cache_exclude',
    ):
        st.session_state.pop(key, None)


def _render_icd_preview_main_panel(lang: str) -> None:
    """Render ICD preview results in the main content area instead of the sidebar."""
    if not _supports_icd_filter(st.session_state.get('database')):
        _clear_icd_preview_state()
        return

    include_query = str(st.session_state.get('cohort_filter', {}).get('icd_include_query', '')).strip()
    exclude_query = str(st.session_state.get('cohort_filter', {}).get('icd_exclude_query', '')).strip()
    preview_specs = [
        ("include", include_query, "Include" if lang == 'en' else "包含"),
        ("exclude", exclude_query, "Exclude" if lang == 'en' else "排除"),
    ]

    active_previews = []
    for preview_key, preview_query, preview_label in preview_specs:
        tokens = _split_query_tokens(preview_query)
        cached = st.session_state.get(f'_icd_preview_cache_{preview_key}')
        if not tokens or not cached or cached.get('tokens') != tokens:
            continue
        active_previews.append((preview_key, preview_label, cached))

    if not active_previews:
        return

    title = "🧾 ICD Match Preview" if lang == 'en' else "🧾 ICD 匹配预览"
    header_cols = st.columns([6, 1.4])
    with header_cols[0]:
        st.markdown(f"#### {title}")
    with header_cols[1]:
        clear_label = "🧹 Clear Preview" if lang == 'en' else "🧹 清除预览"
        if st.button(clear_label, key="clear_icd_preview_main", use_container_width=True):
            _clear_icd_preview_state()
            st.rerun()

    include_cached = next((cached for key, _, cached in active_previews if key == 'include'), None)
    exclude_cached = next((cached for key, _, cached in active_previews if key == 'exclude'), None)
    total_patients = 0
    if include_cached:
        total_patients = int(include_cached.get('total_patients', 0) or 0)
    elif exclude_cached:
        total_patients = int(exclude_cached.get('total_patients', 0) or 0)

    include_ids = set(include_cached.get('matched_ids', [])) if include_cached else set()
    exclude_ids = set(exclude_cached.get('matched_ids', [])) if exclude_cached else set()

    if include_cached:
        final_ids = include_ids - exclude_ids
        final_count = len(final_ids)
    elif exclude_cached:
        final_count = max(total_patients - len(exclude_ids), 0)
    else:
        final_count = 0

    final_pct = final_count / total_patients * 100 if total_patients > 0 else 0
    if lang == 'en':
        st.info(f"🧮 Final cohort after ICD filters: **{final_count:,}** / {total_patients:,} patients ({final_pct:.1f}%)")
    else:
        st.info(f"🧮 ICD 筛选后的最终队列：**{final_count:,}** / {total_patients:,} 位患者 ({final_pct:.1f}%)")

    cols = st.columns(len(active_previews))
    for col, (_, preview_label, preview_result) in zip(cols, active_previews):
        with col:
            if preview_result.get('error'):
                st.warning(preview_result['error'])
                continue

            matched = preview_result.get('matched_patients', 0)
            total = preview_result.get('total_patients', 0)
            pct = matched / total * 100 if total > 0 else 0
            if lang == 'en':
                st.success(f"📊 {preview_label}: matched **{matched:,}** / {total:,} patients ({pct:.1f}%)")
            else:
                st.success(f"📊 {preview_label}: 匹配到 **{matched:,}** / {total:,} 位患者 ({pct:.1f}%)")

            top_codes = preview_result.get('top_codes')
            if top_codes is not None and len(top_codes) > 0:
                table_label = (
                    f"📋 Top matching ICD codes ({preview_label})"
                    if lang == 'en' else
                    f"📋 匹配频率最高的 ICD 编码（{preview_label}）"
                )
                st.markdown(f"**{table_label}**")
                st.dataframe(top_codes, use_container_width=True, hide_index=True)


def _preview_icd_match(data_path: Path, database: str, tokens: list[str]) -> dict:
    """Preview ICD code matching: return matched patient count and top codes."""
    result = {
        'tokens': tokens,
        'matched_patients': 0,
        'matched_ids': [],
        'total_patients': 0,
        'top_codes': None,
        'error': None,
    }
    try:
        DB_META_PREVIEW = {
            'miiv': {'id_col': 'stay_id', 'icu_table': 'icustays.parquet'},
            'mimic': {'id_col': 'icustay_id', 'icu_table': 'icustays.parquet'},
            'eicu': {'id_col': 'patientunitstayid', 'icu_table': 'patient.parquet'},
        }
        meta = DB_META_PREVIEW.get(database)
        if not meta:
            result['error'] = f"ICD preview not supported for {database}"
            return result
        icu_path = data_path / meta['icu_table']
        if not icu_path.exists():
            result['error'] = f"ICU table not found: {icu_path.name}"
            return result
        icu_df = pd.read_parquet(icu_path)
        icu_df.columns = [c.lower() for c in icu_df.columns]
        id_col = meta['id_col'].lower()
        result['total_patients'] = icu_df[id_col].nunique()

        if database in ('miiv', 'mimic'):
            diag_path = data_path / 'diagnoses_icd.parquet'
            if not diag_path.exists():
                result['error'] = f"diagnoses_icd.parquet not found"
                return result
            diag_df = pd.read_parquet(diag_path, columns=['hadm_id', 'icd_code', 'icd_version'] if database == 'miiv' else ['hadm_id', 'icd_code'])
            codes = diag_df['icd_code'].astype(str).str.upper().str.replace('.', '', regex=False)
            norm_tokens = [tok.upper().replace('.', '') for tok in tokens]
            diag_mask = pd.Series(False, index=diag_df.index)
            for tok in norm_tokens:
                diag_mask |= codes.str.startswith(tok)
            matched_diag = diag_df.loc[diag_mask].copy()
            if 'hadm_id' in icu_df.columns:
                matched_hadm = set(matched_diag['hadm_id'].dropna().unique())
                matched_ids = set(icu_df.loc[icu_df['hadm_id'].isin(matched_hadm), id_col].dropna().unique())
                result['matched_patients'] = len(matched_ids)
                result['matched_ids'] = sorted(matched_ids)
            # Top ICD codes
            matched_diag['icd_code_clean'] = codes[diag_mask]
            code_counts = matched_diag['icd_code_clean'].value_counts().head(20).reset_index()
            code_counts.columns = ['ICD Code', 'Count']
            # Try enrich with descriptions
            try:
                d_path = data_path / 'd_icd_diagnoses.parquet'
                if d_path.exists():
                    d_df = pd.read_parquet(d_path)
                    d_df.columns = [c.lower() for c in d_df.columns]
                    if 'icd_code' in d_df.columns and 'long_title' in d_df.columns:
                        d_df['icd_code'] = d_df['icd_code'].astype(str).str.upper().str.replace('.', '', regex=False)
                        desc_map = dict(zip(d_df['icd_code'], d_df['long_title']))
                        code_counts['Description'] = code_counts['ICD Code'].map(desc_map).fillna('')
            except Exception:
                pass
            result['top_codes'] = code_counts

        elif database == 'eicu':
            diag_path = data_path / 'diagnosis.parquet'
            if not diag_path.exists():
                result['error'] = f"diagnosis.parquet not found"
                return result
            diag_df = pd.read_parquet(diag_path)
            diag_df.columns = [c.lower() for c in diag_df.columns]
            if 'patientunitstayid' not in diag_df.columns:
                result['error'] = "patientunitstayid not found in diagnosis table"
                return result
            diag_text = pd.Series('', index=diag_df.index, dtype='object')
            if 'icd9code' in diag_df.columns:
                diag_text = diag_text.str.cat(diag_df['icd9code'].astype(str), sep=' ', na_rep='')
            if 'diagnosisstring' in diag_df.columns:
                diag_text = diag_text.str.cat(diag_df['diagnosisstring'].astype(str), sep=' ', na_rep='')
            diag_text_lower = diag_text.str.lower().str.replace('.', '', regex=False)
            diag_mask = pd.Series(False, index=diag_df.index)
            for tok in tokens:
                diag_mask |= diag_text_lower.str.contains(str(tok).lower().replace('.', ''), na=False)
            matched_diag = diag_df.loc[diag_mask]
            matched_ids = set(matched_diag['patientunitstayid'].dropna().unique())
            result['matched_patients'] = len(matched_ids)
            result['matched_ids'] = sorted(matched_ids)
            # Top codes for eICU
            if 'icd9code' in matched_diag.columns:
                code_counts = matched_diag['icd9code'].dropna().astype(str).value_counts().head(20).reset_index()
                code_counts.columns = ['ICD Code', 'Count']
                if 'diagnosisstring' in matched_diag.columns:
                    ds_map = dict(zip(matched_diag['icd9code'].astype(str), matched_diag['diagnosisstring'].astype(str)))
                    code_counts['Description'] = code_counts['ICD Code'].map(ds_map).fillna('')
                result['top_codes'] = code_counts
    except Exception as e:
        result['error'] = str(e)
    return result
