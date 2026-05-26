"""Data dictionary and feature-definition UI helpers.

Layout mirrors ``easyicu design/page-misc.jsx`` ``PageDataDict``:
design header + bilingual subtitle + search field + concept browser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from easyicu.webapp.cohort_charts import render_design_page_header
from easyicu.webapp.compat import _dataframe_compat as _st_dataframe_compat
from easyicu.webapp.components.constants import get_concept_groups
from easyicu.webapp.concept_catalog import CONCEPT_DESCRIPTIONS, CONCEPT_DICTIONARY

_PROTECTED_NAMES = {
    'render_data_dictionary',
    '_render_category_table',
    '_format_definition_list',
    '_get_table_defaults',
    '_format_source_selector',
    '_collect_recursive_concept_sources',
    '_get_feature_definition_rows',
    '_render_feature_definition_panel',
    'render_home_data_dictionary',
    '_APP_CONTEXT',
    '_PROTECTED_NAMES',
    '_install_app_context',
    'Any',
    'Path',
    'pd',
}
_APP_CONTEXT: dict[str, Any] = {}


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose remaining app globals used by this transitional module."""
    _APP_CONTEXT.clear()
    _APP_CONTEXT.update(app_context)
    for name, value in app_context.items():
        if name.startswith('__') or name in _PROTECTED_NAMES:
            continue
        globals()[name] = value


def render_data_dictionary(app_context: dict[str, Any] | None = None):
    """Render data dictionary (aligned with sidebar groups)."""
    if app_context is not None:
        _install_app_context(app_context)

    lang = st.session_state.get('language', 'en')

    n_concepts = sum(len(c) for c in get_concept_groups().values())
    st.markdown(
        render_design_page_header(
            kicker="DATA DICTIONARY",
            title_en="Data dictionary",
            title_zh="数据字典",
            desc=(
                f"{n_concepts} concepts · abbreviations, full names, units, and "
                "per-database source mapping."
                if lang == 'en' else
                f"{n_concepts} 个 concept · 缩写、全称、单位与按数据库的来源映射。"
            ),
            lang=lang,
        ),
        unsafe_allow_html=True,
    )

    # 🔍 搜索框
    search_placeholder = "Search by code, name or description... (e.g. hr, heart rate, lactate)" if lang == 'en' else "按代码、名称或描述搜索... (如 hr、heart rate、心率)"
    search_query = st.text_input(
        "🔍 Search" if lang == 'en' else "🔍 搜索",
        placeholder=search_placeholder,
        key="dict_page_search_input",
    )

    # 获取双语分组
    concept_groups = get_concept_groups()

    # 如果有搜索词，展示搜索结果
    if search_query and search_query.strip():
        query = search_query.strip().lower()
        matched_rows = []
        for cat_name, concepts in concept_groups.items():
            for concept in concepts:
                if concept in CONCEPT_DICTIONARY:
                    eng_name, chn_name, unit = CONCEPT_DICTIONARY[concept]
                    eng_desc, chn_desc = CONCEPT_DESCRIPTIONS.get(concept, ('', ''))
                    if lang == 'en':
                        searchable = f"{concept} {eng_name} {eng_desc}".lower()
                    else:
                        searchable = f"{concept} {eng_name} {chn_name} {eng_desc} {chn_desc}".lower()
                    if query in searchable:
                        if lang == 'en':
                            matched_rows.append({
                                'Code': concept,
                                'Full Name': eng_name,
                                'Category': cat_name,
                                'Description': eng_desc if eng_desc else eng_name,
                                'Unit': unit if unit else '-'
                            })
                        else:
                            matched_rows.append({
                                '代码': concept,
                                '全称': eng_name,
                                '类别': cat_name,
                                '说明': chn_desc if chn_desc else chn_name,
                                '单位': unit if unit else '-'
                            })

        if matched_rows:
            n = len(matched_rows)
            result_text = f"Found **{n}** matching feature(s)" if lang == 'en' else f"找到 **{n}** 个匹配特征"
            st.success(result_text)
            _st_dataframe_compat(
                st,
                pd.DataFrame(matched_rows),
                width="stretch",
                hide_index=True,
                height=min(400, 50 + 35 * n),
            )
        else:
            no_result = "No matching features found." if lang == 'en' else "未找到匹配的特征。"
            st.warning(no_result)
    else:
        # 无搜索词时，使用分类选择器
        all_label = "All" if lang == 'en' else "全部"
        select_label = "Select Category" if lang == 'en' else "选择类别查看"

        selected_category = st.selectbox(
            select_label,
            options=[all_label] + list(concept_groups.keys()),
            index=0,
            key="dict_category_select"
        )

        if selected_category == all_label:
            # 显示所有类别
            for cat_name, concepts in concept_groups.items():
                feat_label = "features" if lang == 'en' else "个特征"
                with st.expander(f"📁 {cat_name} ({len(concepts)} {feat_label})", expanded=False):
                    _render_category_table(concepts, lang)
        else:
            # 只显示选中的类别
            st.markdown(f"#### {selected_category}")
            _render_category_table(concept_groups[selected_category], lang)


def _render_category_table(concepts, lang='en'):
    """Render feature table for a single category with detailed descriptions."""
    rows = []
    for concept in concepts:
        if concept in CONCEPT_DICTIONARY:
            eng_name, chn_name, unit = CONCEPT_DICTIONARY[concept]
            # 获取详细描述
            if concept in CONCEPT_DESCRIPTIONS:
                eng_desc, chn_desc = CONCEPT_DESCRIPTIONS[concept]
            else:
                eng_desc, chn_desc = '', ''

            if lang == 'en':
                rows.append({
                    'Abbr': concept,
                    'Full Name': eng_name,
                    'Description': eng_desc if eng_desc else chn_name,
                    'Unit': unit if unit else '-'
                })
            else:
                rows.append({
                    '缩写': concept,
                    '全名': eng_name,
                    '详细说明': chn_desc if chn_desc else chn_name,
                    '单位': unit if unit else '-'
                })

    if rows:
        df = pd.DataFrame(rows)
        if lang == 'en':
            _st_dataframe_compat(
                st,
                df,
                width="stretch",
                hide_index=True,
                column_config={
                    'Abbr': st.column_config.TextColumn('Abbr', width='small'),
                    'Full Name': st.column_config.TextColumn('Full Name', width='medium'),
                    'Description': st.column_config.TextColumn('Description', width='large'),
                    'Unit': st.column_config.TextColumn('Unit', width='small'),
                }
            )
        else:
            _st_dataframe_compat(
                st,
                df,
                width="stretch",
                hide_index=True,
                column_config={
                    '缩写': st.column_config.TextColumn('缩写', width='small'),
                    '全名': st.column_config.TextColumn('全名', width='medium'),
                    '详细说明': st.column_config.TextColumn('详细说明', width='large'),
                    '单位': st.column_config.TextColumn('单位', width='small'),
                }
            )


def _format_definition_list(values, limit: int = 6) -> str:
    """Format a short, readable comma-separated preview for metadata fields."""
    cleaned = [str(v).strip() for v in values if v not in (None, "", [], {}) and str(v).strip()]
    if not cleaned:
        return "—"
    if len(cleaned) <= limit:
        return ", ".join(cleaned)
    return f"{', '.join(cleaned[:limit])} (+{len(cleaned) - limit} more)"


@st.cache_data(ttl=3600)
def _get_table_defaults() -> dict:
    """Load per-database per-table default column mappings from data-sources.json."""
    import json as _json
    try:
        ds_path = Path(__file__).resolve().parent.parent / 'data' / 'data-sources.json'
        if not ds_path.exists():
            return {}
        with open(ds_path, encoding='utf-8') as f:
            ds_list = _json.load(f)
        result = {}
        for entry in ds_list:
            if not isinstance(entry, dict):
                continue
            db_name = entry.get('name', '')
            tables = entry.get('tables', {})
            if not isinstance(tables, dict):
                continue
            for tbl_name, tbl_def in tables.items():
                if not isinstance(tbl_def, dict):
                    continue
                defaults = tbl_def.get('defaults', {})
                if not isinstance(defaults, dict):
                    defaults = {}
                val_var = defaults.get('val_var') or tbl_def.get('val_var')
                idx_var = defaults.get('index_var') or tbl_def.get('index_var')
                if val_var or idx_var:
                    result[(db_name, tbl_name)] = {'val_var': val_var, 'index_var': idx_var}
        return result
    except Exception:
        return {}


def _format_source_selector(source) -> str:
    """Summarize the identifying selector used for a raw concept source."""
    selector_parts = []
    if getattr(source, 'sub_var', None):
        if getattr(source, 'ids', None):
            selector_parts.append(f"{source.sub_var}={_format_definition_list(source.ids, limit=8)}")
        elif getattr(source, 'regex', None):
            selector_parts.append(f"{source.sub_var}~/{source.regex}/")
        else:
            selector_parts.append(str(source.sub_var))
    elif getattr(source, 'regex', None):
        selector_parts.append(f"regex=/{source.regex}/")
    if getattr(source, 'class_name', None):
        selector_parts.append(f"class={source.class_name}")
    if getattr(source, 'params', None):
        param_keys = sorted(source.params.keys())
        if param_keys:
            selector_parts.append(f"params={_format_definition_list(param_keys, limit=6)}")
    return " | ".join(selector_parts) if selector_parts else "—"


def _collect_recursive_concept_sources(concept_name: str, database: str, concept_dict: dict, visited=None) -> list[tuple[str, object]]:
    """Collect raw source entries for a concept by recursively traversing sub-concepts."""
    if visited is None:
        visited = set()
    if concept_name in visited:
        return []
    visited.add(concept_name)

    concept_def = concept_dict.get(concept_name)
    if not concept_def:
        return []

    collected = []
    for source in concept_def.sources.get(database, []):
        collected.append((concept_name, source))
    for sub_concept in getattr(concept_def, 'sub_concepts', []) or []:
        collected.extend(_collect_recursive_concept_sources(sub_concept, database, concept_dict, visited))
    return collected


def _get_feature_definition_rows(selected_concepts: list[str], database: str, lang: str, app_context: dict[str, Any] | None = None) -> list[dict]:
    """Build a transparent per-feature definition table for the current database."""
    if app_context is not None:
        _install_app_context(app_context)

    concept_dict = _get_quality_concept_dictionary()
    table_defaults = _get_table_defaults()
    rows = []

    for concept_name in sorted(set(selected_concepts)):
        eng_name, chn_name, dict_unit = CONCEPT_DICTIONARY.get(concept_name, (concept_name, concept_name, ''))
        description_en, description_zh = CONCEPT_DESCRIPTIONS.get(concept_name, ('', ''))
        display_name = eng_name if lang == 'en' else chn_name
        description = description_en if lang == 'en' else description_zh

        concept_def = concept_dict.get(concept_name)
        unit = dict_unit
        if concept_def and getattr(concept_def, 'units', None):
            unit = _format_definition_list(concept_def.units, limit=4)
        unit = unit or "—"

        base_row = {
            'Feature': concept_name,
            'Name': display_name,
            'Unit': unit,
            'Type': "Direct",
            'Table(s)': "—",
            'Selector / ID': "—",
            'Columns': "—",
            'Logic': description or "—",
        }

        if concept_def:
            direct_sources = concept_def.sources.get(database, [])
            if direct_sources:
                for source in direct_sources:
                    logic_parts = []
                    if getattr(source, 'callback', None):
                        logic_parts.append(f"callback: {source.callback}")
                    if getattr(concept_def, 'callback', None):
                        logic_parts.append(f"callback: {concept_def.callback}")
                    if getattr(concept_def, 'sub_concepts', None):
                        logic_parts.append(f"derived: {_format_definition_list(concept_def.sub_concepts, limit=6)}")
                    if description:
                        logic_parts.append(description)

                    # Resolve value_var / index_var from table defaults
                    tbl_name = getattr(source, 'table', None) or ''
                    tbl_def = table_defaults.get((database, tbl_name), {})
                    val_var = getattr(source, 'value_var', None) or tbl_def.get('val_var')
                    idx_var = getattr(source, 'index_var', None) or tbl_def.get('index_var')
                    unit_var = getattr(source, 'unit_var', None)
                    dur_var = getattr(source, 'dur_var', None)

                    col_parts = []
                    if val_var:
                        col_parts.append(f"value={val_var}")
                    if unit_var:
                        col_parts.append(f"unit={unit_var}")
                    if idx_var:
                        col_parts.append(f"time={idx_var}")
                    if dur_var:
                        col_parts.append(f"dur={dur_var}")

                    row = dict(base_row)
                    has_callback = getattr(source, 'callback', None) or getattr(concept_def, 'callback', None)
                    src_class = getattr(source, 'class_name', None) or ''
                    if not tbl_name and src_class == 'fun_itm':
                        row['Type'] = "Function"
                        row['Table(s)'] = "computed"
                    elif not tbl_name and src_class == 'rec_cncpt':
                        row['Type'] = "Derived"
                        row['Table(s)'] = "recursive"
                    else:
                        row['Type'] = "Callback" if has_callback else "Direct"
                        row['Table(s)'] = tbl_name or "—"
                    row['Selector / ID'] = _format_source_selector(source)
                    row['Columns'] = " | ".join(col_parts) if col_parts else "—"
                    row['Logic'] = " ; ".join(logic_parts) if logic_parts else "—"
                    rows.append(row)
                continue

            recursive_sources = _collect_recursive_concept_sources(concept_name, database, concept_dict)
            if recursive_sources:
                table_names = sorted({src.table for _, src in recursive_sources if getattr(src, 'table', None)})
                selectors = []
                for leaf_concept, source in recursive_sources:
                    selector_summary = _format_source_selector(source)
                    if selector_summary != "—":
                        selectors.append(f"{leaf_concept}: {selector_summary}")
                    else:
                        selectors.append(f"{leaf_concept}")

                logic_parts = []
                if getattr(concept_def, 'callback', None):
                    logic_parts.append(f"callback: {concept_def.callback}")
                if getattr(concept_def, 'sub_concepts', None):
                    logic_parts.append(f"derived: {_format_definition_list(concept_def.sub_concepts, limit=8)}")
                if description:
                    logic_parts.append(description)

                row = dict(base_row)
                row['Type'] = "Derived"
                row['Table(s)'] = _format_definition_list(table_names, limit=8)
                row['Selector / ID'] = _format_definition_list(selectors, limit=6)
                row['Logic'] = " ; ".join(logic_parts) if logic_parts else "—"
                rows.append(row)
                continue

            # Concept exists in dict but no source for this database
            if concept_name in SPECIAL_CONCEPTS:
                module_name, func_name, output_cols = SPECIAL_CONCEPTS[concept_name]
                row = dict(base_row)
                row['Type'] = "Special"
                row['Table(s)'] = "loader"
                row['Selector / ID'] = f"{module_name}.{func_name}"
                logic_parts = [f"output: {_format_definition_list(output_cols, limit=4)}"]
                if description:
                    logic_parts.append(description)
                row['Logic'] = " ; ".join(logic_parts)
                rows.append(row)
                continue

            row = dict(base_row)
            no_src_label = f"No source for {database.upper()}" if lang == 'en' else f"{database.upper()} 无数据源"
            row['Type'] = no_src_label
            rows.append(row)
            continue

        # concept_def is None -- check SPECIAL_CONCEPTS
        if concept_name in SPECIAL_CONCEPTS:
            module_name, func_name, output_cols = SPECIAL_CONCEPTS[concept_name]
            row = dict(base_row)
            row['Type'] = "Special"
            row['Table(s)'] = "loader"
            row['Selector / ID'] = f"{module_name}.{func_name}"
            logic_parts = [f"output: {_format_definition_list(output_cols, limit=4)}"]
            if description:
                logic_parts.append(description)
            row['Logic'] = " ; ".join(logic_parts)
            rows.append(row)
            continue

        row = dict(base_row)
        row['Type'] = "Unknown"
        rows.append(row)

    return rows


def _render_feature_definition_panel(lang: str, app_context: dict[str, Any] | None = None) -> None:
    """Render a transparent feature definition panel for the selected database and features."""
    if app_context is not None:
        _install_app_context(app_context)

    if not st.session_state.get('step3_confirmed', False):
        return

    selected_concepts = list(st.session_state.get('selected_concepts', []) or [])
    if not selected_concepts:
        return

    database = str(st.session_state.get('database', '') or '')
    if not database:
        return

    rows = _get_feature_definition_rows(selected_concepts, database, lang)
    if not rows:
        return

    title = "🧬 Feature Definition Transparency" if lang == 'en' else "🧬 特征定义透明化"
    caption = (
        f"Current database: {database.upper()}. This table shows how each selected feature is defined in EasyICU, including raw tables, selectors/item IDs, units, and derived logic."
        if lang == 'en' else
        f"当前数据库：{database.upper()}。该表展示 EasyICU 如何定义你已选特征，包括原始表、选择器/item ID、单位以及派生逻辑。"
    )
    download_label = "⬇️ Download Definition CSV" if lang == 'en' else "⬇️ 下载定义表 CSV"
    n_features = len(set(selected_concepts))
    n_rows = len(rows)
    summary = (
        f"Showing **{n_features}** selected features and **{n_rows}** database-specific definition rows."
        if lang == 'en' else
        f"当前展示 **{n_features}** 个已选特征，对应 **{n_rows}** 条数据库定义记录。"
    )

    with st.expander(title, expanded=False):
        st.caption(caption)
        st.info(summary)
        definition_df = pd.DataFrame(rows)
        st.download_button(
            download_label,
            data=definition_df.to_csv(index=False, encoding='utf-8-sig'),
            file_name=f"easyicu_feature_definition_{database.lower()}.csv",
            mime="text/csv",
            key="download_feature_definition_csv",
        )
        st.dataframe(
            definition_df,
            use_container_width=True,
            hide_index=True,
            height=min(640, 120 + 36 * max(len(definition_df), 1)),
        )


def render_home_data_dictionary(lang, app_context: dict[str, Any] | None = None):
    """在首页渲染完整的数据字典。"""
    if app_context is not None:
        _install_app_context(app_context)

    dict_title = "📖 Complete Data Dictionary" if lang == 'en' else "📖 完整数据字典"
    st.caption(dict_title)

    # Streamlit forbids nested expanders in recent versions, so the section
    # heading remains flat and only the per-category groups use expanders.
    search_placeholder = "Search by code, name or description... (e.g. hr, heart rate, lactate)" if lang == 'en' else "按代码、名称或描述搜索... (如 hr、heart rate、心率)"
    search_query = st.text_input(
        "🔍 Search" if lang == 'en' else "🔍 搜索",
        placeholder=search_placeholder,
        key="dict_search_input",
    )

    concept_groups = get_concept_groups()

    if search_query and search_query.strip():
        query = search_query.strip().lower()
        matched_rows = []
        for group_name, concepts in concept_groups.items():
            for concept in concepts:
                if concept in CONCEPT_DICTIONARY:
                    eng_name, chn_name, unit = CONCEPT_DICTIONARY[concept]
                    eng_desc, chn_desc = CONCEPT_DESCRIPTIONS.get(concept, ('', ''))
                    # 匹配 code、英文名、中文名、描述
                    if lang == 'en':
                        searchable = f"{concept} {eng_name} {eng_desc}".lower()
                    else:
                        searchable = f"{concept} {eng_name} {chn_name} {eng_desc} {chn_desc}".lower()
                    if query in searchable:
                        if lang == 'en':
                            matched_rows.append({
                                'Code': concept,
                                'Full Name': eng_name,
                                'Category': group_name,
                                'Description': eng_desc if eng_desc else eng_name,
                                'Unit': unit if unit else '-'
                            })
                        else:
                            matched_rows.append({
                                '代码': concept,
                                '全称': eng_name,
                                '类别': group_name,
                                '说明': chn_desc if chn_desc else chn_name,
                                '单位': unit if unit else '-'
                            })

        if matched_rows:
            n = len(matched_rows)
            result_text = f"Found **{n}** matching feature(s)" if lang == 'en' else f"找到 **{n}** 个匹配特征"
            st.success(result_text)
            _st_dataframe_compat(
                st,
                pd.DataFrame(matched_rows),
                width="stretch",
                hide_index=True,
                height=min(300, 50 + 35 * n),
            )
        else:
            no_result = "No matching features found." if lang == 'en' else "未找到匹配的特征。"
            st.warning(no_result)
    else:
        categories_title = "📂 Categories" if lang == 'en' else "📂 类别"
        st.markdown(f"#### {categories_title}")

        for group_name in concept_groups.keys():
            feat_text = "features" if lang == 'en' else "个特征"
            with st.expander(f"{group_name} ({len(concept_groups[group_name])} {feat_text})"):
                _render_home_dict_table(concept_groups[group_name], lang)


def _render_home_dict_table(concepts, lang, app_context: dict[str, Any] | None = None):
    """为首页数据字典渲染表格。"""
    if app_context is not None:
        _install_app_context(app_context)

    rows = []
    for concept in concepts:
        if concept in CONCEPT_DICTIONARY:
            eng_name, chn_name, unit = CONCEPT_DICTIONARY[concept]
            # 获取详细描述
            if concept in CONCEPT_DESCRIPTIONS:
                eng_desc, chn_desc = CONCEPT_DESCRIPTIONS[concept]
            else:
                eng_desc, chn_desc = eng_name, chn_name  # 用名称作为默认描述

            if lang == 'en':
                rows.append({
                    'Code': concept,
                    'Full Name': eng_name,
                    'Description': eng_desc,
                    'Unit': unit if unit else '-'
                })
            else:
                rows.append({
                    '代码': concept,
                    '全称': eng_name,
                    '说明': chn_desc,
                    '单位': unit if unit else '-'
                })

    if rows:
        df = pd.DataFrame(rows)
        _st_dataframe_compat(
            st,
            df,
            width="stretch",
            hide_index=True,
            height=300,
        )
