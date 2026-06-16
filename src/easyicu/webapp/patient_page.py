"""Patient detail page rendering for the EasyICU Streamlit app."""

from __future__ import annotations

import html
from typing import Any

from easyicu.webapp.compat import _dataframe_compat as _st_dataframe_compat


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to the extracted patient page."""
    protected = {"render_patient_page", "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def _normalize_patient_view_id(patient_ids: list[Any], current_patient: Any) -> Any | None:
    """Return a valid patient id for the current workspace."""
    if not patient_ids:
        return None
    if current_patient in patient_ids:
        return current_patient
    return patient_ids[0]


def _patient_navigation_target(
    patient_ids: list[Any],
    current_patient: Any,
    action: str,
    *,
    random_choice: Any | None = None,
) -> Any | None:
    """Resolve patient navigation buttons without depending on Streamlit state."""
    current = _normalize_patient_view_id(patient_ids, current_patient)
    if current is None:
        return None

    current_idx = patient_ids.index(current)
    if action == "first":
        return patient_ids[0]
    if action == "previous":
        return patient_ids[max(0, current_idx - 1)]
    if action == "next":
        return patient_ids[min(len(patient_ids) - 1, current_idx + 1)]
    if action == "last":
        return patient_ids[-1]
    if action == "random":
        return random_choice if random_choice in patient_ids else current
    raise ValueError(f"Unknown patient navigation action: {action}")


def _patient_html(value: Any) -> str:
    return html.escape(str(value))


def render_patient_page(app_context: dict[str, Any] | None = None):
    """渲染患者视图页面。"""
    if app_context is not None:
        _install_app_context(app_context)

    lang = st.session_state.get('language', 'en')
    screenshot_mode = _is_screenshot_mode()
    loaded_concepts_map = st.session_state.loaded_concepts
    time_candidates = [
        'time', 'charttime', 'starttime', 'endtime', 'datetime',
        'timestamp', 'Offset', 'measuredat_minutes', 'measuredat',
    ]

    def _patient_concept_frame(concept_name, patient_id, id_col_name):
        frame = loaded_concepts_map.get(concept_name)
        if not isinstance(frame, pd.DataFrame) or id_col_name not in frame.columns:
            return None
        patient_frame = frame[frame[id_col_name] == patient_id].copy()
        if patient_frame.empty:
            return None
        return patient_frame

    def _latest_patient_value(concept_name, patient_id, id_col_name):
        patient_frame = _patient_concept_frame(concept_name, patient_id, id_col_name)
        if patient_frame is None:
            return None
        value_col = _choose_concept_value_column(concept_name, patient_frame)
        if value_col is None:
            excluded_cols = {id_col_name, 'time', 'charttime', 'starttime', 'endtime', 'datetime', 'timestamp', 'Offset', 'measuredat_minutes', 'measuredat'}
            candidates = [c for c in patient_frame.columns if c not in excluded_cols]
            if not candidates:
                return None
            value_col = candidates[-1]
        series = patient_frame[value_col].dropna()
        if series.empty:
            return None
        return series.iloc[-1]

    def _render_patient_micro_heading(kicker: str, title: str, subtitle: str | None = None) -> None:
        subtitle_html = f'<p>{_patient_html(subtitle)}</p>' if subtitle else ""
        st.markdown(
            '<div class="eu-patient-mini-heading">'
            f'<span>{_patient_html(kicker)}</span>'
            f'<b>{_patient_html(title)}</b>'
            f'{subtitle_html}'
            '</div>',
            unsafe_allow_html=True,
        )

    def _patient_signal_grid(cards: list[tuple[str, str, str, str]]) -> None:
        card_html = "".join(
            (
                f'<div class="eu-patient-signal-card {tone}">'
                f'<span>{_patient_html(label)}</span>'
                f'<b>{_patient_html(value)}</b>'
                f'<em>{_patient_html(note)}</em>'
                '</div>'
            )
            for label, value, note, tone in cards
        )
        st.markdown(
            f'<div class="eu-patient-signal-grid">{card_html}</div>',
            unsafe_allow_html=True,
        )

    def _patient_key_fragment(value: Any) -> str:
        fragment = "".join(ch if ch.isalnum() else "_" for ch in str(value).lower()).strip("_")
        return fragment or "item"

    def _patient_time_column(frame) -> str | None:
        for candidate in time_candidates:
            if candidate in frame.columns:
                return candidate
        return None

    def _patient_value_column(concept_name: str, frame, id_col_name: str) -> str | None:
        if concept_name in frame.columns:
            return concept_name
        value_col = _choose_concept_value_column(concept_name, frame)
        if value_col is not None and value_col in frame.columns:
            return value_col
        excluded_cols = {id_col_name, *time_candidates}
        candidates = [column for column in frame.columns if column not in excluded_cols]
        return candidates[-1] if candidates else None

    def _format_patient_category_value(raw_value: Any, decimals: int) -> str:
        try:
            return safe_format_number(float(raw_value), decimals)
        except Exception:
            return str(raw_value)

    def _patient_delta_note(series, decimals: int, lang: str) -> str:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) > 1:
            delta = float(numeric.iloc[-1] - numeric.iloc[0])
            if abs(delta) < 1e-9:
                return "stable" if lang == "en" else "稳定"
            sign = "+" if delta > 0 else ""
            return f"Δ {sign}{safe_format_number(delta, decimals)}"
        return "latest" if lang == "en" else "最新"

    def _patient_category_tone(concept_name: str, raw_value: Any, boolean_active: bool | None = None) -> str:
        if boolean_active is not None:
            if concept_name in {"sep3_sofa1", "sep3_sofa2", "susp_inf", "infection_icd"}:
                return "danger" if boolean_active else "ok"
            if concept_name in {"vent_ind", "mech_vent", "rrt", "vaso_ind"}:
                return "warn" if boolean_active else "ok"
            return "accent" if boolean_active else "neutral"
        try:
            numeric = float(raw_value)
        except Exception:
            return "neutral"
        if concept_name in {"sofa", "sofa2", "qsofa", "sirs", "mews", "news"}:
            return "ok" if numeric < 6 else "warn" if numeric < 10 else "danger"
        if concept_name == "gcs":
            return "ok" if numeric >= 13 else "warn" if numeric >= 9 else "danger"
        return "neutral"

    def _patient_category_notice(tone: str, title: str, body: str) -> None:
        st.markdown(
            f'<div class="eu-patient-category-notice {html.escape(tone)}">'
            f'<b>{_patient_html(title)}</b>'
            f'<p>{_patient_html(body)}</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    def _render_patient_notice(tone: str, kicker: str, title: str, body: str = "", meta: str = "") -> None:
        body_html = f'<p>{_patient_html(body)}</p>' if body else ""
        meta_html = f'<em>{_patient_html(meta)}</em>' if meta else ""
        st.markdown(
            f'<div class="eu-patient-notice {html.escape(tone)}">'
            f'<span>{_patient_html(kicker)}</span>'
            f'<b>{_patient_html(title)}</b>'
            f'{body_html}'
            f'{meta_html}'
            '</div>',
            unsafe_allow_html=True,
        )

    def _collect_patient_category_items(
        concepts: list[str],
        patient_id: Any,
        id_col_name: str,
        *,
        decimals: int,
        boolean_modes: dict[str, str] | None = None,
        include_chart: bool = True,
    ) -> tuple[list[dict[str, str]], list[tuple[str, Any, str, str]]]:
        boolean_modes = boolean_modes or {}
        cards: list[dict[str, str]] = []
        series_items: list[tuple[str, Any, str, str]] = []

        for concept_name in concepts:
            frame = loaded_concepts_map.get(concept_name)
            if not isinstance(frame, pd.DataFrame):
                continue
            patient_frame = frame[frame[id_col_name] == patient_id].copy() if id_col_name in frame.columns else frame.copy()
            if patient_frame.empty:
                continue
            value_col = _patient_value_column(concept_name, patient_frame, id_col_name)
            if value_col is None or value_col not in patient_frame.columns:
                continue
            series = patient_frame[value_col].dropna()
            if series.empty:
                continue

            bool_mode = boolean_modes.get(concept_name)
            if bool_mode:
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                bool_raw = numeric.max() if bool_mode == "max" and not numeric.empty else numeric.iloc[-1] if not numeric.empty else series.iloc[-1]
                try:
                    is_active = float(bool_raw) == 1.0
                except Exception:
                    is_active = str(bool_raw).strip().lower() in {"1", "true", "yes", "y"}
                value = "Yes" if is_active and lang == "en" else "No" if lang == "en" else "是" if is_active else "否"
                note = "ever observed" if bool_mode == "max" and lang == "en" else "latest flag" if lang == "en" else "曾出现" if bool_mode == "max" else "最新标记"
                cards.append({
                    "label": concept_name.upper(),
                    "value": value,
                    "note": note,
                    "tone": _patient_category_tone(concept_name, bool_raw, is_active),
                })
                continue

            latest_value = series.iloc[-1]
            cards.append({
                "label": concept_name.upper(),
                "value": _format_patient_category_value(latest_value, decimals),
                "note": _patient_delta_note(series, decimals, lang),
                "tone": _patient_category_tone(concept_name, latest_value),
            })

            if include_chart:
                time_col = _patient_time_column(patient_frame)
                if time_col is None:
                    continue
                plot_frame = patient_frame[[time_col, value_col]].copy()
                plot_frame[value_col] = pd.to_numeric(plot_frame[value_col], errors="coerce")
                plot_frame = plot_frame.dropna(subset=[time_col, value_col])
                if plot_frame.empty:
                    continue
                try:
                    plot_frame = plot_frame.sort_values(time_col)
                except Exception:
                    pass
                series_items.append((concept_name.upper(), plot_frame.tail(48), time_col, value_col))

        return cards, series_items

    def _render_patient_category_grid(cards: list[dict[str, str]]) -> None:
        card_html = "".join(
            (
                f'<div class="eu-patient-category-card {html.escape(card["tone"])}">'
                f'<span>{_patient_html(card["label"])}</span>'
                f'<b>{_patient_html(card["value"])}</b>'
                f'<em>{_patient_html(card["note"])}</em>'
                '</div>'
            )
            for card in cards
        )
        st.markdown(
            f'<div class="eu-patient-category-grid">{card_html}</div>',
            unsafe_allow_html=True,
        )

    def _render_patient_category_chart(section_key: str, title: str, series_items: list[tuple[str, Any, str, str]]) -> None:
        if not series_items:
            return
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        visible_items = series_items[:6]
        n_cols = min(3, len(visible_items))
        n_rows = (len(visible_items) + n_cols - 1) // n_cols
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[item[0] for item in visible_items],
            vertical_spacing=0.2 if n_rows > 1 else 0.12,
            horizontal_spacing=0.08,
        )
        palette = ["#0f766e", "#334155", "#9f3a57", "#a16207", "#0e7490", "#475569"]
        for idx, (label, plot_frame, time_col, value_col) in enumerate(visible_items):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            fig.add_trace(
                go.Scatter(
                    x=plot_frame[time_col],
                    y=plot_frame[value_col],
                    mode="lines+markers",
                    name=label,
                    line=dict(color=palette[idx % len(palette)], width=2.2, shape="spline", smoothing=0.35),
                    marker=dict(size=4.5, color=palette[idx % len(palette)]),
                    hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )
        fig.update_layout(
            height=168 if n_rows == 1 else 294 if n_rows == 2 else 420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#fbfaf7",
            showlegend=False,
            margin=dict(l=34, r=16, t=34, b=22),
            font=dict(size=11, color="#1f2937"),
        )
        fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(size=9, color="#64748b"))
        fig.update_yaxes(showgrid=True, gridcolor="#ece7df", zeroline=False, tickfont=dict(size=10, color="#64748b"))
        st.markdown(
            f'<div class="eu-patient-category-chart-title">{_patient_html(title)}</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"patient_category_{_patient_key_fragment(section_key)}_sparkline",
            config=_get_plotly_chart_config(),
        )

    def _render_patient_category_section(
        section_key: str,
        title: str,
        subtitle: str,
        concepts: list[str],
        *,
        decimals: int = 1,
        boolean_modes: dict[str, str] | None = None,
        include_chart: bool = True,
        no_data_body: str | None = None,
    ) -> None:
        _render_patient_micro_heading("Category" if lang == "en" else "分类", title, subtitle)
        cards, series_items = _collect_patient_category_items(
            concepts,
            patient_id,
            id_col,
            decimals=decimals,
            boolean_modes=boolean_modes,
            include_chart=include_chart,
        )
        if not cards:
            _patient_category_notice(
                "pending",
                "No signals in this category" if lang == "en" else "该分类暂无信号",
                no_data_body or ("The current workspace has no loaded rows for these concepts." if lang == "en" else "当前工作台未加载这些概念的患者行。"),
            )
            return
        _render_patient_category_grid(cards)
        if include_chart:
            _render_patient_category_chart(section_key, f"{title} · compact trend", series_items)

    _pat_title = "Patient Overview" if lang == 'en' else "患者综合视图"
    _pat_sub = "Multi-dimensional patient dashboard" if lang == 'en' else "多维度患者仪表盘"
    st.markdown(f'''
    <div class="eu-page-marker eu-patient-page-marker"></div>
    <div class="eu-subhead">
        <div class="t">{_pat_title}</div>
        <div class="s">{_pat_sub}</div>
    </div>
    ''', unsafe_allow_html=True)

    if len(st.session_state.loaded_concepts) == 0:
        _msg = "Load data to view patient dashboards." if lang == 'en' else "请先加载数据以查看患者视图。"
        st.markdown(f'''
        <div class="eu-patient-empty-state">
            <div class="eu-patient-empty-icon">Data</div>
            <b>{_patient_html(_msg)}</b>
            <p>{_patient_html("Use Data Extraction or load an exported module folder to open the review workspace." if lang == 'en' else "通过数据提取或已导出的模块目录打开审阅工作台。")}</p>
        </div>
        ''', unsafe_allow_html=True)
        return

    if not st.session_state.patient_ids:
        warn_msg = "No patient data found" if lang == 'en' else "未找到患者数据"
        _render_patient_notice(
            "warning",
            "Patient index" if lang == "en" else "患者索引",
            warn_msg,
            "The loaded tables did not expose any ICU stay identifiers for patient-level review."
            if lang == 'en' else
            "当前已加载表未提供可用于患者级审阅的 ICU stay 标识。",
        )
        return

    # ============ Patient Summary Header (审稿式 Case Review) ============
    def _render_patient_summary_card(pid):
        """渲染患者摘要卡片"""
        loaded = st.session_state.loaded_concepts
        id_col = st.session_state.get('id_col', 'stay_id')
        _age = _los = _mort = _sex = "—"
        _supports = []

        for cname, df in loaded.items():
            if df is None or not hasattr(df, 'columns') or id_col not in df.columns:
                continue
            pdf = df[df[id_col] == pid]
            if pdf.empty or cname not in pdf.columns:
                continue
            v = pdf[cname].dropna()
            if len(v) == 0:
                continue
            if cname == 'age':
                _age = f"{float(v.iloc[0]):.0f}"
            elif cname == 'sex':
                _sex = str(v.iloc[0])
            elif cname == 'los_icu':
                _los = f"{float(v.iloc[0]):.1f}d"
            elif cname == 'death':
                _mort = "Yes" if float(v.iloc[0]) == 1 else "No"
            elif cname == 'mech_vent' and float(v.iloc[0]) > 0:
                _supports.append("MV")
            elif cname == 'rrt' and float(v.iloc[0]) > 0:
                _supports.append("RRT")
            elif cname in ('norepi_rate', 'epi_rate', 'dopa_rate', 'dobu_rate') and float(v.max()) > 0:
                _supports.append("Vasopressors")

        _supports = list(set(_supports))
        _supports_str = ", ".join(_supports) if _supports else "—"
        _demo_lbl = get_text('demographics_header')
        _los_lbl = get_text('icu_los_label')
        _mort_lbl = get_text('mortality_label')
        _sup_lbl = get_text('key_supports')

        st.markdown(f"""
        <div class="eu-patient-summary-card">
            <div class="eu-patient-summary-head">
                <div>
                    <span>{_patient_html("Case review" if lang == 'en' else "病例复核")}</span>
                    <b>{_patient_html(get_text('patient_summary'))} · Patient {_patient_html(pid)}</b>
                </div>
                <em>{_patient_html("local workspace" if lang == 'en' else "本地工作区")}</em>
            </div>
            <div class="eu-patient-summary-grid">
                <div><span>{_patient_html(_demo_lbl)}</span><b>{_patient_html(_sex)}, {_patient_html(_age)}y</b></div>
                <div><span>{_patient_html(_los_lbl)}</span><b>{_patient_html(_los)}</b></div>
                <div><span>{_patient_html(_mort_lbl)}</span><b>{_patient_html(_mort)}</b></div>
                <div><span>{_patient_html(_sup_lbl)}</span><b>{_patient_html(_supports_str)}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 患者选择面板
    if not screenshot_mode:
        select_title = "Patient Selection" if lang == 'en' else "患者选择"
        select_hint = "Move between ICU stays, then inspect the dashboard, category tables, or raw rows." if lang == 'en' else "切换 ICU stay 后查看仪表盘、分类表或原始数据行。"
        st.markdown(f'''
        <div class="eu-patient-control-heading">
            <span>{_patient_html("Case navigator" if lang == 'en' else "病例导航")}</span>
            <b>{_patient_html(select_title)}</b>
            <p>{_patient_html(select_hint)}</p>
        </div>
        ''', unsafe_allow_html=True)

    # 快速导航按钮
    first_btn = "First" if lang == 'en' else "首位"
    prev_btn = "Previous" if lang == 'en' else "上一位"
    next_btn = "Next" if lang == 'en' else "下一位"
    last_btn = "Last" if lang == 'en' else "末位"
    rand_btn = "Random" if lang == 'en' else "随机"
    first_help = "Jump to first patient" if lang == 'en' else "跳转到第一位患者"
    prev_help = "Previous patient" if lang == 'en' else "上一位患者"
    next_help = "Next patient" if lang == 'en' else "下一位患者"
    last_help = "Jump to last patient" if lang == 'en' else "跳转到最后一位患者"
    rand_help = "Random select a patient" if lang == 'en' else "随机选择一位患者"

    current_patient_id = _normalize_patient_view_id(
        st.session_state.patient_ids,
        st.session_state.get('patient_view_id', st.session_state.patient_ids[0]),
    )
    if current_patient_id is not None:
        st.session_state.patient_view_id = current_patient_id
    current_idx = st.session_state.patient_ids.index(current_patient_id)
    if screenshot_mode:
        focus_msg = (
            f"Figure preset: focusing the dashboard on patient {current_idx + 1}/{len(st.session_state.patient_ids)}. Use the selector below to switch cases."
            if lang == 'en'
            else f"截图预设：当前聚焦第 {current_idx + 1}/{len(st.session_state.patient_ids)} 位患者。可通过下方选择器切换病例。"
        )
        st.markdown(f'<div class="compact-inline-notice info">{focus_msg}</div>', unsafe_allow_html=True)
    else:
        nav_cols = st.columns(6)
        with nav_cols[0]:
            if st.button(first_btn, use_container_width=True, help=first_help):
                st.session_state.patient_view_id = _patient_navigation_target(
                    st.session_state.patient_ids,
                    st.session_state.get('patient_view_id'),
                    "first",
                )
                st.rerun()
        with nav_cols[1]:
            if st.button(prev_btn, use_container_width=True, help=prev_help):
                if current_idx > 0:
                    st.session_state.patient_view_id = _patient_navigation_target(
                        st.session_state.patient_ids,
                        st.session_state.get('patient_view_id'),
                        "previous",
                    )
                    st.rerun()
        with nav_cols[2]:
            if st.button(next_btn, use_container_width=True, help=next_help):
                if current_idx < len(st.session_state.patient_ids) - 1:
                    st.session_state.patient_view_id = _patient_navigation_target(
                        st.session_state.patient_ids,
                        st.session_state.get('patient_view_id'),
                        "next",
                    )
                    st.rerun()
        with nav_cols[3]:
            if st.button(last_btn, use_container_width=True, help=last_help):
                st.session_state.patient_view_id = _patient_navigation_target(
                    st.session_state.patient_ids,
                    st.session_state.get('patient_view_id'),
                    "last",
                )
                st.rerun()
        with nav_cols[4]:
            if st.button(rand_btn, use_container_width=True, help=rand_help):
                import random

                st.session_state.patient_view_id = _patient_navigation_target(
                    st.session_state.patient_ids,
                    st.session_state.get('patient_view_id'),
                    "random",
                    random_choice=random.choice(st.session_state.patient_ids),
                )
                st.rerun()
        with nav_cols[5]:
            st.markdown(
                f"<div class='eu-patient-nav-count'><span>{_patient_html('Case' if lang == 'en' else '病例')}</span><b>{current_idx + 1}/{len(st.session_state.patient_ids)}</b></div>",
                unsafe_allow_html=True,
            )

    # ============ Render Patient Summary Card ============
    _current_pid = st.session_state.get('patient_view_id', st.session_state.patient_ids[0] if st.session_state.patient_ids else None)
    if _current_pid is not None:
        try:
            _render_patient_summary_card(_current_pid)
        except Exception:
            pass

    # 判断视图模式
    dashboard_mode = "Dashboard" if lang == 'en' else "综合仪表盘"
    category_mode = "Category View" if lang == 'en' else "分类视图"
    table_mode = "Data Table" if lang == 'en' else "数据表格"

    if screenshot_mode:
        patient_id = _current_pid
        view_mode = dashboard_mode
    else:
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            pat_id_label = "Patient ID" if lang == 'en' else "患者 ID"
            patient_id = _patient_selector(
                patient_ids=st.session_state.patient_ids,
                state_key="patient_view_id",
                label=pat_id_label,
                lang=lang,
                max_display=200,
                default_patient=st.session_state.get('patient_view_id', st.session_state.patient_ids[0]),
            )

        with col2:
            view_label = "View Mode" if lang == 'en' else "显示模式"
            view_options = ["Dashboard", "Category View", "Data Table"] if lang == 'en' else ["综合仪表盘", "分类视图", "数据表格"]
            view_mode = st.selectbox(
                view_label,
                options=view_options,
                key="patient_view_mode"
            )

        with col3:
            # 数据概览 - 显示更详细的可用数据信息
            id_col = st.session_state.id_col
            available_concepts = [k for k, v in st.session_state.loaded_concepts.items()
                                 if isinstance(v, pd.DataFrame) and id_col in v.columns
                                 and patient_id in v[id_col].values]
            n_concepts = len(available_concepts)

            # 统计各类别数据
            vitals_list = ['hr', 'map', 'sbp', 'dbp', 'resp', 'temp', 'spo2']
            labs_list = ['bili', 'crea', 'lac', 'plt', 'wbc', 'hgb', 'inr_pt', 'ptt']
            scores_list = ['sofa', 'sofa2', 'qsofa', 'sirs', 'gcs', 'sep3_sofa1', 'sep3_sofa2']

            n_vitals = len([c for c in available_concepts if c in vitals_list])
            n_labs = len([c for c in available_concepts if c in labs_list])
            n_scores = len([c for c in available_concepts if c in scores_list])

            data_label = "Available Data" if lang == 'en' else "可用数据"
            st.markdown(f'''
            <div class="eu-patient-availability-card metric-card">
                <span>{_patient_html(data_label)}</span>
                <div>
                    <b>{n_concepts}</b><em>{_patient_html("total" if lang == 'en' else "总计")}</em>
                    <b>{n_vitals}</b><em>{_patient_html("vitals" if lang == 'en' else "生命体征")}</em>
                    <b>{n_labs}</b><em>{_patient_html("labs" if lang == 'en' else "实验室")}</em>
                    <b>{n_scores}</b><em>{_patient_html("scores" if lang == 'en' else "评分")}</em>
                </div>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if patient_id:
        st.session_state.selected_patient = patient_id
        id_col = st.session_state.id_col

        if view_mode == dashboard_mode:
            # 自定义综合仪表盘
            dash_title = "Dashboard" if lang == 'en' else "综合仪表盘"
            dash_sub = "Trend tiles, score comparison, and endpoint flags for the selected ICU stay." if lang == 'en' else "当前 ICU stay 的趋势卡片、评分对比与终点标记。"
            st.markdown(
                '<div class="eu-patient-section-heading">'
                f'<span>{_patient_html("Review surface" if lang == "en" else "审阅界面")}</span>'
                f'<b>{_patient_html(dash_title)}</b>'
                f'<p>{_patient_html(dash_sub)}</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            if screenshot_mode:
                dash_focus_note = (
                    "Figure preset: emphasizing SOFA comparison and compact case summary."
                    if lang == 'en'
                    else "截图预设：突出 SOFA 对比和紧凑病例摘要。"
                )
                st.markdown(f'<div class="compact-inline-notice info">{dash_focus_note}</div>', unsafe_allow_html=True)

            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                def _patient_trend_frame(concept_name):
                    frame = _patient_concept_frame(concept_name, patient_id, id_col)
                    if frame is None:
                        return None
                    time_col = None
                    for candidate in time_candidates:
                        if candidate in frame.columns:
                            time_col = candidate
                            break
                    if time_col is None:
                        return None
                    value_col = concept_name if concept_name in frame.columns else _choose_concept_value_column(concept_name, frame)
                    if value_col is None or value_col not in frame.columns:
                        return None
                    plot_frame = frame[[time_col, value_col]].copy()
                    plot_frame[value_col] = pd.to_numeric(plot_frame[value_col], errors='coerce')
                    plot_frame = plot_frame.dropna(subset=[time_col, value_col])
                    if plot_frame.empty:
                        return None
                    try:
                        plot_frame = plot_frame.sort_values(time_col)
                    except Exception:
                        pass
                    return plot_frame.tail(48), time_col, value_col

                def _render_compact_trend_panel(title, concept_specs):
                    trend_items = []
                    for concept_name, display_name, unit in concept_specs:
                        trend_payload = _patient_trend_frame(concept_name)
                        if trend_payload is None:
                            continue
                        trend_items.append((concept_name, display_name, unit, *trend_payload))

                    if not trend_items:
                        return

                    st.markdown(
                        '<div class="eu-patient-mini-heading">'
                        f'<span>{_patient_html("Trend panel" if lang == "en" else "趋势面板")}</span>'
                        f'<b>{_patient_html(title)}</b>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    visible_items = trend_items[:6]
                    n_cols = min(3, len(visible_items))
                    n_rows = (len(visible_items) + n_cols - 1) // n_cols
                    subplot_titles = [
                        f"{display_name}{f' ({unit})' if unit else ''}"
                        for _concept, display_name, unit, _frame, _time_col, _value_col in visible_items
                    ]
                    fig = make_subplots(
                        rows=n_rows,
                        cols=n_cols,
                        subplot_titles=subplot_titles,
                        vertical_spacing=0.18 if n_rows > 1 else 0.12,
                        horizontal_spacing=0.08,
                    )
                    palette = ['#0f766e', '#334155', '#991b1b', '#6b7280', '#a16207', '#0e7490']
                    for idx, (concept_name, display_name, _unit, plot_frame, time_col, value_col) in enumerate(visible_items):
                        row = idx // n_cols + 1
                        col = idx % n_cols + 1
                        fig.add_trace(
                            go.Scatter(
                                x=plot_frame[time_col],
                                y=plot_frame[value_col],
                                mode='lines+markers',
                                name=display_name,
                                line=dict(color=palette[idx % len(palette)], width=2),
                                marker=dict(size=4),
                                hovertemplate=f"{display_name}: %{{y:.2f}}<extra></extra>",
                            ),
                            row=row,
                            col=col,
                        )

                    fig.update_layout(
                        height=240 if n_rows == 1 else 430,
                        template="plotly_white",
                        showlegend=False,
                        margin=dict(l=36, r=14, t=38, b=24),
                        font=dict(size=12, color='#111827'),
                    )
                    fig.update_xaxes(tickfont=dict(size=10, color='#4b5563'), showgrid=True, gridcolor='#ece7df')
                    fig.update_yaxes(tickfont=dict(size=10, color='#4b5563'), showgrid=True, gridcolor='#ece7df')
                    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_chart_config())

                vital_specs = [
                    ('hr', 'Heart rate' if lang == 'en' else '心率', 'bpm'),
                    ('map', 'MAP' if lang == 'en' else '平均动脉压', 'mmHg'),
                    ('sbp', 'SBP' if lang == 'en' else '收缩压', 'mmHg'),
                    ('resp', 'Respiratory rate' if lang == 'en' else '呼吸频率', '/min'),
                    ('temp', 'Temperature' if lang == 'en' else '体温', '°C'),
                    ('spo2', 'SpO2' if lang == 'en' else '血氧饱和度', '%'),
                ]
                lab_specs = [
                    ('lac', 'Lactate' if lang == 'en' else '乳酸', 'mmol/L'),
                    ('crea', 'Creatinine' if lang == 'en' else '肌酐', 'mg/dL'),
                    ('plt', 'Platelets' if lang == 'en' else '血小板', '10^9/L'),
                    ('wbc', 'WBC' if lang == 'en' else '白细胞', '10^9/L'),
                    ('hgb', 'Hemoglobin' if lang == 'en' else '血红蛋白', 'g/dL'),
                    ('bili', 'Bilirubin' if lang == 'en' else '胆红素', 'mg/dL'),
                ]
                _render_compact_trend_panel('Vital Signs Snapshot' if lang == 'en' else '生命体征快照', vital_specs)
                _render_compact_trend_panel('Key Laboratory Snapshot' if lang == 'en' else '关键实验室快照', lab_specs)

                # SOFA 评分趋势
                if 'sofa' in st.session_state.loaded_concepts:
                    sofa_df = st.session_state.loaded_concepts['sofa']
                    if isinstance(sofa_df, pd.DataFrame) and id_col in sofa_df.columns:
                        patient_sofa = sofa_df[sofa_df[id_col] == patient_id]
                        # 检测时间列
                        sofa_time_col = None
                        for tc in time_candidates:
                            if tc in patient_sofa.columns:
                                sofa_time_col = tc
                                break

                        if len(patient_sofa) > 0 and sofa_time_col and not screenshot_mode:
                            _render_patient_micro_heading(
                                "Score trajectory" if lang == 'en' else "评分轨迹",
                                "SOFA Score Trend" if lang == 'en' else "SOFA 评分趋势",
                                "Stacked organ contribution over the selected ICU stay." if lang == 'en' else "当前 ICU stay 内各器官评分贡献的堆叠轨迹。",
                            )

                            # SOFA 分解堆叠图
                            sofa_components = ['sofa_resp', 'sofa_coag', 'sofa_liver',
                                             'sofa_cardio', 'sofa_cns', 'sofa_renal']
                            available_components = [c for c in sofa_components if c in patient_sofa.columns]

                            if available_components:
                                fig = go.Figure()
                                colors = ['#0f766e', '#334155', '#6b7280', '#a16207', '#991b1b', '#0f172a']

                                for i, comp in enumerate(available_components):
                                    fig.add_trace(go.Bar(
                                        x=patient_sofa[sofa_time_col],
                                        y=patient_sofa[comp],
                                        name=comp.replace('sofa_', '').upper(),
                                        marker_color=colors[i]
                                    ))

                                time_label = "Time" if lang == 'en' else "时间"
                                score_label = "SOFA Score" if lang == 'en' else "SOFA 分数"
                                fig.update_layout(
                                    barmode='stack',
                                    template="plotly_white",
                                    height=350,
                                    xaxis_title=time_label,
                                    yaxis_title=score_label,
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                    font=dict(size=12, color='#111827'),
                                    margin=dict(l=52, r=18, t=54, b=44),
                                )
                                fig.update_xaxes(tickfont=dict(size=11, color='#4b5563'), title_font=dict(size=12, color='#4b5563'), gridcolor='#ece7df')
                                fig.update_yaxes(tickfont=dict(size=11, color='#4b5563'), title_font=dict(size=12, color='#4b5563'), gridcolor='#ece7df')

                                st.plotly_chart(fig, use_container_width=True, config=_get_plotly_chart_config())

                # ============ SOFA-1 vs SOFA-2 对比图表 ============
                has_sofa1 = 'sofa' in st.session_state.loaded_concepts
                has_sofa2 = 'sofa2' in st.session_state.loaded_concepts

                if has_sofa1 and has_sofa2:
                    _render_patient_micro_heading(
                        "Comparator" if lang == 'en' else "对照",
                        "SOFA-1 vs SOFA-2 Comparison" if lang == 'en' else "SOFA-1 与 SOFA-2 对比",
                        "Review score drift before using the case in cohort-level interpretation." if lang == 'en' else "在进入队列解释前，先复核评分体系变化。",
                    )

                    sofa1_df = st.session_state.loaded_concepts['sofa']
                    sofa2_df = st.session_state.loaded_concepts['sofa2']

                    # 获取患者数据
                    if isinstance(sofa1_df, pd.DataFrame) and id_col in sofa1_df.columns:
                        patient_sofa1 = sofa1_df[sofa1_df[id_col] == patient_id].copy()
                    else:
                        patient_sofa1 = pd.DataFrame()

                    if isinstance(sofa2_df, pd.DataFrame) and id_col in sofa2_df.columns:
                        patient_sofa2 = sofa2_df[sofa2_df[id_col] == patient_id].copy()
                    else:
                        patient_sofa2 = pd.DataFrame()

                    if len(patient_sofa1) > 0 and len(patient_sofa2) > 0:
                        # 检测时间列
                        time_col1 = None
                        time_col2 = None
                        for tc in time_candidates:
                            if tc in patient_sofa1.columns and time_col1 is None:
                                time_col1 = tc
                            if tc in patient_sofa2.columns and time_col2 is None:
                                time_col2 = tc

                        if time_col1 and time_col2:
                            # 1. 总分对比折线图
                            _render_patient_micro_heading(
                                "Total score" if lang == 'en' else "总分",
                                "Total Score Comparison" if lang == 'en' else "总分对比",
                            )

                            fig_total = go.Figure()

                            # SOFA-1 总分
                            if 'sofa' in patient_sofa1.columns:
                                fig_total.add_trace(go.Scatter(
                                    x=patient_sofa1[time_col1],
                                    y=patient_sofa1['sofa'],
                                    mode='lines+markers',
                                    name='SOFA-1 (Traditional)',
                                    line=dict(color='#0f172a', width=2.5),
                                    marker=dict(size=6)
                                ))

                            # SOFA-2 总分
                            if 'sofa2' in patient_sofa2.columns:
                                fig_total.add_trace(go.Scatter(
                                    x=patient_sofa2[time_col2],
                                    y=patient_sofa2['sofa2'],
                                    mode='lines+markers',
                                    name='SOFA-2 (2025 New)',
                                    line=dict(color='#0f766e', width=2.5, dash='dash'),
                                    marker=dict(size=6, symbol='diamond')
                                ))

                            time_label = "Time (hours from ICU admission)" if lang == 'en' else "时间 (ICU入院后小时)"
                            score_label = "Total SOFA Score" if lang == 'en' else "SOFA 总分"
                            fig_total.update_layout(
                                template="plotly_white",
                                height=300,
                                xaxis_title=time_label,
                                yaxis_title=score_label,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                hovermode='x unified',
                                font=dict(size=12, color='#111827'),
                                margin=dict(l=52, r=18, t=48, b=44),
                            )
                            fig_total.update_xaxes(tickfont=dict(size=11, color='#4b5563'), title_font=dict(size=12, color='#4b5563'), gridcolor='#ece7df')
                            fig_total.update_yaxes(tickfont=dict(size=11, color='#4b5563'), title_font=dict(size=12, color='#4b5563'), gridcolor='#ece7df')

                            st.plotly_chart(fig_total, use_container_width=True, config=_get_plotly_chart_config())

                            # 2. 子器官评分对比（6个子图）
                            _render_patient_micro_heading(
                                "Organ domains" if lang == 'en' else "器官域",
                                "Organ-specific Score Comparison" if lang == 'en' else "各器官评分对比",
                            )

                            # 定义器官映射
                            organ_pairs = [
                                ('sofa_resp', 'sofa2_resp', 'Respiratory', '呼吸'),
                                ('sofa_coag', 'sofa2_coag', 'Coagulation', '凝血'),
                                ('sofa_liver', 'sofa2_liver', 'Liver', '肝脏'),
                                ('sofa_cardio', 'sofa2_cardio', 'Cardiovascular', '心血管'),
                                ('sofa_cns', 'sofa2_cns', 'Neurological', '神经'),
                                ('sofa_renal', 'sofa2_renal', 'Renal', '肾脏'),
                            ]

                            # 🔧 检查器官评分列是否存在于各自的 DataFrame 中
                            # 如果不存在，尝试从其他加载的 concepts 中获取
                            def get_organ_data(patient_df, organ_col, time_col, loaded_concepts, id_col, patient_id):
                                """获取器官评分数据，优先从 sofa/sofa2 DataFrame，否则从单独加载的 concept"""
                                try:
                                    if organ_col in patient_df.columns and time_col in patient_df.columns:
                                        return patient_df[[time_col, organ_col]].copy()
                                    # 尝试从单独加载的 concept 获取
                                    if organ_col in loaded_concepts:
                                        organ_df = loaded_concepts[organ_col]
                                        if isinstance(organ_df, pd.DataFrame) and id_col in organ_df.columns:
                                            patient_organ = organ_df[organ_df[id_col] == patient_id].copy()
                                            if len(patient_organ) > 0 and organ_col in patient_organ.columns:
                                                # 找时间列
                                                for tc in ['time', 'charttime', 'starttime']:
                                                    if tc in patient_organ.columns:
                                                        return patient_organ[[tc, organ_col]].rename(columns={tc: time_col})
                                except Exception:
                                    pass
                                return None

                            # 创建 2x3 子图
                            from plotly.subplots import make_subplots

                            fig_organs = make_subplots(
                                rows=2, cols=3,
                                subplot_titles=[p[2] if lang == 'en' else p[3] for p in organ_pairs],
                                vertical_spacing=0.15,
                                horizontal_spacing=0.08
                            )

                            has_any_data = False
                            for idx, (sofa1_col, sofa2_col, en_name, zh_name) in enumerate(organ_pairs):
                                row = idx // 3 + 1
                                col = idx % 3 + 1

                                # SOFA-1 器官评分
                                sofa1_organ = get_organ_data(patient_sofa1, sofa1_col, time_col1,
                                                            st.session_state.loaded_concepts, id_col, patient_id)
                                if sofa1_organ is not None and len(sofa1_organ) > 0:
                                    has_any_data = True
                                    fig_organs.add_trace(
                                        go.Scatter(
                                            x=sofa1_organ[time_col1],
                                            y=sofa1_organ[sofa1_col],
                                            mode='lines+markers',
                                            name='SOFA-1' if idx == 0 else None,
                                            legendgroup='sofa1',
                                            showlegend=(idx == 0),
                                            line=dict(color='#0f172a', width=2),
                                            marker=dict(size=5)
                                        ),
                                        row=row, col=col
                                    )

                                # SOFA-2 器官评分
                                sofa2_organ = get_organ_data(patient_sofa2, sofa2_col, time_col2,
                                                            st.session_state.loaded_concepts, id_col, patient_id)
                                if sofa2_organ is not None and len(sofa2_organ) > 0:
                                    has_any_data = True
                                    fig_organs.add_trace(
                                        go.Scatter(
                                            x=sofa2_organ[time_col2],
                                            y=sofa2_organ[sofa2_col],
                                            mode='lines+markers',
                                            name='SOFA-2' if idx == 0 else None,
                                            legendgroup='sofa2',
                                            showlegend=(idx == 0),
                                            line=dict(color='#0f766e', width=2, dash='dash'),
                                            marker=dict(size=5, symbol='diamond')
                                        ),
                                        row=row, col=col
                                    )

                            if has_any_data:
                                fig_organs.update_layout(
                                    height=500,
                                    template="plotly_white",
                                    legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
                                    hovermode='x unified',
                                    font=dict(size=12, color='#111827'),
                                    margin=dict(l=44, r=18, t=58, b=42),
                                )

                                # 更新 y 轴范围 (0-4)
                                for i in range(1, 7):
                                    fig_organs.update_yaxes(range=[0, 4.5], row=(i-1)//3+1, col=(i-1)%3+1,
                                                           tickfont=dict(size=10, color='#4b5563'), title_font=dict(size=11, color='#4b5563'), gridcolor='#ece7df')
                                fig_organs.update_xaxes(tickfont=dict(size=10, color='#4b5563'), title_font=dict(size=11, color='#4b5563'), gridcolor='#ece7df')

                                st.plotly_chart(fig_organs, use_container_width=True, config=_get_plotly_chart_config())
                            else:
                                no_organ_msg = "Organ-specific scores are not available in current data. Load individual organ concepts (e.g., sofa_resp, sofa2_resp) to see detailed comparison." if lang == 'en' else "当前数据中无法获取器官子评分。请加载单独的器官概念（如 sofa_resp, sofa2_resp）以查看详细对比。"
                                _render_patient_notice(
                                    "info",
                                    "Organ domains" if lang == "en" else "器官域",
                                    "Organ comparison unavailable" if lang == "en" else "器官对比不可用",
                                    no_organ_msg,
                                )

                            # 3. 差异分析表格
                            _render_patient_micro_heading(
                                "Delta audit" if lang == 'en' else "差异审阅",
                                "Score Difference (SOFA-2 - SOFA-1)" if lang == 'en' else "评分差异 (SOFA-2 - SOFA-1)",
                            )

                            # 计算最新时间点的差异
                            latest_sofa1 = patient_sofa1.iloc[-1] if len(patient_sofa1) > 0 else {}
                            latest_sofa2 = patient_sofa2.iloc[-1] if len(patient_sofa2) > 0 else {}

                            diff_data = []
                            for sofa1_col, sofa2_col, en_name, zh_name in organ_pairs:
                                val1 = latest_sofa1.get(sofa1_col, 0) if isinstance(latest_sofa1, dict) or hasattr(latest_sofa1, 'get') else (latest_sofa1[sofa1_col] if sofa1_col in latest_sofa1.index else 0)
                                val2 = latest_sofa2.get(sofa2_col, 0) if isinstance(latest_sofa2, dict) or hasattr(latest_sofa2, 'get') else (latest_sofa2[sofa2_col] if sofa2_col in latest_sofa2.index else 0)
                                diff = val2 - val1
                                organ_name = en_name if lang == 'en' else zh_name
                                diff_data.append({
                                    'Organ' if lang == 'en' else '器官': organ_name,
                                    'SOFA-1': int(val1),
                                    'SOFA-2': int(val2),
                                    'Diff' if lang == 'en' else '差异': int(diff)
                                })

                            # 总分差异
                            total1 = latest_sofa1.get('sofa', 0) if isinstance(latest_sofa1, dict) or hasattr(latest_sofa1, 'get') else (latest_sofa1['sofa'] if 'sofa' in latest_sofa1.index else 0)
                            total2 = latest_sofa2.get('sofa2', 0) if isinstance(latest_sofa2, dict) or hasattr(latest_sofa2, 'get') else (latest_sofa2['sofa2'] if 'sofa2' in latest_sofa2.index else 0)
                            diff_data.append({
                                'Organ' if lang == 'en' else '器官': 'Total' if lang == 'en' else '总分',
                                'SOFA-1': int(total1),
                                'SOFA-2': int(total2),
                                'Diff' if lang == 'en' else '差异': int(total2 - total1)
                            })

                            diff_df = pd.DataFrame(diff_data)
                            if not screenshot_mode:
                                _st_dataframe_compat(st, diff_df, width="stretch", hide_index=True)
                    else:
                        no_compare = "Need both SOFA-1 and SOFA-2 data for comparison." if lang == 'en' else "需要同时有 SOFA-1 和 SOFA-2 数据才能对比。"
                        _render_patient_notice(
                            "pending",
                            "Comparator" if lang == "en" else "对比器",
                            "SOFA comparison waiting for paired scores" if lang == "en" else "SOFA 对比等待成对评分",
                            no_compare,
                        )

                # Dashboard 快速摘要面板
                _render_patient_micro_heading(
                    "Case signals" if lang == 'en' else "病例信号",
                    "Quick Summary" if lang == 'en' else "快速摘要",
                    "Endpoint and support flags computed from the currently loaded concepts." if lang == 'en' else "根据当前已加载概念计算终点与支持治疗信号。",
                )

                not_selected_status = "Not loaded" if lang == 'en' else "未加载"

                sepsis_status = not_selected_status
                sepsis_note = "sep3_sofa2 / sep3_sofa1" if lang == 'en' else "sep3_sofa2 / sep3_sofa1"
                sepsis_tone = "muted"
                found_sep = False
                concept_key = ''
                if 'sep3_sofa2' in st.session_state.loaded_concepts:
                    sep_df = st.session_state.loaded_concepts['sep3_sofa2']
                    concept_key = 'sep3_sofa2'
                    found_sep = True
                elif 'sep3_sofa1' in st.session_state.loaded_concepts:
                    sep_df = st.session_state.loaded_concepts['sep3_sofa1']
                    concept_key = 'sep3_sofa1'
                    found_sep = True

                if found_sep:
                    sepsis_status = "Unknown" if lang == 'en' else "未知"
                    sepsis_note = concept_key
                    if isinstance(sep_df, pd.DataFrame) and id_col in sep_df.columns:
                        patient_sep = sep_df[sep_df[id_col] == patient_id]
                        if len(patient_sep) > 0 and concept_key in patient_sep.columns:
                            if patient_sep[concept_key].max() == 1:
                                sepsis_status = "Flagged" if lang == 'en' else "已触发"
                                sepsis_tone = "danger"
                            else:
                                sepsis_status = "Clear" if lang == 'en' else "未触发"
                                sepsis_tone = "ok"
                        else:
                            sepsis_status = "No records" if lang == 'en' else "无记录"

                vent_status = not_selected_status
                vent_note = "vent_ind / mech_vent" if lang == 'en' else "vent_ind / mech_vent"
                vent_tone = "muted"
                vent_concepts = ['vent_ind', 'mech_vent', 'vent_start']
                found_vent = any(c in st.session_state.loaded_concepts for c in vent_concepts)
                if found_vent:
                    vent_status = "Unknown" if lang == 'en' else "未知"
                    for vc in vent_concepts:
                        vdf = st.session_state.loaded_concepts.get(vc)
                        if not isinstance(vdf, pd.DataFrame) or id_col not in vdf.columns:
                            continue
                        pvdf = vdf[vdf[id_col] == patient_id]
                        if pvdf.empty:
                            continue
                        val_col = vc if vc in pvdf.columns else pvdf.columns[-1]
                        vent_note = vc
                        try:
                            active = pd.to_numeric(pvdf[val_col], errors='coerce').fillna(0).max() > 0
                        except Exception:
                            active = False
                        vent_status = "Active" if active and lang == 'en' else ("使用中" if active else ("Absent" if lang == 'en' else "未使用"))
                        vent_tone = "warn" if active else "ok"
                        break

                vaso_status = not_selected_status
                vaso_note = "norepi / epi / dopa" if lang == 'en' else "升压药概念"
                vaso_tone = "muted"
                vaso_concepts = ['norepi_rate', 'epi_rate', 'dopa_rate', 'vaso_ind']
                found_vaso = any(c in st.session_state.loaded_concepts for c in vaso_concepts)
                if found_vaso:
                    vaso_status = "Absent" if lang == 'en' else "未使用"
                    vaso_tone = "ok"
                    for vc in vaso_concepts:
                        vdf = st.session_state.loaded_concepts.get(vc)
                        if not isinstance(vdf, pd.DataFrame) or id_col not in vdf.columns:
                            continue
                        pvdf = vdf[vdf[id_col] == patient_id]
                        if pvdf.empty:
                            continue
                        val_col = vc if vc in pvdf.columns else pvdf.columns[-1]
                        try:
                            active = pd.to_numeric(pvdf[val_col], errors='coerce').fillna(0).max() > 0
                        except Exception:
                            active = False
                        if active:
                            vaso_status = "Active" if lang == 'en' else "使用中"
                            vaso_note = vc
                            vaso_tone = "warn"
                            break

                gcs_val = "Not loaded" if lang == 'en' else "未加载"
                gcs_note = "gcs / sofa_cns" if lang == 'en' else "gcs / sofa_cns"
                gcs_tone = "muted"
                if 'gcs' in st.session_state.loaded_concepts:
                    gcs_val = "N/A"
                    gcs_df = st.session_state.loaded_concepts['gcs']
                    if isinstance(gcs_df, pd.DataFrame) and id_col in gcs_df.columns:
                        patient_gcs = gcs_df[gcs_df[id_col] == patient_id]
                        if len(patient_gcs) > 0 and 'gcs' in patient_gcs.columns:
                            val = patient_gcs['gcs'].iloc[-1]
                            try:
                                val_num = float(val)
                                gcs_val = safe_format_number(val_num, 0)
                                gcs_tone = "ok" if val_num >= 13 else ("warn" if val_num >= 9 else "danger")
                            except (ValueError, TypeError):
                                gcs_val = str(val)
                        else:
                            gcs_val = "No records" if lang == 'en' else "无记录"
                elif 'sofa_cns' in st.session_state.loaded_concepts or 'sofa2_cns' in st.session_state.loaded_concepts:
                    cns_col = 'sofa_cns' if 'sofa_cns' in st.session_state.loaded_concepts else 'sofa2_cns'
                    cns_df = st.session_state.loaded_concepts[cns_col]
                    if isinstance(cns_df, pd.DataFrame) and id_col in cns_df.columns:
                        patient_cns = cns_df[cns_df[id_col] == patient_id]
                        if len(patient_cns) > 0 and cns_col in patient_cns.columns:
                            cns_score = patient_cns[cns_col].iloc[-1]
                            gcs_note = f"estimated from {cns_col}" if lang == 'en' else f"由 {cns_col} 估计"
                            if cns_score == 0:
                                gcs_val, gcs_tone = "15 est.", "ok"
                            elif cns_score == 1:
                                gcs_val, gcs_tone = "13-14 est.", "ok"
                            elif cns_score == 2:
                                gcs_val, gcs_tone = "10-12 est.", "warn"
                            elif cns_score == 3:
                                gcs_val, gcs_tone = "6-9 est.", "danger"
                            elif cns_score == 4:
                                gcs_val, gcs_tone = "<6 est.", "danger"

                _patient_signal_grid([
                    ("Sepsis-3" if lang == 'en' else "脓毒症-3", sepsis_status, sepsis_note, sepsis_tone),
                    ("Mechanical vent" if lang == 'en' else "机械通气", vent_status, vent_note, vent_tone),
                    ("Vasopressors" if lang == 'en' else "血管活性药", vaso_status, vaso_note, vaso_tone),
                    ("GCS" if lang == 'en' else "GCS", gcs_val, gcs_note, gcs_tone),
                ])

                snapshot_candidates = []
                snapshot_excluded = {
                    'sep3_sofa1', 'sep3_sofa2', 'vent_ind', 'mech_vent', 'vent_start',
                    'norepi_rate', 'epi_rate', 'dopa_rate', 'vaso_ind', 'gcs'
                }
                for concept_name in available_concepts:
                    if concept_name in snapshot_excluded:
                        continue
                    latest_value = _latest_patient_value(concept_name, patient_id, id_col)
                    if latest_value is None:
                        continue
                    try:
                        formatted = safe_format_number(float(latest_value), 2)
                    except Exception:
                        formatted = str(latest_value)
                    snapshot_candidates.append((concept_name, formatted))

                if snapshot_candidates and not screenshot_mode:
                    _render_patient_micro_heading(
                        "Loaded values" if lang == 'en' else "已加载值",
                        "Feature Snapshot" if lang == 'en' else "特征快照",
                        "Latest patient-level values outside the endpoint/support flags." if lang == 'en' else "除终点和支持治疗外的最新患者级取值。",
                    )
                    visible_snapshots = snapshot_candidates[:8]
                    cards_html = "".join(
                        (
                            '<div class="tiny-stat-card">'
                            f'<div class="tiny-label">{html.escape(str(concept_name))}</div>'
                            f'<div class="tiny-value">{html.escape(str(formatted))}</div>'
                            '</div>'
                        )
                        for concept_name, formatted in visible_snapshots
                    )
                    st.markdown(
                        f'<div class="patient-feature-snapshot-grid">{cards_html}</div>',
                        unsafe_allow_html=True,
                    )
                    if len(snapshot_candidates) > len(visible_snapshots):
                        more_msg = f"Showing {len(visible_snapshots)} of {len(snapshot_candidates)} loaded features for this patient." if lang == 'en' else f"当前展示该患者 {len(snapshot_candidates)} 个已加载特征中的前 {len(visible_snapshots)} 个。"
                        st.markdown(
                            f'<p class="patient-feature-snapshot-caption">{html.escape(more_msg)}</p>',
                            unsafe_allow_html=True,
                        )

            except Exception as e:
                err_msg = f"Dashboard rendering failed: {e}" if lang == 'en' else f"综合仪表盘渲染失败: {e}"
                switch_msg = "Please try switching to 'Category View'" if lang == 'en' else "请尝试切换到「分类视图」"
                _render_patient_notice(
                    "danger",
                    "Dashboard guard" if lang == "en" else "仪表盘保护",
                    "Dashboard rendering failed" if lang == "en" else "综合仪表盘渲染失败",
                    err_msg,
                    switch_msg,
                )

        elif view_mode == category_mode:
            vitals = ['hr', 'map', 'sbp', 'resp', 'temp', 'spo2']
            sofa_concepts = ['sofa', 'sofa2']
            sepsis_concepts = ['sep3_sofa1', 'sep3_sofa2', 'susp_inf', 'infection_icd']
            labs = ['bili', 'crea', 'lac', 'lact', 'plt', 'wbc', 'hgb', 'hct', 'inr_pt', 'ptt', 'alb', 'glu', 'na', 'k', 'cl', 'bun']
            blood_gas = ['ph', 'pco2', 'po2', 'pafi', 'safi', 'be', 'hco3', 'bicar', 'fio2']
            vasopressors = ['norepi_rate', 'epi_rate', 'dopa_rate', 'dobu_rate', 'adh_rate', 'phn_rate', 'vaso_ind']
            resp_support = ['vent_ind', 'fio2', 'spo2', 'pafi', 'safi', 'resp']
            neuro = ['gcs', 'egcs', 'mgcs', 'vgcs', 'rass', 'avpu']
            renal = ['urine', 'urine24', 'crea', 'bun', 'rrt']
            other_scores = ['qsofa', 'sirs', 'mews', 'news']

            loaded_names = {
                name
                for name, frame in loaded_concepts_map.items()
                if isinstance(frame, pd.DataFrame)
            }
            labs_present = set(labs) & loaded_names
            blood_gas_present = set(blood_gas) & loaded_names

            _render_patient_category_section(
                "vital_signs",
                "Vital Signs" if lang == 'en' else "生命体征",
                "Latest observed value with compact trend context." if lang == 'en' else "最新观测值与紧凑趋势上下文。",
                vitals,
                decimals=1,
                no_data_body="No standard vital signs are present in the current loaded features." if lang == 'en' else "当前已加载特征中不包含标准生命体征。",
            )
            _render_patient_category_section(
                "sofa_score",
                "SOFA Score" if lang == 'en' else "SOFA 评分",
                "Severity scores stay in the same review rhythm as other patient signals." if lang == 'en' else "严重程度评分与其他患者信号保持同一审阅节奏。",
                sofa_concepts,
                decimals=0,
            )
            _render_patient_category_section(
                "sepsis_status",
                "Sepsis-3 Status" if lang == 'en' else "Sepsis-3 诊断",
                "Binary infection and organ-dysfunction flags are shown as reviewable status tiles." if lang == 'en' else "感染与器官功能障碍标记以可审阅状态卡展示。",
                sepsis_concepts,
                decimals=0,
                boolean_modes={concept: "last" for concept in sepsis_concepts},
                include_chart=False,
            )
            _render_patient_category_section(
                "laboratory_tests",
                "Laboratory Tests" if lang == 'en' else "实验室检查",
                "Key laboratory values keep their first-to-latest delta visible without default metric chrome." if lang == 'en' else "关键实验室指标保留首末差值，但不再使用默认 metric 外观。",
                labs,
                decimals=2,
            )
            _render_patient_category_section(
                "blood_gas",
                "Blood Gas Analysis" if lang == 'en' else "血气分析",
                "Gas-exchange and acid-base values use the same compact signal grid." if lang == 'en' else "气体交换与酸碱指标使用同一紧凑信号网格。",
                blood_gas,
                decimals=2,
            )
            _render_patient_category_section(
                "vasopressors",
                "Vasopressors" if lang == 'en' else "血管活性药物",
                "Dose traces and exposure flags are grouped together for bedside-style scanning." if lang == 'en' else "剂量轨迹与暴露标记合并展示，便于床旁式浏览。",
                vasopressors,
                decimals=3,
                boolean_modes={"vaso_ind": "max"},
            )
            _render_patient_category_section(
                "respiratory_support",
                "Respiratory Support" if lang == 'en' else "呼吸支持",
                "Ventilation flags and respiratory measurements avoid duplicate blood-gas cards." if lang == 'en' else "通气标记与呼吸测量会避开已在血气中展示的重复项。",
                [concept for concept in resp_support if concept not in blood_gas_present],
                decimals=1,
                boolean_modes={"vent_ind": "max"},
            )
            _render_patient_category_section(
                "neurological",
                "Neurological" if lang == 'en' else "神经系统",
                "Neurological scores use tone-coded tiles so low GCS remains visually legible." if lang == 'en' else "神经评分使用色调编码，低 GCS 保持清晰可见。",
                neuro,
                decimals=0,
            )
            _render_patient_category_section(
                "renal_function",
                "Renal Function" if lang == 'en' else "肾脏功能",
                "Renal output, chemistry, and RRT status are reviewed as one category." if lang == 'en' else "尿量、肾功能化验与 RRT 状态归入同一审阅分类。",
                [concept for concept in renal if concept not in labs_present],
                decimals=1,
                boolean_modes={"rrt": "max"},
            )
            _render_patient_category_section(
                "other_scores",
                "Other Scores" if lang == 'en' else "其他评分",
                "Screening scores are compact because they support context rather than own the page." if lang == 'en' else "筛查评分保持紧凑，用于补充上下文而非占据页面。",
                other_scores,
                decimals=0,
            )

        elif view_mode == table_mode:
            _render_patient_micro_heading(
                "Raw rows" if lang == 'en' else "原始行",
                "Patient Data Table" if lang == 'en' else "患者数据表格",
                "Per-concept extracted rows for the selected ICU stay." if lang == 'en' else "当前 ICU stay 的逐概念抽取行。",
            )
            for concept, df in st.session_state.loaded_concepts.items():
                if id_col in df.columns:
                    patient_df = df[df[id_col] == patient_id]
                else:
                    patient_df = df

                if len(patient_df) > 0:
                    records_label = "records" if lang == 'en' else "条记录"
                    with st.expander(f"{concept} ({len(patient_df)} {records_label})", expanded=False):
                        _st_dataframe_compat(st, patient_df, width="stretch")
