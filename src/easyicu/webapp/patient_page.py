"""Patient detail page rendering for the EasyICU Streamlit app."""

from __future__ import annotations

from typing import Any

from easyicu.webapp.compat import _dataframe_compat as _st_dataframe_compat


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to the extracted patient page."""
    protected = {"render_patient_page", "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def render_patient_page(app_context: dict[str, Any] | None = None):
    """渲染患者视图页面。"""
    if app_context is not None:
        _install_app_context(app_context)

    lang = st.session_state.get('language', 'en')
    screenshot_mode = _is_screenshot_mode()
    loaded_concepts_map = st.session_state.loaded_concepts

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

    _pat_title = "Patient Overview" if lang == 'en' else "患者综合视图"
    _pat_sub = "Multi-dimensional patient dashboard" if lang == 'en' else "多维度患者仪表盘"
    st.markdown(f'''
    <div style="margin-bottom:16px">
        <div style="font-size:1.4rem;font-weight:800;color:#111827">{_pat_title}</div>
        <div style="font-size:.88rem;color:#9ca3af;margin-top:2px">{_pat_sub}</div>
    </div>
    ''', unsafe_allow_html=True)

    if len(st.session_state.loaded_concepts) == 0:
        _msg = "Load data to view patient dashboards." if lang == 'en' else "请先加载数据以查看患者视图。"
        st.markdown(f'''
        <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:14px;padding:28px;text-align:center;margin:20px 0">
            <div style="font-size:2rem;margin-bottom:10px">🏥</div>
            <div style="font-weight:600;color:#111827">{_msg}</div>
        </div>
        ''', unsafe_allow_html=True)
        return

    if not st.session_state.patient_ids:
        warn_msg = "⚠️ No patient data found" if lang == 'en' else "⚠️ 未找到患者数据"
        st.warning(warn_msg)
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
                _mort = "☠️ Yes" if float(v.iloc[0]) == 1 else "✅ No"
            elif cname == 'mech_vent' and float(v.iloc[0]) > 0:
                _supports.append("🌬️ MV")
            elif cname == 'rrt' and float(v.iloc[0]) > 0:
                _supports.append("🚰 RRT")
            elif cname in ('norepi_rate', 'epi_rate', 'dopa_rate', 'dobu_rate') and float(v.max()) > 0:
                _supports.append("💉 Vasopressors")

        _supports = list(set(_supports))
        _supports_str = ", ".join(_supports) if _supports else "—"
        _demo_lbl = get_text('demographics_header')
        _los_lbl = get_text('icu_los_label')
        _mort_lbl = get_text('mortality_label')
        _sup_lbl = get_text('key_supports')

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#f0f9ff,#e0f2fe);border:1px solid #bae6fd;border-radius:14px;padding:16px 20px;margin-bottom:16px">
            <div style="font-size:1.1rem;font-weight:800;color:#0369a1;margin-bottom:10px">📋 {get_text('patient_summary')} — Patient {pid}</div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px">
                <div><div style="font-size:.72rem;color:#6b7280;text-transform:uppercase;font-weight:700">{_demo_lbl}</div><div style="font-weight:600">{_sex}, {_age}y</div></div>
                <div><div style="font-size:.72rem;color:#6b7280;text-transform:uppercase;font-weight:700">{_los_lbl}</div><div style="font-weight:600">{_los}</div></div>
                <div><div style="font-size:.72rem;color:#6b7280;text-transform:uppercase;font-weight:700">{_mort_lbl}</div><div style="font-weight:600">{_mort}</div></div>
                <div><div style="font-size:.72rem;color:#6b7280;text-transform:uppercase;font-weight:700">{_sup_lbl}</div><div style="font-weight:600">{_supports_str}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 患者选择面板
    if not screenshot_mode:
        select_title = "Patient Selection" if lang == 'en' else "患者选择"
        st.markdown(f'''
        <div style="font-size:1.05rem;font-weight:700;color:#111827;margin-bottom:10px">{select_title}</div>
        ''', unsafe_allow_html=True)

    # 快速导航按钮
    first_btn = "⏮️ First" if lang == 'en' else "⏮️ 首位"
    prev_btn = "⬅️ Previous" if lang == 'en' else "⬅️ 上一位"
    next_btn = "➡️ Next" if lang == 'en' else "➡️ 下一位"
    last_btn = "⏭️ Last" if lang == 'en' else "⏭️ 末位"
    rand_btn = "🎲 Random" if lang == 'en' else "🎲 随机"
    first_help = "Jump to first patient" if lang == 'en' else "跳转到第一位患者"
    prev_help = "Previous patient" if lang == 'en' else "上一位患者"
    next_help = "Next patient" if lang == 'en' else "下一位患者"
    last_help = "Jump to last patient" if lang == 'en' else "跳转到最后一位患者"
    rand_help = "Random select a patient" if lang == 'en' else "随机选择一位患者"

    current_idx = st.session_state.patient_ids.index(st.session_state.get('patient_view_id', st.session_state.patient_ids[0]))
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
                st.session_state.patient_view_id = st.session_state.patient_ids[0]
                st.rerun()
        with nav_cols[1]:
            if st.button(prev_btn, use_container_width=True, help=prev_help):
                if current_idx > 0:
                    st.session_state.patient_view_id = st.session_state.patient_ids[current_idx - 1]
                    st.rerun()
        with nav_cols[2]:
            if st.button(next_btn, use_container_width=True, help=next_help):
                if current_idx < len(st.session_state.patient_ids) - 1:
                    st.session_state.patient_view_id = st.session_state.patient_ids[current_idx + 1]
                    st.rerun()
        with nav_cols[3]:
            if st.button(last_btn, use_container_width=True, help=last_help):
                st.session_state.patient_view_id = st.session_state.patient_ids[-1]
                st.rerun()
        with nav_cols[4]:
            if st.button(rand_btn, use_container_width=True, help=rand_help):
                import random
                st.session_state.patient_view_id = random.choice(st.session_state.patient_ids)
                st.rerun()
        with nav_cols[5]:
            st.markdown(f"<div style='text-align:center;padding:0.5rem;background:rgba(30,40,50,0.6);border-radius:4px'>{current_idx + 1}/{len(st.session_state.patient_ids)}</div>", unsafe_allow_html=True)

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
            pat_id_label = "👤 Patient ID" if lang == 'en' else "👤 患者 ID"
            patient_id = _patient_selector(
                patient_ids=st.session_state.patient_ids,
                state_key="patient_view_id",
                label=pat_id_label,
                lang=lang,
                max_display=200,
                default_patient=st.session_state.get('patient_view_id', st.session_state.patient_ids[0]),
            )

        with col2:
            view_label = "📋 View Mode" if lang == 'en' else "📋 显示模式"
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
            <div class="metric-card" style="padding:0.5rem 1rem">
                <div class="stat-label">{data_label}</div>
                <div style="display:flex;gap:1rem;font-size:0.9rem">
                    <span>📊 {n_concepts} total</span>
                    <span>❤️ {n_vitals} vitals</span>
                    <span>🧪 {n_labs} labs</span>
                    <span>📈 {n_scores} scores</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if patient_id:
        st.session_state.selected_patient = patient_id
        id_col = st.session_state.id_col

        if view_mode == dashboard_mode:
            # 自定义综合仪表盘
            dash_title = "### 📊 Dashboard" if lang == 'en' else "### 📊 综合仪表盘"
            st.markdown(dash_title)
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

                # Per-patient vitals trend (HR/MAP/SBP/RESP/SpO2) was
                # removed (2026-05 Phase C de-dup) because **Time Series →
                # Clinical Lanes** already plots the same vitals (and
                # adds clinical threshold annotations like Tachycardia /
                # Fever / Hypoxemia that this view didn't have). Keep
                # Patient Overview focused on per-patient *summary*; send
                # users to Time Series for trend inspection.
                if not screenshot_mode:
                    ts_hint_en = (
                        "Vital-sign trends moved to **Time Series** "
                        "(Clinical Lanes / Single Patient). They share the "
                        "same per-patient data plus clinical threshold "
                        "annotations (Tachycardia, Fever, Hypoxemia…) "
                        "that this view didn't have."
                    )
                    ts_hint_zh = (
                        "生命体征趋势已迁移至 **时间序列**"
                        "（临床通道 / 单患者视图），并附带 "
                        "心动过速 / 发热 / 低氧 等临床阈值标注。"
                    )
                    st.info(ts_hint_en if lang == 'en' else ts_hint_zh, icon="📈")

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
                            sofa_trend = "#### 📈 SOFA Score Trend" if lang == 'en' else "#### 📈 SOFA 评分趋势"
                            st.markdown(sofa_trend)

                            # SOFA 分解堆叠图
                            sofa_components = ['sofa_resp', 'sofa_coag', 'sofa_liver',
                                             'sofa_cardio', 'sofa_cns', 'sofa_renal']
                            available_components = [c for c in sofa_components if c in patient_sofa.columns]

                            if available_components:
                                fig = go.Figure()
                                colors = ['#ff6b6b', '#feca57', '#48dbfb', '#ff9ff3', '#54a0ff', '#5f27cd']

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
                                    font=dict(size=14, color='black'),
                                )
                                fig.update_xaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                                fig.update_yaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))

                                st.plotly_chart(fig, use_container_width=True, config=_get_plotly_chart_config())

                # ============ SOFA-1 vs SOFA-2 对比图表 ============
                has_sofa1 = 'sofa' in st.session_state.loaded_concepts
                has_sofa2 = 'sofa2' in st.session_state.loaded_concepts

                if has_sofa1 and has_sofa2:
                    compare_title = "#### 🔄 SOFA-1 vs SOFA-2 Comparison" if lang == 'en' else "#### 🔄 SOFA-1 与 SOFA-2 对比"
                    st.markdown(compare_title)

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
                            total_compare = "**Total Score Comparison**" if lang == 'en' else "**总分对比**"
                            st.markdown(total_compare)

                            fig_total = go.Figure()

                            # SOFA-1 总分
                            if 'sofa' in patient_sofa1.columns:
                                fig_total.add_trace(go.Scatter(
                                    x=patient_sofa1[time_col1],
                                    y=patient_sofa1['sofa'],
                                    mode='lines+markers',
                                    name='SOFA-1 (Traditional)',
                                    line=dict(color='#1f77b4', width=3),
                                    marker=dict(size=8)
                                ))

                            # SOFA-2 总分
                            if 'sofa2' in patient_sofa2.columns:
                                fig_total.add_trace(go.Scatter(
                                    x=patient_sofa2[time_col2],
                                    y=patient_sofa2['sofa2'],
                                    mode='lines+markers',
                                    name='SOFA-2 (2025 New)',
                                    line=dict(color='#ff7f0e', width=3, dash='dash'),
                                    marker=dict(size=8, symbol='diamond')
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
                                font=dict(size=14, color='black'),
                            )
                            fig_total.update_xaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                            fig_total.update_yaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))

                            st.plotly_chart(fig_total, use_container_width=True, config=_get_plotly_chart_config())

                            # 2. 子器官评分对比（6个子图）
                            organ_compare = "**Organ-specific Score Comparison**" if lang == 'en' else "**各器官评分对比**"
                            st.markdown(organ_compare)

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
                                            line=dict(color='#1f77b4', width=2),
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
                                            line=dict(color='#ff7f0e', width=2, dash='dash'),
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
                                    font=dict(size=14, color='black'),
                                )

                                # 更新 y 轴范围 (0-4)
                                for i in range(1, 7):
                                    fig_organs.update_yaxes(range=[0, 4.5], row=(i-1)//3+1, col=(i-1)%3+1,
                                                           tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
                                fig_organs.update_xaxes(tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))

                                st.plotly_chart(fig_organs, use_container_width=True, config=_get_plotly_chart_config())
                            else:
                                no_organ_msg = "ℹ️ Organ-specific scores not available in current data. Load individual organ concepts (e.g., sofa_resp, sofa2_resp) to see detailed comparison." if lang == 'en' else "ℹ️ 当前数据中无法获取器官子评分。请加载单独的器官概念（如 sofa_resp, sofa2_resp）以查看详细对比。"
                                st.info(no_organ_msg)

                            # 3. 差异分析表格
                            diff_title = "**Score Difference (SOFA-2 - SOFA-1)**" if lang == 'en' else "**评分差异 (SOFA-2 - SOFA-1)**"
                            st.markdown(diff_title)

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
                                'Organ' if lang == 'en' else '器官': '**Total**' if lang == 'en' else '**总分**',
                                'SOFA-1': int(total1),
                                'SOFA-2': int(total2),
                                'Diff' if lang == 'en' else '差异': int(total2 - total1)
                            })

                            diff_df = pd.DataFrame(diff_data)
                            if not screenshot_mode:
                                _st_dataframe_compat(st, diff_df, width="stretch", hide_index=True)
                    else:
                        no_compare = "ℹ️ Need both SOFA-1 and SOFA-2 data for comparison" if lang == 'en' else "ℹ️ 需要同时有 SOFA-1 和 SOFA-2 数据才能对比"
                        st.info(no_compare)

                # Dashboard 快速摘要面板
                summary_title = "#### 📋 Quick Summary" if lang == 'en' else "#### 📋 快速摘要"
                st.markdown(summary_title)

                summary_cols = st.columns(4)
                not_selected_status = "Not selected ⚪" if lang == 'en' else "未选择 ⚪"

                # Sepsis 状态
                with summary_cols[0]:
                    sepsis_status = not_selected_status
                    sepsis_color = "#6c757d"

                    found_sep = False
                    if 'sep3_sofa2' in st.session_state.loaded_concepts:
                        sep_df = st.session_state.loaded_concepts['sep3_sofa2']
                        concept_key = 'sep3_sofa2'
                        found_sep = True
                    elif 'sep3_sofa1' in st.session_state.loaded_concepts:
                        sep_df = st.session_state.loaded_concepts['sep3_sofa1']
                        concept_key = 'sep3_sofa1'
                        found_sep = True

                    if found_sep:
                        sepsis_status = "Unknown"
                        if isinstance(sep_df, pd.DataFrame) and id_col in sep_df.columns:
                            patient_sep = sep_df[sep_df[id_col] == patient_id]
                            if len(patient_sep) > 0 and concept_key in patient_sep.columns:
                                if patient_sep[concept_key].max() == 1:
                                    sepsis_status = "Sepsis ⚠️" if lang == 'en' else "脓毒症 ⚠️"
                                    sepsis_color = "#dc3545"
                                else:
                                    sepsis_status = "No Sepsis ✅" if lang == 'en' else "无脓毒症 ✅"
                                    sepsis_color = "#28a745"
                            else:
                                sepsis_status = "No Records" if lang == 'en' else "无记录"

                    st.markdown(f"**Sepsis-3**" if lang == 'en' else f"**脓毒症-3**")
                    st.markdown(f"<span style='color:{sepsis_color};font-weight:bold'>{sepsis_status}</span>", unsafe_allow_html=True)

                # 机械通气
                with summary_cols[1]:
                    vent_status = not_selected_status
                    vent_concepts = ['vent_ind', 'mech_vent', 'vent_start']

                    # 检查是否有相关 concept 被加载
                    found_vent = any(c in st.session_state.loaded_concepts for c in vent_concepts)

                    if found_vent:
                        vent_status = "Unknown"
                        if 'vent_ind' in st.session_state.loaded_concepts:
                            vent_df = st.session_state.loaded_concepts['vent_ind']
                            if isinstance(vent_df, pd.DataFrame) and id_col in vent_df.columns:
                                patient_vent = vent_df[vent_df[id_col] == patient_id]
                                if len(patient_vent) > 0 and 'vent_ind' in patient_vent.columns:
                                    vent_status = "Yes ✅" if patient_vent['vent_ind'].max() == 1 else "No ❌"
                                else:
                                    vent_status = "No Records" if lang == 'en' else "无记录"

                    st.markdown(f"**Mechanical Vent**" if lang == 'en' else f"**机械通气**")
                    st.markdown(vent_status)

                # 血管活性药物
                with summary_cols[2]:
                    vaso_status = not_selected_status
                    vaso_concepts = ['norepi_rate', 'epi_rate', 'dopa_rate', 'vaso_ind']

                    found_vaso = any(c in st.session_state.loaded_concepts for c in vaso_concepts)

                    if found_vaso:
                        vaso_status = "No ❌"
                        for vc in vaso_concepts:
                            if vc in st.session_state.loaded_concepts:
                                vdf = st.session_state.loaded_concepts[vc]
                                if isinstance(vdf, pd.DataFrame) and id_col in vdf.columns:
                                    pvdf = vdf[vdf[id_col] == patient_id]
                                    if len(pvdf) > 0:
                                        val_col = vc if vc in pvdf.columns else pvdf.columns[-1]
                                        if pvdf[val_col].max() > 0:
                                            vaso_status = "Yes ✅"
                                            break

                    st.markdown(f"**Vasopressors**" if lang == 'en' else f"**血管活性药**")
                    st.markdown(vaso_status)

                # GCS
                with summary_cols[3]:
                    gcs_val = "Not selected" if lang == 'en' else "未选择"
                    gcs_color = "#6c757d"

                    if 'gcs' in st.session_state.loaded_concepts:
                        gcs_val = "N/A"
                        gcs_df = st.session_state.loaded_concepts['gcs']
                        if isinstance(gcs_df, pd.DataFrame) and id_col in gcs_df.columns:
                            patient_gcs = gcs_df[gcs_df[id_col] == patient_id]
                            if len(patient_gcs) > 0 and 'gcs' in patient_gcs.columns:
                                val = patient_gcs['gcs'].iloc[-1]
                                try:
                                    val_num = float(val)
                                    gcs_color = "#28a745" if val_num >= 13 else ("#ffc107" if val_num >= 9 else "#dc3545")
                                    gcs_val = safe_format_number(val_num, 0)
                                except (ValueError, TypeError):
                                    gcs_val = str(val)
                                    gcs_color = "#6c757d"
                            else:
                                gcs_val = "No Records" if lang == 'en' else "无记录"
                    # 尝试从 sofa_cns 推断
                    elif 'sofa_cns' in st.session_state.loaded_concepts or 'sofa2_cns' in st.session_state.loaded_concepts:
                        cns_col = 'sofa_cns' if 'sofa_cns' in st.session_state.loaded_concepts else 'sofa2_cns'
                        cns_df = st.session_state.loaded_concepts[cns_col]
                        if isinstance(cns_df, pd.DataFrame) and id_col in cns_df.columns:
                            patient_cns = cns_df[cns_df[id_col] == patient_id]
                            if len(patient_cns) > 0 and cns_col in patient_cns.columns:
                                cns_score = patient_cns[cns_col].iloc[-1]
                                # 0:15, 1:13-14, 2:10-12, 3:6-9, 4:<6
                                if cns_score == 0: gcs_val, gcs_color = "15 (est)", "#28a745"
                                elif cns_score == 1: gcs_val, gcs_color = "13-14 (est)", "#28a745"
                                elif cns_score == 2: gcs_val, gcs_color = "10-12 (est)", "#ffc107"
                                elif cns_score == 3: gcs_val, gcs_color = "6-9 (est)", "#dc3545"
                                elif cns_score == 4: gcs_val, gcs_color = "<6 (est)", "#dc3545"

                    st.markdown("**GCS**")
                    st.markdown(f"<span style='color:{gcs_color};font-weight:bold;font-size:1.2rem'>{gcs_val}</span>", unsafe_allow_html=True)

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
                    snapshot_title = "#### 🧩 Loaded Feature Snapshot" if lang == 'en' else "#### 🧩 已加载特征快照"
                    st.markdown(snapshot_title)
                    visible_snapshots = snapshot_candidates[:8]
                    snap_cols = st.columns(min(4, len(visible_snapshots)))
                    for idx, (concept_name, formatted) in enumerate(visible_snapshots):
                        with snap_cols[idx % len(snap_cols)]:
                            st.markdown(
                                f'<div class="tiny-stat-card"><div class="tiny-label">{concept_name}</div><div class="tiny-value">{formatted}</div></div>',
                                unsafe_allow_html=True,
                            )
                    if len(snapshot_candidates) > len(visible_snapshots):
                        more_msg = f"Showing {len(visible_snapshots)} of {len(snapshot_candidates)} loaded features for this patient." if lang == 'en' else f"当前展示该患者 {len(snapshot_candidates)} 个已加载特征中的前 {len(visible_snapshots)} 个。"
                        st.caption(more_msg)

            except Exception as e:
                err_msg = f"Dashboard rendering failed: {e}" if lang == 'en' else f"综合仪表盘渲染失败: {e}"
                st.warning(err_msg)
                switch_msg = "Please try switching to 'Category View'" if lang == 'en' else "请尝试切换到「分类视图」"
                st.info(switch_msg)

        elif view_mode == category_mode:
            # 时间列候选（提前定义，避免UnboundLocalError）
            time_candidates = ['time', 'charttime', 'starttime', 'endtime', 'datetime', 'timestamp', 'Offset', 'measuredat_minutes', 'measuredat']

            # 生命体征
            vitals_title = "### ❤️ Vital Signs" if lang == 'en' else "### ❤️ 生命体征"
            st.markdown(vitals_title)
            vitals = ['hr', 'map', 'sbp', 'resp', 'temp', 'spo2']
            vitals_data = {k: v for k, v in st.session_state.loaded_concepts.items()
                          if k in vitals and isinstance(v, pd.DataFrame)}

            if vitals_data:
                cols = st.columns(min(3, len(vitals_data)))

                for i, (concept, df) in enumerate(vitals_data.items()):
                    with cols[i % 3]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df

                        if len(patient_df) > 0:
                            # 显示最新值
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            latest_val = patient_df[value_col].iloc[-1]
                            st.metric(concept.upper(), safe_format_number(latest_val, 1))

                            # 小型趋势图 - 检测时间列
                            time_col = None
                            for tc in time_candidates:
                                if tc in patient_df.columns:
                                    time_col = tc
                                    break
                            if time_col:
                                st.line_chart(patient_df.set_index(time_col)[value_col], height=120)
            else:
                no_vitals = "ℹ️ No standard vital signs are present in the current loaded features" if lang == 'en' else "ℹ️ 当前已加载特征中不包含标准生命体征"
                st.info(no_vitals)

            # SOFA/SOFA2 评分
            sofa_concepts = ['sofa', 'sofa2']
            sofa_data = {k: v for k, v in st.session_state.loaded_concepts.items()
                        if k in sofa_concepts and isinstance(v, pd.DataFrame)}

            if sofa_data:
                sofa_title = "### 📊 SOFA Score" if lang == 'en' else "### 📊 SOFA 评分"
                st.markdown(sofa_title)

                for sofa_key, sofa_df in sofa_data.items():
                    if id_col in sofa_df.columns:
                        patient_sofa = sofa_df[sofa_df[id_col] == patient_id]
                    else:
                        patient_sofa = sofa_df

                    if len(patient_sofa) > 0:
                        latest = patient_sofa.iloc[-1]
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            sofa_val = latest.get(sofa_key, 0)
                            sofa_color = "#28a745" if sofa_val < 6 else ("#ffc107" if sofa_val < 10 else "#dc3545")
                            label = f"Latest {sofa_key.upper()}" if lang == 'en' else f"最新 {sofa_key.upper()}"
                            st.markdown(f'''
                            <div class="metric-card" style="text-align:center">
                                <div class="stat-label">{label}</div>
                                <div class="stat-number" style="color:{sofa_color}">{sofa_val}</div>
                            </div>
                            ''', unsafe_allow_html=True)

                        with col2:
                            sofa_time_col = None
                            for tc in time_candidates:
                                if tc in patient_sofa.columns:
                                    sofa_time_col = tc
                                    break
                            if sofa_key in patient_sofa.columns and sofa_time_col:
                                st.line_chart(patient_sofa.set_index(sofa_time_col)[sofa_key], height=150)

            # Sepsis-3 诊断状态
            sepsis_concepts = ['sep3_sofa1', 'sep3_sofa2', 'susp_inf', 'infection_icd']
            sepsis_data = {k: v for k, v in st.session_state.loaded_concepts.items()
                          if k in sepsis_concepts and isinstance(v, pd.DataFrame)}

            if sepsis_data:
                sepsis_title = "### 🦠 Sepsis-3 Status" if lang == 'en' else "### 🦠 Sepsis-3 诊断"
                st.markdown(sepsis_title)
                cols = st.columns(len(sepsis_data))
                for i, (concept, df) in enumerate(sepsis_data.items()):
                    with cols[i]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df

                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            val = patient_df[value_col].iloc[-1] if len(patient_df) > 0 else 0
                            if val == 1:
                                st.markdown(f"✅ **{concept}**: Yes" if lang == 'en' else f"✅ **{concept}**: 是")
                            else:
                                st.markdown(f"❌ **{concept}**: No" if lang == 'en' else f"❌ **{concept}**: 否")

            # 实验室检查 - 扩展更多指标
            labs = ['bili', 'crea', 'lac', 'lact', 'plt', 'wbc', 'hgb', 'hct', 'inr_pt', 'ptt', 'alb', 'glu', 'na', 'k', 'cl', 'bun']
            labs_data = {k: v for k, v in st.session_state.loaded_concepts.items()
                        if k in labs and isinstance(v, pd.DataFrame)}

            if labs_data:
                labs_title = "### 🧪 Laboratory Tests" if lang == 'en' else "### 🧪 实验室检查"
                st.markdown(labs_title)
                cols = st.columns(min(4, len(labs_data)))
                for i, (concept, df) in enumerate(labs_data.items()):
                    with cols[i % 4]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df

                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            st.metric(
                                label=concept.upper(),
                                value=f"{patient_df[value_col].iloc[-1]:.2f}",
                                delta=f"{patient_df[value_col].iloc[-1] - patient_df[value_col].iloc[0]:.2f}" if len(patient_df) > 1 else None
                            )
                            lab_time_col = None
                            for tc in time_candidates:
                                if tc in patient_df.columns:
                                    lab_time_col = tc
                                    break
                            if lab_time_col:
                                st.line_chart(patient_df.set_index(lab_time_col)[value_col], height=120)

            # 血气分析
            blood_gas = ['ph', 'pco2', 'po2', 'pafi', 'safi', 'be', 'hco3', 'bicar', 'fio2']
            bg_data = {k: v for k, v in st.session_state.loaded_concepts.items()
                      if k in blood_gas and isinstance(v, pd.DataFrame)}

            if bg_data:
                bg_title = "### 🩸 Blood Gas Analysis" if lang == 'en' else "### 🩸 血气分析"
                st.markdown(bg_title)
                cols = st.columns(min(4, len(bg_data)))
                for i, (concept, df) in enumerate(bg_data.items()):
                    with cols[i % 4]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df

                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            st.metric(label=concept.upper(), value=f"{patient_df[value_col].iloc[-1]:.2f}")

            # 血管活性药物
            vasopressors = ['norepi_rate', 'epi_rate', 'dopa_rate', 'dobu_rate', 'adh_rate', 'phn_rate', 'vaso_ind']
            vaso_data = {k: v for k, v in st.session_state.loaded_concepts.items()
                        if k in vasopressors and isinstance(v, pd.DataFrame)}

            if vaso_data:
                vaso_title = "### 💉 Vasopressors" if lang == 'en' else "### 💉 血管活性药物"
                st.markdown(vaso_title)
                cols = st.columns(min(4, len(vaso_data)))
                for i, (concept, df) in enumerate(vaso_data.items()):
                    with cols[i % 4]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df

                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            if concept == 'vaso_ind':
                                val = patient_df[value_col].max()
                                st.markdown(f"**{concept}**: {'Yes ✅' if val == 1 else 'No ❌'}")
                            else:
                                st.metric(label=concept.upper(), value=f"{patient_df[value_col].iloc[-1]:.3f}")
                                vaso_time_col = None
                                for tc in time_candidates:
                                    if tc in patient_df.columns:
                                        vaso_time_col = tc
                                        break
                                if vaso_time_col:
                                    st.line_chart(patient_df.set_index(vaso_time_col)[value_col], height=100)

            # 呼吸支持
            resp_support = ['vent_ind', 'fio2', 'spo2', 'pafi', 'safi', 'resp']
            resp_data = {k: v for k, v in st.session_state.loaded_concepts.items()
                        if k in resp_support and isinstance(v, pd.DataFrame) and k not in bg_data}  # 避免重复

            if resp_data:
                resp_title = "### 💨 Respiratory Support" if lang == 'en' else "### 💨 呼吸支持"
                st.markdown(resp_title)
                cols = st.columns(min(4, len(resp_data)))
                for i, (concept, df) in enumerate(resp_data.items()):
                    with cols[i % 4]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df

                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            if concept == 'vent_ind':
                                val = patient_df[value_col].max()
                                st.markdown(f"**Mechanical Vent**: {'Yes ✅' if val == 1 else 'No ❌'}" if lang == 'en' else f"**机械通气**: {'是 ✅' if val == 1 else '否 ❌'}")
                            else:
                                st.metric(label=concept.upper(), value=safe_format_number(patient_df[value_col].iloc[-1], 1))

            # 神经系统
            neuro = ['gcs', 'egcs', 'mgcs', 'vgcs', 'rass', 'avpu']
            neuro_data = {k: v for k, v in st.session_state.loaded_concepts.items()
                         if k in neuro and isinstance(v, pd.DataFrame)}

            if neuro_data:
                neuro_title = "### 🧠 Neurological" if lang == 'en' else "### 🧠 神经系统"
                st.markdown(neuro_title)
                cols = st.columns(min(4, len(neuro_data)))
                for i, (concept, df) in enumerate(neuro_data.items()):
                    with cols[i % 4]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df

                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            val = patient_df[value_col].iloc[-1]
                            # GCS 颜色编码
                            if concept == 'gcs':
                                try:
                                    val_num = float(val)
                                    color = "#28a745" if val_num >= 13 else ("#ffc107" if val_num >= 9 else "#dc3545")
                                    st.markdown(f"<div style='color:{color};font-size:1.5rem;font-weight:bold'>GCS: {safe_format_number(val_num, 0)}</div>", unsafe_allow_html=True)
                                except (ValueError, TypeError):
                                    st.markdown(f"<div style='font-size:1.5rem;font-weight:bold'>GCS: {val}</div>", unsafe_allow_html=True)
                            else:
                                st.metric(label=concept.upper(), value=safe_format_number(val, 0))

            # 肾脏功能
            renal = ['urine', 'urine24', 'crea', 'bun', 'rrt']
            renal_data = {k: v for k, v in st.session_state.loaded_concepts.items()
                         if k in renal and isinstance(v, pd.DataFrame) and k not in labs_data}

            if renal_data:
                renal_title = "### 🚰 Renal Function" if lang == 'en' else "### 🚰 肾脏功能"
                st.markdown(renal_title)
                cols = st.columns(min(4, len(renal_data)))
                for i, (concept, df) in enumerate(renal_data.items()):
                    with cols[i % 4]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df

                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            if concept == 'rrt':
                                val = patient_df[value_col].max()
                                st.markdown(f"**RRT**: {'Yes ✅' if val == 1 else 'No ❌'}")
                            else:
                                st.metric(label=concept.upper(), value=safe_format_number(patient_df[value_col].iloc[-1], 1))

            # 其他评分
            other_scores = ['qsofa', 'sirs', 'mews', 'news']
            score_data = {k: v for k, v in st.session_state.loaded_concepts.items()
                         if k in other_scores and isinstance(v, pd.DataFrame)}

            if score_data:
                score_title = "### 📈 Other Scores" if lang == 'en' else "### 📈 其他评分"
                st.markdown(score_title)
                cols = st.columns(min(4, len(score_data)))
                for i, (concept, df) in enumerate(score_data.items()):
                    with cols[i % 4]:
                        if id_col in df.columns:
                            patient_df = df[df[id_col] == patient_id]
                        else:
                            patient_df = df

                        if len(patient_df) > 0:
                            value_col = concept if concept in patient_df.columns else patient_df.columns[-1]
                            st.metric(label=concept.upper(), value=safe_format_number(patient_df[value_col].iloc[-1], 0))

        elif view_mode == table_mode:
            table_title = "### 📋 Patient Data Table" if lang == 'en' else "### 📋 患者数据表格"
            st.markdown(table_title)
            for concept, df in st.session_state.loaded_concepts.items():
                if id_col in df.columns:
                    patient_df = df[df[id_col] == patient_id]
                else:
                    patient_df = df

                if len(patient_df) > 0:
                    records_label = "records" if lang == 'en' else "条记录"
                    with st.expander(f"{concept} ({len(patient_df)} {records_label})", expanded=False):
                        _st_dataframe_compat(st, patient_df, width="stretch")
