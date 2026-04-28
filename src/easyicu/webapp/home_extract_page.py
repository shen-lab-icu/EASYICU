"""Home extraction-mode rendering for the EasyICU Streamlit app."""

from __future__ import annotations

from typing import Any


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to this extracted renderer."""
    protected = {'render_home_extract_mode', "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def render_home_extract_mode(lang, app_context: dict[str, Any] | None = None):
    """渲染数据提取导出模式的首页教程。"""
    if app_context is not None:
        _install_app_context(app_context)
    

    # 计算当前步骤完成状态（4个步骤）
    # Step 1: Demo模式需要点击Confirm按钮，Real Data模式需要有效路径
    if st.session_state.get('use_mock_data', False):
        step1_done = st.session_state.get('step1_confirmed', False)
    else:
        step1_done = st.session_state.data_path and Path(st.session_state.data_path).exists()
    step2_done = st.session_state.get('step2_confirmed', False)
    # 🔧 FIX (2026-02-05): Step 3 必须点击确认按钮后才算完成
    step3_done = st.session_state.get('step3_confirmed', False) and len(st.session_state.get('selected_concepts', [])) > 0
    # Step 4 只在真正导出完成后才算完成
    step4_done = st.session_state.get('export_completed', False)

    # ============ 进度指示器 — Stepper 风格 ============
    st.markdown('<div id="progress"></div>', unsafe_allow_html=True)

    # 步骤数据
    _steps = [
        (step1_done, "1", ("Data Source", "配置数据源"), ("Configure database path", "选择数据路径")),
        (step2_done, "2", ("Cohort Selection", "队列筛选"), ("Define patient criteria", "定义患者筛选条件")),
        (step3_done, "3", ("Select Features", "选择特征"), ("Choose clinical variables", "选择临床变量")),
        (step4_done, "4", ("Export Data", "导出数据"), ("Generate output files", "输出数据文件")),
    ]

    # 确定当前活跃步骤
    _current_step = 0
    for i, (done, *_) in enumerate(_steps):
        if not done:
            _current_step = i
            break
    else:
        _current_step = 4  # 全部完成

    _render_extraction_pipeline_figure(
        lang=lang,
        step1_done=bool(step1_done),
        step2_done=bool(step2_done),
        step3_done=bool(step3_done),
        step4_done=bool(step4_done),
    )

    # ============ 动态引导内容 ============
    # 添加引导锚点和动态标题（根据当前步骤变化）
    st.markdown('<div id="guide"></div>', unsafe_allow_html=True)

    # 🆕 动态Guide标题，根据Progress自动转换
    if not step1_done:
        guide_step = "Data Source" if lang == 'en' else "数据源配置"
    elif not step2_done:
        guide_step = "Cohort Selection" if lang == 'en' else "队列筛选"
    elif not step3_done:
        guide_step = "Select Features" if lang == 'en' else "特征选择"
    elif not step4_done:
        guide_step = "Export Data" if lang == 'en' else "数据导出"
    else:
        guide_step = "Export Summary" if lang == 'en' else "导出摘要"

    if step4_done:
        guide_title_text = guide_step
    else:
        guide_title_text = f"Guide: {guide_step}" if lang == 'en' else f"引导: {guide_step}"
    st.markdown(f'<div class="workflow-guide-title">{html.escape(guide_title_text)}</div>', unsafe_allow_html=True)

    if not step1_done:
        # 步骤1引导：配置数据源
        if lang == 'en':
            st.markdown('''
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:24px;margin-bottom:12px">
                <div style="font-weight:800;color:#111827;font-size:1.42rem;margin-bottom:18px;letter-spacing:-0.02em">Configure Data Source in the Sidebar</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
                    <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:16px">
                        <div style="font-weight:700;color:#166534;margin-bottom:10px;font-size:1.12rem">🎭 Demo Mode</div>
                        <ul style="color:#334155;font-size:1.02rem;line-height:1.8;padding-left:20px;margin:0">
                            <li>No real data needed — generates simulated ICU data</li>
                            <li>Adjust patients (50-500) & duration (24-168h)</li>
                            <li>Click <b>"Confirm Data Source"</b> when ready</li>
                        </ul>
                    </div>
                    <div style="background:#eef2ff;border:1px solid #c7d2fe;border-radius:10px;padding:16px">
                        <div style="font-weight:700;color:#3730a3;margin-bottom:10px;font-size:1.12rem">📊 Real Data Mode</div>
                        <ul style="color:#334155;font-size:1.02rem;line-height:1.8;padding-left:20px;margin:0">
                            <li>MIMIC-IV, eICU, AUMC, HiRID, MIMIC-III, SICdb</li>
                            <li>Enter your local database path</li>
                            <li>All processing is local — data stays secure 🔒</li>
                        </ul>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:24px;margin-bottom:12px">
                <div style="font-weight:800;color:#111827;font-size:1.42rem;margin-bottom:18px;letter-spacing:-0.02em">在侧边栏配置数据源</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
                    <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:16px">
                        <div style="font-weight:700;color:#166534;margin-bottom:10px;font-size:1.12rem">🎭 演示模式</div>
                        <ul style="color:#334155;font-size:1.02rem;line-height:1.8;padding-left:20px;margin:0">
                            <li>无需真实数据 — 自动生成模拟ICU数据</li>
                            <li>可调整患者数量(50-500)和时长(24-168h)</li>
                            <li>设置后点击<b>「确认数据源配置」</b></li>
                        </ul>
                    </div>
                    <div style="background:#eef2ff;border:1px solid #c7d2fe;border-radius:10px;padding:16px">
                        <div style="font-weight:700;color:#3730a3;margin-bottom:10px;font-size:1.12rem">📊 真实数据模式</div>
                        <ul style="color:#334155;font-size:1.02rem;line-height:1.8;padding-left:20px;margin:0">
                            <li>支持 MIMIC-IV、eICU、AUMC、HiRID、MIMIC-III、SICdb</li>
                            <li>输入本地数据库路径</li>
                            <li>所有处理本地完成 — 数据安全 🔒</li>
                        </ul>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

    elif not step2_done:
        # 步骤2引导：队列筛选
        if lang == 'en':
            st.markdown('''
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:24px;margin-bottom:12px">
                <div style="font-weight:700;color:#111827;font-size:1.15rem;margin-bottom:14px">Configure Cohort Selection</div>
                <div style="background:#eef2ff;border:1px solid #c7d2fe;border-radius:10px;padding:16px;margin-bottom:12px">
                    <div style="font-weight:600;color:#3730a3;margin-bottom:8px">Available Filters</div>
                    <ul style="color:#4b5563;font-size:.92rem;line-height:1.7;padding-left:18px;margin:0">
                        <li><b>Age Range</b> — e.g., 18-65 years</li>
                        <li><b>Gender</b> — Male, Female, or Any</li>
                        <li><b>Survival Status</b> — Survivors, non-survivors, or all</li>
                        <li><b>ICU Stay</b> — Minimum length of stay</li>
                        <li><b>Clinical Cohorts</b> — Sepsis-3, AKI, circulatory failure, mechanical ventilation, RRT</li>
                        <li><b>ICD filter</b> — For MIMIC/eICU, narrow the cohort by ICD prefixes or diagnosis keywords</li>
                    </ul>
                </div>
                <div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;padding:12px 16px;font-size:.85rem;color:#92400e">
                    💡 Start from your target task, then define the cohort you need. You can still click <b>"Confirm (No Filtering)"</b> to skip this step.
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:24px;margin-bottom:12px">
                <div style="font-weight:700;color:#111827;font-size:1.15rem;margin-bottom:14px">配置队列筛选</div>
                <div style="background:#eef2ff;border:1px solid #c7d2fe;border-radius:10px;padding:16px;margin-bottom:12px">
                    <div style="font-weight:600;color:#3730a3;margin-bottom:8px">可用筛选条件</div>
                    <ul style="color:#4b5563;font-size:.92rem;line-height:1.7;padding-left:18px;margin:0">
                        <li><b>年龄范围</b> — 如 18-65 岁</li>
                        <li><b>性别</b> — 男/女/不限</li>
                        <li><b>存活状态</b> — 存活/死亡/全部</li>
                        <li><b>ICU住院时长</b> — 最短住院时长</li>
                        <li><b>疾病队列</b> — Sepsis-3、AKI、循环衰竭、机械通气、RRT</li>
                        <li><b>ICD 过滤</b> — 对 MIMIC/eICU 可按 ICD 前缀或诊断关键词缩小队列</li>
                    </ul>
                </div>
                <div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;padding:12px 16px;font-size:.85rem;color:#92400e">
                    💡 建议先从研究任务出发定义目标队列；若暂时不需要筛选，也可以点击<b>「确认（不筛选）」</b>跳过此步骤
                </div>
            </div>
            ''', unsafe_allow_html=True)

    elif not step3_done:
        # 步骤3引导：选择特征
        if lang == 'en':
            st.markdown('''
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:24px;margin-bottom:12px">
                <div style="font-weight:700;color:#111827;font-size:1.15rem;margin-bottom:14px">Select Features — 167 ICU Clinical Features</div>
                <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:14px">
                    <div style="background:#eff6ff;border-radius:8px;padding:12px"><b style="color:#1d4ed8">📊 Vital Signs</b><div style="color:#4b5563;font-size:.82rem;margin-top:3px">HR, BP, Temp, SpO2, Resp</div></div>
                    <div style="background:#ecfdf5;border-radius:8px;padding:12px"><b style="color:#047857">🧪 Lab Tests</b><div style="color:#4b5563;font-size:.82rem;margin-top:3px">Chemistry, CBC, Coag, ABG</div></div>
                    <div style="background:#fffbeb;border-radius:8px;padding:12px"><b style="color:#b45309">💊 Medications</b><div style="color:#4b5563;font-size:.82rem;margin-top:3px">Vasopressors, Sedatives, ABX</div></div>
                    <div style="background:#f5f3ff;border-radius:8px;padding:12px"><b style="color:#6d28d9">🏥 Scores</b><div style="color:#4b5563;font-size:.82rem;margin-top:3px">SOFA, GCS, AKI, Sepsis-3</div></div>
                </div>
                <div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;padding:12px 16px;font-size:.85rem;color:#92400e">
                    💡 Select by category or pick individual features. Check the Data Dictionary below for details.
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:24px;margin-bottom:12px">
                <div style="font-weight:700;color:#111827;font-size:1.15rem;margin-bottom:14px">选择特征 — 167 个 ICU 临床特征</div>
                <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:14px">
                    <div style="background:#eff6ff;border-radius:8px;padding:12px"><b style="color:#1d4ed8">📊 生命体征</b><div style="color:#4b5563;font-size:.82rem;margin-top:3px">心率、血压、体温、SpO2、呼吸</div></div>
                    <div style="background:#ecfdf5;border-radius:8px;padding:12px"><b style="color:#047857">🧪 实验室检验</b><div style="color:#4b5563;font-size:.82rem;margin-top:3px">生化、血常规、凝血、血气</div></div>
                    <div style="background:#fffbeb;border-radius:8px;padding:12px"><b style="color:#b45309">💊 药物治疗</b><div style="color:#4b5563;font-size:.82rem;margin-top:3px">血管活性药、镇静药、抗生素</div></div>
                    <div style="background:#f5f3ff;border-radius:8px;padding:12px"><b style="color:#6d28d9">🏥 临床评分</b><div style="color:#4b5563;font-size:.82rem;margin-top:3px">SOFA、GCS、AKI、Sepsis-3</div></div>
                </div>
                <div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;padding:12px 16px;font-size:.85rem;color:#92400e">
                    💡 按类别选择或选取单个特征，查看下方数据字典了解详情
                </div>
            </div>
            ''', unsafe_allow_html=True)

    elif not step4_done:
        # Step 4 Guide: Export Data
        # 🆕 检查是否正在导出或刚完成导出
        exporting_in_progress = st.session_state.get('_exporting_in_progress', False)

        if exporting_in_progress:
            # 🆕 导出正在进行中，显示进度标题
            _exp_msg = ('Export in Progress...', 'Please wait while your data is being exported.') if lang == 'en' else ('导出进行中...', '请稍候，数据正在导出中，进度详情将显示在下方。')
            st.markdown(f'''<div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:14px;padding:20px 24px;margin-bottom:12px">
<div style="font-weight:700;color:#b45309;font-size:1.05rem">⏳ {_exp_msg[0]}</div>
<div style="color:#92400e;font-size:.9rem;margin-top:4px">{_exp_msg[1]}</div>
</div>''', unsafe_allow_html=True)
        else:
            # 显示导出教程
            _steps_html = '<ol style="color:#374151;font-size:.92rem;line-height:2;margin:0;padding-left:18px"><li>{}</li><li>{}</li><li>{}</li><li>{}</li></ol>'
            _tip = ''
            if lang == 'en':
                _steps_html = _steps_html.format('Go to <b>"Data Export"</b> tab above', 'Select export format (CSV / Parquet / Excel)', 'Choose save location', 'Click <b>"Export Data"</b> button')
                _tip = '✅ Best for large datasets — saves directly to disk'
            else:
                _steps_html = _steps_html.format('点击上方 <b>"数据导出"</b> 标签页', '选择导出格式（CSV / Parquet / Excel）', '选择保存位置', '点击 <b>"导出数据"</b> 按钮')
                _tip = '✅ 适合大数据集 — 直接保存到磁盘，不占用内存'
            st.markdown(f'''<div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:24px;margin-bottom:12px">
<div style="font-weight:700;color:#111827;font-size:1.15rem;margin-bottom:14px">{"How to Export Data" if lang=="en" else "如何导出数据"}</div>
{_steps_html}
<div style="background:#ecfdf5;border-radius:8px;padding:10px 14px;margin-top:12px;font-size:.85rem;color:#047857">{_tip}</div>
</div>''', unsafe_allow_html=True)

            # 显示当前选择摘要
            selected = st.session_state.get('selected_concepts', [])
            if st.session_state.get('use_mock_data', False):
                source_info = "Demo Mode" if lang == 'en' else "演示模式"
            else:
                source_info = f"{st.session_state.get('data_path', '')}"

            source_label = "Data Source" if lang == 'en' else "数据源"
            feat_label = "Selected Features" if lang == 'en' else "已选特征"

            st.markdown(f'''
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
                <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;padding:14px">
                    <div style="font-size:.75rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px">{source_label}</div>
                    <div style="font-weight:600;color:#111827;margin-top:4px;font-size:.9rem">{source_info}</div>
                </div>
                <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;padding:14px">
                    <div style="font-size:.75rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px">{feat_label}</div>
                    <div style="font-weight:700;color:#6366f1;margin-top:4px;font-size:1.2rem">{len(selected)}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

        # 🆕 导出进度区域（无论是否正在导出都创建，导出时内容会填充进来）
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        export_section = st.container()
        st.session_state['_export_progress_container'] = export_section

    else:
        # 所有步骤完成 - Guide: Complete

        # 🆕 首先检查是否有刚完成的导出结果要显示
        export_result = st.session_state.get('_export_success_result')
        if export_result:
            # 显示导出成功消息
            exported_files = export_result['files']
            export_dir = export_result['export_dir']
            total_elapsed = export_result['total_time']
            module_times = export_result.get('module_times', {})
            # 🔧 FIX (2026-02-04): 使用保存的概念数
            concept_count = export_result.get('concept_count', len(exported_files))

            success_msg = f"✅ Successfully exported {len(exported_files)} files to `{export_dir}`" if lang == 'en' else f"✅ 成功导出 {concept_count} 个概念（{len(exported_files)} 个文件）到 `{export_dir}`"
            st.success(success_msg)

            # 🆕 显示队列筛选统计（在导出成功消息之后）
            cohort_stats = st.session_state.get('_cohort_stats')
            if cohort_stats and cohort_stats.get('excluded', 0) > 0:
                n_before = cohort_stats['before']
                n_excluded = cohort_stats['excluded']
                n_after = cohort_stats['after']
                details = cohort_stats.get('filter_details', [])
                if lang == 'en':
                    cohort_info = f"👥 **Cohort Selection**: {n_before} candidates → **{n_after} patients** exported ({n_excluded} excluded)"
                    if details:
                        reasons = ", ".join(f"{label_en}: -{cnt}" for label_en, _, cnt in details if cnt > 0)
                        if reasons:
                            cohort_info += f"\n\nExclusion reasons: {reasons}"
                else:
                    cohort_info = f"👥 **队列筛选**: {n_before} 候选 → 最终导出 **{n_after} 位患者**（排除 {n_excluded} 人）"
                    if details:
                        reasons = "、".join(f"{label_cn}: -{cnt}人" for _, label_cn, cnt in details if cnt > 0)
                        if reasons:
                            cohort_info += f"\n\n排除原因: {reasons}"
                st.info(cohort_info)

            # 显示时间统计
            time_stats_title = "⏱️ Export Time Statistics" if lang == 'en' else "⏱️ 导出耗时统计"
            with st.expander(time_stats_title, expanded=False):
                for mod_name, mod_time in module_times.items():
                    if mod_time >= 60:
                        time_str = f"{mod_time/60:.1f} min"
                    else:
                        time_str = f"{mod_time:.1f} s"
                    st.text(f"  • {mod_name}: {time_str}")

                if total_elapsed >= 60:
                    total_str = f"{total_elapsed/60:.1f} min"
                else:
                    total_str = f"{total_elapsed:.1f} s"
                total_msg = f"**Total: {total_str}**" if lang == 'en' else f"**总计: {total_str}**"
                st.markdown(total_msg)

            # 显示导出的文件列表
            view_files_label = "📁 View Exported Files" if lang == 'en' else "📁 查看导出文件"
            with st.expander(view_files_label, expanded=False):
                files_to_show = exported_files[:6]  # 只显示少量代表性文件
                num_cols = 2
                summary_msg = (
                    f"Showing {len(files_to_show)} representative files out of {len(exported_files)} exported files."
                    if lang == 'en' else
                    f"当前展示 {len(files_to_show)} 个代表性文件，共导出 {len(exported_files)} 个文件。"
                )
                st.caption(summary_msg)
                for i in range(0, len(files_to_show), num_cols):
                    cols = st.columns(num_cols)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(files_to_show):
                            with col:
                                st.markdown(f"<p style='color: #1e1e1e; font-size: 0.9rem; margin: 2px 0;'>• {Path(files_to_show[idx]).name}</p>", unsafe_allow_html=True)
                if len(exported_files) > len(files_to_show):
                    more_msg = (
                        f"... and {len(exported_files) - len(files_to_show)} more files"
                        if lang == 'en' else
                        f"... 及其他 {len(exported_files) - len(files_to_show)} 个文件"
                    )
                    st.markdown(f"<p style='color: #1e1e1e; font-size: 0.9rem; margin: 2px 0;'>{more_msg}</p>", unsafe_allow_html=True)

            # 🆕 显示被选择但未能提取的特征（区分无数据源 vs 无数据）
            unavailable_concepts = export_result.get('unavailable_concepts', [])
            unsupported_list = export_result.get('unsupported_concepts', [])
            empty_data_list = export_result.get('empty_data_concepts', [])
            if unavailable_concepts:
                # 🔧 FIX(2026-02-09): 分别显示无数据源和无数据的特征
                unsupported_in_unavail = [c for c in unavailable_concepts if c in unsupported_list]
                empty_or_other = [c for c in unavailable_concepts if c not in unsupported_list]

                parts_html = []
                if unsupported_in_unavail:
                    concepts_formatted = ', '.join(sorted(unsupported_in_unavail))
                    if lang == 'en':
                        parts_html.append(f'<p style="margin-bottom:5px;"><b>🚫 Not configured ({len(unsupported_in_unavail)}):</b> <span style="color:#64748b;">{concepts_formatted}</span></p>')
                    else:
                        parts_html.append(f'<p style="margin-bottom:5px;"><b>🚫 该数据库未配置 ({len(unsupported_in_unavail)})：</b> <span style="color:#64748b;">{concepts_formatted}</span></p>')
                if empty_or_other:
                    concepts_formatted = ', '.join(sorted(empty_or_other))
                    if lang == 'en':
                        parts_html.append(f'<p style="margin-bottom:5px;"><b>📭 No data for selected patients ({len(empty_or_other)}):</b> <span style="color:#64748b;">{concepts_formatted}</span></p>')
                    else:
                        parts_html.append(f'<p style="margin-bottom:5px;"><b>📭 所选患者无数据 ({len(empty_or_other)})：</b> <span style="color:#64748b;">{concepts_formatted}</span></p>')

                body = ''.join(parts_html)
                tip = '💡 <i>Try increasing the patient sample size or selecting All Patients to get more features.</i>' if lang == 'en' else '💡 <i>尝试增大患者样本量或选择全部患者以获取更多特征。</i>'
                unavailable_msg = f"""<div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:12px;padding:12px 18px;margin:8px 0 6px">
<div style="font-weight:600;color:#92400e;margin-bottom:8px">{len(unavailable_concepts)} {"selected features were not extracted" if lang=="en" else "个已选特征未被提取"}:</div>
{body}
<div style="margin-top:10px;font-size:.85rem;color:#6b7280">{tip}</div>
</div>"""
                st.markdown(unavailable_msg, unsafe_allow_html=True)

            st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
            # 🔧 FIX (2026-02-04): 在删除前保存概念数和患者数，供后面的卡片使用
            st.session_state['_last_export_concept_count'] = export_result.get('concept_count', len(exported_files))
            st.session_state['_last_export_patient_count'] = export_result.get('patient_count', 0)
            # 清除导出结果，避免重复显示
            del st.session_state['_export_success_result']

        # 显示状态概览卡片
        db_label = "Database" if lang == 'en' else "数据库"
        feat_label = "Loaded Concepts" if lang == 'en' else "已加载概念"
        patient_label = "Patients" if lang == 'en' else "患者数量"
        status_label = "Status" if lang == 'en' else "数据状态"
        ready_status = "Ready" if lang == 'en' else "就绪"

        db_display = "DEMO" if st.session_state.get('use_mock_data', False) else st.session_state.get('database', 'N/A').upper()

        # 计算概念数
        export_result = st.session_state.get('_export_success_result')
        if export_result and 'concept_count' in export_result:
            n_concepts = export_result['concept_count']
        elif '_last_export_concept_count' in st.session_state:
            n_concepts = st.session_state['_last_export_concept_count']
        elif st.session_state.loaded_concepts:
            n_concepts = len(st.session_state.loaded_concepts)
        elif st.session_state.get('selected_concepts'):
            n_concepts = len(st.session_state.selected_concepts)
        else:
            n_concepts = 0

        # 计算患者数
        n_patients = 0
        id_col = st.session_state.get('id_col', 'stay_id')
        if st.session_state.get('_exported_patient_count'):
            n_patients = st.session_state['_exported_patient_count']
        if n_patients == 0 and st.session_state.loaded_concepts:
            all_ids = set()
            for df in st.session_state.loaded_concepts.values():
                if isinstance(df, pd.DataFrame) and id_col in df.columns:
                    all_ids.update(df[id_col].unique())
            if all_ids:
                n_patients = len(all_ids)
        if n_patients == 0 and st.session_state.patient_ids:
            n_patients = len(st.session_state.patient_ids)
        if n_patients == 0:
            mock_params = st.session_state.get('mock_params', {})
            if mock_params.get('n_patients'):
                n_patients = mock_params['n_patients']

        st.markdown(f'''
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:8px">
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:10px 14px;min-height:72px;display:flex;flex-direction:column;justify-content:center">
                <div style="font-size:.67rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px;line-height:1.1">{db_label}</div>
                <div style="font-weight:700;color:#111827;font-size:1.02rem;margin-top:5px;line-height:1.1">{db_display}</div>
            </div>
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:10px 14px;min-height:72px;display:flex;flex-direction:column;justify-content:center">
                <div style="font-size:.67rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px;line-height:1.1">{feat_label}</div>
                <div style="font-weight:700;color:#6366f1;font-size:1.18rem;margin-top:5px;line-height:1.1">{n_concepts}</div>
            </div>
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:10px 14px;min-height:72px;display:flex;flex-direction:column;justify-content:center">
                <div style="font-size:.67rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px;line-height:1.1">{patient_label}</div>
                <div style="font-weight:700;color:#111827;font-size:1.18rem;margin-top:5px;line-height:1.1">{n_patients:,}</div>
            </div>
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:10px 14px;min-height:72px;display:flex;flex-direction:column;justify-content:center">
                <div style="font-size:.67rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px;line-height:1.1">{status_label}</div>
                <div style="font-weight:700;color:#10b981;font-size:1.02rem;margin-top:5px;line-height:1.1">✓ {ready_status}</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # 🆕 What's Next? 两个选项
        next_step_title = "What's Next?" if lang == 'en' else "下一步？"
        st.markdown(f'<div style="font-weight:700;color:#111827;font-size:1.1rem;margin:6px 0 8px">{next_step_title}</div>', unsafe_allow_html=True)

        col_opt1, col_opt2 = st.columns(2)

        with col_opt1:
            # Option A: Quick Visualization
            if lang == 'en':
                st.markdown('''<div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:14px 18px;min-height:188px;display:flex;flex-direction:column;justify-content:flex-start">
<div style="font-weight:700;color:#0369a1;margin-bottom:6px;line-height:1.2">📈 Quick Visualization</div>
<ul style="color:#374151;margin:0 0 0 16px;padding:0;font-size:.86rem;line-height:1.6">
<li><b>Data Tables Explorer</b> — Browse loaded data by module</li>
<li><b>Time Series Analysis</b> — Clinical trends over time</li>
<li><b>Patient Overview</b> — Single-patient dashboard</li>
<li><b>Data Quality</b> — Missing rates &amp; completeness</li>
</ul>
</div>''', unsafe_allow_html=True)
            else:
                st.markdown('''<div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:14px 18px;min-height:188px;display:flex;flex-direction:column;justify-content:flex-start">
<div style="font-weight:700;color:#0369a1;margin-bottom:6px;line-height:1.2">📈 快速可视化</div>
<ul style="color:#374151;margin:0 0 0 16px;padding:0;font-size:.86rem;line-height:1.6">
<li><b>数据表浏览器</b> — 按模块浏览已加载数据</li>
<li><b>时序分析</b> — 临床指标随时间变化趋势</li>
<li><b>患者概览</b> — 单患者综合仪表盘</li>
<li><b>数据质量</b> — 缺失率与完整性分析</li>
</ul>
</div>''', unsafe_allow_html=True)

            # Option A 按钮
            viz_label = "📈 Go to Visualization" if lang == 'en' else "📈 前往可视化"
            if st.button(viz_label, use_container_width=True, key="goto_viz_home", type="primary"):
                if st.session_state.get('last_export_dir') or st.session_state.get('viz_confirmed_path'):
                    st.session_state['viz_data_source_mode'] = 'exported'
                    st.session_state['_prefer_exported_viz'] = True
                st.session_state['_scroll_to_tab'] = 'viz'
                st.rerun()

        with col_opt2:
            # Option B: Cohort Analysis
            if lang == 'en':
                st.markdown('''<div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:14px 18px;min-height:188px;display:flex;flex-direction:column;justify-content:flex-start">
<div style="font-weight:700;color:#6d28d9;margin-bottom:6px;line-height:1.2">🔬 Cohort Analysis</div>
<ul style="color:#374151;margin:0 0 0 16px;padding:0;font-size:.82rem;line-height:1.55">
<li><b>Group Contrast Table</b> — subgroup balance and tests</li>
<li><b>Coverage Audit</b> — module coverage and eligibility flow</li>
<li><b>Cross-DB Benchmark</b> — harmonized feature shifts</li>
<li><b>Cohort Snapshot</b> — one-cohort phenotype and outcome profile</li>
<li><b>SOFA-1 vs SOFA-2</b> — definition-driven reclassification</li>
</ul>
</div>''', unsafe_allow_html=True)
            else:
                st.markdown('''<div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:14px 18px;min-height:188px;display:flex;flex-direction:column;justify-content:flex-start">
<div style="font-weight:700;color:#6d28d9;margin-bottom:6px;line-height:1.2">🔬 队列分析</div>
<ul style="color:#374151;margin:0 0 0 16px;padding:0;font-size:.82rem;line-height:1.55">
<li><b>组间对照表</b> — 亚组平衡与统计检验</li>
<li><b>覆盖度审计</b> — 模块覆盖度与纳排流程</li>
<li><b>跨库基准</b> — 标准化特征的数据库差异</li>
<li><b>队列快照</b> — 单一队列的表型与结局画像</li>
<li><b>SOFA-1 vs SOFA-2</b> — 定义变化导致的重新分层</li>
</ul>
</div>''', unsafe_allow_html=True)

            # Option B 按钮
            cohort_label = "🔬 Go to Cohort Analysis" if lang == 'en' else "🔬 前往队列分析"
            if st.button(cohort_label, use_container_width=True, key="goto_cohort_home", type="primary"):
                st.session_state['_scroll_to_tab'] = 'cohort'
                st.rerun()

        # 🆕 在 Guide: Complete 下方创建导出进度区域
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        export_section = st.container()
        st.session_state['_export_progress_container'] = export_section

    # ============ 数据字典展示 ============
    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
    # 添加字典锚点和大标题
    st.markdown('<div id="dictionary"></div>', unsafe_allow_html=True)
    dict_header = "📖 Data Dictionary" if lang == 'en' else "📖 数据字典"
    st.markdown(f'<h2 style="background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;border-bottom:3px solid #667eea;padding-bottom:10px;margin-top:10px;font-size:1.6rem;">{dict_header}</h2>', unsafe_allow_html=True)

    # 添加数据字典说明
    if lang == 'en':
        st.markdown('''
        <div style="background: rgba(102, 126, 234, 0.15); padding: 18px; border-radius: 12px; margin-bottom: 14px; border-left: 4px solid #667eea;">
            <p style="color: #333; font-size: 1.15rem; margin: 0; line-height: 1.7;">
                📚 <b>Reference Guide</b>: This dictionary contains all 167 ICU clinical features available in EasyICU, organized into 19 categories.
                Each feature includes its code name, full description, and measurement unit.
                Use this to understand what data you're extracting and make informed selections.
                Note that some features may not be available in all ICU databases.
            </p>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div style="background: rgba(102, 126, 234, 0.15); padding: 18px; border-radius: 12px; margin-bottom: 14px; border-left: 4px solid #667eea;">
            <p style="color: #333; font-size: 1.15rem; margin: 0; line-height: 1.7;">
                📚 <b>参考指南</b>：本字典包含 EasyICU 提供的全部 167 个 ICU 临床特征，分为 19 个类别。
                每个特征包括代码名称、完整描述和测量单位。
                使用此字典了解您正在提取的数据，做出明智的选择。
            </p>
        </div>
        ''', unsafe_allow_html=True)

    render_home_data_dictionary(lang)

    # 页脚信息
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if lang == 'en':
        st.markdown('''
        <div style="text-align:center;color:#aaa;font-size:0.85rem">
            <p>🏥 EasyICU - ICU Data Analysis Toolkit |
            📦 <a href="https://github.com/shen-lab-icu/EASYICU" style="color:#4fc3f7">GitHub</a> |
            📖 <a href="https://github.com/shen-lab-icu/EASYICU/blob/main/README.md" style="color:#4fc3f7">Docs</a></p>
            <p>All data processing is done locally, no data is uploaded to any server 🔒</p>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div style="text-align:center;color:#aaa;font-size:0.85rem">
            <p>🏥 EasyICU - ICU 数据分析工具包 |
            📦 <a href="https://github.com/shen-lab-icu/EASYICU" style="color:#4fc3f7">GitHub</a> |
            📖 <a href="https://github.com/shen-lab-icu/EASYICU/blob/main/README.md" style="color:#4fc3f7">Docs</a></p>
            <p>所有数据处理均在本地完成，不会上传到任何服务器 🔒</p>
        </div>
        ''', unsafe_allow_html=True)
