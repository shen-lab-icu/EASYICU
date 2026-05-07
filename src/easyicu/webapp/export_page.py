"""Export page rendering for the EasyICU Streamlit app."""

from __future__ import annotations

from typing import Any

from easyicu.webapp.compat import _dataframe_compat as _st_dataframe_compat


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to this extracted renderer."""
    protected = {'render_export_page', "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def render_export_page(app_context: dict[str, Any] | None = None):
    """渲染数据导出页面。"""
    if app_context is not None:
        _install_app_context(app_context)
    
    lang = st.session_state.get('language', 'en')
    _exp_title = "Data Export" if lang == 'en' else "数据导出"
    _exp_sub = "Download data in CSV, Parquet, or Excel" if lang == 'en' else "以CSV、Parquet或Excel格式下载数据"
    st.markdown(f'''
    <div style="margin-bottom:16px">
        <div style="font-size:1.4rem;font-weight:800;color:#111827">{_exp_title}</div>
        <div style="font-size:.88rem;color:#9ca3af;margin-top:2px">{_exp_sub}</div>
    </div>
    ''', unsafe_allow_html=True)

    if len(st.session_state.loaded_concepts) == 0:
        _msg = "Load data to enable export." if lang == 'en' else "请先加载数据以启用导出。"
        st.markdown(f'''
        <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:14px;padding:28px;text-align:center;margin:20px 0">
            <div style="font-size:2rem;margin-bottom:10px">💾</div>
            <div style="font-weight:600;color:#111827">{_msg}</div>
        </div>
        ''', unsafe_allow_html=True)
        return

    # 快速导出面板
    quick_title = "⚡ Quick Export" if lang == 'en' else "⚡ 快速导出"
    st.markdown(f"### {quick_title}")
    quick_cols = st.columns(5)

    import io
    from datetime import datetime

    with quick_cols[0]:
        # 一键导出所有CSV
        df_list = [df.assign(concept=name) for name, df in st.session_state.loaded_concepts.items()
                   if isinstance(df, pd.DataFrame) and len(df) > 0]
        if df_list:
            all_data = pd.concat(df_list, ignore_index=True)
            csv_all = all_data.to_csv(index=False, encoding='utf-8-sig')  # 🔧 FIX: 添加 BOM 编码防止中文乱码
            all_csv_label = "📄 All CSV" if lang == 'en' else "📄 全部CSV"
            all_csv_help = "Export all data as CSV" if lang == 'en' else "一键导出所有数据为CSV"
            st.download_button(
                label=all_csv_label,
                data=csv_all,
                file_name=f"easyicu_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                help=all_csv_help
            )
        else:
            no_data_label = "📄 No Data" if lang == 'en' else "📄 无数据"
            st.button(no_data_label, disabled=True, use_container_width=True)

    with quick_cols[1]:
        # 当前选中患者
        if st.session_state.get('selected_patient'):
            patient_id = st.session_state.selected_patient
            patient_data = {}
            for name, df in st.session_state.loaded_concepts.items():
                if isinstance(df, pd.DataFrame) and st.session_state.id_col in df.columns:
                    patient_df = df[df[st.session_state.id_col] == patient_id]
                    if len(patient_df) > 0:
                        patient_data[name] = patient_df

            if patient_data:
                patient_combined = pd.concat(
                    [df.assign(concept=name) for name, df in patient_data.items()],
                    ignore_index=True
                )
                patient_csv = patient_combined.to_csv(index=False, encoding='utf-8-sig')  # 🔧 FIX: BOM编码
                st.download_button(
                    label=f"👤 患者{patient_id}",
                    data=patient_csv,
                    file_name=f"patient_{patient_id}_{datetime.now().strftime('%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help=f"Export all data for patient {patient_id}" if lang == 'en' else f"导出患者 {patient_id} 的所有数据"
                )
            else:
                no_pat = "👤 No Patient" if lang == 'en' else "👤 无患者"
                st.button(no_pat, disabled=True, use_container_width=True)
        else:
            no_sel = "👤 No Selection" if lang == 'en' else "👤 未选患者"
            no_sel_help = "Please select a patient in Patient Overview first" if lang == 'en' else "请先在患者视图中选择一位患者"
            st.button(no_sel, disabled=True, use_container_width=True, help=no_sel_help)

    with quick_cols[2]:
        # 生命体征快速导出
        vitals = ['hr', 'map', 'sbp', 'resp', 'spo2', 'temp']
        vitals_data = {k: v for k, v in st.session_state.loaded_concepts.items()
                      if k in vitals and isinstance(v, pd.DataFrame) and len(v) > 0}
        if vitals_data:
            vitals_combined = pd.concat(
                [df.assign(concept=name) for name, df in vitals_data.items()],
                ignore_index=True
            )
            vitals_csv = vitals_combined.to_csv(index=False, encoding='utf-8-sig')  # 🔧 FIX: BOM编码
            vitals_label = "💓 Vitals" if lang == 'en' else "💓 生命体征"
            vitals_help = "Export all vital signs data" if lang == 'en' else "导出所有生命体征数据"
            st.download_button(
                label=vitals_label,
                data=vitals_csv,
                file_name=f"vitals_{datetime.now().strftime('%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                help=vitals_help
            )
        else:
            no_vitals = "💓 No Vitals" if lang == 'en' else "💓 无体征数据"
            st.button(no_vitals, disabled=True, use_container_width=True)

    with quick_cols[3]:
        # 实验室数据快速导出
        labs = ['bili', 'crea', 'plt', 'lac', 'wbc', 'hgb']
        labs_data = {k: v for k, v in st.session_state.loaded_concepts.items()
                    if k in labs and isinstance(v, pd.DataFrame) and len(v) > 0}
        if labs_data:
            labs_combined = pd.concat(
                [df.assign(concept=name) for name, df in labs_data.items()],
                ignore_index=True
            )
            labs_csv = labs_combined.to_csv(index=False, encoding='utf-8-sig')  # 🔧 FIX: BOM编码
            labs_label = "🧪 Labs" if lang == 'en' else "🧪 实验室"
            labs_help = "Export all laboratory data" if lang == 'en' else "导出所有实验室数据"
            st.download_button(
                label=labs_label,
                data=labs_csv,
                file_name=f"labs_{datetime.now().strftime('%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                help=labs_help
            )
        else:
            no_labs = "🧪 No Labs Data" if lang == 'en' else "🧪 无实验室数据"
            st.button(no_labs, disabled=True, use_container_width=True)

    with quick_cols[4]:
        pdf_label = "📑 PDF Report" if lang == 'en' else "📑 PDF 报告"
        pdf_help = (
            "Download a one-page summary report for the currently loaded Quick Visualization data."
            if lang == 'en' else
            "下载当前 Quick Visualization 已加载数据的一页式摘要报告。"
        )
        try:
            pdf_bytes = _build_quick_viz_pdf_report(
                lang=lang,
                preview_data=st.session_state.loaded_concepts,
                concepts_to_export=list(st.session_state.loaded_concepts.keys()),
            )
            st.download_button(
                label=pdf_label,
                data=pdf_bytes,
                file_name=f"easyicu_quick_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                help=pdf_help,
            )
        except Exception as exc:
            pdf_unavailable = (
                f"📑 PDF unavailable" if lang == 'en' else "📑 PDF 不可用"
            )
            st.button(pdf_unavailable, disabled=True, use_container_width=True, help=str(exc))

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 导出配置面板
    custom_title = "### 🎛️ Custom Export" if lang == 'en' else "### 🎛️ 自定义导出"
    st.markdown(custom_title)

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        concepts_label = "📋 Select Concepts" if lang == 'en' else "📋 选择 Concepts"
        concepts_help = "Select data types to export" if lang == 'en' else "选择要导出的数据类型"
        concepts_to_export = st.multiselect(
            concepts_label,
            options=list(st.session_state.loaded_concepts.keys()),
            default=list(st.session_state.loaded_concepts.keys()),
            help=concepts_help
        )

    with col2:
        format_label = "📁 Export Format" if lang == 'en' else "📁 导出格式"
        format_help = "CSV: Universal format\nExcel: Multi-sheet support\nParquet: Efficient storage" if lang == 'en' else "CSV: 通用格式\nExcel: 支持多Sheet\nParquet: 高效存储"
        export_format = st.selectbox(
            format_label,
            options=['CSV', 'Excel', 'Parquet'],
            help=format_help
        )

        format_icons = {'CSV': '📄', 'Excel': '📊', 'Parquet': '⚡'}
        selected_text = "Selected" if lang == 'en' else "已选择"
        st.markdown(f"<small>{format_icons.get(export_format, '')} {selected_text} {export_format}</small>", unsafe_allow_html=True)

    with col3:
        merge_label = "📦 Merge Mode" if lang == 'en' else "📦 合并模式"
        merge_options = ['Separate Files', 'Merge Into One'] if lang == 'en' else ['分开保存', '合并为一个文件']
        merge_help = "Separate: One file per Concept\nMerge: All data in one file" if lang == 'en' else "分开: 每个Concept一个文件\n合并: 所有数据合并"
        merge_mode = st.selectbox(
            merge_label,
            options=merge_options,
            help=merge_help
        )

    # 高级选项
    adv_label = "⚙️ Advanced Options" if lang == 'en' else "⚙️ 高级选项"
    with st.expander(adv_label, expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            filter_label = "Filter by Patient" if lang == 'en' else "按患者过滤"
            filter_patient = st.checkbox(filter_label, value=False)
            if filter_patient and st.session_state.patient_ids:
                select_patients_label = "Select Patients" if lang == 'en' else "选择患者"
                selected_patients = st.multiselect(
                    select_patients_label,
                    options=st.session_state.patient_ids[:100],
                    default=st.session_state.patient_ids[:5]
                )
            else:
                selected_patients = None

        with col2:
            index_label = "Include Row Index" if lang == 'en' else "包含行索引"
            include_index = st.checkbox(index_label, value=False)
            timestamp_label = "Add Timestamp to Filename" if lang == 'en' else "文件名添加时间戳"
            add_timestamp = st.checkbox(timestamp_label, value=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 导出预览
    preview_title = "### 📋 Export Preview" if lang == 'en' else "### 📋 导出预览"
    st.markdown(preview_title)

    preview_data = {}
    total_rows = 0
    total_cols = 0

    for name in concepts_to_export:
        df = st.session_state.loaded_concepts[name]

        # 确保是 DataFrame
        if not isinstance(df, pd.DataFrame):
            continue

        if selected_patients and st.session_state.id_col in df.columns:
            df = df[df[st.session_state.id_col].isin(selected_patients)]

        preview_data[name] = df
        total_rows += len(df)
        total_cols = max(total_cols, len(df.columns))

    # 预览统计卡片
    total_records_label = "Total Records" if lang == 'en' else "总记录数"
    est_size_label = "Est. Size" if lang == 'en' else "预估大小"
    format_label_2 = "Format" if lang == 'en' else "格式"
    est_size = total_rows * total_cols * 10 / 1024
    size_str = f"{est_size:.0f} KB" if est_size < 1024 else f"{est_size/1024:.1f} MB"

    st.markdown(f'''
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:clamp(8px,.4rem + .4vw,16px);margin-bottom:16px">
        <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px;text-align:center">
            <div style="font-size:.72rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px">Concepts</div>
            <div style="font-weight:700;color:#6366f1;font-size:1.4rem;margin-top:6px">{len(concepts_to_export)}</div>
        </div>
        <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px;text-align:center">
            <div style="font-size:.72rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px">{total_records_label}</div>
            <div style="font-weight:700;color:#111827;font-size:1.4rem;margin-top:6px">{total_rows:,}</div>
        </div>
        <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px;text-align:center">
            <div style="font-size:.72rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px">{est_size_label}</div>
            <div style="font-weight:700;color:#111827;font-size:1.2rem;margin-top:6px">{size_str}</div>
        </div>
        <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px;text-align:center">
            <div style="font-size:.72rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px">{format_label_2}</div>
            <div style="font-weight:700;color:#111827;font-size:1.2rem;margin-top:6px">{export_format}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # 数据预览表格
    if concepts_to_export:
        preview_exp_label = "👁️ Preview Data" if lang == 'en' else "👁️ 预览数据"
        with st.expander(preview_exp_label, expanded=False):
            select_preview_label = "Select Preview" if lang == 'en' else "选择预览"
            preview_concept = st.selectbox(select_preview_label, concepts_to_export)
            if preview_concept in preview_data:
                _st_dataframe_compat(
                    st,
                    preview_data[preview_concept].head(20),
                    width="stretch",
                    hide_index=True,
                )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 导出按钮
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        export_btn_label = "📥 Export Data" if lang == 'en' else "📥 导出数据"
        spinner_text = "Preparing export..." if lang == 'en' else "正在准备导出..."
        merge_single = "Merge Into One" if lang == 'en' else "合并为一个文件"

        if st.button(export_btn_label, type="primary", use_container_width=True):
            with st.spinner(spinner_text):
                import io
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""

                try:
                    filename_base = f"easyicu_export_{timestamp}" if timestamp else "easyicu_export"

                    if export_format == 'CSV':
                        if merge_mode == merge_single:
                            combined = pd.concat(
                                [df.assign(concept=name) for name, df in preview_data.items()],
                                ignore_index=True
                            )
                            csv = combined.to_csv(index=include_index, encoding='utf-8-sig')  # 🔧 FIX: BOM编码防止中文乱码
                            dl_csv = "⬇️ Download CSV" if lang == 'en' else "⬇️ 下载 CSV"
                            st.download_button(
                                label=dl_csv,
                                data=csv,
                                file_name=f"{filename_base}.csv",
                                mime="text/csv",
                            )
                        else:
                            # 分开保存 - 创建 ZIP
                            import zipfile
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                                for name, df in preview_data.items():
                                    csv_data = df.to_csv(index=include_index, encoding='utf-8-sig')  # 🔧 FIX: BOM编码
                                    zf.writestr(f"{name}.csv", csv_data)

                            dl_zip = "⬇️ Download ZIP (Multiple CSVs)" if lang == 'en' else "⬇️ 下载 ZIP (多个CSV)"
                            st.download_button(
                                label=dl_zip,
                                data=zip_buffer.getvalue(),
                                file_name=f"{filename_base}.zip",
                                mime="application/zip",
                            )

                    elif export_format == 'Excel':
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            if merge_mode == merge_single:
                                combined = pd.concat(
                                    [df.assign(concept=name) for name, df in preview_data.items()],
                                    ignore_index=True
                                )
                                combined.to_excel(writer, sheet_name='all_data', index=include_index)
                            else:
                                for name, df in preview_data.items():
                                    sheet_name = name[:31]  # Excel sheet name limit
                                    df.to_excel(writer, sheet_name=sheet_name, index=include_index)

                        dl_excel = "⬇️ Download Excel" if lang == 'en' else "⬇️ 下载 Excel"
                        st.download_button(
                            label=dl_excel,
                            data=output.getvalue(),
                            file_name=f"{filename_base}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )

                    elif export_format == 'Parquet':
                        combined = pd.concat(
                            [df.assign(concept=name) for name, df in preview_data.items()],
                            ignore_index=True
                        )
                        output = io.BytesIO()
                        combined.to_parquet(output, index=include_index)
                        dl_parquet = "⬇️ Download Parquet" if lang == 'en' else "⬇️ 下载 Parquet"
                        st.download_button(
                            label=dl_parquet,
                            data=output.getvalue(),
                            file_name=f"{filename_base}.parquet",
                            mime="application/octet-stream",
                        )

                    success_msg = "✅ Export ready! Click the button above to download" if lang == 'en' else "✅ 导出准备完成！点击上方按钮下载"
                    st.markdown(f'''
                    <div class="success-box">
                        {success_msg}
                    </div>
                    ''', unsafe_allow_html=True)

                except Exception as e:
                    err_msg = f"❌ Export failed: {e}" if lang == 'en' else f"❌ 导出失败: {e}"
                    st.error(err_msg)
