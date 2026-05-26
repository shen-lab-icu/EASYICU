"""CSV and database conversion workflows for the EasyICU webapp.

Header + source/target panel follow ``easyicu design/page-misc.jsx``
``PageConvertDialog``: a card-style dialog with a clear source→target
summary, then progress and per-file status.
"""

from __future__ import annotations

from typing import Any

from easyicu.webapp.cohort_charts import render_design_page_header
from easyicu.webapp.design_primitives import render_card_open, render_card_close


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to extracted workflows."""
    protected = {'render_convert_dialog', 'convert_csv_to_parquet', "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def render_convert_dialog(app_context: dict[str, Any] | None = None):
    """Render CSV to Parquet conversion dialog."""
    if app_context is not None:
        _install_app_context(app_context)


    lang = st.session_state.get('language', 'en')
    source_path = st.session_state.get('convert_source_path', '')

    st.markdown(
        render_design_page_header(
            kicker="CONVERT",
            title_en="CSV → Parquet",
            title_zh="CSV 转换为 Parquet",
            desc=(
                "Convert source files into the prepared Parquet layout the "
                "extraction APIs read from. Runs in place, validates after."
                if lang == 'en' else
                "把源文件就地转为提取 API 读取的 Parquet 布局，转换完成后自动校验。"
            ),
            lang=lang,
        ),
        unsafe_allow_html=True,
    )

    src_label = "Source" if lang == 'en' else "源目录"
    st.markdown(
        render_card_open(padding="12px 14px", extra_style="margin:8px 0 14px") +
        f'<div class="mono" style="font-size:11px;color:var(--ink-4)">{src_label}</div>'
        '<div style="display:flex;align-items:center;gap:8px;'
        'padding:6px 10px;border:1px solid var(--hair-2);background:var(--surface);'
        'border-radius:6px;margin-top:4px;font-size:12px">'
        f'<span class="mono" style="color:var(--ink-2);overflow:hidden;'
        f'text-overflow:ellipsis;white-space:nowrap">{source_path or "—"}</span>'
        '</div>' +
        render_card_close(),
        unsafe_allow_html=True,
    )

    # Parquet 文件就地写入源 CSV 同目录（与提取 API 的数据契约一致）
    overwrite_label = "Overwrite existing Parquet files" if lang == 'en' else "覆盖已存在的Parquet文件"
    overwrite = st.checkbox(overwrite_label, value=False)

    # 扫描可转换文件
    if source_path and Path(source_path).exists():
        csv_files = list(Path(source_path).rglob('*.csv')) + list(Path(source_path).rglob('*.csv.gz'))
        found_msg = f"**Found {len(csv_files)} CSV files to convert**" if lang == 'en' else f"**发现 {len(csv_files)} 个CSV文件可转换**"
        st.markdown(found_msg)

        view_label = "View file list" if lang == 'en' else "查看文件列表"
        with st.expander(view_label, expanded=False):
            for f in csv_files[:20]:
                size_mb = f.stat().st_size / (1024 * 1024)
                st.caption(f"• {f.name} ({size_mb:.1f} MB)")
            if len(csv_files) > 20:
                more_msg = f"... and {len(csv_files) - 20} more files" if lang == 'en' else f"... 及其他 {len(csv_files) - 20} 个文件"
                st.caption(more_msg)

    col1, col2 = st.columns([1, 1])

    with col1:
        start_label = "🚀 Start Conversion" if lang == 'en' else "🚀 开始转换"
        if st.button(start_label, type="primary", use_container_width=True):
            if not source_path or not Path(source_path).exists():
                err_msg = "❌ Source directory does not exist" if lang == 'en' else "❌ 源目录不存在"
                st.error(err_msg)
            else:
                spinner_msg = "Converting..." if lang == 'en' else "正在转换..."
                with st.spinner(spinner_msg):
                    success, failed = convert_csv_to_parquet(
                        source_path,
                        overwrite,
                    )

                # 转换完成后自动重新验证
                _database = st.session_state.get('database', 'miiv')
                _revalidation = validate_database_path(source_path, _database)

                if _revalidation['valid']:
                    # 全部就绪：转换成功 + 验证通过
                    st.session_state.path_validated = True
                    st.session_state.data_path = source_path
                    st.session_state.last_validation = _revalidation
                    st.session_state.last_validated_path = source_path
                    st.session_state.show_convert_dialog = False
                    _done_msg = (f"✅ Setup complete! Converted {success} items, all data validated."
                                 if lang == 'en' else
                                 f"✅ 设置完成！已转换 {success} 项，数据验证通过。")
                    st.success(_done_msg)
                    st.balloons()
                    import time as _t; _t.sleep(1.5)
                    st.rerun()
                elif success > 0:
                    # 部分完成：有转换但验证未完全通过
                    st.session_state.last_validation = _revalidation
                    st.session_state.last_validated_path = source_path
                    _partial_msg = (f"⚠️ Converted {success} items, but some data still needs attention."
                                    if lang == 'en' else
                                    f"⚠️ 已转换 {success} 项，但部分数据仍需处理。")
                    st.warning(_partial_msg)
                    st.error(_revalidation['message'])
                elif failed > 0:
                    fail_msg = (f"⚠️ {failed} files failed to convert. Please check error messages above."
                                if lang == 'en' else
                                f"⚠️ {failed} 个文件转换失败，请查看上方错误信息。")
                    st.warning(fail_msg)
                else:
                    no_files_msg = ("⚠️ No files were converted. Please check your data path."
                                    if lang == 'en' else
                                    "⚠️ 没有文件被转换，请检查数据路径。")
                    st.warning(no_files_msg)

    with col2:
        cancel_label = "❌ Cancel" if lang == 'en' else "❌ 取消"
        if st.button(cancel_label, use_container_width=True):
            st.session_state.show_convert_dialog = False
            st.rerun()


def convert_csv_to_parquet(
    source_dir: str,
    overwrite: bool = False,
    app_context: dict[str, Any] | None = None,
) -> tuple:
    """将源目录下的 CSV/CSV.GZ 文件就地转换为 Parquet。

    统一走 ``DataConverter``:小表写成扁平 ``<table>.parquet``,大表按 ID
    分区为 ricu 风格分片目录 ``<table>/N.parquet``;HiRID 的 tar.gz 归档会
    自动解压。Parquet 写在源 CSV 同目录下,与提取 API 的数据契约一致。
    """
    if app_context is not None:
        _install_app_context(app_context)

    import time

    lang = st.session_state.get('language', 'en')
    database = st.session_state.get('database', 'miiv')

    try:
        from easyicu.data_converter import ConversionStatus, DataConverter
    except ImportError as e:
        st.error(f"Converter not available: {e}")
        return 0, 0

    try:
        converter = DataConverter(
            data_path=str(source_dir), database=database, verbose=False
        )
    except ValueError as e:
        st.error(str(e))
        return 0, 0

    progress_bar = st.progress(0)
    status_text = st.empty()
    details = st.container()
    counters = {'success': 0, 'failed': 0}
    start_time = time.time()

    def _on_progress(info: dict) -> None:
        cur, tot = info['current'], info['total'] or 1
        progress_bar.progress(min(cur / tot, 1.0))
        name = info['file']
        result = info.get('result') or {}
        if info.get('status') == ConversionStatus.FAILED:
            counters['failed'] += 1
            with details:
                st.caption(f"❌ {name}: {(result.get('error') or 'unknown')[:60]}")
        else:
            counters['success'] += 1
            with details:
                rows_label = "rows" if lang == 'en' else "行"
                shards = result.get('shards', 0)
                suffix = f", {shards} shards" if shards else ""
                st.caption(f"✅ {name}: {result.get('row_count', 0):,} {rows_label}{suffix}")
        status_text.markdown(f"**Converting**: `{name}` [{cur}/{tot}]")

    try:
        converter.convert_all(force=overwrite, progress_callback=_on_progress)
    except Exception as e:
        st.error(f"Conversion error: {e}" if lang == 'en' else f"转换过程出错: {e}")
        return counters['success'], counters['failed'] or 1

    total_time = time.time() - start_time
    if total_time < 60:
        time_str = f"{total_time:.1f}s"
    elif total_time < 3600:
        time_str = f"{total_time / 60:.1f}min"
    else:
        time_str = f"{total_time / 3600:.1f}h"

    progress_bar.progress(1.0)
    status_text.empty()
    st.caption(f"✅ Completed in {time_str}" if lang == 'en' else f"✅ 完成，耗时 {time_str}")

    return counters['success'], counters['failed']
