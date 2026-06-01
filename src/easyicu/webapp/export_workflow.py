"""Sidebar export workflow implementation for the EasyICU Streamlit app.

This module is intentionally a transitional extraction: the long export workflow
still uses the app-level helpers it historically depended on, but the bulky
implementation no longer lives in app.py.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path
import html
import os
import re

import numpy as np
import pandas as pd
import streamlit as st

from easyicu.webapp.concept_catalog import CONCEPT_GROUP_NAMES
from easyicu.webapp.services import normalize_column_name


_PROTECTED_CONTEXT_NAMES = {"execute_sidebar_export", "_install_app_context"}
_MODULE_LABEL_PREFIX_RE = re.compile(r"^[^\w\u4e00-\u9fff]+\s*")


def _concept_group_label(group_key: object, lang: str) -> str:
    """Return a user-facing module label without leaking internal keys."""
    raw = str(group_key or "").strip()
    names = globals().get("CONCEPT_GROUP_NAMES", CONCEPT_GROUP_NAMES)
    if isinstance(names, dict) and raw in names:
        en_name, zh_name = names[raw]
        text = str(en_name if lang == "en" else zh_name)
    else:
        text = raw.replace("_", " ").title()
    return _MODULE_LABEL_PREFIX_RE.sub("", text).strip() or text


def _render_export_progress_shell(
    *,
    lang: str,
    export_dir: Path,
    export_format: str,
    selected_concepts: list[str],
    is_preview_context: bool,
) -> None:
    """Render the design-system export progress wrapper before backend work."""
    title = "Packaging export bundle..." if lang == "en" else "正在打包导出包..."
    subtitle = (
        f"local-only · writing to {export_dir}"
        if lang == "en"
        else f"仅本地处理 · 写入 {export_dir}"
    )
    preview_msg = (
        "Preview sample detected. Export will re-extract data from the source database instead of exporting the preview sample."
        if lang == "en"
        else "检测到当前是 Preview 样本。导出会重新从源数据库提取数据，而不是直接导出这批 Preview 样本。"
    )
    rows = "".join(
        '<div class="eu-export-skeleton-row">'
        '<span></span><b></b><em></em>'
        "</div>"
        for _ in range(5)
    )
    preview_html = (
        f'<div class="compact-inline-notice info">{html.escape(preview_msg)}</div>'
        if is_preview_context
        else ""
    )
    st.markdown(
        f"""
        <div class="eu-export-progress-shell" id="export-progress">
          <div class="eu-export-progress-head">
            <span class="eu-spinner"></span>
            <div>
              <b>{html.escape(title)}</b>
              <small>{html.escape(subtitle)}</small>
            </div>
          </div>
          <div class="eu-export-progress-meta">
            <span>{html.escape("Features" if lang == "en" else "特征")} <b>{len(selected_concepts)}</b></span>
            <span>{html.escape("Format" if lang == "en" else "格式")} <b>{html.escape(export_format.upper())}</b></span>
            <span>{html.escape("Privacy" if lang == "en" else "隐私")} <b>{html.escape("local only" if lang == "en" else "仅本地")}</b></span>
          </div>
          <div class="eu-export-indeterminate"></div>
          <div class="eu-export-skeleton-table">{rows}</div>
        </div>
        {preview_html}
        """,
        unsafe_allow_html=True,
    )


def _render_export_conflict_panel(
    *,
    lang: str,
    pending_modules: list[str],
    existing_modules: dict[str, Path],
) -> None:
    """Render file conflicts as a blocked state, while keeping existing actions."""
    title = "Existing files detected" if lang == "en" else "检测到已存在的文件"
    desc = (
        "Choose whether to overwrite or skip matching module files before continuing the export."
        if lang == "en"
        else "继续导出前，请选择覆盖或跳过这些已存在的模块文件。"
    )
    rows = []
    for group_key in pending_modules:
        file_path = existing_modules[group_key]
        rows.append(
            '<div class="eu-export-conflict-row">'
            f'<span>{html.escape(_concept_group_label(group_key, lang))}</span>'
            f'<code>{html.escape(file_path.name)}</code>'
            "</div>"
        )
    question = (
        "How do you want to handle these files?"
        if lang == "en"
        else "请选择如何处理这些文件："
    )
    st.markdown(
        f"""
        <div class="eu-export-conflict-card">
          <div class="eu-export-conflict-glyph">!</div>
          <div class="eu-export-conflict-body">
            <b>{html.escape(title)}</b>
            <p>{html.escape(desc)}</p>
            <div class="eu-export-conflict-list">{''.join(rows)}</div>
            <div class="eu-export-conflict-question">{html.escape(question)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _export_extension_for_format(export_format: str) -> str:
    fmt = str(export_format or "").strip().lower()
    if fmt == "csv":
        return ".csv"
    if fmt in {"excel", "xlsx"}:
        return ".xlsx"
    return ".parquet"


def _terminate_process_tree(proc: Any, *, timeout: float = 2.0) -> None:
    """Terminate a multiprocessing worker and any child processes it spawned."""
    if proc is None:
        return

    pid = getattr(proc, "pid", None)
    if pid:
        try:
            import psutil

            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.Error:
                    pass

            try:
                proc.terminate()
            except Exception:
                pass

            _, alive = psutil.wait_procs(children, timeout=timeout)
            for child in alive:
                try:
                    child.kill()
                except psutil.Error:
                    pass

            try:
                parent.wait(timeout=timeout)
            except psutil.TimeoutExpired:
                try:
                    parent.kill()
                except psutil.Error:
                    pass

            try:
                proc.join(timeout=timeout)
            except Exception:
                pass
            return
        except Exception:
            pass

    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.join(timeout=timeout)
    except Exception:
        pass
    try:
        is_alive = proc.is_alive()
    except Exception:
        is_alive = False
    if is_alive:
        kill = getattr(proc, "kill", None)
        try:
            if callable(kill):
                kill()
            else:
                proc.terminate()
            proc.join(timeout=timeout)
        except Exception:
            pass


def _export_cancel_message(lang: str) -> str:
    return "Export stopped by user." if lang == 'en' else "导出已被用户停止。"


def _queue_export_cancel(state: Any, *, lang: str = 'en') -> None:
    """Return the UI to Step 4 after a user-requested export cancellation."""
    state['_export_cancelled'] = False
    state['trigger_export'] = False
    state['_exporting_in_progress'] = False
    state.pop('_export_conflict_pending', None)
    state.pop('_scroll_to_tab', None)
    state['_active_main_page'] = 'extract'
    state['_main_nav_widget'] = 'extract'
    state['_export_cancel_notice'] = _export_cancel_message(lang)


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to the extracted workflow."""
    for name, value in app_context.items():
        if not name.startswith("__") and name not in _PROTECTED_CONTEXT_NAMES:
            globals()[name] = value


def execute_sidebar_export(app_context: dict[str, Any] | None = None):
    """执行侧边栏触发的数据导出（直接导出到本地目录，带进度条）。

    🔧 进度显示在主内容区的专用容器中。
    🔧 支持三种模式：
        1. 模拟数据模式 (use_mock_data=True)
        2. 真实数据模式 (有有效的 data_path)
        3. 可视化导入模式 (loaded_data_origin='exported_files') - 直接导出已加载的数据
    """
    if app_context is not None:
        _install_app_context(app_context)

    from datetime import datetime

    lang = st.session_state.get('language', 'en')
    export_path = st.session_state.get('export_path', '')
    export_format = st.session_state.get('export_format', 'Parquet').lower()
    selected_concepts = st.session_state.get('selected_concepts', [])
    use_mock = st.session_state.get('use_mock_data', False)

    # 🔧 FIX (2026-02-03): 检测是否是从可视化模式导入数据的场景
    loaded_concepts = st.session_state.get('loaded_concepts', {})
    has_loaded_data = len(loaded_concepts) > 0
    loaded_data_origin = st.session_state.get('loaded_data_origin', 'none')
    preview_like_origins = {'preview', 'quick_preview'}
    is_preview_context = loaded_data_origin in preview_like_origins

    # 只有真正从“已导出文件”加载的数据，才视为 viz import mode。
    # Preview / quick_load / demo_viz 都不应该污染 Confirm Export 的真实提取流程。
    is_viz_import_mode = has_loaded_data and loaded_data_origin == 'exported_files'
    use_loaded_data_export = is_viz_import_mode

    # 🔧 FIX (2026-02-03): 在可视化导入模式下，如果 selected_concepts 为空，
    # 使用 loaded_concepts 的 keys 作为要导出的概念
    if is_viz_import_mode and not selected_concepts:
        selected_concepts = list(loaded_concepts.keys())
        st.session_state.selected_concepts = selected_concepts

    if not export_path or not Path(export_path).exists():
        err_msg = "❌ Please set a valid export path first" if lang == 'en' else "❌ 请先设置有效的导出路径"
        st.error(err_msg)
        return

    if not selected_concepts:
        err_msg = "❌ Please select features to export first" if lang == 'en' else "❌ 请先选择要导出的特征"
        st.error(err_msg)
        return

    try:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

        # 直接使用用户设置的导出路径（已包含数据库子目录）
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        _render_export_progress_shell(
            lang=lang,
            export_dir=export_dir,
            export_format=export_format,
            selected_concepts=selected_concepts,
            is_preview_context=is_preview_context,
        )

        # ── 内存预检：在创建进度条之前，低内存时要求用户确认 ────────────────
        # 实测：流式导出每模块到峰内存约94k患者约 1––1.3 GB，1.5 GB 阈値下才真正属于低内存场景
        _LOW_MEM_THRESHOLD_MB = 1536  # 1.5 GB
        _mem_check_confirmed = st.session_state.get("_low_mem_export_confirmed", False)
        if not _mem_check_confirmed and not is_viz_import_mode:
            try:
                from easyicu.memory_manager import get_available_memory_mb
                _avail_mb_pre = get_available_memory_mb()
                if _avail_mb_pre < _LOW_MEM_THRESHOLD_MB:
                    st.session_state["_exporting_in_progress"] = False
                    if lang == "en":
                        st.warning(
                            f"⚠️ **Low available memory detected: {_avail_mb_pre:.0f} MB** "
                            f"(recommended ≥ 1.5 GB).\n\n"
                            f"With this little RAM the export will be split into many small batches, "
                            f"which is much slower. **Please close other applications to free memory** "
                            f"and then click *Check Again*."
                        )
                        col_check, col_continue = st.columns(2)
                        with col_check:
                            if st.button("🔄 Check Again", key="_low_mem_recheck", use_container_width=True):
                                st.rerun()
                        with col_continue:
                            if st.button(
                                "⚡ Continue Anyway (slow)",
                                key="_low_mem_confirm",
                                use_container_width=True,
                                type="secondary",
                            ):
                                st.session_state["_low_mem_export_confirmed"] = True
                                st.session_state["_exporting_in_progress"] = True
                                st.rerun()
                    else:
                        st.warning(
                            f"⚠️ **检测到可用内存不足：{_avail_mb_pre:.0f} MB**（建议 ≥ 1.5 GB）。\n\n"
                            f"内存不足时导出会被分成很多小批次，速度大幅降低。"
                            f"**建议先关闭其他程序释放内存**，再点击「重新检测」。"
                        )
                        col_check, col_continue = st.columns(2)
                        with col_check:
                            if st.button("🔄 重新检测", key="_low_mem_recheck", use_container_width=True):
                                st.rerun()
                        with col_continue:
                            if st.button(
                                "⚡ 仍然继续（速度较慢）",
                                key="_low_mem_confirm",
                                use_container_width=True,
                                type="secondary",
                            ):
                                st.session_state["_low_mem_export_confirmed"] = True
                                st.session_state["_exporting_in_progress"] = True
                                st.rerun()
                    return  # 等待用户决策，不继续后续流程
            except Exception:
                pass  # 无法检测内存时直接继续

        exported_files = []
        module_times = {}
        all_exported_patient_ids = set()
        total_concepts = len(selected_concepts)

        # 创建进度条和状态显示
        progress_bar = st.progress(0)
        _export_action_cols = st.columns([4, 1])
        status_text = _export_action_cols[0].empty()

        def _set_status(message: str, *, level: str = "info") -> None:
            """Render export status as plain text to avoid markdown/spinner artifacts."""
            plain = str(message).replace("**", "").replace("`", "")
            if level == "success":
                status_text.success(plain)
            elif level == "warning":
                status_text.warning(plain)
            else:
                status_text.info(plain)

        # 🔧 添加取消按钮
        import time as time_module
        cancel_placeholder = _export_action_cols[1].empty()

        # 初始化取消状态
        if '_export_cancelled' not in st.session_state:
            st.session_state._export_cancelled = False

        stop_label = "Stop" if lang == 'en' else "停止"
        stop_help = (
            "Stop the export after the current module finishes."
            if lang == 'en' else
            "在当前模块结束后尽快停止导出。"
        )
        if cancel_placeholder.button(
            stop_label,
            key="stop_export_btn",
            use_container_width=True,
            help=stop_help,
            icon=":material/stop_circle:",
        ):
            _queue_export_cancel(st.session_state, lang=lang)
            st.rerun()

        def check_cancelled():
            """检查是否已取消导出"""
            return st.session_state.get('_export_cancelled', False)

        def _handle_export_cancel():
            """统一处理取消导出后的状态清理。"""
            st.session_state.trigger_export = False
            st.session_state['_exporting_in_progress'] = False
            st.session_state['_export_cancelled'] = False
            st.session_state.pop('_export_conflict_pending', None)
            st.session_state.pop('_scroll_to_tab', None)
            st.session_state['_active_main_page'] = 'extract'
            st.session_state['_main_nav_widget'] = 'extract'
            progress_bar.empty()
            status_text.empty()
            cancel_placeholder.empty()
            st.session_state['_export_cancel_notice'] = _export_cancel_message(lang)
            st.rerun()

        if check_cancelled():
            _handle_export_cancel()
            return

        # ============================================================
        # 🔧 步骤0：检测已存在的文件（适用于模拟数据和真实数据）
        # ============================================================
        # 构建 concept -> group_key 的映射
        concept_to_group = {}
        for group_key in CONCEPT_GROUPS_INTERNAL.keys():
            for c in CONCEPT_GROUPS_INTERNAL[group_key]:
                if c not in concept_to_group:
                    concept_to_group[c] = group_key

        # 找出用户选择的每个模块
        selected_modules = {}  # group_key -> [concepts]
        for c in selected_concepts:
            group_key = concept_to_group.get(c, 'other')
            if group_key not in selected_modules:
                selected_modules[group_key] = []
            selected_modules[group_key].append(c)

        # 检测哪些模块的文件已存在
        # 🔧 FIX (2026-02-05): 使用模块名开头匹配，cohort条件在后缀
        existing_modules = {}  # group_key -> file_path
        cohort_suffix = _generate_cohort_prefix()

        for group_key, group_concepts in selected_modules.items():
            # 🔧 按模块名开头查找已存在的文件
            search_prefix = f"{group_key}_"
            target_ext = _export_extension_for_format(export_format)

            # 检查是否有匹配该模块的文件存在
            matching_files = list(export_dir.glob(f"{search_prefix}*{target_ext}"))
            if matching_files:
                # 找到匹配当前目标格式的文件
                existing_modules[group_key] = matching_files[0]

        # 如果有已存在的模块，显示让用户选择
        # 🔧 FIX (2026-02-03): 在 viz_import_mode 下自动覆盖，跳过对话框
        if existing_modules and not is_viz_import_mode:
            # 检查用户是否已做出所有决定
            skipped_modules = st.session_state.get('_skipped_modules', set())
            overwrite_modules = st.session_state.get('_overwrite_modules', set())

            # 找出尚未决定的模块
            pending_modules = [m for m in existing_modules.keys()
                               if m not in skipped_modules and m not in overwrite_modules]

            if pending_modules:
                # 显示冲突对话框时并未真正在导出：清除进行中标记。
                st.session_state['_exporting_in_progress'] = False
                # 首次检测到冲突时，标记“等待用户选择”并重新进入，
                # 让页面横幅显示等待状态而不是“导出进行中”。
                if not st.session_state.get('_export_conflict_pending'):
                    st.session_state['_export_conflict_pending'] = True
                    st.session_state.trigger_export = True
                    st.rerun()

                _render_export_conflict_panel(
                    lang=lang,
                    pending_modules=pending_modules,
                    existing_modules=existing_modules,
                )

                # 🔧 FIX: 使用 on_click 回调而不是 if st.button，避免页面跳转
                def on_overwrite_all():
                    """覆盖全部的回调函数"""
                    # 将所有 existing_modules 添加到 overwrite 列表
                    all_modules = set(st.session_state.get('_existing_modules_list', []))
                    st.session_state['_overwrite_modules'] = all_modules
                    st.session_state.pop('_export_conflict_pending', None)
                    st.session_state['_exporting_in_progress'] = True
                    # 🔧 FIX: 设置 trigger_export 并让它rerun来继续执行
                    st.session_state.trigger_export = True

                def on_skip_all():
                    """跳过全部的回调函数"""
                    all_modules = set(st.session_state.get('_existing_modules_list', []))
                    st.session_state['_skipped_modules'] = all_modules
                    st.session_state.pop('_export_conflict_pending', None)
                    st.session_state['_exporting_in_progress'] = True
                    # 🔧 FIX: 设置 trigger_export 并让它rerun来继续执行
                    st.session_state.trigger_export = True

                # 🔧 保存 pending_modules 到 session_state 让回调能访问
                st.session_state['_existing_modules_list'] = list(existing_modules.keys())

                col_all_overwrite, col_all_skip = st.columns(2)
                with col_all_overwrite:
                    all_overwrite_btn = "Overwrite all" if lang == 'en' else "全部覆盖"
                    st.button(all_overwrite_btn, key="file_overwrite_all", type="primary",
                             use_container_width=True, on_click=on_overwrite_all)
                with col_all_skip:
                    all_skip_btn = "Skip all" if lang == 'en' else "全部跳过"
                    st.button(all_skip_btn, key="file_skip_all", type="secondary", use_container_width=True,
                             on_click=on_skip_all)

                # 🔧 FIX: 重新检查用户是否已做出决定（回调可能已更新 session_state）
                overwrite_modules = st.session_state.get('_overwrite_modules', set())
                skipped_modules = st.session_state.get('_skipped_modules', set())
                pending_modules = [m for m in existing_modules.keys()
                                   if m not in skipped_modules and m not in overwrite_modules]

                if pending_modules:
                    # 用户尚未做出决定，暂停导出
                    return

        # 根据用户选择，确定要跳过的模块
        skipped_modules = st.session_state.get('_skipped_modules', set())
        concepts_to_skip = set()
        for group_key in skipped_modules:
            if group_key in selected_modules:
                for c in selected_modules[group_key]:
                    concepts_to_skip.add(c)

        # 过滤掉将跳过的概念
        concepts_to_export = [c for c in selected_concepts if c not in concepts_to_skip]

        if not concepts_to_export:
            if concepts_to_skip:
                skip_msg = f"⏭️ All selected modules already exist, nothing to export" if lang == 'en' else "⏭️ 所有选中的模块都已存在，无需导出"
                st.info(skip_msg)
                existing_files = [str(existing_modules[group_key]) for group_key in skipped_modules if group_key in existing_modules]
                existing_patient_count = st.session_state.get('_exported_patient_count') or len(st.session_state.get('patient_ids', []))
                _prime_export_completion(export_dir, existing_files, auto_load=True)
                st.session_state['_export_success_result'] = {
                    'files': existing_files,
                    'export_dir': str(export_dir),
                    'total_time': 0,
                    'module_times': {},
                    'patient_count': existing_patient_count,
                    'concept_count': len(selected_concepts),
                    'unavailable_concepts': [],
                    'unsupported_concepts': [],
                    'empty_data_concepts': [],
                    'note': (
                        "All selected modules already existed. EasyICU reused the current export folder."
                        if lang == 'en' else
                        "所有选中的模块都已存在。EasyICU 已直接复用当前导出目录。"
                    ),
                }
                _write_export_manifest(
                    export_dir,
                    exported_files=existing_files,
                    patient_count=existing_patient_count,
                    concept_count=len(selected_concepts),
                    export_format=export_format,
                    note=st.session_state['_export_success_result']['note'],
                )
            # 清理状态
            if '_skipped_modules' in st.session_state:
                del st.session_state['_skipped_modules']
            if '_overwrite_modules' in st.session_state:
                del st.session_state['_overwrite_modules']
            st.rerun()
            return

        # 显示跳过信息
        if concepts_to_skip:
            skip_count = len(concepts_to_skip)
            load_count = len(concepts_to_export)
            skip_info = f"⏭️ Skipping {skip_count} concepts (files exist), exporting {load_count} concepts" if lang == 'en' else f"⏭️ 跳过 {skip_count} 个概念（文件已存在），导出 {load_count} 个概念"
            st.info(skip_info)

        # 🔧 FIX: 初始化变量，避免 demo 模式下引用未定义变量
        unsupported_concepts = []
        empty_concepts = []
        failed_concepts = []

        if use_mock or use_loaded_data_export:
            # 生成模拟数据并导出
            if use_mock:
                gen_msg = "Generating mock data..." if lang == 'en' else "正在生成模拟数据..."
                _set_status(gen_msg)
                # 🔧 使用 get_mock_params_with_cohort 获取完整参数（包含最新的 cohort_filter）
                params = get_mock_params_with_cohort()
                all_mock_data, patient_ids = generate_mock_data(**params)

                # 保存患者ID列表（用于其他功能）
                st.session_state.patient_ids = patient_ids

                # 🔧 根据要导出的 concepts 过滤数据（排除跳过的）
                data = {}
                for concept in concepts_to_export:
                    if concept in all_mock_data:
                        data[concept] = all_mock_data[concept]

                # 显示加载情况
                loaded_count = len(data)
                if loaded_count < len(concepts_to_export):
                    missing = [c for c in concepts_to_export if c not in all_mock_data]
                    skip_msg = f"⚠️ {len(missing)} concepts not in mock data: {', '.join(missing[:5])}" if lang == 'en' else f"⚠️ 模拟数据中不存在 {len(missing)} 个概念: {', '.join(missing[:5])}"
                    st.warning(skip_msg)
            else:
                _set_status("Using loaded visualization data..." if lang == 'en' else "正在使用已加载的可视化数据...")
                data = {concept: loaded_concepts[concept] for concept in concepts_to_export if concept in loaded_concepts}
                if not data:
                    err_msg = "❌ No loaded visualization data matches the selected features" if lang == 'en' else "❌ 当前已加载的可视化数据中没有匹配所选特征"
                    st.error(err_msg)
                    st.session_state['_exporting_in_progress'] = False
                    return

            progress_bar.progress(0.3)
        else:
            # 加载真实数据并导出（批量并行加载）
            from easyicu import load_concepts
            import os

            # 🔧 FIX (2026-02-15): 每次导出前清除 easyicu 缓存，防止上次导出的患者数据泄漏
            # 问题场景：用户先导出100患者到dir1，再导出"全部数据"到dir2
            # 如果不清缓存，_concept_data_cache 中的旧数据可能被混入新导出
            try:
                from easyicu.cache_manager import clear_easyicu_cache
                clear_easyicu_cache()
            except Exception:
                pass
            try:
                from easyicu.api import clear_global_loader
                clear_global_loader()
            except Exception:
                pass

            # 🔧 FIX: 检查 data_path 是否有效（可视化模式导入数据后可能无效）
            data_path_str = st.session_state.get('data_path', '')
            if not data_path_str or not Path(data_path_str).exists():
                err_msg = "❌ Data path is not set or invalid. Please go back to Tutorial tab and configure a valid database path first." if lang == 'en' else "❌ 数据路径未设置或无效。请返回Tutorial标签页先配置有效的数据库路径。"
                st.error(err_msg)
                st.session_state['_exporting_in_progress'] = False
                return

            # 批量并行加载所有特征
            patient_limit_display = st.session_state.get('patient_limit', 0)
            patient_info = f"({patient_limit_display} patients)" if patient_limit_display else "(all patients)"
            patient_info_cn = f"（{patient_limit_display}患者）" if patient_limit_display else "（全部患者）"
            batch_msg = f"Loading concepts {patient_info}..." if lang == 'en' else f"正在加载概念 {patient_info_cn}..."
            _set_status(batch_msg)

            # 🚀 性能优化：参照 extract_baseline_features.py 的配置
            patient_limit = st.session_state.get('patient_limit', 0)

            patient_ids_filter = None
            id_col = 'stay_id'
            data_path = Path(data_path_str)
            database = st.session_state.get('database', 'miiv')
            id_col_map = {'miiv': 'stay_id', 'eicu': 'patientunitstayid', 'aumc': 'admissionid', 'hirid': 'patientid', 'mimic': 'icustay_id', 'sic': 'CaseID'}
            id_col = id_col_map.get(database, 'stay_id')

            # 👥 先从数据库选patient_limit个患者，再对这些患者做人群筛选
            try:
                # Step 1: 先选patient_limit个患者作为候选集
                candidate_ids = None
                if patient_limit and patient_limit > 0:
                    try:
                        for f in _get_patient_id_table_files(database):
                            fp = data_path / f
                            if fp.exists():
                                icustays_df = pd.read_parquet(fp, columns=[id_col] if id_col else None)
                                if id_col in icustays_df.columns:
                                    all_ids = icustays_df[id_col].unique().tolist()
                                    candidate_ids = _sample_patient_ids_random(all_ids, patient_limit)
                                    break
                    except Exception:
                        pass

                # Step 2: 在候选集上应用人群筛选
                database_for_cohort = st.session_state.get('database', 'miiv')
                cohort_result = apply_cohort_filter(data_path_str, database_for_cohort, candidate_ids=candidate_ids)
                if cohort_result is not None:
                    cohort_id_col = cohort_result['id_col']
                    filtered_ids = cohort_result['filtered_ids']
                    id_col = cohort_id_col
                    patient_ids_filter = {id_col: filtered_ids}
                    # Save cohort stats for completion message
                    n_before = len(candidate_ids) if candidate_ids else 0
                    n_after = len(filtered_ids)
                    st.session_state['_cohort_stats'] = {
                        'before': n_before, 'after': n_after, 'excluded': n_before - n_after,
                        'filter_details': cohort_result.get('filter_details', []),
                    }
                    # 🚫 Zero patients after cohort filter — abort export
                    if n_after == 0:
                        lang = st.session_state.get('language', 'en')
                        details_parts = []
                        for fd in cohort_result.get('filter_details', []):
                            label = fd[0] if lang == 'en' else fd[1]
                            details_parts.append(f"{label} (excluded {fd[2]})")
                        details_str = ", ".join(details_parts) if details_parts else ""
                        if lang == 'en':
                            msg = f"No patients meet the cohort criteria. {n_before} candidates were all excluded."
                            if details_str:
                                msg += f" Filters applied: {details_str}."
                            msg += " Please adjust your cohort filters in Step 2 and try again."
                        else:
                            msg = f"没有患者满足队列筛选条件。{n_before} 名候选患者全部被排除。"
                            if details_str:
                                msg += f" 已应用的筛选条件: {details_str}。"
                            msg += " 请在步骤2中调整筛选条件后重试。"
                        st.error(msg)
                        return
                elif candidate_ids is not None:
                    patient_ids_filter = {id_col: candidate_ids}
                    st.session_state['_cohort_stats'] = None
                else:
                    # 全量提取(patient_limit=0): 加载所有ID以支持分批
                    st.session_state['_cohort_stats'] = None
                    try:
                        for f in _get_patient_id_table_files(database):
                            fp = data_path / f
                            if fp.exists():
                                icustays_df = pd.read_parquet(fp, columns=[id_col] if id_col else None)
                                if id_col in icustays_df.columns:
                                    all_ids = sorted(icustays_df[id_col].unique().tolist())
                                    patient_ids_filter = {id_col: all_ids}
                                    break
                    except Exception:
                        pass
            except Exception as _cohort_err:
                print(f"[COHORT] Error in execute_sidebar_export: {_cohort_err}")
                # Fallback: just apply patient_limit without cohort filter
                if patient_limit and patient_limit > 0:
                    try:
                        for f in _get_patient_id_table_files(database):
                            fp = data_path / f
                            if fp.exists():
                                icustays_df = pd.read_parquet(fp, columns=[id_col] if id_col else None)
                                if id_col in icustays_df.columns:
                                    all_ids = icustays_df[id_col].unique().tolist()
                                    sample_ids = _sample_patient_ids_random(all_ids, patient_limit)
                                    patient_ids_filter = {id_col: sample_ids}
                                    break
                    except Exception:
                        pass

            num_patients = len(patient_ids_filter.get(id_col, [])) if patient_ids_filter else None
            parallel_workers, parallel_backend = get_optimal_parallel_config(num_patients, task_type='export')

            # 显示系统资源信息（包含性能层级）
            resources = get_system_resources()
            perf_tier = resources.get('performance_tier', 'unknown')
            # 🔧 FIX: 显示实际使用的 concept_workers=1（串行），而非 recommended_workers
            # 实际加载时 concept_workers=1 避免死锁，不应显示 64 workers 误导用户
            actual_workers = 1  # 与 load_kwargs['concept_workers'] 一致
            tier_emoji = {
                'high-performance': '🚀',
                'server': '💻',
                'workstation': '🖥️',
                'standard': '💻',
                'limited': '⚠️'
            }.get(perf_tier, '💻')

            n_patients_display = num_patients or 'all'
            if lang == 'en':
                perf_msg = f"{tier_emoji} System: {resources['cpu_count']} cores, {resources['total_memory_gb']}GB RAM → optimized full export for {n_patients_display} patients"
            else:
                perf_msg = f"{tier_emoji} 系统: {resources['cpu_count']} 核心, {resources['total_memory_gb']}GB 内存 → 优化全量导出 {n_patients_display} 患者"
            st.info(perf_msg)

            try:
                # 📝 批量加载所有概念（触发宽表批量加载优化）
                data = {}
                failed_concepts = []
                empty_concepts = []  # 🆕 跟踪返回空结果的概念

                # 🚀 优化：先过滤掉当前数据库不支持的概念，避免批量加载失败
                from easyicu.concept import load_dictionary
                cd = load_dictionary(include_sofa2=True)  # 🔧 FIX: 包含 SOFA2 概念字典
                database = st.session_state.get('database', 'eicu')
                valid_concepts = []
                unsupported_concepts = []
                special_concepts_to_load = []  # 🆕 特殊概念（AKI, circ_failure等）

                # 🔧 FIX (2026-02-09): 递归检查概念是否有数据路径
                # 解决 has_callback=True 但无数据源的概念被误判为有效的问题
                def _has_any_source_recursive(concept_name, db, cd_dict, visited=None):
                    """递归检查概念或其子概念是否在目标数据库有数据源"""
                    if visited is None:
                        visited = set()
                    if concept_name in visited:
                        return False
                    visited.add(concept_name)
                    cdef = cd_dict.get(concept_name)
                    if not cdef:
                        return False
                    if cdef.sources.get(db):
                        return True
                    if cdef.sub_concepts:
                        return any(_has_any_source_recursive(sc, db, cd_dict, visited) for sc in cdef.sub_concepts)
                    return False

                # 🔧 使用 concepts_to_export 而不是 selected_concepts（跳过已存在模块的概念）
                for c in concepts_to_export:
                    # 🆕 先检查是否是特殊概念
                    if c in SPECIAL_CONCEPTS:
                        special_concepts_to_load.append(c)
                        continue

                    concept_def = cd.get(c)
                    if concept_def:
                        # 🔧 FIX 2026-02-09: 使用递归检查替代简单的 has_callback 判断
                        # 旧逻辑: has_sources or has_sub_concepts or has_callback → 很多概念有 callback 但无数据源
                        # 新逻辑: 递归检查概念树中是否有至少一个数据源
                        if _has_any_source_recursive(c, database, cd):
                            valid_concepts.append(c)
                        else:
                            unsupported_concepts.append(c)
                    else:
                        unsupported_concepts.append(c)

                # 🔧 FIX: unsupported_concepts 警告移到 failed_concepts 处统一显示，避免重复
                # 这里只记录，不立即显示
                pass  # unsupported_concepts will be merged with failed_concepts later

                if not valid_concepts and not special_concepts_to_load:
                    st.error("❌ 所选概念在当前数据库中都不可用")
                    return

                # � FIX(2026-02-09): 强制 concept_workers=1，避免多线程加载引起的死锁
                # 原因：多线程同时加载概念会导致 DuckDB 连接竞争、缓存失效、
                #       以及 SIC/MIMIC-III 等数据库的复杂回调函数中的线程安全问题
                # 模块级顺序加载已足够高效（每模块 1-5s），无需概念级并行
                smart_concept_workers = 1

                # 🚀 FIX(2026-02-09): 按模块分组加载，实时显示进度
                # 解决 HiRID/SICdb 加载 100+ 概念时用户无法看到进度的问题
                # 按模块分组：保留同组概念的批量优化（宽表、共享子概念缓存）
                module_concept_map = {}  # {module_key: [concepts]}
                concept_to_module = {}
                for mod_key, mod_concepts in CONCEPT_GROUPS_INTERNAL.items():
                    for c in mod_concepts:
                        concept_to_module[c] = mod_key

                for c in valid_concepts:
                    mod = concept_to_module.get(c, '_other')
                    if mod not in module_concept_map:
                        module_concept_map[mod] = []
                    module_concept_map[mod].append(c)

                # 🔧 模块加载优先级：快的模块先加载（给用户更快的反馈）
                MODULE_PRIORITY = [
                    'vitals', 'demographics', 'outcome',        # 快
                    'chemistry', 'hematology', 'blood_gas',     # 中等
                    'medications', 'ventilator', 'respiratory',  # 中等
                    'vasopressors', 'renal', 'neurological',    # 较慢
                    'other_scores', 'circulatory',               # callback
                    'sofa1_score', 'sofa2_score',                # 慢（SOFA计算）
                    'sepsis3_sofa1', 'sepsis3_sofa2', 'sepsis_shared',  # 慢（依赖SOFA）
                    '_other',
                ]
                ordered_modules = []
                for mod in MODULE_PRIORITY:
                    if mod in module_concept_map:
                        ordered_modules.append(mod)
                # 添加遗漏的module
                for mod in module_concept_map:
                    if mod not in ordered_modules:
                        ordered_modules.append(mod)

                import time as _time_mod
                _export_start = _time_mod.time()

                progress_bar.progress(0.15)

                # ⚡ PERF: 启用跨模块缓存复用 — 共享子概念（MAP/GCS/fio2等）无需重复加载
                # ⚠️ 大量患者时禁用缓存，防止内存膨胀
                #   实测: AUMC 23K患者，缓存200MB但Python内存碎片导致RSS增长56GB
                try:
                    from easyicu.api import _get_global_loader
                    _loader = _get_global_loader(database=database, data_path=st.session_state.data_path,
                                                 use_sofa2=True)
                    _actual_n_patients = len(patient_ids_filter.get(id_col, [])) if patient_ids_filter else None
                    # 只对小规模提取启用缓存（<5000患者），大规模提取禁用以控制RSS
                    if _actual_n_patients is not None and _actual_n_patients <= 5000:
                        _loader.concept_resolver._keep_cache_between_calls = True
                    else:
                        _loader.concept_resolver._keep_cache_between_calls = False
                except Exception:
                    _loader = None
                    _actual_n_patients = None

                # 🧠 内存安全: 碎片感知的分批策略
                # 逐模块加载时，每模块的 pandas/numpy 操作在 pymalloc arena 中产生碎片，
                # gc.collect() + malloc_trim() 无法回收（arena 中只要有1个存活对象就不归还OS）。
                # 19个模块依次执行，碎片累积导致 RSS 远超实际数据量（实测: 3GB 数据 → 22GB RSS）。
                #
                # 分批加载可限制每批的碎片量。
                # 流式导出 (2026-04-13): 每批加载 → 合并 → 追加写入 → 释放，
                # 内存始终受限于单批，无需严格上限。
                #   16GB → 10000, 32GB → 12000, 64GB → 24000, 服务器(1.5TB) → 不分批
                _auto_batch_size = None
                _n_load_patients = _actual_n_patients if _actual_n_patients is not None else (
                    len(patient_ids_filter.get(id_col, [])) if patient_ids_filter else None
                )
                # 🔧 2026-05-11: 默认不分批，追求合理内存下的最优速度。
                # 实测：流式导出每模块到峰内存约94k患者约 1–1.3 GB。
                # 仅在可用内存 < 1.5 GB 时才启用 patient-级分批，
                # 评估依据：单模块 DataFrame 约 1 GB + 处理峰唃 500 MB。
                _LOW_MEM_THRESHOLD_MB = 1536  # 1.5 GB
                # 🔧 FIX 2026-05-xx: 对大量患者（>10k）始终启用流式分批。
                # 原因：非流式路径在子进程中一次性向 DuckDB 发出 94k 患者 IN 子句，
                # 对大型原始数据（chartevents 300M+行）极慢甚至卡死；即使内存充足
                # 也应使用流式路径以保持逐批进度可见、避免单次超大查询。
                # 内存充足时批大小取 50000（通常 1-2 批），低内存时取较小值。
                _LARGE_PATIENT_THRESHOLD = 10000
                if _n_load_patients is not None and _n_load_patients > _LARGE_PATIENT_THRESHOLD:
                    try:
                        from easyicu.memory_manager import get_available_memory_mb
                        _avail_mb = get_available_memory_mb()
                        if _avail_mb < _LOW_MEM_THRESHOLD_MB:
                            # 低内存：分批保守（用户已在启动前确认过，此处静默执行）
                            # 每批大小：用可用内存 * 0.6（索留 40% 给 OS + 处理开销）
                            _frag_safe_max = max(20000, int(_avail_mb * 0.6))
                            _auto_batch_size = _frag_safe_max
                            _n_batches = (_n_load_patients + _auto_batch_size - 1) // _auto_batch_size
                            if lang == 'en':
                                st.info(f"🔀 Low memory ({_avail_mb:.0f}MB): "
                                        f"streaming {_n_load_patients} patients in {_n_batches} batches "
                                        f"of {_auto_batch_size}/module.")
                            else:
                                st.info(f"🔀 低内存 ({_avail_mb:.0f}MB): "
                                        f"{_n_load_patients} 患者分 {_n_batches} 批 ({_auto_batch_size}/模块)。")
                        else:
                            # 内存充足：使用大批量流式加载（避免单次超大 IN 子句卡死）
                            _auto_batch_size = 50000
                            _n_batches = (_n_load_patients + _auto_batch_size - 1) // _auto_batch_size
                            if lang == 'en':
                                st.info(f"🚀 Streaming {_n_load_patients} patients in {_n_batches} batch(es) "
                                        f"of {_auto_batch_size} (available memory {_avail_mb:.0f}MB).")
                            else:
                                st.info(f"🚀 流式加载 {_n_load_patients} 患者，分 {_n_batches} 批 "
                                        f"({_auto_batch_size}/批，可用内存 {_avail_mb:.0f}MB)。")
                    except Exception:
                        # 无法检测内存时，保守起见仍然分批
                        _auto_batch_size = 20000

                # 🚀 FIX: 全模块批量加载 + 改善进度提示
                # 测试结果：逐概念加载比批量慢 3-10x（etco2: 873s 逐概念 vs <100s 批量）
                # 原因：每次 load_concepts() 调用后清除 _table_cache，重新从磁盘读取
                # 方案：保持批量加载（最快），但在每个模块前显示预估时间和概念列表

                total_modules = len(ordered_modules)
                # 加上特殊概念模块（如果有）
                _total_steps = total_modules + (1 if special_concepts_to_load else 0)

                # ──────────────────────────────────────────────
                # 🚀 流式导出: 预构建导出相关变量
                # 每个模块加载完后立即合并+写文件+释放，不在内存中累积
                # ──────────────────────────────────────────────
                import time as time_module
                module_times = {}
                _module_load_times = {}  # 记录每模块的加载耗时

                # 预构建 concept -> group 映射（不依赖加载结果）
                _concept_to_group_pre = {}
                _group_priority = list(CONCEPT_GROUPS_INTERNAL.keys())
                for _gk in _group_priority:
                    _gc = CONCEPT_GROUPS_INTERNAL[_gk]
                    for _c in _gc:
                        if _c not in _concept_to_group_pre:
                            _concept_to_group_pre[_c] = _gk

                # 收集导出文件列表和患者ID
                all_exported_patient_ids = set()
                skipped_modules = st.session_state.get('_skipped_modules', set())
                overwrite_modules = st.session_state.get('_overwrite_modules', set())

                # cohort filter: 在 demographics/outcome 模块加载后计算排除列表
                _cohort_exclude_ids = set()
                _cohort_filter_computed = False

                def _process_result(result, concept_names):
                    """处理 load_concepts 返回结果"""
                    if isinstance(result, dict):
                        for cname, df in result.items():
                            if hasattr(df, 'to_pandas'):
                                df = df.to_pandas()
                            elif hasattr(df, 'dataframe'):
                                df = df.dataframe()
                            elif hasattr(df, 'data') and isinstance(df.data, pd.DataFrame):
                                df = df.data

                            if isinstance(df, pd.DataFrame) and len(df) > 0:
                                data[cname] = df
                            elif isinstance(df, pd.Series):
                                data[cname] = df.to_frame().reset_index()
                            else:
                                if cname not in empty_concepts:
                                    empty_concepts.append(cname)
                    elif isinstance(result, pd.DataFrame):
                        if len(result) > 0:
                            for c in concept_names:
                                if c in result.columns:
                                    data[c] = result
                    # 检查空结果
                    for c in concept_names:
                        if c not in data and c not in empty_concepts:
                            empty_concepts.append(c)

                # ──────────────────────────────────────────────
                # 🚀 流式导出: 定义模块合并+导出辅助函数
                # ──────────────────────────────────────────────
                def _export_module_to_disk(group_name, concept_dfs_dict, step_idx, total_steps):
                    """将一个模块的概念合并为宽表并写入磁盘。

                    Args:
                        group_name: 模块key (e.g. 'vitals')
                        concept_dfs_dict: {concept_name: DataFrame} 该模块的数据
                        step_idx: 当前步骤索引（用于进度条）
                        total_steps: 总步骤数
                    Returns:
                        True if exported, False if skipped/empty
                    """
                    nonlocal all_exported_patient_ids

                    if not concept_dfs_dict:
                        return False

                    # 🔧 检查是否已取消
                    if check_cancelled():
                        return False

                    _mod_export_start = time_module.time()

                    # 显示导出进度
                    concept_list = list(concept_dfs_dict.keys())
                    concepts_str = ', '.join(concept_list[:5]) + (f'... +{len(concept_list)-5}' if len(concept_list) > 5 else '')
                    if lang == 'en':
                        _emsg = f"Exporting: {group_name} ({step_idx+1}/{total_steps}) | {concepts_str}"
                    else:
                        _emsg = f"正在导出: {group_name} ({step_idx+1}/{total_steps}) | {concepts_str}"
                    _set_status(_emsg)

                    # ── 合并为宽表（完整保留原有逻辑） ──
                    id_candidates = ['stay_id', 'hadm_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
                    time_candidates = ['time', 'charttime', 'starttime', 'start', 'endtime', 'itemtime', 'datetime', 'Offset', 'measuredat_minutes', 'measuredat', 'givenat', 'enteredentryat', 'intakeoutputoffset', 'observationoffset', 'nursingchartoffset', 'labresultoffset', 'respchartoffset']
                    unified_time_col = 'charttime'

                    # 统一时间列名称
                    normalized_concept_dfs = {}
                    for cname, cdf in concept_dfs_dict.items():
                        cdf = cdf.copy()
                        if unified_time_col in cdf.columns:
                            other_time_cols = [tc for tc in time_candidates if tc in cdf.columns and tc != unified_time_col]
                            if other_time_cols:
                                cdf = cdf.drop(columns=other_time_cols)
                        else:
                            for tc in time_candidates:
                                if tc in cdf.columns:
                                    cdf = cdf.rename(columns={tc: unified_time_col})
                                    other_time_cols = [t for t in time_candidates if t in cdf.columns and t != unified_time_col]
                                    if other_time_cols:
                                        cdf = cdf.drop(columns=other_time_cols)
                                    break
                        normalized_concept_dfs[cname] = cdf
                    concept_dfs_dict = normalized_concept_dfs

                    # 确定主键列
                    merge_cols = []
                    _id_col = None
                    _time_col = None
                    potential_id_cols = set()
                    potential_time_cols = set()
                    for cname, cdf in concept_dfs_dict.items():
                        for col in id_candidates:
                            if col in cdf.columns:
                                potential_id_cols.add(col)
                                break
                        for col in time_candidates:
                            if col in cdf.columns:
                                potential_time_cols.add(col)
                                break
                    for col in id_candidates:
                        if col in potential_id_cols:
                            _id_col = col
                            merge_cols.append(col)
                            break
                    for col in time_candidates:
                        if col in potential_time_cols:
                            _time_col = col
                            merge_cols.append(col)
                            break

                    if not merge_cols:
                        all_dfs = []
                        for cname, cdf in concept_dfs_dict.items():
                            cdf = cdf.copy()
                            cdf['_concept'] = cname
                            all_dfs.append(cdf)
                        merged_df = pd.concat(all_dfs, ignore_index=True)
                    else:
                        all_concept_dfs = []
                        for concept_name, df in concept_dfs_dict.items():
                            if _id_col and _id_col not in df.columns:
                                continue
                            metadata_cols = ['valueuom', 'unit', 'units', 'category', 'type',
                                            'dur_var', 'entertime',
                                            'intakeoutputentryoffset']  # dur_var/entertime: WinTbl; intakeoutputentryoffset: eICU extra
                            cols_to_drop = [c for c in df.columns if c in metadata_cols]
                            if cols_to_drop:
                                df = df.drop(columns=cols_to_drop)
                            value_cols = [c for c in df.columns if c not in merge_cols]
                            df_to_add = df.copy()
                            if len(value_cols) == 1:
                                df_to_add = df_to_add.rename(columns={value_cols[0]: concept_name})
                            elif len(value_cols) > 1:
                                if concept_name in value_cols:
                                    keep_val_cols = [concept_name]
                                else:
                                    keep_val_cols = value_cols
                                cols_to_keep = merge_cols + keep_val_cols
                                df_to_add = df_to_add[[c for c in cols_to_keep if c in df_to_add.columns]]
                                remaining_val_cols = [c for c in df_to_add.columns if c not in merge_cols]
                                if len(remaining_val_cols) == 1 and remaining_val_cols[0] != concept_name:
                                    df_to_add = df_to_add.rename(columns={remaining_val_cols[0]: concept_name})
                                elif len(remaining_val_cols) > 1:
                                    rename_map = {}
                                    for c in remaining_val_cols:
                                        if c != concept_name and not c.startswith(f"{concept_name}_"):
                                            rename_map[c] = f"{concept_name}_{c}"
                                    if rename_map:
                                        df_to_add = df_to_add.rename(columns=rename_map)
                            for mc in merge_cols:
                                if mc not in df_to_add.columns:
                                    if mc in {'charttime', 'time', 'starttime', 'endtime', 'itemtime'}:
                                        df_to_add[mc] = 0.0
                                    else:
                                        df_to_add[mc] = np.nan
                            keep_cols = merge_cols + [c for c in df_to_add.columns if c not in merge_cols]
                            all_concept_dfs.append(df_to_add[keep_cols])

                        if len(all_concept_dfs) == 0:
                            merged_df = None
                        elif len(all_concept_dfs) == 1:
                            merged_df = all_concept_dfs[0]
                        else:
                            time_related_cols = {'charttime', 'time', 'starttime', 'endtime', 'itemtime'}
                            id_related_cols = {'stay_id', 'hadm_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID'}
                            for i, df in enumerate(all_concept_dfs):
                                for col in merge_cols:
                                    if col in df.columns:
                                        col_dtype = df[col].dtype
                                        if col in time_related_cols:
                                            if col_dtype == 'object' or not pd.api.types.is_numeric_dtype(col_dtype):
                                                all_concept_dfs[i][col] = pd.to_numeric(df[col], errors='coerce')
                                        elif col in id_related_cols:
                                            if col_dtype == 'object':
                                                all_concept_dfs[i][col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                                            elif pd.api.types.is_numeric_dtype(col_dtype):
                                                all_concept_dfs[i][col] = df[col].astype('Int64')
                                        else:
                                            if col_dtype == 'object':
                                                all_concept_dfs[i][col] = pd.to_numeric(df[col], errors='coerce')

                            row_counts = [len(df) for df in all_concept_dfs]
                            for i, df in enumerate(all_concept_dfs):
                                for col in merge_cols:
                                    if col in time_related_cols and pd.api.types.is_float_dtype(df[col]):
                                        all_concept_dfs[i][col] = df[col].round(2)

                            total_rows_sum = sum(row_counts)
                            use_fast_path = (total_rows_sum < 2_000_000)

                            if use_fast_path:
                                try:
                                    processed_dfs = []
                                    static_dfs = []
                                    _empty_concepts_local = []
                                    for df in all_concept_dfs:
                                        df_temp = df.copy()
                                        val_cols = [c for c in df_temp.columns if c not in merge_cols]
                                        if not val_cols:
                                            continue
                                        is_static = False
                                        if _time_col and _time_col not in df_temp.columns:
                                            is_static = True
                                        elif _time_col and _time_col in df_temp.columns:
                                            if df_temp[_time_col].isna().all():
                                                is_static = True
                                        if is_static:
                                            if _id_col and _id_col in df_temp.columns:
                                                static_cols = [_id_col] + val_cols
                                                static_df = df_temp[static_cols].drop_duplicates(subset=[_id_col], keep='last')
                                                static_dfs.append(static_df)
                                        else:
                                            df_temp = df_temp.drop_duplicates(subset=merge_cols, keep='last')
                                            for value_col in val_cols:
                                                if len(df_temp) == 0:
                                                    _empty_concepts_local.append(value_col)
                                                    continue
                                                single_val_df = df_temp[merge_cols + [value_col]].copy()
                                                single_val_df['_concept'] = str(value_col)
                                                single_val_df['_value'] = single_val_df[value_col]
                                                single_val_df.drop(columns=[value_col], inplace=True)
                                                processed_dfs.append(single_val_df)

                                    if not processed_dfs and not static_dfs:
                                        merged_df = None
                                    else:
                                        if processed_dfs:
                                            stacked = pd.concat(processed_dfs, ignore_index=True)
                                            merged_df = stacked.pivot_table(
                                                index=merge_cols, columns='_concept',
                                                values='_value', aggfunc='first'
                                            ).reset_index()
                                            for ec in _empty_concepts_local:
                                                if ec not in merged_df.columns:
                                                    merged_df[ec] = np.nan
                                        else:
                                            merged_df = None
                                        if static_dfs:
                                            from functools import reduce
                                            static_merged = reduce(
                                                lambda left, right: pd.merge(left, right, on=_id_col, how='outer'),
                                                static_dfs
                                            )
                                            if merged_df is not None and _id_col in merged_df.columns:
                                                merged_df = pd.merge(merged_df, static_merged, on=_id_col, how='left')
                                            else:
                                                merged_df = static_merged
                                except Exception:
                                    use_fast_path = False

                            if not use_fast_path:
                                if len(all_concept_dfs) > 10:
                                    _batch_sz = 5
                                    batches = []
                                    for i in range(0, len(all_concept_dfs), _batch_sz):
                                        batch = all_concept_dfs[i:i+_batch_sz]
                                        from functools import reduce
                                        try:
                                            batch_merged = reduce(
                                                lambda left, right: pd.merge(left, right, on=merge_cols, how='outer'),
                                                batch
                                            )
                                            if len(batch_merged) > 0:
                                                batch_merged = batch_merged.drop_duplicates(subset=merge_cols)
                                            batches.append(batch_merged)
                                        except Exception:
                                            continue
                                    if not batches:
                                        merged_df = None
                                    else:
                                        merged_df = reduce(
                                            lambda left, right: pd.merge(left, right, on=merge_cols, how='outer'),
                                            batches
                                        )
                                else:
                                    from functools import reduce
                                    merged_df = reduce(
                                        lambda left, right: pd.merge(left, right, on=merge_cols, how='outer'),
                                        all_concept_dfs
                                    )
                                if merged_df is not None and len(merged_df) > 0:
                                    merged_df = merged_df.drop_duplicates(subset=merge_cols)

                    if merged_df is None:
                        if merge_cols:
                            merged_df = pd.DataFrame(columns=merge_cols + list(concept_dfs_dict.keys()))
                        else:
                            return False

                    # 生成文件名
                    concept_names_sorted = sorted(list(concept_dfs_dict.keys()))
                    if len(concept_names_sorted) <= 5:
                        concepts_suffix = '_'.join(concept_names_sorted)
                    else:
                        concepts_suffix = '_'.join(concept_names_sorted[:4]) + f'_etc{len(concept_names_sorted)}'
                    cohort_suffix = _generate_cohort_prefix()
                    if cohort_suffix:
                        safe_filename = f"{group_name}_{concepts_suffix}_{cohort_suffix}".replace('/', '_').replace('\\', '_')
                    else:
                        safe_filename = f"{group_name}_{concepts_suffix}".replace('/', '_').replace('\\', '_')
                    if len(safe_filename) > 150:
                        safe_filename = safe_filename[:150]

                    if export_format == 'csv':
                        file_path = export_dir / f"{safe_filename}.csv"
                    elif export_format == 'parquet':
                        file_path = export_dir / f"{safe_filename}.parquet"
                    elif export_format == 'excel':
                        file_path = export_dir / f"{safe_filename}.xlsx"
                    else:
                        file_path = export_dir / f"{safe_filename}.parquet"

                    # 覆盖模式
                    _ow_modules = st.session_state.get('_overwrite_modules', set())
                    if group_name in _ow_modules or is_viz_import_mode:
                        target_ext = _export_extension_for_format(export_format)
                        for old_file in export_dir.glob(f"{group_name}_*{target_ext}"):
                            try:
                                old_file.unlink()
                            except Exception:
                                pass

                    # 跳过检查
                    if not use_mock and not is_viz_import_mode and file_path.exists():
                        if group_name in skipped_modules:
                            return False

                    # 收集患者ID
                    if merged_df is not None and len(merged_df) > 0:
                        for _idc in ['stay_id', 'hadm_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID']:
                            if _idc in merged_df.columns:
                                all_exported_patient_ids.update(merged_df[_idc].dropna().unique())
                                break

                    # 写入文件
                    if export_format == 'csv':
                        merged_df.to_csv(file_path, index=False, encoding='utf-8-sig')
                    elif export_format == 'parquet':
                        for col in merged_df.columns:
                            if merged_df[col].dtype == object:
                                numeric_vals = pd.to_numeric(merged_df[col], errors='coerce')
                                orig_valid = merged_df[col].notna().sum()
                                if orig_valid > 0 and numeric_vals.notna().sum() >= orig_valid * 0.5:
                                    merged_df[col] = numeric_vals
                                else:
                                    merged_df[col] = merged_df[col].astype(str)
                        merged_df.to_parquet(file_path, index=False)
                    elif export_format == 'excel':
                        merged_df.to_excel(file_path, index=False)
                    else:
                        for col in merged_df.columns:
                            if merged_df[col].dtype == object:
                                numeric_vals = pd.to_numeric(merged_df[col], errors='coerce')
                                orig_valid = merged_df[col].notna().sum()
                                if orig_valid > 0 and numeric_vals.notna().sum() >= orig_valid * 0.5:
                                    merged_df[col] = numeric_vals
                                else:
                                    merged_df[col] = merged_df[col].astype(str)
                        merged_df.to_parquet(file_path, index=False)

                    exported_files.append(str(file_path))
                    _mod_exp_elapsed = time_module.time() - _mod_export_start
                    # 合并加载+导出耗时，反映该模块的真实总耗时
                    _load_time = _module_load_times.get(group_name, 0)
                    module_times[group_name] = _load_time + _mod_exp_elapsed
                    return True

                # ──────────────────────────────────────────────
                # 🚀 流式导出: cohort filter 计算辅助函数
                # ──────────────────────────────────────────────
                _cf_runtime = st.session_state.get('cohort_filter', {}) or {}
                _cohort_filter_modules = {'demographics', 'outcome'}  # 包含 death/los_icu/age/sex 的模块
                _disease_cohort = _cf_runtime.get('disease_cohort')
                if _disease_cohort and _disease_cohort != 'none':
                    _cohort_filter_modules |= DISEASE_COHORT_CONFIG.get(_disease_cohort, {}).get('required_modules', set())

                def _compute_cohort_exclude_ids_inline():
                    """从 data 中计算要排除的患者ID集合。"""
                    cf = st.session_state.get('cohort_filter', {})
                    if not cf:
                        return set()

                    id_col_map = {
                        'miiv': 'stay_id', 'eicu': 'patientunitstayid', 'aumc': 'admissionid',
                        'hirid': 'patientid', 'mimic': 'icustay_id', 'sic': 'CaseID',
                    }
                    _act_id_col = id_col_map.get(database, 'stay_id')
                    id_cands = [_act_id_col, 'stay_id', 'icustay_id', 'hadm_id',
                                'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
                    actual_id = None
                    for df in data.values():
                        if isinstance(df, pd.DataFrame):
                            for c in id_cands:
                                if c in df.columns:
                                    actual_id = c
                                    break
                            if actual_id:
                                break
                    if not actual_id:
                        return set()

                    all_pids = set()
                    for df in data.values():
                        if isinstance(df, pd.DataFrame) and actual_id in df.columns:
                            all_pids.update(df[actual_id].dropna().unique())
                    if not all_pids:
                        return set()

                    excl = set()
                    if cf.get('survived') is not None and 'death' in data:
                        death_df = data['death']
                        if isinstance(death_df, pd.DataFrame) and actual_id in death_df.columns:
                            val_col = 'death' if 'death' in death_df.columns else death_df.columns[-1]
                            death_valid = death_df[death_df[val_col].notna()].copy()
                            death_vals = pd.to_numeric(death_valid[val_col], errors='coerce')
                            died_ids = set(death_valid.loc[death_vals == 1, actual_id].unique())
                            survived_ids = all_pids - died_ids
                            if not cf['survived']:
                                excl |= survived_ids
                            else:
                                excl |= died_ids

                    if cf.get('los_min') is not None and 'los_icu' in data:
                        los_df = data['los_icu']
                        if isinstance(los_df, pd.DataFrame) and actual_id in los_df.columns:
                            val_col = 'los_icu' if 'los_icu' in los_df.columns else los_df.columns[-1]
                            los_valid = los_df[los_df[val_col].notna()].copy()
                            los_hours = pd.to_numeric(los_valid[val_col], errors='coerce') * 24
                            los_ok = set(los_valid.loc[los_hours >= cf['los_min'], actual_id].unique())
                            excl |= (all_pids - los_ok)

                    if (cf.get('age_min') is not None or cf.get('age_max') is not None) and 'age' in data:
                        age_df = data['age']
                        if isinstance(age_df, pd.DataFrame) and actual_id in age_df.columns:
                            val_col = 'age' if 'age' in age_df.columns else age_df.columns[-1]
                            age_valid = age_df[age_df[val_col].notna()].copy()
                            age_vals = pd.to_numeric(age_valid[val_col], errors='coerce')
                            age_mask = pd.Series(True, index=age_valid.index)
                            if cf.get('age_min') is not None:
                                age_mask &= (age_vals >= cf['age_min'])
                            if cf.get('age_max') is not None:
                                age_mask &= (age_vals <= cf['age_max'])
                            age_ok = set(age_valid.loc[age_mask, actual_id].unique())
                            excl |= (all_pids - age_ok)

                    if cf.get('gender') is not None and 'sex' in data:
                        sex_df = data['sex']
                        if isinstance(sex_df, pd.DataFrame) and actual_id in sex_df.columns:
                            val_col = 'sex' if 'sex' in sex_df.columns else sex_df.columns[-1]
                            sex_valid = sex_df[sex_df[val_col].notna()].copy()
                            sex_vals = sex_valid[val_col].astype(str).str.strip().str.upper()
                            target = cf['gender'].upper()
                            if target == 'M':
                                target_variants = {'M', 'MALE', 'MAN', 'MÄNNLICH'}
                            else:
                                target_variants = {'F', 'FEMALE', 'WOMAN', 'WEIBLICH', 'VROUW', 'W'}
                            sex_ok = set(sex_valid.loc[sex_vals.isin(target_variants), actual_id].unique())
                            excl |= (all_pids - sex_ok)

                    disease_cfg = DISEASE_COHORT_CONFIG.get(cf.get('disease_cohort', 'none'))
                    if disease_cfg and disease_cfg.get('concept_priority'):
                        disease_ids = _get_positive_patient_ids_from_data(
                            data,
                            actual_id_col=actual_id,
                            concept_priority=disease_cfg.get('concept_priority', []),
                        )
                        cohort_stats = st.session_state.get('_cohort_stats')
                        if cohort_stats is not None:
                            _label_en = disease_cfg.get('label_en', 'Disease cohort')
                            _label_zh = disease_cfg.get('label_zh', '疾病队列')
                            _removed = len(all_pids - disease_ids)
                            cohort_stats.setdefault('filter_details', []).append((_label_en, _label_zh, _removed))
                            st.session_state['_cohort_stats'] = cohort_stats
                        excl |= (all_pids - disease_ids)

                    if excl:
                        print(f"[COHORT POST-FILTER] Removing {len(excl)}/{len(all_pids)} patients")
                        cohort_stats = st.session_state.get('_cohort_stats')
                        if cohort_stats:
                            cohort_stats['after'] = len(all_pids) - len(excl)
                            cohort_stats['excluded'] = cohort_stats['before'] - (len(all_pids) - len(excl))
                            n_excl = len(excl)
                            detail_label_en = f"Data consistency check: -{n_excl}"
                            detail_label_cn = f"数据一致性检查: -{n_excl}"
                            cohort_stats.setdefault('filter_details', []).append(
                                (detail_label_en, detail_label_cn, n_excl)
                            )
                            st.session_state['_cohort_stats'] = cohort_stats
                    return excl

                def _apply_cohort_filter_to_dfs(concept_dfs_dict, exclude_ids):
                    """对一组概念数据应用 cohort filter，移除 exclude_ids 中的患者。"""
                    if not exclude_ids:
                        return concept_dfs_dict
                    id_cands = ['stay_id', 'hadm_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
                    filtered = {}
                    for cname, df in concept_dfs_dict.items():
                        if isinstance(df, pd.DataFrame):
                            for idc in id_cands:
                                if idc in df.columns:
                                    filtered[cname] = df[~df[idc].isin(exclude_ids)].copy()
                                    break
                            else:
                                filtered[cname] = df
                        else:
                            filtered[cname] = df
                    return filtered

                # 跟踪已处理的模块和等待 cohort filter 的缓冲模块
                _buffered_mod_keys = []  # 等待 cohort filter 计算后再导出的模块
                _loaded_mod_keys = set()  # 已加载的模块keys
                _export_step_counter = 0  # 导出计数器
                # 🔧 FIX: 检查是否有实际设置的 filter 值，而非仅检查 dict 是否非空
                # cohort_filter 初始化为 {'age_min': None, ...}（8 keys 全 None），
                # bool(non_empty_dict) = True 导致永远为 True，所有模块被缓冲
                _cf = st.session_state.get('cohort_filter', {})
                _has_cohort_filter = False
                if _cf:
                    for _k, _v in _cf.items():
                        if _k in {'disease_cohort'} and _v not in (None, '', 'none'):
                            _has_cohort_filter = True
                            break
                        if _k in {'icd_query', 'icd_include_query', 'icd_exclude_query'} and str(_v).strip():
                            _has_cohort_filter = True
                            break
                        if _k not in {'disease_cohort', 'icd_query', 'icd_include_query', 'icd_exclude_query'} and _v is not None:
                            _has_cohort_filter = True
                            break

                # 模块级别耗时累计（用于动态ETA计算）
                _module_elapsed_list = []

                # 🚀 优化: 缓存特殊概念(AKI/CircFailure)的依赖概念 parquet
                # 在主模块循环中保存，避免特殊概念子进程重新从数据库加载
                _special_dep_concepts = {'crea', 'urine', 'weight', 'rrt',
                                         'lact', 'map', 'norepi_rate', 'epi_rate',
                                         'dobu_rate', 'dopa_rate',
                                         'sofa', 'sofa2', 'susp_inf'}  # 🔧 FIX Bug 53: 包含 SOFA/susp_inf 供 Sep3 复用
                _deps_cache_dir = None
                if special_concepts_to_load:
                    import tempfile as _tmpf_deps
                    _deps_cache_dir = _tmpf_deps.mkdtemp(prefix='easyicu_deps_')

                for mod_idx, mod_key in enumerate(ordered_modules):
                    mod_concepts = module_concept_map[mod_key]
                    mod_display = CONCEPT_GROUP_NAMES.get(mod_key, (mod_key, mod_key))
                    mod_name = mod_display[1] if lang != 'en' else mod_display[0]

                    # 计算已用时间和ETA
                    elapsed = _time_mod.time() - _export_start
                    if mod_idx > 0 and _module_elapsed_list:
                        avg_mod_time = sum(_module_elapsed_list) / len(_module_elapsed_list)
                        remaining_mods = total_modules - mod_idx
                        eta_seconds = int(avg_mod_time * remaining_mods)
                        if eta_seconds >= 60:
                            eta_str = f" | ETA ~{eta_seconds // 60}分{eta_seconds % 60:02d}秒" if lang != 'en' else f" | ETA ~{eta_seconds // 60}m{eta_seconds % 60:02d}s"
                        else:
                            eta_str = f" | ETA ~{eta_seconds}秒" if lang != 'en' else f" | ETA ~{eta_seconds}s"
                    else:
                        eta_str = ""

                    # 概念列表（最多显示8个）
                    if len(mod_concepts) <= 8:
                        concept_list_str = ', '.join(mod_concepts)
                    else:
                        concept_list_str = ', '.join(mod_concepts[:6]) + f', ... +{len(mod_concepts)-6}'

                    # 显示模块加载状态
                    elapsed_str = f"{elapsed:.0f}s"
                    if lang == 'en':
                        status_msg = (
                            f"Loading {mod_name} ({len(mod_concepts)} concepts, "
                            f"module {mod_idx+1}/{total_modules}, elapsed {elapsed_str}){eta_str}\n\n"
                            f"{concept_list_str}"
                        )
                    else:
                        status_msg = (
                            f"正在加载 {mod_name} ({len(mod_concepts)} 个概念, "
                            f"模块 {mod_idx+1}/{total_modules}, 已用 {elapsed_str}){eta_str}\n\n"
                            f"{concept_list_str}"
                        )
                    _set_status(status_msg)

                    # 批量加载整个模块
                    # 🔧 FIX (Bug 52): load + merge + export 全部在子进程内完成
                    # 主进程不接触任何 DataFrame，彻底消除 pymalloc arena 碎片累积
                    # 实测: 旧方案 MIIV 94K × 19模块 RSS 661GB; 新方案主进程 RSS 近零增长
                    mod_start = _time_mod.time()

                    _loaded_mod_keys.add(mod_key)

                    # 判断是否需要走旧路径（cohort filter 需要读回数据计算 exclude_ids）
                    _need_buffered_load = (
                        _has_cohort_filter and not _cohort_filter_computed
                        and mod_key in _cohort_filter_modules
                    )

                    if _need_buffered_load:
                        # ── cohort filter 必需模块: 旧路径（读回小量数据） ──
                        try:
                            import multiprocessing as _mp_mod
                            import tempfile as _tmpf_mod
                            import json as _json_mod

                            _tmp_dir = _tmpf_mod.mkdtemp(prefix='easyicu_mod_')

                            _sub_proc = _mp_mod.Process(
                                target=_subprocess_load_module,
                                args=(mod_concepts, database,
                                      st.session_state.data_path,
                                      patient_ids_filter,
                                      _auto_batch_size, _tmp_dir,
                                      _get_sepsis_runtime_options()),
                                daemon=True
                            )
                            _sub_proc.start()

                            _keepalive_tick = 0
                            while _sub_proc.is_alive():
                                if check_cancelled():
                                    _terminate_process_tree(_sub_proc)
                                    _handle_export_cancel()
                                    return
                                _sub_proc.join(timeout=1)
                                _keepalive_tick += 1
                                _mod_elapsed = _time_mod.time() - mod_start
                                _spinner = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏'][_keepalive_tick % 10]
                                _alive_msg = (
                                    f"{_spinner} {'Loading' if lang == 'en' else '正在加载'} {mod_name} "
                                    f"({len(mod_concepts)} {'concepts' if lang == 'en' else '个概念'}, "
                                    f"{'module' if lang == 'en' else '模块'} {mod_idx+1}/{total_modules}, "
                                    f"{'elapsed' if lang == 'en' else '已用'} {_mod_elapsed:.0f}s){eta_str}"
                                )
                                _set_status(_alive_msg)

                            if _sub_proc.exitcode != 0:
                                raise RuntimeError(f"Subprocess exited with code {_sub_proc.exitcode}")

                            _manifest_path = os.path.join(_tmp_dir, '_manifest.json')
                            if os.path.exists(_manifest_path):
                                with open(_manifest_path) as _mf:
                                    _saved = _json_mod.load(_mf)
                                for _cname, _ppath in _saved.items():
                                    data[_cname] = pd.read_parquet(_ppath)

                            for _cname in mod_concepts:
                                if _cname not in data and _cname not in empty_concepts:
                                    empty_concepts.append(_cname)

                            import shutil as _shutil_mod
                            _shutil_mod.rmtree(_tmp_dir, ignore_errors=True)
                        except Exception:
                            try:
                                _shutil_mod.rmtree(_tmp_dir, ignore_errors=True)
                            except Exception:
                                pass
                            if lang == 'en':
                                _set_status(f"⚠️ Subprocess failed for {mod_name}. Falling back to in-process loading...")
                            else:
                                _set_status(f"⚠️ 子进程失败 ({mod_name})，回退至主进程加载...")
                            for _ci, concept in enumerate(mod_concepts):
                                _fb_elapsed = _time_mod.time() - mod_start
                                if lang == 'en':
                                    _set_status(f"⚠️ Fallback {mod_name}: concept {_ci+1}/{len(mod_concepts)} ({concept}), elapsed {_fb_elapsed:.0f}s...")
                                else:
                                    _set_status(f"⚠️ 回退加载 {mod_name}: 概念 {_ci+1}/{len(mod_concepts)} ({concept})，已用 {_fb_elapsed:.0f}s...")
                                try:
                                    result = load_concepts(
                                        data_path=st.session_state.data_path,
                                        database=database, concepts=[concept],
                                        verbose=False, merge=False, concept_workers=1,
                                        **(dict(patient_ids=patient_ids_filter) if patient_ids_filter else {}),
                                        **_get_sepsis_runtime_options(),
                                    )
                                    _process_result(result, [concept])
                                except Exception:
                                    failed_concepts.append(concept)

                        _buffered_mod_keys.append(mod_key)

                        # 检查是否可以计算 cohort filter 了
                        _filter_ready = (
                            _cohort_filter_modules.issubset(_loaded_mod_keys)
                            or mod_idx == len(ordered_modules) - 1
                        )
                        if _filter_ready:
                            _cohort_exclude_ids = _compute_cohort_exclude_ids_inline()
                            _cohort_filter_computed = True

                            # 导出所有缓冲的模块（demographics/outcome 数据量小，主进程处理可接受）
                            for _buf_key in _buffered_mod_keys:
                                _buf_concepts = module_concept_map.get(_buf_key, [])
                                _buf_dfs = {c: data[c] for c in _buf_concepts if c in data}
                                if _buf_dfs:
                                    _buf_dfs = _apply_cohort_filter_to_dfs(_buf_dfs, _cohort_exclude_ids)
                                    try:
                                        _export_module_to_disk(_buf_key, _buf_dfs, _export_step_counter, _total_steps)
                                        _export_step_counter += 1
                                    except Exception as _exp_e:
                                        import traceback as _tb_mod
                                        _tb_mod.print_exc()
                                        st.warning(f"⚠️ Export failed for module '{_buf_key}': {_exp_e}")
                                    for c in _buf_concepts:
                                        data.pop(c, None)
                            _buffered_mod_keys.clear()
                    else:
                        # ── 正常路径: load + merge + export 全部在子进程内完成 ──
                        # 主进程不接触任何 DataFrame，子进程退出后 OS 完整回收所有内存
                        if not _cohort_filter_computed and not _has_cohort_filter:
                            _cohort_filter_computed = True

                        try:
                            import multiprocessing as _mp_mod
                            import json as _json_mod

                            _overwrite_this = (mod_key in overwrite_modules or is_viz_import_mode)

                            _sub_proc = _mp_mod.Process(
                                target=_subprocess_load_and_export_module,
                                args=(mod_concepts, database,
                                      st.session_state.data_path,
                                      patient_ids_filter, _auto_batch_size,
                                      str(export_dir), export_format, mod_key,
                                      list(_cohort_exclude_ids) if _cohort_exclude_ids else None,
                                      _overwrite_this, cohort_suffix,
                                      _special_dep_concepts if _deps_cache_dir else None,
                                      _deps_cache_dir,
                                      _get_sepsis_runtime_options()),
                                daemon=True
                            )
                            _sub_proc.start()

                            _keepalive_tick = 0
                            while _sub_proc.is_alive():
                                if check_cancelled():
                                    _terminate_process_tree(_sub_proc)
                                    _handle_export_cancel()
                                    return
                                _sub_proc.join(timeout=1)
                                _keepalive_tick += 1
                                _mod_elapsed = _time_mod.time() - mod_start
                                _spinner = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏'][_keepalive_tick % 10]
                                _alive_msg = (
                                    f"{_spinner} {'Loading & exporting' if lang == 'en' else '正在加载并导出'} {mod_name} "
                                    f"({len(mod_concepts)} {'concepts' if lang == 'en' else '个概念'}, "
                                    f"{'module' if lang == 'en' else '模块'} {mod_idx+1}/{total_modules}, "
                                    f"{'elapsed' if lang == 'en' else '已用'} {_mod_elapsed:.0f}s){eta_str}"
                                )
                                _set_status(_alive_msg)

                            if _sub_proc.exitcode != 0:
                                raise RuntimeError(f"Subprocess exited with code {_sub_proc.exitcode}")

                            # 读取子进程元数据（不读 DataFrame！主进程零内存增长）
                            _manifest_path = os.path.join(str(export_dir), f'_manifest_{mod_key}.json')
                            if os.path.exists(_manifest_path):
                                with open(_manifest_path) as _mf:
                                    _meta = _json_mod.load(_mf)
                                if _meta.get('exported_file'):
                                    exported_files.append(_meta['exported_file'])
                                    module_times[mod_key] = _time_mod.time() - mod_start
                                if _meta.get('patient_ids'):
                                    all_exported_patient_ids.update(_meta['patient_ids'])
                                for _ec in _meta.get('empty_concepts', []):
                                    if _ec not in empty_concepts:
                                        empty_concepts.append(_ec)
                                _export_step_counter += 1
                                # 清理 manifest 文件
                                try:
                                    os.unlink(_manifest_path)
                                except Exception:
                                    pass
                            else:
                                # 子进程成功但无 manifest → 所有概念为空
                                for _cname in mod_concepts:
                                    if _cname not in empty_concepts:
                                        empty_concepts.append(_cname)

                        except Exception as _sub_e:
                            # 子进程失败 → 逐概念回退（inprocess）
                            import traceback as _tb_mod
                            _tb_mod.print_exc()
                            _sub_exit = getattr(_sub_proc, 'exitcode', None)
                            _fb_err_str = str(_sub_e)[:200]
                            if lang == 'en':
                                _set_status(
                                    f"⚠️ Subprocess failed (exit={_sub_exit}, {_fb_err_str}). "
                                    f"Falling back to in-process loading for {mod_name}...")
                            else:
                                _set_status(
                                    f"⚠️ 子进程失败 (exit={_sub_exit}, {_fb_err_str})。"
                                    f"回退至主进程加载 {mod_name}...")
                            for _ci, concept in enumerate(mod_concepts):
                                _fb_elapsed = _time_mod.time() - mod_start
                                if lang == 'en':
                                    _set_status(
                                        f"⚠️ Fallback loading {mod_name}: "
                                        f"concept {_ci+1}/{len(mod_concepts)} ({concept}), "
                                        f"elapsed {_fb_elapsed:.0f}s...")
                                else:
                                    _set_status(
                                        f"⚠️ 回退加载 {mod_name}: "
                                        f"概念 {_ci+1}/{len(mod_concepts)} ({concept})，"
                                        f"已用 {_fb_elapsed:.0f}s...")
                                try:
                                    result = load_concepts(
                                        data_path=st.session_state.data_path,
                                        database=database, concepts=[concept],
                                        verbose=False, merge=False, concept_workers=1,
                                        **(dict(patient_ids=patient_ids_filter) if patient_ids_filter else {}),
                                        **_get_sepsis_runtime_options(),
                                    )
                                    _process_result(result, [concept])
                                except Exception:
                                    failed_concepts.append(concept)
                            # 回退模式: 用旧方式导出
                            _cur_dfs = {c: data[c] for c in mod_concepts if c in data}
                            if _cur_dfs:
                                _cur_dfs = _apply_cohort_filter_to_dfs(_cur_dfs, _cohort_exclude_ids)
                                try:
                                    _export_module_to_disk(mod_key, _cur_dfs, _export_step_counter, _total_steps)
                                    _export_step_counter += 1
                                except Exception:
                                    pass
                                for c in mod_concepts:
                                    data.pop(c, None)

                    mod_time = _time_mod.time() - mod_start
                    _module_elapsed_list.append(mod_time)
                    if mod_key not in module_times:
                        _module_load_times[mod_key] = mod_time

                    # 更新进度条
                    progress_bar.progress(0.15 + 0.65 * (mod_idx + 1) / _total_steps)

                progress_bar.progress(0.80)

                # 🆕 加载特殊概念（AKI, circ_failure等）— 子进程隔离
                if special_concepts_to_load:
                    special_msg = "Loading special concepts (AKI, CircFailure)..." if lang == 'en' else "正在加载特殊概念 (AKI, 循环衰竭)..."
                    _set_status(special_msg)

                    try:
                        import multiprocessing as _mp_mod
                        import tempfile as _tmpf_mod
                        import json as _json_mod

                        _sp_tmp_dir = _tmpf_mod.mkdtemp(prefix='easyicu_special_')
                        _sp_start = _time_mod.time()

                        # 构建 concept → group 映射，传给子进程做直接导出
                        _sp_concept_to_group = {c: _concept_to_group_pre.get(c, 'special')
                                                for c in special_concepts_to_load}

                        _sp_proc = _mp_mod.Process(
                            target=_subprocess_load_special,
                            args=(special_concepts_to_load, database,
                                  st.session_state.data_path,
                                  patient_ids_filter,
                                  patient_limit if patient_limit and patient_limit > 0 else None,
                                  _sp_tmp_dir, _deps_cache_dir,
                                  str(export_dir), export_format,
                                  list(_cohort_exclude_ids) if _cohort_exclude_ids else None,
                                  _sp_concept_to_group, cohort_suffix,
                                  _get_sepsis_runtime_options()),
                            daemon=True
                        )
                        _sp_proc.start()

                        _sp_tick = 0
                        while _sp_proc.is_alive():
                            if check_cancelled():
                                _terminate_process_tree(_sp_proc)
                                _handle_export_cancel()
                                return
                            _sp_proc.join(timeout=1)
                            _sp_tick += 1
                            _sp_elapsed = _time_mod.time() - _sp_start
                            _sp_spinner = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏'][_sp_tick % 10]
                            if lang == 'en':
                                _sp_msg = f"{_sp_spinner} Loading special concepts (AKI, CircFailure, elapsed {_sp_elapsed:.0f}s)"
                            else:
                                _sp_msg = f"{_sp_spinner} 正在加载特殊概念 (AKI, 循环衰竭, 已用 {_sp_elapsed:.0f}s)"
                            _set_status(_sp_msg)

                        if _sp_proc.exitcode != 0:
                            raise RuntimeError(f"Special concepts subprocess exited with code {_sp_proc.exitcode}")

                        # 读取子进程导出元数据（不读 DataFrame！主进程零内存增长）
                        _sp_export_manifest = os.path.join(_sp_tmp_dir, '_export_manifest.json')
                        _sp_loaded_concepts = []
                        if os.path.exists(_sp_export_manifest):
                            with open(_sp_export_manifest) as _mf:
                                _sp_exports = _json_mod.load(_mf)
                            for _sg_key, _sg_meta in _sp_exports.items():
                                if _sg_meta.get('exported_file'):
                                    exported_files.append(_sg_meta['exported_file'])
                                    _export_step_counter += 1
                                if _sg_meta.get('patient_ids'):
                                    all_exported_patient_ids.update(_sg_meta['patient_ids'])
                                if _sg_meta.get('concepts'):
                                    _sp_loaded_concepts.extend(_sg_meta['concepts'])
                        else:
                            # Fallback: 旧格式 manifest（个别 parquet）
                            _sp_manifest = os.path.join(_sp_tmp_dir, '_manifest.json')
                            if os.path.exists(_sp_manifest):
                                with open(_sp_manifest) as _mf:
                                    _sp_saved = _json_mod.load(_mf)
                                _sp_loaded_concepts = list(_sp_saved.keys())
                            _sp_error_path = os.path.join(_sp_tmp_dir, '_error.txt')
                            if os.path.exists(_sp_error_path):
                                try:
                                    with open(_sp_error_path) as _ef:
                                        _sp_error_head = _ef.readline().strip()
                                    if _sp_error_head:
                                        st.warning(
                                            f"⚠️ Optional derived concepts were skipped: {_sp_error_head}"
                                            if lang == 'en' else
                                            f"⚠️ 可选派生概念已跳过: {_sp_error_head}"
                                        )
                                except Exception:
                                    pass

                        failed_special = [c for c in special_concepts_to_load if c not in _sp_loaded_concepts]
                        failed_concepts.extend(failed_special)

                        _special_load_elapsed = _time_mod.time() - _sp_start

                        import shutil as _shutil_mod
                        _shutil_mod.rmtree(_sp_tmp_dir, ignore_errors=True)
                        if _deps_cache_dir:
                            _shutil_mod.rmtree(_deps_cache_dir, ignore_errors=True)
                            _deps_cache_dir = None

                    except Exception as special_e:
                        try:
                            _shutil_mod.rmtree(_sp_tmp_dir, ignore_errors=True)
                        except Exception:
                            pass
                        try:
                            if _deps_cache_dir:
                                _shutil_mod.rmtree(_deps_cache_dir, ignore_errors=True)
                                _deps_cache_dir = None
                        except Exception:
                            pass
                        st.warning(f"⚠️ Failed to load special concepts: {special_e}" if lang == 'en' else f"⚠️ 加载特殊概念失败: {special_e}")
                        failed_concepts.extend(special_concepts_to_load)

                    progress_bar.progress(0.90)

                # 🚀 流式导出: 导出 data 中剩余的未导出概念（如果有）
                if data:
                    _remaining_by_group = {}
                    for _rc, _rdf in data.items():
                        if isinstance(_rdf, pd.DataFrame):
                            _rg = _concept_to_group_pre.get(_rc, 'other')
                            if _rg not in _remaining_by_group:
                                _remaining_by_group[_rg] = {}
                            _remaining_by_group[_rg][_rc] = _rdf
                    for _rg_key, _rg_dfs in _remaining_by_group.items():
                        _rg_dfs = _apply_cohort_filter_to_dfs(_rg_dfs, _cohort_exclude_ids)
                        try:
                            _export_module_to_disk(_rg_key, _rg_dfs, _export_step_counter, _total_steps)
                            _export_step_counter += 1
                        except Exception as _exp_e:
                            import traceback as _tb_mod
                            _tb_mod.print_exc()
                            st.warning(f"⚠️ Export failed for remaining module '{_rg_key}': {_exp_e}")
                    data.clear()

                # 🔧 分开展示“无当前数据库映射”和“加载/派生失败”，避免把有定义的派生特征误报为不存在。
                unsupported_unique = sorted(set(unsupported_concepts))
                if unsupported_unique:
                    skip_list = ', '.join(unsupported_unique[:5])
                    more_text = f'... +{len(unsupported_unique)-5}' if len(unsupported_unique) > 5 else ''
                    skip_msg = (
                        f"⚠️ {len(unsupported_unique)} selected concepts have no source mapping for {database.upper()}: {skip_list}{more_text}"
                        if lang == 'en' else
                        f"⚠️ {len(unsupported_unique)} 个所选概念暂无 {database.upper()} 数据源映射: {skip_list}{more_text}"
                    )
                    st.warning(skip_msg)

                failed_unique = sorted(set(failed_concepts) - set(unsupported_concepts))
                if failed_unique:
                    fail_list = ', '.join(failed_unique[:5])
                    more_text = f'... +{len(failed_unique)-5}' if len(failed_unique) > 5 else ''
                    fail_msg = (
                        f"⚠️ {len(failed_unique)} concepts were defined but not exported (empty dependencies or derived calculation failed): {fail_list}{more_text}"
                        if lang == 'en' else
                        f"⚠️ {len(failed_unique)} 个概念已定义但未导出（依赖为空或派生计算失败）: {fail_list}{more_text}"
                    )
                    st.warning(fail_msg)

                # 🆕 显示空结果概念提示
                if empty_concepts:
                    empty_list = ', '.join(empty_concepts[:8])
                    more_text = f'... +{len(empty_concepts)-8}' if len(empty_concepts) > 8 else ''
                    empty_msg = f"ℹ️ {len(empty_concepts)} concepts returned empty (not configured or no data): {empty_list}{more_text}" if lang == 'en' else f"ℹ️ {len(empty_concepts)} 个概念返回空结果（未配置或无数据）: {empty_list}{more_text}"
                    st.info(empty_msg)

                # 概念计数：已导出 = exported_files 中的概念数（后面统计）
                _n_loaded_concepts = len(valid_concepts) - len(failed_concepts) - len(unsupported_concepts)
                loaded_msg = f"✅ Loaded & exported {_n_loaded_concepts} concepts" if lang == 'en' else f"✅ 已加载并导出 {_n_loaded_concepts} 个概念"
                _set_status(loaded_msg, level="success")

            except Exception as e:
                import traceback as _tb_mod
                _tb_mod.print_exc()
                warn_msg = f"⚠️ Batch loading failed: {e}" if lang == 'en' else f"⚠️ 批量加载失败: {e}"
                st.warning(warn_msg)
                data = {}

        # 流式导出已在加载循环中完成，无需旧的 Phase 2 导出循环
        import time as time_module
        # 使用真正的开始时间计算总耗时
        export_start_time = _export_start if '_export_start' in dir() else time_module.time()

        if (use_mock or use_loaded_data_export) and data:
            from functools import reduce
            import warnings as _mock_warnings

            def _export_mock_module_to_disk(group_name, concept_dfs_dict):
                """Merge demo-mode concept frames and export them as one module file."""
                nonlocal all_exported_patient_ids

                if not concept_dfs_dict:
                    return False

                id_candidates = ['stay_id', 'hadm_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
                time_candidates = ['time', 'charttime', 'starttime', 'start', 'endtime', 'itemtime', 'datetime', 'Offset', 'measuredat_minutes', 'measuredat', 'givenat', 'enteredentryat', 'intakeoutputoffset', 'observationoffset', 'nursingchartoffset', 'labresultoffset', 'respchartoffset']
                unified_time_col = 'charttime'

                normalized = {}
                for cname, cdf in concept_dfs_dict.items():
                    if not isinstance(cdf, pd.DataFrame) or cdf.empty:
                        continue
                    frame = cdf.copy()
                    if unified_time_col in frame.columns:
                        other_time_cols = [tc for tc in time_candidates if tc in frame.columns and tc != unified_time_col]
                        if other_time_cols:
                            frame = frame.drop(columns=other_time_cols)
                    else:
                        for tc in time_candidates:
                            if tc in frame.columns:
                                frame = frame.rename(columns={tc: unified_time_col})
                                break
                    normalized[cname] = frame

                if not normalized:
                    return False

                potential_id_cols = set()
                potential_time_cols = set()
                for frame in normalized.values():
                    for col in id_candidates:
                        if col in frame.columns:
                            potential_id_cols.add(col)
                            break
                    if unified_time_col in frame.columns:
                        potential_time_cols.add(unified_time_col)

                merge_cols = []
                id_col = next((col for col in id_candidates if col in potential_id_cols), None)
                if id_col:
                    merge_cols.append(id_col)
                time_col = unified_time_col if unified_time_col in potential_time_cols else None
                if time_col:
                    merge_cols.append(time_col)

                static_frames = []
                ts_frames = []
                metadata_cols = {'valueuom', 'unit', 'units', 'category', 'type',
                                 'dur_var', 'entertime',
                                 'intakeoutputentryoffset'}  # dur_var/entertime: WinTbl; intakeoutputentryoffset: eICU extra

                for concept_name, frame in normalized.items():
                    drop_cols = [c for c in frame.columns if c in metadata_cols]
                    if drop_cols:
                        frame = frame.drop(columns=drop_cols)

                    value_cols = [c for c in frame.columns if c not in merge_cols]
                    if not value_cols:
                        continue

                    if len(value_cols) == 1:
                        frame = frame.rename(columns={value_cols[0]: concept_name})
                        value_cols = [concept_name]

                    is_static = (not time_col) or (time_col not in frame.columns) or (time_col in frame.columns and frame[time_col].isna().all())
                    if is_static:
                        if id_col and id_col in frame.columns:
                            static_cols = [id_col] + [c for c in value_cols if c in frame.columns]
                            static_frames.append(frame[static_cols].drop_duplicates(subset=[id_col], keep='last'))
                    else:
                        keep_cols = merge_cols + [c for c in value_cols if c in frame.columns]
                        ts_frames.append(frame[keep_cols].drop_duplicates(subset=merge_cols, keep='last'))

                merged_df = None
                if ts_frames:
                    # 统一 id_col 为 Int64 避免 outer merge 中 int/float 混合警告
                    if id_col:
                        for i, f in enumerate(ts_frames):
                            if id_col in f.columns and f[id_col].dtype != 'Int64':
                                ts_frames[i] = f.copy()
                                ts_frames[i][id_col] = ts_frames[i][id_col].astype('Int64')
                    with _mock_warnings.catch_warnings():
                        _mock_warnings.simplefilter('ignore', UserWarning)
                        merged_df = ts_frames[0] if len(ts_frames) == 1 else reduce(
                            lambda left, right: pd.merge(left, right, on=merge_cols, how='outer'),
                            ts_frames
                        )
                if static_frames:
                    with _mock_warnings.catch_warnings():
                        _mock_warnings.simplefilter('ignore', UserWarning)
                        static_df = static_frames[0] if len(static_frames) == 1 else reduce(
                            lambda left, right: pd.merge(left, right, on=[id_col], how='outer'),
                            static_frames
                        )
                    if merged_df is not None and id_col and id_col in merged_df.columns:
                        merged_df = pd.merge(merged_df, static_df, on=[id_col], how='left')
                    else:
                        merged_df = static_df

                if merged_df is None or merged_df.empty:
                    return False

                if id_col and id_col in merged_df.columns:
                    all_exported_patient_ids.update(merged_df[id_col].dropna().unique())

                concept_names_sorted = sorted(list(concept_dfs_dict.keys()))
                if len(concept_names_sorted) <= 5:
                    concepts_suffix = '_'.join(concept_names_sorted)
                else:
                    concepts_suffix = '_'.join(concept_names_sorted[:4]) + f'_etc{len(concept_names_sorted)}'

                cohort_suffix = _generate_cohort_prefix()
                safe_filename = f"{group_name}_{concepts_suffix}"
                if cohort_suffix:
                    safe_filename = f"{safe_filename}_{cohort_suffix}"
                safe_filename = safe_filename.replace('/', '_').replace('\\', '_')
                if len(safe_filename) > 150:
                    safe_filename = safe_filename[:150]

                if export_format == 'csv':
                    file_path = export_dir / f"{safe_filename}.csv"
                    merged_df.to_csv(file_path, index=False, encoding='utf-8-sig')
                elif export_format == 'excel':
                    file_path = export_dir / f"{safe_filename}.xlsx"
                    merged_df.to_excel(file_path, index=False)
                else:
                    file_path = export_dir / f"{safe_filename}.parquet"
                    merged_df.to_parquet(file_path, index=False)

                exported_files.append(str(file_path))
                return True

            mock_modules = []
            for group_key, group_concepts in selected_modules.items():
                group_dfs = {c: data[c] for c in group_concepts if c in data}
                if group_dfs:
                    mock_modules.append((group_key, group_dfs))

            total_mock_modules = len(mock_modules) or 1
            for mod_idx, (mod_key, mod_dfs) in enumerate(mock_modules):
                if check_cancelled():
                    _handle_export_cancel()
                    return
                mod_display = CONCEPT_GROUP_NAMES.get(mod_key, (mod_key, mod_key))
                mod_name = mod_display[1] if lang != 'en' else mod_display[0]
                _set_status(
                    f"{'Exporting' if lang == 'en' else '正在导出'} {mod_name} "
                    f"({mod_idx + 1}/{total_mock_modules})"
                )
                mod_start = time_module.time()
                _export_mock_module_to_disk(mod_key, mod_dfs)
                module_times[mod_key] = time_module.time() - mod_start
                progress_bar.progress(0.3 + 0.5 * (mod_idx + 1) / total_mock_modules)

        # 完成
        progress_bar.progress(1.0)
        status_text.empty()
        cancel_placeholder.empty()  # 🔧 清理取消按钮

        # ⚡ PERF: 清理跨模块缓存
        try:
            _loader.concept_resolver._keep_cache_between_calls = False
            _loader.concept_resolver._raw_concept_cache.clear()
            _loader.concept_resolver._table_cache.clear()
        except Exception:
            pass

        # 🔧 清理临时状态
        if '_skipped_modules' in st.session_state:
            del st.session_state['_skipped_modules']
        if '_overwrite_modules' in st.session_state:
            del st.session_state['_overwrite_modules']
        st.session_state.pop('_export_conflict_pending', None)
        if '_export_cancelled' in st.session_state:
            del st.session_state['_export_cancelled']
        if '_low_mem_export_confirmed' in st.session_state:
            del st.session_state['_low_mem_export_confirmed']

        if exported_files:
            _prime_export_completion(export_dir, exported_files, auto_load=True)

            # 🆕 保存实际导出的患者数量（从数据中统计，是 cohort filter 后的真实数量）
            actual_patient_count = len(all_exported_patient_ids)
            st.session_state['_exported_patient_count'] = actual_patient_count

            # 🔧 FIX (2026-02-12): 统计实际导出的概念数量
            # 遍历导出的 parquet 文件，收集所有列名，然后规范化去重
            # 这与 load_from_exported() 的统计方式完全一致
            all_exported_columns = set()
            id_cols_set = {'stay_id', 'hadm_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid'}
            time_cols_set = {'time', 'charttime', 'starttime', 'endtime', 'datetime', 'timestamp', 'index'}
            meta_cols_set = {'_concept'}
            exclude_cols_set = id_cols_set | time_cols_set | meta_cols_set

            for file_path in exported_files:
                try:
                    if file_path.endswith('.parquet'):
                        # 🔧 FIX (2026-04-13): 仅读取 schema（列名），不读取数据
                        # 之前用 pd.read_parquet 读取完整数据仅为获取列名，
                        # 对 200K 患者的大 parquet 文件会在主进程产生 GB 级 pymalloc 碎片
                        import pyarrow.parquet as _pq_stat
                        _schema = _pq_stat.read_schema(file_path)
                        _cols = _schema.names
                    elif file_path.endswith('.csv'):
                        # 只读取列名，不读取全部数据
                        _cols = list(pd.read_csv(file_path, nrows=0).columns)
                    elif file_path.endswith('.xlsx'):
                        _cols = list(pd.read_excel(file_path, nrows=0).columns)
                    else:
                        continue
                    for col in _cols:
                        if col not in exclude_cols_set:
                            # 规范化列名
                            norm_col = normalize_column_name(col)
                            all_exported_columns.add(norm_col)
                except Exception:
                    pass  # 忽略读取错误的文件

            exported_concept_count = len(all_exported_columns)

            # 计算被选择但未能提取的概念列表
            # 这不是错误，只是一些概念在当前数据库中不可用
            selected_but_not_exported = []
            selected_concepts_set = set(selected_concepts) if selected_concepts else set()
            for c in selected_concepts_set:
                # 如果概念不在成功导出的列中，则添加到未提取列表
                norm_c = normalize_column_name(c)
                if norm_c not in all_exported_columns:
                    selected_but_not_exported.append(c)

            # 🆕 保存导出结果到 session state，rerun 后在 Guide: Complete 中显示
            total_elapsed = time_module.time() - export_start_time
            st.session_state['_export_success_result'] = {
                'files': exported_files,
                'export_dir': str(export_dir),
                'total_time': total_elapsed,
                'start_time': export_start_time,
                'module_times': module_times.copy(),
                'patient_count': actual_patient_count,  # 🆕 保存实际患者数
                'concept_count': exported_concept_count,  # 🆕 保存实际概念数
                'unavailable_concepts': selected_but_not_exported,  # 🆕 被选择但未能提取的概念
                'unsupported_concepts': unsupported_concepts,  # 🆕 FIX(2026-02-09): 无数据源的概念
                'empty_data_concepts': empty_concepts,  # 🆕 FIX(2026-02-09): 有数据源但无数据的概念
            }
            manifest_paths = _write_export_manifest(
                export_dir,
                exported_files=exported_files,
                patient_count=actual_patient_count,
                concept_count=exported_concept_count,
                export_format=export_format,
                unavailable_concepts=selected_but_not_exported,
                unsupported_concepts=unsupported_concepts,
                empty_data_concepts=empty_concepts,
                failed_concepts=failed_concepts,
            )
            st.session_state['_export_success_result']['manifest_files'] = manifest_paths
            st.rerun()  # 🆕 立即刷新页面，让 Step 4 变为 DONE
        else:
            st.session_state['_exporting_in_progress'] = False  # 🆕 清除导出进行中标记
            no_data_msg = "⚠️ No data was exported" if lang == 'en' else "⚠️ 没有数据被导出"
            st.warning(no_data_msg)

    except Exception as e:
        st.session_state['_exporting_in_progress'] = False  # 🆕 清除导出进行中标记
        fail_msg = f"❌ Export failed: {e}" if lang == 'en' else f"❌ 导出失败: {e}"
        st.error(fail_msg)
