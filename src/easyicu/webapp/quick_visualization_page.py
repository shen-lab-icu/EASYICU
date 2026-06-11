"""Quick visualization page rendering for the EasyICU Streamlit app."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any


_EXPORT_TABLE_SUFFIXES = {".csv", ".parquet", ".xlsx"}


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to this extracted renderer."""
    protected = {'render_quick_visualization_page', "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def _load_demo_review_workspace_from_state() -> tuple[int, int]:
    """Hydrate Quick Viz on explicit user request."""
    demo_patients = int(globals().get("LIGHTWEIGHT_DEMO_PATIENTS", 24))
    demo_hours = int(globals().get("LIGHTWEIGHT_DEMO_HOURS", 24))
    params_getter = globals().get("get_mock_params_with_cohort")
    if callable(params_getter):
        params = dict(params_getter())
    else:
        params = dict(st.session_state.get("mock_params") or {})

    try:
        params["n_patients"] = int(params.get("n_patients") or demo_patients)
    except (TypeError, ValueError):
        params["n_patients"] = demo_patients
    params["n_patients"] = min(max(1, params["n_patients"]), demo_patients)
    try:
        params["hours"] = int(params.get("hours") or demo_hours)
    except (TypeError, ValueError):
        params["hours"] = demo_hours
    params["hours"] = min(max(24, params["hours"]), demo_hours)
    params["demo_profile"] = "lite"

    demo_generator = globals().get("generate_lightweight_demo_data", generate_mock_data)
    mock_data, patient_ids = demo_generator(**params)
    st.session_state.mock_params = params
    st.session_state.loaded_concepts = mock_data
    st.session_state.loaded_data_origin = "demo_viz"
    st.session_state.patient_ids = sorted(patient_ids) if patient_ids else []
    st.session_state.id_col = "stay_id"
    st.session_state.time_col = "time"
    st.session_state.selected_concepts = list(mock_data.keys())
    st.session_state.trigger_export = False
    st.session_state["_exporting_in_progress"] = False
    st.session_state.viz_data_source_mode = "demo"
    return len(mock_data), len(st.session_state.patient_ids)


def _quick_viz_export_file_count(path: str | Path) -> int:
    """Count table files that Quick Visualization can load from a folder."""
    try:
        folder = Path(path).expanduser()
        if not folder.exists() or not folder.is_dir():
            return 0
        return sum(
            1
            for child in folder.iterdir()
            if child.is_file() and child.suffix.lower() in _EXPORT_TABLE_SUFFIXES
        )
    except OSError:
        return 0


def _quick_viz_export_candidates(state: dict[str, Any], *, limit: int = 4) -> list[dict[str, Any]]:
    """Return remembered or discoverable export folders for the review loader."""
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0

    def _add(path_value: object, source: str) -> None:
        if not path_value:
            return
        path = Path(str(path_value)).expanduser()
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        file_count = _quick_viz_export_file_count(path)
        if file_count <= 0:
            return
        candidates.append({
            "path": key,
            "source": source,
            "file_count": file_count,
            "mtime": _mtime(path),
        })

    _add(state.get("viz_confirmed_path"), "confirmed")
    _add(state.get("last_export_dir"), "last")
    _add(state.get("last_export_full_dir"), "last")
    _add(state.get("viz_export_path"), "current")
    _add(state.get("export_path"), "default")

    roots = [
        state.get("export_path"),
        str(Path.home() / "easyicu_export"),
    ]
    for root_value in roots:
        if not root_value:
            continue
        root = Path(str(root_value)).expanduser()
        if not root.exists() or not root.is_dir():
            continue
        _add(root, "default")
        try:
            child_dirs = [child for child in root.iterdir() if child.is_dir()]
        except OSError:
            continue
        for child in sorted(child_dirs, key=_mtime, reverse=True):
            _add(child, "found")

    candidates.sort(key=lambda item: (item["source"] not in {"confirmed", "last"}, -item["mtime"]))
    return candidates[:limit]


def _apply_quick_viz_export_candidate(state: dict[str, Any], path: str) -> None:
    """Fill the exported-data path controls from a recovered candidate."""
    state["viz_export_path"] = path
    state["viz_export_path_input"] = path
    state["_prefer_exported_viz"] = False


def _render_quick_viz_export_path_recovery(lang: str) -> None:
    candidates = _quick_viz_export_candidates(st.session_state)
    default_export_root = str(Path(st.session_state.get("export_path") or Path.home() / "easyicu_export").expanduser())
    is_en = lang == "en"
    title = "Not sure where the export is?" if is_en else "不确定之前导出到哪里了？"
    subtitle = (
        "Use a remembered export folder below, or browse the default EasyICU export root. This scans local folders only; EasyICU does not upload export history to GitHub."
        if is_en else
        "可以直接使用下方记住的导出目录，或从默认 EasyICU 导出根目录开始找。这里只扫描本机目录；EasyICU 不会把导出历史上传到 GitHub。"
    )
    st.markdown(
        '<div class="eu-qv-export-recovery">'
        f'<div class="eu-qv-export-recovery-title">{html.escape(title)}</div>'
        f'<div class="eu-qv-export-recovery-subtitle">{html.escape(subtitle)}</div>'
        f'<code>{html.escape(default_export_root)}</code>'
        '</div>',
        unsafe_allow_html=True,
    )

    if candidates:
        for idx, candidate in enumerate(candidates[:3]):
            path = str(candidate["path"])
            folder_name = Path(path).name or path
            source = str(candidate["source"])
            if source in {"confirmed", "last"}:
                prefix = "Use last export" if is_en else "使用上次导出"
            elif source == "default":
                prefix = "Use default folder" if is_en else "使用默认目录"
            else:
                prefix = "Use recent folder" if is_en else "使用最近目录"
            file_word = "files" if is_en else "个文件"
            label = f"{prefix}: {folder_name} · {candidate['file_count']} {file_word}"
            if st.button(label, key=f"viz_use_export_candidate_{idx}", use_container_width=True, help=path):
                _apply_quick_viz_export_candidate(st.session_state, path)
                st.rerun()
    else:
        fallback_label = "Start from default export folder" if is_en else "从默认导出目录开始"
        if st.button(
            fallback_label,
            key="viz_use_default_export_root",
            use_container_width=True,
            help=default_export_root,
        ):
            _apply_quick_viz_export_candidate(st.session_state, default_export_root)
            st.rerun()


def _quick_viz_workspace_summary(state: dict[str, Any], lang: str) -> dict[str, Any]:
    """Build a serializable summary of the loaded review workspace."""
    loaded_concepts = state.get("loaded_concepts") if isinstance(state.get("loaded_concepts"), dict) else {}
    patient_ids = list(state.get("patient_ids") or [])
    try:
        all_patient_count = int(state.get("all_patient_count") or 0)
    except (TypeError, ValueError):
        all_patient_count = 0
    loaded_patient_count = len(patient_ids) or all_patient_count
    concept_names = sorted(str(name) for name in loaded_concepts)
    origin = str(state.get("loaded_data_origin") or "none")
    source_labels = {
        "demo_viz": ("Demo review workspace", "演示审阅工作区"),
        "exported_files": ("Exported EasyICU tables", "已导出的 EasyICU 表格"),
        "loaded_exports": ("Loaded export workspace", "已加载导出工作区"),
        "quick_load": ("Quick-loaded local data", "快速加载的本地数据"),
        "quick_preview": ("Preview data", "预览数据"),
        "preview": ("Preview data", "预览数据"),
        "real_sofa_reclassification": ("Real SOFA workspace", "真实 SOFA 工作区"),
        "none": ("No data loaded", "尚未加载数据"),
    }
    source_label_en, source_label_zh = source_labels.get(origin, (origin.replace("_", " ").title(), origin))
    if origin in {"preview", "quick_preview"} and (
        state.get("entry_mode") == "demo" or state.get("use_mock_data") or state.get("database") == "mock"
    ):
        source_label_en, source_label_zh = "Demo review workspace", "演示审阅工作区"
    is_en = lang == "en"
    module_count = _quick_viz_loaded_module_count(concept_names)
    return {
        "loaded": bool(concept_names),
        "origin": origin,
        "source_label": source_label_en if is_en else source_label_zh,
        "concept_count": len(concept_names),
        "module_count": module_count,
        "concepts": concept_names,
        "loaded_patient_count": loaded_patient_count,
        "all_patient_count": all_patient_count or loaded_patient_count,
        "active_panel": str(state.get("quick_viz_active_panel") or "Data Tables"),
        "export_path": str(state.get("viz_confirmed_path") or state.get("last_export_dir") or state.get("viz_export_path") or ""),
        "error_count": int(state.get("quick_viz_error_count") or 0),
    }


def _quick_viz_loaded_module_count(concept_names: list[str]) -> int:
    """Count EasyICU concept modules represented by loaded concepts."""
    groups = globals().get("CONCEPT_GROUPS_INTERNAL")
    if isinstance(groups, dict):
        concept_set = set(concept_names)
        module_count = sum(
            1
            for concepts in groups.values()
            if isinstance(concepts, list) and any(str(concept) in concept_set for concept in concepts)
        )
        if module_count:
            return module_count
    return len(concept_names)


def _quick_viz_export_summary_payload(state: dict[str, Any], lang: str) -> bytes:
    """Return a JSON payload for the loaded review workspace summary."""
    summary = _quick_viz_workspace_summary(state, lang)
    payload = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "workspace": summary,
        "notes": (
            "Summary only; patient-level rows are not included."
            if lang == "en" else
            "仅包含摘要；不包含患者级数据行。"
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _quick_viz_reset_review_workspace(state: dict[str, Any]) -> None:
    """Clear the loaded review workspace and return to the setup loader."""
    state["loaded_concepts"] = {}
    state["loaded_data_origin"] = "none"
    state["patient_ids"] = []
    state["all_patient_count"] = 0
    state["selected_patient"] = None
    state["selected_concepts"] = []
    state.pop("available_patient_ids", None)
    state.pop("patient_view_id", None)
    state.pop("quick_viz_active_panel", None)
    state.pop("_review_expected_export_concepts", None)
    state.pop("_review_source_concept_count", None)
    state.pop("_review_subset_concept_count", None)
    state.pop("_review_missing_export_concepts", None)
    state["_scroll_to_top"] = True


def _render_quick_viz_loaded_bar(lang: str) -> None:
    """Render the loaded-workspace status bar and concrete actions."""
    is_en = lang == "en"
    summary = _quick_viz_workspace_summary(st.session_state, lang)
    selected_source_count = int(st.session_state.get("_review_source_concept_count") or summary["concept_count"])
    selection_suffix = ""
    if selected_source_count and selected_source_count != summary["concept_count"]:
        selection_suffix = (
            f" · from {selected_source_count} selected export concepts"
            if is_en else
            f" · 来自 {selected_source_count} 个已选导出概念"
        )
    status_text = (
        f"{summary['loaded_patient_count']} ICU stays · {summary['concept_count']} review features{selection_suffix} · {summary['error_count']} errors"
        if is_en else
        f"{summary['loaded_patient_count']} 个 ICU stay · {summary['concept_count']} 个审阅特征{selection_suffix} · {summary['error_count']} 个错误"
    )
    is_demo_review = str(summary.get("source_label") or "").startswith("Demo review")
    title = "Demo review workspace ready" if is_demo_review and is_en else summary["source_label"]
    if is_demo_review and not is_en:
        title = "演示审阅工作区已就绪"
    with st.container(key="eu_qv_loaded_bar"):
        st.markdown(
            '<div class="eu-qv-loaded-bar">'
            '<div class="eu-qv-loaded-copy">'
            f'<span class="eu-qv-loaded-pill">{html.escape("Loaded" if is_en else "已加载")}</span>'
            f'<b>{html.escape(str(title))}</b>'
            f'<p>{html.escape(status_text)}</p>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        action_cols = st.columns([1, 1, 4])
        with action_cols[0]:
            if st.button(
                "Edit setup" if is_en else "编辑设置",
                key="quick_viz_edit_setup",
                use_container_width=True,
            ):
                _quick_viz_reset_review_workspace(st.session_state)
                st.rerun()
        with action_cols[1]:
            st.download_button(
                "Export" if is_en else "导出",
                data=_quick_viz_export_summary_payload(st.session_state, lang),
                file_name="easyicu_review_workspace_summary.json",
                mime="application/json",
                key="quick_viz_export_summary",
                use_container_width=True,
            )


def _quick_viz_panel_options(lang: str) -> list[tuple[str, str]]:
    labels = {
        "Data Tables": get_text("review_tables"),
        "Time Series": get_text("review_trends"),
        "Patient Overview": get_text("review_patients"),
        "Data Quality": get_text("review_quality"),
    }
    return [(key, labels[key]) for key in labels]


def _render_quick_viz_panel_switcher(lang: str) -> str:
    """Render a lazy panel switcher and return the active panel key.

    Streamlit tabs eagerly execute every tab body on every rerun. The
    Quick Visualization panels are dataframe/chart heavy, so this
    segmented radio keeps the visual four-panel affordance while rendering
    only the selected panel.
    """
    panel_options = _quick_viz_panel_options(lang)
    panel_keys = [key for key, _label in panel_options]
    label_map = dict(panel_options)
    state_key = "quick_viz_active_panel"
    if st.session_state.get(state_key) not in panel_keys:
        st.session_state[state_key] = panel_keys[0]

    switcher_label = "Review panel" if lang == "en" else "审阅面板"
    with st.container(key="qv_panel_switcher"):
        active_panel = st.radio(
            switcher_label,
            options=panel_keys,
            format_func=lambda key: label_map.get(key, key),
            horizontal=True,
            key=state_key,
            label_visibility="collapsed",
        )
    return active_panel


def render_quick_visualization_page(app_context: dict[str, Any] | None = None):
    """渲染快速可视化主页面 - 包含数据加载区域和四个子模块。"""
    if app_context is not None:
        _install_app_context(app_context)
    
    lang = st.session_state.get('language', 'en')
    entry_mode = st.session_state.get('entry_mode', 'none')
    if _sync_quick_viz_screenshot_mode(st.session_state, lang=lang):
        st.rerun()
    screenshot_mode = bool(st.session_state.get('screenshot_mode', False))
    figure_panel = st.session_state.get('_figure_target_panel') if screenshot_mode else None
    direct_figure_panel = figure_panel in {'Data Tables', 'Time Series', 'Patient Overview', 'Data Quality'}

    # Shell-A declutter: the topbar breadcrumb + each subtab's own header
    # already name the page, so there is no separate page header here. The
    # screenshot-mode toggle was removed at the user's request.

    viz_notices = st.session_state.pop('_viz_notices', [])
    for notice in viz_notices[:3]:
        level = str(notice.get('level') or 'info')
        message = str(notice.get('message') or '').strip()
        if message:
            st.markdown(
                f'<div class="compact-inline-notice {level}">{message}</div>',
                unsafe_allow_html=True,
            )

    if screenshot_mode and not direct_figure_panel:
        screenshot_notice = (
            "Figure preset active: compact layout, hidden side chrome, and screenshot-first defaults."
            if lang == 'en'
            else "截图预设已启用：界面更紧凑、隐藏侧边栏干扰，并自动切到更适合论文配图的默认视图。"
        )
        st.markdown(f'<div class="compact-inline-notice info">{screenshot_notice}</div>', unsafe_allow_html=True)

    if 'viz_export_path' not in st.session_state:
        st.session_state.viz_export_path = ""
    recent_export_path = st.session_state.get('viz_confirmed_path') or st.session_state.get('last_export_dir') or ""
    if recent_export_path and (st.session_state.get('_prefer_exported_viz') or not st.session_state.get('viz_export_path')):
        st.session_state.viz_export_path = recent_export_path
        st.session_state['viz_export_path_input'] = recent_export_path
        st.session_state['viz_data_source_mode'] = "exported"
        st.session_state['_prefer_exported_viz'] = False

    auto_viz_request = st.session_state.pop('_viz_auto_load_export', None)
    if auto_viz_request and auto_viz_request.get('path'):
        auto_path = auto_viz_request.get('path')
        if Path(auto_path).exists():
            with st.spinner("Refreshing with newly exported files..." if lang == 'en' else "正在使用最新导出文件刷新..."):
                load_from_exported(
                    auto_path,
                    max_patients=auto_viz_request.get('max_patients', 100),
                    selected_files=auto_viz_request.get('selected_files'),
                )
            st.session_state['_viz_auto_load_notice'] = (
                f"✅ Auto-loaded exported files from `{auto_path}`"
                if lang == 'en' else
                f"✅ 已自动加载最新导出文件：`{auto_path}`"
            )
            recent_export_path = auto_path

    auto_notice = st.session_state.pop('_viz_auto_load_notice', None)
    if auto_notice:
        st.success(auto_notice)

    data_loaded = len(st.session_state.loaded_concepts) > 0
    show_data_loader = not data_loaded
    if show_data_loader:
        with st.container(key="eu_qv_loader"):
            st.markdown(
                '<div class="eu-qv-loader-head">'
                '<div>'
                f'<div class="k">{html.escape("Quick Visualization" if lang == "en" else "快速可视化")}</div>'
                f'<div class="t">{html.escape("Load a review workspace" if lang == "en" else "加载审阅工作区")}</div>'
                f'<div class="s">{html.escape("Start with exported EasyICU tables or generate a compact demo set; review tabs appear immediately after loading." if lang == "en" else "从已导出的 EasyICU 表格开始，或生成一个紧凑演示集；加载后直接进入审阅子页。")}</div>'
                '</div>'
                '<span class="eu-qv-loader-badge">Data Tables · Time Series · Patient · Quality</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            allow_demo = entry_mode != 'real'
            source_options = ["exported"] + (["demo"] if allow_demo else [])
            source_labels = {
                "exported": "Previously exported data" if lang == 'en' else "加载之前导出的结果文件",
                "demo": "Demo data" if lang == 'en' else "模拟数据",
            }
            st.session_state.viz_data_source_mode = _resolve_viz_data_source_mode(
                current_mode=st.session_state.get('viz_data_source_mode'),
                recent_export_path=recent_export_path,
                allow_demo=allow_demo,
                entry_mode=entry_mode,
            )
            current_source = st.radio(
                "Data Source" if lang == 'en' else "数据来源",
                options=source_options,
                format_func=lambda value: source_labels[value],
                horizontal=True,
                key="viz_data_source_mode",
            )

            if current_source == "exported":
                _render_quick_viz_export_path_recovery(lang)
                export_path = _directory_input(
                    "Folder Containing Exported Data Files" if lang == 'en' else "存放导出结果文件的文件夹",
                    value=st.session_state.get('viz_export_path') or recent_export_path,
                    input_key="viz_export_path_input",
                    button_key="viz_export_path_browse",
                    help="Choose the folder that contains EasyICU exported CSV / Parquet / Excel files" if lang == 'en' else "选择存放 EasyICU 导出 CSV / Parquet / Excel 文件的文件夹",
                )
                st.session_state.viz_export_path = export_path

                if export_path:
                    export_dir = Path(export_path)
                    if export_dir.exists() and export_dir.is_dir():
                        available_files = sorted(
                            list(export_dir.glob('*.csv'))
                            + list(export_dir.glob('*.parquet'))
                            + list(export_dir.glob('*.xlsx')),
                            key=lambda path: path.name,
                        )
                        file_names = list(dict.fromkeys(file.stem for file in available_files))

                        if file_names:
                            st.success(
                                f"✅ Found {len(file_names)} data files" if lang == 'en' else f"✅ 发现 {len(file_names)} 个数据文件"
                            )
                            selected_files = st.multiselect(
                                "Select Tables to Load" if lang == 'en' else "选择要加载的表格",
                                options=file_names,
                                default=file_names,
                                key="viz_selected_files",
                            )

                            patient_options = [50, 100, 200, 500, -1]
                            option_labels = {
                                50: "50 (Recommended)" if lang == 'en' else "50 (推荐)",
                                100: "100",
                                200: "200",
                                500: "500 (Slow)" if lang == 'en' else "500 (较慢)",
                                -1: "All (May Lag)" if lang == 'en' else "全部 (可能卡顿)",
                            }
                            max_patients_opt = st.selectbox(
                                "Max ICU stays to load" if lang == 'en' else "最大加载 ICU stay 数",
                                options=patient_options,
                                index=0,
                                format_func=lambda value: option_labels[value],
                                key="viz_max_patients",
                            )
                            max_patients = None if max_patients_opt == -1 else max_patients_opt

                            if selected_files:
                                if st.button(
                                    "Load selected data" if lang == 'en' else "加载所选数据",
                                    type="primary",
                                    use_container_width=True,
                                    key="viz_load_files",
                                ):
                                    with st.spinner("Loading data..." if lang == 'en' else "正在加载数据..."):
                                        load_from_exported(export_path, max_patients=max_patients, selected_files=selected_files)
                                    st.rerun()
                            else:
                                st.warning("⚠️ Please select at least one file" if lang == 'en' else "⚠️ 请至少选择一个文件")
                        else:
                            st.warning(
                                "⚠️ No data files found in this directory (CSV/Parquet/Excel)"
                                if lang == 'en'
                                else "⚠️ 该目录下未找到数据文件 (CSV/Parquet/Excel)"
                            )
                    else:
                        st.error("❌ Directory does not exist" if lang == 'en' else "❌ 目录不存在")

            elif current_source == "demo":
                demo_patients = int(globals().get("LIGHTWEIGHT_DEMO_PATIENTS", 24))
                demo_hours = int(globals().get("LIGHTWEIGHT_DEMO_HOURS", 24))
                _viz_demo_title = (
                    "Generate a lightweight demo review workspace"
                    if lang == 'en' else
                    "生成轻量演示审阅工作区"
                )
                _viz_demo_subtitle = (
                    "Loads a fast core ICU concept set for tables, trends, patient overview, and quality checks."
                    if lang == 'en' else
                    "加载轻量核心 ICU 概念集，用于表格、趋势、患者概览和质量检查。"
                )
                st.markdown(
                    f"""
                    <div class="viz-demo-load-card">
                        <div class="viz-demo-load-kicker">DEMO REVIEW</div>
                        <div class="viz-demo-load-title">{html.escape(_viz_demo_title)}</div>
                        <div class="viz-demo-load-subtitle">{html.escape(_viz_demo_subtitle)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if int(st.session_state.get("viz_demo_patients", demo_patients) or demo_patients) > 48:
                    st.session_state["viz_demo_patients"] = demo_patients
                if int(st.session_state.get("viz_demo_hours", demo_hours) or demo_hours) > 48:
                    st.session_state["viz_demo_hours"] = demo_hours

                col1, col2 = st.columns(2)
                with col1:
                    n_patients = st.slider(
                        "Number of ICU stays" if lang == 'en' else "ICU stay 数量",
                        10,
                        48,
                        demo_patients,
                        key="viz_demo_patients",
                    )
                with col2:
                    hours = st.slider(
                        "Data Duration (hours)" if lang == 'en' else "数据时长(小时)",
                        24,
                        48,
                        demo_hours,
                        key="viz_demo_hours",
                    )

                feature_hint = (
                    "Fast demo profile: core vitals, labs, SOFA/SOFA-2, Sepsis-3, AKI, interventions, demographics, and outcomes."
                    if lang == 'en'
                    else "轻量演示配置：核心生命体征、实验室、SOFA/SOFA-2、Sepsis-3、AKI、干预、人口学和结局。"
                )
                st.caption(feature_hint)

                if st.button(
                    "Generate and load demo workspace" if lang == 'en' else "生成并加载演示工作区",
                    type="primary",
                    use_container_width=True,
                    key="viz_load_demo",
                ):
                    with st.spinner(
                        "Generating lightweight demo data..." if lang == 'en' else "正在生成轻量演示数据..."
                    ):
                        params = get_mock_params_with_cohort()
                        params['n_patients'] = n_patients
                        params['hours'] = hours
                        params['demo_profile'] = 'lite'
                        demo_generator = globals().get("generate_lightweight_demo_data", generate_mock_data)
                        mock_data, patient_ids = demo_generator(**params)
                        st.session_state.loaded_concepts = mock_data
                        st.session_state.loaded_data_origin = 'demo_viz'
                        st.session_state.patient_ids = patient_ids
                        st.session_state.id_col = 'stay_id'
                        st.session_state.time_col = 'time'
                        st.session_state.selected_concepts = list(mock_data.keys())
                    st.rerun()

    if data_loaded:
        if figure_panel in {'Data Tables', 'Time Series', 'Patient Overview', 'Data Quality'}:
            render_quick_figure_panel(figure_panel)
            return

        _render_quick_viz_loaded_bar(lang)
        active_panel = _render_quick_viz_panel_switcher(lang)
        if active_panel == "Data Tables":
            render_data_table_subtab()
        elif active_panel == "Time Series":
            render_timeseries_page()
        elif active_panel == "Patient Overview":
            render_patient_page()
        elif active_panel == "Data Quality":
            render_quality_page()
    else:
        empty_title = "Preview workspace awaits data" if lang == 'en' else "预览工作区等待数据"
        empty_subtitle = (
            "Generate demo data or load exported files above; the review tabs will appear here as a compact multi-view workspace."
            if lang == 'en' else
            "请在上方生成演示数据或加载导出文件；随后这里会显示紧凑的多视角审阅界面。"
        )
        no_data_msg = f"""
        <div class="viz-empty-state">
            <div class="viz-empty-icon">Data</div>
            <div class="viz-empty-title">{html.escape(empty_title)}</div>
            <div class="viz-empty-subtitle">{html.escape(empty_subtitle)}</div>
        </div>
        """
        st.markdown(no_data_msg, unsafe_allow_html=True)
