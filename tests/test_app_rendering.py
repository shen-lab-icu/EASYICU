from __future__ import annotations

import os
import inspect
import json
import re
import sys
import types
from datetime import date
from pathlib import Path

import easyicu
from easyicu.concept import load_dictionary
import easyicu.webapp.app as app
import easyicu.webapp.concept_catalog as concept_catalog
import easyicu.webapp.cohort_dashboard_page as cohort_dashboard_page
import easyicu.webapp.cohort_filters as cohort_filters
import easyicu.webapp.cohort_group_page as cohort_group_page
import easyicu.webapp.cohort_charts as cohort_charts
import easyicu.webapp.cohort_multidb_page as cohort_multidb_page
import easyicu.webapp.cohort_redesign as cohort_redesign
import easyicu.webapp.cohort_severity_page as cohort_severity_page
import easyicu.webapp.cohort_workspace as cohort_workspace
import easyicu.webapp.data_coverage_audit_page as data_coverage_audit_page
import easyicu.webapp.data_table_page as data_table_page
import easyicu.webapp.demo_data as demo_data
import easyicu.webapp.data_paths as data_paths
import easyicu.webapp.data_workflows as data_workflows
import easyicu.webapp.export_reports as export_reports
import easyicu.webapp.export_workflow as export_workflow
import easyicu.webapp.i18n as i18n
import easyicu.webapp.llm_chat as llm_chat
import easyicu.webapp.page_header as page_header
import easyicu.webapp.pages_redesign as pages_redesign
import easyicu.webapp.patient_page as patient_page
import easyicu.webapp.quality_metrics as quality_metrics
import easyicu.webapp.quality_page as quality_page
import easyicu.webapp.quick_visualization_page as quick_visualization_page
import easyicu.webapp.research_agent as research_agent
import easyicu.webapp.sidebar as sidebar
import easyicu.webapp.shell_styles as shell_styles
import easyicu.webapp.styles as styles
import easyicu.webapp.sofa_reclassification as sofa_reclassification
import easyicu.webapp.subprocess_workers as subprocess_workers
import easyicu.webapp.ui_helpers as ui_helpers
import pandas as pd
import pytest


def test_concept_catalog_helpers_remain_available_to_workflow_context() -> None:
    """Data workflow modules receive these names through app.globals()."""
    assert app._get_patient_id_table_files("miiv")[:2] == [
        "icu/icustays.parquet",
        "icustays.parquet",
    ]
    assert app._get_patient_id_table_files("hirid")[0] == "general.parquet"
    assert app._sample_patient_ids_random([3, 1, 2], 10) == [3, 1, 2]


def test_quick_preview_prefers_lightweight_concepts_when_all_features_selected() -> None:
    selected = ["sofa2", "sofa2_liver", "sep3_sofa2", "hr", "map", "temp", "spo2"]

    preview = data_workflows._select_quick_preview_concepts(selected, limit=5)

    assert preview == ["hr", "map", "temp", "spo2", "sofa2"]


def test_cohort_group_feature_modules_offer_full_catalog_with_light_defaults() -> None:
    modules = cohort_group_page._build_cohort_feature_modules("en")

    assert list(modules) == list(concept_catalog.CONCEPT_GROUPS_INTERNAL)
    assert len(modules) == 19
    assert set(cohort_group_page.COHORT_GROUP_DEFAULT_MODULES) < set(modules)
    assert len(cohort_group_page.COHORT_GROUP_DEFAULT_MODULES) < len(modules)
    assert cohort_group_page._cohort_default_feature_modules(modules) == [
        "demographics",
        "outcome",
        "vitals",
        "sepsis3_sofa2",
    ]
    assert cohort_group_page._cohort_feature_load_concepts(
        cohort_group_page._cohort_default_feature_modules(modules),
        modules,
    ) == ["hr", "map", "sbp", "dbp", "pulse_pressure", "temp", "spo2", "resp", "sep3_sofa2"]
    assert len(
        cohort_group_page._cohort_feature_load_concepts(
            cohort_group_page._cohort_default_feature_modules(modules),
            modules,
        )
    ) <= 10
    assert len(
        cohort_group_page._cohort_feature_load_concepts(
            cohort_group_page.COHORT_GROUP_LEGACY_HEAVY_DEFAULT_MODULES,
            modules,
        )
    ) > 50
    assert modules["respiratory"]["features"]
    assert modules["ventilator"]["features"]
    assert modules["vasopressors"]["features"]
    assert modules["medications"]["features"]
    assert modules["renal"]["features"]
    assert modules["neurological"]["features"]
    assert modules["circulatory"]["features"]
    assert modules["other_scores"]["features"]

    assert cohort_group_page._normalize_cohort_feature_modules(
        ["demographic", "lab", "vital", "sofa", "renal"],
        modules,
    ) == ["demographics", "chemistry", "vitals", "sofa2_score", "renal"]


def test_sofa_delta_intro_uses_stable_heading_block() -> None:
    source = Path(cohort_severity_page.__file__).read_text(encoding="utf-8")
    helper = source[
        source.index("def _render_section_heading"):
        source.index("def _render_chart_heading")
    ]
    render_block = source[
        source.index("title = \"SOFA-1 vs SOFA-2 Definition Sensitivity\""):
        source.index("mode_labels = {")
    ]

    assert "subtitle: str | None = None" in helper
    assert "<p>{html.escape(subtitle)}</p>" in helper
    assert "eu-native-section-heading-after" in helper
    assert "aria-hidden=\"true\"" in helper
    assert "subtitle," in render_block
    assert "st.caption(subtitle)" not in render_block


def test_concept_selection_helpers_select_all_by_default(monkeypatch) -> None:
    concept_groups = {
        "Vitals": ["hr", "map"],
        "Labs": ["lactate"],
        "Scores": ["sofa2", "sofa2_resp"],
    }
    state = _AttrSessionState()
    monkeypatch.setattr(sidebar, "st", _SessionStateStreamlit(state))

    sidebar._reset_concepts_to_groups(concept_groups, sidebar._all_concept_groups(concept_groups))

    assert state["selected_groups"] == ["Vitals", "Labs", "Scores"]
    assert state["selected_concepts"] == ["hr", "lactate", "map", "sofa2", "sofa2_resp"]
    assert state["concept_checkboxes"] == {
        "hr": True,
        "map": True,
        "lactate": True,
        "sofa2": True,
        "sofa2_resp": True,
    }
    assert state["step3_confirmed"] is False
    assert state["_eu_concept_defaults_seeded"] is True


def test_concept_selection_design_exposes_all_action() -> None:
    source = Path(sidebar.__file__).read_text(encoding="utf-8")

    assert 'key="concept_select_all_top"' in source
    assert 'key="concept_clear_all_top"' in source
    assert 'key="concept_recommended_design"' in source
    assert 'key="concept_previous_step"' in source
    assert 'key="concept_clear_design"' not in source
    assert '"Select all" if lang == "en" else "全选"' in source
    assert '"Clear" if lang == "en" else "清空"' in source
    assert '"Recommended" if lang == "en" else "推荐"' in source
    assert '"Previous step" if lang == "en" else "上一步"' in source
    assert "_sidebar_set_extract_step_state(st.session_state, 2)" in source
    assert "_all_concept_groups(concept_groups)" in source


def test_step1_source_banner_uses_noninteractive_status_note() -> None:
    source = Path(sidebar.__file__).read_text(encoding="utf-8")
    css_text = shell_styles._load_shell_overrides_css()

    assert '<span class="source-note">' in source
    assert '"Local demo" if lang == "en" else "本机演示"' in source
    assert "preview the extraction flow" in source
    assert "预览提取流程" in source
    assert "token" not in source[
        source.index('<div class="eu-source-banner">'):
        source.index("st.session_state.database = 'mock'")
    ]
    assert "工作目录" not in source[
        source.index('<div class="eu-source-banner">'):
        source.index("st.session_state.database = 'mock'")
    ]
    assert "working directory" not in source[
        source.index('<div class="eu-source-banner">'):
        source.index("st.session_state.database = 'mock'")
    ]
    assert "Learn more" not in source
    assert "了解更多" not in source
    assert ".eu-source-banner .source-note" in css_text
    assert ".eu-source-banner .learn" not in css_text


def test_step2_confirm_seeds_all_concepts_before_rerun() -> None:
    source = Path(sidebar.__file__).read_text(encoding="utf-8")
    confirm_block = source[
        source.index('key="step2_confirm_design"'):
        source.index("with right:\n        _render_cohort_live_preview")
    ]

    assert "_reset_concepts_to_groups(concept_groups, _all_concept_groups(concept_groups))" in confirm_block
    assert confirm_block.index("_reset_concepts_to_groups") < confirm_block.index("st.rerun()")


def test_step2_cohort_preset_is_restorable_and_visible() -> None:
    source = Path(sidebar.__file__).read_text(encoding="utf-8")
    css_text = shell_styles._load_shell_overrides_css()
    filters = {
        "age_min": 45,
        "disease_cohort": "sepsis",
        "icd_include_query": "A41",
        "unexpected": "ignored",
    }

    snapshot = sidebar._snapshot_step2_cohort_filter(filters)
    restored = sidebar._restore_step2_cohort_filter(snapshot)

    assert snapshot == {
        "age_min": 45,
        "age_max": None,
        "first_icu_stay": None,
        "los_min": None,
        "gender": None,
        "survived": None,
        "has_sepsis": None,
        "disease_cohort": "sepsis",
        "icd_query": "",
        "icd_include_query": "A41",
        "icd_exclude_query": "",
        "icd_mode": "include",
    }
    assert restored["age_min"] == 45
    assert restored["disease_cohort"] == "sepsis"
    assert restored["icd_include_query"] == "A41"
    assert "unexpected" not in restored
    assert 'key="cohort_builder_restore_preset"' in source
    assert "_STEP2_SAVED_PRESET_KEY" in source
    assert "eu-cohort-preset-status" in source
    assert "st-key-cohort_builder_restore_preset" in css_text


def test_concept_search_matches_display_metadata_and_units() -> None:
    assert sidebar._concept_matches_search("lact", "lactate") is True
    assert sidebar._concept_matches_search("lact", "mmol") is True
    assert sidebar._concept_matches_search("lact", "乳酸") is True
    assert sidebar._concept_group_matches_search("blood_gas", ["be", "lact"], "lactate") is True


def test_concept_selected_feature_chips_stay_readable() -> None:
    css_text = shell_styles._load_shell_overrides_css()
    chip_match = re.search(
        r"\.stApp \.eu-concept-chip,\n\.stApp \.eu-concept-more \{(?P<body>.*?)\n\}",
        css_text,
        re.S,
    )
    more_matches = re.findall(
        r"\.stApp \.eu-concept-more \{(?P<body>.*?)\n\}",
        css_text,
        re.S,
    )
    assert chip_match is not None
    assert more_matches
    chip_css = chip_match.group("body")
    more_css = more_matches[-1]

    assert "background: var(--ink)" not in chip_css
    assert "background: #eef6f7 !important" in chip_css
    assert "color: #18343a !important" in chip_css
    assert "-webkit-text-fill-color: #18343a !important" in chip_css
    assert "font-size: 12px" in chip_css
    assert "min-height: 22px" in chip_css
    assert "font-weight: 600" in chip_css
    assert "background: #ffffff !important" in more_css
    assert "-webkit-text-fill-color: var(--ink-3) !important" in more_css


def test_concept_checkbox_and_selection_states_do_not_use_black_token_fills() -> None:
    css_text = shell_styles._load_shell_overrides_css()

    checkbox_code_block = re.search(
        r"\.stApp label\[data-baseweb=\"checkbox\"\] code,\n"
        r"\.stApp \.stCheckbox \[data-baseweb=\"checkbox\"\] code,\n"
        r"\.stApp \.stCheckbox code \{(?P<body>.*?)\n\}",
        css_text,
        re.S,
    )
    checkbox_label_block = re.search(
        r"\.stApp label\[data-baseweb=\"checkbox\"\],\n"
        r"\.stApp \.stCheckbox label,\n"
        r"\.stApp \.stCheckbox \[data-baseweb=\"checkbox\"\] \{(?P<body>.*?)\n\}",
        css_text,
        re.S,
    )
    cohort_chip_block = re.search(r"\.eu-cohort-chip \{(?P<body>.*?)\n\}", css_text, re.S)
    concept_module_block = re.search(
        r"\.stApp \[class\*=\"st-key-concept_module_active_\"\] button \{(?P<body>.*?)\n\}",
        css_text,
        re.S,
    )
    final_guard_css = css_text[css_text.index("/* Final contrast guard.") :]
    inline_code_blocks = re.findall(
        r"\.stApp \[data-testid=\"stMarkdownContainer\"\] code,\n"
        r"\.stApp code:not\(pre code\) \{(?P<body>.*?)\n\}",
        final_guard_css,
        re.S,
    )

    assert checkbox_code_block is not None
    assert checkbox_label_block is not None
    assert cohort_chip_block is not None
    assert concept_module_block is not None
    assert inline_code_blocks
    assert "background: transparent !important" in checkbox_code_block.group("body")
    assert "background: transparent !important" in checkbox_label_block.group("body")
    assert "-webkit-text-fill-color: var(--ink-2) !important" in checkbox_label_block.group("body")
    inline_code_body = inline_code_blocks[-1]
    assert "background: #F4F4F0 !important" in inline_code_body
    assert "color: #2E3338 !important" in inline_code_body
    assert "-webkit-text-fill-color: #2E3338 !important" in inline_code_body
    assert ".stApp [data-testid=\"stExpander\"] label *" in final_guard_css
    assert ".stApp [data-testid=\"stExpanderDetails\"] label code" in final_guard_css
    assert ".stApp details label code" in final_guard_css
    assert ".stApp [data-testid=\"stExpanderDetails\"] *::selection" in final_guard_css
    assert ".stApp details .stCheckbox *::selection" in final_guard_css
    assert ".stApp [data-testid=\"stExpanderDetails\"] *::-moz-selection" in final_guard_css
    assert "background: #DDF1F3 !important" in final_guard_css
    assert "-webkit-text-fill-color: #0F1A23 !important" in final_guard_css
    assert "text-shadow: none !important" in final_guard_css
    assert 'label[data-baseweb="checkbox"]:has(input[aria-checked="true"]) > div:first-child' not in css_text
    assert '.stApp label[data-baseweb="checkbox"] span' in css_text
    assert "background: #eef6f7" in cohort_chip_block.group("body")
    assert "color: #18343a" in cohort_chip_block.group("body")
    assert "font-size: 12px" in cohort_chip_block.group("body")
    assert "min-height: 22px" in cohort_chip_block.group("body")
    assert "background: var(--ink)" not in cohort_chip_block.group("body")
    assert "background: var(--surface) !important" in concept_module_block.group("body")
    assert "border: 2px solid var(--ink) !important" in concept_module_block.group("body")
    assert "background: var(--ink)" not in concept_module_block.group("body")


def test_terminate_process_tree_terminates_descendants(monkeypatch) -> None:
    class _FakePsutilError(Exception):
        pass

    class _FakeTimeoutExpired(_FakePsutilError):
        pass

    class _FakeChild:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    class _FakeParent(_FakeChild):
        def __init__(self, child) -> None:
            super().__init__()
            self._child = child

        def children(self, recursive=False):
            assert recursive is True
            return [self._child]

        def wait(self, timeout=None) -> None:
            raise _FakeTimeoutExpired()

    class _FakeProcess:
        pid = 123

        def __init__(self) -> None:
            self.terminated = False
            self.joined = False

        def terminate(self) -> None:
            self.terminated = True

        def join(self, timeout=None) -> None:
            self.joined = True

    child = _FakeChild()
    parent = _FakeParent(child)
    fake_psutil = types.SimpleNamespace(
        Process=lambda _pid: parent,
        wait_procs=lambda children, timeout=None: ([], children),
        Error=_FakePsutilError,
        TimeoutExpired=_FakeTimeoutExpired,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    proc = _FakeProcess()

    export_workflow._terminate_process_tree(proc, timeout=0.01)

    assert child.terminated is True
    assert child.killed is True
    assert proc.terminated is True
    assert proc.joined is True
    assert parent.killed is True


def test_terminate_process_tree_falls_back_to_process_kill(monkeypatch) -> None:
    class _FakeProcess:
        pid = 123

        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.join_count = 0

        def terminate(self) -> None:
            self.terminated = True

        def join(self, timeout=None) -> None:
            self.join_count += 1

        def is_alive(self) -> bool:
            return not self.killed

        def kill(self) -> None:
            self.killed = True

    monkeypatch.setitem(sys.modules, "psutil", None)
    proc = _FakeProcess()

    export_workflow._terminate_process_tree(proc, timeout=0.01)

    assert proc.terminated is True
    assert proc.killed is True
    assert proc.join_count == 2


class _FakeExpander:
    def __init__(self, streamlit_stub) -> None:
        self._streamlit_stub = streamlit_stub

    def __enter__(self):
        if self._streamlit_stub.expander_depth >= 1:
            raise AssertionError("nested expanders are not allowed")
        self._streamlit_stub.expander_depth += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self._streamlit_stub.expander_depth -= 1
        return False


class _FakeStreamlit:
    def __init__(self) -> None:
        self.expander_depth = 0
        self.expander_labels: list[str] = []
        self.dataframe_calls = 0

    def caption(self, *_args, **_kwargs) -> None:
        pass

    def text_input(self, *_args, **_kwargs) -> str:
        return ""

    def markdown(self, *_args, **_kwargs) -> None:
        pass

    def success(self, *_args, **_kwargs) -> None:
        pass

    def warning(self, *_args, **_kwargs) -> None:
        pass

    def dataframe(self, *_args, **_kwargs) -> None:
        if _kwargs.get("width") == "stretch":
            raise TypeError("width must be an integer")
        self.dataframe_calls += 1

    def expander(self, label, **_kwargs):
        self.expander_labels.append(label)
        return _FakeExpander(self)


class _WarningCaptureStreamlit:
    def __init__(self) -> None:
        self.session_state = {"language": "en"}
        self.warnings: list[str] = []

    def warning(self, message, *_args, **_kwargs) -> None:
        self.warnings.append(message)


class _SessionStateStreamlit:
    def __init__(self, session_state) -> None:
        self.session_state = session_state


class _AttrSessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value) -> None:
        self[key] = value


class _FakeColumn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_research_agent_handoff_from_demo_enters_setup(monkeypatch) -> None:
    class _RerunRequested(RuntimeError):
        pass

    class _HandoffStreamlit:
        def __init__(self) -> None:
            self.session_state = {
                "language": "en",
                "entry_mode": "demo",
                "loaded_concepts": {
                    "hr": pd.DataFrame({
                        "stay_id": [1, 1, 2],
                        "charttime": pd.to_datetime([
                            "2026-01-01 00:00",
                            "2026-01-01 01:00",
                            "2026-01-01 00:00",
                        ]),
                        "value": [80, 82, 90],
                    }),
                    "death": pd.DataFrame({
                        "stay_id": [1, 2],
                        "death": [0, 1],
                    }),
                },
                "patient_ids": [1, 2],
                "id_col": "stay_id",
                "research_agent_preflight_confirmed": True,
                "research_agent_preflight_signature": "old",
            }

        def columns(self, _spec):
            return [_FakeColumn(), _FakeColumn()]

        def markdown(self, *_args, **_kwargs) -> None:
            pass

        def button(self, label, **kwargs) -> bool:
            return label == i18n.TEXTS["en"]["ra_handoff_button"] and kwargs["key"] == "ra_handoff_unit"

        def error(self, *_args, **_kwargs) -> None:
            raise AssertionError("handoff should not error")

        def rerun(self) -> None:
            raise _RerunRequested()

    streamlit_stub = _HandoffStreamlit()
    monkeypatch.setattr(app, "st", streamlit_stub)

    with pytest.raises(_RerunRequested):
        app._render_research_agent_handoff("Loaded concepts", "en", key_suffix="unit")

    state = streamlit_stub.session_state
    inbound = state["research_agent_inbound_cohort"]
    assert isinstance(inbound, pd.DataFrame)
    assert len(inbound) == 2
    assert set(inbound.columns) >= {"stay_id", "hr", "death"}
    assert state["research_agent_inbound_cohort_label"] == "Loaded concepts"
    assert state["research_agent_cohort_source"] == i18n.TEXTS["en"]["ra_source_handoff"]
    assert state["_eu_ra_force_setup_from_handoff"] is True
    assert state["_ra_view"] == "setup"
    assert state["research_agent_preflight_confirmed"] is False
    assert "research_agent_preflight_signature" not in state
    assert app._research_agent_handoff_setup_ready(state) is True
    assert "Research Agent setup" in state["_eu_ra_handoff_success_message"]
    assert "_assistant_notice" not in state


def test_directory_input_does_not_redeclare_widget_value_for_existing_state(monkeypatch) -> None:
    class _DirectoryInputStreamlit:
        def __init__(self) -> None:
            self.session_state = {
                "language": "en",
                "sidebar_export_path_input": "/tmp/export",
            }
            self.text_input_kwargs = None

        def columns(self, _spec):
            return [_FakeColumn(), _FakeColumn()]

        def text_input(self, _label, **kwargs) -> str:
            self.text_input_kwargs = kwargs
            return self.session_state[kwargs["key"]]

        def markdown(self, *_args, **_kwargs) -> None:
            pass

        def button(self, *_args, **_kwargs) -> bool:
            return False

    streamlit_stub = _DirectoryInputStreamlit()
    monkeypatch.setattr(data_paths, "st", streamlit_stub)

    result = data_paths._directory_input(
        "Export Path",
        input_key="sidebar_export_path_input",
        button_key="sidebar_export_path_browse",
        value="/tmp/default",
    )

    assert result == "/tmp/export"
    assert "value" not in streamlit_stub.text_input_kwargs


def test_directory_browser_uses_inline_controlled_panel() -> None:
    source = Path(data_paths.__file__).read_text(encoding="utf-8")
    css_text = shell_styles._load_shell_overrides_css()

    assert "@st.dialog" not in source
    assert "server-browser-inline-title" in source
    assert "native dialog \"X\" has no dismissal callback" in source
    assert ".stApp .server-browser-inline-title" in css_text
    assert ".stApp .server-browser-path" in css_text


def test_directory_browser_create_tools_do_not_nest_columns() -> None:
    source = Path(data_paths.__file__).read_text(encoding="utf-8")
    browser_source = source[
        source.index("def _render_directory_browser_dialog"):
        source.index("def _directory_input")
    ]

    assert "create_cols = st.columns" not in browser_source
    assert "tools_col1, tools_col2, tools_col3 = st.columns([1.4, 2.4, 1])" in browser_source


def test_directory_browser_new_folder_placeholder_uses_current_date() -> None:
    source = Path(data_paths.__file__).read_text(encoding="utf-8")

    assert data_paths._example_export_folder_name(date(2026, 5, 26)) == "exports_20260526"
    assert data_paths._new_folder_name_placeholder("en", date(2026, 5, 26)) == "e.g. exports_20260526"
    assert data_paths._new_folder_name_placeholder("zh", date(2026, 5, 26)) == "例如 exports_20260526"
    assert "exports_20260415" not in source
    assert "placeholder=_new_folder_name_placeholder(lang)" in source


def test_directory_browser_buttons_reopen_before_streamlit_guard(monkeypatch, tmp_path) -> None:
    child_dir = tmp_path / "child"
    child_dir.mkdir()

    class _DirectoryBrowserStreamlit:
        def __init__(self) -> None:
            self.session_state = {
                "language": "en",
                "test_browse_open": True,
                "test_browse_cwd": str(child_dir),
            }
            self.button_kwargs = {}

        def caption(self, *_args, **_kwargs) -> None:
            pass

        def markdown(self, *_args, **_kwargs) -> None:
            pass

        def columns(self, spec, **_kwargs):
            return [_FakeColumn() for _ in spec]

        def button(self, _label, **kwargs) -> bool:
            self.button_kwargs[kwargs["key"]] = kwargs
            return False

        def checkbox(self, *_args, **_kwargs) -> bool:
            return False

        def text_input(self, _label, **kwargs) -> str:
            return str(self.session_state.get(kwargs["key"], ""))

        def container(self, *_args, **_kwargs):
            return _FakeColumn()

        def error(self, *_args, **_kwargs) -> None:
            pass

    streamlit_stub = _DirectoryBrowserStreamlit()
    monkeypatch.setattr(data_paths, "st", streamlit_stub)

    data_paths._render_directory_browser_dialog(
        input_key="test_path",
        button_key="test_browse",
        value=str(child_dir),
    )

    assert streamlit_stub.session_state["test_browse_open"] is False

    up_kwargs = streamlit_stub.button_kwargs["test_browse_dlg_up"]
    up_kwargs["on_click"](*up_kwargs["args"])
    assert streamlit_stub.session_state["test_browse_open"] is True
    assert streamlit_stub.session_state["test_browse_cwd"] == str(tmp_path)

    streamlit_stub.session_state["test_browse_open"] = False
    select_kwargs = streamlit_stub.button_kwargs["test_browse_dlg_select"]
    select_kwargs["on_click"](*select_kwargs["args"])
    assert streamlit_stub.session_state["test_path__pending_value"] == str(child_dir)
    assert streamlit_stub.session_state["test_browse_open"] is False


def test_directory_browser_directory_click_prefills_input_candidate(monkeypatch, tmp_path) -> None:
    database_dir = tmp_path / "databases"
    database_dir.mkdir()

    class _DirectoryBrowserStreamlit:
        def __init__(self) -> None:
            self.session_state = {
                "language": "en",
                "test_browse_open": True,
                "test_browse_cwd": str(tmp_path),
            }
            self.buttons_by_label = {}

        def caption(self, *_args, **_kwargs) -> None:
            pass

        def markdown(self, *_args, **_kwargs) -> None:
            pass

        def columns(self, spec, **_kwargs):
            count = spec if isinstance(spec, int) else len(spec)
            return [_FakeColumn() for _ in range(count)]

        def button(self, label, **kwargs) -> bool:
            self.buttons_by_label[label] = kwargs
            return False

        def checkbox(self, *_args, **_kwargs) -> bool:
            return False

        def text_input(self, _label, **kwargs) -> str:
            return str(self.session_state.get(kwargs["key"], ""))

        def container(self, *_args, **_kwargs):
            return _FakeColumn()

        def error(self, *_args, **_kwargs) -> None:
            pass

    streamlit_stub = _DirectoryBrowserStreamlit()
    monkeypatch.setattr(data_paths, "st", streamlit_stub)

    data_paths._render_directory_browser_dialog(
        input_key="test_path",
        button_key="test_browse",
        value="",
    )

    dir_kwargs = streamlit_stub.buttons_by_label["📁 databases"]
    dir_kwargs["on_click"](*dir_kwargs["args"])

    assert streamlit_stub.session_state["test_browse_open"] is True
    assert streamlit_stub.session_state["test_browse_cwd"] == str(database_dir)
    assert streamlit_stub.session_state["test_path__pending_value"] == str(database_dir)


def test_directory_browser_passive_rerun_clears_open_flag(monkeypatch, tmp_path) -> None:
    class _DirectoryBrowserStreamlit:
        def __init__(self) -> None:
            self.session_state = {
                "language": "en",
                "test_browse_open": True,
                "test_browse_cwd": str(tmp_path),
            }

        def caption(self, *_args, **_kwargs) -> None:
            pass

        def markdown(self, *_args, **_kwargs) -> None:
            pass

        def columns(self, spec, **_kwargs):
            return [_FakeColumn() for _ in spec]

        def button(self, *_args, **_kwargs) -> bool:
            return False

        def checkbox(self, *_args, **_kwargs) -> bool:
            return False

        def text_input(self, _label, **kwargs) -> str:
            return str(self.session_state.get(kwargs["key"], ""))

        def container(self, *_args, **_kwargs):
            return _FakeColumn()

        def error(self, *_args, **_kwargs) -> None:
            pass

    streamlit_stub = _DirectoryBrowserStreamlit()
    monkeypatch.setattr(data_paths, "st", streamlit_stub)

    data_paths._render_directory_browser_dialog(
        input_key="test_path",
        button_key="test_browse",
        value=str(tmp_path),
    )

    assert streamlit_stub.session_state["test_browse_open"] is False


def test_directory_browser_navigation_keeps_dialog_open(monkeypatch, tmp_path) -> None:
    class _DirectoryBrowserStreamlit:
        def __init__(self) -> None:
            self.session_state = {
                "language": "en",
                "test_browse_open": True,
                "test_browse_cwd": str(tmp_path),
            }
            self.clicked_up = False

        def caption(self, *_args, **_kwargs) -> None:
            pass

        def markdown(self, *_args, **_kwargs) -> None:
            pass

        def columns(self, spec, **_kwargs):
            return [_FakeColumn() for _ in spec]

        def button(self, _label, **kwargs) -> bool:
            if kwargs.get("key") == "test_browse_dlg_up" and not self.clicked_up:
                self.clicked_up = True
                kwargs["on_click"](*kwargs["args"])
                return True
            return False

        def checkbox(self, *_args, **_kwargs) -> bool:
            return False

        def text_input(self, _label, **kwargs) -> str:
            return str(self.session_state.get(kwargs["key"], ""))

        def container(self, *_args, **_kwargs):
            return _FakeColumn()

        def error(self, *_args, **_kwargs) -> None:
            pass

    streamlit_stub = _DirectoryBrowserStreamlit()
    monkeypatch.setattr(data_paths, "st", streamlit_stub)

    data_paths._render_directory_browser_dialog(
        input_key="test_path",
        button_key="test_browse",
        value=str(tmp_path),
    )

    assert streamlit_stub.session_state["test_browse_open"] is True
    assert streamlit_stub.session_state["test_browse_cwd"] == str(tmp_path.parent)


def test_export_path_default_reseeds_empty_without_overwriting_custom(monkeypatch) -> None:
    class _SidebarStreamlit:
        def __init__(self) -> None:
            self.session_state = {
                "sidebar_export_path_input": "",
            }

    streamlit_stub = _SidebarStreamlit()
    monkeypatch.setattr(sidebar, "st", streamlit_stub)

    sidebar._ensure_default_directory_input_value(
        input_key="sidebar_export_path_input",
        default_key="_sidebar_export_path_default",
        default_value="/tmp/default-a",
    )
    assert streamlit_stub.session_state["sidebar_export_path_input"] == "/tmp/default-a"

    streamlit_stub.session_state["sidebar_export_path_input"] = ""
    sidebar._ensure_default_directory_input_value(
        input_key="sidebar_export_path_input",
        default_key="_sidebar_export_path_default",
        default_value="/tmp/default-a",
    )
    assert streamlit_stub.session_state["sidebar_export_path_input"] == "/tmp/default-a"

    sidebar._ensure_default_directory_input_value(
        input_key="sidebar_export_path_input",
        default_key="_sidebar_export_path_default",
        default_value="/tmp/default-b",
    )
    assert streamlit_stub.session_state["sidebar_export_path_input"] == "/tmp/default-b"

    streamlit_stub.session_state["sidebar_export_path_input"] = "/tmp/custom"
    sidebar._ensure_default_directory_input_value(
        input_key="sidebar_export_path_input",
        default_key="_sidebar_export_path_default",
        default_value="/tmp/default-c",
    )
    assert streamlit_stub.session_state["sidebar_export_path_input"] == "/tmp/custom"


def test_hide_prefilled_directory_text_keeps_user_absolute_path(monkeypatch) -> None:
    class _SidebarStreamlit:
        def __init__(self) -> None:
            self.session_state = {
                "sidebar_data_path_input": "/Users/haibo/.mounty/新加卷/databases/mimic-iv-3.1",
            }

    streamlit_stub = _SidebarStreamlit()
    monkeypatch.setattr(sidebar, "st", streamlit_stub)

    sidebar._hide_prefilled_directory_text(
        "sidebar_data_path_input",
        "/Users/haibo/.mounty/新加卷/databases/eicu",
    )

    assert streamlit_stub.session_state["sidebar_data_path_input"].endswith("mimic-iv-3.1")


def test_hide_prefilled_directory_text_clears_only_mirrored_value(monkeypatch) -> None:
    class _SidebarStreamlit:
        def __init__(self) -> None:
            self.session_state = {
                "sidebar_data_path_input": "/Users/haibo/.mounty/新加卷/databases/eicu",
            }

    streamlit_stub = _SidebarStreamlit()
    monkeypatch.setattr(sidebar, "st", streamlit_stub)

    sidebar._hide_prefilled_directory_text(
        "sidebar_data_path_input",
        "/Users/haibo/.mounty/新加卷/databases/eicu",
    )

    assert streamlit_stub.session_state["sidebar_data_path_input"] == ""


def test_home_dictionary_avoids_nested_expanders(monkeypatch) -> None:
    streamlit_stub = _FakeStreamlit()

    monkeypatch.setattr(app, "st", streamlit_stub)
    monkeypatch.setattr(app, "get_concept_groups", lambda: {"Vitals": ["hr"], "Labs": ["lactate"]})
    monkeypatch.setattr(
        app,
        "CONCEPT_DICTIONARY",
        {"hr": ("Heart Rate", "心率", "bpm"), "lactate": ("Lactate", "乳酸", "mmol/L")},
    )
    monkeypatch.setattr(
        app,
        "CONCEPT_DESCRIPTIONS",
        {"hr": ("Heart rate", "心率"), "lactate": ("Lactate", "乳酸")},
    )
    monkeypatch.setattr(app, "_render_home_dict_table", lambda *_args, **_kwargs: None)

    app.render_home_data_dictionary("en")

    assert streamlit_stub.expander_labels == ["Vitals (1 features)", "Labs (1 features)"]
    assert streamlit_stub.dataframe_calls == 0


def test_guide_card_renders_html_instead_of_literal_div_tags(monkeypatch) -> None:
    class _GuideStreamlit:
        def __init__(self) -> None:
            self.calls = []

        def markdown(self, body, **kwargs) -> None:
            self.calls.append((body, kwargs))

    streamlit_stub = _GuideStreamlit()
    monkeypatch.setattr(ui_helpers, "st", streamlit_stub)

    ui_helpers.render_guide_card(
        "Select Features",
        mini_cards=[ui_helpers.MiniCard("Vital Signs", "HR, BP", "primary")],
        tip="Select by category.",
    )

    body, kwargs = streamlit_stub.calls[-1]
    assert kwargs["unsafe_allow_html"] is True
    assert body.startswith('<div class="app-guide-card')
    assert '<div class="app-mini-grid">' in body


def test_export_report_cohort_prefix_installs_app_context(monkeypatch) -> None:
    streamlit_stub = _SessionStateStreamlit(
        {
            "cohort_enabled": True,
            "cohort_filter": {
                "age_min": 18,
                "age_max": 80,
                "icd_include_query": "A41",
                "icd_exclude_query": "I50",
            },
        }
    )
    monkeypatch.setattr(export_reports, "st", streamlit_stub)

    prefix = export_reports._generate_cohort_prefix(
        {"_split_query_tokens": cohort_filters._split_query_tokens}
    )

    assert prefix == "age18-80_icdInA41_icdExI50"


def test_export_manifest_keeps_installed_context_when_generating_suffix(tmp_path, monkeypatch) -> None:
    streamlit_stub = _SessionStateStreamlit(
        {
            "cohort_enabled": True,
            "cohort_filter": {
                "icd_include_query": "A41",
                "icd_exclude_query": "I50",
            },
            "database": "miiv",
            "entry_mode": "real",
            "selected_concepts": ["hr"],
            "selected_groups": ["vitals"],
        }
    )
    monkeypatch.setattr(export_reports, "st", streamlit_stub)

    context = {
        "_split_query_tokens": cohort_filters._split_query_tokens,
        "_get_sepsis_runtime_options": lambda: {},
    }
    export_reports._write_export_manifest(
        tmp_path,
        exported_files=[],
        patient_count=1,
        concept_count=1,
        export_format="parquet",
        app_context=context,
    )

    manifest = json.loads((tmp_path / "easyicu_export_manifest.json").read_text())
    assert manifest["cohort_suffix"] == "icdInA41_icdExI50"


def test_special_concept_worker_failure_writes_error_manifest(tmp_path, monkeypatch) -> None:
    def _boom(*_args, **_kwargs):
        raise RuntimeError("derived concept failed")

    monkeypatch.setattr(subprocess_workers, "_subprocess_load_special_impl", _boom)

    subprocess_workers._subprocess_load_special(
        ["aki_stage"],
        "miiv",
        "/tmp/missing",
        None,
        None,
        str(tmp_path),
    )

    assert json.loads((tmp_path / "_manifest.json").read_text()) == {}
    assert "derived concept failed" in (tmp_path / "_error.txt").read_text()


def test_special_concept_worker_computes_sep3_after_fallback_load(tmp_path, monkeypatch) -> None:
    def _fake_load_concepts(*, concepts, **_kwargs):
        frames = {
            "susp_inf": pd.DataFrame(
                {"stay_id": [1, 1, 2], "charttime": [0.0, 1.0, 0.0], "susp_inf": [1, 1, 1]}
            ),
            "sofa": pd.DataFrame(
                {"stay_id": [1, 1, 2], "charttime": [0.0, 1.0, 0.0], "sofa": [1, 3, 4]}
            ),
            "sofa2": pd.DataFrame(
                {"stay_id": [1, 1, 2], "charttime": [0.0, 1.0, 0.0], "sofa2": [2, 1, 5]}
            ),
        }
        return {concept: frames[concept] for concept in concepts}

    import easyicu.api as api_module

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    monkeypatch.setattr(api_module, "load_concepts", _fake_load_concepts)

    subprocess_workers._subprocess_load_special_impl(
        ["sep3_sofa1", "sep3_sofa2"],
        "miiv",
        "/tmp/easyicu-test-data",
        {"stay_id": [1, 2]},
        None,
        str(tmp_path),
        None,
        str(export_dir),
        "parquet",
        None,
        {"sep3_sofa1": "sepsis3_sofa1", "sep3_sofa2": "sepsis3_sofa2"},
        "",
        {},
    )

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert set(manifest) == {"sep3_sofa1", "sep3_sofa2"}

    export_manifest = json.loads((tmp_path / "_export_manifest.json").read_text())
    assert export_manifest["sepsis3_sofa1"]["concepts"] == ["sep3_sofa1"]
    assert export_manifest["sepsis3_sofa2"]["concepts"] == ["sep3_sofa2"]

    sep1 = pd.read_parquet(manifest["sep3_sofa1"])
    sep2 = pd.read_parquet(manifest["sep3_sofa2"])
    assert sep1[["stay_id", "charttime"]].to_dict("records") == [
        {"stay_id": 1, "charttime": 1.0},
        {"stay_id": 2, "charttime": 0.0},
    ]
    assert sep2[["stay_id", "charttime"]].to_dict("records") == [
        {"stay_id": 1, "charttime": 0.0},
        {"stay_id": 2, "charttime": 0.0},
    ]


def test_all_web_dictionary_concepts_are_exposed_in_feature_groups() -> None:
    """CONCEPT_GROUPS_INTERNAL is the single source of the feature count
    (`get_all_concepts()` flattens it, and every "N clinical features"
    label across the webapp derives from that). It must stay in exact
    sync with CONCEPT_DICTIONARY: every dictionary concept lands in
    exactly one group, and no group lists a concept absent from the
    dictionary — otherwise the counts shown in different places drift.
    """
    group_lists = list(concept_catalog.CONCEPT_GROUPS_INTERNAL.values())
    grouped_flat = [concept for concepts in group_lists for concept in concepts]
    grouped = set(grouped_flat)
    dictionary = set(concept_catalog.CONCEPT_DICTIONARY)

    assert dictionary - grouped == set(), "dictionary concepts missing from feature groups"
    assert grouped - dictionary == set(), "feature groups list concepts absent from the dictionary"
    assert len(grouped_flat) == len(grouped), "a concept appears in more than one feature group"


def test_mimiciv_dictionary_sources_include_updated_sparse_features() -> None:
    dictionary = load_dictionary(include_sofa2=True)

    tri_ids = dictionary["tri"].sources["miiv"][0].ids
    assert 52642 in tri_ids

    vent_sources = dictionary["vent_end"].sources["miiv"]
    assert {source.table for source in vent_sources} == {"procedureevents", "chartevents"}


def test_dataframe_compat_falls_back_when_width_stretch_is_unsupported(monkeypatch) -> None:
    streamlit_stub = _FakeStreamlit()

    monkeypatch.setattr(app, "st", streamlit_stub)

    app._dataframe_compat([{"Code": "hr"}], width="stretch", hide_index=True)

    assert streamlit_stub.dataframe_calls == 1


def test_width_normalizer_translates_deprecated_container_flag() -> None:
    assert app._normalize_width_kwargs({"use_container_width": True}) == {"width": "stretch"}
    assert app._normalize_width_kwargs({"use_container_width": False}) == {"width": "content"}
    assert app._normalize_width_kwargs({"width": "stretch", "use_container_width": True}) == {"width": "stretch"}


def test_button_compat_uses_width_instead_of_deprecated_container_flag(monkeypatch) -> None:
    class _ButtonStub:
        def __init__(self) -> None:
            self.kwargs = None

        def button(self, _label, **kwargs):
            self.kwargs = kwargs
            return False

    streamlit_stub = _ButtonStub()
    monkeypatch.setattr(app, "st", streamlit_stub)

    app._button_compat("Run", use_container_width=True, key="run")

    assert streamlit_stub.kwargs == {"key": "run", "width": "stretch"}


def test_conversion_wrapper_forwards_to_data_converter(monkeypatch) -> None:
    """The 2026-05-17 converter consolidation removed bucket optimisation and
    the separate HiRID wrapper; convert_csv_to_parquet now forwards just
    source_dir + overwrite to the unified DataConverter-backed impl (HiRID
    archive extraction is handled inside DataConverter.convert_all)."""
    called: dict[str, object] = {}

    def _fake_convert(source_dir, overwrite=False, app_context=None):
        called.update(
            source_dir=source_dir,
            overwrite=overwrite,
            app_context=app_context,
        )
        return 1, 0

    monkeypatch.setattr(app, "_convert_csv_to_parquet_impl", _fake_convert)

    result = app.convert_csv_to_parquet("/tmp/source", overwrite=True)

    assert result == (1, 0)
    assert called["source_dir"] == "/tmp/source"
    assert called["overwrite"] is True
    assert isinstance(called["app_context"], dict)


def test_cohort_query_tokens_expand_icd_ranges_and_mixed_separators() -> None:
    assert app._split_query_tokens("J12-J14, I50; 428\nN17") == [
        "J12",
        "J13",
        "J14",
        "I50",
        "428",
        "N17",
    ]
    assert app._split_query_tokens("J12-K14") == ["J12-K14"]


def test_supported_disease_cohorts_only_expose_icd_templates_for_icd_databases() -> None:
    miiv_cohorts = set(app._get_supported_disease_cohorts("miiv"))
    aumc_cohorts = set(app._get_supported_disease_cohorts("aumc"))

    assert {"ards", "pneumonia", "heart_failure", "ami", "stroke"}.issubset(miiv_cohorts)
    assert {"sepsis", "aki", "rrt"}.issubset(aumc_cohorts)
    assert "ards" not in aumc_cohorts


def test_death_stay_picker_assigns_multi_stay_death_to_matching_or_last_stay() -> None:
    merged = pd.DataFrame(
        {
            "hadm_id": [10, 10, 20, 20],
            "stay_id": [101, 102, 201, 202],
            "hospital_expire_flag": [1, 1, 1, 1],
            "deathtime": [
                "2024-01-03 08:00",
                "2024-01-03 08:00",
                "2024-02-05 12:00",
                "2024-02-05 12:00",
            ],
            "intime": [
                "2024-01-01 00:00",
                "2024-01-03 00:00",
                "2024-02-01 00:00",
                "2024-02-03 00:00",
            ],
            "outtime": [
                "2024-01-02 00:00",
                "2024-01-04 00:00",
                "2024-02-02 00:00",
                "2024-02-04 00:00",
            ],
        }
    )

    picked = app._pick_death_stay(
        merged,
        merged["hospital_expire_flag"] == 1,
        "stay_id",
        "deathtime",
        "intime",
        "outtime",
    )

    assert picked == {102, 202}


def test_miiv_death_series_marks_only_the_attributed_icu_stay() -> None:
    icu_df = pd.DataFrame(
        {
            "subject_id": [1, 1],
            "hadm_id": [10, 10],
            "stay_id": [101, 102],
            "intime": ["2024-01-01 00:00", "2024-01-03 00:00"],
            "outtime": ["2024-01-02 00:00", "2024-01-04 00:00"],
        }
    )
    admission_df = pd.DataFrame(
        {
            "hadm_id": [10],
            "hospital_expire_flag": [1],
            "deathtime": ["2024-01-03 08:00"],
        }
    )

    death = app._get_death_series(icu_df, "miiv", None, admission_df, "stay_id", "subject_id")

    assert death.tolist() == [False, True]


def test_eicu_death_series_prefers_hospital_status_over_unit_status() -> None:
    icu_df = pd.DataFrame(
        {
            "patientunitstayid": [1, 2],
            "hospitaldischargestatus": ["Alive", "Expired"],
            "unitdischargestatus": ["Expired", "Alive"],
        }
    )

    death = app._get_death_series(
        icu_df,
        "eicu",
        None,
        None,
        "patientunitstayid",
        "uniquepid",
    )

    assert death.tolist() == [False, True]


def test_static_cohort_series_normalize_database_specific_age_los_and_sex() -> None:
    mimic_icu = pd.DataFrame(
        {
            "subject_id": [1],
            "icustay_id": [11],
            "intime": ["2020-01-01"],
        }
    )
    mimic_patients = pd.DataFrame({"subject_id": [1], "dob": ["1900-01-01"]})
    sic_icu = pd.DataFrame({"CaseID": [1, 2], "TimeOfStay": [7200, 10800], "Sex": [735, 736]})

    mimic_age = app._get_age_series(
        mimic_icu,
        "mimic",
        mimic_patients,
        None,
        "icustay_id",
        "subject_id",
    )
    sic_los = app._get_los_hours_series(sic_icu, "sic")
    sic_sex = app._get_sex_series(sic_icu, "sic", None, "CaseID", "subject_id")

    assert mimic_age.tolist() == [90.0]
    assert sic_los.tolist() == [2.0, 3.0]
    assert sic_sex.tolist() == ["M", "F"]


def test_post_filter_cohort_data_applies_disease_concept_and_updates_stats(monkeypatch) -> None:
    streamlit_stub = _SessionStateStreamlit(
        {
            "language": "en",
            "cohort_filter": {"disease_cohort": "aki"},
            "_cohort_stats": {"before": 3, "after": 3, "excluded": 0, "filter_details": []},
        }
    )
    data = {
        "aki": pd.DataFrame({"stay_id": [1, 2, 3], "aki": [0, 1, 1]}),
        "hr": pd.DataFrame({"stay_id": [1, 2, 3], "hr": [80, 90, 100]}),
    }

    monkeypatch.setattr(cohort_filters, "st", streamlit_stub)

    filtered = app._post_filter_cohort_data(data, "miiv")

    assert filtered["aki"]["stay_id"].tolist() == [2, 3]
    assert filtered["hr"]["stay_id"].tolist() == [2, 3]
    assert streamlit_stub.session_state["_cohort_stats"]["after"] == 2
    assert streamlit_stub.session_state["_cohort_stats"]["excluded"] == 1


def test_feature_definition_panel_defaults_to_collapsed(monkeypatch) -> None:
    import easyicu.webapp.data_dictionary_page as data_dictionary_page

    class _FeaturePanelStreamlit:
        def __init__(self) -> None:
            self.session_state = {
                "step3_confirmed": True,
                "selected_concepts": ["hr"],
                "database": "miiv",
            }
            self.expander_kwargs = None

        def expander(self, label, **kwargs):
            self.expander_kwargs = {"label": label, **kwargs}
            return _FakeExpander(_FakeStreamlit())

        def caption(self, *_args, **_kwargs) -> None:
            pass

        def info(self, *_args, **_kwargs) -> None:
            pass

        def download_button(self, *_args, **_kwargs) -> None:
            pass

        def dataframe(self, *_args, **_kwargs) -> None:
            pass

    streamlit_stub = _FeaturePanelStreamlit()

    monkeypatch.setattr(data_dictionary_page, "st", streamlit_stub)
    monkeypatch.setattr(
        data_dictionary_page,
        "_get_feature_definition_rows",
        lambda *_args, **_kwargs: [{"Feature": "hr", "Table": "chartevents"}],
    )

    data_dictionary_page._render_feature_definition_panel("en")

    assert streamlit_stub.expander_kwargs == {
        "label": "🧬 Feature Definition Transparency",
        "expanded": False,
    }


def test_preview_icd_match_counts_hadm_level_matches_across_icu_stays(tmp_path) -> None:
    pd.DataFrame(
        {
            "stay_id": [101, 102, 201],
            "hadm_id": [10, 10, 20],
        }
    ).to_parquet(tmp_path / "icustays.parquet")
    pd.DataFrame(
        {
            "hadm_id": [10, 20],
            "icd_code": ["J12.0", "I50.0"],
            "icd_version": [10, 10],
        }
    ).to_parquet(tmp_path / "diagnoses_icd.parquet")
    pd.DataFrame(
        {
            "icd_code": ["J120", "I500"],
            "long_title": ["Viral pneumonia", "Heart failure"],
        }
    ).to_parquet(tmp_path / "d_icd_diagnoses.parquet")

    result = app._preview_icd_match(tmp_path, "miiv", ["J12"])

    assert result["error"] is None
    assert result["total_patients"] == 3
    assert result["matched_patients"] == 2
    assert result["matched_ids"] == [101, 102]
    assert result["top_codes"].iloc[0].to_dict() == {
        "ICD Code": "J120",
        "Count": 1,
        "Description": "Viral pneumonia",
    }


def test_preview_icd_match_supports_mimiciv_hosp_icu_layout(tmp_path) -> None:
    (tmp_path / "icu").mkdir()
    (tmp_path / "hosp").mkdir()
    pd.DataFrame(
        {
            "stay_id": [101, 102, 201],
            "subject_id": [1, 1, 2],
            "hadm_id": [10, 10, 20],
        }
    ).to_parquet(tmp_path / "icu" / "icustays.parquet")
    pd.DataFrame(
        {
            "hadm_id": [10, 20],
            "icd_code": ["A41.9", "I50.0"],
            "icd_version": [10, 10],
        }
    ).to_parquet(tmp_path / "hosp" / "diagnoses_icd.parquet")
    pd.DataFrame(
        {
            "icd_code": ["A419", "I500"],
            "long_title": ["Sepsis, unspecified organism", "Heart failure"],
        }
    ).to_parquet(tmp_path / "hosp" / "d_icd_diagnoses.parquet")

    result = app._preview_icd_match(tmp_path, "miiv", ["A41"])

    assert result["error"] is None
    assert result["total_patients"] == 3
    assert result["matched_patients"] == 2
    assert result["matched_ids"] == [101, 102]
    assert result["top_codes"].iloc[0].to_dict() == {
        "ICD Code": "A419",
        "Count": 1,
        "Description": "Sepsis, unspecified organism",
    }


def test_apply_cohort_filter_supports_mimiciv_hosp_icu_layout(tmp_path, monkeypatch) -> None:
    (tmp_path / "icu").mkdir()
    (tmp_path / "hosp").mkdir()
    pd.DataFrame(
        {
            "stay_id": [101, 102, 201],
            "subject_id": [1, 1, 2],
            "hadm_id": [10, 10, 20],
            "intime": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "outtime": ["2020-01-03", "2020-02-03", "2020-03-03"],
            "los": [2.0, 2.0, 2.0],
        }
    ).to_parquet(tmp_path / "icu" / "icustays.parquet")
    pd.DataFrame(
        {
            "subject_id": [1, 2],
            "anchor_age": [60, 65],
            "anchor_year": [2020, 2020],
            "gender": ["F", "M"],
        }
    ).to_parquet(tmp_path / "hosp" / "patients.parquet")
    pd.DataFrame(
        {
            "hadm_id": [10, 20],
            "admittime": ["2020-01-01", "2020-03-01"],
            "hospital_expire_flag": [0, 0],
        }
    ).to_parquet(tmp_path / "hosp" / "admissions.parquet")
    pd.DataFrame(
        {
            "hadm_id": [10, 20],
            "icd_code": ["A41.9", "I50.0"],
        }
    ).to_parquet(tmp_path / "hosp" / "diagnoses_icd.parquet")

    streamlit_stub = _SessionStateStreamlit(
        {
            "language": "en",
            "cohort_enabled": True,
            "cohort_filter": {
                "age_min": 0,
                "age_max": 120,
                "first_icu_stay": None,
                "los_min": 0,
                "gender": None,
                "survived": None,
                "disease_cohort": "none",
                "icd_include_query": "A41",
                "icd_exclude_query": "I50",
            },
        }
    )
    monkeypatch.setattr(app, "st", streamlit_stub)

    result = app.apply_cohort_filter(tmp_path, "miiv")

    assert result is not None
    assert result["total_before"] == 3
    assert result["total_after"] == 2
    assert result["filtered_ids"] == [101, 102]


def test_plotly_compat_keeps_plotly_specific_width_api_untouched(monkeypatch) -> None:
    class _PlotlyStub:
        def __init__(self) -> None:
            self.kwargs = None
            self.layout_updates = None

        def update_layout(self, **kwargs):
            self.layout_updates = kwargs
            return self

        def plotly_chart(self, figure, **kwargs):
            self.figure = figure
            self.kwargs = kwargs
            return None

    streamlit_stub = _PlotlyStub()
    monkeypatch.setattr(app, "st", streamlit_stub)

    app._plotly_chart_compat(streamlit_stub, use_container_width=True, config={"displaylogo": False})

    assert streamlit_stub.kwargs == {
        "use_container_width": True,
        "config": {"displaylogo": False},
        "theme": None,
    }
    assert streamlit_stub.layout_updates == {
        "template": "plotly_white",
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FFFFFF",
        "font": {"color": "#111827"},
    }


def test_streamlit_theme_is_fixed_to_light_mode() -> None:
    config_path = Path(app.__file__).resolve().parents[3] / ".streamlit" / "config.toml"
    theme_config = config_path.read_text(encoding="utf-8")
    shell_css = shell_styles._load_shell_overrides_css()
    legacy_dashboard_css = (Path(app.__file__).parent / "visualizations" / "cohort_dashboard.py").read_text(
        encoding="utf-8"
    )

    assert '[theme]' in theme_config
    assert 'base = "light"' in theme_config
    assert "color-scheme: light !important" in shell_css
    assert 'div[data-baseweb="popover"]' in shell_css
    assert 'div[data-baseweb="menu"]' in shell_css
    assert 'div[role="listbox"]' in shell_css
    assert "prefers-color-scheme: dark" not in legacy_dashboard_css


def test_coverage_audit_heatmap_uses_shell_style_and_hides_modebar() -> None:
    source = Path(data_coverage_audit_page.__file__).read_text(encoding="utf-8")
    shell_css = shell_styles._load_shell_overrides_css()
    global_css = Path(styles.__file__).read_text(encoding="utf-8")
    heatmap_source = source[
        source.index("fig = go.Figure(data=go.Heatmap("):
        source.index('key="audit_coverage_heatmap"')
    ]
    audit_css = shell_css[
        shell_css.index(".stApp .audit-panel-title {"):
        shell_css.index('.stApp [data-testid="stMain"] [data-testid="stRadio"]')
    ]
    global_audit_css = global_css[
        global_css.index("        .audit-panel-title {"):
        global_css.index("        .audit-denominator-note {")
    ]

    assert "xgap=4" in heatmap_source
    assert "ygap=4" in heatmap_source
    assert '"displayModeBar": False' in heatmap_source
    assert "paper_bgcolor='#FFFFFF'" in heatmap_source
    assert "font=dict(family='IBM Plex Sans" in heatmap_source
    assert "'#8fbfc7'" in heatmap_source
    assert "'#059669'" not in heatmap_source
    assert "background: var(--accent-soft)" in audit_css
    assert "border-color: var(--hair-2)" in audit_css
    assert "font-size: 16px" in audit_css
    assert "var(--figure-navy)" not in audit_css
    assert "background: var(--accent-soft)" in global_audit_css
    assert "border: 1px solid var(--accent-border)" in global_audit_css
    assert "background: var(--figure-navy)" not in global_audit_css
    assert "var(--figure-orange)" not in global_audit_css


def test_quality_panel_switcher_renders_one_lazy_panel(monkeypatch) -> None:
    class _Panel:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _QualitySwitchStreamlit:
        def __init__(self) -> None:
            self.session_state = {"quality_active_panel": "temporal"}
            self.radio_calls = 0
            self.tabs_called = False

        def container(self, **_kwargs):
            return _Panel()

        def markdown(self, *_args, **_kwargs) -> None:
            pass

        def radio(self, _label, *, options, key, **_kwargs):
            self.radio_calls += 1
            assert options == ["missingness", "outliers", "temporal"]
            return self.session_state[key]

        def tabs(self, *_args, **_kwargs):
            self.tabs_called = True
            return []

    streamlit_stub = _QualitySwitchStreamlit()
    monkeypatch.setattr(quality_page, "st", streamlit_stub, raising=False)

    active_panel = quality_page._render_quality_panel_switcher("en")

    assert active_panel == "temporal"
    assert streamlit_stub.radio_calls == 1
    assert streamlit_stub.tabs_called is False
    assert quality_page._render_quality_panel_switcher("en", screenshot_mode=True) == "missingness"


def test_entry_page_copy_and_cta_spacing_address_review_comments() -> None:
    entry_source = pages_redesign.render_entry_redesign_page.__code__.co_consts
    source_text = "\n".join(str(value) for value in entry_source if isinstance(value, str))
    css_text = shell_styles._load_shell_overrides_css()

    assert "All 19 modules / 167 features available" not in source_text
    assert "Lightweight review dataset opens immediately" not in source_text
    assert "Start at data extraction, then open review panels when ready" in source_text
    assert "Research Agent static gallery viewable" in source_text
    assert "Research Agent setup and local-run handoff preview" not in source_text
    assert "Use local data folder" in source_text
    assert "Generate code only" in source_text
    assert "Let the Research Agent generate a reusable code skeleton" in source_text
    assert "Let the Research Agent prepare extraction settings" not in source_text
    assert "try first" in source_text
    assert "_eu_entry_lang_toggle" in source_text
    assert "Choose how data enters the workspace" in source_text
    assert "EASYICU — ENTRY · MODE CHOICE" not in source_text
    assert "eu-entry-mode-card demo" in source_text
    assert "eu-entry-mode-card real" in source_text
    assert "eu_entry_modes_grid" in source_text
    assert "eu_entry_demo_mode_card" in source_text
    assert "eu_entry_real_mode_card" in source_text
    assert "eu_entry_code_row" in source_text
    assert "中 / EN" not in source_text
    assert "Concept catalog" not in source_text
    assert "Sample cohorts" not in source_text
    assert "eu-entry-next" in source_text
    assert "eu-entry-rail" in source_text
    assert "eu-entry-step" in source_text
    assert "Data gate" in source_text
    assert "Cohort review" in source_text
    assert "Quality checks" in source_text
    assert "Export & handoff" in source_text
    assert ".eu-entry-rail" in css_text
    assert "st-key-eu_entry_modes_grid" in css_text
    assert "grid-template-columns: repeat(4, minmax(0, 1fr))" in css_text
    assert "max-width: 1100px" in css_text
    assert "st-key-eu_entry_topbar_shell" in css_text
    assert "st-key-_eu_entry_lang_toggle" in css_text
    assert "st-key-eu_entry_code_row" in css_text
    assert "margin-top: 40px" in css_text
    assert ".eu-entry-step::before" in css_text
    assert "st-key-_eu_entry_demo" in css_text
    assert "st-key-_eu_entry_real" in css_text
    assert "st-key-_eu_entry_nodata" in css_text
    base_kicker_css = css_text[
        css_text.index(".stApp .eu-start-kicker {"):
        css_text.index(".stApp .eu-start-card.primary .eu-start-kicker")
    ]
    assert "font-size: 16px" in base_kicker_css
    assert "font-weight: 700" in base_kicker_css
    assert "text-transform: none" in base_kicker_css
    assert ".eu-start-card.primary .eu-start-kicker" in css_text
    primary_kicker_css = css_text[
        css_text.index(".stApp .eu-start-card.primary .eu-start-kicker {"):
        css_text.index(".stApp .eu-start-head h3")
    ]
    assert "font-size: 17px" in primary_kicker_css
    assert "font-weight: 760" in primary_kicker_css
    assert "text-transform: none" in primary_kicker_css
    assert "border: 1px solid var(--accent-border)" in primary_kicker_css
    assert ".stApp .eu-start-head h3" in css_text
    assert "font-size: 18px !important" in css_text
    assert "font-size: 21px !important" in css_text
    assert ".stApp .eu-rail-title" in css_text
    assert "font-size: 14.5px" in css_text
    assert ".stApp .eu-flow-title" in css_text
    assert "font-size: 13.8px" in css_text
    assert "st-key-_eu_tutorial_resource_" in css_text
    assert "st-key-eu_tutorial_resources_card" in css_text
    assert "font-size: 13px" in css_text
    assert "margin-top: -49px" not in css_text


def test_get_started_page_matches_latest_print_reference_structure() -> None:
    page_source = Path(pages_redesign.__file__).read_text(encoding="utf-8")
    css_text = shell_styles._load_shell_overrides_css()

    assert 'A quiet, reviewable path from data to draft' in page_source
    assert 'New here? Take the 2-minute demo tour' in page_source
    assert 'How a study moves through EasyICU' in page_source
    assert 'Common questions' in page_source
    assert 'key="eu_getstarted_demo_tour"' in page_source
    assert 'key="eu_getstarted_steps"' in page_source
    assert '_route_to_workspace_states(st.session_state)' in page_source
    assert '_route_to_ai_assistant(' in page_source
    assert "st.columns([0.075, 0.925]" in page_source
    assert "st.columns([0.2, 0.8]" not in page_source
    assert 'Agent gallery' not in page_source

    assert '.stApp [class*="st-key-eu_getstarted_demo_tour"]' in css_text
    assert '.stApp [class*="st-key-eu_getstarted_steps"]' in css_text
    assert '.stApp .eu-guide-step-num' in css_text
    assert '.stApp .eu-faq-card' in css_text
    assert 'grid-template-columns: 56px minmax(0, 1fr)' in css_text


def test_demo_preview_view_all_modules_expands_inline_with_continue_action() -> None:
    sidebar_source = Path(sidebar.__file__).read_text(encoding="utf-8")
    css_text = shell_styles._load_shell_overrides_css()

    assert '<span class="eu-mini-button">' not in sidebar_source
    assert "eu_demo_modules_toggle" in sidebar_source
    assert "eu_demo_module_catalog_panel" in sidebar_source
    assert "eu_demo_modules_continue" in sidebar_source
    assert "_confirm_demo_data_source()" in sidebar_source
    assert "View all modules" in sidebar_source
    assert "Hide modules" in sidebar_source
    assert "Continue to cohort setup" in sidebar_source
    assert "查看全部模块" in sidebar_source
    assert "_demo_module_catalog_html" in sidebar_source

    catalog_html = sidebar._demo_module_catalog_html("en")
    assert "All demo feature modules" in catalog_html
    assert "These modules will be available in Step 3" in catalog_html
    assert "SOFA-2 Scores" in catalog_html
    assert "⭐ SOFA-2 Scores" not in catalog_html
    assert catalog_html.count("eu-module-catalog-row") == len(concept_catalog.CONCEPT_GROUPS_INTERNAL)

    assert "st-key-eu_demo_preview_card" in css_text
    assert "st-key-eu_demo_module_catalog_panel" in css_text
    assert "st-key-eu_demo_modules_continue" in css_text
    assert "-webkit-text-fill-color: var(--ink) !important" in css_text
    assert ".eu-module-catalog-grid" in css_text


def test_main_shell_copy_hides_internal_feature_counts() -> None:
    with open(pages_redesign.__file__, encoding="utf-8") as handle:
        page_source = handle.read()
    with open(cohort_redesign.__file__, encoding="utf-8") as handle:
        cohort_source = handle.read()
    with open(cohort_group_page.__file__, encoding="utf-8") as handle:
        cohort_group_source = handle.read()
    with open(cohort_dashboard_page.__file__, encoding="utf-8") as handle:
        cohort_dashboard_source = handle.read()
    with open(cohort_severity_page.__file__, encoding="utf-8") as handle:
        cohort_severity_source = handle.read()
    with open(sofa_reclassification.__file__, encoding="utf-8") as handle:
        sofa_reclassification_source = handle.read()
    with open(app.__file__, encoding="utf-8") as handle:
        app_source = handle.read()
    with open(i18n.__file__, encoding="utf-8") as handle:
        i18n_source = handle.read()
    with open(patient_page.__file__, encoding="utf-8") as handle:
        patient_source = handle.read()
    with open(quick_visualization_page.__file__, encoding="utf-8") as handle:
        quick_visualization_source = handle.read()
    with open(research_agent.__file__, encoding="utf-8") as handle:
        research_agent_source = handle.read()
    with open(ui_helpers.__file__, encoding="utf-8") as handle:
        ui_helpers_source = handle.read()
    css_text = shell_styles._load_shell_overrides_css()

    assert "167 features" not in page_source
    assert "10 of 167" not in page_source
    assert "Concept catalog · 19" not in page_source
    assert "try first" in page_source
    assert "Lightweight review data is ready immediately" in page_source
    assert "review concept set" in page_source
    assert "Demo set" in cohort_source
    assert "if _active == 'tutorial':" in app_source
    assert "'tutorial':       ('Run'" not in app_source
    assert 'key="_eu_topbar_history"' not in app_source
    assert 'key="_eu_topbar_agent"' not in app_source

    assert "_render_topbar_path_nav" in app_source
    assert "_apply_topbar_breadcrumb_target" in app_source
    assert 'key="eu_extract_breadcrumb_nav"' in app_source
    assert 'key=f"eu_page_breadcrumb_nav_{_active}"' in app_source
    assert "_set_extract_step_state" in app_source
    assert "_switch_extract_entry_mode" in app_source
    assert "[8.55, 1.45]" in app_source
    assert "[8.35, 1.45, 0.7]" not in app_source
    assert "_eu_topbar_cancel" not in app_source
    assert "_eu_topbar_confirm_step" not in app_source
    assert "_eu_topbar_ai_extract" not in app_source
    assert "_eu_topbar_ai_tutorial" not in app_source
    assert "_eu_topbar_ai" not in app_source
    assert "Confirm & continue" not in app_source
    assert "[7.45, 1.35, 0.68, 0.7, 2.0]" not in app_source
    assert "[6.7, 1.25, 0.65, 0.95, 2.25]" not in app_source
    topbar_source = app_source[
        app_source.index("# ============ Shell-A top bar"):
        app_source.index("    _render_global_status_strip")
    ]
    assert "_render_extract_breadcrumb_nav" not in topbar_source
    assert "st.popover(" not in topbar_source
    assert "_eu_bc_step_" not in topbar_source
    assert "_eu_bc_mode_" not in topbar_source
    assert "Global command palette" not in topbar_source
    assert "eu-bc-search" not in topbar_source
    assert "include_search" not in topbar_source
    assert "Session local" not in topbar_source
    assert "Demo · simulated" not in topbar_source
    assert "auto_awesome" not in topbar_source
    assert "play_arrow" not in topbar_source
    assert "_render_topbar_pills_row" not in topbar_source
    assert "Global command palette" not in ui_helpers_source
    assert "eu-kbd-hint" not in ui_helpers_source
    assert "Search<span" not in ui_helpers_source
    assert "⌘K" not in ui_helpers_source
    assert "use_container_width=False" in topbar_source
    assert "### 👥" not in cohort_group_source
    assert "#### 🔀" not in cohort_group_source
    assert "👤 Demographics" not in cohort_group_source
    assert "💀 Survived" not in cohort_group_source
    assert "_render_section_heading" in cohort_group_source
    assert "eu-native-section-heading" in cohort_group_source
    assert "eu-native-section-heading-after" in cohort_group_source
    assert "eu-compact-divider" in app_source
    assert "### 🎯" not in cohort_dashboard_source
    assert "📊 Cohort Snapshot Summary" not in cohort_dashboard_source
    assert "Cohort Snapshot Summary" not in cohort_dashboard_source
    assert "st.caption(snapshot_subtitle)" not in cohort_dashboard_source
    assert "Snapshot summary" in cohort_dashboard_source
    assert cohort_dashboard_source.index("if 'dash_demographics' not in st.session_state:") < cohort_dashboard_source.index('"Snapshot summary"')
    chart_theme_block = cohort_dashboard_source[
        cohort_dashboard_source.index("SHELL_CHART = {"):
        cohort_dashboard_source.index("def _style_readout_figure")
    ]
    phenotype_chart_block = cohort_dashboard_source[
        cohort_dashboard_source.index("phenotype_df = review['phenotype']"):
        cohort_dashboard_source.index("with chart_col2:")
    ]
    severity_chart_block = cohort_dashboard_source[
        cohort_dashboard_source.index("severity_df = review['severity']"):
        cohort_dashboard_source.index('st.warning("No SOFA severity column found"')
    ]
    assert '"plot": "#fbfaf7"' in chart_theme_block
    assert '"teal": "#0f766e"' in chart_theme_block
    assert '"rose": "#9f3a57"' in chart_theme_block
    assert "paper_bgcolor='rgba(0,0,0,0)'" in cohort_dashboard_source
    assert "plot_bgcolor=SHELL_CHART[\"plot\"]" in cohort_dashboard_source
    assert "SHELL_CHART[\"teal_soft\"]" in phenotype_chart_block
    assert "SHELL_CHART[\"teal\"]" in phenotype_chart_block
    assert "#dbeafe" not in phenotype_chart_block
    assert "rgba(96, 142, 239" not in severity_chart_block
    assert "SHELL_CHART[\"rose\"]" in severity_chart_block
    compare_mode_block = cohort_group_source[
        cohort_group_source.index("compare_mode = st.radio("):
        cohort_group_source.index("    # 根据模式显示额外配置")
    ]
    feature_modules_block = cohort_group_source[
        cohort_group_source.index("selected_modules = st.multiselect("):
        cohort_group_source.index("    # 显示将要加载的特征")
    ]
    assert 'label_visibility="collapsed"' in compare_mode_block
    assert 'label_visibility="collapsed"' in feature_modules_block
    assert "FEATURE_MODULES = _build_cohort_feature_modules(lang)" in cohort_group_source
    assert "CONCEPT_GROUPS_INTERNAL" in cohort_group_source
    assert "COHORT_GROUP_DEFAULT_MODULES" in cohort_group_source
    assert "COHORT_GROUP_LEGACY_MODULE_MAP" in cohort_group_source
    assert "COHORT_GROUP_BASE_MODULES" in cohort_group_source
    assert "Open the **SOFA reclassification** panel for the matrix" in cohort_dashboard_source
    assert "SOFA-1 vs SOFA-2 Reclassification" not in cohort_dashboard_source
    assert "dash_reclass_matrix" not in cohort_dashboard_source
    assert "dash_reclass_organ_contrib" not in cohort_dashboard_source
    assert "st.columns(6)" not in cohort_dashboard_source
    assert "_style_readout_figure" in cohort_dashboard_source
    assert "Clinical phenotype prevalence" in cohort_dashboard_source
    assert "reclass_matrix" in cohort_severity_source
    assert "reclass_organ_contrib" in cohort_severity_source
    assert "_style_reclass_figure" in cohort_severity_source
    reclass_chart_block = cohort_severity_source[
        cohort_severity_source.index("RECLASS_CHART = {"):
        cohort_severity_source.index("def _style_reclass_figure")
    ]
    reclass_style_block = cohort_severity_source[
        cohort_severity_source.index("def _style_reclass_figure"):
        cohort_severity_source.index("def render_severity_reclassification_subtab")
    ]
    assert '"plot": "#fbfaf7"' in reclass_chart_block
    assert '"teal": "#0f766e"' in reclass_chart_block
    assert '"rose": "#9f3a57"' in reclass_chart_block
    assert "paper_bgcolor='rgba(0,0,0,0)'" in reclass_style_block
    assert 'plot_bgcolor=RECLASS_CHART["plot"]' in reclass_style_block
    assert "RECLASS_CHART[\"teal_soft\"]" in cohort_severity_source
    assert "RECLASS_CHART[\"rose\"]" in cohort_severity_source
    assert "#dbeafe" not in cohort_severity_source
    assert "#e11d48" not in cohort_severity_source
    assert "rgba(96, 142, 239" not in cohort_severity_source
    assert "Reclassification matrix" in cohort_severity_source
    assert "st.columns(5)" not in sofa_reclassification_source
    assert "eu-cohort-kpi-grid" in sofa_reclassification_source
    assert "### 🧭" not in cohort_severity_source
    assert "_render_section_heading" in cohort_dashboard_source
    assert "_render_section_heading" in cohort_severity_source
    assert "eu-native-section-heading-after" in cohort_severity_source
    assert "st.caption(subtitle)" not in cohort_severity_source
    assert '"👥 Groups"' not in app_source
    assert '"🎯 Snapshot"' not in app_source
    assert '"🧭 SOFA Δ"' not in app_source
    assert 'key_suffix="global"' not in app_source
    assert '"● Setup"' not in app_source
    assert '"● Workbench"' not in app_source
    assert '"● Summary"' not in app_source
    assert "'review_tables': '📋 Data Tables'" not in i18n_source
    assert "'review_trends': '📈 Time Series'" not in i18n_source
    assert "'review_patients': '🏥 Patient Overview'" not in i18n_source
    assert "'review_quality': '📊 Data Quality'" not in i18n_source
    assert "'lane_vitals': '❤️ Vital Signs'" not in i18n_source
    assert "'sub_timeseries': '📈 Time Series'" not in i18n_source
    assert "'sub_data_quality': '📊 Data Quality'" not in i18n_source
    assert "'sub_data_table': '📋 数据大表'" not in i18n_source
    assert "### 📊 Dashboard" not in patient_source
    assert "#### 📈 SOFA Score Trend" not in patient_source
    assert "#### 🔄 SOFA-1 vs SOFA-2 Comparison" not in patient_source
    assert "Vital-sign trends moved" not in patient_source
    assert "Vital Signs Snapshot" in patient_source
    assert "_render_compact_trend_panel" in patient_source
    assert '"👤 Patient ID"' not in patient_source
    assert '"⏮️ First"' not in patient_source
    qv_switcher_source = quick_visualization_source[
        quick_visualization_source.index("def _render_quick_viz_panel_switcher"):
        quick_visualization_source.index("def render_quick_visualization_page")
    ]
    assert "inline-control-label" not in qv_switcher_source
    assert 'icon="🧪"' not in research_agent_source
    assert 'st.button(f"🤖 {label}"' not in app_source
    assert ".eu-native-section-heading" in css_text
    assert ".eu-native-section-heading-after" in css_text
    assert ".eu-native-section-heading p" in css_text
    assert "min-height: 26px" in css_text
    assert "margin-bottom: 10px" in css_text
    assert ".eu-compact-divider" in css_text
    native_heading_css = css_text[
        css_text.index(".stApp .eu-native-section-heading {"):
        css_text.index(".stApp .eu-native-section-heading:first-child")
    ]
    assert "margin: 18px 0 0" in native_heading_css
    assert "box-sizing: border-box" in native_heading_css
    assert ".eu-cohort-kpi-grid" in css_text
    assert ".eu-chart-heading" in css_text
    chart_heading_css = css_text[
        css_text.index(".stApp .eu-chart-heading {"):
        css_text.index(".stApp .eu-chart-row-gap")
    ]
    assert "font-size: 12.5px" in chart_heading_css
    assert "font-size: 17px" in chart_heading_css
    assert "font-size: 12.8px" in chart_heading_css
    assert "font-size: 10px" not in chart_heading_css
    assert "grid-template-columns: repeat(auto-fit" in css_text
    assert 'st-key-qv_panel_switcher' in css_text
    assert ".stApp .inline-control-label" in css_text
    inline_label_css = css_text[
        css_text.index(".stApp .inline-control-label {"):
        css_text.index(".stApp .subtle-preview-note")
    ]
    assert "display: block !important" in inline_label_css
    assert "margin: 0 0 8px !important" in inline_label_css
    qv_switcher_css = css_text[
        css_text.index('.stApp [class*="st-key-qv_panel_switcher"] {'):
        css_text.index('.stApp [class*="st-key-qv_panel_switcher"] [data-testid="stVerticalBlock"]')
    ]
    assert "padding: 0 0 16px !important" in qv_switcher_css
    assert "display: none !important" in qv_switcher_css
    assert "grid-template-columns: repeat(4, minmax(128px, 1fr))" in css_text
    assert "label:has(input:checked) *" in css_text
    assert "-webkit-text-fill-color: #fff" in css_text
    assert ".eu-topbar-stage" in css_text
    topbar_stage_css = css_text[
        css_text.index(".eu-topbar-stage {"): css_text.index(".eu-topbar-stage .eu-pill")
    ]
    assert "position: static" in topbar_stage_css
    assert "position: fixed" not in topbar_stage_css
    assert "right: 390px" not in topbar_stage_css
    assert "top: 21px" not in topbar_stage_css
    assert "st-key-eu_extract_breadcrumb_nav" in css_text
    assert "st-key-eu_page_breadcrumb_nav_" in css_text
    assert ".eu-bc-current" in css_text
    assert ".eu-bc-popover-title" not in css_text
    assert ".eu-bc-search" not in css_text
    breadcrumb_root_css = css_text[
        css_text.index('.stApp [class*="st-key-eu_extract_breadcrumb_nav"],'):
        css_text.index('.stApp [class*="st-key-eu_extract_breadcrumb_nav"] [data-testid="stHorizontalBlock"]')
    ]
    assert "min-height: 38px !important" in breadcrumb_root_css
    assert "align-items: flex-start !important" in breadcrumb_root_css
    assert "justify-content: flex-start !important" in breadcrumb_root_css
    assert "text-align: left !important" in breadcrumb_root_css
    breadcrumb_row_css = css_text[
        css_text.index('.stApp [class*="st-key-eu_extract_breadcrumb_nav"] [data-testid="stHorizontalBlock"],'):
        css_text.index('.stApp [class*="st-key-eu_extract_breadcrumb_nav"] [data-testid="stColumn"],')
    ]
    assert "display: inline-flex !important" in breadcrumb_row_css
    assert "flex: 0 0 auto !important" in breadcrumb_row_css
    assert "justify-content: flex-start !important" in breadcrumb_row_css
    assert "align-self: flex-start !important" in breadcrumb_row_css
    assert "width: auto !important" in breadcrumb_row_css
    breadcrumb_column_css = css_text[
        css_text.index('.stApp [class*="st-key-eu_extract_breadcrumb_nav"] [data-testid="stColumn"],'):
        css_text.index('.stApp [class*="st-key-eu_extract_breadcrumb_nav"] .stButton,')
    ]
    assert "flex: 0 0 auto !important" in breadcrumb_column_css
    assert "width: auto !important" in breadcrumb_column_css
    breadcrumb_button_css = css_text[
        css_text.index('.stApp [class*="st-key-eu_extract_breadcrumb_nav"] .stButton > button,'):
        css_text.index('.stApp [class*="st-key-eu_extract_breadcrumb_nav"] .stButton > button:hover,')
    ]
    assert "width: auto !important" in breadcrumb_button_css
    assert "border: 0 !important" in breadcrumb_button_css
    assert "font-size: 13px !important" in breadcrumb_button_css
    assert "line-height: 22px !important" in breadcrumb_button_css
    breadcrumb_current_css = css_text[
        css_text.index(".eu-bc-current {"):
        css_text.index(".eu-bc-sep {")
    ]
    assert "font-size: 13px !important" in breadcrumb_current_css
    assert "font-weight: 500 !important" in breadcrumb_current_css
    breadcrumb_sep_css = css_text[
        css_text.index(".eu-bc-sep {"):
        css_text.index(".eu-topbar-stage {")
    ]
    assert "font-size: 13px !important" in breadcrumb_sep_css
    assert "font-weight: 500 !important" in breadcrumb_sep_css
    readability_floor_css = css_text[
        css_text.index("/* Global readability floor"):
        css_text.index('.stApp [data-testid="stMetricLabel"],')
    ]
    assert "--eu-readable-min: 12px" in css_text
    assert 'font-size: max(var(--eu-readable-min), 1em) !important' in readability_floor_css
    assert ".inline-control-label" in readability_floor_css
    assert ".viz-demo-load-kicker" in readability_floor_css
    assert ".eu-entry-brand-sub" in readability_floor_css
    assert ".eu-entry-next-head span" in readability_floor_css
    assert '[data-testid="stSidebar"] p' in readability_floor_css
    metric_floor_css = css_text[css_text.index('.stApp [data-testid="stMetricLabel"],'):]
    assert "font-size: var(--eu-readable-min) !important" in metric_floor_css
    assert (
        ".stApp .eu-entry-brand-sub,\n"
        ".stApp .eu-entry-version {\n"
        "  font-size: 12px;"
    ) in css_text
    entry_next_head_css = css_text[
        css_text.index(".stApp .eu-entry-next-head span {"):
        css_text.index(".stApp .eu-entry-next-head b {")
    ]
    assert "font-size: 12px" in entry_next_head_css
    final_breadcrumb_css = css_text[
        css_text.rindex('.stApp [class*="st-key-eu_extract_breadcrumb_nav"] .stButton > button,'):
    ]
    assert css_text.index("/* Global readability floor") < css_text.rindex(
        '.stApp [class*="st-key-eu_extract_breadcrumb_nav"] .stButton > button,'
    )
    assert '.stApp [class*="st-key-eu_extract_breadcrumb_nav"] .stButton > button *' in final_breadcrumb_css
    assert '.stApp [class*="st-key-eu_extract_breadcrumb_nav"] button *' in final_breadcrumb_css
    assert "line-height: 22px !important" in final_breadcrumb_css
    pill_css = css_text[css_text.index(".eu-pill {"): css_text.index(".eu-pill .dot")]
    assert "font-size: 12px" in pill_css
    assert "height: 24px" in pill_css
    assert "st-key-_eu_topbar_cancel" not in css_text
    assert "st-key-_eu_topbar_confirm_" not in css_text
    assert '[data-testid="stMetric"]' in css_text
    assert "font-size: 16px" in css_text
    assert '[data-baseweb="tag"]' in css_text


def test_patient_feature_snapshot_grid_owns_caption_spacing() -> None:
    patient_source = Path(patient_page.__file__).read_text(encoding="utf-8")
    css_text = shell_styles._load_shell_overrides_css()

    assert "patient-feature-snapshot-grid" in patient_source
    assert "patient-feature-snapshot-caption" in patient_source
    assert "st.columns(min(4, len(visible_snapshots)))" not in patient_source
    assert ".patient-feature-snapshot-grid" in css_text
    assert ".patient-feature-snapshot-caption" in css_text
    caption_css = css_text[css_text.index(".stApp .patient-feature-snapshot-caption"):]
    assert "clear: both !important" in caption_css[:400]


def test_page_header_renders_html_without_markdown_code_blocks(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []

    class _StreamlitStub:
        @staticmethod
        def markdown(body: str, unsafe_allow_html: bool = False) -> None:
            calls.append((body, unsafe_allow_html))

    monkeypatch.setattr(page_header, "st", _StreamlitStub)

    page_header.render_page_header(
        "EasyICU Research Agent",
        "Demo Mode is a lightweight preview.",
        icon="",
        kicker="Research Agent",
    )

    assert calls
    body, unsafe = calls[0]
    assert unsafe is True
    assert body.startswith('<div class="app-page-header">')
    assert "\n        <div" not in body
    assert '<div class="app-page-title">EasyICU Research Agent</div>' in body


def test_ui_topbar_helper_does_not_render_fake_search_placeholder(monkeypatch) -> None:
    calls: list[str] = []

    class _StreamlitStub:
        @staticmethod
        def html(body: str) -> None:
            calls.append(body)

    monkeypatch.setattr(ui_helpers, "st", _StreamlitStub)

    ui_helpers.render_topbar(
        ["Home", "Research Agent"],
        pills=[("Demo · simulated", "demo")],
    )

    assert calls
    body = calls[0]
    assert "Search" not in body
    assert "⌘K" not in body
    assert "eu-kbd-hint" not in body
    assert "Global command palette" not in body
    assert "Demo · simulated" in body


def test_design_page_header_spacing_prevents_summary_overlap() -> None:
    body = cohort_charts.render_design_page_header(
        kicker="Research Agent",
        title_en="Characterise whether high lactate, low mean arterial pressure, and vaso...",
        title_zh="研究智能体",
        desc="Analysis-first output summary. Manuscript drafting remains gated by evidence review.",
        lang="en",
    )
    css_text = shell_styles._load_shell_overrides_css()

    assert body.startswith('<div class="eu-design-page-header">')
    assert "eu-design-page-header-spacer" not in body
    assert 'style="margin-bottom:6px"' not in body
    assert ".stApp .eu-design-page-header" in css_text
    header_css = css_text[
        css_text.index(".stApp .eu-design-page-header {"):
        css_text.index(".stApp .app-page-kicker")
    ]
    assert "clear: both !important" in header_css
    assert "margin: 0 0 18px !important" in header_css
    assert "eu-design-page-header-spacer" not in css_text
    summary_css = css_text[
        css_text.index(".eu-summary-page {"):
        css_text.index(".eu-summary-demo-note")
    ]
    assert "clear: both" in summary_css
    assert "margin-top: 16px" in summary_css


def test_tutorial_dictionary_helpers_remain_available_but_not_on_get_started() -> None:
    modules = pages_redesign._tutorial_dictionary_modules("en")
    html = pages_redesign._tutorial_dictionary_module_html(
        "en",
        selected_module="vitals",
    )
    css_text = shell_styles._load_shell_overrides_css()
    dictionary_source_text = "\n".join(
        str(value)
        for value in pages_redesign._render_tutorial_dictionary.__code__.co_consts
        if isinstance(value, str)
    )
    render_source = inspect.getsource(pages_redesign.render_tutorial_redesign_page)
    dictionary_source = inspect.getsource(pages_redesign._render_tutorial_dictionary)

    total_features = sum(len(module["concepts"]) for module in modules)
    selected_module = next(module for module in modules if module["key"] == "vitals")

    assert len(modules) == 19
    assert total_features >= 200
    assert total_features == len(pages_redesign.CONCEPT_DICTIONARY)
    assert len(selected_module["concepts"]) == 8
    assert html.count("eu-dict-list-row") == len(selected_module["concepts"])
    assert html.count('data-active="true"') == 1
    assert "eu-dict-module-preview" in html
    assert "Vital Signs" in html
    assert "SOFA-2 Score" not in html
    assert "Heart Rate" in html
    assert 'data-selected="true"' not in html
    assert "167" not in html
    assert "Browse the complete module-grouped EasyICU dictionary" in dictionary_source_text
    assert "217" not in dictionary_source_text
    assert "_eu_tutorial_dict_module" in dictionary_source
    assert "_eu_tutorial_dict_feature" not in dictionary_source
    assert "Selected concept" not in dictionary_source
    assert "_eu_tutorial_dictionary" not in render_source
    assert "Open selected module -> Concepts" not in render_source
    assert "src/easyicu/data/concept-dict.json" in dictionary_source
    assert "st-key-eu_tutorial_dictionary_panel" in css_text
    dict_kicker_css = css_text[
        css_text.index(".stApp .eu-dict-kicker {"):
        css_text.index(".stApp .eu-dict-head h3")
    ]
    assert "font-size: 15px" in dict_kicker_css
    assert "font-weight: 720" in dict_kicker_css
    assert "text-transform: none" in dict_kicker_css
    assert "letter-spacing: 0" in dict_kicker_css
    dict_title_css = css_text[
        css_text.index(".stApp .eu-dict-head h3 {"):
        css_text.index(".stApp .eu-dict-head p")
    ]
    assert "font-size: 20px !important" in dict_title_css
    assert "font-weight: 650 !important" in dict_title_css
    assert "line-height: 1.24 !important" in dict_title_css
    assert 'content: "⌄"' in css_text
    assert "width: 40px" in css_text
    assert "font-size: 22px" in css_text
    assert "rgba(216, 246, 248" not in css_text
    assert ".eu-dict-module-heading" in css_text


def test_get_started_buttons_route_to_real_destinations() -> None:
    page_source = Path(pages_redesign.__file__).read_text(encoding="utf-8")
    app_source = Path(app.__file__).read_text(encoding="utf-8")
    css_text = shell_styles._load_shell_overrides_css()

    assert 'key="_eu_getstarted_start_demo"' in page_source
    assert 'key="_eu_getstarted_browse_states"' in page_source
    assert 'key=f"_eu_getstarted_step_{number}_{idx}"' in page_source
    assert 'key="eu_getstarted_faq_card"' in page_source
    assert 'key=f"_eu_getstarted_faq_q_{idx}"' in page_source
    assert 'st.session_state["_eu_getstarted_faq_open"] = -1 if is_open else idx' in page_source
    assert '"extract_source"' in page_source
    assert "_route_to_extract_step(st.session_state, 1)" in page_source
    assert "st.columns([0.075, 0.925]" in page_source
    assert "eu-guide-step-title" in page_source
    assert '"assistant"' in page_source
    assert 'state["_active_main_page"] = "assistant"' in page_source
    assert 'state["_inline_ai_panel_open"] = False' in page_source
    render_source = inspect.getsource(pages_redesign.render_tutorial_redesign_page)
    assert 'st.session_state["_main_nav_widget"] = target' not in render_source
    assert "_EXTRA_PAGES = {'extract', 'assistant', 'states', 'settings'}" in app_source
    assert 'mobile_page_keys = ["extract"] + page_keys + ["assistant", "states", "settings"]' in app_source
    assert 'page_labels["extract"] = "Data Extraction" if lang == "en" else "数据提取"' in app_source
    assert 'render_ai_assistant_page(lang)' in app_source
    assert 'render_workspace_states_reference_page(lang)' in app_source
    assert 'render_settings_redesign_page(lang)' in app_source
    assert "'section.stMain'" in app_source
    assert '[data-testid="stMain"]' in app_source
    assert 'eu-resource-row' not in page_source
    assert 'st-key-_eu_tutorial_resource_' in css_text
    assert "st-key-eu_getstarted_faq_card" in css_text
    assert "st-key-_eu_getstarted_faq_q_" in css_text
    assert ".eu-faq-answer" in css_text

    sample_state: dict[str, object] = {
        "_inline_ai_panel_open": True,
        "_floating_ai_open": True,
        "_ai_pending_question": "stale",
    }
    pages_redesign._route_to_workspace_states(sample_state)
    assert sample_state["_active_main_page"] == "states"
    assert "_main_nav_widget" not in sample_state
    assert sample_state["_inline_ai_panel_open"] is False
    assert sample_state["_floating_ai_open"] is False
    assert "_ai_pending_question" not in sample_state
    assert sample_state["_scroll_to_top"] is True

    assistant_state: dict[str, object] = {"_floating_ai_open": True, "_sidebar_ai_open": True}
    pages_redesign._route_to_ai_assistant(assistant_state, "help me")
    assert assistant_state["_active_main_page"] == "assistant"
    assert "_main_nav_widget" not in assistant_state
    assert assistant_state["_inline_ai_panel_open"] is False
    assert assistant_state["_floating_ai_open"] is False
    assert assistant_state["_sidebar_ai_open"] is False
    assert assistant_state["_scroll_to_top"] is True
    assert assistant_state["_ai_pending_question"] == "help me"

    agent_state: dict[str, object] = {
        "research_agent_resume_run_id": "run_old",
        "research_agent_force_manuscript": True,
        "research_agent_resume_mode": "force_manuscript",
        "research_agent_resume_notes": "stale",
        "research_agent_resume_relax_probe": True,
        "research_agent_preflight_confirmed": True,
        "research_agent_preflight_signature": "stale-signature",
        "research_agent_question": "Keep my editable setup question.",
    }
    pages_redesign._route_to_research_agent_setup(agent_state)
    assert agent_state["_active_main_page"] == "research_agent"
    assert agent_state["_ra_view"] == "setup"
    assert agent_state["_scroll_to_top"] is True
    assert agent_state["research_agent_question"] == "Keep my editable setup question."
    assert "_eu_ra_focus_module_folder" not in agent_state
    for key in (
        "research_agent_resume_run_id",
        "research_agent_force_manuscript",
        "research_agent_resume_mode",
        "research_agent_resume_notes",
        "research_agent_resume_relax_probe",
        "research_agent_preflight_confirmed",
        "research_agent_preflight_signature",
    ):
        assert key not in agent_state

    no_data_state: dict[str, object] = {
        "entry_mode": "demo",
        "use_mock_data": True,
        "database": "mock",
        "path_validated": True,
        "last_validated_path": "/tmp/mock",
        "research_agent_resume_run_id": "run_old",
        "research_agent_force_manuscript": True,
        "research_agent_preflight_confirmed": True,
        "research_agent_cohort_source": i18n.TEXTS["en"]["ra_source_synthetic"],
    }
    pages_redesign._route_to_research_agent_no_data_setup(no_data_state)
    assert no_data_state["_active_main_page"] == "research_agent"
    assert no_data_state["_ra_view"] == "setup"
    assert no_data_state["entry_mode"] == "real"
    assert no_data_state["use_mock_data"] is False
    assert no_data_state["database"] == "miiv"
    assert no_data_state["path_validated"] is False
    assert "last_validated_path" not in no_data_state
    assert no_data_state["_eu_ra_focus_no_data"] is True
    assert no_data_state["_eu_ra_no_data_entry"] is True
    assert "research_agent_cohort_source" not in no_data_state
    assert "research_agent_resume_run_id" not in no_data_state
    assert "research_agent_force_manuscript" not in no_data_state
    assert "research_agent_preflight_confirmed" not in no_data_state

    settings_agent_state: dict[str, object] = {
        "entry_mode": "demo",
        "use_mock_data": True,
        "database": "mock",
        "path_validated": True,
        "last_validated_path": "/tmp/mock",
        "research_agent_force_manuscript": True,
    }
    pages_redesign._route_to_research_agent_setup(
        settings_agent_state,
        force_real=True,
        focus_module_folder=True,
    )
    assert settings_agent_state["_active_main_page"] == "research_agent"
    assert settings_agent_state["_ra_view"] == "setup"
    assert settings_agent_state["entry_mode"] == "real"
    assert settings_agent_state["use_mock_data"] is False
    assert settings_agent_state["database"] == "miiv"
    assert settings_agent_state["path_validated"] is False
    assert "last_validated_path" not in settings_agent_state
    assert settings_agent_state["_eu_ra_focus_module_folder"] is True
    assert "research_agent_force_manuscript" not in settings_agent_state

    extract_state: dict[str, object] = {
        "_active_main_page": "cohort",
        "_inline_ai_panel_open": True,
        "_floating_ai_open": True,
        "step1_confirmed": True,
        "step2_confirmed": True,
        "step3_confirmed": True,
        "export_completed": True,
        "trigger_export": True,
    }
    pages_redesign._route_to_extract_step(extract_state, 1)
    assert extract_state["_active_main_page"] == "extract"
    assert extract_state["step1_confirmed"] is False
    assert extract_state["step2_confirmed"] is False
    assert extract_state["step3_confirmed"] is False
    assert extract_state["export_completed"] is False
    assert extract_state["trigger_export"] is False
    assert extract_state["_inline_ai_panel_open"] is False
    assert extract_state["_floating_ai_open"] is False
    assert extract_state["_scroll_to_top"] is True


def test_workspace_states_page_includes_reference_primitives() -> None:
    page_source = Path(pages_redesign.__file__).read_text(encoding="utf-8")
    css_text = shell_styles._load_shell_overrides_css()

    assert "Status primitives" in page_source
    assert "Reusable building blocks" in page_source
    assert "key=\"eu_states_controls\"" in page_source
    assert 'key=f"_eu_states_context_{item[\'key\']}"' in page_source
    assert 'key=f"_eu_states_state_{item[\'key\']}"' in page_source
    assert "_eu_states_primary_action" in page_source
    assert "_workspace_state_preview_html(current_context, current_mode, current_state, lang)" in page_source
    assert "_workspace_state_action_label(current_context, current_mode, lang)" in page_source
    assert "_apply_workspace_state_action(st.session_state, current_context, current_mode)" in page_source
    assert "eu-state-primitive-grid" in page_source
    assert ".stApp .eu-state-primitive-card" in css_text
    assert '.stApp [class*="st-key-eu_states_controls"]' in css_text
    assert ".stApp .eu-state-hero" in css_text
    assert ".stApp .detail-box" in css_text
    assert ".stApp .gate-block" in css_text
    assert ".stApp .eu-state-status-row .passed" in css_text


def test_workspace_state_preview_copy_tracks_selected_context() -> None:
    html = pages_redesign._workspace_state_preview_html("agent", "real", "error", "en")

    assert "Research Agent" in html
    assert "Real Data" in html
    assert "Run failed at analysis step" in html
    assert "LinAlgError: singular matrix" in html
    assert "Patient Review" not in html


def test_workspace_state_primary_action_routes_selected_context() -> None:
    assert pages_redesign._workspace_state_action_label("patient", "demo", "en") == "Open Patient Review demo"
    assert pages_redesign._workspace_state_action_label("crossdb", "real", "en") == "Open Cross-DB setup"
    assert pages_redesign._workspace_state_action_label("agent", "real", "en") == "Open Research Agent setup"

    patient_demo: dict[str, object] = {}
    pages_redesign._apply_workspace_state_action(patient_demo, "patient", "demo")
    assert patient_demo["_active_main_page"] == "quick_viz"
    assert patient_demo["quick_viz_active_panel"] == "data_tables"
    assert patient_demo["_eu_topbar_run_request"] == {
        "page": "quick_viz",
        "requested_at": "workspace_states",
    }
    assert patient_demo["entry_mode"] == "demo"
    assert patient_demo["use_mock_data"] is True

    crossdb_demo: dict[str, object] = {}
    pages_redesign._apply_workspace_state_action(crossdb_demo, "crossdb", "demo")
    assert crossdb_demo["_active_main_page"] == "cross_db"
    assert crossdb_demo["_eu_topbar_run_request"] == {
        "page": "cross_db",
        "requested_at": "workspace_states",
    }

    agent_real: dict[str, object] = {"database": "unknown"}
    pages_redesign._apply_workspace_state_action(agent_real, "agent", "real")
    assert agent_real["_active_main_page"] == "research_agent"
    assert agent_real["_ra_view"] == "setup"
    assert agent_real["entry_mode"] == "real"
    assert agent_real["use_mock_data"] is False
    assert agent_real["database"] == "miiv"
    assert "_eu_topbar_run_request" not in agent_real


def test_demo_entry_routes_to_data_extraction_before_visualization(tmp_path, monkeypatch) -> None:
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    at = streamlit_testing.AppTest.from_file(app.__file__)
    at.session_state["entry_lang_select"] = "EN"
    at.session_state["language"] = "en"
    at.run(timeout=60)
    at.button(key="_eu_entry_demo").click().run(timeout=60)
    assert at.session_state["_active_main_page"] == "extract"
    assert at.session_state["entry_mode"] == "demo"
    assert at.session_state["use_mock_data"] is True
    assert at.session_state["database"] == "mock"
    assert at.session_state["mock_params"]["n_patients"] == demo_data.LIGHTWEIGHT_DEMO_PATIENTS
    assert at.session_state["mock_params"]["hours"] == demo_data.LIGHTWEIGHT_DEMO_HOURS
    assert at.session_state["loaded_data_origin"] == "none"
    assert at.session_state["loaded_concepts"] == {}
    assert at.session_state["patient_ids"] == []
    markdown_text = " ".join(getattr(markdown, "value", "") for markdown in at.markdown)
    assert "Generating lightweight demo data" not in markdown_text
    warning_text = " ".join(getattr(warning, "value", "") for warning in at.warning)
    assert "Dashboard rendering failed" not in warning_text
    assert "time_candidates" not in warning_text


def test_entry_no_data_cta_routes_to_real_agent_extraction_setup(tmp_path, monkeypatch) -> None:
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    at = streamlit_testing.AppTest.from_file(app.__file__)
    at.session_state["entry_lang_select"] = "EN"
    at.session_state["language"] = "en"
    at.session_state["research_agent_resume_run_id"] = "run_old"
    at.session_state["research_agent_force_manuscript"] = True
    at.session_state["research_agent_preflight_confirmed"] = True
    at.run(timeout=60)
    at.button(key="_eu_entry_nodata").click().run(timeout=60)

    assert at.session_state["_active_main_page"] == "research_agent"
    assert at.session_state["_ra_view"] == "setup"
    assert at.session_state["entry_mode"] == "real"
    assert at.session_state["use_mock_data"] is False
    assert at.session_state["database"] == "miiv"
    assert at.session_state["research_agent_cohort_source"] == i18n.TEXTS["en"]["ra_source_no_data"]
    assert at.session_state["_eu_ra_focus_no_data"] is True
    assert "_eu_ra_no_data_entry" not in at.session_state
    assert "research_agent_resume_run_id" not in at.session_state
    assert "research_agent_force_manuscript" not in at.session_state
    assert at.session_state["research_agent_preflight_confirmed"] is False

    page_text = " ".join(
        getattr(element, "value", "")
        for collection in (at.markdown, at.info, at.warning)
        for element in collection
    )
    assert "Demo Mode is a lightweight preview" not in page_text
    assert "Use this when you start from raw ICU tables" in page_text


def test_topbar_render_loads_quick_preview_from_demo_mode(tmp_path, monkeypatch) -> None:
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    at = streamlit_testing.AppTest.from_file(app.__file__)
    at.session_state["entry_lang_select"] = "EN"
    at.session_state["language"] = "en"
    at.run(timeout=60)
    at.button(key="_eu_entry_demo").click().run(timeout=60)
    at.session_state["_active_main_page"] = "quick_viz"
    at.session_state["loaded_concepts"] = {}
    at.session_state["loaded_data_origin"] = "none"
    at.session_state["patient_ids"] = []
    at.run(timeout=60)

    warning_text = " ".join(getattr(warning, "value", "") for warning in at.warning)
    assert "preview_n_patients" not in warning_text
    assert "_eu_topbar_run" in {button.key for button in at.button}

    at.button(key="_eu_topbar_run").click().run(timeout=60)

    assert at.session_state["loaded_data_origin"] == "demo_viz"
    assert len(at.session_state["loaded_concepts"]) > 0
    assert len(at.session_state["loaded_concepts"]) < 90
    assert len(at.session_state["patient_ids"]) == demo_data.LIGHTWEIGHT_DEMO_PATIENTS
    success_text = " ".join(getattr(success, "value", "") for success in at.success)
    assert "Loaded demo review workspace" not in success_text
    assert "Demo review workspace loaded automatically" not in success_text
    info_text = " ".join(getattr(info, "value", "") for info in at.info)
    assert "Preview request received" not in info_text


def test_real_data_mode_requires_data_path_before_validation(tmp_path, monkeypatch) -> None:
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    at = streamlit_testing.AppTest.from_file(app.__file__)
    at.session_state["entry_lang_select"] = "EN"
    at.session_state["language"] = "en"
    at.run(timeout=60)
    at.button(key="_eu_entry_real").click().run(timeout=60)
    at.session_state["_active_main_page"] = "extract"
    at.run(timeout=60)
    at.button(key="validate_path").click().run(timeout=60)

    assert any(error.value == "Please enter a data path" for error in at.error)


def test_validate_data_path_is_accent_secondary_action() -> None:
    source = Path(sidebar.__file__).read_text(encoding="utf-8")
    css_text = shell_styles._load_shell_overrides_css()
    validate_source = source[
        source.index("# 验证按钮"):
        source.index("elif not Path(effective_data_path).exists():")
    ]
    validate_css = css_text[
        css_text.index('.stApp [class*="st-key-validate_path"] button {'):
        css_text.index("/* -------------------------------------------------------------------------- */", css_text.index('.stApp [class*="st-key-validate_path"] button {'))
        if "/* -------------------------------------------------------------------------- */" in css_text[css_text.index('.stApp [class*="st-key-validate_path"] button {'):]
        else len(css_text)
    ]

    assert 'key="validate_path"' in validate_source
    assert 'icon=":material/search:"' in validate_source
    assert 'type="primary"' not in validate_source
    assert "🔍 Validate Data Path" not in validate_source
    assert "validate_spacer, validate_action = st.columns([5, 1.8], gap=\"small\")" in validate_source
    assert "background: var(--accent-soft)" in validate_css
    assert "color: var(--accent-ink)" in validate_css
    assert "stBaseButton-primary" not in validate_css


def test_disabled_primary_buttons_use_disabled_visual_state() -> None:
    css_text = shell_styles._load_shell_overrides_css()

    assert '.stApp .stButton > button[kind="primary"]:disabled' in css_text
    assert '.stApp button[data-testid="stBaseButton-primary"]:disabled' in css_text
    disabled_primary_css = css_text[
        css_text.index('.stApp .stButton > button[kind="primary"]:disabled'):
        css_text.index('.stApp button:disabled *')
    ]
    assert "background: var(--surface-2)" in disabled_primary_css
    assert "border-color: var(--hair)" in disabled_primary_css
    assert "color: var(--ink-4)" in disabled_primary_css


def test_convertible_path_messages_match_convert_setup_button_label() -> None:
    source = Path(data_workflows.__file__).read_text(encoding="utf-8")

    assert 'Convert & Setup" to auto-convert' in source
    assert 'Convert & Setup to auto-convert' in source
    assert "Click Validate & Setup" not in source
    assert '"Validate & Setup"' not in source
    assert "点击「验证并设置」自动转换" not in source


def test_module_preview_metadata_highlights_renal_story() -> None:
    preview = app._build_module_preview_metadata(
        module_key="renal",
        selected_module="🚰 Renal & Urine Output",
        module_concepts=[
            "aki",
            "aki_stage",
            "aki_stage_creat",
            "aki_stage_uo",
            "creat_low_past_48hr",
            "rrt",
            "uo_6h",
            "uo_12h",
            "urine24",
        ],
        lang="en",
    )

    assert "AKI staging" in preview["summary"]
    assert preview["tags"][:6] == [
        "aki",
        "aki_stage",
        "aki_stage_creat",
        "aki_stage_uo",
        "rrt",
        "uo_6h",
    ]


def test_select_preview_columns_prioritizes_representative_module_columns() -> None:
    merged_df = pd.DataFrame(
        columns=[
            "stay_id",
            "charttime",
            "urine24",
            "aki_stage_uo",
            "aki",
            "uo_rt_24hr",
            "rrt",
            "aki_stage",
            "uo_12h",
            "creat_low_past_48hr",
            "uo_6h",
            "aki_stage_creat",
            "extra_flag",
        ]
    )

    preview_columns = app._select_preview_columns(
        merged_df,
        module_key="renal",
        module_concepts=[
            "aki",
            "aki_stage",
            "aki_stage_creat",
            "aki_stage_uo",
            "rrt",
            "uo_6h",
            "uo_12h",
            "creat_low_past_48hr",
            "urine24",
        ],
        id_col="stay_id",
        max_columns=10,
    )

    assert preview_columns[:5] == [
        "stay_id",
        "charttime",
        "aki",
        "aki_stage",
        "aki_stage_creat",
    ]
    assert "uo_6h" in preview_columns
    assert "uo_12h" in preview_columns
    assert len(preview_columns) == 10


def test_load_from_exported_surfaces_parquet_compatibility_failure(monkeypatch, tmp_path) -> None:
    parquet_file = tmp_path / "renal_preview.parquet"
    parquet_file.write_bytes(b"PAR1")

    streamlit_stub = _WarningCaptureStreamlit()

    def _raise_parquet_failure(*_args, **_kwargs):
        raise OSError("Repetition level histogram size mismatch")

    monkeypatch.setattr(app, "st", streamlit_stub)
    monkeypatch.setattr(app.pd, "read_parquet", _raise_parquet_failure)
    monkeypatch.setattr(app, "_get_pyarrow_version", lambda: "19.0.0")
    monkeypatch.setattr(app, "_get_parquet_created_by", lambda _path: "parquet-cpp-arrow version 23.0.1")

    app.load_from_exported(str(tmp_path), max_patients=None, selected_files=[parquet_file.stem])

    assert len(streamlit_stub.warnings) == 1
    assert "failed to read" in streamlit_stub.warnings[0]
    assert "pyarrow=19.0.0" in streamlit_stub.warnings[0]
    assert "23.0.1" in streamlit_stub.warnings[0]


def test_data_table_page_copy_emphasizes_preview_language() -> None:
    english = app._get_data_table_page_copy("en")
    chinese = app._get_data_table_page_copy("zh")

    assert english["title"] == "Module Table Preview"
    assert "Preview loaded tables by module" in english["description"]
    assert chinese["title"] == "模块数据预览"


def test_quick_viz_data_table_overlap_guards_are_in_shell_css() -> None:
    css = shell_styles._load_shell_overrides_css()

    assert ".dt-page-head" in css
    assert "st-key-dt_preview_controls" in css
    assert "st-key-dt_preview_mode" in css
    assert "st-key-dt_preview_summary" in css


def test_patient_feature_snapshot_caption_keeps_bottom_buffer() -> None:
    css = shell_styles._load_shell_overrides_css()
    caption_block = re.search(
        r"\.stApp \.patient-feature-snapshot-caption \{(?P<body>.*?)\n\}",
        css,
        re.S,
    )

    assert caption_block is not None
    assert "margin: 8px 0 32px !important" in caption_block.group("body")


def test_quick_viz_loader_header_stacks_on_mobile() -> None:
    css = shell_styles._load_shell_overrides_css()

    start = css.index("Quick Visualization loader: the reference Patient Review idle state")
    mobile_loader_css = css[start: css.index(".stApp .eu-qv-export-recovery", start)]

    assert "@media (max-width: 640px)" in mobile_loader_css
    assert ".stApp .eu-qv-loader-head" in mobile_loader_css
    assert "display: block !important" in mobile_loader_css
    assert ".stApp .eu-qv-loader-badge" in mobile_loader_css
    assert "margin-top: 10px !important" in mobile_loader_css
    assert "white-space: normal !important" in mobile_loader_css


def test_quick_viz_loader_data_source_uses_reference_pill_radios() -> None:
    css = shell_styles._load_shell_overrides_css()

    assert 'st-key-viz_data_source_mode"] div[role="radiogroup"]' in css
    source_mode_css = css[
        css.index('.stApp [class*="st-key-viz_data_source_mode"] div[role="radiogroup"] {'):
        css.index("/* Quick Visualization loader:", css.index("st-key-viz_data_source_mode"))
    ]

    assert "grid-template-columns: repeat(2, minmax(0, 1fr)) !important" in source_mode_css
    assert "border-radius: var(--r-pill) !important" in source_mode_css
    assert "label:has(input:checked)" in source_mode_css
    assert "background: var(--accent-soft) !important" in source_mode_css
    assert "-webkit-text-fill-color: var(--accent-ink) !important" in source_mode_css
    assert (
        '.stApp [class*="st-key-viz_data_source_mode"] div[role="radiogroup"] {\n'
        "    grid-template-columns: 1fr !important;"
    ) in css


def test_data_table_preview_controls_are_compact_toolbar() -> None:
    source = Path(data_table_page.__file__).read_text()

    assert 'key="dt_preview_controls"' in source
    assert 'key="dt_preview_mode"' in source
    assert 'key="merge_max_rows"' in source
    assert 'key="single_feature_max_rows"' in source
    assert "st.columns([2.15, 0.85]" in source


def test_data_table_preview_has_easyicu_owned_csv_downloads() -> None:
    source = Path(data_table_page.__file__).read_text()

    assert "def _render_preview_csv_download(" in source
    assert "st.download_button(" in source
    assert "Download preview CSV" in source
    assert "下载预览 CSV" in source
    assert 'mime="text/csv"' in source
    assert ".to_csv(index=False).encode(\"utf-8\")" in source
    assert "easyicu_" in source
    assert "data_table_single_feature_csv_" in source
    assert "data_table_merged_preview_csv_" in source


def test_single_feature_preview_copy_matches_preview_style() -> None:
    english = app._get_single_feature_preview_copy("sofa", "en")
    chinese = app._get_single_feature_preview_copy("sofa", "zh")

    assert english["title"] == "Single Feature Preview"
    assert "Inspect `sofa`" in english["description"]
    assert chinese["title"] == "单特征预览"


def test_select_timeseries_screenshot_concepts_prefers_representative_clinical_series() -> None:
    selected = app._select_timeseries_screenshot_concepts(
        ["dbp", "wbc", "spo2", "hr", "sofa2", "map", "crea"],
        max_items=4,
    )

    assert selected == ["hr", "map", "spo2", "crea"]


def test_select_quality_distribution_concept_prefers_interpretable_lab_over_score() -> None:
    loaded_concepts = {
        "sofa": pd.DataFrame({"stay_id": [1], "sofa": [3]}),
        "crea": pd.DataFrame({"stay_id": [1], "crea": [1.2]}),
        "hr": pd.DataFrame({"stay_id": [1], "hr": [88]}),
    }

    assert app._select_quality_distribution_concept(loaded_concepts) == "crea"


def test_normalize_figure_target_maps_short_urls_to_panels() -> None:
    assert app._normalize_figure_target("figure2") == ("paper", "Figure 2")
    assert app._normalize_figure_target("fig3") == ("paper", "Figure 3")
    assert app._normalize_figure_target("figure4") == ("paper", "Figure 4")
    assert app._normalize_figure_target("s1") == ("paper", "Supplementary Figure S1")
    assert app._normalize_figure_target("coverage") == ("cohort", "Coverage Audit")
    assert app._normalize_figure_target("cross-db") == ("cross_db", "Cross-DB Benchmark")
    assert app._normalize_figure_target("quality") == ("viz", "Data Quality")
    assert app._normalize_figure_target("1") == ("", "")
    assert app._normalize_figure_target("unknown") == ("", "")


def test_ensure_cohort_demo_workspace_bootstraps_all_cohort_panels() -> None:
    state = {"mock_params": {"n_patients": 25}}

    assert app._cohort_demo_workspace_ready(state) is False

    app._ensure_cohort_demo_workspace(state, lang="en")

    assert app._cohort_demo_workspace_ready(state) is True
    assert state["grp_is_demo"] is True
    assert state["dash_is_demo"] is True
    assert state["multidb_is_demo"] is True
    assert state["cohort_is_demo"] is True
    assert len(state["grp_demographics"]) == demo_data.COHORT_DEMO_PATIENTS
    assert not state["dash_demographics"].empty
    assert set(state["multidb_data"].keys()) == set(demo_data.COHORT_DEMO_MULTIDB_DATABASES)
    assert len(state["multidb_data"]) == 6
    assert state["multidb_concepts"] == ["hr", "sbp", "map", "temp", "spo2", "lact"]
    assert all(len(frame) <= 6 * demo_data.COHORT_DEMO_MULTIDB_RECORDS_PER_FEATURE for frame in state["multidb_data"].values())


def test_lightweight_demo_data_keeps_review_surface_small() -> None:
    data, patient_ids = demo_data.generate_lightweight_demo_data(n_patients=50, hours=48)

    assert len(patient_ids) == 50
    assert 40 <= len(data) < 90
    assert {"hr", "map", "sofa2", "sep3_sofa2", "aki", "death", "los_icu"} <= set(data)
    assert sum(len(frame) for frame in data.values()) < 50_000
    assert hasattr(demo_data.generate_lightweight_demo_data, "clear")
    assert hasattr(demo_data._generate_mock_multidb_data, "clear")


def test_lightweight_demo_data_keeps_default_group_contrast_comparable() -> None:
    data, patient_ids = demo_data.generate_lightweight_demo_data(n_patients=10, hours=48)

    deaths = data["death"]["death"].astype(int)

    assert len(patient_ids) == 10
    assert int(deaths.sum()) >= 2
    assert int((deaths == 0).sum()) >= 2


def test_cohort_demo_workspace_default_group_contrast_has_both_groups() -> None:
    state: dict[str, object] = {}

    app._ensure_cohort_demo_workspace(state, lang="en", force=True)

    survived = state["grp_demographics"]["survived"].astype(int)
    assert int((survived == 1).sum()) >= 2
    assert int((survived == 0).sum()) >= 2


def test_existing_tiny_demo_group_contrast_repairs_degenerate_survival_split() -> None:
    state: dict[str, object] = {
        "mock_params": {"n_patients": 10},
        "grp_is_demo": True,
        "grp_demographics": pd.DataFrame(
            {
                "stay_id": range(10001, 10011),
                "age": [45, 52, 60, 66, 72, 80, 58, 69, 75, 83],
                "gender": ["M", "F"] * 5,
                "survived": [1] * 10,
                "death": [0] * 10,
                "sofa_max": [1, 2, 3, 4, 12, 10, 3, 8, 9, 11],
            }
        ),
    }

    app._ensure_cohort_demo_workspace(state, lang="en")

    survived = state["grp_demographics"]["survived"].astype(int)
    death = state["grp_demographics"]["death"].astype(int)
    assert int((survived == 1).sum()) >= 2
    assert int((survived == 0).sum()) >= 2
    assert death.tolist() == (1 - survived).tolist()


def test_ensure_cohort_real_workspace_syncs_global_review_state(tmp_path, monkeypatch) -> None:
    class _FakePatientFilter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def _load_demographics(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "stay_id": [39553978, 39553979],
                    "age": [64, 72],
                    "gender": ["M", "F"],
                    "survived": [1, 0],
                }
            )

    fake_pf_module = types.ModuleType("easyicu.patient_filter")
    fake_pf_module.PatientFilter = _FakePatientFilter
    monkeypatch.setitem(sys.modules, "easyicu.patient_filter", fake_pf_module)

    def _fake_load_concepts(*_args, **_kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "stay_id": [39553978, 39553979],
                "charttime": [0, 0],
                "hr": [88, 92],
                "sofa": [3, 8],
            }
        )

    monkeypatch.setattr(easyicu, "load_concepts", _fake_load_concepts)
    monkeypatch.setattr(cohort_workspace, "_default_real_database", lambda: "miiv")
    monkeypatch.setattr(cohort_workspace, "_default_real_data_root", lambda: str(tmp_path))
    monkeypatch.setattr(cohort_workspace, "find_database_path", lambda data_path, _database: data_path)
    monkeypatch.setattr(
        cohort_filters,
        "st",
        _SessionStateStreamlit(
            {
                "sepsis_si_mode": "auto",
                "sepsis_positive_cultures": False,
                "sepsis_abx_win_hours": 24,
                "sepsis_samp_win_hours": 72,
            }
        ),
    )

    state = {
        "patient_ids": [10001],
        "selected_patient": 10001,
        "loaded_concepts": {"old_demo": pd.DataFrame({"stay_id": [10001], "old_demo": [1]})},
        "loaded_data_origin": "demo_viz",
    }

    ok, message = app._ensure_cohort_real_workspace(state, max_patients=1000)

    assert ok is True
    assert "Loaded 2 patients" in message
    assert state["patient_ids"] == [39553978, 39553979]
    assert state["available_patient_ids"] == [39553978, 39553979]
    assert state["all_patient_count"] == 2
    assert state["selected_patient"] == 39553978
    assert state["loaded_data_origin"] == "real_workspace"
    assert set(state["loaded_concepts"]) == {"hr", "sofa"}


def test_snapshot_real_loader_seeds_concepts_not_demographics_only() -> None:
    with open(cohort_dashboard_page.__file__, encoding="utf-8") as handle:
        source = handle.read()

    raw_loader_block = source[
        source.index('elif dash_src == "raw":'):
        source.index('elif dash_src == "exported":')
    ]

    assert "_bundle_from_raw_schema(" in raw_loader_block
    assert "load_concepts=True" in raw_loader_block
    assert "_seed_workspace_state(st.session_state, bundle)" in raw_loader_block
    assert "PatientFilter(database=selected_db" not in raw_loader_block
    assert "正在加载快照所需的人口统计、表型与 SOFA" in raw_loader_block


def test_cohort_dashboard_review_stats_uses_loaded_concepts_for_real_snapshot() -> None:
    cohort = pd.DataFrame(
        {
            "stay_id": [10, 11, 12],
            "age": [64, 72, 81],
            "death": [0, 1, 1],
            "los_icu": [2.0, 4.0, 6.0],
        }
    )
    loaded_concepts = {
        "sofa": pd.DataFrame({"stay_id": [10, 11, 12], "sofa": [2, 7, 10]}),
        "aki": pd.DataFrame({"stay_id": [10, 11, 12], "aki": [0, 1, 1]}),
        "mech_vent": pd.DataFrame({"stay_id": [10, 11, 12], "mech_vent": [0, 1, 1]}),
        "abx": pd.DataFrame({"stay_id": [10, 11, 12], "abx": [1, 1, 0]}),
    }

    review = app._build_cohort_dashboard_review_stats(
        cohort,
        loaded_concepts=loaded_concepts,
        lang="en",
    )

    assert review["metrics"]["median_sofa"] == "7.0"
    assert review["metrics"]["phenotype_burden"] != "NA"
    assert not review["phenotype"].empty
    assert int(review["severity"]["patients"].sum()) == 3
    assert int(review["coverage"]["features"].sum()) == len(loaded_concepts)


def test_real_cohort_page_keeps_panel_import_paths_visible_before_shared_workspace(
    tmp_path, monkeypatch
) -> None:
    class _FakePanel:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeStreamlit:
        def __init__(self) -> None:
            self.session_state = {
                "language": "en",
                "entry_mode": "real",
                "cohort_active_panel": "coverage",
            }
            self.radio_labels: list[str] = []

        def markdown(self, *_args, **_kwargs) -> None:
            pass

        def warning(self, *_args, **_kwargs) -> None:
            pass

        def slider(self, *_args, **_kwargs) -> int:
            return 1000

        def button(self, *_args, **_kwargs) -> bool:
            return False

        def columns(self, spec):
            return [_FakePanel() for _ in spec]

        def radio(self, _label, *, options, key, format_func=None, **_kwargs):
            assert options == ["groups", "coverage", "snapshot", "sofa"]
            self.radio_labels = [format_func(option) for option in options]
            return self.session_state[key]

    streamlit_stub = _FakeStreamlit()
    rendered_panels: list[str] = []

    monkeypatch.setattr(app, "st", streamlit_stub)
    monkeypatch.setattr(app, "_default_real_data_root", lambda: str(tmp_path))
    monkeypatch.setattr(app, "_default_real_database", lambda: "miiv")
    monkeypatch.setattr(app, "render_group_comparison_subtab", lambda _lang: rendered_panels.append("groups"))
    monkeypatch.setattr(app, "render_data_coverage_audit_subtab", lambda _lang: rendered_panels.append("coverage"))
    monkeypatch.setattr(app, "render_multidb_distribution_subtab", lambda _lang: rendered_panels.append("crossdb"))
    monkeypatch.setattr(app, "render_cohort_dashboard_subtab", lambda _lang: rendered_panels.append("snapshot"))
    monkeypatch.setattr(app, "render_severity_reclassification_subtab", lambda _lang: rendered_panels.append("sofa"))

    app.render_cohort_comparison_page()

    assert streamlit_stub.radio_labels == [
        "Group contrast",
        "Coverage audit",
        "Cohort profile",
        "SOFA reclassification",
    ]
    assert rendered_panels == ["coverage"]


def test_real_cohort_page_gates_sub_tabs_when_no_data_path_validated(monkeypatch) -> None:
    class _FakePanel:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeStreamlit:
        def __init__(self) -> None:
            self.session_state = {"language": "en", "entry_mode": "real"}
            self.radio_called = False

        def markdown(self, *_args, **_kwargs) -> None:
            pass

        def warning(self, *_args, **_kwargs) -> None:
            pass

        def info(self, *_args, **_kwargs) -> None:
            pass

        def columns(self, spec):
            return [_FakePanel() for _ in spec]

        def radio(self, *_args, **_kwargs):
            self.radio_called = True
            return "groups"

    streamlit_stub = _FakeStreamlit()
    rendered_panels: list[str] = []

    monkeypatch.setattr(app, "st", streamlit_stub)
    # No validated data path → page must show one guide and gate everything.
    monkeypatch.setattr(app, "_default_real_data_root", lambda: "")
    monkeypatch.setattr(app, "_default_real_database", lambda: "miiv")
    monkeypatch.setattr(app, "render_group_comparison_subtab", lambda _lang: rendered_panels.append("groups"))
    monkeypatch.setattr(app, "render_data_coverage_audit_subtab", lambda _lang: rendered_panels.append("coverage"))
    monkeypatch.setattr(app, "render_multidb_distribution_subtab", lambda _lang: rendered_panels.append("crossdb"))
    monkeypatch.setattr(app, "render_cohort_dashboard_subtab", lambda _lang: rendered_panels.append("snapshot"))
    monkeypatch.setattr(app, "render_severity_reclassification_subtab", lambda _lang: rendered_panels.append("sofa"))

    app.render_cohort_comparison_page()

    assert streamlit_stub.radio_called is False
    assert rendered_panels == []


def test_real_workspace_launcher_defaults_to_fast_import_preview(tmp_path, monkeypatch) -> None:
    class _FakePanel:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeStreamlit:
        def __init__(self) -> None:
            self.markdown_calls: list[str] = []
            self.slider_kwargs = None

        def warning(self, *_args, **_kwargs) -> None:
            pass

        def markdown(self, body, *_args, **_kwargs) -> None:
            self.markdown_calls.append(body)

        def columns(self, spec):
            return [_FakePanel() for _ in spec]

        def slider(self, *_args, **kwargs) -> int:
            self.slider_kwargs = kwargs
            return kwargs["value"]

        def button(self, *_args, **_kwargs) -> bool:
            return False

    streamlit_stub = _FakeStreamlit()

    monkeypatch.setattr(app, "st", streamlit_stub)
    monkeypatch.setattr(app, "_default_real_data_root", lambda: str(tmp_path))
    monkeypatch.setattr(app, "_default_real_database", lambda: "miiv")

    app._render_cohort_real_workspace_launcher("en")

    assert streamlit_stub.slider_kwargs["value"] == 100
    assert "fast import check" in streamlit_stub.slider_kwargs["help"]
    assert any("Quick preview" in call for call in streamlit_stub.markdown_calls)


def test_validate_database_path_resolves_parent_root_with_bucketed_miiv(tmp_path) -> None:
    db_path = tmp_path / "mimic-iv-3.1"
    for subdir in ("hosp", "icu"):
        (db_path / subdir).mkdir(parents=True)

    flat_tables = {
        "hosp": ["admissions", "patients", "prescriptions", "d_labitems"],
        "icu": ["icustays", "outputevents", "ingredientevents", "procedureevents", "d_items"],
    }
    for subdir, tables in flat_tables.items():
        for table in tables:
            (db_path / subdir / f"{table}.parquet").touch()

    for table in ("chartevents", "labevents", "inputevents"):
        bucket_dir = db_path / f"{table}_bucket" / "bucket_id=0"
        bucket_dir.mkdir(parents=True)
        (bucket_dir / "data_0.parquet").touch()

    result = app.validate_database_path(str(tmp_path), "miiv")

    assert result["valid"] is True
    assert "MIMIC-IV" in result["message"]
    assert result["resolved_path"] == str(db_path)


def test_real_data_validation_uses_resolved_database_path() -> None:
    result = {"valid": True, "resolved_path": "/data/mimic-iv-3.1"}
    assert sidebar._validation_resolved_path(result, "/data") == "/data/mimic-iv-3.1"

    convert_result = {"valid": False, "can_convert": True, "csv_path": "/data/eicu-csv"}
    assert sidebar._validation_resolved_path(convert_result, "/data") == "/data/eicu-csv"


def test_real_data_validation_status_accepts_resolved_child_path(tmp_path: Path) -> None:
    parent = tmp_path / "mimic-parent"
    resolved = parent / "mimic-iv-3.1"
    resolved.mkdir(parents=True)
    state = {
        "path_validated": True,
        "data_path": str(resolved),
        "last_validated_path": str(resolved),
        "last_validation": {"valid": True, "resolved_path": str(resolved)},
    }

    assert sidebar._current_input_matches_validated_data_path(state, str(parent)) is True

    stale_state = dict(state)
    stale_state["data_path"] = str(tmp_path / "other-mimic")
    assert sidebar._current_input_matches_validated_data_path(stale_state, str(parent)) is False

    assert sidebar._validation_resolved_path({}, "/fallback") == "/fallback"


def test_sidebar_data_path_updates_are_queued_for_next_rerun(monkeypatch) -> None:
    streamlit_stub = _SessionStateStreamlit(
        _AttrSessionState(
            {
                "sidebar_data_path_input": "/typed/path",
            }
        )
    )
    monkeypatch.setattr(sidebar, "st", streamlit_stub)

    sidebar._queue_sidebar_data_path_input("/resolved/mimic-iv-3.1")

    assert streamlit_stub.session_state["sidebar_data_path_input"] == "/typed/path"
    assert streamlit_stub.session_state["sidebar_data_path_input__pending_value"] == "/resolved/mimic-iv-3.1"


def test_apply_quick_viz_screenshot_defaults_focuses_figure_friendly_views() -> None:
    state = {
        "patient_ids": [101, 202],
        "loaded_concepts": {
            "sofa": pd.DataFrame({"stay_id": [101], "sofa": [3]}),
            "crea": pd.DataFrame({"stay_id": [101], "crea": [1.1]}),
        },
    }

    app._apply_quick_viz_screenshot_defaults(state, lang="en")

    assert state["lane_patient_select"] == 101
    assert state["patient_view_id"] == 101
    assert state["ts_mode"] == "Clinical Lanes"
    assert state["patient_view_mode"] == "Dashboard"
    assert state["missing_chart_sort_order"] == "desc"
    assert state["quality_concept"] == "crea"
    assert state["data_table_view_mode"] == "Merge All (Wide Table)"


def test_resolve_viz_data_source_mode_keeps_session_state_valid_without_widget_index() -> None:
    assert app._resolve_viz_data_source_mode(
        current_mode="invalid",
        recent_export_path="",
        allow_demo=True,
        entry_mode="demo",
    ) == "demo"

    assert app._resolve_viz_data_source_mode(
        current_mode="demo",
        recent_export_path="/tmp/export",
        allow_demo=False,
        entry_mode="real",
    ) == "exported"


def test_quick_viz_export_recovery_finds_recent_export_folders(tmp_path) -> None:
    last_export = tmp_path / "external_export"
    last_export.mkdir()
    (last_export / "vitals.parquet").write_text("stub", encoding="utf-8")

    default_root = tmp_path / "easyicu_export"
    default_root.mkdir()
    recent_child = default_root / "exports_20260526"
    recent_child.mkdir()
    (recent_child / "labs.csv").write_text("stub", encoding="utf-8")
    empty_child = default_root / "empty"
    empty_child.mkdir()

    state = {
        "last_export_dir": str(last_export),
        "export_path": str(default_root),
    }

    candidates = quick_visualization_page._quick_viz_export_candidates(state, limit=6)
    candidate_paths = [item["path"] for item in candidates]

    assert candidate_paths[0] == str(last_export)
    assert str(recent_child) in candidate_paths
    assert str(empty_child) not in candidate_paths
    assert {item["path"]: item["file_count"] for item in candidates}[str(last_export)] == 1

    quick_visualization_page._apply_quick_viz_export_candidate(state, str(recent_child))
    assert state["viz_export_path"] == str(recent_child)
    assert state["viz_export_path_input"] == str(recent_child)
    assert "viz_data_source_mode" not in state


def test_quick_viz_loader_surfaces_export_path_recovery_controls() -> None:
    source = Path(quick_visualization_page.__file__).read_text(encoding="utf-8")
    css_text = shell_styles._load_shell_overrides_css()

    assert "_render_quick_viz_export_path_recovery(lang)" in source
    assert "Not sure where the export is?" in source
    assert "local folders only" in source
    assert "does not upload export history to GitHub" in source
    assert "不会把导出历史上传到 GitHub" in source
    assert "viz_use_export_candidate_" in source
    assert "viz_use_default_export_root" in source
    assert ".eu-qv-export-recovery" in css_text
    assert "st-key-viz_use_export_candidate_" in css_text


def test_local_user_history_outputs_are_gitignored() -> None:
    repo_root = Path(easyicu.__file__).parents[2]
    gitignore = (repo_root / ".gitignore").read_text(encoding="utf-8")

    assert "Local user histories and private app outputs" in gitignore
    for pattern in [
        "/easyicu_export/",
        "/exports_*/",
        "/agent_runs/",
        "/easyicu_agent_runs/",
        "/run_history/",
        "/.easyicu/",
        "*.easyicu-history.json",
    ]:
        assert pattern in gitignore


def test_apply_screenshot_mode_ui_state_closes_floating_ai_and_clears_ai_jump() -> None:
    state = {
        "_floating_ai_open": True,
        "_sidebar_ai_open": True,
        "_eu_sidebar_settings_open": True,
        "_inline_ai_panel_open": True,
        "_scroll_to_tab": "ai_assistant",
    }

    app._apply_screenshot_mode_ui_state(state)

    assert state["_floating_ai_open"] is False
    assert state["_sidebar_ai_open"] is False
    assert state["_eu_sidebar_settings_open"] is False
    assert state["_inline_ai_panel_open"] is False
    assert "_scroll_to_tab" not in state


def test_clear_assistant_surfaces_removes_stale_inline_panel_state() -> None:
    state = {
        "_inline_ai_panel_open": True,
        "_floating_ai_open": True,
        "_sidebar_ai_open": True,
        "_ai_pending_question": "stale prompt",
    }

    app._clear_assistant_surfaces(state, clear_pending=True)

    assert state["_inline_ai_panel_open"] is False
    assert state["_floating_ai_open"] is False
    assert state["_sidebar_ai_open"] is False
    assert "_ai_pending_question" not in state


def test_open_embedded_ai_assistant_targets_standalone_page() -> None:
    state: dict[str, object] = {"llm_enabled": False, "_floating_ai_open": True}

    app._open_embedded_ai_assistant(state, "How should I configure SOFA?")

    assert state["llm_enabled"] is True
    assert state["_llm_toggle"] is True
    assert state["_active_main_page"] == "assistant"
    assert "_main_nav_widget" not in state
    assert state["_scroll_to_top"] is True
    assert state["_inline_ai_panel_open"] is False
    assert state["_sidebar_ai_open"] is False
    assert state["_floating_ai_open"] is False
    assert state["_ai_pending_question"] == "How should I configure SOFA?"


def test_ai_assistant_handoff_seeds_research_agent_question_without_overwriting() -> None:
    state: dict[str, object] = {
        "_ai_pending_question": "Does first-24h lactate improve sepsis mortality prediction?",
        "_inline_ai_panel_open": True,
        "_floating_ai_open": True,
        "_sidebar_ai_open": True,
        "research_agent_resume_run_id": "run_old",
        "research_agent_preflight_confirmed": True,
    }

    seeded = llm_chat._prepare_research_agent_handoff_from_ai(state)

    assert seeded is True
    assert state["_active_main_page"] == "research_agent"
    assert state["_ra_view"] == "setup"
    assert state["_scroll_to_top"] is True
    assert state["research_agent_question"] == (
        "Does first-24h lactate improve sepsis mortality prediction?"
    )
    assert state["_research_agent_question_handoff_notice"] is True
    assert "_ai_pending_question" not in state
    assert "research_agent_resume_run_id" not in state
    assert "research_agent_preflight_confirmed" not in state
    assert state["_inline_ai_panel_open"] is False
    assert state["_floating_ai_open"] is False
    assert state["_sidebar_ai_open"] is False

    existing_state: dict[str, object] = {
        "research_agent_question": "Keep this manually edited Agent question.",
        "llm_messages": [
            {"role": "assistant", "content": "I can help."},
            {"role": "user", "content": "Replace this if overwrite were allowed."},
        ],
    }

    seeded_existing = llm_chat._prepare_research_agent_handoff_from_ai(existing_state)

    assert seeded_existing is False
    assert existing_state["research_agent_question"] == "Keep this manually edited Agent question."
    assert "_research_agent_question_handoff_notice" not in existing_state


def test_topbar_assistant_open_agent_reuses_ai_handoff_seed() -> None:
    state: dict[str, object] = {
        "_eu_topbar_run_request": {"page": "assistant"},
        "_active_main_page": "assistant",
        "llm_messages": [
            {"role": "assistant", "content": "Let's frame the cohort."},
            {"role": "user", "content": "Does SOFA-2 trajectory predict ICU mortality?"},
        ],
        "research_agent_preflight_confirmed": True,
    }

    result = app._consume_topbar_run_request(state, "assistant", "en")

    assert result == {
        "level": "info",
        "message": "Opened Research Agent setup with the latest assistant question.",
    }
    assert state["_active_main_page"] == "research_agent"
    assert state["_ra_view"] == "setup"
    assert state["research_agent_question"] == "Does SOFA-2 trajectory predict ICU mortality?"
    assert state["_research_agent_question_handoff_notice"] is True
    assert "research_agent_preflight_confirmed" not in state
    assert "_ai_pending_question" not in state


def test_research_agent_external_llm_opt_in_defers_sidebar_toggle_sync() -> None:
    source = Path(research_agent.__file__).read_text(encoding="utf-8")
    opt_in_start = source.index("external_llm_selected =")
    opt_in_source = source[
        opt_in_start:
        source.index("request_ready =", opt_in_start)
    ]

    assert 'st.session_state["llm_enabled"] = True' in opt_in_source
    assert 'st.session_state["_llm_toggle_sync_pending"] = True' in opt_in_source
    assert 'st.session_state["_llm_toggle"] = True' not in opt_in_source


def test_research_agent_raw_extract_defaults_to_all_modules_and_hands_off() -> None:
    modules = {
        "demographics": ["age"],
        "outcome": ["death"],
        "sofa2_score": ["sofa2"],
        "sepsis3_sofa2": ["sep3_sofa2"],
        "vitals": ["hr"],
        "blood_gas": ["ph"],
        "renal": ["crea"],
    }

    assert research_agent._default_extract_module_selection(modules) == list(modules)
    assert research_agent._raw_extract_module_selection_for_preset(modules, "all") == list(modules)
    assert research_agent._raw_extract_module_selection_for_preset(modules, "core") == [
        "demographics",
        "outcome",
        "sofa2_score",
        "sepsis3_sofa2",
        "vitals",
        "blood_gas",
    ]
    assert research_agent._raw_extract_module_selection_for_preset(
        modules,
        "custom",
        ["outcome", "unknown", "renal"],
    ) == ["outcome", "renal"]

    legacy_state = {
        "research_agent_extract_modules": [
            "demographics",
            "outcome",
            "sofa2_score",
            "sepsis3_sofa2",
            "vitals",
            "blood_gas",
        ]
    }
    research_agent._migrate_legacy_extract_module_selection(legacy_state, modules)
    assert legacy_state["research_agent_extract_modules"] == list(modules)

    manual_subset = {"research_agent_extract_modules": ["demographics", "vitals"]}
    research_agent._migrate_legacy_extract_module_selection(manual_subset, modules)
    assert manual_subset["research_agent_extract_modules"] == ["demographics", "vitals"]

    source = Path(research_agent.__file__).read_text(encoding="utf-8")
    raw_extract_source = source[
        source.index("if source == source_no_data:"):
        source.index("    # Cross-DB cohort builder")
    ]
    assert "_default_extract_module_selection(modules)" in raw_extract_source
    assert "_raw_extract_module_selection_for_preset(" in raw_extract_source
    assert "_migrate_legacy_extract_module_selection(st.session_state, modules)" in raw_extract_source
    assert "_queue_raw_extract_handoff(" in raw_extract_source
    assert 'state["_active_main_page"] = "extract"' in source
    assert 'state["trigger_export"] = False' in source
    assert 'state["_exporting_in_progress"] = False' in source
    assert 'state["_scroll_to_top"] = True' in source
    assert "raw_path_exists = bool(data_path and Path(data_path).expanduser().exists())" in raw_extract_source
    assert "start_export_disabled = not picked_modules or (db != \"mock\" and not raw_path_exists)" in raw_extract_source
    assert 'st.caption(_ra_text("start_export_needs_path"))' in raw_extract_source


def test_research_agent_raw_extract_db_follows_active_data_source_until_manual_override() -> None:
    options = ["miiv", "mimic", "eicu", "aumc", "hirid", "sic", "mock"]

    state = {"database": "mimic"}
    assert research_agent._sync_extract_db_with_active_data_source(state, options) == "mimic"
    assert "research_agent_extract_db" not in state
    assert state["_research_agent_extract_db_source"] == "mimic"

    state["research_agent_extract_db"] = "mimic"
    state["database"] = "eicu"
    assert research_agent._sync_extract_db_with_active_data_source(state, options) == "eicu"
    assert "research_agent_extract_db" not in state
    assert state["_research_agent_extract_db_source"] == "eicu"

    state["research_agent_extract_db"] = "aumc"
    state["database"] = "hirid"
    assert research_agent._sync_extract_db_with_active_data_source(state, options) == "aumc"
    assert state["research_agent_extract_db"] == "aumc"
    assert state["_research_agent_extract_db_source"] == "eicu"

    old_session_state = {"database": "mimic", "research_agent_extract_db": "miiv"}
    assert research_agent._sync_extract_db_with_active_data_source(old_session_state, options) == "mimic"
    assert "research_agent_extract_db" not in old_session_state
    assert old_session_state["_research_agent_extract_db_source"] == "mimic"


def test_research_agent_raw_extract_copy_explains_handoff_and_full_default() -> None:
    en = i18n.TEXTS["en"]
    zh = i18n.TEXTS["zh"]

    assert "All available modules are selected by default" in en["ra_no_data_info"]
    assert "Data Extraction" in en["ra_no_data_info"]
    assert en["ra_start_export"] == "Open Data Extraction with these settings"
    assert "Enter a raw database path first" in en["ra_start_export_needs_path"]
    assert "Select at least one module" in en["ra_start_export_needs_modules"]
    assert "final review" in en["ra_export_queued"]
    assert en["ra_module_preset"] == "Module preset"
    assert en["ra_module_preset_all"] == "All modules"
    assert en["ra_module_preset_core"] == "Core quick set"
    assert en["ra_module_preset_custom"] == "Custom"
    assert "full context" in en["ra_module_preset_all_help"]
    assert "missing modules" in en["ra_module_preset_custom_help"]

    assert "默认选择全部可用模块" in zh["ra_no_data_info"]
    assert "数据提取" in zh["ra_no_data_info"]
    assert zh["ra_start_export"] == "用这些设置打开数据提取"
    assert "请先填写原始数据库路径" in zh["ra_start_export_needs_path"]
    assert "请至少选择一个模块" in zh["ra_start_export_needs_modules"]
    assert "最终复核" in zh["ra_export_queued"]
    assert zh["ra_module_preset"] == "模块预设"
    assert zh["ra_module_preset_all"] == "全部模块"
    assert zh["ra_module_preset_core"] == "核心快速集"
    assert zh["ra_module_preset_custom"] == "自定义"
    assert "完整上下文" in zh["ra_module_preset_all_help"]
    assert "缺失模块" in zh["ra_module_preset_custom_help"]


def test_research_agent_crossdb_requires_distinct_database_tags() -> None:
    duplicate_exports = [
        ("miiv", Path("/tmp/easyicu_export/miiv_20260427")),
        ("miiv", Path("/tmp/easyicu_export/miiv_20260502")),
    ]
    distinct_exports = [
        ("miiv", Path("/tmp/easyicu_export/miiv_20260427")),
        ("eicu", Path("/tmp/easyicu_export/eicu_20260502")),
    ]

    assert research_agent._duplicate_db_tags(duplicate_exports) == ["miiv"]
    assert research_agent._has_min_distinct_db_tags(duplicate_exports, min_count=2) is False
    assert research_agent._multi_db_label_is_distinct("multi_db:miiv,miiv") is False
    assert research_agent._multi_db_label_is_distinct("multi_db:miiv") is False
    assert research_agent._duplicate_db_tags(distinct_exports) == []
    assert research_agent._has_min_distinct_db_tags(distinct_exports, min_count=2) is True
    assert research_agent._multi_db_label_is_distinct("multi_db:eicu,miiv") is True

    en = i18n.TEXTS["en"]
    zh = i18n.TEXTS["zh"]
    assert "different database tag" in en["ra_multi_db_duplicate_tags"]
    assert "不同数据库标签" in zh["ra_multi_db_duplicate_tags"]

    source = Path(research_agent.__file__).read_text(encoding="utf-8")
    picker_source = source[
        source.index("def _render_db_exports_multipicker"):
        source.index("def _build_multi_db_cohort")
    ]
    build_source = source[
        source.index("def _build_multi_db_cohort"):
        source.index("def _section_cohort_picker")
    ]
    page_source = source[
        source.index("def render_research_agent_page"):
        source.index("external_llm_selected =", source.index("def render_research_agent_page"))
    ]
    assert "_duplicate_db_tags(chosen)" in picker_source
    assert "_clear_research_agent_preflight_confirmation()" in picker_source
    assert "_duplicate_db_tags(chosen)" in build_source
    assert "_has_min_distinct_db_tags(chosen, min_count=2)" in build_source
    assert "loaded_unique_tags" in build_source
    assert "multi_db_loaded_need_distinct" in build_source
    assert "_multi_db_label_is_distinct(cohort_label)" in page_source


def test_research_agent_history_is_separate_and_setup_has_claude_reference_shell() -> None:
    app_source = Path(app.__file__).read_text(encoding="utf-8")
    ra_source = Path(research_agent.__file__).read_text(encoding="utf-8")
    css_source = Path(research_agent.__file__).with_name("shell_overrides.css").read_text(encoding="utf-8")

    app_start = app_source.index('elif active_page == "research_agent":')
    app_branch = app_source[
        app_start:
        app_source.index("_handle_sidebar_export_trigger(default_export_container)", app_start)
    ]
    assert "_eu_ra_view_history" in app_branch
    assert "_ra_view == 'history'" in app_branch
    assert "_render_research_agent_reference_header(lang, view=_ra_view)" in app_branch
    assert app_branch.index("_render_research_agent_reference_header(lang, view=_ra_view)") < app_branch.index('st.container(key="_eu_ra_tabs")')
    assert "_research_agent_active_run_context(st.session_state)" in app_branch
    assert "_eu_ra_header_rerun" in app_branch
    assert "_prime_research_agent_header_rerun(st.session_state, _ra_run_context)" in app_branch
    assert 'icon=":material/tune:"' in app_branch
    assert 'icon=":material/grid_view:"' in app_branch
    assert 'icon=":material/history:"' in app_branch
    assert 'icon=":material/shield:"' in app_branch
    assert 'icon=":material/replay:"' in app_branch
    assert "render_research_agent_history_page(lang, show_header=False)" in app_branch
    assert 'view: str = "setup"' in app_source
    assert 'view == "history"' in app_source
    assert "local history" in app_source
    assert "Local manifests only; nothing leaves your machine." in app_source

    setup_source = ra_source[
        ra_source.index("def render_research_agent_page"):
        ra_source.index("external_llm_selected =", ra_source.index("def render_research_agent_page"))
    ]
    assert "_render_research_agent_setup_overview(" in setup_source
    assert "_render_run_history(" not in setup_source
    assert "_render_replication_section(" not in setup_source

    history_source = ra_source[
        ra_source.index("def render_research_agent_history_page"):
        ra_source.index("def _render_resume_panel")
    ]
    assert "Run history" in history_source
    assert "Local manifests only" in history_source
    assert "Export ledger" in history_source
    assert "research_agent_history_export_ledger" in history_source
    assert "research_agent_history_page_pick" in history_source
    assert "_format_history_findings(" in history_source
    assert "_history_selected_summary_html(" in history_source
    assert 'data-label="{html.escape(headers[0])}"' in ra_source
    assert 'data-label="{html.escape(headers[5])}"' in ra_source
    assert "Selected run utilities" in history_source
    assert "Resume controls" in history_source
    assert "Detailed report and artefacts" in history_source
    assert "finding_errors']}E / {row['finding_warnings']}W" not in history_source
    assert 'finding_errors", 0)}E / {selected_run.get("finding_warnings", 0)}W' not in history_source
    assert "Show resume controls" not in history_source
    assert "Show detailed report and artefacts" not in history_source
    assert 'with st.expander("Detailed report and artefacts"' not in history_source
    assert "_section_request_picker" not in history_source
    assert "_section_llm_picker" not in history_source
    assert "_render_replication_section(default_workdir=workdir)" in history_source

    assert ".ra-setup-overview" in css_source
    assert ".ra-setup-stage-list" in css_source
    assert ".ra-setup-operating" in css_source
    assert ".ra-setup-split" in css_source
    assert ".eu-ref-setup-split .eu-ref-context-grid" in css_source
    assert "grid-template-columns: 1fr" in css_source
    assert ".ra-setup-plan-list" in css_source
    assert ".ra-setup-gate-strip" in css_source
    assert ".ra-history-table" in css_source
    assert ".ra-history-pill" in css_source
    assert ".ra-history-selected" in css_source
    assert ".ra-history-selected.compact" in css_source
    assert ".ra-history-table td::before" in css_source
    assert "content: attr(data-label)" in css_source
    assert "grid-template-columns: 78px minmax(0, 1fr)" in css_source
    assert ".ra-history-utilities-head" in css_source
    assert 'st-key-ra_history_utilities' in css_source
    assert "st-key-_eu_ra_header_rerun" in css_source
    i18n_source = Path(i18n.__file__).read_text(encoding="utf-8")
    assert "🧪 Reproduce published findings" not in i18n_source
    assert "🧪 复现已发表" not in i18n_source


def test_research_agent_header_rerun_routes_to_checkpoint_setup() -> None:
    state = {
        "_agent_workbench": {
            "run_id": "run_20260601T010203_abcd",
            "run_dir": "/tmp/run_20260601T010203_abcd",
            "steps": [{"id": "01"}],
            "research_question": "Does lactate predict ICU mortality?",
        },
        "research_agent_force_manuscript": True,
        "research_agent_resume_mode": "force_manuscript",
        "research_agent_preflight_confirmed": True,
        "research_agent_preflight_signature": "stale",
        "_ra_view": "summary",
    }

    context = app._research_agent_active_run_context(state)
    app._prime_research_agent_header_rerun(state, context)

    assert context == {
        "run_id": "run_20260601T010203_abcd",
        "question": "Does lactate predict ICU mortality?",
        "run_dir": "/tmp/run_20260601T010203_abcd",
    }
    assert state["research_agent_resume_run_id"] == "run_20260601T010203_abcd"
    assert state["research_agent_resume_run_dir"] == "/tmp/run_20260601T010203_abcd"
    assert state["research_agent_force_manuscript"] is False
    assert state["research_agent_resume_mode"] == "continue"
    assert state["research_agent_resume_notes"] == ""
    assert state["research_agent_resume_relax_probe"] is False
    assert state["research_agent_question"] == "Does lactate predict ICU mortality?"
    assert state["research_agent_preflight_confirmed"] is False
    assert "research_agent_preflight_signature" not in state
    assert state["_active_main_page"] == "research_agent"
    assert state["_ra_view"] == "setup"


def test_research_agent_manifest_step_records_do_not_nest_expanders() -> None:
    ra_source = Path(research_agent.__file__).read_text(encoding="utf-8")
    step_source = ra_source[
        ra_source.index("def _render_step_records"):
        ra_source.index("def _render_artifact_gallery")
    ]

    assert "with st.expander(title" in step_source
    assert 'with st.expander(_ra_text("full_step_summary")' not in step_source
    assert 'st.markdown(f"**{_ra_text(\'full_step_summary\')}**")' in step_source
    assert "st.json(summary)" in step_source


def test_research_agent_demo_setup_uses_claude_reference_overview() -> None:
    ra_source = Path(research_agent.__file__).read_text(encoding="utf-8")
    css_source = Path(research_agent.__file__).with_name("shell_overrides.css").read_text(encoding="utf-8")
    demo_source = ra_source[
        ra_source.index("def _render_research_agent_demo_visuals"):
        ra_source.index("def render_research_agent_demo_page")
    ]
    demo_page_source = ra_source[
        ra_source.index("def render_research_agent_demo_page"):
        ra_source.index("def _render_replication_section")
    ]

    assert "Operating model" in demo_source
    assert "Context pack" in demo_source
    assert "Plan preview · 6 steps" in demo_source
    assert "Preflight gate" in demo_source
    assert "ra-setup-stage-list" in demo_source
    assert "pl-step" not in demo_source
    assert "eu-ref-setup-split" in demo_source
    assert "eu-ref-setup-stack" in demo_source
    assert "ra-context-pack-card" in demo_source
    assert "ra-question-card" in demo_source
    assert "One sentence. The agent drafts a plan first" in demo_source
    assert "Claude reference structure adapted" not in demo_source
    assert "No LLM call, no token use, no fabricated analysis pack." in demo_page_source
    assert "render_design_page_header" in demo_page_source
    assert "Question + EasyICU data -> evidence-bound research output" not in demo_source
    assert "Demo guide" not in demo_source
    assert "ra-demo-hero" not in demo_source
    assert ".eu-ref-agent-setup" in css_source
    assert ".ra-setup-stage-list" in css_source
    assert ".pipeline" in css_source
    assert ".eu-ref-setup-split" in css_source
    assert ".eu-ref-setup-stack" in css_source
    assert ".ra-context-pack-card" in css_source
    assert ".ra-question-helper" in css_source
    assert ".eu-ref-question-box" in css_source


def test_research_agent_real_setup_groups_controls_and_defers_data_recipe() -> None:
    ra_source = Path(research_agent.__file__).read_text(encoding="utf-8")
    css_source = Path(research_agent.__file__).with_name("shell_overrides.css").read_text(encoding="utf-8")
    page_source = ra_source[
        ra_source.index("def render_research_agent_page"):
        ra_source.index("external_llm_selected =", ra_source.index("def render_research_agent_page"))
    ]
    overview_source = ra_source[
        ra_source.index("def _render_research_agent_setup_overview"):
        ra_source.index("def _preflight_signature")
    ]
    preflight_source = ra_source[
        ra_source.index("def _render_execution_preflight"):
        ra_source.index("def _render_setup_controls_intro")
    ]

    assert 'st.container(key="eu_ra_setup_controls")' in page_source
    assert 'st.container(key="eu_ra_preflight_panel")' in ra_source
    assert "Operating model" in overview_source
    assert "Context pack" in overview_source
    assert "context_badge" in overview_source
    assert '"awaiting cohort" if is_en else "等待队列"' in overview_source
    assert '<span>{"handed off" if is_en else "已交接"}</span>' not in overview_source
    assert "Plan preview · 6 steps" in overview_source
    assert "Preflight gate" in overview_source
    assert "Ready to run" in overview_source
    assert "ra-setup-stage-list" in overview_source
    assert "pl-step" not in overview_source
    assert "ra-setup-split" in overview_source
    assert "ra-setup-plan-list" in overview_source
    assert "ra-setup-gate-strip" in overview_source
    assert "ra-setup-grid" not in overview_source
    assert "_render_setup_controls_intro(is_en=_is_en)" in page_source
    assert "_render_preflight_controls_intro(is_en=_is_en)" in ra_source
    assert "expanded=bool(question_hint)" in page_source
    assert 'with st.expander(_step_titles[2], expanded=True)' not in page_source
    assert "Complete the missing fields" in ra_source
    assert "local setup" in ra_source
    assert "backend logic unchanged" not in ra_source
    assert "human-controlled run" in ra_source
    assert "Confirm inputs, files, and evidence gates" in ra_source
    assert "Launch review" in ra_source
    assert "Review the current run contract" in ra_source
    assert "ra-preflight-steps" in ra_source
    assert "Confirm launch review" in ra_source
    assert "ra-context-policy" in ra_source
    assert "disable_icu_context = False" in ra_source
    assert "research_agent_disable_icu" not in ra_source
    assert "Disable ICU-aware context" not in ra_source
    assert ".ra-context-policy" in css_source
    assert 'st.session_state["research_agent_preflight_signature"] = signature\n            st.rerun()' in preflight_source
    assert 'st.session_state["research_agent_preflight_confirmed"] = False\n            st.rerun()' in preflight_source
    assert "llm_ready, llm_issue = _llm_run_readiness" in page_source
    assert "preview_signature = _preflight_signature(preview_contract)" in page_source
    assert 'st.session_state["research_agent_preflight_confirmed"] = False' in page_source
    assert "consent_ready = not external_llm_selected or bool(st.session_state.get(\"llm_enabled\", False))" in ra_source
    assert "disabled=cohort is None or not request_ready or not preflight_confirmed or not llm_ready or not consent_ready" in ra_source
    assert "run_clicked = run_button_clicked" in ra_source
    assert "or (force_manuscript and preflight_confirmed and llm_ready)" not in ra_source
    assert "Execution preflight" not in ra_source
    assert "Confirm what the agent will read, write, and gate" not in ra_source
    assert "height=112" in ra_source
    assert "st-key-eu_ra_setup_controls" in css_source
    assert "st-key-eu_ra_preflight_panel" in css_source
    assert ".ra-setup-controls-intro" in css_source
    assert ".ra-preflight-step" in css_source
    assert ".ra-request-brief" in css_source
    assert "st-key-research_agent_question" in css_source


def test_sync_quick_viz_screenshot_mode_requests_rerun_on_toggle_transition() -> None:
    state = {
        "screenshot_mode": True,
        "_screenshot_mode_last_value": False,
        "patient_ids": [101],
        "loaded_concepts": {
            "crea": pd.DataFrame({"stay_id": [101], "crea": [1.1]}),
        },
    }

    should_rerun = app._sync_quick_viz_screenshot_mode(state, lang="en")

    assert should_rerun is True
    assert state["_screenshot_mode_last_value"] is True
    assert state["_floating_ai_open"] is False
    assert state["quality_concept"] == "crea"


def test_compute_quality_missing_rate_preserves_small_real_null_rates() -> None:
    df = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "time": [0, 1, 2, 3, 0, 1, 2, 3],
            "spo2": [98.0, 97.0, None, 96.0, 99.0, 98.0, 97.0, 96.0],
        }
    )

    missing = app._compute_quality_missing_rate(
        concept="spo2",
        df=df,
        id_col="stay_id",
        cohort_patient_count=2,
        time_grid_size=4,
    )

    assert missing == pytest.approx(12.5)


def test_compute_quality_missing_rate_uses_time_coverage_for_sparse_series_without_null_rows() -> None:
    df = pd.DataFrame(
        {
            "stay_id": [1, 1, 2],
            "time": [0, 1, 0],
            "hr": [80, 82, 76],
        }
    )

    missing = app._compute_quality_missing_rate(
        concept="hr",
        df=df,
        id_col="stay_id",
        cohort_patient_count=2,
        time_grid_size=4,
    )

    assert missing == pytest.approx(62.5)


def test_compute_quality_missing_rate_uses_cohort_patients_for_static_boolean_events() -> None:
    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 5],
            "abx": [1, 1, 1],
        }
    )

    missing = app._compute_quality_missing_rate(
        concept="abx",
        df=df,
        id_col="stay_id",
        cohort_patient_count=10,
        time_grid_size=72,
    )

    assert missing == pytest.approx(70.0)


def test_compute_quality_missing_rate_treats_time_stamped_abx_as_sparse_event_series() -> None:
    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 5],
            "time": [0, 0, 0],
            "abx": [1, 1, 1],
        }
    )

    missing = app._compute_quality_missing_rate(
        concept="abx",
        df=df,
        id_col="stay_id",
        cohort_patient_count=10,
        time_grid_size=4,
    )

    assert missing == pytest.approx(92.5)


def test_expected_observation_count_prefers_patient_los_for_real_time_series() -> None:
    patient_df = pd.DataFrame({"stay_id": [1, 1], "time": [0, 12], "hr": [80, 85]})

    expected, source = app._expected_observation_count(
        concept="hr",
        patient_df=patient_df,
        los_icu=2.5,
    )

    assert expected == 60
    assert source == "los"


def test_expected_observation_count_falls_back_to_demo_hours_then_72h() -> None:
    patient_df = pd.DataFrame({"stay_id": [1], "time": [0], "hr": [80]})

    demo_expected, demo_source = app._expected_observation_count(
        concept="hr",
        patient_df=patient_df,
        los_icu=None,
        demo_hours=48,
    )
    fallback_expected, fallback_source = app._expected_observation_count(
        concept="hr",
        patient_df=patient_df,
        los_icu=None,
    )

    assert demo_expected == 48
    assert demo_source == "demo"
    assert fallback_expected == 72
    assert fallback_source == "72h"


def test_compute_quality_out_of_physio_rate_uses_harmonized_ranges() -> None:
    df = pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 2],
            "time": [0, 1, 0, 1],
            "hr": [80, 500, -5, None],
        }
    )

    rate = app._compute_quality_out_of_physio_rate("hr", df)

    assert rate == pytest.approx(66.7, abs=0.05)


def test_compute_quality_duplicate_timestamp_rate_counts_extra_rows_with_same_patient_time() -> None:
    df = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 2],
            "time": [0, 0, 1, 0],
            "hr": [80, 82, 81, 90],
        }
    )

    duplicate_rate = app._compute_quality_duplicate_timestamp_rate(
        concept="hr",
        df=df,
        id_col="stay_id",
    )

    assert duplicate_rate == pytest.approx(25.0)


def test_summarize_quality_temporal_density_reports_median_and_iqr() -> None:
    df = pd.DataFrame(
        {
            "stay_id": [1] * 24 + [2] * 12 + [3] * 6,
            "time": list(range(24)) + list(range(12)) + list(range(6)),
            "hr": [80] * 42,
        }
    )
    los_by_patient = pd.Series({1: 1.0, 2: 1.0, 3: 1.0})

    summary = app._summarize_quality_temporal_density(
        concept="hr",
        df=df,
        id_col="stay_id",
        los_by_patient=los_by_patient,
    )

    assert summary["median"] == pytest.approx(0.5)
    assert summary["q25"] == pytest.approx(0.375)
    assert summary["q75"] == pytest.approx(0.75)


def test_choose_concept_value_column_prefers_primary_map_for_abp_like_frames() -> None:
    df = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "time": [0, 1],
            "sbp": [120, 118],
            "dbp": [70, 68],
            "map": [87, 85],
        }
    )

    assert app._choose_concept_value_column("abp", df) == "map"


def test_choose_concept_value_column_uses_single_numeric_value_when_only_one_exists() -> None:
    df = pd.DataFrame({"stay_id": [1, 1], "time": [0, 1], "crea": [1.1, 1.3]})

    assert app._choose_concept_value_column("crea", df) == "crea"


def test_filter_patient_selector_options_respects_search_and_cap() -> None:
    patient_ids = list(range(1000, 1300))

    default_options = app._filter_patient_selector_options(patient_ids, query="", max_display=200)
    searched_options = app._filter_patient_selector_options(patient_ids, query="12", max_display=5)

    assert len(default_options) == 200
    assert default_options[:3] == [1000, 1001, 1002]
    assert searched_options == [1012, 1112, 1120, 1121, 1122]


def test_patient_selector_does_not_mix_session_state_with_default_index(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class _FakeStreamlit:
        session_state = {"patient_view_id": 10002}

        @staticmethod
        def text_input(*_args, **_kwargs) -> str:
            return ""

        @staticmethod
        def selectbox(**kwargs):
            calls.update(kwargs)
            return kwargs["options"][0]

    monkeypatch.setattr(quality_metrics, "st", _FakeStreamlit)

    quality_metrics._patient_selector(
        patient_ids=[10001, 10002],
        state_key="patient_view_id",
        label="Patient ID",
        lang="en",
        default_patient=10002,
    )

    assert "index" not in calls
    assert calls["key"] == "patient_view_id"


def test_compute_smd_continuous_uses_pooled_standard_deviation() -> None:
    smd = app._compute_smd_continuous(pd.Series([1, 2, 3]), pd.Series([2, 3, 4]))

    assert smd == pytest.approx(-1.0)


def test_compute_smd_binary_uses_pooled_proportion() -> None:
    smd = app._compute_smd_binary(pd.Series([1, 1, 0, 0]), pd.Series([1, 0, 0, 0]))

    assert smd == pytest.approx(0.5164, abs=1e-4)


def test_smd_severity_tag_keeps_threshold_semantics_for_non_table_use() -> None:
    assert app._smd_severity_tag(1.25, "en") == "🔴 large"
    assert app._smd_severity_tag(0.18, "en") == "🟠 mild"
    assert app._smd_severity_tag(0.05, "en") == "🟢 balanced"


def test_cohort_group_table_reports_numeric_smd_without_inline_labels() -> None:
    source = Path(cohort_group_page.__file__).read_text(encoding="utf-8")
    format_smd_source = source[
        source.index("def _format_smd_value"):
        source.index("default_modules = _cohort_default_feature_modules")
    ]

    assert "min n < 10" in source
    assert "SMD is reported as a numeric effect-size distance" in source
    assert "stronger imbalance" in source
    assert "often treated as large" not in source
    assert 'return f"{value:.2f}"' in format_smd_source
    assert "_smd_severity_tag" not in format_smd_source
    assert "_smd_min_group_n" not in source
    assert "⚪ small n" not in source


def test_cohort_group_survival_split_falls_back_when_outcome_is_degenerate() -> None:
    all_survived = pd.DataFrame({
        "stay_id": [1, 2, 3, 4],
        "age": [42, 58, 71, 83],
        "gender": ["M", "F", "M", "F"],
        "los_hours": [24, 48, 96, 120],
        "survived": [1, 1, 1, 1],
    })

    ready, zh_notice = cohort_group_page._survival_contrast_status(all_survived, lang="zh")

    assert ready is False
    assert "没有死亡病例" in zh_notice
    assert (
        cohort_group_page._fallback_compare_mode_for_degenerate_survival(
            all_survived,
            age_threshold=65,
        )
        == "age"
    )

    death_only = all_survived.drop(columns=["survived"]).assign(death=[0, 0, 0, 0])
    ready_from_death, death_notice = cohort_group_page._survival_contrast_status(death_only, lang="zh")

    assert ready_from_death is False
    assert "没有死亡病例" in death_notice
    assert cohort_group_page._cohort_mortality_display(death_only) == "0.0%"
    assert cohort_group_page._cohort_mortality_display(all_survived.drop(columns=["survived"])) == "—"
    assert (
        cohort_group_page._cohort_resolve_compare_mode(
            "survival",
            "age",
            ["survival", "age", "gender"],
        )
        == "age"
    )
    assert (
        cohort_group_page._cohort_resolve_compare_mode(
            "gender",
            "stale",
            ["survival", "age", "gender"],
        )
        == "gender"
    )

    source = Path(cohort_group_page.__file__).read_text(encoding="utf-8")
    assert "已切换到" in source
    assert "无存活状态数据" not in source


def test_build_group_feature_data_from_loaded_concepts_reuses_loaded_demo_frames() -> None:
    loaded_concepts = {
        "hr": pd.DataFrame(
            {
                "stay_id": [1, 1, 2, 2],
                "time": [0, 1, 0, 1],
                "hr": [80, 100, 70, 90],
            }
        ),
        "sep3_sofa2": pd.DataFrame(
            {
                "stay_id": [1, 1, 2, 2],
                "time": [0, 1, 0, 1],
                "sep3_sofa2": [0, 1, 0, 0],
            }
        ),
    }

    feature_data = app._build_group_feature_data_from_loaded_concepts(
        [1, 2],
        ["hr", "sep3_sofa2"],
        loaded_concepts,
        id_col="stay_id",
    )

    assert set(feature_data.keys()) == {"hr", "sep3_sofa2"}
    assert feature_data["hr"].set_index("stay_id")["hr"].to_dict() == {1: 90.0, 2: 80.0}
    assert feature_data["sep3_sofa2"].set_index("stay_id")["sep3_sofa2"].to_dict() == {1: 1, 2: 0}


def test_build_cohort_dashboard_review_stats_summarizes_clinical_signal() -> None:
    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "age": [45, 66, 72, 81],
            "los_hours": [24, 96, 48, 240],
            "sofa_max": [2, 7, 10, 4],
            "survived": [1, 0, 0, 1],
            "sepsis": [False, True, True, False],
            "aki": [False, True, True, False],
            "rrt": [False, False, True, False],
            "mech_vent": [False, True, True, False],
            "vasopressors": [False, True, False, False],
        }
    )

    stats = app._build_cohort_dashboard_review_stats(df, loaded_concepts={}, lang="en")
    phenotype = stats["phenotype"]
    severity = stats["severity"]

    assert stats["metrics"]["patients"] == "4"
    assert stats["metrics"]["median_sofa"] == "5.5"
    assert phenotype.loc[phenotype["label"] == "Sepsis", "pct"].item() == pytest.approx(50.0)
    assert phenotype.loc[phenotype["label"] == "RRT", "count"].item() == 1
    assert severity.loc[severity["sofa_group"] == "6-9", "mortality"].item() == pytest.approx(100.0)
    assert severity.loc[severity["sofa_group"] == ">=10", "patients"].item() == 1


def test_build_cohort_dashboard_review_stats_reports_loaded_module_coverage(monkeypatch) -> None:
    df = pd.DataFrame({"stay_id": [1, 2, 3]})
    loaded_concepts = {
        "hr": pd.DataFrame({"stay_id": [1, 2], "hr": [80, 90]}),
        "map": pd.DataFrame({"stay_id": [1, 3], "map": [70, 75]}),
        "aki": pd.DataFrame({"stay_id": [2], "aki": [1]}),
    }

    monkeypatch.setattr(
        app,
        "get_concept_groups",
        lambda: {"Vitals": ["hr", "map"], "Renal": ["aki"], "Labs": ["crea"]},
    )

    stats = app._build_cohort_dashboard_review_stats(df, loaded_concepts=loaded_concepts, lang="en")
    coverage = stats["coverage"]

    assert stats["metrics"]["features"] == "3"
    assert coverage.loc[coverage["module"] == "Vitals", "features"].item() == 2
    assert coverage.loc[coverage["module"] == "Vitals", "patients"].item() == 3
    assert coverage.loc[coverage["module"] == "Renal", "rows"].item() == 1


def test_build_cohort_dashboard_review_stats_uses_loaded_concepts_for_clinical_panels(monkeypatch) -> None:
    df = pd.DataFrame({"stay_id": [1, 2, 3]})
    loaded_concepts = {
        "sofa2": pd.DataFrame({"stay_id": [1, 2, 3], "sofa2": [1, 7, 10]}),
        "death": pd.DataFrame({"stay_id": [1, 2, 3], "death": [0, 1, 1]}),
        "aki": pd.DataFrame({"stay_id": [2, 3], "aki": [1, 1]}),
        "vaso_ind": pd.DataFrame({"stay_id": [2], "vaso_ind": [1]}),
    }

    monkeypatch.setattr(app, "get_concept_groups", lambda: {"Scores": ["sofa2"], "Outcome": ["death"]})

    stats = app._build_cohort_dashboard_review_stats(df, loaded_concepts=loaded_concepts, lang="en")
    phenotype = stats["phenotype"]
    severity = stats["severity"]

    assert stats["metrics"]["median_sofa"] == "7.0"
    assert phenotype.loc[phenotype["label"] == "AKI", "pct"].item() == pytest.approx(66.7)
    assert phenotype.loc[phenotype["label"] == "Vasopressors", "count"].item() == 1
    assert severity.loc[severity["sofa_group"] == "6-9", "mortality"].item() == pytest.approx(100.0)
    assert severity.loc[severity["sofa_group"] == ">=10", "patients"].item() == 1


def test_generate_mock_cohort_dashboard_data_includes_sofa_reclassification_inputs() -> None:
    df = app._generate_mock_cohort_dashboard_data(lang="en")

    assert {"sofa1_max", "sofa2_max", "sofa1_resp", "sofa2_resp"}.issubset(df.columns)
    assert len(df) == 500
    assert (df["sofa2_max"] - df["sofa1_max"]).abs().sum() > 0

    review = app._build_cohort_dashboard_review_stats(df, lang="en")
    severity = review["severity"].set_index("sofa_group")
    reclass = review["reclassification"]["summary"].set_index("group")

    assert severity.loc["6-9", "mortality"] > severity.loc["3-5", "mortality"]
    assert severity.loc[">=10", "mortality"] > severity.loc["6-9", "mortality"]
    assert reclass.loc["Up-classified", "pct"] == pytest.approx(48.4)
    assert reclass.loc["Up-classified", "mortality"] > reclass.loc["Same", "mortality"]


def test_build_sofa_reclassification_stats_classifies_patient_level_changes() -> None:
    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "sofa1_max": [2, 4, 8, 12],
            "sofa2_max": [4, 4, 5, 13],
            "survived": [0, 1, 1, 0],
            "los_hours": [120, 48, 72, 240],
            "sofa1_resp": [1, 1, 2, 3],
            "sofa2_resp": [2, 1, 1, 4],
            "sofa1_renal": [0, 1, 2, 3],
            "sofa2_renal": [1, 1, 1, 3],
        }
    )

    stats = app._build_sofa_reclassification_stats(df, lang="en")
    summary = stats["summary"]
    matrix = stats["matrix"]
    organ = stats["organ"]

    assert stats["available"] is True
    assert summary.loc[summary["group"] == "Up-classified", "patients"].item() == 2
    assert summary.loc[summary["group"] == "Down-classified", "patients"].item() == 1
    assert summary.loc[summary["group"] == "Same", "patients"].item() == 1
    assert summary.loc[summary["group"] == "Up-classified", "mortality"].item() == pytest.approx(100.0)
    assert matrix.loc[(matrix["SOFA-1"] == "6-9") & (matrix["SOFA-2"] == "3-5"), "patients"].item() == 1
    assert organ.loc[organ["organ"] == "Respiratory", "mean_abs_delta"].item() == pytest.approx(0.75)


def test_build_reclassification_df_from_loaded_concepts_supports_first24_aligned_worst() -> None:
    loaded = {
        "sofa": pd.DataFrame(
            {
                "stay_id": [1, 1, 1, 1, 2, 2],
                "charttime": [-1, 0, 10, 30, 2, 30],
                "sofa": [1, 2, 5, 9, 3, 8],
            }
        ),
        "sofa2": pd.DataFrame(
            {
                "stay_id": [1, 1, 1, 2, 2],
                "charttime": [0, 10, 30, 2, 30],
                "sofa2": [3, 4, 10, 7, 9],
            }
        ),
        "death": pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}),
        "los_icu": pd.DataFrame({"stay_id": [1, 2], "los_icu": [2.5, 4.0]}),
    }

    result = app._build_reclassification_df_from_loaded_concepts(loaded, mode="first24_worst")

    assert result["analysis_unit"].unique().tolist() == ["patients"]
    assert result.loc[result["stay_id"] == 1, "sofa1_max"].item() == 5
    assert result.loc[result["stay_id"] == 1, "sofa2_max"].item() == 4
    assert result.loc[result["stay_id"] == 2, "sofa1_max"].item() == 3
    assert result.loc[result["stay_id"] == 2, "sofa2_max"].item() == 7
    assert result.loc[result["stay_id"] == 2, "mortality"].item() == 1


def test_build_reclassification_df_from_loaded_concepts_supports_time_aligned_points() -> None:
    loaded = {
        "sofa": pd.DataFrame(
            {
                "stay_id": [1, 1, 1],
                "charttime": [0, 10, 30],
                "sofa": [2, 5, 9],
            }
        ),
        "sofa2": pd.DataFrame(
            {
                "stay_id": [1, 1, 1],
                "charttime": [0, 10, 30],
                "sofa2": [3, 4, 10],
            }
        ),
    }

    result = app._build_reclassification_df_from_loaded_concepts(loaded, mode="time_aligned")

    assert result["analysis_unit"].unique().tolist() == ["timepoints"]
    assert result["charttime"].tolist() == [0, 10, 30]
    assert result["sofa1_max"].tolist() == [2, 5, 9]
    assert result["sofa2_max"].tolist() == [3, 4, 10]


def test_build_sofa_reclassification_stats_labels_time_aligned_units() -> None:
    df = pd.DataFrame(
        {
            "stay_id": [1, 1, 2],
            "charttime": [0, 10, 0],
            "analysis_unit": ["timepoints", "timepoints", "timepoints"],
            "sofa1_max": [2, 5, 7],
            "sofa2_max": [3, 4, 7],
        }
    )

    stats = app._build_sofa_reclassification_stats(df, lang="en")

    assert stats["metrics"]["denominator"] == "3"
    assert stats["metrics"]["denominator_label"] == "Paired points"
    assert stats["metrics"]["patient_count"] == "2"


def test_generate_mock_sofa_timeseries_concepts_unlocks_all_sensitivity_modes() -> None:
    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "sofa1_max": [4, 8],
            "sofa2_max": [5, 7],
            "sofa1_resp": [1, 2],
            "sofa2_resp": [2, 1],
            "sofa1_coag": [1, 1],
            "sofa2_coag": [1, 1],
            "sofa1_liver": [0, 1],
            "sofa2_liver": [0, 1],
            "sofa1_cardio": [1, 1],
            "sofa2_cardio": [1, 1],
            "sofa1_cns": [1, 1],
            "sofa2_cns": [1, 1],
            "sofa1_renal": [0, 2],
            "sofa2_renal": [0, 2],
            "mortality": [0, 1],
            "los_days": [3.0, 5.0],
        }
    )

    concepts = app._generate_mock_sofa_timeseries_concepts(cohort)
    availability = app._get_sofa_reclassification_mode_availability(concepts)
    first24 = app._build_reclassification_df_from_loaded_concepts(concepts, mode="first24_worst")
    time_aligned = app._build_reclassification_df_from_loaded_concepts(concepts, mode="time_aligned")

    assert availability["available"] == ["worst_icu", "first24_worst", "time_aligned"]
    assert availability["locked"] == []
    assert set(concepts).issuperset({"sofa", "sofa2", "sofa_resp", "sofa2_resp", "death", "los_icu"})
    assert first24["analysis_unit"].unique().tolist() == ["patients"]
    assert time_aligned["analysis_unit"].unique().tolist() == ["timepoints"]
    assert time_aligned["stay_id"].nunique() == 2


def test_get_sofa_reclassification_mode_availability_locks_time_series_without_data() -> None:
    availability = app._get_sofa_reclassification_mode_availability({})

    assert availability["available"] == ["worst_icu"]
    assert availability["locked"] == ["first24_worst", "time_aligned"]


def test_get_sofa_reclassification_source_uses_demo_timeseries_for_time_aligned(monkeypatch) -> None:
    cohort = app._generate_mock_cohort_dashboard_data(lang="en").head(12)
    streamlit_stub = _SessionStateStreamlit(
        {
            "dash_demographics": cohort,
            "dash_is_demo": True,
            "loaded_concepts": {},
        }
    )
    monkeypatch.setattr(app, "st", streamlit_stub)

    source_df, source_label = app._get_sofa_reclassification_source(lang="en", mode="time_aligned")
    stats = app._build_sofa_reclassification_stats(source_df, lang="en")

    assert source_label.startswith("Demo SOFA time series")
    assert not source_df.empty
    assert "charttime" in source_df.columns
    assert stats["metrics"]["denominator_label"] == "Paired points"


def test_get_sofa_reclassification_mode_availability_unlocks_time_series_when_loaded() -> None:
    loaded = {
        "sofa": pd.DataFrame(
            {
                "stay_id": [1, 1],
                "charttime": [0, 10],
                "sofa": [2, 5],
            }
        ),
        "sofa2": pd.DataFrame(
            {
                "stay_id": [1, 1],
                "charttime": [0, 10],
                "sofa2": [3, 4],
            }
        ),
    }

    availability = app._get_sofa_reclassification_mode_availability(loaded)

    assert availability["available"] == ["worst_icu", "first24_worst", "time_aligned"]
    assert availability["locked"] == []


def test_topbar_render_action_loads_quick_viz_demo_workspace() -> None:
    def fake_generate_mock_data(**params):
        assert params["n_patients"] == 2
        return (
            {
                "age": pd.DataFrame({"stay_id": [2, 1], "age": [70, 60]}),
                "hr": pd.DataFrame({"stay_id": [2, 1], "time": [0, 0], "hr": [82, 91]}),
            },
            [2, 1],
        )

    state = {
        "_eu_topbar_run_request": {"page": "quick_viz"},
        "entry_mode": "demo",
        "mock_params": {"n_patients": 2, "hours": 24},
    }

    result = app._consume_topbar_run_request(
        state,
        "quick_viz",
        "en",
        generate_data_func=fake_generate_mock_data,
    )

    assert result["level"] == "success"
    assert state["loaded_data_origin"] == "demo_viz"
    assert state["patient_ids"] == [1, 2]
    assert state["selected_concepts"] == ["age", "hr"]
    assert "_eu_topbar_run_request" not in state
    assert state["_eu_action_log"]


def test_topbar_cohort_action_refreshes_demo_workspace() -> None:
    called = {}

    def fake_ensure_demo(state, **kwargs):
        called.update(kwargs)
        state["cohort_is_demo"] = True

    state = {
        "_eu_topbar_run_request": {"page": "cohort"},
        "entry_mode": "demo",
    }

    result = app._consume_topbar_run_request(
        state,
        "cohort",
        "en",
        ensure_demo_workspace_fn=fake_ensure_demo,
    )

    assert result["level"] == "success"
    assert called == {"lang": "en", "force": True}
    assert state["cohort_is_demo"] is True


def test_real_data_step1_requires_current_validated_path(tmp_path) -> None:
    data_root = tmp_path / "miiv"
    data_root.mkdir()

    state = {
        "entry_mode": "real",
        "data_path": str(data_root),
        "path_validated": False,
    }
    assert app._real_data_source_ready_for_step1(state) is False

    state["path_validated"] = True
    state["last_validated_path"] = str(tmp_path / "other")
    assert app._real_data_source_ready_for_step1(state) is False

    state["last_validated_path"] = str(data_root)
    assert app._real_data_source_ready_for_step1(state) is True


def test_real_entry_cta_resets_demo_progress_and_database() -> None:
    state = {
        "entry_mode": "demo",
        "use_mock_data": True,
        "database": "mock",
        "path_validated": True,
        "last_validated_path": "/tmp/demo",
        "step1_confirmed": True,
        "step2_confirmed": True,
        "step3_confirmed": True,
        "export_completed": True,
        "trigger_export": True,
        "_exporting_in_progress": True,
        "loaded_concepts": {"hr": object()},
        "loaded_data_origin": "demo_viz",
        "patient_ids": [10001],
        "all_patient_count": 1,
        "selected_patient": 10001,
        "selected_concepts": ["hr"],
        "quick_viz_active_panel": "Time Series",
        "_export_failure_result": {"type": "no_data"},
        "_active_main_page": "quick_viz",
    }

    pages_redesign._route_to_extract_entry_mode(state, "real")

    assert state["entry_mode"] == "real"
    assert state["use_mock_data"] is False
    assert state["database"] == "miiv"
    assert state["path_validated"] is False
    assert "last_validated_path" not in state
    assert state["step1_confirmed"] is False
    assert state["step2_confirmed"] is False
    assert state["step3_confirmed"] is False
    assert state["export_completed"] is False
    assert state["trigger_export"] is False
    assert state["_exporting_in_progress"] is False
    assert state["loaded_concepts"] == {}
    assert state["loaded_data_origin"] == "none"
    assert state["patient_ids"] == []
    assert state["all_patient_count"] == 0
    assert state["selected_patient"] is None
    assert state["selected_concepts"] == []
    assert "quick_viz_active_panel" not in state
    assert "_export_failure_result" not in state
    assert state["_active_main_page"] == "extract"


def test_topbar_breadcrumb_targets_navigate_to_parent_paths(monkeypatch) -> None:
    state = {
        "_active_main_page": "quick_viz",
        "entry_mode": "demo",
        "use_mock_data": True,
        "database": "mock",
        "loaded_concepts": {"hr": object()},
        "patient_ids": [10001],
        "step1_confirmed": True,
        "step2_confirmed": True,
        "step3_confirmed": True,
        "export_completed": True,
    }
    clear_calls: list[str] = []

    monkeypatch.setattr(app, "clear_run_state", lambda scope="all": clear_calls.append(scope))

    app._apply_topbar_breadcrumb_target(state, "data_visualization")
    assert state["_active_main_page"] == "quick_viz"

    app._apply_topbar_breadcrumb_target(state, "cohort")
    assert state["_active_main_page"] == "cohort"

    app._apply_topbar_breadcrumb_target(state, "data_extraction")
    assert state["_active_main_page"] == "extract"

    app._apply_topbar_breadcrumb_target(state, "entry")
    assert clear_calls == ["all"]
    assert state["entry_mode"] == "none"
    assert state["use_mock_data"] is False
    assert state["_active_main_page"] == "tutorial"


def test_extract_workflow_helpers_keep_state_consistent() -> None:
    source = Path(app.__file__).read_text(encoding="utf-8")
    state = {
        "_active_main_page": "quick_viz",
        "entry_mode": "demo",
        "use_mock_data": True,
        "database": "mock",
        "step1_confirmed": True,
        "step2_confirmed": True,
        "step3_confirmed": True,
        "export_completed": True,
    }

    assert app._extract_step_unlocked(state, 4) is True

    app._set_extract_step_state(state, 2)
    assert state["_active_main_page"] == "extract"
    assert state["step1_confirmed"] is True
    assert state["step2_confirmed"] is False
    assert state["step3_confirmed"] is False
    assert state["_scroll_to_top"] is True
    assert state["export_completed"] is False
    assert app._extract_step_unlocked(state, 3) is False
    assert "scrollEasyICUToTop" in source
    assert "[0, 80, 240, 600, 1200]" in source

    state["_export_failure_result"] = {"type": "no_data"}
    app._switch_extract_entry_mode(state, "real")
    assert state["entry_mode"] == "real"
    assert state["use_mock_data"] is False
    assert state["database"] == "miiv"
    assert state["path_validated"] is False
    assert state["step1_confirmed"] is False
    assert state["step2_confirmed"] is False
    assert state["loaded_concepts"] == {}
    assert state["loaded_data_origin"] == "none"
    assert state["patient_ids"] == []
    assert "_export_failure_result" not in state
    assert state["_active_main_page"] == "extract"

    app._switch_extract_entry_mode(state, "demo")
    assert state["entry_mode"] == "demo"
    assert state["use_mock_data"] is True
    assert state["database"] == "mock"


def test_export_completion_routes_to_visualization_data_tables(monkeypatch, tmp_path) -> None:
    state = _AttrSessionState({
        "viz_max_patients": 10,
        "_post_export_guidance_dismissed": True,
        "_export_failure_result": {"type": "no_data"},
    })
    monkeypatch.setattr(app, "st", _SessionStateStreamlit(state))
    exported_files = [
        str(tmp_path / "vitals_hr_map.parquet"),
        str(tmp_path / "labs_creatinine.parquet"),
    ]

    app._prime_export_completion(tmp_path, exported_files, auto_load=True)

    assert state["export_completed"] is True
    assert state["trigger_export"] is False
    assert state["_exporting_in_progress"] is False
    assert state["last_export_dir"] == str(tmp_path)
    assert state["viz_export_path"] == str(tmp_path)
    assert "_export_failure_result" not in state
    assert state["viz_data_source_mode"] == "exported"
    assert state["_prefer_exported_viz"] is True
    assert state["_active_main_page"] == "quick_viz"
    assert state["_post_export_navigation_pending"] is True
    assert state["_post_export_target_panel"] == "Data Tables"
    assert "_main_nav_widget" not in state
    assert "quick_viz_active_panel" not in state
    assert "_post_export_guidance_dismissed" not in state
    assert state["_viz_auto_load_export"] == {
        "path": str(tmp_path),
        "selected_files": ["vitals_hr_map", "labs_creatinine"],
        "max_patients": 10,
    }


def test_post_export_next_step_actions_route_to_real_destinations(tmp_path) -> None:
    export_dir = str(tmp_path / "easyicu_export")
    state = {
        "export_completed": True,
        "last_export_dir": export_dir,
        "_post_export_navigation_pending": True,
        "quick_viz_active_panel": "Time Series",
    }

    app._apply_post_export_next_step(state, "review", lang="en")
    assert state["_active_main_page"] == "quick_viz"
    assert "_main_nav_widget" not in state
    assert state["quick_viz_active_panel"] == "Data Tables"
    assert state["_scroll_to_top"] is True
    assert state["_post_export_guidance_dismissed"] is True
    assert "_post_export_navigation_pending" not in state

    agent_state = {
        "export_completed": True,
        "last_export_dir": export_dir,
        "entry_mode": "demo",
        "use_mock_data": True,
        "database": "mock",
        "path_validated": True,
        "last_validated_path": "/tmp/mock",
        "research_agent_resume_run_id": "run_old",
        "research_agent_force_manuscript": True,
        "research_agent_preflight_confirmed": True,
        "research_agent_module_dir_pick": "research_output/webapp",
    }
    app._apply_post_export_next_step(agent_state, "agent", lang="zh")
    assert agent_state["_active_main_page"] == "research_agent"
    assert "_main_nav_widget" not in agent_state
    assert agent_state["_ra_view"] == "setup"
    assert agent_state["entry_mode"] == "real"
    assert agent_state["use_mock_data"] is False
    assert agent_state["database"] == "miiv"
    assert agent_state["path_validated"] is False
    assert "last_validated_path" not in agent_state
    assert agent_state["_eu_ra_focus_module_folder"] is True
    assert agent_state["_eu_ra_module_pick_force_manual"] is True
    assert agent_state["_eu_ra_apply_export_file_selection"] is True
    assert agent_state["research_agent_module_dir_text"] == export_dir
    assert agent_state["research_agent_cohort_source"] == "选择 EasyICU 模块导出文件夹"
    assert "research_agent_module_dir_pick" not in agent_state
    assert "research_agent_resume_run_id" not in agent_state
    assert "research_agent_force_manuscript" not in agent_state
    assert "research_agent_preflight_confirmed" not in agent_state

    cohort_state = {
        "export_completed": True,
        "entry_mode": "real",
        "database": "miiv",
        "id_col": "stay_id",
        "loaded_concepts": {
            "age": pd.DataFrame({"stay_id": [11, 12], "age": [62, 71]}),
            "death": pd.DataFrame({"stay_id": [11, 12], "death": [0, 1]}),
        },
    }
    app._apply_post_export_next_step(cohort_state, "cohort", lang="en")
    assert cohort_state["_active_main_page"] == "cohort"
    assert "_main_nav_widget" not in cohort_state
    assert cohort_state["_cohort_real_ws_ready"] is True
    assert cohort_state["_cohort_real_ws_origin"] == "loaded_exports"
    assert list(cohort_state["grp_demographics"]["stay_id"]) == [11, 12]
    assert list(cohort_state["dash_demographics"]["stay_id"]) == [11, 12]


def test_post_export_guidance_copy_exposes_visualization_and_agent_actions() -> None:
    app_source = Path(app.__file__).read_text(encoding="utf-8")
    sidebar_source = Path(sidebar.__file__).read_text(encoding="utf-8")
    research_agent_source = Path(research_agent.__file__).read_text(encoding="utf-8")
    app_handoff_source = app_source[
        app_source.index("def _apply_post_export_next_step"):
        app_source.index("def _render_post_export_guidance")
    ]
    sidebar_handoff_source = sidebar_source[
        sidebar_source.index("def _apply_sidebar_post_export_next_step"):
        sidebar_source.index("def _ensure_default_directory_input_value")
    ]

    assert "_render_post_export_guidance(" in app_source
    assert "_post_export_open_review" in app_source
    assert "_post_export_open_agent" in app_source
    assert "Review tables" in app_source
    assert "Use export in Agent" in app_source
    assert "eu-post-export-hero" in app_source
    assert "post_export_completed_open_review" in sidebar_source
    assert "post_export_completed_open_agent" in sidebar_source
    assert "Use export in Agent" in sidebar_source
    assert "_main_nav_widget" not in app_handoff_source
    assert "_main_nav_widget" not in sidebar_handoff_source
    assert "research_agent_module_dir_pick" in app_handoff_source
    assert "research_agent_module_dir_pick" in sidebar_handoff_source
    assert "_eu_ra_module_pick_force_manual" in app_handoff_source
    assert "_eu_ra_module_pick_force_manual" in sidebar_handoff_source
    assert "_eu_ra_apply_export_file_selection" in app_handoff_source
    assert "_eu_ra_apply_export_file_selection" in sidebar_handoff_source
    assert "show_handoff_path" in research_agent_source
    assert "ra-export-handoff" in research_agent_source
    assert "post_export_handoff_title" in research_agent_source
    assert "post_export_handoff_files" in research_agent_source
    assert "_eu_ra_focus_module_folder" in research_agent_source
    assert "_export_result_file_labels_for_folder" in research_agent_source


def test_export_in_progress_uses_quiet_wait_mode_before_main_body() -> None:
    app_source = Path(app.__file__).read_text(encoding="utf-8")
    quiet_block = app_source[
        app_source.index("default_export_container = st.container()"):
        app_source.index("# ============ Shell-A top bar")
    ]
    helper_source = app_source[
        app_source.index("def _handle_sidebar_export_trigger"):
        app_source.index("def _get_pyarrow_version")
    ]

    assert "if export_in_progress:" in quiet_block
    assert "_handle_sidebar_export_trigger(default_export_container)" in quiet_block
    assert "or st.session_state.get('_export_conflict_pending', False)" in app_source
    assert "return" in quiet_block
    assert "render_tutorial_redesign_page(lang)" not in quiet_block
    assert "tabs[0].click()" not in helper_source
    assert "不再切回 Tutorial 正文" in helper_source


def test_completed_export_navigation_pending_is_consumed_once() -> None:
    state = {
        "export_completed": True,
        "_post_export_navigation_pending": True,
        "_post_export_target_panel": "Data Tables",
        "_active_main_page": "extract",
        "_scroll_to_tab": "export_progress",
    }

    assert app._consume_completed_export_navigation(state) is True
    assert state["_active_main_page"] == "quick_viz"
    assert state["_main_nav_widget"] == "quick_viz"
    assert state["quick_viz_active_panel"] == "Data Tables"
    assert "_post_export_navigation_pending" not in state
    assert "_post_export_target_panel" not in state
    assert "_scroll_to_tab" not in state
    assert app._consume_completed_export_navigation(state) is False


def test_export_cancel_queue_keeps_user_on_extraction_page() -> None:
    state = {
        "trigger_export": True,
        "_exporting_in_progress": True,
        "_export_conflict_pending": True,
        "_scroll_to_tab": "export_progress",
    }

    export_workflow._queue_export_cancel(state, lang="en")

    assert state["_export_cancelled"] is False
    assert state["trigger_export"] is False
    assert state["_exporting_in_progress"] is False
    assert "_export_conflict_pending" not in state
    assert "_scroll_to_tab" not in state
    assert state["_active_main_page"] == "extract"
    assert state["_main_nav_widget"] == "extract"
    assert state["_export_cancel_notice"] == "Export stopped by user."


def test_no_data_export_failure_returns_to_recoverable_step4(tmp_path) -> None:
    state = _AttrSessionState({
        "export_completed": True,
        "trigger_export": True,
        "_exporting_in_progress": True,
        "_export_success_result": {"files": [str(tmp_path / "old.parquet")]},
        "_viz_auto_load_export": {"path": "/tmp/old"},
        "_post_export_navigation_pending": True,
        "_post_export_target_panel": "Data Tables",
        "_post_export_guidance_dismissed": False,
        "_scroll_to_tab": "export_progress",
        "_active_main_page": "quick_viz",
    })

    failure = export_workflow._record_no_data_export_failure(
        state,
        export_dir=tmp_path,
        selected_concepts=["vent_start", "age", "death"],
        unsupported_concepts=["vent_start"],
        empty_concepts=["age"],
        failed_concepts=["death"],
        lang="en",
    )

    assert failure["type"] == "no_data"
    assert failure["selected_count"] == 3
    assert state["export_completed"] is False
    assert state["trigger_export"] is False
    assert state["_exporting_in_progress"] is False
    assert state["_active_main_page"] == "extract"
    assert state["_main_nav_widget"] == "extract"
    assert state["_scroll_to_top"] is True
    assert "_export_success_result" not in state
    assert "_viz_auto_load_export" not in state
    assert "_post_export_navigation_pending" not in state
    assert "_post_export_target_panel" not in state
    assert "_post_export_guidance_dismissed" not in state
    assert "_scroll_to_tab" not in state
    assert state["_export_failure_result"]["unsupported_concepts"] == ["vent_start"]
    assert state["_export_failure_result"]["empty_data_concepts"] == ["age"]
    assert state["_export_failure_result"]["failed_concepts"] == ["death"]


def test_completed_export_restart_resets_all_extraction_steps() -> None:
    source = Path(sidebar.__file__).read_text(encoding="utf-8")
    restart_block = source[
        source.index('key="restart_extraction"'):
        source.index("# 返回首页按钮")
    ]

    assert "st.session_state.step1_confirmed = False" in restart_block
    assert "st.session_state.step2_confirmed = False" in restart_block
    assert "st.session_state.step3_confirmed = False" in restart_block
    assert "st.session_state[_STEP2_RESET_PENDING_KEY] = True" in restart_block
    assert "st.session_state.pop('_eu_concept_defaults_seeded', None)" in restart_block
    assert "st.session_state.pop('_post_export_navigation_pending', None)" in restart_block
    assert "st.session_state.pop('_post_export_target_panel', None)" in restart_block


def test_sidebar_pipeline_steps_are_sequential_click_targets() -> None:
    source = Path(sidebar.__file__).read_text(encoding="utf-8")
    css_text = shell_styles._load_shell_overrides_css()
    state = {
        "_active_main_page": "quick_viz",
        "step1_confirmed": False,
        "step2_confirmed": False,
        "step3_confirmed": False,
        "export_completed": False,
    }

    assert sidebar._sidebar_extract_step_unlocked(state, 1) is True
    assert sidebar._sidebar_extract_step_unlocked(state, 2) is False
    assert sidebar._sidebar_extract_step_unlocked(state, 3) is False
    assert sidebar._sidebar_extract_step_unlocked(state, 4) is False

    state["step1_confirmed"] = True
    assert sidebar._sidebar_extract_step_unlocked(state, 2) is True
    assert sidebar._sidebar_extract_step_unlocked(state, 3) is False

    state["step2_confirmed"] = True
    state["step3_confirmed"] = True
    state["export_completed"] = True
    sidebar._sidebar_set_extract_step_state(state, 2)

    assert state["_active_main_page"] == "extract"
    assert state["step1_confirmed"] is True
    assert state["step2_confirmed"] is False
    assert state["step3_confirmed"] is False
    assert state["export_completed"] is False
    assert "eu_pipeline_step_" in source
    assert "eu_pipeline_jump_" in source
    assert "_sidebar_extract_step_unlocked" in source
    assert "_sidebar_set_extract_step_state" in source
    assert "`{step.meta}`" not in source
    assert "**{step.title}**  \\n{step.meta}" in source
    pipeline_css = css_text[
        css_text.index("[class*=\"st-key-eu_pipeline_step_\"]"):
        css_text.index(".eu-sidebar-footer-rule")
    ]
    assert "[class*=\"st-key-eu_pipeline_step_\"] .stButton" in css_text
    assert "[class*=\"st-key-eu_pipeline_step_\"] button" in css_text
    assert "st-key-eu_pipeline_step_active_" in css_text
    assert "st-key-eu_pipeline_step_locked_" in css_text
    assert "span[data-testid=\"stIconMaterial\"]" in pipeline_css
    assert "color: var(--accent) !important" in pipeline_css
    assert "color: var(--ink-4) !important" in pipeline_css
    assert "white-space: pre-line !important" in css_text
    assert "opacity: 0 !important" not in pipeline_css


def test_sidebar_cohort_meta_ignores_default_internal_filter_values(monkeypatch) -> None:
    streamlit_stub = _SessionStateStreamlit(
        _AttrSessionState(
            {
                "language": "en",
                "entry_mode": "demo",
                "mock_params": {"n_patients": 10},
                "step1_confirmed": True,
                "step2_confirmed": True,
                "step3_confirmed": False,
                "export_completed": False,
                "cohort_enabled": True,
                "cohort_filter": {
                    "age_min": None,
                    "age_max": None,
                    "first_icu_stay": None,
                    "los_min": None,
                    "los_max": None,
                    "gender": None,
                    "survived": None,
                    "has_sepsis": None,
                    "disease_cohort": "none",
                    "icd_query": "",
                    "icd_include_query": "",
                    "icd_exclude_query": "",
                    "icd_mode": "include",
                },
            }
        )
    )
    monkeypatch.setattr(sidebar, "st", streamlit_stub)

    steps = sidebar._compute_pipeline_steps()

    assert sidebar._active_step2_filter_chips("en") == []
    assert steps[1].key == "cohort"
    assert steps[1].meta == "all stays"
    assert "2 filters" not in steps[1].meta


def test_sidebar_cohort_meta_counts_only_effective_filters(monkeypatch) -> None:
    streamlit_stub = _SessionStateStreamlit(
        _AttrSessionState(
            {
                "language": "en",
                "entry_mode": "demo",
                "mock_params": {"n_patients": 10},
                "step1_confirmed": True,
                "step2_confirmed": True,
                "step3_confirmed": False,
                "export_completed": False,
                "cohort_enabled": True,
                "cohort_filter": {
                    "age_min": 65,
                    "age_max": None,
                    "first_icu_stay": None,
                    "los_min": None,
                    "los_max": None,
                    "gender": None,
                    "survived": None,
                    "has_sepsis": True,
                    "disease_cohort": "sepsis",
                    "icd_query": "A41",
                    "icd_include_query": "A41",
                    "icd_exclude_query": "",
                    "icd_mode": "include",
                },
            }
        )
    )
    monkeypatch.setattr(sidebar, "st", streamlit_stub)
    monkeypatch.setattr(sidebar, "DISEASE_COHORT_CONFIG", {"sepsis": {"label_en": "Sepsis-3"}}, raising=False)

    chips = sidebar._active_step2_filter_chips("en")
    steps = sidebar._compute_pipeline_steps()

    assert chips == ["age 65-120", "Sepsis-3", "ICD + A41"]
    assert steps[1].meta == "3 filters"


def test_sidebar_pipeline_concept_meta_uses_fresh_group_selection(monkeypatch) -> None:
    streamlit_stub = _SessionStateStreamlit(
        _AttrSessionState(
            {
                "language": "en",
                "entry_mode": "demo",
                "mock_params": {"n_patients": 10},
                "step1_confirmed": True,
                "step2_confirmed": True,
                "step3_confirmed": False,
                "export_completed": False,
                "selected_groups": ["Core", "Renal"],
                "selected_concepts": ["hr"],
                "concept_checkboxes": {
                    "hr": True,
                    "map": True,
                    "creatinine": True,
                    "urine_output": True,
                },
            }
        )
    )
    monkeypatch.setattr(sidebar, "st", streamlit_stub)
    monkeypatch.setattr(
        sidebar,
        "get_concept_groups",
        lambda: {"Core": ["hr", "map"], "Renal": ["creatinine", "urine_output"]},
        raising=False,
    )

    steps = sidebar._compute_pipeline_steps()

    assert steps[2].key == "concepts"
    assert steps[2].meta == "4 features"


def test_concept_module_toggle_callback_updates_summary_before_render(monkeypatch) -> None:
    class _ConceptToggleStreamlit:
        def __init__(self) -> None:
            self.session_state = _AttrSessionState(
                {
                    "selected_groups": ["Core"],
                    "selected_concepts": ["hr"],
                    "concept_checkboxes": {"hr": True},
                    "step3_confirmed": True,
                }
            )
            self.rerun_called = False

        def rerun(self) -> None:
            self.rerun_called = True

    streamlit_stub = _ConceptToggleStreamlit()
    monkeypatch.setattr(sidebar, "st", streamlit_stub)
    concept_groups = {"Core": ["hr"], "Renal": ["creatinine", "urine_output"]}

    sidebar._toggle_concept_group_for_design(concept_groups, "Renal")

    assert streamlit_stub.session_state["selected_groups"] == ["Core", "Renal"]
    assert streamlit_stub.session_state["selected_concepts"] == ["creatinine", "hr", "urine_output"]
    assert streamlit_stub.session_state["step3_confirmed"] is False
    assert streamlit_stub.rerun_called is False


def test_topbar_crossdb_real_action_opens_loader_when_data_missing() -> None:
    state = {
        "_eu_topbar_run_request": {"page": "cross_db"},
        "entry_mode": "real",
    }

    result = app._consume_topbar_run_request(state, "cross_db", "en")

    assert result["level"] == "warning"
    assert "_eu_crossdb_advanced_open" not in state
    assert state["_eu_crossdb_distribution_open"] is True
    assert "Detailed Cross-DB distribution panel is open below" in result["message"]
    assert "_eu_topbar_run_request" not in state


def test_crossdb_page_defaults_to_summary_with_opt_in_details(monkeypatch) -> None:
    class _FakeStreamlit:
        def __init__(self, session_state) -> None:
            self.session_state = session_state
            self.button_labels: list[str] = []
            self.button_kwargs: list[dict] = []
            self.markdown_calls: list[str] = []

        def markdown(self, body, *_args, **_kwargs) -> None:
            self.markdown_calls.append(str(body))

        def button(self, label, **kwargs) -> bool:
            self.button_labels.append(label)
            self.button_kwargs.append(kwargs)
            return False

        def rerun(self) -> None:
            raise AssertionError("rerun should not be called")

    rendered: list[str] = []
    streamlit_stub = _FakeStreamlit({"language": "en", "entry_mode": "demo"})
    monkeypatch.setattr(cohort_redesign, "st", streamlit_stub)

    cohort_redesign.render_cross_db_redesign_page(
        "en",
        multidb_fn=lambda _lang: rendered.append("loader"),
    )

    assert rendered == []
    assert streamlit_stub.button_labels == ["Open detailed distributions"]
    page_source = "\n".join(streamlit_stub.markdown_calls)
    assert "Detailed distributions" in page_source
    assert "collapsed by default" in page_source
    assert "Open detailed loader" not in page_source
    assert "Hide detailed loader" not in page_source
    assert "Load data / detailed distributions" not in page_source
    assert "Connect real multi-database roots here" not in page_source
    assert "Use this advanced panel" not in page_source
    assert "详细加载器" not in page_source
    assert "加载数据 / 详细分布" not in page_source
    assert "eu-agent-gate" not in page_source
    assert "Agent preflight" not in page_source
    assert '_render_agent_gate_strip(lang, context="Cross-DB benchmark")' not in Path(cohort_redesign.__file__).read_text(encoding="utf-8")

    assert streamlit_stub.button_kwargs[0]["on_click"] is cohort_redesign._toggle_crossdb_distribution_panel

    rendered.clear()
    streamlit_stub = _FakeStreamlit({
        "language": "en",
        "entry_mode": "demo",
        "_eu_crossdb_distribution_open": True,
    })
    monkeypatch.setattr(cohort_redesign, "st", streamlit_stub)

    cohort_redesign.render_cross_db_redesign_page(
        "en",
        multidb_fn=lambda _lang: rendered.append("loader"),
    )

    assert rendered == ["loader"]
    assert streamlit_stub.button_labels == ["Hide detailed distributions"]
    page_source = "\n".join(streamlit_stub.markdown_calls)
    assert "open" in page_source


def test_crossdb_chinese_copy_uses_natural_product_labels(monkeypatch) -> None:
    class _FakeStreamlit:
        def __init__(self, session_state) -> None:
            self.session_state = session_state
            self.markdown_calls: list[str] = []

        def markdown(self, body, *_args, **_kwargs) -> None:
            self.markdown_calls.append(str(body))

    streamlit_stub = _FakeStreamlit({
        "language": "zh",
        "entry_mode": "demo",
        "multidb_is_demo": True,
        "multidb_data": {"miiv": pd.DataFrame({"stay_id": [1, 2], "hr": [80, 90]})},
        "multidb_concepts": ["hr"],
        "_eu_shell_only": True,
    })
    monkeypatch.setattr(cohort_redesign, "st", streamlit_stub)

    cohort_redesign.render_cross_db_redesign_page("zh", multidb_fn=lambda _lang: None)

    page_source = "\n".join(streamlit_stub.markdown_calls)
    assert "跨数据库对比" in page_source
    assert "工作区" in page_source
    assert "演示队列" in page_source
    assert "已选择数据库" in page_source
    assert "就绪" in page_source
    assert "跨库基准" not in page_source
    assert "跨库分布基准" not in page_source
    assert "WORKSPACE" not in page_source
    assert "CURRENT COHORT" not in page_source
    assert "ACTIVE DATABASES" not in page_source
    assert ">ready<" not in page_source

    source_bundle = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in [
            cohort_redesign.__file__,
            cohort_multidb_page.__file__,
            i18n.__file__,
        ]
    )
    assert "跨库基准" not in source_bundle
    assert "跨库分布基准" not in source_bundle


def test_crossdb_unloaded_state_does_not_show_fabricated_results(monkeypatch) -> None:
    class _FakeStreamlit:
        def __init__(self, session_state) -> None:
            self.session_state = session_state
            self.markdown_calls: list[str] = []

        def markdown(self, body, *_args, **_kwargs) -> None:
            self.markdown_calls.append(str(body))

    streamlit_stub = _FakeStreamlit({
        "language": "zh",
        "entry_mode": "real",
        "multidb_selected": ["miiv", "eicu", "aumc"],
        "_eu_shell_only": True,
    })
    monkeypatch.setattr(cohort_redesign, "st", streamlit_stub)

    cohort_redesign.render_cross_db_redesign_page("zh", multidb_fn=lambda _lang: None)

    page_source = "\n".join(streamlit_stub.markdown_calls)
    assert "尚未加载数据库" in page_source
    assert "跨数据库摘要待生成" in page_source
    assert "概念可用性矩阵尚未生成" in page_source
    assert "占位百分比" in page_source
    assert "ACTIVE DATABASES" not in page_source
    assert "MIMIC-IV</div>" not in page_source
    assert "eICU-CRD" not in page_source
    assert "AmsterdamUMCdb" not in page_source
    assert "可用于对比" not in page_source
    assert "70%" not in page_source
    assert "90%" not in page_source
    assert "20%" not in page_source


def test_crossdb_single_loaded_database_is_not_treated_as_comparison(monkeypatch) -> None:
    class _FakeStreamlit:
        def __init__(self, session_state) -> None:
            self.session_state = session_state
            self.markdown_calls: list[str] = []

        def markdown(self, body, *_args, **_kwargs) -> None:
            self.markdown_calls.append(str(body))

    streamlit_stub = _FakeStreamlit({
        "language": "zh",
        "entry_mode": "real",
        "multidb_data": {"miiv": pd.DataFrame({"stay_id": [1, 2], "hr": [80, 90]})},
        "multidb_concepts": ["hr"],
        "multidb_is_demo": False,
        "_eu_shell_only": True,
    })
    monkeypatch.setattr(cohort_redesign, "st", streamlit_stub)

    cohort_redesign.render_cross_db_redesign_page("zh", multidb_fn=lambda _lang: None)

    page_source = "\n".join(streamlit_stub.markdown_calls)
    assert "还需要另一个数据库" in page_source
    assert "跨数据库摘要待生成" in page_source
    assert "概念可用性矩阵尚未生成" in page_source
    assert "已加载的跨数据库分布摘要" not in page_source
    assert "概念在不同数据库的可用性" not in page_source
    assert "70%" not in page_source
    assert "90%" not in page_source
    assert "20%" not in page_source


def test_crossdb_loader_avoids_streamlit_app_import_side_effects() -> None:
    viz_source = Path(easyicu.__file__).with_name("cohort_visualization.py").read_text(encoding="utf-8")
    multidb_source = Path(cohort_multidb_page.__file__).read_text(encoding="utf-8")

    assert "from easyicu.webapp.app import find_database_path" not in viz_source
    assert "from easyicu.webapp.data_paths import find_database_path" in viz_source
    assert "len(selected_dbs or []) < 2" in multidb_source
    assert "Loaded fewer than two databases" in multidb_source
    assert "st.session_state.pop('multidb_data'" in multidb_source


def test_crossdb_sidebar_path_hint_keeps_single_detected_root(tmp_path) -> None:
    mimiciv = tmp_path / "mimic-iv-3.1"
    mimiciv.mkdir()

    root, siblings = cohort_multidb_page._detect_sibling_database_root(str(mimiciv))

    assert root == str(tmp_path)
    assert siblings == ["mimic-iv-3.1"]


def test_crossdb_distribution_section_has_spacing_guard() -> None:
    redesign_source = Path(cohort_redesign.__file__).read_text(encoding="utf-8")
    multidb_source = Path(cohort_multidb_page.__file__).read_text(encoding="utf-8")
    css_text = shell_styles._load_shell_overrides_css()

    assert "eu-crossdb-distribution-boundary" in redesign_source
    assert "eu-crossdb-distribution-heading" in multidb_source
    assert "📈 {title}" not in multidb_source
    assert ".stApp .eu-crossdb-distribution-boundary" in css_text
    assert ".stApp .eu-crossdb-distribution-heading" in css_text
    assert "clear: both" in css_text
    assert "min-height: 24px" in css_text
    assert "list(data.items())[:4]" not in multidb_source
    assert "list(data.items())[:6]" in multidb_source


def test_crossdb_real_mode_clears_seeded_demo_frames() -> None:
    state = {
        "entry_mode": "real",
        "multidb_data": {"miiv": pd.DataFrame({"hr": [80]})},
        "multidb_concepts": ["hr"],
        "multidb_is_demo": True,
    }

    assert cohort_redesign._clear_demo_crossdb_state_for_real_mode(state) is True

    assert "multidb_data" not in state
    assert "multidb_concepts" not in state
    assert "multidb_is_demo" not in state

    real_state = {
        "entry_mode": "real",
        "multidb_data": {"miiv": pd.DataFrame({"hr": [80]})},
        "multidb_concepts": ["hr"],
        "multidb_is_demo": False,
    }

    assert cohort_redesign._clear_demo_crossdb_state_for_real_mode(real_state) is False
    assert "multidb_data" in real_state
    assert real_state["multidb_is_demo"] is False


def test_crossdb_demo_workspace_keeps_six_database_story(monkeypatch) -> None:
    state = {"entry_mode": "demo"}
    app._ensure_cohort_demo_workspace(state, lang="en")
    monkeypatch.setattr(cohort_redesign, "st", _SessionStateStreamlit(state))

    cards = cohort_redesign._crossdb_active_databases("en")
    columns, rows = cohort_redesign._crossdb_kpi_rows("en")

    assert [card[0] for card in cards] == [
        "MIMIC-IV",
        "eICU-CRD",
        "AmsterdamUMCdb",
        "HiRID",
        "MIMIC-III",
        "SICdb",
    ]
    assert all("seeded feature rows" in card[1] for card in cards)
    assert all("demo rows" not in card[1] for card in cards)
    assert columns == [
        "Metric",
        "MIMIC-IV",
        "eICU-CRD",
        "AmsterdamUMCdb",
        "HiRID",
        "MIMIC-III",
        "SICdb",
        "Δ range",
    ]
    assert rows[0][1:-1] == ["144", "144", "144", "144", "144", "144"]


def test_crossdb_page_labels_demo_source_before_summary(monkeypatch) -> None:
    class _FakeStreamlit:
        def __init__(self, session_state) -> None:
            self.session_state = session_state
            self.markdown_calls: list[str] = []

        def markdown(self, body, *_args, **_kwargs) -> None:
            self.markdown_calls.append(str(body))

    streamlit_stub = _FakeStreamlit({
        "language": "en",
        "entry_mode": "demo",
        "multidb_is_demo": True,
        "multidb_data": {
            "miiv": pd.DataFrame({"hr": [80, 90]}),
            "eicu": pd.DataFrame({"hr": [82, 92]}),
        },
        "multidb_concepts": ["hr"],
        "_eu_shell_only": True,
    })
    monkeypatch.setattr(cohort_redesign, "st", streamlit_stub)

    cohort_redesign.render_cross_db_redesign_page("en", multidb_fn=lambda _lang: None)

    page_source = "\n".join(streamlit_stub.markdown_calls)
    assert "Demo simulated data" in page_source
    assert "independent seeded feature frames for each database" in page_source
    assert "not the 10-patient review demo or a user database" in page_source
    assert "Loaded cross-database distribution summary" in page_source


def test_cohort_redesign_defaults_to_real_panel_body(monkeypatch) -> None:
    class _FakeStreamlit:
        def __init__(self) -> None:
            self.session_state = {
                "entry_mode": "demo",
                "cohort_active_panel": "coverage",
            }
            self.markdown_calls: list[str] = []
            self.radio_labels: list[str] = []

        def markdown(self, body, *_args, **_kwargs) -> None:
            self.markdown_calls.append(str(body))

        def radio(self, _label, *, options, key, format_func=None, **_kwargs):
            assert options == ["groups", "coverage", "snapshot", "sofa"]
            self.radio_labels = [format_func(option) for option in options]
            return self.session_state[key]

    streamlit_stub = _FakeStreamlit()
    rendered: list[str] = []
    monkeypatch.setattr(cohort_redesign, "st", streamlit_stub)

    cohort_redesign.render_cohort_redesign_page(
        "en",
        group_fn=lambda _lang: rendered.append("groups"),
        coverage_fn=lambda _lang: rendered.append("coverage"),
        snapshot_fn=lambda _lang: rendered.append("snapshot"),
        sofa_fn=lambda _lang: rendered.append("sofa"),
    )

    assert rendered == ["coverage"]
    assert streamlit_stub.radio_labels == [
        "Group contrast",
        "Coverage audit",
        "Cohort profile",
        "SOFA reclassification",
    ]
    page_source = "\n".join(streamlit_stub.markdown_calls)
    assert "Cohort readiness" in page_source
    assert "Review state" in page_source
    assert "10 stays · demo concept set" in page_source
    assert "ready for cohort review" in page_source
    assert "current session" in page_source
    assert "Agent preflight" not in page_source
    assert "Draft gate" not in page_source
    assert "agent drafts only after review" not in page_source
    assert "cohort_statistics:250:0" not in page_source
    assert "250 stays" not in page_source
    cohort_source = Path(cohort_redesign.__file__).read_text(encoding="utf-8")
    assert "_render_cohort_readiness_strip(lang)" in cohort_source
    assert "_render_agent_gate_strip" not in cohort_source

    css_text = shell_styles._load_shell_overrides_css()
    assert "st-key-cohort_active_panel" in css_text
    assert "eu-readiness-strip" in css_text
    assert "grid-template-columns: repeat(4, minmax(0, 1fr))" in css_text
    cohort_panel_css = css_text[
        css_text.index('[class*="st-key-cohort_active_panel"] div[role="radiogroup"]'):
        css_text.index("@media (max-width: 900px)", css_text.index('[class*="st-key-cohort_active_panel"]'))
    ]
    assert "gap: 8px !important" in cohort_panel_css
    assert "display: flex !important" in cohort_panel_css
    assert "display: none !important" not in cohort_panel_css


def test_cohort_redesign_real_unconfigured_does_not_show_demo_denominators(monkeypatch) -> None:
    class _FakeStreamlit:
        def __init__(self) -> None:
            self.session_state = {
                "entry_mode": "real",
                "cohort_active_panel": "groups",
            }
            self.markdown_calls: list[str] = []

        def markdown(self, body, *_args, **_kwargs) -> None:
            self.markdown_calls.append(str(body))

        def radio(self, _label, *, options, key, format_func=None, **_kwargs):
            assert options == ["groups", "coverage", "snapshot", "sofa"]
            return self.session_state[key]

    streamlit_stub = _FakeStreamlit()
    rendered: list[str] = []
    monkeypatch.setattr(cohort_redesign, "st", streamlit_stub)

    cohort_redesign.render_cohort_redesign_page(
        "en",
        group_fn=lambda _lang: rendered.append("groups"),
        coverage_fn=lambda _lang: rendered.append("coverage"),
        snapshot_fn=lambda _lang: rendered.append("snapshot"),
        sofa_fn=lambda _lang: rendered.append("sofa"),
    )

    assert rendered == ["groups"]
    page_source = "\n".join(streamlit_stub.markdown_calls)
    assert "Cohort statistics" in page_source
    assert "Cohort readiness" in page_source
    assert "waiting for local cohort" in page_source
    assert "load data for denominators" in page_source
    assert "review unlocks after data load" in page_source
    assert "Agent preflight" not in page_source
    assert "Draft gate" not in page_source
    assert "agent drafts only after review" not in page_source
    assert "demo concept set" not in page_source
    assert "250 stays" not in page_source
    assert "Sepsis vs Non-sepsis" not in page_source


def test_cohort_redesign_real_loaded_export_uses_patient_ids_for_readiness(monkeypatch) -> None:
    class _FakeStreamlit:
        def __init__(self) -> None:
            self.session_state = {
                "entry_mode": "real",
                "cohort_active_panel": "coverage",
                "patient_ids": [10001, 10002, 10003],
                "loaded_concepts": {
                    "hr": pd.DataFrame({"stay_id": [10001, 10002, 10003]}),
                    "map": pd.DataFrame({"stay_id": [10001, 10002, 10003]}),
                },
            }
            self.markdown_calls: list[str] = []

        def markdown(self, body, *_args, **_kwargs) -> None:
            self.markdown_calls.append(str(body))

        def radio(self, _label, *, options, key, format_func=None, **_kwargs):
            assert options == ["groups", "coverage", "snapshot", "sofa"]
            return self.session_state[key]

    streamlit_stub = _FakeStreamlit()
    rendered: list[str] = []
    monkeypatch.setattr(cohort_redesign, "st", streamlit_stub)

    cohort_redesign.render_cohort_redesign_page(
        "en",
        group_fn=lambda _lang: rendered.append("groups"),
        coverage_fn=lambda _lang: rendered.append("coverage"),
        snapshot_fn=lambda _lang: rendered.append("snapshot"),
        sofa_fn=lambda _lang: rendered.append("sofa"),
    )

    page_source = "\n".join(streamlit_stub.markdown_calls)
    assert rendered == ["coverage"]
    assert "3 stays · 2 concepts" in page_source
    assert "coverage + denominators ready" in page_source
    assert "ready for cohort review" in page_source
    assert "0 stays" not in page_source


def test_cohort_redesign_shell_only_keeps_design_preview_available(monkeypatch) -> None:
    class _FakeStreamlit:
        def __init__(self) -> None:
            self.session_state = {
                "entry_mode": "demo",
                "cohort_active_panel": "groups",
                "_eu_shell_only": True,
            }
            self.markdown_calls: list[str] = []
            self.radio_labels: list[str] = []

        def markdown(self, body, *_args, **_kwargs) -> None:
            self.markdown_calls.append(str(body))

        def radio(self, _label, *, options, key, format_func=None, **_kwargs):
            assert options == ["groups", "coverage", "snapshot", "sofa"]
            self.radio_labels = [format_func(option) for option in options]
            return self.session_state[key]

    streamlit_stub = _FakeStreamlit()
    rendered: list[str] = []
    monkeypatch.setattr(cohort_redesign, "st", streamlit_stub)

    cohort_redesign.render_cohort_redesign_page(
        "en",
        group_fn=lambda _lang: rendered.append("groups"),
        coverage_fn=lambda _lang: rendered.append("coverage"),
        snapshot_fn=lambda _lang: rendered.append("snapshot"),
        sofa_fn=lambda _lang: rendered.append("sofa"),
    )

    assert rendered == []
    assert streamlit_stub.radio_labels == [
        "Group contrast",
        "Coverage audit",
        "Cohort profile",
        "SOFA reclassification",
    ]
    assert any("Mortality by SOFA quartile" in body for body in streamlit_stub.markdown_calls)


def test_crossdb_summary_uses_loaded_data_instead_of_static_design_numbers(monkeypatch) -> None:
    streamlit_stub = _SessionStateStreamlit({
        "multidb_data": {
            "miiv": pd.DataFrame({
                "stay_id": [1, 1, 2, 2],
                "concept": ["hr", "lact", "hr", "lact"],
                "value": [80, 1.2, 100, 2.2],
            }),
            "eicu": pd.DataFrame({
                "stay_id": [10, 11, 12],
                "concept": ["hr", "hr", "lact"],
                "value": [70, 90, 3.1],
            }),
        },
        "multidb_concepts": ["hr", "lact"],
        "multidb_is_demo": True,
    })
    monkeypatch.setattr(cohort_redesign, "st", streamlit_stub)

    columns, rows = cohort_redesign._crossdb_kpi_rows("en")
    payload = json.dumps({"columns": columns, "rows": rows}, ensure_ascii=False)

    assert columns == ["Metric", "MIMIC-IV", "eICU-CRD", "Δ range"]
    assert rows[0] == ["Rows", "4", "3", "1"]
    assert rows[1] == ["Concepts present", "2", "2", "0"]
    assert any(row[0] == "hr median" and row[1:] == ["90.00", "80.00", "10.00"] for row in rows)
    assert "Patients" not in payload
    assert "Distinct IDs" not in payload
    assert "2,481" not in payload
    assert "12,083" not in payload
    assert "Sepsis-3 mortality benchmark" not in payload


def test_topbar_research_agent_demo_action_opens_guide() -> None:
    state = {
        "_eu_topbar_run_request": {"page": "research_agent"},
        "entry_mode": "demo",
    }

    result = app._consume_topbar_run_request(state, "research_agent", "en")

    assert result["level"] == "success"
    assert state["_ra_view"] == "setup"
    assert "guide" in result["message"]
    assert "_eu_topbar_run_request" not in state


def test_topbar_research_agent_action_uses_reference_guide_label_but_opens_setup() -> None:
    assert app._topbar_primary_action_label("research_agent", "en", entry_mode="demo") == (
        "Agent guide",
        "Agent 导览",
    )
    assert app._topbar_primary_action_label("research_agent", "zh", entry_mode="real") == (
        "Agent guide",
        "Agent 导览",
    )

    state = {
        "_eu_topbar_run_request": {"page": "research_agent"},
        "entry_mode": "real",
    }

    result = app._consume_topbar_run_request(state, "research_agent", "en")

    assert result["level"] == "info"
    assert state["_ra_view"] == "setup"
    assert state["_eu_ra_launch_requested"] is True
    assert "run controls" in result["message"]
    assert "_eu_topbar_run_request" not in state


def test_topbar_settings_action_matches_reference_and_resets_defaults() -> None:
    page_source = Path(pages_redesign.__file__).read_text(encoding="utf-8")
    css_text = shell_styles._load_shell_overrides_css()

    assert '_T(lang, "Change", "修改")' in page_source
    assert "_eu_settings_module_folder_mode" in page_source
    assert "Turn on to open Research Agent setup and choose a module export folder." in page_source
    assert "st.toggle(" in page_source
    assert "st.columns([0.72, 0.28]" in page_source
    assert 'st-key-_eu_settings_demo_patients_' in css_text
    assert 'background: white !important;' in css_text
    assert "Reduce motion" in page_source
    assert "Disable shimmer and progress animations." in page_source
    assert "st-key-_eu_settings_module_folder_mode" in css_text
    assert 'key="_eu_settings_density_compact"' in page_source
    assert 'key="_eu_settings_reduce_motion"' in page_source
    assert "def _settings_start_path_edit" in page_source
    assert "def _settings_apply_path_edit" in page_source
    assert 'key="_eu_settings_workdir_input"' in page_source
    assert 'key="_eu_settings_save_workdir"' in page_source
    assert 'key="_eu_settings_export_path_input"' in page_source
    assert 'key="_eu_settings_save_export_path"' in page_source
    assert 'state_key="research_agent_workdir"' in page_source
    assert 'state_key="export_path"' in page_source
    assert 'state["sidebar_export_path_input"] = normalized' in page_source
    assert "def _settings_reduce_motion_changed" in page_source
    assert 'if bool(state.get("_eu_settings_allow_outbound_model_calls", False)) != outbound_enabled:' in page_source
    assert "_settings_outbound_model_calls_changed()\n                st.rerun()" in page_source
    assert "st-key-_eu_settings_density_" in css_text
    assert "st-key-_eu_settings_reduce_motion" in css_text
    assert "_route_to_research_agent_setup(\n                st.session_state," in page_source
    assert "_route_to_research_agent_setup(state, force_real=True)" in page_source
    assert "focus_module_folder=True" in page_source
    assert "_eu_ra_focus_module_folder" in Path(research_agent.__file__).read_text(encoding="utf-8")
    assert '"Pick an EasyICU module export folder"' in page_source
    assert "Release notes" in page_source
    assert "Documentation" in page_source
    assert "Export diagnostics" in page_source
    assert "_settings_diagnostics_json" in page_source
    assert "secrets_included" in page_source
    assert "patient_rows_included" in page_source
    assert 'key="_eu_settings_diagnostics_download"' in page_source
    assert "eu_settings_env_actions" in css_text
    assert ".stDownloadButton > button" in css_text

    assert app._topbar_primary_action_label("settings", "en") == (
        "Reset to defaults",
        "恢复默认",
    )
    assert app._topbar_primary_action_icon("settings") == ":material/refresh:"

    state = {
        "_eu_topbar_run_request": {"page": "settings"},
        "entry_mode": "real",
        "use_mock_data": False,
        "database": "miiv",
        "demo_mode_patients": 50,
        "demo_mode_hours": 168,
        "mock_params": {"n_patients": 50, "hours": 168, "demo_profile": "full"},
        "llm_enabled": True,
        "llm_provider": "openrouter",
        "llm_api_key": "sk-test",
        "llm_model": "custom-model",
        "llm_base_url": "https://example.invalid/v1",
        "llm_configured": True,
        "_eu_settings_allow_outbound_model_calls": True,
        "_eu_settings_reduce_motion": True,
        "ui_density": "compact",
        "reduce_motion": True,
        "_llm_provider_sel": "openrouter",
        "_llm_api_key_inp": "sk-test",
        "_llm_base_url_inp": "https://example.invalid/v1",
        "_llm_model_inp": "custom-model",
        "data_path": "/tmp/old-real-source",
        "path_validated": True,
        "last_validated_path": "/tmp/old-real-source",
        "sidebar_data_path_input": "/tmp/old-real-source",
        "research_agent_extract_data_path": "/tmp/old-agent-source",
        "research_agent_extract_db": "mimic",
        "_research_agent_extract_db_source": "mimic",
        "research_agent_resume_run_id": "run_20260531T121512_3a91c8",
        "research_agent_force_manuscript": True,
        "research_agent_resume_mode": "force_manuscript",
        "research_agent_resume_notes": "stale review note",
        "research_agent_resume_relax_probe": True,
        "research_agent_preflight_confirmed": True,
        "research_agent_preflight_signature": "stale-signature",
        "_agent_workbench": {"run_id": "run_20260531T121512_3a91c8"},
        "_agent_workbench_source_run_dir": "/tmp/run_old",
        "_agent_workbench_is_active_selection": True,
        "_eu_ra_launch_requested": True,
        "_eu_ra_focus_module_folder": True,
        "_eu_ra_module_pick_force_manual": True,
        "_eu_ra_apply_export_file_selection": True,
        "_eu_wb_findings_acked": {"finding-a"},
        "_eu_wb_findings_acked_run_dir": "/tmp/run_old",
        "_eu_wb_review_details_expanded": True,
        "_eu_wb_action_panel": "plan",
        "_eu_summary_review_note_run_20260531T121512_3a91c8": "stale note",
        "_eu_wb_ev_sha_show_step_key": True,
        "_eu_wb_ev_id_show_step_key": True,
        "_eu_wb_evidence_pick_step_key": "01",
        "_eu_wb_timeline_jump_run_key": "03",
        "_ra_view": "summary",
        "export_completed": True,
        "_post_export_navigation_pending": True,
        "_post_export_target_panel": "Data Tables",
        "_post_export_guidance_dismissed": False,
        "_export_cancel_notice": "Export stopped by user.",
    }

    result = app._consume_topbar_run_request(state, "settings", "en")

    assert result == {"level": "success", "message": "Settings reset to workspace defaults."}
    assert state["entry_mode"] == "demo"
    assert state["use_mock_data"] is True
    assert state["database"] == "mock"
    assert state["demo_mode_patients"] == app.LIGHTWEIGHT_DEMO_PATIENTS
    assert state["demo_mode_hours"] == app.LIGHTWEIGHT_DEMO_HOURS
    assert state["mock_params"] == {
        "n_patients": app.LIGHTWEIGHT_DEMO_PATIENTS,
        "hours": app.LIGHTWEIGHT_DEMO_HOURS,
        "demo_profile": "lite",
    }
    assert state["llm_enabled"] is False
    assert state["llm_provider"] in {"easyicu_hosted", "openrouter"}
    assert state["llm_api_key"] == ""
    assert state["llm_model"] == ""
    assert state["llm_base_url"] == ""
    assert state["llm_configured"] is False
    assert state["_llm_toggle"] is False
    assert state["_llm_toggle_sync_pending"] is True
    assert state["_eu_settings_allow_outbound_model_calls"] is False
    assert state["_eu_settings_reduce_motion"] is False
    assert state["ui_density"] == "comfortable"
    assert state["reduce_motion"] is False
    assert state["_llm_provider_sel"] == state["llm_provider"]
    assert state["_llm_api_key_inp"] == ""
    assert state["_llm_base_url_inp"] == ""
    assert state["_llm_model_inp"] == ""
    assert state["data_path"] is None
    assert state["path_validated"] is False
    assert "last_validated_path" not in state
    assert "sidebar_data_path_input" not in state
    assert "research_agent_extract_data_path" not in state
    assert "research_agent_extract_db" not in state
    assert "_research_agent_extract_db_source" not in state
    assert state["export_completed"] is True
    assert state["_post_export_guidance_dismissed"] is True
    assert "_post_export_navigation_pending" not in state
    assert "_post_export_target_panel" not in state
    assert "_export_cancel_notice" not in state
    assert state["_ra_view"] == "setup"
    for key in (
        "research_agent_resume_run_id",
        "research_agent_force_manuscript",
        "research_agent_resume_mode",
        "research_agent_resume_notes",
        "research_agent_resume_relax_probe",
        "research_agent_preflight_confirmed",
        "research_agent_preflight_signature",
        "_agent_workbench",
        "_agent_workbench_source_run_dir",
        "_agent_workbench_is_active_selection",
        "_eu_ra_launch_requested",
        "_eu_ra_focus_module_folder",
        "_eu_ra_module_pick_force_manual",
        "_eu_ra_apply_export_file_selection",
        "_eu_wb_findings_acked",
        "_eu_wb_findings_acked_run_dir",
        "_eu_wb_review_details_expanded",
        "_eu_wb_action_panel",
        "_eu_summary_review_note_run_20260531T121512_3a91c8",
        "_eu_wb_ev_sha_show_step_key",
        "_eu_wb_ev_id_show_step_key",
        "_eu_wb_evidence_pick_step_key",
        "_eu_wb_timeline_jump_run_key",
    ):
        assert key not in state
    assert "_eu_topbar_run_request" not in state


def test_settings_diagnostics_payload_omits_secrets_and_patient_rows() -> None:
    payload_text = pages_redesign._settings_diagnostics_json(
        {
            "entry_mode": "real",
            "database": "miiv",
            "use_mock_data": False,
            "demo_mode_patients": 50,
            "demo_mode_hours": 168,
            "llm_enabled": True,
            "llm_provider": "openrouter",
            "llm_api_key": "sk-secret-should-not-export",
            "llm_model": "anthropic/claude-sonnet",
            "patient_ids": ["patient-001", "patient-002"],
            "research_agent_resume_run_id": "run_20260531T000000_demo",
        },
        lang="en",
        workdir="/tmp/easyicu-workspace",
        export_hint="/tmp/easyicu-export",
        provider_label="OpenRouter",
        model_label="anthropic/claude-sonnet",
        base_url_label="https://openrouter.ai/api/v1",
        provider_needs_key=True,
        api_key_present=True,
        agent_run_value="Ready",
    )

    payload = json.loads(payload_text)

    assert "sk-secret-should-not-export" not in payload_text
    assert "llm_api_key" not in payload_text
    assert "patient-001" not in payload_text
    assert payload["llm"]["credential_state"] == "present"
    assert payload["privacy"]["secrets_included"] is False
    assert payload["privacy"]["patient_rows_included"] is False


def test_settings_diagnostics_uses_visible_workbench_run_over_stale_resume(tmp_path: Path) -> None:
    visible_run_dir = tmp_path / "run_visible"
    visible_run_dir.mkdir()
    stale_last_dir = tmp_path / "run_last_fallback"
    stale_last_dir.mkdir()

    payload = json.loads(
        pages_redesign._settings_diagnostics_json(
            {
                "_ra_view": "workbench",
                "_agent_workbench": {"run_id": "run_visible"},
                "_agent_workbench_source_run_dir": str(visible_run_dir),
                "_agent_workbench_is_active_selection": True,
                "research_agent_resume_run_id": "run_deleted_resume",
                "research_agent_last_run_id": "run_last_fallback",
                "llm_provider": "openrouter",
            },
            lang="en",
            workdir=str(tmp_path),
            export_hint="/tmp/easyicu-export",
            provider_label="OpenRouter",
            model_label="openai/gpt-oss-120b:free",
            base_url_label="https://openrouter.ai/api/v1",
            provider_needs_key=True,
            api_key_present=False,
            agent_run_value="Ready",
        )
    )

    assert payload["research_agent"]["last_run_id"] == "run_visible"


def test_settings_diagnostics_ignores_deleted_resume_run(tmp_path: Path) -> None:
    fallback_dir = tmp_path / "run_existing_last"
    fallback_dir.mkdir()

    payload = json.loads(
        pages_redesign._settings_diagnostics_json(
            {
                "_ra_view": "workbench",
                "_agent_workbench": {"run_id": "run_deleted_workbench"},
                "_agent_workbench_source_run_dir": str(tmp_path / "run_deleted_workbench"),
                "_agent_workbench_is_active_selection": True,
                "research_agent_resume_run_id": "run_deleted_resume",
                "research_agent_last_run_id": "run_existing_last",
            },
            lang="en",
            workdir=str(tmp_path),
            export_hint="/tmp/easyicu-export",
            provider_label="OpenRouter",
            model_label="openai/gpt-oss-120b:free",
            base_url_label="https://openrouter.ai/api/v1",
            provider_needs_key=True,
            api_key_present=False,
            agent_run_value="Ready",
        )
    )

    assert payload["research_agent"]["last_run_id"] == "run_existing_last"


def test_history_open_workbench_clears_resume_markers_and_records_last_run() -> None:
    source = Path(research_agent.__file__).read_text(encoding="utf-8")

    assert "clear_agent_continuation_state(st.session_state)" in source
    assert 'st.session_state["research_agent_last_run_id"] = run_id' in source


def test_settings_reset_request_reruns_shell_and_preserves_notice(monkeypatch) -> None:
    class _RerunRequested(Exception):
        pass

    class _SettingsResetStreamlit:
        def __init__(self) -> None:
            self.session_state = {
                "_eu_topbar_run_request": {"page": "settings"},
                "entry_mode": "real",
                "use_mock_data": False,
                "database": "miiv",
            }
            self.toasts: list[str] = []

        def toast(self, message: str) -> None:
            self.toasts.append(message)

        def rerun(self) -> None:
            raise _RerunRequested

    streamlit_stub = _SettingsResetStreamlit()
    monkeypatch.setattr(app, "st", streamlit_stub)

    with pytest.raises(_RerunRequested):
        app._handle_topbar_run_request("settings", "en")

    assert streamlit_stub.session_state["entry_mode"] == "demo"
    assert streamlit_stub.session_state["database"] == "mock"
    assert streamlit_stub.session_state["_eu_topbar_notice_pending"] == (
        "Settings reset to workspace defaults."
    )

    assert app._handle_topbar_run_request("settings", "en") is None
    assert streamlit_stub.toasts == ["Settings reset to workspace defaults."]
    assert "_eu_topbar_notice_pending" not in streamlit_stub.session_state


def test_topbar_quick_viz_request_reruns_shell_after_state_change(monkeypatch) -> None:
    class _RerunRequested(Exception):
        pass

    class _QuickVizStreamlit:
        def __init__(self) -> None:
            self.session_state = {
                "_eu_topbar_run_request": {"page": "quick_viz"},
                "entry_mode": "demo",
            }
            self.toasts: list[str] = []

        def toast(self, message: str) -> None:
            self.toasts.append(message)

        def rerun(self) -> None:
            raise _RerunRequested

    def _consume_request(state, active_page, lang):
        assert active_page == "quick_viz"
        assert lang == "en"
        state.pop("_eu_topbar_run_request", None)
        state["loaded_concepts"] = {"age": object(), "hr": object()}
        state["selected_concepts"] = ["age", "hr"]
        return {"level": "success", "message": "Loaded lightweight demo review workspace."}

    streamlit_stub = _QuickVizStreamlit()
    monkeypatch.setattr(app, "st", streamlit_stub)
    monkeypatch.setattr(app, "_consume_topbar_run_request", _consume_request)

    with pytest.raises(_RerunRequested):
        app._handle_topbar_run_request("quick_viz", "en")

    assert streamlit_stub.session_state["selected_concepts"] == ["age", "hr"]
    assert streamlit_stub.session_state["_eu_topbar_notice_pending"] == (
        "Loaded lightweight demo review workspace."
    )
    assert streamlit_stub.toasts == []


def test_sidebar_brand_home_button_returns_to_entry(monkeypatch) -> None:
    class _AttrState(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        def __setattr__(self, key, value):
            self[key] = value

    state = _AttrState({
        "entry_mode": "demo",
        "use_mock_data": True,
        "_active_main_page": "cohort",
    })
    streamlit_stub = _SessionStateStreamlit(state)
    clear_calls: list[str] = []

    monkeypatch.setattr(sidebar, "st", streamlit_stub)
    monkeypatch.setattr(sidebar, "clear_run_state", lambda scope="all": clear_calls.append(scope))

    sidebar._return_to_entry_home()

    assert clear_calls == ["all"]
    assert state["entry_mode"] == "none"
    assert state["use_mock_data"] is False
    assert state["_active_main_page"] == "tutorial"

    sidebar_source = Path(sidebar.__file__).read_text(encoding="utf-8")
    css_text = shell_styles._load_shell_overrides_css()
    assert 'key="_eu_brand_home_button"' in sidebar_source
    assert "_return_to_entry_home()" in sidebar_source
    assert "st-key-eu_brand_home" in css_text
    assert "st-key-_eu_brand_home_button" in css_text
    assert 'icon("flask")' in Path(ui_helpers.__file__).read_text(encoding="utf-8")
    assert "background: var(--ink);" in css_text
    assert ".eu-brand .logo svg" in css_text
    assert "cursor: pointer !important" in css_text
    assert "line-height: 0 !important" in css_text
    assert "-webkit-text-fill-color: transparent !important" in css_text
    assert 'st-key-_eu_brand_home_button"] button:focus-visible' in css_text
    assert 'st-key-_eu_brand_home_button"] button:active' in css_text
    assert 'st-key-_eu_brand_home_button"] button::before' in css_text
    assert "background-color: transparent !important" in css_text
    assert "outline: none !important" in css_text


def test_sidebar_workspace_session_card_is_removed_from_extract_shell() -> None:
    sidebar_source = Path(sidebar.__file__).read_text(encoding="utf-8")
    css_text = shell_styles._load_shell_overrides_css()

    assert not hasattr(sidebar, "_session_summary_html")
    assert "eu-session-card" not in sidebar_source
    assert "Demo workspace" not in sidebar_source
    assert "Ready demo cohort" not in sidebar_source
    assert "eu-workspace-label" not in sidebar_source
    assert ".eu-session-card" not in css_text


def test_sidebar_context_summary_uses_plain_setup_language(monkeypatch) -> None:
    streamlit_stub = _SessionStateStreamlit({
        "mock_params": {"n_patients": 64},
        "step2_confirmed": False,
        "selected_concepts": [],
    })
    monkeypatch.setattr(sidebar, "st", streamlit_stub)

    html = sidebar._context_summary_html("demo", "en")

    assert "Current setup" in html
    assert "Dataset" in html
    assert "Demo · 64 patients" in html
    assert "demo defaults" in html
    assert "Data context" not in html
    assert "not a project" not in html

    streamlit_stub.session_state.update(
        {
            "step2_confirmed": True,
            "cohort_enabled": True,
            "cohort_filter": {
                "age_min": None,
                "age_max": None,
                "first_icu_stay": None,
                "los_min": None,
                "los_max": None,
                "gender": None,
                "survived": None,
                "has_sepsis": None,
                "disease_cohort": "none",
                "icd_query": "",
                "icd_include_query": "",
                "icd_exclude_query": "",
                "icd_mode": "include",
            },
        }
    )

    confirmed_html = sidebar._context_summary_html("demo", "en")

    assert "all stays" in confirmed_html
    assert "configured" not in confirmed_html


def test_sidebar_agent_page_uses_reference_agent_state_rail(monkeypatch) -> None:
    streamlit_stub = _SessionStateStreamlit({
        "mock_params": {"n_patients": 10},
        "_active_main_page": "research_agent",
        "_agent_workbench": {
            "run_id": "run_20260528T184052_adaf4d",
            "evidence_total": 122,
            "audit": {"counts": {"warnings": 1}},
            "is_demo": False,
        },
    })
    monkeypatch.setattr(sidebar, "st", streamlit_stub)

    html = sidebar._agent_state_summary_html("demo", "en")
    source = Path(sidebar.__file__).read_text(encoding="utf-8")
    css_text = Path(sidebar.__file__).with_name("shell_overrides.css").read_text(encoding="utf-8")

    assert "Agent state" in html
    assert "Needs review" in html
    assert "Mode" in html
    assert "Local run" in html
    assert "Evidence" in html
    assert "122 evidence" in html
    assert "Demo" not in html
    assert "sepsis · 10" not in html
    assert "Last run" in html
    assert "Guarantees" in html
    assert "Local-first · no upload" in html
    assert "Draft gated on evidence" in html
    assert "Human confirms each run" in html
    assert 'active == "research_agent"' in source
    assert "eu_context_edit_setup" not in html
    assert ".eu-agent-state-pill" in css_text
    assert ".eu-agent-guarantee-row" in css_text


def test_sidebar_agent_setup_view_describes_new_setup_not_previous_run(monkeypatch) -> None:
    streamlit_stub = _SessionStateStreamlit({
        "entry_mode": "real",
        "_active_main_page": "research_agent",
        "_ra_view": "setup",
        "_agent_workbench": {
            "run_id": "run_20260601T090510_c7b109",
            "evidence_total": 48,
            "audit": {"counts": {"errors": 3, "warnings": 9}},
            "is_demo": False,
        },
        "research_agent_cohort_source": i18n.TEXTS["en"]["ra_source_synthetic"],
        "research_agent_synth_n": 800,
    })
    monkeypatch.setattr(sidebar, "st", streamlit_stub)

    html = sidebar._agent_state_summary_html("real", "en")

    assert "Setup" in html
    assert "Mode" in html
    assert "Real" in html
    assert "Cohort" in html
    assert "800 rows" in html
    assert "Last run" in html
    assert "run_202…c7b109" in html
    assert "Local run" not in html
    assert "Evidence" not in html
    assert "48 evidence" not in html
    assert "Review" not in html


def test_sidebar_agent_page_keeps_demo_context_when_no_manifest(monkeypatch) -> None:
    streamlit_stub = _SessionStateStreamlit({
        "mock_params": {"n_patients": 10},
        "_active_main_page": "research_agent",
        "_agent_workbench": {"is_demo": True, "audit": {"counts": {}}},
    })
    monkeypatch.setattr(sidebar, "st", streamlit_stub)

    html = sidebar._agent_state_summary_html("demo", "en")

    assert "Mode" in html
    assert "Demo" in html
    assert "Cohort" in html
    assert "sepsis · 10" in html
    assert "Last run" in html
    assert "preview" in html
    assert "Local run" not in html


def test_sidebar_agent_rail_reports_agent_cohort_not_extraction_step(monkeypatch) -> None:
    streamlit_stub = _SessionStateStreamlit({
        "_active_main_page": "research_agent",
        "_agent_workbench": {"is_demo": True, "audit": {"counts": {}}},
        "step2_confirmed": True,
        "selected_concepts": ["hr", "death"],
        "loaded_concepts": {},
    })
    monkeypatch.setattr(sidebar, "st", streamlit_stub)

    html = sidebar._agent_state_summary_html("real", "en")

    assert "Cohort" in html
    assert "not selected" in html
    assert "configured" not in html

    streamlit_stub.session_state.update({
        "loaded_concepts": {"hr": object()},
        "patient_ids": [1, 2, 3, 4],
        "research_agent_cohort_source": i18n.TEXTS["en"]["ra_source_module_folder"],
        "research_agent_module_dir_text": "/tmp/easyicu_export",
    })

    module_pending_html = sidebar._agent_state_summary_html("real", "en")

    assert "export ready to build" in module_pending_html
    assert "4 loaded rows" not in module_pending_html
    assert "not selected" not in module_pending_html

    streamlit_stub.session_state["research_agent_module_built"] = {
        "df": pd.DataFrame({"stay_id": [1, 2, 3]}),
    }

    built_html = sidebar._agent_state_summary_html("real", "en")

    assert "3 rows" in built_html
    assert "not selected" not in built_html

    streamlit_stub.session_state.pop("research_agent_module_built", None)
    streamlit_stub.session_state["research_agent_cohort_source"] = i18n.TEXTS["en"]["ra_source_synthetic"]
    streamlit_stub.session_state["research_agent_synth_n"] = 800

    synthetic_html = sidebar._agent_state_summary_html("real", "en")

    assert "800 rows" in synthetic_html
    assert "not selected" not in synthetic_html


def test_sidebar_agent_state_uses_reviewed_warning_and_signoff(monkeypatch) -> None:
    finding = {"severity": "warning", "validator": "critic", "message": "Check table 1."}
    from easyicu.webapp.agent_workbench import _finding_review_id, _finding_review_state_summary

    reviewed_id = _finding_review_id(finding)
    workbench = {
        "run_id": "run_20260528T184052_adaf4d",
        "audit": {
            "counts": {"warnings": 1},
            "findings": [finding],
            "review_decision": {},
        },
        "reviewed_finding_ids": [reviewed_id],
    }
    finding_state = _finding_review_state_summary(workbench)
    workbench["audit"]["review_decision"] = {
        "decision": "approved",
        "finding_review_signature": finding_state["finding_review_signature"],
    }

    streamlit_stub = _SessionStateStreamlit({
        "mock_params": {"n_patients": 10},
        "_eu_wb_findings_acked": [reviewed_id],
        "_agent_workbench": workbench,
    })
    monkeypatch.setattr(sidebar, "st", streamlit_stub)

    html = sidebar._agent_state_summary_html("real", "en")

    assert "Ready" in html
    assert "Needs review" not in html
    assert "Sign-off" not in html


def test_sidebar_agent_state_separates_backend_gate_followup(monkeypatch) -> None:
    streamlit_stub = _SessionStateStreamlit({
        "mock_params": {"n_patients": 10},
        "_agent_workbench": {
            "run_id": "run_backend_gate",
            "audit": {
                "counts": {"errors": 0, "warnings": 0},
                "gates": [{"label": "numeric verified", "ok": False}],
                "review_decision": {},
            },
        },
    })
    monkeypatch.setattr(sidebar, "st", streamlit_stub)

    html = sidebar._agent_state_summary_html("real", "en")

    assert "Gate follow-up" in html
    assert "Review" not in html
    assert "Ready" not in html


def test_research_agent_mobile_view_switcher_matches_handoff_stack() -> None:
    css_text = shell_styles._load_shell_overrides_css()

    assert "Research Agent has dense setup" in css_text
    assert '@media (max-width: 640px)' in css_text
    assert 'st-key-_eu_ra_tabs' in css_text
    assert "flex-direction: column !important" in css_text
    assert "width: 100% !important" in css_text
    assert "border-color: var(--ink) !important" in css_text
    assert "background: var(--ink) !important" in css_text


def test_mobile_shell_hides_native_sidebar_toggle_and_keeps_topbar_crumbs_inline() -> None:
    css_text = shell_styles._load_shell_overrides_css()
    native_toggle_idx = css_text.index(
        'button[data-testid="stBaseButton-headerNoPadding"][kind="headerNoPadding"]'
    )
    mobile_css = css_text[
        css_text.rindex("@media (max-width: 900px)", 0, native_toggle_idx):
        css_text.index("@media (max-width: 520px)", native_toggle_idx)
    ]

    assert '.stApp button[data-testid="stBaseButton-headerNoPadding"][kind="headerNoPadding"]' in mobile_css
    assert "pointer-events: none !important" in mobile_css
    assert ".stApp .eu-topbar .bc" in mobile_css
    assert ".stApp .eu-topbar .crumbs" in mobile_css
    assert "overflow-x: auto" in mobile_css
    assert "flex-wrap: nowrap" in mobile_css


def test_sidebar_spacing_and_removes_noninteractive_rail_guide(monkeypatch) -> None:
    streamlit_stub = _SessionStateStreamlit({
        "language": "en",
        "mock_params": {"n_patients": 64},
        "step2_confirmed": False,
        "selected_concepts": [],
    })
    monkeypatch.setattr(sidebar, "st", streamlit_stub)

    nav_items = sidebar._shell_nav_items("demo")
    nav_by_key = {item.key: item for item in nav_items}

    assert [item.key for item in nav_items] == [
        "extract",
        "quick_viz",
        "cohort",
        "cross_db",
        "research_agent",
    ]
    assert nav_by_key["extract"].label == "Data Extraction"
    assert nav_by_key["quick_viz"].label == "Patient Review"
    assert nav_by_key["quick_viz"].level == "child"
    assert nav_by_key["cohort"].level == "child"
    assert nav_by_key["cross_db"].level == "child"
    assert nav_by_key["research_agent"].level == "top"

    assert not hasattr(sidebar, "_sidebar_next_steps_html")
    css_text = shell_styles._load_shell_overrides_css()

    assert ".eu-context-label" in css_text
    assert ".eu-context-card" in css_text
    assert "background: transparent;" in css_text
    assert "padding-top: 14px" in css_text
    assert "margin-top: 14px !important" in css_text
    assert "st-key-eunavrow_extract" in css_text
    assert ".eu-nav-group-label" in css_text
    assert ".eu-nav-group-label.design-section" in css_text
    assert ".eu-nav-item.level-child" in css_text
    assert "st-key-eunavrow_visualization" in css_text
    assert "st-key-eunavchildren_visualization" in css_text
    assert ".eu-nav-children-title" not in css_text
    assert "left: calc(100% - 8px)" not in css_text
    assert "visibility: hidden !important" not in css_text
    assert "st-key-eunavrow_research_agent" in css_text
    assert "margin-top: 12px !important" in css_text
    assert "@media (max-width: 900px)" in css_text
    assert "display: none !important;" in css_text
    assert "min-height: 40px !important" in css_text
    assert "eu-side-guide" not in css_text
    sidebar_text = Path(sidebar.__file__).read_text(encoding="utf-8")
    assert "_eu_visualization_nav_open" in sidebar_text
    assert "def _render_shell_aux_nav" in sidebar_text
    assert "_render_shell_primary_nav()\n                _render_shell_aux_nav()" in sidebar_text
    assert 'if active_main_page in {"assistant", "tutorial", "states"}' not in sidebar_text
    assert 'key=f"eunavrow_{item.key}"' in sidebar_text
    assert "assistant" in sidebar_text
    assert "tutorial" in sidebar_text
    assert "states" in sidebar_text
    assert '"AI Assistant" if lang == "en" else "AI 助手"' in sidebar_text
    assert '"Get Started" if lang == "en" else "开始使用"' in sidebar_text
    assert '"Workspace States" if lang == "en" else "工作区状态"' in sidebar_text
    assert 'st.session_state["_active_main_page"] = "assistant"' in sidebar_text
    assert 'st.session_state["_main_nav_widget"] = "assistant"' in sidebar_text
    assert 'st.session_state["_inline_ai_panel_open"] = False' in sidebar_text
    assert 'st.session_state["_active_main_page"] = item.key' in sidebar_text
    assert "eu_footer_help_active" in sidebar_text
    assert "eu_footer_settings_active" in sidebar_text
    assert 'count="3"' not in sidebar_text
    assert "Choose view" not in sidebar_text
    assert 'key="_eu_footer_settings"' in sidebar_text
    assert 'st.session_state["_active_main_page"] = "settings"' in sidebar_text
    assert "API Key 只保存在当前会话" in sidebar_text
    assert "render_sidebar_chat_widget" not in sidebar_text
    assert "_real_data_source_ready()" in sidebar_text
    assert "Path after setup" not in sidebar_text

    app_text = Path(app.__file__).read_text(encoding="utf-8")
    llm_text = Path(app.__file__).with_name("llm_chat.py").read_text(encoding="utf-8")
    ai_optin_text = Path(app.__file__).with_name("ai_optin.py").read_text(encoding="utf-8")
    assert "render_floating_chat_dock()" not in app_text
    assert "render_inline_ai_panel()" not in app_text
    assert "render_ai_assistant_page(lang)" in app_text
    assert "_open_embedded_ai_assistant" in app_text
    assert "_inline_ai_panel_open" in app_text
    assert 'if active_page != "assistant":' in app_text
    assert '_clear_assistant_surfaces(st.session_state, clear_pending=True)' in app_text
    assert "Show floating AI assistant" not in llm_text
    assert "Show floating AI assistant" not in ai_optin_text
    assert "per-run external LLM opt-in" in ai_optin_text
    assert "Use the bottom-right chat button" not in llm_text
    assert "render_inline_ai_panel" in llm_text
    assert "def render_ai_assistant_page" in llm_text
    assert "force_open: bool = False" in llm_text
    assert "allow_close: bool = True" in llm_text
    assert 'active_page = st.session_state.get("_active_main_page")' in llm_text
    assert 'if active_page != "assistant":' in llm_text
    assert ".eu-ai-page-head" in shell_styles._load_shell_overrides_css()
    assert "st-key-ai_assistant_page_panel" in shell_styles._load_shell_overrides_css()
    assert "st-key-inline_ai_assistant_panel" in llm_text
    assert "EasyICU Assistant" in llm_text
    assert "evidence-bound" in llm_text
    assert "EasyICU hosted · evidence-bound" in llm_text
    assert "EasyICU · research helper" in llm_text
    assert "GPT-OSS · 研究助手" not in llm_text
    assert "gpt-oss · local · evidence-bound" not in llm_text
    assert "_render_ai_assistant_workspace_page" in llm_text
    assert "def _submit_prompt_background" in llm_text
    assert "background_pending_prompts=True" in llm_text
    assert "You can switch pages while I work" in llm_text
    assert "render_inline_ai_panel(force_open=True" not in llm_text
    assert '[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])' in llm_text
    assert "max-width: min(650px, 68%)" in llm_text
    assert "inline-ai-status-strip" in llm_text
    assert ".st-key-_inline_ai_close" in llm_text
    assert "flex-wrap: nowrap !important" in llm_text
    assert 'st.chat_message(role, avatar=avatar)' in llm_text
    assert 'avatar = ":material/person:" if role == "user" else ":material/smart_toy:"' in llm_text
    assert '[data-testid="stChatMessageAvatarUser"]::after' in llm_text
    assert 'content: "YOU";' in llm_text
    assert 'content: "AI";' in llm_text
    assert 'st.form_submit_button(\n                "→"' in llm_text
    assert '"Export Chat" if lang == "en" else "导出对话"' in llm_text
    assert 'icon=":material/download:"' in llm_text
    assert '"Clear Chat" if lang == "en" else "清空对话"' in llm_text
    assert 'icon=":material/delete:"' in llm_text
    assert "📄 Export Chat" not in llm_text
    assert "🗑️ Clear Chat" not in llm_text
    assert "🤔 Thinking..." not in llm_text
    assert "✍️ Generating response..." not in llm_text
    for legacy_emoji in ("🤖", "💬", "💡", "🛠️", "📦", "🔑", "⚙️", "✅", "⚠️", "❌", "⏳"):
        assert legacy_emoji not in llm_text
    assert 'icon=":material/smart_toy:"' in llm_text
    assert "_render_inline_ai_context_and_handoff" in llm_text


def test_ai_assistant_real_context_card_does_not_invent_demo_counts() -> None:
    real_empty_state = {"entry_mode": "real", "database": "miiv"}

    html = llm_chat._inline_ai_context_html("en", state=real_empty_state)

    assert "No cohort loaded" in html
    assert "real data · waiting for local export" in html
    assert "MIIV" in html
    assert "10 stays" not in html
    assert "19 modules" not in html
    assert "sepsis_mortality_demo" not in html

    mock_handoff_html = llm_chat._inline_ai_context_html(
        "en",
        state={"entry_mode": "real", "database": "mock", "use_mock_data": True},
    )
    assert "No cohort loaded" in mock_handoff_html
    assert "mock extraction · waiting for local export" in mock_handoff_html
    assert "MOCK" in mock_handoff_html
    assert "real data · waiting for local export" not in mock_handoff_html
    assert "10 stays" not in mock_handoff_html
    assert "19 modules" not in mock_handoff_html
    assert "sepsis_mortality_demo" not in mock_handoff_html

    demo_html = llm_chat._inline_ai_context_html("en", state={"entry_mode": "demo"})
    assert "sepsis_mortality_demo" in demo_html
    assert "demo · 10 stays · 19 modules" in demo_html

    demo_loaded_html = llm_chat._inline_ai_context_html(
        "en",
        state={
            "entry_mode": "demo",
            "patient_ids": [10001, 10002],
            "loaded_concepts": {"hr": object(), "map": object()},
        },
    )
    assert "demo · 2 stays · 2 features" in demo_loaded_html
    assert "demo · 2 stays · 2 modules" not in demo_loaded_html


def test_sidebar_settings_gear_opens_full_settings_page() -> None:
    sidebar_text = Path(sidebar.__file__).read_text(encoding="utf-8")
    llm_text = Path(sidebar.__file__).with_name("llm_chat.py").read_text(encoding="utf-8")
    pages_text = Path(pages_redesign.__file__).read_text(encoding="utf-8")
    app_text = Path(app.__file__).read_text(encoding="utf-8")
    css_text = shell_styles._load_shell_overrides_css()
    footer_source = sidebar_text[
        sidebar_text.index("def _render_shell_footer_icons"):
        sidebar_text.index("def _render_sidebar_settings_panel")
    ]
    settings_external_model_source = pages_text[
        pages_text.index('key="_eu_settings_model_external"'):
        pages_text.index("st.markdown(\n                '<div class=\"eu-settings-route-note\">")
    ]

    assert 'key="_eu_footer_settings"' in sidebar_text
    assert "st.popover" not in footer_source
    assert 'st.session_state["_active_main_page"] = "settings"' in footer_source
    assert 'render_settings_redesign_page(lang)' in app_text
    assert "def render_settings_redesign_page" in pages_text
    assert "Local paths" in pages_text
    assert "Defaults for new sessions" in pages_text
    assert "Local-first guarantees" in pages_text
    assert "Run behavior" in pages_text
    assert "key=\"_eu_settings_allow_outbound_model_calls\"" in pages_text
    assert "def _settings_outbound_model_calls_changed" in pages_text
    assert "_settings_outbound_model_calls_changed()\n                st.rerun()" in pages_text
    llm_settings_source = Path(llm_chat.__file__).read_text(encoding="utf-8")
    assert "provider_changed = provider != current_provider" in llm_settings_source
    assert 'st.session_state["_llm_base_url_inp"] = default_url' in llm_settings_source
    assert "st.rerun()" in llm_settings_source[
        llm_settings_source.index("if provider_changed:"):
        llm_settings_source.index("desc = p_info")
    ]
    assert "Shared outbound calls are on; Research Agent still shows a per-run disclosure gate." in llm_settings_source
    assert "Provider details can be prepared here; calls still require the shared outbound toggle." in llm_settings_source
    assert "render_llm_settings(" in pages_text
    assert "show_status_card=False" in pages_text
    assert "controls_only=True" in pages_text
    assert "show_enable_toggle=False" in pages_text
    assert "open_sidebar_on_enable=False" in pages_text
    assert "AI / API connection" in pages_text
    assert "These controls are the real shared settings used by the assistant and Research Agent." in pages_text
    assert "Hosted relay is reserved for assistant/internal use" in pages_text
    assert "Shared external ready" in pages_text
    assert "ui_density" in pages_text
    assert "reduce_motion" in pages_text
    assert "eu-display-preferences" in app_text
    assert "data-reduce-motion" in app_text
    assert 'state["_llm_provider_sel"] = "easyicu_hosted"' in pages_text
    assert 'state["_llm_provider_sel"] = external_provider' in settings_external_model_source
    assert 'state["llm_base_url"] = external_base_url' in settings_external_model_source
    assert 'state["llm_model"] = external_model' in settings_external_model_source
    assert 'state["_llm_api_key_inp"] = ""' in settings_external_model_source
    assert 'state["_eu_settings_allow_outbound_model_calls"]' not in settings_external_model_source
    assert "Hosted (assistant)" in pages_text
    assert "gpt-oss · local" not in pages_text
    assert "EasyICU 不会写入本地文件" in sidebar_text
    assert "font-size:12px" in footer_source
    assert "font-size:10.5px" not in footer_source
    assert "controls_only: bool = False" in llm_text
    assert "show_enable_toggle: bool = True" in llm_text
    assert "open_sidebar_on_enable: bool = True" in llm_text
    assert "eu-llm-settings-status" in llm_text
    assert "if enabled and open_sidebar_on_enable:" in llm_text
    assert "Provider configured. Turn on outbound model calls above" in llm_text
    assert "API keys stay in this browser session only" in llm_text
    agent_text = Path(research_agent.__file__).read_text(encoding="utf-8")
    assert "elif sidebar_hosted_blocked:" in agent_text
    assert "default_index = options.index(mock_choice)" in agent_text
    assert "defaulting from\n    # Settings/Hosted straight into a Custom external endpoint" in agent_text
    assert ".eu-settings-page-head" in css_text
    assert ".eu-settings-card" in css_text
    assert ".eu-settings-toggle" in css_text
    assert "st-key-eu_settings_privacy_card" in css_text
    assert "st-key-_eu_settings_allow_outbound_model_calls" in css_text
    assert ".eu-llm-settings-status" in css_text
    assert ".eu-settings-status-grid" in css_text
    assert ".eu-settings-route-note" in css_text


def test_step1_source_mode_tabs_only_show_actionable_modes() -> None:
    sidebar_text = Path(sidebar.__file__).read_text(encoding="utf-8")
    mode_tabs_source = sidebar_text[
        sidebar_text.index("def _render_source_mode_tabs"):
        sidebar_text.index("def _render_data_source_page_header")
    ]

    assert '("demo", "Demo", "模拟数据", ":material/science:")' in mode_tabs_source
    assert '("real", "Real Data", "真实数据", ":material/database:")' in mode_tabs_source
    assert "st.columns(2, gap=\"small\")" in mode_tabs_source
    assert "_eu_source_mode_none" not in mode_tabs_source
    assert "No Data" not in mode_tabs_source
    assert "仅生成代码" not in mode_tabs_source
    assert "disabled=target == \"none\"" not in mode_tabs_source
    assert "Code-only mode is surfaced through Research Agent" not in mode_tabs_source


def test_extract_footer_action_buttons_keep_confirm_label_on_one_line() -> None:
    sidebar_text = Path(sidebar.__file__).read_text(encoding="utf-8")
    app_text = Path(app.__file__).read_text(encoding="utf-8")
    css_text = shell_styles._load_shell_overrides_css()

    assert '<div class="banner-icon" aria-hidden="true">' in sidebar_text
    assert '<svg viewBox="0 0 24 24"' in sidebar_text
    assert '<div class="banner-icon">⚗</div>' not in sidebar_text
    assert '"🤖 Ask AI about Sepsis settings"' not in app_text
    assert 'icon=":material/smart_toy:"' in app_text
    assert sidebar_text.count("st.columns([5, 1.45, 2.25], gap=\"small\")") == 2
    assert "footer_l, prev_col, reset_col, preset_col, restore_col, confirm_col = st.columns(" in sidebar_text
    assert "[1.65, 1.25, 1.1, 1.25, 1.25, 1.45]" in sidebar_text
    assert "st.columns([4.2, 1.45, 1.45], gap=\"small\")" in sidebar_text
    assert 'key="step1_reset_real"' in sidebar_text
    assert 'key="step1_confirm_real"' in sidebar_text
    assert "disabled=not real_ready" in sidebar_text
    assert "_confirm_real_data_source()" in sidebar_text
    assert "st.columns([5, 1.7, 1.7], gap=\"small\")" not in sidebar_text
    assert "st.columns([5, 1, 2.4], gap=\"small\")" not in sidebar_text
    assert "st.columns([2.45, 0.72, 1.15, 1.85], gap=\"small\")" not in sidebar_text
    assert "st.columns([4.2, 1.0, 1.65], gap=\"small\")" not in sidebar_text
    assert 'class*="st-key-step1_reset_demo"' in css_text
    assert 'class*="st-key-step1_confirm_demo"' in css_text
    assert 'class*="st-key-step1_reset_real"' in css_text
    assert 'class*="st-key-step1_confirm_real"' in css_text
    assert 'class*="st-key-step2_confirm_design"' in css_text
    assert 'class*="st-key-cohort_builder_restore_preset"' in css_text
    assert 'class*="st-key-cohort_builder_previous_step"' in css_text
    assert 'key="cohort_builder_previous_step"' in sidebar_text
    assert "_sidebar_set_extract_step_state(st.session_state, 1)" in sidebar_text
    assert 'class*="st-key-step3_confirm_design"' in css_text
    assert 'class*="st-key-concept_previous_step"' in css_text
    assert "height: 45px !important" in css_text


def test_step4_export_footer_actions_have_overlap_guard() -> None:
    sidebar_text = Path(sidebar.__file__).read_text(encoding="utf-8")
    i18n_text = Path(i18n.__file__).read_text(encoding="utf-8")
    data_paths_text = Path(data_paths.__file__).read_text(encoding="utf-8")
    css_text = shell_styles._load_shell_overrides_css()

    assert 'key="eu_export_footer_actions"' in sidebar_text
    assert "Package & export" in sidebar_text
    assert "Export contents" in sidebar_text
    assert "Agent code, figures, evidence ledger" in sidebar_text
    assert 'icon=":material/arrow_back:"' in sidebar_text
    assert 'icon=":material/check:"' in sidebar_text
    assert "'sanity_back': 'Previous step'" in i18n_text
    assert "'sanity_back': '上一步'" in i18n_text
    assert "_sidebar_set_extract_step_state(st.session_state, 3)" in sidebar_text
    assert "✅ Confirm & Export" not in i18n_text
    assert "↩️ Go Back & Modify" not in i18n_text
    assert 'browse_label = ""' in data_paths_text
    assert 'icon=":material/folder_open:"' in data_paths_text
    assert 'browse_label = "📂"' not in data_paths_text
    assert "_module_display_name(group, lang)" in sidebar_text
    assert "sofa2 score" not in sidebar._module_display_name("sofa2_score", "en")
    assert 'class*="st-key-eu_export_footer_actions"' in css_text
    assert ".eu-export-contents-card" in css_text
    export_footer_css = css_text[
        css_text.index('.stApp [class*="st-key-eu_export_footer_actions"] {'):
        css_text.index(".eu-performance-strip {")
    ]

    assert "clear: both" in export_footer_css
    assert "margin-top: 30px !important" in export_footer_css
    assert "align-items: center !important" in export_footer_css


def test_export_runtime_states_use_design_system_surfaces() -> None:
    export_text = Path(export_workflow.__file__).read_text(encoding="utf-8")
    sidebar_text = Path(sidebar.__file__).read_text(encoding="utf-8")
    app_text = Path(app.__file__).read_text(encoding="utf-8")
    css_text = shell_styles._load_shell_overrides_css()

    assert "_render_export_progress_shell(" in export_text
    assert "eu-export-progress-shell" in export_text
    assert "Packaging export bundle..." in export_text
    assert "_render_export_conflict_panel(" in export_text
    assert "eu-export-conflict-card" in export_text
    assert "Overwrite all" in export_text
    assert "Skip all" in export_text
    assert "OVERWRITE ALL" not in export_text
    assert "SKIP ALL" not in export_text
    assert export_workflow._concept_group_label("sofa2_score", "en") == "SOFA-2 Scores"
    assert "sofa2_score" not in export_workflow._concept_group_label("sofa2_score", "en")

    assert "eu-export-complete-hero" in sidebar_text
    assert "eu-export-ledger-grid" in sidebar_text
    assert "Export complete" in sidebar_text
    assert "Everything stayed on your machine" in sidebar_text
    assert "Packaging export bundle..." in app_text

    assert ".eu-export-progress-shell" in css_text
    assert ".eu-export-conflict-card" in css_text
    assert ".eu-export-complete-hero" in css_text
    assert ".eu-export-ledger-grid" in css_text
    assert ".eu-post-export-hero" in css_text
    assert 'class*="st-key-file_overwrite_all"' in css_text
    assert 'class*="st-key-post_export_completed_open_"' in css_text
    assert 'class*="st-key-_post_export_open_"' in css_text


def test_export_patient_limit_hint_tracks_selected_limit() -> None:
    assert sidebar._export_patient_limit_label(0, "en") == "All"
    assert sidebar._export_patient_limit_label(1000, "en") == "1k"
    assert sidebar._export_patient_limit_hint(0, "en") == "All patients for final runs"
    assert sidebar._export_patient_limit_hint(100, "en") == "Export first 100 patients"
    assert sidebar._export_patient_limit_hint(5000, "en") == "Export first 5k patients"


def test_export_conflicts_are_scoped_to_selected_format() -> None:
    assert export_workflow._export_extension_for_format("CSV") == ".csv"
    assert export_workflow._export_extension_for_format("Excel") == ".xlsx"
    assert export_workflow._export_extension_for_format("Parquet") == ".parquet"

    export_text = Path(export_workflow.__file__).read_text(encoding="utf-8")
    conflict_block = export_text[
        export_text.index("existing_modules = {}"):
        export_text.index("# 如果有已存在的模块，显示让用户选择")
    ]
    overwrite_block = export_text[
        export_text.index("if group_name in _ow_modules or is_viz_import_mode:"):
        export_text.index("# 跳过检查")
    ]

    assert "target_ext = _export_extension_for_format(export_format)" in conflict_block
    assert "glob(f\"{search_prefix}*{target_ext}\")" in conflict_block
    assert "for ext in ['.parquet', '.csv', '.xlsx']" not in conflict_block
    assert "glob(f\"{group_name}_*{target_ext}\")" in overwrite_block


def test_step2_reset_defers_cohort_enabled_until_next_render(monkeypatch) -> None:
    class _RerunRequested(Exception):
        pass

    class _GuardedSessionState(_AttrSessionState):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            object.__setattr__(self, "_locked_widget_keys", set())

        def lock_widget_key(self, key: str) -> None:
            self._locked_widget_keys.add(key)

        def clear_widget_locks(self) -> None:
            self._locked_widget_keys.clear()

        def __setitem__(self, key, value) -> None:
            if key in self._locked_widget_keys:
                raise AssertionError(f"{key} was modified after widget instantiation")
            super().__setitem__(key, value)

    class _Step2ResetStreamlit:
        def __init__(self) -> None:
            self.session_state = _GuardedSessionState(
                {
                    "language": "en",
                    "entry_mode": "real",
                    "use_mock_data": False,
                    "database": "miiv",
                    "cohort_enabled": True,
                    "cohort_age_min_design": 65,
                    "cohort_icd_include_query_design": "A41",
                    "cohort_filter": {
                        "age_min": 65,
                        "age_max": None,
                        "first_icu_stay": None,
                        "los_min": None,
                        "los_max": None,
                        "gender": None,
                        "survived": None,
                        "has_sepsis": True,
                        "disease_cohort": "sepsis",
                        "icd_query": "A41",
                        "icd_include_query": "A41",
                        "icd_exclude_query": "",
                        "icd_mode": "include",
                    },
                }
            )

        def columns(self, spec, **_kwargs):
            count = spec if isinstance(spec, int) else len(spec)
            return [_FakeColumn() for _ in range(count)]

        def container(self, **_kwargs):
            return _FakeColumn()

        def markdown(self, *_args, **_kwargs) -> None:
            pass

        def warning(self, *_args, **_kwargs) -> None:
            pass

        def caption(self, *_args, **_kwargs) -> None:
            pass

        def toggle(self, *_args, key, **_kwargs) -> bool:
            self.session_state.lock_widget_key(key)
            return bool(self.session_state.get(key, False))

        def number_input(self, *_args, **kwargs) -> int:
            return kwargs["value"]

        def selectbox(self, *_args, options, index=0, **_kwargs):
            return options[index]

        def text_input(self, *_args, **kwargs) -> str:
            return str(kwargs.get("value", ""))

        def button(self, *_args, **kwargs) -> bool:
            return kwargs.get("key") == "cohort_builder_reset"

        def rerun(self) -> None:
            raise _RerunRequested

    streamlit_stub = _Step2ResetStreamlit()
    monkeypatch.setattr(sidebar, "st", streamlit_stub)
    monkeypatch.setattr(sidebar, "_real_data_source_ready", lambda: True)
    monkeypatch.setattr(sidebar, "_get_supported_disease_cohorts", lambda _db: ["none", "sepsis"], raising=False)
    monkeypatch.setattr(sidebar, "_supports_icd_filter", lambda _db: True, raising=False)
    monkeypatch.setattr(sidebar, "DISEASE_COHORT_CONFIG", {"sepsis": {"label_en": "Sepsis-3"}}, raising=False)
    monkeypatch.setattr(
        sidebar,
        "_clear_icd_preview_state",
        lambda: streamlit_stub.session_state.__setitem__("_icd_preview_cleared", True),
        raising=False,
    )

    with pytest.raises(_RerunRequested):
        sidebar._render_step2_cohort_builder_design()

    assert streamlit_stub.session_state[sidebar._STEP2_RESET_PENDING_KEY] is True
    assert streamlit_stub.session_state["cohort_enabled"] is True

    streamlit_stub.session_state.clear_widget_locks()
    sidebar._ensure_step2_state_defaults()

    assert sidebar._STEP2_RESET_PENDING_KEY not in streamlit_stub.session_state
    assert streamlit_stub.session_state["cohort_enabled"] is False
    assert "cohort_age_min_design" not in streamlit_stub.session_state
    assert "cohort_icd_include_query_design" not in streamlit_stub.session_state
    assert streamlit_stub.session_state["cohort_filter"]["age_min"] is None
    assert streamlit_stub.session_state["cohort_filter"]["disease_cohort"] == "none"
    assert streamlit_stub.session_state["_icd_preview_cleared"] is True


def test_real_step2_preview_does_not_claim_demo_results(monkeypatch) -> None:
    rendered: list[str] = []

    class _PreviewStreamlit:
        session_state = _AttrSessionState(
            {
                "language": "en",
                "entry_mode": "real",
                "database": "mimic",
                "cohort_enabled": True,
                "cohort_filter": {
                    "age_min": 65,
                    "age_max": None,
                    "first_icu_stay": None,
                    "los_min": None,
                    "gender": None,
                    "survived": None,
                    "disease_cohort": "none",
                    "icd_include_query": "",
                    "icd_exclude_query": "",
                },
            }
        )

        @staticmethod
        def markdown(value: str, **_kwargs) -> None:
            rendered.append(value)

    monkeypatch.setattr(sidebar, "st", _PreviewStreamlit())

    sidebar._render_cohort_live_preview("en")
    preview_html = "\n".join(rendered)

    assert "Local source · MIMIC-III" in preview_html
    assert "Preview available after extraction" in preview_html
    assert "pending extraction" in preview_html
    assert "Clear all" not in preview_html
    assert "Sample of demo cohort" not in preview_html
    assert "seed=42" not in preview_html
    assert "18.0%" not in preview_html


def test_demo_step2_preview_does_not_render_negative_zero_drop(monkeypatch) -> None:
    rendered: list[str] = []

    class _PreviewStreamlit:
        session_state = _AttrSessionState(
            {
                "language": "en",
                "entry_mode": "demo",
                "mock_params": {"n_patients": 10},
                "cohort_enabled": True,
                "cohort_filter": {
                    "age_min": None,
                    "age_max": None,
                    "first_icu_stay": None,
                    "los_min": None,
                    "gender": None,
                    "survived": None,
                    "disease_cohort": "none",
                    "icd_include_query": "",
                },
            }
        )

        @staticmethod
        def markdown(value: str, **_kwargs) -> None:
            rendered.append(value)

    monkeypatch.setattr(sidebar, "st", _PreviewStreamlit())

    sidebar._render_cohort_live_preview("en")
    preview_html = "\n".join(rendered)

    assert "of 10 stays · 0.0%" in preview_html
    assert "Clear all" not in preview_html
    assert "-0.0%" not in preview_html


def test_demo_step2_live_preview_applies_icd_exclude_estimate(monkeypatch) -> None:
    rendered: list[str] = []

    class _PreviewStreamlit:
        session_state = _AttrSessionState(
            {
                "language": "en",
                "entry_mode": "demo",
                "mock_params": {"n_patients": 10},
                "cohort_enabled": True,
                "cohort_filter": {
                    "age_min": None,
                    "age_max": None,
                    "first_icu_stay": None,
                    "los_min": None,
                    "gender": None,
                    "survived": None,
                    "disease_cohort": "none",
                    "icd_include_query": "A41",
                    "icd_exclude_query": "I50,C34",
                },
            }
        )

        @staticmethod
        def markdown(value: str, **_kwargs) -> None:
            rendered.append(value)

    monkeypatch.setattr(sidebar, "st", _PreviewStreamlit())

    sidebar._render_cohort_live_preview("en")
    preview_html = "\n".join(rendered)

    assert "of 10 stays · -90.0%" in preview_html
    assert "after filters: 1" in preview_html
    assert "ICD + A41" in preview_html
    assert "ICD - I50,C34" in preview_html


def test_step2_database_display_names_cover_real_sources() -> None:
    assert sidebar._step2_database_display_name("miiv") == "MIMIC-IV"
    assert sidebar._step2_database_display_name("mimic") == "MIMIC-III"
    assert sidebar._step2_database_display_name("eicu") == "eICU-CRD"


def test_demo_step2_icd_preview_summarizes_include_exclude_and_net_counts() -> None:
    preview = sidebar._demo_step2_icd_preview_counts(10, "A41", "I50,C34")

    assert preview["mode"] == "demo"
    assert preview["total"] == 10
    assert preview["include_tokens"] == ["A41"]
    assert preview["exclude_tokens"] == ["I50", "C34"]
    assert preview["include_count"] == 4
    assert preview["exclude_count"] == 3
    assert preview["retained_count"] == 1

    preview_html = sidebar._step2_icd_preview_html("zh", preview)
    assert "ICD 条件预览" in preview_html
    assert "演示估算" in preview_html
    assert "包含匹配" in preview_html
    assert "排除匹配" in preview_html
    assert "ICD 净保留" in preview_html


def test_real_step2_icd_preview_uses_local_match_id_sets(tmp_path, monkeypatch) -> None:
    class _PreviewStreamlit:
        session_state = _AttrSessionState(
            {
                "data_path": str(tmp_path),
                "database": "miiv",
            }
        )

    calls: list[tuple[str, tuple[str, ...]]] = []

    def _fake_preview(_path: Path, database: str, tokens: list[str]) -> dict:
        calls.append((database, tuple(tokens)))
        if tokens == ["A41"]:
            return {
                "tokens": tokens,
                "matched_patients": 3,
                "matched_ids": [101, 102, 103],
                "total_patients": 10,
                "top_codes": None,
                "error": None,
            }
        if tokens == ["I50"]:
            return {
                "tokens": tokens,
                "matched_patients": 2,
                "matched_ids": [102, 200],
                "total_patients": 10,
                "top_codes": None,
                "error": None,
            }
        raise AssertionError(tokens)

    monkeypatch.setattr(sidebar, "st", _PreviewStreamlit())
    monkeypatch.setattr(sidebar, "_preview_icd_match", _fake_preview, raising=False)

    preview = sidebar._real_step2_icd_preview_counts("A41", "I50")

    assert preview["mode"] == "real"
    assert preview["total"] == 10
    assert preview["include_count"] == 3
    assert preview["exclude_count"] == 2
    assert preview["retained_count"] == 2
    assert calls == [("miiv", ("A41",)), ("miiv", ("I50",))]

    sidebar._real_step2_icd_preview_counts("A41", "I50")
    assert calls == [("miiv", ("A41",)), ("miiv", ("I50",))]


def test_step2_icd_preview_css_is_present() -> None:
    css_text = shell_styles._load_shell_overrides_css()

    assert ".eu-icd-preview" in css_text
    assert ".eu-icd-preview-grid" in css_text
    assert "exact from local tables" not in css_text


def test_dataframe_toolbar_is_kept_inside_clickable_table_surface() -> None:
    css_text = shell_styles._load_shell_overrides_css()

    toolbar_rule = css_text[
        css_text.index('.stApp [data-testid="stDataFrame"] [data-testid="stElementToolbar"]'):
        css_text.index("/* Legacy hero / header blocks")
    ]
    assert "top: 6px !important" in toolbar_rule
    assert "right: 8px !important" in toolbar_rule
    assert "z-index: 120 !important" in toolbar_rule
    assert 'button[aria-label="Download as CSV"]' in toolbar_rule
    assert ':has([data-testid="search-input"]) [data-testid="stElementToolbar"]' in toolbar_rule
    assert "display: none !important" in toolbar_rule


def test_large_desktop_density_keeps_sidebar_readable() -> None:
    css_text = shell_styles._load_shell_overrides_css()

    assert "@media (min-width: 1500px)" in css_text
    large_density_css = css_text[
        css_text.index("/* Large desktop comfort density."):
        css_text.index("/* Tablet (901-1100px)")
    ]
    assert "--eu-sidebar-w: 280px" in large_density_css
    assert "max-width: 1840px !important" in large_density_css
    assert "font-size: 15px;" in large_density_css
    assert "font-size: 14.5px !important" in large_density_css
    assert "font-size: 13.8px !important" in large_density_css
    assert "font-size: 12.8px" in large_density_css
    assert "width: 18px !important" in large_density_css
    assert "padding-left: 44px !important" in large_density_css
    assert "padding-right: 44px !important" in large_density_css
    assert "vw" not in large_density_css


def test_mobile_bottom_nav_labels_clip_with_ellipsis() -> None:
    css_text = shell_styles._load_shell_overrides_css()

    assert 'st-key-main_nav_bar"] div[role="radiogroup"] > label p' in css_text
    assert 'label:has(input:checked) *' in css_text
    assert "-webkit-text-fill-color: var(--ink) !important" in css_text
    assert "flex: 1 1 0% !important" in css_text
    assert "min-width: 0 !important" in css_text
    assert "width: 100% !important" in css_text
    assert "width: 72px !important" in css_text
    assert "max-width: 72px !important" in css_text
    assert "text-overflow: ellipsis !important" in css_text
    assert "overflow: hidden !important" in css_text


def test_narrow_view_notice_is_scoped_to_dense_visualization_pages(monkeypatch) -> None:
    global_css = Path(styles.__file__).read_text(encoding="utf-8")

    assert ".eu-narrow-view-note" in global_css
    assert "body::before" not in global_css
    assert "interrupting every workflow" in global_css

    class _FakeStreamlit:
        def __init__(self) -> None:
            self.markdown_calls: list[str] = []

        def markdown(self, body, *_args, **_kwargs) -> None:
            self.markdown_calls.append(str(body))

    fake_st = _FakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)

    app._render_narrow_view_notice("research_agent", "en")
    app._render_narrow_view_notice("extract", "en")
    assert fake_st.markdown_calls == []

    app._render_narrow_view_notice("quick_viz", "en")
    app._render_narrow_view_notice("cohort", "zh")

    assert len(fake_st.markdown_calls) == 2
    assert "eu-narrow-view-note" in fake_st.markdown_calls[0]
    assert "dense chart comparison" in fake_st.markdown_calls[0]
    assert "密集图表对比" in fake_st.markdown_calls[1]
