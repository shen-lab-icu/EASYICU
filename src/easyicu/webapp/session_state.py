"""Shared Streamlit session-state cleanup helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, MutableMapping
from typing import Iterable, Literal

import streamlit as st


Scope = Literal["all", "agent", "cohort"]


STATE_KEY_OWNERS = {
    "entry_mode": "entry_page",
    "data_path": "sidebar.data_source",
    "database": "sidebar.data_source",
    "loaded_concepts": "data_workflows",
    "loaded_data_origin": "data_workflows",
    "patient_ids": "data_workflows",
    "all_patient_count": "data_workflows",
    "selected_patient": "patient_page",
    "use_mock_data": "entry_page",
    "id_col": "data_workflows",
    "selected_concepts": "sidebar.feature_selection",
    "export_completed": "export_workflow",
    "trigger_export": "sidebar.export",
    "export_format": "sidebar.export",
    "export_path": "sidebar.export",
    "language": "entry_page.sidebar",
    "sidebar_expanded": "sidebar.layout",
    "screenshot_mode": "bootstrap.screenshot",
    "_scroll_to_tab": "navigation",
    "_figure_target_section": "figure_capture",
    "_figure_target_panel": "figure_capture",
}


@dataclass
class AppState:
    """Typed access shim for the shared Streamlit state mapping.

    This is a migration bridge: new pages should depend on this object instead
    of scattering raw ``st.session_state`` reads and writes.
    """

    raw: MutableMapping[str, Any]

    @property
    def language(self) -> str:
        return str(self.raw.get("language", "en"))

    @property
    def entry_mode(self) -> str:
        return str(self.raw.get("entry_mode", "none"))

    @property
    def loaded_concepts(self) -> dict[str, Any]:
        value = self.raw.get("loaded_concepts", {})
        return value if isinstance(value, dict) else {}

    @property
    def patient_ids(self) -> list[Any]:
        value = self.raw.get("patient_ids", [])
        return value if isinstance(value, list) else []

    @property
    def id_col(self) -> str:
        return str(self.raw.get("id_col", "stay_id"))

    @property
    def screenshot_mode(self) -> bool:
        return bool(self.raw.get("screenshot_mode", False))

    def clear_navigation_request(self) -> tuple[str | None, bool]:
        """Pop one-shot navigation requests after a render pass."""
        return self.raw.pop("_scroll_to_tab", None), bool(self.raw.pop("_scroll_to_top", None))


def get_state() -> AppState:
    """Return typed access to the current Streamlit session state."""
    return AppState(st.session_state)


COHORT_STATE_KEYS = {
    "group_a_data",
    "group_b_data",
    "multidb_data",
    "dash_demographics",
    "multidb_is_demo",
    "dash_is_demo",
    "cohort_is_demo",
}


AGENT_STATE_KEYS = {
    "research_agent_last_result",
    "research_agent_resume_run_id",
    "research_agent_force_manuscript",
    "research_agent_cohort_source",
    "research_agent_module_built",
    "research_agent_inbound_cohort",
    "research_agent_inbound_cohort_label",
    "research_agent_inbound_signature",
    "research_agent_progress_events",
}


AGENT_CONTINUATION_STATE_KEYS = {
    "research_agent_resume_run_id",
    "research_agent_force_manuscript",
    "research_agent_resume_mode",
    "research_agent_resume_notes",
    "research_agent_resume_relax_probe",
    "research_agent_preflight_confirmed",
    "research_agent_preflight_signature",
}


RUN_STATE_KEYS = {
    "loaded_concepts",
    "loaded_data_origin",
    "patient_ids",
    "use_mock_data",
    "trigger_export",
    "export_completed",
    "_exporting_in_progress",
    "_preview_requested",
    "_viz_import_export_auto_trigger",
    "_viz_notices",
    "_scroll_to_tab",
}


def clear_agent_continuation_state(state: MutableMapping[str, Any]) -> None:
    """Clear resume/draft markers before opening a fresh Agent setup."""
    for key in AGENT_CONTINUATION_STATE_KEYS:
        state.pop(key, None)


def _drop_many(keys: Iterable[str]) -> None:
    for key in keys:
        st.session_state.pop(key, None)


def init_session_state() -> None:
    """Initialize Streamlit session-state defaults used across the app."""
    if "entry_mode" not in st.session_state:
        st.session_state.entry_mode = "none"
    if "data_path" not in st.session_state:
        st.session_state.data_path = None
    if "database" not in st.session_state:
        st.session_state.database = "miiv"
    if "loaded_concepts" not in st.session_state:
        st.session_state.loaded_concepts = {}
    if "loaded_data_origin" not in st.session_state:
        st.session_state.loaded_data_origin = "none"
    if "patient_ids" not in st.session_state:
        st.session_state.patient_ids = []
    if "all_patient_count" not in st.session_state:
        st.session_state.all_patient_count = 0
    if "selected_patient" not in st.session_state:
        st.session_state.selected_patient = None
    if "use_mock_data" not in st.session_state:
        st.session_state.use_mock_data = False
    if "id_col" not in st.session_state:
        st.session_state.id_col = "stay_id"
    if "selected_concepts" not in st.session_state:
        st.session_state.selected_concepts = []
    if "export_completed" not in st.session_state:
        st.session_state.export_completed = False
    if "mock_params" not in st.session_state:
        st.session_state.mock_params = {"n_patients": 100, "hours": 72}
    if "trigger_export" not in st.session_state:
        st.session_state.trigger_export = False
    if "export_format" not in st.session_state:
        st.session_state.export_format = "Parquet"
    if "export_path" not in st.session_state:
        st.session_state.export_path = os.path.expanduser("~/easyicu_export")
    if "path_validated" not in st.session_state:
        st.session_state.path_validated = False
    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "entry_lang_select" not in st.session_state:
        st.session_state.entry_lang_select = "EN" if st.session_state.language == "en" else "ZH"
    if "patient_limit" not in st.session_state:
        st.session_state.patient_limit = 0
    if "available_patient_ids" not in st.session_state:
        st.session_state.available_patient_ids = None
    if "step1_confirmed" not in st.session_state:
        st.session_state.step1_confirmed = False
    if "step2_confirmed" not in st.session_state:
        st.session_state.step2_confirmed = False
    if "sidebar_expanded" not in st.session_state:
        st.session_state.sidebar_expanded = False
    if "sidebar_preview_enabled" not in st.session_state:
        st.session_state.sidebar_preview_enabled = False


def clear_run_state(scope: Scope = "all") -> None:
    """Clear app run state while preserving language and mode selection.

    ``scope="cohort"`` only removes cohort-comparison caches.
    ``scope="agent"`` only removes research-agent run/cached cohort state.
    ``scope="all"`` removes loaded data, cohort caches, and agent caches.
    """
    if scope == "cohort":
        _drop_many(COHORT_STATE_KEYS)
        return
    if scope == "agent":
        _drop_many(AGENT_STATE_KEYS)
        return
    if scope != "all":
        raise ValueError(f"Unknown session-state cleanup scope: {scope}")

    _drop_many(RUN_STATE_KEYS)
    _drop_many(COHORT_STATE_KEYS)
    _drop_many(AGENT_STATE_KEYS)
    st.session_state.loaded_concepts = {}
    st.session_state.loaded_data_origin = "none"
    st.session_state.patient_ids = []
