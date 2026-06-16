from __future__ import annotations

import base64
import inspect
import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from easyicu.webapp import agent_workbench as wb_page
from easyicu.webapp import research_agent as ra_page
from easyicu.webapp.agent_workbench import (
    _demo_state,
    _result_cards_from_evidence,
    _resolve_workbench_state,
    _step_button_label,
    _step_flow_html,
    _step_legend_html,
    build_workbench_state_from_manifest,
)
from easyicu.webapp.workspace_snapshots import build_agent_project_snapshot


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}


def test_module_export_folder_builds_filtered_stay_level_cohort(tmp_path: Path) -> None:
    folder = tmp_path / "mimiciv_export"
    (folder / "sepsis3_sofa2").mkdir(parents=True)
    (folder / "outcome").mkdir()
    (folder / "vitals").mkdir()

    pd.DataFrame({
        "stay_id": [1, 2, 3],
        "sep3_sofa2": [1, 0, 1],
    }).to_parquet(folder / "sepsis3_sofa2" / "sep3_sofa2.parquet", index=False)
    pd.DataFrame({
        "stay_id": [1, 2, 3],
        "death": [1, 0, 0],
    }).to_parquet(folder / "outcome" / "death.parquet", index=False)
    pd.DataFrame({
        "stay_id": [1, 1, 2, 3],
        "charttime": pd.to_datetime([
            "2024-01-01 00:00",
            "2024-01-01 01:00",
            "2024-01-01 00:00",
            "2024-01-01 00:00",
        ]),
        "hr": [70, 80, 90, 100],
    }).to_parquet(folder / "vitals" / "hr.parquet", index=False)

    selected = [
        folder / "sepsis3_sofa2" / "sep3_sofa2.parquet",
        folder / "outcome" / "death.parquet",
        folder / "vitals" / "hr.parquet",
    ]
    cohort = ra_page._build_stay_level_from_module_folder(
        folder=folder,
        selected_files=selected,
        id_col="stay_id",
        filter_spec=(folder / "sepsis3_sofa2" / "sep3_sofa2.parquet", "sep3_sofa2", "nonzero / true", ""),
    )

    assert set(cohort["stay_id"]) == {1, 3}
    assert cohort.loc[cohort["stay_id"] == 1, "hr"].iloc[0] == 80
    assert set(["sep3_sofa2", "death", "hr"]) <= set(cohort.columns)


def test_module_export_folder_builds_from_csv_exports(tmp_path: Path) -> None:
    folder = tmp_path / "csv_export"
    folder.mkdir()
    sepsis = folder / "sepsis3_sofa2_sep3_sofa2.csv"
    outcome = folder / "outcome_death.csv"
    vitals = folder / "vitals_hr.csv"
    pd.DataFrame({
        "stay_id": [1, 2, 3],
        "sep3_sofa2": [1, 0, 1],
    }).to_csv(sepsis, index=False)
    pd.DataFrame({
        "stay_id": [1, 2, 3],
        "death": [1, 0, 0],
    }).to_csv(outcome, index=False)
    pd.DataFrame({
        "stay_id": [1, 1, 2, 3],
        "charttime": [
            "2024-01-01 00:00",
            "2024-01-01 01:00",
            "2024-01-01 00:00",
            "2024-01-01 00:00",
        ],
        "hr": [70, 80, 90, 100],
    }).to_csv(vitals, index=False)

    assert ra_page._list_module_parquets(folder) == [
        outcome.resolve(),
        sepsis.resolve(),
        vitals.resolve(),
    ]

    cohort = ra_page._build_stay_level_from_module_folder(
        folder=folder,
        selected_files=[sepsis, outcome, vitals],
        id_col="stay_id",
        filter_spec=(sepsis, "sep3_sofa2", "nonzero / true", ""),
    )

    assert set(cohort["stay_id"]) == {1, 3}
    assert cohort.loc[cohort["stay_id"] == 1, "hr"].iloc[0] == 80
    assert {"sep3_sofa2", "death", "hr"} <= set(cohort.columns)


def test_module_export_folder_normalizes_numeric_id_dtype_before_merge(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "mock_export"
    folder.mkdir()
    left = folder / "vitals.parquet"
    right = folder / "labs.parquet"
    pd.DataFrame({
        "stay_id": pd.Series([1, 2], dtype="int64"),
        "charttime": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "hr": [70, 80],
    }).to_parquet(left, index=False)
    pd.DataFrame({
        "stay_id": pd.Series([1.0, 2.0], dtype="float64"),
        "charttime": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "lactate": [1.2, 2.4],
    }).to_parquet(right, index=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cohort = ra_page._build_stay_level_from_module_folder(
            folder=folder,
            selected_files=[left, right],
            id_col="stay_id",
        )

    merge_warnings = [
        warning
        for warning in caught
        if "merging on int and float columns" in str(warning.message)
    ]
    assert not merge_warnings
    assert list(cohort["stay_id"]) == [1, 2]
    assert {"hr", "lactate"} <= set(cohort.columns)


def test_agent_demo_to_real_mode_clears_mock_database_context() -> None:
    state: dict[str, object] = {
        "entry_mode": "demo",
        "use_mock_data": True,
        "database": "mock",
        "path_validated": True,
        "last_validated_path": "/tmp/mock",
        "step1_confirmed": True,
        "step2_confirmed": True,
        "step3_confirmed": True,
        "export_completed": True,
        "trigger_export": True,
        "_exporting_in_progress": True,
        "loaded_concepts": {"hr": object()},
        "loaded_data_origin": "demo_viz",
        "patient_ids": [20001],
        "all_patient_count": 1,
        "selected_patient": 20001,
        "selected_concepts": ["hr"],
        "_active_main_page": "quick_viz",
        "_ra_view": "summary",
    }

    ra_page._activate_real_data_mode_from_agent(state)

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
    assert state["_active_main_page"] == "research_agent"
    assert state["_ra_view"] == "setup"


def test_raw_extract_handoff_sets_extraction_steps_and_clears_conflict_state(tmp_path: Path) -> None:
    state: dict[str, object] = {
        "entry_mode": "real",
        "use_mock_data": False,
        "database": "miiv",
        "data_path": "/old/raw",
        "path_validated": False,
        "loaded_concepts": {"hr": object()},
        "loaded_data_origin": "exported_files",
        "patient_ids": [1, 2],
        "all_patient_count": 2,
        "selected_patient": 1,
        "_skipped_modules": {"vitals"},
        "_overwrite_modules": {"outcome"},
        "_existing_modules_list": ["vitals", "outcome"],
        "_export_conflict_pending": True,
        "_export_cancel_notice": "Export stopped by user.",
        "_post_export_navigation_pending": True,
        "_post_export_target_panel": "Data Tables",
        "_post_export_guidance_dismissed": True,
        "_export_success_result": {"files": []},
        "_export_failure_result": {"type": "no_data"},
    }

    ra_page._queue_raw_extract_handoff(
        state,
        database="mock",
        data_path="",
        output_dir=str(tmp_path / "easyicu_export"),
        concepts=["hr", "death"],
        modules=["vitals", "outcome"],
        patient_limit=0,
    )

    assert state["entry_mode"] == "real"
    assert state["database"] == "mock"
    assert state["use_mock_data"] is True
    assert state["data_path"] == ""
    assert state["path_validated"] is True
    assert state["step1_confirmed"] is True
    assert state["step2_confirmed"] is True
    assert state["step3_confirmed"] is True
    assert state["selected_concepts"] == ["hr", "death"]
    assert state["selected_groups"] == ["vitals", "outcome"]
    assert state["export_format"] == "Parquet"
    assert state["patient_limit"] == 0
    assert state["loaded_concepts"] == {}
    assert state["loaded_data_origin"] == "none"
    assert state["patient_ids"] == []
    assert state["all_patient_count"] == 0
    assert state["selected_patient"] is None
    for key in (
        "_skipped_modules",
        "_overwrite_modules",
        "_existing_modules_list",
        "_export_conflict_pending",
        "_export_cancel_notice",
        "_post_export_navigation_pending",
        "_post_export_target_panel",
        "_post_export_guidance_dismissed",
        "_export_success_result",
        "_export_failure_result",
    ):
        assert key not in state
    assert state["trigger_export"] is False
    assert state["_exporting_in_progress"] is False
    assert state["_active_main_page"] == "extract"
    assert "_main_nav_widget" not in state
    assert "_scroll_to_tab" not in state
    assert state["_scroll_to_top"] is True


def test_module_file_multiselect_defaults_reset_when_folder_changes(tmp_path: Path) -> None:
    folder_a = tmp_path / "miiv_20260427"
    folder_b = tmp_path / "eicu_20260428"
    labels = [
        "demographics_adm_age.parquet",
        "vitals_hr.parquet",
        "outcome_death.parquet",
    ]
    state = {
        "research_agent_module_files": [labels[0]],
    }

    ra_page._sync_module_file_multiselect_defaults(
        state,
        key="research_agent_module_files",
        signature_key="research_agent_module_files_folder",
        folder=folder_a,
        labels=labels,
    )

    assert state["research_agent_module_files"] == labels
    assert state["research_agent_module_files_folder"] == str(folder_a)

    state["research_agent_module_files"] = [labels[0]]
    ra_page._sync_module_file_multiselect_defaults(
        state,
        key="research_agent_module_files",
        signature_key="research_agent_module_files_folder",
        folder=folder_a,
        labels=labels,
    )

    assert state["research_agent_module_files"] == [labels[0]]

    ra_page._sync_module_file_multiselect_defaults(
        state,
        key="research_agent_module_files",
        signature_key="research_agent_module_files_folder",
        folder=folder_b,
        labels=labels,
    )

    assert state["research_agent_module_files"] == labels
    assert state["research_agent_module_files_folder"] == str(folder_b)


def test_module_file_selection_recovers_once_after_build_rerun(tmp_path: Path) -> None:
    folder = tmp_path / "miiv_20260427"
    folder.mkdir()
    labels = [
        "demographics_adm_age.parquet",
        "outcome_death.parquet",
        "vitals_hr.parquet",
    ]
    built_labels = [labels[0], labels[2]]
    state = {
        "research_agent_module_files": [],
        "research_agent_module_files_folder": str(folder),
        "research_agent_module_built": {
            "signature": {
                "folder": str(folder),
                "files": [str(folder / label) for label in built_labels],
                "id_col": "stay_id",
                "filter": None,
                "join_how": "outer",
            },
            "df": pd.DataFrame({"stay_id": [1], "age": [70], "hr": [88]}),
        },
        "_research_agent_module_restore_built_selection": True,
    }

    ra_page._restore_module_file_selection_after_build_rerun(
        state,
        key="research_agent_module_files",
        signature_key="research_agent_module_files_folder",
        folder=folder,
        labels=labels,
    )

    assert state["research_agent_module_files"] == built_labels
    assert state["research_agent_module_files_folder"] == str(folder)
    assert "_research_agent_module_restore_built_selection" not in state

    state["research_agent_module_files"] = []
    ra_page._restore_module_file_selection_after_build_rerun(
        state,
        key="research_agent_module_files",
        signature_key="research_agent_module_files_folder",
        folder=folder,
        labels=labels,
    )

    assert state["research_agent_module_files"] == []


def test_module_file_selection_survives_apply_question_rerun(tmp_path: Path) -> None:
    folder = tmp_path / "miiv_20260427"
    folder.mkdir()
    labels = [
        "demographics_adm_age.parquet",
        "outcome_death.parquet",
        "vitals_hr.parquet",
    ]
    selected = [labels[0], labels[2]]
    state = {
        "research_agent_module_files": selected.copy(),
        "research_agent_module_files_folder": str(folder),
        "research_agent_cohort_source": "Pick an EasyICU module export folder",
    }

    ra_page._preserve_module_file_selection_for_next_rerun(state)

    assert state["_research_agent_module_pending_selection_restore"] == {
        "folder": str(folder),
        "labels": selected,
        "source": "Pick an EasyICU module export folder",
    }

    state["research_agent_cohort_source"] = "I haven't extracted data yet — help me do it"
    ra_page._restore_pending_module_source(
        state,
        options=[
            "I haven't extracted data yet — help me do it",
            "Pick an EasyICU module export folder",
        ],
    )
    assert state["research_agent_cohort_source"] == "Pick an EasyICU module export folder"

    state["research_agent_module_dir_text"] = str(tmp_path / "easyicu_export")
    state["research_agent_module_dir_pick"] = "some detected export"
    ra_page._restore_pending_module_folder_path(
        state,
        manual_path_label="(type a path manually)",
    )

    assert state["research_agent_module_dir_text"] == str(folder)
    assert state["_research_agent_module_dir_restore_folder"] == str(folder)
    assert "research_agent_module_dir_pick" not in state

    state["research_agent_module_files"] = []
    ra_page._restore_pending_module_file_selection(
        state,
        key="research_agent_module_files",
        signature_key="research_agent_module_files_folder",
        folder=folder,
        labels=labels,
    )

    assert state["research_agent_module_files"] == selected
    assert state["research_agent_module_files_folder"] == str(folder)
    assert "_research_agent_module_pending_selection_restore" not in state


def test_question_filter_suggestion_keeps_existing_unfiltered_build() -> None:
    base_signature = {
        "folder": "/tmp/easyicu_export",
        "files": ["/tmp/easyicu_export/sepsis3_sofa2.csv"],
        "id_col": "stay_id",
        "join_how": "outer",
    }
    cached_build = {
        "signature": {
            **base_signature,
            "filter": None,
        },
        "df": pd.DataFrame({"stay_id": [1, 2]}),
    }

    assert not ra_page._module_filter_default_for_question(
        filter_suggested=True,
        state={},
        cached_build=cached_build,
        base_signature=base_signature,
    )
    assert ra_page._module_build_signature_matches_base(
        cached_build,
        base_signature,
        filter_value=None,
    )


def test_question_filter_suggestion_applies_before_first_build() -> None:
    base_signature = {
        "folder": "/tmp/easyicu_export",
        "files": ["/tmp/easyicu_export/sepsis3_sofa2.csv"],
        "id_col": "stay_id",
        "join_how": "outer",
    }

    assert ra_page._module_filter_default_for_question(
        filter_suggested=True,
        state={},
        cached_build=None,
        base_signature=base_signature,
    )
    assert not ra_page._module_filter_default_for_question(
        filter_suggested=True,
        state={"research_agent_module_filter_enabled": False},
        cached_build=None,
        base_signature=base_signature,
    )


def test_quick_cohort_source_shortcut_resets_launch_review_and_focus() -> None:
    state: dict[str, object] = {
        "research_agent_preflight_confirmed": True,
        "research_agent_preflight_ack": True,
        "research_agent_question": "Does SOFA-2 improve mortality prediction?",
    }

    ra_page._activate_research_agent_cohort_source(
        state,
        "Pick an EasyICU module export folder",
        focus_module=True,
    )

    assert state["research_agent_cohort_source"] == "Pick an EasyICU module export folder"
    assert state["research_agent_preflight_confirmed"] is False
    assert state["research_agent_preflight_ack"] is False
    assert state["_eu_ra_focus_module_folder"] is True
    assert "_eu_ra_focus_no_data" not in state
    assert state["research_agent_question"] == "Does SOFA-2 improve mortality prediction?"


def test_cohort_source_quick_actions_render_before_full_radio() -> None:
    section_source = inspect.getsource(ra_page._section_cohort_picker)
    helper_source = inspect.getsource(ra_page._render_cohort_source_quick_actions)

    assert section_source.index("_render_cohort_source_quick_actions(") < section_source.index(
        "st.radio(",
    )
    assert "entry_mode" in section_source
    assert "source_handoff=source_handoff" in section_source
    assert "research_agent_quick_synthetic" in helper_source
    assert "research_agent_quick_module_folder" in helper_source
    assert "research_agent_quick_no_data" in helper_source
    assert "_activate_research_agent_test_cohort(" in helper_source
    assert "_activate_research_agent_cohort_source(" in helper_source
    assert "st.rerun()" in helper_source


def test_quick_test_cohort_prepares_ready_session_handoff() -> None:
    state: dict[str, object] = {
        "research_agent_question": "Does SOFA predict mortality?",
        "research_agent_preflight_confirmed": True,
        "research_agent_preflight_ack": True,
        "research_agent_preflight_signature": "stale",
        "_eu_ra_focus_module_folder": True,
        "_eu_ra_focus_no_data": True,
    }

    ra_page._activate_research_agent_test_cohort(
        state,
        source_handoff="Use cohort prepared elsewhere in this session",
        is_en=True,
    )

    cohort = state["research_agent_inbound_cohort"]
    assert isinstance(cohort, pd.DataFrame)
    assert cohort.shape[0] == 800
    assert {"stay_id", "sofa2", "death"} <= set(cohort.columns)
    assert state["research_agent_inbound_cohort_label"] == "Test SOFA cohort"
    assert state["research_agent_cohort_source"] == "Use cohort prepared elsewhere in this session"
    assert state["_research_agent_previous_cohort_source"] is None
    assert state["research_agent_preflight_confirmed"] is False
    assert state["research_agent_preflight_ack"] is False
    assert "research_agent_preflight_signature" not in state
    assert "_eu_ra_focus_module_folder" not in state
    assert "_eu_ra_focus_no_data" not in state
    assert state["research_agent_question"] == "Does SOFA predict mortality?"


def test_quick_test_cohort_seeds_starter_question_when_empty() -> None:
    state: dict[str, object] = {
        "research_agent_question": "",
        "research_agent_preflight_confirmed": True,
        "research_agent_preflight_ack": True,
        "research_agent_preflight_signature": "stale",
    }

    ra_page._activate_research_agent_test_cohort(
        state,
        source_handoff="Use cohort prepared elsewhere in this session",
        is_en=True,
    )

    assert "SOFA-2" in str(state["research_agent_question"])
    assert "ICU mortality" in str(state["research_agent_question"])
    assert state["_research_agent_default_question_notice"] is True
    assert state["research_agent_preflight_confirmed"] is False
    assert state["research_agent_preflight_ack"] is False
    assert "research_agent_preflight_signature" not in state


def test_quick_test_cohort_ui_shortcut_can_skip_question_widget_mutation() -> None:
    state: dict[str, object] = {
        "research_agent_question": "",
        "research_agent_preflight_confirmed": True,
        "research_agent_preflight_ack": True,
        "research_agent_preflight_signature": "stale",
    }

    ra_page._activate_research_agent_test_cohort(
        state,
        source_handoff="Use cohort prepared elsewhere in this session",
        is_en=True,
        seed_question=False,
    )
    quick_actions_source = inspect.getsource(ra_page._render_cohort_source_quick_actions)

    assert state["research_agent_question"] == ""
    assert "_research_agent_default_question_notice" not in state
    assert isinstance(state["research_agent_inbound_cohort"], pd.DataFrame)
    assert "seed_question=False" in quick_actions_source
    assert state["research_agent_preflight_confirmed"] is False
    assert state["research_agent_preflight_ack"] is False
    assert "research_agent_preflight_signature" not in state


def test_detected_module_folder_defaults_to_most_complete_export(tmp_path: Path) -> None:
    sparse = tmp_path / "miiv_20260427"
    complete = tmp_path / "mock_20260424"
    sparse.mkdir()
    complete.mkdir()

    pd.DataFrame({"stay_id": [1], "age": [70]}).to_parquet(
        sparse / "demographics.parquet",
        index=False,
    )
    for name in ("demographics", "vitals", "outcome"):
        pd.DataFrame({"stay_id": [1], name: [1]}).to_parquet(
            complete / f"{name}.parquet",
            index=False,
        )

    options = ["Manual path", str(sparse), str(complete)]

    assert ra_page._default_module_dir_pick_index(options, [sparse, complete]) == 2


def test_detected_module_folder_avoids_generic_container_root(tmp_path: Path) -> None:
    container = tmp_path / "easyicu_export"
    complete = container / "mock_20260424"
    complete.mkdir(parents=True)

    for idx in range(4):
        pd.DataFrame({"stay_id": [1], f"root_{idx}": [idx]}).to_parquet(
            container / f"root_{idx}.parquet",
            index=False,
        )
    for name in ("demographics", "vitals", "outcome"):
        pd.DataFrame({"stay_id": [1], name: [1]}).to_parquet(
            complete / f"{name}.parquet",
            index=False,
        )

    options = ["Manual path", str(container), str(complete)]

    assert ra_page._default_module_dir_pick_index(options, [container, complete]) == 2


def test_module_folder_manual_handoff_skips_generic_export_path_container(
    tmp_path: Path,
) -> None:
    container = tmp_path / "easyicu_export"
    complete = container / "mock_20260424"
    complete.mkdir(parents=True)
    (container / "root.parquet").write_bytes(b"")
    (complete / "vitals.parquet").write_bytes(b"")

    state = {"export_path": str(container)}

    assert ra_page._module_folder_manual_handoff_dir(state) == ""


def test_module_folder_manual_handoff_preserves_latest_export_root_with_manifest(
    tmp_path: Path,
) -> None:
    container = tmp_path / "easyicu_export"
    complete = container / "mock_20260424"
    complete.mkdir(parents=True)
    current = container / "vitals.parquet"
    current.write_bytes(b"")
    (complete / "demographics.parquet").write_bytes(b"")
    state = {
        "last_export_dir": str(container),
        "export_path": str(container),
        "_export_success_result": {"files": [str(current)]},
    }

    assert ra_page._module_folder_manual_handoff_dir(state) == str(container)


def test_generic_module_folder_manual_default_is_released(tmp_path: Path) -> None:
    container = tmp_path / "easyicu_export"
    complete = container / "mock_20260424"
    complete.mkdir(parents=True)
    (container / "root.parquet").write_bytes(b"")
    (complete / "vitals.parquet").write_bytes(b"")
    state = {
        "export_path": str(container),
        "research_agent_module_dir_text": str(container),
        "research_agent_module_dir_pick": "(type a path manually)",
    }

    ra_page._clear_generic_module_folder_manual_default(state)

    assert "research_agent_module_dir_text" not in state
    assert "research_agent_module_dir_pick" not in state


def test_module_folder_scan_excludes_agent_run_history_container(
    tmp_path: Path,
) -> None:
    research_webapp = tmp_path / "research_output" / "webapp"
    run_dir = research_webapp / "run_20260601T010101_abcd12"
    run_dir.mkdir(parents=True)
    (run_dir / "cohort.parquet").write_bytes(b"")
    (run_dir / "run_status.json").write_text("{}", encoding="utf-8")

    assert ra_page._scan_workspace_for_module_dirs([tmp_path / "research_output"]) == []


def test_module_folder_scan_prefers_export_child_over_agent_history(
    tmp_path: Path,
) -> None:
    export_root = tmp_path / "easyicu_export"
    export_child = export_root / "mock_20260424"
    export_child.mkdir(parents=True)
    for name in ("demographics", "vitals", "outcome"):
        (export_child / f"{name}.parquet").write_bytes(b"")
    research_webapp = tmp_path / "research_output" / "webapp"
    run_dir = research_webapp / "run_20260601T010101_abcd12"
    run_dir.mkdir(parents=True)
    for idx in range(12):
        (run_dir / f"cohort_{idx}.parquet").write_bytes(b"")
    (run_dir / "run_status.json").write_text("{}", encoding="utf-8")

    dirs = ra_page._scan_workspace_for_module_dirs([export_root, tmp_path / "research_output"])

    assert dirs == [export_child.resolve()]


def test_module_parquet_listing_uses_direct_export_files_before_child_runs(tmp_path: Path) -> None:
    export_root = tmp_path / "easyicu_export"
    child_run = export_root / "miiv_20260427"
    child_run.mkdir(parents=True)
    direct = export_root / "outcome.parquet"
    child = child_run / "demographics.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(direct, index=False)
    pd.DataFrame({"stay_id": [1], "age": [70]}).to_parquet(child, index=False)

    assert ra_page._list_module_parquets(export_root) == [direct.resolve()]


def test_latest_export_file_labels_only_include_current_export_folder(tmp_path: Path) -> None:
    export_root = tmp_path / "easyicu_export"
    other_root = tmp_path / "research_output" / "webapp"
    export_root.mkdir(parents=True)
    other_root.mkdir(parents=True)
    current_file = export_root / "vitals_hr.parquet"
    nested_file = export_root / "nested" / "outcome.parquet"
    other_file = other_root / "cohort.parquet"
    nested_file.parent.mkdir()
    current_file.write_bytes(b"")
    nested_file.write_bytes(b"")
    other_file.write_bytes(b"")
    state = {
        "_export_success_result": {
            "files": [
                str(current_file),
                str(nested_file),
                str(other_file),
                str(export_root / "manifest.json"),
            ]
        }
    }

    labels = ra_page._export_result_file_labels_for_folder(state, export_root)

    assert labels == ["vitals_hr.parquet", "nested/outcome.parquet"]


def test_latest_export_file_labels_include_current_csv_export(tmp_path: Path) -> None:
    export_root = tmp_path / "easyicu_export"
    export_root.mkdir()
    current_csv = export_root / "vitals_hr.csv"
    stale_parquet = export_root / "vitals_hr.parquet"
    current_csv.write_text("stay_id,hr\n1,80\n", encoding="utf-8")
    stale_parquet.write_bytes(b"")
    state = {
        "_export_success_result": {
            "files": [
                str(current_csv),
                str(export_root / "easyicu_export_manifest.json"),
            ]
        }
    }

    labels = ra_page._export_result_file_labels_for_folder(state, export_root)

    assert labels == ["vitals_hr.csv"]


def test_module_folder_filter_defaults_are_limited_to_selected_files() -> None:
    source = Path(ra_page.__file__).read_text(encoding="utf-8")

    assert "selected_summaries = [" in source
    assert "_infer_filter_defaults(\n            selected_summaries," in source
    assert "filter_labels = selected_labels" in source
    assert "filter_path = selected_files[filter_labels.index(filter_label)]" in source
    assert "filter_path = module_files[filter_labels.index(filter_label)]" not in source


def test_module_folder_force_manual_does_not_preset_selectbox_widget_key() -> None:
    source = Path(ra_page.__file__).read_text(encoding="utf-8")
    module_source = source[
        source.index('if source == source_module:'):
        source.index('if source == source_no_data:')
    ]

    assert "folder_pick_index" in module_source
    assert "if force_manual_pick" in module_source
    assert 'st.session_state["research_agent_module_dir_pick"] = manual_path_label' not in module_source


def test_module_folder_handoff_anchor_survives_setup_reruns() -> None:
    source = Path(ra_page.__file__).read_text(encoding="utf-8")
    module_source = source[
        source.index('if source == source_module:'):
        source.index('if source == source_no_data:')
    ]

    assert "handoff_manual_active" in module_source
    assert "force_manual_pick or handoff_manual_active" in module_source
    assert "st.session_state.pop(\"research_agent_module_dir_pick\", None)" in module_source
    assert "not handoff_manual_active" in module_source
    assert "on_change=_clear_module_folder_handoff_focus" in module_source


def test_module_folder_handoff_notice_explains_export_preselection() -> None:
    source = Path(ra_page.__file__).read_text(encoding="utf-8")
    module_source = source[
        source.index('if source == source_module:'):
        source.index('if source == source_no_data:')
    ]
    webapp_dir = Path(ra_page.__file__).parent
    i18n_source = (webapp_dir / "i18n.py").read_text(encoding="utf-8")
    css_source = (webapp_dir / "shell_overrides.css").read_text(encoding="utf-8")

    assert "ra-export-handoff" in module_source
    assert "post_export_handoff_title" in module_source
    assert "post_export_handoff_body" in module_source
    assert "post_export_handoff_files" in module_source
    assert "post_export_handoff_path" in module_source
    assert "handoff_folder_for_note = Path(selected_folder_value)" in module_source
    assert "_export_result_file_labels_for_folder(" in module_source
    assert "_ra_text(\"post_export_handoff_no_files\")" in module_source
    assert "已接收导出数据" in i18n_source
    assert "Export handoff received" in i18n_source
    assert ".ra-export-handoff" in css_source


def test_clear_module_folder_handoff_focus_releases_export_file_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _FakeStreamlit()
    fake_st.session_state.update({
        "_eu_ra_focus_module_folder": True,
        "_eu_ra_apply_export_file_selection": True,
    })
    monkeypatch.setattr(ra_page, "st", fake_st)

    ra_page._clear_module_folder_handoff_focus()

    assert fake_st.session_state["_eu_ra_focus_module_folder"] is False
    assert "_eu_ra_apply_export_file_selection" not in fake_st.session_state


def test_preflight_confirm_copy_respects_external_consent_gate() -> None:
    source = Path(ra_page.__file__).read_text(encoding="utf-8")

    assert "external_consent_needed" in source
    assert "Enable external LLM calls below to unlock Run." in source
    assert source.index("external_consent_needed") < source.index("Plan confirmed. The run button is enabled.")
    assert 'st.session_state["_eu_ra_external_llm_enabled_notice"] = True' in source
    assert "st.rerun()" in source[source.index('key="research_agent_enable_external_llm_for_run"'):]
    assert 'st.session_state.pop("_eu_ra_external_llm_enabled_notice", False)' in source


def test_infers_sepsis_filter_defaults_from_question(tmp_path: Path) -> None:
    path = tmp_path / "sep3_sofa2.parquet"
    pd.DataFrame({"stay_id": [1], "sep3_sofa2": [1]}).to_parquet(path, index=False)
    summary = ra_page._parquet_file_summary(path)

    filter_path, filter_col = ra_page._infer_filter_defaults(
        [summary],
        question="Do sepsis patients have higher hospital mortality?",
    )

    assert filter_path == path
    assert filter_col == "sep3_sofa2"


def test_scans_research_agent_history_from_final_and_partial_manifests(tmp_path: Path) -> None:
    workdir = tmp_path / "webapp"
    final_dir = workdir / "run_20260101T000000_final"
    partial_dir = workdir / "run_20260102T000000_partial"
    final_dir.mkdir(parents=True)
    partial_dir.mkdir(parents=True)

    (final_dir / "manifest.json").write_text(
        json.dumps({
            "run_id": "run_final",
            "research_question": "Does lactate predict mortality?",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:01:00+00:00",
            "per_step_records": [
                {"step_id": "00_probe", "status": "ok"},
                {"step_id": "01_model", "status": "ok"},
            ],
            "evidence": [
                {"evidence_id": "fig1", "kind": "figure"},
                {"evidence_id": "tbl1", "kind": "table"},
            ],
            "findings": [{"severity": "warning"}],
        }),
        encoding="utf-8",
    )
    (partial_dir / "manifest_partial.json").write_text(
        json.dumps({
            "run_id": "run_partial",
            "research_question": "Build a cohort audit.",
            "started_at": "2026-01-02T00:00:00+00:00",
            "per_step_records": [
                {"step_id": "00_probe", "status": "ok"},
                {"step_id": "01_table", "status": "execution_failed"},
            ],
            "evidence": [],
            "findings": [{"severity": "error"}],
        }),
        encoding="utf-8",
    )

    rows = ra_page._scan_research_agent_runs(workdir)

    assert {row["run_id"] for row in rows} == {"run_final", "run_partial"}
    partial = next(row for row in rows if row["run_id"] == "run_partial")
    assert partial["manifest_partial"] is True
    assert partial["step_ok"] == 1
    assert partial["step_failed"] == 1
    final = next(row for row in rows if row["run_id"] == "run_final")
    assert final["figure_count"] == 1
    assert final["table_count"] == 1


def test_discovers_nested_research_agent_workdir_candidates(tmp_path: Path) -> None:
    aware_dir = tmp_path / "research_output" / "q1_e2e" / "bench" / "E1" / "aware"
    run_dir = aware_dir / "run_20260611T101151_2bdd3d"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "run_id": "run_real_manifest",
            "research_question": "Does Sepsis-3 predict mortality?",
            "per_step_records": [{"step_id": "00_probe", "status": "ok"}],
            "evidence": [{"evidence_id": "tbl1", "kind": "table"}],
        }),
        encoding="utf-8",
    )
    ignored_dir = tmp_path / "research_output" / "dry_run"
    ignored_dir.mkdir(parents=True)
    (ignored_dir / "source_snapshot_manifest.json").write_text("{}", encoding="utf-8")

    candidates = ra_page._discover_research_agent_workdir_candidates(
        tmp_path / "research_output",
        limit=3,
    )

    assert len(candidates) == 1
    assert candidates[0]["workdir"] == aware_dir.resolve()
    assert candidates[0]["run_count"] == 1
    assert candidates[0]["latest_run"] == run_dir.name
    assert "source_snapshot" not in json.dumps(candidates, default=str)


def test_research_agent_history_empty_state_surfaces_manifest_discovery(tmp_path: Path) -> None:
    workdir = tmp_path / "research_output" / "webapp"
    candidate = tmp_path / "research_output" / "q1_e2e" / "bench" / "E1" / "aware"
    html_text = ra_page._history_empty_discovery_html(
        workdir=workdir,
        candidates=[{
            "workdir": candidate,
            "label": ".../bench/E1/aware",
            "run_count": 2,
            "latest_run": "run_20260611T101151_2bdd3d",
        }],
        is_en=True,
    )

    assert "ra-history-discovery" in html_text
    assert "Manifest discovery" in html_text
    assert "Scanning now" in html_text
    assert str(workdir) in html_text
    assert ".../bench/E1/aware" in html_text
    assert "2 runs" in html_text
    assert "run_20260611T101151_2bdd3d" in html_text


def test_run_summary_counts_failed_steps_and_missing_outputs(tmp_path: Path) -> None:
    manifest = {
        "run_id": "run_failed",
        "per_step_records": [
            {"step_id": "00_probe", "status": "ok"},
            {"step_id": "01_clustering", "status": "execution_failed"},
            {"step_id": "02_plot", "status": "blocked_by_concept_audit"},
        ],
        "evidence": [{"evidence_id": "log1", "kind": "log"}],
        "findings": [{"severity": "error"}, {"severity": "warning"}],
    }

    summary = ra_page._run_summary_from_manifest(tmp_path / "run_failed", manifest, partial=False)

    assert summary["status"] == "blocked"
    assert summary["step_ok"] == 1
    assert summary["step_failed"] == 2
    assert summary["figure_count"] == 0
    assert summary["table_count"] == 0
    assert summary["finding_errors"] == 1
    assert summary["finding_warnings"] == 1


def test_run_summary_preserves_backend_analysis_only_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_analysis_only"
    run_dir.mkdir()
    (run_dir / "run_status.json").write_text(
        json.dumps({"status": "analysis_only"}),
        encoding="utf-8",
    )
    manifest = {
        "run_id": "run_analysis_only",
        "per_step_records": [{"step_id": "00_probe", "status": "ok"}],
        "evidence": [],
        "findings": [],
    }

    summary = ra_page._run_summary_from_manifest(run_dir, manifest, partial=False)

    assert summary["status"] == "analysis_only"
    assert "analysis_only" in ra_page._format_history_label(summary)


def test_review_decision_roundtrip_updates_run_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_reviewed"
    run_dir.mkdir()
    manifest = {"run_id": "run_reviewed", "per_step_records": [], "evidence": [], "findings": []}

    path = ra_page._write_review_decision(
        run_dir,
        decision="approved",
        note="Checked evidence links.",
        manifest=manifest,
    )
    loaded = ra_page._load_review_decision(run_dir)
    summary = ra_page._run_summary_from_manifest(run_dir, manifest, partial=False)

    assert path.name == "review_decision.json"
    assert loaded["decision"] == "approved"
    assert loaded["note"] == "Checked evidence links."
    assert summary["review_decision"] == "approved"


def test_execution_preflight_contract_and_signature_change_with_question(tmp_path: Path) -> None:
    cohort = pd.DataFrame({"stay_id": [1, 2], "lactate": [2.0, 3.5]})

    contract = ra_page._build_execution_preflight_contract(
        free_question="Build a mortality model.",
        target_outcome="death",
        cohort=cohort,
        cohort_label="synthetic",
        llm_choice="OpenAI",
        model="gpt-test",
        workdir_text=str(tmp_path),
        stop_after_analysis=True,
        force_manuscript=False,
        template_key="prediction",
        language="en",
        llm_ready=False,
        llm_issue="api_key_missing",
    )
    changed = dict(contract)
    changed["question"] = "Run a data quality audit."

    assert contract["external_llm"] is True
    assert contract["llm_ready"] is False
    assert contract["llm_issue"] == "api_key_missing"
    assert contract["cohort_rows"] == 2
    assert contract["template_contract"]["label"] == "Prediction model"
    assert any("manifest.json" in target for target in contract["write_targets"])
    assert ra_page._preflight_signature(contract) != ra_page._preflight_signature(changed)
    ready_contract = dict(contract)
    ready_contract["llm_ready"] = True
    ready_contract["llm_issue"] = ""
    assert ra_page._preflight_signature(contract) != ra_page._preflight_signature(ready_contract)


def test_llm_run_readiness_requires_key_for_external_provider() -> None:
    assert ra_page._llm_run_readiness("MockLLMClient (offline, deterministic)", "", "") == (True, "")
    assert ra_page._llm_run_readiness("OpenAI", "", "gpt-4o-mini") == (False, "api_key_missing")
    assert ra_page._llm_run_readiness("OpenRouter", "sk-test", "") == (False, "model_missing")
    assert ra_page._llm_run_readiness("Custom OpenAI-compatible", "sk-test", "deepseek-chat") == (True, "")


def test_llm_picker_preserves_explicit_choice_before_defaulting() -> None:
    source = inspect.getsource(ra_page._section_llm_picker)
    assert "prior_choice = st.session_state.get(\"research_agent_llm_choice\")" in source
    assert "if prior_choice in options:" in source
    assert "default_index = options.index(prior_choice)" in source


def test_workbench_state_builds_from_real_run_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_20260103T000000_abcd12"
    run_dir.mkdir()
    (run_dir / "analysis.py").write_text("print('ok')\n", encoding="utf-8")
    (run_dir / "review_decision.json").write_text(
        json.dumps({"decision": "repair_requested", "note": "Fix model step."}),
        encoding="utf-8",
    )
    (run_dir / "run_status.json").write_text(
        json.dumps({
            "status": "diagnostic_only",
            "gates": {
                "execution_complete": False,
                "evidence_complete": True,
                "numeric_verified": False,
            },
        }),
        encoding="utf-8",
    )
    manifest = {
        "run_id": "run_manifest_bound",
        "research_question": "Does lactate predict mortality?",
        "context_path": "context.json",
        "plan_path": "analysis_plan.json",
        "per_step_records": [
            {"step_id": "01_table_one", "status": "ok", "generation_mode": "system"},
            {
                "step_id": "02_model_training",
                "status": "execution_failed",
                "returncode": 1,
                "evidence_ids": ["script_1"],
            },
        ],
        "evidence": [
            {
                "evidence_id": "script_1",
                "kind": "code",
                "relative_path": "analysis.py",
                "produced_by_step": "02_model_training",
            },
            {
                "evidence_id": "fig_1",
                "kind": "figure",
                "relative_path": "figures/roc.svg",
                "produced_by_step": "02_model_training",
            },
        ],
        "findings": [
            {"severity": "error", "validator": "runner", "message": "model failed"},
            {"severity": "warning", "validator": "cohort_auditor", "message": "missingness high"},
        ],
    }

    state = build_workbench_state_from_manifest(run_dir, manifest, partial=False)

    assert state["run_id"] == "run_manifest_bound"
    assert state["research_question"] == "Does lactate predict mortality?"
    assert state["status"] == "blocked"
    assert [step["status"] for step in state["steps"]] == ["ok", "fail"]
    assert "print('ok')" in state["code"]
    assert state["steps"][1]["step_id"] == "02_model_training"
    assert state["step_details"][1]["code_path"] == "analysis.py"
    assert state["step_details"][1]["results"][0]["kind"] == "figure"
    assert state["source_label"] == "Real manifest"
    assert state["audit"]["counts"] == {"errors": 1, "warnings": 1, "info": 0}
    assert any(gate["label"] == "numeric verified" and gate["ok"] is False for gate in state["audit"]["gates"])
    assert state["results"][0]["kind"] == "figure"
    assert state["evidence"][0]["tag"] == "code"
    assert state["summary_outputs"][0]["kind"] == "figure"
    assert state["execution_contract"]["workdir"] == str(run_dir)
    assert state["review_gate_actions"][0]["state"] == "blocked"
    assert state["step_details"][1]["step_contract"]["method"]["label"] == "Statistical association model"
    assert state["step_details"][1]["step_contract"]["outputs"][0]["path"] == "analysis.py"
    assert any(item["ok"] is False for item in state["step_details"][1]["step_contract"]["checkpoints"])
    assert state["review_decisions"][0]["label"] == "Saved: repair_requested"
    assert state["audit_tasks"][0]["tone"] == "danger"


def test_agent_project_snapshot_summarizes_active_run_and_history() -> None:
    state = {
        "_ra_view": "workbench",
        "research_agent_cohort_source": "Module export folder",
        "_agent_workbench": {
            "run_id": "run_snapshot",
            "run_dir": "/tmp/run_snapshot",
            "research_question": "Does lactate predict mortality?",
            "steps": [
                {"status": "ok"},
                {"status": "fail"},
            ],
            "evidence_total": 7,
            "artifact_counts": {"figures": 2, "tables": 1},
            "audit": {
                "counts": {"errors": 1, "warnings": 2},
                "gates": [{"label": "numeric verified", "ok": False}, {"label": "evidence", "ok": True}],
                "review_decision": {"decision": "repair_requested"},
            },
        },
    }
    snapshot = build_agent_project_snapshot(
        state,
        history_runs=[{"run_id": "run_prev", "status": "complete", "evidence_count": 4}],
        lang="en",
    )

    assert snapshot.project_id == "run_snapshot"
    assert snapshot.project_title == "Does lactate predict mortality?"
    assert snapshot.active_view == "workbench"
    assert snapshot.status == "Needs review"
    assert snapshot.step_total == 2
    assert snapshot.step_ok == 1
    assert snapshot.step_failed == 1
    assert snapshot.evidence_count == 7
    assert snapshot.figure_count == 2
    assert snapshot.table_count == 1
    assert snapshot.finding_errors == 1
    assert snapshot.finding_warnings == 2
    assert snapshot.gates_blocked == 1
    assert snapshot.review_decision == "repair_requested"
    assert snapshot.history_runs[0]["run_id"] == "run_prev"

    zh_empty = build_agent_project_snapshot({}, lang="zh")
    assert zh_empty.project_title == "未命名研究项目"


def test_workbench_state_disambiguates_repeated_manifest_step_labels(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_duplicate_steps"
    run_dir.mkdir()
    manifest = {
        "run_id": "run_duplicate_steps",
        "research_question": "Audit cohort setup.",
        "per_step_records": [
            {"step_id": "00_probe", "status": "ok"},
            {"step_id": "00_probe", "status": "ok"},
            {"step_id": "01_cohort_summary", "status": "ok"},
        ],
        "evidence": [],
        "findings": [],
    }

    state = build_workbench_state_from_manifest(run_dir, manifest, partial=False)

    assert [step["label"] for step in state["steps"]] == [
        "Probe 1",
        "Probe 2",
        "Cohort Summary",
    ]
    assert [detail["label"] for detail in state["step_details"][:2]] == ["Probe 1", "Probe 2"]
    html = wb_page._agent_reference_workbench_html(state, "en")
    assert "Probe 1" in html
    assert "Probe 2" in html


def test_workbench_review_summary_counts_manifest_repair_events(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_repair_count"
    run_dir.mkdir()
    manifest = {
        "run_id": "run_repair_count",
        "research_question": "Audit repair visibility.",
        "per_step_records": [
            {"step_id": "00_probe", "status": "ok"},
            {"step_id": "01_model", "status": "ok", "code_repair_attempts": 1},
            {"step_id": "02_table", "status": "running"},
        ],
        "evidence": [],
        "findings": [],
    }

    state = build_workbench_state_from_manifest(run_dir, manifest, partial=True)

    assert state["steps"][1]["repair_count"] == 1
    assert wb_page._review_steps_summary_text(state["steps"], "en") == "2 done · 1 repair · 1 running"


def test_workbench_finding_open_step_infers_target_from_message() -> None:
    steps = [
        {"step_id": "00_probe", "label": "Probe 1"},
        {"step_id": "00_probe", "label": "Probe 2"},
        {"step_id": "04_component_qc", "label": "Component QC"},
        {"step_id": "04_component_qc_figure", "label": "Component QC Figure"},
    ]
    step_ids = list(wb_page._step_id_to_first_index(steps))

    assert wb_page._step_id_to_first_index(steps)["00_probe"] == 0
    assert wb_page._finding_target_step_id(
        {
            "validator": "critic_agent",
            "message": "CriticAgent marked 00_probe as needs_revision.",
        },
        step_ids,
    ) == "00_probe"
    assert wb_page._finding_target_step_id(
        {
            "validator": "critic_agent",
            "message": "CriticAgent marked 04_component_qc_figure as needs_revision.",
        },
        step_ids,
    ) == "04_component_qc_figure"
    assert wb_page._finding_target_step_id(
        {"validator": "critic_agent", "message": "No step marker here."},
        step_ids,
    ) == ""


def test_workbench_finding_queue_rows_carry_review_and_step_state() -> None:
    findings = [
        {
            "severity": "warning",
            "validator": "critic_agent",
            "message": "CriticAgent marked 04_component_qc_figure as needs_revision.",
        },
        {
            "severity": "warning",
            "validator": "statistical_guard",
            "message": "Multiplicity check needs manual review.",
        },
        {
            "severity": "error",
            "validator": "numeric_validator",
            "message": "Primary effect mismatch in 05_primary_association.",
        },
        {"severity": "info", "validator": "replanner", "message": "Plan revised."},
    ]
    reviewed = {wb_page._finding_review_id(findings[0])}
    state = {
        "steps": [
            {"step_id": "04_component_qc", "label": "Component QC"},
            {"step_id": "04_component_qc_figure", "label": "Component QC Figure"},
            {"step_id": "05_primary_association", "label": "Primary Association"},
        ],
        "audit": {"findings": findings},
    }

    rows = wb_page._finding_queue_rows(state, reviewed_ids=reviewed)
    stats = wb_page._finding_queue_stats(rows)

    assert len(rows) == 3
    assert rows[0]["reviewed"] is True
    assert rows[0]["target_index"] == 1
    assert rows[0]["target_label"] == "02 · Component QC Figure"
    assert rows[1]["target_index"] is None
    assert rows[2]["severity"] == "error"
    assert stats == {"total": 3, "reviewed": 1, "errors": 1, "warnings": 2, "linked": 2}


def test_workbench_finding_queue_defaults_to_compact_preview() -> None:
    rows = [{"index": i, "review_id": f"finding-{i}"} for i in range(12)]

    preview = wb_page._visible_finding_queue_rows(rows, show_all=False, limit=5)
    expanded = wb_page._visible_finding_queue_rows(rows, show_all=True, limit=5)

    assert [row["index"] for row in preview] == [0, 1, 2, 3, 4]
    assert expanded == rows


def test_workbench_reviewed_findings_persist_per_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_persisted_review"
    run_dir.mkdir()
    findings = [
        {
            "severity": "warning",
            "validator": "critic_agent",
            "message": "CriticAgent marked 01_plan as needs_revision.",
        },
        {
            "severity": "warning",
            "validator": "statistical_guard",
            "message": "Multiplicity check needs manual review.",
        },
    ]
    reviewed_id = wb_page._finding_review_id(findings[0])

    path = wb_page._write_reviewed_finding_ids(
        run_dir,
        [reviewed_id],
        run_id="run_persisted_review",
    )
    manifest = {
        "run_id": "run_persisted_review",
        "per_step_records": [{"step_id": "01_plan", "status": "ok"}],
        "evidence": [],
        "findings": findings,
    }
    state = wb_page.build_workbench_state_from_manifest(run_dir, manifest, lang="en")
    rows = wb_page._finding_queue_rows(state)

    assert path == run_dir / "finding_review_state.json"
    assert wb_page._load_reviewed_finding_ids(run_dir) == [reviewed_id]
    assert state["reviewed_finding_ids"] == [reviewed_id]
    assert state["finding_review_state_path"].endswith("finding_review_state.json")
    assert rows[0]["reviewed"] is True
    assert rows[1]["reviewed"] is False


def test_workbench_finding_queue_uses_design_queue_surface() -> None:
    source = Path(wb_page.__file__).read_text(encoding="utf-8")
    css = Path(wb_page.__file__).with_name("shell_overrides.css").read_text(encoding="utf-8")
    render_source = source[
        source.index("def _render_audit_actions"):
        source.index("# ---------------------------------------------------------------------", source.index("def _render_audit_actions"))
    ]

    assert "Finding queue" in render_source
    assert "Review warnings before Summary sign-off" in render_source
    assert "Triage findings" not in render_source
    assert "_finding_queue_rows(state, reviewed_ids=acked)" in render_source
    assert "_visible_finding_queue_rows(rows, show_all=show_all)" in render_source
    assert "Show all" in render_source
    assert "No linked step" in render_source
    assert "manual review across the run evidence" in render_source
    assert "_store_reviewed_findings_for_state(state, acked)" in render_source
    assert 'st.session_state["_active_main_page"] = "research_agent"' in render_source
    assert 'st.session_state["_ra_view"] = "workbench"' in render_source
    assert "_finding_queue_stats(rows)" in render_source
    assert "_REVIEW_DETAILS_EXPANDED_KEY" in source
    assert "_render_workbench_technical_details_gate(lang)" in source
    assert "expanded=True" in source
    assert "eu-finding-card" in render_source
    assert "eu-finding-target" in render_source
    assert "eu-finding-queue-head" in css
    assert "st-key-_eu_wb_finding_row_" in css
    assert "st-key-_eu_wb_finding_open_" in css
    assert "st-key-_eu_wb_finding_ack_" in css
    assert '[data-testid="stTabs"] [role="tablist"]' in css
    assert "flex-wrap: wrap !important;" in css


def test_workbench_step_selection_keeps_review_details_expanded(monkeypatch) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(wb_page, "st", fake_st)

    wb_page._set_selected_step("_step_key", 2)

    assert fake_st.session_state["_step_key"] == 2
    assert fake_st.session_state[wb_page._REVIEW_DETAILS_EXPANDED_KEY] is True


def test_workbench_audit_gate_delegates_warnings_to_finding_queue() -> None:
    css = Path(wb_page.__file__).with_name("shell_overrides.css").read_text(encoding="utf-8")
    audit = {
        "run_status": "analysis_only",
        "counts": {"errors": 0, "warnings": 2, "info": 1},
        "gates": [
            {"label": "evidence complete", "ok": False},
            {"label": "numeric verified", "ok": False},
            {"label": "analysis validated", "ok": True},
            {"label": "publication ready", "ok": False},
        ],
        "findings": [
            {"severity": "info", "validator": "replanner", "message": "Plan revised after probe."},
            {"severity": "warning", "validator": "statistical_guard", "message": "Multiplicity needs review."},
            {"severity": "warning", "validator": "visual_qa", "message": "Figure label may crop."},
        ],
        "reproducibility": "deterministic fallback · no external LLM",
    }
    tasks = wb_page._audit_tasks_from_audit(audit, lang="en")
    html = wb_page._audit_review_html({
        "audit": audit,
        "audit_tasks": tasks,
        "review_decisions": [],
    }, "en")

    assert [task["title"] for task in tasks] == [
        "Review readiness gates",
        "Review warning queue",
    ]
    assert tasks[0]["action"] == "Gate follow-up"
    assert tasks[1]["action"] == "Finding queue"
    assert "Resolve gate:" not in json.dumps(tasks)
    assert "Audit notes" in html
    assert "Plan revised after probe." in html
    assert "Multiplicity needs review." not in html
    assert "Reviewable findings are handled in the Finding queue below." not in html
    assert "deterministic fallback · no external LLM" in html
    assert ".eu-audit-review-grid" in css
    assert "grid-template-columns: minmax(0, 1fr);" in css


def test_workbench_audit_payload_hides_internal_writer_gates(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_analysis_only"
    run_dir.mkdir()
    (run_dir / "run_status.json").write_text(json.dumps({
        "status": "analysis_only",
        "gates": {
            "execution_complete": True,
            "evidence_complete": False,
            "numeric_verified": False,
            "analysis_validated": True,
            "manuscript_ready": False,
            "publication_ready": False,
            "manuscript_generated": False,
            "writer_probe_mode": False,
        },
    }), encoding="utf-8")

    audit = wb_page._audit_payload(
        manifest={"findings": []},
        run_dir=run_dir,
        partial=False,
    )
    labels = [gate["label"] for gate in audit["gates"]]
    tasks = wb_page._audit_tasks_from_audit(audit, lang="en")

    assert "manuscript generated" not in labels
    assert "writer probe mode" not in labels
    assert "evidence complete" in labels
    assert "numeric verified" in labels
    assert "publication ready" in labels
    assert tasks[0]["detail"].startswith("4 readiness gate(s)")


def test_demo_workbench_summary_slots_do_not_expose_fake_metrics() -> None:
    state = _demo_state("en")
    payload = json.dumps(state, ensure_ascii=False)

    assert state["is_demo"] is True
    assert "does not fabricate cohort metrics" in state["summary_outputs"][1]["sub"]
    assert state["execution_contract"]["provider"] == "No LLM call"
    assert state["review_gate_actions"][0]["state"] == "ready"
    assert state["review_decisions"][0]["label"] == "Preview only"
    assert state["audit_tasks"][0]["title"] == "Open a real manifest"
    assert state["step_details"][0]["step_contract"]["method"]["label"] == "Demo method slot"
    assert "AUC" not in payload
    assert "Brier" not in payload
    assert "2,481" not in payload
    assert all(not result.get("svg") for result in state["results"])
    assert all("No generated output" in result.get("preview_html", "") for result in state["results"])


def test_workbench_uses_reference_empty_state_for_demo_preview(monkeypatch) -> None:
    class _StreamlitStub:
        session_state = {
            "entry_mode": "demo",
            "_agent_workbench": {
                "is_demo": True,
                "steps": [{"label": "Fake", "status": "running"}],
            },
        }

    monkeypatch.setattr(wb_page, "st", _StreamlitStub)

    state = _resolve_workbench_state("en")
    empty_html = wb_page._workbench_empty_html("en")

    assert state == {}
    assert "No active run" in empty_html
    assert "eu-agent-empty-glyph" in empty_html
    assert "Start with one research question" in empty_html
    assert "Setup" in empty_html
    assert "Run" in empty_html
    assert "Review" in empty_html
    assert "Local run history stays on this machine" in empty_html
    assert "Static preview" not in empty_html
    assert "Task map" not in empty_html
    assert "Analysis outputs" not in empty_html


def test_empty_workbench_removes_ambiguous_open_latest_run_action() -> None:
    source = Path(wb_page.__file__).read_text(encoding="utf-8")

    assert "Open latest run" not in source
    assert "打开最近 run" not in source
    assert "_eu_wb_empty_latest" not in source
    assert "_latest_real_workbench_state" not in source
    assert "Open setup" in source
    assert "打开配置" in source
    assert "Open local saved runs" in source
    assert "查看本机历史运行" in source
    assert "Local run history stays on this machine" in source
    assert "本机运行历史只保留在这台机器上" in source
    assert "Open manifest history" not in source
    assert "打开 manifest 历史" not in source
    assert "_eu_wb_empty_history" in source
    empty_source = source[
        source.index("def _render_workbench_empty_state"):
        source.index("def _step_contract_html")
    ]
    assert '_route_to_agent_empty_state_target("history")' in empty_source
    assert '_route_to_agent_empty_state_target("setup")' in empty_source
    assert 'st.session_state["_ra_view"] = "setup"' not in empty_source[empty_source.index("_eu_wb_empty_history"):]
    assert 'st.session_state.pop("_research_agent_expand_history", None)' in empty_source


def test_empty_workbench_actions_stay_inside_agent_workspace(monkeypatch) -> None:
    fake_st = _FakeStreamlit()
    fake_st.session_state.update({"entry_mode": "none"})
    monkeypatch.setattr(wb_page, "st", fake_st)

    wb_page._route_to_agent_empty_state_target("history")

    assert fake_st.session_state["entry_mode"] == "real"
    assert fake_st.session_state["use_mock_data"] is False
    assert fake_st.session_state["_active_main_page"] == "research_agent"
    assert "_main_nav_widget" not in fake_st.session_state
    assert fake_st.session_state["_ra_view"] == "history"


def test_research_agent_history_copy_is_local_only() -> None:
    source = Path(ra_page.__file__).read_text(encoding="utf-8")
    i18n_source = Path(ra_page.__file__).with_name("i18n.py").read_text(encoding="utf-8")

    assert "Local run history is loaded on demand from this workdir only" in source
    assert "not uploaded to GitHub" in source
    assert "本机运行历史只会按需从当前工作目录读取" in source
    assert "Load local recent runs" in source
    assert "加载本机最近 run" in source
    assert "'ra_history_title': 'Local analysis records'" in i18n_source
    assert "'ra_history_title': '本机分析记录'" in i18n_source


def test_resume_panel_does_not_nest_streamlit_expanders() -> None:
    source = Path(ra_page.__file__).read_text(encoding="utf-8")
    resume_source = source[
        source.index("def _render_resume_panel"):
        source.index("def _render_research_agent_demo_visuals")
    ]

    assert 'with st.expander(_ra_text("resume_section")' in resume_source
    assert 'resume_expanded = can_resume and (status_label != "complete" or bool(failed))' in resume_source
    assert 'with st.expander(_ra_text("resume_failed_steps")' not in resume_source
    assert 'with st.expander(_ra_text("resume_findings_summary")' not in resume_source
    assert resume_source.index("prior_question =") < resume_source.index("action_cols =")
    assert resume_source.count('st.session_state["research_agent_question"] = prior_question') == 2
    assert resume_source.count('st.session_state["_active_main_page"] = "research_agent"') >= 2


def test_empty_workbench_actions_use_responsive_nowrap_buttons() -> None:
    source = Path(wb_page.__file__).read_text(encoding="utf-8")
    css = Path(wb_page.__file__).with_name("shell_overrides.css").read_text(encoding="utf-8")

    assert 'st.container(key=f"eu_wb_empty_panel_{summary}")' in source
    assert 'st.container(key=f"eu_wb_empty_actions_{summary}")' in source
    assert "st.columns(2)" in source
    assert "st.columns([1.4, 1.75, 6.15])" not in source
    assert "eu-agent-empty-glyph" in source
    assert "eu-agent-empty-path" in source
    assert "st-key-eu_wb_empty_actions" in css
    assert "st-key-eu_wb_empty_panel" in css
    assert 'class*="st-key-_eu_wb_empty_"' in css
    assert "max-width: 560px" in css
    assert "margin: 14px 0 24px" in css
    assert "margin: 22px auto 0" in css
    assert "height: 38px" in css
    assert "white-space: nowrap" in css
    assert "text-overflow: ellipsis" in css
    assert "min-height: 326px" in css
    assert "border: 1px dashed var(--hair-2)" in css
    assert ".eu-agent-empty-glyph" in css
    assert ".eu-agent-empty small" in css
    assert ".eu-agent-empty-path" in css
    assert "st-key-_eu_wb_open_step_details" in css
    assert "margin: 16px auto 0" in css
    assert "@media (max-width: 560px)" in css


def test_workbench_default_tab_is_step_review_not_code() -> None:
    source = Path(wb_page.__file__).read_text(encoding="utf-8")
    css = Path(wb_page.__file__).with_name("shell_overrides.css").read_text(encoding="utf-8")
    tabs_source = source[
        source.index("def _render_code_panel_tabs"):
        source.index("def _render_evidence_drilldown")
    ]

    assert "def _step_review_html" in source
    assert "_step_review_html(active_state, full_state, lang)" in tabs_source
    assert "include_review: bool = True" in tabs_source
    assert "if include_review:" in tabs_source
    assert tabs_source.index("Review") < tabs_source.index("Script")
    assert "Activity" in tabs_source
    assert "Issues" in tabs_source
    assert "All steps" in tabs_source
    assert ".eu-wb-step-review" in css
    assert ".eu-wb-step-check" in css
    assert ".eu-wb-step-artifact" in css


def test_summary_review_gate_uses_checklist_and_locked_draft_card() -> None:
    source = Path(wb_page.__file__).read_text(encoding="utf-8")
    css = Path(wb_page.__file__).with_name("shell_overrides.css").read_text(encoding="utf-8")
    summary_source = source[
        source.index("def _output_summary_html"):
        source.index("def render_agent_output_summary")
    ]
    render_source = source[
        source.index("def render_agent_output_summary"):
        source.index("def _resolve_workbench_state")
    ]

    assert "Manuscript draft is locked until checks pass" in summary_source
    assert "eu-summary-checklist" in summary_source
    assert "eu-summary-check-row" in summary_source
    assert "Draft methods + results" in summary_source
    assert "Output bundle" not in summary_source
    assert "Export package" in summary_source
    assert "Summary export terminal" in source
    assert "eu-summary-export-terminal" in source
    assert "eu-summary-export-row" in source
    assert "_summary_export_terminal_html" in summary_source
    assert "Export package is demo-only" in summary_source
    assert "Export review ZIP" in source
    assert "One reviewer sign-off outstanding" in summary_source
    assert "eu-summary-review-terminal" in source
    assert "eu-summary-review-action-row" in source
    assert "_summary_review_action_terminal_html" in summary_source
    assert "_remember_summary_review_receipt" in summary_source
    assert "st.columns([1, 1, 1.15])" not in summary_source
    assert "st.success(" not in summary_source
    assert "st.warning(" not in summary_source
    assert "Add reviewer note" in summary_source
    assert "_eu_summary_review_note_visible_" in summary_source
    assert "note_visible = st.toggle(" in summary_source
    assert "review_decision.json and unlocks the draft gate" in summary_source
    assert "height=68" in summary_source
    assert "height=58" not in summary_source
    assert "def _prime_summary_draft_setup" in source
    assert "research_agent_force_manuscript" in source
    assert "research_agent_resume_mode" in source
    assert "force_manuscript" in source
    assert "Draft methods + results" in source[source.index("def _render_summary_review_controls"):source.index("def render_agent_output_summary")]
    assert "def _summary_empty_html" in source
    assert "locked until review" in source
    assert "Results and export appear here after a run" in source
    assert "_render_summary_empty_state(lang, show_header=show_header)" in render_source
    assert "_render_workbench_empty_state(lang, summary=True)" not in render_source
    assert "eu-summary-reference-grid" in summary_source
    assert "_summary_bundle_counts(state)" in summary_source
    assert "summary_outputs" in source
    assert "Inbound cohort" not in summary_source
    assert "Research question" not in summary_source
    assert 'title_en="EasyICU Research Agent"' in render_source
    assert "An auditable, evidence-bound workflow" in render_source
    assert "Analysis-first output summary" not in render_source
    assert "review checks outstanding" in summary_source
    assert "Resolve the pending checks before preparing methods and results" in summary_source
    assert ".eu-summary-checklist" in css
    assert ".eu-summary-check-row" in css
    assert ".eu-summary-action-token" in css
    assert ".eu-summary-action-token.ready" in css
    assert ".eu-summary-reference-grid" in css
    assert ".eu-summary-export-terminal" in css
    assert ".eu-summary-export-row" in css
    assert ".eu-summary-export-button" in css
    assert ".eu-summary-review-terminal" in css
    assert ".eu-summary-review-action-row" in css
    assert ".eu-summary-review-receipt" in css
    assert ".eu-summary-bundle-details:not([open]) > :not(summary)" in css
    assert ".eu-summary-bundle-summary" in css
    assert ".eu-summary-bundle-row" in css
    assert ".eu-summary-bundle-button" in css
    assert "width: min(100%, calc(100% - 316px))" not in css
    assert "grid-column: 1 / -1" in css


def test_summary_draft_cta_routes_to_force_manuscript_setup(monkeypatch) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(wb_page, "st", fake_st)

    wb_page._prime_summary_draft_setup({
        "run_id": "run_20260528T184052_adaf4d",
        "run_dir": "/tmp/run_20260528T184052_adaf4d",
        "research_question": "Does lactate predict mortality?",
    })

    assert fake_st.session_state["research_agent_resume_run_id"] == "run_20260528T184052_adaf4d"
    assert fake_st.session_state["research_agent_resume_run_dir"] == "/tmp/run_20260528T184052_adaf4d"
    assert fake_st.session_state["research_agent_workdir"] == str(Path("/tmp").resolve())
    assert fake_st.session_state["research_agent_force_manuscript"] is True
    assert fake_st.session_state["research_agent_resume_mode"] == "force_manuscript"
    assert fake_st.session_state["research_agent_question"] == "Does lactate predict mortality?"
    assert fake_st.session_state["_active_main_page"] == "research_agent"
    assert fake_st.session_state["_ra_view"] == "setup"
    assert fake_st.session_state["_research_agent_expand_history"] is False


def test_resume_run_seeds_cohort_from_local_run_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_20260601T010203_abcd"
    run_dir.mkdir()
    cohort_path = run_dir / "cohort.parquet"
    pd.DataFrame({
        "stay_id": [1, 2],
        "death": [0, 1],
        "sofa2": [3, 8],
    }).to_parquet(cohort_path, index=False)
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "run_id": "run_20260601T010203_abcd",
            "research_question": "Does SOFA predict mortality?",
        }),
        encoding="utf-8",
    )

    state: dict[str, object] = {
        "research_agent_resume_run_id": "run_20260601T010203_abcd",
        "research_agent_resume_run_dir": str(run_dir),
        "research_agent_resume_mode": "continue",
    }

    restored = ra_page._restore_resume_cohort_handoff(state)

    assert restored is True
    assert isinstance(state["research_agent_inbound_cohort"], pd.DataFrame)
    assert state["research_agent_inbound_cohort"].shape == (2, 3)
    assert state["research_agent_inbound_cohort_label"] == "resume:run_20260601T010203_abcd:cohort.parquet"
    assert str(state["research_agent_resume_cohort_signature"]).startswith("resume:run_20260601T010203_abcd:")
    assert state["research_agent_inbound_signature"] == state["research_agent_resume_cohort_signature"]


def test_force_manuscript_seeds_cohort_from_copied_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "copied_reviewed_run"
    run_dir.mkdir()
    cohort_path = run_dir / "cohort.parquet"
    pd.DataFrame({
        "stay_id": [1, 2, 3],
        "death": [0, 1, 0],
        "sofa2": [2, 9, 5],
    }).to_parquet(cohort_path, index=False)
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "run_id": "run_20260601T035358_d282cb",
            "research_question": "Does SOFA predict mortality?",
        }),
        encoding="utf-8",
    )

    state: dict[str, object] = {
        "research_agent_resume_run_id": "run_20260601T035358_d282cb",
        "research_agent_resume_run_dir": str(run_dir),
        "research_agent_resume_mode": "force_manuscript",
    }

    restored = ra_page._restore_resume_cohort_handoff(state)

    assert restored is True
    assert isinstance(state["research_agent_inbound_cohort"], pd.DataFrame)
    assert state["research_agent_inbound_cohort"].shape == (3, 3)
    assert state["research_agent_inbound_cohort_label"] == "resume:run_20260601T035358_d282cb:cohort.parquet"
    assert str(state["research_agent_resume_cohort_signature"]).startswith("resume:run_20260601T035358_d282cb:")


def test_resume_run_context_sets_followup_workdir_to_run_parent(tmp_path: Path) -> None:
    run_dir = tmp_path / "external_history" / "copied_run_folder"
    run_dir.mkdir(parents=True)
    state: dict[str, object] = {}

    ra_page._store_resume_run_dir_context(state, run_dir)

    assert state["research_agent_resume_run_dir"] == str(run_dir)
    assert state["research_agent_workdir"] == str(run_dir.resolve().parent)


def test_resume_run_context_can_defer_workdir_until_next_render(tmp_path: Path) -> None:
    run_dir = tmp_path / "external_history" / "copied_run_folder"
    run_dir.mkdir(parents=True)
    state: dict[str, object] = {}

    ra_page._store_resume_run_dir_context(state, run_dir, defer_workdir=True)

    assert state["research_agent_resume_run_dir"] == str(run_dir)
    assert "research_agent_workdir" not in state
    assert state["_research_agent_workdir_pending"] == str(run_dir.resolve().parent)
    ra_page._apply_pending_research_agent_workdir(state)
    assert state["research_agent_workdir"] == str(run_dir.resolve().parent)
    assert "_research_agent_workdir_pending" not in state


def test_history_resume_buttons_use_selected_run_workdir_context() -> None:
    source = Path(ra_page.__file__).read_text(encoding="utf-8")
    panel_source = source[
        source.index("def _render_resume_panel"):
        source.index("def _render_research_agent_demo_visuals")
    ]

    assert panel_source.count("_store_resume_run_dir_context(st.session_state, run_dir, defer_workdir=True)") == 2


def test_research_agent_page_restores_cohort_for_force_manuscript_resume() -> None:
    source = Path(ra_page.__file__).read_text(encoding="utf-8")
    render_source = source[
        source.index("def render_research_agent_page"):
        source.index("_step_titles = [", source.index("def render_research_agent_page"))
    ]

    assert 'resume_mode in {"continue", "force_manuscript"}' in render_source
    assert "_restore_resume_cohort_handoff(st.session_state)" in render_source


def test_cancel_resume_clears_only_resume_loaded_cohort() -> None:
    df = pd.DataFrame({"stay_id": [1]})
    state: dict[str, object] = {
        "research_agent_inbound_cohort": df,
        "research_agent_inbound_cohort_label": "resume:run_1:cohort.parquet",
        "research_agent_inbound_signature": "resume:run_1:/tmp/cohort.parquet:1:2",
        "research_agent_resume_cohort_signature": "resume:run_1:/tmp/cohort.parquet:1:2",
    }

    ra_page._clear_resume_cohort_handoff(state)

    assert "research_agent_inbound_cohort" not in state
    assert "research_agent_inbound_cohort_label" not in state
    assert "research_agent_inbound_signature" not in state
    assert "research_agent_resume_cohort_signature" not in state


def test_research_agent_question_widget_uses_session_state_without_duplicate_default() -> None:
    source = Path(ra_page.__file__).read_text(encoding="utf-8")
    request_source = source[
        source.index("def _section_request_picker"):
        source.index("def _section_method_preferences")
    ]

    assert 'st.session_state.setdefault("research_agent_question", "")' in request_source
    assert 'key="research_agent_question"' in request_source
    assert "on_change=_on_research_agent_question_change" in request_source
    assert '_research_agent_question_applied_notice' in request_source
    assert '_research_agent_question_empty_notice' in request_source
    assert 'value=st.session_state.get("research_agent_question", "")' not in request_source


def test_research_agent_renders_copilot_handoff_packet_before_fallback_notice() -> None:
    source = Path(ra_page.__file__).read_text(encoding="utf-8")
    request_source = source[
        source.index("def _section_request_picker"):
        source.index("def _section_method_preferences")
    ]
    helper_source = inspect.getsource(ra_page._render_copilot_handoff_packet)
    css_source = Path(ra_page.__file__).with_name("shell_overrides.css").read_text(encoding="utf-8")

    assert "research_agent_copilot_handoff_packet" in helper_source
    assert "Copilot handoff packet" in helper_source
    assert "ra-copilot-handoff" in helper_source
    assert "ra-copilot-handoff-ledger" in helper_source
    assert "ra-copilot-handoff-node" in helper_source
    assert "Setup packet landing" in helper_source
    assert "preflight_status" in helper_source
    assert "_render_copilot_handoff_packet(" in request_source
    assert "question_handoff_notice and not packet_rendered" in request_source
    assert ".ra-copilot-handoff" in css_source
    assert ".ra-copilot-handoff-facts" in css_source
    assert ".ra-copilot-handoff-node" in css_source
    assert ".ra-copilot-handoff-receipt" in css_source
    assert "border-left: 4px solid var(--accent)" in css_source


def test_research_agent_handoff_receipt_summarizes_loaded_export_without_patient_ids(
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / "easyicu_export"
    export_dir.mkdir()
    files = [
        export_dir / "demographics.parquet",
        export_dir / "vitals.parquet",
        export_dir / "outcome.parquet",
    ]
    for path in files:
        path.write_bytes(b"stub")
    state = {
        "_eu_ra_focus_module_folder": True,
        "_eu_ra_apply_export_file_selection": True,
        "research_agent_module_dir_text": str(export_dir),
        "research_agent_cohort_source": "Pick an EasyICU module export folder",
        "patient_ids": [10001, 10002, 10003],
        "loaded_concepts": {
            "age": pd.DataFrame({"stay_id": [10001, 10002, 10003]}),
            "hr": pd.DataFrame({"stay_id": [10001, 10002, 10003]}),
        },
        "_review_source_concept_count": 3,
        "_review_subset_concept_count": 2,
        "_export_success_result": {
            "files": [str(path) for path in files],
            "patient_count": 3,
        },
    }

    summary = ra_page._agent_handoff_context_summary(state, is_en=True)
    html = ra_page._agent_handoff_context_html(summary, is_en=True)
    render_source = inspect.getsource(ra_page.render_research_agent_page)
    css_source = Path(ra_page.__file__).with_name("shell_overrides.css").read_text(encoding="utf-8")

    assert summary["active"] is True
    assert summary["patient_count"] == 3
    assert summary["feature_count"] == 2
    assert summary["selected_count"] == 3
    assert summary["file_count"] == 3
    assert summary["export_dir"] == str(export_dir)
    assert "ra-agent-handoff-receipt" in html
    assert "Patient Review -&gt; Agent Projects" in html
    assert "Loaded context ready for Agent Projects" in html
    assert "ICU stays" in html
    assert "Review features" in html
    assert "2 / 3" in html
    assert "Export files" in html
    assert str(export_dir) in html
    assert "10001" not in html
    assert "10002" not in html
    assert "10003" not in html
    assert "_render_agent_handoff_context_receipt(st.session_state, is_en=_is_en)" in render_source
    assert "_eu_ra_handoff_receipt_rendered_in_shell" in inspect.getsource(
        ra_page._render_agent_handoff_context_receipt
    )
    assert ".ra-agent-handoff-receipt" in css_source
    assert ".ra-agent-handoff-receipt-shell" in css_source
    assert ".ra-agent-handoff-stats" in css_source
    assert ".ra-agent-handoff-path code" in css_source


def test_research_agent_method_widgets_use_session_state_without_duplicate_defaults() -> None:
    source = Path(ra_page.__file__).read_text(encoding="utf-8")
    preferences_source = source[
        source.index("def _section_method_preferences"):
        source.index("def _section_llm_picker")
    ]

    for key in (
        "research_agent_method_preferences_text",
        "research_agent_evaluation_focus",
        "research_agent_subgroup_sensitivity",
        "research_agent_timing_design",
        "research_agent_data_constraints",
        "research_agent_must_have_outputs",
        "research_agent_covariates",
        "research_agent_extra_notes",
    ):
        assert f'"{key}"' in preferences_source
    assert "preference_widget_keys" in preferences_source
    assert "st.session_state.setdefault(widget_key, \"\")" in preferences_source
    assert "value=st.session_state.get(\"research_agent_method_preferences_text\"" not in preferences_source
    assert "value=st.session_state.get(\"research_agent_evaluation_focus\"" not in preferences_source
    assert 'value=""' not in preferences_source


def test_summary_reference_uses_manifest_evidence_total_and_icons() -> None:
    state = {
        "run_id": "run_demo",
        "run_dir": "/tmp/run_demo",
        "source_label": "Real manifest",
        "status": "done",
        "steps": [{"label": "Cohort summary", "status": "ok"}],
        "evidence": [{"label": f"row {i}", "tag": "table"} for i in range(12)],
        "evidence_total": 122,
        "artifact_counts": {"figures": 17, "tables": 11, "code": 4, "evidence": 122},
        "summary_outputs": [
            {"kind": "table", "title": "Table 1"},
            {"kind": "figure", "title": "Missingness figure"},
        ],
        "audit": {
            "counts": {"errors": 0, "warnings": 15},
            "findings": [
                {"severity": "warning", "validator": "critic", "message": "Check table 1."},
                {"severity": "warning", "validator": "critic", "message": "Check model card."},
            ],
            "review_decision": {},
        },
        "is_demo": False,
    }
    html = wb_page._output_summary_html(state, "en")

    assert "122 manifest rows" in html
    assert "12 manifest rows" not in html
    assert "17 figures" in html
    assert "11 tables" in html
    assert "4 code artifacts" in html
    assert "0 error(s) · 15 warning(s)" in html
    assert "0/2 reviewed" in html
    assert "2 review checks outstanding" in html
    assert "One reviewer sign-off outstanding" not in html
    assert "0E / 15W" not in html
    assert "0E/15W" not in html
    assert "Summary export terminal" in html
    assert "eu-summary-export-terminal available" in html
    assert "eu-summary-export-row available" in html
    assert "run_demo_output_bundle.zip" in html
    assert "Export review ZIP" in html
    assert "eu-summary-bundle-details" not in html
    assert "eu-summary-bundle-ico\"><svg" not in html
    assert ">F</span>" not in html
    assert ".eu-summary-bundle-ico svg" in Path(wb_page.__file__).with_name("shell_overrides.css").read_text(encoding="utf-8")


def test_summary_denominator_gate_accepts_table_one_and_locked_cohort_evidence() -> None:
    state = {
        "run_id": "run_table_one",
        "run_dir": "/tmp/run_table_one",
        "source_label": "Real manifest",
        "status": "done",
        "steps": [{"step_id": "01_table_one", "label": "Table One", "status": "ok"}],
        "evidence": [
            {"kind": "log", "relative_path": "evidence/cohort_locked__cohort_locked.json"},
            {"kind": "table", "relative_path": "evidence/table_table_one_74af4152__table_one.csv"},
        ],
        "artifact_counts": {"figures": 1, "tables": 1, "code": 1, "evidence": 2},
        "audit": {
            "counts": {"errors": 0, "warnings": 0},
            "findings": [],
            "review_decision": {},
        },
        "reviewed_finding_ids": [],
        "is_demo": False,
    }

    checks = wb_page._summary_review_checks(state, "en")
    html = wb_page._output_summary_html(state, "en")

    cohort_check = next(check for check in checks if check["label"] == "Cohort denominators resolved")
    assert cohort_check["ok"] is True
    assert "One reviewer sign-off outstanding" in html
    assert "1 review checks outstanding" not in html
    assert "2 review checks outstanding" not in html


def test_summary_gate_advances_when_warning_findings_are_marked_reviewed() -> None:
    findings = [
        {"severity": "warning", "validator": "critic", "message": "Check table 1."},
        {"severity": "warning", "validator": "critic", "message": "Check model card."},
    ]
    reviewed = [wb_page._finding_review_id(finding) for finding in findings]
    state = {
        "run_id": "run_reviewed_warnings",
        "run_dir": "/tmp/run_reviewed_warnings",
        "source_label": "Real manifest",
        "status": "done",
        "steps": [{"label": "Cohort summary", "status": "ok"}],
        "evidence": [{"label": "row", "tag": "table"}],
        "artifact_counts": {"figures": 1, "tables": 1, "code": 1, "evidence": 1},
        "audit": {
            "counts": {"errors": 0, "warnings": 2},
            "findings": findings,
            "review_decision": {},
        },
        "reviewed_finding_ids": reviewed,
        "is_demo": False,
    }

    checks = wb_page._summary_review_checks(state, "en")
    html = wb_page._output_summary_html(state, "en")

    validator_check = next(check for check in checks if check["label"] == "Validator findings reviewed")
    assert validator_check["ok"] is True
    assert validator_check["status"] == "2/2 reviewed"
    assert "One reviewer sign-off outstanding" in html
    assert "2/2 reviewed" in html


def test_summary_gate_unlocks_after_saved_reviewer_signoff() -> None:
    findings = [
        {"severity": "warning", "validator": "critic", "message": "Check table 1."},
        {"severity": "warning", "validator": "critic", "message": "Check model card."},
    ]
    reviewed = [wb_page._finding_review_id(finding) for finding in findings]
    state = {
        "run_id": "run_signed_off",
        "run_dir": "/tmp/run_signed_off",
        "source_label": "Real manifest",
        "status": "done",
        "steps": [{"label": "Cohort summary", "status": "ok"}],
        "evidence": [{"label": "row", "tag": "table"}],
        "artifact_counts": {"figures": 1, "tables": 1, "code": 1, "evidence": 1},
        "audit": {
            "counts": {"errors": 0, "warnings": 2},
            "findings": findings,
            "review_decision": {"decision": "approved", "note": "Checked evidence."},
        },
        "reviewed_finding_ids": reviewed,
        "is_demo": False,
    }
    finding_state = wb_page._finding_review_state_summary(state)
    state["audit"]["review_decision"] = {
        "decision": "approved",
        "note": "Checked evidence.",
        "finding_review_signature": finding_state["finding_review_signature"],
        "reviewable_finding_count": finding_state["reviewable_finding_count"],
        "reviewed_finding_count": finding_state["reviewed_finding_count"],
    }

    checks = wb_page._summary_review_checks(state, "en")
    html = wb_page._output_summary_html(state, "en")
    href, filename = wb_page._summary_bundle_index_download(state, "en")
    payload = json.loads(base64.b64decode(href.split(",", 1)[1]).decode("utf-8"))

    assert all(check["ok"] is True for check in checks)
    assert "Reviewer gate ready" in html
    assert "<button>Draft methods + results</button>" not in html
    assert "eu-summary-action-token ready" in html
    assert "Summary export terminal" in html
    assert "run_signed_off_output_bundle.zip" in html
    assert filename == "run_signed_off_bundle_index.json"
    assert all(check["ok"] is True for check in payload["review_checks"])
    assert payload["review_checks"][3]["status"] == "2/2 reviewed"


def test_summary_gate_invalidates_saved_signoff_when_findings_change() -> None:
    old_finding = {"severity": "warning", "validator": "critic", "message": "Check table 1."}
    new_finding = {"severity": "warning", "validator": "critic", "message": "Check figure canvas."}
    old_reviewed = [wb_page._finding_review_id(old_finding)]
    old_state = {
        "audit": {"findings": [old_finding]},
        "reviewed_finding_ids": old_reviewed,
    }
    old_signature = wb_page._finding_review_state_summary(old_state)["finding_review_signature"]
    state = {
        "run_id": "run_stale_signoff",
        "run_dir": "/tmp/run_stale_signoff",
        "source_label": "Real manifest",
        "status": "done",
        "steps": [{"label": "Cohort summary", "status": "ok"}],
        "evidence": [{"label": "row", "tag": "table"}],
        "artifact_counts": {"figures": 1, "tables": 1, "code": 1, "evidence": 1},
        "audit": {
            "counts": {"errors": 0, "warnings": 2},
            "findings": [old_finding, new_finding],
            "review_decision": {
                "decision": "approved",
                "note": "Previous review.",
                "finding_review_signature": old_signature,
            },
        },
        "reviewed_finding_ids": old_reviewed,
        "is_demo": False,
    }

    checks = wb_page._summary_review_checks(state, "en")
    html = wb_page._output_summary_html(state, "en")
    validator_check = next(check for check in checks if check["label"] == "Validator findings reviewed")
    reviewer_check = next(check for check in checks if check["label"] == "Reviewer sign-off")
    review_decisions = wb_page._review_decisions_for_state(state, lang="en")

    assert validator_check["ok"] is False
    assert validator_check["status"] == "1/2 reviewed"
    assert reviewer_check["ok"] is False
    assert reviewer_check["status"] == "refresh required"
    assert review_decisions[0]["label"] == "Saved sign-off needs refresh"
    assert review_decisions[0]["state"] == "warning"
    assert "2 / 5 checks" not in html
    assert "2 review checks outstanding" in html
    assert "eu-summary-action-token disabled\">Draft methods + results" in html


def test_summary_review_decision_writer_uses_shared_schema(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_20260530T010101_test"
    payload = wb_page._write_summary_review_decision(
        run_dir,
        decision="approved",
        note="Summary gate reviewed.",
        run_id="run_signed_off",
        finding_review_state={
            "finding_review_signature": "abc123",
            "reviewable_finding_count": 2,
            "reviewed_finding_count": 2,
            "warning_finding_count": 2,
            "reviewed_warning_count": 2,
            "error_finding_count": 0,
        },
    )
    saved = json.loads((run_dir / "review_decision.json").read_text(encoding="utf-8"))

    assert payload == saved
    assert saved["decision"] == "approved"
    assert saved["note"] == "Summary gate reviewed."
    assert saved["run_id"] == "run_signed_off"
    assert saved["source"] == "easyicu_web_research_agent"
    assert saved["updated_at"]
    assert saved["finding_review_signature"] == "abc123"
    assert saved["reviewable_finding_count"] == 2
    assert saved["reviewed_finding_count"] == 2


def test_summary_gate_keeps_error_findings_blocked_even_if_reviewed() -> None:
    finding = {"severity": "error", "validator": "stat", "message": "Numeric claim mismatch."}
    state = {
        "steps": [{"label": "Cohort summary", "status": "ok"}],
        "evidence": [{"label": "row", "tag": "table"}],
        "artifact_counts": {"figures": 1, "tables": 1, "code": 1, "evidence": 1},
        "audit": {
            "counts": {"errors": 1, "warnings": 0},
            "findings": [finding],
            "review_decision": {},
        },
        "reviewed_finding_ids": [wb_page._finding_review_id(finding)],
        "is_demo": False,
    }

    validator_check = next(
        check
        for check in wb_page._summary_review_checks(state, "en")
        if check["label"] == "Validator findings reviewed"
    )

    assert validator_check["ok"] is False
    assert validator_check["status"] == "error unresolved"


def test_workbench_uses_claude_reference_overview_before_detail_inspector() -> None:
    source = Path(wb_page.__file__).read_text(encoding="utf-8")
    css = Path(wb_page.__file__).with_name("shell_overrides.css").read_text(encoding="utf-8")
    workbench_source = source[
        source.index("def render_agent_workbench"):
        source.index("def render_agent_live_workbench")
    ]
    overview_source = source[
        source.index("def _agent_reference_workbench_html"):
        source.index("def _review_gate_actions_from_audit")
    ]
    primary_actions_source = source[
        source.index("def _render_workbench_primary_actions"):
        source.index("# ---------------------------------------------------------------------", source.index("def _render_workbench_primary_actions"))
    ]

    assert "_agent_reference_workbench_html(state, lang)" in workbench_source
    assert 'state["reviewed_finding_ids"] = sorted(_sync_reviewed_findings_to_session(state))' in workbench_source
    assert 'st.session_state["_active_main_page"] = "research_agent"' in primary_actions_source
    assert "EasyICU Research Agent" in workbench_source
    assert "An auditable, evidence-bound workflow" in workbench_source
    assert "Full audit trail" in workbench_source
    assert "details_expanded = bool(st.session_state.get(_REVIEW_DETAILS_EXPANDED_KEY))" in workbench_source
    assert "_render_workbench_technical_details_gate(lang)" in workbench_source
    assert "if not details_expanded:" in workbench_source
    assert "expanded=True" in workbench_source
    assert "_render_workbench_primary_actions(state, active_state, lang)" in workbench_source
    assert workbench_source.index("_render_workbench_primary_actions(state, active_state, lang)") < workbench_source.index("_render_workbench_technical_details_gate(lang)")
    assert workbench_source.index("_render_workbench_technical_details_gate(lang)") < workbench_source.index("_step_review_html(active_state, state, lang)")
    assert "_step_review_html(active_state, state, lang)" in workbench_source
    assert "Advanced step inspector" in workbench_source
    assert "key=_DETAIL_ADVANCED_KEY" in workbench_source
    assert "_render_code_panel_tabs(active_state, state, lang, include_review=False)" in workbench_source
    assert workbench_source.index("_step_review_html(active_state, state, lang)") < workbench_source.index("Advanced step inspector")
    assert workbench_source.index("Advanced step inspector") < workbench_source.index("_render_code_panel_tabs(active_state, state, lang, include_review=False)")
    assert "Task map" in source
    assert "Evidence checks" in source
    assert "Analysis outputs" not in overview_source
    assert "Run map" in source
    assert "steps ·" in source
    assert "Task map, evidence ledger, and generated outputs" not in source
    assert "Findings · review before drafting" in source
    assert 'st.session_state["_eu_wb_action_panel"] = "evidence"' in primary_actions_source
    assert "Evidence snapshot" in source
    assert "eu-wb-snapshot-list" in source
    assert ".eu-wb-snapshot-list" in css
    assert "grid-template-columns: repeat(4, minmax(0, 1fr));" in css
    assert ".eu-wb-snapshot-list {\n    grid-template-columns: 1fr;" in css
    assert "No findings have been generated in this static preview" in source
    assert "checks clear" in source
    assert "error(s)" in source
    assert "warning(s)" in source
    assert "errors}E/{warnings}W" not in overview_source
    assert ".eu-ref-run-strip" in css
    assert ".eu-ref-split" in css
    assert ".eu-ref-out-grid" in css
    assert ".eu-ref-more-details" in css
    assert 'st-key-_eu_wb_primary_actions' in css
    assert ".eu-ref-note-meta" in css
    assert ".eu-ref-ledger-ico svg" in css
    assert ".stApp .stElementContainer[class*=\"st-key-_eu_ra_view_\"] .stButton > button" in css
    assert "title card first, then icon tabs with an" in css
    assert "border-bottom: 2px solid transparent !important" in css
    assert "border-bottom-color: var(--ink) !important" in css
    assert "background: transparent !important" in css
    assert "color: var(--ink) !important" in css


def test_research_agent_shell_identity_card_precedes_view_tabs() -> None:
    app_source = Path(ra_page.__file__).with_name("app.py").read_text(encoding="utf-8")
    css = Path(wb_page.__file__).with_name("shell_overrides.css").read_text(encoding="utf-8")
    header_source = Path(wb_page.__file__).with_name("cohort_charts.py").read_text(encoding="utf-8")
    agent_branch = app_source[
        app_source.index('elif active_page == "research_agent"'):
        app_source.index("    _handle_sidebar_export_trigger", app_source.index('elif active_page == "research_agent"'))
    ]

    assert 'def _render_research_agent_reference_header(lang: str, *, view: str = "setup")' in app_source
    assert "state_is_demo = bool(state.get(\"is_demo\"))" in app_source
    assert "handoff_setup_ready = _research_agent_handoff_setup_ready(st.session_state)" in app_source
    assert "is_static_guide = (entry_is_demo and not handoff_setup_ready) if state_is_demo is None else state_is_demo" in app_source
    assert 'if view == "history":' in app_source
    assert "local history" in app_source
    assert "eu-ra-reference-header" in app_source
    assert agent_branch.index("_render_research_agent_reference_header(lang, view=_ra_view)") < agent_branch.index("_render_agent_projects_shell(lang, _ra_snapshot, _ra_run_context)")
    assert "_research_agent_active_run_context(st.session_state)" in agent_branch
    assert "_prime_research_agent_header_rerun(st.session_state, run_context)" in app_source
    # View buttons are now subviews inside the Agent Projects shell.
    assert 'key=f"_eu_agent_project_view_{view_key}"' in app_source
    assert '"Workbench" if is_en else "工作台"' in app_source
    assert '"history", "History" if is_en else "历史"' in app_source
    assert 'key="_eu_agent_project_rerun"' in app_source
    assert "render_agent_workbench(lang, show_header=False)" in agent_branch
    assert "render_agent_output_summary(lang, show_header=False)" in agent_branch
    assert "render_research_agent_history_page(lang, show_header=False)" in agent_branch
    assert 'st.session_state["_active_main_page"] = "research_agent"' in agent_branch
    assert "_draft_resume_pending = bool(" in agent_branch
    assert "_handoff_setup_pending = _research_agent_handoff_setup_ready(st.session_state)" in agent_branch
    assert "research_agent_force_manuscript" in agent_branch
    assert "and not _handoff_setup_pending" in agent_branch
    assert "render_research_agent_page(show_header=False)" in agent_branch
    assert "render_research_agent_demo_page(show_header=False)" in agent_branch
    assert "st-key-_eu_ra_tabs" in css
    assert ".eu-design-page-header.eu-ra-reference-header" in css
    assert 'class="eu-design-page-header-row"' in header_source
    assert 'class="eu-design-page-header-actions"' in header_source
    assert ".eu-design-page-header-actions" in css
    assert ".eu-design-page-header.eu-ra-reference-header .eu-design-page-header-actions" in css
    assert 'st.container(key="_eu_ra_header_actions")' not in agent_branch
    assert "_render_agent_projects_shell(lang, _ra_snapshot, _ra_run_context)" in agent_branch
    assert 'key="_eu_agent_project_rerun"' in app_source
    assert ".eu-agent-project-panel" in css
    assert ".eu-agent-project-timeline" in css
    assert "st-key-_eu_agent_project_rerun" in css
    assert "return ('Project guide', '项目导览')" in app_source
    assert '_topbar_type = "secondary"' not in app_source
    assert 'type="primary"' in app_source


def test_workbench_reference_overview_hides_internal_step_metadata() -> None:
    assert wb_page._agent_ref_step_meta({
        "sub": "llm · rc=0 · 9 evidence",
        "status": "ok",
        "evidence_count": 9,
        "repair_count": 1,
    }, "en") == "1 repair logged · 9 evidence"

    assert "rc=0" not in wb_page._agent_ref_step_meta({
        "sub": "llm · rc=0",
        "status": "ok",
    }, "en")


def test_workbench_reference_overview_uses_manifest_evidence_total() -> None:
    state = {
        "steps": [{"label": "Plan", "status": "ok", "evidence_count": 3}],
        "evidence": [{"label": "row"} for _ in range(12)],
        "evidence_total": 122,
        "audit": {"counts": {"errors": 0, "warnings": 15}, "findings": []},
        "summary_outputs": [],
        "run_id": "run_demo",
        "is_demo": False,
    }

    html = wb_page._agent_reference_workbench_html(state, "en")

    assert "122 evidence" in html
    assert "12 evidence" not in html
    assert "Review needed" in html
    assert 'eu-ref-note warn' in html
    assert "Findings · review before drafting" in html
    assert '<details class="eu-ref-more-details">' in html
    assert html.index("Findings · review before drafting") < html.index("Run map")
    assert "Ready for review" not in html
    assert "0E/15W" not in html


def test_workbench_reference_overview_surfaces_finding_review_progress() -> None:
    linked_finding = {"severity": "warning", "validator": "critic", "step_id": "02_model", "message": "Check model card."}
    reviewed_finding = {"severity": "warning", "validator": "critic", "message": "Check table 1."}
    state = {
        "steps": [
            {"label": "Cohort", "step_id": "01_cohort", "status": "ok"},
            {"label": "Model", "step_id": "02_model", "status": "ok"},
        ],
        "evidence": [{"label": "row"}],
        "audit": {
            "counts": {"errors": 0, "warnings": 2},
            "findings": [linked_finding, reviewed_finding],
        },
        "reviewed_finding_ids": [wb_page._finding_review_id(reviewed_finding)],
        "summary_outputs": [],
        "run_id": "run_review_progress",
        "is_demo": False,
    }

    html = wb_page._agent_reference_workbench_html(state, "en")

    assert "1/2 reviewed · 0 error(s) · 2 warning(s)" in html
    assert "1/2 findings reviewed · 1/2 linked to a step" in html
    assert "02 · Model: Check model card." in html
    assert 'eu-ref-pill review">1/2 reviewed' in html


def test_workbench_reference_overview_routes_reviewed_warnings_to_summary() -> None:
    findings = [
        {"severity": "warning", "validator": "critic", "step_id": "02_model", "message": "Check model card."},
        {"severity": "warning", "validator": "critic", "message": "Check table 1."},
    ]
    state = {
        "steps": [
            {"label": "Cohort", "step_id": "01_cohort", "status": "ok"},
            {"label": "Model", "step_id": "02_model", "status": "ok"},
        ],
        "evidence": [{"label": "row"}],
        "audit": {
            "counts": {"errors": 0, "warnings": 2},
            "findings": findings,
        },
        "reviewed_finding_ids": [wb_page._finding_review_id(finding) for finding in findings],
        "summary_outputs": [],
        "run_id": "run_reviewed_warnings",
        "is_demo": False,
    }

    html = wb_page._agent_reference_workbench_html(state, "en")

    assert "Summary sign-off" in html
    assert "Warnings reviewed · Summary sign-off next" in html
    assert "Open Summary to record reviewer sign-off before drafting." in html
    assert 'eu-ref-pill ok">2/2 reviewed' in html
    assert "Review needed" not in html
    assert "Findings · review before drafting" not in html


def test_workbench_reference_overview_status_reflects_blocked_gates() -> None:
    state = {
        "steps": [{"label": "Plan", "status": "ok"}],
        "evidence": [{"label": "row"}],
        "audit": {
            "counts": {"errors": 0, "warnings": 0},
            "gates": [{"label": "numeric verified", "ok": False}],
        },
        "summary_outputs": [],
        "run_id": "run_blocked",
        "is_demo": False,
    }

    html = wb_page._agent_reference_workbench_html(state, "en")

    assert "Gate follow-up" in html
    assert 'eu-ref-pill warn' in html
    assert 'eu-ref-note warn' in html
    assert "Gate follow-up required" in html
    assert "Ready for review" not in html
    assert "Review blocked" not in html


def test_workbench_reference_overview_warnings_take_priority_over_backend_gates() -> None:
    state = {
        "steps": [{"label": "Plan", "status": "ok"}],
        "evidence": [{"label": "row"}],
        "audit": {
            "counts": {"errors": 0, "warnings": 2},
            "gates": [{"label": "publication ready", "ok": False}],
            "findings": [{"severity": "warning", "validator": "critic", "message": "Review lab summary."}],
        },
        "summary_outputs": [],
        "run_id": "run_warning_gate",
        "is_demo": False,
    }

    html = wb_page._agent_reference_workbench_html(state, "en")

    assert "Review needed" in html
    assert "Findings · review before drafting" in html
    assert "Review lab summary." in html
    assert "Review blocked" not in html
    assert "Findings · resolve before drafting" not in html


def test_workbench_review_actions_keep_warning_flow_separate_from_backend_gates() -> None:
    audit = {
        "counts": {"errors": 0, "warnings": 2},
        "gates": [{"label": "publication ready", "ok": False}],
    }

    actions = wb_page._review_gate_actions_from_audit(audit, lang="en")
    decisions = wb_page._review_decisions_from_audit(audit, lang="en")

    assert actions[0]["label"] == "Review warnings"
    assert actions[0]["state"] == "review"
    assert decisions[0]["label"] == "Conditional approve"
    assert decisions[0]["state"] == "selected"
    assert not any(action["label"] == "Resolve audit blockers" for action in actions)


def test_workbench_reference_overview_success_note_is_not_warning() -> None:
    state = {
        "steps": [{"label": "Plan", "status": "ok"}, {"label": "Analyze", "status": "ok"}],
        "evidence": [{"label": "table"}, {"label": "figure"}],
        "audit": {"counts": {"errors": 0, "warnings": 0}, "gates": [{"label": "numeric verified", "ok": True}]},
        "summary_outputs": [],
        "run_id": "run_ready",
        "is_demo": False,
    }

    html = wb_page._agent_reference_workbench_html(state, "en")

    assert "Ready for review" in html
    assert "Evidence gate clear" in html
    assert 'eu-ref-note ok' in html
    assert 'eu-ref-note warn' not in html


def test_demo_agent_copy_matches_backend_scope() -> None:
    source = Path(ra_page.__file__).read_text(encoding="utf-8")
    demo_source = source[
        source.index("def _render_research_agent_demo_visuals"):
        source.index("def render_research_agent_demo_page")
    ]

    # Demo copy now reports backend scope dynamically (stays + review features)
    # instead of a hard-coded "demo · N stays · M modules" string.
    assert "review workspace · " in demo_source
    assert "ICU stays · " in demo_source
    assert "review features" in demo_source
    assert "selected export concepts · " in demo_source
    assert "Demo · no LLM call" in demo_source
    assert "demo · 10 stays · 8 modules" not in demo_source
    assert "gpt-oss · sidebar AI" not in demo_source


def test_workbench_demotes_internal_trace_and_jargon() -> None:
    source = Path(wb_page.__file__).read_text(encoding="utf-8")
    controls_source = source[
        source.index("def _render_process_graph_controls"):
        source.index("def _step_flow_html")
    ]

    assert "Review steps" in controls_source
    assert "Select a step to review checklist, outputs, evidence, and activity." in controls_source
    assert "cols_per_row = 2" in controls_source
    assert "_process_minimap_svg_html" not in controls_source
    assert "Activity notes" in source
    assert "Review timeline" in source
    assert "step timing · review states" in source
    assert "Run trace" not in source
    assert "step timing · audit states" not in source
    assert "error(s) ·" in source
    assert "Pipeline map" not in source
    assert "retry arcs" not in source
    assert "auto-patch" not in source
    assert "PlanAgent lanes" not in source
    assert "State track" not in source


def test_real_figure_result_card_uses_bound_artifact_not_placeholder_chart(tmp_path: Path) -> None:
    fig = tmp_path / "figures" / "roc.svg"
    fig.parent.mkdir()
    fig.write_text(
        '<svg viewBox="0 0 10 10"><path d="M1 9 L5 5 L9 1" /></svg>',
        encoding="utf-8",
    )
    evidence = [{
        "evidence_id": "fig_1",
        "kind": "figure",
        "relative_path": "figures/roc.svg",
        "produced_by_step": "02_plot",
    }]

    cards = _result_cards_from_evidence(evidence, run_dir=tmp_path, lang="en")

    assert cards[0]["metric"] == "rendered"
    assert cards[0]["relative_path"] == "figures/roc.svg"
    assert cards[0]["path"] == "figures/roc.svg"
    assert cards[0]["artifact_path"] == str(fig)
    assert cards[0]["evidence_id"] == "fig_1"
    assert '<svg viewBox="0 0 10 10"' in cards[0]["preview_html"]
    assert "render_tile" not in cards[0]["preview_html"]


def test_workbench_download_empty_state_uses_specific_result_context() -> None:
    source = Path(wb_page.__file__).read_text(encoding="utf-8")

    assert "Results carry no local file paths (demo or unresolved)." not in source
    assert "_result_download_ledger_html" in source
    assert "Result download ledger" in source
    assert "Output bundle terminal" in source
    assert "Demo preview does not write downloadable result files." in source
    assert "No downloadable result files are registered for this selected step." in source
    assert "Registered result paths are not available on disk from this run directory." in source


def test_workbench_timeline_jump_key_tracks_selected_step() -> None:
    source = Path(wb_page.__file__).read_text(encoding="utf-8")
    resolve_source = source[
        source.index("def _resolve_selected_step"):
        source.index("def _state_for_selected_step")
    ]
    timeline_source = source[
        source.index("def _render_timeline_jump"):
        source.index("def _render_audit_actions")
    ]

    assert "st.session_state[key] = selected" in resolve_source
    assert 'key=f"_eu_wb_timeline_jump_{select_key}_{current}"' in timeline_source
    assert 'key=f"_eu_wb_timeline_jump_{select_key}"' not in timeline_source


def test_live_workbench_state_builds_from_progress_events(tmp_path: Path) -> None:
    progress_events = [
        {
            "stage": "run",
            "message": "Starting research-agent run.",
            "status": "running",
            "timestamp": "2026-05-22T00:00:00+00:00",
        },
        {
            "stage": "step",
            "step_id": "01_table_one",
            "message": "Step 1/3 started: 01_table_one.",
            "status": "running",
            "current_step": 1,
            "total_steps": 3,
            "timestamp": "2026-05-22T00:00:01+00:00",
        },
    ]
    state = build_workbench_state_from_manifest(
        tmp_path / "run_pending_webapp",
        {
            "run_id": "run_pending_webapp",
            "research_question": "Short live run",
            "per_step_records": [],
            "evidence": [],
            "findings": [],
        },
        partial=True,
        progress_events=progress_events,
    )
    dag_html = _step_flow_html(state, "en")

    assert state["status"] == "running"
    assert state["steps"][-1]["step_id"] == "01_table_one"
    assert state["steps"][-1]["status"] == "running"
    assert "eu-agent-flow-node running" in dag_html
    assert "Evidence review" in dag_html


def test_workbench_step_labels_explain_status_semantics() -> None:
    assert "review · needs review" in _step_button_label({"label": "Model", "status": "fail", "sub": "ValueError"}, 3, "en")
    assert "queued · reviewable step" in _step_button_label({"label": "Findings", "status": "pending"}, 8, "en")

    legend_html = _step_legend_html("en")
    assert "Done" in legend_html
    assert "Running" in legend_html
    assert "Queued" in legend_html
    assert "Needs fix" in legend_html
    assert "Retrying" in legend_html


def test_step_evidence_links_explicit_ids_and_produced_by_step(tmp_path: Path) -> None:
    manifest = {
        "evidence": [
            {
                "evidence_id": "script_1",
                "kind": "code",
                "produced_by_step": "01_model",
            },
            {
                "evidence_id": "table_1",
                "kind": "table",
                "produced_by_step": "01_model",
            },
            {
                "evidence_id": "figure_2",
                "kind": "figure",
                "produced_by_step": "02_plot",
            },
        ],
    }
    record = {"step_id": "01_model", "evidence_ids": ["script_1"]}

    linked = ra_page._evidence_for_step(record, manifest)

    assert [rec["evidence_id"] for rec in linked] == ["script_1", "table_1"]


def test_workbench_evidence_rows_preserve_paths_for_inspector() -> None:
    rows = wb_page._evidence_rows_from_records(
        [
            {
                "evidence_id": "script_1",
                "kind": "code",
                "description": "Generated analysis script.",
                "relative_path": "evidence/code_analysis.py",
                "sha256": "abcdef1234567890",
            }
        ],
        fallback_label="01_model",
    )

    assert rows[0]["label"] == "Generated analysis script."
    assert rows[0]["relative_path"] == "evidence/code_analysis.py"
    assert rows[0]["path"] == "evidence/code_analysis.py"
    assert rows[0]["sha8"] == "abcdef12"
    assert rows[0]["sha256"] == "abcdef1234567890"
    assert rows[0]["evidence_id"] == "script_1"


def test_workbench_evidence_rows_bind_real_artifact_file_state(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_01"
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    table_path = evidence_dir / "table_one.csv"
    table_path.write_text("stay_id,age\n1,62\n", encoding="utf-8")

    rows = wb_page._evidence_rows_from_records(
        [
            {
                "evidence_id": "table_one",
                "kind": "table",
                "description": "Table 1 export.",
                "relative_path": "evidence/table_one.csv",
                "produced_by_step": "02_table_one",
                "sha256": "1234567890abcdef",
            },
            {
                "evidence_id": "missing_plot",
                "kind": "figure",
                "relative_path": "evidence/missing_plot.png",
            },
        ],
        fallback_label="02_table_one",
        run_dir=run_dir,
    )

    assert rows[0]["file_exists"] is True
    assert rows[0]["file_state"] == "available"
    assert rows[0]["artifact_path"] == str(table_path)
    assert rows[0]["file_name"] == "table_one.csv"
    assert rows[0]["suffix"] == "csv"
    assert rows[0]["size_label"].endswith("B")
    assert rows[0]["produced_by_step"] == "02_table_one"
    assert rows[1]["file_exists"] is False
    assert rows[1]["file_state"] == "missing"


def test_workbench_artifact_action_contract_maps_real_file_states(tmp_path: Path) -> None:
    table_path = tmp_path / "evidence" / "table_one.csv"
    table_path.parent.mkdir()
    table_path.write_text("stay_id,age\n1,62\n", encoding="utf-8")
    record = {
        "evidence_id": "table_one",
        "kind": "table",
        "suffix": "csv",
        "produced_by_step": "02_table_one",
        "sha256": "1234567890abcdef",
    }

    available_html = wb_page._artifact_action_contract_html(
        record,
        raw_path="evidence/table_one.csv",
        target_path=table_path,
        file_state="available",
        lang="en",
    )
    assert "eu-wb-artifact-contract available" in available_html
    assert "Local artifact contract" in available_html
    assert "eu-wb-artifact-tile available" in available_html
    assert "eu-wb-artifact-node" in available_html
    assert ">01<" in available_html
    assert "eu-wb-artifact-copy" in available_html
    assert "eu-wb-artifact-action-strip" in available_html
    assert 'eu-wb-artifact-action enabled">Open file' in available_html
    assert 'eu-wb-artifact-action enabled">Download' in available_html
    assert "available on disk" in available_html
    assert "Open and download enabled" in available_html
    assert "02_table_one" in available_html
    assert "1234567890ab" in available_html

    missing_html = wb_page._artifact_action_contract_html(
        record,
        raw_path="evidence/missing.csv",
        target_path=tmp_path / "evidence" / "missing.csv",
        file_state="missing",
        lang="en",
    )
    assert "eu-wb-artifact-contract missing" in missing_html
    assert "eu-wb-artifact-tile missing" in missing_html
    assert 'eu-wb-artifact-action disabled">Open file' in missing_html
    assert 'eu-wb-artifact-action disabled">Download' in missing_html
    assert 'eu-wb-artifact-action enabled">SHA' in missing_html
    assert 'eu-wb-artifact-action enabled">Copy ID' in missing_html
    assert "registered but missing" in missing_html
    assert "Metadata only until the file is present" in missing_html

    unbound_html = wb_page._artifact_action_contract_html(
        {},
        raw_path="",
        target_path=None,
        file_state="unbound",
        lang="en",
    )
    assert "eu-wb-artifact-contract unbound" in unbound_html
    assert "eu-wb-artifact-action-strip" in unbound_html
    assert 'eu-wb-artifact-action disabled">Open file' in unbound_html
    assert 'eu-wb-artifact-action disabled">Download' in unbound_html
    assert "no file path bound" in unbound_html
    assert "This evidence row has no registered file path." in unbound_html

    terminal_html = wb_page._artifact_action_terminal_html(
        receipt={
            "action": "download",
            "label": "Download",
            "detail": str(table_path),
        },
        file_exists=True,
        has_sha=True,
        has_id=True,
        raw_path="evidence/table_one.csv",
        lang="en",
    )
    assert "eu-wb-artifact-terminal" in terminal_html
    assert "Action terminal" in terminal_html
    assert "Open / download receipt" in terminal_html
    assert "Download: " in terminal_html
    assert "eu-wb-artifact-terminal-row done" in terminal_html
    assert "Requests the local desktop to open" in terminal_html
    assert "Reveals the evidence ID" in terminal_html

    disabled_terminal_html = wb_page._artifact_action_terminal_html(
        receipt=None,
        file_exists=False,
        has_sha=False,
        has_id=False,
        raw_path="",
        lang="en",
    )
    assert "No artifact action has been requested" in disabled_terminal_html
    assert "eu-wb-artifact-terminal-row disabled" in disabled_terminal_html
    assert "Download unlocks when the local file exists" in disabled_terminal_html


def test_workbench_result_download_ledger_maps_file_states() -> None:
    rows = [
        {
            "idx": 1,
            "title": "Primary table",
            "file_name": "table_one.csv",
            "meta": "csv · 4.2 KB",
            "file_state": "available",
            "state_label": "ready to download",
            "action": "download_1",
        },
        {
            "idx": 2,
            "title": "Forest plot",
            "file_name": "forest.svg",
            "meta": "figure",
            "file_state": "missing",
            "state_label": "missing on disk",
            "action": "download_2",
        },
        {
            "idx": 3,
            "title": "Unbound result",
            "file_name": "No file path registered",
            "meta": "artifact",
            "file_state": "unbound",
            "state_label": "metadata only",
            "action": "download_3",
        },
    ]

    html = wb_page._result_download_ledger_html(
        rows,
        lang="en",
        receipt={"action": "download_1", "label": "Download 01", "detail": "table_one.csv"},
    )

    assert "eu-wb-result-download-ledger" in html
    assert "Output bundle terminal" in html
    assert "Result download ledger" in html
    assert "1 ready / 1 missing" in html
    assert "eu-wb-result-download-row available done" in html
    assert "eu-wb-result-download-row missing" in html
    assert "eu-wb-result-download-row unbound" in html
    assert "Download 01: table_one.csv" in html
    assert ">01<" in html
    assert "ready to download" in html


def test_workbench_evidence_ledger_uses_project_artifact_inspector_surface() -> None:
    source = Path(wb_page.__file__).read_text(encoding="utf-8")
    css = Path(wb_page.__file__).with_name("shell_overrides.css").read_text(encoding="utf-8")
    drilldown_source = source[
        source.index("def _render_evidence_drilldown"):
        source.index("def _render_result_downloads")
    ]

    assert "eu-wb-evidence-ledger" in source
    assert "eu-wb-evidence-row" in source
    assert "eu-wb-evidence-inspector" in drilldown_source
    assert "_artifact_action_contract_html" in drilldown_source
    assert "_artifact_action_terminal_html" in drilldown_source
    assert "_remember_artifact_action_receipt" in drilldown_source
    assert "eu-wb-artifact-contract" in source
    assert "_artifact_file_meta" in source
    assert "_artifact_mime(target_path)" in drilldown_source
    assert "st.download_button" in drilldown_source
    assert "on_click=_remember_artifact_action_receipt" in drilldown_source
    assert ".eu-wb-evidence-ledger" in css
    assert ".eu-wb-evidence-inspector" in css
    assert ".eu-wb-evidence-state.available" in css
    assert ".eu-wb-artifact-contract" in css
    assert ".eu-wb-artifact-grid" in css
    assert ".eu-wb-artifact-node" in css
    assert ".eu-wb-artifact-copy" in css
    assert ".eu-wb-artifact-action-strip" in css
    assert ".eu-wb-artifact-terminal" in css
    assert ".eu-wb-artifact-terminal-row.done" in css
    assert ".eu-wb-result-download-ledger" in css
    assert ".eu-wb-result-download-row.available" in css
    assert 'key=f"_eu_wb_result_download_' in source
    assert "on_click=_remember_artifact_action_receipt" in source
    artifact_grid_css = css[
        css.index(".eu-wb-artifact-grid {"): css.index(".eu-wb-artifact-tile {")
    ]
    assert "grid-template-columns: 1fr" in artifact_grid_css


def test_step_view_hides_raw_logs_json_and_code_from_result_artifacts() -> None:
    records = [
        {"kind": "table", "relative_path": "evidence/table_one.csv"},
        {"kind": "figure", "relative_path": "evidence/plot.png"},
        {"kind": "log", "relative_path": "evidence/log_run__run.log"},
        {"kind": "code", "relative_path": "evidence/code_analysis.py"},
        {"kind": "statistic", "relative_path": "evidence/step_summary.json"},
    ]

    visible = [rec for rec in records if ra_page._is_user_facing_step_artifact(rec)]
    debug = [rec for rec in records if ra_page._is_debug_artifact(rec)]

    assert [rec["kind"] for rec in visible] == ["table", "figure"]
    assert [rec["kind"] for rec in debug] == ["log", "code", "statistic"]


def test_research_grounding_json_payloads_load_from_registered_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    payload_path = evidence_dir / "analysis_plan__analysis_plan.json"
    payload_path.write_text(
        json.dumps({"steps": [{"step_id": "01_table", "intent": "Summarize cohort."}]}),
        encoding="utf-8",
    )
    manifest = {
        "evidence": [
            {
                "evidence_id": "analysis_plan",
                "kind": "log",
                "relative_path": "evidence/analysis_plan__analysis_plan.json",
            }
        ]
    }

    payload = ra_page._json_payload_for_evidence(run_dir, manifest, "analysis_plan")

    assert payload["steps"][0]["step_id"] == "01_table"


def test_table_preview_reads_only_preview_rows(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    pd.DataFrame({"a": range(100), "b": range(100, 200)}).to_csv(path, index=False)

    preview = ra_page._read_table_preview(path, n=10)

    assert preview.shape == (10, 2)
    assert preview["a"].tolist() == list(range(10))


def test_resolve_llm_disables_qwen3_thinking_for_openai_compatible_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Socket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(ra_page.socket, "create_connection", lambda *_args, **_kwargs: _Socket())

    client = ra_page._resolve_llm(
        {"MockLLMClient": object, "OpenAIClient": FakeOpenAIClient},
        "Custom OpenAI-compatible",
        api_key="vllm",
        model="qwen3-8b",
        base_url="http://127.0.0.1:8000/v1",
    )

    assert client.kwargs["extra_body"]["enable_thinking"] is False
    assert client.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_local_llm_endpoint_health_check_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    def _refused(*_args, **_kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(ra_page.socket, "create_connection", _refused)

    with pytest.raises(RuntimeError, match="Local LLM endpoint is unreachable"):
        ra_page._assert_local_llm_endpoint_reachable("http://127.0.0.1:8787/v1", timeout=0.01)


def test_local_llm_endpoint_health_check_ignores_remote_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unexpected(*_args, **_kwargs):  # pragma: no cover - fails if called
        raise AssertionError("remote endpoints should not be socket-probed")

    monkeypatch.setattr(ra_page.socket, "create_connection", _unexpected)

    ra_page._assert_local_llm_endpoint_reachable("https://api.openai.com/v1", timeout=0.01)


def test_run_pipeline_enables_deterministic_planner_fallback(tmp_path: Path) -> None:
    class FakePipeline:
        init_kwargs = {}

        def __init__(self, **kwargs):
            FakePipeline.init_kwargs = kwargs

        def run(self, **kwargs):
            return {"ok": True, "run_kwargs": kwargs}

    result = ra_page._run_pipeline(
        handles={"ResearchAgentPipeline": FakePipeline},
        cohort=pd.DataFrame({"stay_id": [1]}),
        skill_key=None,
        question="Does lactate predict mortality?",
        target_outcome=None,
        workdir=tmp_path,
        llm=object(),
        disable_icu_context=False,
    )

    assert FakePipeline.init_kwargs["enable_deterministic_planner_fallback"] is True
    assert FakePipeline.init_kwargs["enable_deterministic_code_fallback"] is True
    assert result["ok"] is True


def test_loaded_concepts_handoff_respects_current_patient_ids() -> None:
    loaded = {
        "age": pd.DataFrame({"stay_id": [1, 2, 3], "age": [60, 70, 80]}),
        "hr": pd.DataFrame({
            "stay_id": [1, 1, 2, 3],
            "time": [0, 1, 0, 0],
            "hr": [70, 75, 80, 90],
        }),
    }

    cohort = ra_page._stay_level_from_loaded_concepts(
        loaded,
        id_col="stay_id",
        patient_ids=[1, 3],
    )

    assert cohort is not None
    assert set(cohort["stay_id"]) == {1, 3}
    assert cohort.loc[cohort["stay_id"] == 1, "hr"].iloc[0] == 75


def test_research_agent_idea_workflow_contract_targets_dry_run_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    fake_st.session_state = {"language": "en"}
    monkeypatch.setattr(ra_page, "st", fake_st)

    examples = ra_page._request_examples()
    contract = ra_page._build_execution_preflight_contract(
        free_question="Explore review-derived ICU ideas.",
        target_outcome="",
        cohort=pd.DataFrame({"stay_id": [1], "lact": [1.2], "death": [0]}),
        cohort_label="synthetic web cohort",
        llm_choice="MockLLMClient (offline, deterministic)",
        model="",
        workdir_text=str(tmp_path),
        stop_after_analysis=True,
        force_manuscript=False,
        template_key="idea_exploration",
        workflow_mode="Idea exploration",
    )

    assert examples[0]["key"] == "idea_exploration"
    assert contract["workflow_mode"] == "idea_exploration"
    assert contract["mode"] == "idea_exploration_dry_run"
    assert any(
        "source_snapshot_manifest.json" in target
        for target in contract["write_targets"]
    )
    assert any(
        "candidate_triage_report.json" in target
        for target in contract["write_targets"]
    )
    assert any("idea_registry.json" in target for target in contract["write_targets"])
    assert any("discovery_report.md" in target for target in contract["write_targets"])
    assert not any(
        "manuscript_scaffold" in target for target in contract["write_targets"]
    )


def test_idea_exploration_panel_keeps_core_triage_first() -> None:
    source = inspect.getsource(ra_page._render_idea_exploration_core_fields)
    panel_source = inspect.getsource(ra_page._render_idea_exploration_panel)
    picker_source = inspect.getsource(ra_page._section_request_picker)

    core_idx = source.index("Core idea triage")
    advanced_idx = source.index("Advanced source and mapping")
    result_idx = panel_source.index("Latest idea triage")
    details_idx = panel_source.index("Review triage details")
    prior_art_idx = panel_source.index("Prior art")
    broad_exact_idx = panel_source.index("Broad/exact")

    assert source.index("Review/editorial excerpt") > core_idx
    assert source.index("Traceable source quote") > core_idx
    assert source.index("Exposure / predictor") > core_idx
    assert source.index("Outcome") > core_idx
    assert source.index("Exposure / predictor") < advanced_idx
    assert source.index("Outcome") < advanced_idx
    assert source.index("Citation key") > advanced_idx
    assert source.index("Analysis family") > advanced_idx
    assert source.index("Aggregation") > advanced_idx
    assert 'key="research_agent_idea_show_advanced_mapping"' in source
    assert "_render_idea_exploration_core_fields(is_en=is_en)" in picker_source
    assert "return (" in picker_source[picker_source.index("_render_idea_exploration_core_fields(is_en=is_en)"):]
    assert "result = _run_idea_exploration_dry_run(" in panel_source
    assert "Stage top candidate as analysis question" in panel_source
    assert panel_source.index("Stage top candidate as analysis question") < details_idx
    assert result_idx < details_idx
    assert prior_art_idx > result_idx
    assert prior_art_idx < details_idx
    assert broad_exact_idx < details_idx
    assert "Prior-art query records" in panel_source
    assert "_idea_prior_art_rows(result)" in panel_source
    assert 'key="research_agent_idea_show_triage_details"' in panel_source
    assert "st.dataframe(pd.DataFrame(candidate_rows)" in panel_source
    assert "st.dataframe(pd.DataFrame(prior_art_rows)" in panel_source
    assert panel_source.index("st.dataframe(pd.DataFrame(candidate_rows)") > details_idx
    assert panel_source.index("st.dataframe(pd.DataFrame(prior_art_rows)") > details_idx
    assert "show_triage_details = st.toggle(" in panel_source
    assert "if show_triage_details:" in panel_source


def test_web_idea_exploration_dry_run_uses_backend_registry_without_pipeline(
    tmp_path: Path,
) -> None:
    cohort = pd.DataFrame({
        "stay_id": [1, 2, 3],
        "lact": [1.2, None, 2.4],
        "death": [0, 1, 0],
    })
    form = {
        "citation_key": "neutral_review_2026",
        "title": "Review of ICU research directions",
        "year": "2026",
        "venue": "Critical Care Review",
        "source_text": (
            "The review highlights lactate and mortality as an unresolved ICU "
            "research direction."
        ),
        "population": "adult ICU patients",
        "predictor": "lactate",
        "outcome": "mortality",
        "rationale": "The source describes this candidate as open.",
        "source_quote": "lactate and mortality",
        "analysis_family": "association",
        "time_window_hint": "early ICU stay",
        "aggregation_hint": "stay-level summary",
    }

    result = ra_page._run_idea_exploration_dry_run(
        form=form,
        cohort=cohort,
        cohort_label="web test cohort",
        workdir_text=str(tmp_path),
    )

    out_dir = Path(result.triage_report_path).parent
    triage = json.loads(Path(result.triage_report_path).read_text(encoding="utf-8"))
    manifest_text = Path(result.manifest_path).read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    registry = json.loads(Path(result.registry_path).read_text(encoding="utf-8"))
    discovery_report = Path(str(result.discovery_report_path))

    assert result.yield_report.n_literature_ideas == 1
    assert result.yield_report.n_executable == 1
    assert result.candidate_records[0].registry_selection_status == "proposed"
    assert result.feasibility_signals[0].source == "webapp_active_cohort_pairwise_probe"
    assert result.feasibility_signals[0].n_joint_complete == 2
    summary = ra_page._idea_result_summary(result)
    assert summary["prior_art_assessments"] == 1
    assert summary["prior_art_label"] == "apparently_gap"
    assert summary["prior_art_broad_hits"] == 0
    assert summary["prior_art_exact_hits"] == 0
    assert summary["prior_art_screen_status"] == "no_hits_to_screen"
    prior_rows = ra_page._idea_prior_art_rows(result)
    assert [row["query_type"] for row in prior_rows] == ["broad", "exact"]
    assert "lactate[Title/Abstract]" in prior_rows[0]["query"]
    assert "mortality[Title/Abstract]" in prior_rows[0]["query"]
    assert triage["schema_version"] == "easyicu.idea_mining_dry_run/1"
    assert triage["candidate_records"][0]["registry_selection_status"] == "proposed"
    assert registry["entries"][0]["selection_status"] == "proposed"
    assert manifest["items"][0]["source_text_stored"] is False
    assert "lactate and mortality as an unresolved" not in manifest_text
    assert discovery_report.exists()
    assert not (out_dir / "manifest.json").exists()
    assert not (out_dir / "manuscript_scaffold_bound.md").exists()


def test_idea_candidate_handoff_sets_analysis_question_and_resets_gate(
    tmp_path: Path,
) -> None:
    cohort = pd.DataFrame({
        "stay_id": [1, 2, 3],
        "lact": [1.2, None, 2.4],
        "death": [0, 1, 0],
    })
    result = ra_page._run_idea_exploration_dry_run(
        form={
            "citation_key": "neutral_review_2026",
            "title": "Review of ICU research directions",
            "year": "2026",
            "venue": "Critical Care Review",
            "source_text": (
                "The review highlights lactate and mortality as an unresolved ICU "
                "research direction."
            ),
            "population": "adult ICU patients",
            "predictor": "lactate",
            "outcome": "mortality",
            "rationale": "The source describes this candidate as open.",
            "source_quote": "lactate and mortality",
            "analysis_family": "association",
            "time_window_hint": "early ICU stay",
            "aggregation_hint": "stay-level summary",
        },
        cohort=cohort,
        cohort_label="web test cohort",
        workdir_text=str(tmp_path),
    )
    state: dict[str, object] = {
        "research_agent_workflow_mode": "Idea exploration",
        "research_agent_preflight_confirmed": True,
        "research_agent_preflight_signature": "stale",
    }

    assert ra_page._stage_idea_candidate_as_research_question(
        state,
        result,
        is_en=True,
    )

    assert "lactate" in str(state["research_agent_question"])
    assert "mortality" in str(state["research_agent_question"])
    assert state["research_agent_target_outcome"] == "death"
    assert state["research_agent_workflow_mode"] == "analysis_run"
    assert "research_agent_workflow_mode_pick" not in state
    assert state["research_agent_preflight_confirmed"] is False
    assert "research_agent_preflight_signature" not in state
    assert state["_research_agent_idea_handoff_candidate_id"]


def test_idea_candidate_ui_handoff_queues_until_before_question_widget(
    tmp_path: Path,
) -> None:
    cohort = pd.DataFrame({
        "stay_id": [1, 2, 3],
        "lact": [1.2, None, 2.4],
        "death": [0, 1, 0],
    })
    result = ra_page._run_idea_exploration_dry_run(
        form={
            "citation_key": "neutral_review_2026",
            "title": "Review of ICU research directions",
            "year": "2026",
            "venue": "Critical Care Review",
            "source_text": (
                "The review highlights lactate and mortality as an unresolved ICU "
                "research direction."
            ),
            "population": "adult ICU patients",
            "predictor": "lactate",
            "outcome": "mortality",
            "rationale": "The source describes this candidate as open.",
            "source_quote": "lactate and mortality",
            "analysis_family": "association",
            "time_window_hint": "early ICU stay",
            "aggregation_hint": "stay-level summary",
        },
        cohort=cohort,
        cohort_label="web test cohort",
        workdir_text=str(tmp_path),
    )
    state: dict[str, object] = {
        "research_agent_question": "",
        "research_agent_workflow_mode": "Idea exploration",
        "research_agent_preflight_confirmed": True,
        "research_agent_preflight_signature": "stale",
    }

    assert ra_page._queue_idea_candidate_as_research_question(state, result)
    assert state["research_agent_question"] == ""
    assert "_research_agent_pending_idea_handoff" in state

    assert ra_page._apply_pending_idea_candidate_handoff(state, is_en=True)

    assert "lactate" in str(state["research_agent_question"])
    assert state["research_agent_target_outcome"] == "death"
    assert state["research_agent_workflow_mode"] == "analysis_run"
    assert "research_agent_preflight_signature" not in state
    assert "_research_agent_pending_idea_handoff" not in state
