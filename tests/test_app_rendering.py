from __future__ import annotations

import os
import sys
import types

import easyicu
import easyicu.webapp.app as app
import pandas as pd
import pytest


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


def test_plotly_compat_keeps_plotly_specific_width_api_untouched(monkeypatch) -> None:
    class _PlotlyStub:
        def __init__(self) -> None:
            self.kwargs = None

        def plotly_chart(self, _figure, **kwargs):
            self.kwargs = kwargs
            return None

    streamlit_stub = _PlotlyStub()
    monkeypatch.setattr(app, "st", streamlit_stub)

    app._plotly_chart_compat("figure", use_container_width=True, config={"displaylogo": False})

    assert streamlit_stub.kwargs == {"use_container_width": True, "config": {"displaylogo": False}}


def test_quick_visualization_demo_loads_after_entry_selection(tmp_path, monkeypatch) -> None:
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    at = streamlit_testing.AppTest.from_file(app.__file__)
    at.session_state["entry_lang_select"] = "EN"
    at.session_state["language"] = "en"
    at.run(timeout=60)
    at.button[0].click().run(timeout=60)
    at.button(key="viz_load_demo").click().run(timeout=60)

    assert any("Data Ready" in markdown.value for markdown in at.markdown)


def test_real_data_mode_requires_data_path_before_validation(tmp_path, monkeypatch) -> None:
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    at = streamlit_testing.AppTest.from_file(app.__file__)
    at.session_state["entry_lang_select"] = "EN"
    at.session_state["language"] = "en"
    at.run(timeout=60)
    at.button(key="entry_real_btn").click().run(timeout=60)
    at.button(key="validate_path").click().run(timeout=60)

    assert any(error.value == "❌ Please enter data path" for error in at.error)


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

    assert english["title"] == "📋 Module Table Preview"
    assert "Preview loaded tables by module" in english["description"]
    assert chinese["title"] == "📋 模块数据预览"


def test_single_feature_preview_copy_matches_preview_style() -> None:
    english = app._get_single_feature_preview_copy("sofa", "en")
    chinese = app._get_single_feature_preview_copy("sofa", "zh")

    assert english["title"] == "🧪 Single Feature Preview"
    assert "Inspect `sofa`" in english["description"]
    assert chinese["title"] == "🧪 单特征预览"


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
    assert app._normalize_figure_target("cross-db") == ("cohort", "Cross-DB Benchmark")
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
    assert len(state["grp_demographics"]) == 25
    assert not state["dash_demographics"].empty
    assert set(state["multidb_data"].keys()) == {"miiv", "eicu", "aumc", "hirid", "mimic", "sic"}
    assert state["multidb_concepts"][:4] == ["hr", "sbp", "dbp", "map"]


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
    monkeypatch.setattr(app, "_default_real_database", lambda: "miiv")
    monkeypatch.setattr(app, "_default_real_data_root", lambda: str(tmp_path))
    monkeypatch.setattr(app, "find_database_path", lambda data_path, _database: data_path)

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


def test_apply_screenshot_mode_ui_state_closes_floating_ai_and_clears_ai_jump() -> None:
    state = {
        "_floating_ai_open": True,
        "_scroll_to_tab": "ai_assistant",
    }

    app._apply_screenshot_mode_ui_state(state)

    assert state["_floating_ai_open"] is False
    assert "_scroll_to_tab" not in state


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


def test_compute_smd_continuous_uses_pooled_standard_deviation() -> None:
    smd = app._compute_smd_continuous(pd.Series([1, 2, 3]), pd.Series([2, 3, 4]))

    assert smd == pytest.approx(-1.0)


def test_compute_smd_binary_uses_pooled_proportion() -> None:
    smd = app._compute_smd_binary(pd.Series([1, 1, 0, 0]), pd.Series([1, 0, 0, 0]))

    assert smd == pytest.approx(0.5164, abs=1e-4)


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
