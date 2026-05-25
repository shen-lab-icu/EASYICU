from __future__ import annotations

import os
import json
import re
import sys
import types
from pathlib import Path

import easyicu
from easyicu.concept import load_dictionary
import easyicu.webapp.app as app
import easyicu.webapp.concept_catalog as concept_catalog
import easyicu.webapp.cohort_dashboard_page as cohort_dashboard_page
import easyicu.webapp.cohort_filters as cohort_filters
import easyicu.webapp.cohort_group_page as cohort_group_page
import easyicu.webapp.cohort_redesign as cohort_redesign
import easyicu.webapp.cohort_severity_page as cohort_severity_page
import easyicu.webapp.cohort_workspace as cohort_workspace
import easyicu.webapp.demo_data as demo_data
import easyicu.webapp.data_paths as data_paths
import easyicu.webapp.data_workflows as data_workflows
import easyicu.webapp.export_reports as export_reports
import easyicu.webapp.export_workflow as export_workflow
import easyicu.webapp.i18n as i18n
import easyicu.webapp.page_header as page_header
import easyicu.webapp.pages_redesign as pages_redesign
import easyicu.webapp.patient_page as patient_page
import easyicu.webapp.quality_page as quality_page
import easyicu.webapp.research_agent as research_agent
import easyicu.webapp.sidebar as sidebar
import easyicu.webapp.shell_styles as shell_styles
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


class _FakeColumn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


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

    assert '[theme]' in theme_config
    assert 'base = "light"' in theme_config


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
    assert "Lightweight review dataset opens immediately" in source_text
    assert "Use local data folder" in source_text
    assert "try first" in source_text
    assert "_eu_entry_lang_toggle" in source_text
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
    assert "Export handoff" in source_text
    assert "repeat(4, minmax(0, 1fr))" in css_text
    assert "st-key-eu_entry_topbar_shell" in css_text
    assert "st-key-_eu_entry_lang_toggle" in css_text
    assert "st-key-eu_entry_code_row" in css_text
    assert "margin-top: 68px" in css_text
    assert ".eu-entry-step::before" in css_text
    assert "st-key-_eu_entry_demo" in css_text
    assert "st-key-_eu_entry_real" in css_text
    assert "st-key-_eu_entry_nodata" in css_text
    assert "margin-top: -49px" not in css_text


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
    with open(research_agent.__file__, encoding="utf-8") as handle:
        research_agent_source = handle.read()
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
    assert "### 👥" not in cohort_group_source
    assert "#### 🔀" not in cohort_group_source
    assert "👤 Demographics" not in cohort_group_source
    assert "💀 Survived" not in cohort_group_source
    assert "_render_section_heading" in cohort_group_source
    assert "eu-native-section-heading" in cohort_group_source
    assert "### 🎯" not in cohort_dashboard_source
    assert "📊 Cohort Snapshot Summary" not in cohort_dashboard_source
    assert "Open the **SOFA Δ** tab for the matrix" in cohort_dashboard_source
    assert "SOFA-1 vs SOFA-2 Reclassification" not in cohort_dashboard_source
    assert "dash_reclass_matrix" not in cohort_dashboard_source
    assert "dash_reclass_organ_contrib" not in cohort_dashboard_source
    assert "st.columns(6)" not in cohort_dashboard_source
    assert "_style_readout_figure" in cohort_dashboard_source
    assert "Clinical phenotype prevalence" in cohort_dashboard_source
    assert "reclass_matrix" in cohort_severity_source
    assert "reclass_organ_contrib" in cohort_severity_source
    assert "_style_reclass_figure" in cohort_severity_source
    assert "Reclassification matrix" in cohort_severity_source
    assert "st.columns(5)" not in sofa_reclassification_source
    assert "eu-cohort-kpi-grid" in sofa_reclassification_source
    assert "### 🧭" not in cohort_severity_source
    assert "_render_section_heading" in cohort_dashboard_source
    assert "_render_section_heading" in cohort_severity_source
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
    assert 'icon="🧪"' not in research_agent_source
    assert 'st.button(f"🤖 {label}"' not in app_source
    assert ".eu-native-section-heading" in css_text
    assert ".eu-cohort-kpi-grid" in css_text
    assert ".eu-chart-heading" in css_text
    assert "grid-template-columns: repeat(auto-fit" in css_text
    assert '[data-testid="stMetric"]' in css_text
    assert "font-size: 16px" in css_text
    assert '[data-baseweb="tag"]' in css_text


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


def test_tutorial_surfaces_existing_data_dictionary_preview() -> None:
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
    source_text = "\n".join(
        str(value)
        for value in (
            pages_redesign.render_tutorial_redesign_page.__code__.co_consts
            + pages_redesign._render_tutorial_dictionary.__code__.co_consts
        )
        if isinstance(value, str)
    )

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
    assert "_eu_tutorial_dict_module" in source_text
    assert "_eu_tutorial_dict_feature" not in source_text
    assert "Selected concept" not in source_text
    assert "_eu_tutorial_dictionary" in source_text
    assert "Open selected module -> Concepts" in source_text
    assert "src/easyicu/data/concept-dict.json" in source_text
    assert "st-key-eu_tutorial_dictionary_panel" in css_text
    assert 'content: "⌄"' in css_text
    assert "width: 40px" in css_text
    assert "font-size: 22px" in css_text
    assert "rgba(216, 246, 248" not in css_text
    assert ".eu-dict-module-heading" in css_text


def test_quick_visualization_demo_entry_autoloads_light_review_workspace(tmp_path, monkeypatch) -> None:
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    at = streamlit_testing.AppTest.from_file(app.__file__)
    at.session_state["entry_lang_select"] = "EN"
    at.session_state["language"] = "en"
    at.run(timeout=60)
    at.button(key="_eu_entry_demo").click().run(timeout=60)
    assert at.session_state["_active_main_page"] == "quick_viz"
    assert at.session_state["loaded_data_origin"] == "demo_viz"
    assert at.session_state["mock_params"]["demo_profile"] == "lite"
    assert 0 < len(at.session_state["loaded_concepts"]) < 90
    assert len(at.session_state["patient_ids"]) == 50
    assert "viz_load_demo" not in {button.key for button in at.button}
    markdown_text = " ".join(getattr(markdown, "value", "") for markdown in at.markdown)
    assert "Generating lightweight demo data" not in markdown_text
    warning_text = " ".join(getattr(warning, "value", "") for warning in at.warning)
    assert "Dashboard rendering failed" not in warning_text
    assert "time_candidates" not in warning_text


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
    assert len(at.session_state["patient_ids"]) == 50
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

    assert english["title"] == "Module Table Preview"
    assert "Preview loaded tables by module" in english["description"]
    assert chinese["title"] == "模块数据预览"


def test_quick_viz_data_table_overlap_guards_are_in_shell_css() -> None:
    css = shell_styles._load_shell_overrides_css()

    assert ".dt-page-head" in css
    assert "st-key-dt_preview_mode" in css
    assert "st-key-dt_preview_summary" in css


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
    assert len(state["grp_demographics"]) == 25
    assert not state["dash_demographics"].empty
    assert set(state["multidb_data"].keys()) == {"miiv", "eicu", "aumc", "hirid", "mimic", "sic"}
    assert state["multidb_concepts"][:4] == ["hr", "sbp", "dbp", "map"]


def test_lightweight_demo_data_keeps_review_surface_small() -> None:
    data, patient_ids = demo_data.generate_lightweight_demo_data(n_patients=50, hours=48)

    assert len(patient_ids) == 50
    assert 40 <= len(data) < 90
    assert {"hr", "map", "sofa2", "sep3_sofa2", "aki", "death", "los_icu"} <= set(data)
    assert sum(len(frame) for frame in data.values()) < 50_000


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
            self.session_state = {"language": "en", "entry_mode": "real"}
            self.tabs_labels: list[str] = []

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

        def tabs(self, labels):
            self.tabs_labels = labels
            return [_FakePanel() for _ in labels]

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

    assert streamlit_stub.tabs_labels == [
        "Groups",
        "Coverage",
        "Snapshot",
        "SOFA Δ",
    ]
    assert rendered_panels == ["groups", "coverage", "snapshot", "sofa"]


def test_real_cohort_page_gates_sub_tabs_when_no_data_path_validated(monkeypatch) -> None:
    class _FakePanel:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeStreamlit:
        def __init__(self) -> None:
            self.session_state = {"language": "en", "entry_mode": "real"}
            self.tabs_called = False

        def markdown(self, *_args, **_kwargs) -> None:
            pass

        def warning(self, *_args, **_kwargs) -> None:
            pass

        def info(self, *_args, **_kwargs) -> None:
            pass

        def columns(self, spec):
            return [_FakePanel() for _ in spec]

        def tabs(self, labels):
            self.tabs_called = True
            return [_FakePanel() for _ in labels]

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

    assert streamlit_stub.tabs_called is False
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


def test_open_embedded_ai_assistant_targets_main_workspace_panel() -> None:
    state: dict[str, object] = {"llm_enabled": False, "_floating_ai_open": True}

    app._open_embedded_ai_assistant(state, "How should I configure SOFA?")

    assert state["llm_enabled"] is True
    assert state["_llm_toggle"] is True
    assert state["_inline_ai_panel_open"] is True
    assert state["_sidebar_ai_open"] is False
    assert state["_floating_ai_open"] is False
    assert state["_ai_pending_question"] == "How should I configure SOFA?"


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


def test_topbar_crossdb_real_action_opens_loader_when_data_missing() -> None:
    state = {
        "_eu_topbar_run_request": {"page": "cross_db"},
        "entry_mode": "real",
    }

    result = app._consume_topbar_run_request(state, "cross_db", "en")

    assert result["level"] == "warning"
    assert state["_eu_crossdb_advanced_open"] is True
    assert "_eu_topbar_run_request" not in state


def test_crossdb_page_does_not_render_advanced_loader_until_opened(monkeypatch) -> None:
    class _FakePanel:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeStreamlit:
        def __init__(self, session_state) -> None:
            self.session_state = session_state
            self.button_labels: list[str] = []

        def markdown(self, *_args, **_kwargs) -> None:
            pass

        def button(self, label, **_kwargs) -> bool:
            self.button_labels.append(label)
            return False

        def expander(self, *_args, **_kwargs):
            return _FakePanel()

        def rerun(self) -> None:
            raise AssertionError("rerun should not be called without a click")

    rendered: list[str] = []
    streamlit_stub = _FakeStreamlit({"language": "en", "entry_mode": "demo"})
    monkeypatch.setattr(cohort_redesign, "st", streamlit_stub)

    cohort_redesign.render_cross_db_redesign_page(
        "en",
        multidb_fn=lambda _lang: rendered.append("loader"),
    )

    assert rendered == []
    assert "Open detailed loader" in streamlit_stub.button_labels

    streamlit_stub = _FakeStreamlit({
        "language": "en",
        "entry_mode": "demo",
        "_eu_crossdb_advanced_open": True,
    })
    monkeypatch.setattr(cohort_redesign, "st", streamlit_stub)

    cohort_redesign.render_cross_db_redesign_page(
        "en",
        multidb_fn=lambda _lang: rendered.append("loader"),
    )

    assert rendered == ["loader"]


def test_cohort_redesign_defaults_to_real_panel_body(monkeypatch) -> None:
    class _FakeStreamlit:
        def __init__(self) -> None:
            self.session_state = {
                "entry_mode": "demo",
                "cohort_active_panel": "coverage",
            }
            self.markdown_calls: list[str] = []

        def markdown(self, body, *_args, **_kwargs) -> None:
            self.markdown_calls.append(str(body))

        def radio(self, _label, *, options, key, **_kwargs):
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

    assert rendered == ["coverage"]


def test_cohort_redesign_shell_only_keeps_design_preview_available(monkeypatch) -> None:
    class _FakeStreamlit:
        def __init__(self) -> None:
            self.session_state = {
                "entry_mode": "demo",
                "cohort_active_panel": "groups",
                "_eu_shell_only": True,
            }
            self.markdown_calls: list[str] = []

        def markdown(self, body, *_args, **_kwargs) -> None:
            self.markdown_calls.append(str(body))

        def radio(self, _label, *, options, key, **_kwargs):
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

    assert rendered == []
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


def test_sidebar_session_summary_returns_html_without_none_leak(monkeypatch) -> None:
    streamlit_stub = _SessionStateStreamlit({
        "mock_params": {"n_patients": 42},
        "demo_mode_patients": 42,
    })
    monkeypatch.setattr(sidebar, "st", streamlit_stub)

    html = sidebar._session_summary_html("demo", "en")

    assert html.startswith('<div class="eu-session-card">')
    assert "Ready demo cohort · 42 patients" in html
    assert "simulated" in html
    assert ">None<" not in html
    assert "not an active project" not in html


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
        "tutorial",
        "quick_viz",
        "cohort",
        "cross_db",
        "research_agent",
    ]
    assert nav_by_key["tutorial"].label == "Data Extraction"
    assert nav_by_key["quick_viz"].label == "Patient Review"
    assert nav_by_key["quick_viz"].level == "child"
    assert nav_by_key["cohort"].level == "child"
    assert nav_by_key["cross_db"].level == "child"
    assert nav_by_key["research_agent"].level == "top"

    assert not hasattr(sidebar, "_sidebar_next_steps_html")
    css_text = shell_styles._load_shell_overrides_css()

    assert "border-left: 3px solid var(--accent)" in css_text
    assert ".eu-context-label" in css_text
    assert "padding-top: 14px" in css_text
    assert "margin-top: 14px !important" in css_text
    assert "st-key-eunavrow_tutorial" in css_text
    assert ".eu-nav-group-label" in css_text
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
    assert "min-height: 34px !important" in css_text
    assert "eu-side-guide" not in css_text
    sidebar_text = Path(sidebar.__file__).read_text(encoding="utf-8")
    assert "_eu_visualization_nav_open" in sidebar_text
    assert 'count="3"' not in sidebar_text
    assert "Choose view" not in sidebar_text
    assert "_eu_sidebar_settings_open" in sidebar_text
    assert "_render_sidebar_ai_and_lang()" in sidebar_text
    assert "render_sidebar_chat_widget" not in sidebar_text
    assert "_real_data_source_ready()" in sidebar_text
    assert "Path after setup" not in sidebar_text

    app_text = Path(app.__file__).read_text(encoding="utf-8")
    llm_text = Path(app.__file__).with_name("llm_chat.py").read_text(encoding="utf-8")
    ai_optin_text = Path(app.__file__).with_name("ai_optin.py").read_text(encoding="utf-8")
    assert "render_floating_chat_dock()" not in app_text
    assert "render_inline_ai_panel()" in app_text
    assert "_open_embedded_ai_assistant" in app_text
    assert "_inline_ai_panel_open" in app_text
    assert "Show floating AI assistant" not in llm_text
    assert "Show floating AI assistant" not in ai_optin_text
    assert "per-run external LLM opt-in" in ai_optin_text
    assert "Use the bottom-right chat button" not in llm_text
    assert "render_inline_ai_panel" in llm_text
    assert "st-key-inline_ai_assistant_panel" in llm_text
