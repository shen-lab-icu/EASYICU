from __future__ import annotations

import os
import json
import sys
import types

import easyicu
from easyicu.concept import load_dictionary
import easyicu.webapp.app as app
import easyicu.webapp.concept_catalog as concept_catalog
import easyicu.webapp.cohort_filters as cohort_filters
import easyicu.webapp.cohort_workspace as cohort_workspace
import easyicu.webapp.data_paths as data_paths
import easyicu.webapp.data_workflows as data_workflows
import easyicu.webapp.export_reports as export_reports
import easyicu.webapp.export_workflow as export_workflow
import easyicu.webapp.sidebar as sidebar
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
    at.button(key="entry_demo_btn").click().run(timeout=60)
    # Main nav is a st.radio since the tabs→radio refactor: only the active
    # page renders, so switch to the quick-viz page before its demo-load
    # button (`viz_load_demo`) is reachable.
    at.session_state["_active_main_page"] = "quick_viz"
    at.run(timeout=60)
    at.button(key="viz_load_demo").click().run(timeout=60)

    rendered_text = " ".join(
        getattr(element, "value", "")
        for collection_name in ("markdown", "html")
        for element in getattr(at, collection_name, [])
    )
    assert "Data Loaded" in rendered_text
    assert len(at.session_state["loaded_concepts"]) > 0
    assert len(at.session_state["patient_ids"]) == 50


def test_sidebar_quick_preview_is_one_click_after_feature_confirmation(tmp_path, monkeypatch) -> None:
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    at = streamlit_testing.AppTest.from_file(app.__file__)
    at.session_state["entry_lang_select"] = "EN"
    at.session_state["language"] = "en"
    at.run(timeout=60)
    at.button(key="entry_demo_btn").click().run(timeout=60)
    at.button(key="step1_confirm_demo").click().run(timeout=60)
    at.button(key="step2_confirm_no_filter").click().run(timeout=60)
    at.button(key="select_all_groups").click().run(timeout=60)
    at.button(key="step3_confirm_selection").click().run(timeout=60)

    warning_text = " ".join(getattr(warning, "value", "") for warning in at.warning)
    assert "preview_n_patients" not in warning_text
    assert "sidebar_preview_btn" in {button.key for button in at.button}

    at.button(key="sidebar_preview_btn").click().run(timeout=60)

    assert at.session_state["loaded_data_origin"] == "preview"
    assert len(at.session_state["loaded_concepts"]) > 0
    assert len(at.session_state["patient_ids"]) == at.session_state["_preview_n"]
    assert at.session_state["_preview_requested"] is False
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
        "👥 Groups",
        "🧾 Coverage",
        "📈 Cross-DB",
        "🎯 Snapshot",
        "🧭 SOFA Δ",
    ]
    assert rendered_panels == ["groups", "coverage", "crossdb", "snapshot", "sofa"]


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
