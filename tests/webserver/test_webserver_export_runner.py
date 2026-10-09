"""The native export runner applies the requested cohort and concept contract.

It interrupts the current module on cancel, applies the native cohort, ICD
and concept-derived selections before loading concepts, honors per-module
concept choices, writes the digest-bound column metadata, and publishes only
outputs whose primary bindings and structural availability it can name.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.webserver import dataio


@pytest.fixture(autouse=True)
def _disable_real_provider_env_file(monkeypatch) -> None:
    monkeypatch.setenv("EASYICU_DISABLE_PROVIDER_ENV_FILE", "1")


def test_an_export_without_a_folder_stays_in_the_servers_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "real"))
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "isolated"))

    out = dataio._resolve_export_out_dir(
        out_dir=None,
        database="miiv",
        export_format="parquet",
        create_run_subdir=False,
    )

    assert out == tmp_path / "isolated" / ".easyicu" / "exports"


class _ExportJob:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def emit(self, payload: dict[str, object]) -> None:
        self.events.append(payload)


def test_export_runner_interrupts_current_duckdb_module_on_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager

    import easyicu.api as api_module
    import easyicu.datasource as datasource_module
    from easyicu.webserver.jobs import Job

    @contextmanager
    def fake_keep_cache(**_: object):
        yield None

    job = Job("cancel-current-query", "extract")
    loaded: list[list[str]] = []
    interrupt_calls: list[str] = []

    def fake_load_concepts(concepts, **kwargs):
        loaded.append(list(concepts))
        ids = (kwargs.get("patient_ids") or {}).get("stay_id", [])
        if len(loaded) == 1:
            return pd.DataFrame({"stay_id": ids, "age": [65] * len(ids)})
        job.request_cancel("user_requested")
        raise datasource_module.DuckDBQueryInterrupted(
            "DuckDB query interrupted"
        )

    monkeypatch.setattr(api_module, "keep_cache", fake_keep_cache)
    monkeypatch.setattr(api_module, "load_concepts", fake_load_concepts)
    monkeypatch.setattr(
        api_module,
        "get_all_patient_ids",
        lambda *_, **__: ([1, 2], "stay_id"),
    )
    monkeypatch.setattr(
        datasource_module,
        "get_duckdb_interrupt_callback",
        lambda: lambda: interrupt_calls.append("interrupt"),
    )

    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["demographics", "vitals"],
        concepts={"demographics": ["age"], "vitals": ["hr", "map"]},
        export_format="csv",
        out_dir=str(tmp_path / "out"),
    )

    result = runner(job)

    assert interrupt_calls == ["interrupt"]
    assert len(loaded) == 2
    assert result["cancelled_at"] == "modules"
    assert result["file_count"] == 1
    assert result["files"][0]["module"] == "demographics"
    assert result["manifest"] is None
    assert not (tmp_path / "out" / "_manifest.json").exists()


def _patch_export_api(
    monkeypatch: pytest.MonkeyPatch, loaded: list[dict[str, object]]
) -> None:
    from contextlib import contextmanager
    import easyicu.api as api_module
    from easyicu.resources import load_dictionary

    dictionary = load_dictionary(include_sofa2=True)

    @contextmanager
    def fake_keep_cache(**_: object):
        yield None

    def fake_load_concepts(concepts, **kwargs):
        loaded.append({"concepts": concepts, "kwargs": kwargs})
        ids = (kwargs.get("patient_ids") or {}).get("stay_id", [])
        payload: dict[str, object] = {"stay_id": ids}
        for concept in concepts:
            definition = dictionary.get(concept)
            if definition is not None and definition.class_name == "lgl_cncpt":
                payload[concept] = [index % 2 for index in range(len(ids))]
            else:
                payload[concept] = [65.0] * len(ids)
        return pd.DataFrame(payload)

    monkeypatch.setattr(api_module, "keep_cache", fake_keep_cache)
    monkeypatch.setattr(api_module, "load_concepts", fake_load_concepts)


def test_export_runner_applies_native_cohort_contract_to_patient_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.patient_filter as patient_filter_module

    class FakePatientFilter:
        def __init__(
            self, database: str, data_path: str, verbose: bool = False
        ) -> None:
            self.database = database
            self.data_path = data_path
            self.verbose = verbose

        def filter(self, **kwargs):
            assert kwargs["age_min"] == 40
            assert kwargs["age_max"] == 80
            assert kwargs["first_icu_stay"] is True
            assert kwargs["los_min"] == 24
            assert kwargs["return_dataframe"] is True
            return pd.DataFrame({"patient_id": [2]})

    loaded: list[dict[str, object]] = []
    _patch_export_api(monkeypatch, loaded)
    monkeypatch.setattr(patient_filter_module, "PatientFilter", FakePatientFilter)

    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["demographics"],
        export_format="csv",
        out_dir=str(tmp_path / "out"),
        max_patients=500,
        cohort={
            "preset": "adult_first",
            "age_min": 40,
            "age_max": 80,
            "min_icu_los_hours": 24,
            "observation_window_hours": 48,
            "exclude_readmissions": True,
        },
    )

    result = runner(_ExportJob())
    manifest = json.loads(
        (tmp_path / "out" / "_manifest.json").read_text(encoding="utf-8")
    )

    assert result["file_count"] == 1
    assert loaded[0]["kwargs"]["patient_ids"] == {"stay_id": [2]}
    assert "win_length" not in loaded[0]["kwargs"]
    assert manifest["cohort_contract"]["age_min"] == 40
    assert manifest["cohort_report"]["selected"] == 1


def test_export_runner_applies_icd_include_exclude_before_loading_concepts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.patient_filter as patient_filter_module

    pd.DataFrame({"stay_id": [1, 2, 3], "hadm_id": [10, 20, 30]}).to_csv(
        tmp_path / "icustays.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "hadm_id": [10, 20, 30, 30],
            "icd_code": ["A419", "J189", "A410", "R650"],
        }
    ).to_csv(tmp_path / "diagnoses_icd.csv", index=False)

    class FakePatientFilter:
        def __init__(
            self, database: str, data_path: str, verbose: bool = False
        ) -> None:
            pass

        def filter(self, **kwargs):
            return pd.DataFrame({"patient_id": [1, 2, 3]})

    loaded: list[dict[str, object]] = []
    _patch_export_api(monkeypatch, loaded)
    monkeypatch.setattr(patient_filter_module, "PatientFilter", FakePatientFilter)

    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["demographics"],
        export_format="csv",
        out_dir=str(tmp_path / "out"),
        cohort={
            "preset": "icd",
            "icd_enabled": True,
            "icd_include": "A41",
            "icd_exclude": "R65",
        },
    )

    runner(_ExportJob())
    manifest = json.loads(
        (tmp_path / "out" / "_manifest.json").read_text(encoding="utf-8")
    )

    assert loaded[0]["kwargs"]["patient_ids"] == {"stay_id": [1]}
    assert manifest["cohort_report"]["icd"]["include_matches"] == 2
    assert manifest["cohort_report"]["icd"]["exclude_matches"] == 1


def test_export_runner_keeps_legacy_all_icu_default_without_cohort_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.api as api_module

    loaded: list[dict[str, object]] = []
    _patch_export_api(monkeypatch, loaded)
    monkeypatch.setattr(
        api_module, "get_all_patient_ids", lambda *_, **__: ([3, 4], "stay_id")
    )

    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["demographics"],
        export_format="csv",
        out_dir=str(tmp_path / "out"),
        max_patients=2,
    )

    runner(_ExportJob())

    assert loaded[0]["kwargs"]["patient_ids"] == {"stay_id": [3, 4]}
    assert "win_length" not in loaded[0]["kwargs"]
    manifest = json.loads(
        (tmp_path / "out" / "_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["cohort_contract"]["preset"] == "all_icu"
    assert manifest["cohort_contract"]["observation_window_hours"] == 720


def test_export_runner_honors_module_specific_concept_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.api as api_module

    loaded: list[dict[str, object]] = []
    _patch_export_api(monkeypatch, loaded)
    monkeypatch.setattr(
        api_module, "get_all_patient_ids", lambda *_, **__: ([1, 2], "stay_id")
    )

    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["demographics", "vitals"],
        concepts={"demographics": ["age"], "vitals": ["hr", "map"]},
        export_format="csv",
        out_dir=str(tmp_path / "out"),
        max_patients=2,
    )

    result = runner(_ExportJob())
    manifest = json.loads(
        (tmp_path / "out" / "_manifest.json").read_text(encoding="utf-8")
    )
    readme = (tmp_path / "out" / "README.md").read_text(encoding="utf-8")

    assert result["file_count"] == 2
    assert [row["concepts"] for row in loaded] == [["age"], ["hr", "map"]]
    assert manifest["concept_selection"]["mode"] == "explicit"
    assert manifest["concept_selection"]["modules"] == {
        "demographics": ["age"],
        "vitals": ["hr", "map"],
    }
    assert manifest["files"][0]["concept_ids"] == ["age"]
    assert manifest["files"][1]["concept_ids"] == ["hr", "map"]
    assert "Concepts selected: `3`" in readme


@pytest.mark.parametrize("include_feature_definitions", [True, False])
def test_export_runner_writes_digest_bound_column_metadata_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_feature_definitions: bool,
) -> None:
    import easyicu.api as api_module
    from easyicu.concept.metadata_projection import ConceptColumnRole
    from easyicu.concept.metadata_sidecar import read_content_addressed_sidecar
    from easyicu.research_agent.intake.export_package import open_export_package

    loaded: list[dict[str, object]] = []
    _patch_export_api(monkeypatch, loaded)

    def exact_age_export(_concepts, **kwargs):
        ids = (kwargs.get("patient_ids") or {}).get("stay_id", [])
        return pd.DataFrame({"stay_id": ids, "age": [65] * len(ids)})

    monkeypatch.setattr(api_module, "load_concepts", exact_age_export)
    monkeypatch.setattr(
        api_module, "get_all_patient_ids", lambda *_, **__: ([1, 2], "stay_id")
    )
    out = tmp_path / "out"
    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["demographics"],
        concepts={"demographics": ["age"]},
        export_format="csv",
        out_dir=str(out),
        include_feature_definitions=include_feature_definitions,
    )

    result = runner(_ExportJob())
    manifest = json.loads((out / "_manifest.json").read_text(encoding="utf-8"))
    descriptor = manifest["column_metadata"]

    assert result["column_metadata"] == descriptor["file"]
    assert result["column_metadata_sha256"] == descriptor["sha256"]
    assert descriptor["file"] == (f"column_metadata.sha256-{descriptor['sha256']}.json")
    assert manifest["schema_version"] == "easyicu_native_export_v2"
    assert manifest["files"][0]["column_metadata_columns"] == ["age"]
    sidecar = read_content_addressed_sidecar(
        out / descriptor["file"],
        expected_sha256=descriptor["sha256"],
        expected_size=descriptor["size"],
    )
    assert sidecar.source_database == "miiv"
    assert sidecar.record_count == descriptor["record_count"] == 1
    file_binding = sidecar.files[0]
    assert file_binding.relative_path == "demographics.csv"
    assert file_binding.identity_column == "stay_id"
    age = file_binding.columns["age"]
    assert age.metadata.source_concept == "age"
    assert age.metadata.role is ConceptColumnRole.VALUE
    assert age.representation_transform is None
    assert manifest["feature_definitions"]["included"] is bool(
        include_feature_definitions
    )
    with open_export_package(out) as package:
        assert package.column_metadata_sha256 == descriptor["sha256"]
        assert set(package.concept_index) == {"age"}


def test_export_runner_loads_composite_source_and_publishes_public_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.api as api_module
    from easyicu.concept.metadata_projection import ConceptColumnRole
    from easyicu.concept.metadata_sidecar import read_content_addressed_sidecar

    loaded: list[dict[str, object]] = []
    _patch_export_api(monkeypatch, loaded)

    def exact_sep3_export(concepts, **kwargs):
        loaded.append({"concepts": list(concepts), "kwargs": kwargs})
        ids = (kwargs.get("patient_ids") or {}).get("stay_id", [])
        return pd.DataFrame(
            {
                "stay_id": ids,
                "charttime": [0.0] * len(ids),
                "sep3": [index % 2 == 0 for index in range(len(ids))],
            }
        )

    monkeypatch.setattr(api_module, "load_concepts", exact_sep3_export)
    monkeypatch.setattr(
        api_module, "get_all_patient_ids", lambda *_, **__: ([1, 2], "stay_id")
    )
    out = tmp_path / "out"
    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["sepsis3_sofa1"],
        export_format="csv",
        out_dir=str(out),
        include_feature_definitions=False,
    )

    result = runner(_ExportJob())
    exported = pd.read_csv(out / "sepsis3_sofa1.csv")
    manifest = json.loads((out / "_manifest.json").read_text(encoding="utf-8"))
    descriptor = manifest["column_metadata"]
    sidecar = read_content_addressed_sidecar(
        out / descriptor["file"],
        expected_sha256=descriptor["sha256"],
        expected_size=descriptor["size"],
    )

    assert result["file_count"] == 1
    assert loaded[-1]["concepts"] == ["sep3"]
    assert "sep3_sofa1" in exported.columns
    assert "sep3" not in exported.columns
    assert manifest["files"][0]["concept_ids"] == ["sep3_sofa1"]
    binding = sidecar.files[0].columns["sep3_sofa1"]
    assert binding.metadata.source_concept == "sep3_sofa1"
    assert binding.metadata.role is ConceptColumnRole.EVENT_STATUS


def test_export_runner_does_not_publish_v2_without_selected_primary_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.api as api_module

    loaded: list[dict[str, object]] = []
    _patch_export_api(monkeypatch, loaded)

    def unrelated_output(_concepts, **kwargs):
        ids = (kwargs.get("patient_ids") or {}).get("stay_id", [])
        return pd.DataFrame({"stay_id": ids, "height": [180.0] * len(ids)})

    monkeypatch.setattr(api_module, "load_concepts", unrelated_output)
    monkeypatch.setattr(
        api_module, "get_all_patient_ids", lambda *_, **__: ([1, 2], "stay_id")
    )
    out = tmp_path / "out"
    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["demographics"],
        concepts={"demographics": ["age"]},
        export_format="csv",
        out_dir=str(out),
        include_feature_definitions=False,
    )

    with pytest.raises(dataio.ExportCohortError) as exc_info:
        runner(_ExportJob())
    assert exc_info.value.detail["error"] == "column_metadata_primary_binding_missing"
    assert exc_info.value.detail["concepts"] == ["age"]
    assert not (out / "_manifest.json").exists()


def test_export_runner_records_owner_confirmed_structural_unavailability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.api as api_module

    loaded: list[dict[str, object]] = []
    _patch_export_api(monkeypatch, loaded)

    def no_mimic_ventilator_days(concepts, **kwargs):
        loaded.append({"concepts": list(concepts), "kwargs": kwargs})
        ids = (kwargs.get("patient_ids") or {}).get("stay_id", [])
        return pd.DataFrame({"stay_id": ids})

    monkeypatch.setattr(api_module, "load_concepts", no_mimic_ventilator_days)
    monkeypatch.setattr(
        api_module, "get_all_patient_ids", lambda *_, **__: ([1, 2], "stay_id")
    )
    out = tmp_path / "out"
    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["outcome"],
        concepts={"outcome": ["vent_free_days_28"]},
        export_format="csv",
        out_dir=str(out),
        include_feature_definitions=True,
    )

    result = runner(_ExportJob())
    manifest = json.loads((out / "_manifest.json").read_text(encoding="utf-8"))
    definitions = json.loads(
        (out / "feature_definitions.json").read_text(encoding="utf-8")
    )

    assert result["file_count"] == 1
    assert manifest["concept_availability"] == {
        "structurally_unavailable_count": 1,
        "structurally_unavailable": [
            {
                "concept_id": "vent_free_days_28",
                "module": "outcome",
                "database": "miiv",
                "status": "structurally_unavailable",
                "reason_code": "outcome_concept_structurally_unavailable",
                "supported_databases": [],
            }
        ],
    }
    assert definitions["records"][0]["availability"]["status"] == (
        "structurally_unavailable"
    )


def test_export_runner_explains_a_source_native_renal_column_its_profile_lacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The renal bundle's shape follows each database's pinned AKI profile.

    eICU's official tree publishes urine components and no AKI stage, so the
    per-component stage columns cannot exist there.  Without the profile
    owner's receipt this export fails as an unexplained gap and the renal
    module cannot be extracted for eICU at all.
    """

    import easyicu.api as api_module

    loaded: list[dict[str, object]] = []
    _patch_export_api(monkeypatch, loaded)

    def urine_components_only(concepts, **kwargs):
        loaded.append({"concepts": list(concepts), "kwargs": kwargs})
        ids = (kwargs.get("patient_ids") or {}).get("stay_id", [])
        return pd.DataFrame(
            {"stay_id": ids, "aki_stage_source_native": [0] * len(ids)}
        )

    monkeypatch.setattr(api_module, "load_concepts", urine_components_only)
    monkeypatch.setattr(
        api_module, "get_all_patient_ids", lambda *_, **__: ([1, 2], "stay_id")
    )
    out = tmp_path / "out"
    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="eicu",
        modules=["renal"],
        concepts={
            "renal": ["aki_stage_source_native", "aki_stage_creat_source_native"]
        },
        export_format="csv",
        out_dir=str(out),
        include_feature_definitions=False,
    )

    result = runner(_ExportJob())
    manifest = json.loads((out / "_manifest.json").read_text(encoding="utf-8"))

    assert result["file_count"] == 1
    assert manifest["concept_availability"] == {
        "structurally_unavailable_count": 1,
        "structurally_unavailable": [
            {
                "concept_id": "aki_stage_creat_source_native",
                "module": "renal",
                "database": "eicu",
                "status": "structurally_unavailable",
                "reason_code": "source_native_profile_publishes_no_such_component",
                "supported_databases": ["miiv", "mimic"],
                "source_native_profile": "EICU_OFFICIAL_RENAL_COMPONENTS_34CECE8C",
                "source_native_output_kind": (
                    "URINE_COMPONENT_ONLY_NO_OFFICIAL_AKI_STAGE"
                ),
            }
        ],
    }


def test_export_runner_explains_the_renal_column_of_the_adapter_branch_not_taken(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MIMIC-IV publishes its components or, lacking a CRRT source, a reason.

    A source with the CRRT table (the official demo included) takes the
    component branch, so the reason column cannot exist; before the profile
    declared its modes every current MIMIC-IV renal export failed on it.
    """

    import easyicu.api as api_module

    loaded: list[dict[str, object]] = []
    _patch_export_api(monkeypatch, loaded)

    def component_branch(concepts, **kwargs):
        loaded.append({"concepts": list(concepts), "kwargs": kwargs})
        ids = (kwargs.get("patient_ids") or {}).get("stay_id", [])
        return pd.DataFrame(
            {
                "stay_id": ids,
                "aki_stage_source_native": [0] * len(ids),
                "aki_stage_crrt_source_native": [0] * len(ids),
            }
        )

    monkeypatch.setattr(api_module, "load_concepts", component_branch)
    monkeypatch.setattr(
        api_module, "get_all_patient_ids", lambda *_, **__: ([1, 2], "stay_id")
    )
    out = tmp_path / "out"
    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["renal"],
        concepts={
            "renal": [
                "aki_stage_source_native",
                "aki_stage_crrt_source_native",
                "aki_source_native_reason",
            ]
        },
        export_format="csv",
        out_dir=str(out),
        include_feature_definitions=False,
    )

    runner(_ExportJob())
    manifest = json.loads((out / "_manifest.json").read_text(encoding="utf-8"))

    assert manifest["concept_availability"]["structurally_unavailable"] == [
        {
            "concept_id": "aki_source_native_reason",
            "module": "renal",
            "database": "miiv",
            "status": "structurally_unavailable",
            "reason_code": "source_native_profile_mode_emits_no_such_column",
            "supported_databases": ["aumc", "hirid", "miiv", "sic"],
            "source_native_profile": "MIMIC_IV_MIT_LCP_KDIGO_D20B49A7",
            "source_native_output_kind": "DYNAMIC_KDIGO_STAGE_0_TO_3",
            "source_native_mode": "components",
        }
    ]


def test_export_runner_still_fails_on_a_renal_column_the_profile_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MIMIC-IV does publish the CRRT component, so its absence stays loud."""

    import easyicu.api as api_module

    loaded: list[dict[str, object]] = []
    _patch_export_api(monkeypatch, loaded)

    def missing_crrt_component(concepts, **kwargs):
        loaded.append({"concepts": list(concepts), "kwargs": kwargs})
        ids = (kwargs.get("patient_ids") or {}).get("stay_id", [])
        return pd.DataFrame(
            {"stay_id": ids, "aki_stage_source_native": [0] * len(ids)}
        )

    monkeypatch.setattr(api_module, "load_concepts", missing_crrt_component)
    monkeypatch.setattr(
        api_module, "get_all_patient_ids", lambda *_, **__: ([1, 2], "stay_id")
    )
    out = tmp_path / "out"
    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["renal"],
        concepts={
            "renal": ["aki_stage_source_native", "aki_stage_crrt_source_native"]
        },
        export_format="csv",
        out_dir=str(out),
        include_feature_definitions=False,
    )

    with pytest.raises(dataio.ExportCohortError) as exc_info:
        runner(_ExportJob())

    assert exc_info.value.detail["error"] == "column_metadata_primary_binding_missing"
    assert exc_info.value.detail["concepts"] == ["aki_stage_crrt_source_native"]


def test_export_runner_rejects_unknown_selected_concepts(
    tmp_path: Path,
) -> None:
    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["vitals"],
        concepts={"vitals": ["hr", "not_a_real_concept"]},
        export_format="csv",
        out_dir=str(tmp_path / "out"),
    )

    with pytest.raises(dataio.ExportCohortError) as exc:
        runner(_ExportJob())

    assert exc.value.detail["error"] == "invalid_selected_concepts"
    assert exc.value.detail["invalid"] == ["vitals:not_a_real_concept"]
    assert not (tmp_path / "out").exists()


def test_export_runner_can_create_timestamped_run_folder_with_readme(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.api as api_module

    root = tmp_path / "exports"
    loaded: list[dict[str, object]] = []
    _patch_export_api(monkeypatch, loaded)
    monkeypatch.setattr(
        api_module, "get_all_patient_ids", lambda *_, **__: ([7, 8], "stay_id")
    )

    runner = dataio.make_export_runner(
        data_path=str(tmp_path / "source"),
        database="miiv",
        modules=["demographics"],
        export_format="parquet",
        out_dir=str(root),
        create_run_subdir=True,
        max_patients=2,
    )

    result = runner(_ExportJob())
    out = Path(result["out_dir"])

    assert out.parent == root
    assert out.name.startswith("easyicu_export_")
    assert out.name.endswith("_miiv_parquet")
    assert (out / "_manifest.json").exists()
    assert (out / "README.md").exists()
    assert result["manifest"] == "_manifest.json"
    assert result["readme"] == "README.md"
    manifest = json.loads((out / "_manifest.json").read_text(encoding="utf-8"))
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert manifest["export_folder"]["run_subdir"] is True
    assert manifest["export_folder"]["label"] == out.name
    assert manifest["cohort_contract"]["observation_window_hours"] == 720
    assert "Observation window: `720 hours`" in readme
    assert "`demographics.parquet`" in readme
    assert "No patient rows are included in this README" in readme


def test_export_runner_ignores_stale_icd_tokens_when_preset_is_not_icd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.patient_filter as patient_filter_module

    class FakePatientFilter:
        def __init__(
            self, database: str, data_path: str, verbose: bool = False
        ) -> None:
            pass

        def filter(self, **kwargs):
            return pd.DataFrame({"patient_id": [1, 2]})

    loaded: list[dict[str, object]] = []
    _patch_export_api(monkeypatch, loaded)
    monkeypatch.setattr(patient_filter_module, "PatientFilter", FakePatientFilter)

    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["demographics"],
        export_format="csv",
        out_dir=str(tmp_path / "out"),
        cohort={
            "preset": "adult_first",
            "icd_enabled": False,
            "icd_include": "A41",
            "icd_exclude": "R65",
        },
    )

    runner(_ExportJob())
    manifest = json.loads(
        (tmp_path / "out" / "_manifest.json").read_text(encoding="utf-8")
    )

    assert loaded[0]["kwargs"]["patient_ids"] == {"stay_id": [1, 2]}
    assert "win_length" not in loaded[0]["kwargs"]
    assert manifest["cohort_contract"]["icd_include"] == []
    assert manifest["cohort_contract"]["observation_window_hours"] == 720
    assert manifest["cohort_report"]["applied_filters"] == ["demographics"]


@pytest.mark.parametrize(
    ("preset", "concepts", "positive_column"),
    [
        ("sepsis3", ["sep3_sofa2"], "sep3_sofa2"),
        ("aki", ["aki"], "aki"),
        ("ventilation", ["mech_vent", "vent_ind"], "mech_vent"),
        ("vasopressor", ["vaso_ind"], "vaso_ind"),
        (
            "respiratory",
            ["adv_resp", "mech_vent", "vent_ind", "pafi", "safi"],
            "adv_resp",
        ),
    ],
)
def test_export_runner_applies_concept_derived_cohort_prefilter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preset: str,
    concepts: list[str],
    positive_column: str,
) -> None:
    from contextlib import contextmanager
    import easyicu.api as api_module
    import easyicu.patient_filter as patient_filter_module
    from easyicu.resources import load_dictionary

    class FakePatientFilter:
        def __init__(
            self, database: str, data_path: str, verbose: bool = False
        ) -> None:
            pass

        def filter(self, **kwargs):
            return pd.DataFrame({"patient_id": [1, 2, 3]})

    loaded: list[dict[str, object]] = []
    dictionary = load_dictionary(include_sofa2=True)

    @contextmanager
    def fake_keep_cache(**_: object):
        yield None

    def fake_load_concepts(concepts, **kwargs):
        loaded.append({"concepts": concepts, "kwargs": kwargs})
        if concepts == loaded_concepts:
            return pd.DataFrame({"stay_id": [1, 2, 3], "charttime": [0.0, 1.0, 71.0], positive_column: [0, 1, True]})
        ids = (kwargs.get("patient_ids") or {}).get("stay_id", [])
        payload: dict[str, object] = {"stay_id": ids}
        for concept in concepts:
            definition = dictionary.get(concept)
            if definition is not None and definition.class_name == "lgl_cncpt":
                payload[concept] = [index % 2 for index in range(len(ids))]
            else:
                payload[concept] = [65.0] * len(ids)
        return pd.DataFrame(payload)

    loaded_concepts = list(concepts)
    monkeypatch.setattr(patient_filter_module, "PatientFilter", FakePatientFilter)
    monkeypatch.setattr(api_module, "keep_cache", fake_keep_cache)
    monkeypatch.setattr(api_module, "load_concepts", fake_load_concepts)

    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["demographics"],
        export_format="csv",
        out_dir=str(tmp_path / "out"),
        cohort={"preset": preset, "observation_window_hours": 72},
    )

    job = _ExportJob()
    runner(job)
    manifest = json.loads(
        (tmp_path / "out" / "_manifest.json").read_text(encoding="utf-8")
    )

    assert loaded[0]["concepts"] == concepts
    assert loaded[0]["kwargs"]["patient_ids"] == {"stay_id": [1, 2, 3]}
    assert "win_length" not in loaded[0]["kwargs"]
    assert loaded[1]["kwargs"]["patient_ids"] == {"stay_id": [2, 3]}
    assert manifest["cohort_report"]["applied_filters"] == [
        "demographics",
        "concept_prefilter",
    ]
    assert manifest["cohort_report"]["concept_matches"] == 2
    stages = [
        event.get("stage") for event in job.events if event.get("phase") == "cohort"
    ]
    assert "concept_prefilter" in stages
    assert "cohort_selected" in stages
    assert all(
        "patient_ids" not in event and "stay_id" not in event for event in job.events
    )


def test_export_runner_fails_closed_for_unsupported_native_cohort_preset(
    tmp_path: Path,
) -> None:
    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["demographics"],
        out_dir=str(tmp_path / "out"),
        cohort={"preset": "obesity"},
    )

    with pytest.raises(dataio.ExportCohortError) as exc:
        runner(_ExportJob())

    assert exc.value.detail["error"] == "unsupported_cohort_preset"
