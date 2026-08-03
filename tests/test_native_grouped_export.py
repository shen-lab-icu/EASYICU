"""Native-v2 sealing for the high-throughput grouped module extractor."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from easyicu.api import extraction as api
from easyicu.research_agent.intake.export_package import open_export_package


def _completed_result(module: str = "demographics") -> dict[str, object]:
    return {"modules": {module: {"errors": []}}}


def test_native_time_axis_uses_los_and_normalises_stay_level_outcomes() -> None:
    outcome = pd.DataFrame(
        {
            "admissionid": [1, 2],
            "los_icu": [2.0, float("nan")],
        }
    )
    bounds = api._native_export_stay_time_upper_bounds(outcome)
    assert bounds == {1: 72.0}

    longitudinal = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 2],
            "charttime": [-25.0, -24.0, 73.0, 9_000.0],
            "hr": [70.0, 71.0, 72.0, 73.0],
        }
    )
    filtered, audit = api._enforce_native_export_time_axis(
        longitudinal,
        module="vitals",
        stay_time_upper_bounds=bounds,
    )
    assert filtered["charttime"].tolist() == [-24.0]
    assert audit["excluded_rows"] == 3

    stay_level, stay_audit = api._enforce_native_export_time_axis(
        pd.DataFrame(
            {
                "stay_id": [1, 2],
                "charttime": [127_664.0, float("nan")],
                "mort_28d": [False, True],
            }
        ),
        module="outcome",
        stay_time_upper_bounds=bounds,
    )
    assert stay_level["charttime"].tolist() == [0.0, 0.0]
    assert stay_audit["normalized_stay_level_rows"] == 2


def test_native_renal_publication_drops_untimed_negative_rrt_merge_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        api,
        "EXTRACT_MODULES",
        {"renal": ["urine", "rrt_criteria"]},
    )
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "charttime": [0.0, None, None],
            "urine": [100.0, None, None],
            "rrt_criteria": [False, False, None],
        }
    ).to_parquet(tmp_path / "renal.parquet", index=False)

    api._publish_native_export_v2(
        database="aumc",
        data_path="/raw/source-must-not-be-read",
        output_dir=str(tmp_path),
        modules=["renal"],
        max_patients=None,
        result=_completed_result("renal"),
    )

    exported = pd.read_parquet(tmp_path / "renal.parquet")
    assert exported[["stay_id", "charttime"]].to_dict("records") == [
        {"stay_id": 1, "charttime": 0.0}
    ]
    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    audit = manifest["files"][0]["time_axis_audit"]
    assert audit["excluded_untimed_negative_rrt_criteria_rows"] == 1
    assert audit["excluded_untimed_empty_rows"] == 1
    assert manifest["files"][0]["row_grain_audit"][
        "null_charttime_rows_after"
    ] == 0


def test_native_renal_publication_rejects_positive_untimed_rrt_criteria(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "EXTRACT_MODULES", {"renal": ["rrt_criteria"]})
    pd.DataFrame(
        {
            "stay_id": [1],
            "charttime": [None],
            "rrt_criteria": [True],
        }
    ).to_parquet(tmp_path / "renal.parquet", index=False)

    with pytest.raises(ValueError, match="positive rrt_criteria rows"):
        api._publish_native_export_v2(
            database="aumc",
            data_path="/raw/source-must-not-be-read",
            output_dir=str(tmp_path),
            modules=["renal"],
            max_patients=None,
            result=_completed_result("renal"),
        )

    assert not (tmp_path / "_manifest.json").exists()


def test_grouped_output_is_sealed_without_accessing_the_raw_data_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "EXTRACT_MODULES", {"demographics": ["age"]})
    (tmp_path / "demographics.parquet").write_bytes(b"")
    pd.DataFrame({"stay_id": [1, 2], "age": [63.0, 72.0]}).to_parquet(
        tmp_path / "demographics.parquet", index=False
    )

    published = api._publish_native_export_v2(
        database="miiv",
        data_path="/raw/source-must-not-be-read",
        output_dir=str(tmp_path),
        modules=["demographics"],
        max_patients=None,
        result=_completed_result(),
    )

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert manifest["schema_version"] == "easyicu_native_export_v2"
    assert manifest["files"][0]["file"] == "demographics.parquet"
    assert manifest["files"][0]["primary_key"] == ["stay_id"]
    assert manifest["files"][0]["row_grain"] == "one_row_per_icu_stay"
    assert manifest["files"][0]["parquet_sha256"] == hashlib.sha256(
        (tmp_path / "demographics.parquet").read_bytes()
    ).hexdigest()
    assert manifest["files"][0]["parquet_bytes"] == (
        tmp_path / "demographics.parquet"
    ).stat().st_size
    assert manifest["canonical_physical_schema"]["identity_column"] == "stay_id"
    assert manifest["runtime_provenance"]["easyicu_import_path"].endswith(
        "/src/easyicu"
    )
    assert isinstance(
        manifest["runtime_provenance"]["easyicu_git_dirty"], bool
    )
    if manifest["runtime_provenance"]["easyicu_git_dirty"]:
        assert manifest["runtime_provenance"]["easyicu_git_diff_sha256"]
    else:
        assert manifest["runtime_provenance"]["easyicu_git_diff_sha256"] is None
    assert list(pd.read_parquet(tmp_path / "demographics.parquet").columns) == [
        "stay_id",
        "age",
    ]
    assert published["output_validation_reads"] == 1
    with open_export_package(tmp_path) as package:
        assert package.database == "miiv"
        assert package.column_metadata_sha256 == published["column_metadata_sha256"]
        assert set(package.concept_index) == {"age"}


def test_grouped_output_never_publishes_native_manifest_without_primary_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "EXTRACT_MODULES", {"demographics": ["age"]})
    pd.DataFrame({"stay_id": [1], "height": [180.0]}).to_parquet(
        tmp_path / "demographics.parquet", index=False
    )

    with pytest.raises(ValueError, match="primary physical binding"):
        api._publish_native_export_v2(
            database="miiv",
            data_path="/raw/source-must-not-be-read",
            output_dir=str(tmp_path),
            modules=["demographics"],
            max_patients=None,
            result=_completed_result(),
        )

    assert not (tmp_path / "_manifest.json").exists()
    assert not list(tmp_path.glob("column_metadata.sha256-*.json"))


def test_grouped_export_records_structural_unavailability_without_selecting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        api,
        "EXTRACT_MODULES",
        {"demographics": ["age"], "sepsis_shared": ["susp_inf"]},
    )
    pd.DataFrame({"stay_id": [1], "age": [63.0]}).to_parquet(
        tmp_path / "demographics.parquet", index=False
    )

    published = api._publish_native_export_v2(
        database="miiv",
        data_path="/raw/source-must-not-be-read",
        output_dir=str(tmp_path),
        modules=["demographics", "sepsis_shared"],
        max_patients=None,
        result={
            "modules": {
                "demographics": {"errors": []},
                "sepsis_shared": {"errors": []},
            }
        },
    )

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert manifest["concept_selection"]["modules"] == {
        "demographics": ["age"],
        "sepsis_shared": [],
    }
    assert manifest["unavailable_modules"] == [
        {
            "module": "sepsis_shared",
            "reason": "producer_returned_no_physical_output",
            "concept_ids": ["susp_inf"],
        }
    ]
    placeholder = pd.read_parquet(tmp_path / "sepsis_shared.parquet")
    assert list(placeholder.columns) == ["stay_id", "charttime", "susp_inf"]
    assert placeholder.empty
    assert placeholder["stay_id"].dtype == "int64"
    assert placeholder["charttime"].dtype == "float64"
    assert str(placeholder["susp_inf"].dtype) == "boolean"
    assert published["output_validation_reads"] == 2
    with open_export_package(tmp_path) as package:
        assert set(package.concept_index) == {"age"}

    # Re-sealing an existing typed placeholder must preserve its structural
    # unavailability rather than treating file existence as source evidence.
    (tmp_path / "sepsis_shared.manifest.json").write_text(
        json.dumps({"saved": {}, "errors": []}),
        encoding="utf-8",
    )
    sidecar_file = manifest["column_metadata"]["file"]
    (tmp_path / "_manifest.json").unlink()
    (tmp_path / sidecar_file).unlink()
    api._publish_native_export_v2(
        database="miiv",
        data_path="/raw/source-must-not-be-read",
        output_dir=str(tmp_path),
        modules=["demographics", "sepsis_shared"],
        max_patients=None,
        result={
            "modules": {
                "demographics": {"errors": []},
                "sepsis_shared": {"errors": []},
            }
        },
    )

    republished = json.loads((tmp_path / "_manifest.json").read_text())
    assert republished["unavailable_modules"] == manifest["unavailable_modules"]
    assert next(
        file
        for file in republished["files"]
        if file["module"] == "sepsis_shared"
    )["availability"] == "structurally_unavailable"
    with open_export_package(tmp_path) as package:
        assert set(package.concept_index) == {"age"}


def test_grouped_export_records_missing_concept_inside_physical_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    pd.DataFrame({"stay_id": [1], "mort_28d": [False]}).to_parquet(
        output / "outcome.parquet", index=False
    )
    (output / "outcome.manifest.json").write_text(
        json.dumps(
            {
                "saved": {
                    "outcome": {
                        "concepts": ["mort_28d"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    result = {"modules": {"outcome": {"errors": []}}}
    monkeypatch.setitem(
        api.EXTRACT_MODULES, "outcome", ["mort_28d", "vent_free_days_28"]
    )

    api._publish_native_export_v2(
        database="miiv",
        data_path=str(tmp_path / "source"),
        output_dir=str(output),
        modules=["outcome"],
        max_patients=None,
        result=result,
    )

    manifest = json.loads((output / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["concept_selection"]["modules"]["outcome"] == ["mort_28d"]
    assert manifest["unavailable_concepts"] == [
        {
            "module": "outcome",
            "concept": "vent_free_days_28",
            "reason": "producer_returned_no_physical_column",
        }
    ]
    exported = pd.read_parquet(output / "outcome.parquet")
    assert list(exported.columns) == [
        "stay_id",
        "charttime",
        "mort_28d",
        "vent_free_days_28",
    ]
    assert exported["stay_id"].dtype == "int64"
    assert exported["charttime"].dtype == "float64"
    assert str(exported["mort_28d"].dtype) == "boolean"
    assert exported["vent_free_days_28"].dtype == "float64"
    status = manifest["files"][0]["concept_status"]
    assert status["mort_28d"] == {
        "availability": "available",
        "non_null": 1,
        "excluded_out_of_bounds": 0,
    }
    assert status["vent_free_days_28"] == {
        "availability": "structurally_unavailable_placeholder",
        "non_null": 0,
        "excluded_out_of_bounds": 0,
    }
    with open_export_package(output) as package:
        assert set(package.concept_index) == {"mort_28d"}


def test_grouped_export_normalises_source_identity_and_categorical_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        api, "EXTRACT_MODULES", {"demographics": ["age", "sex"]}
    )
    pd.DataFrame({"patientunitstayid": [11], "age": [63]}).to_parquet(
        tmp_path / "demographics.parquet", index=False
    )
    (tmp_path / "demographics.manifest.json").write_text(
        json.dumps(
            {"saved": {"demographics": {"concepts": ["age"]}}}
        ),
        encoding="utf-8",
    )

    api._publish_native_export_v2(
        database="eicu",
        data_path="/raw/source-must-not-be-read",
        output_dir=str(tmp_path),
        modules=["demographics"],
        max_patients=None,
        result=_completed_result(),
    )

    exported = pd.read_parquet(tmp_path / "demographics.parquet")
    assert list(exported.columns) == ["stay_id", "age", "sex"]
    assert exported["stay_id"].tolist() == [11]
    assert exported["age"].dtype == "float64"
    assert str(exported["sex"].dtype) == "string"
    assert exported["sex"].isna().all()


def test_native_schema_preserves_numeric_logical_and_categorical_families() -> None:
    dictionary = api.load_dictionary(include_sofa2=True)
    frame = pd.DataFrame(
        {
            "patientid": [101, 102],
            "charttime": [0, 1.5],
            "gcs": [15, 10],
            "delirium_positive": [0, 1],
            "avpu": ["A", "V"],
            "ecmo_indication": ["cardiac", None],
        }
    )

    canonical = api._canonicalise_native_export_frame(
        frame,
        module="neurological",
        requested_concepts=[
            "gcs",
            "delirium_positive",
            "avpu",
            "ecmo_indication",
        ],
        dictionary=dictionary,
    )

    assert canonical["stay_id"].dtype == "int64"
    assert canonical["charttime"].dtype == "float64"
    assert canonical["gcs"].dtype == "float64"
    assert str(canonical["delirium_positive"].dtype) == "boolean"
    assert str(canonical["avpu"].dtype) == "string"
    assert str(canonical["ecmo_indication"].dtype) == "string"


def test_native_schema_accepts_multi_class_numeric_concept() -> None:
    dictionary = api.load_dictionary(include_sofa2=True)
    assert isinstance(dictionary["dex"].class_name, list)

    canonical = api._canonicalise_native_export_frame(
        pd.DataFrame(
            {
                "stay_id": [101],
                "charttime": [0.5],
                "dex": [0.7],
            }
        ),
        module="medications",
        requested_concepts=["dex"],
        dictionary=dictionary,
    )

    assert canonical["dex"].dtype == "float64"
    assert canonical["dex"].tolist() == [0.7]


def test_grouped_export_nulls_declared_bound_violations_and_audits_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        api,
        "EXTRACT_MODULES",
        {"demographics": ["height", "weight", "bmi"]},
    )
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "height": [0.0, 170.0, 612.6],
            "weight": [0.0, 75.0, 5864.0],
            "bmi": [7.0, 26.0, 98.0],
        }
    ).to_parquet(tmp_path / "demographics.parquet", index=False)

    api._publish_native_export_v2(
        database="miiv",
        data_path="/raw/source-must-not-be-read",
        output_dir=str(tmp_path),
        modules=["demographics"],
        max_patients=None,
        result=_completed_result(),
    )

    exported = pd.read_parquet(tmp_path / "demographics.parquet")
    assert exported.loc[1, "height"] == 170.0
    assert exported.loc[1, "weight"] == 75.0
    assert exported.loc[1, "bmi"] == pytest.approx(75.0 / 1.7**2)
    assert exported.loc[[0, 2], ["height", "weight", "bmi"]].isna().all().all()

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert (
        manifest["canonical_physical_schema"]["declared_bounds_policy"]
        == "out_of_range_to_null"
    )
    status = manifest["files"][0]["concept_status"]
    assert status["height"] == {
        "availability": "available",
        "non_null": 1,
        "excluded_out_of_bounds": 2,
        "declared_bounds": {"minimum": 10.0, "maximum": 230.0},
    }
    assert status["weight"]["excluded_out_of_bounds"] == 2
    assert status["bmi"]["excluded_out_of_bounds"] == 2


def test_demographics_consolidates_nearest_static_values_and_recomputes_bmi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        api,
        "EXTRACT_MODULES",
        {"demographics": ["age", "bmi", "height", "sex", "weight", "adm"]},
    )
    pd.DataFrame(
        {
            "stay_id": [10, 10, 10, 20],
            "charttime": [-12.0, 2.0, 1.0, 0.0],
            "age": [59.0, 60.0, None, 70.0],
            "bmi": [99.0, 98.0, 97.0, 18.0],
            "height": [180.0, None, 170.0, 160.0],
            "sex": ["M", "M", None, "F"],
            "weight": [80.0, 70.0, None, 50.0],
            "adm": ["emergency", "elective", "urgent", "emergency"],
        }
    ).to_parquet(tmp_path / "demographics.parquet", index=False)

    api._publish_native_export_v2(
        database="hirid",
        data_path="/raw/source-must-not-be-read",
        output_dir=str(tmp_path),
        modules=["demographics"],
        max_patients=None,
        result=_completed_result(),
    )

    exported = pd.read_parquet(tmp_path / "demographics.parquet")
    assert exported["stay_id"].tolist() == [10, 20]
    stay10 = exported.set_index("stay_id").loc[10]
    # Each static concept is selected independently at its nearest non-null
    # source time: age/weight from +2 h, height/adm from +1 h.
    assert stay10["age"] == 60.0
    assert stay10["height"] == 170.0
    assert stay10["weight"] == 70.0
    assert stay10["adm"] == "urgent"
    assert stay10["bmi"] == pytest.approx(70.0 / 1.7**2)

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    audit = manifest["files"][0]["row_grain_audit"]
    assert audit["primary_key"] == ["stay_id"]
    assert audit["source_rows"] == 4
    assert audit["published_rows"] == 2
    assert audit["duplicate_excess_rows_before"] == 2
    assert audit["rows_consolidated"] == 2
    assert audit["recomputed_bmi_rows"] == 2


def test_longitudinal_null_time_boolean_conflicts_use_any_and_are_unique(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "EXTRACT_MODULES", {"sepsis_shared": ["samp"]})
    pd.DataFrame(
        {
            "patientunitstayid": [1, 1, 1, 2],
            "charttime": [None, None, None, None],
            "samp": [False, True, None, False],
        }
    ).to_parquet(tmp_path / "sepsis_shared.parquet", index=False)

    api._publish_native_export_v2(
        database="eicu",
        data_path="/raw/source-must-not-be-read",
        output_dir=str(tmp_path),
        modules=["sepsis_shared"],
        max_patients=None,
        result=_completed_result("sepsis_shared"),
    )

    exported = pd.read_parquet(tmp_path / "sepsis_shared.parquet")
    assert len(exported) == 2
    assert exported.set_index("stay_id")["samp"].to_dict() == {1: True, 2: False}
    assert not exported.duplicated(["stay_id", "charttime"], keep=False).any()
    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    audit = manifest["files"][0]["row_grain_audit"]
    assert audit["null_key_equality"] == "nulls_equal"
    assert audit["null_charttime_rows_before"] == 3
    assert audit["null_charttime_rows_after"] == 2
    assert audit["duplicate_key_groups_before"] == 1
    assert audit["rows_consolidated"] == 1
    assert manifest["files"][0]["time_axis_audit"][
        "excluded_untimed_empty_rows"
    ] == 1


def test_longitudinal_conflicting_strings_fail_closed() -> None:
    dictionary = api.load_dictionary(include_sofa2=True)
    canonical = api._canonicalise_native_export_frame(
        pd.DataFrame(
            {
                "stay_id": [1, 1],
                "charttime": [0.0, 0.0],
                "avpu": ["A", "V"],
            }
        ),
        module="neurological",
        requested_concepts=["avpu"],
        dictionary=dictionary,
    )

    with pytest.raises(ValueError, match="conflicting string concept 'avpu'"):
        api._consolidate_native_export_row_grain(
            canonical,
            module="neurological",
            requested_concepts=["avpu"],
            dictionary=dictionary,
        )


def test_native_bounds_survive_sofa2_overlay_redefinition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "_CONCEPT_BOUNDS_CACHE", None)
    dictionary = api.load_dictionary(include_sofa2=True)
    assert dictionary["uo_6h"].minimum is None
    assert api._load_concept_bounds_map()["uo_6h"] == (0.0, 20.0)
    frame = pd.DataFrame({"uo_6h": [0.0, 1.5, 20.0, 20.1, 1886.0]})

    audit = api._enforce_native_export_concept_bounds(
        frame,
        requested_concepts=["uo_6h"],
        dictionary=dictionary,
    )

    assert frame["uo_6h"].tolist()[:3] == [0.0, 1.5, 20.0]
    assert frame["uo_6h"].iloc[3:].isna().all()
    assert audit["uo_6h"] == {
        "excluded_out_of_bounds": 2,
        "declared_bounds": {"minimum": 0.0, "maximum": 20.0},
    }


def test_grouped_export_reads_special_concept_from_saved_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "EXTRACT_MODULES", {"sepsis3_sofa1": ["sep3_sofa1"]})
    pd.DataFrame(
        {"stay_id": [1, 2], "charttime": [0.0, 1.0], "sep3_sofa1": [0, 1]}
    ).to_parquet(tmp_path / "sepsis3_sofa1.parquet", index=False)
    (tmp_path / "sepsis3_sofa1.manifest.json").write_text(
        json.dumps({"saved": {"sep3_sofa1": {"rows": 2}}}), encoding="utf-8"
    )

    api._publish_native_export_v2(
        database="miiv",
        data_path=str(tmp_path / "source"),
        output_dir=str(tmp_path),
        modules=["sepsis3_sofa1"],
        max_patients=None,
        result={"modules": {"sepsis3_sofa1": {"errors": []}}},
    )

    with open_export_package(tmp_path) as package:
        assert set(package.concept_index) == {"sep3_sofa1"}


def test_unique_longitudinal_publication_never_materialises_full_pandas_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The normal publisher path must remain Arrow-batched across row groups."""
    rows = 40_000
    monkeypatch.setattr(
        api,
        "EXTRACT_MODULES",
        {
            "neurological": [
                "gcs",
                "delirium_positive",
                "avpu",
            ]
        },
    )
    monkeypatch.setenv("EASYICU_NATIVE_PUBLISH_BATCH_ROWS", "16384")
    source = pd.DataFrame(
        {
            "patientid": range(rows),
            "charttime": [0.0] * rows,
            "gcs": [15.0] * rows,
            "delirium_positive": [0, 1] * (rows // 2),
            "avpu": ["A"] * rows,
        }
    )
    source.to_parquet(tmp_path / "neurological.parquet", index=False)
    real_read_parquet = pd.read_parquet

    def _forbid_pandas_payload_read(*_args, **_kwargs):
        raise AssertionError("unique longitudinal payload entered pandas")

    monkeypatch.setattr(pd, "read_parquet", _forbid_pandas_payload_read)
    api._publish_native_export_v2(
        database="miiv",
        data_path="/raw/source-must-not-be-read",
        output_dir=str(tmp_path),
        modules=["neurological"],
        max_patients=None,
        result=_completed_result("neurological"),
    )

    exported = real_read_parquet(tmp_path / "neurological.parquet")
    assert len(exported) == rows
    assert exported["stay_id"].tolist() == list(range(rows))
    assert str(exported["delirium_positive"].dtype) == "boolean"
    assert str(exported["avpu"].dtype) == "string"
    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    entry = manifest["files"][0]
    audit = entry["row_grain_audit"]
    assert audit["publication_backend"] == "pyarrow_record_batches"
    assert audit["uniqueness_memory_limit_mb"] == 512
    assert audit["duplicate_excess_rows_after"] == 0
    assert entry["physical_schema"] == {
        "stay_id": "int64",
        "charttime": "double",
        "gcs": "double",
        "delirium_positive": "bool",
        "avpu": "string",
    }
    assert entry["parquet_sha256"] == hashlib.sha256(
        (tmp_path / "neurological.parquet").read_bytes()
    ).hexdigest()
    assert entry["parquet_bytes"] == (
        tmp_path / "neurological.parquet"
    ).stat().st_size


def test_arrow_publication_matches_legacy_canonical_values_order_and_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        api,
        "EXTRACT_MODULES",
        {"neurological": ["gcs", "delirium_positive", "avpu"]},
    )
    source = pd.DataFrame(
        {
            "patientid": [30, 10, 20, 40, 50],
            "charttime": [-25.0, -24.0, 0.0, 9_000.0, None],
            "gcs": [15.0, 99.0, 12.0, 8.0, None],
            "delirium_positive": [0.0, 1.0, None, 1.0, 0.0],
            # Exercise pandas' exact numeric-to-string representation inside a
            # bounded Arrow batch (1.0 must remain "1.0", not Arrow's "1").
            "avpu": [1.0, 2.0, None, 4.0, 5.0],
        }
    )
    source.to_parquet(tmp_path / "neurological.parquet", index=False)
    dictionary = api.load_dictionary(include_sofa2=True)
    expected = api._canonicalise_native_export_frame(
        pd.read_parquet(tmp_path / "neurological.parquet"),
        module="neurological",
        requested_concepts=["gcs", "delirium_positive", "avpu"],
        dictionary=dictionary,
    )
    expected, expected_time_audit = api._enforce_native_export_time_axis(
        expected,
        module="neurological",
        stay_time_upper_bounds={},
    )
    expected_bounds_audit = api._enforce_native_export_concept_bounds(
        expected,
        requested_concepts=["gcs", "delirium_positive", "avpu"],
        dictionary=dictionary,
    )
    expected, expected_grain_audit = api._consolidate_native_export_row_grain(
        expected,
        module="neurological",
        requested_concepts=["gcs", "delirium_positive", "avpu"],
        dictionary=dictionary,
    )
    import pyarrow as pa
    import pyarrow.parquet as pq

    expected_arrow_schema = pa.Table.from_pandas(
        expected,
        preserve_index=False,
    ).schema

    api._publish_native_export_v2(
        database="miiv",
        data_path="/raw/source-must-not-be-read",
        output_dir=str(tmp_path),
        modules=["neurological"],
        max_patients=None,
        result=_completed_result("neurological"),
    )

    exported = pd.read_parquet(tmp_path / "neurological.parquet")
    pd.testing.assert_frame_equal(exported, expected)
    assert pq.read_schema(tmp_path / "neurological.parquet").equals(
        expected_arrow_schema,
        check_metadata=True,
    )
    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    entry = manifest["files"][0]
    assert entry["time_axis_audit"] == expected_time_audit
    assert {
        key: entry["row_grain_audit"][key]
        for key in expected_grain_audit
    } == expected_grain_audit
    assert {
        concept: {
            "excluded_out_of_bounds": status["excluded_out_of_bounds"],
            **(
                {"declared_bounds": status["declared_bounds"]}
                if "declared_bounds" in status
                else {}
            ),
        }
        for concept, status in entry["concept_status"].items()
    } == expected_bounds_audit


def test_large_duplicate_grain_fails_closed_before_unbounded_pandas_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "EXTRACT_MODULES", {"sepsis_shared": ["samp"]})
    monkeypatch.setattr(api, "_NATIVE_EXPORT_PANDAS_FALLBACK_MAX_ROWS", 1)
    source = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [None, None],
            "samp": [False, True],
        }
    )
    path = tmp_path / "sepsis_shared.parquet"
    source.to_parquet(path, index=False)
    original_digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="exceeds the bounded pandas fallback"):
        api._publish_native_export_v2(
            database="eicu",
            data_path="/raw/source-must-not-be-read",
            output_dir=str(tmp_path),
            modules=["sepsis_shared"],
            max_patients=None,
            result=_completed_result("sepsis_shared"),
        )

    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_digest
    assert not (tmp_path / "_manifest.json").exists()
    assert not (tmp_path / ".sepsis_shared.native-v2.tmp.parquet").exists()
