"""Native-v2 sealing for the high-throughput grouped module extractor."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.api import extraction as api
from easyicu.research_agent.intake.export_package import open_export_package


def _completed_result(module: str = "demographics") -> dict[str, object]:
    return {"modules": {module: {"errors": []}}}


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
    assert status["mort_28d"] == {"availability": "available", "non_null": 1}
    assert status["vent_free_days_28"] == {
        "availability": "structurally_unavailable_placeholder",
        "non_null": 0,
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
