"""Native-v2 sealing for the high-throughput grouped module extractor."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu import api
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

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert manifest["concept_selection"]["modules"] == {"demographics": ["age"]}
    assert manifest["unavailable_modules"] == [
        {
            "module": "sepsis_shared",
            "reason": "producer_returned_no_physical_output",
            "concept_ids": ["susp_inf"],
        }
    ]
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
    with open_export_package(output) as package:
        assert set(package.concept_index) == {"mort_28d"}
