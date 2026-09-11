"""Legacy exports apply declared category maps before cohort materialization."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.cohort import materializer
from easyicu.research_agent.intake import export_package as intake
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedMetadataError,
)


def _legacy_export(root: Path, values: list[object]) -> Path:
    root.mkdir()
    frame = pd.DataFrame({"stay_id": range(1, len(values) + 1), "adm": values})
    frame.to_parquet(root / "admission.parquet", index=False)
    (root / intake.LEGACY_MANIFEST).write_text(
        json.dumps(
            {
                "database": "miiv",
                "export_format": "parquet",
                "modules": [
                    {
                        "file": "admission.parquet",
                        "module": "admission",
                        "rows": len(frame),
                    }
                ],
                "exported_files": ["admission.parquet"],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_legacy_export_applies_declared_category_map_and_records_receipt(
    tmp_path: Path,
) -> None:
    root = _legacy_export(tmp_path / "export", ["EYE", "med", "other", "EYE"])

    cohort, provenance = materializer.materialize_cohort(
        data_path=root,
        database="miiv",
        feature_concepts=(),
        static_concepts=("adm",),
        outcome_concepts=(),
    )

    assert cohort["adm"].tolist() == ["surg", "med", "other", "surg"]
    receipt = provenance["legacy_export_domain_normalizations"]["adm"]
    assert receipt["policy"] == "declared_apply_map_for_legacy_export"
    assert receipt["declared_levels"] == ["med", "surg", "other"]
    assert receipt["raw_counts"]["EYE"] == 2
    assert receipt["normalized_counts"]["surg"] == 2
    assert receipt["changed_rows"] == 2
    assert len(receipt["callback_sha256"]) == 64
    assert (
        receipt["manifest_sha256"]
        == provenance["export_authority"]["manifest_sha256"]
    )


def test_legacy_export_still_fails_closed_for_unmapped_levels(
    tmp_path: Path,
) -> None:
    root = _legacy_export(tmp_path / "export", ["EYE", "UNKNOWN"])

    with pytest.raises(
        MaterializedMetadataError, match="still contains undeclared levels"
    ):
        materializer.materialize_cohort(
            data_path=root,
            database="miiv",
            feature_concepts=(),
            static_concepts=("adm",),
            outcome_concepts=(),
        )


def test_typed_export_is_not_rewritten_by_the_legacy_map() -> None:
    frame = pd.DataFrame({"stay_id": [1, 2], "adm": ["EYE", "med"]})
    package = SimpleNamespace(column_metadata_sha256="a" * 64)

    normalized, receipt = materializer._normalize_legacy_export_categorical(
        frame,
        concept="adm",
        database="miiv",
        package=package,
    )

    assert normalized is frame
    assert receipt is None
