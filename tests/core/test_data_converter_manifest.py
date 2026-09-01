"""DataConverter conversion manifests are evidence-bindable."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from easyicu.io.data_converter import DataConverter


def test_data_converter_writes_conversion_manifest_and_evidence(tmp_path: Path) -> None:
    data_dir = tmp_path / "miiv"
    data_dir.mkdir()
    csv_path = data_dir / "sample.csv"
    pd.DataFrame({"stay_id": [1, 2], "value": [3.0, 4.0]}).to_csv(
        csv_path,
        index=False,
    )

    evidence_root = tmp_path / "run"
    converter = DataConverter(
        data_dir,
        database="miiv",
        parallel_workers=1,
        verbose=False,
    )
    results = converter.convert_all(force=True, evidence_root=evidence_root)

    assert results["sample.csv"]["status"] == "completed"
    manifest_path = data_dir / "conversion_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "easyicu.conversion_manifest/1"
    assert manifest["database"] == "miiv"
    table = manifest["tables"][0]
    assert table["input"]["sha256"]
    assert table["outputs"][0]["sha256"]
    assert table["outputs"][0]["relative_path"] == "sample.parquet"

    aliases = json.loads(
        (evidence_root / "evidence" / "evidence_aliases.json").read_text(
            encoding="utf-8"
        )
    )
    assert "conversion_manifest" in aliases
