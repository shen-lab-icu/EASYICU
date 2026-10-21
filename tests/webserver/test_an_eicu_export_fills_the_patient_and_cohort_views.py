"""An eICU export's patient and cohort views show its values.

An export keeps its database's own stay key (eICU ``patientunitstayid``). The
views name that key ``stay_id``; reading module files by the canonical name
alone found no key, so a real eICU export summarised to blank ages, scores and
outcomes ("Unknown", 0 survived / 0 deceased) although every module was full.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.webserver import dataio

_KEY = {"eicu": "patientunitstayid", "miiv": "stay_id"}


def _export(root: Path, database: str, fmt: str) -> Path:
    key = _KEY[database]
    tables = {
        "demographics": pd.DataFrame({key: [11, 12, 13], "age": [50.0, 70.0, 60.0], "sex": ["Female", "Male", "Female"]}),
        "outcome": pd.DataFrame({key: [11, 12, 13], "death": [False, True, False], "los_icu": [2.0, 5.0, 1.0]}),
        "sofa2_score": pd.DataFrame({key: [11, 11, 12], "charttime": [0.0, 60.0, 0.0], "sofa2": [4.0, 5.0, 8.0]}),
        "vitals": pd.DataFrame({key: [11, 12], "charttime": [0.0, 0.0], "hr": [90.0, 80.0], "map": [70.0, 75.0],
                                "spo2": [97.0, 96.0], "temp": [37.0, 36.8]}),
    }
    root.mkdir()
    files = []
    for module, frame in tables.items():
        name = f"{module}.{fmt}"
        if fmt == "parquet":
            frame.to_parquet(root / name, index=False)
        else:
            frame.to_csv(root / name, index=False)
        files.append({"file": name, "module": module, "rows": len(frame), "column_metadata_columns": list(frame.columns)})
    (root / "_manifest.json").write_text(json.dumps(
        {"database": database, "generated": "2026-10-21T16:39:10", "format": fmt, "files": files}), encoding="utf-8")
    return root


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
@pytest.mark.parametrize("database", ["eicu", "miiv"])
def test_the_views_read_every_module_under_the_databases_own_key(tmp_path: Path, database: str, fmt: str) -> None:
    result = dataio.summarize_export_workspace(str(_export(tmp_path / "export", database, fmt)))
    assert result["ok"] is True
    rows = {row["stay_id"]: row for row in result["tableRows"]}
    assert set(rows) == {"11", "12", "13"}
    assert (rows["11"]["age"], rows["11"]["sofa2"], rows["11"]["los_icu"]) == (50.0, 5.0, 2.0)
    assert rows["12"]["outcome"] != "Unknown" and rows["12"]["sofa2"] == 8.0
    cohort = result["cohort"]
    assert (cohort["survived"], cohort["deceased"]) == (2, 1)


def test_the_stay_key_reads_back_as_stay_id(tmp_path: Path) -> None:
    root = _export(tmp_path / "export", "eicu", "parquet")
    frame = dataio._read_export_frame(root / "outcome.parquet", columns=["stay_id", "death"], stay_ids={"11", "13"})
    assert list(frame.columns) == ["stay_id", "death"]
    assert sorted(frame["stay_id"].tolist()) == [11, 13]
    # Without a filter the whole file reads, key renamed.
    assert "patientunitstayid" not in dataio._read_export_frame(root / "demographics.parquet").columns
