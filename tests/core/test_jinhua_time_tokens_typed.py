"""Jinhua relative-minute columns: a mixed column is refused, never guessed.

Jinhua declares its ``*_base`` time columns as relative minute offsets. A
column Arrow inferred as strings used to be read as numbers only when every
non-blank token parsed, and otherwise as calendar datetimes, so one stray token
in a batch changed the reading of every row in it (magnitude/format sweep #4,
lead-r13). A mix of numeric and non-numeric tokens is now a typed error. The
all-numeric case is unchanged. Synthetic rows only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource, SourceTimeFormatError
from easyicu.storage_layout import StorageIntegrityError


def _jinhua(root: Path, charttime_base) -> ICUDataSource:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "subject_id": [1], "hadm_id": ["IP1"], "stay_id": ["IP1"],
        "intime": [1.0], "outtime": [12.0],
    }).to_parquet(root / "stays.parquet", index=False)
    events = root / "vital_signs"
    events.mkdir()
    pd.DataFrame({
        "hadm_id": ["IP1"] * len(charttime_base),
        "charttime_base": pd.Series(charttime_base, dtype="string"),
        "subcategory_name_en": ["Pulse"] * len(charttime_base),
        "value": [88.0] * len(charttime_base),
        "subcategory_unit": ["bpm"] * len(charttime_base),
    }).to_parquet(events / "part-00000.parquet", index=False)
    (root / "community_preparation_manifest.json").write_text(
        json.dumps({"database": "jinhua"}), encoding="utf-8"
    )
    return ICUDataSource(load_src_cfg("jinhua"), base_path=root, enable_cache=False)


def _load(source):
    table = source.load_table("vital_signs", verbose=False)
    return table.data if hasattr(table, "data") else table


def test_numeric_string_offsets_stay_numeric_and_blanks_stay_missing(tmp_path):
    frame = _load(_jinhua(tmp_path / "j", ["120", " ", "180"]))
    values = pd.to_numeric(frame["charttime_base"], errors="coerce")
    assert pd.api.types.is_numeric_dtype(frame["charttime_base"])
    assert values.tolist()[0::2] == [120.0, 180.0] and values.isna().sum() == 1


def test_a_column_mixing_offsets_and_other_tokens_is_refused_typed(tmp_path):
    with pytest.raises(SourceTimeFormatError) as caught:
        _load(_jinhua(tmp_path / "j", ["120", "not-an-offset", "180"]))
    error = caught.value
    assert isinstance(error, StorageIntegrityError) and not isinstance(error, ValueError)
    assert error.reason == "mixed_time_tokens"
    assert (error.table, error.column) == ("vital_signs", "charttime_base")
    assert "2 numeric and 1 non-numeric" in str(error)
