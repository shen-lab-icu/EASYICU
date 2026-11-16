"""Conversion-manifest paths use "/" on every OS.

v0.1.0 Windows acceptance: the converter wrote ``chartevents\\1.parquet`` on
Windows while the scan index compares ``chartevents/1.parquet``, so every
partitioned table was refused as ``table_scan_unregistered``.  The writer now
emits POSIX paths and readers normalise manifests written before the fix.
All rows are invented.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.datasource import ICUDataSource, load_wide_table_aggregated
from easyicu.resources import load_data_sources


def _vitals(ids):
    return pd.DataFrame({"patientunitstayid": ids, "observationoffset": [30] * len(ids),
                         "heartrate": [80.0] * len(ids)})


def test_a_manifest_written_on_windows_registers_its_partitioned_table(tmp_path):
    (tmp_path / "vitalperiodic").mkdir()
    for index, ids in enumerate(([1], [2]), start=1):
        _vitals(ids).to_parquet(tmp_path / "vitalperiodic" / f"{index}.parquet")
    (tmp_path / "conversion_manifest.json").write_text(json.dumps({"tables": [{
        "table": "vitalperiodic", "row_count": 2,
        "outputs": [{"relative_path": f"vitalperiodic\\{i}.parquet"} for i in (1, 2)],
    }]}))
    source = ICUDataSource(config=load_data_sources().get("eicu"), base_path=tmp_path)
    frame = load_wide_table_aggregated(source, "vitalperiodic", ["heartrate"])
    assert sorted(frame["patientunitstayid"]) == [1, 2]


def test_the_converter_records_relative_paths_with_forward_slashes(tmp_path):
    from easyicu.io.data_converter import DataConverter

    nested = tmp_path / "chartevents" / "1.parquet"
    nested.parent.mkdir()
    nested.write_bytes(b"x")
    converter = object.__new__(DataConverter)
    converter.data_path = tmp_path
    record = converter._file_manifest_record(nested, hash_files=False)
    assert record["relative_path"] == "chartevents/1.parquet"


def test_an_input_manifest_cannot_leave_its_root_through_backslashes(tmp_path):
    from easyicu.formal_inputs import InputSnapshotError, _relative

    with pytest.raises(InputSnapshotError):
        _relative(tmp_path, "..\\outside.parquet", tmp_path / "m.json")
    assert _relative(tmp_path, "a\\b.parquet", tmp_path / "m.json") == Path(tmp_path, "a", "b.parquet")
