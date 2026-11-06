"""Optional offline check: the variable × stay layout and batch order on the eICU demo.

A per-file symlink assembly of the converted public eICU demo (v2.0.1) gets
variable × stay installations for nursecharting, lab and infusiondrug, with
stay ranges scaled to the demo so file-level stay pruning is exercised. The
published module must be byte-identical to the flat source, and contiguous
and interleaved stay batches must publish the same bytes.

Set EASYICU_EICU_DEMO_RAW to the demo parquet directory. It is read only;
everything is written below pytest's temporary directory. Only hashes and
counts are compared, never rows.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from easyicu.api import extraction as api

RAW = Path(os.environ.get(
    "EASYICU_EICU_DEMO_RAW",
    str(Path.home() / ".easyicu" / "demo_sources" / "eicu_demo_v2_0_1" / "raw"),
))
TABLES = {
    "nursecharting": "nursingchartcelltypevallabel",
    "lab": "labname",
    "infusiondrug": "drugname",
}
BATCH_SIZE = 700


def _assembly(root: Path, *, variable: bool) -> Path:
    from scripts.build_itemid_bucket_cache import VARIABLE_PROFILES, build_variable_cache

    root.mkdir(parents=True)
    for path in RAW.iterdir():
        if path.is_file():
            (root / path.name).symlink_to(path)
    if variable:
        for table, key in TABLES.items():
            build_variable_cache(
                root, table=table, key=key, stay_key="patientunitstayid",
                secondary_keys=VARIABLE_PROFILES[("eicu", table)][1],
                rows_per_bucket=8192, memory_limit="1GB", threads=2,
            )
            manifest = json.loads((root / f"{table}_bucket" / "_BUCKET_MANIFEST.json").read_text())
            assert manifest["derived_from"]["files"][0]["path"] == f"{table}.parquet"
    return root


def _publish(data: Path, out: Path, monkeypatch, *, interleaved: bool) -> dict[str, str]:
    if interleaved:
        def plan(patient_ids, batch_size, _database):
            ordered, planned = api._interleave_stream_patient_ids(list(patient_ids), batch_size)
            return ordered, planned, "source_order_interleaved_v1"
        monkeypatch.setattr(api, "_order_stream_patient_ids", plan)
    else:
        monkeypatch.undo()
    result = api.extract_database(
        "eicu", data_path=str(data), output_dir=str(out),
        modules=["outcome", "circulatory"], native_export_v2=True,
        batch_size=BATCH_SIZE, stream_output_batches=True, verbose=False,
    )
    assert not result.get("storage_fallbacks")
    return {
        p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*.parquet"))
    }


@pytest.mark.slow
@pytest.mark.skipif(
    not (RAW / "patient.parquet").is_file(),
    reason="Converted public eICU demo parquet directory is absent",
)
def test_variable_layout_and_batch_order_publish_the_flat_bytes(tmp_path, monkeypatch):
    flat = _publish(_assembly(tmp_path / "flat", variable=False), tmp_path / "out_flat",
                    monkeypatch, interleaved=False)
    variable = _assembly(tmp_path / "variable", variable=True)
    contiguous = _publish(variable, tmp_path / "out_contiguous", monkeypatch, interleaved=False)
    interleaved = _publish(variable, tmp_path / "out_interleaved", monkeypatch, interleaved=True)
    assert flat and len(flat) == len(contiguous)
    assert contiguous == flat
    assert interleaved == contiguous
    ids = pd.read_parquet(RAW / "patient.parquet", columns=["patientunitstayid"])
    assert len(ids) > BATCH_SIZE  # more than one batch, so the two orders differ
