"""Published module bytes must not depend on the execution profile.

Streamed extraction interleaves patients across batches (density balancing)
and appends each batch to the module file, so the published row order used to
follow the batch size, and the batch size follows the memory budget. A 14 GB
laptop and a large server then wrote different bytes for equal values, and a
computation-order -0.0 could replace +0.0. Publication now sorts every module
file by its row keys (then every column) and writes +0.0 for zero.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, frame, metadata):
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **metadata})
    pq.write_table(table, path, compression="snappy")


def test_canonical_order_is_independent_of_input_row_order_and_signed_zero(tmp_path):
    from easyicu.api.extraction import _canonicalise_native_export_row_order

    frame = pd.DataFrame({
        "stay_id": pd.array([2, 1, 2, 1, 3], dtype="int64"),
        "charttime": [1.0, 0.0, 0.0, 1.0, np.nan],
        "anion_gap": [-0.0, 12.0, np.nan, 0.0, 5.5],
        "flag": pd.array([True, None, False, True, None], dtype="boolean"),
        "label": ["b", "a", None, "a", "c"],
    })
    metadata = {b"easyicu_probe": b"kept"}
    first, second = tmp_path / "first.parquet", tmp_path / "second.parquet"
    _write(first, frame, metadata)
    permuted = frame.iloc[[4, 2, 0, 3, 1]].copy()
    permuted["anion_gap"] = permuted["anion_gap"].replace(0.0, -0.0)
    _write(second, permuted, metadata)
    assert _sha(first) != _sha(second)

    for path in (first, second):
        _canonicalise_native_export_row_order(path, module="chemistry")

    assert _sha(first) == _sha(second)
    published = pq.read_table(first)
    assert published.schema.metadata[b"easyicu_probe"] == b"kept"
    assert published.schema == pq.read_table(tmp_path / "first.parquet").schema
    out = published.to_pandas()
    assert out["stay_id"].tolist() == [1, 1, 2, 2, 3]
    assert out["charttime"].tolist()[:4] == [0.0, 1.0, 0.0, 1.0]
    zeros = out["anion_gap"].to_numpy()
    assert not np.signbit(zeros[np.nan_to_num(zeros, nan=1.0) == 0.0]).any()
    pd.testing.assert_frame_equal(
        out,
        frame.sort_values(["stay_id", "charttime"], na_position="last")
        .reset_index(drop=True)
        .assign(anion_gap=lambda f: f["anion_gap"] + 0.0),
        check_exact=True,
    )


def test_canonical_order_breaks_key_ties_on_every_column(tmp_path):
    from easyicu.api.extraction import _canonicalise_native_export_row_order

    frame = pd.DataFrame({"stay_id": [1, 1, 1], "value": [3.0, 1.0, 2.0]})
    first, second = tmp_path / "a.parquet", tmp_path / "b.parquet"
    _write(first, frame, {})
    _write(second, frame.iloc[::-1], {})
    for path in (first, second):
        _canonicalise_native_export_row_order(path, module="outcome")
    assert _sha(first) == _sha(second)
    assert pq.read_table(first).column("value").to_pylist() == [1.0, 2.0, 3.0]


def test_streamed_export_bytes_do_not_depend_on_batch_size_or_workers(tmp_path, monkeypatch):
    from easyicu.api.concepts import clear_global_loader
    from easyicu.api.extraction import EXTRACT_MODULE_ORDER, extract_database
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / "source"
    generate_source(root, "hirid", stays=6)
    stays = list(range(1, 7))
    outputs = {}
    try:
        for label, batch_size, workers in (("one_thread", 1, 1), ("four_threads", 6, 4)):
            monkeypatch.setenv("EASYICU_DUCKDB_THREADS", str(workers))
            monkeypatch.setenv("EASYICU_ARROW_THREADS", str(workers))
            clear_global_loader()
            result = extract_database(
                "hirid",
                data_path=str(root),
                output_dir=str(tmp_path / label),
                modules=list(EXTRACT_MODULE_ORDER),
                patient_ids={"patientid": stays},
                batch_size=batch_size,
                max_workers=workers,
                native_export_v2=True,
                stream_output_batches=True,
                resource_budget_mb=2048,
                verbose=False,
            )
            assert all(not state["errors"] for state in result["modules"].values())
            outputs[label] = tmp_path / label
    finally:
        clear_global_loader()
    differing = [
        module
        for module in EXTRACT_MODULE_ORDER
        if _sha(outputs["one_thread"] / f"{module}.parquet")
        != _sha(outputs["four_threads"] / f"{module}.parquet")
    ]
    assert differing == []


@pytest.mark.parametrize("module", ["vitals", "outcome"])
def test_publication_calls_the_canonical_order_for_every_module(module):
    import inspect

    from easyicu.api import extraction

    source = inspect.getsource(extraction._publish_native_export_v2)
    assert "_canonicalise_native_export_row_order(" in source
