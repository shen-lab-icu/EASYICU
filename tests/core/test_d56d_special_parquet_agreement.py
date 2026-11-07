"""Special readers consume the owner's whole suffix-aware inventory.

Each suffix case fails at 01af932fe; successful and typed-refusal agreement
are checked together. All tables are synthetic.
"""

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pytest

from easyicu import storage_layout as sl
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource


@pytest.mark.parametrize("database,backend", [("eicu", "duckdb"), ("mimic", "duckdb"), ("mimic", "arrow")])
@pytest.mark.parametrize("suffix", [".pq", ".PQ", ".PARQUET", ".PaRqUeT"])
def test_owner_resolver_and_special_reader_agree_on_every_suffix(tmp_path, monkeypatch, database, backend, suffix):
    table = "intakeoutput" if database == "eicu" else "chartevents"
    directory = tmp_path / table
    directory.mkdir()
    if database == "eicu":
        frame = pd.DataFrame({
            "patientunitstayid": range(100, 120), "intakeoutputid": range(20),
            "intakeoutputoffset": range(20), "cellpath": ["synthetic"] * 20,
            "cellvaluenumeric": [60.0] * 20,
        })
        columns = ["patientunitstayid", "cellvaluenumeric"]
    else:
        frame = pd.DataFrame({
            "icustay_id": range(100, 120), "itemid": [220045] * 20,
            "value": ["60"] * 20, "error": [0] * 20,
        })
        columns = ["icustay_id", "itemid", "value"]
    # Both the variant and conventional suffix must be consumed, in order.
    frame.iloc[:10].to_parquet(directory / ("part-a" + suffix), index=False)
    frame.iloc[10:].to_parquet(directory / "part-b.parquet", index=False)
    source = ICUDataSource(load_src_cfg(database), base_path=tmp_path, enable_cache=False)
    decision = sl.table_decision(source, table)
    resolved = source.resolve_target_files(table)
    assert decision.files == resolved
    consumed = []
    inspected = []
    relation = sl.parquet_relation
    schemas = source._get_parquet_columns_for_files
    dataset = ds.dataset

    def observe_relation(files, **kwargs):
        files = tuple(Path(p) for p in files)
        consumed.append(files)
        return relation(files, **kwargs)

    def observe_schemas(files):
        files = tuple(Path(p) for p in files)
        inspected.append(files)
        return schemas(files)

    def observe_dataset(files, **kwargs):
        consumed.append(tuple(Path(p) for p in files))
        return dataset(files, **kwargs)

    monkeypatch.setattr(sl, "parquet_relation", observe_relation)
    monkeypatch.setattr(source, "_get_parquet_columns_for_files", observe_schemas)
    monkeypatch.setattr(ds, "dataset", observe_dataset)
    if backend == "arrow":
        # A legacy fallback adapter may forward only columns and predicates.
        # Optional quality projection must already be retained at dispatch.
        monkeypatch.setattr(source, "_read_partitioned_data_duckdb", lambda *args, **kwargs:
                            source._read_partitioned_data_optimized(
                                *args[:3], itemid_filter_config=kwargs.get("itemid_filter_config"),
                            ))
    count = len(source._load_raw_frame(table, columns))
    assert count == len(frame)
    assert consumed and all(files == resolved for files in consumed)
    if database == "mimic":
        assert inspected and all(files == resolved for files in inspected)
    assert tuple(Path(p) for p in sl.enumerate_parquet_files(directory)) == resolved
    assert sl.flat_table_files(source, table) == resolved

    # Broken inventory must be the same typed refusal at all three entries.
    (directory / "locked").mkdir()
    (directory / "locked").chmod(0)
    try:
        broken = sl.table_decision(source, table)
        assert broken.state == sl.PRESENCE_BROKEN
        for read in (lambda: source.resolve_target_files(table),
                     lambda: source._load_raw_frame(table, columns)):
            with pytest.raises(sl.StorageIntegrityError) as error:
                read()
            assert error.value.reason == broken.error.reason == sl.FILE_UNREADABLE
    finally:
        (directory / "locked").chmod(0o700)
