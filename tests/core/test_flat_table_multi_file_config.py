"""A table configured as several Parquet files resolves to all of them.

MIIV registers ``labevents`` as ``labevents/1.parquet`` .. ``10.parquet`` and
``chartevents`` as 30 files.  The storage-policy owner once returned the first
configured file alone, so DuckDB aggregations read a tenth of labevents and
most lab concepts came back empty without an error (lead-r13).  ``load_table``
always read the directory; every reader must see the same rows.
"""

from types import SimpleNamespace

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from easyicu.config import TableConfig
from easyicu.storage_layout import (
    TableScanError,
    flat_table_files,
    parquet_relation,
    verify_flat_scan,
)


def _source(base, files):
    table = TableConfig(name="labevents", files=files)
    config = SimpleNamespace(name="miiv", get_table=lambda _name: table)
    return SimpleNamespace(base_path=base, config=config)


def _write_chunks(base, count):
    directory = base / "labevents"
    directory.mkdir(parents=True)
    for index in range(1, count + 1):
        pq.write_table(
            pa.table({"itemid": [50000 + index], "valuenum": [float(index)]}),
            directory / f"{index}.parquet",
        )
    return [f"labevents/{index}.parquet" for index in range(1, count + 1)]


def test_multi_file_config_resolves_every_chunk(tmp_path):
    files = _write_chunks(tmp_path, 10)
    resolved = flat_table_files(_source(tmp_path, files), "labevents")
    assert len(resolved) == 10
    assert {path.name for path in resolved} == {f"{i}.parquet" for i in range(1, 11)}
    with duckdb.connect() as conn:
        items = conn.execute(
            f"SELECT itemid FROM {parquet_relation(resolved)}"
        ).fetchall()
    assert sorted(row[0] for row in items) == list(range(50001, 50011))


def test_single_file_config_keeps_its_file(tmp_path):
    pq.write_table(pa.table({"itemid": [1]}), tmp_path / "labevents.parquet")
    resolved = flat_table_files(_source(tmp_path, "labevents.parquet"), "labevents")
    assert resolved == (tmp_path / "labevents.parquet",)


def test_csv_multi_file_config_falls_back_to_the_parquet_directory(tmp_path):
    # HiRID registers observation_tables/csv/part-*.csv; installations hold
    # the converted Parquet under the table name.
    _write_chunks(tmp_path, 3)
    files = [f"observation_tables/csv/part-{i}.csv" for i in range(3)]
    resolved = flat_table_files(_source(tmp_path, files), "labevents")
    assert len(resolved) == 3


# --- runtime invariant: a flat scan covers the table's whole inventory -------


def test_scan_of_the_first_chunk_alone_fails_closed(tmp_path):
    files = _write_chunks(tmp_path, 10)
    source = _source(tmp_path, files)
    with pytest.raises(TableScanError) as caught:
        verify_flat_scan(source, "labevents", (tmp_path / "labevents" / "1.parquet",))
    assert caught.value.reason == "table_scan_incomplete"
    assert "1 files/1 rows of 10 files/10 rows" in str(caught.value)


def test_chunks_named_unlike_the_registry_list_are_read_whole(tmp_path):
    # The registry lists ricu's chunk names; an installation may name them
    # differently.  The directory inventory, not the list, is the table.
    directory = tmp_path / "labevents"
    directory.mkdir()
    for index in range(3):
        pq.write_table(pa.table({"itemid": [index]}), directory / f"part-{index}.parquet")
    source = _source(tmp_path, [f"labevents/{i}.parquet" for i in range(1, 11)])
    resolved = verify_flat_scan(source, "labevents", flat_table_files(source, "labevents"))
    assert len(resolved) == 3


def test_nested_parquet_outside_the_scan_fails_closed(tmp_path):
    files = _write_chunks(tmp_path, 2)
    nested = tmp_path / "labevents" / "part=3"
    nested.mkdir()
    pq.write_table(pa.table({"itemid": [9], "valuenum": [9.0]}), nested / "3.parquet")
    source = _source(tmp_path, files)
    with pytest.raises(TableScanError):
        verify_flat_scan(source, "labevents", flat_table_files(source, "labevents"))


def test_hidden_unpublished_files_and_extra_installed_chunks_are_fine(tmp_path):
    files = _write_chunks(tmp_path, 3)
    extra = tmp_path / "labevents" / "4.parquet"  # newer release, more chunks
    pq.write_table(pa.table({"itemid": [4], "valuenum": [4.0]}), extra)
    hidden = tmp_path / "labevents" / ".easyicu-unpublished"
    hidden.mkdir()
    pq.write_table(pa.table({"itemid": [5], "valuenum": [5.0]}), hidden / "x.parquet")
    source = _source(tmp_path, files)
    resolved = verify_flat_scan(source, "labevents", flat_table_files(source, "labevents"))
    assert len(resolved) == 4


def test_single_file_table_beside_other_tables_is_complete(tmp_path):
    pq.write_table(pa.table({"itemid": [1]}), tmp_path / "labevents.parquet")
    pq.write_table(pa.table({"itemid": [2]}), tmp_path / "chartevents.parquet")
    source = _source(tmp_path, "labevents.parquet")
    assert verify_flat_scan(source, "labevents", flat_table_files(source, "labevents")) == (
        tmp_path / "labevents.parquet",
    )
