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


# --- the conversion manifest is the inventory when present ------------------


def _write_manifest(base, chunks, rows):
    import json

    entry = {
        "table": "labevents",
        "row_count": rows,
        "outputs": [{"relative_path": f"labevents/{name}"} for name in chunks],
    }
    (base / "conversion_manifest.json").write_text(json.dumps({"tables": [entry]}))


def test_missing_converted_chunk_fails_against_the_conversion_manifest(tmp_path):
    files = _write_chunks(tmp_path, 10)
    _write_manifest(tmp_path, [f"{i}.parquet" for i in range(1, 11)], 10)
    (tmp_path / "labevents" / "7.parquet").unlink()  # interrupted conversion
    source = _source(tmp_path, files)
    with pytest.raises(TableScanError) as caught:
        verify_flat_scan(source, "labevents", flat_table_files(source, "labevents"))
    assert caught.value.reason == "table_scan_manifest_mismatch"
    assert "scanned 9 files/9 rows" in str(caught.value)


def test_truncated_converted_chunk_fails_on_rows(tmp_path):
    files = _write_chunks(tmp_path, 3)
    _write_manifest(tmp_path, ["1.parquet", "2.parquet", "3.parquet"], 4)
    source = _source(tmp_path, files)
    with pytest.raises(TableScanError) as caught:
        verify_flat_scan(source, "labevents", flat_table_files(source, "labevents"))
    assert "3 files/4 rows" in str(caught.value)


def test_complete_conversion_passes_through_a_symlink_farm(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    _write_chunks(real, 3)
    _write_manifest(real, ["1.parquet", "2.parquet", "3.parquet"], 3)
    farm = tmp_path / "farm"
    farm.mkdir()
    for entry in real.iterdir():
        (farm / entry.name).symlink_to(entry)
    source = _source(farm, [f"labevents/{i}.parquet" for i in range(1, 11)])
    assert len(verify_flat_scan(source, "labevents", flat_table_files(source, "labevents"))) == 3


def test_unmanifested_installation_falls_back_to_the_directory_with_a_warning(tmp_path, caplog):
    files = _write_chunks(tmp_path, 3)
    source = _source(tmp_path, files)
    with caplog.at_level("WARNING", logger="easyicu.storage_layout"):
        assert len(verify_flat_scan(source, "labevents", flat_table_files(source, "labevents"))) == 3
    assert "table_scan_unmanifested" in caplog.text


def test_a_table_converted_part_by_part_is_one_inventory(tmp_path):
    import json

    files = _write_chunks(tmp_path, 3)
    entries = [
        {"table": "labevents", "row_count": 1, "outputs": [{"relative_path": f"labevents/{i}.parquet"}]}
        for i in range(1, 4)
    ]
    (tmp_path / "conversion_manifest.json").write_text(json.dumps({"tables": entries}))
    source = _source(tmp_path, files)
    assert len(verify_flat_scan(source, "labevents", flat_table_files(source, "labevents"))) == 3
    (tmp_path / "labevents" / "2.parquet").unlink()
    with pytest.raises(TableScanError):
        verify_flat_scan(source, "labevents", flat_table_files(source, "labevents"))


def _farm_with_derived_manifest(tmp_path):
    import hashlib
    import json

    view = tmp_path / "view"
    view.mkdir()
    pq.write_table(pa.table({"itemid": [1, 2]}), view / "labevents.parquet")
    source = view / "conversion_manifest.json"
    source.write_text(json.dumps({"tables": [{
        "table": "labevents", "row_count": 2, "outputs": [{"relative_path": "labevents.parquet"}],
    }]}))
    farm = tmp_path / "farm"
    (farm / "labevents").mkdir(parents=True)
    (farm / "labevents" / "1.parquet").symlink_to(view / "labevents.parquet")
    (farm / "conversion_manifest.json").write_text(json.dumps({
        "tables": [{"table": "labevents", "row_count": 2,
                    "outputs": [{"relative_path": "labevents/1.parquet"}]}],
        "derived_from": {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
    }))
    return view, farm, source


def test_derived_manifest_passes_while_its_source_is_unchanged(tmp_path):
    _view, farm, _source_manifest = _farm_with_derived_manifest(tmp_path)
    source = _source(farm, "labevents.parquet")
    assert len(verify_flat_scan(source, "labevents", flat_table_files(source, "labevents"))) == 1


def test_derived_manifest_is_stale_once_its_source_changes(tmp_path):
    import json

    _view, farm, source_manifest = _farm_with_derived_manifest(tmp_path)
    source = _source(farm, "labevents.parquet")
    verify_flat_scan(source, "labevents", flat_table_files(source, "labevents"))  # cached
    payload = json.loads(source_manifest.read_text())
    payload["rebuilt"] = True  # same rows, rebuilt view
    source_manifest.write_text(json.dumps(payload))
    with pytest.raises(TableScanError) as caught:
        verify_flat_scan(source, "labevents", flat_table_files(source, "labevents"))
    assert caught.value.reason == "table_scan_manifest_stale"


def test_derived_manifest_with_a_missing_source_fails_closed(tmp_path):
    _view, farm, source_manifest = _farm_with_derived_manifest(tmp_path)
    source_manifest.unlink()
    source = _source(farm, "labevents.parquet")
    with pytest.raises(TableScanError) as caught:
        verify_flat_scan(source, "labevents", flat_table_files(source, "labevents"))
    assert caught.value.reason == "conversion_manifest_source_missing"


@pytest.mark.parametrize(
    "binding",
    [None, "elsewhere.json", ["x"], {"path": "x"}, {"path": "", "sha256": "0" * 64},
     {"path": "x", "sha256": "ABC"}],
)
def test_a_derived_binding_that_cannot_be_verified_is_not_unbound(tmp_path, binding):
    import json

    files = _write_chunks(tmp_path, 1)
    (tmp_path / "conversion_manifest.json").write_text(json.dumps({
        "tables": [{"table": "labevents", "row_count": 1,
                    "outputs": [{"relative_path": "labevents/1.parquet"}]}],
        "derived_from": binding,
    }))
    source = _source(tmp_path, files)
    with pytest.raises(TableScanError) as caught:
        verify_flat_scan(source, "labevents", flat_table_files(source, "labevents"))
    assert caught.value.reason == "conversion_manifest_derived_invalid"


# --- a manifest is the whole inventory (final review B-STORAGE-INVENTORY) ---


def test_a_registered_table_with_no_files_fails_instead_of_reading_empty(tmp_path):
    import shutil

    _write_chunks(tmp_path, 2)
    _write_manifest(tmp_path, ["1.parquet", "2.parquet"], 2)
    shutil.rmtree(tmp_path / "labevents")
    source = _source(tmp_path, ["labevents/1.parquet", "labevents/2.parquet"])
    assert flat_table_files(source, "labevents") == ()
    with pytest.raises(TableScanError) as caught:
        verify_flat_scan(source, "labevents", ())
    assert caught.value.reason == "table_scan_files_missing"


def test_an_unregistered_table_without_files_is_simply_absent(tmp_path):
    _write_chunks(tmp_path, 1)
    _write_manifest(tmp_path, ["1.parquet"], 1)
    source = _source(tmp_path, "chartevents.parquet")
    assert verify_flat_scan(source, "chartevents", ()) == ()


def _single_file_beside_a_manifest(tmp_path):
    _write_chunks(tmp_path, 2)
    _write_manifest(tmp_path, ["1.parquet", "2.parquet"], 2)
    pq.write_table(pa.table({"itemid": [7]}), tmp_path / "chartevents.parquet")
    return _source(tmp_path, "chartevents.parquet")


def test_files_no_manifest_entry_owns_fail_closed(tmp_path, monkeypatch):
    monkeypatch.delenv("EASYICU_ALLOW_UNREGISTERED_SCAN", raising=False)
    source = _single_file_beside_a_manifest(tmp_path)
    with pytest.raises(TableScanError) as caught:
        verify_flat_scan(source, "chartevents", flat_table_files(source, "chartevents"))
    assert caught.value.reason == "table_scan_unregistered"


def test_an_unregistered_scan_is_accepted_only_by_explicit_opt_in_and_recorded(tmp_path, monkeypatch):
    from easyicu.storage_layout import scan_fallbacks

    monkeypatch.setenv("EASYICU_ALLOW_UNREGISTERED_SCAN", "1")
    source = _single_file_beside_a_manifest(tmp_path)
    assert verify_flat_scan(source, "chartevents", flat_table_files(source, "chartevents")) == (
        tmp_path / "chartevents.parquet",
    )
    assert {"reason": "table_scan_unregistered", "root": str(tmp_path),
            "table": "chartevents"} in scan_fallbacks()


def test_the_no_manifest_fallback_is_recorded(tmp_path):
    from easyicu.storage_layout import scan_fallbacks

    files = _write_chunks(tmp_path, 2)
    source = _source(tmp_path, files)
    assert len(verify_flat_scan(source, "labevents", flat_table_files(source, "labevents"))) == 2
    assert {"reason": "table_scan_unmanifested", "root": str(tmp_path),
            "table": "labevents"} in scan_fallbacks()


def test_storage_integrity_errors_are_not_missing_source_errors():
    from easyicu.storage_layout import BucketLayoutError, StorageIntegrityError

    for kind in (TableScanError, BucketLayoutError):
        assert issubclass(kind, StorageIntegrityError)
        assert not issubclass(kind, (ValueError, KeyError, OSError))
