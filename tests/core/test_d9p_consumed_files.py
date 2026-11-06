"""The run binds what it actually reads: one consumed-files source (Codex delta9p B9P-CSV-BINDING).

On a converted installation a CSV read used to call the Parquet resolver as
an integrity precheck, and that precheck was what got bound and named. The
reader then read the CSV. Now the file actually read is recorded at the
point of reading (``storage_layout.record_consumed``), with its realpath and
format. It is bound, kept in the unit (``consumed_files``) and logged
(``read_consumed_log``). A precheck of another candidate binds nothing.

Consumed CSV and flat Parquet files must resolve under the data root, which
is the real directory of the conversion manifest. Otherwise a declared view
must admit them. A symlink farm declares its view through ``derived_from``
{path, sha256} of the source installation's conversion manifest. A link is
admitted when that manifest still has the sha256 and lists the link target
for the same table with the same size_bytes. Anything else is
``storage_read_outside_data_root``. Clock and identity tables that a query
opens by name are recorded the same way. Rows are synthetic.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from easyicu.config import DataSourceRegistry
from easyicu.datasource import FilterOp, FilterSpec, ICUDataSource, _sic_cases_relation_sql
from easyicu.storage_binding import (
    StorageSourceChanged,
    binding_unit,
    child_binding,
    consumed_files,
)
from easyicu.storage_layout import (
    SCAN_FALLBACK_LOG_ENV,
    TableScanError,
    admit_consumed_realpath,
    declared_view_chain,
    read_consumed_log,
)

_SERVICES = pd.DataFrame({
    "ROW_ID": [1, 2], "SUBJECT_ID": [10, 11], "HADM_ID": [100, 101],
    "TRANSFERTIME": ["2100-01-01 00:00:00", "2100-01-02 00:00:00"],
    "CURR_SERVICE": ["MED", "SURG"],
})
_ICUSTAYS = pd.DataFrame({
    "subject_id": [10, 11], "hadm_id": [100, 101], "icustay_id": [1000, 1001],
    "intime": pd.to_datetime(["2100-01-01", "2100-01-02"]),
    "outtime": pd.to_datetime(["2100-01-03", "2100-01-04"]),
})


def _config(name: str):
    path = Path(__file__).resolve().parents[2] / "src/easyicu/data/data-sources.json"
    return DataSourceRegistry.from_json(path).get(name)


def _manifest(root: Path, tables: dict[str, int], **extra) -> Path:
    entries = [
        {"table": t, "row_count": rows,
         "outputs": [{"relative_path": f"{t}.parquet",
                      "size_bytes": (root / f"{t}.parquet").stat().st_size}]}
        for t, rows in tables.items()
    ]
    path = root / "conversion_manifest.json"
    path.write_text(json.dumps({"schema_version": "easyicu.conversion_manifest/1",
                                "tables": entries, **extra}))
    return path


def _converted_mimic(root: Path, csv_target: Path | None = None) -> Path:
    """A converted MIMIC-III install: services as Parquet and as CSV beside it."""
    root.mkdir(parents=True)
    _SERVICES.to_parquet(root / "services.parquet", index=False)
    _ICUSTAYS.to_parquet(root / "icustays.parquet", index=False)
    if csv_target is None:
        _SERVICES.to_csv(root / "SERVICES.csv.gz", index=False)
    else:
        _SERVICES.to_csv(csv_target, index=False)
        (root / "SERVICES.csv.gz").symlink_to(csv_target)
    _manifest(root, {"services": 2, "icustays": 2})
    return root


def _services(source):
    return source.load_table(
        "services", columns=["hadm_id", "curr_service"],
        filters=[FilterSpec(column="hadm_id", op=FilterOp.IN, value=[100])],
    ).data


def _touch(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


def test_a_preferred_csv_is_what_the_binding_and_the_event_name(tmp_path, monkeypatch):
    root = _converted_mimic(tmp_path / "mimic")
    log = tmp_path / "run.log"
    monkeypatch.setenv(SCAN_FALLBACK_LOG_ENV, str(log))
    with binding_unit():
        assert _services(ICUDataSource(_config("mimic"), base_path=root))["curr_service"].tolist() == ["MED"]
        consumed = [e for e in consumed_files() if e["table"] == "services"]
    csv = str(root / "SERVICES.csv.gz")
    assert consumed == [{"table": "services", "path": csv, "realpath": os.path.realpath(csv),
                         "format": "csv"}]
    events = [e for e in read_consumed_log(log) if e["table"] == "services"]
    assert [(e["path"], e["format"]) for e in events] == [(csv, "csv")]


def test_only_the_file_read_is_bound_when_csv_and_parquet_sit_side_by_side(tmp_path):
    root = _converted_mimic(tmp_path / "mimic")
    source = ICUDataSource(_config("mimic"), base_path=root, enable_cache=False)
    with binding_unit():
        _services(source)
        snapshot = child_binding(source)
        assert str(root / "services.parquet") not in snapshot["versions"]
        _touch(root / "services.parquet")  # the precheck candidate: not read, not bound
        _services(source)
        _touch(root / "SERVICES.csv.gz")  # what is read: bound
        with pytest.raises(StorageSourceChanged):
            _services(source)


def test_a_csv_symlink_outside_the_data_root_is_refused(tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    root = _converted_mimic(tmp_path / "mimic", csv_target=outside / "SERVICES.csv.gz")
    with binding_unit(), pytest.raises(TableScanError) as error:
        _services(ICUDataSource(_config("mimic"), base_path=root))
    assert error.value.reason == "storage_read_outside_data_root"


def _farm(tmp_path: Path, *, size_skew: int = 0, unlisted: bool = False) -> tuple[Path, Path]:
    """A CareVue-style symlink farm declaring its view through derived_from."""
    source = tmp_path / "view"
    source.mkdir()
    _ICUSTAYS.to_parquet(source / "icustays.parquet", index=False)
    _SERVICES.to_parquet(source / "services.parquet", index=False)
    source_manifest = _manifest(source, {"icustays": 2} if unlisted else {"icustays": 2, "services": 2})
    farm = tmp_path / "farm"
    farm.mkdir()
    for name in ("icustays.parquet", "services.parquet"):
        (farm / name).symlink_to(source / name)
    tables = [
        {"table": t, "row_count": 2,
         "outputs": [{"relative_path": f"{t}.parquet",
                      "size_bytes": (source / f"{t}.parquet").stat().st_size + size_skew}]}
        for t in ("icustays", "services")
    ]
    (farm / "conversion_manifest.json").write_text(json.dumps({
        "schema_version": "easyicu.conversion_manifest/1", "tables": tables,
        "derived_from": {"note": "symlink farm of the view", "path": str(source_manifest),
                         "sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest()},
    }))
    return farm, source_manifest


def test_a_declared_view_admits_its_links_and_names_its_chain(tmp_path):
    farm, source_manifest = _farm(tmp_path)
    admitted = admit_consumed_realpath(farm, "icustays", farm / "icustays.parquet")
    assert admitted["admitted_by"] == "declared_view"
    assert admitted["target_realpath"] == os.path.realpath(farm / "icustays.parquet")
    chain = declared_view_chain(farm)
    assert chain == {
        "farm_manifest_sha256": hashlib.sha256((farm / "conversion_manifest.json").read_bytes()).hexdigest(),
        "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
    }
    source = ICUDataSource(_config("mimic"), base_path=farm)
    with binding_unit():
        assert source.resolve_target_files("icustays")
        [entry] = [e for e in consumed_files() if e["table"] == "icustays"]
    assert entry["realpath"] == admitted["target_realpath"] and entry["format"] == "parquet"


def test_a_tampered_source_manifest_admits_nothing(tmp_path):
    farm, source_manifest = _farm(tmp_path)
    source_manifest.write_text(source_manifest.read_text() + " ")
    assert declared_view_chain(farm) is None
    with pytest.raises(TableScanError) as error:
        admit_consumed_realpath(farm, "icustays", farm / "icustays.parquet")
    assert error.value.reason == "storage_read_outside_data_root"


def test_a_link_the_source_manifest_does_not_list_is_refused(tmp_path):
    farm, _ = _farm(tmp_path, unlisted=True)
    assert admit_consumed_realpath(farm, "icustays", farm / "icustays.parquet")
    with pytest.raises(TableScanError):
        admit_consumed_realpath(farm, "services", farm / "services.parquet")


def test_a_size_mismatch_is_refused(tmp_path):
    farm, _ = _farm(tmp_path, size_skew=1)
    with pytest.raises(TableScanError):
        admit_consumed_realpath(farm, "icustays", farm / "icustays.parquet")


def test_a_clock_table_opened_by_name_is_recorded(tmp_path):
    root = tmp_path / "sic"
    root.mkdir()
    pd.DataFrame({"CaseID": [1], "ICUOffset": [0]}).to_parquet(root / "cases.parquet", index=False)
    source = ICUDataSource(_config("sic"), base_path=root)
    with binding_unit():
        assert "read_parquet" in _sic_cases_relation_sql(source)
        [entry] = consumed_files()
    assert (entry["table"], entry["format"]) == ("cases", "parquet")


# Codex delta10d D10D-VIEW-ADMISSION: admission is per (table, realpath) at
# the current size, and the view is read once per run, never across runs.


def test_a_warm_view_does_not_admit_a_link_for_another_table(tmp_path):
    farm, _ = _farm(tmp_path)
    with binding_unit():
        assert admit_consumed_realpath(farm, "icustays", farm / "icustays.parquet")
        with pytest.raises(TableScanError) as error:
            admit_consumed_realpath(farm, "services", farm / "icustays.parquet")
    assert error.value.reason == "storage_read_outside_data_root"


def test_a_warm_view_checks_the_current_size(tmp_path):
    farm, _ = _farm(tmp_path)
    with binding_unit():
        assert admit_consumed_realpath(farm, "icustays", farm / "icustays.parquet")
        with open(os.path.realpath(farm / "icustays.parquet"), "ab") as handle:
            handle.write(b"x")
        with pytest.raises(TableScanError):
            admit_consumed_realpath(farm, "icustays", farm / "icustays.parquet")


def test_a_changed_source_manifest_is_not_admitted_by_an_earlier_run(tmp_path):
    farm, source_manifest = _farm(tmp_path)
    with binding_unit():
        assert admit_consumed_realpath(farm, "icustays", farm / "icustays.parquet")
    source_manifest.write_text(source_manifest.read_text() + " ")  # farm manifest untouched
    with binding_unit(), pytest.raises(TableScanError):
        admit_consumed_realpath(farm, "icustays", farm / "icustays.parquet")


def test_a_source_manifest_changed_within_the_run_is_a_changed_source(tmp_path):
    farm, source_manifest = _farm(tmp_path)
    with binding_unit():
        assert admit_consumed_realpath(farm, "icustays", farm / "icustays.parquet")
        source_manifest.write_text(source_manifest.read_text() + " ")
        _touch(source_manifest)
        with pytest.raises(StorageSourceChanged):
            admit_consumed_realpath(farm, "icustays", farm / "icustays.parquet")


def test_a_clock_helper_link_is_admitted_like_any_consumed_file(tmp_path):
    view = tmp_path / "view"
    view.mkdir()
    pd.DataFrame({"CaseID": [1], "ICUOffset": [0]}).to_parquet(view / "cases.parquet", index=False)
    source_manifest = _manifest(view, {"cases": 1})
    farm = tmp_path / "farm"
    farm.mkdir()
    (farm / "cases.parquet").symlink_to(view / "cases.parquet")
    (farm / "conversion_manifest.json").write_text(json.dumps({
        "schema_version": "easyicu.conversion_manifest/1",
        "tables": [{"table": "cases", "row_count": 1, "outputs": [
            {"relative_path": "cases.parquet",
             "size_bytes": (view / "cases.parquet").stat().st_size}]}],
        "derived_from": {"path": str(source_manifest),
                         "sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest()},
    }))
    source = ICUDataSource(_config("sic"), base_path=farm)
    with binding_unit():
        assert "read_parquet" in _sic_cases_relation_sql(source)  # admitted: view is warm
    with open(view / "cases.parquet", "ab") as handle:
        handle.write(b"x")  # no longer the declared size
    with binding_unit(), pytest.raises(TableScanError) as error:
        _sic_cases_relation_sql(source)
    assert error.value.reason == "storage_read_outside_data_root"


# Codex delta10d: a warm raw-frame cache hit records the files it was read from.


def test_a_cached_csv_payload_is_recorded_again_on_a_hit(tmp_path):
    root = _converted_mimic(tmp_path / "mimic")
    source = ICUDataSource(_config("mimic"), base_path=root)
    csv = str(root / "SERVICES.csv.gz")
    with binding_unit():
        _services(source)
    with binding_unit():
        assert _services(source)["curr_service"].tolist() == ["MED"]
        consumed = [(e["table"], e["path"], e["format"]) for e in consumed_files()]
    assert ("services", csv, "csv") in consumed


def test_a_cached_payload_whose_file_changed_is_read_again(tmp_path):
    root = _converted_mimic(tmp_path / "mimic")
    source = ICUDataSource(_config("mimic"), base_path=root)
    with binding_unit():
        assert _services(source)["curr_service"].tolist() == ["MED"]
    changed = _SERVICES.assign(CURR_SERVICE=["CSURG", "SURG"])
    changed.to_csv(root / "SERVICES.csv.gz", index=False)
    _touch(root / "SERVICES.csv.gz")
    with binding_unit():
        assert _services(source)["curr_service"].tolist() == ["CSURG"]
