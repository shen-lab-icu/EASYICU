"""delta47d: reader, cohort and resolver must take one complete source decision.

All data are synthetic. Each regression fails at 761fd4c53; independent-copy
and absent-CSV controls are included in the regression that protects them.
"""

import os
import shutil
import types

import pandas as pd
import pytest

from easyicu import storage_binding as sb
from easyicu import storage_layout as sl
from easyicu.api import get_all_patient_ids
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from tests.core.test_d39d_table_source_matcher import IDS, _bucket, _parquet


def _source(root, database="mimic"):
    return ICUDataSource(load_src_cfg(database), base_path=root, enable_cache=False)


def _csv(root, database="mimic"):
    column = "patientid" if database == "hirid" else "icustay_id"
    name = "general_table.csv" if database == "hirid" else "icustays.csv"
    pd.DataFrame({column: [900, 901], "value": [99.0, 99.0]}).to_csv(root / name, index=False)


def _readers(root, database="mimic"):
    table = "general" if database == "hirid" else "icustays"
    column = "patientid" if database == "hirid" else "icustay_id"
    source = _source(root, database)
    listing = {}
    cohort, _ = get_all_patient_ids(root, database=database, listing=listing)
    raw = source._load_raw_frame(table, None)
    files = source.resolve_target_files(table)
    resolved = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    assert listing["source_format"] == "parquet"
    assert cohort == raw[column].tolist() == resolved[column].tolist() == IDS
    assert raw["value"].tolist() == resolved["value"].tolist() == [60.0] * len(IDS)


def _refusals(root, database="mimic"):
    source = _source(root, database)
    table = "general" if database == "hirid" else "icustays"
    calls = (
        lambda: source._load_raw_frame(table, None),
        lambda: get_all_patient_ids(root, database=database),
        lambda: source.resolve_target_files(table),
    )
    reasons = []
    for call in calls:
        try:
            call()
        except sl.StorageIntegrityError as exc:
            reasons.append(exc.reason)
        else:
            reasons.append("accepted")
    return reasons


@pytest.mark.parametrize("relative", [
    "custom/icustays.parquet", "custom/deeper/icustays.pq",
    "custom/deeper/IcuStays/part.parquet",
])
def test_one_nested_source_is_read_by_every_entry_point(tmp_path, relative):
    _csv(tmp_path)
    _parquet(tmp_path / relative)
    _readers(tmp_path)


def test_root_and_independent_nested_source_are_refused_everywhere(tmp_path):
    _csv(tmp_path)
    _parquet(tmp_path / "icustays.parquet")
    _parquet(tmp_path / "custom" / "icustays.parquet", range(200, 220))
    assert _refusals(tmp_path) == [sl.SOURCE_AMBIGUOUS] * 3


@pytest.mark.parametrize("relative", [
    "General/part.parquet", "GENERAL_TABLE/part.parquet", "custom/general.parquet",
])
def test_configured_hirid_csv_is_used_only_when_parquet_is_absent(tmp_path, relative):
    _csv(tmp_path, "hirid")
    source = _source(tmp_path, "hirid")
    assert source._load_raw_frame("general", None)["patientid"].tolist() == [900, 901]
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"patientid": IDS, "value": [60.0] * len(IDS)}).to_parquet(path, index=False)
    _readers(tmp_path, "hirid")


@pytest.mark.parametrize("denial", ["chmod", "listing_error"])
@pytest.mark.parametrize("location", ["custom", "icustays/locked", "GENERAL_TABLE/locked"])
def test_unreadable_nested_directory_never_becomes_absent_csv(tmp_path, monkeypatch, denial, location):
    database = "hirid" if location.startswith("GENERAL_TABLE/") else "mimic"
    _csv(tmp_path, database)
    blocked = tmp_path / location
    _parquet(blocked / "deeper" / "icustays.parquet")
    if location.startswith("icustays/"):
        _parquet(tmp_path / "icustays" / "part.parquet")
    elif database == "hirid":
        pd.DataFrame({"patientid": IDS, "value": [60.0] * len(IDS)}).to_parquet(
            tmp_path / "GENERAL_TABLE" / "part.parquet", index=False,
        )
    if denial == "chmod":
        blocked.chmod(0)
    else:
        real = sl.os.listdir

        def listing(path):
            if os.path.realpath(path) == str(blocked):
                raise PermissionError("synthetic listing denial")
            return real(path)

        monkeypatch.setattr(sl.os, "listdir", listing)
    try:
        assert _refusals(tmp_path, database) == [sl.FILE_UNREADABLE] * 3
    finally:
        blocked.chmod(0o700)


def test_a_supplied_loader_cannot_override_the_shared_absent_decision(tmp_path):
    _csv(tmp_path)
    other = tmp_path / "custom" / "unrelated.parquet"
    _parquet(other)
    with pytest.raises(sl.TableScanError) as error:
        _source(tmp_path).resolve_target_files("icustays", loader=other)
    assert error.value.reason == "table_scan_loader_mismatch"


def test_verified_derived_pair_uses_inode_even_when_alias_sorts_first(tmp_path):
    _csv(tmp_path)
    flat = tmp_path / "icustays.parquet"
    _parquet(flat)
    _bucket(tmp_path, derived=True)
    alias = tmp_path / "ICUSTAYS.parquet"
    os.link(flat, alias)
    _readers(tmp_path)
    alias.unlink()
    shutil.copyfile(flat, alias)
    assert _refusals(tmp_path) == [sl.SOURCE_AMBIGUOUS] * 3


@pytest.mark.parametrize("bind_alias", [True, False])
def test_racy_csv_alias_stays_unmemoized_after_the_window(tmp_path, monkeypatch, bind_alias):
    path = tmp_path / "table.csv"
    path.write_text("value\n60\n")
    alias = tmp_path / "alias.csv"
    alias.symlink_to(path)
    bound, scanned = (alias, path) if bind_alias else (path, alias)
    version = sb.stat_identity("t", bound)
    clock = [max(version[1], version[2])]
    monkeypatch.setattr(sb, "time", types.SimpleNamespace(time_ns=lambda: clock[0]))
    monkeypatch.setattr(ICUDataSource, "_CSV_VALIDATED", set())

    class Scan:
        def execute(self, query):
            return self

        def fetchone(self):
            return (1, 1)

    with sb.binding_unit():
        sb.consume("t", [bound], sl.FORMAT_CSV)
        clock[0] += 3_000_000_000
        ICUDataSource._validate_mimic3_csv(Scan(), scanned, "t", ["value"], "")
        with sb.capturing_consumed() as captured:
            sb.consume("t", [scanned], sl.FORMAT_CSV)
        assert not ICUDataSource._CSV_VALIDATED
        assert sb.captured_racily_clean(captured)
