"""delta47d: reader, cohort and resolver must take one complete source decision.

All data are synthetic. Each regression fails at 761fd4c53; independent-copy
and absent-CSV controls are included in the regression that protects them.
"""

import os
import json
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


def _readers(root, database="mimic", source=None):
    table = "general" if database == "hirid" else "icustays"
    column = "patientid" if database == "hirid" else "icustay_id"
    source = source or _source(root, database)
    listing = {}
    with sb.binding_unit():
        cohort, _ = get_all_patient_ids(root, database=database, listing=listing)
        assert any(e["table"] == table and e["format"] in {sl.FORMAT_PARQUET, sl.FORMAT_BUCKET_PART}
                   for e in sb.consumed_files())
    raw = source._load_raw_frame(table, None)
    files = source.resolve_target_files(table)
    resolved = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    assert listing["source_format"] == "parquet"
    assert cohort == raw[column].tolist() == resolved[column].tolist() == IDS
    assert raw["value"].tolist() == resolved["value"].tolist() == [60.0] * len(IDS)


def _refusals(root, database="mimic", source=None):
    source = source or _source(root, database)
    table = "general" if database == "hirid" else "icustays"
    calls = (
        lambda: source._load_raw_frame(table, None),
        lambda: get_all_patient_ids(root, database=database),
        lambda: source.resolve_target_files(table),
        lambda: sl.flat_table_files(source, table),
        lambda: source.resolve_flat_parquet_directory(table),
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
    assert _refusals(tmp_path) == [sl.SOURCE_AMBIGUOUS] * 5


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
        assert _refusals(tmp_path, database) == [sl.FILE_UNREADABLE] * 5
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
    (tmp_path / sl.CONVERSION_MANIFEST).write_text(json.dumps({"tables": [{
        "table": "icustays", "row_count": len(IDS),
        "outputs": [{"relative_path": "icustays.parquet", "size_bytes": flat.stat().st_size}],
    }]}))
    alias = tmp_path / "ICUSTAYS.parquet"
    os.link(flat, alias)
    _readers(tmp_path)
    assert sl.flat_table_files(_source(tmp_path), "icustays") == (flat,)
    alias.unlink()
    shutil.copyfile(flat, alias)
    assert _refusals(tmp_path) == [sl.SOURCE_AMBIGUOUS] * 5


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


def test_services_automatic_selection_prefers_parquet_to_its_old_csv_exception(tmp_path):
    from tests.core.test_d9p_consumed_files import _converted_mimic, _services, _SERVICES

    root = _converted_mimic(tmp_path / "mimic")
    _SERVICES.assign(CURR_SERVICE=["CSURG", "NSURG"]).to_csv(root / "SERVICES.csv.gz", index=False)
    source = _source(root)
    assert _services(source)["curr_service"].tolist() == ["MED"]
    assert source.resolve_target_files("services") == (root / "services.parquet",)


@pytest.mark.parametrize("alias", [False, True])
def test_owner_selected_chart_csv_keeps_strict_validation_without_an_item_filter(tmp_path, alias):
    import copy
    from tests.core.test_d30d_racy_age import _csv as chart_csv

    path = chart_csv(tmp_path)
    config = copy.deepcopy(load_src_cfg("mimic"))
    if alias:
        path = path.rename(tmp_path / "custom_chart.csv")
        config.get_table("chartevents").files = [{"path": path.name}]
    path.write_text(path.read_text().replace(",60,bpm,", ",xx,bpm,", 1))
    with pytest.raises(sl.TableScanError) as error:
        ICUDataSource(config, base_path=tmp_path, enable_cache=False)._load_raw_frame(
            "chartevents", ["itemid", "value"],
        )
    assert error.value.reason == "csv_rows_rejected"


def test_owner_selected_chart_csv_alias_keeps_declared_value_strings(tmp_path):
    import copy
    from tests.core.test_d30d_racy_age import _csv as chart_csv

    path = chart_csv(tmp_path).rename(tmp_path / "custom_chart.csv")
    config = copy.deepcopy(load_src_cfg("mimic"))
    config.get_table("chartevents").files = [{"path": path.name}]
    frame = ICUDataSource(config, base_path=tmp_path, enable_cache=False)._load_raw_frame("chartevents", ["value"])
    assert frame["value"].tolist() == ["60"] * 5


def test_empty_flat_leftover_beside_a_healthy_standalone_bucket_is_refused(tmp_path):
    _parquet(tmp_path / "icustays.parquet")
    _bucket(tmp_path, derived=False)
    (tmp_path / "icustays.parquet").unlink()
    (tmp_path / "icustays").mkdir()
    assert _refusals(tmp_path) == [sl.LAYOUT_BROKEN] * 5


@pytest.mark.parametrize("relative", [
    "IcuStays.parquet", "icustays/part.PARQUET", "icustays/part.pq", "custom/icustays.pq",
])
def test_direct_flat_helper_and_directory_enumerator_share_the_owner(tmp_path, relative):
    _csv(tmp_path)
    _parquet(tmp_path / relative)
    source = _source(tmp_path)
    files = sl.flat_table_files(source, "icustays")
    assert files
    assert pd.concat([pd.read_parquet(p) for p in files])["icustay_id"].tolist() == IDS
    loader = source._resolve_loader_from_disk("icustays")
    if loader.is_dir():
        assert sl.enumerate_parquet_files(loader) == [p.as_posix() for p in files]
        assert source.resolve_flat_parquet_directory("icustays") == loader
    _readers(tmp_path)


@pytest.mark.parametrize("relative", [".", "general_table.csv"])
def test_configured_directory_cannot_be_absent_when_its_loader_reads_parquet(tmp_path, monkeypatch, relative):
    import copy
    from easyicu import datasource as ds

    config = copy.deepcopy(load_src_cfg("hirid"))
    config.get_table("general").files = [{"path": relative}]
    directory = tmp_path / relative
    directory.mkdir(exist_ok=True)
    pd.DataFrame({"patientid": [900, 901], "value": [99.0, 99.0]}).to_csv(tmp_path / "general.csv", index=False)
    for number, ids in enumerate((IDS[:10], IDS[10:])):
        pd.DataFrame({"patientid": ids, "value": [60.0] * len(ids)}).to_parquet(directory / f"part{number}.parquet", index=False)
    monkeypatch.setattr(ds, "storage_source", lambda _db, path: ICUDataSource(config, base_path=path, enable_cache=False))
    _readers(tmp_path, "hirid", ICUDataSource(config, base_path=tmp_path, enable_cache=False))


def test_configured_outside_parquet_is_refused_instead_of_falling_back_to_csv(tmp_path, monkeypatch):
    import copy
    from easyicu import datasource as ds

    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "external_general.parquet"
    pd.DataFrame({"patientid": IDS, "value": [60.0] * len(IDS)}).to_parquet(outside, index=False)
    pd.DataFrame({"patientid": [900, 901], "value": [99.0, 99.0]}).to_csv(root / "general.csv", index=False)
    config = copy.deepcopy(load_src_cfg("hirid"))
    config.get_table("general").files = [{"path": str(outside)}]
    monkeypatch.setattr(ds, "storage_source", lambda _db, path: ICUDataSource(config, base_path=path, enable_cache=False))
    assert _refusals(root, "hirid", ICUDataSource(config, base_path=root, enable_cache=False)) == ["storage_read_outside_data_root"] * 5


def test_configured_stem_directory_never_hides_a_nested_inventory_part(tmp_path):
    _csv(tmp_path, "hirid")
    directory = tmp_path / "GENERAL_TABLE"
    (directory / "deep").mkdir(parents=True)
    for part in (directory / "part.parquet", directory / "deep" / "extra.parquet"):
        pd.DataFrame({"patientid": IDS, "value": [60.0] * len(IDS)}).to_parquet(part, index=False)
    assert _refusals(tmp_path, "hirid") == ["table_scan_incomplete"] * 5


def test_verified_legacy_adoption_records_the_flat_source_binding(tmp_path):
    from scripts import build_itemid_bucket_cache as builder
    from tests.core.test_bucket_storage_b6 import _legacy_aumc

    source, bucket = _legacy_aumc(tmp_path)
    expected = builder._derived_from(
        tmp_path.resolve(), "numericitems", sl.bucket_source_files(tmp_path, "numericitems"), bucket,
    )
    legacy = json.loads((bucket / sl.MANIFEST).read_text())
    legacy["derived_from"] = expected
    (bucket / sl.MANIFEST).write_text(json.dumps(legacy))
    adopted = builder.adopt_bucket_cache(tmp_path, threads=1, memory_limit="256MB")
    assert adopted["derived_from"] == expected
    files = source.resolve_target_files("numericitems")
    assert files and all(file.is_relative_to(bucket) for file in files)
    assert sl.flat_table_files(source, "numericitems") == sl.bucket_source_files(tmp_path, "numericitems")
