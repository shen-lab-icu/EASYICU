"""Variable × stay-range layout (lead-r13 #13): builder, contract, selection.

One partition per value of a variable key (eICU labels, integer itemids),
then stay ranges shared by the whole layout. Readers get the narrowed file
set from ``storage_layout`` and still apply their own predicates, so values
match flat sources (the cross-layout byte checks live in
test_bucket_storage_b6.py). Rows are synthetic.
"""

from __future__ import annotations

from bisect import bisect_left
import hashlib
import json
from pathlib import Path
import shutil

import duckdb
import pandas as pd
import pytest

from easyicu.storage_layout import (
    MANIFEST,
    ORDINAL,
    PARTITION_PREFIX,
    VARIABLE_SCHEME,
    BucketLayoutError,
    StorageIntegrityError,
    read_bucket_contract,
)
from scripts.build_itemid_bucket_cache import build_bucket_cache, build_variable_cache
from tests.support.bucket_storage import b6_dictionary, b6_source, install_variable_layout


class _Selector:
    """select() needs a source only for hash layouts."""


def _events(root: Path, *, integer_keys: bool) -> pd.DataFrame:
    keys = [7, 3, 11, 3, 7, 7, 5, 3, 7, 11] if integer_keys else [
        "Heart Rate", "GCS", "Temp", "GCS", "Heart Rate", "Heart Rate", None,
        "GCS", "Heart Rate", "Temp",
    ]
    frame = pd.DataFrame(
        {
            "stay_id": [40, 10, 30, 20, 10, 20, 30, 40, 30, 10],
            "item": keys,
            "detail": ["a", "eyes", "c", "motor", "a", "a", "z", "eyes", "a", "c"],
            "charttime": [5, 1, 4, 2, 1, 2, 9, 3, 3, 8],
            "value": [float(i) for i in range(10)],
        }
    )
    (root / "events").mkdir(parents=True)
    frame.iloc[:6].to_parquet(root / "events" / "1.parquet", index=False)
    frame.iloc[6:].to_parquet(root / "events" / "2.parquet", index=False)
    return frame


def _build(root: Path, **kwargs):
    defaults = dict(
        table="events", key="item", stay_key="stay_id", bucket_count=4,
        rows_per_bucket=1, secondary_keys=["detail"], memory_limit="256MB", threads=1,
    )
    defaults.update(kwargs)
    return build_variable_cache(root, **defaults)


def _read_all(files) -> pd.DataFrame:
    paths = ", ".join("'" + str(p) + "'" for p in files)
    return duckdb.sql(f"SELECT * FROM read_parquet([{paths}], hive_partitioning=false)").df()


@pytest.mark.parametrize("integer_keys", [False, True])
def test_builder_installs_exact_partitions_inventory_and_source_binding(tmp_path, integer_keys):
    source = _events(tmp_path, integer_keys=integer_keys)
    receipt = _build(tmp_path)
    root = tmp_path / "events_bucket"
    assert not (tmp_path / ".events_bucket_build").exists()
    manifest = json.loads((root / MANIFEST).read_text())
    assert manifest["scheme"] == VARIABLE_SCHEME and manifest["modulus"] == 4
    assert len(manifest["stay_breakpoints"]) == 3
    values = [p["value"] for p in manifest["partitions"]]
    distinct = sorted(set(source["item"].dropna()))
    assert values == distinct + ([None] if source["item"].isna().any() else [])
    for part in manifest["partitions"]:
        assert part["modulus"] in (1, 2, 4)
    on_disk = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.parquet"))
    assert sorted(e["relative_path"] for e in manifest["inventory"]) == on_disk
    assert sum(e["row_count"] for e in manifest["inventory"]) == len(source) == receipt["rows"]
    # The installation holds the source rows exactly, in recoverable order.
    stored = _read_all(root.rglob("*.parquet")).sort_values(ORDINAL)
    pd.testing.assert_frame_equal(
        stored.drop(columns=ORDINAL).reset_index(drop=True), source, check_dtype=False
    )
    # Every row sits in its variable's partition and its stay's range bucket.
    breaks = manifest["stay_breakpoints"]
    for part in manifest["partitions"]:
        span = 4 // part["modulus"]
        for path in (root / f"{PARTITION_PREFIX}{part['id']}").rglob("*.parquet"):
            rows = pd.read_parquet(path)
            assert set(rows["item"].dropna()) <= ({part["value"]} - {None})
            bucket = int(path.parent.name.split("=")[1])
            assert {bisect_left(breaks, s) // span for s in rows["stay_id"]} == {bucket}
            assert rows[["stay_id", "charttime", ORDINAL]].equals(
                rows[["stay_id", "charttime", ORDINAL]].sort_values(
                    ["stay_id", "charttime", ORDINAL]
                )
            )
    # Secondary selector values are recorded per partition.
    gcs = next(p for p in manifest["partitions"] if p["value"] in ("GCS", 3))
    assert gcs["secondary"]["detail"] == ["eyes", "motor"]
    # derived_from binds each source file's bytes.
    files = manifest["derived_from"]["files"]
    assert [f["path"] for f in files] == ["events/1.parquet", "events/2.parquet"]
    for entry in files:
        data = (tmp_path / entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
        assert entry["size"] == len(data)
    assert receipt["derived_from"] == manifest["derived_from"]
    assert read_bucket_contract(root).scheme == VARIABLE_SCHEME


def test_single_key_builds_also_bind_their_source_bytes(tmp_path):
    _events(tmp_path, integer_keys=True)
    build_bucket_cache(tmp_path, table="events", key="item", bucket_count=4,
                       memory_limit="256MB", threads=1)
    manifest = json.loads((tmp_path / "events_bucket" / MANIFEST).read_text())
    assert len(manifest["derived_from"]["files"]) == 2
    assert read_bucket_contract(tmp_path / "events_bucket").scheme == "hash"


def test_builder_refuses_an_existing_target_and_removes_a_failed_build(tmp_path, monkeypatch):
    import scripts.build_itemid_bucket_cache as builder

    _events(tmp_path, integer_keys=False)
    _build(tmp_path)
    with pytest.raises(FileExistsError):
        _build(tmp_path)

    def broken(*_args, **_kwargs):
        raise RuntimeError("bucket_sort_verification_failed: injected")

    monkeypatch.setattr(builder, "_verify_bucket", broken)
    with pytest.raises(RuntimeError, match="injected"):
        _build(tmp_path, output_root=tmp_path / "other_bucket")
    assert not (tmp_path / "other_bucket").exists()
    assert not (tmp_path / ".other_bucket_build").exists()


@pytest.mark.parametrize("integer_keys", [False, True])
def test_selection_by_variable_stay_both_and_neither(tmp_path, integer_keys):
    source = _events(tmp_path, integer_keys=integer_keys)
    _build(tmp_path)
    layout = read_bucket_contract(tmp_path / "events_bucket")
    everything = set(layout.files)
    wanted = ["GCS"] if not integer_keys else [3]
    stays = [10, 40]

    def expected(values, stay_values):
        chosen = set()
        for part in layout.partitions:
            if values is not None and part.value not in values:
                continue
            span = layout.modulus // part.modulus
            buckets = None if stay_values is None else {
                bisect_left(layout.breakpoints, s) // span for s in stay_values
            }
            for (pid, bucket), files in layout.files_by_partition.items():
                if pid == part.pid and (buckets is None or bucket in buckets):
                    chosen.update(files)
        return chosen

    def select(values=None, stay_values=None, column="item"):
        return set(layout.select(
            _Selector(), column if values is not None else None, values,
            stay_column="stay_id" if stay_values is not None else None,
            stay_values=stay_values,
        )[1])

    assert select() == everything
    assert select(wanted) == expected(wanted, None) < everything
    assert select(stay_values=stays) == expected(None, stays)
    both = select(wanted, stays)
    assert both == expected(wanted, stays)
    # The narrowed files hold every matching row.
    matching = source[source["item"].isin(wanted) & source["stay_id"].isin(stays)]
    read = _read_all(both)
    assert len(read[read["item"].isin(wanted) & read["stay_id"].isin(stays)]) == len(matching)
    # A different stay column, or a type-mismatched value, narrows nothing.
    assert set(layout.select(_Selector(), None, None, stay_column="other",
                             stay_values=stays)[1]) == everything
    assert set(layout.select(_Selector(), None, None, stay_column="stay_id",
                             stay_values=["10"])[1]) == everything
    # Secondary selector column.
    assert select(["motor"], column="detail") == expected(wanted if integer_keys else ["GCS"], None)
    # An absent value selects one file, which the caller's predicate empties.
    absent = layout.select(_Selector(), "item", [999] if integer_keys else ["Nope"])[1]
    assert len(absent) == 1


def test_single_key_layouts_ignore_the_stay_predicate(tmp_path):
    _events(tmp_path, integer_keys=True)
    build_bucket_cache(tmp_path, table="events", key="item", bucket_count=3, scheme="range",
                       range_breakpoints=[4, 8], memory_limit="256MB", threads=1)
    layout = read_bucket_contract(tmp_path / "events_bucket")
    plain = layout.select(_Selector(), "item", [3])
    narrowed = layout.select(_Selector(), "item", [3], stay_column="stay_id", stay_values=[10])
    assert plain == narrowed and len(plain[1]) == 1
    assert layout.select(_Selector(), None, None, stay_column="stay_id",
                         stay_values=[10]) == layout.select(_Selector(), None, None)


def _break(root: Path, defect: str) -> None:
    manifest = json.loads((root / MANIFEST).read_text())
    if defect == "partition_missing":
        shutil.rmtree(root / f"{PARTITION_PREFIX}0")
    elif defect == "partition_extra":
        (root / f"{PARTITION_PREFIX}{len(manifest['partitions'])}" / "bucket_id=0").mkdir(parents=True)
    elif defect == "file_extra":
        first = root / manifest["inventory"][0]["relative_path"]
        shutil.copy(first, first.with_name("extra.parquet"))
    elif defect == "file_missing":
        (root / manifest["inventory"][0]["relative_path"]).unlink()
    elif defect == "derived":
        manifest["derived_from"]["files"][0]["sha256"] = "not-a-digest"
        (root / MANIFEST).write_text(json.dumps(manifest))
    elif defect == "breakpoints":
        manifest["stay_breakpoints"] = manifest["stay_breakpoints"][:-1]
        (root / MANIFEST).write_text(json.dumps(manifest))


_DEFECTS = [
    ("partition_missing", "bucket_partition_missing"),
    ("partition_extra", "bucket_partition_extra"),
    ("file_extra", "bucket_inventory_mismatch"),
    ("file_missing", "bucket_inventory_mismatch"),
    ("derived", "bucket_derived_from_invalid"),
    ("breakpoints", "bucket_modulus_mismatch"),
]


@pytest.mark.parametrize("defect,reason", _DEFECTS)
def test_a_damaged_installation_is_refused_with_a_stable_reason(tmp_path, defect, reason):
    _events(tmp_path, integer_keys=False)
    _build(tmp_path)
    _break(tmp_path / "events_bucket", defect)
    with pytest.raises(BucketLayoutError) as error:
        read_bucket_contract(tmp_path / "events_bucket")
    assert error.value.reason == reason


@pytest.mark.parametrize("defect,reason", _DEFECTS[:4])
def test_without_a_flat_source_the_refusal_reaches_public_concept_loading(tmp_path, defect, reason):
    from easyicu.concept import ConceptResolver

    source = b6_source(tmp_path, "eicu")
    install_variable_layout(source)
    flat = [p for p in (tmp_path / "nursecharting", tmp_path / "nursecharting.parquet") if p.exists()]
    for path in flat:
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    _break(tmp_path / "nursecharting_bucket", defect)
    with pytest.raises(StorageIntegrityError) as error:
        ConceptResolver(b6_dictionary("eicu")).load_concepts(
            ["hr"], source, merge=False, interval=pd.Timedelta(hours=1),
            concept_workers=1, verbose=False,
        )
    assert error.value.reason == reason


@pytest.mark.parametrize("defect,reason", _DEFECTS[:2])
def test_beside_its_flat_source_a_damaged_layout_publishes_with_the_fallback_marked(
    tmp_path, defect, reason
):
    from easyicu.api.concepts import clear_global_loader
    from easyicu.api.extraction import extract_database
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / "hirid"
    generate_source(root, "hirid", stays=3)
    build_variable_cache(root, table="observations", key="variableid", stay_key="patientid",
                         bucket_count=2, rows_per_bucket=1, memory_limit="256MB", threads=1)
    _break(root / "observations_bucket", defect)
    output = tmp_path / "export"
    try:
        result = extract_database(
            "hirid", data_path=str(root), output_dir=str(output),
            modules=["outcome"], native_export_v2=True, verbose=False,
        )
    finally:
        clear_global_loader()
    expected = {"reason": "bucket_cache_ignored", "root": str(root),
                "table": "observations", "bucket_reason": reason}
    assert expected in result["storage_fallbacks"]
    published = json.loads((output / "_manifest.json").read_text())
    assert expected in published["storage_fallbacks"]


def test_stream_batches_are_contiguous_in_stay_order_for_every_database():
    from easyicu.api.extraction import _order_stream_patient_ids

    ids = [30, 10, 50, 20, 40]
    for database in ("eicu", "miiv", "mimic", "hirid", "aumc", "sic"):
        ordered, batches, strategy = _order_stream_patient_ids(ids, 2, database)
        assert ordered == [10, 20, 30, 40, 50] and batches == 3
        assert strategy == "stay_order_contiguous_prunable_v1"
    mixed, _, _ = _order_stream_patient_ids(["b", 1], 1, "eicu")
    assert mixed == ["b", 1]  # ids that do not sort together keep their order


@pytest.mark.parametrize("database", ["eicu", "miiv", "hirid"])
def test_published_bytes_do_not_depend_on_contiguous_or_interleaved_batches(
    tmp_path, database, monkeypatch
):
    from easyicu.api import extraction
    from easyicu.api.concepts import clear_global_loader
    from scripts.build_itemid_bucket_cache import TABLE_CONFIGS

    source = b6_source(tmp_path / "source", database)
    install_variable_layout(source)
    stay = TABLE_CONFIGS[database][2]
    stays = sorted(pd.concat(
        [pd.read_parquet(p, columns=[stay]) for p in
         (tmp_path / "source").glob(f"{TABLE_CONFIGS[database][0]}*/**/*.parquet")]
    )[stay].dropna().unique().tolist()) or [1, 2, 3]

    def interleaved(patient_ids, batch_size, _database):
        ordered, planned = extraction._interleave_stream_patient_ids(patient_ids, batch_size)
        return ordered, planned, "source_order_interleaved_v1"

    outputs = []
    try:
        for name in ("contiguous", "interleaved"):
            if name == "interleaved":
                monkeypatch.setattr(extraction, "_order_stream_patient_ids", interleaved)
            output = tmp_path / name
            output.mkdir()
            clear_global_loader()
            extraction._run_module_extraction(
                module_name="vitals", concepts=["hr", "map", "temp"], database=database,
                data_path=str(source.base_path), patient_ids_filter={stay: list(reversed(stays))},
                batch_size=1, output_dir=str(output), stream_output_batches=True,
            )
            module = json.loads((output / "_manifest.json").read_text())
            assert module["errors"] == []
            (output / "_manifest.json").rename(output / "vitals.manifest.json")
            extraction._publish_native_export_v2(
                database=database, data_path=str(source.base_path), output_dir=str(output),
                modules=["vitals"], max_patients=None, result={"modules": {"vitals": module}},
            )
            outputs.append((output / "vitals.parquet").read_bytes())
    finally:
        clear_global_loader()
    assert outputs[0] == outputs[1]
