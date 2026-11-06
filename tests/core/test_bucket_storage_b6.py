"""Synthetic storage contracts must never change the logical source table."""

from pathlib import Path

import pandas as pd

from easyicu.config import DataSourceConfig
from easyicu.datasource import ICUDataSource


import json
from dataclasses import replace
import shutil

import duckdb
import pyarrow.parquet as pq
import pytest

from easyicu.concept import ConceptResolver
from easyicu.datasource import (
    load_bucketed_table_aggregated,
    load_bucketed_table_multi_aggregated,
)
from easyicu.storage_layout import (
    MANIFEST,
    ORDINAL,
    BucketLayoutError,
    parquet_relation,
    read_bucket_contract,
)
from scripts import build_itemid_bucket_cache as builder
from tests.support.bucket_storage import (
    with_value_column_rows,
    b6_dictionary,
    b6_source,
    install_layout,
    write_contract,
)


def test_missing_manifest_falls_back_to_flat_without_reader_writes(tmp_path: Path):
    flat = tmp_path / "chartevents"
    bucket = tmp_path / "chartevents_bucket" / "bucket_id=0"
    flat.mkdir()
    bucket.mkdir(parents=True)
    pd.DataFrame({"itemid": [220045], "valuenum": [71.0]}).to_parquet(
        flat / "1.parquet"
    )
    pd.DataFrame({"itemid": [220045], "valuenum": [999.0]}).to_parquet(
        bucket / "1.parquet"
    )
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    source = ICUDataSource(DataSourceConfig(name="miiv"), base_path=tmp_path)
    assert source.resolve_bucket_directory("chartevents") is None
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


def _frames(source, names, *, single=False, aggregate="median"):
    resolver = ConceptResolver(b6_dictionary(source.config.name))
    result = {}
    for group in [[name] for name in names] if single else [names]:
        result.update(
            resolver.load_concepts(
                group,
                source,
                merge=False,
                r_compatible=False,
                interval=pd.Timedelta(hours=1),
                concept_workers=1,
                aggregate=aggregate,
                verbose=False,
            )
        )
    return {name: value.data for name, value in result.items()}


def _assert_frames(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        pd.testing.assert_frame_equal(actual[name], expected[name], check_exact=True)
        assert actual[name].attrs == expected[name].attrs


@pytest.mark.parametrize("database", ["miiv", "mimic", "eicu", "hirid", "aumc"])
@pytest.mark.parametrize("scheme", ["hash", "range"])
def test_logical_table_and_concepts_are_layout_independent(tmp_path, database, scheme):
    source = b6_source(tmp_path, database)
    table, key, stay, clock = builder.TABLE_CONFIGS[database]
    raw = source.load_table(table).data
    names = ["hr", "map", "temp", "b6_first", "b6_id", "b6_callback"]
    flat = _frames(source, names)
    single = _frames(source, names, single=True)
    install_layout(source, scheme)
    pd.testing.assert_frame_equal(source.load_table(table).data, raw, check_exact=True)
    _assert_frames(_frames(source, names), flat)
    _assert_frames(_frames(source, names, single=True), single)
    assert ORDINAL not in source.load_table(table).data.columns
    layout = read_bucket_contract(tmp_path / f"{table}_bucket")
    assert layout.source_order == ORDINAL and layout.modulus == (
        64 if scheme == "hash" else 2
    )
    assert (
        layout.files and len(layout.files) < layout.modulus
        if scheme == "hash"
        else layout.files
    )


@pytest.mark.parametrize("database", ["miiv", "eicu", "hirid", "aumc"])
@pytest.mark.parametrize("aggregate", ["first", "median", "sum", "mean"])
def test_single_and_multi_sql_aggregates_preserve_ties(tmp_path, database, aggregate):
    source = b6_source(tmp_path, database)
    table, key, stay, clock = builder.TABLE_CONFIGS[database]
    definition = b6_dictionary(database)["hr"].sources[database][-1]
    ids = definition.ids
    value = (
        "nursingchartvalue"
        if database == "eicu"
        else "valuenum"
        if database == "miiv"
        else "value"
    )
    # The eICU aggregate accepts numeric VARCHAR through its single owner;
    # exercise a numeric source column for the multi aggregate as in production.
    if database == "eicu":
        path = source.resolve_target_files(table)[0]
        frame = pd.read_parquet(path)
        frame["numeric_value"] = pd.to_numeric(frame.nursingchartvalue)
        frame.to_parquet(path, index=False)
        value = "numeric_value"
    kwargs = dict(
        value_column=value,
        agg_func=aggregate,
        id_col=stay,
        time_col=clock,
        itemid_col=key,
    )
    flat = load_bucketed_table_aggregated(source, table, itemids=ids, **kwargs)
    multi = load_bucketed_table_multi_aggregated(source, table, {"hr": ids}, **kwargs)
    install_layout(source, "hash")
    pd.testing.assert_frame_equal(
        load_bucketed_table_aggregated(source, table, itemids=ids, **kwargs),
        flat,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        load_bucketed_table_multi_aggregated(source, table, {"hr": ids}, **kwargs),
        multi,
        check_exact=True,
    )


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("missing", "bucket_manifest_missing"),
        ("modulus", "bucket_modulus_mismatch"),
        ("inventory", "bucket_inventory_mismatch"),
        ("file", "bucket_inventory_mismatch"),
        ("type", "bucket_key_type_mismatch"),
        ("complete", "bucket_incomplete"),
        ("hash", "bucket_hash_mismatch"),
        ("json", "bucket_manifest_invalid"),
    ],
)
def test_invalid_contract_falls_back_or_fails_with_stable_reason(
    tmp_path, defect, reason
):
    source = b6_source(tmp_path, "miiv")
    install_layout(source, "hash")
    root = tmp_path / "chartevents_bucket"
    manifest = json.loads((root / MANIFEST).read_text())
    if defect == "missing":
        (root / MANIFEST).unlink()
    elif defect == "file":
        (root / manifest["inventory"][0]["relative_path"]).unlink()
    elif defect == "complete":
        (root / "_COMPLETE").unlink()
    else:
        if defect == "modulus":
            manifest["modulus"] = 63
        elif defect == "inventory":
            manifest["inventory"][0]["row_count"] += 1
        elif defect == "type":
            manifest["key_type"] = "VARCHAR"
        elif defect == "hash":
            manifest["hash"]["duckdb_version"] = "other"
        (root / MANIFEST).write_text("{" if defect == "json" else json.dumps(manifest))
    before = {
        p.relative_to(tmp_path): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    assert source.resolve_bucket_directory("chartevents") is None
    assert source.resolve_target_files("chartevents") == (
        tmp_path / "chartevents/1.parquet",
    )
    shutil.rmtree(tmp_path / "chartevents")
    with pytest.raises(BucketLayoutError) as error:
        source.resolve_target_files("chartevents", "itemid", [220045])
    assert error.value.reason == reason
    assert before == {
        p.relative_to(tmp_path): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


def test_empty_last_bucket_uses_contract_modulus_and_reader_is_read_only(tmp_path):
    source = b6_source(tmp_path, "miiv")
    install_layout(source, "hash")
    root = tmp_path / "chartevents_bucket"
    assert not list((root / "bucket_id=63").glob("*.parquet"))
    source_files = source.resolve_target_files("chartevents", "itemid", [220045])
    expected = int(duckdb.sql("SELECT hash(220045::BIGINT) % 64").fetchone()[0])
    assert {p.parent.name for p in source_files} == {f"bucket_id={expected}"}
    before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in root.rglob("*")}
    source.load_table("chartevents")
    assert before == {
        p: (p.stat().st_mtime_ns, p.stat().st_size) for p in root.rglob("*")
    }
    (root / "bucket_id=63").rmdir()
    assert source.resolve_bucket_directory("chartevents") is None


def test_string_key_pruning_requires_same_column_and_original_type(tmp_path):
    source = b6_source(tmp_path, "eicu")
    install_layout(source, "hash")
    all_files = source.resolve_target_files("nursecharting")
    selected = source.resolve_target_files(
        "nursecharting", "nursingchartcelltypevallabel", ["Heart Rate"]
    )
    assert 0 < len(selected) < len(all_files)
    assert (
        source.resolve_target_files(
            "nursecharting", "nursingchartcelltypevalname", ["Invasive BP Mean"]
        )
        == all_files
    )
    assert (
        source.resolve_target_files(
            "nursecharting", "nursingchartcelltypevallabel", [123]
        )
        == all_files
    )
    expected = source._read_partitioned_data_duckdb(
        tmp_path / "nursecharting",
        None,
        itemid_filter_config=("nursingchartcelltypevalname", {"Invasive BP Mean"}),
    )
    actual = source._read_partitioned_data_duckdb(
        tmp_path / "nursecharting_bucket",
        None,
        itemid_filter_config=("nursingchartcelltypevalname", {"Invasive BP Mean"}),
    )
    pd.testing.assert_frame_equal(actual, expected)
    assert len(actual) == 2


def test_numeric_type_mismatch_never_hashes_a_cast_value(tmp_path):
    source = b6_source(tmp_path, "miiv")
    install_layout(source, "hash")
    assert source.resolve_target_files(
        "chartevents", "itemid", [220045.0]
    ) == source.resolve_target_files("chartevents")
    assert source.resolve_target_files(
        "chartevents", "itemid", ["220045"]
    ) == source.resolve_target_files("chartevents")


def test_arrow_fallback_uses_same_files_and_source_order(tmp_path):
    source = b6_source(tmp_path, "miiv")
    columns = ["stay_id", "charttime", "itemid", "valuenum"]
    expected = source._read_partitioned_data_optimized(
        tmp_path / "chartevents", columns, itemid_filter_config=("itemid", {220045})
    )
    install_layout(source, "hash")
    actual = source._read_partitioned_data_optimized(
        tmp_path / "chartevents_bucket",
        columns,
        itemid_filter_config=("itemid", {220045}),
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_id_table_fast_path_reads_only_target_files(tmp_path, monkeypatch):
    import easyicu.storage_layout as storage

    source = b6_source(tmp_path, "miiv")
    install_layout(source, "hash")
    expected = source.resolve_target_files("chartevents", "itemid", [226512])
    all_files = source.resolve_target_files("chartevents")
    calls = []
    original = storage.parquet_relation

    def counted(files, **kwargs):
        files = tuple(files)
        calls.append(files)
        return original(files, **kwargs)

    monkeypatch.setattr(storage, "parquet_relation", counted)
    result = _frames(source, ["b6_id"])
    assert not result["b6_id"].empty
    assert calls == [expected]
    assert len(expected) < len(all_files)


def test_builder_declares_and_preserves_stable_exact_ties(tmp_path):
    source = b6_source(tmp_path, "miiv")
    receipt = install_layout(source, "hash")
    assert receipt["sort_keys"][-1] == ORDINAL
    layout = read_bucket_contract(tmp_path / "chartevents_bucket")
    assert layout.source_order == ORDINAL
    with duckdb.connect() as con:
        for path in layout.files:
            frame = con.execute(
                f"SELECT * FROM {parquet_relation([path], keep_order_column=True)}"
            ).df()
            assert frame[ORDINAL].is_monotonic_increasing
            physical = pq.read_table(path).to_pandas()
            expected = physical.sort_values(layout.key_column, kind="stable")
            assert (
                physical[layout.key_column].tolist()
                == expected[layout.key_column].tolist()
            )
            assert receipt["rows"] == 8


def test_builder_rejects_injected_unsorted_bucket_and_cleans_install(
    tmp_path, monkeypatch
):
    b6_source(tmp_path, "miiv")
    original = builder._verify_bucket

    def corrupt(con, path, sort_keys, expected_rows):
        frame = pq.read_table(path).to_pandas().iloc[::-1]
        frame.to_parquet(path, index=False)
        original(con, path, sort_keys, expected_rows)

    monkeypatch.setattr(builder, "_verify_bucket", corrupt)
    with pytest.raises(RuntimeError, match="bucket_sort_verification_failed"):
        builder.build_bucket_cache(
            tmp_path, table="chartevents", key="itemid", bucket_count=1
        )
    assert not (tmp_path / "chartevents_bucket").exists()
    assert not (tmp_path / ".chartevents_bucket_build").exists()


def test_unverifiable_legacy_cache_cannot_be_used(tmp_path):
    root = tmp_path / "numericitems_bucket"
    root.mkdir()
    (root / MANIFEST).write_text(json.dumps({"num_buckets": 64, "files_by_bucket": {}}))
    (root / "_COMPLETE").touch()
    source = ICUDataSource(DataSourceConfig(name="aumc"), base_path=tmp_path)
    with pytest.raises(BucketLayoutError, match="bucket_legacy_unverifiable"):
        source.resolve_target_files("numericitems")


def _legacy_aumc(root, *, receipted=True):
    source = b6_source(root, "aumc")
    table = "numericitems"
    files = source.resolve_target_files(table)
    frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    cache = root / "numericitems_bucket"
    cache.mkdir()
    with duckdb.connect() as con:
        for bid in range(8):
            directory = cache / f"bucket_id={bid}"
            directory.mkdir()
            selected = f"SELECT * FROM {builder._relation(files)} WHERE hash(itemid) % 8 = {bid}"
            if con.execute(f"SELECT count(*) FROM ({selected})").fetchone()[0]:
                con.execute(
                    f"COPY ({selected}) TO {builder._sql_string(directory / 'part.parquet')} (FORMAT PARQUET)"
                )
        schema = builder._schema(con, builder._relation(files))
        fingerprint = builder._row_fingerprint(
            con, builder._relation(files), list(frame)
        )
    receipt = dict(
        schema=builder.RECEIPT_SCHEMA,
        key="itemid",
        bucket_count=8,
        duckdb_version=duckdb.__version__,
        source_schema=schema,
        rows=len(frame),
        row_multiset_fingerprint=fingerprint,
    )
    (cache / "_COMPLETE").write_text("complete\n")
    marker = (cache / "_COMPLETE").stat()
    legacy = dict(
        num_buckets=8,
        complete_mtime_ns=marker.st_mtime_ns,
        complete_size=marker.st_size,
        files_by_bucket={
            str(i): [
                p.relative_to(cache).as_posix()
                for p in (cache / f"bucket_id={i}").glob("*.parquet")
            ]
            for i in range(8)
        },
    )
    (cache / MANIFEST).write_text(json.dumps(legacy))
    if receipted:
        (cache / "_BUCKET_BUILD_RECEIPT.json").write_text(json.dumps(receipt))
    return source, cache


def test_current_receipted_aumc_legacy_is_read_only_and_adoptable(tmp_path):
    source, cache = _legacy_aumc(tmp_path)
    before = {p: p.read_bytes() for p in cache.rglob("*") if p.is_file()}
    selected = source.resolve_target_files("numericitems", "itemid", [6640])
    assert selected and len(selected) < len(source.resolve_target_files("numericitems"))
    assert before == {p: p.read_bytes() for p in cache.rglob("*") if p.is_file()}
    manifest = builder.adopt_bucket_cache(tmp_path, threads=1, memory_limit="256MB")
    assert manifest["schema"] == "easyicu.bucket_storage/1"
    assert read_bucket_contract(cache).modulus == 8
    assert selected == source.resolve_target_files("numericitems", "itemid", [6640])
    assert all(
        p.read_bytes() == content for p, content in before.items() if p.name != MANIFEST
    )


def test_adoption_fails_on_wrong_row_multiset_and_does_not_install(tmp_path):
    _, cache = _legacy_aumc(tmp_path)
    before = (cache / MANIFEST).read_bytes()
    path = next(cache.glob("bucket_id=*/*.parquet"))
    frame = pd.read_parquet(path)
    frame.loc[0, "value"] = 1234567.0
    frame.to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="Row-multiset fingerprint mismatch"):
        builder.adopt_bucket_cache(tmp_path, threads=1)
    assert before == (cache / MANIFEST).read_bytes()


@pytest.mark.parametrize("scheme", ["hash", "range", "variable"])
@pytest.mark.parametrize("database", ["miiv", "eicu", "hirid", "aumc"])
def test_streamed_native_module_parquet_bytes_and_metadata_match(
    tmp_path, database, scheme
):
    from easyicu.api import extraction
    from easyicu.api.concepts import clear_global_loader

    source = b6_source(tmp_path / "source", database)
    _, _, stay, _ = builder.TABLE_CONFIGS[database]
    outputs = []
    try:
        for layout_name in ("flat", scheme):
            if layout_name != "flat":
                install_layout(source, scheme)
            output = tmp_path / layout_name
            output.mkdir()
            clear_global_loader()
            extraction._run_module_extraction(
                module_name="vitals",
                concepts=["hr", "map", "temp"],
                database=database,
                data_path=str(source.base_path),
                patient_ids_filter={stay: [1, 2, 3]},
                batch_size=1,
                output_dir=str(output),
                stream_output_batches=True,
            )
            module = json.loads((output / "_manifest.json").read_text())
            assert module["errors"] == []
            (output / "_manifest.json").rename(output / "vitals.manifest.json")
            extraction._publish_native_export_v2(
                database=database,
                data_path=str(source.base_path),
                output_dir=str(output),
                modules=["vitals"],
                max_patients=None,
                result={"modules": {"vitals": module}},
            )
            outputs.append(output / "vitals.parquet")
        assert outputs[0].read_bytes() == outputs[1].read_bytes()
        assert (
            pq.read_schema(outputs[0]).metadata == pq.read_schema(outputs[1]).metadata
        )
        pd.testing.assert_frame_equal(
            pd.read_parquet(outputs[0]), pd.read_parquet(outputs[1]), check_exact=True
        )
    finally:
        clear_global_loader()


@pytest.mark.parametrize("database", ["hirid", "aumc"])
def test_all_nineteen_native_module_files_are_byte_identical(tmp_path, database):
    from easyicu.api.extraction import EXTRACT_MODULE_ORDER, extract_database
    from easyicu.api.concepts import clear_global_loader
    from easyicu.config import load_src_cfg
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / "source"
    generate_source(root, database, stays=3)
    source = ICUDataSource(load_src_cfg(database), base_path=root, enable_cache=False)
    _, _, stay, _ = builder.TABLE_CONFIGS[database]
    try:
        for layout in ("flat", "hash"):
            if layout == "hash":
                install_layout(source, "hash")
            clear_global_loader()
            result = extract_database(
                database,
                data_path=str(root),
                output_dir=str(tmp_path / layout),
                modules=list(EXTRACT_MODULE_ORDER),
                patient_ids={stay: [1, 2, 3]},
                batch_size=2,
                native_export_v2=True,
                stream_output_batches=True,
                resource_budget_mb=2048,
                verbose=False,
            )
            assert all(not state["errors"] for state in result["modules"].values())
        for module in EXTRACT_MODULE_ORDER:
            flat, hashed = (
                tmp_path / layout / f"{module}.parquet" for layout in ("flat", "hash")
            )
            pd.testing.assert_frame_equal(
                pd.read_parquet(flat),
                pd.read_parquet(hashed),
                check_exact=True,
                obj=module,
            )
            assert pq.read_schema(flat).metadata == pq.read_schema(hashed).metadata
            assert flat.read_bytes() == hashed.read_bytes(), module
    finally:
        clear_global_loader()


def test_mimic_uppercase_physical_columns_keep_logical_selector_pruning(tmp_path):
    source = b6_source(tmp_path, "mimic")
    path = tmp_path / "chartevents/1.parquet"
    frame = pd.read_parquet(path).rename(columns=str.upper)
    frame.to_parquet(path, index=False)
    builder.build_bucket_cache(
        tmp_path,
        table="chartevents",
        key="itemid",
        bucket_count=64,
        memory_limit="256MB",
        threads=1,
    )
    selected = source.resolve_target_files("chartevents", "itemid", [220045])
    assert 0 < len(selected) < len(source.resolve_target_files("chartevents"))
    layout = read_bucket_contract(tmp_path / "chartevents_bucket")
    assert layout.key_column == "ITEMID"
    manifest = json.loads((layout.root / MANIFEST).read_text())
    assert manifest["sort_keys"] == ["ITEMID", "ICUSTAY_ID", "CHARTTIME", ORDINAL]
    assert _frames(source, ["hr"])["hr"].hr.tolist() == [84.0]


def test_invalid_contract_is_visible_through_public_concept_loading(tmp_path):
    source = b6_source(tmp_path, "miiv")
    install_layout(source, "hash")
    shutil.rmtree(tmp_path / "chartevents")
    (tmp_path / "chartevents_bucket" / MANIFEST).unlink()
    with pytest.raises(BucketLayoutError, match="bucket_manifest_missing"):
        _frames(source, ["hr"])


def test_hirid_dose_callback_uses_same_weight_owner_in_every_layout(tmp_path):
    from easyicu.resources import load_dictionary

    source = b6_source(tmp_path, "hirid")
    names = ["norepi_rate", "epi_rate", "dobu_rate"]

    def load():
        result = ConceptResolver(load_dictionary()).load_concepts(
            names,
            source,
            merge=False,
            r_compatible=False,
            interval=pd.Timedelta(hours=1),
            patient_ids={"patientid": [1, 2, 3]},
            concept_workers=1,
            verbose=False,
        )
        return {name: value.data for name, value in result.items()}

    flat = load()
    install_layout(source, "hash")
    _assert_frames(load(), flat)


def test_range_breakpoint_equality_selects_the_earlier_range(tmp_path):
    source = b6_source(tmp_path, "eicu")
    install_layout(source, "range")
    layout = read_bucket_contract(tmp_path / "nursecharting_bucket")
    boundary = layout.breakpoints[0]
    selected = source.resolve_target_files(
        "nursecharting", layout.key_column, [boundary]
    )
    assert {p.parent.name for p in selected} == {"bucket_id=0"}
    expected = source._read_partitioned_data_duckdb(
        tmp_path / "nursecharting",
        None,
        itemid_filter_config=(layout.key_column, {boundary}),
    )
    actual = source._read_partitioned_data_duckdb(
        tmp_path / "nursecharting_bucket",
        None,
        itemid_filter_config=(layout.key_column, {boundary}),
    )
    pd.testing.assert_frame_equal(actual, expected)
    assert not actual.empty


def test_dictionary_encoded_string_keys_use_their_logical_duckdb_type(tmp_path):
    source = b6_source(tmp_path, "eicu")
    path = tmp_path / "nursecharting/part.parquet"
    frame = pd.read_parquet(path)
    frame["nursingchartcelltypevallabel"] = frame.nursingchartcelltypevallabel.astype(
        "category"
    )
    frame.to_parquet(path, index=False)
    expected = source.load_table("nursecharting").data
    install_layout(source, "hash")
    selected = source.resolve_target_files(
        "nursecharting", "nursingchartcelltypevallabel", ["Heart Rate"]
    )
    assert 0 < len(selected) < len(source.resolve_target_files("nursecharting"))
    pd.testing.assert_frame_equal(
        source.load_table("nursecharting").data, expected, check_exact=True
    )


_ID_TBL_ITEMS = {
    # database: (concept, item) for every id_tbl source on the bucketed table
    "miiv": [("weight", 226512), ("height", 226707)],
    "mimic": [("weight", 580), ("height", 920)],
    "hirid": [("weight", 10000400), ("height", 10000450)],
    "aumc": [("weight", 6638)],
}


def _layout_frames(root, database, concepts, interval=None):
    """Load each concept on its own; an error is part of the output."""
    from easyicu.config import load_src_cfg
    from easyicu.resources import load_dictionary

    source = ICUDataSource(load_src_cfg(database), base_path=root, enable_cache=False)
    resolver = ConceptResolver(load_dictionary())
    frames = {}
    for concept in concepts:
        try:
            loaded = resolver.load_concepts(
                [concept], source, merge=False, concept_workers=1, verbose=False,
                interval=interval,
            )
        except Exception as exc:  # compared across layouts, never swallowed
            frames[concept] = f"{type(exc).__name__}: {str(exc)[:200]}"
            continue
        if isinstance(loaded, dict):
            loaded = loaded[concept]
        frames[concept] = getattr(loaded, "data", loaded).reset_index(drop=True)
    return frames


def _fixture_table(root, database):
    """The fixture's bucketed table as one frame, plus its first and last stay."""
    b6_source(root, database)
    table, _key, stay, _clock = builder.TABLE_CONFIGS[database]
    parts = sorted((root / table).glob("*.parquet"))
    frame = pd.concat([pd.read_parquet(part) for part in parts], ignore_index=True)
    for part in parts:
        part.unlink()  # one file, so the single-file layout holds every row
    stays = sorted(frame[stay].dropna().unique().tolist())
    return frame, stays


def _each_layout(root, database, layouts=("directory", "single", "hash")):
    """Yield each layout name with ``root`` rearranged into it, in order."""
    from easyicu.config import load_src_cfg

    table = builder.TABLE_CONFIGS[database][0]
    path = root / table / "1.parquet"
    for layout in layouts:
        if layout == "single":
            shutil.move(path, root / f"{table}.parquet")
            (root / table).rmdir()
        elif (root / f"{table}.parquet").exists():
            (root / table).mkdir()
            shutil.move(root / f"{table}.parquet", path)
        if layout in ("hash", "variable"):
            bucket = root / f"{table}_bucket"
            if bucket.exists():
                shutil.rmtree(bucket)
            install_layout(
                ICUDataSource(load_src_cfg(database), base_path=root, enable_cache=False),
                layout,
            )
        yield layout


def _assert_same(name, actual, expected):
    if isinstance(expected, str) or isinstance(actual, str):
        assert actual == expected, name
    else:
        pd.testing.assert_frame_equal(actual, expected, check_exact=True, obj=name)


@pytest.mark.parametrize("database", sorted(_ID_TBL_ITEMS))
def test_id_tbl_concepts_are_identical_in_file_directory_and_bucket_layouts(tmp_path, database):
    """lead-r13 class a: id_tbl values depended on the storage layout.

    The former DuckDB fast path (directory/bucketed tables) read the string
    ``value`` column instead of ``valuenum``, and the raw path (single files)
    clipped rows to the ICU episode window.  The first stay's only row carries
    just ``valuenum``; the last stay has one row inside the window and one
    200 h before admission.  Every layout must keep all of them and aggregate
    one median per stay, as ricu does.
    """
    from easyicu.resources import load_dictionary

    dictionary = load_dictionary()
    frame, stays = _fixture_table(tmp_path, database)
    table, key, stay, _clock = builder.TABLE_CONFIGS[database]
    concepts = [concept for concept, _ in _ID_TBL_ITEMS[database]]
    own_items = {
        item
        for concept in concepts
        for source in dictionary[concept].sources.get(database, [])
        if source.table == table
        for item in (source.ids if isinstance(source.ids, list) else [source.ids])
    }
    frame = frame[~frame[key].isin(own_items)]
    rows = []
    clock = builder.TABLE_CONFIGS[database][3]
    before_admission = (
        pd.Timedelta(hours=200) if pd.api.types.is_datetime64_any_dtype(frame[clock])
        else 200 * 3600 * 1000  # AUMC measuredat: ms from admission
    )
    for concept, item in _ID_TBL_ITEMS[database]:
        normal = 61.0 if concept == "weight" else 171.0
        for stay_id, value, early in (
            (stays[0], normal, False), (stays[-1], normal - 10, False), (stays[-1], normal - 5, True),
        ):
            row = frame.iloc[[0]].copy()
            row[stay] = stay_id
            row[key] = item
            if early:
                row[clock] = row[clock] - before_admission
            if "valuenum" in row:
                row["valuenum"] = value
                row["value"] = None if stay_id == stays[0] else str(value)
            else:
                row["value"] = value
            rows.append(row)
    pd.concat([frame, *rows], ignore_index=True).to_parquet(
        tmp_path / table / "1.parquet", index=False
    )
    loaded = {
        layout: _layout_frames(tmp_path, database, concepts)
        for layout in _each_layout(tmp_path, database)
    }
    reference = loaded["single"]
    for (concept, _item), normal in zip(_ID_TBL_ITEMS[database], (61.0, 171.0)):
        if "valuenum" in frame:  # MIMIC: only these sources feed the stays
            frame_ = reference[concept].set_index(reference[concept].columns[0])[concept]
            assert frame_.get(stays[0]) == normal, concept  # ``valuenum`` only
            assert frame_.get(stays[-1]) == normal - 7.5, concept  # early row kept
        for layout in ("directory", "hash"):
            _assert_same(concept, loaded[layout][concept], reference[concept])


@pytest.mark.parametrize("database", sorted(builder.TABLE_CONFIGS))
def test_every_concept_on_the_bucketed_table_is_layout_independent(tmp_path, database):
    """lead-r13 class a: each source's own value column, in every layout.

    Every dictionary source on the bucketed table gets rows whose value sits
    only in that source's value column; the directory, single-file and
    bucketed layouts must return the same rows, or the same error.
    """
    from easyicu.resources import load_dictionary

    frame, stays = _fixture_table(tmp_path, database)
    frame, concepts = with_value_column_rows(frame, database, load_dictionary(), stays)
    assert concepts, database
    table = builder.TABLE_CONFIGS[database][0]
    frame.to_parquet(tmp_path / table / "1.parquet", index=False)
    loaded = {
        layout: _layout_frames(tmp_path, database, concepts, pd.Timedelta(hours=1))
        for layout in _each_layout(
            tmp_path, database, ("directory", "single", "hash", "variable")
        )
    }
    single = loaded["single"]
    produced = [c for c in concepts if not isinstance(single[c], str) and not single[c].empty]
    assert len(produced) >= len(concepts) // 2, (database, len(produced), len(concepts))
    mismatched = []
    for concept in concepts:
        for layout in ("directory", "hash", "variable"):
            try:
                _assert_same(concept, loaded[layout][concept], single[concept])
            except AssertionError:
                mismatched.append(f"{concept}:{layout}")
    assert not mismatched, mismatched


def _module_exports(root, out, database, concepts, stays):
    """Every module that uses ``concepts``: published bytes, or its errors."""
    from easyicu.api import extraction
    from easyicu.api.concepts import clear_global_loader

    stay = builder.TABLE_CONFIGS[database][2]
    exports = {}
    modules = [
        name for name, members in extraction.EXTRACT_MODULES.items()
        if set(members) & set(concepts)
    ]
    for module in modules:
        target = out / module
        target.mkdir(parents=True)
        clear_global_loader()
        try:
            extraction._run_module_extraction(
                module_name=module,
                concepts=list(extraction.EXTRACT_MODULES[module]),
                database=database,
                data_path=str(root),
                patient_ids_filter={stay: stays},
                batch_size=1,
                output_dir=str(target),
                stream_output_batches=True,
            )
        except Exception as exc:  # compared across layouts, never swallowed
            exports[module] = f"{type(exc).__name__}: {str(exc)[:200]}"
            continue
        manifest = json.loads((target / "_manifest.json").read_text())
        if manifest["errors"]:
            exports[module] = [str(error)[:200] for error in manifest["errors"]]
            continue
        (target / "_manifest.json").rename(target / f"{module}.manifest.json")
        extraction._publish_native_export_v2(
            database=database,
            data_path=str(root),
            output_dir=str(target),
            modules=[module],
            max_patients=None,
            result={"modules": {module: manifest}},
        )
        exports[module] = (target / f"{module}.parquet").read_bytes()
    clear_global_loader()
    return exports


@pytest.mark.slow
@pytest.mark.parametrize("database", sorted(builder.TABLE_CONFIGS))
def test_every_module_export_is_byte_identical_across_layouts(tmp_path, database):
    """lead-r13 class a: storage layout must not reach the exported bytes.

    A single parquet file used to take the raw pandas path (float64 and a
    per-patient table cache) while directories and buckets took DuckDB
    (float32), so derived columns such as shock indices and white-cell
    ratios differed between layouts.  Batches of one stay with several
    concepts on one table exercise both the path choice and the cache.
    HiRID observations have no single-file layout (detection refuses it).
    """
    from easyicu.resources import load_dictionary

    root = tmp_path / "source"
    frame, stays = _fixture_table(root, database)
    frame, concepts = with_value_column_rows(frame, database, load_dictionary(), stays)
    table = builder.TABLE_CONFIGS[database][0]
    frame.to_parquet(root / table / "1.parquet", index=False)
    layouts = (
        ("directory", "hash", "variable")
        if database == "hirid"
        else ("directory", "single", "hash", "variable")
    )
    exports = {
        layout: _module_exports(root, tmp_path / layout, database, concepts, stays)
        for layout in _each_layout(root, database, layouts)
    }
    published = [m for m, value in exports["directory"].items() if isinstance(value, bytes)]
    assert published, exports["directory"]
    for layout in layouts[1:]:
        assert exports[layout].keys() == exports["directory"].keys()
        different = [m for m in exports[layout] if exports[layout][m] != exports["directory"][m]]
        assert not different, (layout, different)
