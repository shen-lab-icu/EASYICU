"""Explicit contracts for small, invented Parquet fixtures (never real sources)."""

from dataclasses import replace
import json
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.storage_layout import MANIFEST, SCHEMA, arrow_key_type
from scripts.build_itemid_bucket_cache import (
    STAY_KEYS,
    TABLE_CONFIGS,
    VARIABLE_PROFILES,
    build_bucket_cache,
    build_variable_cache,
)


def write_contract(
    root: Path, *, key: str, modulus: int = 1, key_type: str | None = None
):
    """Declare an explicit fixture modulus; never derive it from directory names."""
    for i in range(modulus):
        (root / f"bucket_id={i}").mkdir(parents=True, exist_ok=True)
    files = sorted(
        p for p in root.glob("bucket_id=*/*.parquet") if not p.name.startswith(".")
    )
    assert files, "A fixture needs a typed Parquet file, even for zero rows"
    if key_type is None:
        key_type = arrow_key_type(pq.read_schema(files[0]).field(key).type)
    manifest = dict(
        schema=SCHEMA,
        scheme="hash",
        key_column=key,
        key_type=key_type,
        modulus=modulus,
        hash={"name": "duckdb.hash", "duckdb_version": duckdb.__version__},
        sort_keys=[],
        source_order=None,
        row_group_target=131072,
        builder_version="synthetic-fixture/1",
        inventory=[
            dict(
                relative_path=p.relative_to(root).as_posix(),
                size=p.stat().st_size,
                row_count=pq.ParquetFile(p).metadata.num_rows,
            )
            for p in files
        ],
    )
    (root / MANIFEST).write_text(json.dumps(manifest))
    (root / "_COMPLETE").touch()
    return manifest


def declare_single_bucket_fixtures(root: Path):
    """Upgrade old one-bucket test fixtures to the explicit storage contract."""
    for directory in Path(root).glob("**/*_bucket"):
        files = list(directory.glob("bucket_id=0/*.parquet"))
        if not files:
            continue
        assert {p.name for p in directory.glob("bucket_id=*")} == {"bucket_id=0"}
        schema = pq.read_schema(files[0])
        key = next(
            (
                k
                for k in (
                    "itemid",
                    "variableid",
                    "pharmaid",
                    "DataID",
                    "LaboratoryID",
                    "nursingchartcelltypevallabel",
                    "labname",
                    "celllabel",
                    "patientunitstayid",
                    "admissionid",
                )
                if k in schema.names
                and arrow_key_type(schema.field(k).type) != "unsupported"
            ),
            None,
        )
        if key is None:
            key = next(
                k
                for k in schema.names
                if arrow_key_type(schema.field(k).type) != "unsupported"
            )
        write_contract(directory, key=key)


def b6_source(root: Path, database: str):
    """Four native long-table schemas with intentionally interleaved exact ties."""
    root.mkdir(parents=True, exist_ok=True)
    table, key, stay, clock = TABLE_CONFIGS[database]
    if database in {"aumc", "hirid"}:
        from easyicu.synthetic_sources import generate_source

        root.rmdir()  # generate_source requires a new destination
        generate_source(root, database, stays=3)
        files = [root / table / "1.parquet"]
        frame = pd.read_parquet(files[0])
        ties = frame.iloc[[0, 0]].copy()
        ties["value"] = [77.0, 91.0]
        frame = pd.concat(
            [ties.iloc[[1]], frame.iloc[::-1], ties.iloc[[0]]], ignore_index=True
        )
        frame.to_parquet(files[0], index=False)
    elif database in {"miiv", "mimic"}:
        from tests.support.mimic_synthetic_source import mimic_synthetic_source

        items = [220045, 220052, 220181, 223762, 226512, 220045, 220052, 223761]
        frame = pd.DataFrame(
            {
                stay: [2, 1, 1, 1, 1, 2, 1, 1],
                "itemid": items,
                "valuenum": [91.0, 65.0, 79.0, 37.0, 82.0, 77.0, 67.0, 98.6],
            }
        )
        frame["charttime"] = pd.Timestamp("2150-01-01") + pd.Timedelta(minutes=5)
        frame["storetime"] = frame.charttime
        frame["value"] = frame.valuenum.astype(str)
        frame["valueuom"] = ""
        frame["subject_id"] = frame[stay] + 10
        frame["hadm_id"] = frame[stay] + 20
        frame["error"] = 0
        mimic_synthetic_source(root, database, chartevents=frame)
        # Use the same multi-file reader for all three physical layouts.
        (root / table).mkdir()
        (root / f"{table}.parquet").rename(root / table / "1.parquet")
    else:
        from tests.support.eicu_source import _eicu_source

        frame = pd.DataFrame(
            {
                "nursingchartcelltypevallabel": [
                    "Heart Rate",
                    "Arterial Line MAP (mmHg)",
                    "Other",
                    "Heart Rate",
                    "Temperature",
                ],
                "nursingchartcelltypevalname": [
                    "Value",
                    "Invasive BP Mean",
                    "Invasive BP Mean",
                    "Value",
                    "Temperature (C)",
                ],
                "nursingchartvalue": ["91", "65", "79", "77", "37"],
                "nursingchartoffset": [5] * 5,
                "nursingchartentryoffset": [6] * 5,
            }
        )
        _eicu_source(root, database, nursecharting=frame)
    return ICUDataSource(load_src_cfg(database), base_path=root, enable_cache=False)


def install_variable_layout(source, *, bucket_count=4, rows_per_bucket=1):
    """Variable × stay-range layout of the fixture's long table.

    ``rows_per_bucket=1`` gives every partition as many stay buckets as its
    rows allow (up to ``bucket_count``), so stay narrowing is exercised.
    """
    database = source.config.name
    table, key, stay, _ = TABLE_CONFIGS[database]
    variable, secondary = VARIABLE_PROFILES.get((database, table), (key, []))
    return build_variable_cache(
        source.base_path,
        table=table,
        key=variable,
        stay_key=STAY_KEYS.get(database, stay),
        bucket_count=bucket_count,
        rows_per_bucket=rows_per_bucket,
        secondary_keys=secondary,
        memory_limit="256MB",
        threads=2,
    )


def install_layout(source, scheme):
    if scheme == "variable":
        return install_variable_layout(source)
    table, key, _, _ = TABLE_CONFIGS[source.config.name]
    key_schema = source.resolve_target_files(table)
    sample = pd.read_parquet(key_schema[0], columns=[key])[key].dropna().tolist()
    breaks = [sorted(set(sample))[len(set(sample)) // 2]]
    receipt = build_bucket_cache(
        source.base_path,
        table=table,
        key=key,
        bucket_count=64,
        scheme=scheme,
        range_breakpoints=breaks,
        memory_limit="256MB",
        threads=2,
    )
    return receipt


def b6_dictionary(database):
    from easyicu.concept import ConceptDictionary
    from easyicu.resources import load_dictionary

    original = load_dictionary()
    hr = original["hr"]
    sources = {
        db: [replace(s, callback="transform_fun(binary_op(`*`, 1))") for s in rows]
        for db, rows in hr.sources.items()
    }
    return ConceptDictionary(
        {
            "hr": hr,
            "map": original["map"],
            "temp": original["temp"],
            "b6_first": replace(hr, name="b6_first", aggregate="first"),
            "b6_id": replace(original["weight"], name="b6_id"),
            "b6_callback": replace(hr, name="b6_callback", sources=sources),
        }
    )


def synthetic_data_source(*args, **kwargs):
    """Construct an ordinary reader after explicitly declaring fixture layouts."""
    if kwargs.get("base_path") is not None:
        declare_single_bucket_fixtures(Path(kwargs["base_path"]))
    return ICUDataSource(*args, **kwargs)


def write_fixture_parquet(frame, path, *args, **kwargs):
    """Write fixture rows and refresh their explicit single-bucket inventory."""
    frame.to_parquet(path, *args, **kwargs)
    target = Path(path)
    if target.parent.name == "bucket_id=0":
        declare_single_bucket_fixtures(target.parent.parent.parent)


def move_fixture_parquet(path, target):
    """Move a synthetic fixture into a bucket and declare the destination."""
    result = path.rename(target)
    destination = Path(target)
    if destination.parent.name == "bucket_id=0":
        declare_single_bucket_fixtures(destination.parent.parent.parent)
    return result


def _source_value_column(database, source):
    from easyicu.config import load_src_cfg

    if source.value_var:
        return source.value_var
    return load_src_cfg(database).get_table(source.table).defaults.val_var


def with_value_column_rows(frame, database, dictionary, stays):
    """Add rows for every dictionary source on the bucketed table.

    Each row carries its value only in that source's own value column
    (explicit ``val_var`` or the table default); the other value column is
    empty.  Returns the frame, cast back to the fixture dtypes, and the
    concepts those sources feed.
    """
    table, key, stay, _clock = TABLE_CONFIGS[database]
    rows, concepts = [], set()
    for name in sorted(dictionary.keys()):
        definition = dictionary[name]
        for source in definition.sources.get(database, []):
            if source.table != table or source.ids is None:
                continue
            column = _source_value_column(database, source)
            if column not in frame.columns:
                continue
            item = source.ids[0] if isinstance(source.ids, list) else source.ids
            low, high = definition.minimum, definition.maximum
            if low is not None and high is not None:
                value = (low + high) / 2
            elif low is not None:
                value = low + 1
            else:
                value = high - 1 if high is not None else 1.0
            # A tenth is not exact in float32, so float32/float64 paths differ.
            value = round(value, 3) + 0.1
            for stay_id, offset in ((stays[0], 0.0), (stays[-1], 0.0), (stays[-1], 1.0)):
                row = frame.iloc[[0]].copy()
                row[stay] = stay_id
                row[source.sub_var or key] = item
                for other in ("value", "valuenum"):
                    if other in row and other != column:
                        row[other] = None
                row[column] = value + offset
                rows.append(row)
            concepts.add(name)
    result = pd.concat([frame, *rows], ignore_index=True)
    for column, dtype in frame.dtypes.items():
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            result[column] = result[column].map(
                lambda x: None if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)
            )
        elif pd.api.types.is_float_dtype(dtype):
            result[column] = pd.to_numeric(result[column]).astype(dtype)
    return result, sorted(concepts)
