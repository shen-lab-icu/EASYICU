"""Synthetic MIMIC-III inputs for the storage and CSV reader contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from easyicu import storage_layout as sl
from easyicu.datasource import FilterOp, FilterSpec


IDS = list(range(100, 120))


def icustays_parquet(path, ids=IDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"icustay_id": list(ids), "value": np.repeat(60.0, len(ids))}).to_parquet(path, index=False)


def icustays_bucket(root, *, derived: bool):
    """A range bucket of icustays beside its flat source, bound by derived_from or not."""
    flat = root / "icustays.parquet"
    frame = pd.read_parquet(flat).assign(**{sl.ORDINAL: np.arange(len(IDS), dtype="int64")})
    bucket = root / "icustays_bucket"
    (bucket / "bucket_id=0").mkdir(parents=True)
    (bucket / "_COMPLETE").write_text("synthetic")
    (bucket / sl.BUILD_RECEIPT).write_text("{}")
    part = bucket / "bucket_id=0" / "part.parquet"
    frame.to_parquet(part, index=False)
    manifest = {
        "schema": sl.SCHEMA, "scheme": "range", "key_column": "icustay_id", "key_type": "BIGINT",
        "range_breakpoints": [], "sort_keys": ["icustay_id", sl.ORDINAL],
        "source_order": sl.ORDINAL, "row_group_target": 65536,
        "builder_version": sl.BUILDER_VERSION,
        "inventory": [{"relative_path": "bucket_id=0/part.parquet",
                       "size": part.stat().st_size, "row_count": len(IDS)}],
    }
    if derived:
        stat = flat.stat()
        manifest["derived_from"] = {"table": "icustays", "files": [{
            "path": "icustays.parquet", "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": hashlib.sha256(flat.read_bytes()).hexdigest(),
        }]}
    (bucket / sl.MANIFEST).write_text(json.dumps(manifest))


SERVICES = pd.DataFrame({
    "ROW_ID": [1, 2], "SUBJECT_ID": [10, 11], "HADM_ID": [100, 101],
    "TRANSFERTIME": ["2100-01-01 00:00:00", "2100-01-02 00:00:00"],
    "CURR_SERVICE": ["MED", "SURG"],
})


ICUSTAYS = pd.DataFrame({
    "subject_id": [10, 11], "hadm_id": [100, 101], "icustay_id": [1000, 1001],
    "intime": pd.to_datetime(["2100-01-01", "2100-01-02"]),
    "outtime": pd.to_datetime(["2100-01-03", "2100-01-04"]),
})


def conversion_manifest(root: Path, tables: dict[str, int], **extra) -> Path:
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


def converted_mimic(root: Path, csv_target: Path | None = None) -> Path:
    """A converted MIMIC-III install: services as Parquet and as CSV beside it."""
    root.mkdir(parents=True)
    SERVICES.to_parquet(root / "services.parquet", index=False)
    ICUSTAYS.to_parquet(root / "icustays.parquet", index=False)
    if csv_target is None:
        SERVICES.to_csv(root / "SERVICES.csv.gz", index=False)
    else:
        SERVICES.to_csv(csv_target, index=False)
        (root / "SERVICES.csv.gz").symlink_to(csv_target)
    conversion_manifest(root, {"services": 2, "icustays": 2})
    return root


def load_services(source):
    return source.load_table(
        "services", columns=["hadm_id", "curr_service"],
        filters=[FilterSpec(column="hadm_id", op=FilterOp.IN, value=[100])],
    ).data


def constant_chart_csv(root):
    header = (
        "ROW_ID,SUBJECT_ID,HADM_ID,ICUSTAY_ID,ITEMID,CHARTTIME,STORETIME,CGID,VALUE,"
        "VALUENUM,VALUEUOM,WARNING,ERROR,RESULTSTATUS,STOPPED"
    )
    rows = [
        f"{i},1,11,101,220045,2140-01-02 01:00:00,2140-01-02 01:05:00,17,60,60,bpm,0,0,,"
        for i in range(1, 6)
    ]
    path = root / "CHARTEVENTS.csv"
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


HEADER = (
    "ROW_ID,SUBJECT_ID,HADM_ID,ICUSTAY_ID,ITEMID,CHARTTIME,STORETIME,CGID,VALUE,"
    "VALUENUM,VALUEUOM,WARNING,ERROR,RESULTSTATUS,STOPPED"
)


def chart_row(i, *, valuenum=None, charttime="2140-01-02 01:00:00", cgid="17"):
    valuenum = str(60 + i % 5) if valuenum is None else valuenum
    return f"{i},1,11,101,220045,{charttime},2140-01-02 01:05:00,{cgid},{60 + i % 5},{valuenum},bpm,0,0,,"
