"""Synthetic worlds for the seal-tip bucket tool (tools/seal_buckets): a git mirror whose builder is a
small fake with the real builder's output contract, current formal roots (one plain root, one CareVue-style
link farm around a read-only view), and a config that runs the tool unconfined in a temp directory.

The fake builder writes the same files the real one does (parts with the source ordinal, manifest,
build receipt, _COMPLETE last, atomic rename from a hidden build directory) and can be told to misbehave
(MODE) so that each typed failure of the seal tool can be provoked. Rows are synthetic.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ORDINAL = "_easyicu_source_ordinal"

FAKE_BUILDER = r'''#!/usr/bin/env python3
"""Fake bucket builder for seal-tool tests (contract mark: bucket_float_fidelity_mismatch)."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import duckdb, numpy as np, pyarrow as pa, pyarrow.parquet as pq

MODE = "@MODE@"
ORDINAL = "_easyicu_source_ordinal"
DUCK = {"double": "DOUBLE", "float": "FLOAT", "int64": "BIGINT", "string": "VARCHAR", "large_string": "VARCHAR"}

ap = argparse.ArgumentParser()
for flag in ("--data-path", "--database", "--table", "--key", "--stay-key", "--scheme", "--memory-limit",
             "--sort-keys", "--output-root", "--row-group-target", "--threads"):
    ap.add_argument(flag)
ap.add_argument("--modulus", "--bucket-count", type=int, default=2)
a = ap.parse_args()
if os.environ.get("FAKE_PRELOAD"):  # stands for any start-up hook that preloads a foreign submodule
    import importlib.util
    spec = importlib.util.spec_from_file_location("easyicu.storage_layout", os.environ["FAKE_PRELOAD"])
    sys.modules["easyicu.storage_layout"] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules["easyicu.storage_layout"])
if MODE == "fail":
    print("FloatFidelityError: bucket_float_fidelity_mismatch: column 'x'", file=sys.stderr)
    sys.exit(1)
data, table = Path(a.data_path), a.table
files = sorted((data / table).glob("*.parquet")) if (data / table).is_dir() else [data / f"{table}.parquet"]
t = pa.concat_tables([pq.read_table(f) for f in files])
t = t.append_column(ORDINAL, pa.array(np.arange(t.num_rows), pa.int64()))
key = a.key or "itemid"
mod = a.modulus if (a.scheme or "hash") == "hash" else 2
target, tmp = data / f"{table}_bucket", data / f".{table}_bucket_build"
if target.exists() or tmp.exists():
    sys.exit("refusing to overwrite")
tmp.mkdir()
keys = t.column(key).to_numpy(zero_copy_only=False)
inventory = []
variable = (a.scheme or "hash") == "variable_stay_range"


def write_part(part, rel):
    (tmp / rel).mkdir(parents=True)
    if part.num_rows:
        if MODE == "nondeterministic":
            part = part.replace_schema_metadata({"built_at_ns": str(time.time_ns())})
        path = tmp / rel / "part.parquet"
        pq.write_table(part, path, use_dictionary=False)
        inventory.append({"relative_path": f"{rel}/part.parquet", "row_count": part.num_rows,
                          "size": path.stat().st_size})


partitions = []
if variable:  # one partition per key value, two stay ranges (stay <= 10 | stay > 10) in each
    stays = t.column(a.stay_key).to_numpy(zero_copy_only=False)
    for pid, value in enumerate(sorted(set(keys.tolist()))):
        partitions.append({"id": pid, "value": int(value), "modulus": 2, "secondary": {}})
        for b in range(2):
            mask = (keys == value) & ((stays > 10) == bool(b))
            write_part(t.filter(pa.array(mask)), f"var_id={pid}/bucket_id={b}")
else:
    for b in range(mod):
        write_part(t.filter(pa.array(keys % mod == b)), f"bucket_id={b}")
schema = [[f.name, DUCK.get(str(f.type), str(f.type).upper())] for f in t.schema if f.name != ORDINAL]
floats = [c for c, k in schema if k in ("DOUBLE", "FLOAT")]
columns = {}
for c in floats:
    v = t.column(c).to_numpy(zero_copy_only=False)
    columns[c] = {"values": int(v.size), "nulls": 0, "neg_zero": int(((v == 0) & np.signbit(v)).sum()),
                  "pos_zero": int(((v == 0) & ~np.signbit(v)).sum()), "nan": int(np.isnan(v).sum()),
                  "sha256": hashlib.sha256(v.tobytes()).hexdigest()}
fidelity = {"algorithm": "fake", "dictionary_encoding": "default" if MODE == "dict_encoding" else "disabled",
            "columns": columns, "rows": t.num_rows}
repo = Path(__file__).resolve().parents[1]
commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo,
                            capture_output=True, text=True).stdout.strip())
code = None if MODE == "no_builder_code" else {"source": "git", "commit": commit, "dirty": dirty}
sort = ([s for s in a.sort_keys.split(",")] if a.sort_keys else [key]) + [ORDINAL]
types = dict(schema)
manifest = {"schema": "easyicu.bucket_storage/1", "scheme": a.scheme or "hash", "key_column": key,
            "key_type": types[key], "builder_version": "easyicu.bucket_builder/2",
            "modulus": 2 if variable else mod, "row_group_target": int(a.row_group_target or 131072),
            "sort_keys": sort, "source_order": ORDINAL, "inventory": inventory, "float_fidelity": fidelity,
            "derived_from": {"table": table, "files": [{"path": str(f.relative_to(data)),
                                                        "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                                                        "size": f.stat().st_size,
                                                        "mtime_ns": f.stat().st_mtime_ns}
                                                       for f in files]}}
if variable:
    manifest.update(stay_column=a.stay_key, stay_key_type=types[a.stay_key], stay_breakpoints=[10],
                    secondary_keys=[], partitions=partitions)
else:
    manifest["hash"] = {"name": "duckdb.hash", "duckdb_version": duckdb.__version__}
if code is not None:
    manifest["builder_code"] = code
rows = t.num_rows + (1 if MODE == "bad_rows" else 0)
receipt = {"rows": rows, "threads": int(a.threads or 2), "memory_limit": a.memory_limit or "2GB",
           "source_schema": schema, "float_fidelity": fidelity, "elapsed_seconds": 0.01,
           "source": str(data / table)}
if code is not None:
    receipt["builder_code"] = code
(tmp / "_BUCKET_BUILD_RECEIPT.json").write_text(json.dumps(receipt, sort_keys=True))
(tmp / "_BUCKET_MANIFEST.json").write_text(json.dumps(manifest, sort_keys=True))
(tmp / "_COMPLETE").write_text("complete\n")
os.replace(tmp, target)
'''

SIC_TABLES = [
    [
        "medication",
        "--key DrugID --stay-key CaseID --scheme variable_stay_range --threads 1 --memory-limit 1GB",
    ],
    [
        "laboratory",
        "--key LaboratoryID --scheme hash --modulus 2 --threads 1 --memory-limit 1GB",
    ],
]
CV_TABLES = [
    [
        "chartevents",
        "--key itemid --scheme hash --modulus 2 --threads 1 --memory-limit 1GB",
    ]
]


def _git(args, cwd):
    return subprocess.run(
        ["git", "-c", "user.name=seal-test", "-c", "user.email=seal@test", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def make_mirror(root: Path, mode: str = "ok", mark: bool = True) -> tuple[Path, str]:
    """A bare mirror with one commit whose scripts/build_itemid_bucket_cache.py is the fake builder."""
    repo = root / f"repo_{mode}"
    (repo / "scripts").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "src" / "README").write_text("synthetic\n")
    (repo / "src" / "easyicu").mkdir()
    (repo / "src" / "easyicu" / "__init__.py").write_text(
        '"""synthetic seal-clone package"""\n'
    )
    text = FAKE_BUILDER.replace("@MODE@", mode)
    if not mark:
        text = text.replace("bucket_float_fidelity_mismatch", "no_float_check")
    (repo / "scripts" / "build_itemid_bucket_cache.py").write_text(text)
    _git(["init", "-q"], repo)
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "synthetic builder"], repo)
    sha = _git(["rev-parse", "HEAD"], repo)
    mirror = root / f"mirror_{mode}.git"
    _git(["clone", "-q", "--bare", str(repo), str(mirror)], root)
    return mirror, sha


def _table(n: int, key: str, stay: str, value: str, seed: int) -> pa.Table:
    rng = np.random.default_rng(seed)
    v = rng.choice([0.0, 1.5, 2.5], n)
    v[rng.random(n) < 0.2] = -0.0
    return pa.table(
        {
            stay: pa.array(rng.integers(1, 20, n), pa.int64()),
            key: pa.array(rng.integers(1, 7, n), pa.int64()),
            value: pa.array(v, pa.float64()),
        }
    )


def _run_builder(builder: Path, data: Path, table: str, args: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(builder),
            "--data-path",
            str(data),
            "--table",
            table,
            *args.split(),
        ],
        check=True,
        capture_output=True,
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_world(root: Path, mode: str = "ok", mark: bool = True) -> dict:
    """Mirror, sources, the two current formal roots and an unconfined config; returns paths and the SHA."""
    root.mkdir(parents=True, exist_ok=True)
    mirror, sha = make_mirror(root, mode, mark)
    ok_mirror = mirror if mode == "ok" else make_mirror(root, "ok")[0]
    builder = root / "repo_ok" / "scripts" / "build_itemid_bucket_cache.py"
    databases, view, b6 = root / "databases", root / "view" / "raw", root / "b6"
    for d in (
        databases / "sicx" / "medication",
        databases / "sicx" / "laboratory",
        view,
        b6,
    ):
        d.mkdir(parents=True)
    pq.write_table(
        _table(300, "DrugID", "CaseID", "Amount", 1),
        databases / "sicx/medication/1.parquet",
        use_dictionary=False,
    )
    pq.write_table(
        _table(300, "LaboratoryID", "CaseID", "LaboratoryValue", 2),
        databases / "sicx/laboratory/1.parquet",
        use_dictionary=False,
    )
    (databases / "sicx" / "conversion_manifest.json").write_text(
        json.dumps({"tables": ["medication"]})
    )
    # current SIC-style root: per-file links into databases/, one real conversion manifest, built buckets
    sic = b6 / "sic_cur"
    for t in ("medication", "laboratory"):
        (sic / t).mkdir(parents=True)
        os.symlink(databases / "sicx" / t / "1.parquet", sic / t / "1.parquet")
    (sic / "conversion_manifest.json").write_text(
        json.dumps(
            {
                "data_path": str(sic),
                "tables": [{"table": "medication"}],
                "derived_from": {
                    "path": str(databases / "sicx/conversion_manifest.json"),
                    "sha256": _sha(databases / "sicx/conversion_manifest.json"),
                },
            }
        )
    )
    (sic / "_lead_r13_build_medication.log").write_text("log\n")
    for t, args in SIC_TABLES:
        _run_builder(builder, sic, t, args)
    # CareVue: a read-only view, the bucket root it was built in, and the formal farm around it
    pq.write_table(
        _table(400, "itemid", "icustay_id", "valuenum", 3),
        view / "chartevents.parquet",
        use_dictionary=False,
    )
    pq.write_table(pa.table({"subject_id": [1, 2]}), view / "admissions.parquet")
    pq.write_table(pa.table({"amount": [1.0, 2.0]}), view / "inputevents_cv.parquet")
    (view / "conversion_manifest.json").write_text(
        json.dumps({"tables": ["chartevents"]})
    )
    cvb = b6 / "mimic_cv"
    cvb.mkdir()
    os.symlink(view / "chartevents.parquet", cvb / "chartevents.parquet")
    _run_builder(builder, cvb, "chartevents", CV_TABLES[0][1])
    farm = b6 / "mimic_cv_dir"
    farm.mkdir()
    for name in ("admissions.parquet", "chartevents.parquet"):
        os.symlink(view / name, farm / name)
    (farm / "inputevents_cv").mkdir()
    os.symlink(view / "inputevents_cv.parquet", farm / "inputevents_cv" / "1.parquet")
    os.symlink(cvb / "chartevents_bucket", farm / "chartevents_bucket")
    (farm / "conversion_manifest.json").write_text(
        json.dumps(
            {
                "data_path": str(farm),
                "tables": [{"table": "chartevents"}, {"table": "admissions"}],
                "derived_from": {
                    "path": str(view / "conversion_manifest.json"),
                    "sha256": _sha(view / "conversion_manifest.json"),
                },
            }
        )
    )
    config = {
        "b6_root": str(b6),
        "work_root": str(root / "work"),
        "mirror": str(mirror),
        "python": sys.executable,
        "allowed_roots": [str(b6), str(databases), str(view)],
        "confine": "none",
        "memgate": None,
        "order": ["sic", "cv"],
        "dbs": {
            "sic": {
                "current_root": str(sic),
                "database": "sic",
                "mem_gib": 1,
                "threads": 1,
                "timeout_s": 120,
                "tables": SIC_TABLES,
            },
            "cv": {
                "current_root": str(farm),
                "database": "mimic",
                "mem_gib": 1,
                "threads": 1,
                "timeout_s": 120,
                "tables": CV_TABLES,
                "farm": {"name": "mimic_cv_dir", "bucket_root": "mimic_cv_bucket"},
            },
        },
    }
    config_path = root / "seal_config.json"
    config_path.write_text(json.dumps(config, indent=1))
    return {
        "root": root,
        "b6": b6,
        "work": root / "work",
        "sha": sha,
        "config": config_path,
        "mirror": mirror,
        "ok_mirror": ok_mirror,
        "view": view,
        "databases": databases,
    }


def seal_args(world: dict, mode: str, *extra: str) -> list[str]:
    return [
        world["sha"],
        mode,
        "--config",
        str(world["config"]),
        "--cores",
        "sic=188-191",
        "--cores",
        "cv=192-195",
        *extra,
    ]


def rewrite_config(world: dict, **changes) -> None:
    config = json.loads(Path(world["config"]).read_text())
    config.update(changes)
    Path(world["config"]).write_text(json.dumps(config, indent=1))


def make_writable(path: Path) -> None:
    """Let pytest's tmp cleanup remove read-only trees the tool produced."""
    for dirpath, dirnames, filenames in os.walk(path):
        os.chmod(dirpath, 0o700)
        for name in filenames:
            p = os.path.join(dirpath, name)
            if not os.path.islink(p):
                os.chmod(p, 0o600)


def make_bucket(root: Path, rows: int = 40) -> Path:
    """A two-part hash bucket satisfying the full storage contract (src/easyicu/storage_layout.py):
    manifest, receipt and _COMPLETE agree, and derived_from binds a source file beside it."""
    import duckdb

    bucket = root / "t_bucket"
    source = root / "t" / "1.parquet"
    source.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"k": pa.array([1], pa.int64())}), source)
    inventory = []
    for b in range(2):
        (bucket / f"bucket_id={b}").mkdir(parents=True)
        n = np.arange(rows // 2)
        table = pa.table(
            {
                "k": pa.array(n * 2 + b, pa.int64()),
                "v": pa.array(np.where(n % 3 == 0, -0.0, 1.5), pa.float64()),
                ORDINAL: pa.array(n, pa.int64()),
            }
        )
        path = bucket / f"bucket_id={b}" / "part.parquet"
        pq.write_table(table, path, use_dictionary=False)
        inventory.append(
            {
                "relative_path": f"bucket_id={b}/part.parquet",
                "row_count": rows // 2,
                "size": path.stat().st_size,
            }
        )
    st = source.stat()
    manifest = {
        "schema": "easyicu.bucket_storage/1",
        "scheme": "hash",
        "key_column": "k",
        "key_type": "BIGINT",
        "modulus": 2,
        "hash": {"name": "duckdb.hash", "duckdb_version": duckdb.__version__},
        "row_group_target": 131072,
        "builder_version": "easyicu.bucket_builder/2",
        "sort_keys": ["k", ORDINAL],
        "source_order": ORDINAL,
        "inventory": inventory,
        "float_fidelity": {},
        "derived_from": {
            "table": "t",
            "files": [
                {
                    "path": "t/1.parquet",
                    "sha256": _sha(source),
                    "size": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                }
            ],
        },
    }
    (bucket / "_BUCKET_MANIFEST.json").write_text(json.dumps(manifest))
    (bucket / "_BUCKET_BUILD_RECEIPT.json").write_text(json.dumps({"rows": rows}))
    (bucket / "_COMPLETE").write_text("complete\n")
    return bucket
