#!/usr/bin/env python3
"""Write RUN_RECEIPT.json for one formal bench extraction run (executor #739).

The receipt is written once, in the run root, after every database has
exited. It binds the run to the sealed code and to its inputs:

* the sealed 40-character SHA, confirmed against the code actually imported
  (``easyicu.code_identity``: a clean ``git_checkout`` of exactly that SHA);
* the run parameters (modules, batch policy, threads, workers, MemoryMax,
  cores) and, per database: the input realpath and its manifest sha256, the
  exit status, wall time, cgroup and process-tree peaks, the native manifest
  sha256 and its commit, and per table actually read the layout used with
  any fallback reason (``storage_reads``, from the storage owner's run log);
* for SICdb, ``cases_binding``: the prepared conversion manifest records
  ``cases.parquet`` from ``cases.csv.gz`` without content digests and without
  a converter commit, so the receipt carries both sha256s, the manifest record
  and a reproduction: ``cases.csv.gz`` is re-converted with the sealed
  converter into scratch and must be byte-identical, or value-identical (row
  count and a digest per column), to the prepared ``cases.parquet``.

Anything else stops with a :class:`ReceiptError` reason code; nothing is
papered over. The tool reads inputs and outputs only (it never writes under
an input root) and prints no rows.

Usage (bench7.sh calls it with the sealed worktree's ``src`` on PYTHONPATH)::

    formal_run_receipt.py --run-root R --sealed-sha SHA --scratch S \\
        --started-at EPOCH [--param KEY=VALUE ...] DB:DIR:CORES [...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

RECEIPT_FILE = "RUN_RECEIPT.json"
RECEIPT_SCHEMA = "easyicu.formal_run_receipt/1"
SIC_DATABASES = {"sic", "sicdb"}
_COMMIT = re.compile(r"[0-9a-f]{40}")


class ReceiptError(Exception):
    """A formal-run receipt cannot be written; ``code`` names the cause."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.details = details


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sealed_code_identity(sealed_sha: str) -> Dict[str, Any]:
    """The imported EasyICU must be a clean checkout of exactly ``sealed_sha``."""

    import easyicu
    from easyicu.code_identity import SOURCE_GIT_CHECKOUT, resolve_code_identity

    if not isinstance(sealed_sha, str) or _COMMIT.fullmatch(sealed_sha) is None:
        raise ReceiptError("sealed_sha_invalid", "the sealed SHA must be 40 lowercase hex")
    identity = resolve_code_identity(Path(easyicu.__file__).resolve().parent)
    if (
        identity.get("source") != SOURCE_GIT_CHECKOUT
        or identity.get("commit") != sealed_sha
        or identity.get("dirty") is not False
    ):
        raise ReceiptError(
            "code_identity_not_sealed",
            "the imported EasyICU is not a clean checkout of the sealed SHA",
            source=identity.get("source"), commit=identity.get("commit"),
            dirty=identity.get("dirty"), reason=identity.get("reason_code"),
        )
    return {
        "source": identity["source"],
        "commit": identity["commit"],
        "dirty": identity["dirty"],
        "import_path": str(Path(easyicu.__file__).resolve().parent),
    }


def _canonical(value: Any) -> Any:
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def value_digest(parquet: Path) -> Dict[str, Any]:
    """Row count and, per column, its Arrow type and a digest of its values.

    Values are hashed in row order, one canonical JSON token each (floats by
    ``repr``, so the digest is exact; nulls as ``null``). No value leaves the
    function.
    """

    import pyarrow.parquet as pq

    table = pq.read_table(parquet)
    columns = []
    for name, column in zip(table.column_names, table.columns):
        digest = hashlib.sha256()
        for chunk in column.chunks:
            for value in chunk.to_pylist():
                digest.update(json.dumps(_canonical(value)).encode("utf-8"))
                digest.update(b"\n")
        columns.append({"name": name, "type": str(column.type), "sha256": digest.hexdigest()})
    return {"rows": table.num_rows, "columns": columns}


def _manifest_record(prepared: Path, csv: Path, parquet: Path) -> Dict[str, Any]:
    manifest_path = prepared / "conversion_manifest.json"
    if not manifest_path.is_file():
        raise ReceiptError("sic_cases_manifest_missing", "prepared conversion_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tables = manifest.get("tables")
    matches = [
        (index, entry) for index, entry in enumerate(tables if isinstance(tables, list) else [])
        if isinstance(entry, dict) and entry.get("table") == "cases"
    ]
    if len(matches) != 1:
        raise ReceiptError("sic_cases_manifest_record_missing",
                           "the conversion manifest has no single cases record")
    index, entry = matches[0]
    record_input = entry.get("input") or {}
    outputs = [o for o in entry.get("outputs") or [] if isinstance(o, dict)]
    csv_stat, parquet_stat = csv.stat(), parquet.stat()
    agrees = (
        entry.get("status") == "completed"
        and record_input.get("relative_path") == csv.name
        and record_input.get("size_bytes") == csv_stat.st_size
        and record_input.get("mtime_ns") == csv_stat.st_mtime_ns
        and outputs
        and all(o.get("relative_path") == parquet.name for o in outputs)
        and all(o.get("size_bytes") == parquet_stat.st_size for o in outputs)
    )
    if not agrees:
        raise ReceiptError("sic_cases_manifest_record_mismatch",
                           "the cases record does not describe the files on disk")
    return {
        "realpath": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "record_key": f"tables[{index}]",
        "record": {
            "table": entry.get("table"),
            "file": entry.get("file"),
            "status": entry.get("status"),
            "row_count": entry.get("row_count"),
            "input": {k: record_input.get(k) for k in ("relative_path", "size_bytes", "mtime_ns")},
            "outputs": [{k: o.get(k) for k in ("relative_path", "size_bytes", "mtime_ns")}
                        for o in outputs],
            "converter_git_sha": (entry.get("storage_provenance") or {}).get("git_sha"),
        },
    }


def _reconvert_cases(csv: Path, work: Path) -> Path:
    from easyicu.io.data_converter import DataConverter

    work.mkdir(parents=True)
    copy = work / csv.name
    shutil.copy2(csv, copy)
    if sha256_file(copy) != sha256_file(csv):
        raise ReceiptError("sic_cases_copy_mismatch", "the scratch copy of cases.csv.gz differs")
    DataConverter(work, database="sic", parallel_workers=1, verbose=False).convert_all(
        write_manifest=False
    )
    output = work / "cases.parquet"
    if not output.is_file():
        raise ReceiptError("sic_cases_reconversion_missing",
                           "the sealed converter wrote no single cases.parquet")
    return output


def sic_cases_binding(input_dir: Path, scratch: Path, code: Mapping[str, Any]) -> Dict[str, Any]:
    """Bind ``cases.parquet`` to ``cases.csv.gz`` by content and by reproduction."""

    csv = (Path(input_dir) / "cases.csv.gz").resolve(strict=True)
    parquet = (Path(input_dir) / "cases.parquet").resolve(strict=True)
    prepared = parquet.parent
    if csv.parent != prepared:
        raise ReceiptError("sic_cases_split_installation",
                           "cases.csv.gz and cases.parquet resolve to different directories")
    record = _manifest_record(prepared, csv, parquet)
    prepared_sha = sha256_file(parquet)

    work = Path(scratch) / "sic_cases_reconvert"
    if work.exists():
        raise ReceiptError("sic_cases_scratch_not_fresh", "the reconversion scratch already exists")
    try:
        reconverted = _reconvert_cases(csv, work)
        reconverted_sha = sha256_file(reconverted)
        reproduction: Dict[str, Any] = {"reconverted_sha256": reconverted_sha}
        if reconverted_sha == prepared_sha:
            reproduction["result"] = "byte_identical"
        else:
            prepared_values, reconverted_values = value_digest(parquet), value_digest(reconverted)
            if prepared_values != reconverted_values:
                differing = sorted(
                    {c["name"] for c in prepared_values["columns"]}
                    ^ {c["name"] for c in reconverted_values["columns"]}
                    | {a["name"] for a, b in zip(prepared_values["columns"],
                                                  reconverted_values["columns"]) if a != b}
                )
                raise ReceiptError(
                    "sic_cases_reproduction_mismatch",
                    "the sealed converter does not reproduce the prepared cases.parquet",
                    prepared_rows=prepared_values["rows"],
                    reconverted_rows=reconverted_values["rows"],
                    differing_columns=differing,
                )
            reproduction["result"] = "value_identical"
            reproduction["rows"] = prepared_values["rows"]
            reproduction["column_digests"] = prepared_values["columns"]
    finally:
        if work.exists():
            shutil.rmtree(work)
    return {
        "cases_csv_gz": {"realpath": str(csv), "sha256": sha256_file(csv)},
        "cases_parquet": {"realpath": str(parquet), "sha256": prepared_sha},
        "conversion_manifest": record,
        "converter_identity": dict(code),
        "reproduction": reproduction,
    }


def _status_exits(run_root: Path) -> Dict[str, Any]:
    exits: Dict[str, Any] = {}
    status = run_root / "status"
    for line in status.read_text(encoding="utf-8").splitlines() if status.is_file() else []:
        match = re.fullmatch(r"(\S+) exit=(\S+)", line.strip())
        if match:
            exits[match.group(1)] = match.group(2)
    return exits


def _database_section(run_root: Path, db: str, input_dir: str, cores: str,
                      exit_status: Any, sealed_sha: str) -> Dict[str, Any]:
    out = run_root / db
    input_real = Path(input_dir).resolve()
    input_manifest = input_real / "conversion_manifest.json"
    section: Dict[str, Any] = {
        "input_realpath": str(input_real),
        "input_manifest_sha256": sha256_file(input_manifest) if input_manifest.is_file() else None,
        "cores": cores,
        "exit_status": exit_status,
        "output_root": str((out / "export").resolve()),
    }
    if str(exit_status) != "0":
        raise ReceiptError("database_run_failed", f"{db} exited {exit_status}", database=db)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    if summary.get("exception") or summary.get("errors") or summary.get("worker_failures"):
        raise ReceiptError("database_run_reported_errors", f"{db} reported errors", database=db)
    section.update({
        "wall_seconds": summary.get("secs"),
        "cgroup_memory_max": summary.get("cgroup_memory_max"),
        "cgroup_peak_gib": summary.get("cgroup_peak_gib"),
        "peak_tree_rss_gib": summary.get("peak_tree_rss_gib"),
        "batch_size": summary.get("batch_size"),
    })
    manifest_path = out / "export" / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = manifest.get("runtime_provenance") or {}
    if (
        provenance.get("easyicu_git_commit") != sealed_sha
        or provenance.get("easyicu_git_dirty") is not False
        or provenance.get("easyicu_code_identity_source", "git_checkout") != "git_checkout"
    ):
        raise ReceiptError("native_manifest_not_sealed",
                           f"{db} native manifest was not produced by the sealed checkout",
                           database=db)
    section["native_manifest"] = {
        "sha256": sha256_file(manifest_path),
        "easyicu_git_commit": provenance.get("easyicu_git_commit"),
        "easyicu_git_dirty": provenance.get("easyicu_git_dirty"),
    }
    if "storage_reads" not in manifest:
        raise ReceiptError("storage_reads_missing",
                           f"{db} native manifest lists no per-table storage reads", database=db)
    section["storage_reads"] = manifest["storage_reads"]
    section["storage_fallbacks"] = manifest.get("storage_fallbacks") or []
    return section


def build_run_receipt(
    run_root: Path,
    sealed_sha: str,
    specs: Iterable[str],
    *,
    scratch: Path,
    started_at: Optional[float] = None,
    parameters: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    run_root = Path(run_root)
    code = sealed_code_identity(sealed_sha)
    exits = _status_exits(run_root)
    databases: Dict[str, Any] = {}
    for spec in specs:
        db, input_dir, cores = spec.split(":", 2)
        section = _database_section(run_root, db, input_dir, cores, exits.get(db), sealed_sha)
        if db in SIC_DATABASES:
            section["cases_binding"] = sic_cases_binding(Path(input_dir), Path(scratch), code)
        databases[db] = section
    finished = time.time()
    peaks = [s["cgroup_peak_gib"] for s in databases.values() if s.get("cgroup_peak_gib") is not None]
    return {
        "schema_version": RECEIPT_SCHEMA,
        "sealed_sha": sealed_sha,
        "code_identity": code,
        "parameters": dict(parameters or {}),
        "run_root": str(run_root.resolve()),
        "wall_seconds": None if started_at is None else round(finished - float(started_at), 1),
        "max_database_cgroup_peak_gib": max(peaks) if peaks else None,
        "databases": databases,
    }


def write_run_receipt(run_root: Path, receipt: Mapping[str, Any]) -> Path:
    target = Path(run_root) / RECEIPT_FILE
    if target.exists():
        raise ReceiptError("run_receipt_exists", "RUN_RECEIPT.json is written once")
    temporary = target.with_name(f".{RECEIPT_FILE}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--sealed-sha", required=True)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--started-at", type=float)
    parser.add_argument("--param", action="append", default=[])
    parser.add_argument("specs", nargs="+")
    args = parser.parse_args(argv)
    parameters = dict(item.split("=", 1) for item in args.param)
    try:
        receipt = build_run_receipt(
            args.run_root, args.sealed_sha, args.specs, scratch=args.scratch,
            started_at=args.started_at, parameters=parameters,
        )
        path = write_run_receipt(args.run_root, receipt)
    except ReceiptError as error:
        sys.stderr.write(json.dumps({"code": error.code, "message": str(error),
                                     "details": error.details}, default=str) + "\n")
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
