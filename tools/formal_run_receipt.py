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
* the inputs did not change: ``INPUT_SNAPSHOT.json`` (taken before any job,
  ``easyicu.formal_inputs``) and ``INPUT_SNAPSHOT_POST.json`` (the same set
  hashed after every job exited) must exist and agree per database; the
  receipt records both digests, the files, bytes and hash seconds, and a
  declared view's chain;
* every file the run consumed (``storage_consumed``, the storage owner's
  ``table_consumed`` events from every worker) is in the snapshot by real
  path, its link still resolves to that real path, and it carries the
  snapshot's sha256 and how it was admitted (``admitted_by``: ``data_root``
  or ``declared_view``, from the storage owner). An empty set is refused.
  The log's own keys (``event``, ``root``) are checked (the event is
  ``table_consumed``, the root is this database's input) and not carried;
* per table read, the directory it was served from and its real path; for a
  bucketed table, ``bucket_receipt`` (the storage owner's
  ``bucket_source_receipt``: manifest and build-receipt sha256, the
  ``derived_from`` sources by real path); the installation must be one the
  snapshot covers, where the builder's sha256 were compared with the hashes;
* the parent run binding's digest from the native manifest
  (``storage_binding``, see its ``note``) and its per-table layout record
  (``storage_binding["tables"]``): every consumed or read table must have one
  (``storage_layout_unrecorded``) and every read in it is checked against the
  snapshot (flat under this input; bucket installations, manifests and
  ``derived_from`` sources as snapshotted);
* no storage fallback: a formal run that read through one is
  ``storage_fallback_in_formal_run`` (with counts per reason);
* ``REQUIRED_LAYOUT.json`` in the run root (``complete: true``, per database
  {table: {layout: flat|bucket, root_realpath}}, schema /2): its sha256 is
  recorded and every read of every table must have been served from the
  required layout at the required root (a flat table defaults to this input),
  else ``layout_deviation``;
* the code carries no working-tree diff: the code identity's and the native
  manifest's ``diff_sha256`` are copied and must be null;
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
REQUIRED_LAYOUT_FILE = "REQUIRED_LAYOUT.json"
REQUIRED_LAYOUT_SCHEMA = "easyicu.required_layout/2"
LAYOUT_KINDS = ("flat", "bucket")
# A consumed entry in the receipt: the storage owner's table_consumed event
# without the log's own keys (``event``, ``root``), which are checked and dropped.
CONSUMED_KEYS = ("table", "path", "realpath", "format")
CONSUMED_LOG_KEYS = ("event", "root")
RECEIPT_SCHEMA = "easyicu.formal_run_receipt/1"
SIC_DATABASES = {"sic", "sicdb"}
_COMMIT = re.compile(r"[0-9a-f]{40}")


class ReceiptError(Exception):
    """A formal-run receipt cannot be written; ``code`` names the cause."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.details = details


def load_json_object(path: Path, what: str) -> Dict[str, Any]:
    """A JSON object input of the receipt; anything else is a typed stop."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReceiptError("receipt_input_missing", f"{what} is missing",
                           input=what, path=str(path)) from exc
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ReceiptError("receipt_input_malformed", f"{what} is not readable JSON",
                           input=what, path=str(path)) from exc
    if not isinstance(payload, dict):
        raise ReceiptError("receipt_input_malformed", f"{what} is not a JSON object",
                           input=what, path=str(path))
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sealed_code_identity(sealed_sha: str) -> Dict[str, Any]:
    """The imported EasyICU must be a clean checkout of exactly ``sealed_sha``."""

    import easyicu
    from easyicu.code_identity import (
        SOURCE_GIT_CHECKOUT,
        installed_dist_version,
        resolve_code_identity,
    )

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
    # G1: a sealed checkout carries no working-tree diff; the field is
    # copied, and anything but an explicit null is a stop.
    if "diff_sha256" not in identity or identity["diff_sha256"] is not None:
        raise ReceiptError(
            "code_identity_diff_present",
            "the imported EasyICU checkout records a working-tree diff (or none at all)",
            diff_sha256=identity.get("diff_sha256", "<unrecorded>"),
        )
    return {
        "source": identity["source"],
        "commit": identity["commit"],
        "dirty": identity["dirty"],
        "diff_sha256": identity["diff_sha256"],
        "version": identity.get("version"),
        "installed_dist_version": installed_dist_version(),
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
    manifest = load_json_object(manifest_path, "prepared conversion_manifest.json")
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


def _within(path: str, root: Path) -> bool:
    try:
        Path(path).relative_to(root)
    except ValueError:
        return False
    return True


def _storage_reads(db: str, reads: List[Dict[str, Any]],
                   snapshot: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Each table read with where it was served from; bucket reads with their binding."""
    from easyicu.storage_layout import LAYOUT_FLAT, StorageIntegrityError, bucket_source_receipt

    snapshotted = {b["layout_realpath"] for b in snapshot.get("buckets") or ()}
    receipt = []
    for read in reads:
        entry = dict(read)
        realpath = entry.get("layout_realpath")
        if not entry.get("layout_path") or not realpath:
            raise ReceiptError("storage_read_layout_missing",
                               f"{db} {entry.get('table')} read names no layout directory",
                               database=db, table=entry.get("table"))
        if entry.get("layout") != LAYOUT_FLAT:
            try:
                binding = bucket_source_receipt(entry["layout_path"])
            except (StorageIntegrityError, OSError) as exc:
                raise ReceiptError("bucket_receipt_unverifiable",
                                   f"{db} {entry['table']}: {getattr(exc, 'reason', exc)}",
                                   database=db, table=entry["table"],
                                   reason=getattr(exc, "reason", None)) from exc
            if binding["layout_realpath"] != realpath or realpath not in snapshotted:
                raise ReceiptError("storage_read_outside_snapshot",
                                   f"{db} {entry['table']} bucket is not one the input "
                                   "snapshot covers", database=db, table=entry["table"])
            entry["bucket_receipt"] = binding
        receipt.append(entry)
    return receipt


def _admit_consumed(base: str, table: str, path: str) -> Dict[str, Any]:
    """How the storage owner admits a consumed file (the single call site of its rule)."""
    from easyicu.storage_layout import admit_consumed_realpath

    return admit_consumed_realpath(base, table, path)


def _consumed(db: str, input_dir: str, consumed: Any,
              snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    """Every consumed file is a snapshotted file, by real path; none is an empty set."""
    from easyicu.storage_layout import StorageIntegrityError

    if not isinstance(consumed, list) or not consumed:
        raise ReceiptError("storage_consumed_empty",
                           f"{db} native manifest lists no consumed files", database=db)
    hashed = {f["realpath"]: f for f in snapshot["files"]}
    # The same file under another spelling (letter case on a case-insensitive
    # file system) is the same (device, inode).
    by_identity = {(f.get("dev"), f.get("ino")): f for f in snapshot["files"]
                   if f.get("dev") is not None}
    from easyicu.storage_layout import CONSUMED_EVENT

    files, declared = [], False
    input_real = os.path.realpath(input_dir)
    for raw in consumed:
        if not isinstance(raw, dict) or not all(
            isinstance(raw.get(k), str) for k in CONSUMED_LOG_KEYS + CONSUMED_KEYS
        ) or raw["event"] != CONSUMED_EVENT:
            raise ReceiptError("receipt_input_malformed",
                               f"{db} consumed entry is not a {CONSUMED_EVENT} event "
                               "{event, root, table, path, realpath, format}", database=db)
        # G5: the log's own keys are checked here and not carried into the
        # receipt: every entry was read under this database's input.
        if os.path.realpath(raw["root"]) != input_real:
            raise ReceiptError("storage_consumed_other_root",
                               f"{db} {raw['table']} was consumed under another data root",
                               database=db, table=raw["table"], root=raw["root"])
        item = {k: raw[k] for k in CONSUMED_KEYS}
        if os.path.realpath(item["path"]) != item["realpath"]:
            raise ReceiptError("storage_consumed_realpath_changed",
                               f"{db} {item['table']} consumed file no longer resolves to "
                               "what the run read", database=db, table=item["table"],
                               path=item["path"])
        held = hashed.get(item["realpath"])
        identity_note = None
        if held is None:
            # Only the very same file qualifies: same device and inode as a
            # snapshotted entry (a different file never does, whatever its name).
            try:
                stat = os.stat(item["realpath"])
            except OSError:
                stat = None
            if stat is not None:
                held = by_identity.get((stat.st_dev, stat.st_ino))
            if held is not None:
                identity_note = (
                    "matched by (device, inode): the reader named this file "
                    f"{item['realpath']}, the snapshot {held['realpath']}"
                )
        if held is None:
            raise ReceiptError("storage_consumed_outside_snapshot",
                               f"{db} {item['table']} consumed a file outside the input "
                               "snapshot", database=db, table=item["table"], path=item["path"])
        entry = {**item, "sha256": held["sha256"], "size": held["size"]}
        if identity_note is not None:
            entry["snapshot_realpath"] = held["realpath"]
            entry["note"] = identity_note
        if item["format"] in {"csv", "parquet"}:
            try:
                admission = _admit_consumed(input_dir, item["table"], item["path"])
            except StorageIntegrityError as exc:
                raise ReceiptError("storage_consumed_not_admitted",
                                   f"{db} {item['table']}: {getattr(exc, 'reason', exc)}",
                                   database=db, table=item["table"], path=item["path"],
                                   reason=getattr(exc, "reason", None)) from exc
            if admission["target_realpath"] != item["realpath"]:
                raise ReceiptError("storage_consumed_realpath_changed",
                                   f"{db} {item['table']} admission names another file",
                                   database=db, table=item["table"], path=item["path"])
            entry["admitted_by"] = admission["admitted_by"]
            declared = declared or admission["admitted_by"] == "declared_view"
        files.append(entry)
    result: Dict[str, Any] = {
        "files": sorted(files, key=lambda e: (e["table"], e["path"], e["format"])),
        "count": len(files),
        "bytes": sum(e["size"] for e in {e["realpath"]: e for e in files}.values()),
    }
    if declared:
        chain = snapshot.get("declared_view_chain")
        if not chain:
            raise ReceiptError("declared_view_chain_missing",
                               f"{db} read through a declared view the snapshot has no chain "
                               "for", database=db)
        result["declared_view_chain"] = chain
    return result


def _read_kind(layout: Any) -> str:
    """``flat`` or ``bucket`` for a storage_reads layout (``flat``, ``bucket_<scheme>``)."""
    return "flat" if layout == "flat" else "bucket"


def _layout_tables(db: str, input_dir: str, binding: Mapping[str, Any],
                   consumed: List[Dict[str, Any]], reads: List[Dict[str, Any]],
                   snapshot: Mapping[str, Any]) -> Dict[str, str]:
    """G2: every consumed or read table has a layout record, and every record is checked.

    Keyed on ``storage_binding["tables"]`` (the storage owner's per-table
    record of what served each consumed table). Each of a table's ``reads`` is
    checked, not only the primary one: a flat read must be served under this
    database's input (with the snapshot's declared-view chain if it names
    one); a bucket read must be an installation the snapshot covers, record
    its ``derived_from``, and name a manifest and sources whose sha256 are the
    snapshot's. Returns {table: [{layout, root_realpath}, ...]}: every distinct
    read of the table, for the layout requirement.
    """
    tables = binding.get("tables")
    if not isinstance(tables, dict):
        raise ReceiptError("storage_layout_unrecorded",
                           f"{db} native manifest records no per-table layouts", database=db)
    for table in sorted({e["table"] for e in consumed} | {r["table"] for r in reads}):
        if table not in tables:
            raise ReceiptError("storage_layout_unrecorded",
                               f"{db} {table} was read but no layout record names what served it",
                               database=db, table=table)
    hashed = {f["realpath"]: f for f in snapshot["files"]}
    buckets = {b["layout_realpath"] for b in snapshot.get("buckets") or ()}
    input_real = os.path.realpath(input_dir)
    chain = snapshot.get("declared_view_chain")
    observed: Dict[str, List[Dict[str, str]]] = {}
    for table, record in tables.items():
        records = record.get("reads") if isinstance(record, dict) else None
        if not isinstance(records, list) or not records:
            raise ReceiptError("storage_layout_unrecorded",
                               f"{db} {table} layout record lists no reads",
                               database=db, table=table)
        for read in records:
            kind = read.get("layout") if isinstance(read, dict) else None
            if kind not in LAYOUT_KINDS or not isinstance(read.get("root"), str):
                raise ReceiptError("storage_layout_unrecorded",
                                   f"{db} {table} has a read without a layout and root",
                                   database=db, table=table)
            real = os.path.realpath(read["root"])
            if kind == "flat":
                if real != input_real:
                    raise ReceiptError("storage_read_other_root",
                                       f"{db} {table} flat read was served under another root",
                                       database=db, table=table)
                if read.get("derived_from") is not None and read["derived_from"] != chain:
                    raise ReceiptError("declared_view_chain_mismatch",
                                       f"{db} {table} flat read names a declared-view chain "
                                       "the snapshot does not hold", database=db, table=table)
                continue
            if real not in buckets:
                raise ReceiptError("storage_read_outside_snapshot",
                                   f"{db} {table} bucket is not one the input snapshot covers",
                                   database=db, table=table)
            derived, manifest = read.get("derived_from"), read.get("manifest")
            if read.get("derived_from_recorded") is not True or not isinstance(derived, dict):
                raise ReceiptError("bucket_derived_from_unrecorded",
                                   f"{db} {table} bucket read records no derived_from",
                                   database=db, table=table)
            held = hashed.get(os.path.realpath(manifest["path"])) if isinstance(
                manifest, dict) and isinstance(manifest.get("path"), str) else None
            if held is None or held["sha256"] != manifest.get("sha256"):
                raise ReceiptError("bucket_manifest_not_snapshotted",
                                   f"{db} {table} bucket manifest the run read is not the "
                                   "snapshotted one", database=db, table=table)
            for source in derived.get("files") or ():
                held = hashed.get(source.get("realpath"))
                if held is None:
                    raise ReceiptError("bucket_derived_from_outside_snapshot",
                                       f"{db} {table} bucket source is not in the snapshot",
                                       database=db, table=table)
                if held["sha256"] != source.get("sha256"):
                    raise ReceiptError("bucket_derived_from_sha_mismatch",
                                       f"{db} {table} bucket source differs from the snapshot",
                                       database=db, table=table)
        primary = record.get("layout")
        if primary not in LAYOUT_KINDS or not any(r["layout"] == primary for r in records):
            raise ReceiptError("storage_layout_unrecorded",
                               f"{db} {table} layout record names no primary layout",
                               database=db, table=table)
        observed[table] = sorted(
            {(r["layout"], os.path.realpath(r["root"])) for r in records})
        observed[table] = [{"layout": k, "root_realpath": r} for k, r in observed[table]]
    # The run log's per-read layout (storage_reads) agrees with the record.
    for read in reads:
        kind = _read_kind(read.get("layout"))
        if not any(
            r["layout"] == kind
            and (kind == "flat" or os.path.realpath(r["root"]) == read.get("layout_realpath"))
            for r in tables[read["table"]]["reads"]
        ):
            raise ReceiptError("storage_layout_disagrees",
                               f"{db} {read['table']} read as {read.get('layout')} has no "
                               "matching layout record", database=db, table=read["table"])
    return observed


def _refuse_fallbacks(db: str, fallbacks: Any) -> None:
    """G3: a formal run reads only what its layout says; any storage fallback is a stop."""
    if not fallbacks:
        return
    kinds: Dict[str, int] = {}
    for entry in fallbacks if isinstance(fallbacks, list) else [fallbacks]:
        reason = entry.get("reason") if isinstance(entry, dict) else None
        kinds[str(reason)] = kinds.get(str(reason), 0) + 1
    raise ReceiptError("storage_fallback_in_formal_run",
                       f"{db} read through {sum(kinds.values())} storage fallback(s)",
                       database=db, count=sum(kinds.values()), kinds=dict(sorted(kinds.items())))


def load_required_layout(run_root: Path, databases: Iterable[str]) -> tuple[Dict[str, Any], str]:
    """G4: the run's REQUIRED_LAYOUT.json and its sha256.

    {"schema": "easyicu.required_layout/2", "complete": true, "databases":
    {db: {table: {"layout": "flat" | "bucket", "root_realpath": "<abs>"}}}}.
    ``root_realpath`` is required for a bucket (the installation it must be
    served from) and optional for a flat table (default: the database's
    input). ``complete`` states the list names every table the run may read.
    Anything else, schema /1 included, or a database the run extracts
    without an entry, is a stop.
    """
    path = Path(run_root) / REQUIRED_LAYOUT_FILE
    if not path.is_file():
        raise ReceiptError("required_layout_missing",
                           f"the run root has no {REQUIRED_LAYOUT_FILE}", path=str(path))
    payload = load_json_object(path, REQUIRED_LAYOUT_FILE)
    listed = payload.get("databases")
    if payload.get("schema") != REQUIRED_LAYOUT_SCHEMA or not isinstance(listed, dict):
        raise ReceiptError("receipt_input_malformed",
                           f"{REQUIRED_LAYOUT_FILE} is not a {REQUIRED_LAYOUT_SCHEMA} list",
                           input=REQUIRED_LAYOUT_FILE)
    if payload.get("complete") is not True:
        raise ReceiptError("required_layout_incomplete",
                           f"{REQUIRED_LAYOUT_FILE} does not state complete: true")
    for db in databases:
        tables = listed.get(db)
        if not isinstance(tables, dict) or not tables:
            raise ReceiptError("required_layout_missing",
                               f"{REQUIRED_LAYOUT_FILE} lists no layout for {db}", database=db)
        names = [str(t).lower() for t in tables]
        if len(set(names)) != len(names) or not all(
                _required_entry_valid(v) for v in tables.values()):
            raise ReceiptError("receipt_input_malformed",
                               f"{REQUIRED_LAYOUT_FILE} {db}: each table once, as "
                               "{layout: flat|bucket, root_realpath} (required for a bucket)",
                               input=REQUIRED_LAYOUT_FILE, database=db)
    return payload, sha256_file(path)


def _required_entry_valid(entry: Any) -> bool:
    if not isinstance(entry, dict) or set(entry) - {"layout", "root_realpath"}:
        return False
    root = entry.get("root_realpath")
    if entry.get("layout") == "bucket":
        return isinstance(root, str) and os.path.isabs(root)
    return entry.get("layout") == "flat" and (
        root is None and "root_realpath" not in entry
        or isinstance(root, str) and os.path.isabs(root))


def _layout_requirement(db: str, input_dir: str, required: Optional[Mapping[str, Any]],
                        observed: Mapping[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    """G4: every read of every table matches its required layout and root realpath.

    Both sides are compared after resolving symlinks; a flat table without
    ``root_realpath`` must be served under the database's input.
    """
    if not isinstance(required, Mapping):
        raise ReceiptError("required_layout_missing",
                           f"{REQUIRED_LAYOUT_FILE} lists no layout for {db}", database=db)
    wanted = {}
    for table, entry in required.items():
        root = entry.get("root_realpath") or input_dir
        wanted[str(table).lower()] = {"layout": entry["layout"],
                                      "root_realpath": os.path.realpath(root)}
    deviations = []
    for table, reads in sorted(observed.items()):
        need = wanted.get(table.lower())
        for read in reads:
            if need != read:
                deviations.append({"table": table, "required": need, "observed": read})
    if deviations:
        raise ReceiptError("layout_deviation",
                           f"{db}: {len(deviations)} read(s) not served from the required "
                           "layout and root", database=db, deviations=deviations)
    read_tables = {t.lower() for t in observed}
    return {
        "required": {t: wanted[str(t).lower()] for t in sorted(required)},
        "observed": dict(sorted(observed.items())),
        "not_read": sorted(t for t in required if str(t).lower() not in read_tables),
    }


def _snapshots(run_root: Path) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """The pre- and post-run input snapshots, and their file digests."""
    from easyicu.formal_inputs import SNAPSHOT_SCHEMA

    pre_path, post_path = run_root / "INPUT_SNAPSHOT.json", run_root / "INPUT_SNAPSHOT_POST.json"
    pre = load_json_object(pre_path, "INPUT_SNAPSHOT.json")
    post = load_json_object(post_path, "INPUT_SNAPSHOT_POST.json")
    for name, payload in (("INPUT_SNAPSHOT.json", pre), ("INPUT_SNAPSHOT_POST.json", post)):
        if payload.get("schema") != SNAPSHOT_SCHEMA or not isinstance(
                payload.get("databases"), dict):
            raise ReceiptError("receipt_input_malformed", f"{name} is not an input snapshot",
                               input=name)
    return pre, post, {"sha256": sha256_file(pre_path), "post_sha256": sha256_file(post_path)}


def _snapshot_section(db: str, input_dir: str, pre: Mapping[str, Any],
                      post: Mapping[str, Any]) -> Dict[str, Any]:
    before, after = pre["databases"].get(db), post["databases"].get(db)
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ReceiptError("input_snapshot_missing",
                           f"{db} has no pre- and post-run input snapshot", database=db)
    if os.path.realpath(before.get("data_path", "")) != os.path.realpath(input_dir):
        raise ReceiptError("input_snapshot_other_input",
                           f"{db} snapshot is of another input", database=db)
    if before.get("digest") != after.get("digest"):
        raise ReceiptError("input_changed_during_run",
                           f"{db} inputs differ before and after the run", database=db)
    return {
        "digest": before["digest"],
        "post_digest": after["digest"],
        "post_equal": True,
        "files": before.get("file_count"),
        "bytes": before.get("bytes"),
        "hash_seconds": before.get("hash_seconds"),
        "post_hash_seconds": after.get("hash_seconds"),
        "declared_view_chain": before.get("declared_view_chain"),
        "buckets": [
            {"table": b["table"], "layout_realpath": b["layout_realpath"],
             "derived_from_matches_snapshot": all(
                 s.get("sha256_matches_snapshot") for s in b["derived_from"])}
            for b in before.get("buckets") or ()
        ],
    }


def _database_section(run_root: Path, db: str, input_dir: str, cores: str,
                      exit_status: Any, sealed_sha: str,
                      version: Optional[str] = None,
                      snapshots: Optional[tuple] = None,
                      required: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    out = run_root / db
    input_real = Path(input_dir).resolve()
    input_manifest = input_real / "conversion_manifest.json"
    data_root = input_manifest.resolve().parent if input_manifest.is_file() else input_real
    section: Dict[str, Any] = {
        "input_path": str(input_dir),
        "input_realpath": str(input_real),
        "data_root_realpath": str(data_root),
        "input_manifest_sha256": sha256_file(input_manifest) if input_manifest.is_file() else None,
        "cores": cores,
        "exit_status": exit_status,
        "output_root": str((out / "export").resolve()),
    }
    if str(exit_status) != "0":
        raise ReceiptError("database_run_failed", f"{db} exited {exit_status}", database=db)
    summary = load_json_object(out / "summary.json", f"{db} summary.json")
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
    manifest = load_json_object(manifest_path, f"{db} native _manifest.json")
    provenance = manifest.get("runtime_provenance") or {}
    if not isinstance(provenance, dict):
        raise ReceiptError("receipt_input_malformed", f"{db} runtime_provenance is not an object",
                           input=f"{db} native _manifest.json", path=str(manifest_path))
    if (
        provenance.get("easyicu_git_commit") != sealed_sha
        or provenance.get("easyicu_git_dirty") is not False
        or provenance.get("easyicu_code_identity_source") != "git_checkout"
    ):
        raise ReceiptError("native_manifest_not_sealed",
                           f"{db} native manifest was not produced by the sealed checkout",
                           database=db)
    if "easyicu_git_diff_sha256" not in provenance or (
        provenance["easyicu_git_diff_sha256"] is not None
    ):
        raise ReceiptError("native_manifest_diff_present",
                           f"{db} native manifest records a working-tree diff (or none at all)",
                           database=db,
                           diff_sha256=provenance.get("easyicu_git_diff_sha256", "<unrecorded>"))
    if provenance.get("easyicu_version") != version:
        raise ReceiptError("native_manifest_version_mismatch",
                           f"{db} native manifest names another EasyICU version than the "
                           "sealed checkout", database=db,
                           manifest_version=provenance.get("easyicu_version"),
                           sealed_version=version)
    section["native_manifest"] = {
        "sha256": sha256_file(manifest_path),
        "easyicu_git_commit": provenance.get("easyicu_git_commit"),
        "easyicu_git_dirty": provenance.get("easyicu_git_dirty"),
        "easyicu_git_diff_sha256": provenance["easyicu_git_diff_sha256"],
        "easyicu_version": provenance.get("easyicu_version"),
        # Diagnostic only: the environment's installed distribution.
        "easyicu_installed_dist_version": provenance.get("easyicu_installed_dist_version"),
    }
    if "storage_reads" not in manifest:
        raise ReceiptError("storage_reads_missing",
                           f"{db} native manifest lists no per-table storage reads", database=db)
    pre, post = snapshots if snapshots is not None else _snapshots(run_root)[:2]
    section["input_snapshot"] = _snapshot_section(db, input_dir, pre, post)
    before = pre["databases"][db]
    section["storage_reads"] = _storage_reads(db, list(manifest["storage_reads"]), before)
    section["storage_consumed"] = _consumed(db, input_dir, manifest.get("storage_consumed"),
                                            before)
    binding = manifest.get("storage_binding")
    if not isinstance(binding, dict) or not isinstance(binding.get("digest"), str):
        raise ReceiptError("storage_binding_missing",
                           f"{db} native manifest records no run binding digest", database=db)
    section["storage_binding"] = {
        **binding,
        "note": "digest of the parent extraction's run binding, recorded at publication; "
                "workers and isolated children verify against it and extend it in their "
                "own processes (not merged into this digest)",
    }
    observed = _layout_tables(db, input_dir, binding, section["storage_consumed"]["files"],
                              section["storage_reads"], before)
    _refuse_fallbacks(db, manifest.get("storage_fallbacks"))
    section["storage_fallbacks"] = []
    section["layout_requirement"] = _layout_requirement(db, input_dir, required, observed)
    return section


def _input_roots(input_dir: str) -> List[Path]:
    """Every real directory a database input reads from (its root, its data root,
    and a SIC prepared tree behind cases.*), for the containment check."""
    base = Path(input_dir)
    roots = [base.resolve()]
    for name in ("conversion_manifest.json", "cases.csv.gz", "cases.parquet"):
        if (base / name).exists():
            roots.append((base / name).resolve().parent)
    return roots


def _refuse_paths_inside_inputs(paths: Mapping[str, Path], specs: List[str]) -> None:
    """NB9P-SCRATCH-CONTAINMENT: nothing the receipt writes may sit in an input."""
    for spec in specs:
        _db, input_dir, _cores = spec.split(":", 2)
        for root in _input_roots(input_dir):
            for kind, path in paths.items():
                if _within(str(Path(path).resolve()), root):
                    raise ReceiptError(f"{kind}_inside_input",
                                       f"the receipt's {kind} lies inside an input",
                                       path=str(path), input=str(root))


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
    specs = list(specs)
    _refuse_paths_inside_inputs({"scratch": Path(scratch), "run_root": run_root}, specs)
    code = sealed_code_identity(sealed_sha)
    pre, post, snapshot_files = _snapshots(run_root)
    required, required_sha = load_required_layout(
        run_root, [spec.split(":", 2)[0] for spec in specs])
    exits = _status_exits(run_root)
    databases: Dict[str, Any] = {}
    for spec in specs:
        db, input_dir, cores = spec.split(":", 2)
        try:
            section = _database_section(run_root, db, input_dir, cores, exits.get(db),
                                        sealed_sha, code.get("version"), (pre, post),
                                        required["databases"][db])
            if db in SIC_DATABASES:
                section["cases_binding"] = sic_cases_binding(Path(input_dir), Path(scratch), code)
        except (KeyError, TypeError, AttributeError) as exc:
            # A field of an input has the wrong shape: a typed stop, never a pass.
            raise ReceiptError("receipt_input_malformed", f"{db}: {type(exc).__name__}: {exc}",
                               database=db) from exc
        databases[db] = section
    finished = time.time()
    peaks = [s["cgroup_peak_gib"] for s in databases.values() if s.get("cgroup_peak_gib") is not None]
    return {
        "schema_version": RECEIPT_SCHEMA,
        "sealed_sha": sealed_sha,
        "easyicu_version": code.get("version"),
        "code_identity": code,
        "input_snapshot": snapshot_files,
        "required_layout": {"path": str((run_root / REQUIRED_LAYOUT_FILE).resolve()),
                            "sha256": required_sha, "complete": True},
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
