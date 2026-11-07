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
  hashed after every job exited) must exist and agree per database: each is
  recomputed from its own file list (digest, file count, bytes, database and
  input root) and the full (real path, size, sha256) sets must be equal; the
  receipt records both digests, the files, bytes and hash seconds, and a
  declared view's chain;
* a CSV read is allowed only as the table's input CSV listed in the
  conversion manifest (its own read type ``csv``, consumed and snapshotted);
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
* no declared table read as empty for want of a physical source unless the
  root's REQUIRED_LAYOUT.json ``declared_absent`` allowlist excuses it: the
  native manifest's ``storage_unavailable`` list is copied (an empty list
  included) with the entry that excused each record, and any other record is
  ``storage_unavailable_in_formal_run``, naming the tables and their causes;
* ``REQUIRED_LAYOUT.json`` in the run root (``complete: true``, per database
  {table: {layout: flat|csv|bucket, root_realpath}}, schema /2): its sha256 is
  recorded and every read of every table must have been served from the
  required layout at the required root (a flat table defaults to this input),
  else ``layout_deviation``;
* the code carries no working-tree diff: the code identity's and the native
  manifest's ``diff_sha256`` are copied and must be null;
* copy-then-exec: the run executed only staged copies of the sealed tool
  set (``tools/formal/FORMAL_TOOLS.json``: the wrapper, the boot, both
  drivers, memgate.py, this tool and its allowlist). ``.formal_tools`` in
  the run root holds exactly those copies, read-only, each equal to the
  seal tree's blob and to the sha the wrapper staged, and this tool runs as
  its copy, else ``formal_tool_script_mismatch`` (``tool_scripts``); every
  Python step ran under the staged boot (``python -S -P formal_boot.py``)
  with easyicu from the clone's src at the seal, duckdb/pyarrow from the
  named site-packages and nothing from elsewhere (``import_origin``; this
  step checked live), else ``formal_import_origin_mismatch``;
* REQUIRED_LAYOUT.json and the approved allowlist are parsed rejecting a
  duplicate key at any level (``receipt_input_duplicate_key``); names are
  deduplicated by casefold;
* for SICdb, ``cases_binding``: the prepared conversion manifest records
  ``cases.parquet`` from ``cases.csv.gz`` without content digests and without
  a converter commit, so the receipt carries both sha256s, the manifest record
  and a reproduction: ``cases.csv.gz`` is re-converted with the sealed
  converter into scratch and must be byte-identical, or value-identical (row
  count and a digest per column), to the prepared ``cases.parquet``.

Anything else stops with a :class:`ReceiptError` reason code; nothing is
papered over. The tool reads inputs and outputs only (it never writes under
an input root) and prints no rows.

Usage (bench7v3f.sh runs its staged copy under the staged boot, ``formal_boot.py``)::

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
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

RECEIPT_FILE = "RUN_RECEIPT.json"
REQUIRED_LAYOUT_FILE = "REQUIRED_LAYOUT.json"
REQUIRED_LAYOUT_SCHEMA = "easyicu.required_layout/2"
LAYOUT_KINDS = ("flat", "csv", "bucket")
FLAT_KINDS = ("flat", "csv")  # served under the input root (a listed CSV is its own kind)
# A consumed entry in the receipt: the storage owner's table_consumed event
# without the log's own keys (``event``, ``root``), which are checked and dropped.
CONSUMED_KEYS = ("table", "path", "realpath", "format")
DECLARED_ABSENT_KEYS = ("db", "table", "reason", "evidence", "readers")
DECLARED_ABSENT_TEXT_KEYS = ("db", "table", "reason", "evidence")
# The user-approved allowlist, sealed with this tool: a run's REQUIRED_LAYOUT.json
# may excuse a table only with an entry equal to one of these.
APPROVED_DECLARED_ABSENT_FILE = Path(__file__).with_name("formal_declared_absent.json")


def approved_declared_absent() -> tuple[List[Dict[str, Any]], str]:
    """The approved declared_absent entries and the sha256 of their file."""
    payload, digest = load_json_object_and_sha(APPROVED_DECLARED_ABSENT_FILE,
                                               APPROVED_DECLARED_ABSENT_FILE.name)
    entries = payload.get("entries")
    if payload.get("schema") != "easyicu.declared_absent/1" or not isinstance(entries, list):
        raise ReceiptError("receipt_input_malformed",
                           f"{APPROVED_DECLARED_ABSENT_FILE.name} is not an approved allowlist",
                           input=APPROVED_DECLARED_ABSENT_FILE.name)
    return entries, digest


def _declared_absent_entry_valid(entry: Any) -> bool:
    return (
        isinstance(entry, dict) and set(entry) == set(DECLARED_ABSENT_KEYS)
        and all(isinstance(entry[k], str) and entry[k].strip() for k in DECLARED_ABSENT_TEXT_KEYS)
        and type(entry["readers"]) is int and entry["readers"] == 0
    )
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


class _DuplicateKey(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


def _unique_object(pairs: List[tuple[str, Any]]) -> Dict[str, Any]:
    """``object_pairs_hook``: a key repeated in one object, at any level, is a stop."""
    out: Dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _DuplicateKey(key)
        out[key] = value
    return out


def _parse_json_object(raw: bytes, path: Path, what: str, *, unique_keys: bool) -> Dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"),
                             object_pairs_hook=_unique_object if unique_keys else None)
    except _DuplicateKey as exc:
        raise ReceiptError("receipt_input_duplicate_key",
                           f"{what} repeats the key {exc.key!r} in one object",
                           input=what, path=str(path), key=exc.key) from exc
    except (UnicodeDecodeError, ValueError) as exc:
        raise ReceiptError("receipt_input_malformed", f"{what} is not readable JSON",
                           input=what, path=str(path)) from exc
    if not isinstance(payload, dict):
        raise ReceiptError("receipt_input_malformed", f"{what} is not a JSON object",
                           input=what, path=str(path))
    return payload


def _read_input(path: Path, what: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except FileNotFoundError as exc:
        raise ReceiptError("receipt_input_missing", f"{what} is missing",
                           input=what, path=str(path)) from exc
    except OSError as exc:
        raise ReceiptError("receipt_input_malformed", f"{what} is not readable JSON",
                           input=what, path=str(path)) from exc


def load_json_object(path: Path, what: str) -> Dict[str, Any]:
    """A JSON object input of the receipt; anything else is a typed stop."""
    return _parse_json_object(_read_input(path, what), path, what, unique_keys=False)


def load_json_object_and_sha(path: Path, what: str) -> tuple[Dict[str, Any], str]:
    """A contract input: a JSON object with no key repeated in any object, and
    the sha256 of exactly the bytes parsed."""
    raw = _read_input(path, what)
    return (_parse_json_object(raw, path, what, unique_keys=True),
            hashlib.sha256(raw).hexdigest())


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


# Formal tool execution identity (delta32p ruling: copy-then-exec). The
# wrapper stages every tool of the sealed manifest by content into the run's
# private FORMAL_TOOLS_DIR before anything runs, records their sha256 in
# STAGED_FILE, and executes only those copies, every Python one under the
# staged boot (tools/formal/formal_boot.py). The receipt is itself a staged
# copy. Nothing here looks at processes, links or timestamps.
TOOLS_MANIFEST_SEAL_PATH = "tools/formal/FORMAL_TOOLS.json"
TOOLS_MANIFEST_SCHEMA = "easyicu.formal_tools/1"
FORMAL_TOOLS_DIR = ".formal_tools"
STAGED_FILE = "FORMAL_TOOLS.staged"
IMPORT_ORIGIN_FILE = "IMPORT_ORIGIN.json"
INPUT_STEPS = ("snapshot", "verify")
BOOT_LIVE_MODULE = "formal_boot_live"


def _seal_blob(repo: Path, sealed_sha: str, seal_path: str) -> Optional[bytes]:
    """The file at ``seal_path`` in the sealed commit's tree, or None if it has none."""
    try:
        done = subprocess.run(["git", "-C", str(repo), "cat-file", "blob",
                               f"{sealed_sha}:{seal_path}"], capture_output=True, check=False)
    except OSError as exc:
        raise ReceiptError("formal_tool_script_unverifiable", "git is not runnable",
                           seal_path=seal_path) from exc
    return done.stdout if done.returncode == 0 else None


def _seal_tree_root(code: Mapping[str, Any]) -> Path:
    import_path = code.get("import_path")
    done = subprocess.run(["git", "-C", str(import_path), "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True, check=False) if import_path else None
    if done is None or done.returncode != 0 or not done.stdout.strip():
        raise ReceiptError("formal_tool_script_unverifiable",
                           "the sealed checkout's git tree is not readable",
                           import_path=import_path)
    return Path(done.stdout.strip())


def sealed_tool_manifest(repo: Path, sealed_sha: str) -> List[Dict[str, str]]:
    """The sealed manifest's tools [{name, path, sha256}], each equal to its blob at the seal."""
    raw = _seal_blob(repo, sealed_sha, TOOLS_MANIFEST_SEAL_PATH)
    if raw is None:
        raise ReceiptError("formal_tool_script_unverifiable",
                           f"the seal tree has no {TOOLS_MANIFEST_SEAL_PATH}", sealed_sha=sealed_sha)
    payload = _parse_json_object(raw, Path(TOOLS_MANIFEST_SEAL_PATH), TOOLS_MANIFEST_SEAL_PATH,
                                 unique_keys=True)
    tools = payload.get("tools")
    names = [t.get("name") for t in tools if isinstance(t, dict)] if isinstance(tools, list) else []
    if payload.get("schema") != TOOLS_MANIFEST_SCHEMA or not names or len(names) != len(tools) or (
            len(set(names)) != len(names)) or not all(
            isinstance(t.get(k), str) and t[k] for t in tools for k in ("name", "path", "sha256")) or (
            any("/" in n or n in {".", ".."} for n in names)) or (
            "formal_run_receipt.py" not in names or "formal_boot.py" not in names):
        raise ReceiptError("receipt_input_malformed",
                           f"{TOOLS_MANIFEST_SEAL_PATH} is not a {TOOLS_MANIFEST_SCHEMA} manifest",
                           input=TOOLS_MANIFEST_SEAL_PATH)
    for tool in tools:
        blob = _seal_blob(repo, sealed_sha, tool["path"])
        seal_digest = None if blob is None else hashlib.sha256(blob).hexdigest()
        if seal_digest != tool["sha256"]:
            raise ReceiptError("formal_tool_script_mismatch",
                               f"the sealed manifest's {tool['name']} is not the seal tree's "
                               f"{tool['path']}", role=tool["name"], reason="manifest_not_seal",
                               seal_path=tool["path"], sha256=tool["sha256"],
                               seal_sha256=seal_digest)
    return [{k: t[k] for k in ("name", "path", "sha256")} for t in tools]


def _staged_record(run_root: Path) -> Dict[str, str]:
    path = run_root / STAGED_FILE
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReceiptError("formal_tools_not_staged", f"the run root has no {STAGED_FILE}",
                           path=str(path)) from exc
    staged: Dict[str, str] = {}
    for line in lines:
        digest, _, name = line.partition("  ")
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or not name or name in staged:
            raise ReceiptError("receipt_input_malformed", f"{STAGED_FILE} is malformed",
                               input=STAGED_FILE)
        staged[name] = digest
    return staged


def formal_tool_scripts(sealed_sha: str, code: Mapping[str, Any], run_root: Path,
                        approved_sha256: str) -> List[Dict[str, Any]]:
    """Every formal tool the run executed was a staged copy equal to the seal.

    ``<run_root>/.formal_tools`` must be a real, read-only directory holding
    exactly the sealed manifest's tools (no other entry: no ``.pyc``, no
    ``__pycache__``), each a regular read-only file with one link whose
    sha256 equals the manifest's, the seal tree's blob and the sha the
    wrapper recorded when it staged it (``FORMAL_TOOLS.staged``), so the
    copies are unchanged at the end. This receipt must be running as the
    staged copy, and the allowlist it parsed (``approved_sha256``) must be
    the staged one. Returns one record per tool; any difference is
    ``formal_tool_script_mismatch`` (``reason`` says which).
    """
    repo = _seal_tree_root(code)
    tools = sealed_tool_manifest(repo, sealed_sha)
    root = Path(run_root)
    folder = root / FORMAL_TOOLS_DIR
    try:
        info = os.lstat(folder)
    except OSError as exc:
        raise ReceiptError("formal_tools_not_staged", f"the run root has no {FORMAL_TOOLS_DIR}",
                           path=str(folder)) from exc
    if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o222 or (
            os.path.realpath(folder) != os.path.join(os.path.realpath(root), FORMAL_TOOLS_DIR)):
        raise ReceiptError("formal_tools_dir_invalid",
                           f"{FORMAL_TOOLS_DIR} must be a real read-only directory of the run root",
                           path=str(folder))
    names = {t["name"] for t in tools}
    present = set(os.listdir(folder))
    if present != names:
        raise ReceiptError("formal_tools_dir_unexpected",
                           f"{FORMAL_TOOLS_DIR} does not hold exactly the sealed tools",
                           path=str(folder), unexpected=sorted(present - names),
                           missing=sorted(names - present))
    staged = _staged_record(root)
    records = []
    for tool in tools:
        name, copy = tool["name"], folder / tool["name"]
        entry = os.lstat(copy)
        if not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1 or entry.st_mode & 0o222:
            raise ReceiptError("formal_tool_script_mismatch",
                               f"the staged {name} is not a single-link read-only regular file",
                               role=name, path=str(copy), reason="not_a_staged_copy")
        digest = sha256_file(copy)
        if not digest == tool["sha256"] == staged.get(name):
            raise ReceiptError("formal_tool_script_mismatch",
                               f"the staged {name} is not the sealed {tool['path']} it was "
                               "staged as", role=name, path=str(copy), seal_path=tool["path"],
                               sha256=digest, seal_sha256=tool["sha256"],
                               staged_sha256=staged.get(name),
                               reason="changed_since_staged" if staged.get(name) == tool["sha256"]
                               else "staged_not_seal")
        records.append({"name": name, "seal_path": tool["path"], "path": str(copy),
                        "sha256": digest})
    if set(staged) != names:
        raise ReceiptError("formal_tool_script_mismatch",
                           f"{STAGED_FILE} does not list exactly the sealed tools",
                           reason="staged_set_differs", staged=sorted(staged))
    running = Path(__file__).resolve()
    if running != (folder / "formal_run_receipt.py").resolve():
        raise ReceiptError("formal_tool_script_mismatch",
                           "the receipt is not running as the run's staged copy",
                           role="formal_run_receipt.py", path=str(running), reason="not_staged")
    if APPROVED_DECLARED_ABSENT_FILE.resolve().parent != folder.resolve() or (
            approved_sha256 != staged.get(APPROVED_DECLARED_ABSENT_FILE.name)):
        raise ReceiptError("formal_tool_script_mismatch",
                           "the approved allowlist the receipt read is not the staged copy",
                           role=APPROVED_DECLARED_ABSENT_FILE.name, reason="not_staged")
    return records


def _origin_ok(record: Any, sealed_sha: str, src: str, tools: str, script: str) -> bool:
    return (
        isinstance(record, dict) and record.get("complete") is True
        and record.get("mismatched") == [] and record.get("head") == sealed_sha
        and record.get("src") == src and record.get("tools") == tools
        and isinstance(record.get("path"), str) and record["path"].startswith(src + os.sep)
        and record.get("script") == os.path.join(tools, script)
    )


def formal_import_origins(run_root: Path, sealed_sha: str, code: Mapping[str, Any],
                          databases: Iterable[str]) -> Dict[str, Any]:
    """Every Python step ran under the staged boot with clean import origins.

    The boot's records: each database's extraction (``<db>/IMPORT_ORIGIN.json``,
    script ``bench_local.py``) and both input-snapshot steps
    (``IMPORT_ORIGIN.inputs_<step>.json``, ``inputs_bench_local.py``) must be
    complete, at the sealed HEAD, with easyicu from the clone's src and no
    mismatched module. The receipt's own step is checked live through the
    boot (``formal_boot_live``) before RUN_RECEIPT.json is written; outside
    the boot it is ``formal_boot_missing``.
    """
    root = Path(run_root)
    src = os.path.realpath(Path(code["import_path"]).parent)
    tools = os.path.realpath(root / FORMAL_TOOLS_DIR)
    steps = {f"extract:{db}": (root / db / IMPORT_ORIGIN_FILE, "bench_local.py")
             for db in databases}
    steps.update({f"inputs:{step}": (root / f"IMPORT_ORIGIN.inputs_{step}.json",
                                     "inputs_bench_local.py") for step in INPUT_STEPS})
    out: Dict[str, Any] = {}
    for step, (path, script) in steps.items():
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ReceiptError("formal_import_origin_missing",
                               f"{step} left no readable import-origin record", step=step,
                               path=str(path)) from exc
        if not _origin_ok(record, sealed_sha, src, tools, script):
            raise ReceiptError("formal_import_origin_mismatch",
                               f"{step} did not run as the staged {script} under the boot with "
                               "clean import origins", step=step,
                               mismatched=record.get("mismatched") if isinstance(record, dict)
                               else None)
        out[step] = {"path": str(path), "sha256": sha256_file(path),
                     "modules": len(record.get("modules") or {})}
    live = sys.modules.get(BOOT_LIVE_MODULE)
    if live is None:
        raise ReceiptError("formal_boot_missing",
                           "the receipt is not running under the staged boot")
    mismatched = live.check()
    record = live.record
    if mismatched or not _origin_ok({**record, "complete": True}, sealed_sha, src, tools,
                                    "formal_run_receipt.py"):
        raise ReceiptError("formal_import_origin_mismatch",
                           "the receipt step's import origins are not clean", step="receipt",
                           mismatched=list(mismatched))
    out["receipt"] = {"live": True, "modules": len(record.get("modules") or {})}
    return out



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


def _refuse_unused_installation(db: str, table: Any, realpath: Any,
                                snapshot: Mapping[str, Any]) -> None:
    """delta16v: an installation the snapshot only listed as unused was never
    validated; a run that read it anyway is refused."""
    unused = {u.get("realpath") for u in snapshot.get("unused_installations") or ()}
    if isinstance(realpath, str) and os.path.realpath(realpath) in unused:
        raise ReceiptError("storage_read_unvalidated_installation",
                           f"{db} {table} was read from a bucket installation the snapshot "
                           "listed as unused (never validated)", database=db, table=table)


def _storage_reads(db: str, reads: List[Dict[str, Any]],
                   snapshot: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Each table read with where it was served from; bucket reads with their binding."""
    from easyicu.storage_layout import (
        LAYOUT_CSV, LAYOUT_FLAT, StorageIntegrityError, bucket_source_receipt,
    )

    snapshotted = {b["layout_realpath"] for b in snapshot.get("buckets") or ()}
    receipt = []
    for read in reads:
        _refuse_unused_installation(db, read.get("table"), read.get("layout_realpath"), snapshot)
        entry = dict(read)
        realpath = entry.get("layout_realpath")
        if not entry.get("layout_path") or not realpath:
            raise ReceiptError("storage_read_layout_missing",
                               f"{db} {entry.get('table')} read names no layout directory",
                               database=db, table=entry.get("table"))
        if entry.get("layout") not in (LAYOUT_FLAT, LAYOUT_CSV):
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


def _listed_csv(base: str, table: str, path: str) -> bool:
    """Whether the input's conversion manifest lists this CSV as the table's input
    (the storage owner's rule; a listed CSV read is a read type, not a fallback)."""
    from easyicu.storage_layout import listed_csv_input

    return listed_csv_input(base, table, path)


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
            if item["format"] == "csv" and not _listed_csv(input_dir, item["table"], item["path"]):
                raise ReceiptError("storage_consumed_csv_unlisted",
                                   f"{db} {item['table']} consumed a CSV its conversion manifest "
                                   "does not list as the table's input", database=db,
                                   table=item["table"], path=item["path"])
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
    """``flat``, ``csv`` or ``bucket`` for a storage_reads layout (``bucket_<scheme>``)."""
    return layout if layout in FLAT_KINDS else "bucket"


def _layout_tables(db: str, input_dir: str, binding: Mapping[str, Any],
                   consumed: List[Dict[str, Any]], reads: List[Dict[str, Any]],
                   snapshot: Mapping[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """G2: every consumed or read table has a layout record, and every record is checked.

    Keyed on ``storage_binding["tables"]`` (the storage owner's per-table
    record of what served each consumed table). Each of a table's ``reads`` is
    checked, not only the primary one:

    * a flat or listed-CSV read must be served under this database's input;
      its declared-view chain must be the snapshot's, and present whenever a
      file of the table was admitted through the declared view;
    * a bucket read must be an installation the snapshot covers that the
      snapshot bound to this very table, record its ``derived_from`` for this
      table, and name a manifest and a non-empty source set equal to the
      snapshot's (real path and sha256).

    Every consumed table has a ``storage_reads`` entry, and each entry agrees
    with the record (a flat or CSV entry by the directory of the files it
    consumed). Returns {table: [{layout, root_realpath}, ...]}: every
    distinct read of the table, for the layout requirement.
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
    read_tables = {r["table"] for r in reads}
    for table in sorted(tables):
        if table not in read_tables:
            raise ReceiptError("storage_reads_missing",
                               f"{db} {table} was consumed but the run log has no read of it",
                               database=db, table=table)
    hashed = {f["realpath"]: f for f in snapshot["files"]}
    buckets = {b["layout_realpath"]: b for b in snapshot.get("buckets") or ()}
    input_real = os.path.realpath(input_dir)
    chain = snapshot.get("declared_view_chain")
    through_view = {e["table"] for e in consumed if e.get("admitted_by") == "declared_view"}
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
            if kind in FLAT_KINDS:
                if real != input_real:
                    raise ReceiptError("storage_read_other_root",
                                       f"{db} {table} {kind} read was served under another root",
                                       database=db, table=table)
                named = read.get("derived_from")
                if (named is not None and named != chain) or (
                        named is None and table in through_view):
                    raise ReceiptError("declared_view_chain_mismatch",
                                       f"{db} {table} {kind} read does not name the snapshot's "
                                       "declared-view chain", database=db, table=table)
                continue
            _check_bucket_read(db, table, read, real, buckets, hashed, snapshot)
        primary = record.get("layout")
        if primary not in LAYOUT_KINDS or not any(r["layout"] == primary for r in records):
            raise ReceiptError("storage_layout_unrecorded",
                               f"{db} {table} layout record names no primary layout",
                               database=db, table=table)
        distinct = sorted({(r["layout"], os.path.realpath(r["root"])) for r in records})
        observed[table] = [{"layout": k, "root_realpath": r} for k, r in distinct]
    # The run log's per-read layout (storage_reads) agrees with the record.
    for read in reads:
        kind = _read_kind(read.get("layout"))
        if not any(
            r["layout"] == kind
            and (kind in FLAT_KINDS or os.path.realpath(r["root"]) == read.get("layout_realpath"))
            for r in tables[read["table"]]["reads"]
        ):
            raise ReceiptError("storage_layout_disagrees",
                               f"{db} {read['table']} read as {read.get('layout')} has no "
                               "matching layout record", database=db, table=read["table"])
        if kind in FLAT_KINDS:
            # Exactly the common parent of the files it consumed, as named and
            # by real path (an ancestor such as "/" is not enough). The two may
            # differ legitimately: a link farm names files that live elsewhere.
            fmt = "csv" if kind == "csv" else "parquet"
            files = [e for e in consumed if e["table"] == read["table"] and e["format"] == fmt]
            if not files or (
                read.get("layout_realpath")
                != os.path.commonpath([os.path.dirname(e["realpath"]) for e in files])
                or read.get("layout_path")
                != os.path.commonpath([os.path.dirname(e["path"]) for e in files])
            ):
                raise ReceiptError("storage_layout_disagrees",
                                   f"{db} {read['table']} {kind} read does not name the "
                                   "common directory of the files it consumed", database=db,
                                   table=read["table"])
    return observed


def _check_bucket_read(db: str, table: str, read: Mapping[str, Any], real: str,
                       buckets: Mapping[str, Any], hashed: Mapping[str, Any],
                       snapshot: Mapping[str, Any]) -> None:
    """A bucket read: the snapshot's installation of THIS table, its manifest and sources."""
    _refuse_unused_installation(db, table, real, snapshot)
    bucket = buckets.get(real)
    if bucket is None:
        raise ReceiptError("storage_read_outside_snapshot",
                           f"{db} {table} bucket is not one the input snapshot covers",
                           database=db, table=table)
    derived, manifest = read.get("derived_from"), read.get("manifest")
    if read.get("derived_from_recorded") is not True or not isinstance(derived, dict):
        raise ReceiptError("bucket_derived_from_unrecorded",
                           f"{db} {table} bucket read records no derived_from",
                           database=db, table=table)
    # B-TABLE-SOURCE: the requested table, the snapshot's binding of the
    # installation, and the manifest's source table are one and the same.
    names = {str(table).lower(), str(bucket.get("table")).lower(),
             str(bucket.get("derived_table", bucket.get("table"))).lower(),
             str(derived.get("table")).lower()}
    if len(names) != 1:
        raise ReceiptError("bucket_derived_from_other_table",
                           f"{db} {table} bucket read is bound to another table's sources",
                           database=db, table=table)
    from easyicu.storage_layout import MANIFEST

    named = os.path.realpath(manifest["path"]) if isinstance(
        manifest, dict) and isinstance(manifest.get("path"), str) else None
    held = hashed.get(named) if named == os.path.join(real, MANIFEST) else None
    if held is None or held["sha256"] != manifest.get("sha256"):
        raise ReceiptError("bucket_manifest_not_snapshotted",
                           f"{db} {table} bucket manifest the run read is not the "
                           "snapshotted one", database=db, table=table)
    files = derived.get("files")
    if not isinstance(files, list) or not files:
        raise ReceiptError("bucket_derived_from_empty",
                           f"{db} {table} bucket read lists no sources", database=db, table=table)
    for source in files:
        held = hashed.get(source.get("realpath"))
        if held is None:
            raise ReceiptError("bucket_derived_from_outside_snapshot",
                               f"{db} {table} bucket source is not in the snapshot",
                               database=db, table=table)
        if held["sha256"] != source.get("sha256"):
            raise ReceiptError("bucket_derived_from_sha_mismatch",
                               f"{db} {table} bucket source differs from the snapshot",
                               database=db, table=table)
    run = sorted((s.get("realpath"), s.get("sha256")) for s in files)
    snapped = sorted((s["realpath"], s["builder_sha256"]) for s in bucket["derived_from"])
    if run != snapped:
        raise ReceiptError("bucket_derived_from_not_snapshot",
                           f"{db} {table} bucket read names other sources than the snapshot "
                           "bound for the installation", database=db, table=table)


def _storage_unavailable(db: str, manifest: Mapping[str, Any],
                         declared_absent: Iterable[Mapping[str, str]] = ()) -> List[Dict[str, Any]]:
    """Declared tables the run read as empty because they have no physical source.

    The storage owner records each one (``table_unavailable``, delta17w) and
    the native manifest publishes them as ``storage_unavailable``. The list is
    always copied into the receipt, each record with the ``declared_absent``
    entry of the run's REQUIRED_LAYOUT.json that excuses it. In a formal run a
    record no entry excuses is a stop (``storage_unavailable_in_formal_run``,
    naming the tables and keeping each record's cause), and a manifest that
    does not record the list at all cannot show there were none
    (``storage_unavailable_missing``).
    """
    if "storage_unavailable" not in manifest:
        raise ReceiptError("storage_unavailable_missing",
                           f"{db} native manifest does not record which declared tables had "
                           "no physical source", database=db)
    records = manifest["storage_unavailable"]
    if not isinstance(records, list) or not all(
            isinstance(r, dict) and isinstance(r.get("table"), str) for r in records):
        raise ReceiptError("receipt_input_malformed",
                           f"{db} native manifest storage_unavailable is not a list of records",
                           database=db)
    allowed = {str(e["table"]).casefold(): dict(e) for e in declared_absent if e.get("db") == db}
    excused, refused = [], []
    for record in records:
        entry = allowed.get(record["table"].casefold())
        (refused if entry is None else excused).append({**record, "excused_by": entry})
    if refused:
        tables = sorted({r["table"] for r in refused})
        raise ReceiptError(
            "storage_unavailable_in_formal_run",
            f"{db} read {len(tables)} declared table(s) with no physical source as empty and "
            "not declared absent for this root: " + ", ".join(tables),
            database=db, tables=tables,
            causes=[{k: r.get(k) for k in ("table", "reason_code", "availability", "manifest",
                                            "module")} for r in refused],
        )
    return excused


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


def load_required_layout(run_root: Path, databases: Iterable[str],
                         approved: Optional[tuple[List[Dict[str, Any]], str]] = None,
                         ) -> tuple[Dict[str, Any], str]:
    """G4: the run's REQUIRED_LAYOUT.json and its sha256.

    {"schema": "easyicu.required_layout/2", "complete": true, "databases":
    {db: {table: {"layout": "flat" | "csv" | "bucket", "root_realpath": "<abs>"}}}}.
    ``root_realpath`` is required for a bucket (the installation it must be
    served from) and optional for a flat table (default: the database's
    input). ``complete`` states the list names every table the run may read.
    Optional ``declared_absent``: the root's allowlist of declared tables a
    formal run may read as empty, [{"db", "table", "reason", "evidence",
    "readers": 0}] (non-empty strings, each (db, table) once); absent means
    empty. Each entry must equal one of the user-approved entries sealed with
    this tool (``formal_declared_absent.json``), else
    ``declared_absent_not_approved``. Only these may appear in a formal run's
    ``storage_unavailable``. Anything else,
    schema /1 included, or a database the run extracts without an entry, is
    a stop.
    """
    path = Path(run_root) / REQUIRED_LAYOUT_FILE
    if not path.is_file():
        raise ReceiptError("required_layout_missing",
                           f"the run root has no {REQUIRED_LAYOUT_FILE}", path=str(path))
    payload, digest = load_json_object_and_sha(path, REQUIRED_LAYOUT_FILE)
    listed = payload.get("databases")
    if payload.get("schema") != REQUIRED_LAYOUT_SCHEMA or not isinstance(listed, dict) or (
            set(payload) - {"schema", "complete", "databases", "declared_absent"}) or (
            len({str(db).casefold() for db in listed}) != len(listed)):
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
        names = [str(t).casefold() for t in tables]
        if len(set(names)) != len(names) or not all(
                _required_entry_valid(v) for v in tables.values()):
            raise ReceiptError("receipt_input_malformed",
                               f"{REQUIRED_LAYOUT_FILE} {db}: each table once, as "
                               "{layout: flat|csv|bucket, root_realpath} (required for a bucket)",
                               input=REQUIRED_LAYOUT_FILE, database=db)
    absent = payload.get("declared_absent", [])
    keys = [(str(e.get("db")).casefold(), str(e.get("table")).casefold()) for e in absent
            if isinstance(e, dict)] if isinstance(absent, list) else []
    if not isinstance(absent, list) or len(keys) != len(absent) or len(set(keys)) != len(keys) or (
        not all(_declared_absent_entry_valid(e) for e in absent)
    ):
        raise ReceiptError("receipt_input_malformed",
                           f"{REQUIRED_LAYOUT_FILE} declared_absent must list each (db, table) "
                           "once as {db, table, reason, evidence, readers: 0}",
                           input=REQUIRED_LAYOUT_FILE)
    allowlist, _sha = approved if approved is not None else approved_declared_absent()
    unapproved = [e for e in absent if e not in allowlist]
    if unapproved:
        raise ReceiptError("declared_absent_not_approved",
                           f"{REQUIRED_LAYOUT_FILE} declared_absent names "
                           f"{len(unapproved)} entr(y/ies) the approved allowlist does not hold",
                           input=REQUIRED_LAYOUT_FILE,
                           entries=[{"db": e["db"], "table": e["table"]} for e in unapproved])
    return payload, digest


def _required_entry_valid(entry: Any) -> bool:
    if not isinstance(entry, dict) or set(entry) - {"layout", "root_realpath"}:
        return False
    root = entry.get("root_realpath")
    if entry.get("layout") == "bucket":
        return isinstance(root, str) and os.path.isabs(root)
    return entry.get("layout") in FLAT_KINDS and (
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
        wanted[str(table).casefold()] = {"layout": entry["layout"],
                                      "root_realpath": os.path.realpath(root)}
    deviations = []
    for table, reads in sorted(observed.items()):
        need = wanted.get(table.casefold())
        for read in reads:
            if need != read:
                deviations.append({"table": table, "required": need, "observed": read})
    if deviations:
        raise ReceiptError("layout_deviation",
                           f"{db}: {len(deviations)} read(s) not served from the required "
                           "layout and root", database=db, deviations=deviations)
    read_tables = {t.casefold() for t in observed}
    return {
        "required": {t: wanted[str(t).casefold()] for t in sorted(required)},
        "observed": dict(sorted(observed.items())),
        "not_read": sorted(t for t in required if str(t).casefold() not in read_tables),
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


def _checked_snapshot(db: str, input_dir: str, name: str, snap: Mapping[str, Any]) -> set:
    """One database snapshot, recomputed from its own file list (B-SNAPSHOT).

    The self-reported digest, file count and bytes must be what the files
    say, the snapshot must be of this database and input, and it must list
    files. Returns the (real path, size, sha256) set.
    """
    from easyicu.formal_inputs import snapshot_digest

    input_real = os.path.realpath(input_dir)
    if snap.get("database", db) != db or os.path.realpath(str(snap.get("data_path", ""))) != (
            input_real) or snap.get("data_path_realpath") != input_real:
        raise ReceiptError("input_snapshot_other_input",
                           f"{db} {name}-run snapshot is of another input", database=db)
    files = snap.get("files")
    if not isinstance(files, list) or not files or not all(
        isinstance(f, dict) and isinstance(f.get("realpath"), str)
        and isinstance(f.get("size"), int) and isinstance(f.get("sha256"), str)
        for f in files
    ):
        raise ReceiptError("input_snapshot_inconsistent",
                           f"{db} {name}-run snapshot lists no well-formed files", database=db)
    triples = {(f["realpath"], f["size"], f["sha256"]) for f in files}
    if (
        len(triples) != len(files)
        or snapshot_digest(files) != snap.get("digest")
        or snap.get("file_count") != len(files)
        or snap.get("bytes") != sum(f["size"] for f in files)
    ):
        raise ReceiptError("input_snapshot_inconsistent",
                           f"{db} {name}-run snapshot's digest, count or bytes are not what "
                           "its files say", database=db)
    return triples


def _snapshot_section(db: str, input_dir: str, pre: Mapping[str, Any],
                      post: Mapping[str, Any]) -> Dict[str, Any]:
    before, after = pre["databases"].get(db), post["databases"].get(db)
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ReceiptError("input_snapshot_missing",
                           f"{db} has no pre- and post-run input snapshot", database=db)
    sets = {}
    for name, snap in (("pre", before), ("post", after)):
        sets[name] = _checked_snapshot(db, input_dir, name, snap)
    if sets["pre"] != sets["post"] or any(
            before.get(k) != after.get(k)
            for k in ("declared_view_chain", "buckets", "unused_installations")):
        changed = len(sets["pre"] ^ sets["post"])
        raise ReceiptError("input_changed_during_run",
                           f"{db} inputs differ before and after the run", database=db,
                           changed_count=changed)
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
                      required: Optional[Mapping[str, str]] = None,
                      declared_absent: Iterable[Mapping[str, str]] = ()) -> Dict[str, Any]:
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
    section["storage_unavailable"] = _storage_unavailable(db, manifest, declared_absent)
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
    approved = approved_declared_absent()
    approved_sha = approved[1]
    tool_scripts = formal_tool_scripts(sealed_sha, code, run_root, approved_sha)
    pre, post, snapshot_files = _snapshots(run_root)
    required, required_sha = load_required_layout(
        run_root, [spec.split(":", 2)[0] for spec in specs], approved)
    exits = _status_exits(run_root)
    databases: Dict[str, Any] = {}
    for spec in specs:
        db, input_dir, cores = spec.split(":", 2)
        try:
            section = _database_section(run_root, db, input_dir, cores, exits.get(db),
                                        sealed_sha, code.get("version"), (pre, post),
                                        required["databases"][db],
                                        required.get("declared_absent", []))
            if db in SIC_DATABASES:
                section["cases_binding"] = sic_cases_binding(Path(input_dir), Path(scratch), code)
        except (KeyError, TypeError, AttributeError) as exc:
            # A field of an input has the wrong shape: a typed stop, never a pass.
            raise ReceiptError("receipt_input_malformed", f"{db}: {type(exc).__name__}: {exc}",
                               database=db) from exc
        databases[db] = section
    # Last: every Python step ran as a staged copy under the boot (this one checked live).
    import_origin = formal_import_origins(run_root, sealed_sha, code, list(databases))
    finished = time.time()
    peaks = [s["cgroup_peak_gib"] for s in databases.values() if s.get("cgroup_peak_gib") is not None]
    return {
        "schema_version": RECEIPT_SCHEMA,
        "sealed_sha": sealed_sha,
        "easyicu_version": code.get("version"),
        "code_identity": code,
        "tool_scripts": tool_scripts,
        "import_origin": import_origin,
        "input_snapshot": snapshot_files,
        "required_layout": {"path": str((run_root / REQUIRED_LAYOUT_FILE).resolve()),
                            "sha256": required_sha, "complete": True,
                            "declared_absent": list(required.get("declared_absent", [])),
                            "approved_declared_absent_sha256": approved_sha},
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
