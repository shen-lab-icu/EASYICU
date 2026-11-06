"""Formal runs: the inputs did not change, by hashing before and after the run.

A formal run snapshots every input file of each database before any job
starts (sha256 plus size, mtime_ns and inode), and hashes the same set again
after every job has exited and before the run receipt is written. Any
difference makes the run invalid. Runtime source binding
(``easyicu.storage_binding``) stays as defence in depth; it is best-effort,
this is the guarantee.

The set for one database input (its ``data_path``):

* ``conversion_manifest.json`` and, relative to the data path, every output
  (flat shards) and every existing input (the CSVs) it lists. A formal input
  without a conversion manifest is refused;
* for each bucket installation the storage owner chooses, its manifest, build
  receipt and every part file; the builder's ``derived_from`` sha256 of each
  source must equal that source's hash in the snapshot;
* for a declared view (a symlink farm whose manifest names the manifest it was
  derived from, ``storage_layout.declared_view_chain``): the source manifest
  and every output it lists; a declared chain the storage owner does not
  accept is ``declared_view_chain_broken``.

Files are identified by real path (and recorded with their device and inode,
so a consumer can match a file named with other letter case on a
case-insensitive file system); a file reached through several links is hashed
once. Equality is by (real path, size, sha256): a file modified and
restored to the same bytes during the run is not detected (an accepted,
documented limitation). The snapshot is written outside the input.

Inputs must not change while a run reads them; runtime source binding catches
many changes but is best-effort. The formal guarantee comes only from bench7v2 /
``extract_database`` with ``RUN_RECEIPT``: there the snapshot is combined with
the run's consumed files, every one of which must be in it.

``easyicu extract --input-snapshot`` is a narrower opt-in for end users: it
checks that the listed input files are unchanged before and after the run. It
does NOT publish a consumed set or a RUN_RECEIPT, and it does NOT prove that
every file the run read was in the snapshot.

This module's command line (bench7v2 calls it for every formal run)::

    python -m easyicu.formal_inputs snapshot --out PRE.json DB:DATA_PATH ...
    python -m easyicu.formal_inputs verify --pre PRE.json --out POST.json
    python -m easyicu.formal_inputs plan DB:DATA_PATH ...     # sizes, no hashing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional

SNAPSHOT_SCHEMA = "easyicu.input_snapshot/1"
CONVERSION_MANIFEST = "conversion_manifest.json"
BUCKET_SUFFIX = "_bucket"


class InputSnapshotError(Exception):
    """The input set cannot be snapshotted, or changed; ``code`` names why."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.details = details


def _load_manifest(path: Path) -> Dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputSnapshotError("input_manifest_missing", f"{path.name} is missing",
                                 path=str(path)) from exc
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise InputSnapshotError("input_manifest_malformed", f"{path.name} is not JSON",
                                 path=str(path)) from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tables"), list):
        raise InputSnapshotError("input_manifest_malformed", f"{path.name} lists no tables",
                                 path=str(path))
    return manifest


def _relative(base: Path, relative: Any, manifest: Path) -> Path:
    if not isinstance(relative, str):
        raise InputSnapshotError("input_manifest_malformed", "a listed path is not a string",
                                 path=str(manifest))
    parts = PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or ".." in parts.parts:
        raise InputSnapshotError("input_manifest_malformed", "a listed path leaves its root",
                                 path=str(manifest), relative=relative)
    return base.joinpath(*parts.parts)


def _manifest_files(base: Path, manifest_path: Path, role: str) -> List[Dict[str, str]]:
    """The manifest itself, every output it lists, and every listed input that exists."""
    manifest = _load_manifest(manifest_path)
    files = [{"role": f"{role}_manifest", "path": str(manifest_path)}]
    missing_outputs = []
    for entry in manifest["tables"]:
        if not isinstance(entry, dict):
            raise InputSnapshotError("input_manifest_malformed", "a table entry is not an object",
                                     path=str(manifest_path))
        for output in entry.get("outputs") or ():
            if not isinstance(output, dict):
                raise InputSnapshotError("input_manifest_malformed", "an output is not an object",
                                         path=str(manifest_path))
            path = _relative(base, output.get("relative_path"), manifest_path)
            if not path.is_file():
                missing_outputs.append(output.get("relative_path"))
                continue
            files.append({"role": f"{role}_output", "path": str(path)})
        source = entry.get("input")
        if isinstance(source, dict) and source.get("relative_path") is not None:
            path = _relative(base, source.get("relative_path"), manifest_path)
            if path.is_file():
                files.append({"role": f"{role}_input", "path": str(path)})
    if missing_outputs:
        raise InputSnapshotError("input_output_missing",
                                 f"{len(missing_outputs)} listed outputs do not exist",
                                 path=str(manifest_path), outputs=sorted(missing_outputs)[:20])
    return files


def _declared_view_chain(base: Path) -> Optional[Dict[str, str]]:
    """The declared-view chain of ``base`` (single call site of the storage owner's rule).

    A manifest that declares ``derived_from`` but whose chain the owner does
    not accept (the source manifest's sha256 no longer matches) is refused.
    """
    from .storage_layout import declared_view_chain

    chain = declared_view_chain(base)
    if chain is None and _load_manifest(base / CONVERSION_MANIFEST).get("derived_from"):
        raise InputSnapshotError(
            "declared_view_chain_broken",
            "the input declares a derived_from chain the storage owner does not accept",
            base=str(base),
        )
    return chain


def _bucket_tables(source, base: Path) -> List[str]:
    """Every table name the storage owner could resolve a bucket for (B-NESTED).

    The configured table set, plus every ``<name>_bucket`` directory under the
    roots the resolver searches (``base``, ``base/icu``, ``base/hosp``), so an
    installation on disk is never missed by the snapshot.
    """
    names = {str(t) for t in source.config.tables}
    for root in (base, base / "icu", base / "hosp"):
        if root.is_dir():
            names.update(c.name[: -len(BUCKET_SUFFIX)] for c in root.glob(f"*{BUCKET_SUFFIX}"))
    return sorted(names)


def _bucket_owner_outputs(table: str, source_root: Path) -> set:
    """Real paths of the outputs the source installation's conversion manifest
    lists for ``table`` (B-TABLE-SOURCE: the owner of a bucket's sources)."""
    manifest_path = source_root / CONVERSION_MANIFEST
    if not manifest_path.is_file():
        raise InputSnapshotError("bucket_source_unmanifested",
                                 f"{table} bucket sources have no conversion manifest",
                                 table=table, source_root=str(source_root))
    manifest = _load_manifest(manifest_path)
    outputs = set()
    for entry in manifest["tables"]:
        if isinstance(entry, dict) and str(entry.get("table")).lower() == table.lower():
            for output in entry.get("outputs") or ():
                path = _relative(source_root, (output or {}).get("relative_path"), manifest_path)
                outputs.add(os.path.realpath(path))
    return outputs


def _bucket_files(database: str, base: Path) -> tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Every file of each bucket installation the storage owner chooses, and its sources.

    Installations are enumerated through the reader's own resolver (root,
    ``icu/`` and ``hosp/``) over every table it could serve, once per real
    path. Each must be a bucket OF that table: its manifest's
    ``derived_from.table`` is the table, and its sources are exactly the
    outputs the source installation's conversion manifest lists for it
    (non-empty, complete, nothing else).
    """
    from .datasource import storage_source
    from .storage_layout import StorageIntegrityError, bucket_source_receipt

    source = storage_source(database, str(base))
    files: List[Dict[str, str]] = []
    buckets: List[Dict[str, Any]] = []
    seen: Dict[str, str] = {}
    for table in _bucket_tables(source, base):
        try:
            chosen = source.resolve_bucket_directory(table)
        except StorageIntegrityError as exc:
            raise InputSnapshotError("input_bucket_invalid", f"{table}: {exc}",
                                     table=table, reason=getattr(exc, "reason", None)) from exc
        if chosen is None:
            continue
        chosen = Path(chosen)
        real = Path(os.path.realpath(chosen))
        if str(real) in seen:
            if seen[str(real)].lower() != table.lower():
                raise InputSnapshotError(
                    "bucket_derived_from_other_table",
                    f"one bucket installation is resolved for {seen[str(real)]} and {table}",
                    table=table)
            continue
        seen[str(real)] = table
        try:
            receipt = bucket_source_receipt(chosen)
        except StorageIntegrityError as exc:
            raise InputSnapshotError("input_bucket_invalid", f"{table}: {exc}",
                                     table=table, reason=getattr(exc, "reason", None)) from exc
        derived = receipt["derived_from"]
        if str(derived.get("table")).lower() != table.lower():
            raise InputSnapshotError(
                "bucket_derived_from_other_table",
                f"{table} bucket was built from another table's sources",
                table=table, derived_table=derived.get("table"))
        sources = {f["realpath"] for f in derived["files"]}
        owned = _bucket_owner_outputs(table, Path(derived["source_root"]))
        if not sources or len(sources) != len(derived["files"]) or sources != owned:
            raise InputSnapshotError(
                "bucket_derived_from_not_owner_outputs",
                f"{table} bucket sources are not exactly the conversion manifest's outputs "
                "for the table", table=table, sources=len(sources), owned=len(owned))
        for directory, _dirs, names in os.walk(real):
            for name in names:
                files.append({"role": "bucket_file", "path": str(Path(directory) / name)})
        buckets.append({
            "table": table,
            "derived_table": derived["table"],
            "layout_path": str(chosen),
            "layout_realpath": str(real),
            "source_manifest": str(Path(derived["source_root"]) / CONVERSION_MANIFEST),
            "derived_from": [
                {"path": f["path"], "realpath": f["realpath"], "builder_sha256": f["sha256"]}
                for f in derived["files"]
            ],
        })
        files.append({"role": "bucket_source_manifest",
                      "path": str(Path(derived["source_root"]) / CONVERSION_MANIFEST)})
    return files, buckets


def plan_input_files(database: str, data_path: str | Path) -> Dict[str, Any]:
    """Which files a formal run of ``database`` on ``data_path`` must snapshot."""
    base = Path(data_path)
    files = _manifest_files(base, base / CONVERSION_MANIFEST, "conversion")
    chain = _declared_view_chain(base)
    if chain is not None:
        source_manifest = Path(chain["source_manifest_path"])
        files += _manifest_files(source_manifest.parent, source_manifest, "view_source")
    bucket_files, buckets = _bucket_files(database, base)
    files += bucket_files
    unique: Dict[str, Dict[str, Any]] = {}
    for item in files:
        real = os.path.realpath(item["path"])
        entry = unique.setdefault(real, {"realpath": real, "path": item["path"], "roles": []})
        if item["role"] not in entry["roles"]:
            entry["roles"].append(item["role"])
    return {
        "database": database,
        "data_path": str(base),
        "data_path_realpath": os.path.realpath(base),
        "declared_view_chain": chain,
        "buckets": buckets,
        "files": [unique[key] for key in sorted(unique)],
    }


def _hash(path: str) -> Dict[str, Any]:
    before = os.stat(path)
    with open(path, "rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    after = os.stat(path)
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino
    ):
        raise InputSnapshotError("input_changed_while_hashed", "an input changed while hashed",
                                 realpath=path)
    return {"size": after.st_size, "mtime_ns": after.st_mtime_ns, "ino": after.st_ino,
            "dev": after.st_dev, "sha256": digest}


def snapshot_digest(files: Iterable[Mapping[str, Any]]) -> str:
    """Digest of the set by (real path, size, sha256); mtime and inode are recorded only."""
    payload = sorted([f["realpath"], f["size"], f["sha256"]] for f in files)
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _snapshot_one(database: str, data_path: str | Path) -> Dict[str, Any]:
    plan = plan_input_files(database, data_path)
    started = time.monotonic()
    files = []
    for entry in plan["files"]:
        try:
            files.append({**entry, **_hash(entry["realpath"])})
        except FileNotFoundError as exc:
            raise InputSnapshotError("input_missing", "a planned input vanished",
                                     realpath=entry["realpath"]) from exc
    by_real = {f["realpath"]: f for f in files}
    for bucket in plan["buckets"]:
        for source in bucket["derived_from"]:
            held = by_real.get(source["realpath"])
            if held is None:
                raise InputSnapshotError(
                    "bucket_derived_from_outside_snapshot",
                    f"{bucket['table']} bucket was built from a file outside the input set",
                    table=bucket["table"], source=source["path"])
            if held["sha256"] != source["builder_sha256"]:
                raise InputSnapshotError(
                    "bucket_derived_from_sha_mismatch",
                    f"{bucket['table']} bucket source differs from what the builder bound",
                    table=bucket["table"], source=source["path"])
            source["sha256_matches_snapshot"] = True
    return {
        **{k: plan[k] for k in ("database", "data_path", "data_path_realpath",
                                "declared_view_chain", "buckets")},
        "files": files,
        "file_count": len(files),
        "bytes": sum(f["size"] for f in files),
        "hash_seconds": round(time.monotonic() - started, 3),
        "digest": snapshot_digest(files),
    }


def _spec(spec: str) -> tuple[str, str]:
    """``DB:DATA_PATH`` or bench's ``DB:DATA_PATH:CORES``."""
    database, _, data_path = spec.partition(":")
    head, _, tail = data_path.rpartition(":")
    if head and re.fullmatch(r"\d+(-\d+)?", tail):
        data_path = head
    if not database or not data_path:
        raise InputSnapshotError("input_spec_invalid", f"{spec!r} is not DB:DATA_PATH[:CORES]")
    return database, data_path


def take_snapshot(specs: Iterable[str]) -> Dict[str, Any]:
    """Snapshot every database input of a run (``DB:DATA_PATH[:CORES]`` specs)."""
    databases = {}
    for spec in specs:
        database, data_path = _spec(spec)
        databases[database] = _snapshot_one(database, data_path)
    return {"schema": SNAPSHOT_SCHEMA, "databases": databases}


def verify_snapshot(pre: Mapping[str, Any]) -> Dict[str, Any]:
    """Hash the same inputs again; any difference is ``input_changed_during_run``."""
    if pre.get("schema") != SNAPSHOT_SCHEMA or not isinstance(pre.get("databases"), dict):
        raise InputSnapshotError("input_snapshot_malformed", "not an input snapshot")
    post = {"schema": SNAPSHOT_SCHEMA, "databases": {}}
    for database, before in pre["databases"].items():
        after = _snapshot_one(database, before["data_path"])
        post["databases"][database] = after
        old = {f["realpath"]: (f["size"], f["sha256"]) for f in before["files"]}
        new = {f["realpath"]: (f["size"], f["sha256"]) for f in after["files"]}
        changed = sorted(p for p in old.keys() | new.keys() if old.get(p) != new.get(p))
        if changed or after["digest"] != before["digest"]:
            raise InputSnapshotError(
                "input_changed_during_run",
                f"{database}: {len(changed)} input files differ from the pre-run snapshot",
                database=database, changed=changed[:20], changed_count=len(changed),
            )
    return post


def write_snapshot(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a snapshot once, atomically, and never inside an input it covers."""
    path = Path(path)
    _refuse_inside_inputs(path, payload)
    if path.exists():
        raise InputSnapshotError("input_snapshot_exists", f"{path.name} is written once")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _refuse_inside_inputs(out: Path, snapshot: Mapping[str, Any]) -> None:
    target = Path(os.path.realpath(out.parent))
    for entry in snapshot["databases"].values():
        roots = {entry["data_path_realpath"]} | {
            os.path.dirname(f["realpath"]) for f in entry["files"]
        }
        for root in roots:
            if target == Path(root) or Path(root) in target.parents:
                raise InputSnapshotError("input_snapshot_inside_input",
                                         "the snapshot would be written inside an input",
                                         out=str(out), input=root)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m easyicu.formal_inputs",
                                     description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snapshot")
    snap.add_argument("--out", required=True, type=Path)
    snap.add_argument("specs", nargs="+")
    check = commands.add_parser("verify")
    check.add_argument("--pre", required=True, type=Path)
    check.add_argument("--out", required=True, type=Path)
    plan = commands.add_parser("plan")
    plan.add_argument("specs", nargs="+")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            for spec in args.specs:
                database, data_path = _spec(spec)
                planned = plan_input_files(database, data_path)
                size = sum(os.stat(f["realpath"]).st_size for f in planned["files"])
                print(json.dumps({"database": database, "files": len(planned["files"]),
                                  "bytes": size, "buckets": len(planned["buckets"]),
                                  "declared_view": planned["declared_view_chain"] is not None}))
            return 0
        if args.command == "snapshot":
            result = take_snapshot(args.specs)
            write_snapshot(args.out, result)
        else:
            pre = json.loads(args.pre.read_text(encoding="utf-8"))
            result = verify_snapshot(pre)
            write_snapshot(args.out, result)
    except InputSnapshotError as error:
        sys.stderr.write(json.dumps({"code": error.code, "message": str(error),
                                     "details": error.details}, default=str) + "\n")
        return 3
    print(json.dumps({db: {"files": e["file_count"], "bytes": e["bytes"],
                           "hash_seconds": e["hash_seconds"], "digest": e["digest"]}
                      for db, e in result["databases"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
