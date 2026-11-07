#!/usr/bin/env python3
"""Footer-validity scan for one data root (lead-r13 seal tool; metadata only, never reads data pages).

For every *.parquet under ROOT (symlinked flat files are followed; the files are opened read-only):
  * the file must start and end with the parquet magic, and its footer must parse (pyarrow.read_metadata);
  * row count = footer num_rows.
For every <table>_bucket directory with a _BUCKET_MANIFEST.json:
  * the parts on disk must equal the manifest inventory (relative_path set);
  * the footer row count of each part must equal its inventory row_count;
  * the sum must equal the manifest inventory total and, when present, the build receipt's `rows`;
  * the _COMPLETE marker must be present.
Output: JSON with counts and paths relative to ROOT only (no values, no IDs, no dates).
Exit 0 = all ok, 1 = at least one failure (listed under `failures` with a typed reason), 2 = usage error.

Containment (the seal tool's contract, shared): every directory and file reached -- following links --
must resolve inside ROOT or an --allowed-root; anything else is a `path_escapes_allowed_roots` failure
and is not read (bucket metadata included). Loop protection is the set of real directories on the
current walk path, so every logical path of an alias is scanned (each real file's footer is read once).

  python footer_scan.py ROOT --out scan.json [--buckets-only] [--allowed-root DIR ...]
"""

from __future__ import annotations

# Import contract: this tool never imports easyicu (pinned by tests/core/test_seal_bucket_round2.py), so
# seal_rebuild.py may run it with `python -I`; it needs no seal-clone import origin.
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

MAGIC = b"PAR1"
TOOL_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _footer(path: Path) -> tuple[int | None, str | None]:
    try:
        size = path.stat().st_size
        if size < 12:
            return None, "parquet_too_small"
        with open(path, "rb") as fh:
            head = fh.read(4)
            fh.seek(size - 4)
            tail = fh.read(4)
        if head != MAGIC or tail != MAGIC:
            return None, "parquet_magic_missing"
        return pq.read_metadata(path).num_rows, None
    except Exception as exc:  # noqa: BLE001 - every unreadable footer is a typed failure
        return None, f"parquet_footer_unreadable:{type(exc).__name__}"


def _inside(path, roots) -> bool:
    real = os.path.realpath(path)
    for root in roots:
        r = os.path.realpath(root)
        if real == r or real.startswith(r.rstrip("/") + "/"):
            return True
    return False


def _parquet_files(root: Path, buckets_only: bool, allowed, failures: list):
    """Every logical *.parquet path under ROOT, links followed. Loop protection is the set of real
    directories on the CURRENT walk path (a link back to an ancestor ends there), so a legitimate alias
    -- a second link to the same directory -- is walked under its own logical path too (D29-FOOTER-ALIAS)."""
    root = Path(root)
    stack = [(root, frozenset())]
    while stack:
        directory, ancestors = stack.pop()
        rel_dir = directory.relative_to(root)
        if not _inside(directory, allowed):
            failures.append(
                {"path": rel_dir.as_posix(), "reason": "path_escapes_allowed_roots"}
            )
            continue
        real = os.path.realpath(directory)
        if real in ancestors:  # a link loop: this directory is already on the walk path
            continue
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError as exc:
            failures.append(
                {
                    "path": rel_dir.as_posix(),
                    "reason": f"directory_unreadable:{type(exc).__name__}",
                }
            )
            continue
        subdirs = [
            e.name
            for e in entries
            if not e.name.startswith(".") and e.is_dir(follow_symlinks=True)
        ]
        skip_files = buckets_only and (
            not rel_dir.parts or not rel_dir.parts[0].endswith("_bucket")
        )
        for entry in [] if skip_files else entries:
            name = entry.name
            if name.endswith(".parquet") and not name.startswith("."):
                path = directory / name
                if not _inside(path, allowed):
                    failures.append(
                        {
                            "path": path.relative_to(root).as_posix(),
                            "reason": "path_escapes_allowed_roots",
                        }
                    )
                    continue
                if path.is_dir():
                    continue
                yield path
        here = ancestors | {real}
        for name in reversed(subdirs):
            stack.append((directory / name, here))


def scan(root: Path, buckets_only: bool = False, allowed_roots=()) -> dict:
    failures: list[dict] = []
    files = 0
    rows_total = 0
    footer_rows: dict[str, int] = {}
    allowed = [root, *allowed_roots]
    footers: dict[str, tuple[int | None, str | None]] = {}  # read once per real file
    for path in _parquet_files(root, buckets_only, allowed, failures):
        files += 1
        rel = path.relative_to(root).as_posix()
        real = os.path.realpath(path)
        if real not in footers:
            footers[real] = _footer(path)
        n, err = footers[real]
        if err:
            failures.append({"path": rel, "reason": err})
            continue
        footer_rows[rel] = n
        rows_total += n
    buckets = {}
    for bucket in sorted(
        p for p in root.iterdir() if p.is_dir() and p.name.endswith("_bucket")
    ):
        mpath = bucket / "_BUCKET_MANIFEST.json"
        receipt = bucket / "_BUCKET_BUILD_RECEIPT.json"
        escaped = [
            m
            for m in (bucket, mpath, receipt)
            if m.exists() and not _inside(m, allowed)
        ]
        if (
            escaped
        ):  # metadata is read only from inside the allowed roots, like the parts
            failures.append(
                {
                    "path": escaped[0].relative_to(root).as_posix(),
                    "reason": "path_escapes_allowed_roots",
                }
            )
            continue
        if not mpath.is_file():
            failures.append({"path": bucket.name, "reason": "bucket_manifest_missing"})
            continue
        manifest = json.loads(mpath.read_text())
        inventory = {
            e["relative_path"]: int(e["row_count"])
            for e in manifest.get("inventory") or []
        }
        on_disk = {
            k[len(bucket.name) + 1 :]: v
            for k, v in footer_rows.items()
            if k.startswith(bucket.name + "/")
        }
        unreadable = [f for f in failures if f["path"].startswith(bucket.name + "/")]
        missing = sorted(
            set(inventory)
            - set(on_disk)
            - {f["path"][len(bucket.name) + 1 :] for f in unreadable}
        )
        extra = sorted(set(on_disk) - set(inventory))
        mismatched = sorted(
            k for k in set(inventory) & set(on_disk) if inventory[k] != on_disk[k]
        )
        rec = {
            "parts_manifest": len(inventory),
            "parts_on_disk_readable": len(on_disk),
            "parts_unreadable": len(unreadable),
            "rows_manifest": sum(inventory.values()),
            "rows_footer": sum(on_disk.values()),
            "complete_marker": (bucket / "_COMPLETE").is_file(),
            "float_fidelity_recorded": "float_fidelity" in manifest,
        }
        if receipt.is_file():
            rec["rows_receipt"] = json.loads(receipt.read_text()).get("rows")
        for name, items in (
            ("bucket_parts_missing", missing),
            ("bucket_parts_unlisted", extra),
            ("bucket_part_rowcount_mismatch", mismatched),
        ):
            if items:
                failures.append(
                    {"path": bucket.name, "reason": name, "parts": len(items)}
                )
        if rec["rows_manifest"] != rec["rows_footer"]:
            failures.append(
                {"path": bucket.name, "reason": "bucket_rows_footer_vs_manifest"}
            )
        if (
            rec.get("rows_receipt") is not None
            and int(rec["rows_receipt"]) != rec["rows_manifest"]
        ):
            failures.append(
                {"path": bucket.name, "reason": "bucket_rows_manifest_vs_receipt"}
            )
        if not rec["complete_marker"]:
            failures.append(
                {"path": bucket.name, "reason": "bucket_complete_marker_missing"}
            )
        buckets[bucket.name] = rec
    return {
        "tool": "footer_scan.py",
        "tool_sha256": TOOL_SHA256,
        "root": str(root),
        "buckets_only": buckets_only,
        "parquet_files": files,
        "parquet_rows_total": rows_total,
        "buckets": buckets,
        "failures": failures,
        "ok": not failures,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--buckets-only", action="store_true")
    ap.add_argument("--allowed-root", action="append", default=[], type=Path)
    args = ap.parse_args()
    if not args.root.is_dir():
        print(f"not a directory: {args.root}", file=sys.stderr)
        return 2
    result = scan(
        args.root.resolve() if not args.root.is_symlink() else args.root,
        args.buckets_only,
        args.allowed_root,
    )
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True))
    nb = len(result["buckets"])
    print(
        f"footer_scan ok={result['ok']} files={result['parquet_files']} rows={result['parquet_rows_total']} "
        f"buckets={nb} failures={len(result['failures'])} tool_sha256={TOOL_SHA256[:12]}"
    )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
