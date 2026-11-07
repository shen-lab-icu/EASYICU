"""Exact part-by-part comparison of two bucket layouts (lead-r13, supervisor ruling 2026-11-06).

This is bucket_compare_v3.py plus two seal-tool rules: every file read must resolve inside the
root being compared (no symlink may lead out of it, and a symlinked directory is a finding), and
the exit code is 0 only for IDENTICAL.

v3 = v2 plus a sub-label: a DIFFERS part whose only differences are +0.0/-0.0 flips (in values and/or
row-group min/max statistics) is labelled SIGNED_ZERO_ONLY, with the flip count. It is NOT a pass:
the verdict stays DIFFERS (bit-exact rule) and the exit code stays 1.

usage: bucket_compare.py CURRENT_BUCKET NEW_BUCKET --out RESULT.json [--seal-sha SHA]
exit:  0 every part IDENTICAL (bytes) and no containment finding; 1 anything else (LOGICALLY_IDENTICAL,
       DIFFERS, a path escaping its root); 2 usage error. The JSON keeps the full verdict.

Per part (every part is decoded and compared, byte-identical or not):
  * bytes        : full byte equality of the parquet file;
  * schema       : Arrow schema equal with check_metadata=True (names, exact types, nullability, metadata);
  * row groups   : same number of row groups and the same row count in each;
  * values       : per row group, per column, positionally (so row order is covered): identical null
                   positions; non-null values equal, floating point compared by bit pattern with every NaN
                   equal to every NaN (so -0.0 != 0.0, NaN == NaN);
  * statistics   : per row group, per column, the parquet statistics present/absent alike, and
                   has_min_max, min, max, null_count, num_values, distinct_count equal (NaN-aware);
  verdict: IDENTICAL (bytes equal, and all of the above), LOGICALLY_IDENTICAL (bytes differ, all of the
  above equal), DIFFERS (anything else, including a part missing on either side).
Layout level: part names, manifest inventory (sizes excluded only for LOGICALLY_IDENTICAL parts),
partitions/stay breakpoints, derived_from source (path, sha256) sets.
Output: counts and part paths only (no values, IDs or dates); includes this tool's own sha256.
"""

import argparse
import filecmp
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

TOOL_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _float_bits(arr):
    width = arr.type.bit_width
    as_int = {16: pa.uint16(), 32: pa.uint32(), 64: pa.uint64()}[width]
    return arr.view(as_int)


def _column_equal(a, b):
    return _column_diff(a, b)[0] == "equal"


def _column_diff(a, b):
    """('equal'|'signed_zero'|'other', signed-zero flips). Null positions first, then values
    (floats by bit pattern, NaN == NaN); a float mismatch where both sides are zero is a sign flip."""
    if len(a) != len(b) or a.null_count != b.null_count:
        return "other", 0
    if a.null_count and not pc.all(pc.equal(pc.is_null(a), pc.is_null(b))).as_py():
        return "other", 0
    t = a.type
    if pa.types.is_floating(t):
        na, nb = pc.fill_null(pc.is_nan(a), False), pc.fill_null(pc.is_nan(b), False)
        if not pc.all(pc.equal(na, nb)).as_py():
            return "other", 0
        bits_eq = pc.equal(
            _float_bits(a.fill_null(0) if a.null_count else a),
            _float_bits(b.fill_null(0) if b.null_count else b),
        )
        ok = pc.or_(bits_eq, pc.and_(na, nb))
        if a.null_count:
            ok = pc.or_(ok, pc.is_null(a))
        if pc.all(ok).as_py():
            return "equal", 0
        bad = pc.invert(ok)
        fa, fb = pc.filter(a, bad), pc.filter(b, bad)
        zero = pc.and_(pc.equal(fa, 0.0), pc.equal(fb, 0.0))
        flips = int(pc.sum(pc.cast(zero, pa.int64())).as_py() or 0)
        return ("signed_zero" if flips == len(fa) else "other"), flips
    if pa.types.is_nested(t):
        return ("equal" if a.equals(b) else "other"), 0
    eq = pc.equal(a, b)
    if a.null_count:
        eq = pc.fill_null(eq, True)
    return ("equal" if pc.all(eq).as_py() else "other"), 0


def _same_scalar(x, y):
    if (
        isinstance(x, float)
        and isinstance(y, float)
        and math.isnan(x)
        and math.isnan(y)
    ):
        return True
    if isinstance(x, float) and isinstance(y, float):
        return x == y and math.copysign(1.0, x) == math.copysign(1.0, y)
    return x == y


def _stats(col):
    s = col.statistics
    if s is None:
        return None
    out = {
        "has_min_max": s.has_min_max,
        "null_count": s.null_count if s.has_null_count else None,
        "num_values": s.num_values,
        "distinct_count": s.distinct_count if s.has_distinct_count else None,
    }
    if s.has_min_max:
        out["min"], out["max"] = s.min, s.max
    return out


def _stats_equal(sa, sb):
    return _stats_diff(sa, sb) == "equal"


def _stats_diff(sa, sb):
    """'equal' | 'signed_zero' (only min/max differ, by the sign of a zero) | 'other'."""
    if sa is None or sb is None:
        return "equal" if sa is None and sb is None else "other"
    if set(sa) != set(sb):
        return "other"
    bad = [k for k in sa if not _same_scalar(sa[k], sb[k])]
    if not bad:
        return "equal"

    def zero_flip(x, y):
        return isinstance(x, float) and isinstance(y, float) and x == 0.0 and y == 0.0

    return (
        "signed_zero"
        if all(k in ("min", "max") and zero_flip(sa[k], sb[k]) for k in bad)
        else "other"
    )


def compare_part(pa_path, pb_path):
    res = {"bytes_equal": filecmp.cmp(pa_path, pb_path, shallow=False)}
    fa, fb = pq.ParquetFile(pa_path), pq.ParquetFile(pb_path)
    res["schema_equal"] = fa.schema_arrow.equals(fb.schema_arrow, check_metadata=True)
    ma, mb = fa.metadata, fb.metadata
    rga = [ma.row_group(i).num_rows for i in range(ma.num_row_groups)]
    rgb = [mb.row_group(i).num_rows for i in range(mb.num_row_groups)]
    res["row_groups_equal"] = rga == rgb
    res["rows"] = sum(rga)
    shape_ok = (
        res["schema_equal"]
        and res["row_groups_equal"]
        and ma.num_columns == mb.num_columns
    )
    value_kinds, stat_kinds, flips = set(), set(), 0
    if shape_ok:
        for i in range(ma.num_row_groups):
            ta, tb = fa.read_row_group(i), fb.read_row_group(i)
            for j in range(ma.num_columns):
                stat_kinds.add(
                    _stats_diff(
                        _stats(ma.row_group(i).column(j)),
                        _stats(mb.row_group(i).column(j)),
                    )
                )
                kind, n = _column_diff(
                    ta.column(j).combine_chunks(), tb.column(j).combine_chunks()
                )
                value_kinds.add(kind)
                flips += n
    values_ok = shape_ok and value_kinds <= {"equal"}
    stats_ok = shape_ok and stat_kinds <= {"equal"}
    res["values_equal"] = bool(values_ok)
    res["statistics_equal"] = bool(stats_ok)
    res["signed_zero_flips"] = flips
    logical = (
        res["schema_equal"]
        and res["row_groups_equal"]
        and res["values_equal"]
        and res["statistics_equal"]
    )
    if (
        shape_ok
        and not logical
        and (value_kinds | stat_kinds) <= {"equal", "signed_zero"}
    ):
        res["sub_label"] = "SIGNED_ZERO_ONLY"
    res["verdict"] = (
        "IDENTICAL"
        if res["bytes_equal"] and logical
        else "LOGICALLY_IDENTICAL"
        if logical
        else "DIFFERS"
    )
    return res


def _contained(path: Path, root: Path) -> bool:
    real, base = os.path.realpath(path), os.path.realpath(root)
    return real == base or real.startswith(base.rstrip("/") + "/")


def _parts(root: Path, findings: list) -> list[str]:
    """Parquet parts under ``root`` without following symlinks out of it."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames:
            if os.path.islink(os.path.join(dirpath, name)):
                rel = os.path.relpath(os.path.join(dirpath, name), root)
                findings.append(
                    {
                        "root": str(root),
                        "path": rel,
                        "reason": "symlinked_dir_in_bucket",
                    }
                )
        for name in filenames:
            if not name.endswith(".parquet") or name.startswith("."):
                continue
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            if not _contained(path, root):
                findings.append(
                    {"root": str(root), "path": rel, "reason": "path_escapes_root"}
                )
                continue
            out.append(rel)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("current", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seal-sha")
    args = ap.parse_args()
    cur, new = args.current, args.new
    findings = []
    a, b = _parts(cur, findings), _parts(new, findings)
    for root in (cur, new):
        if not _contained(root / "_BUCKET_MANIFEST.json", root):
            findings.append(
                {
                    "root": str(root),
                    "path": "_BUCKET_MANIFEST.json",
                    "reason": "path_escapes_root",
                }
            )
    parts = {}
    for n in sorted(set(a) | set(b)):
        if n in a and n in b:
            parts[n] = compare_part(cur / n, new / n)
        else:
            parts[n] = {
                "verdict": "DIFFERS",
                "missing_in": "new" if n in a else "current",
            }
    logical_only = {
        n for n, r in parts.items() if r["verdict"] == "LOGICALLY_IDENTICAL"
    }
    ma, mb = (json.loads((r / "_BUCKET_MANIFEST.json").read_text()) for r in (cur, new))

    def strip(inv):
        return [
            {
                k: v
                for k, v in e.items()
                if not (k == "size" and e.get("relative_path") in logical_only)
            }
            for e in inv or []
        ]

    def src(m):
        return sorted(
            (f["path"], f["sha256"])
            for f in (m.get("derived_from") or {}).get("files", [])
        )

    bc = (mb.get("builder_code") or {}).get("commit")
    layout = {
        "names_equal": a == b,
        "inventory_equal_excluding_logical_part_sizes": strip(ma.get("inventory"))
        == strip(mb.get("inventory")),
        "partitions_equal": ma.get("partitions") == mb.get("partitions")
        and ma.get("stay_breakpoints") == mb.get("stay_breakpoints"),
        "derived_from_sources_equal": bool(src(mb))
        and (not src(ma) or src(ma) == src(mb)),
        "new_builder_code_commit": None
        if bc is None
        else ("seal" if args.seal_sha and bc.startswith(args.seal_sha[:9]) else bc),
    }
    counts = {
        v: sum(1 for r in parts.values() if r["verdict"] == v)
        for v in ("IDENTICAL", "LOGICALLY_IDENTICAL", "DIFFERS")
    }
    counts["DIFFERS_SIGNED_ZERO_ONLY"] = sum(
        1 for r in parts.values() if r.get("sub_label") == "SIGNED_ZERO_ONLY"
    )
    layout_ok = all(
        layout[k]
        for k in (
            "names_equal",
            "inventory_equal_excluding_logical_part_sizes",
            "partitions_equal",
            "derived_from_sources_equal",
        )
    )
    verdict = (
        "DIFFERS"
        if counts["DIFFERS"] or not layout_ok or findings
        else "LOGICALLY_IDENTICAL"
        if counts["LOGICALLY_IDENTICAL"]
        else "IDENTICAL"
    )
    result = {
        "tool": "bucket_compare.py",
        "tool_sha256": TOOL_SHA256,
        "containment_findings": findings,
        "current": str(cur),
        "new": str(new),
        "verdict": verdict,
        "counts": counts,
        "rows_total": sum(r.get("rows", 0) for r in parts.values()),
        "differing_parts": sorted(
            n for n, r in parts.items() if r["verdict"] != "IDENTICAL"
        ),
        "signed_zero_only_parts": sorted(
            n for n, r in parts.items() if r.get("sub_label") == "SIGNED_ZERO_ONLY"
        ),
        "signed_zero_flips_total": sum(
            r.get("signed_zero_flips", 0) for r in parts.values()
        ),
        "layout": layout,
        "parts": parts,
    }
    args.out.write_text(
        json.dumps(result, indent=1, sort_keys=True, default=str) + "\n"
    )
    print(
        f"verdict={verdict} parts={len(parts)} identical={counts['IDENTICAL']} "
        f"logically_identical={counts['LOGICALLY_IDENTICAL']} differs={counts['DIFFERS']} "
        f"(signed_zero_only={counts['DIFFERS_SIGNED_ZERO_ONLY']}, flips={result['signed_zero_flips_total']}) "
        f"rows={result['rows_total']} layout_ok={layout_ok} containment_findings={len(findings)} "
        f"tool_sha256={TOOL_SHA256[:12]}"
    )
    sys.exit(0 if verdict == "IDENTICAL" else 1)


if __name__ == "__main__":
    main()
