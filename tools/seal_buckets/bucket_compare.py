"""Exact part-by-part comparison of two bucket layouts (lead-r13, supervisor ruling 2026-11-06).

This is bucket_compare_v3.py plus two seal-tool rules: every file read must resolve inside the
root being compared (no symlink may lead out of it, and a symlinked directory is a finding), and
the exit code is 0 only for IDENTICAL.

v3 = v2 plus a sub-label: a DIFFERS part whose only differences are +0.0/-0.0 flips (in values and/or
row-group min/max statistics) is labelled SIGNED_ZERO_ONLY, with the flip count. It is NOT a pass:
the verdict stays DIFFERS (bit-exact rule) and the exit code stays 1.

Before anything is compared, each side is validated against its OWN installation with the full bucket
storage contract of src/easyicu/storage_layout.py (ported, so this tool stays easyicu-free; a parity test
runs both on the same installs): schema, scheme/key/hash/builder fields, EVERY bucket directory the
modulus implies and every variable partition with its bucket directories (empty ones included),
canonical part paths, each listed part a real file of its inventory size, footer rows and key types,
the on-disk part set equal to the inventory, and the derived_from sources; then the inventory total must
equal the build receipt's rows. Any violation is INVALID (a part or an empty directory missing on both
sides is never IDENTICAL). The first containment finding (a path leading out of its root) stops the run
before any manifest is read.

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

# Import contract: this tool never imports easyicu (pinned by tests/core/test_seal_bucket_round2.py), so
# seal_rebuild.py may run it with `python -I`; it needs no seal-clone import origin.
import argparse
import filecmp
import hashlib
import json
import math
import numbers
import os
import sys
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# seal_rebuild.py executes this module from verified bytes (no __file__); the CLI run hashes its file.
TOOL_SHA256 = (
    hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if "__file__" in globals()
    else None
)


class StorageIntegrityError(Exception):
    """A bucket contradicts its own contract, inventory or receipt; ``reason`` is the stable code."""

    def __init__(self, reason: str, root: Path, detail: str = ""):
        super().__init__(f"{reason}: {root}{' ' + detail if detail else ''}")
        self.reason, self.root, self.detail = reason, root, detail


# ---- the bucket storage contract, ported from src/easyicu/storage_layout.py (read_bucket_contract and
# _read_variable_contract, allow_legacy=False, no cache) so this tool stays easyicu-free. The two must
# not diverge: tests/core/test_seal_bucket_round3.py runs both on the same synthetic installs and requires
# the same accept/refuse decision and reason code.
SCHEMA = "easyicu.bucket_storage/1"
ORDINAL = "_easyicu_source_ordinal"
VARIABLE_SCHEME = "variable_stay_range"
PARTITION_PREFIX = "var_id="
INTEGER_TYPES = frozenset(
    {
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
    }
)
KEY_TYPES = INTEGER_TYPES | {"VARCHAR", "FLOAT", "DOUBLE"}
_SHA256_HEX = frozenset("0123456789abcdef")


class _Invalid(ValueError):
    """A malformed field: reported as bucket_manifest_invalid, like the resolver's generic catch."""


def _matching_values(values, key_type: str) -> bool:
    if key_type == "VARCHAR":
        return all(isinstance(v, str) for v in values)
    if key_type in INTEGER_TYPES:
        return all(
            isinstance(v, numbers.Integral) and not isinstance(v, bool) for v in values
        )
    return all(
        isinstance(v, numbers.Real)
        and not isinstance(v, (numbers.Integral, bool))
        and math.isfinite(v)
        for v in values
    )


def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_power_of_two(value) -> bool:
    return _positive_int(value) and value & (value - 1) == 0


def _arrow_key_type(kind) -> str:
    if pa.types.is_dictionary(kind):
        kind = kind.value_type
    integers = {
        pa.int8(): "TINYINT",
        pa.int16(): "SMALLINT",
        pa.int32(): "INTEGER",
        pa.int64(): "BIGINT",
        pa.uint8(): "UTINYINT",
        pa.uint16(): "USMALLINT",
        pa.uint32(): "UINTEGER",
        pa.uint64(): "UBIGINT",
    }
    if kind in integers:
        return integers[kind]
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        return "VARCHAR"
    return {pa.float32(): "FLOAT", pa.float64(): "DOUBLE"}.get(kind, "unsupported")


def _sie(reason: str, root: Path, detail: str = "") -> StorageIntegrityError:
    return StorageIntegrityError(reason, root, detail)


def _hash_spec() -> dict:
    import duckdb  # the hash contract names the duckdb version that placed the keys

    return {"name": "duckdb.hash", "duckdb_version": duckdb.__version__}


def _check_derived_from(root: Path, manifest: dict) -> None:
    if "derived_from" not in manifest:
        return
    derived = manifest["derived_from"]
    files = derived.get("files") if isinstance(derived, dict) else None
    if not isinstance(derived, dict) or not isinstance(derived.get("table"), str):
        raise _sie("bucket_derived_from_invalid", root)
    if not isinstance(files, list) or not files:
        raise _sie("bucket_derived_from_invalid", root)
    for entry in files:
        digest = entry.get("sha256") if isinstance(entry, dict) else None
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("size"), int)
            or isinstance(entry.get("size"), bool)
            or entry["size"] < 0
            or not isinstance(entry.get("mtime_ns"), int)
            or not isinstance(digest, str)
            or len(digest) != 64
            or not set(digest) <= _SHA256_HEX
        ):
            raise _sie("bucket_derived_from_invalid", root)


def _verify_sources(root: Path, manifest: dict) -> None:
    derived = manifest.get("derived_from")
    if derived is None:
        return
    for entry in derived["files"]:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise _sie("bucket_derived_from_invalid", root)
        source = root.parent.joinpath(*relative.parts)
        try:
            st = source.stat()
        except OSError as exc:
            raise _sie(
                "bucket_derived_from_source_missing", root, entry["path"]
            ) from exc
        if st.st_size != entry["size"] or st.st_mtime_ns != entry["mtime_ns"]:
            raise _sie("bucket_derived_from_source_changed", root, entry["path"])


def _check_part(
    root: Path, path: Path, entry: dict, columns, sort, order, seen: set
) -> None:
    """One listed part, in the resolver's order: size/row_count types (a missing field is a manifest
    defect), then a duplicate path, a non-file or a size mismatch, then footer rows and key types."""
    if any(
        not isinstance(entry[k], int) or isinstance(entry[k], bool) or entry[k] < 0
        for k in ("size", "row_count")
    ):
        raise _sie("bucket_inventory_mismatch", root, entry["relative_path"])
    if path in seen or not path.is_file() or path.stat().st_size != entry["size"]:
        raise _sie("bucket_inventory_mismatch", root, entry["relative_path"])
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != entry["row_count"]:
        raise _sie("bucket_inventory_mismatch", root, entry["relative_path"])
    schema = parquet.schema_arrow
    for column, kind in columns:
        if (
            column not in schema.names
            or _arrow_key_type(schema.field(column).type) != kind
        ):
            raise _sie("bucket_key_type_mismatch", root, entry["relative_path"])
    if any(k not in schema.names for k in sort) or (
        order and schema.field(order).type != pa.int64()
    ):
        raise _sie("bucket_manifest_invalid", root, entry["relative_path"])


def _hash_or_range(root: Path, manifest: dict) -> None:
    scheme, key, kind = (manifest[k] for k in ("scheme", "key_column", "key_type"))
    if (
        scheme not in {"hash", "range"}
        or not isinstance(key, str)
        or not key
        or kind not in KEY_TYPES
    ):
        raise _Invalid("unsupported key or scheme")
    breaks = manifest.get("range_breakpoints", [])
    if scheme == "hash":
        count = manifest["modulus"]
        if not _positive_int(count):
            raise _sie("bucket_modulus_mismatch", root)
        if manifest["hash"] != _hash_spec():
            raise _sie("bucket_hash_mismatch", root)
    else:
        if (
            not isinstance(breaks, list)
            or not _matching_values(breaks, kind)
            or any(a >= b for a, b in zip(breaks, breaks[1:]))
        ):
            raise _Invalid("invalid range breakpoints")
        count = len(breaks) + 1
    sort, order = manifest["sort_keys"], manifest["source_order"]
    if (
        not isinstance(sort, list)
        or any(not isinstance(k, str) for k in sort)
        or order not in {None, ORDINAL}
        or (order and (not sort or sort[-1] != order))
        or not _positive_int(manifest["row_group_target"])
        or not isinstance(manifest["builder_version"], str)
        or not manifest["builder_version"]
    ):
        raise _Invalid("invalid builder contract")
    # every bucket directory the modulus implies, empty ones included (D29-EMPTY-STRUCT)
    dirs = {d.name for d in root.glob("bucket_id=*") if d.is_dir()}
    if dirs != {f"bucket_id={i}" for i in range(count)}:
        raise _sie("bucket_modulus_mismatch", root, "bucket directories")
    inventory = manifest["inventory"]
    if not isinstance(inventory, list) or not inventory:
        raise _Invalid("missing inventory")
    seen = set()
    for entry in inventory:
        if not isinstance(entry, dict):
            raise _sie("bucket_inventory_mismatch", root)
        relative = entry["relative_path"]
        if not isinstance(relative, str):
            raise _sie("bucket_inventory_mismatch", root)
        parts = PurePosixPath(relative).parts
        if (
            len(parts) != 2
            or not parts[0].startswith("bucket_id=")
            or not parts[1].endswith(".parquet")
            or parts[1].startswith(".")
        ):
            raise _sie("bucket_inventory_mismatch", root, relative)
        try:
            bid = int(parts[0].split("=", 1)[1])
        except ValueError as exc:
            raise _sie("bucket_inventory_mismatch", root, relative) from exc
        if parts[0] != f"bucket_id={bid}" or not 0 <= bid < count:
            raise _sie("bucket_modulus_mismatch", root, relative)
        path = root.joinpath(*parts)
        if path.is_symlink() or path.parent.is_symlink():
            raise _sie("bucket_inventory_mismatch", root, relative)
        _check_part(root, path, entry, [(key, kind)], sort, order, seen)
        seen.add(path)
    actual = {p for p in root.rglob("*.parquet") if not p.name.startswith(".")}
    if actual != seen:
        raise _sie("bucket_inventory_mismatch", root, "parts on disk != inventory")


def _variable(root: Path, manifest: dict) -> None:
    key, kind = manifest["key_column"], manifest["key_type"]
    stay, stay_kind = manifest["stay_column"], manifest["stay_key_type"]
    count = manifest["modulus"]
    if (
        not isinstance(key, str)
        or not key
        or kind not in KEY_TYPES
        or not isinstance(stay, str)
        or not stay
        or stay_kind not in KEY_TYPES
        or stay.casefold() == key.casefold()
    ):
        raise _Invalid("unsupported key or stay column")
    if not _is_power_of_two(count):
        raise _sie("bucket_modulus_mismatch", root)
    breaks = manifest["stay_breakpoints"]
    if (
        not isinstance(breaks, list)
        or len(breaks) != count - 1
        or not _matching_values(breaks, stay_kind)
        or any(a > b for a, b in zip(breaks, breaks[1:]))
    ):
        raise _sie("bucket_modulus_mismatch", root)
    sort, order = manifest["sort_keys"], manifest["source_order"]
    if (
        not isinstance(sort, list)
        or any(not isinstance(k, str) for k in sort)
        or order != ORDINAL
        or not sort
        or sort[-1] != order
        or not _positive_int(manifest["row_group_target"])
        or not isinstance(manifest["builder_version"], str)
        or not manifest["builder_version"]
    ):
        raise _Invalid("invalid builder contract")
    secondary_columns = manifest.get("secondary_keys", [])
    if not isinstance(secondary_columns, list) or any(
        not isinstance(c, str) or not c for c in secondary_columns
    ):
        raise _Invalid("invalid secondary keys")
    raw = manifest["partitions"]
    if not isinstance(raw, list) or not raw:
        raise _Invalid("missing partitions")
    moduli, seen_values = [], set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise _sie("bucket_partition_invalid", root)
        if entry["id"] != index or not _is_power_of_two(entry["modulus"]):
            raise _sie("bucket_partition_invalid", root)
        if count % entry["modulus"]:
            raise _sie("bucket_modulus_mismatch", root)
        value = entry["value"]
        if value is not None and not _matching_values((value,), kind):
            raise _Invalid("partition value does not match the key type")
        marker = ("null",) if value is None else ("value", value)
        if marker in seen_values:
            raise _sie("bucket_partition_invalid", root)
        seen_values.add(marker)
        secondary = entry.get("secondary", {})
        if (
            not isinstance(secondary, dict)
            or set(secondary) != set(secondary_columns)
            or any(
                not isinstance(v, list) or any(isinstance(x, (list, dict)) for x in v)
                for v in secondary.values()
            )
        ):
            raise _sie("bucket_partition_invalid", root)
        moduli.append(entry["modulus"])
    # every partition and every bucket directory inside it, empty ones included (D29-EMPTY-STRUCT)
    expected = {f"{PARTITION_PREFIX}{i}" for i in range(len(moduli))}
    actual_dirs = {d.name for d in root.glob(f"{PARTITION_PREFIX}*") if d.is_dir()}
    if expected - actual_dirs:
        raise _sie("bucket_partition_missing", root)
    if actual_dirs - expected:
        raise _sie("bucket_partition_extra", root)
    if any(d.is_dir() for d in root.glob("bucket_id=*")):
        raise _sie("bucket_partition_extra", root, "bucket_id at the top level")
    for pid, modulus in enumerate(moduli):
        directory = root / f"{PARTITION_PREFIX}{pid}"
        if directory.is_symlink():
            raise _sie("bucket_inventory_mismatch", root, directory.name)
        buckets = {d.name for d in directory.iterdir() if d.is_dir()}
        if buckets != {f"bucket_id={b}" for b in range(modulus)}:
            raise _sie("bucket_modulus_mismatch", root, directory.name)
    inventory = manifest["inventory"]
    if not isinstance(inventory, list) or not inventory:
        raise _Invalid("missing inventory")
    seen: set[Path] = set()
    for entry in inventory:
        if not isinstance(entry, dict):
            raise _sie("bucket_inventory_mismatch", root)
        relative = entry["relative_path"]
        parts = PurePosixPath(relative).parts if isinstance(relative, str) else ()
        if (
            len(parts) != 3
            or not parts[0].startswith(PARTITION_PREFIX)
            or not parts[1].startswith("bucket_id=")
            or not parts[2].endswith(".parquet")
            or parts[2].startswith(".")
        ):
            raise _sie("bucket_inventory_mismatch", root, str(relative))
        try:
            pid = int(parts[0][len(PARTITION_PREFIX) :])
            bucket = int(parts[1].split("=", 1)[1])
        except ValueError as exc:
            raise _sie("bucket_inventory_mismatch", root, relative) from exc
        if (
            parts[0] != f"{PARTITION_PREFIX}{pid}"
            or parts[1] != f"bucket_id={bucket}"
            or not 0 <= pid < len(moduli)
        ):
            raise _sie("bucket_inventory_mismatch", root, relative)
        if not 0 <= bucket < moduli[pid]:
            raise _sie("bucket_modulus_mismatch", root, relative)
        path = root.joinpath(*parts)
        if path.is_symlink() or path.parent.is_symlink():
            raise _sie("bucket_inventory_mismatch", root, relative)
        _check_part(
            root, path, entry, [(key, kind), (stay, stay_kind)], sort, order, seen
        )
        seen.add(path)
    actual = {p for p in root.rglob("*.parquet") if not p.name.startswith(".")}
    if actual != seen:
        raise _sie("bucket_inventory_mismatch", root, "parts on disk != inventory")


def validate_contract(root: Path) -> dict:
    """The resolver's full contract on one bucket: manifest, _COMPLETE, schema, every partition and
    bucket directory (empty ones included), canonical part paths, footer rows, key types, the on-disk
    part set and the derived_from sources. Returns the manifest."""
    if not (root / "_BUCKET_MANIFEST.json").is_file():
        raise _sie("bucket_manifest_missing", root)
    if not (root / "_COMPLETE").is_file():
        raise _sie("bucket_incomplete", root)
    try:
        manifest = json.loads(
            (root / "_BUCKET_MANIFEST.json").read_text(encoding="utf-8")
        )
        if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
            raise _Invalid(
                "not a bucket_storage/1 manifest"
            )  # no legacy layouts in a seal
        _check_derived_from(root, manifest)
        if manifest.get("scheme") == VARIABLE_SCHEME:
            _variable(root, manifest)
        else:
            _hash_or_range(root, manifest)
        _verify_sources(root, manifest)
        return manifest
    except StorageIntegrityError:
        raise
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
        raise _sie("bucket_manifest_invalid", root, type(exc).__name__) from exc


def validate_own_inventory(root: Path) -> dict:
    """The full contract, then this tool's receipt rules: every metadata file is contained, and the
    inventory total equals the build receipt's rows. Returns the manifest."""
    for name in ("_BUCKET_MANIFEST.json", "_BUCKET_BUILD_RECEIPT.json", "_COMPLETE"):
        if (root / name).exists() and not _contained(root / name, root):
            raise _sie("bucket_install_incomplete", root, name)
    manifest = validate_contract(root)
    try:
        receipt = json.loads((root / "_BUCKET_BUILD_RECEIPT.json").read_text())
    except (OSError, ValueError) as exc:
        raise _sie(
            "bucket_install_incomplete", root, "_BUCKET_BUILD_RECEIPT.json"
        ) from exc
    total = sum(e["row_count"] for e in manifest["inventory"])
    if not isinstance(receipt, dict) or receipt.get("rows") != total:
        raise _sie("bucket_rows_vs_receipt", root)
    return manifest


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


def _refuse(out: Path, cur: Path, new: Path, verdict: str, **extra) -> None:
    result = {
        "tool": "bucket_compare.py",
        "tool_sha256": TOOL_SHA256,
        "current": str(cur),
        "new": str(new),
        "verdict": verdict,
        "containment_findings": extra.pop("findings", []),
        **extra,
    }
    out.write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    print(
        f"verdict={verdict} {json.dumps(extra, sort_keys=True)} tool_sha256={TOOL_SHA256[:12]}"
    )
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("current", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seal-sha")
    args = ap.parse_args()
    cur, new = args.current, args.new
    findings = []
    listed = {}
    for root in (cur, new):
        # the first out-of-root finding stops the run: nothing further is read from either side
        listed[root] = _parts(root, findings)
        if not findings and not _contained(root / "_BUCKET_MANIFEST.json", root):
            findings.append(
                {
                    "root": str(root),
                    "path": "_BUCKET_MANIFEST.json",
                    "reason": "path_escapes_root",
                }
            )
        if findings:
            _refuse(args.out, cur, new, "REFUSED", findings=findings)
    for root in (cur, new):
        try:
            validate_own_inventory(root)
        except StorageIntegrityError as exc:
            _refuse(
                args.out,
                cur,
                new,
                "INVALID",
                integrity={
                    "root": str(root),
                    "reason": exc.reason,
                    "detail": exc.detail,
                },
            )
    a, b = listed[cur], listed[new]
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
