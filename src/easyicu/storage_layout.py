"""Read-only, versioned physical storage contracts and selector file resolution.

Bucket directories are immutable installations. A manifest is an assertion to
validate, never a hint from which a reader may reconstruct a missing layout.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import numbers
import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = "easyicu.bucket_storage/1"
BUILDER_VERSION = "easyicu.bucket_builder/2"
ORDINAL = "_easyicu_source_ordinal"
MANIFEST = "_BUCKET_MANIFEST.json"
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


class StorageIntegrityError(Exception):
    """An installation contradicts its own inventory; never "source absent".

    Deliberately not a ``ValueError``/``KeyError``/``OSError``: concept and
    table readers catch those as "this source is not available here" and move
    on, which turned a missing chunk into a silently absent concept.  Every
    subclass carries a stable ``reason`` code.
    """

    reason: str


class BucketLayoutError(StorageIntegrityError):
    """A stable, data-independent reason for rejecting a physical layout."""

    def __init__(self, reason: str, root: Path):
        self.reason = reason
        super().__init__(
            f"{reason}: {root.name}; use a complete flat source or rebuild/adopt the cache"
        )


@dataclass(frozen=True)
class BucketLayout:
    root: Path
    scheme: str
    key_column: str
    key_type: str
    modulus: int
    breakpoints: tuple
    files_by_bucket: dict[int, tuple[Path, ...]]
    source_order: str | None

    @property
    def files(self) -> tuple[Path, ...]:
        return tuple(
            sorted(p for paths in self.files_by_bucket.values() for p in paths)
        )

    def select(
        self, source, column: str | None, values: Iterable | None
    ) -> tuple[set, tuple[Path, ...]]:
        values = None if values is None else tuple(values)
        if (
            column is None
            or column.casefold() != self.key_column.casefold()
            or values is None
            or not matching_values(values, self.key_type)
        ):
            return set(self.files_by_bucket), self.files
        if self.scheme == "hash":
            targets = source._compute_target_buckets(
                set(values), self.modulus, duckdb, key_type=self.key_type
            )
        else:
            targets = {bisect_left(self.breakpoints, v) for v in values}
        return targets, tuple(
            p for i in sorted(targets) for p in self.files_by_bucket.get(i, ())
        )


def matching_values(values: Iterable, key_type: str) -> bool:
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


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _legacy_manifest(root: Path, manifest: dict, marker, allow: bool) -> dict:
    """Only the receipted AUMC itemid cache may use the old index format."""
    if not allow or root.name != "numericitems_bucket":
        raise BucketLayoutError("bucket_manifest_invalid", root)
    try:
        receipt = json.loads((root / "_BUCKET_BUILD_RECEIPT.json").read_text())
        count = receipt["bucket_count"]
        schema = dict(receipt["source_schema"])
        if not (receipt["schema"] == "easyicu_itemid_bucket_cache_v1"):
            raise ValueError("legacy receipt mismatch")
        if not (receipt["key"] == "itemid" and schema["itemid"] in INTEGER_TYPES):
            raise ValueError("legacy receipt mismatch")
        if not (receipt["duckdb_version"] == duckdb.__version__):
            raise ValueError("legacy receipt mismatch")
        if not (receipt["row_multiset_fingerprint"]["rows"] == receipt["rows"]):
            raise ValueError("legacy receipt mismatch")
        if not (_positive_int(count) and count == manifest["num_buckets"]):
            raise ValueError("legacy receipt mismatch")
        if not (manifest["complete_mtime_ns"] == marker.st_mtime_ns):
            raise ValueError("legacy receipt mismatch")
        if not (manifest["complete_size"] == marker.st_size):
            raise ValueError("legacy receipt mismatch")
        inventory = []
        for bid, paths in manifest["files_by_bucket"].items():
            for relative in paths:
                path = _inventory_path(root, relative, count)
                if not (int(bid) == int(path.parent.name.split("=")[1])):
                    raise ValueError("legacy receipt mismatch")
                inventory.append(
                    {
                        "relative_path": relative,
                        "size": path.stat().st_size,
                        "row_count": pq.ParquetFile(path).metadata.num_rows,
                    }
                )
        if not (sum(f["row_count"] for f in inventory) == receipt["rows"]):
            raise ValueError("legacy receipt mismatch")
        return dict(
            schema=SCHEMA,
            scheme="hash",
            key_column="itemid",
            key_type=schema["itemid"],
            modulus=count,
            hash={"name": "duckdb.hash", "duckdb_version": receipt["duckdb_version"]},
            sort_keys=[],
            source_order=None,
            row_group_target=122880,
            builder_version="legacy-aumc-v1",
            inventory=inventory,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise BucketLayoutError("bucket_legacy_unverifiable", root) from exc


def _inventory_path(root: Path, relative: str, count: int) -> Path:
    if not isinstance(relative, str):
        raise BucketLayoutError("bucket_inventory_mismatch", root)
    parts = PurePosixPath(relative).parts
    if (
        len(parts) != 2
        or not parts[0].startswith("bucket_id=")
        or not parts[1].endswith(".parquet")
        or parts[1].startswith(".")
    ):
        raise BucketLayoutError("bucket_inventory_mismatch", root)
    try:
        bid = int(parts[0].split("=", 1)[1])
    except ValueError as exc:
        raise BucketLayoutError("bucket_inventory_mismatch", root) from exc
    if parts[0] != f"bucket_id={bid}" or not 0 <= bid < count:
        raise BucketLayoutError("bucket_modulus_mismatch", root)
    path = root.joinpath(*parts)
    if path.is_symlink() or path.parent.is_symlink():
        raise BucketLayoutError("bucket_inventory_mismatch", root)
    return path


def read_bucket_contract(
    root: Path, *, allow_legacy: bool = False, cache: dict | None = None
) -> BucketLayout:
    """Validate contract, complete marker, directories, schemas and inventory on open.

    Only footer validation is cached; file sizes and directory membership are
    checked on every open, including for an already resolved data source.
    """
    root = Path(root)
    if not (root / MANIFEST).is_file():
        raise BucketLayoutError("bucket_manifest_missing", root)
    if not (root / "_COMPLETE").is_file():
        raise BucketLayoutError("bucket_incomplete", root)
    try:
        manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
        if manifest.get("schema") != SCHEMA:
            manifest = _legacy_manifest(
                root, manifest, (root / "_COMPLETE").stat(), allow_legacy
            )
        scheme, key, kind = (manifest[k] for k in ("scheme", "key_column", "key_type"))
        if (
            scheme not in {"hash", "range"}
            or not isinstance(key, str)
            or not key
            or kind not in KEY_TYPES
        ):
            raise ValueError("unsupported key or scheme")
        breaks = manifest.get("range_breakpoints", [])
        if scheme == "hash":
            count = manifest["modulus"]
            if not _positive_int(count):
                raise BucketLayoutError("bucket_modulus_mismatch", root)
            if manifest["hash"] != {
                "name": "duckdb.hash",
                "duckdb_version": duckdb.__version__,
            }:
                raise BucketLayoutError("bucket_hash_mismatch", root)
        else:
            if (
                not isinstance(breaks, list)
                or not matching_values(breaks, kind)
                or any(a >= b for a, b in zip(breaks, breaks[1:]))
            ):
                raise ValueError("invalid range breakpoints")
            count = len(breaks) + 1
        sort = manifest["sort_keys"]
        order = manifest["source_order"]
        if (
            not isinstance(sort, list)
            or any(not isinstance(k, str) for k in sort)
            or order not in {None, ORDINAL}
            or (order and (not sort or sort[-1] != order))
            or not _positive_int(manifest["row_group_target"])
            or not isinstance(manifest["builder_version"], str)
            or not manifest["builder_version"]
        ):
            raise ValueError("invalid builder contract")
        dirs = {d.name for d in root.glob("bucket_id=*") if d.is_dir()}
        if dirs != {f"bucket_id={i}" for i in range(count)}:
            raise BucketLayoutError("bucket_modulus_mismatch", root)
        inventory = manifest["inventory"]
        if not isinstance(inventory, list) or not inventory:
            raise ValueError("missing inventory")
        by_bucket: dict[int, list[Path]] = {i: [] for i in range(count)}
        seen = set()
        for entry in inventory:
            path = _inventory_path(root, entry["relative_path"], count)
            if any(
                not isinstance(entry[k], int)
                or isinstance(entry[k], bool)
                or entry[k] < 0
                for k in ("size", "row_count")
            ):
                raise BucketLayoutError("bucket_inventory_mismatch", root)
            if (
                path in seen
                or not path.is_file()
                or path.stat().st_size != entry["size"]
            ):
                raise BucketLayoutError("bucket_inventory_mismatch", root)
            seen.add(path)
            stat = path.stat()
            signature = (
                str(path),
                stat.st_size,
                stat.st_mtime_ns,
                kind,
                key,
                tuple(sort),
                order,
                entry["row_count"],
            )
            if cache is None or signature not in cache:
                parquet = pq.ParquetFile(path)
                if parquet.metadata.num_rows != entry["row_count"]:
                    raise BucketLayoutError("bucket_inventory_mismatch", root)
                schema = parquet.schema_arrow
                if (
                    key not in schema.names
                    or arrow_key_type(schema.field(key).type) != kind
                ):
                    raise BucketLayoutError("bucket_key_type_mismatch", root)
                if any(k not in schema.names for k in sort) or (
                    order and schema.field(order).type != pa.int64()
                ):
                    raise BucketLayoutError("bucket_manifest_invalid", root)
                if cache is not None:
                    cache[signature] = True
            by_bucket[int(path.parent.name.split("=")[1])].append(path)
        actual = {p for p in root.rglob("*.parquet") if not p.name.startswith(".")}
        if actual != seen:
            raise BucketLayoutError("bucket_inventory_mismatch", root)
        return BucketLayout(
            root,
            scheme,
            key,
            kind,
            count,
            tuple(breaks),
            {i: tuple(sorted(paths)) for i, paths in by_bucket.items()},
            order,
        )
    except BucketLayoutError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise BucketLayoutError("bucket_manifest_invalid", root) from exc


def arrow_key_type(kind: pa.DataType) -> str:
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


def flat_table_files(source, table: str) -> tuple[Path, ...]:
    """Locate only ordinary Parquet sources, without inspecting bucket policy."""
    if source.base_path is None:
        # An in-memory source (frames handed to ICUDataSource) has no files:
        # absence, as the resolver contract states, not a layout error.
        return ()
    roots = (source.base_path, source.base_path / "icu", source.base_path / "hosp")
    candidates = []
    try:
        entries = [
            entry.get("path") or entry.get("file")
            for entry in source.config.get_table(table).files
        ]
        entries = [entry for entry in entries if entry]
        # A table configured as several files (MIIV labevents/1..10.parquet)
        # is the whole directory, as in load_table; its first file alone is
        # a silent subset of the table.
        if len(entries) > 1:
            candidates.append(source.base_path / Path(entries[0]).parent)
        elif entries:
            candidates.append(source.base_path / entries[0])
    except (AttributeError, KeyError, ValueError):
        pass
    for base in roots:
        for name in dict.fromkeys((table, table.lower(), table.upper())):
            candidates.extend(
                (base / name, base / f"{name}.parquet", base / f"{name}.pq")
            )
    for path in candidates:
        if path.is_dir():
            files = tuple(
                sorted(p for p in path.glob("*.parquet") if not p.name.startswith("."))
            )
            if files:
                return files
        elif path.is_file() and path.suffix.lower() in {".parquet", ".pq"}:
            return (path,)
    return ()


class TableScanError(StorageIntegrityError):
    """A flat table scan does not cover the table's complete on-disk inventory."""

    def __init__(self, reason: str, table: str, detail: str):
        self.reason = reason
        self.table = table
        super().__init__(f"{reason}: {table}: {detail}")


_FOOTER_ROWS: dict[tuple[str, int, int], int] = {}


def _footer_rows(path: Path) -> int:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    rows = _FOOTER_ROWS.get(key)
    if rows is None:
        rows = _FOOTER_ROWS[key] = pq.ParquetFile(path).metadata.num_rows
    return rows


def _visible_parquet(directory: Path) -> set[Path]:
    return {
        p
        for p in directory.rglob("*")
        if p.suffix.lower() in {".parquet", ".pq"}
        and p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(directory).parts)
    }


CONVERSION_MANIFEST = "conversion_manifest.json"  # written by io.data_converter
_CONVERSION_INDEX: dict[tuple[str, int, int], dict[str, tuple[str, frozenset, int | None]]] = {}
_FALLBACK_WARNED: set[tuple[str, str]] = set()


_DERIVED_FROM: dict[tuple[str, int, int], dict | None] = {}
_SOURCE_SHA: dict[tuple[str, int, int], str] = {}


_SHA256_HEX = frozenset("0123456789abcdef")


def _derived_binding(path: Path, manifest: dict) -> dict | None:
    """Validate the optional ``derived_from`` binding of a conversion manifest.

    Absent means unbound.  Present means bound: an object naming the source
    manifest ``path`` and its 64-hex ``sha256``.  Any other shape is an
    unverifiable binding, not an absent one.
    """
    if "derived_from" not in manifest:
        return None
    derived = manifest["derived_from"]
    source = derived.get("path") if isinstance(derived, dict) else None
    digest = derived.get("sha256") if isinstance(derived, dict) else None
    if (
        not isinstance(source, str)
        or not source
        or not isinstance(digest, str)
        or len(digest) != 64
        or not set(digest) <= _SHA256_HEX
    ):
        raise TableScanError(
            "conversion_manifest_derived_invalid", path.name,
            "derived_from must be an object with a source path and a lowercase "
            "hex sha256; regenerate the derived manifest",
        )
    return {"path": source, "sha256": digest}


def _check_derived_from(path: Path, derived: dict | None) -> None:
    """A manifest derived from another (a symlink farm's) must match its source.

    A rebuilt source view leaves the derived manifest stale even when row
    counts happen to agree, so the source bytes are compared by SHA-256.  A
    source that cannot be read cannot be compared, so the binding fails.
    """
    if derived is None:
        return
    source = Path(derived["path"])
    try:
        stat = source.stat()
        key = (str(source), stat.st_size, stat.st_mtime_ns)
        digest = _SOURCE_SHA.get(key)
        if digest is None:
            digest = _SOURCE_SHA[key] = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as exc:
        raise TableScanError(
            "conversion_manifest_source_missing", path.name,
            f"derived from {source}, which cannot be read ({type(exc).__name__}); "
            "restore the source or regenerate the derived manifest",
        ) from exc
    if digest != derived["sha256"]:
        raise TableScanError(
            "table_scan_manifest_stale", path.name,
            f"derived from {source} (sha256 {derived['sha256'][:12]}), "
            f"which is now {digest[:12]}; regenerate the derived manifest",
        )


def _conversion_index(base: Path) -> dict[str, tuple[str, frozenset, int | None]] | None:
    """Map each converted output (path relative to the root) to its table entry."""
    path = Path(base) / CONVERSION_MANIFEST
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    index = _CONVERSION_INDEX.get(key)
    if index is not None:
        _check_derived_from(path, _DERIVED_FROM.get(key))
    if index is None:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            entries = manifest["tables"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise TableScanError(
                "conversion_manifest_unreadable", path.name, str(exc)[:200]
            ) from exc
        if not isinstance(manifest, dict) or not isinstance(entries, list):
            raise TableScanError(
                "conversion_manifest_unreadable", path.name, "tables must be a list"
            )
        derived = _derived_binding(path, manifest)
        _check_derived_from(path, derived)
        _DERIVED_FROM[key] = derived
        # One table may span several entries (HiRID converts each CSV part
        # separately); its inventory is the union, its rows the sum.
        tables: dict[str, tuple[set, int | None]] = {}
        for entry in entries:
            name = str(entry.get("table"))
            outputs, rows = tables.setdefault(name, (set(), 0))
            outputs.update(
                PurePosixPath(o["relative_path"]).as_posix()
                for o in entry.get("outputs") or ()
                if str(o.get("relative_path", "")).lower().endswith((".parquet", ".pq"))
            )
            count = entry.get("row_count")
            valid = isinstance(count, int) and not isinstance(count, bool)
            tables[name] = (outputs, rows + count if valid and rows is not None else None)
        index = {}
        for name, (outputs, rows) in tables.items():
            frozen = frozenset(outputs)
            for relative in frozen:
                index[relative] = (name, frozen, rows)
        _CONVERSION_INDEX[key] = index
    return index


def _relative(base: Path, path: Path) -> str | None:
    try:
        return Path(path).relative_to(base).as_posix()
    except ValueError:
        return None


ALLOW_UNREGISTERED_SCAN_ENV = "EASYICU_ALLOW_UNREGISTERED_SCAN"
_SCAN_FALLBACKS: list[dict[str, str]] = []


def scan_fallbacks() -> tuple[dict[str, str], ...]:
    """Every flat scan accepted without a conversion-manifest entry, in order.

    ``table_scan_unmanifested``: the installation has no manifest at all (the
    compatibility path for older or hand-made installations).
    ``table_scan_unregistered``: a manifest exists but does not own the scan,
    accepted only because ``EASYICU_ALLOW_UNREGISTERED_SCAN=1`` was set.
    Both were checked against the directory inventory only.
    """
    return tuple(dict(entry) for entry in _SCAN_FALLBACKS)


def _record_fallback(reason: str, base: Path, table: str) -> None:
    warned = (str(base), table)
    if warned in _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED.add(warned)
    _SCAN_FALLBACKS.append({"reason": reason, "root": str(base), "table": table})
    logging.getLogger(__name__).warning(
        "%s: %s under %s has no conversion-manifest entry; checking the "
        "directory only. Re-run or verify the conversion to detect a missing "
        "chunk.", reason, table, base,
    )


def _registered_outputs(index: dict, table: str) -> frozenset:
    wanted = table.lower()
    for name, outputs, _rows in index.values():
        if name.lower() == wanted:
            return outputs
    return frozenset()


def verify_flat_scan(source, table: str, files: tuple[Path, ...]) -> tuple[Path, ...]:
    """Fail closed unless a flat scan covers its table's whole inventory.

    The conversion manifest written when the installation was converted
    (``conversion_manifest.json``: each table's output files and total rows)
    is the inventory: the scan must be exactly the outputs of the table entry
    that owns its files, and the Parquet footer rows must equal the entry's
    row count, so a missing or truncated chunk fails the load.  A table the
    manifest lists with outputs cannot resolve to no files, and when a
    manifest exists every scanned file must belong to one of its entries
    (``EASYICU_ALLOW_UNREGISTERED_SCAN=1`` downgrades the latter to the
    directory check, recorded in :func:`scan_fallbacks`).  Without any
    manifest (older installations, hand-made views) the inventory is every
    visible Parquet file under the directory holding the scan, or the single
    file itself, and the fallback is warned and recorded.  The registry's
    file list names ricu's CSV-era chunks and is not an installation
    manifest.  Bucket layouts carry their own manifest inventory, validated
    on open (lead-r13: MIIV labevents read 1 of 10 files).
    """
    files = tuple(files)
    if source.base_path is None:
        return files
    base = Path(source.base_path)
    index = _conversion_index(base)
    if not files:
        registered = _registered_outputs(index, table) if index else frozenset()
        if registered:
            raise TableScanError(
                "table_scan_files_missing", table,
                f"conversion manifest lists {len(registered)} files; none found",
            )
        return files
    parents = {p.parent for p in files}
    if len(parents) != 1:
        raise TableScanError("table_scan_mixed_roots", table, f"{len(parents)} directories")
    scanned_rows = sum(_footer_rows(p) for p in files)
    relatives = {_relative(base, p) for p in files}
    owners = {index.get(r) for r in relatives} if index is not None and None not in relatives else {None}
    if None not in owners and len(owners) == 1:
        owner, outputs, rows = next(iter(owners))
        if relatives != set(outputs) or (rows is not None and scanned_rows != rows):
            raise TableScanError(
                "table_scan_manifest_mismatch", table,
                f"scanned {len(relatives)} files/{scanned_rows} rows; conversion "
                f"manifest {owner!r} lists {len(outputs)} files/{rows} rows",
            )
        return files
    if None not in owners:
        raise TableScanError("table_scan_mixed_roots", table, f"{len(owners)} manifest tables")
    if index is None:
        _record_fallback("table_scan_unmanifested", base, table)
    elif os.environ.get(ALLOW_UNREGISTERED_SCAN_ENV) == "1":
        _record_fallback("table_scan_unregistered", base, table)
    else:
        raise TableScanError(
            "table_scan_unregistered", table,
            f"{len(files)} scanned files are not outputs of any {CONVERSION_MANIFEST} "
            f"entry; regenerate the manifest (or set {ALLOW_UNREGISTERED_SCAN_ENV}=1 "
            "to accept a recorded directory-only check)",
        )
    parent = next(iter(parents))
    names = {table.lower(), f"{table.lower()}_bucket"}
    single = len(files) == 1 and parent.name.lower() not in names
    inventory = set(files) if single else _visible_parquet(parent)
    inventory_rows = sum(_footer_rows(p) for p in inventory)
    if set(files) != inventory or scanned_rows != inventory_rows:
        raise TableScanError(
            "table_scan_incomplete", table,
            f"scanned {len(files)} files/{scanned_rows} rows of "
            f"{len(inventory)} files/{inventory_rows} rows",
        )
    return files


def _file_identity(path: Path) -> tuple[int, int]:
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino


def _loader_scan(table: str, loader: Path, files: tuple[Path, ...]) -> tuple[Path, ...]:
    """Bind the file set a reader opened (``loader``) to the policy's file set.

    The disk loader and the policy locate files independently (registry
    mappings, case-insensitive exports).  The reader must consume the policy's
    validated set; a loader the policy cannot see is validated as the table's
    scan, and a loader that disagrees with the policy fails closed.
    """
    if loader.is_dir():
        own = tuple(sorted(p for p in loader.glob("*.parquet") if not p.name.startswith(".")))
    elif loader.is_file() and loader.suffix.lower() in {".parquet", ".pq"}:
        own = (loader,)
    else:
        own = ()
    if not files:
        return own
    if {_file_identity(p) for p in own} != {_file_identity(p) for p in files}:
        raise TableScanError(
            "table_scan_loader_mismatch", table,
            f"the reader opened {loader.name} ({len(own)} files) but the storage "
            f"policy resolves {len(files)} files under {files[0].parent.name}",
        )
    return files


def resolve_table_files(
    source,
    table: str,
    selector_column: str | None = None,
    selector_values: Iterable | None = None,
    *,
    loader: Path | None = None,
) -> tuple[Path, ...]:
    """The sole table + selector → files policy, shared by all read paths.

    ``loader`` is the path a disk reader resolved for the table; the returned
    set is what it must read (see :func:`_loader_scan`).
    """
    root = source.resolve_bucket_directory(table)
    if root is None:
        files = flat_table_files(source, table)
        if loader is not None:
            files = _loader_scan(table, Path(loader), files)
        return verify_flat_scan(source, table, files)
    if loader is not None and Path(loader) != root:
        raise TableScanError(
            "table_scan_loader_mismatch", table,
            f"the reader opened {Path(loader).name} but the storage policy "
            f"resolves the bucket installation {root.name}",
        )
    layout = read_bucket_contract(
        root,
        allow_legacy=source.config.name == "aumc",
        cache=source._bucket_layout_cache,
    )
    return layout.select(source, selector_column, selector_values)[1]


def validate_concept_layouts(source, dictionary, names: Iterable[str]) -> None:
    """Surface storage corruption before concept availability fallbacks catch it."""
    pending, visited, tables = list(names), set(), set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        definition = dictionary.get(name)
        if definition is None:
            continue
        pending.extend(definition.depends_on or ())
        pending.extend(definition.sub_concepts or ())
        tables.update(
            s.table for s in definition.for_data_source(source.config) if s.table
        )
    for table in sorted(tables):
        if table not in source._table_sources and table not in source._dataset_sources:
            source.resolve_bucket_directory(table)


def enumerate_parquet_files(directory: Path) -> list[str]:
    root = Path(directory)
    if not root.exists():
        return []
    if (
        root.name.endswith("_bucket")
        or (root / MANIFEST).exists()
        or any(root.glob("bucket_id=*"))
    ):
        try:
            paths = read_bucket_contract(
                root, allow_legacy=root.name == "numericitems_bucket"
            ).files
        except BucketLayoutError:
            flat = root.with_name(root.name.removesuffix("_bucket"))
            if flat != root and flat.is_dir() and any(flat.glob("*.parquet")):
                return enumerate_parquet_files(flat)
            raise
    else:
        paths = sorted(p for p in root.rglob("*.parquet") if not p.name.startswith("."))
    return [p.as_posix() for p in paths]


def files_for_directory(
    source, directory, column=None, values=None
) -> tuple[Path, ...]:
    root = Path(directory)
    if (
        root.name.endswith("_bucket")
        or (root / MANIFEST).exists()
        or any(root.glob("bucket_id=*"))
    ):
        layout = read_bucket_contract(
            root,
            allow_legacy=source.config.name == "aumc",
            cache=source._bucket_layout_cache,
        )
        return layout.select(source, column, values)[1]
    return tuple(Path(p) for p in enumerate_parquet_files(root))


def parquet_columns(path):
    return pq.read_schema(path).names


def restore_frame_order(frame):
    if ORDINAL in frame.columns:
        return (
            frame.sort_values(ORDINAL, kind="stable")
            .drop(columns=ORDINAL)
            .reset_index(drop=True)
        )
    return frame


def empty_frame(directory, columns):
    """Use the installation schema even when the selected buckets are empty."""
    import pandas as pd

    files = enumerate_parquet_files(directory)
    if not files:
        return pd.DataFrame(columns=list(columns) if columns else [])
    schema = pq.read_schema(files[0])
    wanted = list(columns) if columns else [c for c in schema.names if c != ORDINAL]
    schema = pa.schema([schema.field(c) for c in wanted])
    return pa.Table.from_batches([], schema=schema).to_pandas()


def parquet_relation(
    files: Iterable[Path], *, keep_order_column: bool = False, ordered: bool = True
) -> str:
    """Restore source order before scientific operations and hide physical fields.

    ``ordered=False`` is for order-insensitive aggregates only (for example
    ``MAX(datetime) ... GROUP BY patientid``): the ORDER BY wraps the whole
    relation, so filters such as a cohort semi-join run after a full-table
    sort.  On HiRID observations that sort took minutes and over 20 GiB.
    """
    files = tuple(files)
    paths = (
        "["
        + ", ".join("'" + Path(p).as_posix().replace("'", "''") + "'" for p in files)
        + "]"
    )
    relation = f"read_parquet({paths}, union_by_name=true, hive_partitioning=false)"
    if not ordered:
        if files and ORDINAL in pq.read_schema(files[0]).names and not keep_order_column:
            return f'(SELECT * EXCLUDE("{ORDINAL}") FROM {relation})'
        return relation
    if files and ORDINAL in pq.read_schema(files[0]).names:
        projection = "*" if keep_order_column else f'* EXCLUDE("{ORDINAL}")'
        return f'(SELECT {projection} FROM {relation} ORDER BY "{ORDINAL}")'
    if files and Path(files[0]).parent.name.startswith("bucket_id="):
        # Adoption retains the existing AUMC cache's physical scan order.
        return relation
    # Explicit file order is the logical source order used by the builder.
    # Parallel Parquet scans may interleave files even when the list is sorted.
    # Keep that physical scheduling out of FIRST and exact-time tie handling.
    relation = relation[:-1] + ", filename=true, file_row_number=true)"
    return f"(SELECT * EXCLUDE(filename, file_row_number) FROM {relation} ORDER BY filename, file_row_number)"
