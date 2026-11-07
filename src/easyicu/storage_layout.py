"""Read-only, versioned physical storage contracts and selector file resolution.

Bucket directories are immutable installations. A manifest is an assertion to
validate, never a hint from which a reader may reconstruct a missing layout.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import contextlib
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
BUILD_RECEIPT = "_BUCKET_BUILD_RECEIPT.json"
# Two-level layout: one partition per value of a variable key (label/itemid),
# then stay ranges inside each partition. The layout has M stay ranges (M a
# power of two, M - 1 shared breakpoints, ``v <= b`` belongs to the earlier
# range); a partition with modulus m merges M/m consecutive ranges, so a
# stay's bucket is ``bisect_left(breakpoints, stay) // (M // m)``. Files are
# sorted by stay, so contiguous stay batches also prune row groups.
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
class VariablePartition:
    """One variable-key value (``None`` for rows whose key is null)."""

    pid: int
    value: Any
    modulus: int
    secondary: dict  # selector column -> frozenset of values in this partition


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
    stay_column: str | None = None
    stay_key_type: str | None = None
    partitions: tuple = ()
    files_by_partition: dict | None = None  # (pid, bucket) -> files

    @property
    def files(self) -> tuple[Path, ...]:
        return tuple(
            sorted(p for paths in self.files_by_bucket.values() for p in paths)
        )

    def select(
        self,
        source,
        column: str | None,
        values: Iterable | None,
        *,
        stay_column: str | None = None,
        stay_values: Iterable | None = None,
    ) -> tuple[set, tuple[Path, ...]]:
        """Files a reader must scan; callers still apply the same predicates.

        ``stay_column``/``stay_values`` narrow only a variable layout whose
        stay key is that column; single-key layouts select exactly as before.
        Selecting is not consuming: a reader that reads the selection records
        it (:func:`record_bucket_consumed`).
        """
        values = None if values is None else tuple(values)
        if self.scheme == VARIABLE_SCHEME:
            return self._select_variable(source, column, values, stay_column, stay_values)
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

    def _select_variable(self, source, column, values, stay_column, stay_values):
        partitions = self.partitions
        if column is not None and values is not None:
            wanted = set(values)
            if column.casefold() == self.key_column.casefold():
                if matching_values(wanted, self.key_type):
                    partitions = tuple(
                        p for p in partitions if p.value is not None and p.value in wanted
                    )
            else:
                for name in partitions[0].secondary if partitions else ():
                    if name.casefold() == column.casefold():
                        partitions = tuple(
                            p for p in partitions if p.secondary[name] & wanted
                        )
                        break
        ranges = None
        if (
            stay_column is not None
            and stay_values is not None
            and stay_column.casefold() == (self.stay_column or "").casefold()
        ):
            stays = {v for v in stay_values if v is not None}
            if stays and matching_values(stays, self.stay_key_type):
                ranges = {bisect_left(self.breakpoints, v) for v in stays}
        targets, files = set(), []
        for part in partitions:
            span = self.modulus // part.modulus
            buckets = (
                range(part.modulus)
                if ranges is None
                else sorted({r // span for r in ranges})
            )
            for bucket in buckets:
                found = self.files_by_partition.get((part.pid, bucket), ())
                if found:
                    targets.add((part.pid, bucket))
                    files.extend(found)
        if not files and self.files:
            # An empty selection still has the installation's schema: hand the
            # reader one file, which the caller's own predicates empty, so the
            # answer has the same columns and dtypes as on a flat source.
            files = [self.files[0]]
        return targets, tuple(sorted(files))


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
        receipt = json.loads((root / BUILD_RECEIPT).read_text())
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
        if not isinstance(manifest["files_by_bucket"], dict):
            raise ValueError("files_by_bucket must be an object")
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
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
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
    Opening is not consuming: the manifest and build receipt are bound with
    the parts a reader reads (:func:`record_bucket_consumed`).
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
        _check_bucket_derived_from(root, manifest)
        if manifest.get("scheme") == VARIABLE_SCHEME:
            layout = _read_variable_contract(root, manifest, cache)
            _verify_bucket_sources(root, manifest)
            return layout
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
            if not isinstance(entry, dict):
                raise BucketLayoutError("bucket_inventory_mismatch", root)
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
        _verify_bucket_sources(root, manifest)
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
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
        # AttributeError: a field of the wrong JSON type (``[]`` where an
        # object belongs) is the manifest's defect, not an untyped crash.
        raise BucketLayoutError("bucket_manifest_invalid", root) from exc


def _is_power_of_two(value: Any) -> bool:
    return _positive_int(value) and value & (value - 1) == 0


def _json_key(value: Any, kind: str) -> Any:
    """A manifest key value, checked against the declared key type."""
    if value is None:
        return None
    if not matching_values((value,), kind):
        raise ValueError("partition value does not match the key type")
    return value


def _check_bucket_derived_from(root: Path, manifest: dict) -> None:
    """A bucket manifest's optional ``derived_from`` must be well formed.

    Present means bound to exact source files (relative path, size, mtime and
    sha256 of each); any other shape is an unverifiable binding. The current
    sources are compared with it by ``_verify_bucket_sources``.
    """
    if "derived_from" not in manifest:
        return
    derived = manifest["derived_from"]
    files = derived.get("files") if isinstance(derived, dict) else None
    table = derived.get("table") if isinstance(derived, dict) else None
    if (
        not isinstance(table, str)
        or PurePosixPath(table).parts != (table,)
        or table in {".", ".."}
        or "\\" in table
    ):
        raise BucketLayoutError("bucket_derived_from_invalid", root)
    if not isinstance(files, list) or not files:
        raise BucketLayoutError("bucket_derived_from_invalid", root)
    source_root = derived.get("source_root")
    if "source_root" in derived and (not isinstance(source_root, str) or not source_root):
        raise BucketLayoutError("bucket_derived_from_invalid", root)
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
            raise BucketLayoutError("bucket_derived_from_invalid", root)


def bucket_source_files(data_path: Path, table: str) -> tuple[Path, ...]:
    """The flat source files a bucket installation of ``table`` is built from.

    The single discovery rule shared by the builder (what it binds in
    ``derived_from``) and the reader (which requires the current set to be
    exactly that set): the non-hidden ``*.parquet`` files directly in
    ``<data_path>/<table>/``, else the single file ``<data_path>/<table>.parquet``;
    empty when neither exists.
    """
    data_path = Path(data_path)
    files = sorted(
        p for p in (data_path / table).glob("*.parquet") if not p.name.startswith(".")
    )
    if not files and (data_path / f"{table}.parquet").is_file():
        files = [data_path / f"{table}.parquet"]
    return tuple(files)


def _derived_from_base(root: Path, derived: dict) -> Path:
    """The directory ``derived_from``'s source paths are relative to.

    A layout built beside its data (the builder's default output) records no
    ``source_root``: its sources are relative to the bucket directory's
    physical (resolved) parent. A layout built to another output root records
    ``source_root``, the builder's data path relative to the real (resolved)
    parent of the bucket directory, or absolute when no relative path
    exists; the reader resolves it the same way, so opening the layout
    directly or through a bucket-only symlink farm finds the same files.
    Recorded file paths stay confined below this directory.
    """
    recorded = derived.get("source_root")
    if recorded is None:
        return root.resolve().parent
    relative = PurePosixPath(recorded)
    if relative.is_absolute():
        return Path(recorded)
    return root.resolve().parent.joinpath(*relative.parts)


def _derived_source(root: Path, derived: dict, entry: dict) -> Path:
    """Where one ``derived_from`` file lives: the single derived-source path rule.

    The recorded path is relative to :func:`_derived_from_base` and confined
    below it; any other path is ``bucket_derived_from_invalid``.
    """
    relative = PurePosixPath(entry["path"])
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise BucketLayoutError("bucket_derived_from_invalid", root)
    return _derived_from_base(root, derived).joinpath(*relative.parts)


def _verify_bucket_sources(root: Path, manifest: dict) -> None:
    """The flat source ``derived_from`` binds is still exactly what it recorded.

    Checked on every open, after the layout's own structure, so structural
    defects keep their reasons. Each recorded path is relative to the
    builder's data path, where the flat source lives: the bucket directory's
    parent, or the recorded ``source_root`` of a layout built to another
    output root (:func:`_derived_from_base`); a symlinked source is compared
    through its link, as the builder stat'ed it. No bytes are read; the
    recorded sha256 stays the builder's binding:
    - a recorded source that no longer exists is
      ``bucket_derived_from_source_missing``: the layout claims a provenance
      it can no longer show, so it is not served (rebuild the layout or
      restore the source);
    - a current source set (:func:`bucket_source_files`, the builder's own
      discovery rule) that is not exactly the recorded set, e.g. a shard
      added beside the recorded ones, is
      ``bucket_derived_from_source_set_changed``: the layout no longer holds
      every source row;
    - a changed size or mtime is ``bucket_derived_from_source_changed``.
    A manifest without ``derived_from`` (layouts built before the binding)
    is not checked here, as before.
    """
    derived = manifest.get("derived_from")
    if derived is None:
        return
    base = _derived_from_base(root, derived)
    recorded: dict[str, tuple[Path, dict]] = {
        PurePosixPath(entry["path"]).as_posix(): (_derived_source(root, derived, entry), entry)
        for entry in derived["files"]
    }
    stats = {}
    for name, (source, _entry) in recorded.items():
        try:
            stats[name] = source.stat()
        except OSError as exc:
            raise BucketLayoutError("bucket_derived_from_source_missing", root) from exc
    try:
        current = {
            PurePosixPath(*p.relative_to(base).parts).as_posix()
            for p in bucket_source_files(base, derived["table"])
        }
    except (OSError, ValueError) as exc:
        raise BucketLayoutError("bucket_derived_from_source_set_changed", root) from exc
    if current != set(recorded):
        raise BucketLayoutError("bucket_derived_from_source_set_changed", root)
    for name, (_source, entry) in recorded.items():
        stat = stats[name]
        if stat.st_size != entry["size"] or stat.st_mtime_ns != entry["mtime_ns"]:
            raise BucketLayoutError("bucket_derived_from_source_changed", root)


def _read_variable_contract(root: Path, manifest: dict, cache: dict | None) -> BucketLayout:
    """Validate a variable × stay installation exactly as hash/range ones.

    The partition list, each partition's bucket directories and the file
    inventory must all agree with the disk; a missing or extra partition is
    its own stable reason.
    """
    key, kind = manifest["key_column"], manifest["key_type"]
    stay, stay_kind = manifest["stay_column"], manifest["stay_key_type"]
    count = manifest["modulus"]
    if (
        not isinstance(key, str) or not key or kind not in KEY_TYPES
        or not isinstance(stay, str) or not stay or stay_kind not in KEY_TYPES
        or stay.casefold() == key.casefold()
    ):
        raise ValueError("unsupported key or stay column")
    if not _is_power_of_two(count):
        raise BucketLayoutError("bucket_modulus_mismatch", root)
    breaks = manifest["stay_breakpoints"]
    if (
        not isinstance(breaks, list)
        or len(breaks) != count - 1
        or not matching_values(breaks, stay_kind)
        or any(a > b for a, b in zip(breaks, breaks[1:]))
    ):
        raise BucketLayoutError("bucket_modulus_mismatch", root)
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
        raise ValueError("invalid builder contract")
    secondary_columns = manifest.get("secondary_keys", [])
    if not isinstance(secondary_columns, list) or any(
        not isinstance(c, str) or not c for c in secondary_columns
    ):
        raise ValueError("invalid secondary keys")
    raw_partitions = manifest["partitions"]
    if not isinstance(raw_partitions, list) or not raw_partitions:
        raise ValueError("missing partitions")
    partitions, seen_values = [], set()
    for index, entry in enumerate(raw_partitions):
        if not isinstance(entry, dict):
            raise BucketLayoutError("bucket_partition_invalid", root)
        if entry["id"] != index or not _is_power_of_two(entry["modulus"]):
            raise BucketLayoutError("bucket_partition_invalid", root)
        if count % entry["modulus"]:
            raise BucketLayoutError("bucket_modulus_mismatch", root)
        value = _json_key(entry["value"], kind)
        marker = ("null",) if value is None else ("value", value)
        if marker in seen_values:
            raise BucketLayoutError("bucket_partition_invalid", root)
        seen_values.add(marker)
        secondary = entry.get("secondary", {})
        # Types first: ``secondary`` is an object of selector column -> list
        # of scalar values ([] or a string has no ``values()``).
        if (
            not isinstance(secondary, dict)
            or set(secondary) != set(secondary_columns)
            or any(
                not isinstance(v, list)
                or any(isinstance(x, (list, dict)) for x in v)
                for v in secondary.values()
            )
        ):
            raise BucketLayoutError("bucket_partition_invalid", root)
        partitions.append(
            VariablePartition(
                index, value, entry["modulus"],
                {c: frozenset(secondary[c]) for c in secondary_columns},
            )
        )
    expected_dirs = {f"{PARTITION_PREFIX}{p.pid}" for p in partitions}
    actual_dirs = {d.name for d in root.glob(f"{PARTITION_PREFIX}*") if d.is_dir()}
    if expected_dirs - actual_dirs:
        raise BucketLayoutError("bucket_partition_missing", root)
    if actual_dirs - expected_dirs:
        raise BucketLayoutError("bucket_partition_extra", root)
    if any(d.is_dir() for d in root.glob("bucket_id=*")):
        raise BucketLayoutError("bucket_partition_extra", root)
    for part in partitions:
        directory = root / f"{PARTITION_PREFIX}{part.pid}"
        if directory.is_symlink():
            raise BucketLayoutError("bucket_inventory_mismatch", root)
        buckets = {d.name for d in directory.iterdir() if d.is_dir()}
        if buckets != {f"bucket_id={b}" for b in range(part.modulus)}:
            raise BucketLayoutError("bucket_modulus_mismatch", root)
    inventory = manifest["inventory"]
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("missing inventory")
    by_partition: dict[tuple[int, int], list[Path]] = {}
    seen: set[Path] = set()
    for entry in inventory:
        if not isinstance(entry, dict):
            raise BucketLayoutError("bucket_inventory_mismatch", root)
        relative = entry["relative_path"]
        parts = PurePosixPath(relative).parts if isinstance(relative, str) else ()
        if (
            len(parts) != 3
            or not parts[0].startswith(PARTITION_PREFIX)
            or not parts[1].startswith("bucket_id=")
            or not parts[2].endswith(".parquet")
            or parts[2].startswith(".")
        ):
            raise BucketLayoutError("bucket_inventory_mismatch", root)
        try:
            pid = int(parts[0][len(PARTITION_PREFIX):])
            bucket = int(parts[1].split("=", 1)[1])
        except ValueError as exc:
            raise BucketLayoutError("bucket_inventory_mismatch", root) from exc
        if (
            parts[0] != f"{PARTITION_PREFIX}{pid}"
            or parts[1] != f"bucket_id={bucket}"
            or not 0 <= pid < len(partitions)
        ):
            raise BucketLayoutError("bucket_inventory_mismatch", root)
        if not 0 <= bucket < partitions[pid].modulus:
            raise BucketLayoutError("bucket_modulus_mismatch", root)
        path = root.joinpath(*parts)
        if path.is_symlink() or path.parent.is_symlink():
            raise BucketLayoutError("bucket_inventory_mismatch", root)
        if any(
            not isinstance(entry[k], int) or isinstance(entry[k], bool) or entry[k] < 0
            for k in ("size", "row_count")
        ):
            raise BucketLayoutError("bucket_inventory_mismatch", root)
        if path in seen or not path.is_file() or path.stat().st_size != entry["size"]:
            raise BucketLayoutError("bucket_inventory_mismatch", root)
        seen.add(path)
        stat = path.stat()
        signature = (
            str(path), stat.st_size, stat.st_mtime_ns, kind, key, stay, stay_kind,
            tuple(sort), order, entry["row_count"],
        )
        if cache is None or signature not in cache:
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows != entry["row_count"]:
                raise BucketLayoutError("bucket_inventory_mismatch", root)
            schema = parquet.schema_arrow
            for column, column_kind in ((key, kind), (stay, stay_kind)):
                if (
                    column not in schema.names
                    or arrow_key_type(schema.field(column).type) != column_kind
                ):
                    raise BucketLayoutError("bucket_key_type_mismatch", root)
            if any(k not in schema.names for k in sort) or schema.field(order).type != pa.int64():
                raise BucketLayoutError("bucket_manifest_invalid", root)
            if cache is not None:
                cache[signature] = True
        by_partition.setdefault((pid, bucket), []).append(path)
    actual = {p for p in root.rglob("*.parquet") if not p.name.startswith(".")}
    if actual != seen:
        raise BucketLayoutError("bucket_inventory_mismatch", root)
    files_by_partition = {k: tuple(sorted(v)) for k, v in by_partition.items()}
    flat_index = {
        index: files
        for index, (_, files) in enumerate(sorted(files_by_partition.items()))
    }
    return BucketLayout(
        root,
        VARIABLE_SCHEME,
        key,
        kind,
        count,
        tuple(breaks),
        flat_index,
        order,
        stay_column=stay,
        stay_key_type=stay_kind,
        partitions=tuple(partitions),
        files_by_partition=files_by_partition,
    )


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
                sorted(path / name for name in _listdir(path)
                       if not name.startswith(".") and name.casefold().endswith(_PARQUET_SUFFIXES)
                       and (path / name).is_file())
            )
            if files:
                return files
        elif path.is_file() and path.suffix.lower() in {".parquet", ".pq"}:
            return (path,)
    return ()


PRESENCE_ABSENT = "absent"
PRESENCE_VALID = "valid"
PRESENCE_BROKEN = "broken"
LAYOUT_BROKEN = "table_layout_broken"


@dataclass(frozen=True)
class TablePresence:
    """Whether a table has a Parquet installation, from the owner's own contract.

    ``state`` is ``absent`` (no candidate exists), ``valid`` (``files`` is
    the owner's validated set, possibly a healthy zero-row file) or
    ``broken`` (a candidate exists but is a dangling link, or holds no
    file the owner can read). ``layout`` is ``bucket`` or ``flat`` for a
    valid table.
    """

    state: str
    files: tuple = ()
    layout: str | None = None
    path: Path | None = None
    loader: Path | None = None
    reason: str | None = None
    error: StorageIntegrityError | None = None
    contract: BucketLayout | None = None
    candidates: tuple = ()


SOURCE_AMBIGUOUS = "table_source_ambiguous"
_SOURCE_SUFFIXES = ("", ".parquet", ".pq", "_bucket")
_PARQUET_SUFFIXES = (".parquet", ".pq")


@dataclass(frozen=True)
class TableMatch:
    """One actual directory entry that names a table (any case, any contract root)."""

    path: Path
    kind: str  # "file" | "directory" | "bucket" | "broken"


@dataclass(frozen=True)
class TableMatches:
    """Every entry the matcher found for a table, deduplicated by physical source."""

    flat: tuple  # TableMatch: Parquet files and table directories
    bucket: tuple  # TableMatch: <name>_bucket directories
    broken: tuple  # TableMatch: dangling links, empty or untyped entries
    case_variants: int  # entries whose name differs from the canonical spelling


def _listdir(directory: Path) -> list:
    try:
        return sorted(os.listdir(directory))
    except (FileNotFoundError, NotADirectoryError):
        return []


def _table_keys(source, table: str) -> set:
    """The table name and a single configured file's stem, as spelled."""
    keys = {table}
    if getattr(source.config, "name", None) == "miiv":
        mapped = {"procedureevents_mv": "procedureevents", "inputevents_mv": "inputevents",
                  "inputevents_cv": "inputevents"}.get(table)
        if mapped:
            keys.add(mapped)
    try:
        entries = [
            entry.get("path") or entry.get("file")
            for entry in source.config.get_table(table).files
        ]
    except (AttributeError, KeyError, ValueError):
        entries = []
    entries = [e for e in entries if e]
    if len(entries) == 1:
        name = PurePosixPath(entries[0]).name
        for suffix in (".csv.gz", ".csv", ".parquet", ".pq"):
            if name.casefold().endswith(suffix):
                name = name[: -len(suffix)]
                break
        keys.add(name)
    return keys


def _classify(path: Path) -> TableMatch:
    if os.path.islink(path) and not path.exists():
        return TableMatch(path, "broken")
    name = path.name.casefold()
    if path.is_dir():
        if name.endswith("_bucket"):
            return TableMatch(path, "bucket")
        # Establish the whole directory inventory even for a configured stem
        # differing from the table name (e.g. HiRID general_table/). Otherwise
        # verify_flat_scan's single-file shortcut could hide an unreadable child.
        has_parquet = bool(_visible_parquet(path))
        return TableMatch(path, "directory" if has_parquet else "broken")
    if path.is_file() and name.endswith(_PARQUET_SUFFIXES):
        return TableMatch(path, "file")
    return TableMatch(path, "broken")


def _physical_identity(path: Path):
    """The same physical source: a file's (dev, ino), a directory's realpath."""
    if path.is_file():
        stat = path.stat()
        return ("file", stat.st_dev, stat.st_ino)
    return ("dir", os.path.realpath(path))


def match_table_entries(source, table: str, *, subdirectories: bool = True) -> TableMatches:
    """The one matcher of a table's on-disk entries (delta39d).

    It lists the actual directory entries, matched case-insensitively
    (casefold), under the contract roots: the installation, and its ``icu``
    and ``hosp`` directories in any case. Each entry is matched against the
    table name and a single configured file's stem, with the suffix '' (a
    table directory), ``.parquet``, ``.pq`` or ``_bucket`` in any case. With
    ``subdirectories``, all supported names in nested directories match too.
    Named table sources are leaves; symlink cycles are visited once. Each match is typed
    first: a dangling link, an empty directory or an entry of another type
    is broken. Aliases of one physical source (the same realpath, or a
    hardlink with the same dev and inode) count once.
    """
    base = Path(source.base_path)
    spelled = _table_keys(source, table)
    keys = {k.casefold() for k in spelled}
    wanted = {k + suffix for k in keys for suffix in _SOURCE_SUFFIXES}
    canonical = {k + suffix for k in spelled for suffix in _SOURCE_SUFFIXES}
    roots, found = [base], []
    for name in _listdir(base):
        if name.casefold() in {"icu", "hosp"}:
            path = base / name
            if os.path.islink(path) and not path.exists():
                found.append(TableMatch(path, "broken"))
            elif path.is_dir():
                roots.append(path)
    for root in roots:
        found += [_classify(root / n) for n in _listdir(root) if n.casefold() in wanted]
    # Registry chunks can name a directory differently from the logical table.
    # Keep that directory as one complete candidate, never its first chunk alone.
    try:
        entries = source.config.get_table(table).files
    except (AttributeError, KeyError, ValueError):
        entries = ()
    if len(entries) > 1:
        first = entries[0].get("path") or entries[0].get("file")
        directory = base / PurePosixPath(first).parent if first else base
        if directory != base and directory.is_dir() and any(
            n.casefold().endswith(_PARQUET_SUFFIXES) for n in _listdir(directory)
        ):
            found.append(_classify(directory))
    if subdirectories:
        visited = {os.path.realpath(root) for root in roots}

        def descend(directory):
            for name in _listdir(directory):
                if name.startswith(".") or name.casefold() in wanted:
                    continue
                path = directory / name
                if os.path.islink(path) and not path.exists():
                    found.append(TableMatch(path, "broken"))
                elif path.is_dir() and not name.casefold().endswith("_bucket"):
                    identity = os.path.realpath(path)
                    if identity in visited:
                        continue
                    visited.add(identity)
                    found.extend(_classify(path / n) for n in _listdir(path)
                                 if n.casefold() in wanted)
                    descend(path)

        for root in roots:
            descend(root)
    seen, flat, bucket, broken, variants = set(), [], [], [], 0
    for match in found:
        if match.path.name not in canonical:
            variants += 1
        if match.kind == "broken":
            broken.append(match)
            continue
        identity = _physical_identity(match.path)
        if identity in seen:
            continue
        seen.add(identity)
        (bucket if match.kind == "bucket" else flat).append(match)
    return TableMatches(tuple(flat), tuple(bucket), tuple(broken), variants)


def _bucket_derives_from(bucket: Path, flat: Path) -> bool:
    """Whether the bucket's verified ``derived_from`` binds exactly this flat source."""
    try:
        manifest = json.loads((bucket / MANIFEST).read_text())
        if not isinstance(manifest, dict):
            return False
        derived = manifest.get("derived_from")
        if not derived:
            return False
        _check_bucket_derived_from(bucket, manifest)
        _verify_bucket_sources(bucket, manifest)
        recorded = {
            _physical_identity(_derived_source(bucket, derived, entry)) for entry in derived["files"]
        }
    except (OSError, ValueError, KeyError, TypeError, BucketLayoutError):
        return False
    files = [flat] if flat.is_file() else [
        flat / n for n in _listdir(flat) if n.casefold().endswith(_PARQUET_SUFFIXES)
    ]
    return bool(files) and {_physical_identity(f) for f in files} == recorded


def table_source(source, table: str, *, subdirectories: bool = True) -> TableMatches:
    """The table's one source, or a typed refusal (delta39d).

    Two or more independent sources of one table (case variants, base and
    ``icu``, two nested directories, links to different sources) are
    ``table_source_ambiguous``. The only exemption is a flat source beside
    the bucket whose ``derived_from`` binds exactly that flat source (the
    bucket contract verifies the binding when it is opened). Broken entries
    are returned for the caller to refuse.
    """
    found = match_table_entries(source, table, subdirectories=subdirectories)
    pair = (
        len(found.flat) == 1 and len(found.bucket) == 1
        and _bucket_derives_from(found.bucket[0].path, found.flat[0].path)
    )
    if (len(found.flat) + len(found.bucket) > 1) and not pair:
        names = sorted(m.path.name for m in (*found.flat, *found.bucket))
        raise TableScanError(
            SOURCE_AMBIGUOUS, table,
            f"{len(names)} independent sources ({', '.join(names)}); remove all but one",
        )
    return found


def _csv_loader(source, table: str) -> Path | None:
    """CSV fallback paths belong to the decision owner too."""
    base = Path(source.base_path)
    try:
        explicit = source.config.get_table(table).first_file()
    except (AttributeError, KeyError, ValueError):
        explicit = None
    if explicit and str(explicit).casefold().endswith((".csv", ".csv.gz")):
        path = base / explicit
        if path.is_file():
            return path
    names = {key.casefold() + suffix for key in _table_keys(source, table)
             for suffix in (".csv", ".csv.gz")}
    return next((base / name for name in _listdir(base)
                 if name.casefold() in names and (base / name).is_file()), None)


def table_decision(source, table: str, *, listings_only: bool = False) -> TablePresence:
    """Own the complete Parquet decision and loader for reader, cohort and resolver.

    Every runtime caller uses recursive nested search. Only ABSENT permits
    the returned CSV loader. BROKEN carries the typed refusal and its reason;
    callers rethrow it with :func:`refuse_broken_presence`. A corrupt bucket
    may use a complete validated flat source, as before; two healthy independent
    sources refuse. No source-choice cache is maintained here.

    ``listings_only`` is for read-only installation censuses: it opens no files,
    returns the same candidates, and conservatively refuses multiple sources
    without inspecting a possible derived binding. Its files are unvalidated
    and must never be used by a reader.
    """
    if source.base_path is None:
        return TablePresence(PRESENCE_ABSENT, reason="parquet_source_absent")
    from .storage_binding import (
        StorageSourceChanged, bound_layout, bound_set, is_bound, source_key,
    )

    candidates = ()
    path = Path(source.base_path)
    try:
        chosen = None if listings_only else bound_layout(source_key(source, table))
        if chosen is not None and chosen["layout"] == "flat":
            for held in bound_set(source_key(source, table)):
                if not Path(held).is_file():
                    raise StorageSourceChanged("storage_source_missing", table, held)
        if chosen is not None and chosen["layout"] == "bucket":
            path = Path(chosen["root"])
            if not path.is_dir():
                raise StorageSourceChanged("storage_source_missing", table, str(path / MANIFEST))
            if os.path.realpath(path) != chosen["realpath"]:
                raise StorageSourceChanged("storage_source_changed", table, str(path))
        found = match_table_entries(source, table)
        candidates = (*found.flat, *found.bucket, *found.broken)
        if found.broken:
            path = found.broken[0].path
            refuse_broken_presence(table, TablePresence(PRESENCE_BROKEN, path=path))
        if len(found.flat) > 1 or len(found.bucket) > 1:
            raise TableScanError(SOURCE_AMBIGUOUS, table, "independent table sources")
        flat = found.flat[0].path if found.flat else None
        bucket = found.bucket[0].path if found.bucket else None
        if listings_only:
            if flat is not None and bucket is not None:
                raise TableScanError(SOURCE_AMBIGUOUS, table, "binding not inspected by listing probe")
            loader = bucket or flat
            return TablePresence(
                PRESENCE_VALID if loader else PRESENCE_ABSENT,
                layout="bucket" if bucket else "flat" if flat else None,
                path=loader, loader=loader or _csv_loader(source, table),
                reason="listings_only", candidates=candidates,
            )
        contract = None
        if bucket is not None:
            path = bucket
            try:
                contract = read_bucket_contract(
                    bucket, allow_legacy=source.config.name == "aumc",
                    cache=source._bucket_layout_cache,
                )
            except BucketLayoutError as exc:
                if (chosen is not None and chosen["layout"] == "bucket") or is_bound(bucket / MANIFEST):
                    raise StorageSourceChanged("storage_source_changed", table, str(bucket / MANIFEST)) from exc
                if flat is None:
                    if not _listdir(bucket):
                        raise TableScanError(LAYOUT_BROKEN, table, "empty bucket directory") from exc
                    raise
                # Compatibility: reject the cache and validate the entire flat
                # source below. An invalid bucket never makes a CSV eligible.
                record_bucket_fallback(source, table, exc)
            else:
                if flat is not None and not _bucket_derives_from(bucket, flat):
                    raise TableScanError(SOURCE_AMBIGUOUS, table, "bucket is independent of flat source")
                if chosen is not None and chosen["layout"] == "flat":
                    raise StorageSourceChanged("storage_source_changed", table, str(bucket))
        if contract is not None:
            source._bucket_dir_cache[str(table).lower()] = bucket
            return TablePresence(
                PRESENCE_VALID, tuple(contract.files), "bucket", bucket,
                loader=bucket, reason="parquet_source_valid", contract=contract,
                candidates=candidates,
            )
        files = verify_flat_scan(source, table, () if flat is None else _loader_scan(table, flat, ()))
        if not files:
            if chosen is not None:
                held = bound_set(source_key(source, table))
                raise StorageSourceChanged("storage_source_missing", table, held[0] if held else table)
            if flat is not None:
                raise TableScanError(LAYOUT_BROKEN, table, "source has no readable Parquet files")
            return TablePresence(PRESENCE_ABSENT, loader=_csv_loader(source, table),
                                 reason="parquet_source_absent", candidates=candidates)
        return TablePresence(PRESENCE_VALID, tuple(files), "flat", flat, loader=flat,
                             reason="parquet_source_valid", candidates=candidates)
    except StorageIntegrityError as exc:
        return TablePresence(PRESENCE_BROKEN, path=path, reason=exc.reason,
                             error=exc, candidates=candidates)
    except OSError as exc:
        error = unreadable_file(table, path, exc)
        error.__cause__ = exc
        return TablePresence(PRESENCE_BROKEN, path=path, reason=error.reason,
                             error=error, candidates=candidates)


def table_presence(source, table: str, *, subdirectories: bool = True) -> TablePresence:
    """Compatibility entry point; all callers share the owner's nested policy."""
    decision = table_decision(source, table)
    if decision.error is not None:
        refuse_broken_presence(table, decision)
    return decision


def refuse_broken_presence(table: str, presence: TablePresence) -> None:
    """A present table with nothing readable is an integrity failure, never absence."""
    if presence.state != PRESENCE_BROKEN:
        return
    if presence.error is not None:
        raise presence.error
    if os.path.lexists(presence.path) and not presence.path.exists():
        raise TableScanError(FILE_UNREADABLE, table, f"{presence.path.name}: dangling link")
    raise TableScanError(
        LAYOUT_BROKEN, table, f"{presence.path.name}: present without a readable Parquet file",
    )


class TableScanError(StorageIntegrityError):
    """A flat table scan does not cover the table's complete on-disk inventory."""

    def __init__(self, reason: str, table: str, detail: str):
        self.reason = reason
        self.table = table
        super().__init__(f"{reason}: {table}: {detail}")


FILE_UNREADABLE = "table_scan_file_unreadable"


def unreadable_file_errors() -> tuple[type[BaseException], ...]:
    """What opening or decoding a raw Parquet file raises when the file is bad.

    ``OSError`` covers a missing or unreadable file; ``pyarrow.ArrowException``
    covers a truncated or corrupt footer, page or schema.
    """
    import pyarrow as pa

    return (OSError, pa.ArrowException)


def unreadable_file(table: str, path, exc: BaseException) -> TableScanError:
    """A raw file the storage owner cannot read: a typed refusal, never an exclusion.

    Raise it ``from exc`` so the reader's own error stays the cause.
    """
    return TableScanError(FILE_UNREADABLE, str(table), f"{path}: {type(exc).__name__}: {exc}")


_FOOTER_ROWS: dict[tuple[str, int, int], int] = {}


def _footer_rows(path: Path, table: str) -> int:
    try:
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        rows = _FOOTER_ROWS.get(key)
        if rows is None:
            rows = _FOOTER_ROWS[key] = pq.ParquetFile(path).metadata.num_rows
    except unreadable_file_errors() as exc:
        raise unreadable_file(table, path, exc) from exc
    return rows


def _visible_parquet(directory: Path) -> set[Path]:
    # Path.rglob suppresses listing PermissionError. An unreadable subtree
    # cannot establish a complete inventory and must propagate to the owner.
    files = set()
    pending = [directory]
    while pending:
        parent = pending.pop()
        for name in _listdir(parent):
            if name.startswith("."):
                continue
            path = parent / name
            if path.is_dir() and not path.is_symlink():
                pending.append(path)
            elif path.suffix.casefold() in _PARQUET_SUFFIXES and path.is_file():
                files.add(path)
    return files


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
    except FileNotFoundError:
        # Only an absent manifest means "not a converted installation".
        return None
    except OSError as exc:
        # A manifest that exists but cannot be stat'ed (permissions, I/O,
        # a dangling symlink loop) must not downgrade the installation to
        # the unmanifested directory check.
        raise TableScanError(
            "conversion_manifest_unreadable", path.name, str(exc)[:200]
        ) from exc
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
            outputs_list = entry.get("outputs") if isinstance(entry, dict) else None
            if not isinstance(entry, dict) or not (
                outputs_list is None
                or (isinstance(outputs_list, list)
                    and all(isinstance(o, dict) for o in outputs_list))
            ):
                raise TableScanError(
                    "conversion_manifest_unreadable", path.name,
                    "each table entry must be an object whose outputs are objects",
                )
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
    """Every read the storage policy admitted without full verification, in order.

    ``table_scan_unmanifested``: the installation has no manifest at all (the
    compatibility path for older or hand-made installations).
    ``table_scan_unregistered``: a manifest exists but does not own the scan,
    accepted only because ``EASYICU_ALLOW_UNREGISTERED_SCAN=1`` was set.
    Both were checked against the directory inventory only.
    ``table_read_csv_unmanifested`` / ``table_read_csv_beside_manifest``: a
    CSV read (see :func:`csv_table_scan`).
    ``bucket_cache_ignored``: a bucket cache failed validation
    (``bucket_reason``) and the flat source was read instead.
    """
    return tuple(dict(entry) for entry in _SCAN_FALLBACKS)


_FALLBACK_DETAIL = {
    "table_read_csv_unmanifested": "is read from CSV in an installation without a "
    "conversion manifest; nothing verifies it",
    "table_read_csv_beside_manifest": "is read from CSV although its converted "
    "Parquet is listed (and was verified complete); the CSV itself is not verified",
    "bucket_cache_ignored": "has a bucket cache that failed validation; the "
    "flat table, its true source, is read instead",
}


def record_bucket_fallback(source, table: str, failure: BucketLayoutError) -> None:
    """Record that a corrupt bucket cache was bypassed for the flat source.

    Policy: a corrupt cache falls back to its true source, so the read (and
    publication) proceeds, but the fallback is a receipt fact, not a log line.
    """
    base = None if source.base_path is None else Path(source.base_path)
    _record_fallback("bucket_cache_ignored", base, table, bucket_reason=failure.reason)


SCAN_FALLBACK_LOG_ENV = "EASYICU_SCAN_FALLBACK_LOG"
_FALLBACK_LOGGED: set[tuple[str, tuple]] = set()


def _log_fallback(entry: dict[str, str]) -> None:
    """Append ``entry`` to the run's fallback log, shared with worker processes."""
    target = os.environ.get(SCAN_FALLBACK_LOG_ENV)
    if not target:
        return
    key = (target, json.dumps(entry, sort_keys=True))
    if key in _FALLBACK_LOGGED:
        return
    _FALLBACK_LOGGED.add(key)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


CONSUMED_EVENT = "table_consumed"
LAYOUT_EVENT = "table_layout"
FORMAT_PARQUET, FORMAT_CSV = "parquet", "csv"
FORMAT_BUCKET_PART, FORMAT_BUCKET_METADATA = "bucket_part", "bucket_metadata"


def _declared_view(manifest: Path) -> dict[tuple[str, str], int]:
    """Link targets a declared view admits: (table, realpath) -> size_bytes.

    A symlink farm declares its view with ``derived_from`` = {path, sha256}
    of the source installation's conversion manifest. A farm output is
    admitted when the source manifest still has that sha256 and the link
    resolves to an output of the same table there with the same size_bytes.
    Anything else admits nothing.

    The view is read once per run unit, never across runs (D10D-VIEW-ADMISSION):
    both manifests are bound with their sha256, so a later change in the run
    is ``storage_source_changed``. Outside a unit it is re-read every time.
    """
    from .storage_binding import active_unit, bind_files

    unit = active_unit()
    views = None if unit is None else unit.caches.setdefault("declared_views", {})
    if views is not None and str(manifest) in views:
        bind_files("conversion_manifest", views[str(manifest)][1], sha256=True)
        return views[str(manifest)][0]
    targets: dict[tuple[str, str], int] = {}
    bound: tuple = ()
    try:
        raw_own = manifest.read_bytes()
        own = json.loads(raw_own)
        derived = own.get("derived_from")
        if isinstance(derived, dict) and derived.get("path") and derived.get("sha256"):
            source = Path(derived["path"])
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() == derived["sha256"]:
                declared = {
                    (str(entry["table"]).lower(),
                     os.path.realpath(source.parent / out["relative_path"])): out.get("size_bytes")
                    for entry in json.loads(raw).get("tables", ())
                    for out in entry.get("outputs") or ()
                }
                for entry in own.get("tables", ()):
                    for out in entry.get("outputs") or ():
                        table = str(entry["table"]).lower()
                        real = os.path.realpath(manifest.parent / out["relative_path"])
                        size = out.get("size_bytes")
                        if isinstance(size, int) and declared.get((table, real), object()) == size:
                            targets[(table, real)] = size
                bound = (str(manifest), str(source))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        targets, bound = {}, ()
    if bound:
        # What admitted the view is what this run read: a later rewrite of
        # either manifest in the run is a changed source.
        bind_files("conversion_manifest", bound, sha256=True)
    if views is not None:
        views[str(manifest)] = (targets, bound)
    return targets


def declared_view_chain(base) -> dict[str, str] | None:
    """The chain a declared view is admitted by, for the run receipt.

    {farm_manifest_sha256, source_manifest_path, source_manifest_sha256}
    when ``base``'s conversion manifest declares ``derived_from`` {path,
    sha256} and the source manifest still has that sha256; None otherwise.
    """
    manifest = Path(base) / CONVERSION_MANIFEST
    try:
        raw = manifest.read_bytes()
        derived = json.loads(raw).get("derived_from")
        if not (isinstance(derived, dict) and derived.get("path") and derived.get("sha256")):
            return None
        source = hashlib.sha256(Path(derived["path"]).read_bytes()).hexdigest()
    except (OSError, ValueError, AttributeError):
        return None
    if source != derived["sha256"]:
        return None
    return {
        "farm_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "source_manifest_path": str(derived["path"]),
        "source_manifest_sha256": source,
    }


def admit_consumed_realpath(base, table: str, path) -> dict[str, Any]:
    """Admit the file a reader consumes for ``table`` by where it really lives.

    It must resolve under the data root: the real directory of the
    installation's ``conversion_manifest.json`` (a manifest linked in from
    the prepared installation names that installation), or ``realpath(base)``
    without a manifest. Otherwise a declared view must list it for this
    ``table`` with its current size (:func:`_declared_view`). Anything else
    is ``storage_read_outside_data_root``.
    """
    real = os.path.realpath(path)
    manifest = Path(base) / CONVERSION_MANIFEST
    root = (
        os.path.dirname(os.path.realpath(manifest)) if manifest.is_file()
        else os.path.realpath(base)
    )
    if os.path.commonpath([real, root]) == root:
        return {"target_realpath": real, "admitted_by": "data_root", "size": None, "sha256": None}
    declared = (
        _declared_view(manifest).get((str(table).lower(), real)) if manifest.is_file() else None
    )
    if declared is not None:
        try:
            size = os.stat(real).st_size
        except OSError:
            size = None
        # Admitted for this table only, and only at the declared size now.
        if size == declared:
            return {"target_realpath": real, "admitted_by": "declared_view",
                    "size": declared, "sha256": None}
    raise TableScanError(
        "storage_read_outside_data_root", table,
        f"{Path(path).name} resolves outside the data root and no declared view lists it",
    )


def record_consumed(
    where, table: str, files, fmt: str, *, key: tuple | None = None, sha256: bool = False
) -> tuple[dict[str, str], ...]:
    """The one consumed-files source: record what a reader actually reads.

    ``where`` is the data source (or its base path). A CSV or flat Parquet
    file must be admitted (:func:`admit_consumed_realpath`). The files are
    bound to the run (``storage_binding.consume``; ``key`` also binds a
    flat table's whole set) and logged as ``table_consumed`` events, so
    worker processes report into one receipt (:func:`read_consumed_log`).
    Integrity prechecks of other candidates never call this.
    """
    from .storage_binding import consume

    base = getattr(where, "base_path", where)
    files = tuple(Path(p) for p in files)
    if base is not None and fmt in (FORMAT_PARQUET, FORMAT_CSV):
        for path in files:
            admit_consumed_realpath(base, table, path)
    entries = consume(table, files, fmt, key=key, sha256=sha256)
    for entry in entries:
        _log_fallback({"event": CONSUMED_EVENT, "root": str(base), **entry})
    if fmt in (FORMAT_PARQUET, FORMAT_CSV) and files:
        _log_table_layout(base, table, "flat", base, lambda: _flat_derived_from(base, files))
    return entries


def _flat_derived_from(base, files) -> dict | None:
    """A flat read's declared-view chain when a file lies outside the data root, else None.

    Such a file was admitted only through the declared view
    (:func:`admit_consumed_realpath`), so the chain is what admitted it.
    """
    if base is None:
        return None
    manifest = Path(base) / CONVERSION_MANIFEST
    root = (
        os.path.dirname(os.path.realpath(manifest)) if manifest.is_file()
        else os.path.realpath(base)
    )
    if all(os.path.commonpath([os.path.realpath(p), root]) == root for p in files):
        return None
    return declared_view_chain(base)


def _bucket_record(layout: BucketLayout) -> dict:
    """A bucket read's manifest and ``derived_from`` sources (null for a legacy layout)."""
    from .storage_binding import bound_sha256

    manifest_path = layout.root / MANIFEST
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, ValueError):
        manifest, raw = None, None
    if not isinstance(manifest, dict):  # a legacy layout without a manifest
        return {"manifest": None, "derived_from": None, "derived_from_recorded": False}
    digest = bound_sha256(manifest_path) or hashlib.sha256(raw).hexdigest()
    derived = manifest.get("derived_from")
    if not isinstance(derived, dict):  # never guessed: the receipt refuses it
        return {"manifest": {"path": str(manifest_path), "sha256": digest},
                "derived_from": None, "derived_from_recorded": False}
    return {
        "manifest": {"path": str(manifest_path), "sha256": digest},
        "derived_from_recorded": True,
        "derived_from": {
            "table": derived["table"],
            "files": [
                {
                    "path": entry["path"],
                    "realpath": os.path.realpath(_derived_source(layout.root, derived, entry)),
                    "size": entry["size"],
                    "mtime_ns": entry["mtime_ns"],
                    "sha256": entry["sha256"],
                }
                for entry in derived["files"]
            ],
        },
    }


def _log_table_layout(base, table: str, layout: str, layout_root, details) -> None:
    """Log which layout a read of ``table`` was served from (``table_layout``).

    Written once per unit per (table, layout, root); ``details`` computes the
    rest of the record (bucket manifest and ``derived_from``, or a flat
    read's declared-view chain) only then.
    """
    from .storage_binding import active_unit

    if not os.environ.get(SCAN_FALLBACK_LOG_ENV):
        return
    unit = active_unit()
    seen = None if unit is None else unit.caches.setdefault("table_layouts_logged", set())
    marker = (os.environ.get(SCAN_FALLBACK_LOG_ENV), str(base), table, layout, str(layout_root))
    if seen is not None and marker in seen:
        return
    extra = details()
    if layout == "flat":
        extra = {"derived_from": extra}
    _log_fallback({"event": LAYOUT_EVENT, "root": str(base), "table": table,
                   "layout": layout, "layout_root": str(layout_root), **extra})
    if seen is not None:
        seen.add(marker)


def read_layout_log(path) -> tuple[dict, ...]:
    """The ``table_layout`` events of a run log (every worker's reads)."""
    return tuple(e for e in _read_run_log(path) if e.get("event") == LAYOUT_EVENT)


def consumed_table_layouts(consumed, layouts) -> dict[str, dict]:
    """The per-table layout record of a run: what served every consumed table.

    {table: {layout, root, derived_from[, manifest], reads}}. ``reads`` is
    every distinct layout record of the table (a flat read beside its bucket
    is legitimate); the top-level fields are the bucket's when there is one.
    A consumed table with no layout record is ``storage_layout_unrecorded``.
    """
    records: dict[str, dict] = {}
    for event in layouts:
        record = {k: v for k, v in event.items() if k not in ("event", "root", "table", "layout_root")}
        record["root"] = event["layout_root"]
        reads = records.setdefault(str(event["table"]), {"reads": []})["reads"]
        if record not in reads:
            reads.append(record)
    tables: dict[str, dict] = {}
    for entry in consumed:
        table = str(entry["table"])
        if table in tables:
            continue
        if table not in records:
            raise TableScanError(
                "storage_layout_unrecorded", table,
                "a file of this table was consumed but no layout record names what served it",
            )
        reads = sorted(records[table]["reads"], key=lambda r: json.dumps(r, sort_keys=True))
        primary = next((r for r in reads if r["layout"] == "bucket"), reads[0])
        tables[table] = {**primary, "reads": reads}
    return {table: tables[table] for table in sorted(tables)}


def replay_consumed(where, captured) -> None:
    """Record again what a cached payload was read from (a payload cache hit).

    ``captured`` is what :func:`storage_binding.capturing_consumed` collected
    when the payload was read. The same files are admitted, bound and
    verified as if they were read now, so the consumed set names the CSV a
    cached frame came from, never only a Parquet candidate.
    """
    groups: dict[tuple, list] = {}
    for table, path, fmt, sha256, _version, _racy in captured:
        groups.setdefault((table, fmt, bool(sha256)), []).append(path)
    for (table, fmt, sha256), paths in groups.items():
        record_consumed(where, table, paths, fmt, sha256=sha256)


def _file_sha256(path) -> str | None:
    try:
        with open(path, "rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError:
        return None


def payload_identity(source, captured) -> tuple[dict, ...] | None:
    """What a cached raw payload was read from, per table and kind (D14D).

    ``captured`` is what ``capturing_consumed`` collected while the payload
    was read. Each entry names the table, its kind (``flat``, ``csv`` or
    ``bucket``), the files read and the metadata identity: the conversion
    manifest (flat) or bucket manifest (bucket) with its sha256, and, for a
    bucket, every part its contract lists. None when a bucket read cannot be
    identified; such a payload is never served from the cache.
    """
    from .storage_binding import bound_layout, bound_sha256, source_key

    groups: dict[tuple[str, str], set] = {}
    for table, path, fmt, _sha, _version, _racy in captured:
        kind = {FORMAT_CSV: "csv", FORMAT_PARQUET: "flat"}.get(fmt, "bucket")
        groups.setdefault((table, kind), set()).add(str(path))
    base = getattr(source, "base_path", None)
    identity = []
    for (table, kind), paths in sorted(groups.items()):
        entry: dict[str, Any] = {"table": table, "kind": kind, "paths": tuple(sorted(paths))}
        if kind == "bucket":
            manifest = next((p for p in paths if Path(p).name == MANIFEST), None)
            if manifest is None:
                return None
            root = Path(manifest).parent
            layout = read_bucket_contract(
                root, allow_legacy=source.config.name == "aumc",
                cache=source._bucket_layout_cache,
            )
            entry.update(root=str(root), manifest=manifest,
                         manifest_sha256=bound_sha256(manifest) or _file_sha256(manifest),
                         files=tuple(sorted(str(f) for f in layout.files)))
        else:
            manifest = None if base is None else str(Path(base) / CONVERSION_MANIFEST)
            entry.update(manifest=manifest, manifest_sha256=None if manifest is None else _file_sha256(manifest))
        if kind == "flat":
            own = tuple(sorted(str(f) for f in flat_table_files(source, table)))
            # The table's own flat set: revalidated through the resolver (or the
            # deliberate flat scan beside a bucket). Any other directory read is
            # revalidated file by file.
            # A read of the table's flat set while a bucket serves the table
            # is the deliberate flat scan beside it, not the resolver's choice.
            chosen = bound_layout(source_key(source, table))
            if own != entry["paths"]:
                entry["via"] = "paths"
            elif chosen is not None:
                entry["via"] = "flat_scan" if chosen["layout"] == "bucket" else "resolver"
            else:
                entry["via"] = (
                    "flat_scan" if source.resolve_bucket_directory(table) is not None else "resolver"
                )
        identity.append(entry)
    return tuple(identity)


def _revalidate_entry(source, entry: dict) -> bool:
    """The owner's own check of one cached source; False = it no longer matches."""
    from .storage_binding import StorageSourceChanged

    table, kind = entry["table"], entry["kind"]
    if entry.get("manifest") and entry.get("manifest_sha256") != _file_sha256(entry["manifest"]):
        return False
    if kind == "csv":
        for path in entry["paths"]:
            csv_table_scan(source, table, Path(path))
        return True
    if kind == "bucket":
        root = source.resolve_bucket_directory(table)
        if root is None or str(root) != entry["root"]:
            return False
        try:
            layout = read_bucket_contract(
                root, allow_legacy=source.config.name == "aumc",
                cache=source._bucket_layout_cache,
            )
        except BucketLayoutError as exc:
            raise StorageSourceChanged("storage_source_changed", table, entry["manifest"]) from exc
        return tuple(sorted(str(f) for f in layout.files)) == entry["files"]
    if entry["via"] == "resolver":
        files = resolve_table_files(source, table, consume=False)
    elif entry["via"] == "flat_scan":
        files = verify_flat_scan(source, table, flat_table_files(source, table))
    else:
        files = tuple(Path(p) for p in entry["paths"] if Path(p).is_file())
    return tuple(sorted(str(f) for f in files)) == entry["paths"]


def revalidate_payload(source, identity, captured) -> bool:
    """Whether a cached raw payload may be served now; a hit records its sources.

    Every hit revalidates each source through its owner: the resolver (layout
    choice, file set, empty resolve), the bucket contract (full part
    inventory with sizes), or the CSV admission, plus the metadata sha256
    read from the bytes. No hit skips the owner: a stat signature cannot see
    a bucket appearing beside a flat table, a nested bucket part changing,
    or a manifest rewritten in place (delta17d). In a unit that has already
    bound the source, a mismatch raises
    ``storage_source_changed`` / ``storage_source_missing``: never a miss
    that falls through to an empty table, and never a stale payload. Only a
    unit that has not bound it may refresh (False: read again).
    """
    from .storage_binding import (
        StorageSourceChanged, active_unit, bound_layout, captured_still_current,
        is_bound, source_key,
    )

    if identity is None:
        return False
    unit = active_unit()
    for entry in identity:
        key = source_key(source, entry["table"])
        bound = unit is not None and (
            bound_layout(key) is not None or any(is_bound(p) for p in entry["paths"])
        )
        if not _revalidate_entry(source, entry):
            if bound:
                missing = [p for p in entry["paths"] if not Path(p).exists()]
                raise StorageSourceChanged(
                    "storage_source_missing" if missing else "storage_source_changed",
                    entry["table"], missing[0] if missing else entry["paths"][0],
                )
            return False
    if not captured_still_current(captured):
        if unit is not None and any(is_bound(c[1]) for c in captured):
            replay_consumed(source, captured)  # raises the bound file's change
        return False
    replay_consumed(source, captured)
    return True


def record_bucket_consumed(source, table: str, layout: BucketLayout, files) -> None:
    """A bucket read consumes its selected parts, its manifest and build receipt.

    It is also the table's layout choice for the run: bucket at this root.
    """
    from .storage_binding import bind_layout, is_bound, source_key

    bind_layout(source_key(source, table), table, "bucket", layout.root)

    receipt = layout.root / BUILD_RECEIPT
    # A receipt bound earlier must still be there (a deleted one is missing).
    has_receipt = receipt.is_file() or is_bound(receipt)
    metadata = (layout.root / MANIFEST, *((receipt,) if has_receipt else ()))
    record_consumed(source, table, metadata, FORMAT_BUCKET_METADATA, sha256=True)
    record_consumed(source, table, files, FORMAT_BUCKET_PART)
    _log_table_layout(
        getattr(source, "base_path", source), table, "bucket", layout.root,
        lambda: _bucket_record(layout),
    )


UNAVAILABLE_EVENT = "table_unavailable"
STORAGE_MODULE_ENV = "EASYICU_STORAGE_MODULE"
_UNAVAILABLE: list[dict] = []
# Where each open storage_module context's records start in _UNAVAILABLE,
# so a run without a run log never reports an earlier run's records.
_MODULE_STARTS: list[int] = []


@contextlib.contextmanager
def storage_module(module: str):
    """Tag this process's (and its children's) availability records with ``module``.

    Set and restored around each module's load; a child process started
    inside inherits the tag. A record made without a tag says module=None.
    """
    previous = os.environ.get(STORAGE_MODULE_ENV)
    os.environ[STORAGE_MODULE_ENV] = str(module)
    _MODULE_STARTS.append(len(_UNAVAILABLE))
    try:
        yield
    finally:
        _MODULE_STARTS.pop()
        if previous is None:
            os.environ.pop(STORAGE_MODULE_ENV, None)
        else:
            os.environ[STORAGE_MODULE_ENV] = previous


def record_table_unavailable(source, table: str) -> dict:
    """Record that a declared table has no physical source here (delta17w).

    Called only after :func:`verify_flat_scan` has refused a table the
    conversion manifest lists (``table_scan_files_missing``): what remains is
    a table the manifest does not list, or an installation without a
    manifest. Its read may give empty, but never silently: the record
    (``availability`` source_unavailable, ``missing_tables``) goes to the
    run log, the module receipt and the run manifest. It is written before
    the caller returns empty or raises.
    """
    base = getattr(source, "base_path", None)
    index = None if base is None else _conversion_index(Path(base))
    record = {
        "event": UNAVAILABLE_EVENT,
        "root": str(base),
        "table": str(table),
        "availability": "source_unavailable",
        "reason_code": "declared_table_not_installed",
        "manifest": "absent" if index is None else "unlisted",
        "missing_tables": [str(table)],
        "module": os.environ.get(STORAGE_MODULE_ENV) or None,
    }
    if record not in _UNAVAILABLE[_MODULE_STARTS[-1] if _MODULE_STARTS else 0:]:
        _UNAVAILABLE.append(record)
    _log_fallback(record)
    logging.getLogger(__name__).warning(
        "declared table %s has no physical source under %s (manifest %s): recorded "
        "source_unavailable", table, base, record["manifest"],
    )
    return record


def read_unavailable_log(path) -> tuple[dict, ...]:
    """The ``table_unavailable`` records of a run log (every worker's)."""
    return tuple(e for e in _read_run_log(path) if e.get("event") == UNAVAILABLE_EVENT)


def module_unavailability(module: str) -> list[dict]:
    """The availability records tagged with ``module``.

    From the run log when one is set (every worker's records); otherwise this
    process's records since the innermost open :func:`storage_module` began.
    """
    target = os.environ.get(SCAN_FALLBACK_LOG_ENV)
    if target:
        records = read_unavailable_log(target)
    else:
        records = tuple(_UNAVAILABLE[_MODULE_STARTS[-1] if _MODULE_STARTS else 0:])
    return [dict(r) for r in records if r.get("module") == module]


def read_consumed_log(path) -> tuple[dict[str, str], ...]:
    """The ``table_consumed`` events of a run log (every worker's reads)."""
    return tuple(e for e in _read_run_log(path) if e.get("event") == CONSUMED_EVENT)


def read_scan_fallback_log(path) -> tuple[dict[str, str], ...]:
    """The distinct fallbacks a run logged (see ``EASYICU_SCAN_FALLBACK_LOG``).

    The same log carries read events (``table_consumed``); those are not
    fallbacks (:func:`read_consumed_log`).
    """
    return tuple(e for e in _read_run_log(path) if "event" not in e)


def _read_run_log(path) -> tuple[dict[str, str], ...]:
    """Every distinct entry of a run log, fallbacks and read events."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return ()
    entries = {}
    for line in lines:
        if line.strip():
            entry = json.loads(line)
            entries.setdefault(json.dumps(entry, sort_keys=True), entry)
    return tuple(entries.values())


def _record_fallback(reason: str, base: Path, table: str, **extra: str) -> None:
    entry = {"reason": reason, "root": str(base), "table": table, **extra}
    _log_fallback(entry)
    warned = (str(base), table, reason)
    if warned in _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED.add(warned)
    _SCAN_FALLBACKS.append(entry)
    detail = _FALLBACK_DETAIL.get(
        reason,
        "has no conversion-manifest entry; checking the directory only. Re-run "
        "or verify the conversion to detect a missing chunk",
    )
    logging.getLogger(__name__).warning("%s: %s under %s %s.", reason, table, base, detail)


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
    scanned_rows = sum(_footer_rows(p, table) for p in files)
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
    inventory_rows = sum(_footer_rows(p, table) for p in inventory)
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
        own = tuple(loader / name for name in _listdir(loader)
                    if not name.startswith(".") and name.casefold().endswith(_PARQUET_SUFFIXES))
    elif loader.suffix.lower() in {".parquet", ".pq"}:
        own = (loader,) if loader.is_file() else ()
    else:
        # A CSV loader is admitted by csv_table_scan, never bound to Parquet.
        raise TypeError(f"{loader.name} is not a Parquet file or directory")
    if not files:
        return own
    if {_file_identity(p) for p in own} != {_file_identity(p) for p in files}:
        raise TableScanError(
            "table_scan_loader_mismatch", table,
            f"the reader opened {loader.name} ({len(own)} files) but the storage "
            f"policy resolves {len(files)} files under {files[0].parent.name}",
        )
    return files


def flat_table_scan(source, table: str, loader: Path) -> tuple[Path, ...]:
    """The validated flat installation of ``table`` at ``loader``.

    For readers that deliberately read the flat Parquet even when a bucket
    cache sits beside it (a patient-partitioned subset, for example); the
    flat set is still bound to the policy's location and to the manifest.
    """
    from .storage_binding import source_key

    files = verify_flat_scan(
        source, table, _loader_scan(table, Path(loader), flat_table_files(source, table))
    )
    if files:
        record_consumed(source, table, files, FORMAT_PARQUET, key=source_key(source, table))
    return files


def csv_table_scan(source, table: str, path: Path) -> None:
    """Admit a CSV read of ``table``; the conversion manifest lists Parquet only.

    Without a manifest the read is a legacy compatibility fallback and is
    recorded.  In a converted installation the table's Parquet inventory must
    still be complete (a listed table cannot fall back to CSV because its
    outputs vanished) and the CSV read is recorded; a table the manifest does
    not list is refused like an unregistered Parquet scan.
    """
    base = None if source.base_path is None else Path(source.base_path)
    index = _conversion_index(base) if base is not None else None
    if index is None:
        _record_fallback("table_read_csv_unmanifested", base, table)
    elif _registered_outputs(index, table):
        # An integrity precheck of the Parquet candidate: it is not what is
        # read, so it is not consumed (B9P-CSV-BINDING).
        resolve_table_files(source, table, consume=False)
        _record_fallback("table_read_csv_beside_manifest", base, table)
    elif os.environ.get(ALLOW_UNREGISTERED_SCAN_ENV) == "1":
        _record_fallback("table_scan_unregistered", base, table)
    else:
        raise TableScanError(
            "table_scan_unregistered", table,
            f"{Path(path).name} is not an output of any {CONVERSION_MANIFEST} entry; "
            f"regenerate the manifest (or set {ALLOW_UNREGISTERED_SCAN_ENV}=1)",
        )
    # The CSV is what is read: it is the consumed file (B9P-CSV-BINDING).
    record_consumed(source, table, (path,), FORMAT_CSV)


def resolve_table_files(
    source,
    table: str,
    selector_column: str | None = None,
    selector_values: Iterable | None = None,
    *,
    loader: Path | None = None,
    stay_column: str | None = None,
    stay_values: Iterable | None = None,
    consume: bool = True,
) -> tuple[Path, ...]:
    """The sole table + selector → files policy, shared by all read paths.

    ``loader`` is the path a disk reader resolved for the table; the returned
    set is what it must read (see :func:`_loader_scan`). ``stay_column`` and
    ``stay_values`` are the reader's own direct stay predicate
    (``column IN values``); a variable layout narrows its stay buckets with it.
    The returned files are what the caller reads, so they are recorded as
    consumed (:func:`record_consumed`). An integrity precheck or existence
    probe passes ``consume=False``.
    """
    from .storage_binding import (
        StorageSourceChanged, bind_layout, bound_layout, bound_set, source_key,
    )

    key = source_key(source, table)
    # The layout this run chose for the table is checked before any
    # existence branch (D10D): a bound bucket that vanished is missing, and
    # a bound flat table never turns into a bucket mid-run. Other tables'
    # reads, and a deliberate flat read beside a bucket, are not a choice.
    chosen = bound_layout(key)
    decision = table_decision(source, table)
    refuse_broken_presence(table, decision)
    root = decision.loader if decision.layout == "bucket" else None
    if chosen is not None and chosen["layout"] == "bucket" and root is None:
        raise StorageSourceChanged("storage_source_missing", table, chosen["root"])
    if chosen is not None and chosen["layout"] == "flat" and root is not None:
        raise StorageSourceChanged("storage_source_changed", table, str(root))
    if root is None:
        files = decision.files
        if loader is not None:
            if decision.state == PRESENCE_ABSENT:
                raise TableScanError(
                    "table_scan_loader_mismatch", table,
                    "a Parquet loader cannot override the owner's absent decision",
                )
            files = _loader_scan(table, Path(loader), files)
        files = verify_flat_scan(source, table, files)
        if consume and files:
            record_consumed(source, table, files, FORMAT_PARQUET, key=key)
            bind_layout(key, table, "flat", None)
        elif not files and chosen is not None:
            # Read non-empty earlier in this run, gone now (D9B-EMPTY-RESOLVE).
            held = bound_set(key)
            raise StorageSourceChanged(
                "storage_source_missing", table, held[0] if held else table
            )
        return files
    if loader is not None and Path(loader) != root:
        raise TableScanError(
            "table_scan_loader_mismatch", table,
            f"the reader opened {Path(loader).name} but the storage policy "
            f"resolves the bucket installation {root.name}",
        )
    layout = decision.contract
    selected = layout.select(
        source, selector_column, selector_values,
        stay_column=stay_column, stay_values=stay_values,
    )[1]
    if consume:
        record_bucket_consumed(source, table, layout, selected)
    return selected


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
        or any(root.glob(f"{PARTITION_PREFIX}*"))
    ):
        try:
            paths = read_bucket_contract(
                root, allow_legacy=root.name == "numericitems_bucket"
            ).files
        except BucketLayoutError:
            flat = root.with_name(root.name.removesuffix("_bucket"))
            if flat != root and flat.is_dir() and _visible_parquet(flat):
                return enumerate_parquet_files(flat)
            raise
    else:
        paths = sorted(_visible_parquet(root))
    return [p.as_posix() for p in paths]


def files_for_directory(
    source, directory, column=None, values=None, *, stay_column=None, stay_values=None
) -> tuple[Path, ...]:
    root = Path(directory)
    if (
        root.name.endswith("_bucket")
        or (root / MANIFEST).exists()
        or any(root.glob("bucket_id=*"))
        or any(root.glob(f"{PARTITION_PREFIX}*"))
    ):
        layout = read_bucket_contract(
            root,
            allow_legacy=source.config.name == "aumc",
            cache=source._bucket_layout_cache,
        )
        files = layout.select(
            source, column, values, stay_column=stay_column, stay_values=stay_values
        )[1]
        record_bucket_consumed(source, root.name.removesuffix("_bucket"), layout, files)
        return files
    files = tuple(Path(p) for p in enumerate_parquet_files(root))
    if files:
        record_consumed(source, root.name, files, FORMAT_PARQUET)
    return files


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
