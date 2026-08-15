"""Host-observed physical schema receipts for typed products.

This module reports representation facts for one digest-bound table or
structured JSON document. It never assigns an exposure, outcome, cohort,
estimator, estimand, or semantic role. The legacy tabular public import remains
re-exported by ``declared_product``.
"""

from __future__ import annotations

from ..canonical_json import sha256_file as _sha256_file

import csv
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

_MAX_TYPED_TABLE_COLUMNS = 2048
_MAX_TYPED_TABLE_HEADER_CHARS = 256
_MAX_TYPED_TABLE_RECEIPT_BYTES = 64 * 1024
_MAX_TYPED_TABLE_DTYPE_PROFILE_BYTES = 16 * 1024 * 1024

#: A non-numeric column is reported as a closed value set only when it looks
#: like a category vocabulary rather than free text or an identifier. Above
#: either bound the column is simply omitted -- omission is the pre-existing
#: state and says nothing, which is the fail-closed answer.
_MAX_TYPED_TABLE_CATEGORY_CARDINALITY = 24
_MAX_TYPED_TABLE_CATEGORY_VALUE_CHARS = 64

_MAX_TYPED_JSON_ARTIFACT_BYTES = 4 * 1024 * 1024
_MAX_TYPED_JSON_RECEIPT_BYTES = 64 * 1024
_MAX_TYPED_JSON_PATHS = 128
_MAX_TYPED_JSON_DEPTH = 3
_MAX_TYPED_JSON_KEYS_PER_OBJECT = 128
_MAX_TYPED_JSON_KEY_CHARS = 256
_JSON_VALUE_TYPES = frozenset(
    {"array", "boolean", "null", "number", "object", "string"}
)


def _tabular_artifact_columns(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> list[str]:
    """Read only a verified artifact's ordered physical columns."""

    suffix = path.suffix.lower()
    try:
        with path.open("rb") as handle:
            if expected_sha256 is not None:
                digest = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                if digest.hexdigest() != expected_sha256:
                    return []
                handle.seek(0)
            if suffix in {".csv", ".tsv"}:
                delimiter = "\t" if suffix == ".tsv" else ","
                with io.TextIOWrapper(
                    handle,
                    encoding="utf-8-sig",
                    newline="",
                ) as text_handle:
                    return [
                        str(value)
                        for value in next(
                            csv.reader(text_handle, delimiter=delimiter),
                            [],
                        )
                    ]
            if suffix in {".parquet", ".pq"}:
                import pyarrow.parquet as pq

                return [str(value) for value in pq.read_schema(handle).names]
    except Exception:
        return []
    return []


def _serialized_json_size(value: object) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _json_value_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def _safe_json_object_keys(value: Mapping[object, object]) -> list[str] | None:
    keys = list(value)
    if (
        len(keys) > _MAX_TYPED_JSON_KEYS_PER_OBJECT
        or not all(isinstance(key, str) for key in keys)
        or any(
            not key
            or len(key) > _MAX_TYPED_JSON_KEY_CHARS
            or any(ord(character) < 32 for character in key)
            for key in keys
        )
    ):
        return None
    return [str(key) for key in keys]


def _json_pointer_child(path: str, key: str) -> str:
    escaped = key.replace("~", "~0").replace("/", "~1")
    return f"{path}/{escaped}" if path else f"/{escaped}"


def _valid_json_structure_receipt(
    receipt: object,
    *,
    expected_sha256: str,
) -> bool:
    if not isinstance(receipt, Mapping):
        return False
    if set(receipt) != {"json_format", "source_sha256", "root_type", "paths"}:
        return False
    if (
        receipt.get("json_format") != "json"
        or receipt.get("source_sha256") != expected_sha256
        or receipt.get("root_type") not in _JSON_VALUE_TYPES
    ):
        return False
    paths = receipt.get("paths")
    if not isinstance(paths, Mapping) or len(paths) > _MAX_TYPED_JSON_PATHS:
        return False
    for path, entry in paths.items():
        if (
            not isinstance(path, str)
            or len(path) > 1024
            or (path and not path.startswith("/"))
            or any(ord(character) < 32 for character in path)
            or not isinstance(entry, Mapping)
            or entry.get("type") not in _JSON_VALUE_TYPES
        ):
            return False
        entry_type = entry.get("type")
        if entry_type == "object":
            allowed_fields = {"type", "keys"}
        elif entry_type == "array":
            allowed_fields = {
                "type",
                "length",
                "item_types",
                "object_item_keys",
                "object_item_keys_consistent",
            }
        else:
            allowed_fields = {"type"}
        if not set(entry) <= allowed_fields:
            return False
        keys = entry.get("keys")
        if keys is not None and (
            not isinstance(keys, list)
            or not all(isinstance(key, str) for key in keys)
            or _safe_json_object_keys({key: None for key in keys}) != keys
        ):
            return False
        if entry_type == "object" and keys is None:
            return False
        if entry.get("type") == "array":
            length = entry.get("length")
            item_types = entry.get("item_types")
            if (
                isinstance(length, bool)
                or not isinstance(length, int)
                or length < 0
                or not isinstance(item_types, list)
                or not all(value in _JSON_VALUE_TYPES for value in item_types)
            ):
                return False
            object_item_keys = entry.get("object_item_keys")
            if object_item_keys is not None and (
                not isinstance(object_item_keys, list)
                or not all(isinstance(key, str) for key in object_item_keys)
                or _safe_json_object_keys(
                    {key: None for key in object_item_keys}
                )
                != object_item_keys
                or not isinstance(entry.get("object_item_keys_consistent"), bool)
            ):
                return False
            if (object_item_keys is None) != (
                "object_item_keys_consistent" not in entry
            ):
                return False
    return _serialized_json_size(receipt) <= _MAX_TYPED_JSON_RECEIPT_BYTES


def typed_json_structure_receipt(
    *,
    artifact_path: Path,
    expected_sha256: str,
) -> dict[str, Any] | None:
    """Seal bounded structural coordinates for exact JSON bytes.

    Values are intentionally excluded. Paths and keys describe only where a
    consumer can find values in the digest-bound serialization. Object ``keys``
    already enumerate scalar children, so only the root and nested containers
    consume path-budget entries; repeating every scalar leaf would add no
    coordinate and can hide useful nested containers behind the bounded path
    limit.
    """

    artifact_path = Path(artifact_path)
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256 or "")) is None
        or artifact_path.suffix.lower() != ".json"
    ):
        return None
    try:
        if artifact_path.stat().st_size > _MAX_TYPED_JSON_ARTIFACT_BYTES:
            return None
        raw = artifact_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            return None
        payload = json.loads(raw)
    except (OSError, ValueError, TypeError):
        return None

    paths: dict[str, dict[str, Any]] = {}
    valid = True

    def visit(value: object, path: str, depth: int) -> None:
        nonlocal valid
        if not valid or len(paths) >= _MAX_TYPED_JSON_PATHS:
            valid = False
            return
        value_type = _json_value_type(value)
        entry: dict[str, Any] = {"type": value_type}
        if isinstance(value, Mapping):
            keys = _safe_json_object_keys(value)
            if keys is None:
                valid = False
                return
            entry["keys"] = keys
        elif isinstance(value, list):
            entry["length"] = len(value)
            item_types = sorted({_json_value_type(item) for item in value})
            entry["item_types"] = item_types
            object_items = [item for item in value if isinstance(item, Mapping)]
            if object_items and len(object_items) == len(value):
                key_lists = [_safe_json_object_keys(item) for item in object_items]
                if any(keys is None for keys in key_lists):
                    valid = False
                    return
                concrete_key_lists = [list(keys or []) for keys in key_lists]
                union_keys: list[str] = []
                for keys in concrete_key_lists:
                    for key in keys:
                        if key not in union_keys:
                            union_keys.append(key)
                entry["object_item_keys"] = union_keys
                entry["object_item_keys_consistent"] = all(
                    set(keys) == set(concrete_key_lists[0])
                    for keys in concrete_key_lists[1:]
                )
        paths[path] = entry
        if depth >= _MAX_TYPED_JSON_DEPTH or not isinstance(value, Mapping):
            return
        for key, child in value.items():
            if not isinstance(child, (Mapping, list)):
                continue
            visit(child, _json_pointer_child(path, str(key)), depth + 1)

    try:
        root_type = _json_value_type(payload)
        visit(payload, "", 0)
    except (TypeError, ValueError):
        return None
    if not valid:
        return None
    receipt: dict[str, Any] = {
        "json_format": "json",
        "source_sha256": expected_sha256,
        "root_type": root_type,
        "paths": paths,
    }
    return (
        receipt
        if _valid_json_structure_receipt(
            receipt,
            expected_sha256=expected_sha256,
        )
        else None
    )


def _tabular_artifact_pandas_dtypes(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any] | None:
    """Return bounded, digest-stable pandas representation facts."""

    try:
        if path.stat().st_size > _MAX_TYPED_TABLE_DTYPE_PROFILE_BYTES:
            return None
        if _sha256_file(path) != expected_sha256:
            return None
        import pandas as pd

        suffix = path.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(path, low_memory=False)
        elif suffix == ".tsv":
            frame = pd.read_csv(path, sep="\t", low_memory=False)
        elif suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        else:  # pragma: no cover - the caller keeps this capability closed
            return None
        if _sha256_file(path) != expected_sha256:
            return None
    except Exception:
        # Dtype facts are optional. Ordered columns remain useful authority
        # when a bounded standard adapter is unavailable.
        return None

    # Pandas 3 reports inferred text as ``str`` while older supported releases
    # report the same physical values as ``object``.  Keep the receipt
    # vocabulary stable across the supported dependency matrix.
    column_dtypes = {
        str(column): (
            "object"
            if pd.api.types.is_string_dtype(dtype)
            and not pd.api.types.is_numeric_dtype(dtype)
            else str(dtype)
        )
        for column, dtype in zip(frame.columns, frame.dtypes, strict=True)
    }
    numeric_columns = [
        str(column)
        for column, dtype in zip(frame.columns, frame.dtypes, strict=True)
        if pd.api.types.is_numeric_dtype(dtype)
    ]
    # Naming a column without naming what may appear in it publishes half a
    # contract: a consumer that has to select rows on that column is left to
    # invent the value. This reports the observed vocabulary for the columns
    # where a vocabulary is what the column is -- the same representation
    # fact as a dtype, bounded the same way.
    numeric_set = set(numeric_columns)
    categorical_values: dict[str, list[str]] = {}
    for column in frame.columns:
        name = str(column)
        if name in numeric_set:
            continue
        try:
            observed = frame[column].dropna().unique().tolist()
        except Exception:
            continue
        if not observed or len(observed) > _MAX_TYPED_TABLE_CATEGORY_CARDINALITY:
            continue
        rendered = [str(value) for value in observed]
        if any(
            len(value) > _MAX_TYPED_TABLE_CATEGORY_VALUE_CHARS
            or any(ord(character) < 32 for character in value)
            for value in rendered
        ):
            continue
        categorical_values[name] = sorted(set(rendered))
    profile: dict[str, Any] = {
        "column_dtypes": column_dtypes,
        "numeric_columns": numeric_columns,
    }
    if categorical_values:
        profile["categorical_values"] = categorical_values
    return profile


def _tabular_artifact_row_count(
    path: Path,
    *,
    expected_sha256: str,
) -> int | None:
    """Return the physical data-row count for exact digest-bound bytes."""

    try:
        if _sha256_file(path) != expected_sha256:
            return None
        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            import pyarrow.parquet as pq

            row_count = int(pq.ParquetFile(path).metadata.num_rows)
        elif suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = csv.reader(handle, delimiter=delimiter)
                next(rows, None)
                row_count = sum(1 for _ in rows)
        else:  # pragma: no cover - caller keeps formats closed
            return None
        if _sha256_file(path) != expected_sha256:
            return None
        return row_count
    except Exception:
        return None


def typed_product_schema_receipt(
    *,
    artifact_path: Path,
    expected_sha256: str,
) -> dict[str, Any] | None:
    """Seal a digest-verified table's bounded physical representation facts."""

    artifact_path = Path(artifact_path)
    if re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256 or "")) is None:
        return None
    if artifact_path.suffix.lower() not in {".csv", ".tsv", ".parquet", ".pq"}:
        return None
    columns = _tabular_artifact_columns(
        artifact_path,
        expected_sha256=expected_sha256,
    )
    if not columns:
        return None
    stripped_columns = [column.strip() for column in columns]
    if (
        len(columns) > _MAX_TYPED_TABLE_COLUMNS
        or any(
            not stripped
            or len(column) > _MAX_TYPED_TABLE_HEADER_CHARS
            or "\ufeff" in column
            or any(ord(character) < 32 for character in column)
            for column, stripped in zip(columns, stripped_columns, strict=True)
        )
        or len(set(columns)) != len(columns)
        or len(set(stripped_columns)) != len(stripped_columns)
    ):
        return None
    receipt: dict[str, Any] = {
        "tabular_format": artifact_path.suffix.lower().lstrip("."),
        "column_count": len(columns),
        "columns": list(columns),
    }
    row_count = _tabular_artifact_row_count(
        artifact_path,
        expected_sha256=expected_sha256,
    )
    if row_count is None:
        return None
    receipt["row_count"] = row_count
    dtype_profile = _tabular_artifact_pandas_dtypes(
        artifact_path,
        expected_sha256=expected_sha256,
    )
    if dtype_profile is not None:
        profiled_receipt = {**receipt, **dtype_profile}
        if _serialized_json_size(profiled_receipt) <= _MAX_TYPED_TABLE_RECEIPT_BYTES:
            return profiled_receipt
        # Value sets are the optional part. Dropping them must not also cost
        # the dtype facts a consumer already relied on.
        without_values = {
            key: value
            for key, value in profiled_receipt.items()
            if key != "categorical_values"
        }
        if _serialized_json_size(without_values) <= _MAX_TYPED_TABLE_RECEIPT_BYTES:
            return without_values
    return (
        receipt
        if _serialized_json_size(receipt) <= _MAX_TYPED_TABLE_RECEIPT_BYTES
        else None
    )


def merge_host_table_contract(
    producer_contract: Mapping[str, Any] | None,
    schema_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge producer coordinates with reserved host representation facts."""

    contract = dict(producer_contract or {})
    for reserved in (
        "semantic_roles",
        "semantic_roles_scope",
        "column_dtypes",
        "numeric_columns",
        "categorical_values",
        "row_count",
    ):
        contract.pop(reserved, None)
    contract.update(schema_receipt)
    contract["schema_version"] = "easyicu.host_typed_product.v4"
    return contract


def merge_host_json_contract(
    producer_contract: Mapping[str, Any] | None,
    structure_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge producer coordinates with reserved host JSON structure facts."""

    contract = dict(producer_contract or {})
    # Preserve producer-declared executable coordinates. Only this structural
    # receipt is host-reserved and must never be accepted from producer output.
    contract.pop("json_structure", None)
    contract["json_structure"] = dict(structure_receipt)
    contract["schema_version"] = "easyicu.host_typed_product.v1"
    return contract


def typed_json_structure_prompt_facts(
    contract: Mapping[str, Any],
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Return only a valid host-sealed JSON structure receipt for prompting."""

    receipt = contract.get("json_structure")
    if contract.get("schema_version") != "easyicu.host_typed_product.v1":
        return {}
    if not _valid_json_structure_receipt(
        receipt,
        expected_sha256=expected_sha256,
    ):
        return {}
    return dict(receipt)  # type: ignore[arg-type]


def typed_product_prompt_facts(
    contract: Mapping[str, Any],
    prompt_columns: Sequence[str],
) -> dict[str, Any]:
    """Return prompt-bounded facts only from a complete host v3 receipt."""

    if contract.get("schema_version") not in {
        "easyicu.host_typed_product.v3",
        "easyicu.host_typed_product.v4",
    }:
        return {}
    columns = contract.get("columns")
    column_dtypes = contract.get("column_dtypes")
    numeric_columns = contract.get("numeric_columns")
    if (
        not isinstance(columns, list)
        or not all(isinstance(value, str) for value in columns)
        or not isinstance(column_dtypes, Mapping)
        or set(column_dtypes) != set(columns)
        or not all(isinstance(value, str) for value in column_dtypes.values())
        or not isinstance(numeric_columns, list)
        or not all(isinstance(value, str) for value in numeric_columns)
        or numeric_columns != [value for value in columns if value in numeric_columns]
    ):
        return {}
    selected = [value for value in prompt_columns if value in column_dtypes]
    selected_set = set(selected)
    facts: dict[str, Any] = {
        "column_dtypes": {value: column_dtypes[value] for value in selected},
        "numeric_columns": [
            value for value in numeric_columns if value in selected_set
        ],
    }
    categorical_values = contract.get("categorical_values")
    if isinstance(categorical_values, Mapping):
        projected = {
            column: list(values)
            for column, values in categorical_values.items()
            if column in selected_set
            and isinstance(values, list)
            and all(isinstance(value, str) for value in values)
        }
        if projected:
            facts["categorical_values"] = projected
    return facts


__all__ = [
    "merge_host_json_contract",
    "merge_host_table_contract",
    "typed_json_structure_prompt_facts",
    "typed_json_structure_receipt",
    "typed_product_prompt_facts",
    "typed_product_schema_receipt",
]
