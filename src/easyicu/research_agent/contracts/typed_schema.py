"""Host-observed physical schema receipts for typed tabular products.

This module reports representation facts for one digest-bound table. It never
assigns an exposure, outcome, cohort, estimator, estimand, or semantic column
role. The legacy public import remains re-exported by ``declared_product``.
"""

from __future__ import annotations

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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    "merge_host_table_contract",
    "typed_product_prompt_facts",
    "typed_product_schema_receipt",
]
