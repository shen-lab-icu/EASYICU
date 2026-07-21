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

    column_dtypes = {
        str(column): str(dtype)
        for column, dtype in zip(frame.columns, frame.dtypes, strict=True)
    }
    numeric_columns = [
        str(column)
        for column, dtype in zip(frame.columns, frame.dtypes, strict=True)
        if pd.api.types.is_numeric_dtype(dtype)
    ]
    return {
        "column_dtypes": column_dtypes,
        "numeric_columns": numeric_columns,
    }


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
    return {
        "column_dtypes": {value: column_dtypes[value] for value in selected},
        "numeric_columns": [
            value for value in numeric_columns if value in selected_set
        ],
    }


__all__ = [
    "merge_host_table_contract",
    "typed_product_prompt_facts",
    "typed_product_schema_receipt",
]
