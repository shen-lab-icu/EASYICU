"""Deterministic adapters for digest-bound non-tabular typed products."""

from __future__ import annotations

import ast

_JSON_LOADER_ANCHOR = """    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported tabular input format: {path}")
"""

_JSON_LOADER_REPLACEMENT = """    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Bound JSON input must contain an object: {path}")
        return pd.DataFrame([payload])
    raise ValueError(f"Unsupported tabular input format: {path}")
"""

_SCHEMA_ANCHOR = """    expected_columns = contract.get("columns")
    if not isinstance(expected_columns, list) or not expected_columns:
        raise ValueError(f"Missing product_contract.columns for {expected_key}")
"""

_SCHEMA_REPLACEMENT = """    expected_columns = contract.get("columns")
    if (
        (not isinstance(expected_columns, list) or not expected_columns)
        and path.suffix.lower() == ".json"
        and contract.get("schema_version") == "easyicu.host_typed_product.v1"
    ):
        expected_columns = list(table.columns)
    elif not isinstance(expected_columns, list) or not expected_columns:
        raise ValueError(f"Missing product_contract.columns for {expected_key}")
"""


def patch_resolved_json_document_adapter(code: str, run_log: str) -> str:
    """Adapt one proven v1 JSON product without weakening table contracts.

    The repair activates only after the runner reports that a digest-bound
    ``.json`` artifact reached the generated tabular loader.  It preserves the
    existing CSV/Parquet paths and permits a JSON *object* only when the host
    product contract is the non-tabular v1 shape.  Newer tabular contracts
    still require their exact declared columns.
    """

    lowered = (run_log or "").lower()
    if "unsupported tabular input format:" not in lowered or ".json" not in lowered:
        return code
    if code.count(_JSON_LOADER_ANCHOR) != 1 or code.count(_SCHEMA_ANCHOR) != 1:
        return code

    repaired = code.replace(
        _JSON_LOADER_ANCHOR,
        _JSON_LOADER_REPLACEMENT,
        1,
    ).replace(
        _SCHEMA_ANCHOR,
        _SCHEMA_REPLACEMENT,
        1,
    )
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_resolved_json_document_adapter"]
