"""Bounded Coder rendering of host-verified typed representation facts.

The binding authority remains machine-readable.  This leaf module renders only
the bounded table/JSON schema, row-count, and Planner-owned consumption facts
needed by a Coder; it performs no evidence selection or scientific role
assignment.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..contracts.typed_schema import (
    typed_json_structure_prompt_facts,
    typed_product_prompt_facts,
)

_CODER_PARENT_SCHEMA_PROMPT_COLUMN_LIMIT = 32
_CODER_PARENT_SCHEMA_CONTEXT_BYTE_LIMIT = 16 * 1024


def typed_parent_schema_context_block(
    bindings: Mapping[str, Mapping[str, Any]],
) -> str:
    """Render bounded host facts about typed parent representations for Coder."""

    def render(selected: Mapping[str, Mapping[str, Any]], omitted_n: int) -> str:
        payload: dict[str, Any] = {"receipts": dict(selected)}
        if omitted_n:
            payload.update(
                {
                    "omitted_typed_parent_receipt_n": omitted_n,
                    "full_receipts_location": (
                        "EASYICU_RESOLVED_INPUTS_JSON inputs.*.product_contract "
                        "and inputs.*.consumption_contract"
                    ),
                }
            )
        return (
            "HOST-VERIFIED TYPED PARENT SCHEMAS (binding facts only):\n"
            + json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\nColumn order/names and json_structure paths/keys are physical "
            "schema facts, not scientific role assignments. JSON paths use JSON "
            "Pointer syntax; the empty path is the document root. Each `paths` "
            "mapping key is the JSON Pointer to load from the document; its mapping "
            "value is that location's type/shape descriptor and is not itself a "
            "pointer. Object `keys` also enumerate scalar children even when those "
            "leaves have no separate `paths` entry. "
            "object_item_keys list keys observed across an object array, while "
            "object_item_keys_consistent states whether every item has that key "
            "set. No JSON data values are promoted into this receipt. "
            "column_dtypes/numeric_columns, when present, are "
            "host-observed pandas representation facts for the exact artifact, not "
            "scientific roles. Choose columns only inside the Planner-declared typed "
            "product using the Planner-owned method and scientific context. Do not "
            "use first-numeric, dtype-order, or nonexistent-column fallbacks; fail "
            "closed when the schema cannot support the declared product. A present "
            "consumption_contract is mandatory: all_rows means preserve every row, "
            "single_row is valid only for the verified singleton, and one_per_role "
            "requires every declared role exactly once."
        )

    receipts: dict[str, dict[str, Any]] = {}
    omitted_n = 0
    for input_key in sorted(bindings):
        binding = bindings[input_key]
        contract = binding.get("product_contract")
        if not isinstance(contract, Mapping):
            continue
        json_structure = typed_json_structure_prompt_facts(
            contract,
            expected_sha256=str(binding.get("sha256") or ""),
        )
        if json_structure:
            receipt = {"json_structure": json_structure}
            consumption_contract = binding.get("consumption_contract")
            if isinstance(consumption_contract, Mapping):
                receipt["consumption_contract"] = dict(consumption_contract)
            candidate = {**receipts, input_key: receipt}
            if (
                len(render(candidate, omitted_n).encode("utf-8"))
                > _CODER_PARENT_SCHEMA_CONTEXT_BYTE_LIMIT
            ):
                omitted_n += 1
                continue
            receipts[input_key] = receipt
            continue
        columns = contract.get("columns")
        column_count = contract.get("column_count")
        row_count = contract.get("row_count")
        schema_version = contract.get("schema_version")
        tabular_format = contract.get("tabular_format")
        if not isinstance(columns, list) or any(
            not isinstance(value, str) for value in columns
        ):
            continue
        if (
            isinstance(column_count, bool)
            or not isinstance(column_count, int)
            or column_count != len(columns)
            or (
                schema_version == "easyicu.host_typed_product.v4"
                and (
                    isinstance(row_count, bool)
                    or not isinstance(row_count, int)
                    or row_count < 0
                )
            )
            or not isinstance(tabular_format, str)
            or not tabular_format.strip()
        ):
            continue
        prompt_columns = list(columns[:_CODER_PARENT_SCHEMA_PROMPT_COLUMN_LIMIT])
        receipt: dict[str, Any] = {
            "tabular_format": tabular_format,
            "column_count": column_count,
            "columns": prompt_columns,
        }
        if isinstance(row_count, int) and not isinstance(row_count, bool):
            receipt["row_count"] = row_count
        consumption_contract = binding.get("consumption_contract")
        if isinstance(consumption_contract, Mapping):
            receipt["consumption_contract"] = dict(consumption_contract)
        receipt.update(typed_product_prompt_facts(contract, prompt_columns))
        if len(prompt_columns) != len(columns):
            receipt["columns_omitted_from_prompt_n"] = len(columns) - len(
                prompt_columns
            )
            receipt["full_schema_location"] = (
                "EASYICU_RESOLVED_INPUTS_JSON product_contract.columns"
            )
        candidate = {**receipts, input_key: receipt}
        if (
            len(render(candidate, omitted_n).encode("utf-8"))
            > _CODER_PARENT_SCHEMA_CONTEXT_BYTE_LIMIT
        ):
            omitted_n += 1
            continue
        receipts[input_key] = receipt
    if not receipts and not omitted_n:
        return ""
    block = render(receipts, omitted_n)
    while (
        len(block.encode("utf-8")) > _CODER_PARENT_SCHEMA_CONTEXT_BYTE_LIMIT
        and receipts
    ):
        receipts.popitem()
        omitted_n += 1
        block = render(receipts, omitted_n)
    if len(block.encode("utf-8")) > _CODER_PARENT_SCHEMA_CONTEXT_BYTE_LIMIT:
        return (
            "HOST-VERIFIED TYPED PARENT SCHEMAS: prompt receipt omitted "
            "because it exceeded the transport limit. Load exact product contracts "
            "from EASYICU_RESOLVED_INPUTS_JSON; do not guess columns."
        )
    return block


__all__ = ["typed_parent_schema_context_block"]
