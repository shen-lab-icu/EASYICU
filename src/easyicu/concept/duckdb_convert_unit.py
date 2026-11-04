"""SQL value expressions for ricu ``convert_unit`` callbacks on the DuckDB id_tbl path.

``convert_unit(fun, new, rgx)`` applies ``fun`` to the rows whose unit matches
``rgx``; without ``rgx`` it applies ``fun`` to every row (ricu semantics). The
id_tbl fast path aggregates in SQL, so the callback must be expressed here or
the source must fall back to the pandas path.
"""

from __future__ import annotations

import re
from typing import List, Optional

_BINARY_OP = re.compile(r"binary_op\(`(.+?)`,\s*(.+)\)")


def split_callback_args(text: str) -> List[str]:
    """Split top-level comma-separated arguments, keeping nested calls intact."""
    args, level, current = [], 0, []
    for char in text:
        if char == "(":
            level += 1
        elif char == ")":
            level -= 1
        elif char == "," and level == 0:
            args.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        args.append("".join(current).strip())
    return args


def convert_unit_value_sql(callback: str, value_column: str, unit_column: Optional[str]) -> Optional[str]:
    """Return the SQL value expression for a ``convert_unit`` callback, or ``None``.

    ``None`` means the callback cannot be expressed here; the caller must not mark
    it as applied.
    """
    match = re.match(r"convert_unit\((.+)\)", callback.strip(), re.DOTALL)
    if not match:
        return None
    args = split_callback_args(match.group(1))
    if not args:
        return None
    raw = f'TRY_CAST("{value_column}" AS DOUBLE)'
    first = args[0].strip()
    binary = _BINARY_OP.match(first)
    if len(args) == 2:
        # No unit regex: every row is converted (2026-11-04 audit round 3,
        # R3-01: MIIV/MIMIC-III 226531 weight in lbs).
        if binary:
            return f"({raw} {binary.group(1)} {binary.group(2)})"
        if first == "set_val(NA)":
            return "CAST(NULL AS DOUBLE)"
        if first == "fahr_to_cels":
            return f"(({raw} - 32.0) * 5.0 / 9.0)"
        return None
    unit_filter = args[2].strip().strip("'\"") if len(args) > 2 else None
    if not unit_filter or not unit_column:
        return None
    matches = f"REGEXP_MATCHES(LOWER(\"{unit_column}\"), '{unit_filter}')"
    if binary:
        converted = f"{raw} {binary.group(1)} {binary.group(2)}"
    elif first == "set_val(NA)":
        converted = "NULL"
    elif first == "fahr_to_cels":
        converted = f"({raw} - 32.0) * 5.0 / 9.0"
    else:
        return None
    return f"CASE WHEN {matches} THEN {converted} ELSE {raw} END"
