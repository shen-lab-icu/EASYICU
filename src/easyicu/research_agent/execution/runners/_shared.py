"""Shared small helpers for deterministic execution runners (Owner: execution).

This module is a pure relocation target for the tiny predicates and JSON
helpers that were copy-pasted across runner modules (``_method_head``,
``_figure_product``, ``_typed_product``, ``_report_product``,
``_is_safe_figure_product_id``, ``_read_json``, ``_write_json``).  Runners
import these names (usually aliased back to their historical private spelling)
instead of redefining them, so ownership-check semantics stay identical in one
place.

Unification notes (fail-closed direction only):

* ``method_head`` uses ``casefold`` (the strictest historical spelling;
  identical to ``lower`` for ASCII method names).
* ``figure_product`` enforces the bounded ``[a-z][a-z0-9_]{0,127}`` family
  already used at the render boundary.  Runners that previously accepted
  unbounded ids now fail closed at ownership for pathological >127-char ids
  instead of later at render.
* ``typed_product_value`` keeps the generic unbounded spelling for non-figure
  kinds (``table``/``statistic``/``report`` inputs); it is the single owner.
* ``write_json`` is canonical sorted-keys UTF-8 with ``default=str`` tolerance
  (union of the historical variants).

``gates/preflight.py`` is intentionally NOT split by this change.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping

#: Canonical figure product-id shape (bounded; enforced at the render boundary).
FIGURE_PRODUCT_ID = re.compile(r"[a-z][a-z0-9_]{0,127}")

#: Canonical ``kind:product`` token shape for generic inputs.
TYPED_KEY = re.compile(r"([a-z][a-z0-9_]*):([a-z][a-z0-9_]*)")


def method_head(value: Any) -> str:
    """Return the method family head (text before ``" with "``)."""

    return str(value or "").strip().casefold().split(" with ", 1)[0]


def figure_product(value: Any) -> str | None:
    """Return the figure product id for a ``figure:<id>`` token, else ``None``."""

    kind, separator, product = str(value or "").strip().partition(":")
    if kind != "figure" or not separator or not FIGURE_PRODUCT_ID.fullmatch(product):
        return None
    return product


def typed_product_value(value: Any, expected_kind: str) -> str | None:
    """Return the product id for a ``<expected_kind>:<id>`` token, else ``None``."""

    kind, separator, product = str(value or "").strip().partition(":")
    if (
        kind != expected_kind
        or not separator
        or not re.fullmatch(r"[a-z][a-z0-9_]*", product)
    ):
        return None
    return product


def report_product(step: Any) -> str | None:
    """Return the single ``report:<id>`` product a reporting step declares."""

    if len(step.expected_outputs or ()) != 1:
        return None
    match = TYPED_KEY.fullmatch(str(step.expected_outputs[0] or "").strip())
    if match is None or match.group(1) != "report":
        return None
    return match.group(2)


def is_safe_figure_product_id(value: Any) -> bool:
    """Return whether ``value`` is a safe figure product id."""

    return bool(FIGURE_PRODUCT_ID.fullmatch(str(value or "")))


def read_json_object(path: Path) -> Dict[str, Any]:
    """Read a JSON object, returning ``{}`` when unreadable or not an object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write ``value`` as sorted-keys UTF-8 JSON (``default=str`` tolerant)."""

    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


__all__ = [
    "FIGURE_PRODUCT_ID",
    "TYPED_KEY",
    "figure_product",
    "is_safe_figure_product_id",
    "method_head",
    "read_json_object",
    "report_product",
    "typed_product_value",
    "write_json",
]
