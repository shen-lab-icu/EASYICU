"""Canonical identity grammar for typed scientific products."""

from __future__ import annotations

import re
from pathlib import Path

from .product_files import KNOWN_FILE_SUFFIXES


FIGURE_KIND_ALIASES = frozenset({"figure", "plot", "chart", "fig", "heatmap"})
CANONICAL_TYPED_PRODUCT_TOKEN_PATTERN = (
    r"^[a-z][a-z0-9_]*:[a-z][a-z0-9_]*$"
)


def is_canonical_typed_product_token(value: object) -> bool:
    """Return whether *value* already uses the exact wire identity grammar."""

    return isinstance(value, str) and re.fullmatch(
        CANONICAL_TYPED_PRODUCT_TOKEN_PATTERN, value
    ) is not None


def normalize_product_token(value: object) -> str:
    """Normalize one user/model-authored identity token without inferring scope."""

    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def canonical_product_kind(value: object) -> str:
    """Canonicalize representation aliases to one physical product family."""

    kind = normalize_product_token(value)
    if kind in FIGURE_KIND_ALIASES:
        return "figure"
    if kind == "cohort":
        return "dataset"
    if kind in {"metric", "statistics"}:
        return "statistic"
    return kind


def typed_product(value: object) -> tuple[str, str] | None:
    """Return the canonical identity for one exact ``kind:product`` token."""

    kind, separator, product = str(value or "").strip().partition(":")
    if not separator:
        return None
    canonical_kind = canonical_product_kind(kind)
    product_name = normalize_product_token(Path(product).name)
    for suffix in sorted(KNOWN_FILE_SUFFIXES, key=len, reverse=True):
        normalized_suffix = normalize_product_token(suffix)
        if product_name.endswith(f"_{normalized_suffix}"):
            product_name = product_name[: -(len(normalized_suffix) + 1)]
            break
    if not canonical_kind or not product_name:
        return None
    return canonical_kind, product_name


__all__ = [
    "CANONICAL_TYPED_PRODUCT_TOKEN_PATTERN",
    "FIGURE_KIND_ALIASES",
    "canonical_product_kind",
    "is_canonical_typed_product_token",
    "normalize_product_token",
    "typed_product",
]
