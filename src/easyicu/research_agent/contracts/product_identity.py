"""Canonical identity grammar for typed scientific products."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

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


def normalised_expected_output_names(
    expected_outputs: Sequence[str] | str,
) -> set[str]:
    """Return representation-agnostic names from declared outputs."""

    values = (
        re.split(r"[\s,]+", expected_outputs)
        if isinstance(expected_outputs, str)
        else [str(value or "") for value in (expected_outputs or [])]
    )
    names: set[str] = set()
    for raw in values:
        value = str(raw or "").strip().lower()
        if not value:
            continue
        name = value.split(":", 1)[-1].rsplit("/", 1)[-1]
        names.add(
            re.sub(r"\.(?:csv|json|parquet|png|svg|pdf|tiff?)$", "", name)
        )
    return names


_STRUCTURED_CONTRACT_OUTPUT_KINDS = frozenset(
    {"", "statistic", "table", "model", "manifest", "dataset", "artifact"}
)


def normalised_structured_output_names(
    expected_outputs: Sequence[str] | str,
) -> set[str]:
    """Return names only for outputs that carry structured plan authority."""

    values = (
        re.split(r"[\s,]+", expected_outputs)
        if isinstance(expected_outputs, str)
        else [str(value or "") for value in (expected_outputs or [])]
    )
    names: set[str] = set()
    for raw in values:
        value = str(raw or "").strip().lower()
        if not value:
            continue
        parsed = typed_product(value)
        if parsed is not None and parsed[0] in _STRUCTURED_CONTRACT_OUTPUT_KINDS:
            names.add(parsed[1])
            continue
        kind, separator, product = value.partition(":")
        if separator and kind not in _STRUCTURED_CONTRACT_OUTPUT_KINDS:
            continue
        name = (product if separator else kind).rsplit("/", 1)[-1]
        names.add(re.sub(r"\.(?:csv|json|parquet)$", "", name))
    return names


def normalised_method_head(method: str) -> str:
    """Return the normalized method head before an optional ``with`` rider."""

    normalized = normalize_product_token(method)
    return normalized.split("_with_", 1)[0]


__all__ = [
    "CANONICAL_TYPED_PRODUCT_TOKEN_PATTERN",
    "FIGURE_KIND_ALIASES",
    "canonical_product_kind",
    "is_canonical_typed_product_token",
    "normalize_product_token",
    "normalised_expected_output_names",
    "normalised_method_head",
    "normalised_structured_output_names",
    "typed_product",
]
