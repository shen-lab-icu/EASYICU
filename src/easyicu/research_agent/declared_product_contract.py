"""Execution-time boundary between planned and produced step products.

The planner owns scientific scope through typed ``kind:product`` entries in
``AnalysisStep.expected_outputs``.  A successful script must realise those
products in its machine-readable summary, and it may not silently widen a
non-figure/non-effect step into a publication figure or effect analysis.

This module only validates declarations and registrations.  It never chooses
an exposure, outcome, cohort, estimator, or analysis method.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schema import AnalysisStep, ValidationFinding
from .trajectory_plan_contract import (
    trajectory_role_result_findings,
    trajectory_role_scope_summary_findings,
)


_FIGURE_KINDS = frozenset({"figure", "plot", "chart", "fig", "heatmap"})
_FAILED_STATUSES = frozenset(
    {
        "blocked",
        "error",
        "failed",
        "execution_failed",
        "contract_failed",
        "fail_closed",
        "failed_closed",
        "repair_failed",
        "skipped_dependency_failed",
    }
)
_OUTPUT_CONTAINER_KEYS = frozenset({"output_files", "outputs", "figure_files"})
_DIRECT_FIGURE_KEYS = frozenset({"figure_file", "figure_path"})
_FIGURE_SUFFIXES = frozenset({".png", ".svg", ".pdf", ".tif", ".tiff"})
_KNOWN_FILE_SUFFIXES = frozenset(
    {
        *_FIGURE_SUFFIXES,
        ".csv",
        ".tsv",
        ".parquet",
        ".feather",
        ".json",
        ".jsonl",
        ".md",
        ".txt",
        ".log",
        ".pkl",
        ".pickle",
        ".joblib",
        ".npy",
        ".npz",
    }
)
_EFFECT_PRODUCT_BASES = frozenset(
    {
        "adjusted_effect",
        "adjusted_effect_estimate",
        "adjusted_effect_estimates",
        "adjusted_association_estimate",
        "adjusted_association_estimates",
        "adjusted_odds_ratio",
        "adjusted_odds_ratios",
        "adjusted_or",
        "association_estimate",
        "association_estimates",
        "causal_effect",
        "coefficient",
        "coefficients",
        "hazard_ratio",
        "interaction_pvalue",
        "odds_ratio",
        "overall_effect",
        "primary_association",
        "primary_association_estimate",
        "primary_effect",
        "primary_hr",
        "primary_or",
        "risk_difference",
        "risk_ratio",
        "subgroup_effect",
        "subgroup_effects",
        "treatment_effect",
    }
)


def _normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip(
        "_"
    )


def _canonical_kind(value: object) -> str:
    kind = _normalise(value)
    if kind in _FIGURE_KINDS:
        return "figure"
    if kind in {"metric", "statistics"}:
        return "statistic"
    return kind


def typed_product(value: object) -> tuple[str, str] | None:
    """Return the shared canonical identity for a ``kind:product`` token."""

    kind, separator, product = str(value or "").strip().partition(":")
    if not separator:
        return None
    canonical_kind = _canonical_kind(kind)
    product_name = _normalise(Path(product).name)
    for suffix in sorted(_KNOWN_FILE_SUFFIXES, key=len, reverse=True):
        normalised_suffix = _normalise(suffix)
        if product_name.endswith(f"_{normalised_suffix}"):
            product_name = product_name[: -(len(normalised_suffix) + 1)]
            break
    if not canonical_kind or not product_name:
        return None
    return canonical_kind, product_name


# Internal compatibility alias; lineage and product-scope validation must share
# the public parser above rather than growing independent token grammars.
_typed_product = typed_product


def _file_stem(value: object) -> str:
    name = Path(str(value or "").strip()).name
    suffix = Path(name).suffix.lower()
    if suffix in _KNOWN_FILE_SUFFIXES:
        name = name[: -len(suffix)]
    return _normalise(name)


def _file_kinds(value: object) -> frozenset[str]:
    suffix = Path(str(value or "").strip()).suffix.lower()
    if suffix in _FIGURE_SUFFIXES:
        return frozenset({"figure"})
    if suffix in {".csv", ".tsv"}:
        return frozenset({"table", "artifact", "dataset", "test"})
    if suffix in {".parquet", ".feather"}:
        return frozenset({"artifact", "dataset", "table"})
    if suffix in {".pkl", ".pickle", ".joblib"}:
        return frozenset({"model", "artifact"})
    if suffix in {".npy", ".npz"}:
        return frozenset({"artifact", "dataset", "model"})
    if suffix in {".md", ".txt", ".log", ".jsonl"}:
        return frozenset({"log", "artifact"})
    if suffix == ".json":
        return frozenset({"artifact", "manifest", "log", "model", "test"})
    return frozenset()


def _is_file_path(value: object) -> bool:
    return isinstance(value, str) and Path(value.strip()).suffix.lower() in (
        _KNOWN_FILE_SUFFIXES
    )


def _iter_paths(value: Any) -> Iterable[str]:
    if _is_file_path(value):
        yield str(value).strip()
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_paths(child)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _iter_paths(child)


def _summary_scalar_products(value: Any) -> set[tuple[str, str]]:
    """Return exact statistic/log keys backed by non-null scalar values."""

    products: set[tuple[str, str]] = set()

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for raw_key, child in node.items():
                key = _normalise(raw_key)
                if isinstance(child, Mapping) or isinstance(child, (list, tuple)):
                    visit(child)
                    continue
                valid = child is not None and child != ""
                if isinstance(child, float):
                    valid = math.isfinite(child)
                if valid and key:
                    products.add(("statistic", key))
                    products.add(("log", key))
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(value)
    return products


def _registered_products(
    summary: Mapping[str, Any],
    *,
    out_dir: Path | None = None,
) -> tuple[set[tuple[str, str]], list[tuple[str, bool]]]:
    """Collect typed/file products and figure paths from output containers."""

    products: set[tuple[str, str]] = set()
    figure_paths: list[tuple[str, bool]] = []

    def is_actual_output(path: str) -> bool:
        if out_dir is None:
            return True
        root = out_dir.resolve()
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (FileNotFoundError, OSError, ValueError):
            return False
        return resolved.is_file()

    def add_path(path: str, *, explicit_figure_list: bool = False) -> None:
        kinds = _file_kinds(path)
        stem = _file_stem(path)
        products.update((kind, stem) for kind in kinds if stem)
        if "figure" in kinds:
            figure_paths.append((path, explicit_figure_list))

    def add_container(value: Any, *, explicit_figure_list: bool = False) -> None:
        if isinstance(value, Mapping):
            for raw_role, child in value.items():
                role = _typed_product(raw_role)
                paths = [path for path in _iter_paths(child) if is_actual_output(path)]
                if role is not None:
                    role_kind, role_name = role
                    compatible_path = any(
                        role_kind in _file_kinds(path) for path in paths
                    )
                    scalar_registration = (
                        role_kind in {"statistic", "log"}
                        and not isinstance(child, (Mapping, list, tuple, set))
                        and child is not None
                        and child != ""
                    )
                    if compatible_path or scalar_registration:
                        products.add(role)
                elif paths:
                    role_name = _normalise(raw_role)
                    for path in paths:
                        products.update(
                            (kind, role_name) for kind in _file_kinds(path) if role_name
                        )
                for path in paths:
                    add_path(path, explicit_figure_list=explicit_figure_list)
            return
        for path in _iter_paths(value):
            add_path(path, explicit_figure_list=explicit_figure_list)

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for raw_key, child in node.items():
                key = _normalise(raw_key)
                if key in _OUTPUT_CONTAINER_KEYS:
                    add_container(child, explicit_figure_list=key == "figure_files")
                elif key in _DIRECT_FIGURE_KEYS:
                    add_container(child, explicit_figure_list=True)
                if isinstance(child, (Mapping, list, tuple)):
                    visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(summary)
    products.update(_summary_scalar_products(summary))
    return products, figure_paths


def _has_product_registry(value: Any) -> bool:
    """Whether a summary opted into the machine-readable output registry."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if _normalise(raw_key) in {"output_files", "outputs"}:
                return True
            if isinstance(child, (Mapping, list, tuple)) and _has_product_registry(
                child
            ):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_has_product_registry(child) for child in value)
    return False


def _effect_bearing_name(name: str) -> bool:
    normalised = _normalise(name)
    return any(
        normalised == base or normalised.startswith(f"{base}_")
        for base in _EFFECT_PRODUCT_BASES
    )


def _effect_summary_paths(summary: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []

    def visit(node: Any, prefix: str = "") -> None:
        if isinstance(node, Mapping):
            for raw_key, child in node.items():
                key = _normalise(raw_key)
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(child, Mapping) or isinstance(child, (list, tuple)):
                    visit(child, path)
                elif (
                    key
                    and _effect_bearing_name(key)
                    and child is not None
                    and child != ""
                ):
                    paths.append(path)
        elif isinstance(node, (list, tuple)):
            for index, child in enumerate(node):
                visit(child, f"{prefix}[{index}]")

    visit(summary)
    return sorted(set(paths))


def _undeclared_figure_bundle(
    figure_paths: Sequence[tuple[str, bool]],
) -> dict[str, list[str]]:
    by_stem: dict[str, set[str]] = defaultdict(set)
    explicit_stems: set[str] = set()
    for path, explicit in figure_paths:
        stem = _file_stem(path)
        if not stem:
            continue
        by_stem[stem].add(Path(path).suffix.lower())
        if explicit:
            explicit_stems.add(stem)
    return {
        stem: sorted(suffixes)
        for stem, suffixes in by_stem.items()
        if len(suffixes) >= 2 or stem in explicit_stems
    }


def declared_product_contract_findings(
    *,
    step: AnalysisStep,
    step_summary: Mapping[str, Any],
    effect_method_authorized: bool,
    out_dir: Path | None = None,
) -> list[ValidationFinding]:
    """Validate declared-product realization and scientific output scope."""

    reported_status = _normalise(step_summary.get("status"))
    if reported_status in _FAILED_STATUSES:
        return []

    declared = {
        product
        for raw in (step.expected_outputs or [])
        if (product := _typed_product(raw)) is not None
    }
    registered, figure_paths = _registered_products(step_summary, out_dir=out_dir)
    findings: list[ValidationFinding] = []
    findings.extend(
        trajectory_role_scope_summary_findings(
            step=step,
            step_summary=step_summary,
        )
    )
    findings.extend(
        trajectory_role_result_findings(
            step=step,
            step_summary=step_summary,
        )
    )

    # Older direct unit fixtures predate ``output_files`` and validate only
    # their own numeric payload.  A real execution supplies ``out_dir`` and is
    # always held to the product boundary, even if its script tries to evade the
    # gate by omitting the modern registry entirely.
    missing = (
        sorted(declared - registered)
        if out_dir is not None or _has_product_registry(step_summary)
        else []
    )
    if missing:
        findings.append(
            ValidationFinding(
                validator="declared_product_contract",
                severity="error",
                message=(
                    f"Step {step.step_id} did not realise every typed product "
                    "declared by the plan in step_summary output registrations."
                ),
                detail={
                    "kind": "declared_product_missing",
                    "step_id": step.step_id,
                    "missing_products": [f"{kind}:{name}" for kind, name in missing],
                    "declared_products": [
                        f"{kind}:{name}" for kind, name in sorted(declared)
                    ],
                    "registered_products": [
                        f"{kind}:{name}" for kind, name in sorted(registered)
                    ],
                },
            )
        )

    declares_figure = any(kind == "figure" for kind, _name in declared)
    figure_bundle = _undeclared_figure_bundle(figure_paths)
    if figure_bundle and not declares_figure:
        findings.append(
            ValidationFinding(
                validator="declared_product_contract",
                severity="error",
                message=(
                    f"Step {step.step_id} produced a figure bundle without a "
                    "typed figure product in expected_outputs. Figure rendering "
                    "must remain in its declared figure owner step."
                ),
                detail={
                    "kind": "undeclared_figure_bundle",
                    "step_id": step.step_id,
                    "figure_bundle": figure_bundle,
                },
            )
        )

    if not effect_method_authorized:
        declared_effects = sorted(
            f"{kind}:{name}"
            for kind, name in declared
            if _effect_bearing_name(name)
        )
        registered_effects = sorted(
            f"{kind}:{name}"
            for kind, name in registered
            if kind != "log" and _effect_bearing_name(name)
        )
        summary_effects = _effect_summary_paths(step_summary)
        if declared_effects or registered_effects or summary_effects:
            findings.append(
                ValidationFinding(
                    validator="declared_product_contract",
                    severity="error",
                    message=(
                        f"Step {step.step_id} uses a non-effect method but "
                        "declared or registered effect-bearing scientific output. "
                        "Move effect estimation to an agent-planned effect-method owner."
                    ),
                    detail={
                        "kind": "unauthorized_effect_product",
                        "step_id": step.step_id,
                        "planned_method": step.method,
                        "declared_effect_products": declared_effects,
                        "registered_effect_products": registered_effects,
                        "summary_effect_paths": summary_effects,
                    },
                )
            )

    return findings


__all__ = ["declared_product_contract_findings"]
