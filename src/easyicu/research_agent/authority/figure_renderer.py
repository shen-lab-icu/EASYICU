"""Digest-bound direct-parent authority for sealed figure renderers.

This module owns structural joins between Planner-declared parent products,
registered evidence digests, and closed renderer schemas. It selects no
scientific method and imports neither :mod:`pipeline` nor the execute phase.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional

from ..contracts.declared_product import (
    read_digest_bound_artifact_snapshot,
    typed_product,
)
from ..figures.ordered_distribution import (
    ordered_distribution_availability_snapshot_is_valid,
)
from ..repair_registry import is_sealed_renderer_repair, repair_metadata_for
from ..schema import AnalysisStep
from .parent_artifact import (
    _resolve_upstream_manifest_step,
    _verified_direct_parent_artifact_digests,
)


def _ordered_distribution_availability_parent_digest_seal(
    run_dir: Path,
    figure_step_id: str,
) -> Optional[dict[str, str]]:
    """Seal one typed distribution table and one typed availability table.

    Planner product roles select the two logical inputs, the digest-bound
    parent summary maps those roles to physical files, and the renderer schema
    validates ordinal levels, counts, percentages, and denominators before the
    sealed adapter is authorized. A model-authored method string is never a
    routing input.
    """

    digests = _verified_direct_parent_artifact_digests(run_dir, figure_step_id)
    request_step = _resolve_upstream_manifest_step(run_dir, figure_step_id)
    if (
        not digests
        or "step_summary.json" not in digests
        or not isinstance(request_step, Mapping)
    ):
        return None
    products = [
        parsed
        for raw in (request_step.get("expected_outputs") or [])
        if (parsed := typed_product(raw)) is not None
        and parsed[0] in {"table", "artifact", "dataset"}
    ]
    distribution = [
        product for product in products if product[1].endswith("_distribution")
    ]
    availability = [
        product
        for product in products
        if product[1] == "availability" or product[1].endswith("_availability")
    ]
    if len(distribution) != 1 or len(availability) != 1:
        return None

    parent_out = (
        Path(run_dir)
        / "steps"
        / str(figure_step_id).removesuffix("_figure")
        / "outputs"
    )
    try:
        summary_snapshot = read_digest_bound_artifact_snapshot(
            parent_out=parent_out,
            artifact_digests={
                "step_summary.json": digests["step_summary.json"],
            },
        )
        summary = json.loads(summary_snapshot["step_summary.json"].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    output_files = summary.get("output_files") if isinstance(summary, Mapping) else None
    if not isinstance(output_files, Mapping):
        return None

    selected_names: list[str] = []
    for product in (distribution[0], availability[0]):
        raw_name = output_files.get(f"{product[0]}:{product[1]}")
        name = str(raw_name or "").strip()
        if (
            not name
            or Path(name).name != name
            or Path(name).suffix.lower() != ".csv"
            or name not in digests
        ):
            return None
        selected_names.append(name)
    if len(set(selected_names)) != 2:
        return None

    sealed = {
        name: digests[name] for name in sorted({"step_summary.json", *selected_names})
    }
    try:
        snapshot = read_digest_bound_artifact_snapshot(
            parent_out=parent_out,
            artifact_digests=sealed,
        )
    except ValueError:
        return None
    if not ordered_distribution_availability_snapshot_is_valid(snapshot):
        return None
    return sealed


def _sealed_renderer_figure_step_matches_parent(
    run_dir: Path,
    step: AnalysisStep,
    renderer_repair_id: str,
) -> bool:
    """Require a Planner-owned structural edge for one sealed renderer.

    Modern split steps consume exact logical parent products registered for the
    renderer. Legacy split steps repeat the parent method and inputs exactly.
    A ``*_figure`` sibling name alone is never authority.
    """

    if not is_sealed_renderer_repair(renderer_repair_id):
        return False
    metadata = repair_metadata_for(renderer_repair_id)
    request_step = _resolve_upstream_manifest_step(run_dir, step.step_id)
    if not isinstance(request_step, Mapping):
        return False
    planner_method = str(request_step.get("method") or "").strip().lower()
    if (
        metadata.planner_method_required
        and planner_method not in metadata.planner_methods
    ):
        return False

    parent_products = {
        parsed
        for raw in (request_step.get("expected_outputs") or [])
        if (parsed := typed_product(raw)) is not None
        and parsed[0] in {"table", "artifact", "dataset"}
    }
    required_products: set[tuple[str, str]] = set()
    for role_alternatives in metadata.planner_parent_output_role_groups:
        matches = {
            product
            for product in parent_products
            if any(
                len(product[1].split("_")) >= len(suffix)
                and tuple(product[1].split("_")[-len(suffix) :]) == tuple(suffix)
                for suffix in role_alternatives
            )
        }
        if not matches:
            return False
        required_products.update(matches)

    child_typed_inputs = {
        parsed
        for raw in (step.inputs or [])
        if (parsed := typed_product(raw)) is not None
    }
    if required_products <= child_typed_inputs:
        return True
    return str(step.method or "").strip().lower() == planner_method and tuple(
        str(value) for value in (step.inputs or [])
    ) == tuple(str(value) for value in (request_step.get("inputs") or []))


__all__ = [
    "_ordered_distribution_availability_parent_digest_seal",
    "_sealed_renderer_figure_step_matches_parent",
]
