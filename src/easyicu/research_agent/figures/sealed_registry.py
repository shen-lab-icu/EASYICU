"""Closed registry for sealed, cross-file deterministic figure renderers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

from ..authority.figure_renderer import (
    _ordered_distribution_availability_parent_digest_seal,
)
from .continuous_measurement_audit import (
    _continuous_measurement_audit_parent_digest_seal,
    render_continuous_measurement_audit_bundle,
)
from .distribution_availability import (
    _distribution_availability_parent_digest_seal,
    render_distribution_availability_bundle_from_prior_outputs,
)
from .missingness_source import (
    REPAIR_ID as MISSINGNESS_REPAIR_ID,
    missingness_source_parent_digest_seal,
    render_missingness_source_bundle,
)
from .ordered_distribution import (
    render_ordered_distribution_bundle_from_prior_outputs,
)

Seal = Callable[[Path, str], Optional[dict[str, str]]]
Render = Callable[[Path, str, Path, Mapping[str, bytes]], Optional[str]]

_ORDERED_DISTRIBUTION_AVAILABILITY_V2 = (
    "ordered_category_distribution_availability_publication_bundle_v2"
)


@dataclass(frozen=True)
class SealedRendererAdapter:
    seal: Seal
    render: Render


def _render_missingness(
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    snapshot: Mapping[str, bytes],
) -> Optional[str]:
    return render_missingness_source_bundle(
        run_dir=run_dir,
        current_step_id=current_step_id,
        out_dir=out_dir,
        preverified_parent_artifacts=snapshot,
    )


def _render_distribution_availability(
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    snapshot: Mapping[str, bytes],
) -> Optional[str]:
    return render_distribution_availability_bundle_from_prior_outputs(
        run_dir=run_dir,
        current_step_id=current_step_id,
        out_dir=out_dir,
        preverified_parent_artifacts=snapshot,
    )


def _render_continuous_measurement_audit(
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    snapshot: Mapping[str, bytes],
) -> Optional[str]:
    return render_continuous_measurement_audit_bundle(
        run_dir=run_dir,
        current_step_id=current_step_id,
        out_dir=out_dir,
        preverified_parent_artifacts=snapshot,
    )


def _render_ordered_distribution_availability_v2(
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    snapshot: Mapping[str, bytes],
) -> Optional[str]:
    return render_ordered_distribution_bundle_from_prior_outputs(
        run_dir=run_dir,
        current_step_id=current_step_id,
        out_dir=out_dir,
        preverified_parent_artifacts=snapshot,
        authorized_repair_id=_ORDERED_DISTRIBUTION_AVAILABILITY_V2,
    )


# Only repair ids whose seal AND render semantics are identical at every
# pipeline call site may live here. NOT migrated on purpose:
# * ordered_category_distribution_publication_bundle_v1 — its upstream
#   selection deliberately returns without a seal probe; the adapter path
#   would add one and change selection behavior.
# * absolute_risk / cohort_flow / sensitivity / association — their seal or
#   render implementations are still pipeline-local functions; registering
#   them would require figures -> pipeline imports (a forbidden direction).
#   Migrate them only after extracting those implementations into figures/.
_ADAPTERS = {
    MISSINGNESS_REPAIR_ID: SealedRendererAdapter(
        seal=missingness_source_parent_digest_seal,
        render=_render_missingness,
    ),
    "distribution_availability_publication_bundle_from_parent_outputs_v1": (
        SealedRendererAdapter(
            seal=_distribution_availability_parent_digest_seal,
            render=_render_distribution_availability,
        )
    ),
    "continuous_measurement_audit_publication_bundle_v1": SealedRendererAdapter(
        seal=_continuous_measurement_audit_parent_digest_seal,
        render=_render_continuous_measurement_audit,
    ),
    _ORDERED_DISTRIBUTION_AVAILABILITY_V2: SealedRendererAdapter(
        seal=_ordered_distribution_availability_parent_digest_seal,
        render=_render_ordered_distribution_availability_v2,
    ),
}


def sealed_renderer_adapter(repair_id: str) -> Optional[SealedRendererAdapter]:
    """Return only an explicitly registered host-owned renderer adapter."""

    return _ADAPTERS.get(str(repair_id))


__all__ = ["SealedRendererAdapter", "sealed_renderer_adapter"]
