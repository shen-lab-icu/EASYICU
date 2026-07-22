"""Closed registry for sealed, cross-file deterministic figure renderers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

from .missingness_source import (
    REPAIR_ID as MISSINGNESS_REPAIR_ID,
    missingness_source_parent_digest_seal,
    render_missingness_source_bundle,
)

Seal = Callable[[Path, str], Optional[dict[str, str]]]
Render = Callable[[Path, str, Path, Mapping[str, bytes]], Optional[str]]


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


_ADAPTERS = {
    MISSINGNESS_REPAIR_ID: SealedRendererAdapter(
        seal=missingness_source_parent_digest_seal,
        render=_render_missingness,
    )
}


def sealed_renderer_adapter(repair_id: str) -> Optional[SealedRendererAdapter]:
    """Return only an explicitly registered host-owned renderer adapter."""

    return _ADAPTERS.get(str(repair_id))


__all__ = ["SealedRendererAdapter", "sealed_renderer_adapter"]
