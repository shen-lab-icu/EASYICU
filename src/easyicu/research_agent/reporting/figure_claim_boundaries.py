"""Per-figure support and cannot-prove boundaries for reportability review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..canonical_json import canonical_sha256
from ..figures.contracts import (
    figure_contract_paths,
    figure_contract_tier,
    relative_to_run,
)
from ..schema import AnalysisPlan


class PanelClaimBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    panel_id: str
    supports: str
    cannot_prove: str


class FigureClaimBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_path: str
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    figure_id: str
    tier: Literal["primary_publication", "supporting_step", "other"]
    supports: str
    cannot_prove: str
    figure_role: str | None = None
    boundary_source: Literal["selected_research_design", "legacy_analysis_only"]
    panels: list[PanelClaimBoundary]


class FigureClaimBoundaryAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.figure_claim_boundary_audit/1"] = (
        "easyicu.figure_claim_boundary_audit/1"
    )
    status: Literal["no_figures", "legacy_analysis_only", "complete", "incomplete"]
    claim_ceiling: Literal["analysis_only"] = "analysis_only"
    plan_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    design_selection_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    boundary_ready: bool
    figures: list[FigureClaimBoundary]
    errors: list[str]


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def build_figure_claim_boundary_audit(
    *,
    plan: AnalysisPlan | None,
    run_dir: Path,
    per_step_records: Sequence[Mapping[str, Any]] | None = None,
) -> FigureClaimBoundaryAudit:
    """Bind every current FigureContract to an explicit claim ceiling."""

    paths = figure_contract_paths(run_dir, per_step_records=per_step_records)
    if not paths:
        return FigureClaimBoundaryAudit(
            status="no_figures",
            boundary_ready=False,
            figures=[],
            errors=["No current FigureContract is available for claim-boundary review."],
        )
    selected = (
        plan.design_selection.selected
        if plan is not None and plan.design_selection is not None
        else None
    )
    fallback = (
        "This legacy analysis-only figure cannot authorize a manuscript claim "
        "beyond the exact registered evidence and source data."
    )
    cannot_prove = _text(selected.cannot_prove) if selected is not None else fallback
    figure_role = _text(selected.figure_role) if selected is not None else None
    entries: list[FigureClaimBoundary] = []
    errors: list[str] = []
    for path in paths:
        try:
            raw_bytes = path.read_bytes()
            raw = json.loads(raw_bytes)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"Unreadable FigureContract {path.name}: {type(exc).__name__}")
            continue
        if not isinstance(raw, Mapping):
            errors.append(f"FigureContract {path.name} is not a JSON object.")
            continue
        supports = _text(raw.get("core_claim"))
        if not supports:
            errors.append(f"FigureContract {path.name} has no core_claim.")
            continue
        raw_panels = raw.get("panels")
        if not isinstance(raw_panels, list) or not raw_panels:
            errors.append(f"FigureContract {path.name} has no panels.")
            continue
        panels: list[PanelClaimBoundary] = []
        malformed = False
        for index, raw_panel in enumerate(raw_panels):
            if not isinstance(raw_panel, Mapping):
                malformed = True
                errors.append(f"FigureContract {path.name} panel {index} is malformed.")
                continue
            panel_id = _text(raw_panel.get("panel_id"))
            panel_supports = _text(raw_panel.get("claim"))
            if not panel_id or not panel_supports:
                malformed = True
                errors.append(
                    f"FigureContract {path.name} panel {index} lacks id or claim."
                )
                continue
            panels.append(
                PanelClaimBoundary(
                    panel_id=panel_id,
                    supports=panel_supports,
                    cannot_prove=cannot_prove,
                )
            )
        if malformed:
            continue
        entries.append(
            FigureClaimBoundary(
                contract_path=relative_to_run(path, run_dir),
                contract_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                figure_id=_text(raw.get("figure_id")) or path.stem,
                tier=figure_contract_tier(path, run_dir),
                supports=supports,
                cannot_prove=cannot_prove,
                figure_role=figure_role,
                boundary_source=(
                    "selected_research_design"
                    if selected is not None
                    else "legacy_analysis_only"
                ),
                panels=panels,
            )
        )
    complete = not errors and len(entries) == len(paths)
    status = (
        "complete"
        if complete and selected is not None
        else "legacy_analysis_only"
        if complete
        else "incomplete"
    )
    return FigureClaimBoundaryAudit(
        status=status,
        plan_sha256=(
            canonical_sha256(plan.model_dump(mode="json")) if plan is not None else None
        ),
        design_selection_sha256=(
            canonical_sha256(plan.design_selection.model_dump(mode="json"))
            if plan is not None and plan.design_selection is not None
            else None
        ),
        boundary_ready=status == "complete",
        figures=entries,
        errors=errors,
    )


__all__ = [
    "FigureClaimBoundary",
    "FigureClaimBoundaryAudit",
    "PanelClaimBoundary",
    "build_figure_claim_boundary_audit",
]
