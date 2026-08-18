"""Build run-level figure contracts from registered step-level contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .publication import make_figure_contract


_PROMOTION_NOTE = (
    "This run-level bundle promotes a registered step-level publication "
    "figure and preserves its source evidence."
)


def contract_promoted_from_source(
    contract_path: Path,
    *,
    source_ids: Sequence[str],
):
    """Preserve usable source-contract semantics in a run-level contract."""

    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}
    source_statistics_note = str(payload.get("statistics_note") or "").strip()
    statistics_note = (
        f"{source_statistics_note} {_PROMOTION_NOTE}"
        if source_statistics_note
        else _PROMOTION_NOTE
    )
    source_panels = payload.get("panels")
    panels: List[Dict[str, Any]] = []
    if isinstance(source_panels, list):
        for idx, panel in enumerate(source_panels, start=1):
            if not isinstance(panel, dict):
                continue
            panel_id = str(panel.get("panel_id") or panel.get("id") or idx)
            metadata = dict(panel.get("metadata") or {})
            for key in ("chart_type", "scale", "visual_role"):
                if panel.get(key) is not None and key not in metadata:
                    metadata[key] = panel.get(key)
            promoted_panel = {
                "panel_id": panel_id,
                "title": str(panel.get("title") or f"Panel {panel_id}"),
                "role": panel.get("role") or "validation",
                "claim": str(
                    panel.get("claim")
                    or panel.get("purpose")
                    or "This panel is promoted from registered figure evidence."
                ),
                "evidence_ids": list(source_ids),
                "review_risk": panel.get("review_risk"),
            }
            if metadata:
                promoted_panel["metadata"] = metadata
            panels.append(promoted_panel)
    if not panels:
        panels = [
            {
                "panel_id": "A",
                "title": "Registered manuscript figure",
                "role": "validation",
                "claim": (
                    "The run-level figure is promoted from registered step-level "
                    "figure evidence with source data."
                ),
                "evidence_ids": list(source_ids),
                "review_risk": (
                    "Interpretation depends on the upstream figure contract and "
                    "source-data table."
                ),
            }
        ]
    try:
        return make_figure_contract(
            figure_id="easyicu_publication_figure",
            core_claim=str(
                payload.get("core_claim")
                or "The manuscript-facing figure is promoted from registered step-level evidence."
            ),
            panels=panels,
            source_data=list(source_ids),
            statistics_note=statistics_note,
        )
    except Exception:
        return make_figure_contract(
            figure_id="easyicu_publication_figure",
            core_claim=(
                "The manuscript-facing figure is promoted from registered "
                "step-level evidence."
            ),
            panels=[
                {
                    "panel_id": "A",
                    "title": "Registered manuscript figure",
                    "role": "validation",
                    "claim": (
                        "The figure is copied from a registered step-level "
                        "figure bundle."
                    ),
                    "evidence_ids": list(source_ids),
                }
            ],
            source_data=list(source_ids),
            statistics_note=_PROMOTION_NOTE,
        )


__all__ = ["contract_promoted_from_source"]
