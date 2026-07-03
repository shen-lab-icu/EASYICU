"""Reviewer-facing artifact manifests for research-agent runs.

This module keeps review entry-point policy out of the readiness/reporting
orchestrator. It turns figure contracts already classified by readiness gates
into two durable payloads:

* ``review_artifacts.json``: full primary/supporting/archive manifest.
* ``figure_gallery.json``: concise gallery for human review surfaces.

The policy is case-neutral. It separates canonical run-level publication
figures from step-level supporting artifacts, and archives support figures that
are duplicates or are already covered by the primary publication figure.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .figure_contracts import (
    EXPORT_SUFFIXES,
    figure_contract_export_paths,
    figure_contract_label,
    figure_contract_panel_summaries,
    read_figure_contract,
    relative_to_run,
)


_PRIMARY_PUBLICATION_FIGURE_IDS = {
    "publication_figure",
    "easyicu_publication_figure",
}
_PRIMARY_LIKE_ROLES = {
    "descriptive_result",
    "primary_estimand",
    "relationship",
}


# Inline previews are duplicated across review_artifacts.json,
# figure_gallery.json and their evidence copies; cap the embedded PNG so one
# high-dpi export cannot balloon every artifact. Larger figures fall back to
# the relative_path/exports references already present on the row.
_MAX_INLINE_PNG_BYTES = 1_500_000


def _png_data_url(path: Path) -> Optional[str]:
    if not path.exists() or path.suffix.lower() != ".png":
        return None
    try:
        if path.stat().st_size > _MAX_INLINE_PNG_BYTES:
            return None
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None
    return f"data:image/png;base64,{payload}"


def _figure_artifact_row(
    *,
    contract_path: Path,
    run_dir: Path,
    tier: str,
    include_data_url: bool = False,
) -> Dict[str, Any]:
    raw = read_figure_contract(contract_path)
    panel_summaries = figure_contract_panel_summaries(raw)
    figure_id = str(raw.get("figure_id") or "").strip()
    exports = figure_contract_export_paths(contract_path)
    preferred = exports.get("png") or exports.get("svg") or contract_path
    row: Dict[str, Any] = {
        "label": (
            "Primary publication figure"
            if tier == "primary_publication"
            else figure_contract_label(contract_path)
        ),
        "figure_id": figure_id,
        "tier": tier,
        "relative_path": relative_to_run(preferred, run_dir),
        "contract_path": relative_to_run(contract_path, run_dir),
        "panel_count": len(panel_summaries),
        "panel_roles": sorted(
            {panel["role"] for panel in panel_summaries if panel["role"]}
        ),
        "chart_types": sorted(
            {panel["chart_type"] for panel in panel_summaries if panel["chart_type"]}
        ),
        "exports": {
            key: relative_to_run(path, run_dir)
            for key, path in sorted(exports.items())
            if key != "contract"
        },
        "status": "canonical_main" if tier == "primary_publication" else "supporting",
        "review_recommendation": (
            "review_first"
            if tier == "primary_publication"
            else "supporting_context_not_primary"
        ),
    }
    if include_data_url and (png_path := exports.get("png")):
        data_url = _png_data_url(png_path)
        if data_url:
            row["data_url"] = data_url
    return row


def _supporting_review_group_key(row: Dict[str, Any]) -> Optional[str]:
    figure_id = str(row.get("figure_id") or "").strip().lower()
    if figure_id and figure_id not in _PRIMARY_PUBLICATION_FIGURE_IDS:
        return f"figure_id:{figure_id}"
    return None


def _archive_supporting_figure_reason(
    *,
    row: Dict[str, Any],
    primary_roles: set[str],
    seen_supporting_group_keys: set[str],
) -> Optional[str]:
    roles = {
        str(role).strip().lower()
        for role in (row.get("panel_roles") or [])
        if str(role).strip()
    }
    group_key = _supporting_review_group_key(row)
    if group_key and group_key in seen_supporting_group_keys:
        return "duplicate_supporting_figure_id"

    figure_id = str(row.get("figure_id") or "").strip().lower()
    contract_name = Path(str(row.get("contract_path") or "")).name
    generic_publication_name = contract_name.startswith(
        ("publication_figure.", "easyicu_publication_figure.")
    )
    if (
        roles
        and roles <= primary_roles
        and (
            figure_id in _PRIMARY_PUBLICATION_FIGURE_IDS
            or generic_publication_name
            or bool(roles & _PRIMARY_LIKE_ROLES)
        )
    ):
        return "covered_by_primary_publication_figure"
    return None


def _partition_supporting_review_figures(
    *,
    primary_figures: Sequence[Dict[str, Any]],
    supporting_figures: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    primary_roles = {
        str(role).strip().lower()
        for row in primary_figures
        for role in (row.get("panel_roles") or [])
        if str(role).strip()
    }
    seen_supporting_group_keys: set[str] = set()
    visible: List[Dict[str, Any]] = []
    archived: List[Dict[str, Any]] = []
    for row in supporting_figures:
        reason = _archive_supporting_figure_reason(
            row=row,
            primary_roles=primary_roles,
            seen_supporting_group_keys=seen_supporting_group_keys,
        )
        group_key = _supporting_review_group_key(row)
        if reason is None:
            if group_key:
                seen_supporting_group_keys.add(group_key)
            visible.append(row)
            continue
        archived.append(
            {
                **row,
                "status": "archived_supporting",
                "review_recommendation": "archived_context_not_primary",
                "archive_reason": reason,
            }
        )
    return visible, archived


def build_review_artifact_payloads(
    *,
    run_dir: Path,
    gates: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    """Build reviewer manifest, gallery payload, and canonical main-figure paths."""

    primary_paths = [
        run_dir / str(path)
        for path in (gates.get("display_primary_publication_contract_paths") or [])
    ]
    supporting_paths = [
        run_dir / str(path)
        for path in (gates.get("display_supporting_figure_contract_paths") or [])
    ]
    primary_figures = [
        _figure_artifact_row(
            contract_path=path,
            run_dir=run_dir,
            tier="primary_publication",
            include_data_url=True,
        )
        for path in primary_paths
        if path.exists()
    ]
    supporting_figures = [
        _figure_artifact_row(
            contract_path=path,
            run_dir=run_dir,
            tier="supporting_step",
            include_data_url=False,
        )
        for path in supporting_paths
        if path.exists()
    ]
    visible_supporting_figures, archived_supporting_figures = (
        _partition_supporting_review_figures(
            primary_figures=primary_figures,
            supporting_figures=supporting_figures,
        )
    )
    review_payload = {
        "schema_version": "easyicu.review_artifacts/1",
        "primary_publication_figures": primary_figures,
        "supporting_figures": visible_supporting_figures,
        "archived_supporting_figures": archived_supporting_figures,
        "review_order": [
            *(row["relative_path"] for row in primary_figures),
            *(row["relative_path"] for row in visible_supporting_figures),
            "display_suite_audit.json",
            "article_figure_strategy_audit.json",
            "author_review_note.md",
        ],
        "policy": {
            "primary_publication_source": "publication_figures/",
            "supporting_step_source": "steps/*/outputs/",
            "supporting_step_figures_are_not_canonical_main_figures": True,
            "covered_or_duplicate_supporting_figures_are_archived": True,
        },
    }
    gallery_payload = {
        "kind": "figure_gallery",
        "schema_version": "easyicu.figure_gallery/1",
        "status": "ok" if primary_figures else "no_primary_publication_figure",
        "figures": [*primary_figures, *visible_supporting_figures],
        "primary_count": len(primary_figures),
        "supporting_count": len(visible_supporting_figures),
        "archived_supporting_count": len(archived_supporting_figures),
        "archived_supporting_figures": archived_supporting_figures,
    }
    canonical_paths: Dict[str, str] = {}
    if primary_figures:
        primary = primary_figures[0]
        canonical_paths["primary_publication_figure"] = str(primary["relative_path"])
        canonical_paths["primary_publication_figure_contract"] = str(
            primary["contract_path"]
        )
        exports = primary.get("exports") if isinstance(primary.get("exports"), dict) else {}
        for key in EXPORT_SUFFIXES:
            if key in exports:
                canonical_paths[f"primary_publication_figure_{key}"] = str(
                    exports[key]
                )
    return review_payload, gallery_payload, canonical_paths
