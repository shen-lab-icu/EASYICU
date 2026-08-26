"""Inspect an article-level display package without inventing displays.

The run-level figure gallery answers which execution figures exist. This
module owns the adjacent, submission-facing question: how are separately
rendered main and supplementary displays distributed across an article?

Counts are planning signals, not scientific acceptance gates. A package may
legitimately contain fewer displays. A single main figure is therefore
reported explicitly, while a multi-panel scientific figure with more than one
bound source is not mechanically treated as missing merely because it is one
file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping


ARTICLE_DISPLAY_PACKAGE_SCHEMA_VERSION = "easyicu.article_display_package/2"
MAIN_FIGURE_PLANNING_TARGET = (2, 4)
MAIN_TABLE_PLANNING_TARGET = (2, 3)
DISPLAY_PURPOSES = frozenset({"scientific_result", "diagnostic", "context", "audit"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _panel_placement(contract: Mapping[str, Any]) -> str:
    placements: set[str] = set()
    for panel in contract.get("panels") or ():
        if not isinstance(panel, Mapping):
            continue
        metadata = panel.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        placement = str(metadata.get("placement") or "").strip().casefold()
        if placement in {"main", "supplementary"}:
            placements.add(placement)
    if len(placements) == 1:
        return next(iter(placements))
    if len(placements) > 1:
        return "mixed"
    return "unresolved"


def _panel_display_purpose(contract: Mapping[str, Any]) -> str:
    purposes: set[str] = set()
    for panel in contract.get("panels") or ():
        if not isinstance(panel, Mapping):
            continue
        metadata = panel.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        purpose = str(metadata.get("display_purpose") or "").strip().casefold()
        if purpose in DISPLAY_PURPOSES:
            purposes.add(purpose)
    if len(purposes) == 1:
        return next(iter(purposes))
    if purposes and purposes <= {"scientific_result", "context"}:
        return "scientific_result"
    if purposes and purposes <= {"diagnostic", "context"}:
        return "diagnostic"
    if purposes and purposes <= {"audit", "context"}:
        return "audit"
    if len(purposes) > 1:
        return "mixed"
    return "unresolved"


def _contract_stem(path: Path, suffix: str) -> str:
    return path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem


def _figure_row(contract_path: Path, package_dir: Path) -> dict[str, Any]:
    contract = _load_object(contract_path)
    stem = _contract_stem(contract_path, ".figure_contract.json")
    placement = _panel_placement(contract)
    if placement == "unresolved" and "supplement" in contract_path.name.casefold():
        placement = "supplementary"
    exports = {
        suffix: contract_path.with_name(f"{stem}.{suffix}")
        for suffix in ("png", "svg", "pdf", "tiff", "tif")
        if contract_path.with_name(f"{stem}.{suffix}").is_file()
    }
    png_path = exports.get("png")

    def relative(path: Path) -> str:
        return str(path.relative_to(package_dir))

    return {
        "display_id": str(contract.get("figure_id") or stem),
        "kind": "figure",
        "placement": placement,
        "display_purpose": _panel_display_purpose(contract),
        "label": str(contract.get("title") or contract.get("core_claim") or stem),
        "supports": str(contract.get("core_claim") or ""),
        "cannot_prove": str(contract.get("cannot_prove") or ""),
        "contract_path": relative(contract_path),
        "contract_sha256": _sha256(contract_path),
        "preferred_preview_path": relative(png_path) if png_path else None,
        "preferred_preview_sha256": _sha256(png_path) if png_path else None,
        "exports": {
            suffix: {"path": relative(path), "sha256": _sha256(path)}
            for suffix, path in sorted(exports.items())
        },
        "source_data": list(contract.get("source_data") or ()),
        "panel_count": len(
            [
                panel
                for panel in contract.get("panels") or ()
                if isinstance(panel, Mapping)
            ]
        ),
    }


def _table_row(contract_path: Path, package_dir: Path) -> dict[str, Any]:
    contract = _load_object(contract_path)
    stem = _contract_stem(contract_path, ".table_contract.json")
    placement = str(contract.get("placement") or "unresolved").strip().casefold()
    if placement not in {"main", "supplementary"}:
        placement = "unresolved"
    display_purpose = str(contract.get("display_purpose") or "").strip().casefold()
    if display_purpose not in DISPLAY_PURPOSES:
        display_purpose = "unresolved"

    def relative(path: Path) -> str:
        return str(path.relative_to(package_dir))

    source_name = str(contract.get("source_path") or "")
    source_path = contract_path.with_name(source_name) if source_name else None
    source_present = source_path is not None and source_path.is_file()
    return {
        "display_id": str(contract.get("table_id") or stem),
        "kind": "table",
        "placement": placement,
        "display_purpose": display_purpose,
        "label": str(contract.get("title") or stem),
        "supports": str(contract.get("supports") or ""),
        "cannot_prove": str(contract.get("cannot_prove") or ""),
        "contract_path": relative(contract_path),
        "contract_sha256": _sha256(contract_path),
        "source_path": relative(source_path)
        if source_present and source_path
        else None,
        "source_sha256": _sha256(source_path)
        if source_present and source_path
        else None,
    }


def _count(rows: list[dict[str, Any]], kind: str, placement: str) -> int:
    return sum(
        1
        for row in rows
        if row.get("kind") == kind and row.get("placement") == placement
    )


def _count_purpose(
    rows: list[dict[str, Any]], kind: str, placement: str, display_purpose: str
) -> int:
    return sum(
        1
        for row in rows
        if row.get("kind") == kind
        and row.get("placement") == placement
        and row.get("display_purpose") == display_purpose
    )


def _single_composite_has_scientific_coverage(
    rows: list[dict[str, Any]],
) -> bool:
    """Return whether one main figure carries real multi-panel result coverage.

    Figure-file count is a poor proxy for scientific completeness: journals
    routinely accept one multi-panel primary figure for a narrow estimand.  To
    avoid blessing a decorative or diagnostic composite, coverage requires at
    least two panels, a scientific-result purpose, and at least two separately
    declared source-data bindings.
    """

    main_figure_rows = [
        row
        for row in rows
        if row.get("kind") == "figure" and row.get("placement") == "main"
    ]
    if len(main_figure_rows) != 1:
        return False
    figure = main_figure_rows[0]
    sources = {
        str(source).strip()
        for source in figure.get("source_data") or ()
        if str(source).strip()
    }
    return (
        figure.get("display_purpose") == "scientific_result"
        and int(figure.get("panel_count") or 0) >= 2
        and len(sources) >= 2
    )


def inspect_article_display_package(package_dir: Path) -> dict[str, Any]:
    """Return a digest-bound inventory for one article display directory."""

    root = Path(package_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"article display package is not a directory: {root}")
    status_contract = _load_object(root / "article_display_status.json")
    scientific_status = str(
        status_contract.get("scientific_status") or "analysis_only"
    ).strip()
    planning_targets_applicable = scientific_status not in {
        "failed_closed",
        "blocked",
    }
    rows = [
        _figure_row(path, root)
        for path in sorted(root.rglob("*.figure_contract.json"))
        if path.is_file()
    ]
    rows.extend(
        _table_row(path, root)
        for path in sorted(root.rglob("*.table_contract.json"))
        if path.is_file()
    )
    main_figures = _count(rows, "figure", "main")
    supplementary_figures = _count(rows, "figure", "supplementary")
    main_tables = _count(rows, "table", "main")
    supplementary_tables = _count(rows, "table", "supplementary")
    unresolved = [
        str(row["contract_path"])
        for row in rows
        if row.get("placement") in {"unresolved", "mixed"}
    ]
    single_composite_scientific_coverage = (
        planning_targets_applicable
        and _single_composite_has_scientific_coverage(rows)
    )
    planning_gaps: list[str] = []
    if (
        planning_targets_applicable
        and main_figures == 1
        and not single_composite_scientific_coverage
    ):
        planning_gaps.append("single_composite_only")
    if planning_targets_applicable and (
        main_figures > MAIN_FIGURE_PLANNING_TARGET[1]
        or (
            main_figures < MAIN_FIGURE_PLANNING_TARGET[0]
            and not single_composite_scientific_coverage
        )
    ):
        planning_gaps.append("main_figure_count_outside_planning_target")
    if planning_targets_applicable and (
        not MAIN_TABLE_PLANNING_TARGET[0]
        <= main_tables
        <= MAIN_TABLE_PLANNING_TARGET[1]
    ):
        planning_gaps.append("main_table_count_outside_planning_target")
    if unresolved:
        planning_gaps.append("display_placement_unresolved")
    unresolved_purpose = [
        str(row["contract_path"])
        for row in rows
        if row.get("display_purpose") in {"unresolved", "mixed"}
    ]
    if unresolved_purpose:
        planning_gaps.append("display_purpose_unresolved")
    failed_closed_main = [row for row in rows if row.get("placement") == "main"]
    if scientific_status in {"failed_closed", "blocked"}:
        if any(
            row.get("display_purpose") == "scientific_result"
            for row in failed_closed_main
        ):
            planning_gaps.append("failed_closed_main_scientific_result_forbidden")
        if any(
            row.get("display_purpose") in {"unresolved", "mixed"}
            for row in failed_closed_main
        ):
            planning_gaps.append("failed_closed_main_display_purpose_unresolved")
    return {
        "schema_version": ARTICLE_DISPLAY_PACKAGE_SCHEMA_VERSION,
        "authority": "analysis_only_display_inventory",
        "scientific_status": scientific_status,
        "package_dir": str(root),
        "counts": {
            "main_figures": main_figures,
            "supplementary_figures": supplementary_figures,
            "main_tables": main_tables,
            "supplementary_tables": supplementary_tables,
            "main_scientific_result_figures": _count_purpose(
                rows, "figure", "main", "scientific_result"
            ),
            "main_scientific_result_tables": _count_purpose(
                rows, "table", "main", "scientific_result"
            ),
            "main_diagnostic_figures": _count_purpose(
                rows, "figure", "main", "diagnostic"
            ),
            "main_diagnostic_tables": _count_purpose(
                rows, "table", "main", "diagnostic"
            ),
            "main_unresolved_purpose_figures": _count_purpose(
                rows, "figure", "main", "unresolved"
            )
            + _count_purpose(rows, "figure", "main", "mixed"),
            "main_unresolved_purpose_tables": _count_purpose(
                rows, "table", "main", "unresolved"
            )
            + _count_purpose(rows, "table", "main", "mixed"),
            "unresolved_placements": len(unresolved),
        },
        "planning_targets": {
            "main_figures": list(MAIN_FIGURE_PLANNING_TARGET),
            "main_tables": list(MAIN_TABLE_PLANNING_TARGET),
            "are_acceptance_gates": False,
            "applicable_to_scientific_status": planning_targets_applicable,
        },
        "single_composite_only": planning_targets_applicable and main_figures == 1,
        "single_composite_scientific_coverage": (
            single_composite_scientific_coverage
        ),
        "planning_target_gaps": list(dict.fromkeys(planning_gaps)),
        "unresolved_contract_paths": unresolved,
        "unresolved_purpose_contract_paths": unresolved_purpose,
        "displays": rows,
        "claim_boundary": (
            "This inventory proves which digest-bound display contracts and exports "
            "exist and whether a main display is a scientific result or a diagnostic. "
            "A main diagnostic does not become an effect estimate, a selected class "
            "solution, proof of publication readiness, or a mandatory display count."
        ),
    }


def reader_figure_rows(
    inventory: Mapping[str, Any],
    *,
    placement: Literal["main", "supplementary"],
) -> list[dict[str, Any]]:
    """Return previewable figures in deterministic manuscript order."""

    return [
        dict(row)
        for row in inventory.get("displays") or ()
        if isinstance(row, Mapping)
        and row.get("kind") == "figure"
        and row.get("placement") == placement
        and row.get("preferred_preview_path")
    ]


__all__ = [
    "ARTICLE_DISPLAY_PACKAGE_SCHEMA_VERSION",
    "MAIN_FIGURE_PLANNING_TARGET",
    "MAIN_TABLE_PLANNING_TARGET",
    "inspect_article_display_package",
    "reader_figure_rows",
]
