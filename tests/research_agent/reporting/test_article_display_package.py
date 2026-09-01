from __future__ import annotations

import json
from pathlib import Path

from easyicu.research_agent.reporting.article_display_package import (
    inspect_article_display_package,
    reader_figure_rows,
)


def _write_figure(
    root: Path,
    stem: str,
    placement: str,
    *,
    display_purpose: str | None = None,
    panel_count: int = 1,
    source_count: int = 0,
) -> None:
    metadata = {"placement": placement}
    if display_purpose is not None:
        metadata["display_purpose"] = display_purpose
    (root / f"{stem}.png").write_bytes(b"\x89PNG\r\n\x1a\npreview")
    (root / f"{stem}.figure_contract.json").write_text(
        json.dumps(
            {
                "figure_id": f"figure:{stem}",
                "core_claim": f"{stem} supports its declared result.",
                "panels": [
                    {
                        "panel_id": chr(ord("a") + index),
                        "metadata": metadata,
                    }
                    for index in range(panel_count)
                ],
                "source_data": [
                    f"source_{index + 1}.csv" for index in range(source_count)
                ],
            }
        ),
        encoding="utf-8",
    )


def test_article_display_inventory_separates_main_and_supplement(
    tmp_path: Path,
) -> None:
    _write_figure(tmp_path, "main_figure_1", "main")
    _write_figure(tmp_path, "main_figure_2", "main")
    _write_figure(tmp_path, "supplementary_figure_s1", "supplementary")

    inventory = inspect_article_display_package(tmp_path)

    assert inventory["counts"] == {
        "main_figures": 2,
        "supplementary_figures": 1,
        "main_tables": 0,
        "supplementary_tables": 0,
        "main_scientific_result_figures": 0,
        "main_scientific_result_tables": 0,
        "main_diagnostic_figures": 0,
        "main_diagnostic_tables": 0,
        "main_unresolved_purpose_figures": 2,
        "main_unresolved_purpose_tables": 0,
        "unresolved_placements": 0,
    }
    assert inventory["single_composite_only"] is False
    assert (
        "main_figure_count_outside_planning_target"
        not in inventory["planning_target_gaps"]
    )
    assert len(reader_figure_rows(inventory, placement="main")) == 2
    assert len(reader_figure_rows(inventory, placement="supplementary")) == 1


def test_article_display_inventory_flags_single_composite_without_calling_it_invalid(
    tmp_path: Path,
) -> None:
    _write_figure(tmp_path, "only_main", "main")

    inventory = inspect_article_display_package(tmp_path)

    assert inventory["single_composite_only"] is True
    assert inventory["single_composite_scientific_coverage"] is False
    assert "single_composite_only" in inventory["planning_target_gaps"]
    assert inventory["planning_targets"]["are_acceptance_gates"] is False
    assert "publication readiness" in inventory["claim_boundary"]


def test_article_display_inventory_accepts_one_bound_multipanel_scientific_figure(
    tmp_path: Path,
) -> None:
    _write_figure(
        tmp_path,
        "narrow_primary_result",
        "main",
        display_purpose="scientific_result",
        panel_count=3,
        source_count=3,
    )

    inventory = inspect_article_display_package(tmp_path)

    assert inventory["single_composite_only"] is True
    assert inventory["single_composite_scientific_coverage"] is True
    assert "single_composite_only" not in inventory["planning_target_gaps"]
    assert (
        "main_figure_count_outside_planning_target"
        not in inventory["planning_target_gaps"]
    )


def test_article_display_inventory_allows_main_diagnostics_when_failed_closed(
    tmp_path: Path,
) -> None:
    (tmp_path / "article_display_status.json").write_text(
        json.dumps({"scientific_status": "failed_closed"}), encoding="utf-8"
    )
    _write_figure(
        tmp_path,
        "main_identifiability_diagnostic",
        "main",
        display_purpose="diagnostic",
    )
    _write_figure(
        tmp_path,
        "supplementary_audit",
        "supplementary",
        display_purpose="audit",
    )

    inventory = inspect_article_display_package(tmp_path)

    assert inventory["scientific_status"] == "failed_closed"
    assert inventory["counts"]["main_figures"] == 1
    assert inventory["counts"]["main_tables"] == 0
    assert inventory["counts"]["main_scientific_result_figures"] == 0
    assert inventory["counts"]["main_diagnostic_figures"] == 1
    assert inventory["counts"]["main_unresolved_purpose_figures"] == 0
    assert inventory["planning_target_gaps"] == []
    assert inventory["planning_targets"]["applicable_to_scientific_status"] is False


def test_article_display_inventory_rejects_failed_closed_main_scientific_result(
    tmp_path: Path,
) -> None:
    (tmp_path / "article_display_status.json").write_text(
        json.dumps({"scientific_status": "failed_closed"}), encoding="utf-8"
    )
    _write_figure(
        tmp_path,
        "unauthorized_effect",
        "main",
        display_purpose="scientific_result",
    )

    inventory = inspect_article_display_package(tmp_path)

    assert inventory["counts"]["main_scientific_result_figures"] == 1
    assert (
        "failed_closed_main_scientific_result_forbidden"
        in inventory["planning_target_gaps"]
    )
