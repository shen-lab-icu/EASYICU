from __future__ import annotations

import json
from pathlib import Path

from easyicu.research_agent.reporting.article_display_package import (
    inspect_article_display_package,
    reader_figure_rows,
)


def _write_figure(root: Path, stem: str, placement: str) -> None:
    (root / f"{stem}.png").write_bytes(b"\x89PNG\r\n\x1a\npreview")
    (root / f"{stem}.figure_contract.json").write_text(
        json.dumps(
            {
                "figure_id": f"figure:{stem}",
                "core_claim": f"{stem} supports its declared result.",
                "panels": [
                    {
                        "panel_id": "a",
                        "metadata": {"placement": placement},
                    }
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
    assert "single_composite_only" in inventory["planning_target_gaps"]
    assert inventory["planning_targets"]["are_acceptance_gates"] is False
    assert "publication readiness" in inventory["claim_boundary"]


def test_article_display_inventory_does_not_demand_main_results_when_failed_closed(
    tmp_path: Path,
) -> None:
    (tmp_path / "article_display_status.json").write_text(
        json.dumps({"scientific_status": "failed_closed"}), encoding="utf-8"
    )
    _write_figure(tmp_path, "supplementary_diagnostic", "supplementary")

    inventory = inspect_article_display_package(tmp_path)

    assert inventory["scientific_status"] == "failed_closed"
    assert inventory["counts"]["main_figures"] == 0
    assert inventory["counts"]["main_tables"] == 0
    assert inventory["planning_target_gaps"] == []
    assert inventory["planning_targets"]["applicable_to_scientific_status"] is False
