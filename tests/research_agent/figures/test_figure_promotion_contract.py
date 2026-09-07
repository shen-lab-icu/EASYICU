from __future__ import annotations

import json

from easyicu.research_agent.figures.promotion_contract import (
    contract_promoted_from_source,
)


def test_promotion_preserves_source_claim_panel_and_statistics_note(tmp_path) -> None:
    contract_path = tmp_path / "figure_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "core_claim": "Source claim",
                "statistics_note": "Counts only; no inference.",
                "reader_caption": "A: observed counts. No confidence intervals are shown.",
                "archetype": "quantitative_grid",
                "width_mm": 183.0,
                "height_mm": 86.0,
                "panels": [
                    {
                        "panel_id": "A",
                        "title": "Distribution",
                        "claim": "Observed counts",
                        "chart_type": "bar",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    promoted = contract_promoted_from_source(
        contract_path,
        source_ids=["evidence:counts"],
    )

    assert promoted.core_claim == "Source claim"
    assert promoted.panels[0].claim == "Observed counts"
    assert promoted.panels[0].metadata["chart_type"] == "bar"
    assert promoted.panels[0].evidence_ids == ["evidence:counts"]
    assert promoted.statistics_note.startswith("Counts only; no inference.")
    assert promoted.reader_caption == "A: observed counts. No confidence intervals are shown."
    assert promoted.archetype == "quantitative_grid"
    assert (promoted.width_mm, promoted.height_mm) == (183.0, 86.0)
    assert "promotes a registered step-level" in promoted.statistics_note


def test_promotion_fails_closed_to_a_source_bound_default_contract(tmp_path) -> None:
    contract_path = tmp_path / "invalid.json"
    contract_path.write_text("not json", encoding="utf-8")

    promoted = contract_promoted_from_source(
        contract_path,
        source_ids=["evidence:figure", "evidence:table"],
    )

    assert promoted.panels[0].evidence_ids == [
        "evidence:figure",
        "evidence:table",
    ]
    assert promoted.source_data == ["evidence:figure", "evidence:table"]
    assert promoted.reader_caption is None
    assert "reader_caption" not in promoted.model_dump(mode="json")
