from __future__ import annotations

import pandas as pd

from easyicu.research_agent.figures.display_labels import (
    binary_contrast_label,
    binary_scope_label,
    display_label,
    scoped_label_lookup,
)
from easyicu.research_agent.figures.strata import normalise_strata_frame


def test_strata_projection_excludes_overall_audit_row_and_scales_percent() -> None:
    frame = pd.DataFrame(
        {
            "row_role": ["overall", "stratum", "stratum"],
            "exposure_level": [None, 0, 1],
            "outcome_rate_pct": [20.0, 10.0, 30.0],
            "outcome_denominator": [100, 60, 40],
        }
    )

    result = normalise_strata_frame(frame)

    assert result["score"].tolist() == ["Unexposed", "Exposed"]
    assert result["rate"].tolist() == [0.1, 0.3]
    assert result["n"].tolist() == [60, 40]
    assert result.attrs == {
        "score_label": "Exposure group",
        "score_is_numeric": False,
    }


def test_strata_projection_uses_planner_owned_display_label() -> None:
    frame = pd.DataFrame(
        {
            "status": [0, 1],
            "outcome_rate": [0.1, 0.2],
        }
    )

    result = normalise_strata_frame(
        frame,
        display_labels={"status": "Sepsis-3"},
    )

    assert result["score"].tolist() == ["Sepsis-3 negative", "Sepsis-3 positive"]
    assert result.attrs["score_label"] == "Sepsis-3 status"


def test_display_label_fallback_does_not_invent_endpoint_semantics() -> None:
    assert display_label("death") == "Death"
    assert display_label("death", {"death": "In-hospital mortality"}) == (
        "In-hospital mortality"
    )


def test_scoped_binary_labels_remain_bound_to_their_planner_variable() -> None:
    labels = {
        "marker_a=0": "Marker A absent",
        "marker_a=1": "Marker A present",
        "marker_b=1": "Marker B high",
    }

    assert scoped_label_lookup("marker_a", 0.0, labels) == "Marker A absent"
    assert scoped_label_lookup("marker_a", True, labels) == "Marker A present"
    assert scoped_label_lookup("marker_b", 0, labels) is None
    assert binary_contrast_label("marker_a", labels) == (
        "Marker A present vs Marker A absent"
    )
    assert binary_scope_label("marker_a", labels) == "Marker A"
    assert binary_contrast_label("marker_b", labels) is None
    assert binary_scope_label("marker_b", labels) is None
