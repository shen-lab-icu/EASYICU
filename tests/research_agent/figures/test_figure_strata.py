from __future__ import annotations

import pandas as pd
import pytest

from easyicu.research_agent.figures.display_labels import (
    binary_contrast_label,
    binary_scope_label,
    display_label,
    figure_language_labels,
    label_lookup,
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


def test_fallback_display_labels_expand_common_icu_tokens() -> None:
    assert display_label("sep3_sofa2") == "Sepsis-3 SOFA-2"
    assert display_label("sofa2_resp") == "SOFA-2 respiratory"
    assert display_label("lact") == "Lactate"
    assert display_label("spo2") == "SpO2"


@pytest.mark.parametrize(("identifier", "declared", "expected"), [
    ("icu_readmission", "ICU Readmission", "ICU readmission"),
    ("icu_readmission", "Icu Readmission", "ICU readmission"),
    ("in_hospital_mortality", "In Hospital Mortality", "In-hospital mortality"),
    ("in_hospital_mortality", "In-Hospital Mortality", "In-hospital mortality"),
    ("sofa_score", "SOFA Score", "SOFA score"),
    ("sofa_score", "Sofa Score", "SOFA score"),
    ("kdigo_aki_stage", "KDIGO AKI Stage", "KDIGO AKI stage"),
    ("lactate_tertile", "Lactate Tertile", "Lactate tertile"),
    ("map", "MAP", "MAP"),
    ("ph", "pH", "pH"),
    ("ph", "Ph", "pH"),
    ("spo2", "SpO2", "SpO2"),
    ("spo2", "Spo2", "SpO2"),
    ("crp_level", "CRP Level", "CRP level"),
    ("egfr", "eGFR", "eGFR"),
])
def test_figure_labels_use_sentence_case_and_preserve_acronyms(identifier, declared, expected):
    labels = {identifier: declared}
    assert display_label(identifier, labels) == expected
    assert label_lookup(identifier, labels) == expected
    assert scoped_label_lookup("outcome", 1, {"outcome=1": declared}) == expected
    assert figure_language_labels(labels) == {identifier: expected}
    assert figure_language_labels({identifier: "结局"}, {identifier: declared}) == {identifier: expected}
    assert figure_language_labels({}, {identifier: declared, f"{identifier}=1": declared}) == {
        identifier: expected, f"{identifier}=1": expected,
    }


@pytest.mark.parametrize(("identifier", "expected"), [
    ("icu_readmission", "ICU readmission"),
    ("in_hospital_mortality", "In-hospital mortality"),
    ("sofa_score", "SOFA score"),
    ("kdigo_aki_stage", "KDIGO AKI stage"),
    ("sofa2_resp", "SOFA-2 respiratory"),
    ("lactate_tertile", "Lactate tertile"),
    ("ph", "pH"),
    ("spo2", "SpO2"),
])
def test_identifier_label_fallback_uses_sentence_case(identifier, expected):
    assert display_label(identifier) == expected


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
