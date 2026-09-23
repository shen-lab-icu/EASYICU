"""Observability-preserving KDIGO staging derived from evidence receipts.

The contract under test is a scientific one: a stage of ``0`` must mean "injury
was ruled out", never "nothing was measured".  These tests fix the rule against
the published renal bundle's receipts and against the retired strict profile's
spelling, so a later export rename cannot quietly reintroduce the collapse.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.scores.aki_strict import (
    STRICT_KDIGO_SCHEMA_VERSION,
    StrictKdigoError,
    derive_strict_kdigo_stage,
    strict_kdigo_receipt_columns,
    strict_kdigo_summary,
    summarize_strict_kdigo_window,
)


def _rows(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    for column in (
        "aki_stage_creat_reference",
        "aki_stage_uo_reference",
        "aki_stage_rrt_reference",
    ):
        if column not in frame:
            frame[column] = pd.Series([None] * len(frame), dtype="Int64")
    for column in (
        "creatinine_evidence_status",
        "urine_evidence_status",
        "rrt_evidence_status",
    ):
        if column not in frame:
            frame[column] = "indeterminate"
    return frame


def test_stage_zero_requires_complete_negative_evidence() -> None:
    frame = _rows(
        [
            # every component observed and negative -> a real stage 0
            {
                "stay_id": 1,
                "charttime": 1.0,
                "aki_stage_creat_reference": 0,
                "aki_stage_uo_reference": 0,
                "aki_stage_rrt_reference": 0,
                "creatinine_evidence_status": "negative",
                "urine_evidence_status": "negative",
                "rrt_evidence_status": "negative",
            },
            # one component never assessed -> unknown, not the reference group
            {
                "stay_id": 2,
                "charttime": 1.0,
                "aki_stage_creat_reference": 0,
                "creatinine_evidence_status": "negative",
                "urine_evidence_status": "indeterminate",
                "rrt_evidence_status": "negative",
            },
            # nothing assessed at all
            {"stay_id": 3, "charttime": 1.0},
        ]
    )

    derived = derive_strict_kdigo_stage(frame)

    assert derived["aki_stage_strict"].tolist()[0] == 0
    assert derived["aki_stage_strict"].isna().tolist() == [False, True, True]
    assert derived["aki_ascertainment"].tolist() == [
        "negative_complete",
        "partial_no_observed_positive",
        "indeterminate",
    ]


def test_a_positive_component_sets_the_stage_and_never_reads_the_combined_stage() -> None:
    frame = _rows(
        [
            {
                "stay_id": 1,
                "charttime": 4.0,
                "aki_stage_creat_reference": 1,
                "aki_stage_uo_reference": 3,
                "creatinine_evidence_status": "positive",
                "urine_evidence_status": "positive",
                "rrt_evidence_status": "negative",
                # A reference stage that disagrees must not leak into the result.
                "aki_stage_reference": 1,
            }
        ]
    )

    derived = derive_strict_kdigo_stage(frame)

    assert derived["aki_stage_strict"].tolist() == [3]
    assert derived["aki_ascertainment"].tolist() == ["positive"]


def test_a_positive_component_without_a_magnitude_stays_positive_not_zero() -> None:
    """eICU RRT positives carry no reference component stage; they are not stage 0."""

    frame = _rows(
        [
            {
                "stay_id": 1,
                "charttime": 2.0,
                "aki_stage_creat_reference": 0,
                "aki_stage_uo_reference": 0,
                "creatinine_evidence_status": "negative",
                "urine_evidence_status": "negative",
                "rrt_evidence_status": "positive",
            }
        ]
    )

    derived = derive_strict_kdigo_stage(frame)

    assert derived["aki_stage_strict"].isna().all()
    assert derived["aki_ascertainment"].tolist() == ["positive_stage_unresolved"]

    window = summarize_strict_kdigo_window(derived, id_column="stay_id")
    assert window["aki_stage_strict"].isna().all()
    assert window["aki_ascertainment"].tolist() == ["positive_stage_unresolved"]


def test_window_summary_keeps_an_unassessed_stay_out_of_the_reference_group() -> None:
    frame = _rows(
        [
            # stay 1: positive inside the window
            {
                "stay_id": 1,
                "charttime": 6.0,
                "aki_stage_uo_reference": 2,
                "creatinine_evidence_status": "indeterminate",
                "urine_evidence_status": "positive",
                "rrt_evidence_status": "negative",
            },
            # stay 1: complete negative earlier; the positive must win
            {
                "stay_id": 1,
                "charttime": 1.0,
                "aki_stage_creat_reference": 0,
                "aki_stage_uo_reference": 0,
                "aki_stage_rrt_reference": 0,
                "creatinine_evidence_status": "negative",
                "urine_evidence_status": "negative",
                "rrt_evidence_status": "negative",
            },
            # stay 2: positive only AFTER the window closes
            {
                "stay_id": 2,
                "charttime": 30.0,
                "aki_stage_creat_reference": 3,
                "creatinine_evidence_status": "positive",
                "urine_evidence_status": "negative",
                "rrt_evidence_status": "negative",
            },
            {
                "stay_id": 2,
                "charttime": 2.0,
                "aki_stage_creat_reference": 0,
                "aki_stage_uo_reference": 0,
                "aki_stage_rrt_reference": 0,
                "creatinine_evidence_status": "negative",
                "urine_evidence_status": "negative",
                "rrt_evidence_status": "negative",
            },
            # stay 3: only partial evidence in the window
            {"stay_id": 3, "charttime": 3.0, "urine_evidence_status": "negative"},
        ]
    )

    window = summarize_strict_kdigo_window(
        frame, id_column="stay_id", window_start_hours=0.0, window_end_hours=24.0
    )

    assert window["stay_id"].tolist() == [1, 2, 3]
    assert window["aki_stage_strict"].tolist()[:2] == [2, 0]
    assert bool(window["aki_stage_strict"].isna().tolist()[2])
    # Stay 3 was partly assessed (urine only) and never positive: still unknown
    # for the stage, but distinguishable from a stay nothing was measured on.
    assert window["aki_ascertainment"].tolist() == [
        "positive",
        "negative_complete",
        "partial_no_observed_positive",
    ]
    assert window["kidney_window_row_count"].tolist() == [2, 1, 1]


def test_the_retired_strict_profile_spelling_is_still_readable() -> None:
    frame = pd.DataFrame(
        {
            "stay_id": [1],
            "charttime": [1.0],
            "aki_stage_creat": [0],
            "aki_stage_uo": [0],
            "aki_stage_rrt": [0],
            "creatinine_ascertainment": ["negative"],
            "urine_ascertainment": ["negative"],
            "rrt_ascertainment": ["negative"],
        }
    )

    assert strict_kdigo_receipt_columns(frame.columns) is not None
    derived = derive_strict_kdigo_stage(frame)
    assert derived["aki_stage_strict"].tolist() == [0]


def test_receipts_are_required_and_contradictions_fail_closed() -> None:
    bare = pd.DataFrame(
        {"stay_id": [1], "charttime": [1.0], "aki_stage": [0], "aki_stage_creat": [0]}
    )
    assert strict_kdigo_receipt_columns(bare.columns) is None
    with pytest.raises(StrictKdigoError, match="no per-component"):
        derive_strict_kdigo_stage(bare)

    contradictory = _rows(
        [
            {
                "stay_id": 1,
                "charttime": 1.0,
                "aki_stage_creat_reference": 2,
                "creatinine_evidence_status": "negative",
                "urine_evidence_status": "negative",
                "rrt_evidence_status": "negative",
            }
        ]
    )
    with pytest.raises(StrictKdigoError, match="contradicts"):
        derive_strict_kdigo_stage(contradictory)

    unsupported = _rows(
        [{"stay_id": 1, "charttime": 1.0, "urine_evidence_status": "maybe"}]
    )
    with pytest.raises(StrictKdigoError, match="unsupported evidence states"):
        derive_strict_kdigo_stage(unsupported)


def test_summary_is_aggregate_only() -> None:
    frame = _rows(
        [
            {
                "stay_id": 1,
                "charttime": 1.0,
                "aki_stage_creat_reference": 0,
                "aki_stage_uo_reference": 0,
                "aki_stage_rrt_reference": 0,
                "creatinine_evidence_status": "negative",
                "urine_evidence_status": "negative",
                "rrt_evidence_status": "negative",
            },
            {"stay_id": 2, "charttime": 1.0},
        ]
    )

    summary = strict_kdigo_summary(derive_strict_kdigo_stage(frame))

    assert summary["schema_version"] == STRICT_KDIGO_SCHEMA_VERSION
    assert summary["rows"] == 2 and summary["unknown_rows"] == 1
    assert summary["stage_counts"] == {"0": 1}
    assert summary["ascertainment_counts"] == {
        "negative_complete": 1,
        "indeterminate": 1,
    }
    assert "stay_id" not in summary
