from __future__ import annotations

import pandas as pd
import pytest

from tools.e3_strict_kdigo_window import (
    StrictKdigoWindowError,
    derive_strict_kdigo_window,
    strict_kdigo_summary,
)


def _strict_row(
    stay_id: int,
    charttime: float,
    *,
    stage: int | None,
    ascertainment: str,
    creat_stage: int | None = 0,
    uo_stage: int | None = 0,
    rrt_stage: int | None = 0,
    creat_status: str = "negative",
    uo_status: str = "negative",
    rrt_status: str = "negative",
) -> dict:
    return {
        "stay_id": stay_id,
        "charttime": charttime,
        "aki_stage": stage,
        "aki_ascertainment": ascertainment,
        "aki_stage_creat": creat_stage,
        "aki_stage_uo": uo_stage,
        "aki_stage_rrt": rrt_stage,
        "creatinine_ascertainment": creat_status,
        "urine_ascertainment": uo_status,
        "rrt_ascertainment": rrt_status,
    }


def _reference(*rows: tuple[int, float, int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"stay_id": stay_id, "charttime": time, "aki_stage_reference": stage}
            for stay_id, time, stage in rows
        ]
    )


def test_strict_window_does_not_turn_partial_evidence_into_stage_zero() -> None:
    strict = pd.DataFrame(
        [
            _strict_row(
                1,
                12,
                stage=0,
                ascertainment="partial_no_observed_positive",
                uo_stage=None,
                uo_status="indeterminate",
            ),
            _strict_row(2, 12, stage=0, ascertainment="negative_complete"),
            _strict_row(
                3,
                12,
                stage=2,
                ascertainment="positive",
                creat_stage=2,
                creat_status="positive",
                uo_stage=None,
                uo_status="indeterminate",
            ),
        ]
    )

    result = derive_strict_kdigo_window(
        strict,
        reference_profile=_reference((1, 12, 0), (2, 12, 0), (3, 12, 2)),
    ).set_index("stay_id")

    assert pd.isna(result.loc[1, "aki_stage_strict"])
    assert result.loc[1, "aki_stage_reference"] == 0
    assert result.loc[1, "aki_ascertainment"] == "indeterminate"
    assert result.loc[2, "aki_stage_strict"] == 0
    assert result.loc[2, "aki_ascertainment"] == "negative_complete"
    assert result.loc[3, "aki_stage_strict"] == 2
    assert result.loc[3, "aki_ascertainment"] == "positive"


def test_strict_window_uses_any_positive_and_ignores_rows_after_24_hours() -> None:
    strict = pd.DataFrame(
        [
            _strict_row(1, 1, stage=0, ascertainment="negative_complete"),
            _strict_row(
                1,
                20,
                stage=1,
                ascertainment="positive",
                uo_stage=1,
                uo_status="positive",
            ),
            _strict_row(
                1,
                30,
                stage=3,
                ascertainment="positive",
                rrt_stage=3,
                rrt_status="positive",
            ),
        ]
    )

    result = derive_strict_kdigo_window(
        strict,
        reference_profile=_reference((1, 1, 0), (1, 20, 1), (1, 30, 3)),
    )

    assert result["aki_stage_strict"].item() == 1
    assert result["aki_stage_reference"].item() == 1
    assert result["kidney_window_row_count"].item() == 2


def test_positive_authority_requires_a_positive_stage() -> None:
    strict = pd.DataFrame(
        [_strict_row(1, 12, stage=0, ascertainment="positive")]
    )

    with pytest.raises(StrictKdigoWindowError, match="requires a positive"):
        derive_strict_kdigo_window(strict)


def test_summary_reports_unknown_denominator_explicitly() -> None:
    strict = pd.DataFrame(
        [
            _strict_row(
                1,
                12,
                stage=0,
                ascertainment="partial_no_observed_positive",
                uo_stage=None,
                uo_status="indeterminate",
            ),
            _strict_row(2, 12, stage=0, ascertainment="negative_complete"),
        ]
    )
    result = derive_strict_kdigo_window(
        strict,
        reference_profile=_reference((1, 12, 0), (2, 12, 0)),
    )

    summary = strict_kdigo_summary(result)

    assert summary["rows"] == 2
    assert summary["strict_stage_counts"] == {"missing": 1, "0": 1}
    assert summary["public_reference_stage_counts"] == {"0": 2}
    assert summary["strict_missing_not_recoded_to_zero"] is True
