"""SICdb's own KDIGO_AKI_168 field is the native AKI label, at its window close.

2026-11-03 extraction audit 2 (#25): ``aki_source_native`` was empty for
every SICdb case although ``data_ref`` FieldID 3113 carries the stage.
"""

from __future__ import annotations

import pandas as pd

from easyicu.scores.aki_profiles import build_renal_aki_bundle
from easyicu.scores.sicdb_native_aki import sicdb_native_case_labels


def _sources():
    data_ref = pd.DataFrame(
        {
            "CaseID": [1, 2, 3, 4, 4, 5],
            "FieldID": [3113, 3113, 3113, 3113, 3113, 3099],
            # 3115..3118 = stage 0..3; case 4 has two different stages.
            "RefID": [3117, 3115, 3118, 3116, 3117, 2302],
        }
    )
    cases = pd.DataFrame(
        {
            "CaseID": [1, 2, 3, 4, 5],
            # seconds: 10 days, 30.5 hours, 5 days, 2 days, 2 days
            "TimeOfStay": [864000, 109800, 432000, 172800, 172800],
        }
    )
    return data_ref, cases


def test_field_3113_reads_the_documented_stage_and_window_close():
    labels = sicdb_native_case_labels(*_sources()).set_index("CaseID")

    assert labels["aki_stage_source_native"].to_dict() == {1: 2, 2: 0, 3: 3}
    # 168 h for a long stay, the discharge hour for a shorter one.
    assert labels["window_close_hours"].to_dict() == {1: 168.0, 2: 30.0, 3: 120.0}


def test_native_label_lands_on_its_window_close_hour_only():
    creatinine = pd.DataFrame(
        {"CaseID": [1, 1, 2], "Offset": [0.0, 1.0, 0.0], "crea": [1.0, 4.5, 1.0]}
    )
    result = build_renal_aki_bundle(
        "sic",
        crea_df=creatinine,
        id_col="CaseID",
        time_col="Offset",
        time_unit="hours",
        native_case_labels=sicdb_native_case_labels(*_sources()),
    )

    labelled = result[result["aki_stage_source_native"].notna()]
    assert labelled[["CaseID", "Offset"]].values.tolist() == [[1, 168.0], [2, 30.0], [3, 120.0]]
    assert labelled["aki_stage_source_native"].tolist() == [2, 0, 3]
    assert labelled["aki_source_native"].tolist() == [True, False, True]
    assert labelled["aki_severe_source_native"].tolist() == [True, False, True]
    early = result[result["Offset"] < 30]
    assert early["aki_stage_source_native"].isna().all()
    assert early["aki_source_native"].isna().all()
    assert result["aki_source_native_time_scale"].eq(
        "CASE_LEVEL_MAXIMUM_STAGE_OVER_FIRST_168H"
    ).all()
    assert result["aki_source_native_uses_future"].all()
    assert result["aki_source_native_profile"].eq(
        "SICDB_NATIVE_KDIGO_AKI_168_44A27CC8"
    ).all()
