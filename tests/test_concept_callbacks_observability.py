import logging

import pandas as pd

from easyicu.concept_callbacks import _match_fio2_fallback_loop_original


def test_merge_asof_fallback_logs_skipped_patient_groups(caplog):
    left = pd.DataFrame({"stay_id": [1], "charttime": ["bad-time"], "pao2": [80.0]})
    right = pd.DataFrame({"stay_id": [1], "charttime": ["bad-time"], "fio2": [40.0]})

    caplog.set_level(logging.WARNING, logger="easyicu.concept_callbacks")
    result = _match_fio2_fallback_loop_original(
        left,
        right,
        ["stay_id"],
        "charttime",
        "pao2",
        "fio2",
        pd.Timedelta(hours=1),
    )

    assert result.empty
    assert "merge_asof fallback skipped 1/1 patient groups" in caplog.text
