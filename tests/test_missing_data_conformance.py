from __future__ import annotations

import pandas as pd
import pytest

from easyicu.callbacks import safi


@pytest.mark.clinical_conformance
def test_safi_room_air_imputation_is_machine_readable() -> None:
    spo2 = pd.DataFrame(
        {"stay_id": [1, 2], "charttime": [0.0, 0.0], "spo2": [96.0, 96.0]}
    )
    fio2 = pd.DataFrame(
        {"stay_id": [1], "charttime": [0.0], "fio2": [40.0]}
    )

    result = safi(spo2, fio2, fix_na_fio2=True)

    by_stay = result.set_index("stay_id")
    assert bool(by_stay.loc[1, "fio2_observed"]) is True
    assert bool(by_stay.loc[1, "fio2_imputed"]) is False
    assert by_stay.loc[1, "fio2_assessment_reason"] == "observed"
    assert bool(by_stay.loc[2, "fio2_observed"]) is False
    assert bool(by_stay.loc[2, "fio2_imputed"]) is True
    assert by_stay.loc[2, "fio2_assessment_reason"] == "room_air_assumption"
