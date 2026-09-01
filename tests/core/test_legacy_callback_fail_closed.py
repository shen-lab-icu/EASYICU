from __future__ import annotations

import pandas as pd
import pytest

from easyicu.callbacks import UnsupportedClinicalScoreError, vaso60, vent_ind
from easyicu.utils.callback_utils import aumc_drug


@pytest.mark.parametrize(
    ("callback", "reason_code"),
    [
        (lambda: vaso60(pd.DataFrame({"norepi_rate": [0.1]}), pd.DataFrame()),
         "vaso60_duration_contract_not_implemented"),
        (lambda: vent_ind(pd.DataFrame({"vent_start": [1]}), pd.DataFrame()),
         "vent_ind_window_contract_not_implemented"),
    ],
)
def test_unimplemented_clinical_callbacks_fail_with_stable_reason(
    callback, reason_code: str
) -> None:
    with pytest.raises(UnsupportedClinicalScoreError) as caught:
        callback()

    assert caught.value.code == reason_code


def test_unimplemented_aumc_drug_callback_does_not_return_raw_doses() -> None:
    raw = pd.DataFrame({"itemid": [1], "value": [2.5], "unit": ["mg"]})

    with pytest.raises(NotImplementedError, match="dose-to-rate conversion"):
        aumc_drug(raw)
