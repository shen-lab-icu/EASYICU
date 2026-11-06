"""vaso60 does not depend on which stays share a patient batch.

MIMIC-III's MetaVision rate source keeps ``linkorderid`` only while it has no
rows, so a batch without MetaVision rows hands vaso60 an all-NA infusion
column and a larger batch hands it none.  Rows without an infusion id must be
treated the same either way (lead-r13: epi60 differed between one-shot and
1,000-stay batches for 16 of 3,000 MIMIC-III sample stays).
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.concept.callbacks import ConceptCallbackContext, _callback_vaso60
from easyicu.table import ICUTable


def _ctx():
    return ConceptCallbackContext(
        concept_name="epi60",
        target=None,
        interval=pd.Timedelta(hours=1),
        resolver=None,
        data_source=SimpleNamespace(config=SimpleNamespace(name="mimic")),
        patient_ids=None,
        kwargs={},
    )


def _carevue_rate():
    # CareVue charting: rate changes at irregular times, no infusion id.
    return pd.DataFrame(
        {
            "icustay_id": 1,
            "charttime": [137.0, 138.0, 143.0, 144.0, 145.0, 146.0],
            "epi_rate": [0.0747, 0.0747, 0.0747, 0.0712, 0.0589, 0.3607],
            "rateuom": "mcg/kg/min",
        }
    )


def _duration():
    return pd.DataFrame(
        {"icustay_id": [1, 1], "charttime": [137.8167, 144.0167], "epi_dur": [6.2, 3.2833]}
    )


def _run(rate, duration=None):
    tables = {
        "epi_rate": ICUTable(rate, id_columns=["icustay_id"], index_column="charttime", value_column="epi_rate"),
        "epi_dur": ICUTable(
            _duration() if duration is None else duration,
            id_columns=["icustay_id"], index_column="charttime", value_column="epi_dur",
        ),
    }
    out = _callback_vaso60(tables, _ctx()).data
    return out.sort_values(list(out.columns[:2])).reset_index(drop=True)


def test_an_all_na_infusion_column_is_no_infusion_column():
    without = _run(_carevue_rate())
    with_na = _run(_carevue_rate().assign(linkorderid=np.nan))
    pd.testing.assert_frame_equal(with_na, without, check_dtype=False)
    assert not without.empty


@pytest.mark.parametrize("infusion_col", ["linkorderid", "orderid", "infusionid"])
def test_a_stay_with_infusions_does_not_change_a_stay_without(infusion_col):
    carevue = _carevue_rate()
    metavision = pd.DataFrame(
        {
            "icustay_id": 2,
            "charttime": [10.0, 10.5, 12.0],
            "epi_rate": [0.05, 0.08, 0.10],
            "rateuom": "mcg/kg/min",
            infusion_col: [7.0, 7.0, 8.0],
        }
    )
    durations = pd.concat(
        [_duration(), pd.DataFrame({"icustay_id": [2], "charttime": [10.0], "epi_dur": [3.0]})],
        ignore_index=True,
    )
    together = _run(pd.concat([carevue, metavision], ignore_index=True), durations)
    alone_1 = _run(carevue.assign(**{infusion_col: np.nan}), _duration())
    alone_2 = _run(metavision, durations[durations["icustay_id"] == 2].reset_index(drop=True))
    batched = pd.concat([alone_1, alone_2], ignore_index=True)
    batched = batched.sort_values(list(batched.columns[:2])).reset_index(drop=True)
    pd.testing.assert_frame_equal(together, batched, check_dtype=False)
