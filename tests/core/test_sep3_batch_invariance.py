"""Sepsis-3 labels do not depend on which stays share a batch.

Streamed exports derive SOFA-1 and SOFA-2 Sepsis-3 per patient batch; a one-shot
load derives them for the whole cohort at once.  Each stay's onset must be the
same either way, including stays whose 48 h look-back before suspected
infection reaches into unscored pre-admission hours (lead-r13).
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept.callbacks import (
    ConceptCallbackContext,
    _callback_sep3,
    _callback_sep3_sofa2,
)
from easyicu.table import ICUTable

NAN = float("nan")


def _context(name):
    return ConceptCallbackContext(
        concept_name=name,
        target=None,
        interval=None,
        resolver=None,
        data_source=SimpleNamespace(config=SimpleNamespace(name="miiv")),
        patient_ids=None,
        kwargs={},
    )


def _stay(stay_id, scores, start=-10.0):
    return pd.DataFrame(
        {
            "stay_id": stay_id,
            "charttime": [start + h for h in range(len(scores))],
            "score": scores,
        }
    )


def _cohort():
    # 1: unscored pre-admission hours, SI 2 h after admission (look-back
    #    window starts 46 h before admission); rise at admission.
    # 2: scored throughout, SI late; rise inside the window 30 h before SI.
    # 3: unscored hours inside the window and a rise after SI (+20 h).
    # 4: no suspected infection.
    scores = pd.concat(
        [
            _stay(1, [NAN] * 10 + [3.0] * 12),
            _stay(2, [1.0] * 40 + [4.0] * 30, start=0.0),
            _stay(3, [0.0] * 5 + [NAN] * 10 + [1.0] * 30 + [5.0] * 10, start=-5.0),
            _stay(4, [0.0] * 5 + [6.0] * 5, start=0.0),
        ],
        ignore_index=True,
    )
    susp = pd.DataFrame(
        {"stay_id": [1, 2, 3], "charttime": [2.0, 60.0, 20.0], "susp_inf": [True] * 3}
    )
    return scores, susp


def _run(callback, score_name, scores, susp, stays):
    frame = scores[scores["stay_id"].isin(stays)].rename(columns={"score": score_name})
    tables = {
        score_name: ICUTable(
            frame.reset_index(drop=True), id_columns=["stay_id"],
            index_column="charttime", value_column=score_name,
        ),
        "susp_inf": ICUTable(
            susp[susp["stay_id"].isin(stays)].reset_index(drop=True),
            id_columns=["stay_id"], index_column="charttime", value_column="susp_inf",
        ),
    }
    concept = "sep3" if score_name == "sofa" else "sep3_sofa2"
    data = callback(tables, _context(concept)).data
    value = [c for c in data.columns if c not in ("stay_id", "charttime")][0]
    data = data[data[value].fillna(False).astype(bool)]
    return data[["stay_id", "charttime"]].sort_values(["stay_id", "charttime"]).reset_index(drop=True)


@pytest.mark.parametrize(
    "callback,score_name", [(_callback_sep3, "sofa"), (_callback_sep3_sofa2, "sofa2")]
)
@pytest.mark.parametrize("batches", [[[1, 2, 3, 4]], [[1], [2], [3], [4]], [[1, 3], [2, 4]], [[4, 3], [2, 1]]])
def test_onsets_do_not_depend_on_batch_membership(callback, score_name, batches):
    scores, susp = _cohort()
    one_shot = _run(callback, score_name, scores, susp, [1, 2, 3, 4])
    batched = pd.concat(
        [_run(callback, score_name, scores, susp, stays) for stays in batches],
        ignore_index=True,
    ).sort_values(["stay_id", "charttime"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(batched, one_shot, check_dtype=False)
    assert set(one_shot["stay_id"]) <= {1, 2, 3}


def test_sofa2_unscored_look_back_hours_count_as_zero_in_every_batch():
    scores, susp = _cohort()
    for stays in ([1], [1, 2, 3, 4]):
        onset = _run(_callback_sep3_sofa2, "sofa2", scores, susp, stays)
        assert onset.loc[onset["stay_id"] == 1, "charttime"].tolist() == [0.0]
