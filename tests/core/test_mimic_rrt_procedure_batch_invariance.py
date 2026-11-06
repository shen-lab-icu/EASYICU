"""CareVue RRT procedure hours do not depend on which stays share a batch.

``rrt_procedure_hours`` expands closed CareVue 582 courses and unclosed
starts separately. An empty expansion is untyped, so a batch whose stays had
only one kind of course used to publish ``charttime`` as object Timestamps,
while a batch with both kinds published datetime64. The MIMIC-III resolver
routes object time columns through a different clock conversion, so one stay's
RRT hours (and SOFA-2 renal) changed with its batch peers (lead-r13: 1 stay in
the MIMIC-III demo, batch 50 vs one-shot). Every record here is synthetic.
"""

import pandas as pd
import pytest

from easyicu.concept.mimic_support import other_vaso_points, rrt_procedure_hours
from easyicu.concept.schema import ConceptSource

T0 = pd.Timestamp("2100-01-01 08:00")
RRT = ConceptSource(table="chartevents", value_var="value", index_var="charttime", sub_var="itemid")
VASO_MV = ConceptSource(table="inputevents_mv", index_var="starttime", sub_var="itemid")


def _icu(ids):
    return pd.DataFrame({
        "icustay_id": ids,
        "intime": [T0 - pd.Timedelta(minutes=20)] * len(ids),
        "outtime": [T0 + pd.Timedelta(days=3)] * len(ids),
    })


def _chart():
    # Stay 1 has a closed hemodialysis course; stay 2 only an unclosed start.
    return pd.DataFrame({
        "icustay_id": [1, 1, 2],
        "charttime": [T0, T0 + pd.Timedelta(hours=5), T0 + pd.Timedelta(hours=1, minutes=30)],
        "value": ["Hemodialysis st", "Hemodialysis end", "Hemodialysis st"],
        "itemid": [582, 582, 582],
    })


def _rows(out, stay):
    return out.loc[out["icustay_id"].eq(stay)].sort_values("charttime").reset_index(drop=True)


@pytest.mark.parametrize("stay", [1, 2])
def test_rrt_hours_are_identical_alone_and_in_a_shared_batch(stay):
    chart = _chart()
    together = rrt_procedure_hours(chart, RRT, "rrt", icu_stays=_icu([1, 2]))
    alone = rrt_procedure_hours(chart.loc[chart["icustay_id"].eq(stay)], RRT, "rrt", icu_stays=_icu([stay]))

    assert pd.api.types.is_datetime64_any_dtype(alone["charttime"])
    pd.testing.assert_frame_equal(_rows(alone, stay), _rows(together, stay))


def test_rrt_without_any_course_keeps_the_source_clock():
    chart = _chart().assign(value="Hemodialysis end")
    out = rrt_procedure_hours(chart, RRT, "rrt", icu_stays=_icu([1, 2]))

    assert out.empty
    assert pd.api.types.is_datetime64_any_dtype(out["charttime"])
    assert out["rrt"].dtype == bool


def test_other_vaso_without_support_keeps_the_source_clock():
    frame = pd.DataFrame({
        "icustay_id": [1],
        "starttime": [T0],
        "endtime": [T0 + pd.Timedelta(hours=2)],
        "rate": [float("nan")],
        "amount": [float("nan")],
        "itemid": [221662],
    })
    out = other_vaso_points(frame, VASO_MV, "vaso", icu_stays=_icu([1]))

    assert out.empty
    assert pd.api.types.is_datetime64_any_dtype(out["starttime"])
