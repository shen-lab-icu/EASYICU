"""Urine criteria must refer to one completely covered patient/time window."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.concept.callbacks import (
    ConceptCallbackContext,
    _callback_uo_window,
    _callback_sofa_component,
    _callback_rrt_criteria,
)
from easyicu.io.ts_utils import change_interval
from easyicu.scores.sofa2 import sofa2_renal
from easyicu.scores.urine_windows import assess_urine_windows, urine_evidence_columns
from easyicu.table import ICUTable


def evidence(times, amounts, *, interval=1, rate=False):
    urine = pd.DataFrame(
        {"stay_id": 1, "charttime": np.array(times, dtype=float), "urine": amounts}
    )
    weight = pd.DataFrame({"stay_id": [1], "weight": [1.0]})
    return assess_urine_windows(
        urine,
        weight,
        id_columns=["stay_id"],
        time_column="charttime",
        interval=pd.Timedelta(hours=interval),
        source_is_rate=rate,
    )


@pytest.mark.parametrize(
    "times,amounts,coverage",
    [
        (list(range(1, 24, 2)), [0.2] * 12, 6),
        (list(range(1, 7)) + list(range(20, 26)), [0.2] * 12, 6),
        (list(range(1, 13)), [0.0] * 11 + [np.nan], 11),
        (list(range(1, 13)) * 2, [0.0] * 24, 12),
    ],
)
def test_real_window_coverage(times, amounts, coverage):
    row = evidence(times, amounts).iloc[-1]
    assert row.uo_12h_covered_h == coverage
    assert pd.notna(row.uo_12h_assessment_rate) == (coverage == 12)


def test_nondividing_volume_bin_is_not_uniformly_apportioned():
    row = evidence([5, 10, 15], [2.75] * 3, interval=5).iloc[-1]
    assert row.uo_6h_covered_h == 6
    assert pd.isna(row.uo_6h_assessment_rate)
    assert row.uo_6h_assessment_reason == "partial_volume_bin"


def test_two_low_moving_averages_do_not_prove_low_union_rate():
    # Each endpoint 6 h average is .25, but the whole 7 h average is 3/7.
    frame = evidence(range(1, 8), [1.5, 0, 0, 0, 0, 0, 1.5])
    assert frame.uo_6h_assessment_rate.iloc[-2:].tolist() == [0.25, 0.25]
    assert not frame.uo_6h_oliguria_gt6h.iloc[-1]
    assert frame.uo_6h_oliguria_rate.iloc[-1] == pytest.approx(3 / 7)


def test_irregular_rate_chart_intervals_prove_eight_hours():
    result = evidence([0, 6, 8], [0.2, 0.2, 0.2], rate=True)
    assert result.uo_6h_oliguria_gt6h.tolist() == [False, False, True]
    assert result.uo_6h_oliguria_duration_h.iloc[-1] == 8
    assert result.uo_6h_assessment_rate.iloc[-1] == pytest.approx(0.2)


def tbl(name, values, times=None):
    values = list(values)
    return ICUTable(
        pd.DataFrame(
            {
                "stay_id": 1,
                "charttime": np.array(
                    times or list(range(1, len(values) + 1)), dtype=float
                ),
                name: values,
            }
        ),
        id_columns=["stay_id"],
        index_column="charttime",
        value_column=name,
    )


def ctx(name):
    return ConceptCallbackContext(
        concept_name=name,
        target="ts_tbl",
        patient_ids=None,
        resolver=SimpleNamespace(),
        data_source=SimpleNamespace(config=SimpleNamespace(name="miiv")),
        interval=pd.Timedelta(hours=1),
    )


@pytest.mark.parametrize(
    "amount,expected6,expected12", [(0.8, 0, 0), (0.0, 1, 3), (0.4, 1, 2)]
)
def test_callback_merge_resample_score_keeps_window_evidence(
    amount, expected6, expected12
):
    tables = {
        "urine": tbl("urine", [amount] * 24),
        "weight": ICUTable(
            pd.DataFrame({"stay_id": [1], "weight": [1.0]}),
            id_columns=["stay_id"],
            value_column="weight",
        ),
    }
    scoring = {"crea": tbl("crea", [1.0] * 24)}
    for hours in (6, 12, 24):
        name = f"uo_{hours}h"
        result = _callback_uo_window(tables.copy(), ctx(name), hours, name)
        scoring[name] = change_interval(
            result,
            interval=pd.Timedelta(hours=1),
            aggregation="sum",
            time_unit="hours",
            row_evidence_columns=urine_evidence_columns(name),
        )
    score = _callback_sofa_component(sofa2_renal)(scoring, ctx("sofa2_renal")).data
    assert score.loc[score.charttime == 6, "sofa2_renal"].item() == expected6
    assert score.loc[score.charttime == 12, "sofa2_renal"].item() == expected12
    assert (
        score.loc[score.charttime == 3, "sofa2_renal_available"].item() == 1
    )  # Independent creatinine.


def test_both_rrt_paths_require_more_than_six_hours():
    base = {
        "urine": tbl("urine", [0.2] * 8),
        "weight": ICUTable(
            pd.DataFrame({"stay_id": [1], "weight": [1.0]}),
            id_columns=["stay_id"],
            value_column="weight",
        ),
    }
    tables = {
        name: tbl(name, [value] * 8)
        for name, value in [
            ("crea", 1.0),
            ("potassium", 6.0),
            ("ph", 7.4),
            ("bicarb", 24.0),
            ("rrt", 0),
        ]
    }
    for hours in (6, 12, 24):
        name = f"uo_{hours}h"
        tables[name] = _callback_uo_window(base.copy(), ctx(name), hours, name)
    scored = _callback_sofa_component(sofa2_renal)(tables, ctx("sofa2_renal")).data
    rrt = _callback_rrt_criteria(tables, ctx("rrt_criteria")).data
    assert scored.sofa2_renal.tolist() == [0, 0, 0, 0, 0, 1, 4, 4]
    assert rrt.rrt_criteria.tolist() == [False] * 6 + [True] * 2


def test_missing_evidence_disables_urine_only():
    assert sofa2_renal(
        pd.Series([1.0, 4.0]), uo_12h=pd.Series([0.0, 0.0])
    ).tolist() == [0, 3]


def test_resampling_conflicting_evidence_fails_instead_of_sum():
    result = tbl("uo_6h", [0.1, 0.2], [1.0, 1.0])
    result.data["uo_6h_covered_h"] = [6.0, 6.0]
    with pytest.raises(ValueError, match="Conflicting window evidence"):
        change_interval(
            result,
            interval=pd.Timedelta(hours=1),
            aggregation="sum",
            time_unit="hours",
            row_evidence_columns=["uo_6h_covered_h"],
        )


def test_resolver_cold_and_warm_keep_patient_window_authority():
    from easyicu.concept import (
        ConceptDefinition,
        ConceptDictionary,
        ConceptResolver,
        ConceptSource,
    )
    from easyicu.config import DataSourceConfig
    from easyicu.datasource import ICUDataSource

    frames = {}
    definitions = {}
    configs = {}
    for name, values in {
        "urine": [0.0] * 24 + [80.0] * 24,
        "weight": [100.0] * 48,
        "crea": [1.0] * 48,
    }.items():
        frames[name] = pd.DataFrame(
            {
                "stay_id": [1] * 24 + [2] * 24,
                "charttime": list(range(1, 25)) * 2,
                name: values,
            }
        )
        configs[name] = {
            "defaults": {"id_var": "stay_id", "index_var": "charttime", "val_var": name}
        }
        definitions[name] = ConceptDefinition(
            name=name, target="id_tbl" if name == "weight" else "ts_tbl", sources={"unit": [ConceptSource(table=name, value_var=name)]}
        )
    for hours in (6, 12, 24):
        name = f"uo_{hours}h"
        definitions[name] = ConceptDefinition(
            name=name, sources={}, sub_concepts=["urine", "weight"], callback=name
        )
    definitions["sofa2_renal"] = ConceptDefinition(
        name="sofa2_renal",
        sources={},
        sub_concepts=["crea", "uo_6h", "uo_12h", "uo_24h"],
        callback="sofa2_renal",
    )
    resolver = ConceptResolver(ConceptDictionary(definitions))
    source = ICUDataSource(
        DataSourceConfig(name="unit", tables=configs), table_sources={name: (lambda frame=frame: frame.copy()) for name, frame in frames.items()}
    )
    outputs = []
    for _ in range(2):
        loaded = resolver.load_concepts(
            ["sofa2_renal"],
            source,
            merge=False,
            interval=pd.Timedelta(hours=1),
            r_compatible=False,
            verbose=False,
            concept_workers=1,
        )
        outputs.append(
            loaded["sofa2_renal"]
            .data.sort_values(["stay_id", "charttime"])
            .reset_index(drop=True)
        )
    pd.testing.assert_frame_equal(*outputs)
    result = outputs[0]
    assert result.loc[result.stay_id == 2, "sofa2_renal"].eq(0).all()
    assert (
        result.loc[
            (result.stay_id == 1) & (result.charttime == 6), "sofa2_renal"
        ].item()
        == 1
    )
    assert (
        result.loc[
            (result.stay_id == 1) & (result.charttime == 12), "sofa2_renal"
        ].item()
        == 3
    )


@pytest.mark.parametrize('dtype', ['Float64', 'float64[pyarrow]'])
def test_nullable_urine_hole_does_not_gain_coverage(dtype):
    urine = pd.DataFrame({'stay_id':[1]*12,'charttime':range(1,13),
                          'urine':pd.Series([0.]*11+[pd.NA], dtype=dtype)})
    result = assess_urine_windows(urine, pd.DataFrame({'stay_id':[1],'weight':[70.]}),
        id_columns=['stay_id'],time_column='charttime',interval=pd.Timedelta(hours=1))
    assert result.uo_12h_covered_h.iloc[-1] == 11.
    assert pd.isna(result.uo_12h_assessment_rate.iloc[-1])
