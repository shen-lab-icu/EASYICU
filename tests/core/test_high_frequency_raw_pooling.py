"""AUMC raw pooling must preserve bounds, patient clocks and source weights."""

import pandas as pd
import pytest

from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.concept.schema import ConceptDefinition, ConceptSource
from easyicu.config import DataSourceConfig
from easyicu.table import ICUTable


class RawSource:
    base_path = None
    config = DataSourceConfig(
        name="aumc",
        tables={"events": {"defaults": {
            "id_var": "admissionid", "index_var": "measuredat",
            "val_var": "value",
        }}},
    )

    def __init__(self, frame, admittedat=(0., 0.)):
        self.frame = frame
        self.admittedat = admittedat

    def load_table(self, table_name, columns=None, filters=None, verbose=False):
        if table_name == "admissions":
            return ICUTable(pd.DataFrame({"admissionid": [1, 2], "admittedat": self.admittedat}),
                            id_columns=["admissionid"], index_column="admittedat")
        frame = self.frame.copy()
        for spec in filters or []:
            frame = spec.apply(frame)
        return ICUTable(frame, id_columns=["admissionid"],
                        index_column="measuredat", value_column="value")


def extract(frame, *, bounded, multi_source):
    sources = [ConceptSource(table="events", sub_var="itemid", ids=[1])]
    if multi_source:
        sources.append(ConceptSource(table="events", sub_var="itemid", ids=[2]))
    definition = ConceptDefinition(
        name="tidal_vol", minimum=0 if bounded else None,
        maximum=2000 if bounded else None, sources={"aumc": sources},
    )
    result = ConceptResolver(ConceptDictionary({"tidal_vol": definition})).load_concepts(
        ["tidal_vol"], RawSource(frame), merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), verbose=False, concept_workers=1,
    )["tidal_vol"]
    return result.data if hasattr(result, "data") else result


@pytest.mark.parametrize("bounded,multi_source", [(True, False), (False, True), (True, True)])
def test_raw_pooling_is_invariant_to_unrelated_stays(bounded, multi_source):
    # Unequal source multiplicities distinguish pooled from median-of-medians.
    if bounded:
        items, values, expected = [1, 1], [0., 2849.], 0.
        if multi_source:
            items, values, expected = [1, 1, 2], [0., 2849., 100.], 50.
    else:
        items, values, expected = [1, 1, 1, 2], [0., 0., 0., 1000.], 0.
    target = pd.DataFrame({
        "admissionid": [1] * len(items), "measuredat": [float(i+1) for i in range(len(items))],
        "itemid": items, "value": values,
    })
    padding = pd.DataFrame({
        "admissionid": [2] * 1001, "measuredat": [1.] * 1001,
        "itemid": [1] * 1001, "value": [500.] * 1001,
    })
    small = extract(target, bounded=bounded, multi_source=multi_source)
    large = extract(pd.concat([target, padding], ignore_index=True),
                    bounded=bounded, multi_source=multi_source)
    small = small[small.admissionid == 1].reset_index(drop=True)
    large = large[large.admissionid == 1].reset_index(drop=True)
    pd.testing.assert_frame_equal(small, large)
    assert small.tidal_vol.tolist() == [expected]


@pytest.mark.parametrize(
    "admission,minutes,items,values,aggregator,expected",
    [
        # Both points belong to ICU hour zero despite crossing wall-clock hour.
        (20., [50., 70.], [1, 1], [40., 60.], "median", 50.),
        # Bounds apply to raw values, before aggregation.
        (0., [1., 2.], [1, 1], [0., 40.], "median", 40.),
        # Callback converts fraction to percent before pooling all sources.
        (0., [1., 2., 3., 4.], [1, 1, 1, 2], [40., 40., 40., .8], "median", 40.),
        # P/F requests max FiO2; a premature median destroys the maximum.
        (0., [1., 2., 3.], [1, 1, 1], [40., 40., 80.], "max", 80.),
    ],
)
def test_fio2_preserves_raw_clock_bounds_and_source_weights(
    admission, minutes, items, values, aggregator, expected
):
    definition = ConceptDefinition(
        name="fio2", minimum=21, maximum=100,
        sources={"aumc": [
            ConceptSource(table="events", sub_var="itemid", ids=[1]),
            ConceptSource(table="events", sub_var="itemid", ids=[2],
                          callback="transform_fun(percent_as_numeric)"),
        ]},
    )
    target = pd.DataFrame({
        "admissionid": [1] * len(items), "measuredat": minutes,
        "itemid": items, "value": values,
    })
    padding = pd.DataFrame({
        "admissionid": [2] * 1001, "measuredat": [1.] * 1001,
        "itemid": [1] * 1001, "value": [50.] * 1001,
    })
    outputs = []
    for frame in [target, pd.concat([target, padding], ignore_index=True)]:
        result = ConceptResolver(ConceptDictionary({"fio2": definition})).load_concepts(
            ["fio2"], RawSource(frame, admittedat=(admission, 0.)),
            aggregate=aggregator,
            merge=False, r_compatible=False, interval=pd.Timedelta(hours=1),
            verbose=False, concept_workers=1,
        )["fio2"].data
        outputs.append(result[result.admissionid == 1].reset_index(drop=True))
    pd.testing.assert_frame_equal(*outputs)
    assert outputs[0].measuredat.tolist() == [0.]
    assert outputs[0].fio2.tolist() == [expected]
