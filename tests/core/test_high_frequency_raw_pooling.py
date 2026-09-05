"""AUMC tidal-volume results cannot depend on the resampling row threshold."""

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

    def __init__(self, frame):
        self.frame = frame

    def load_table(self, table_name, columns=None, filters=None, verbose=False):
        if table_name == "admissions":
            return ICUTable(pd.DataFrame({"admissionid": [1, 2], "admittedat": [0., 0.]}),
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
