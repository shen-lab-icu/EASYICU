"""Ranked channels survive the AUMC/HiRID high-frequency fallback (audit #16).

Once a frame passes 1,000 rows the raw loader used to pre-sample each source
by median on the absolute database clock.  For a ranked-channel concept such
as MAP that moved values of a non-hour admission into the previous ICU hour
and turned a requested hourly minimum (SOFA cardiovascular) into a median.
"""

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
            "id_var": "admissionid", "index_var": "measuredat", "val_var": "value",
        }}},
    )

    def __init__(self, frame):
        self.frame = frame

    def load_table(self, table_name, columns=None, filters=None, verbose=False):
        if table_name == "admissions":
            # Stay 1 is admitted 20 minutes past a database-clock hour.
            return ICUTable(pd.DataFrame({"admissionid": [1, 2], "admittedat": [20., 0.]}),
                            id_columns=["admissionid"], index_column="admittedat")
        frame = self.frame.copy()
        for spec in filters or []:
            frame = spec.apply(frame)
        return ICUTable(frame, id_columns=["admissionid"],
                        index_column="measuredat", value_column="value")


def _definition():
    return ConceptDefinition(
        name="map", minimum=0, maximum=250,
        source_channel_priority=["invasive", "nibp"],
        sources={"aumc": [
            ConceptSource(table="events", sub_var="itemid", ids=[1],
                          params={"source_channel": "invasive"}),
            ConceptSource(table="events", sub_var="itemid", ids=[2],
                          params={"source_channel": "nibp"}),
        ]},
    )


@pytest.mark.parametrize("aggregator,expected_hour_one", [("min", 60.), ("median", 80.)])
def test_ranked_channels_keep_icu_hours_and_the_requested_aggregate(aggregator, expected_hour_one):
    # Minutes on the database clock; stay 1's ICU hour 1 is minutes 80-140.
    target = pd.DataFrame({
        "admissionid": [1] * 5,
        "measuredat": [30., 85., 90., 95., 100.],
        "itemid": [2, 1, 1, 1, 2],
        "value": [90., 60., 80., 100., 120.],
    })
    padding = pd.DataFrame({"admissionid": [2] * 1001, "measuredat": [1.] * 1001,
                            "itemid": [1] * 1001, "value": [70.] * 1001})
    outputs = []
    for frame in (target, pd.concat([target, padding], ignore_index=True)):
        result = ConceptResolver(ConceptDictionary({"map": _definition()})).load_concepts(
            ["map"], RawSource(frame), aggregate=aggregator, merge=False, r_compatible=False,
            interval=pd.Timedelta(hours=1), verbose=False, concept_workers=1,
        )["map"]
        data = result.data if hasattr(result, "data") else result
        outputs.append(data[data.admissionid == 1].sort_values("measuredat").reset_index(drop=True))
    pd.testing.assert_frame_equal(*outputs)
    # Hour 0: cuff only.  Hour 1: arterial wins over the cuff (120).
    assert outputs[1].measuredat.tolist() == [0., 1.]
    assert outputs[1]["map"].tolist() == [90., expected_hour_one]
