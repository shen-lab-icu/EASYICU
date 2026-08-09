from __future__ import annotations

import pandas as pd

from easyicu.concept.callbacks import ConceptCallbackContext, _callback_vaso60
from easyicu.table import ICUTable


class _Resolver:
    def __init__(self, refreshed: ICUTable) -> None:
        self.refreshed = refreshed
        self.calls: list[dict] = []

    def load_concepts(self, names, data_source, **kwargs):
        self.calls.append({"names": list(names), "data_source": data_source, **kwargs})
        return {names[0]: self.refreshed}


def _table(frame: pd.DataFrame, value: str) -> ICUTable:
    return ICUTable(
        frame,
        id_columns=["patientunitstayid"],
        index_column="charttime",
        value_column=value,
    )


def test_vaso60_uncached_reload_restores_missing_eicu_identifier() -> None:
    malformed_rate = _table(
        pd.DataFrame({"charttime": [0.0], "dobu_rate": [5.0]}),
        "dobu_rate",
    )
    refreshed_rate = _table(
        pd.DataFrame(
            {
                "patientunitstayid": [101],
                "charttime": [0.0],
                "dobu_rate": [5.0],
            }
        ),
        "dobu_rate",
    )
    duration = _table(
        pd.DataFrame(
            {
                "patientunitstayid": [101],
                "charttime": [0.0],
                "dobu_dur": [2.0],
            }
        ),
        "dobu_dur",
    )
    resolver = _Resolver(refreshed_rate)
    data_source = object()
    ctx = ConceptCallbackContext(
        concept_name="dobu60",
        target=None,
        interval=pd.Timedelta(hours=1),
        resolver=resolver,
        data_source=data_source,
        patient_ids={"patientunitstayid": [101]},
    )

    result = _callback_vaso60(
        {"dobu_rate": malformed_rate, "dobu_dur": duration},
        ctx,
    )

    assert resolver.calls[0]["names"] == ["dobu_rate"]
    assert resolver.calls[0]["_skip_concept_cache"] is True
    assert result.data["patientunitstayid"].tolist() == [101]
    assert result.data["dobu60"].tolist() == [5.0]
