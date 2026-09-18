from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept.callbacks import (
    ConceptCallbackContext,
    _callback_rrt_criteria,
)
from easyicu.table import ICUTable


def _table(name: str, values: list[object]) -> ICUTable:
    return ICUTable(
        pd.DataFrame(
            {
                "stay_id": [1, 2],
                "charttime": [0.0, None],
                name: values,
            }
        ),
        id_columns=["stay_id"],
        index_column="charttime",
        value_column=name,
    )


def _context() -> ConceptCallbackContext:
    return ConceptCallbackContext(
        concept_name="rrt_criteria",
        target=None,
        interval=pd.Timedelta(hours=1),
        resolver=None,
        data_source=SimpleNamespace(config=SimpleNamespace(name="eicu")),
        patient_ids=None,
    )


def _tables(*, positive_null_time: bool = False) -> dict[str, ICUTable]:
    return {
        "crea": _table("crea", [2.0, 2.0 if positive_null_time else None]),
        "potassium": _table(
            "potassium", [6.0, 6.0 if positive_null_time else None]
        ),
        "ph": _table("ph", [7.4, None]),
        "bicarb": _table("bicarb", [24.0, None]),
        "rrt": _table("rrt", [False, False]),
        "uo_6h": _table("uo_6h", [1.0, None]),
        "uo_12h": _table("uo_12h", [1.0, None]),
        "uo_24h": _table("uo_24h", [1.0, None]),
    }


def test_rrt_criteria_drops_false_outer_merge_row_without_time() -> None:
    result = _callback_rrt_criteria(_tables(), _context())

    assert result.index_column == "charttime"
    assert result.data[["stay_id", "charttime", "rrt_criteria"]].to_dict(
        "records"
    ) == [{"stay_id": 1, "charttime": 0.0, "rrt_criteria": True}]


def test_rrt_criteria_rejects_positive_result_without_time() -> None:
    with pytest.raises(ValueError, match="positive rows without an event time"):
        _callback_rrt_criteria(_tables(positive_null_time=True), _context())


def test_rrt_criteria_is_longitudinal_not_stay_level() -> None:
    """2026-09-19 v6: rrt_criteria must stay timed at the producer layer.

    eICU staging carried 2 untimed TRUE rows (stays 560871/1073908, all
    other renal columns NULL) plus 843 untimed FALSE merge artifacts. The
    timed labs behind them (e.g. 560871 crea 6.48/k 6.5 at 28 h, upper
    25.86 h from discharge 112 min + 24 h) are outside the ICU episode and
    are dropped at publication as out-of-window; collapsing the survivors
    via id_tbl ANY to a stay-level TRUE resurrects them as untimed events
    that trip the publication fail-closed. Declaring rrt_criteria
    longitudinal (no id_tbl target) keeps out-of-window TRUE timed so the
    publication window drops it as excluded_rows instead of raising, while
    the publication fail-closed itself stays intact for genuine bugs.
    """

    import json
    from pathlib import Path

    for dict_name in ("concept-dict.json", "sofa2-dict.json"):
        path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "easyicu"
            / "data"
            / dict_name
        )
        with open(path) as handle:
            entry = json.load(handle).get("rrt_criteria", {})
        assert entry.get("target") != "id_tbl", (
            f"{dict_name} rrt_criteria must not aggregate to stay-level"
        )
        assert entry.get("callback") == "rrt_criteria"
