"""Row-filter argument handling in ``easyicu.io.data_load.load_src``."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from easyicu.config import DataSourceConfig
from easyicu.datasource import FilterOp, ICUDataSource
from easyicu.io.data_load import load_src


def test_load_src_accepts_mapping_filters_before_generic_iterables(monkeypatch) -> None:
    source = ICUDataSource(DataSourceConfig(name="unit"))
    captured: dict[str, object] = {}

    def fake_load_table(name, *, columns=None, filters=None, **_kwargs):
        captured.update(name=name, columns=columns, filters=filters)
        return SimpleNamespace(data=pd.DataFrame({"stay_id": [1, 2]}))

    monkeypatch.setattr(source, "load_table", fake_load_table)

    result = load_src(
        "events",
        src=source,
        rows={"stay_id": [1, 2], "site": "icu"},
    )

    filters = captured["filters"]
    assert result["stay_id"].tolist() == [1, 2]
    assert [(item.column, item.op, item.value) for item in filters] == [
        ("stay_id", FilterOp.IN, [1, 2]),
        ("site", FilterOp.EQ, "icu"),
    ]
