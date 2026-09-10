"""Executable contracts for repaired table facades and renal evidence alignment."""

import pandas as pd
import pytest

from easyicu.scores.sofa2 import sofa2_renal
from easyicu.scores.sofa2_validation import SOFA2InputError
from easyicu.table import IdTbl, TsTbl


@pytest.mark.parametrize(
    "evidence",
    [True, pd.Series([True], index=[99]), pd.Series([True, False], index=[1, 99])],
)
def test_oliguria_requires_exact_series_alignment(evidence):
    with pytest.raises(SOFA2InputError):
        sofa2_renal(crea=pd.Series([1.0], index=[1]), oliguria_gt6h=evidence)


@pytest.mark.parametrize(
    "value, score", [(True, 4), (False, 0), (pd.NA, 0), ("false", 0), ("true", 0)]
)
def test_oliguria_unknown_values_do_not_manufacture_rrt(value, score):
    def s(x):
        return pd.Series([x], index=[1])
    result = sofa2_renal(
        crea=s(1.0),
        uo_6h=s(1.0),
        potassium=s(6.5),
        ph=s(7.4),
        bicarb=s(24.0),
        oliguria_gt6h=s(value),
    )
    assert result.tolist() == [score]


def test_table_configuration_and_source_facades():
    import easyicu
    from easyicu.table.convert import as_col_cfg, as_src_cfg
    from easyicu.table.meta import default_vars
    from easyicu.datasource import ICUDataSource

    cfg = as_src_cfg("miiv")
    assert cfg.name == "miiv"
    assert isinstance(easyicu.new_src_tbl("patients", "miiv"), ICUDataSource)
    table = TsTbl(
        pd.DataFrame({"stay_id": [1], "time": [pd.Timedelta("1h")], "x": [2.0]}),
        id_vars="stay_id",
        index_var="time",
    )
    assert as_col_cfg(table) == {"id_var": "stay_id", "index_var": "time"}
    assert default_vars(table) == as_col_cfg(table)
    assert default_vars(cfg.get_table("icustays"))["id_var"] == "stay_id"


def test_table_time_facade_floors_without_mutating():
    from easyicu.table.utils import change_interval

    frame = pd.DataFrame({"time": pd.to_timedelta([0.5, 1.5], unit="h"), "x": [1, 2]})
    result = change_interval(frame, pd.Timedelta("1h"), "time")
    assert result.time.tolist() == pd.to_timedelta([0, 1], unit="h").tolist()
    assert frame.time.tolist() == pd.to_timedelta([0.5, 1.5], unit="h").tolist()


def test_id_map_facade_uses_declared_granular_table(monkeypatch):
    from easyicu.table.utils import id_map_helper
    from easyicu.datasource import ICUDataSource
    from easyicu.resources import load_data_sources

    source = ICUDataSource(load_data_sources().get("miiv"))
    calls = []

    def load(name, *, columns):
        calls.append((name, columns))
        return IdTbl(
            pd.DataFrame(
                {
                    "subject_id": [1, 1, 1],
                    "stay_id": [10, 11, 11],
                    "intime": [0.0, 3.0, 3.0],
                }
            ),
            id_vars="stay_id",
        )

    monkeypatch.setattr(source, "load_table", load)
    result = id_map_helper(source, "subject_id", "stay_id", index_col="intime")
    assert calls == [("icustays", ["subject_id", "stay_id", "intime"])]
    assert result.to_dict("list") == {
        "subject_id": [1, 1],
        "stay_id": [10, 11],
        "intime": [0.0, 3.0],
    }
    with pytest.raises(TypeError, match="ICUDataSource"):
        id_map_helper(source.config, "subject_id", "stay_id")
