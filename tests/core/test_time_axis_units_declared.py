"""Time-axis units and intervals are declared, never inferred from the values.

lead-r13 magnitude-inference sweep (time-axis family). Each former rule read a
quantile, a median or a minimum gap of the frame it was handed, so its answer
depended on which stays shared the batch:

* #2  ``_infer_numeric_time_unit`` set slide/expand window widths;
* #3  ``eicu_dex_inf`` took the smallest pooled infusion-offset gap as dur_var;
* #5  ``_aumc_admittedat_scale`` chose ms vs minutes from a median;
* #9  an env opt-in re-enabled a dur_var distribution guess;
* #10 KDIGO guessed a numeric axis unit from its name and spacing;
* #11 ``meta.interval`` fell back to a minimum gap or a 1 h default;
* #11b ``TsTbl``/``WinTbl`` derived ``interval`` from the rows, and the raw
  concept cache compared that value with each request;
* #12 the SOFA grid fell back to the median observed spacing;
* #13 ``vaso_ind`` read a datetime-typed duration as seconds or wiped it.

Synthetic rows only.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import easyicu
from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.concept.callbacks import (
    ScoreGridUndeclaredError,
    _callback_vaso_ind,
    _score_grid_and_window,
)
from easyicu.io.ts_utils import _declared_numeric_time_unit, expand, slide
from easyicu.scores.kdigo_aki import _resolve_time_unit
from easyicu.table import ICUTable, TsTbl, WinTbl
from easyicu.table.duration import DurationUnitError
from easyicu.table.meta import IntervalUndeclaredError, interval as table_interval, time_step, time_unit
from easyicu.utils.callback_utils import eicu_dex_inf
from tests.support.aumc_source import aumc_db, aumc_drug

SRC = Path(easyicu.__file__).parent
DATA = SRC / "data"


# --- #2: slide / expand -------------------------------------------------------

def _frame(stays, name):
    rows = []
    for stay, times in stays.items():
        for k, t in enumerate(times):
            rows.append({"stay": stay, name: float(t), "v": float(len(times) - k)})
    return pd.DataFrame(rows)


def test_slide_numeric_axis_is_hours_whatever_the_spacing():
    # Two values 100 apart on an *offset axis. The old rule read the 100-unit
    # spacing and took minutes, so a 24 h window (1,440 units) reached back to
    # 0. The declared concept-layer clock is hours: the window excludes it.
    data = _frame({1: [0, 100]}, "labresultoffset")
    out = slide(data, ["stay"], "labresultoffset", before=pd.Timedelta(hours=24), agg_func={"v": "max"})
    assert out.sort_values("labresultoffset")["v"].tolist() == [2.0, 1.0]


def test_slide_honours_an_explicit_or_declared_unit():
    data = _frame({1: [0, 100]}, "labresultoffset")
    explicit = slide(data, ["stay"], "labresultoffset", before=pd.Timedelta(hours=24),
                     agg_func={"v": "max"}, numeric_time_unit="min")
    # Declared minutes: the 24 h window (1,440 units) reaches back to 0.
    assert explicit.sort_values("labresultoffset")["v"].tolist() == [2.0, 2.0]
    declared = data.copy()
    declared.attrs["easyicu_time_columns"] = {"labresultoffset": "icu_relative_hours"}
    assert _declared_numeric_time_unit(declared, "labresultoffset") == "h"
    bad = data.copy()
    bad.attrs["easyicu_time_columns"] = {"labresultoffset": "fortnights"}
    with pytest.raises(ValueError, match="unknown declared time unit"):
        _declared_numeric_time_unit(bad, "labresultoffset")
    with pytest.raises(ValueError, match="numeric_time_unit must be"):
        _declared_numeric_time_unit(data, "labresultoffset", "days")


def test_slide_window_does_not_depend_on_the_batch():
    # 24 dense stays (spacing 1) take the bulk path; a sparse stay alone takes
    # the per-stay path. The old per-stay inference read spacing 30 as minutes
    # while the pooled one read hours.
    dense = {s: list(range(0, 30)) for s in range(1, 25)}
    sparse = {99: [0, 30, 60]}
    pooled = slide(_frame({**dense, **sparse}, "intakeoutputoffset"), ["stay"], "intakeoutputoffset",
                   before=pd.Timedelta(hours=24), agg_func={"v": "max"})
    alone = slide(_frame(sparse, "intakeoutputoffset"), ["stay"], "intakeoutputoffset",
                  before=pd.Timedelta(hours=24), agg_func={"v": "max"})
    key = ["stay", "intakeoutputoffset"]
    in_pool = pooled[pooled["stay"] == 99].sort_values(key).reset_index(drop=True)[key + ["v"]]
    pd.testing.assert_frame_equal(in_pool, alone.sort_values(key).reset_index(drop=True)[key + ["v"]],
                                  check_dtype=False)
    assert in_pool["v"].tolist() == [3.0, 2.0, 1.0]


def test_expand_numeric_axis_uses_the_declared_unit():
    win = pd.DataFrame({"stay": [1], "measuredat_minutes": [0.0], "stop": [3.0], "x": [1.0]})
    out = expand(win, start_var="measuredat_minutes", end_var="stop", step_size=pd.Timedelta(hours=1),
                 id_cols=["stay"], keep_vars=["x"], end_inclusive=False)
    assert sorted(out["measuredat_minutes"].astype(float)) == [0.0, 1.0, 2.0]


# --- #3: eICU dextrose infusion -------------------------------------------------

def _dex_rows(offsets):
    return pd.DataFrame({"patientunitstayid": [s for s, _ in offsets],
                         "infusionoffset": [o for _, o in offsets], "drugrate": 50.0})


def test_dex_infusion_duration_is_the_load_interval():
    frame = _dex_rows([(1, 0), (1, 20), (1, 40), (2, 0), (2, 60)])
    hourly = eicu_dex_inf(frame, "drugrate", "infusionoffset", interval=pd.Timedelta(hours=1))
    assert set(hourly["dur_var"]) == {60.0}
    assert set(eicu_dex_inf(frame, "drugrate", "infusionoffset")["dur_var"]) == {60.0}
    half = eicu_dex_inf(frame, "drugrate", "infusionoffset", interval=pd.Timedelta(minutes=30))
    assert set(half["dur_var"]) == {30.0}
    with pytest.raises(ValueError, match="positive load interval"):
        eicu_dex_inf(frame, "drugrate", "infusionoffset", interval=pd.Timedelta(0))


def test_dex_infusion_duration_does_not_depend_on_the_batch():
    pooled = eicu_dex_inf(_dex_rows([(1, 0), (1, 20), (2, 0), (2, 60)]), "drugrate", "infusionoffset",
                          interval=pd.Timedelta(hours=1))
    alone = eicu_dex_inf(_dex_rows([(2, 0), (2, 60)]), "drugrate", "infusionoffset",
                         interval=pd.Timedelta(hours=1))
    assert pooled.loc[pooled["patientunitstayid"] == 2, "dur_var"].tolist() == alone["dur_var"].tolist()


# --- #5: AUMC clock -------------------------------------------------------------

def test_aumc_clock_scale_is_declared_not_a_median():
    from easyicu.concept import __init__ as resolver_module  # noqa: F401

    import easyicu.concept as concept

    assert concept._AUMC_LOADED_UNITS_PER_HOUR == 60.0
    assert not hasattr(concept, "_aumc_admittedat_scale")


# --- #10: KDIGO -----------------------------------------------------------------

@pytest.mark.parametrize("name", ["charttime", "measuredat_minutes", "labresultoffset", "time"])
def test_kdigo_numeric_axis_needs_an_explicit_unit(name):
    series = pd.Series([0.0, 30.0, 60.0], name=name)
    with pytest.raises(ValueError, match="pass time_unit explicitly"):
        _resolve_time_unit(series, name, None)
    assert _resolve_time_unit(series, name, "min") == "minutes"
    assert _resolve_time_unit(series, name, "hours") == "hours"


# --- #11 / #11b: table intervals ------------------------------------------------

def _minute_rows():
    return pd.DataFrame({"stay": [1, 1, 2], "t": pd.to_timedelta([0, 1, 4], unit="m"), "x": [1.0, 2.0, 3.0]})


def test_tables_never_derive_an_interval_from_their_rows():
    ts = TsTbl(_minute_rows(), id_vars=["stay"], index_var="t")
    assert ts.interval is None
    win = WinTbl(_minute_rows().assign(d=pd.to_timedelta(5, unit="m")), id_vars=["stay"], index_var="t", dur_var="d")
    assert win.interval is None
    declared = TsTbl(_minute_rows(), id_vars=["stay"], index_var="t", interval=pd.Timedelta(hours=1))
    assert declared.interval == pd.Timedelta(hours=1)


def test_value_producing_readers_refuse_an_undeclared_interval():
    """Enforcing: reading an interval that was never declared fails typed."""
    ts = TsTbl(_minute_rows(), id_vars=["stay"], index_var="t")
    for reader in (table_interval, time_step, time_unit):
        with pytest.raises(IntervalUndeclaredError):
            reader(ts)
    declared = TsTbl(_minute_rows(), id_vars=["stay"], index_var="t", interval=pd.Timedelta(hours=1))
    assert table_interval(declared) == pd.Timedelta(hours=1)
    from easyicu.concept.merge_time_index import RELATIVE_HOURS, MergeTimeIndexError, merge_index_name

    a = ICUTable(pd.DataFrame({"stay": [1], "charttime": [0.0], "a": [1.0]}), id_columns=["stay"],
                 index_column="charttime", value_column="a")
    b = ICUTable(pd.DataFrame({"stay": [1], "starttime": [0.0], "b": [1.0]}), id_columns=["stay"],
                 index_column="starttime", value_column="b")
    a.time_basis = b.time_basis = RELATIVE_HOURS
    a.interval = pd.Timedelta(hours=1)
    with pytest.raises(MergeTimeIndexError, match="time_interval_unknown"):
        merge_index_name({"a": a, "b": b})


_TABLE_CLASSES = {"TsTbl", "WinTbl"}


def test_production_time_tables_declare_their_interval():
    """Enforcing: every TsTbl/WinTbl built in the package names its interval.

    A construction without ``interval=`` would now carry None, which the
    readers above refuse; this keeps a new producer from silently relying on
    a derived grid. ``table/`` itself (the classes and their copy/rbind
    helpers, which forward the source table's interval) is exempt.
    """
    missing = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith("table/"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
                if name in _TABLE_CLASSES and not any(k.arg == "interval" for k in node.keywords) and len(node.args) < 4:
                    missing.append(f"{rel}:{node.lineno}")
    assert missing == [], missing


# --- #11b: raw concept cache ----------------------------------------------------

def _vaso_source(tmp_path):
    return aumc_db(tmp_path, drugs=[aumc_drug(.25, 120, 7229), aumc_drug(5, 20, 19929), aumc_drug(40, 121, 19929)])


def _load(resolver, db, names, step):
    return resolver.load_concepts(
        names, db, merge=False, r_compatible=False, interval=step,
        patient_ids={"admissionid": [1]}, concept_workers=1, verbose=False,
    )


def _dictionary():
    dictionary = ConceptDictionary.from_json(DATA / "concept-dict.json")
    dictionary.update(ConceptDictionary.from_json(DATA / "sofa2-dict.json"))
    return dictionary


def _canonical(table):
    frame = table.data
    return frame.sort_values(list(frame.columns)).reset_index(drop=True)


def test_a_per_minute_raw_table_does_not_answer_an_hourly_request(tmp_path):
    db = _vaso_source(tmp_path)
    shared = ConceptResolver(_dictionary())
    minute = _load(shared, db, ["other_vaso"], pd.Timedelta(minutes=1))["other_vaso"]
    hourly = _load(shared, db, ["other_vaso"], pd.Timedelta(hours=1))["other_vaso"]
    fresh = _load(ConceptResolver(_dictionary()), db, ["other_vaso"], pd.Timedelta(hours=1))["other_vaso"]
    assert minute.interval == pd.Timedelta(minutes=1)
    assert hourly.interval == pd.Timedelta(hours=1)
    pd.testing.assert_frame_equal(_canonical(hourly), _canonical(fresh), check_exact=True)


@pytest.mark.parametrize("order", [["vaso_ind", "other_vaso"], ["other_vaso", "vaso_ind"]])
def test_raw_cache_reuse_equals_a_fresh_load(tmp_path, order):
    db = _vaso_source(tmp_path)
    together = _load(ConceptResolver(_dictionary()), db, order, pd.Timedelta(hours=1))
    for name in order:
        alone = _load(ConceptResolver(_dictionary()), db, [name], pd.Timedelta(hours=1))[name]
        pd.testing.assert_frame_equal(_canonical(together[name]), _canonical(alone), check_exact=True)


# --- #12: SOFA grid -------------------------------------------------------------

def test_score_grid_is_the_requested_interval_only():
    ctx = SimpleNamespace(interval=pd.Timedelta(hours=1), kwargs={})
    assert _score_grid_and_window(pd.Timedelta(hours=3), ctx, pd.Timedelta(hours=24)) == (
        pd.Timedelta(hours=1), pd.Timedelta(hours=23))
    with pytest.raises(ScoreGridUndeclaredError):
        _score_grid_and_window(pd.Timedelta(hours=3), SimpleNamespace(interval=None, kwargs={}),
                               pd.Timedelta(hours=24))


# --- #13: vaso_ind datetime duration ------------------------------------------

def test_vaso_ind_refuses_a_datetime_typed_duration():
    frame = pd.DataFrame({"stay_id": [1], "charttime": [0.0],
                          "dopa_dur": pd.to_datetime(["2000-01-01 02:00"])})
    tables = {"dopa_dur": ICUTable(frame, id_columns=["stay_id"], index_column="charttime",
                                   value_column="dopa_dur")}
    ctx = SimpleNamespace(concept_name="vaso_ind", target=None, interval=None, resolver=None,
                          data_source=SimpleNamespace(config=SimpleNamespace(name="miiv")),
                          patient_ids=None, kwargs={})
    with pytest.raises(DurationUnitError, match="datetime-typed duration"):
        _callback_vaso_ind(tables, ctx)


def test_no_value_derived_unit_helpers_remain():
    import easyicu.io.ts_utils as ts_utils
    import easyicu.utils.callback_utils as cu
    from easyicu.table import duration

    assert not hasattr(ts_utils, "_infer_numeric_time_unit")
    assert not hasattr(ts_utils, "infer_numeric_time_unit")
    assert not hasattr(cu, "_infer_interval_from_series")
    assert not hasattr(duration, "_heuristic_unit_is_hours")
    assert not hasattr(TsTbl, "_detect_interval")
    assert np.isfinite(60.0)
