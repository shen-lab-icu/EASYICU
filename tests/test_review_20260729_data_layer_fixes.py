"""Regressions for the 2026-07-29 external review of the data layer.

Every test here drives the real public function against a real
``DataSourceConfig`` from ``data-sources.json``, so the id-system wiring under
test is the one production reads. The tables are synthetic because the point is
the loader's arithmetic and its refusals, not the contents of any one database;
the column sets are taken from the config so the loader's own column
resolution runs unmodified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd
import pytest

from easyicu.datasource import ICUDataSource
from easyicu.io.data_load import (
    TimeOriginError,
    load_difftime,
    load_id,
    load_ts,
    load_win,
)
from easyicu.resources import load_data_sources
from easyicu.table import (
    IdMapRelationError,
    IdTbl,
    KeyAlignmentError,
    TsTbl,
    WinTbl,
    cbind_tbl,
    change_id,
    classify_id_relation,
)


def _stamps(*values: str) -> pd.Series:
    return pd.to_datetime(list(values), format="%Y-%m-%d %H:%M:%S")


def _source(
    name: str, tables: Mapping[str, Dict[str, Any]], base: Path
) -> ICUDataSource:
    """A data source of ``name`` backed by parquet files holding ``tables``.

    Columns the config declares but the test does not care about are filled
    with nulls so the loader's own column selection runs unchanged.
    """

    config = load_data_sources().get(name)
    assert config is not None, name
    source = ICUDataSource(config, base_path=base)
    for table, rows in tables.items():
        declared = list(config.get_table(table).columns.keys())
        frame = pd.DataFrame(rows)
        for column in declared:
            if column not in frame.columns:
                frame[column] = pd.Series([None] * len(frame), dtype="object")
        path = base / f"{table}.parquet"
        frame.to_parquet(path)
        source.register_table_source(table, path)
    return source


def _miiv(base: Path) -> ICUDataSource:
    return _source(
        "miiv",
        {
            "icustays": {
                "subject_id": [1, 1, 2],
                "hadm_id": [10, 10, 20],
                "stay_id": [100, 101, 200],
                "intime": _stamps(
                    "2180-01-01 00:00:00",
                    "2180-02-01 00:00:00",
                    "2180-03-01 00:00:00",
                ),
                "outtime": _stamps(
                    "2180-01-05 00:00:00",
                    "2180-02-05 00:00:00",
                    "2180-03-05 00:00:00",
                ),
            },
            "chartevents": {
                "subject_id": [1, 1, 2],
                "hadm_id": [10, 10, 20],
                "stay_id": [100, 100, 200],
                "charttime": _stamps(
                    "2180-01-01 06:00:00",
                    "2180-01-01 12:00:00",
                    "2180-03-01 03:00:00",
                ),
                "itemid": [220045, 220045, 220045],
                "valuenum": [80.0, 85.0, 90.0],
            },
        },
        base,
    )


# --------------------------------------------------------------------------
# P0-1  load_difftime / load_id / load_ts / load_win
# --------------------------------------------------------------------------


def test_absolute_timestamps_become_relative_to_icu_admission(tmp_path: Path) -> None:
    """The conversion the function is named for actually happens.

    The previous body set ``origin_df = None`` behind a "will be implemented
    later" note, so ``charttime`` came back as the calendar date it started as
    — the same name, the same dtype a converted column would have, and no
    signal anywhere that the clock had not been moved.
    """

    source = _miiv(tmp_path)

    table = load_difftime(
        "chartevents", src=source, id_hint="stay_id", time_vars=["charttime"]
    )

    assert table.data["charttime"].tolist() == [
        pd.Timedelta(hours=6),
        pd.Timedelta(hours=12),
        pd.Timedelta(hours=3),
    ]
    assert table.id_vars == ["stay_id"]


def test_a_source_with_no_declared_origin_is_refused(tmp_path: Path) -> None:
    source = _miiv(tmp_path)

    with pytest.raises(TimeOriginError) as excinfo:
        load_difftime(
            "chartevents", src=source, id_hint="itemid", time_vars=["charttime"]
        )

    assert "itemid" in str(excinfo.value)


def test_a_numeric_offset_needs_its_unit_declared(tmp_path: Path) -> None:
    """eICU's offsets are bare numbers; guessing their unit is the 60x bug.

    ``ts_utils`` carries a scar from exactly this: a name-based rule read
    already-hourly values as minutes and shrank the KDIGO urine-output window
    sixty-fold. So a numeric time column without a declared unit is refused
    rather than assumed to be minutes, which is what the old exception branch
    did.
    """

    source = _source(
        "eicu",
        {
            "patient": {
                "patientunitstayid": [11, 12],
                "patienthealthsystemstayid": [1, 2],
                "unitadmitoffset": [0, -60],
                "hospitaladmitoffset": [-120, -240],
                "hospitaldischargeoffset": [3000, 4000],
                "unitdischargeoffset": [1440, 2880],
            },
            "vitalperiodic": {
                "patientunitstayid": [11, 11, 12],
                "observationoffset": [30, 90, 0],
                "heartrate": [80, 85, 90],
            },
        },
        tmp_path,
    )

    with pytest.raises(TimeOriginError) as excinfo:
        load_difftime(
            "vitalperiodic",
            src=source,
            id_hint="patientunitstayid",
            time_vars=["observationoffset"],
        )
    assert "time_unit" in str(excinfo.value)

    table = load_difftime(
        "vitalperiodic",
        src=source,
        id_hint="patientunitstayid",
        time_vars=["observationoffset"],
        time_unit="minutes",
    )
    assert table.data["observationoffset"].tolist() == [
        pd.Timedelta(minutes=30),
        pd.Timedelta(minutes=90),
        pd.Timedelta(minutes=60),
    ]


def test_load_ts_returns_a_series_that_knows_whose_it_is(tmp_path: Path) -> None:
    """``as_ts_tbl`` requires ``id_vars``; the call site omitted it entirely."""

    source = _miiv(tmp_path)

    table = load_ts(
        "chartevents",
        src=source,
        id_var="stay_id",
        index_var="charttime",
        time_vars=["charttime"],
    )

    assert isinstance(table, TsTbl)
    assert table.id_vars == ["stay_id"]
    assert table.index_var == "charttime"


def test_load_win_carries_the_id_and_index_into_the_window(tmp_path: Path) -> None:
    source = _source(
        "miiv",
        {
            "icustays": {
                "subject_id": [1],
                "hadm_id": [10],
                "stay_id": [100],
                "intime": _stamps("2180-01-01 00:00:00"),
                "outtime": _stamps("2180-01-05 00:00:00"),
            },
            "procedureevents": {
                "subject_id": [1, 1],
                "hadm_id": [10, 10],
                "stay_id": [100, 100],
                "starttime": _stamps("2180-01-01 06:00:00", "2180-01-02 06:00:00"),
                "endtime": _stamps("2180-01-01 12:00:00", "2180-01-02 12:00:00"),
                "itemid": [225792, 225792],
                "value": [360.0, 360.0],
            },
        },
        tmp_path,
    )

    table = load_win(
        "procedureevents",
        src=source,
        id_var="stay_id",
        index_var="starttime",
        dur_var="value",
        time_vars=["starttime"],
        duration_unit="minutes",
    )

    assert isinstance(table, WinTbl)
    assert table.id_vars == ["stay_id"]
    assert table.index_var == "starttime"
    assert table.dur_var == "value"
    assert table.dur_unit == "minutes"


def test_load_id_moves_between_id_systems(tmp_path: Path) -> None:
    """``change_id(tbl, id_var)`` could never have run: it takes four arguments.

    The map now comes from the id-system table that carries both identifiers.
    """

    source = _miiv(tmp_path)

    table = load_id(
        "chartevents",
        src=source,
        id_var="hadm_id",
        time_vars=["charttime"],
        agg_funcs={"itemid": "first", "valuenum": "mean"},
    )

    assert isinstance(table, IdTbl)
    assert table.id_vars == ["hadm_id"]
    assert sorted(table.data["hadm_id"].tolist()) == [10, 20]


def test_collapsing_stays_will_not_average_a_code_or_a_score(
    tmp_path: Path,
) -> None:
    """P1-4: "numeric" was taken to mean "safe to average".

    Two ICU stays of one admission collapse into one row. ``itemid`` is a
    category code — the mean of two item ids is not an item — and the same
    default would have averaged a SOFA component or a 0/1 mortality flag.
    """

    source = _miiv(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        load_id("chartevents", src=source, id_var="hadm_id", time_vars=["charttime"])

    assert "itemid" in str(excinfo.value)
    assert "agg_funcs" in str(excinfo.value)


def test_load_id_leaves_a_table_that_already_has_the_id_alone(
    tmp_path: Path,
) -> None:
    source = _miiv(tmp_path)

    table = load_id(
        "chartevents", src=source, id_var="stay_id", time_vars=["charttime"]
    )

    assert table.id_vars == ["stay_id"]
    assert len(table.data) == 3


# --------------------------------------------------------------------------
# P0-2  cbind_tbl
# --------------------------------------------------------------------------


def _id_tbl(**columns: Any) -> IdTbl:
    return IdTbl(pd.DataFrame(columns), id_vars="stay_id")


def test_binding_differently_ordered_tables_is_refused() -> None:
    """The silent patient mix-up this whole check exists for.

    Both inputs are legitimate typed tables over the same two stays. Bound by
    row position, row 0 held stay 1's exposure next to stay 2's outcome, and
    the result was a well-formed ``IdTbl`` that no downstream check could
    distinguish from a correct one.
    """

    left = _id_tbl(stay_id=[1, 2], exposure=[0, 1])
    right = _id_tbl(stay_id=[2, 1], outcome=[9, 8])

    with pytest.raises(KeyAlignmentError) as excinfo:
        cbind_tbl(left, right)

    assert "stay_id" in str(excinfo.value)


def test_binding_aligned_tables_keeps_one_copy_of_the_keys() -> None:
    left = _id_tbl(stay_id=[1, 2], exposure=[0, 1])
    right = _id_tbl(stay_id=[1, 2], outcome=[8, 9])

    out = cbind_tbl(left, right)

    assert list(out.data.columns) == ["stay_id", "exposure", "outcome"]
    assert out.data["outcome"].tolist() == [8, 9]


def test_a_typed_table_without_the_key_cannot_be_bound() -> None:
    left = _id_tbl(stay_id=[1, 2], exposure=[0, 1])
    right = IdTbl(
        pd.DataFrame({"other_id": [1, 2], "outcome": [8, 9]}), id_vars="other_id"
    )

    with pytest.raises(KeyAlignmentError):
        cbind_tbl(left, right)


def test_a_bare_frame_of_derived_columns_still_binds() -> None:
    """The ricu use this function exists for keeps working."""

    left = _id_tbl(stay_id=[1, 2], exposure=[0, 1])
    derived = pd.DataFrame({"risk": [0.2, 0.7]})

    out = cbind_tbl(left, derived)

    assert out.data["risk"].tolist() == [0.2, 0.7]


def test_a_bare_frame_of_the_wrong_length_is_refused() -> None:
    left = _id_tbl(stay_id=[1, 2], exposure=[0, 1])

    with pytest.raises(KeyAlignmentError):
        cbind_tbl(left, pd.DataFrame({"risk": [0.2]}))


def test_mismatched_row_labels_are_refused_not_padded() -> None:
    """``pd.concat`` joins on the index, so this used to produce NaN rows."""

    left = _id_tbl(stay_id=[1, 2], exposure=[0, 1])
    right = pd.DataFrame({"risk": [0.2, 0.7]}, index=[5, 6])

    with pytest.raises(KeyAlignmentError):
        cbind_tbl(left, right)


def test_duplicate_value_columns_are_rejected_by_default() -> None:
    left = _id_tbl(stay_id=[1, 2], value=[0, 1])
    right = _id_tbl(stay_id=[1, 2], value=[8, 9])

    with pytest.raises(ValueError, match="Duplicate column names"):
        cbind_tbl(left, right)


# --------------------------------------------------------------------------
# P0-3  change_id
# --------------------------------------------------------------------------


def test_equal_cardinality_does_not_mean_one_to_one() -> None:
    """``A→X, A→Y, B→X, B→Y``: two on each side, and not a function either way.

    The old direction test compared ``nunique`` on each side, read this as
    one-to-one, and mapped it with ``dict(zip(...))`` — which keeps one target
    per source and drops the rest without a word.
    """

    id_map = pd.DataFrame(
        {"from_id": ["A", "A", "B", "B"], "to_id": ["X", "Y", "X", "Y"]}
    )

    assert classify_id_relation(id_map, "from_id", "to_id") == "many_to_many"

    with pytest.raises(IdMapRelationError) as excinfo:
        change_id(
            pd.DataFrame({"from_id": ["A", "B"], "value": [1, 2]}),
            id_map,
            "from_id",
            "to_id",
        )

    assert "on_many_to_many" in str(excinfo.value)


def test_many_to_many_runs_once_a_strategy_is_named() -> None:
    id_map = pd.DataFrame(
        {"from_id": ["A", "A", "B", "B"], "to_id": ["X", "Y", "X", "Y"]}
    )
    data = pd.DataFrame({"from_id": ["A", "B"], "value": [1, 2]})

    expanded = change_id(data, id_map, "from_id", "to_id", on_many_to_many="expand")

    assert len(expanded) == 4


@pytest.mark.parametrize(
    ("pairs", "expected"),
    [
        ({"from_id": ["A", "B"], "to_id": ["X", "Y"]}, "one_to_one"),
        ({"from_id": ["A", "A"], "to_id": ["X", "Y"]}, "one_to_many"),
        ({"from_id": ["A", "B"], "to_id": ["X", "X"]}, "many_to_one"),
    ],
)
def test_the_other_three_relations_are_named_correctly(
    pairs: Dict[str, Any], expected: str
) -> None:
    assert classify_id_relation(pd.DataFrame(pairs), "from_id", "to_id") == expected


def test_a_one_to_one_change_still_maps_values() -> None:
    id_map = pd.DataFrame({"from_id": ["A", "B"], "to_id": ["X", "Y"]})

    out = change_id(
        pd.DataFrame({"from_id": ["A", "B"], "value": [1, 2]}),
        id_map,
        "from_id",
        "to_id",
    )

    assert out["to_id"].tolist() == ["X", "Y"]
    assert "from_id" not in out.columns


# --------------------------------------------------------------------------
# P1-3  dtype compression must not change reported numbers
# --------------------------------------------------------------------------


def test_statistical_columns_keep_their_precision() -> None:
    """float32 carries about seven significant digits.

    That is ample for a heart rate and not for ``p = 3.2e-9``, a CI bound
    compared against a threshold, or a coefficient that has to match the R
    implementation. The blanket downcast changed reported numbers on the way
    to the manuscript.
    """

    import numpy as np

    from easyicu.api import _compress_dtypes

    frame = pd.DataFrame(
        {
            "p_value": [3.2e-9, 1.1e-12],
            "auroc": [0.8123456789, 0.7654321098],
            "ci_lower": [0.8012345678, 0.7512345678],
            "risk": [0.5000000001, 0.4999999999],
        }
    )
    original = frame.copy()

    out = _compress_dtypes(frame)

    for column in original.columns:
        assert out[column].dtype == np.float64, column
        assert out[column].tolist() == original[column].tolist()


def test_a_measurement_column_is_still_compressed() -> None:
    """The memory saving this exists for keeps working."""

    import numpy as np

    from easyicu.api import _compress_dtypes

    out = _compress_dtypes(pd.DataFrame({"creatinine": [1.234567, 2.345678]}))

    assert out["creatinine"].dtype == np.float32


# --------------------------------------------------------------------------
# P1-5  an error message is not a place to put a patient record
# --------------------------------------------------------------------------


def test_the_window_guard_names_the_fields_but_not_the_values() -> None:
    from easyicu.api import WindowExpansionError, _guard_window_expansion

    with pytest.raises(WindowExpansionError) as excinfo:
        _guard_window_expansion(
            10_001,
            concept_name="vent_ind",
            duration=525600.0,
            unit="minutes",
            row={"stay_id": 30042318, "charttime": "2180-01-01T12:30:00"},
        )

    message = str(excinfo.value)
    assert "stay_id" in message and "charttime" in message
    assert "30042318" not in message
    assert "2180-01-01" not in message
