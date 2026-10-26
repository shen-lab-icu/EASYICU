"""Index aliases must preserve entity identity and relative-time semantics."""

import pandas as pd
import pytest

from easyicu.concept import ConceptResolver
from easyicu.concept.merge_time_index import (
    RELATIVE_HOURS,
    MergeTimeIndexError,
    declare_resolved_time_index,
)
from easyicu.table import ICUTable, IdTbl, TsTbl, WinTbl


HOUR = pd.Timedelta(hours=1)


def _table(name, index="time", *, times=(0.0, 2.0), interval=HOUR):
    return ICUTable(
        pd.DataFrame({"entity": [7, 7], index: times, name: [1.0, 2.0]}),
        id_columns=["entity"], index_column=index, value_column=name,
        time_basis=RELATIVE_HOURS, interval=interval,
    )


@pytest.mark.parametrize("first_index,second_index", [("time", "recorded"), ("recorded", "time")])
@pytest.mark.parametrize("interval", [HOUR, pd.Timedelta(minutes=15)])
def test_merge_aligns_sparse_aliases_without_changing_inputs(first_index, second_index, interval):
    step = interval / HOUR
    first = _table("a", first_index, times=(0.0, 2.0 * step), interval=interval)
    second = _table("b", second_index, times=(step, 2.0 * step), interval=interval)
    originals = [table.copy() for table in (first, second)]
    actual = ConceptResolver({})._merge_tables({"a": first, "b": second})
    expected = pd.DataFrame({
        "entity": [7, 7, 7], first_index: [0.0, step, 2.0 * step],
        "a": [1.0, None, 2.0], "b": [None, 1.0, 2.0],
    })
    pd.testing.assert_frame_equal(actual.sort_values(first_index).reset_index(drop=True), expected)
    for table, original in zip((first, second), originals):
        pd.testing.assert_frame_equal(table.data, original.data)
        assert table.index_column == original.index_column
        assert table.time_basis == original.time_basis
        assert table.interval == original.interval


@pytest.mark.parametrize("window_first", [False, True])
def test_window_table_alias_uses_the_same_guard(window_first):
    point = _table("a")
    window = WinTbl(
        pd.DataFrame({"entity": [7], "begin": [0.0], "duration": [1.0], "b": [True]}),
        id_vars=["entity"], index_var="begin", dur_var="duration", interval=HOUR,
    )
    declare_resolved_time_index(window, align_to_admission=True, interval=HOUR)
    tables = {"b": window, "a": point} if window_first else {"a": point, "b": window}
    result = ConceptResolver({})._merge_tables(tables)
    assert result.columns[1] == ("begin" if window_first else "time")
    assert result["b"].dropna().tolist() == [True]
    assert window.index_var == "begin"


@pytest.mark.parametrize("second_index", ["time", "recorded"])
def test_different_identifiers_are_still_refused(second_index):
    second = _table("b", second_index)
    second.data = second.data.rename(columns={"entity": "other_entity"})
    second.id_columns = ["other_entity"]
    with pytest.raises(ValueError, match="identical identifier columns"):
        ConceptResolver({})._merge_tables({"a": _table("a"), "b": second})


@pytest.mark.parametrize("field,value,reason", [
    ("time_basis", None, "time_basis_unknown"),
    ("time_basis", "hospital_admission_hours", "time_basis_mismatch"),
    ("interval", None, "time_interval_unknown"),
    ("interval", pd.NaT, "time_interval_unknown"),
    ("interval", pd.Timedelta(0), "time_interval_unknown"),
    ("interval", -HOUR, "time_interval_unknown"),
    ("interval", 2 * HOUR, "time_interval_mismatch"),
])
def test_alias_refusal_has_a_typed_specific_reason(field, value, reason):
    second = _table("b", "recorded")
    setattr(second, field, value)
    with pytest.raises(MergeTimeIndexError) as caught:
        ConceptResolver({})._merge_tables({"a": _table("a"), "b": second})
    assert caught.value.reason_code == reason
    assert caught.value.concept == "b"
    assert reason in str(caught.value)


@pytest.mark.parametrize("times", [(0.5, 2.0), (0.0, float("inf"))])
def test_off_grid_alias_is_refused(times):
    with pytest.raises(MergeTimeIndexError) as caught:
        ConceptResolver({})._merge_tables({"a": _table("a"), "b": _table("b", "recorded", times=times)})
    assert caught.value.reason_code == "time_grid_mismatch"


@pytest.mark.parametrize("absolute_values", [False, True])
def test_absolute_time_is_never_renamed_as_relative_time(absolute_values):
    tables = {"a": _table("a"), "b": _table("b", "recorded")}
    for table in tables.values():
        if absolute_values:
            table.data[table.index_column] = pd.date_range("2020-01-01", periods=2, freq="h")
        else:
            table.time_basis = "unix_epoch_hours"
    with pytest.raises(MergeTimeIndexError) as caught:
        ConceptResolver({})._merge_tables(tables)
    assert caught.value.reason_code == "time_basis_not_relative"


def test_every_indexed_table_is_checked_including_equal_named_tables():
    third = _table("c")
    third.interval = 2 * HOUR
    with pytest.raises(MergeTimeIndexError) as caught:
        ConceptResolver({})._merge_tables({"a": _table("a"), "b": _table("b", "recorded"), "c": third})
    assert caught.value.reason_code == "time_interval_mismatch"
    assert caught.value.concept == "c"


def test_alias_never_overwrites_a_value_column():
    second = _table("b", "recorded")
    second.data["time"] = 12.0
    with pytest.raises(MergeTimeIndexError) as caught:
        ConceptResolver({})._merge_tables({"a": _table("a"), "b": second})
    assert caught.value.reason_code == "time_index_collision"
    assert second.data["time"].tolist() == [12.0, 12.0]


def test_equal_names_keep_the_existing_merge_without_new_metadata():
    first, second = _table("a"), _table("b")
    for table in (first, second):
        table.time_basis = table.interval = None
    result = ConceptResolver({})._merge_tables({"a": first, "b": second})
    assert result.columns.tolist() == ["entity", "time", "a", "b"]


def test_static_tables_keep_the_existing_id_only_merge():
    tables = {
        name: ICUTable(pd.DataFrame({"entity": [7], name: [value]}), ["entity"], value_column=name)
        for name, value in (("age", 55), ("height", 170))
    }
    result = ConceptResolver({})._merge_tables(tables)
    assert result.to_dict("list") == {"entity": [7], "age": [55], "height": [170]}


@pytest.mark.parametrize("align,interval", [(False, HOUR), (True, None)])
def test_unaligned_or_unresampled_loads_do_not_gain_time_metadata(align, interval):
    table = _table("a")
    table.time_basis = table.interval = None
    declare_resolved_time_index(table, align_to_admission=align, interval=interval)
    assert table.time_basis is None
    assert table.interval is None


def test_declaration_preserves_producer_metadata_and_table_copies():
    table = _table("a", interval=2 * HOUR)
    table.time_basis = "another_origin_hours"
    declare_resolved_time_index(table, align_to_admission=True, interval=HOUR)
    for selected in (table, table.copy(), table.select_ids([7])):
        assert selected.time_basis == "another_origin_hours"
        assert selected.interval == 2 * HOUR


@pytest.mark.parametrize("window", [False, True])
def test_time_series_copies_preserve_the_producers_time_basis(window):
    frame = pd.DataFrame({"entity": [7], "time": [0.0], "duration": [1.0], "a": [True]})
    table = (
        WinTbl(frame, ["entity"], "time", "duration", 2 * HOUR)
        if window else TsTbl(frame, ["entity"], "time", 2 * HOUR)
    )
    table.time_basis = "hospital_admission_hours"
    copied = table.copy()
    declare_resolved_time_index(copied, align_to_admission=True, interval=HOUR)
    assert copied.time_basis == "hospital_admission_hours"
    assert copied.interval == 2 * HOUR


def test_id_only_declaration_does_not_add_a_time_axis():
    table = IdTbl(pd.DataFrame({"entity": [7], "age": [55]}), ["entity"])
    declare_resolved_time_index(table, align_to_admission=True, interval=HOUR)
    assert not hasattr(table, "time_basis")
