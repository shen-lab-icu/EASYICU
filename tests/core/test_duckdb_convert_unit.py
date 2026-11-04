"""convert_unit callbacks on the DuckDB id_tbl path follow ricu semantics."""

import duckdb
import pytest

from easyicu.concept.duckdb_convert_unit import convert_unit_value_sql


def _values(expression):
    connection = duckdb.connect()
    connection.execute("CREATE TABLE t AS SELECT * FROM (VALUES (200.0, 'lbs'), (100.0, 'kg')) v(value, valueuom)")
    return [row[0] for row in connection.execute(f"SELECT {expression} FROM t").fetchall()]


def test_two_argument_convert_unit_converts_every_row():
    # 2026-11-04 audit round 3, R3-01: 226531 'Admission Weight (lbs.)' carries
    # no unit regex; before, the fast path published pounds as kilograms.
    sql = convert_unit_value_sql("convert_unit(binary_op(`*`, 0.45359237), 'kg')", "value", "valueuom")
    assert _values(sql) == pytest.approx([90.718474, 45.359237])


def test_three_argument_convert_unit_converts_only_matching_units():
    sql = convert_unit_value_sql("convert_unit(binary_op(`*`, 0.45359237), 'kg', 'lb')", "value", "valueuom")
    assert _values(sql) == pytest.approx([90.718474, 100.0])


def test_unexpressible_callback_is_not_reported_as_applied():
    assert convert_unit_value_sql("convert_unit(some_other_fun, 'kg')", "value", "valueuom") is None
    assert convert_unit_value_sql("convert_unit(binary_op(`*`, 2), 'x', 'y')", "value", None) is None


def test_dictionary_weight_sources_are_expressible():
    from easyicu.concept import load_dictionary

    weight = load_dictionary().get("weight")
    for database in ("miiv", "mimic"):
        callbacks = [s.callback for s in weight.sources[database] if s.callback and s.callback.startswith("convert_unit(")]
        assert callbacks, database
        assert all(convert_unit_value_sql(c, "value", "valueuom") for c in callbacks)
