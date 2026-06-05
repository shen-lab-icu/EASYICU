"""Smoke tests for ``easyicu.callback_utils``.

Background
----------
``callback_utils.py`` (~5,283 LOC) holds the R-ricu-style callback
factories and helpers used by every concept callback in the dictionary.
Despite its size and its central role it currently has **no direct
unit tests** — only transitive coverage via concept-level tests.

These tests pin the contracts of the pure / closure-factory subset of
the public surface, so that the planned ``utils/callback`` namespace
consolidation has something to break against. The DataFrame-mutating
callbacks (rate / duration / interval expansion) are NOT covered here
— they require ICU-shaped fixtures that belong in concept-level tests.

Scope
-----
Pure functions and closure factories from the top of the module:

* ``transform_fun``, ``binary_op``, ``comp_na``, ``set_val``
* ``apply_map``, ``combine_callbacks``, ``force_type``
* ``fahr_to_cels``, ``percent_as_numeric``, ``silent_as_numeric``
* ``eicu_extract_unit``, ``sub_trans``
* ``get_one_unique``, ``units_to_unit``
"""

from __future__ import annotations

import operator

import numpy as np
import pandas as pd
import pytest

from easyicu.callback_utils import (
    apply_map,
    binary_op,
    combine_callbacks,
    comp_na,
    convert_unit,
    eicu_extract_unit,
    fahr_to_cels,
    force_type,
    get_one_unique,
    percent_as_numeric,
    set_val,
    silent_as_numeric,
    sub_trans,
    transform_fun,
    units_to_unit,
)


# ---------------------------------------------------------------------------
# fahr_to_cels — pure math, scalar + Series
# ---------------------------------------------------------------------------


class TestFahrToCels:
    def test_freezing_point(self):
        assert fahr_to_cels(32) == pytest.approx(0.0)

    def test_body_temp(self):
        assert fahr_to_cels(98.6) == pytest.approx(37.0, abs=1e-3)

    def test_series_input(self):
        result = fahr_to_cels(pd.Series([32.0, 212.0]))
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[1] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# percent_as_numeric / silent_as_numeric
# ---------------------------------------------------------------------------


class TestPercentAsNumeric:
    def test_scalar_string(self):
        assert percent_as_numeric("50%") == 50.0

    def test_scalar_without_percent(self):
        assert percent_as_numeric("42") == 42.0

    def test_series(self):
        result = percent_as_numeric(pd.Series(["10%", "25%", "100%"]))
        assert list(result) == [10.0, 25.0, 100.0]


class TestSilentAsNumeric:
    def test_series_non_numeric_becomes_nan(self):
        result = silent_as_numeric(pd.Series(["1", "bad", "3"]))
        assert result.iloc[0] == 1.0
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 3.0

    def test_ndarray(self):
        result = silent_as_numeric(np.array(["2.5", "x"]))
        assert isinstance(result, np.ndarray)
        assert result[0] == 2.5
        assert pd.isna(result[1])

    def test_scalar_convertible(self):
        assert silent_as_numeric("3.14") == pytest.approx(3.14)

    def test_scalar_non_convertible_is_nan(self):
        assert pd.isna(silent_as_numeric("not a number"))


# ---------------------------------------------------------------------------
# eicu_extract_unit
# ---------------------------------------------------------------------------


class TestEicuExtractUnit:
    def test_scalar_with_parens(self):
        assert eicu_extract_unit("Norepinephrine (mcg/kg/min)") == "mcg/kg/min"

    def test_scalar_no_parens_returns_nan(self):
        assert pd.isna(eicu_extract_unit("Drug"))

    def test_series(self):
        result = eicu_extract_unit(
            pd.Series(["Drug A (mg/hr)", "No parens", "X (units/min)"])
        )
        assert result.iloc[0] == "mg/hr"
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == "units/min"


# ---------------------------------------------------------------------------
# get_one_unique
# ---------------------------------------------------------------------------


class TestGetOneUnique:
    def test_all_same_returns_value(self):
        assert get_one_unique([1, 1, 1]) == 1

    def test_multiple_distinct_returns_nan(self):
        assert pd.isna(get_one_unique([1, 2, 3]))

    def test_series_input(self):
        assert get_one_unique(pd.Series(["a", "a", "a"])) == "a"

    def test_na_rm_collapses_to_single_value(self):
        # Without na_rm: NaN counts as a distinct value → multiple → NaN.
        # With na_rm=True: NaN is dropped → only [1] left → returns 1.
        s = pd.Series([1.0, 1.0, np.nan])
        assert get_one_unique(s, na_rm=True) == 1.0


# ---------------------------------------------------------------------------
# units_to_unit
# ---------------------------------------------------------------------------


class TestUnitsToUnit:
    """Pin units_to_unit across pandas versions.

    Until 2026-05 this helper silently returned 'hour' for every sub-day
    precision under pandas >=1.5 because ``resolution_string`` switched
    to lower-case ('h'/'min'/'s') while the lookup table only knew the
    pandas <1.5 upper-case codes ('H'/'T'/'S'). The bug was found while
    adding these smoke tests; the lookup table now accepts both forms.
    """

    def test_day(self):
        assert units_to_unit(pd.Timedelta(days=1)) == "day"

    def test_hour(self):
        assert units_to_unit(pd.Timedelta(hours=1)) == "hour"

    def test_minute(self):
        assert units_to_unit(pd.Timedelta(minutes=15)) == "min"

    def test_second(self):
        assert units_to_unit(pd.Timedelta(seconds=30)) == "sec"


# ---------------------------------------------------------------------------
# binary_op closure
# ---------------------------------------------------------------------------


class TestBinaryOp:
    def test_multiplication(self):
        times_2 = binary_op(operator.mul, 2)
        assert times_2(5) == 10

    def test_addition(self):
        plus_3 = binary_op(operator.add, 3)
        assert plus_3(4) == 7

    def test_none_input_returns_none(self):
        assert binary_op(operator.mul, 2)(None) is None

    def test_division_by_zero_returns_none(self):
        # Documented behaviour: division-by-zero is coerced to None.
        div = binary_op(operator.truediv, 0)
        assert div(5) is None

    def test_division_by_nan_returns_none(self):
        div = binary_op(operator.truediv, np.nan)
        assert div(5) is None

    def test_division_series_preserves_values_and_na(self):
        div = binary_op(operator.truediv, 60)
        result = div(pd.Series([120, 60, np.nan, "bad"]))
        assert result.iloc[0] == pytest.approx(2.0)
        assert result.iloc[1] == pytest.approx(1.0)
        assert pd.isna(result.iloc[2])
        assert pd.isna(result.iloc[3])

    def test_division_series_by_zero_returns_nan_series(self):
        div = binary_op(operator.truediv, 0)
        result = div(pd.Series([120, 60], index=["a", "b"]))
        assert list(result.index) == ["a", "b"]
        assert result.isna().all()


# ---------------------------------------------------------------------------
# convert_unit — direct public helper path
# ---------------------------------------------------------------------------


class TestConvertUnit:
    def test_regex_division_only_converts_matching_units(self):
        cb = convert_unit(binary_op(operator.truediv, 60), "hour", regex="min")
        df = pd.DataFrame(
            {
                "value": [120.0, 60.0, 30.0],
                "unit": ["min", "min", "sec"],
            }
        )

        result = cb(df)

        assert result["value"].tolist() == [2.0, 1.0, 30.0]
        assert result["unit"].tolist() == ["hour", "hour", "sec"]

    def test_division_without_regex_converts_whole_column(self):
        cb = convert_unit(binary_op(operator.truediv, 60), "hour")
        df = pd.DataFrame({"value": [120.0, 60.0], "unit": ["min", "min"]})

        result = cb(df)

        assert result["value"].tolist() == [2.0, 1.0]
        assert result["unit"].tolist() == ["hour", "hour"]


# ---------------------------------------------------------------------------
# comp_na closure (NaN-aware comparison)
# ---------------------------------------------------------------------------


class TestCompNa:
    def test_scalar_gte(self):
        gte_4 = comp_na(operator.ge, 4)
        assert gte_4(5) is True
        assert gte_4(3) is False

    def test_scalar_na_is_false(self):
        gte_4 = comp_na(operator.ge, 4)
        assert gte_4(np.nan) is False

    def test_series_treats_na_as_false(self):
        ge_4 = comp_na(operator.ge, 4)
        result = ge_4(pd.Series([1, 4, 5, np.nan]))
        assert list(result) == [False, True, True, False]


# ---------------------------------------------------------------------------
# set_val closure
# ---------------------------------------------------------------------------


class TestSetVal:
    def test_scalar_input_returns_scalar(self):
        assert set_val(7)(99) == 7

    def test_series_input_replaces_every_row(self):
        result = set_val(True)(pd.Series([1, 2, 3]))
        assert list(result) == [True, True, True]
        assert len(result) == 3

    def test_series_preserves_index(self):
        s = pd.Series([10, 20], index=["a", "b"])
        result = set_val(0)(s)
        assert list(result.index) == ["a", "b"]


# ---------------------------------------------------------------------------
# combine_callbacks — composition order
# ---------------------------------------------------------------------------


def test_combine_callbacks_applies_left_to_right():
    cb1 = transform_fun(lambda x: x * 2)
    cb2 = transform_fun(lambda x: x + 1)
    combined = combine_callbacks(cb1, cb2)
    df = pd.DataFrame({"value": [1, 2, 3]})
    result = combined(df)
    # Documented: cb1 first, then cb2. So (1*2)+1 = 3, (2*2)+1 = 5, (3*2)+1 = 7.
    assert list(result["value"]) == [3, 5, 7]


def test_combine_callbacks_empty_is_passthrough():
    df = pd.DataFrame({"value": [1, 2]})
    result = combine_callbacks()(df)
    pd.testing.assert_frame_equal(result, df)


# ---------------------------------------------------------------------------
# transform_fun — missing column is a no-op (documented)
# ---------------------------------------------------------------------------


class TestTransformFun:
    def test_applies_to_default_value_column(self):
        double = transform_fun(lambda x: x * 2)
        df = pd.DataFrame({"value": [1, 2, 3]})
        result = double(df)
        assert list(result["value"]) == [2, 4, 6]

    def test_missing_column_is_passthrough(self):
        double = transform_fun(lambda x: x * 2)
        df = pd.DataFrame({"other": [1, 2]})
        result = double(df, val_col="value")
        pd.testing.assert_frame_equal(result, df)

    def test_does_not_mutate_input(self):
        double = transform_fun(lambda x: x * 2)
        df = pd.DataFrame({"value": [1, 2]})
        original_values = list(df["value"])
        _ = double(df)
        assert list(df["value"]) == original_values


# ---------------------------------------------------------------------------
# force_type
# ---------------------------------------------------------------------------


class TestForceType:
    def test_int(self):
        result = force_type("int")(pd.Series(["1", "2", "x"]))
        # Int64 nullable: non-coercible becomes pd.NA
        assert result.iloc[0] == 1
        assert result.iloc[1] == 2
        assert pd.isna(result.iloc[2])

    def test_float(self):
        result = force_type("float")(pd.Series(["1.5", "bad"]))
        assert result.iloc[0] == pytest.approx(1.5)
        assert pd.isna(result.iloc[1])

    def test_str(self):
        result = force_type("str")(pd.Series([1, 2]))
        assert list(result) == ["1", "2"]

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown type"):
            force_type("complex128")


# ---------------------------------------------------------------------------
# apply_map
# ---------------------------------------------------------------------------


class TestApplyMap:
    def test_numeric_to_label(self):
        mapper = apply_map({1: "male", 2: "female"})
        df = pd.DataFrame({"sex_code": [1, 2, 1]})
        result = mapper(df, val_col="sex_code")
        assert list(result["sex_code"]) == ["male", "female", "male"]

    def test_string_key_fallback_for_numeric_source(self):
        # Documented quirk: many ricu dicts ship string keys against
        # numeric source columns. Apply_map tries the string form too.
        mapper = apply_map({"1": "yes", "2": "no"})
        df = pd.DataFrame({"flag": [1, 2]})
        result = mapper(df, val_col="flag")
        assert list(result["flag"]) == ["yes", "no"]

    def test_missing_target_column_is_passthrough(self):
        mapper = apply_map({1: "a"})
        df = pd.DataFrame({"other": [1]})
        result = mapper(df, val_col="value")
        pd.testing.assert_frame_equal(result, df)

    def test_unmapped_values_left_untouched(self):
        # Documented: mask is only set where mapping existed; unmapped
        # rows keep their original value.
        mapper = apply_map({1: "a"})
        df = pd.DataFrame({"value": [1, 99]})
        result = mapper(df, val_col="value")
        assert result["value"].iloc[0] == "a"
        # 99 was never in the mapping → it stays put.
        assert result["value"].iloc[1] == 99


# ---------------------------------------------------------------------------
# sub_trans
# ---------------------------------------------------------------------------


class TestSubTrans:
    def test_scalar_substitution(self):
        hr_to_min = sub_trans(r"/hr$", "/min")
        assert hr_to_min("mg/hr") == "mg/min"

    def test_series_substitution(self):
        hr_to_min = sub_trans(r"/hr$", "/min")
        result = hr_to_min(pd.Series(["mg/hr", "ug/hr", "ml/min"]))
        assert list(result) == ["mg/min", "ug/min", "ml/min"]

    def test_case_insensitive(self):
        # Documented: regex match is case-insensitive.
        hr_to_min = sub_trans(r"/HR$", "/min")
        assert hr_to_min("mg/hr") == "mg/min"
