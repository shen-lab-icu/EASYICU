"""Per-row unit rules: each value is judged alone, so batches cannot move it.

The lead-r13 magnitude/format sweep classified these conversions as LOW risk.
Each decides a unit from a single value inside a declared physiological
window, the way ricu's own per-value conversions do: °F in a declared range,
L vs mL, fraction vs percent, decimal-entry repair, cm vs m, and the FiO2 domain
check. These tests pin two things for each rule:
- the boundary values;
- partition invariance: the union of two row sets converts exactly as each set
  does alone.

A future edit that swaps a row rule for a frame statistic (max, median, share
of rows) fails here. Synthetic values only.

Sites #17-#24 and #26 are pinned in this file; #25 (the MIMIC-III epoch-us
charttime guard) needs a concept load, so it is pinned in
test_mimic_epoch_us_rule_pinned.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import ConceptSource, _apply_callback
from easyicu.concept.aumc_sources import tidal_volume
from easyicu.concept.callbacks import (
    ConceptCallbackContext,
    _callback_bmi,
    _callback_oxygenation_index,
)
from easyicu.concept.eicu_source_qc import fio2_values
from easyicu.scores.sofa2_validation import SOFA2InputError, normalize_fio2_input
from easyicu.table import ICUTable
from easyicu.utils.callback_utils import percent_as_numeric


def _partition_invariant(rule, left: pd.Series, right: pd.Series) -> pd.Series:
    """``rule`` on the union equals ``rule`` on each part (exact)."""
    both = pd.concat([left, right], ignore_index=True)
    whole = rule(both).reset_index(drop=True)
    parts = pd.concat([rule(left), rule(right)], ignore_index=True)
    pd.testing.assert_series_equal(whole, parts, check_names=False, check_exact=True)
    return whole


# --- #17 percent_as_numeric: fraction in (0, 1] means percent/100 ------------


def test_percent_as_numeric_series_is_row_wise():
    left = pd.Series([0.5, "40%", 1.0])  # a batch of mostly fractions
    right = pd.Series([50.0, "21%", 1.5, 0.0])  # a batch of percentages
    out = _partition_invariant(percent_as_numeric, left, right)
    assert out.tolist() == [50.0, 40.0, 100.0, 50.0, 21.0, 1.5, 0.0]
    assert percent_as_numeric("0.21") == pytest.approx(21.0)
    assert percent_as_numeric("35%") == 35.0


def test_transform_percent_as_numeric_callback_is_row_wise():
    def rule(values):
        frame = pd.DataFrame({"fio2": values.to_numpy()})
        out = _apply_callback(
            frame, ConceptSource(callback="transform_fun(percent_as_numeric)", value_var="fio2"),
            concept_name="fio2",
        )
        return pd.to_numeric(out["fio2"], errors="coerce")

    out = _partition_invariant(rule, pd.Series([0.4, "0.6"]), pd.Series(["60%", 80.0]))
    assert out.tolist() == [40.0, 60.0, 60.0, 80.0]


# --- #19 eICU FiO2 sources and the oxygenation index -------------------------


def test_eicu_fio2_values_converts_each_fraction_alone():
    def rule(values):
        frame = pd.DataFrame({"v": values.astype(object).to_numpy()})
        return fio2_values(frame, "v", "chart")["v"].reset_index(drop=True)

    out = _partition_invariant(rule, pd.Series([0.21, 0.5, 1.0]), pd.Series([40.0, "60%", 0.2, 101.0]))
    # 0.2 is neither a fraction in [.21, 1] nor a percentage in [21, 100]; 101 is out of range.
    assert out.tolist() == [21.0, 50.0, 100.0, 40.0, 60.0]


def _ctx(name: str) -> ConceptCallbackContext:
    return ConceptCallbackContext(
        concept_name=name, target=None, interval=pd.Timedelta(hours=1),
        resolver=None, data_source=None, patient_ids=None,
    )


def _ts(col, stays, values):
    return ICUTable(
        pd.DataFrame({"stay_id": stays, "charttime": [2.0] * len(stays), col: values}),
        id_columns=["stay_id"], index_column="charttime", value_column=col,
    )


def test_oxygenation_index_fio2_rule_does_not_depend_on_other_stays():
    def oi(stays, fio2):
        out = _callback_oxygenation_index({
            "fio2": _ts("fio2", stays, fio2),
            "mean_airway_pres": _ts("mean_airway_pres", stays, [12.0] * len(stays)),
            "po2": _ts("po2", stays, [60.0] * len(stays)),
        }, _ctx("oxygenation_index")).data
        return out.set_index("stay_id")["oxygenation_index"]

    together = oi([1, 2, 3], [0.5, 50.0, 1.5])
    alone = pd.concat([oi([1], [0.5]), oi([2], [50.0]), oi([3], [1.5])])
    pd.testing.assert_series_equal(together, alone, check_exact=True)
    # 0.5 and 50 are both 50 %; 1.5 sits inside the declared fraction window (<= 1.5) -> 150 % -> OI 30.
    assert together.tolist() == pytest.approx([10.0, 10.0, 30.0])


# --- #20 eICU ionized calcium: mg/dL vs mmol/L per row -----------------------


def test_eicu_ionized_calcium_is_row_wise():
    def rule(values):
        frame = pd.DataFrame({
            "labresult": values.to_numpy(),
            "labmeasurenameinterface": ["mg/dL" if v in (2.2,) else "mmol/L" for v in values],
        })
        out = _apply_callback(
            frame, ConceptSource(callback="eicu_ionized_calcium", value_var="labresult"),
            concept_name="ionized_calcium",
        )
        return pd.to_numeric(out["labresult"], errors="coerce")

    out = _partition_invariant(rule, pd.Series([1.1, 2.2]), pd.Series([4.6, 1.3]))
    assert out.tolist() == pytest.approx([1.1, 2.2 / 4.008, 4.6 / 4.008, 1.3])


# --- #21 fahr_between: Fahrenheit only inside the declared window ------------


def test_fahr_between_is_row_wise():
    def rule(values):
        frame = pd.DataFrame({"temp": values.to_numpy()})
        out = _apply_callback(
            frame, ConceptSource(callback="transform_fun(fahr_between(85, 110))", value_var="temp"),
            concept_name="temp",
        )
        return pd.to_numeric(out["temp"], errors="coerce")

    out = _partition_invariant(rule, pd.Series([37.0, 98.6]), pd.Series([85.0, 110.0, 84.9, 110.1]))
    assert out.tolist() == pytest.approx([37.0, 37.0, (85 - 32) * 5 / 9, (110 - 32) * 5 / 9, 84.9, 110.1])


# --- #22 AUMC 8851 tidal volume: L vs mL per row -----------------------------


def test_aumc_tidal_volume_is_row_wise():
    def rule(values):
        return tidal_volume(pd.DataFrame({"value": values.to_numpy()}), "value")["value"]

    out = _partition_invariant(rule, pd.Series([0.45, 1.5]), pd.Series([450.0, 2.0, 50.0, 0.1]))
    assert out.iloc[:2].tolist() == pytest.approx([450.0, 1500.0])
    assert out.iloc[2] == 450.0 and pd.isna(out.iloc[3]) and pd.isna(out.iloc[4])
    assert out.iloc[5] == pytest.approx(100.0)


# --- #23 AUMC urine decimal-entry repair --------------------------------------


def test_aumc_urine_decimal_repair_is_row_wise():
    def rule(values):
        out = _apply_callback(
            pd.DataFrame({"value": values.to_numpy()}),
            ConceptSource(callback="aumc_urine_output", value_var="value"),
            concept_name="kdigo_urine_input",
        )
        return pd.to_numeric(out["value"], errors="coerce")

    out = _partition_invariant(rule, pd.Series([100.0, 2500.0]), pd.Series([2600.0, 45_000.0, 50_000.0]))
    assert out.iloc[:4].tolist() == [100.0, 2500.0, 260.0, 4500.0] and pd.isna(out.iloc[4])


# --- #24 BMI height: cm vs m per row -------------------------------------------


def test_bmi_height_unit_is_judged_per_stay():
    def bmi(stays, heights):
        tables = {
            "weight": ICUTable(pd.DataFrame({"stay_id": stays, "weight": [80.0] * len(stays)}),
                               id_columns=["stay_id"], index_column=None, value_column="weight"),
            "height": ICUTable(pd.DataFrame({"stay_id": stays, "height": heights}),
                               id_columns=["stay_id"], index_column=None, value_column="height"),
        }
        return _callback_bmi(tables, _ctx("bmi")).data.set_index("stay_id")["bmi"]

    together = bmi([1, 2], [180.0, 1.8])
    alone = pd.concat([bmi([1], [180.0]), bmi([2], [1.8])])
    pd.testing.assert_series_equal(together, alone, check_exact=True)
    assert together.tolist() == pytest.approx([80 / 1.8**2] * 2)


# --- #26 SOFA-2 FiO2 domain: disjoint windows; a mixed call fails closed -------


def test_sofa2_fio2_fractions_do_not_depend_on_partition_and_a_mix_is_refused():
    index_a, index_b = pd.RangeIndex(2), pd.RangeIndex(2)
    frac = normalize_fio2_input(pd.Series([0.21, 0.5]), index=index_a)
    pct = normalize_fio2_input(pd.Series([21.0, 50.0]), index=index_b)
    assert (frac.source_unit, pct.source_unit) == ("fraction", "percent")
    np.testing.assert_allclose(frac.values, pct.values)
    with pytest.raises(SOFA2InputError) as caught:
        normalize_fio2_input(pd.Series([0.5, 50.0]), index=pd.RangeIndex(2))
    assert caught.value.reason_code == "sofa2_resp_fio2_units_mixed"


# --- #18 the row-wise FiO2 guards (standardization tail and the P/F guard) --


def test_fio2_standardization_row_guard_is_row_wise_without_a_database_branch():
    from easyicu.concept.callbacks import _standardize_fio2_units

    def rule(values):
        frame = pd.DataFrame({"fio2": values.to_numpy(dtype="float64")})
        return _standardize_fio2_units(frame, "fio2", "sic")["fio2"]

    # 0.16 -> 16 % is converted and then dropped below 21; 0.15 is not a fraction.
    out = _partition_invariant(rule, pd.Series([0.4, 0.21, 1.0, 0.16]), pd.Series([40.0, 0.15, 101.0, 20.0]))
    assert out.iloc[[0, 1, 2, 4]].tolist() == pytest.approx([40.0, 21.0, 100.0, 40.0])
    assert out.iloc[[3, 5, 6, 7]].isna().all()


def test_pafi_fio2_guard_is_row_wise_when_the_unit_is_declared():
    from easyicu.concept.callbacks import _callback_pafi

    def pafi(stays, fio2):
        table = lambda name, values: ICUTable(  # noqa: E731
            pd.DataFrame({"stay_id": stays, "charttime": [0.0] * len(stays), name: values}),
            id_columns=["stay_id"], index_column="charttime", value_column=name,
        )
        ctx = ConceptCallbackContext(
            concept_name="pafi", target=None, interval=None, resolver=None,
            data_source=None, patient_ids=None, kwargs={"fio2_unit": "percentage"},
        )
        out = _callback_pafi({"po2": table("po2", [80.0] * len(stays)), "fio2": table("fio2", fio2)},
                             ctx, source_col_a="po2", source_col_b="fio2", output_col="pafi").data
        return out.set_index("stay_id")["pafi"]

    together = pafi([1, 2, 3], [0.4, 40.0, 0.1])
    alone = pd.concat([pafi([1], [0.4]), pafi([2], [40.0]), pafi([3], [0.1])])
    pd.testing.assert_series_equal(together, alone, check_exact=True)
    assert together.iloc[:2].tolist() == pytest.approx([200.0, 200.0]) and pd.isna(together.iloc[2])
