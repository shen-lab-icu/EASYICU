"""Value-based unit guesses refuse; declared units and per-row rules remain.

These are the lead-r13 magnitude/format sweep MEDIUM sites. Instrumentation of
586a93a29 on the 3,000-stay sample of all six databases (one-shot and batch
50) never reached a guessed branch, so the outputs are unchanged. Each guess
used to decide from a frame statistic (max, median, share of values), which
depends on the rows sharing the batch. Reaching one now raises
``UndeclaredUnitError``; a declared unit is applied as declared. Synthetic
values only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.callbacks import pafi as public_pafi
from easyicu.callbacks import safi as public_safi
from easyicu.concept.callback_apply import _normalize_eicu_tidal_volume_frame
from easyicu.concept.callbacks import (
    ConceptCallbackContext,
    _callback_pafi,
    _standardize_fio2_units,
)
from easyicu.table import ICUTable
from easyicu.utils.declared_units import UndeclaredUnitError


# --- #6 concept FiO2 standardization -----------------------------------------


@pytest.mark.parametrize(
    "database,values",
    [
        ("miiv", [0.4, 0.5, 0.6]),  # max <= 1, median > 0.1
        ("miiv", [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.3, 0.4, 0.5, 1.2]),
        ("eicu", [0.3, 0.5]),
        ("aumc", [0.4, 0.5, 40.0]),  # > 20 % fraction-like values
    ],
)
def test_a_batch_that_looks_fractional_is_refused_not_rescaled(database, values):
    frame = pd.DataFrame({"fio2": values})
    with pytest.raises(UndeclaredUnitError) as caught:
        _standardize_fio2_units(frame, "fio2", database)
    assert (caught.value.reason, caught.value.quantity) == ("fio2_unit_undeclared", "fio2")
    assert database in str(caught.value)


def test_percent_batches_and_isolated_fractions_keep_the_row_guard():
    # A percent batch with one isolated fraction never reached a guessed branch:
    # the row guard converts the fraction alone (as before).
    frame = pd.DataFrame({"fio2": [40.0, 60.0, 0.5, 21.0]})
    out = _standardize_fio2_units(frame, "fio2", "miiv")["fio2"]
    assert out.tolist() == [40.0, 60.0, 50.0, 21.0]
    # AUMC's minority fraction branch is per row: (0.1, 1) x100, zero -> NaN.
    out = _standardize_fio2_units(pd.DataFrame({"fio2": [40.0] * 9 + [0.5, 0.0]}), "fio2", "aumc")["fio2"]
    assert out.iloc[:10].tolist() == [40.0] * 9 + [50.0] and pd.isna(out.iloc[10])


def test_a_declared_unit_skips_the_guess_and_is_validated():
    frame = pd.DataFrame({"fio2": [0.4, 0.5, 0.6]})
    assert _standardize_fio2_units(frame, "fio2", "miiv", fio2_unit="fraction").equals(frame)
    with pytest.raises(UndeclaredUnitError) as caught:
        _standardize_fio2_units(frame, "fio2", "miiv", fio2_unit="ratio")
    assert caught.value.reason == "fio2_unit_invalid"


# --- #7 concept pafi/safi ---------------------------------------------------------


def _table(name, values):
    return ICUTable(
        pd.DataFrame({"stay_id": list(range(1, len(values) + 1)), "charttime": [0.0] * len(values), name: values}),
        id_columns=["stay_id"], index_column="charttime", value_column=name,
    )


def _pafi(fio2, **kwargs):
    ctx = ConceptCallbackContext(
        concept_name="pafi", target=None, interval=None, resolver=None,
        data_source=None, patient_ids=None, kwargs=kwargs,
    )
    return _callback_pafi({"po2": _table("po2", [80.0] * len(fio2)), "fio2": _table("fio2", fio2)},
                          ctx, source_col_a="po2", source_col_b="fio2", output_col="pafi").data["pafi"]


def test_pafi_majority_of_fractions_is_refused_without_a_declared_unit():
    with pytest.raises(UndeclaredUnitError) as caught:
        _pafi([0.4, 0.5, 60.0])
    assert caught.value.reason == "fio2_unit_undeclared" and "2 of 3" in str(caught.value)


def test_pafi_declared_units_and_percent_batches():
    assert _pafi([0.4, 0.5], fio2_unit="fraction").tolist() == pytest.approx([200.0, 160.0])
    assert _pafi([40.0, 50.0], fio2_unit="percentage").tolist() == pytest.approx([200.0, 160.0])
    # Undeclared percent batch (the production case): unchanged, row guard intact.
    assert _pafi([40.0, 50.0, 0.4]).tolist() == pytest.approx([200.0, 160.0, 200.0])
    with pytest.raises(UndeclaredUnitError):
        _pafi([40.0], fio2_unit="ratio")


# --- #8 public easyicu.callbacks.pafi / safi ---------------------------------------


def _frames(fio2):
    o2 = pd.DataFrame({"stay_id": [1, 2], "charttime": [0.0, 0.0], "po2": [80.0, 80.0]})
    f = pd.DataFrame({"stay_id": [1, 2], "charttime": [0.0, 0.0], "fio2": fio2})
    return o2, f


@pytest.mark.parametrize("fio2", [[0.4, 0.5], [0.4, 50.0]])  # pure and mixed
def test_public_pafi_refuses_fraction_like_values_without_a_unit(fio2):
    po2, f = _frames(fio2)
    with pytest.raises(UndeclaredUnitError) as caught:
        public_pafi(po2, f)
    assert caught.value.reason == "fio2_unit_undeclared"
    with pytest.raises(UndeclaredUnitError):
        public_pafi(po2, f, database="miiv")  # a database name is not a unit


def test_public_pafi_and_safi_apply_the_declared_unit():
    po2, f = _frames([0.4, 0.5])
    out = public_pafi(po2, f, fio2_unit="fraction").set_index("stay_id")["pafi"]
    assert out.tolist() == pytest.approx([200.0, 160.0])
    po2, f = _frames([40.0, 50.0])
    assert public_pafi(po2, f).set_index("stay_id")["pafi"].tolist() == pytest.approx([200.0, 160.0])
    spo2 = pd.DataFrame({"stay_id": [1], "charttime": [0.0], "spo2": [96.0]})
    fio2 = pd.DataFrame({"stay_id": [1], "charttime": [0.0], "fio2": [0.4]})
    assert public_safi(spo2, fio2, fio2_unit="fraction")["safi"].tolist() == pytest.approx([240.0])


# --- #15 eICU tidal volume: the litre/millilitre reference is per stay -------------


def test_eicu_tidal_low_value_reference_is_judged_within_its_own_stay():
    def normalise(stays, values, ages):
        frame = pd.DataFrame({
            "patientunitstayid": stays,
            "respchartvaluelabel": ["Exhaled TV (patient)"] * len(stays),
            "tidal_vol": values,
        })
        out = _normalize_eicu_tidal_volume_frame(
            frame, concept_name="tidal_vol", ages=pd.Series(ages, index=frame.index),
        )
        return out.set_index([out["patientunitstayid"], out.index])["tidal_vol"].droplevel(1)

    # Stay 1 (age unknown) carries an mL reference; stay 2 (child) does not.
    stays, values, ages = [1, 1, 2, 2], [450.0, 0.4, 30.0, 0.4], [float("nan")] * 2 + [10.0] * 2
    together = normalise(stays, values, ages)
    alone = pd.concat([normalise(stays[:2], values[:2], ages[:2]), normalise(stays[2:], values[2:], ages[2:])])
    pd.testing.assert_series_equal(together, alone, check_exact=True)
    assert together.loc[1].tolist() == pytest.approx([450.0, 400.0])
    assert together.loc[2].isna().all()  # 30 is unit-ambiguous; 0.4 has no reference
