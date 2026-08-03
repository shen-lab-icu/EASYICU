"""Regression tests for grouped amount-to-rate dextrose callbacks.

MIMIC-III CareVue amounts can arrive as strings.  Grouping before numeric
coercion concatenates those strings (for example ``"300" + "300"``), and the
old dispatcher then swallowed the resulting division error and returned the
raw amounts as if they were rates.  These tests pin the full callback order:
numeric amount aggregation, rate calculation, then concentration conversion.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from easyicu.concept.callback_apply import _apply_callback
from easyicu.concept.schema import ConceptSource
from easyicu.utils.callback_utils import grp_mount_to_rate


def _mimic_cv_source(callback: str | None = None) -> ConceptSource:
    return ConceptSource(
        table="inputevents_cv",
        sub_var="itemid",
        value_var="amount",
        unit_var="amountuom",
        callback=callback
        or "combine_callbacks("
        "grp_mount_to_rate(mins(1L), hours(1L)), "
        "dex_to_10(30017L, 2))",
        params={"grp_var": "linkorderid"},
    )


def _hirid_source() -> ConceptSource:
    return ConceptSource(
        table="pharma",
        sub_var="pharmaid",
        callback=(
            "combine_callbacks("
            "grp_mount_to_rate(mins(1L), hours(1L)), "
            "dex_to_10("
            "list(c(1000689L, 1000544L, 1000746L, 1000835L), "
            "1000060L, 1000545L, 1000567L), c(2, 3, 4, 5)))"
        ),
        params={"grp_var": "infusionid"},
    )


def test_mimic_cv_string_amounts_become_rates_before_d10_conversion():
    """Pin real MIMIC-III spot values, including the 30017 x2 order."""

    frame = pd.DataFrame(
        {
            "icustay_id": [270233] * 6,
            "linkorderid": [9305607, 9305607, 1293322, 1293322, 8141493, 8141493],
            "itemid": [30016, 30016, 30017, 30017, 30017, 30017],
            "charttime": pd.to_datetime(
                [
                    "2100-01-01 00:00:00",
                    "2100-01-01 04:00:00",
                    "2100-01-02 00:00:00",
                    "2100-01-02 07:00:00",
                    "2100-01-03 00:00:00",
                    "2100-01-04 20:30:00",
                ]
            ),
            # These are deliberately strings, matching the failing extraction.
            "dex": ["150", "200", "300", "300", "2500", "2687.5"],
            "amountuom": ["ml"] * 6,
        }
    )

    result = _apply_callback(
        frame,
        _mimic_cv_source(),
        concept_name="dex",
        unit_column="amountuom",
    ).set_index("linkorderid")

    assert pd.api.types.is_numeric_dtype(result["dex"])
    assert result.loc[9305607, "itemid"] == 30016
    assert result.loc[9305607, "dex"] == pytest.approx(70.0)
    assert result.loc[1293322, "itemid"] == 30017
    assert result.loc[1293322, "dex"] == pytest.approx(150.0)
    assert result.loc[8141493, "dex"] == pytest.approx(228.02197802197804)
    assert result["dex"].max() < 1_000


def test_mixed_mimic_itemids_are_split_and_keep_the_conversion_key():
    """A shared linkorderid must not erase which concentration a row used."""

    frame = pd.DataFrame(
        {
            "icustay_id": [1, 1, 1, 1],
            "linkorderid": [99, 99, 99, 99],
            "itemid": [30016, 30016, 30017, 30017],
            "charttime": pd.to_datetime(
                ["2020-01-01 00:00", "2020-01-01 01:00"] * 2
            ),
            "dex": ["10", "10", "10", "10"],
            "amountuom": ["ml"] * 4,
        }
    )

    result = _apply_callback(
        frame,
        _mimic_cv_source(),
        concept_name="dex",
        unit_column="amountuom",
    ).sort_values("itemid")

    assert result["itemid"].tolist() == [30016, 30017]
    assert result["dex"].tolist() == pytest.approx([10.0, 20.0])


def test_hirid_nested_concentration_factors_run_after_grouped_rate():
    pharmaids = [1000022, 1000689, 1000060, 1000545, 1000567]
    frame = pd.DataFrame(
        {
            "patientid": np.repeat(7, len(pharmaids) * 2),
            "infusionid": np.repeat(np.arange(1, len(pharmaids) + 1), 2),
            "pharmaid": np.repeat(pharmaids, 2),
            "givenat": pd.to_datetime(
                ["2020-01-01 00:00", "2020-01-01 01:00"] * len(pharmaids)
            ),
            "dex": ["10", "10"] * len(pharmaids),
            "doseunit": ["ml"] * (len(pharmaids) * 2),
        }
    )

    result = _apply_callback(
        frame,
        _hirid_source(),
        concept_name="dex",
        unit_column="doseunit",
    ).set_index("pharmaid")

    # Raw grouped rate is 20 ml / (1 h span + 1 h padding) = 10 ml/h.
    assert result.loc[1000022, "dex"] == pytest.approx(10.0)
    assert result.loc[1000689, "dex"] == pytest.approx(20.0)
    assert result.loc[1000060, "dex"] == pytest.approx(30.0)
    assert result.loc[1000545, "dex"] == pytest.approx(40.0)
    assert result.loc[1000567, "dex"] == pytest.approx(50.0)


def test_non_numeric_amount_rows_are_dropped_with_an_audit_warning(caplog):
    callback = grp_mount_to_rate(
        min_dur=pd.Timedelta(hours=1),
        extra_dur=pd.Timedelta(0),
        grp_var="linkorderid",
    )
    frame = pd.DataFrame(
        {
            "icustay_id": [1, 1, 1, 1],
            "linkorderid": [8, 8, 8, 8],
            "itemid": [30016] * 4,
            "charttime": pd.to_datetime(
                [
                    "2020-01-01 00:00",
                    "2020-01-01 01:00",
                    "2020-01-01 02:00",
                    "2020-01-01 03:00",
                ]
            ),
            "amount": ["10", "not-a-number", None, "inf"],
        }
    )

    with caplog.at_level(logging.WARNING, logger="easyicu.utils.callback_utils"):
        result = callback(
            frame,
            val_col="amount",
            index_var="charttime",
            id_cols=["icustay_id"],
            sub_var="itemid",
        )

    # Invalid amount values do not enter the numerator, but their timestamps
    # still define the recorded three-hour infusion span.
    assert result["amount"].tolist() == pytest.approx([10.0 / 3.0])
    assert "dropped 3/4 rows" in caplog.text
    assert "non-numeric, or non-finite amount values" in caplog.text


def test_all_invalid_amounts_return_empty_not_raw_values():
    callback = grp_mount_to_rate(
        min_dur=pd.Timedelta(minutes=1),
        extra_dur=pd.Timedelta(hours=1),
        grp_var="linkorderid",
    )
    frame = pd.DataFrame(
        {
            "icustay_id": [1, 1],
            "linkorderid": [8, 8],
            "itemid": [30016, 30016],
            "charttime": pd.to_datetime(
                ["2020-01-01 00:00", "2020-01-01 01:00"]
            ),
            "amount": ["bad", None],
        }
    )

    result = callback(
        frame,
        val_col="amount",
        index_var="charttime",
        id_cols=["icustay_id"],
        sub_var="itemid",
    )

    assert result.empty


def test_grouped_rate_dispatch_failure_does_not_return_raw_amounts():
    frame = pd.DataFrame(
        {
            "icustay_id": [1],
            "linkorderid": [8],
            "itemid": [30016],
            "dex": ["150"],
        }
    )

    with pytest.raises(ValueError, match="grp_mount_to_rate conversion failed"):
        _apply_callback(
            frame,
            _mimic_cv_source(callback="grp_mount_to_rate(mins(1L), hours(1L))"),
            concept_name="dex",
        )


def test_grouped_rate_rejects_unparseable_duration_instead_of_guessing():
    frame = pd.DataFrame(
        {
            "icustay_id": [1],
            "linkorderid": [8],
            "itemid": [30016],
            "charttime": pd.to_datetime(["2020-01-01"]),
            "dex": ["150"],
        }
    )

    with pytest.raises(ValueError, match="unsupported duration expression"):
        _apply_callback(
            frame,
            _mimic_cv_source(
                callback="grp_mount_to_rate(fortnights(1L), hours(1L))"
            ),
            concept_name="dex",
        )
