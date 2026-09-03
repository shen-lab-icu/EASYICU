"""Regression tests for ``ConceptLoader._convert_units``.

The method used to be a no-op pass-through, which silently dropped the
``concept.unit`` contract. These tests pin the new behaviour: rows are
converted to the target unit when the source unit is recognised, mixed
units in the same frame are each converted independently, and unknown
units are left untouched (no exceptions).
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.load_concepts import ConceptLoader


def _make_loader() -> ConceptLoader:
    # Bypass __init__ side effects (which would require a configured
    # datasource); only the unbound ``_convert_units`` helper is exercised.
    return ConceptLoader.__new__(ConceptLoader)


def test_convert_units_no_op_when_unit_column_missing() -> None:
    loader = _make_loader()
    df = pd.DataFrame({"value": [1.0, 2.0]})
    out = loader._convert_units(df, target_unit="mmol/l")
    pd.testing.assert_frame_equal(out, df)


def test_convert_units_converts_glucose_mg_dl_to_mmol_l() -> None:
    loader = _make_loader()
    df = pd.DataFrame(
        {
            "value": [180.16, 90.08],
            "unit": ["mg/dl", "mg/dl"],
        }
    )
    out = loader._convert_units(df, target_unit="mmol/l")
    assert list(out["unit"]) == ["mmol/l", "mmol/l"]
    assert out["value"].iloc[0] == pytest.approx(10.0, rel=1e-3)
    assert out["value"].iloc[1] == pytest.approx(5.0, rel=1e-3)


def test_convert_units_handles_mixed_source_units() -> None:
    loader = _make_loader()
    df = pd.DataFrame(
        {
            "value": [180.16, 5.0, 7.5],
            "unit": ["mg/dl", "mmol/l", "unknown_unit"],
        }
    )
    out = loader._convert_units(df, target_unit="mmol/l")
    # Convertible row is converted.
    assert out.loc[0, "unit"] == "mmol/l"
    assert out.loc[0, "value"] == pytest.approx(10.0, rel=1e-3)
    # Same-unit row is preserved exactly.
    assert out.loc[1, "unit"] == "mmol/l"
    assert out.loc[1, "value"] == pytest.approx(5.0, rel=1e-3)
    # Unknown unit is left as-is, never raises.
    assert out.loc[2, "unit"] == "unknown_unit"
    assert out.loc[2, "value"] == pytest.approx(7.5, rel=1e-3)


def test_convert_units_returns_input_when_target_unit_blank() -> None:
    loader = _make_loader()
    df = pd.DataFrame({"value": [1.0], "unit": ["mg/dl"]})
    out = loader._convert_units(df, target_unit="")
    pd.testing.assert_frame_equal(out, df)
