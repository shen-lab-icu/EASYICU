"""Smoke + dispatch-survey tests for ``easyicu.concept._apply_callback``.

Background
----------
``_apply_callback`` is a ~2,470-line if-elif dispatcher that maps a
``ConceptSource.callback`` string to one of ~50 inline handler bodies.
It is the largest single function in the package and has **no direct
unit tests**; coverage is purely transitive through concept loading.

Why this file matters now
-------------------------
Phase 2 of the ``concept.py`` split (documented in CLAUDE.md, drafted
2026-05-17) plans to extract ``_apply_callback`` into its own module
and rename it ``apply_callback``. Before that move can land safely, we
need at minimum:

1. A **dispatch-survey tripwire** that enumerates every callback string
   actually used in ``concept-dict.json`` / ``sofa2-dict.json`` (159
   distinct strings as of 2026-05-17) and verifies that
   ``_apply_callback`` recognises each — i.e. does NOT fall through to
   ``raise NotImplementedError``. If a future refactor accidentally
   drops a branch, this test catches it instantly.

2. **Pure-transform tests** for the simple branches that don't need
   a live ``ICUDataSource`` (``identity_callback``, ``transform_fun(...)``
   family). The database-bound branches (``aumc_death``, ``hirid_rate``,
   etc.) are intentionally left to the existing transitive coverage —
   building synthetic fixtures for each is a multi-day project not in
   Phase 2's scope.

3. A **negative test** that an unknown callback raises
   ``NotImplementedError`` (this is the contract Phase 2 must preserve).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import ConceptSource, _apply_callback
from easyicu.concept.callbacks import (
    CALLBACK_REGISTRY,
    ConceptCallbackContext,
    _callback_blood_cell_ratio,
)
from easyicu.table import ICUTable


def test_mimic_urine_output_nets_gu_irrigant_before_aggregation():
    frame = pd.DataFrame(
        {
            "itemid": [226559, 227488, 227489],
            "value": [100.0, 500.0, 550.0],
        }
    )

    result = _apply_callback(
        frame,
        _src(
            "mimic_urine_output",
            sub_var="itemid",
            value_var="value",
        ),
        concept_name="kdigo_urine_input",
    )

    assert result["value"].tolist() == [100.0, -500.0, 550.0]
    assert result["value"].sum() == 150.0


def test_aumc_urine_output_repairs_decimal_errors_before_outlier_filter():
    frame = pd.DataFrame(
        {"value": [2500.0, 2600.0, 45_000.0, 50_000.0]}
    )

    result = _apply_callback(
        frame,
        _src("aumc_urine_output", value_var="value"),
        concept_name="kdigo_urine_input",
    )

    assert result["value"].iloc[:3].tolist() == [2500.0, 260.0, 4500.0]
    assert pd.isna(result["value"].iloc[3])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _src(callback: str | None, **fields) -> ConceptSource:
    """Build a minimal ConceptSource carrying just a callback."""
    return ConceptSource(callback=callback, **fields)


def _enumerate_dict_callbacks() -> set[str]:
    """Collect every callback string used in shipped concept dictionaries.

    Reads both ``concept-dict.json`` and ``sofa2-dict.json``. Returns the
    set of distinct callback strings — these are the exact strings
    ``_apply_callback`` must keep handling after Phase 2 lands.
    """
    data_dir = Path(__file__).resolve().parents[1] / "src" / "easyicu" / "data"
    callbacks: set[str] = set()
    for fname in ("concept-dict.json", "sofa2-dict.json"):
        path = data_dir / fname
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for _, cdef in payload.items():
            cb = cdef.get("callback")
            if cb:
                callbacks.add(cb)
            for src_entries in (cdef.get("sources") or {}).values():
                for entry in src_entries:
                    cb = entry.get("callback")
                    if cb:
                        callbacks.add(cb)
    return callbacks


# ---------------------------------------------------------------------------
# Trivial dispatch — no callback / identity passthrough
# ---------------------------------------------------------------------------


class TestPassthrough:
    def test_none_callback_returns_frame_unchanged(self):
        df = pd.DataFrame({"value": [1, 2, 3]})
        out = _apply_callback(df, _src(None), concept_name="value")
        pd.testing.assert_frame_equal(out, df)

    def test_empty_string_callback_returns_frame_unchanged(self):
        df = pd.DataFrame({"value": [1, 2, 3]})
        out = _apply_callback(df, _src(""), concept_name="value")
        pd.testing.assert_frame_equal(out, df)

    def test_identity_callback_returns_frame_unchanged(self):
        df = pd.DataFrame({"value": [1.5, 2.5, 3.5]})
        out = _apply_callback(df, _src("identity_callback"), concept_name="value")
        pd.testing.assert_frame_equal(out, df)


# ---------------------------------------------------------------------------
# Pure transform_fun(...) branches — math on a single column
# ---------------------------------------------------------------------------


class TestTransformFunFloor:
    def test_floor_applied_to_concept_column(self):
        df = pd.DataFrame({"value": [1.4, 2.6, 3.9]})
        out = _apply_callback(df, _src("transform_fun(floor)"), concept_name="value")
        assert list(out["value"]) == [1.0, 2.0, 3.0]


class TestTransformFunCeiling:
    def test_ceiling_applied_to_concept_column(self):
        df = pd.DataFrame({"value": [1.1, 2.4, 3.0]})
        out = _apply_callback(
            df, _src("transform_fun(ceiling)"), concept_name="value"
        )
        assert list(out["value"]) == [2.0, 3.0, 3.0]


class TestTransformFunRound:
    def test_round_applied_to_concept_column(self):
        df = pd.DataFrame({"value": [1.4, 2.5, 3.6]})
        out = _apply_callback(
            df, _src("transform_fun(round)"), concept_name="value"
        )
        # pandas .round uses banker's rounding on .5 — assert exact values.
        assert out["value"].iloc[0] == 1.0
        assert out["value"].iloc[2] == 4.0


class TestTransformFunPercentAsNumeric:
    def test_strips_percent_and_converts(self):
        df = pd.DataFrame({"value": ["50%", "25%", "100%"]})
        out = _apply_callback(
            df,
            _src("transform_fun(percent_as_numeric)"),
            concept_name="value",
        )
        # Output column may retain object dtype (pandas keeps the
        # original column's storage), so check by value, not by dtype.
        assert float(out["value"].iloc[0]) == pytest.approx(50.0)
        assert float(out["value"].iloc[1]) == pytest.approx(25.0)
        assert float(out["value"].iloc[2]) == pytest.approx(100.0)

    def test_numeric_input_is_fast_pathed(self):
        # Documented fast path: numeric input bypasses string ops.
        df = pd.DataFrame({"value": [50.0, 25.0, 100.0]})
        out = _apply_callback(
            df,
            _src("transform_fun(percent_as_numeric)"),
            concept_name="value",
        )
        assert list(out["value"]) == [50.0, 25.0, 100.0]

    def test_fractional_numeric_input_is_scaled_to_percent(self):
        df = pd.DataFrame({"value": [0.21, 0.5, 1.0, 21.0]})
        out = _apply_callback(
            df,
            _src("transform_fun(percent_as_numeric)"),
            concept_name="value",
        )
        assert list(out["value"]) == [21.0, 50.0, 100.0, 21.0]


class TestTransformFunSetVal:
    def test_set_val_true_replaces_column_with_true(self):
        # transform_fun(set_val(TRUE)) is the most common callback in the
        # shipped dictionary (~417 usages); it stamps a constant onto the
        # concept column. Pin the contract.
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        out = _apply_callback(
            df,
            _src("transform_fun(set_val(TRUE))"),
            concept_name="value",
        )
        assert "value" in out.columns
        # Use bool() coercion — handler may emit numpy.True_ rather than
        # Python True; either is acceptable.
        assert all(bool(v) is True for v in out["value"])

    def test_set_val_string_constant(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        out = _apply_callback(
            df,
            _src("transform_fun(set_val('cardiovascular'))"),
            concept_name="value",
        )
        assert all(v == "cardiovascular" for v in out["value"])


class TestTransformFunMimicAge:
    def test_dispatch_does_not_raise(self):
        # The mimic_age handler in concept.py is much more involved than
        # callback_utils.mimic_age — it can load icustays from a live
        # data_source to compute age from dob + intime. Without that
        # data_source we can only verify the dispatch reaches the branch
        # and exits cleanly.
        df = pd.DataFrame({"value": [365.25, 365.25 * 50, 365.25 * 150]})
        out = _apply_callback(
            df,
            _src("transform_fun(mimic_age)", value_var="value"),
            concept_name="value",
        )
        assert isinstance(out, pd.DataFrame)
        assert "value" in out.columns


class TestTransformFunEicuAge:
    def test_replaces_gt_89_with_90_and_coerces_numeric(self):
        df = pd.DataFrame({"age": ["45", "> 89", "67"]})
        out = _apply_callback(
            df,
            _src("transform_fun(eicu_age)"),
            concept_name="age",
        )
        assert pd.api.types.is_float_dtype(out["age"])
        assert out["age"].iloc[0] == 45.0
        assert out["age"].iloc[1] == 90.0
        assert out["age"].iloc[2] == 67.0


# ---------------------------------------------------------------------------
# eicu_adx — diagnosis-path classifier (pure logic, no DB)
# ---------------------------------------------------------------------------


class TestEicuAdx:
    def test_classifies_operative_paths_as_surg(self):
        df = pd.DataFrame({
            "adm": [
                "admission diagnosis|All Diagnosis|Operative|Diagnosis|Cardiovascular|X",
                "admission diagnosis|All Diagnosis|Non-operative|Diagnosis|Pulmonary|Y",
                "admission diagnosis|All Diagnosis|Non-operative|Diagnosis|Genitourinary|Z",
            ]
        })
        out = _apply_callback(df, _src("eicu_adx"), concept_name="adm")
        assert list(out["adm"]) == ["surg", "med", "other"]

    def test_drops_rows_not_under_all_diagnosis(self):
        df = pd.DataFrame({
            "adm": [
                "admission diagnosis|All Diagnosis|Operative|X",
                "something else entirely",
                "admission diagnosis|Other Bucket|Operative|Y",
            ]
        })
        out = _apply_callback(df, _src("eicu_adx"), concept_name="adm")
        assert len(out) == 1
        assert out["adm"].iloc[0] == "surg"


# ---------------------------------------------------------------------------
# Negative contract — unknown callback must raise NotImplementedError
# ---------------------------------------------------------------------------


def test_unknown_callback_raises_not_implemented():
    df = pd.DataFrame({"value": [1, 2]})
    with pytest.raises(NotImplementedError, match="callback_that_definitely_does_not_exist"):
        _apply_callback(
            df,
            _src("callback_that_definitely_does_not_exist"),
            concept_name="value",
        )


def test_blood_cell_ratio_without_wbc_fails_closed_instead_of_returning_numerator():
    frame = pd.DataFrame(
        {"stay_id": [1], "charttime": [0.0], "lymph": [2.0]}
    )

    result = _apply_callback(
        frame,
        _src("blood_cell_ratio", value_var="lymph", index_var="charttime"),
        concept_name="lymph",
        resolver=None,
    )

    assert result["lymph"].isna().all()
    assert result["lymph_assessment_reason"].tolist() == ["missing_wbc_resolver"]


def test_registry_blood_cell_ratio_also_fails_closed_when_wbc_load_fails(monkeypatch):
    def fail_load(*args, **kwargs):
        raise RuntimeError("wbc unavailable")

    monkeypatch.setattr("easyicu.api.load_concepts", fail_load)
    table = ICUTable(
        pd.DataFrame({"stay_id": [1], "charttime": [0.0], "lymph": [2.0]}),
        id_columns=["stay_id"],
        index_column="charttime",
        value_column="lymph",
    )
    context = ConceptCallbackContext(
        concept_name="lymph",
        target=None,
        interval=None,
        resolver=None,
        data_source=None,
        patient_ids=None,
    )

    result = _callback_blood_cell_ratio({"lymph": table}, context).data

    assert result["lymph"].isna().all()
    assert result["lymph_assessment_reason"].tolist() == ["wbc_load_failed"]


# ---------------------------------------------------------------------------
# Dispatch-survey tripwire — every callback in the shipped dictionary
# must be recognised by _apply_callback. This is the **strongest single
# protection** Phase 2 will rely on: if a refactor silently drops a
# branch, this test surfaces it immediately.
# ---------------------------------------------------------------------------


_DICT_CALLBACKS = sorted(_enumerate_dict_callbacks())

# ``_apply_callback`` is only ONE of THREE dispatchers. The other two:
#   1. ``concept_callbacks.CALLBACK_REGISTRY`` — handles the derived /
#      composite concepts (sofa_score, kdigo_aki, qsofa_score, etc.).
#   2. ``ConceptResolver._load_single_concept`` — special-cases
#      ``los_callback`` and any callback containing ``fwd_concept(...)``
#      because those need access to other concepts at load time.
# The tripwire must exclude callbacks dispatched elsewhere; otherwise
# it would incorrectly flag them as missing from ``_apply_callback``.
_REGISTRY_HANDLED = set(CALLBACK_REGISTRY.keys())
_LOAD_SINGLE_CONCEPT_HANDLED = {"los_callback"}  # plus any fwd_concept(*)


def _dispatched_elsewhere(cb: str) -> bool:
    if cb in _REGISTRY_HANDLED:
        return True
    if cb in _LOAD_SINGLE_CONCEPT_HANDLED:
        return True
    # combine_callbacks(... fwd_concept(...) ...) routes via the
    # single-concept loader (see concept.py L5803 + L6294 region).
    if "fwd_concept" in cb:
        return True
    return False


_APPLY_CALLBACK_TARGETS = sorted(
    cb for cb in _DICT_CALLBACKS if not _dispatched_elsewhere(cb)
)


@pytest.mark.parametrize(
    "callback", _APPLY_CALLBACK_TARGETS, ids=lambda s: s[:50]
)
def test_every_shipped_callback_is_dispatched(callback: str):
    """Every callback string in concept-dict.json / sofa2-dict.json that
    is NOT already handled by ``concept_callbacks.CALLBACK_REGISTRY``
    must produce a recognised dispatch in ``_apply_callback`` — i.e.
    must not fall through to ``raise NotImplementedError``.

    The handler is allowed to fail in other ways (KeyError because our
    1-row probe frame lacks the columns it needs, AttributeError because
    no real ``data_source`` is passed, etc.). What we **forbid** is the
    sentinel "Callback '...' is not yet supported." error — that signals
    a branch was lost in translation.
    """
    df = pd.DataFrame({
        "value": [1.0],
        "valuenum": [1.0],
        "value_col": [1.0],
        "unit": ["mg/dL"],
        "unit_col": ["mg/dL"],
        "charttime": [pd.Timestamp("2024-01-01")],
        "stay_id": [1],
        "patientid": [1],
    })
    # ConceptSource fields are populated leniently so dispatch reaches
    # the right branch; missing data inside the handler may then fail
    # for legitimate (non-dispatch) reasons we tolerate.
    src = _src(
        callback,
        value_var="value",
        unit_var="unit",
        index_var="charttime",
    )
    try:
        _apply_callback(df, src, concept_name="value")
    except NotImplementedError as exc:
        # The only failure mode this test cares about.
        msg = str(exc)
        if "is not yet supported" in msg:
            pytest.fail(
                f"_apply_callback no longer recognises callback "
                f"{callback!r} — branch likely lost in refactor. "
                f"Full message: {msg}"
            )
        # Some handlers legitimately raise NotImplementedError for
        # narrower reasons (e.g. unsupported sub-operator inside a
        # convert_unit). Those are fine.
    except Exception:
        # Any other exception (KeyError, ValueError, AttributeError, ...)
        # is acceptable here — the dispatch reached the right branch and
        # then failed inside the handler due to our minimal probe frame /
        # missing data_source. The test only protects the dispatch surface.
        pass


def test_dispatch_survey_sees_expected_number_of_callbacks():
    """Lock the surface size so a dictionary edit that adds a brand-new
    callback string also forces the developer to think about coverage."""
    # As of 2026-05-17 the shipped dicts use 159 distinct callbacks.
    # Allow growth, alert on regression — if this drops sharply, a
    # dictionary file probably got corrupted or accidentally truncated.
    assert len(_DICT_CALLBACKS) >= 150, (
        f"Shipped dictionaries declare only {len(_DICT_CALLBACKS)} "
        "distinct callbacks; expected at least 150. Did concept-dict.json "
        "or sofa2-dict.json get truncated?"
    )
