"""E-value sensitivity to unmeasured confounding (VanderWeele & Ding 2017).

These lock the hand-verified reference values from the paper's canonical
smoking/lung-cancer example plus the closed-form ``RR + sqrt(RR*(RR-1))``
identities, and the confidence-interval logic (nearest-to-null limit, with a
crossing CI collapsing to an E-value of 1).
"""

from __future__ import annotations

import math

import pytest

from easyicu.research_agent.evalue import (
    EValueResult,
    evalue,
    evalue_ci,
    evalue_point,
)


# ---------------------------------------------------------------------------
# Reference values (hand-verified ground truth)
# ---------------------------------------------------------------------------


def test_point_canonical_smoking_example():
    # VanderWeele & Ding's RR = 3.9 smoking / lung-cancer example.
    assert evalue_point(3.9, "rr") == pytest.approx(7.26, abs=1e-2)


def test_point_rr_two():
    # 2 + sqrt(2 * 1) = 3.41421356...
    assert evalue_point(2.0, "rr") == pytest.approx(3.4142, abs=1e-2)


def test_point_symmetry_below_null():
    # 1 / 0.5 = 2, so the E-value matches RR = 2.
    assert evalue_point(0.5, "rr") == pytest.approx(3.4142, abs=1e-2)


def test_point_at_null_is_one():
    assert evalue_point(1.0, "rr") == pytest.approx(1.0, abs=1e-2)


def test_point_or_common_sqrt_conversion():
    # sqrt(4) = 2, so an OR of 4 (common outcome) matches an RR of 2.
    assert evalue_point(4.0, "or_common") == pytest.approx(
        evalue_point(2.0, "rr"), abs=1e-2
    )


def test_ci_above_null_uses_lower_limit():
    # 1.5 + sqrt(1.5 * 0.5) = 2.36602540...
    assert evalue_ci(1.5, 2.5, "rr") == pytest.approx(2.3660, abs=1e-2)


def test_ci_crossing_null_is_one():
    assert evalue_ci(0.9, 1.8, "rr") == pytest.approx(1.0, abs=1e-2)


def test_ci_below_null_uses_upper_limit():
    # Entirely below 1: nearest-to-null limit is the upper bound (0.8).
    assert evalue_ci(0.3, 0.8, "rr") == pytest.approx(
        evalue_point(0.8, "rr"), abs=1e-2
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_convenience_bundles_point_and_ci():
    result = evalue(3.9, 3.1, 4.9, "rr")
    assert isinstance(result, EValueResult)
    assert result.point_evalue == pytest.approx(7.26, abs=1e-2)
    # CI wholly above the null -> E-value from the lower limit 3.1.
    assert result.ci_evalue == pytest.approx(evalue_point(3.1, "rr"), abs=1e-2)
    # Frozen dataclass: fields are immutable.
    with pytest.raises(Exception):
        result.point_evalue = 0.0  # type: ignore[misc]


def test_hr_and_or_rare_treated_as_direct_risk_ratio():
    # hr and or_rare read the estimate directly as RR, so all three agree.
    assert evalue_point(2.5, "hr") == pytest.approx(evalue_point(2.5, "rr"))
    assert evalue_point(2.5, "or_rare") == pytest.approx(
        evalue_point(2.5, "rr")
    )


def test_ci_limit_exactly_at_null_collapses_to_one():
    # ci_low == 1 is not strictly above the null -> crossing branch.
    assert evalue_ci(1.0, 2.0, "rr") == pytest.approx(1.0, abs=1e-2)
    # ci_high == 1 is not strictly below the null -> crossing branch.
    assert evalue_ci(0.4, 1.0, "rr") == pytest.approx(1.0, abs=1e-2)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        evalue_point(2.0, "bogus")  # unknown kind
    with pytest.raises(ValueError):
        evalue_point(0.0, "rr")  # non-positive ratio
    with pytest.raises(ValueError):
        evalue_point(math.inf, "rr")  # non-finite
    with pytest.raises(ValueError):
        evalue_ci(2.5, 1.5, "rr")  # low > high
