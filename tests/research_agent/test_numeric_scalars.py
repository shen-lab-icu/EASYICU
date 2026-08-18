from __future__ import annotations

import pytest

from easyicu.research_agent.numeric_scalars import (
    coerce_finite_float,
    coerce_optional_finite_float,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, 1.0), ("2.5", 2.5), (True, 1.0), (None, None), ("x", None)],
)
def test_optional_finite_float_preserves_legacy_coercion(value, expected) -> None:
    assert coerce_optional_finite_float(value) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "-inf"])
def test_optional_finite_float_rejects_nonfinite_values(value) -> None:
    assert coerce_optional_finite_float(value) is None


def test_required_finite_float_uses_stable_diagnostic_labels() -> None:
    with pytest.raises(ValueError, match="estimate is not numeric"):
        coerce_finite_float("x", label="estimate")
    with pytest.raises(ValueError, match="estimate is not finite"):
        coerce_finite_float(float("nan"), label="estimate")
