from __future__ import annotations

import pytest

from benchmarks.figure2_canonical9.prompt_preflight import (
    PromptPreflightError,
    _planner_limit_bytes,
)


def test_planner_preflight_uses_the_production_owned_limit() -> None:
    assert _planner_limit_bytes({"limit_bytes": 120_000}) == 120_000


@pytest.mark.parametrize(
    "metrics",
    ({}, {"limit_bytes": None}, {"limit_bytes": 0}, {"limit_bytes": -1}),
)
def test_planner_preflight_rejects_missing_or_invalid_production_limit(
    metrics: dict[str, object],
) -> None:
    with pytest.raises(PromptPreflightError, match="production limit"):
        _planner_limit_bytes(metrics)
