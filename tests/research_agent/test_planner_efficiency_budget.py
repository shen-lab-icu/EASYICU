"""Development Planner efficiency hard-stop contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from easyicu.research_agent.providers.efficiency_budget import (
    PlannerEfficiencyBudgetClient,
    PlannerEfficiencyBudgetExhausted,
    PlannerEfficiencyLimits,
)


class _UsageClient:
    def __init__(self, usages: Sequence[Mapping[str, int] | None]) -> None:
        self.usages = list(usages)
        self.calls = 0

    def complete_with_usage(
        self,
        messages: Sequence[Any],
        **kwargs: Any,
    ) -> tuple[str, Mapping[str, int] | None]:
        del messages, kwargs
        usage = self.usages[self.calls]
        self.calls += 1
        return "ok", usage


def _client(
    usages: Sequence[Mapping[str, int] | None],
    *,
    max_calls: int = 2,
    max_reported_tokens: int = 100,
    max_wall_seconds: float = 60.0,
) -> tuple[PlannerEfficiencyBudgetClient, _UsageClient]:
    inner = _UsageClient(usages)
    return (
        PlannerEfficiencyBudgetClient(
            inner,
            limits=PlannerEfficiencyLimits(
                max_calls=max_calls,
                max_reported_tokens=max_reported_tokens,
                max_wall_seconds=max_wall_seconds,
            ),
        ),
        inner,
    )


def test_planner_efficiency_budget_stops_before_extra_transport_call() -> None:
    client, inner = _client(
        [{"total_tokens": 4}, {"total_tokens": 5}, {"total_tokens": 6}],
    )

    assert client.complete([]) == "ok"
    assert client.complete([]) == "ok"
    with pytest.raises(PlannerEfficiencyBudgetExhausted) as caught:
        client.complete([])

    assert caught.value.reason == "call_limit"
    assert inner.calls == 2
    assert caught.value.easyicu_safe_diagnostic["calls"] == 2
    assert caught.value.easyicu_safe_diagnostic["reported_tokens"] == 9


def test_planner_efficiency_budget_stops_after_reported_token_limit() -> None:
    client, inner = _client(
        [{"total_tokens": 11}, {"total_tokens": 1}],
        max_calls=5,
        max_reported_tokens=10,
    )

    assert client.complete([]) == "ok"
    with pytest.raises(PlannerEfficiencyBudgetExhausted) as caught:
        client.complete([])

    assert caught.value.reason == "reported_token_limit"
    assert inner.calls == 1
    assert client.efficiency_snapshot()["reported_tokens"] == 11


def test_planner_efficiency_budget_fails_closed_without_usage() -> None:
    client, inner = _client([None, {"total_tokens": 1}], max_calls=5)

    assert client.complete([]) == "ok"
    with pytest.raises(PlannerEfficiencyBudgetExhausted) as caught:
        client.complete([])

    assert caught.value.reason == "provider_usage_unavailable"
    assert inner.calls == 1
    assert client.efficiency_snapshot()["usage_available"] is False


def test_planner_efficiency_budget_stops_on_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((0.0, 2.0, 2.0))
    monkeypatch.setattr(
        "easyicu.research_agent.providers.efficiency_budget.time.monotonic",
        lambda: next(ticks),
    )
    client, inner = _client(
        [{"total_tokens": 1}],
        max_wall_seconds=1.0,
    )

    with pytest.raises(PlannerEfficiencyBudgetExhausted) as caught:
        client.complete([])

    assert caught.value.reason == "wall_clock_limit"
    assert inner.calls == 0
    assert caught.value.easyicu_safe_diagnostic["elapsed_seconds"] == 2.0


@pytest.mark.parametrize(
    "limits",
    [
        {"max_calls": 0, "max_reported_tokens": 1, "max_wall_seconds": 1.0},
        {"max_calls": 1, "max_reported_tokens": 0, "max_wall_seconds": 1.0},
        {"max_calls": 1, "max_reported_tokens": 1, "max_wall_seconds": 0.0},
    ],
)
def test_planner_efficiency_limits_must_be_positive(
    limits: dict[str, int | float],
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        PlannerEfficiencyLimits(**limits)
