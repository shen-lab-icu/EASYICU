from __future__ import annotations

from types import SimpleNamespace

import pytest

from easyicu.research_agent.orchestration.progress import (
    planner_retry_progress_callback,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.providers.structured_retry import StructuredRetryProgress


def test_planner_retry_projection_exposes_only_bounded_progress() -> None:
    observed: list[tuple[tuple[object, ...], dict[str, object]]] = []
    callback = planner_retry_progress_callback(
        lambda *args, **kwargs: observed.append((args, kwargs)),
        run_id="run-safe",
    )

    callback(
        StructuredRetryProgress(
            role="planner",
            phase="rejected",
            attempt=2,
            total_attempts=3,
            error_class="private-validator-detail",
        )
    )

    args, kwargs = observed[0]
    assert args == (
        "planning",
        "Plan draft 2/3 did not satisfy the scientific contract; retrying.",
    )
    assert kwargs == {
        "current": 2,
        "total": 3,
        "status": "running",
        "run_id": "run-safe",
    }
    assert "private-validator-detail" not in repr(observed)


def test_runtime_authority_pair_preserves_the_owner_error() -> None:
    class AuthorityError(ValueError):
        pass

    class FailingAuthority:
        def validate_plan(self, _plan: object) -> None:
            raise AuthorityError("trajectory-plan-drift")

    authorities = ScientificRuntimeAuthorities(
        trajectory=FailingAuthority(),  # type: ignore[arg-type]
        current_case=None,
    )

    with pytest.raises(AuthorityError, match="trajectory-plan-drift"):
        authorities.validate_plan(SimpleNamespace())  # type: ignore[arg-type]
