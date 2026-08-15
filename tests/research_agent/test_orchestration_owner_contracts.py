from __future__ import annotations

from types import SimpleNamespace

import pytest

from easyicu.research_agent.orchestration.progress import (
    NonFatalProgressCallbackError,
    ProgressControlSignal,
    ResumableProgressChannel,
    planner_retry_progress_callback,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.providers.structured_retry import StructuredRetryProgress


def test_resumable_progress_channel_projects_heartbeat_audit_and_ui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.research_agent.orchestration.progress as progress_module

    heartbeats: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    monkeypatch.setattr(
        progress_module,
        "record_active_run_progress",
        lambda **kwargs: heartbeats.append(kwargs),
    )

    class _AuditLogger:
        def emit(self, **kwargs):
            audits.append(kwargs)

    channel = ResumableProgressChannel(events.append)
    channel.bind_audit_logger(_AuditLogger())  # type: ignore[arg-type]
    channel.emit(
        "analysis",
        "Running the exact approved step.",
        status="complete",
        step_id="02_primary",
        run_id="run-safe",
    )

    assert heartbeats == [
        {
            "stage": "analysis",
            "message": "Running the exact approved step.",
            "status": "complete",
            "step_id": "02_primary",
            "phase_timeout_seconds": None,
            "run_id": "run-safe",
        }
    ]
    assert audits[0]["phase"] == "analysis"
    assert audits[0]["detail"] == {"run_id": "run-safe"}
    assert events[0]["status"] == "complete"
    assert events[0]["timestamp"]


def test_progress_callback_control_signal_propagates() -> None:
    class HostCancellation(ProgressControlSignal):
        pass

    def cancel(_event: dict[str, object]) -> None:
        raise HostCancellation("host requested cancellation")

    channel = ResumableProgressChannel(cancel)

    with pytest.raises(HostCancellation, match="host requested cancellation"):
        channel.emit("analysis", "Running the exact approved step.")


def test_generic_progress_callback_failure_is_ignored() -> None:
    def broken_observer(_event: dict[str, object]) -> None:
        raise RuntimeError("optional observer disconnected")

    channel = ResumableProgressChannel(broken_observer)

    channel.emit("analysis", "Running without the optional observer.")


def test_typed_nonfatal_progress_callback_failure_is_ignored() -> None:
    def unavailable_ui(_event: dict[str, object]) -> None:
        raise NonFatalProgressCallbackError("UI transport disconnected")

    channel = ResumableProgressChannel(unavailable_ui)

    channel.emit("analysis", "Running without the optional UI projection.")


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
