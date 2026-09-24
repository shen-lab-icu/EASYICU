"""A failed step's retry basis is read from the run's own attempt records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from easyicu.research_agent.execution.retry_basis import (
    AttemptIdentity,
    load_failed_step_retry_basis,
)


def _run(tmp_path: Path, records: list[dict[str, Any]], **payload: Any) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"per_step_records": records, **payload}), encoding="utf-8"
    )
    return run_dir


def _failed(**fields: Any) -> dict[str, Any]:
    return {
        "step_id": "figure",
        "status": "contract_failed",
        "step_llm_repair_attempts": 2,
        "step_llm_repair_budget": 2,
        "step_provider_call_attempts": 4,
        "step_provider_call_budget": 9,
        "step_provider_call_remaining": 5,
        **fields,
    }


def test_a_code_failure_with_spent_repairs_is_exhausted(tmp_path: Path) -> None:
    basis = load_failed_step_retry_basis(_run(tmp_path, [_failed()]), "figure")

    assert basis is not None
    assert (basis.status, basis.failure_class) == ("contract_failed", "code")
    assert (basis.repair_attempts, basis.repair_limit) == (2, 2)
    assert (basis.provider_calls, basis.provider_limit) == (4, 9)
    assert basis.repair_budget_exhausted is True
    # No capsule reference: the identity is unknown, never guessed.
    assert basis.identity is None


def test_the_budget_is_the_monotonic_maximum_over_attempts(tmp_path: Path) -> None:
    # A resumed attempt that stopped before copying its counter must not buy
    # a fresh budget: the earlier attempt's two repairs still count.
    history = [_failed(), _failed(step_llm_repair_attempts=0)]
    run_dir = _run(
        tmp_path,
        [_failed(step_llm_repair_attempts=0)],
        step_attempt_history=history,
    )

    basis = load_failed_step_retry_basis(run_dir, "figure")

    assert basis is not None
    assert basis.repair_attempts == 2
    assert basis.repair_budget_exhausted is True


def test_repairs_left_or_a_spent_provider_budget_are_read_as_recorded(
    tmp_path: Path,
) -> None:
    remaining = load_failed_step_retry_basis(
        _run(tmp_path, [_failed(step_llm_repair_attempts=1)]), "figure"
    )
    assert remaining is not None and remaining.repair_budget_exhausted is False

    spent = load_failed_step_retry_basis(
        _run(
            tmp_path / "spent",
            [_failed(step_llm_repair_attempts=1, step_provider_call_remaining=0)],
        ),
        "figure",
    )
    assert spent is not None and spent.repair_budget_exhausted is True


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({"timed_out": True}, "timeout"),
        ({"runtime_failure_class": "execution_timeout"}, "timeout"),
        ({"returncode": 137}, "infrastructure"),
        ({"runner_failure_code": "docker_unavailable"}, "infrastructure"),
        ({"status": "execution_environment_failed"}, "infrastructure"),
        ({"status": "execution_raised"}, "host_exception"),
        ({"runtime_repair_route": "fail_closed"}, "fail_closed"),
        ({"capsule_pending_execution": True}, "pending"),
        ({"step_llm_repair_history_invalid": True}, "ledger_invalid"),
        ({"status": "blocked"}, "other"),
        ({"status": "execution_failed"}, "code"),
        ({"status": "repair_failed"}, "code"),
    ],
)
def test_only_a_script_that_ran_and_failed_is_a_code_failure(
    tmp_path: Path, fields: dict[str, Any], expected: str
) -> None:
    basis = load_failed_step_retry_basis(_run(tmp_path, [_failed(**fields)]), "figure")

    assert basis is not None
    assert basis.failure_class == expected


def test_a_step_without_a_record_or_a_readable_ledger_has_no_basis(
    tmp_path: Path,
) -> None:
    assert load_failed_step_retry_basis(_run(tmp_path, [_failed()]), "other") is None

    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "manifest.json").write_text("{", encoding="utf-8")
    assert load_failed_step_retry_basis(corrupt, "figure") is None

    assert load_failed_step_retry_basis(tmp_path / "missing", "figure") is None


def test_only_components_known_on_both_sides_can_count_as_changed() -> None:
    failed = AttemptIdentity("a" * 64, "b" * 64, None, "sha256:old")

    assert AttemptIdentity("a" * 64, "b" * 64, "k" * 64, None).changes_since(failed) == ()
    assert AttemptIdentity("c" * 64, "b" * 64, None, "sha256:new").changes_since(
        failed
    ) == ("engine_code_sha256", "image_id")
