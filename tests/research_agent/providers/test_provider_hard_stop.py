"""Durable run/batch Provider stop-loss regressions."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest


def _limits(**overrides):
    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopLimits,
    )

    values = {
        "max_provider_attempts_per_run": 3,
        "max_provider_attempts_per_batch": 6,
        "max_total_tokens_per_run": 1_000_000,
        "max_total_tokens_per_batch": 2_000_000,
        "max_estimated_cost_usd_per_batch": 10.0,
        "max_wall_clock_seconds_per_task": 60.0,
        "input_cost_usd_per_million_tokens": 1.0,
        "output_cost_usd_per_million_tokens": 2.0,
    }
    values.update(overrides)
    return ProviderHardStopLimits(**values)


def _ledger(tmp_path, *, task_ids=("E1",), limits=None):
    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopLedger,
    )

    return ProviderHardStopLedger(
        path=(tmp_path / "provider_progress.json").resolve(),
        task_ids=task_ids,
        limits=limits or _limits(),
        batch_id="test-batch",
    )


def _message(text="hello"):
    from easyicu.research_agent.providers.protocol import LLMMessage

    return [LLMMessage(role="user", content=text)]


def _pipeline_limit_options(limits):
    return {
        "max_provider_attempts_per_run": limits.max_provider_attempts_per_run,
        "max_provider_attempts_per_batch": limits.max_provider_attempts_per_batch,
        "max_total_tokens_per_run": limits.max_total_tokens_per_run,
        "max_total_tokens_per_batch": limits.max_total_tokens_per_batch,
        "max_estimated_cost_usd_per_batch": (
            limits.max_estimated_cost_usd_per_batch
        ),
        "max_wall_clock_seconds_per_task": limits.max_wall_clock_seconds_per_task,
        "provider_input_cost_usd_per_million_tokens": (
            limits.input_cost_usd_per_million_tokens
        ),
        "provider_output_cost_usd_per_million_tokens": (
            limits.output_cost_usd_per_million_tokens
        ),
    }


def _reserve_from_independent_process(
    path,
    limits,
    task_id,
    barrier,
    results,
):
    from pathlib import Path

    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopExceeded,
        ProviderHardStopLedger,
    )

    try:
        ledger = ProviderHardStopLedger(
            path=Path(path),
            task_ids=("E1", "E2"),
            limits=limits,
            batch_id="test-batch",
            resume_existing=True,
        )
        barrier.wait(timeout=10)
        ledger.start_task(task_id)
        barrier.wait(timeout=10)
        attempt_id = ledger.reserve_transport_attempt(
            task_id=task_id,
            role="planner",
            model="test-model",
            messages=[SimpleNamespace(content=task_id)],
            max_tokens=8,
            prior_attempt_id=None,
        )
        results.put((task_id, "authorized", attempt_id))
    except ProviderHardStopExceeded as exc:
        results.put((task_id, "blocked", exc.code))
    except BaseException as exc:  # pragma: no cover - reported to the parent
        results.put((task_id, "error", f"{type(exc).__name__}: {exc}"))


def test_pipeline_requires_matching_declarative_limits_and_live_service(
    tmp_path, ra
):
    from easyicu.research_agent.orchestration.services import PipelineServices
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    limits = _limits()
    task = _ledger(tmp_path, limits=limits).start_task("E1")
    config = ra.PipelineConfig(
        workdir=tmp_path / "run",
        **_pipeline_limit_options(limits),
    )
    explicit_auditor = ScriptedMockLLMClient(["{}"])
    explicit_vlm = ScriptedMockLLMClient(["{}"])
    pipeline = ra.ResearchAgentPipeline.from_config(
        config,
        services=PipelineServices(
            llm=ScriptedMockLLMClient(["{}"]),
            vlm_client=explicit_vlm,
            llm_concept_auditor_client=explicit_auditor,
            provider_hard_stop=task,
        ),
    )
    assert isinstance(pipeline._llm_concept_auditor_client, HardStopClient)
    assert isinstance(pipeline._vlm_client, HardStopClient)
    assert pipeline._vlm_client._inner is explicit_vlm

    with pytest.raises(ValueError, match="supplied together"):
        ra.ResearchAgentPipeline.from_config(
            config,
            services=PipelineServices(llm=ScriptedMockLLMClient(["{}"])),
        )
    with pytest.raises(ValueError, match="supplied together"):
        ra.ResearchAgentPipeline.from_config(
            ra.PipelineConfig(workdir=tmp_path / "missing-config"),
            services=PipelineServices(
                llm=ScriptedMockLLMClient(["{}"]),
                provider_hard_stop=task,
            ),
        )


def test_legacy_reviewed_client_is_reserved_before_invocation(tmp_path):
    from easyicu.research_agent.authority.provider_hard_stop import (
        load_provider_hard_stop_ledger,
    )
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    ledger = _ledger(tmp_path)
    task = ledger.start_task("E1")
    inner = ScriptedMockLLMClient(["ok"])
    client = HardStopClient(inner, role="planner", task=task)

    assert client.complete(_message(), max_tokens=32) == "ok"
    task.finish(score={"aware": {"run_id": "run-1"}})

    payload = load_provider_hard_stop_ledger(ledger.path)
    assert payload["terminal"] is True
    assert payload["totals"]["provider_attempts"] == 1
    call = payload["tasks"][0]["calls"][0]
    assert call["state"] == "completed_usage_unreported"
    assert call["role"] == "planner"
    assert "hello" not in json.dumps(payload)


def test_metadata_only_usage_cannot_release_the_transport_reservation(tmp_path):
    from easyicu.research_agent.providers.hard_stop import HardStopClient

    class MetadataOnlyUsageClient:
        name = "metadata-only"

        def complete_with_usage(self, messages, **kwargs):
            return "ok", {"actual_model": "provider/model"}

    ledger = _ledger(tmp_path)
    task = ledger.start_task("E1")
    response = HardStopClient(
        MetadataOnlyUsageClient(),
        role="planner",
        task=task,
    ).complete(_message(), max_tokens=32)

    assert response == "ok"
    call = ledger.snapshot()["tasks"][0]["calls"][0]
    assert call["state"] == "completed_usage_unreported"
    assert call["accounted_tokens"] > 0
    assert call["accounted_estimated_cost_usd"] > 0


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 4},
        {"completion_tokens": 4},
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        {"prompt_tokens": "invalid", "completion_tokens": 4},
        {"prompt_tokens": True, "completion_tokens": True},
        {"prompt_tokens": 4.9, "completion_tokens": 4},
        {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 2},
        {"prompt_tokens": 100, "total_tokens": 1},
        {"completion_tokens": 100, "total_tokens": 1},
        {"prompt_tokens": 6, "completion_tokens": 6, "total_tokens": 10},
    ],
)
def test_partial_or_zero_usage_cannot_release_the_transport_reservation(
    tmp_path,
    usage,
):
    from easyicu.research_agent.providers.hard_stop import HardStopClient

    class PartialUsageClient:
        name = "partial-usage"

        def complete_with_usage(self, messages, **kwargs):
            return "ok", usage

    ledger = _ledger(tmp_path)
    task = ledger.start_task("E1")
    response = HardStopClient(
        PartialUsageClient(),
        role="planner",
        task=task,
    ).complete(_message(), max_tokens=32)

    assert response == "ok"
    call = ledger.snapshot()["tasks"][0]["calls"][0]
    assert call["state"] == "completed_usage_unreported"
    assert call["accounted_tokens"] > 0
    assert call["accounted_estimated_cost_usd"] > 0


def test_explicit_resume_reopens_completed_task_without_resetting_limits(
    tmp_path,
):
    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopExceeded,
        ProviderHardStopLedger,
    )
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    limits = _limits(
        max_provider_attempts_per_run=2,
        max_provider_attempts_per_batch=2,
    )
    ledger = _ledger(tmp_path, limits=limits)
    first = ledger.start_task("E1")
    assert (
        HardStopClient(
            ScriptedMockLLMClient(["first"]),
            role="planner",
            task=first,
        ).complete(_message(), max_tokens=8)
        == "first"
    )
    first.finish(score={"aware": {"run_id": "run-first"}})
    first_terminal = ledger.snapshot()
    ledger = ProviderHardStopLedger(
        path=ledger.path,
        task_ids=("E1",),
        limits=limits,
        batch_id="test-batch",
        resume_existing=True,
    )

    # Ordinary reuse never reopens or downgrades a completed task.
    completed = ledger.start_task("E1")
    completed.finish(error="RuntimeError: later caller failure")
    assert ledger.snapshot()["tasks"][0]["status"] == "completed"

    resumed = ledger.start_task("E1", reopen_terminal=True)
    reopened = ledger.snapshot()
    reopened_task = reopened["tasks"][0]
    assert reopened_task["status"] == "running"
    assert reopened["terminal"] is False
    assert reopened_task["resume_count"] == 1
    assert len(reopened_task["calls"]) == 1
    assert reopened_task["terminal_attempts"][0]["status"] == "completed"
    assert (
        reopened_task["terminal_attempts"][0]["score_summary"]["run_id"]
        == "run-first"
    )

    second_client = HardStopClient(
        ScriptedMockLLMClient(["second", "must-not-run"]),
        role="coder",
        task=resumed,
    )
    assert second_client.complete(_message("resume"), max_tokens=8) == "second"
    with pytest.raises(ProviderHardStopExceeded, match="RUN_PROVIDER_ATTEMPT_LIMIT"):
        second_client.complete(_message("third"), max_tokens=8)
    resumed.finish(score={"aware": {"run_id": "run-first"}})

    final = ledger.snapshot()
    assert final["tasks"][0]["status"] == "completed"
    assert final["totals"]["provider_attempts"] == 2
    assert (
        final["tasks"][0]["elapsed_seconds"]
        >= first_terminal["tasks"][0]["elapsed_seconds"]
    )


def test_run_attempt_limit_blocks_before_second_client_call(tmp_path):
    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopExceeded,
    )
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    ledger = _ledger(
        tmp_path,
        limits=_limits(
            max_provider_attempts_per_run=1,
            max_provider_attempts_per_batch=1,
        ),
    )
    task = ledger.start_task("E1")
    inner = ScriptedMockLLMClient(["one", "must-not-run"])
    client = HardStopClient(inner, role="coder", task=task)

    assert client.complete(_message(), max_tokens=8) == "one"
    with pytest.raises(ProviderHardStopExceeded, match="RUN_PROVIDER_ATTEMPT_LIMIT"):
        client.complete(_message("second"), max_tokens=8)

    assert len(inner.calls) == 1
    assert ledger.snapshot()["totals"]["provider_attempts"] == 1


def test_batch_attempt_limit_blocks_next_task_before_client_call(tmp_path):
    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopExceeded,
    )
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    ledger = _ledger(
        tmp_path,
        task_ids=("E1", "E2"),
        limits=_limits(
            max_provider_attempts_per_run=1,
            max_provider_attempts_per_batch=1,
        ),
    )
    first = HardStopClient(
        ScriptedMockLLMClient(["one"]),
        role="planner",
        task=ledger.start_task("E1"),
    )
    second_inner = ScriptedMockLLMClient(["must-not-run"])
    second = HardStopClient(
        second_inner,
        role="planner",
        task=ledger.start_task("E2"),
    )

    assert first.complete(_message(), max_tokens=8) == "one"
    with pytest.raises(ProviderHardStopExceeded, match="BATCH_PROVIDER_ATTEMPT_LIMIT"):
        second.complete(_message(), max_tokens=8)

    assert second_inner.calls == []


def test_one_attempt_batch_serializes_independently_loaded_processes(tmp_path):
    import multiprocessing

    from easyicu.research_agent.authority.provider_hard_stop import (
        load_provider_hard_stop_ledger,
    )

    limits = _limits(
        max_provider_attempts_per_run=1,
        max_provider_attempts_per_batch=1,
    )
    ledger = _ledger(tmp_path, task_ids=("E1", "E2"), limits=limits)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_reserve_from_independent_process,
            args=(str(ledger.path), limits, task_id, barrier, results),
        )
        for task_id in ("E1", "E2")
    ]

    for process in processes:
        process.start()
    outcomes = [results.get(timeout=20) for _ in processes]
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    results.close()
    results.join_thread()

    assert sorted(outcome[1] for outcome in outcomes) == ["authorized", "blocked"]
    assert [outcome[2] for outcome in outcomes if outcome[1] == "blocked"] == [
        "BATCH_PROVIDER_ATTEMPT_LIMIT"
    ]
    payload = load_provider_hard_stop_ledger(ledger.path)
    tasks = {task["task_id"]: task for task in payload["tasks"]}
    assert {task_id: task["status"] for task_id, task in tasks.items()} == {
        "E1": "running",
        "E2": "running",
    }
    assert all(task["started_at"] is not None for task in tasks.values())
    assert sum(len(task["calls"]) for task in tasks.values()) == 1
    assert payload["totals"]["provider_attempts"] == 1


def test_windows_sidecar_locks_before_initializing_first_byte(
    tmp_path, monkeypatch
) -> None:
    from easyicu.research_agent.authority import provider_hard_stop as owner

    events: list[str] = []

    class _Msvcrt:
        LK_LOCK = 1
        LK_UNLCK = 2

        @staticmethod
        def locking(descriptor, operation, _length):
            if operation == _Msvcrt.LK_LOCK:
                assert owner.os.fstat(descriptor).st_size == 0
                events.append("lock")
            else:
                events.append("unlock")

    monkeypatch.setattr(owner, "fcntl", None)
    monkeypatch.setattr(owner, "msvcrt", _Msvcrt)
    ledger_path = (tmp_path / "windows-first-create.json").resolve()

    with owner._exclusive_ledger_file_lock(ledger_path):
        lock_path = ledger_path.parent / f".{ledger_path.name}.lock"
        assert lock_path.read_bytes() == b"\0"
        assert events == ["lock"]

    assert events == ["lock", "unlock"]


def test_pending_review_reconciles_a_post_resume_crash_to_the_original_pause(
    tmp_path, monkeypatch
):
    from easyicu.research_agent.authority import provider_hard_stop as owner

    wall = [datetime(2026, 8, 14, tzinfo=timezone.utc)]
    monotonic = [100.0]

    class _ClockDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = wall[0]
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr(owner, "datetime", _ClockDateTime)
    monkeypatch.setattr(owner.time, "monotonic", lambda: monotonic[0])
    ledger = _ledger(tmp_path)
    task = ledger.start_task("E1")
    wall[0] += timedelta(seconds=5)
    monotonic[0] += 5.0
    checkpoint_at = wall[0].isoformat()
    wall[0] += timedelta(seconds=2)
    monotonic[0] += 2.0
    task.pause()

    assert ledger.snapshot()["tasks"][0]["elapsed_seconds"] == pytest.approx(7.0)

    task.reconcile_review_pause(paused_at=checkpoint_at)
    reconciled_pause = ledger.snapshot()["tasks"][0]
    paused_elapsed = reconciled_pause["elapsed_seconds"]
    assert paused_elapsed == pytest.approx(5.0)
    assert reconciled_pause["paused_at"] == checkpoint_at
    assert reconciled_pause["review_checkpoint_at"] == checkpoint_at
    task.resume()
    assert ledger.snapshot()["tasks"][0]["status"] == "running"

    wrong_checkpoint = (wall[0] + timedelta(seconds=1)).isoformat()
    with pytest.raises(
        owner.ProviderHardStopLedgerError,
        match="review checkpoint changed",
    ):
        task.reconcile_review_pause(paused_at=wrong_checkpoint)
    assert ledger.snapshot()["tasks"][0]["status"] == "running"

    task.reconcile_review_pause(paused_at=checkpoint_at)

    recovered = ledger.snapshot()["tasks"][0]
    assert recovered["status"] == "paused"
    assert recovered["review_checkpoint_at"] == checkpoint_at
    assert recovered["elapsed_seconds"] == paused_elapsed


@pytest.mark.parametrize(
    ("limits", "expected_code"),
    [
        (
            _limits(
                max_total_tokens_per_run=100,
                max_total_tokens_per_batch=100,
            ),
            "RUN_TOKEN_LIMIT",
        ),
        (
            _limits(
                max_estimated_cost_usd_per_batch=0.000001,
                input_cost_usd_per_million_tokens=100.0,
                output_cost_usd_per_million_tokens=100.0,
            ),
            "BATCH_COST_LIMIT",
        ),
    ],
)
def test_token_and_cost_reservations_block_before_client_call(
    tmp_path, limits, expected_code
):
    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopExceeded,
    )
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    ledger = _ledger(tmp_path, limits=limits)
    inner = ScriptedMockLLMClient(["must-not-run"])
    client = HardStopClient(
        inner,
        role="writer",
        task=ledger.start_task("E1"),
    )

    with pytest.raises(ProviderHardStopExceeded, match=expected_code):
        client.complete(_message("x"), max_tokens=32)

    assert inner.calls == []
    assert ledger.snapshot()["totals"]["provider_attempts"] == 0


def test_denied_retry_closes_prior_attempt_without_masking_limit(tmp_path):
    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopExceeded,
        consume_active_provider_hard_stop_attempt,
        provider_hard_stop_call_scope,
    )

    ledger = _ledger(
        tmp_path,
        limits=_limits(
            max_provider_attempts_per_run=1,
            max_provider_attempts_per_batch=1,
        ),
    )
    task = ledger.start_task("E1")

    with provider_hard_stop_call_scope(
        task=task,
        role="coder",
        model="test-model",
        messages=_message(),
        max_tokens=8,
    ):
        consume_active_provider_hard_stop_attempt()
        with pytest.raises(
            ProviderHardStopExceeded, match="RUN_PROVIDER_ATTEMPT_LIMIT"
        ):
            consume_active_provider_hard_stop_attempt()

    call = ledger.snapshot()["tasks"][0]["calls"][0]
    assert call["state"] == "failed_usage_unknown"
    assert call["error_type"] == "TransportRetry"


def test_failed_attempt_remains_conservatively_accounted(tmp_path):
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.mocks import (
        BudgetAwareScriptedMockLLMClient,
    )

    ledger = _ledger(tmp_path)
    inner = BudgetAwareScriptedMockLLMClient([RuntimeError("network failed")])
    client = HardStopClient(
        inner,
        role="planner",
        task=ledger.start_task("E1"),
    )

    with pytest.raises(RuntimeError, match="network failed"):
        client.complete(_message(), max_tokens=64)

    snapshot = ledger.snapshot()
    assert snapshot["totals"]["provider_attempts"] == 1
    assert snapshot["totals"]["accounted_tokens"] > 64
    assert snapshot["tasks"][0]["calls"][0]["state"] == "failed_usage_unknown"


def test_task_accounting_separates_reported_unknown_and_upper_bound(tmp_path):
    from easyicu.research_agent.authority.provider_hard_stop import (
        consume_active_provider_hard_stop_attempt,
        provider_hard_stop_call_scope,
    )

    ledger = _ledger(tmp_path)
    task = ledger.start_task("E1")
    with provider_hard_stop_call_scope(
        task=task,
        role="planner",
        model="test-model",
        messages=_message(),
        max_tokens=32,
    ) as call:
        consume_active_provider_hard_stop_attempt()
        call.complete(
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            }
        )
    with provider_hard_stop_call_scope(
        task=task,
        role="repair",
        model="test-model",
        messages=_message(),
        max_tokens=32,
    ) as call:
        consume_active_provider_hard_stop_attempt()
        call.fail("KeyboardInterrupt")

    accounting = task.accounting_summary()

    assert accounting["provider_reported"] == {
        "n_calls": 1,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "estimated_cost_usd": pytest.approx(0.00014),
    }
    assert accounting["usage_unknown"]["n_calls"] == 1
    assert accounting["usage_unknown"]["states"] == {"failed_usage_unknown": 1}
    assert accounting["conservative_upper_bound"]["n_calls"] == 2
    assert accounting["conservative_upper_bound"]["total_tokens"] > 120
    assert (
        accounting["conservative_upper_bound"]["source"]
        == "durable_provider_hard_stop_ledger"
    )


def test_total_only_usage_is_counted_at_the_more_expensive_rate(tmp_path):
    from easyicu.research_agent.authority.provider_hard_stop import (
        consume_active_provider_hard_stop_attempt,
        provider_hard_stop_call_scope,
    )

    ledger = _ledger(tmp_path)
    task = ledger.start_task("E1")
    with provider_hard_stop_call_scope(
        task=task,
        role="writer",
        model="test-model",
        messages=_message(),
        max_tokens=32,
    ) as call:
        consume_active_provider_hard_stop_attempt()
        call.complete({"total_tokens": 7})

    record = ledger.snapshot()["tasks"][0]["calls"][0]
    assert record["accounted_tokens"] == 7
    assert record["reported_total_tokens"] == 7
    assert record["accounted_estimated_cost_usd"] == pytest.approx(0.000014)


def test_provider_side_prompt_overhead_is_reserved_and_released(tmp_path):
    from easyicu.research_agent.authority.provider_hard_stop import (
        PROVIDER_PROMPT_OVERHEAD_TOKEN_RESERVATION,
        consume_active_provider_hard_stop_attempt,
        provider_hard_stop_call_scope,
    )

    ledger = _ledger(tmp_path)
    task = ledger.start_task("E1")
    with provider_hard_stop_call_scope(
        task=task,
        role="analyzer",
        model="gpt-5.6-luna",
        messages=_message("Return exactly READY"),
        max_tokens=32,
    ) as call:
        consume_active_provider_hard_stop_attempt()
        call.complete(
            {
                "prompt_tokens": 316,
                "completion_tokens": 5,
                "total_tokens": 321,
            }
        )

    record = ledger.snapshot()["tasks"][0]["calls"][0]
    assert record["state"] == "completed"
    assert record["accounted_tokens"] == 321
    assert (
        record["provider_prompt_overhead_token_reservation"]
        == PROVIDER_PROMPT_OVERHEAD_TOKEN_RESERVATION
    )
    assert record["prompt_token_reservation"] > record["reported_prompt_tokens"]


def test_usage_beyond_provider_overhead_reservation_still_fails_closed(tmp_path):
    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopExceeded,
        consume_active_provider_hard_stop_attempt,
        provider_hard_stop_call_scope,
    )

    ledger = _ledger(tmp_path)
    task = ledger.start_task("E1")
    with provider_hard_stop_call_scope(
        task=task,
        role="analyzer",
        model="test-model",
        messages=_message(),
        max_tokens=32,
    ) as call:
        consume_active_provider_hard_stop_attempt()
        with pytest.raises(
            ProviderHardStopExceeded,
            match="PROVIDER_USAGE_EXCEEDED_RESERVATION",
        ):
            call.complete({"total_tokens": 1_000_000})

    record = ledger.snapshot()["tasks"][0]["calls"][0]
    assert record["state"] == "completed_usage_overflow"
    assert record["accounted_tokens"] == 1_000_000


def test_unenforced_completion_cap_is_pre_reserved_at_provider_maximum(tmp_path):
    from easyicu.research_agent.authority.provider_hard_stop import (
        PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR,
        consume_active_provider_hard_stop_attempt,
        provider_hard_stop_call_scope,
    )

    ledger = _ledger(tmp_path)
    task = ledger.start_task("E1")
    with provider_hard_stop_call_scope(
        task=task,
        role="writer",
        model="gpt-5.6-luna",
        messages=_message("repeat ALPHA"),
        max_tokens=24,
    ) as call:
        consume_active_provider_hard_stop_attempt()
        call.complete(
            {
                "prompt_tokens": 324,
                "completion_tokens": 500,
                "total_tokens": 824,
            }
        )

    record = ledger.snapshot()["tasks"][0]["calls"][0]
    assert record["state"] == "completed"
    assert record["reported_completion_tokens"] == 500
    assert record["requested_completion_tokens"] == 24
    assert (
        record["completion_token_reservation"]
        == PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR
    )


def test_completion_beyond_provider_maximum_still_fails_closed(tmp_path):
    from easyicu.research_agent.authority.provider_hard_stop import (
        PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR,
        ProviderHardStopExceeded,
        consume_active_provider_hard_stop_attempt,
        provider_hard_stop_call_scope,
    )

    ledger = _ledger(tmp_path)
    task = ledger.start_task("E1")
    with provider_hard_stop_call_scope(
        task=task,
        role="writer",
        model="gpt-5.6-luna",
        messages=_message(),
        max_tokens=24,
    ) as call:
        consume_active_provider_hard_stop_attempt()
        with pytest.raises(
            ProviderHardStopExceeded,
            match="PROVIDER_COMPLETION_USAGE_EXCEEDED_RESERVATION",
        ):
            call.complete(
                {
                    "prompt_tokens": 100,
                    "completion_tokens": (
                        PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR + 1
                    ),
                }
            )

    assert (
        ledger.snapshot()["tasks"][0]["calls"][0]["state"]
        == "completed_usage_overflow"
    )


def test_provider_request_timeout_is_capped_by_task_wall_clock(
    tmp_path, monkeypatch
):
    from easyicu.research_agent.providers.factory import build_provider_client
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.llm import OpenAIClient

    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            choice = SimpleNamespace(
                message=SimpleNamespace(content="ok"),
                finish_reason="stop",
            )
            return SimpleNamespace(choices=[choice], usage=None)

    transport = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions())
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=lambda **_kwargs: transport),
    )
    inner = build_provider_client(
        provider="openai",
        model="test-model",
        request_timeout=900.0,
        title="hard-stop timeout test",
        client_cls=OpenAIClient,
        environment={"OPENAI_BASE_URL": "http://127.0.0.1:8317/v1"},
        max_retries=1,
        stream_enabled=False,
        allow_environment_overrides=False,
    )
    task = _ledger(
        tmp_path,
        limits=_limits(max_wall_clock_seconds_per_task=60.0),
    ).start_task("E1")
    client = HardStopClient(inner, role="planner", task=task)

    assert client.complete(_message(), max_tokens=8) == "ok"
    assert 0.0 < captured["timeout"] <= 60.0


def test_each_web_500_retry_is_charged_to_provider_hard_stop(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.factory import build_provider_client
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.llm import OpenAIClient

    class _Completions:
        def __init__(self):
            self.calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                failure = RuntimeError("provider internal failure")
                failure.status_code = 500
                raise failure
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok"),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )

    completions = _Completions()
    transport = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=lambda **_kwargs: transport),
    )
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    inner = build_provider_client(
        provider="openai",
        model="gpt-5.6-luna",
        request_timeout=30.0,
        title="Web retry hard-stop contract",
        client_cls=OpenAIClient,
        environment={"OPENAI_BASE_URL": "http://127.0.0.1:8317/v1"},
        max_retries=1,
        retryable_http_status_codes=(500, 502, 503, 504),
        stream_enabled=False,
        allow_environment_overrides=False,
    )
    ledger = _ledger(tmp_path)
    client = HardStopClient(inner, role="planner", task=ledger.start_task("E1"))

    assert client.complete(_message(), max_tokens=8) == "ok"
    assert completions.calls == 2
    assert ledger.snapshot()["totals"]["provider_attempts"] == 2


def test_vision_transport_is_reserved_settled_and_stopped_before_overrun(
    tmp_path, monkeypatch
):
    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopExceeded,
    )
    from easyicu.research_agent.providers.factory import build_provider_client
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.llm import OpenAIClient

    class _Completions:
        def __init__(self):
            self.calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"findings": []}'),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=20,
                    completion_tokens=4,
                    total_tokens=24,
                ),
            )

    completions = _Completions()
    transport = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=lambda **_kwargs: transport),
    )
    inner = build_provider_client(
        provider="openai",
        model="gpt-4o",
        request_timeout=30.0,
        title="vision hard-stop contract",
        client_cls=OpenAIClient,
        environment={"OPENAI_BASE_URL": "http://127.0.0.1:8317/v1"},
        max_retries=0,
        stream_enabled=False,
        allow_environment_overrides=False,
    )
    ledger = _ledger(
        tmp_path,
        limits=_limits(
            max_provider_attempts_per_run=1,
            max_provider_attempts_per_batch=1,
        ),
    )
    client = HardStopClient(inner, role="visual_qa", task=ledger.start_task("E1"))
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(b"not-real-image-bytes")

    assert (
        client.complete_with_images(
            prompt="Review this figure.",
            image_paths=[image_path],
            max_tokens=64,
        )
        == '{"findings": []}'
    )
    first = ledger.snapshot()
    call = first["tasks"][0]["calls"][0]
    assert completions.calls == 1
    assert first["totals"]["provider_attempts"] == 1
    assert call["state"] == "completed"
    assert call["reported_total_tokens"] == 24
    assert call["prompt_payload_bytes"] == len("Review this figure.".encode())
    assert str(image_path) not in json.dumps(first)

    with pytest.raises(ProviderHardStopExceeded, match="RUN_PROVIDER_ATTEMPT_LIMIT"):
        client.complete_with_images(
            prompt="Do not send this request.",
            image_paths=[image_path],
            max_tokens=64,
        )
    assert completions.calls == 1


def test_hard_stop_wrapper_does_not_promote_a_text_client_to_vision(
    tmp_path,
):
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.llm import llm_supports_vision
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    client = HardStopClient(
        ScriptedMockLLMClient(['{"findings": []}']),
        role="visual_qa",
        task=_ledger(tmp_path).start_task("E1"),
    )

    assert llm_supports_vision(client) is False


def test_human_review_pause_does_not_consume_active_execution_time(
    tmp_path, monkeypatch
):
    from easyicu.research_agent.authority import provider_hard_stop as owner

    clock = [100.0]
    monkeypatch.setattr(owner.time, "monotonic", lambda: clock[0])
    ledger = _ledger(
        tmp_path,
        limits=_limits(max_wall_clock_seconds_per_task=60.0),
    )
    task = ledger.start_task("E1")
    clock[0] = 105.0
    task.pause()
    paused = ledger.snapshot()["tasks"][0]
    assert paused["status"] == "paused"
    assert paused["elapsed_seconds"] == pytest.approx(5.0)
    assert ledger.snapshot()["terminal"] is False

    # A long human wait changes wall time but is not active Provider execution.
    clock[0] = 10_005.0
    task.resume()
    clock[0] = 10_006.0
    assert task.assert_active() == pytest.approx(54.0)
    task.finish(error="test_finished")
    final = ledger.snapshot()["tasks"][0]
    assert final["status"] == "failed"
    assert final["elapsed_seconds"] == pytest.approx(6.0)


def test_restarted_host_attaches_to_paused_task_without_resetting_budget(
    tmp_path,
):
    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopLedger,
    )

    path = tmp_path / "restarted-ledger.json"
    limits = _limits(max_wall_clock_seconds_per_task=60.0)
    first = ProviderHardStopLedger(
        path=path,
        task_ids=("task-a",),
        limits=limits,
        batch_id="task-a",
        declaration_sha256="a" * 64,
    )
    first.start_task("task-a").pause()

    reopened = ProviderHardStopLedger(
        path=path,
        task_ids=("task-a",),
        limits=limits,
        batch_id="task-a",
        declaration_sha256="a" * 64,
        resume_existing=True,
    )
    attached = reopened.start_task("task-a")
    assert reopened.snapshot()["tasks"][0]["status"] == "paused"
    attached.resume()
    assert attached.assert_active() > 0


def test_restart_after_pause_resume_uses_persisted_active_wall_clock_anchor(
    tmp_path,
    monkeypatch,
):
    from easyicu.research_agent.authority import provider_hard_stop as owner
    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopLedger,
    )

    wall = [datetime(2026, 8, 14, tzinfo=timezone.utc)]
    monotonic = [100.0]

    class _ClockDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = wall[0]
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr(owner, "datetime", _ClockDateTime)
    monkeypatch.setattr(owner.time, "monotonic", lambda: monotonic[0])
    path = tmp_path / "resume-anchor-ledger.json"
    limits = _limits(max_wall_clock_seconds_per_task=60.0)
    first = ProviderHardStopLedger(
        path=path,
        task_ids=("task-a",),
        limits=limits,
        batch_id="task-a",
        declaration_sha256="a" * 64,
    )
    task = first.start_task("task-a")
    monotonic[0] += 5.0
    wall[0] += timedelta(seconds=5)
    task.pause()

    monotonic[0] += 10_000.0
    wall[0] += timedelta(seconds=10_000)
    task.resume()
    resumed_row = first.snapshot()["tasks"][0]
    assert resumed_row["elapsed_seconds"] == pytest.approx(5.0)
    assert resumed_row["active_started_at"] is not None

    monotonic[0] += 2.0
    wall[0] += timedelta(seconds=2)
    reopened = ProviderHardStopLedger(
        path=path,
        task_ids=("task-a",),
        limits=limits,
        batch_id="task-a",
        declaration_sha256="a" * 64,
        resume_existing=True,
    )

    attached = reopened.start_task("task-a")
    assert attached.assert_active() == pytest.approx(53.0)


def test_restart_reconciles_checkpoint_pause_without_charging_downtime(
    tmp_path,
    monkeypatch,
):
    from easyicu.research_agent.authority import provider_hard_stop as owner
    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopLedger,
    )

    wall = [datetime(2026, 8, 14, tzinfo=timezone.utc)]
    monotonic = [100.0]

    class _ClockDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = wall[0]
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr(owner, "datetime", _ClockDateTime)
    monkeypatch.setattr(owner.time, "monotonic", lambda: monotonic[0])
    path = tmp_path / "crash-window-ledger.json"
    limits = _limits(max_wall_clock_seconds_per_task=60.0)
    first = ProviderHardStopLedger(
        path=path,
        task_ids=("task-a",),
        limits=limits,
        batch_id="task-a",
        declaration_sha256="a" * 64,
    )
    first.start_task("task-a")
    monotonic[0] += 5.0
    wall[0] += timedelta(seconds=5)
    checkpoint_created_at = wall[0].isoformat()

    # The checkpoint reached disk, but the process died before task.pause().
    monotonic[0] += 10_000.0
    wall[0] += timedelta(seconds=10_000)
    reopened = ProviderHardStopLedger(
        path=path,
        task_ids=("task-a",),
        limits=limits,
        batch_id="task-a",
        declaration_sha256="a" * 64,
        resume_existing=True,
    )
    reopened.reconcile_review_pause(
        "task-a",
        paused_at=checkpoint_created_at,
    )

    row = reopened.snapshot()["tasks"][0]
    assert row["status"] == "paused"
    assert row["elapsed_seconds"] == pytest.approx(5.0)
    task = reopened.start_task("task-a")
    task.resume()
    assert task.assert_active() == pytest.approx(55.0)


def test_wall_clock_exhaustion_is_terminal_and_persisted(tmp_path):
    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopExceeded,
        ProviderHardStopLedgerError,
    )

    ledger = _ledger(
        tmp_path,
        limits=_limits(max_wall_clock_seconds_per_task=0.01),
    )
    task = ledger.start_task("E1")
    ledger._task_started_monotonic["E1"] = time.monotonic() - 1.0

    with pytest.raises(ProviderHardStopExceeded, match="TASK_WALL_CLOCK_EXHAUSTED"):
        task.cap_timeout(900.0)

    snapshot = ledger.snapshot()
    assert snapshot["tasks"][0]["status"] == "budget_exhausted"
    assert snapshot["terminal"] is True
    with pytest.raises(
        ProviderHardStopLedgerError,
        match="cannot start from 'budget_exhausted'",
    ):
        ledger.start_task("E1", reopen_terminal=True)


def test_tampered_ledger_and_nonfinite_limits_fail_closed(tmp_path):
    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopLedgerError,
        load_provider_hard_stop_ledger,
    )

    ledger = _ledger(tmp_path)
    payload = json.loads(ledger.path.read_text(encoding="utf-8"))
    payload["totals"]["provider_attempts"] = -100
    ledger.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProviderHardStopLedgerError):
        load_provider_hard_stop_ledger(ledger.path)
    with pytest.raises(ValueError, match="finite"):
        _limits(max_estimated_cost_usd_per_batch=float("inf"))


def test_task_failure_persists_only_type_and_digest(tmp_path):
    ledger = _ledger(tmp_path)
    task = ledger.start_task("E1")
    task.finish(error="ValueError: secret-token patient-value-123")

    raw = ledger.path.read_text(encoding="utf-8")
    assert "secret-token" not in raw
    assert "patient-value-123" not in raw
    error = json.loads(raw)["tasks"][0]["error"]
    assert error["type"] == "ValueError"
    assert len(error["message_sha256"]) == 64


def test_strict_schema_bytes_are_reserved_and_forwarded(tmp_path):
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from easyicu.research_agent.providers.protocol import StructuredOutputRequest

    request = StructuredOutputRequest.from_schema(
        name="probe",
        schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
    )
    inner = ScriptedMockLLMClient(['{"ok":true}'])
    task = _ledger(tmp_path).start_task("E1")
    client = HardStopClient(inner, role="planner", task=task)

    assert client.complete(
        _message("schema-bound"),
        max_tokens=8,
        structured_output=request,
    ) == '{"ok":true}'

    call = task.ledger.snapshot()["tasks"][0]["calls"][0]
    assert call["prompt_payload_bytes"] == (
        len("schema-bound".encode("utf-8")) + request.payload_bytes
    )
    assert inner.calls[0][1]["structured_output"] is request
