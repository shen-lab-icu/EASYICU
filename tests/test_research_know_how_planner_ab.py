from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "run_research_know_how_planner_ab.py"
)
SPEC = importlib.util.spec_from_file_location("research_know_how_planner_ab", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_schedule_is_balanced_deterministic_and_interleaved() -> None:
    first = MODULE.build_schedule(3, 42)
    second = MODULE.build_schedule(3, 42)

    assert first == second
    assert first.count("off") == 3
    assert first.count("on") == 3
    assert len(set(first)) == 2
    assert all(
        set(first[index : index + 2]) == {"off", "on"} for index in range(0, 6, 2)
    )


def test_schedule_requires_two_runs_per_arm() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        MODULE.build_schedule(1, 42)


def test_blinded_labels_do_not_reveal_arm() -> None:
    labels = [MODULE.blinded_label(seed=7, trial_index=index) for index in range(4)]

    assert len(set(labels)) == 4
    assert all(label.startswith("plan_") for label in labels)
    assert all("off" not in label and "on" not in label for label in labels)


def test_blind_rubric_catches_scientific_authority_failures() -> None:
    assert "data_answerability_and_stop_conditions" in MODULE.BLIND_RUBRIC["dimensions"]
    assert (
        "unsupported_disease_specific_exclusion"
        in MODULE.BLIND_RUBRIC["critical_errors"]
    )


def test_online_run_rejects_curated_card_without_explicit_override() -> None:
    binding = SimpleNamespace(selected_ids=("early_peak_lactate_association",))
    registry = SimpleNamespace(
        get=lambda _card_id: SimpleNamespace(review_status="curated_mvp")
    )
    prepared = SimpleNamespace(registry=registry)

    with pytest.raises(RuntimeError, match="refused unreviewed"):
        MODULE.require_reviewed_cards(
            binding,
            prepared,
            allow_curated_development_card=False,
        )

    MODULE.require_reviewed_cards(
        binding,
        prepared,
        allow_curated_development_card=True,
    )


def test_counting_client_records_each_retry_call() -> None:
    from easyicu.research_agent.providers.mocks import MockLLMClient

    inner = MockLLMClient()
    client = MODULE.CountingClient(inner)

    assert isinstance(client.complete([]), str)
    assert client.calls[0]["usage"] == inner.last_usage
    assert client.calls[0]["usage"]["prompt_tokens"] > 0
    assert client.calls[0]["usage"]["completion_tokens"] > 0
    assert len(client.calls[0]["raw_sha256"]) == 64


def test_counting_client_fails_before_exceeding_trial_budget() -> None:
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    inner = ScriptedMockLLMClient(["{}"], repeat_last=True)
    client = MODULE.CountingClient(inner, max_calls=1)
    client.complete([])

    with pytest.raises(RuntimeError, match="budget exhausted"):
        client.complete([])

    assert len(client.calls) == 1
