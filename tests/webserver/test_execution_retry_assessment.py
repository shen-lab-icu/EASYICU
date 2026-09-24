"""A retry is withheld only when it provably repeats the failure."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from uuid import uuid4

import pytest

from easyicu.research_agent.execution.retry_basis import (
    AttemptIdentity,
    FailedStepRetryBasis,
    current_attempt_identity,
)
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver import execution_retry
from easyicu.webserver.execution_retry import (
    ExecutionRetryAssessment,
    assess_execution_retry,
    assess_failed_step,
)
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

IMAGE = "sha256:" + "1" * 64


def _basis(
    identity: Optional[AttemptIdentity] = None, **fields: Any
) -> FailedStepRetryBasis:
    values: dict[str, Any] = {
        "step_id": "figure",
        "status": "contract_failed",
        "failure_class": "code",
        "repair_attempts": 2,
        "repair_limit": 2,
        "provider_calls": 4,
        "provider_limit": 9,
        "repair_budget_exhausted": True,
        "identity": identity,
    }
    values.update(fields)
    return FailedStepRetryBasis(**values)


def _unasked() -> Optional[str]:
    pytest.fail("the runner image must not be inspected")


def test_spent_repairs_on_unchanged_code_and_image_are_futile() -> None:
    assessment = assess_failed_step(
        _basis(current_attempt_identity(image_id=IMAGE)), read_image_id=lambda: IMAGE
    )

    assert assessment.public() == {
        "state": "futile",
        "reason_code": "execution_retry_repeats_failure",
        "failed_step_id": "figure",
        "repair_attempts": 2,
        "repair_limit": 2,
        "changed_components": [],
        "image_checked": True,
    }


@pytest.mark.parametrize(
    "component",
    ["engine_code_sha256", "prompt_pack_sha256", "execution_kernel_identity_sha256"],
)
def test_changed_code_offers_the_retry_without_asking_docker(component: str) -> None:
    failed = replace(current_attempt_identity(image_id=IMAGE), **{component: "0" * 64})

    assessment = assess_failed_step(_basis(failed), read_image_id=_unasked)

    assert (assessment.state, assessment.reason_code) == (
        "available",
        "execution_retry_code_changed",
    )
    assert assessment.changed_components == (component,)


def test_a_rebuilt_runner_image_offers_the_retry() -> None:
    assessment = assess_failed_step(
        _basis(current_attempt_identity(image_id=IMAGE)),
        read_image_id=lambda: "sha256:" + "2" * 64,
    )

    assert (assessment.state, assessment.changed_components) == ("available", ("image_id",))
    assert assessment.reason_code == "execution_retry_runner_image_changed"


def test_an_unreadable_runner_image_is_doubt_not_futility() -> None:
    assessment = assess_failed_step(
        _basis(current_attempt_identity(image_id=IMAGE)), read_image_id=lambda: None
    )

    assert (assessment.state, assessment.reason_code) == (
        "unknown",
        "execution_retry_runner_image_unreadable",
    )


def test_an_attempt_without_a_recorded_image_is_judged_on_its_code() -> None:
    assessment = assess_failed_step(
        _basis(current_attempt_identity(image_id=None)), read_image_id=_unasked
    )

    assert (assessment.state, assessment.image_checked) == ("futile", False)


@pytest.mark.parametrize(
    "fields,state,reason",
    [
        ({"repair_budget_exhausted": False}, "available", "execution_retry_repair_budget_remaining"),
        ({"failure_class": "timeout"}, "available", "execution_retry_failure_timeout"),
        ({"failure_class": "infrastructure"}, "available", "execution_retry_failure_infrastructure"),
        ({"failure_class": "ledger_invalid"}, "available", "execution_retry_failure_ledger_invalid"),
        ({"identity": None}, "unknown", "execution_retry_failed_identity_unknown"),
    ],
)
def test_every_other_basis_leaves_the_retry_offered(
    fields: dict[str, Any], state: str, reason: str
) -> None:
    basis = _basis(**{"identity": current_attempt_identity(image_id=IMAGE), **fields})

    assessment = assess_failed_step(basis, read_image_id=_unasked)

    assert (assessment.state, assessment.reason_code) == (state, reason)
    assert assess_failed_step(None).state == "unknown"


def test_the_resume_owner_refusal_and_unreadable_records_are_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def refused() -> tuple[Path, Optional[str]]:
        raise ResearchPipelineRunError(
            "research_pipeline_execution_retry_checkpoint_missing", "missing"
        )

    assessment = assess_execution_retry(
        source_run_id="run-refused", configuration_sha256="c" * 64,
        resolve_failed_step=refused, max_age_seconds=0.0,
    )
    assert (assessment.state, assessment.reason_code) == (
        "unknown",
        "research_pipeline_execution_retry_checkpoint_missing",
    )

    def unreadable(*_args: Any) -> None:
        raise OSError("disk")

    monkeypatch.setattr(execution_retry, "load_failed_step_retry_basis", unreadable)
    assessment = assess_execution_retry(
        source_run_id="run-unreadable", configuration_sha256="c" * 64,
        resolve_failed_step=lambda: (tmp_path, "figure"), max_age_seconds=0.0,
    )
    assert (assessment.state, assessment.reason_code) == (
        "unknown",
        "execution_retry_records_unreadable",
    )


def test_the_policy_reads_the_failed_steps_own_attempt_records(tmp_path: Path) -> None:
    def failed_run(name: str, repairs: int) -> Path:
        run_dir = tmp_path / name
        run_dir.mkdir()
        record = {
            "step_id": "figure",
            "status": "contract_failed",
            "step_llm_repair_attempts": repairs,
            "step_llm_repair_budget": 2,
        }
        (run_dir / "manifest.json").write_text(
            json.dumps({"per_step_records": [record]}), encoding="utf-8"
        )
        return run_dir

    def assess(run_dir: Path) -> ExecutionRetryAssessment:
        return assess_execution_retry(
            source_run_id=f"run-{uuid4().hex}", configuration_sha256="e" * 64,
            resolve_failed_step=lambda: (run_dir, "figure"), max_age_seconds=0.0,
        )

    repairs_left = assess(failed_run("repairs-left", 1))
    spent = assess(failed_run("spent", 2))

    assert (repairs_left.state, repairs_left.reason_code) == (
        "available",
        "execution_retry_repair_budget_remaining",
    )
    assert (repairs_left.repair_attempts, repairs_left.repair_limit) == (1, 2)
    # Spent repairs without a verifiable capsule are doubt, never futility.
    assert (spent.state, spent.reason_code) == (
        "unknown",
        "execution_retry_failed_identity_unknown",
    )


def test_projection_polls_reuse_one_reading_and_the_launch_rereads(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    run_id = f"run-{uuid4().hex}"

    def failed_step() -> tuple[Path, Optional[str]]:
        calls.append("resolved")
        return tmp_path / "no-ledger", "figure"

    for _ in range(2):
        assess_execution_retry(
            source_run_id=run_id, configuration_sha256="d" * 64,
            resolve_failed_step=failed_step,
        )
    assert calls == ["resolved"]

    assessment = assess_execution_retry(
        source_run_id=run_id, configuration_sha256="d" * 64,
        resolve_failed_step=failed_step, max_age_seconds=0.0,
    )
    assert calls == ["resolved", "resolved"]
    assert (assessment.state, assessment.reason_code) == (
        "unknown",
        "execution_retry_basis_unavailable",
    )


def _factory(monkeypatch: pytest.MonkeyPatch, state: str) -> list[dict[str, Any]]:
    asked: list[dict[str, Any]] = []
    prepared = SimpleNamespace(
        scientific=SimpleNamespace(study={"id": "study-1"}),
        authority=SimpleNamespace(),
        execution=SimpleNamespace(
            execution_resume_source_run_id="run-failed",
            project_root="/server/workspace",
            runner_image="easyicu-research-agent:test",
        ),
    )
    monkeypatch.setattr(
        agent_pipeline_runs, "prepare_research_pipeline_run", lambda _request: prepared
    )

    def assessment(**kwargs: Any) -> ExecutionRetryAssessment:
        asked.append(kwargs)
        return ExecutionRetryAssessment(
            state, "execution_retry_repeats_failure", failed_step_id="figure",
            repair_attempts=2, repair_limit=2,
        )

    monkeypatch.setattr(agent_pipeline_runs, "execution_retry_assessment", assessment)
    return asked


def _launch(**overrides: Any) -> Any:
    return agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path="/typed/demo",
        study_context={"id": "study-1"},
        project_root="/server/workspace",
        provider={"provider": "openai"},
        execution_resume_source_run_id="run-failed",
        **overrides,
    )


def test_a_futile_retry_is_refused_before_any_job_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = _factory(monkeypatch, "futile")

    with pytest.raises(ResearchPipelineRunError) as raised:
        _launch()

    assert raised.value.code == "research_pipeline_execution_retry_futile"
    assert raised.value.details["repair_attempts"] == 2
    assert asked == [
        {
            "study": {"id": "study-1"},
            "project_root": "/server/workspace",
            "source_run_id": "run-failed",
            "runner_image": "easyicu-research-agent:test",
            "max_age_seconds": 0.0,
        }
    ]


def test_a_retry_that_could_change_the_outcome_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _factory(monkeypatch, "available")

    assert callable(_launch())
