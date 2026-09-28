"""An explicit retry after an engine change runs the changed engine.

Retrying a failed approved execution is offered when the research-agent code
or runner image changed since the failure.  Two things must then hold: the
retry's fresh repair budget is actually granted (the resumed run's own ledger
no longer holds the failed record, so the grant must read the checkpoint the
resume started from), and a step whose code the host generates runs the
host's regenerated code rather than a model repair of the old host code.
All run directories and capsules here are synthetic.
"""

from __future__ import annotations

import inspect
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from easyicu.research_agent.authority import step_runtime
from easyicu.research_agent.execution import phase_support, step_authority_resume
from easyicu.research_agent.execution.budget_epoch import (
    EPOCH_FIELD,
    AttemptIdentity,
    checkpoint_capsule_identity,
)
from easyicu.research_agent.execution.step_attempt_bootstrap import (
    prepare_step_attempt_bootstrap,
)
from easyicu.research_agent.execution.step_authority_resume import (
    EXPLICIT_RETRY_ENGINE_CHANGED,
    EXPLICIT_RETRY_HOST_REGENERATED,
    explicit_retry_host_owned_code,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

STEP = "s1"
OLD_ENGINE = "a" * 64
NEW_ENGINE = "b" * 64
PROMPTS = "p" * 64
CAPSULE_REF = {
    "schema_version": "easyicu.step_authority_capsule_ref/1",
    "step_id": STEP,
    "capsule_sha256": "c" * 64,
}


def _failed_record(**fields: Any) -> dict[str, Any]:
    return {
        "step_id": STEP,
        "status": "execution_failed",
        "attempt_sequence": 1,
        "step_llm_repair_attempts": 2,
        "step_llm_repair_budget": 2,
        "step_llm_repair_classes": ["contract", "runtime"],
        "step_authority_capsule_ref": dict(CAPSULE_REF),
        **fields,
    }


def _failed_execution(returncode: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        returncode=returncode,
        timed_out=False,
        outputs_safe_to_collect=True,
        runner_failure_code=None,
        runtime_provenance=None,
    )


def _verified_capsule(*, engine: str = OLD_ENGINE, execution: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        ref=SimpleNamespace(**CAPSULE_REF),
        candidate_code="# the model's rewrite of the old host script\n",
        capsule=SimpleNamespace(
            engine_code_sha256=engine,
            prompt_pack_sha256=PROMPTS,
            execution=execution,
        ),
    )


# ---------------------------------------------------------------------------
# The fresh budget reads the checkpoint the resume started from
# ---------------------------------------------------------------------------


@pytest.fixture
def resumed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A failed run's checkpoint, superseded on disk by its resumed ledger."""

    failed_checkpoint = {"checkpoint_sequence": 5, "per_step_records": [_failed_record()]}
    (tmp_path / "manifest.json").write_text(json.dumps(failed_checkpoint), encoding="utf-8")
    # The resume restores only the ok records before its cut, so the ledger it
    # flushes before the retried step runs no longer holds the failed record.
    (tmp_path / "manifest_partial.json").write_text(
        json.dumps({"checkpoint_sequence": 6, "per_step_records": []}), encoding="utf-8"
    )
    monkeypatch.setattr(
        step_runtime,
        "load_verified_step_authority_capsule",
        lambda *_args, **_kwargs: _verified_capsule(),
    )
    return {"run_dir": tmp_path, "checkpoint": failed_checkpoint}


def test_the_newest_ledger_no_longer_selects_the_failed_capsule(resumed_run) -> None:
    assert checkpoint_capsule_identity(resumed_run["run_dir"], STEP) is None
    assert checkpoint_capsule_identity(
        resumed_run["run_dir"], STEP, checkpoint=resumed_run["checkpoint"]
    ) == AttemptIdentity(OLD_ENGINE, PROMPTS, None, None)


def _bootstrap(resumed_run: dict[str, Any], *, findings: list) -> Any:
    step = AnalysisStep(
        step_id=STEP,
        intent="Summarize the locked cohort.",
        inputs=["stay_id"],
        expected_outputs=["table:summary"],
        method="descriptive_summary",
    )
    run_dir = resumed_run["run_dir"]
    universe = run_dir / "cohort_universe.parquet"
    cohort = run_dir / "cohort_analysis.parquet"
    universe.write_bytes(b"universe")
    cohort.write_bytes(b"cohort")
    checkpoint = resumed_run["checkpoint"]
    return prepare_step_attempt_bootstrap(
        resume_state={**checkpoint, "step_attempt_history": list(checkpoint["per_step_records"])},
        per_step_records=[],
        shared_lock=threading.Lock(),
        step=step,
        plan=AnalysisPlan(research_question="Summarize the cohort.", steps=[step]),
        run_id="run",
        run_dir=run_dir,
        universe_path=universe,
        cohort_path=cohort,
        plan_scientific_signature=[{"step_id": STEP}],
        findings=findings,
        max_provider_calls=5,
        max_llm_repairs=2,
        reserve_concept_audit=False,
        allow_terminal_initial_generation_restart=True,
        explicit_rerun=True,
        current_identity=AttemptIdentity(NEW_ENGINE, PROMPTS, None, None),
    )


def test_an_explicit_retry_of_a_resumed_run_is_granted_its_fresh_budget(resumed_run) -> None:
    findings: list = []

    result = _bootstrap(resumed_run, findings=findings)

    repair = result.budget_runtime.repair_budget
    assert result.budget_runtime.integrity_error is None
    assert result.step_record[EPOCH_FIELD] == 1
    assert repair.llm_repair_attempts == 0 and repair.available("runtime")
    assert [finding.validator for finding in findings] == ["step_budget_epoch"]


# ---------------------------------------------------------------------------
# The retry records whether the engine changed since the failed candidate
# ---------------------------------------------------------------------------


def _select(monkeypatch: pytest.MonkeyPatch, *, capsule: Any, engine: str) -> dict[str, Any]:
    monkeypatch.setattr(
        step_authority_resume,
        "load_checkpoint_selected_step_capsule",
        lambda *_args, **_kwargs: capsule,
    )
    step_record: dict[str, Any] = {}
    request = SimpleNamespace(
        run_dir=Path("/unused"),
        step=SimpleNamespace(step_id=STEP),
        resume_state={"per_step_records": [_failed_record()]},
        requested_resume_from_step_id=STEP,
        prior_step_record=_failed_record(),
        prior_attempt_records=[_failed_record()],
        step_attempt_state=SimpleNamespace(selected_resume_capsule=None),
        step_record=step_record,
        engine_code_sha256=engine,
    )
    step_authority_resume._select_resume_candidate(
        request, gate_stamp={"deterministic_gate_fingerprint": "g" * 64}
    )
    return step_record


def test_a_retry_under_a_new_engine_is_marked(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _select(
        monkeypatch, capsule=_verified_capsule(execution=_failed_execution()), engine=NEW_ENGINE
    )

    assert record["explicit_failed_execution_retry"] is True
    assert record[EXPLICIT_RETRY_ENGINE_CHANGED] is True


def test_a_retry_under_the_same_engine_reruns_its_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _select(
        monkeypatch, capsule=_verified_capsule(execution=_failed_execution()), engine=OLD_ENGINE
    )

    assert record["explicit_failed_execution_retry"] is True
    assert EXPLICIT_RETRY_ENGINE_CHANGED not in record


def test_a_step_whose_execution_succeeded_is_not_an_explicit_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _select(
        monkeypatch,
        capsule=_verified_capsule(execution=_failed_execution(returncode=0)),
        engine=NEW_ENGINE,
    )

    assert "explicit_failed_execution_retry" not in record
    assert EXPLICIT_RETRY_ENGINE_CHANGED not in record


# ---------------------------------------------------------------------------
# The host regenerates the code it owns
# ---------------------------------------------------------------------------


def _generators(calls: list[str], owner: str | None):
    def generator(name: str):
        def generate() -> str | None:
            calls.append(name)
            return f"# host {name} runner\n" if name == owner else None

        return generate

    return tuple(generator(name) for name in ("absolute_risk", "robustness", "missingness"))


def test_the_owning_generator_runs_and_later_ones_are_not_asked() -> None:
    calls: list[str] = []
    record = {"explicit_failed_execution_retry": True, EXPLICIT_RETRY_ENGINE_CHANGED: True}

    code = explicit_retry_host_owned_code(record, _generators(calls, "robustness"))

    assert code == "# host robustness runner\n"
    assert calls == ["absolute_risk", "robustness"]
    assert record[EXPLICIT_RETRY_HOST_REGENERATED] is True


@pytest.mark.parametrize(
    "record",
    [
        {"explicit_failed_execution_retry": True},
        {EXPLICIT_RETRY_ENGINE_CHANGED: True},
        {},
    ],
    ids=["same-engine", "not-an-explicit-retry", "ordinary-attempt"],
)
def test_without_an_engine_change_nothing_is_regenerated(record) -> None:
    calls: list[str] = []

    assert explicit_retry_host_owned_code(record, _generators(calls, "robustness")) is None
    assert calls == [] and EXPLICIT_RETRY_HOST_REGENERATED not in record


def test_a_step_the_host_does_not_own_keeps_its_candidate() -> None:
    calls: list[str] = []
    record = {"explicit_failed_execution_retry": True, EXPLICIT_RETRY_ENGINE_CHANGED: True}

    assert explicit_retry_host_owned_code(record, _generators(calls, None)) is None
    assert calls == ["absolute_risk", "robustness", "missingness"]
    assert EXPLICIT_RETRY_HOST_REGENERATED not in record


def _resolve(*, step_record: dict[str, Any], robustness_code: str | None):
    """Resolve a retried step's code with only the robustness runner owning it."""

    worker_progress = SimpleNamespace(resumed_code_reuse_used=False)
    kwargs = {name: None for name in inspect.signature(phase_support._step_resolve_initial_code).parameters}
    kwargs.update(
        failed_contract_code_preflight_reuse=False,
        step_attempt_state=SimpleNamespace(selected_resume_capsule=_verified_capsule()),
        step=SimpleNamespace(step_id=STEP),
        step_record=step_record,
        worker_progress=worker_progress,
        findings=[],
        shared_lock=threading.Lock(),
        prior_attempt_records=[],
        _deterministic_absolute_risk_context_code=lambda *_args, **_kwargs: None,
        _deterministic_robustness_sensitivity_code=lambda *_args, **_kwargs: robustness_code,
        _deterministic_missingness_audit_code=lambda *_args, **_kwargs: None,
    )
    code, terminal = phase_support._step_resolve_initial_code(**kwargs)
    assert terminal is None
    return code, worker_progress


def test_a_retried_host_owned_step_runs_the_regenerated_host_code() -> None:
    record = {"explicit_failed_execution_retry": True, EXPLICIT_RETRY_ENGINE_CHANGED: True}

    code, progress = _resolve(step_record=record, robustness_code="# host robustness runner\n")

    assert code == "# host robustness runner\n"
    assert progress.resumed_code_reuse_used is False
    assert record[EXPLICIT_RETRY_HOST_REGENERATED] is True


def test_a_retry_under_the_same_engine_still_reruns_the_failed_candidate() -> None:
    record = {"explicit_failed_execution_retry": True}

    code, progress = _resolve(step_record=record, robustness_code="# host robustness runner\n")

    assert code == "# the model's rewrite of the old host script\n"
    assert progress.resumed_code_reuse_used is True
    assert record["generation_mode"] == "resumed_code_reuse"
