"""Analyzer failures cannot publish success or disappear across resume."""

from functools import partial
import inspect
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopExceeded
from easyicu.research_agent.execution import phase
from easyicu.research_agent.execution.run_coordination import (
    RunCoordinator,
    RunExecutionState,
)
from easyicu.research_agent.schema import AnalysisStep


@pytest.mark.parametrize("hard_stop", [False, True])
def test_finalize_failure_is_sealed_without_analyzer_evidence(
    monkeypatch, tmp_path, hard_stop
):
    error = (
        ProviderHardStopExceeded(code="RUN_PROVIDER_ATTEMPT_LIMIT", detail="test")
        if hard_stop
        else RuntimeError("analyzer unavailable")
    )
    gates = SimpleNamespace(
        **{
            name: []
            for name in (
                "stat_findings",
                "clinical_findings",
                "guard_findings",
                "contract_findings",
                "figure_source_findings",
            )
        }
    )
    monkeypatch.setattr(
        phase, "_evaluate_final_deterministic_gates", lambda **kw: gates
    )
    records, findings, flushes, attempted = [], [], [], []
    evidence = Mock()
    step = AnalysisStep(step_id="model", intent="Interpret synthetic results")
    args = {
        name: None for name in inspect.signature(phase._step_finalize_step).parameters
    }
    sync = Mock()
    args.update(
        step=step,
        step_record={"step_id": step.step_id},
        step_summary={},
        step_summary_record_id="summary",
        evidence_ids_for_step=["summary"],
        code="print('synthetic')",
        findings=findings,
        per_step_records=records,
        evidence=evidence,
        shared_lock=RLock(),
        _sync_provider_budget=sync,
        _flush_partial_manifest=lambda *a: flushes.append(list(records)),
        _validator_messages=lambda *a: [],
        worker_progress=SimpleNamespace(
            llm_repair_used=False, generation_mode=lambda: "llm"
        ),
        typed_binding_resolver=SimpleNamespace(
            resolve_names=lambda *a, **kw: ([], [], [])
        ),
        supervisor=SimpleNamespace(
            critique_step=lambda **kw: SimpleNamespace(critique=None)
        ),
        plausibility_authority=SimpleNamespace(scope=None),
        run_result=SimpleNamespace(out_dir=tmp_path),
        plan_result=SimpleNamespace(agent_context=None),
        pipeline=SimpleNamespace(_enable_llm_concept_audit=False),
        concept_audit=SimpleNamespace(tokens_by_digest={}),
        analyzer=SimpleNamespace(run=Mock(side_effect=error)),
    )

    def execute(current):
        attempted.append(current.step_id)
        return phase._step_finalize_step(**args)

    state = RunExecutionState(
        remaining_steps=[step, AnalysisStep(step_id="later", intent="Must not run")],
        executed_step_ids=set(),
    )
    seal = partial(
        phase._step_record_step_exception,
        shared_lock=RLock(),
        findings=findings,
        _flush_partial_manifest=lambda *a: flushes.append(list(records)),
        per_step_records=records,
        _append_terminal_step_record=phase._append_terminal_step_record,
    )

    def run():
        return RunCoordinator().run_sequential(
            state=state,
            execute_step=execute,
            resolve_transition=lambda *a: pytest.fail("failed step cannot transition"),
            apply_revised_plan=lambda *a: [],
            on_step_exception=seal,
        )

    if hard_stop:
        with pytest.raises(ProviderHardStopExceeded) as caught:
            run()
        assert caught.value is error
    else:
        run()
    assert attempted == ["model"]
    assert records[-1]["status"] == "execution_raised"
    assert records[-1]["error_type"] == type(error).__name__
    assert any(f.severity == "error" for f in findings)
    assert flushes and sync.called
    evidence.register_text.assert_not_called()


def test_parallel_hard_stop_blocks_queued_steps_and_preserves_type():
    attempted, failures = [], []
    stop = ProviderHardStopExceeded(code="BATCH_COST_LIMIT", detail="test")

    def execute(step):
        attempted.append(step.step_id)
        raise stop

    with pytest.raises(ProviderHardStopExceeded) as caught:
        RunCoordinator().run_parallel(
            steps=[AnalysisStep(step_id=str(i), intent="synthetic") for i in range(8)],
            max_workers=1,
            execute_step=execute,
            submit_step=lambda pool, fn, step: pool.submit(fn, step),
            on_worker_error=lambda step, error: failures.append((step.step_id, error)),
        )
    assert caught.value is stop
    assert attempted == ["0"]
    assert failures == [("0", stop)]


@pytest.mark.parametrize(
    "text, valid",
    [
        ("(analyzer failed: old failure)", False),
        ("", False),
        ("Observed association with limitations.", True),
    ],
)
def test_resume_rejects_legacy_failed_interpretation(tmp_path, text, valid):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.authority.run_input import (
        _explicit_step_authority_error,
    )

    store = EvidenceStore(root=tmp_path)
    authorities = {}
    checkpoint = {"generation_mode": "llm", "status": "ok"}
    for field, kind, producer, content in (
        ("script_evidence_id", "code", "coder", "print(1)"),
        ("step_summary_evidence_id", "statistic", "runner", "{}"),
        ("interpretation_evidence_id", "log", "analyzer", text),
    ):
        record = store.register_text(
            kind=kind,
            description=field,
            text=content,
            filename=field + ".txt",
            produced_by_step="s",
            producer=producer,
            generation_mode="system",
        )
        authorities[record.evidence_id] = record.model_dump(mode="json")
        checkpoint[field] = record.evidence_id
    error = _explicit_step_authority_error(
        record=checkpoint,
        evidence_ids=list(authorities),
        step_id="s",
        run_dir=tmp_path,
        records=authorities,
    )
    assert (error is None) is valid
    if not valid:
        assert "empty or records a failed call" in error
