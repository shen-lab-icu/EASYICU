"""Selective deterministic resume revalidation regression tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.contracts import ValidationFinding
from easyicu.research_agent.evidence import EvidenceStore
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ResearchContext,
)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Does the planned analysis run?",
        cohort=CohortDescriptor(
            cohort_name="fixture",
            database="synthetic",
            n_patients=2,
            n_stays=2,
        ),
        variables=[],
    )


def _register_success(
    *,
    run_dir: Path,
    evidence: EvidenceStore,
    step: AnalysisStep,
    summary_inputs: tuple[str, ...] = (),
    summary_payload: dict | None = None,
    aliases: tuple[str, ...] = (),
) -> tuple[dict, object, object]:
    script = evidence.register_text(
        kind="code",
        description="Sealed analysis script.",
        text="value = 1\n",
        filename=f"{step.step_id}_analysis.py",
        evidence_id=f"{step.step_id}_script",
        produced_by_step=step.step_id,
        producer="coder",
        generation_mode="llm",
        aliases=[*aliases, f"{step.step_id}_code"],
    )
    payload = summary_payload or {"status": "ok", "outputs": []}
    summary = evidence.register_text(
        kind="statistic",
        description="Sealed step summary.",
        text=json.dumps(payload, sort_keys=True),
        filename="step_summary.json",
        evidence_id=f"{step.step_id}_summary",
        produced_by_step=step.step_id,
        inputs=list(summary_inputs),
        script_evidence_id=script.evidence_id,
        producer="runner",
        generation_mode="executed",
        aliases=[f"{step.step_id}_summary_alias"],
    )
    record = {
        "step_id": step.step_id,
        "status": "ok",
        "evidence_ids": [script.evidence_id, summary.evidence_id],
        "script_evidence_id": script.evidence_id,
        "step_summary_evidence_id": summary.evidence_id,
        "step_summary": {"status": "forged", "outputs": ["mutable.csv"]},
        "analysis_request": {"step": step.model_dump(mode="json")},
    }
    return record, script, summary


def _empty_gates(pipeline_execute):
    return pipeline_execute._FinalDeterministicGateFindings((), (), (), (), ())


@pytest.fixture
def replay_environment(monkeypatch, tmp_path):
    from easyicu.research_agent import pipeline_execute

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    monkeypatch.setattr(
        pipeline_execute,
        "_deterministic_code_gate_findings",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        pipeline_execute,
        "CriticAgent",
        lambda *_args, **_kwargs: SimpleNamespace(
            review_step=lambda **_review_kwargs: SimpleNamespace(
                status="pass",
                model_dump=lambda **_dump_kwargs: {
                    "status": "pass",
                    "reviewer": "deterministic-fixture",
                },
            )
        ),
    )
    return pipeline_execute, run_dir, evidence


def _revalidate(
    pipeline_execute,
    *,
    run_dir: Path,
    evidence,
    records: list[dict],
    plan: AnalysisPlan,
    resume_from_step_id: str | None = None,
    cohort_path: Path | None = None,
    universe_path: Path | None = None,
):
    return pipeline_execute._selectively_revalidate_resume_successes(
        resume_state={"per_step_records": records, "findings": []},
        plan=plan,
        context=_context(),
        evidence=evidence,
        run_dir=run_dir,
        cohort_path=cohort_path or run_dir / "cohort.parquet",
        universe_path=universe_path or cohort_path or run_dir / "cohort.parquet",
        resume_from_step_id=resume_from_step_id,
    )


def test_current_fingerprint_is_a_true_zero_work_fast_path(monkeypatch, tmp_path):
    from easyicu.research_agent import pipeline_execute

    class EvidenceMustNotBeRead:
        def records(self):
            pytest.fail("current fingerprints must not touch evidence")

    step = AnalysisStep(step_id="01_model", intent="Fit the planned model.")
    record = {"step_id": step.step_id, "status": "ok"}
    record.update(pipeline_execute._deterministic_gate_stamp())

    result = _revalidate(
        pipeline_execute,
        run_dir=tmp_path,
        evidence=EvidenceMustNotBeRead(),
        records=[record],
        plan=AnalysisPlan(research_question="Question", steps=[step]),
    )

    assert result.resume_state["per_step_records"] == [record]
    assert result.revalidated_step_ids == ()
    assert result.invalidated_step_ids == ()


def test_legacy_success_revalidates_once_without_execution(
    replay_environment,
    monkeypatch,
):
    pipeline_execute, run_dir, evidence = replay_environment
    step = AnalysisStep(step_id="01_model", intent="Fit the planned model.")
    plan = AnalysisPlan(research_question="Question", steps=[step])
    record, _, _ = _register_success(
        run_dir=run_dir,
        evidence=evidence,
        step=step,
    )
    calls = []
    monkeypatch.setattr(
        pipeline_execute,
        "_evaluate_final_deterministic_gates",
        lambda **kwargs: calls.append(kwargs["step"].step_id)
        or _empty_gates(pipeline_execute),
    )

    first = _revalidate(
        pipeline_execute,
        run_dir=run_dir,
        evidence=evidence,
        records=[record],
        plan=plan,
    )
    latest = first.resume_state["per_step_records"][-1]

    assert calls == [step.step_id]
    assert latest["status"] == "ok"
    assert latest["revalidated_without_execution"] is True
    assert latest["step_summary"] == {"status": "ok", "outputs": []}
    assert latest["deterministic_gate_fingerprint"]
    assert latest["attempt_id"].startswith(f"{step.step_id}:resume_revalidation:")

    second = _revalidate(
        pipeline_execute,
        run_dir=run_dir,
        evidence=evidence,
        records=first.resume_state["per_step_records"],
        plan=plan,
    )
    assert second.resume_state["per_step_records"] == first.resume_state[
        "per_step_records"
    ]
    assert calls == [step.step_id]


def test_missing_or_tampered_summary_fails_closed(
    replay_environment,
    monkeypatch,
):
    pipeline_execute, run_dir, evidence = replay_environment
    step = AnalysisStep(step_id="01_model", intent="Fit the planned model.")
    record, _, summary = _register_success(
        run_dir=run_dir,
        evidence=evidence,
        step=step,
    )
    from easyicu.research_agent.runtime_artifacts import verified_run_evidence_path

    summary_path = verified_run_evidence_path(run_dir, summary)
    assert summary_path is not None
    summary_path.write_text('{"status":"tampered"}', encoding="utf-8")
    monkeypatch.setattr(
        pipeline_execute,
        "_evaluate_final_deterministic_gates",
        lambda **_kwargs: pytest.fail("corrupt authority must not reach gates"),
    )

    result = _revalidate(
        pipeline_execute,
        run_dir=run_dir,
        evidence=evidence,
        records=[record],
        plan=AnalysisPlan(research_question="Question", steps=[step]),
    )

    latest = result.resume_state["per_step_records"][-1]
    assert latest["status"] == "resume_validator_invalid"
    assert "digest verification" in latest["resume_invalidation_reason"]
    assert f"{step.step_id}_summary_alias" not in evidence.aliases()
    assert summary.evidence_id not in evidence.aliases().values()


def test_historical_invalid_upstream_blocks_later_explicit_cut(tmp_path):
    from easyicu.research_agent import pipeline_execute
    from easyicu.research_agent.run_input_capsule import RunInputIdentityError

    upstream = AnalysisStep(step_id="01_upstream", intent="Produce inputs.")
    downstream = AnalysisStep(step_id="02_downstream", intent="Consume inputs.")

    class EvidenceMustNotBeRead:
        def records(self):
            pytest.fail("explicit cut must fail before evidence replay")

    with pytest.raises(RunInputIdentityError, match="already-invalid upstream"):
        _revalidate(
            pipeline_execute,
            run_dir=tmp_path,
            evidence=EvidenceMustNotBeRead(),
            records=[
                {
                    "step_id": upstream.step_id,
                    "status": "resume_validator_invalid",
                }
            ],
            plan=AnalysisPlan(
                research_question="Question",
                steps=[upstream, downstream],
            ),
            resume_from_step_id=downstream.step_id,
        )


def test_checkpoint_cannot_swap_in_same_step_decoy_script(
    replay_environment,
    monkeypatch,
):
    pipeline_execute, run_dir, evidence = replay_environment
    step = AnalysisStep(step_id="01_model", intent="Fit the planned model.")
    record, actual_script, _ = _register_success(
        run_dir=run_dir,
        evidence=evidence,
        step=step,
    )
    decoy = evidence.register_text(
        kind="code",
        description="Benign decoy code.",
        text="BENIGN = True\n",
        filename="decoy.py",
        evidence_id="01_model_decoy",
        produced_by_step=step.step_id,
        producer="coder",
        generation_mode="llm",
    )
    record["evidence_ids"].append(decoy.evidence_id)
    record["script_evidence_id"] = decoy.evidence_id
    monkeypatch.setattr(
        pipeline_execute,
        "_evaluate_final_deterministic_gates",
        lambda **_kwargs: pytest.fail("decoy lineage must fail before gates"),
    )

    result = _revalidate(
        pipeline_execute,
        run_dir=run_dir,
        evidence=evidence,
        records=[record],
        plan=AnalysisPlan(research_question="Question", steps=[step]),
    )

    latest = result.resume_state["per_step_records"][-1]
    assert latest["status"] == "resume_validator_invalid"
    assert actual_script.evidence_id in latest["resume_invalidation_reason"]
    assert decoy.evidence_id in latest["resume_invalidation_reason"]


def test_host_probe_invalid_json_fails_closed(replay_environment):
    pipeline_execute, run_dir, evidence = replay_environment
    summary = evidence.register_text(
        kind="statistic",
        description="Probe summary.",
        text="not-json",
        filename="probe_summary.json",
        evidence_id="probe_summary",
        produced_by_step="00_probe",
        producer="pipeline",
        generation_mode="deterministic_probe",
    )
    table = evidence.register_text(
        kind="table",
        description="Probe table.",
        text="variable,n\nx,2\n",
        filename="probe_variable_profile.csv",
        evidence_id="probe_table",
        produced_by_step="00_probe",
        producer="pipeline",
        generation_mode="deterministic_probe",
    )
    record = {
        "step_id": "00_probe",
        "status": "ok",
        "step_authority_kind": pipeline_execute._HOST_PROBE_AUTHORITY_KIND,
        "evidence_ids": [summary.evidence_id, table.evidence_id],
        "probe_summary_evidence_id": summary.evidence_id,
        "probe_table_evidence_id": table.evidence_id,
    }

    result = _revalidate(
        pipeline_execute,
        run_dir=run_dir,
        evidence=evidence,
        records=[record],
        plan=AnalysisPlan(research_question="Question", steps=[]),
    )

    latest = result.resume_state["per_step_records"][-1]
    assert latest["status"] == "resume_validator_invalid"
    assert "not readable JSON" in latest["resume_invalidation_reason"]


def _register_host_cohort_materializer(
    *,
    pipeline_execute,
    run_dir: Path,
    evidence: EvidenceStore,
    step: AnalysisStep,
    source_name: str = "cohort_analysis.parquet",
):
    import pandas as pd

    cohort_path = run_dir / source_name
    pd.DataFrame({"stay_id": [1, 2]}).to_parquet(cohort_path, index=False)
    cohort = evidence.register_file(
        kind="table",
        description="Host materialized cohort.",
        source_path=cohort_path,
        evidence_id="analysis_cohort_execute_repair",
        produced_by_step=step.step_id,
        producer="cohort_repair",
        generation_mode="llm",
        metadata={"reason": "agent cohort prose translated to locked predicates"},
    )
    record = {
        "step_id": step.step_id,
        "status": "ok",
        "generation_mode": pipeline_execute._HOST_COHORT_MATERIALIZER_GENERATION_MODE,
        "step_authority_kind": (
            pipeline_execute._HOST_COHORT_MATERIALIZER_AUTHORITY_KIND
        ),
        pipeline_execute._HOST_COHORT_MATERIALIZER_AUTHORITY_FIELD: (
            cohort.evidence_id
        ),
        "step_summary": {
            "output_files": {"table:analysis_cohort": "cohort_analysis.parquet"},
            "n_universe": 3,
            "n_analysis_cohort": 2,
        },
        "evidence_ids": [cohort.evidence_id],
    }
    return record, cohort_path


def test_host_cohort_materializer_revalidates_without_script_or_llm(
    replay_environment,
    monkeypatch,
):
    import pandas as pd

    pipeline_execute, run_dir, evidence = replay_environment
    step = AnalysisStep(
        step_id="01_cohort",
        intent="Materialize the Agent-selected cohort.",
        expected_outputs=["table:analysis_cohort"],
    )
    record, cohort_path = _register_host_cohort_materializer(
        pipeline_execute=pipeline_execute,
        run_dir=run_dir,
        evidence=evidence,
        step=step,
    )
    universe_path = run_dir / "universe.parquet"
    pd.DataFrame({"stay_id": [1, 2, 3]}).to_parquet(universe_path, index=False)
    monkeypatch.setattr(
        pipeline_execute,
        "_evaluate_final_deterministic_gates",
        lambda **_kwargs: pytest.fail("host materializer must not enter agent gates"),
    )

    result = _revalidate(
        pipeline_execute,
        run_dir=run_dir,
        evidence=evidence,
        records=[record],
        plan=AnalysisPlan(research_question="Question", steps=[step]),
        cohort_path=cohort_path,
        universe_path=universe_path,
    )

    latest = result.resume_state["per_step_records"][-1]
    assert latest["status"] == "ok"
    assert latest["revalidated_without_execution"] is True
    assert latest["deterministic_gate_fingerprint"]


def test_arbitrary_table_cannot_impersonate_host_cohort_materializer(
    replay_environment,
    monkeypatch,
):
    pipeline_execute, run_dir, evidence = replay_environment
    step = AnalysisStep(
        step_id="01_cohort",
        intent="Materialize the Agent-selected cohort.",
        expected_outputs=["table:analysis_cohort"],
    )
    record, _ = _register_host_cohort_materializer(
        pipeline_execute=pipeline_execute,
        run_dir=run_dir,
        evidence=evidence,
        step=step,
        source_name="arbitrary.parquet",
    )
    monkeypatch.setattr(
        pipeline_execute,
        "_evaluate_final_deterministic_gates",
        lambda **_kwargs: pytest.fail("forged host record must fail exact contract"),
    )

    result = _revalidate(
        pipeline_execute,
        run_dir=run_dir,
        evidence=evidence,
        records=[record],
        plan=AnalysisPlan(research_question="Question", steps=[step]),
    )

    latest = result.resume_state["per_step_records"][-1]
    assert latest["status"] == "resume_validator_invalid"
    assert "canonical cohort" in latest["resume_invalidation_reason"] or (
        "canonical cohort product" in latest["resume_invalidation_reason"]
    )


def test_deterministic_error_invalidates_evidence_dependent_downstream(
    replay_environment,
    monkeypatch,
):
    pipeline_execute, run_dir, evidence = replay_environment
    upstream = AnalysisStep(step_id="01_upstream", intent="Produce inputs.")
    downstream = AnalysisStep(step_id="02_downstream", intent="Consume inputs.")
    upstream_record, _, upstream_summary = _register_success(
        run_dir=run_dir,
        evidence=evidence,
        step=upstream,
        aliases=("current_primary_result",),
    )
    downstream_record, _, _ = _register_success(
        run_dir=run_dir,
        evidence=evidence,
        step=downstream,
        summary_inputs=(upstream_summary.evidence_id,),
    )

    def gates(**kwargs):
        if kwargs["step"].step_id == upstream.step_id:
            finding = ValidationFinding(
                validator="test_gate",
                severity="error",
                message="Current deterministic contract fails.",
            )
            return pipeline_execute._FinalDeterministicGateFindings(
                (), (), (), (finding,), ()
            )
        pytest.fail("dependent downstream must be invalidated without replay")

    monkeypatch.setattr(
        pipeline_execute,
        "_evaluate_final_deterministic_gates",
        gates,
    )
    result = _revalidate(
        pipeline_execute,
        run_dir=run_dir,
        evidence=evidence,
        records=[upstream_record, downstream_record],
        plan=AnalysisPlan(
            research_question="Question",
            steps=[upstream, downstream],
        ),
    )

    current = {
        record["step_id"]: record
        for record in pipeline_execute.current_step_records(
            result.resume_state["per_step_records"]
        )
    }
    assert current[upstream.step_id]["status"] == "resume_validator_invalid"
    assert current[downstream.step_id]["status"] == "resume_validator_invalid"
    assert "invalidated upstream" in current[downstream.step_id][
        "resume_invalidation_reason"
    ]
    assert "current_primary_result" not in evidence.aliases()


def test_replay_ignores_mutable_checkpoint_summary_and_binding_receipts(
    replay_environment,
    monkeypatch,
):
    pipeline_execute, run_dir, evidence = replay_environment
    step = AnalysisStep(step_id="01_model", intent="Fit the planned model.")
    record, _, _ = _register_success(
        run_dir=run_dir,
        evidence=evidence,
        step=step,
        summary_payload={"status": "ok", "outputs": ["sealed.csv"]},
    )
    record["resolved_inputs_path"] = "/tmp/forged.json"
    record["resolved_input_bindings"] = {"artifact:forged": {"path": "/tmp/x"}}
    captured = {}

    def gates(**kwargs):
        captured["summary"] = kwargs["step_summary"]
        captured["bindings"] = kwargs["resolved_input_bindings"]
        return _empty_gates(pipeline_execute)

    monkeypatch.setattr(pipeline_execute, "_evaluate_final_deterministic_gates", gates)
    result = _revalidate(
        pipeline_execute,
        run_dir=run_dir,
        evidence=evidence,
        records=[record],
        plan=AnalysisPlan(research_question="Question", steps=[step]),
    )

    assert captured == {
        "summary": {"status": "ok", "outputs": ["sealed.csv"]},
        "bindings": {},
    }
    latest = result.resume_state["per_step_records"][-1]
    assert "resolved_inputs_path" not in latest
    assert "resolved_input_bindings" not in latest


def test_prior_blocking_critic_cannot_be_upgraded_by_replay(
    replay_environment,
    monkeypatch,
):
    pipeline_execute, run_dir, evidence = replay_environment
    step = AnalysisStep(step_id="01_model", intent="Fit the planned model.")
    record, _, _ = _register_success(
        run_dir=run_dir,
        evidence=evidence,
        step=step,
    )
    record["critique_report"] = {"status": "blocked"}
    monkeypatch.setattr(
        pipeline_execute,
        "_evaluate_final_deterministic_gates",
        lambda **_kwargs: _empty_gates(pipeline_execute),
    )

    result = _revalidate(
        pipeline_execute,
        run_dir=run_dir,
        evidence=evidence,
        records=[record],
        plan=AnalysisPlan(research_question="Question", steps=[step]),
    )

    latest = result.resume_state["per_step_records"][-1]
    assert latest["status"] == "resume_validator_invalid"
    assert "prior deterministic Critic status remains blocked" in latest[
        "resume_invalidation_reason"
    ]


def test_invalid_checkpoint_inherits_monotonic_provider_and_repair_budgets(
    replay_environment,
    monkeypatch,
):
    pipeline_execute, run_dir, evidence = replay_environment
    step = AnalysisStep(step_id="01_model", intent="Fit the planned model.")
    record, _, _ = _register_success(
        run_dir=run_dir,
        evidence=evidence,
        step=step,
    )
    inherited = {
        "step_provider_call_budget": 5,
        "step_provider_call_attempts": 4,
        "step_provider_call_categories": ["coder", "concept_auditor"],
        "step_provider_call_receipt_version": 1,
        "step_provider_call_receipt": {"attempts": 4},
        "step_llm_repair_budget": 2,
        "step_llm_repair_attempts": 2,
        "step_llm_repair_classes": ["contract", "runtime"],
    }
    record.update(inherited)
    error = ValidationFinding(
        validator="test_gate",
        severity="error",
        message="Current gate fails.",
    )
    monkeypatch.setattr(
        pipeline_execute,
        "_evaluate_final_deterministic_gates",
        lambda **_kwargs: pipeline_execute._FinalDeterministicGateFindings(
            (), (), (), (error,), ()
        ),
    )

    result = _revalidate(
        pipeline_execute,
        run_dir=run_dir,
        evidence=evidence,
        records=[record],
        plan=AnalysisPlan(research_question="Question", steps=[step]),
    )
    latest = result.resume_state["per_step_records"][-1]

    assert latest["status"] == "resume_validator_invalid"
    assert {key: latest[key] for key in inherited} == inherited


def test_alias_retirement_failure_rolls_back_manifest_and_alias_authority(
    replay_environment,
    monkeypatch,
):
    pipeline_execute, run_dir, evidence = replay_environment
    step = AnalysisStep(step_id="01_model", intent="Fit the planned model.")
    record, _, _ = _register_success(
        run_dir=run_dir,
        evidence=evidence,
        step=step,
        aliases=("current_result",),
    )
    initial_state = {"per_step_records": [record], "findings": []}
    pipeline_execute.write_run_checkpoint(
        run_dir / "manifest_partial.json",
        initial_state,
    )
    aliases_before = evidence.aliases()
    error = ValidationFinding(
        validator="test_gate",
        severity="error",
        message="Current gate fails.",
    )
    monkeypatch.setattr(
        pipeline_execute,
        "_evaluate_final_deterministic_gates",
        lambda **_kwargs: pipeline_execute._FinalDeterministicGateFindings(
            (), (), (), (error,), ()
        ),
    )
    monkeypatch.setattr(
        evidence,
        "retire_steps_current_aliases",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("save failed")),
    )

    with pytest.raises(RuntimeError, match="manifest was rolled back"):
        _revalidate(
            pipeline_execute,
            run_dir=run_dir,
            evidence=evidence,
            records=[record],
            plan=AnalysisPlan(research_question="Question", steps=[step]),
        )

    assert evidence.aliases() == aliases_before
    rolled_back = json.loads((run_dir / "manifest_partial.json").read_text())
    assert rolled_back["per_step_records"] == initial_state["per_step_records"]
    assert rolled_back["findings"] == initial_state["findings"]


def test_checkpoint_write_failure_never_retires_aliases(
    replay_environment,
    monkeypatch,
):
    pipeline_execute, run_dir, evidence = replay_environment
    step = AnalysisStep(step_id="01_model", intent="Fit the planned model.")
    record, _, _ = _register_success(
        run_dir=run_dir,
        evidence=evidence,
        step=step,
        aliases=("current_result",),
    )
    aliases_before = evidence.aliases()
    error = ValidationFinding(
        validator="test_gate",
        severity="error",
        message="Current gate fails.",
    )
    monkeypatch.setattr(
        pipeline_execute,
        "_evaluate_final_deterministic_gates",
        lambda **_kwargs: pipeline_execute._FinalDeterministicGateFindings(
            (), (), (), (error,), ()
        ),
    )
    monkeypatch.setattr(
        pipeline_execute,
        "write_run_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        _revalidate(
            pipeline_execute,
            run_dir=run_dir,
            evidence=evidence,
            records=[record],
            plan=AnalysisPlan(research_question="Question", steps=[step]),
        )

    assert evidence.aliases() == aliases_before


def test_replay_uses_shared_gates_and_never_constructs_llm_auditor():
    import inspect

    from easyicu.research_agent import pipeline_execute

    replay_source = inspect.getsource(
        pipeline_execute._selectively_revalidate_resume_successes
    )
    fresh_source = inspect.getsource(pipeline_execute.run_execute_phase)

    assert "_deterministic_code_gate_findings(" in replay_source
    assert "_evaluate_final_deterministic_gates(" in replay_source
    assert "LLMConceptAuditor(" not in replay_source
    assert "_deterministic_code_gate_findings(" in fresh_source
    assert "_evaluate_final_deterministic_gates(" in fresh_source


def test_resume_application_preserves_append_only_revalidation_history(tmp_path):
    from easyicu.research_agent.pipeline_resume import ResumeController

    step = AnalysisStep(step_id="01_model", intent="Fit the planned model.")
    history = [
        {"step_id": step.step_id, "status": "ok", "attempt_id": "original"},
        {
            "step_id": step.step_id,
            "status": "ok",
            "attempt_id": "revalidated",
            "revalidated_without_execution": True,
        },
    ]
    applied = ResumeController(
        plan=AnalysisPlan(research_question="Question", steps=[step]),
        run_dir=tmp_path,
        resume_state={"per_step_records": history, "findings": []},
    ).apply()

    assert applied.audit_history == history
    assert applied.per_step_records == [history[-1]]


def test_execute_phase_writes_resume_audit_history_separately_from_authority_view():
    import inspect

    from easyicu.research_agent import pipeline_execute

    source = inspect.getsource(pipeline_execute.run_execute_phase)

    assert "step_attempt_history.extend(resume_application.audit_history)" in source
    assert '"step_attempt_history": step_attempt_history' in source
    assert "per_step_records.extend(resume_application.per_step_records)" in source
