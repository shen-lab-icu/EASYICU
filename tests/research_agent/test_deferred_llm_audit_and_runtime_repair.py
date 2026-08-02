from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.authority.provider_budget import (
    PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION,
)
from easyicu.research_agent.providers.mocks import (
    PatternScriptedMockLLMClient,
    ScriptedMockLLMClient,
)


def _prompt_text(messages) -> str:
    return "\n".join(str(message.content or "") for message in messages)


def _call_count(client, marker: str) -> int:
    folded = marker.casefold()
    return sum(
        folded in _prompt_text(messages).casefold()
        for messages, _kwargs in client.calls
    )


def _matching_user_prompts(client, marker: str) -> list[str]:
    folded = marker.casefold()
    prompts = []
    for messages, _kwargs in client.calls:
        full_prompt = _prompt_text(messages)
        if folded not in full_prompt.casefold():
            continue
        prompts.append(
            next(
                (
                    str(message.content or "")
                    for message in reversed(messages)
                    if message.role == "user"
                ),
                "",
            )
        )
    return prompts


def _isolate_article_suite_contract(monkeypatch) -> None:
    """Keep orchestration regressions independent of article-suite breadth."""

    from easyicu.research_agent.agents.core import PlannerAgent

    original_run = PlannerAgent.run

    def run_without_article_suite(self, context, **kwargs):
        kwargs["enforce_article_contract"] = False
        return original_run(self, context, **kwargs)

    monkeypatch.setattr(PlannerAgent, "run", run_without_article_suite)


def _summary_script(*, phase: str) -> str:
    return f"""
import json
import os
import pandas as pd

out = os.environ["STEP_OUT_DIR"]
summary = {{
    "phase": {phase!r},
    "n": 3,
    "output_files": {{"table:cohort_summary": "cohort_summary.csv"}},
}}
pd.DataFrame([summary]).to_csv(
    os.path.join(out, "cohort_summary.csv"), index=False
)
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle)
"""


def _plan() -> str:
    return json.dumps(
        {
            "research_question": "Summarize the ICU cohort.",
            "steps": [
                {
                    "step_id": "01_summary",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Produce a descriptive cohort summary.",
                    "inputs": ["stay_id"],
                    "expected_outputs": ["table:cohort_summary"],
                    "method": "descriptive_summary",
                    "icu_rule_refs": [],
                }
            ],
            "rationale": "deferred concept-audit orchestration regression",
        }
    )


def _latest_step_record(run_dir: Path) -> dict:
    partial = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    return [
        record
        for record in partial["per_step_records"]
        if record.get("step_id") == "01_summary"
    ][-1]


def test_provider_budget_failure_is_not_replayed_as_concept_authority(
    monkeypatch, tmp_path: Path
) -> None:
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.execution import concept_audit

    audit = SimpleNamespace(
        auditor_identity_sha256="a" * 64,
        environment_sha256="b" * 64,
        validator_implementation_sha256="c" * 64,
        audit_key="d" * 64,
    )
    verified = SimpleNamespace(
        capsule=SimpleNamespace(concept_audit=audit),
    )
    monkeypatch.setattr(
        concept_audit,
        "read_concept_audit_findings",
        lambda *_args, **_kwargs: [
            ValidationFinding(
                validator="provider_call_budget",
                severity="error",
                message="budget exhausted before audit",
                detail={"step_id": "01_summary"},
            )
        ],
    )

    assert (
        concept_audit.verified_capsule_concept_audit_replay(
            verified,
            run_dir=tmp_path,
            auditor_identity_sha256="a" * 64,
            environment_sha256="b" * 64,
            validator_implementation_sha256="c" * 64,
        )
        is None
    )


def test_llm_concept_audit_runs_once_only_after_local_contracts_pass(
    ra, tmp_path: Path, monkeypatch
) -> None:
    _isolate_article_suite_contract(monkeypatch)
    from easyicu.research_agent.audits.validators import PrimaryModelContractValidator
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.execution.runner import CodeRunner

    initial_code = _summary_script(phase="INITIAL_CONTRACT_ERROR")
    repaired_code = _summary_script(phase="FINAL_CONTRACT_VALID")

    llm = PatternScriptedMockLLMClient(
        [
            ("ICU-AWARE RESEARCH PLAN", [_plan()]),
            ("WRITE THE PYTHON CODE", [initial_code]),
            ("REPAIR THE PYTHON CODE", [repaired_code] * 2),
            (
                "CONSERVATIVE ICU CONCEPT-USE AUDITOR",
                [json.dumps({"findings": []})],
            ),
            ("INTERPRET THE RESULTS", ["The cohort summary is available."]),
        ]
    )

    def contract_audit(self, *, step, step_summary, **kwargs):
        del self, kwargs
        if step_summary.get("phase") != "INITIAL_CONTRACT_ERROR":
            return []
        return [
            ValidationFinding(
                validator="primary_model_contract",
                severity="error",
                message="Force one deterministic-contract repair.",
                detail={"step_id": step.step_id},
            )
        ]

    executed_code: list[str] = []
    original_run = CodeRunner.run

    def recording_run(self, *, step_id, code, resolved_inputs_path=None):
        executed_code.append(code)
        return original_run(
            self,
            step_id=step_id,
            code=code,
            resolved_inputs_path=resolved_inputs_path,
        )

    monkeypatch.setattr(PrimaryModelContractValidator, "audit", contract_audit)
    monkeypatch.setattr(CodeRunner, "run", recording_run)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=True,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
        max_code_repair_attempts=1,
        runner_kind="subprocess",
    )
    result = pipeline.run(
        question="Summarize the ICU cohort.",
        cohort=pd.DataFrame({"stay_id": [1, 2, 3], "death": [0, 1, 0]}),
        cohort_name="deferred_audit_test",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )

    record = _latest_step_record(Path(result.workdir))
    assert len(executed_code) == 2
    assert "INITIAL_CONTRACT_ERROR" in executed_code[0]
    assert "FINAL_CONTRACT_VALID" in executed_code[1]
    audit_prompts = _matching_user_prompts(llm, "CONSERVATIVE ICU CONCEPT-USE AUDITOR")
    assert len(audit_prompts) == 1
    assert "FINAL_CONTRACT_VALID" in audit_prompts[0]
    assert "INITIAL_CONTRACT_ERROR" not in audit_prompts[0]
    assert record["status"] == "ok"
    assert record["llm_concept_audit_status"] == "completed"
    assert record["llm_concept_approved_code_sha256"] == record["executed_code_sha256"]
    assert (
        record["deterministic_contract_approved_code_sha256"]
        == record["executed_code_sha256"]
    )


def test_automatic_contract_repair_does_not_consume_llm_contract_allowance(
    ra, tmp_path: Path, monkeypatch
) -> None:
    _isolate_article_suite_contract(monkeypatch)
    from easyicu.research_agent.audits.validators import PrimaryModelContractValidator
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.execution import phase

    initial_code = _summary_script(phase="INITIAL_CONTRACT_ERROR")
    structural_code = _summary_script(phase="STRUCTURAL_CONTRACT_ERROR")
    repaired_code = _summary_script(phase="FINAL_CONTRACT_VALID")
    llm = PatternScriptedMockLLMClient(
        [
            ("ICU-AWARE RESEARCH PLAN", [_plan()]),
            ("WRITE THE PYTHON CODE", [initial_code]),
            ("REPAIR THE PYTHON CODE", [repaired_code] * 2),
            (
                "CONSERVATIVE ICU CONCEPT-USE AUDITOR",
                [json.dumps({"findings": []})],
            ),
            ("INTERPRET THE RESULTS", ["The cohort summary is available."]),
        ]
    )

    def contract_audit(self, *, step, step_summary, **kwargs):
        del self, kwargs
        if step_summary.get("phase") == "FINAL_CONTRACT_VALID":
            return []
        return [
            ValidationFinding(
                validator="primary_model_contract",
                severity="error",
                message="Force deterministic then LLM contract repair.",
                detail={"step_id": step.step_id},
            )
        ]

    def one_structural_repair(*, code, findings, previous_repair=None):
        del findings
        if (
            "INITIAL_CONTRACT_ERROR" in code
            and previous_repair != "render_only_effect_echo_suppression_v1"
        ):
            return "render_only_effect_echo_suppression_v1", structural_code
        return None

    monkeypatch.setattr(PrimaryModelContractValidator, "audit", contract_audit)
    monkeypatch.setattr(phase, "deterministic_contract_repair", one_structural_repair)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=True,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=True,
        max_code_repair_attempts=1,
        runner_kind="subprocess",
    )

    result = pipeline.run(
        question="Summarize the ICU cohort.",
        cohort=pd.DataFrame({"stay_id": [1, 2, 3], "death": [0, 1, 0]}),
        cohort_name="independent_contract_budget_test",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )

    record = _latest_step_record(Path(result.workdir))
    assert record["status"] == "ok"
    assert record["contract_repair_attempts"] == 2
    assert record["llm_contract_repair_attempts"] == 1
    assert record["code_repair_attempts"] == 2
    assert record["step_llm_repair_classes"] == ["contract"]
    assert _call_count(llm, "REPAIR THE PYTHON CODE") >= 1


def test_disabled_llm_audit_records_final_gate_without_claiming_llm_approval(
    ra, tmp_path: Path, monkeypatch
) -> None:
    _isolate_article_suite_contract(monkeypatch)
    llm = PatternScriptedMockLLMClient(
        [
            ("ICU-AWARE RESEARCH PLAN", [_plan()]),
            ("WRITE THE PYTHON CODE", [_summary_script(phase="AUDIT_DISABLED")]),
            ("INTERPRET THE RESULTS", ["The cohort summary is available."]),
        ]
    )

    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
        runner_kind="subprocess",
    )
    result = pipeline.run(
        question="Summarize the ICU cohort.",
        cohort=pd.DataFrame({"stay_id": [1, 2, 3], "death": [0, 1, 0]}),
        cohort_name="audit_disabled_test",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )

    record = _latest_step_record(Path(result.workdir))
    assert record["status"] == "ok"
    assert record["llm_concept_audit_status"] == "disabled"
    assert "llm_concept_approved_code_sha256" not in record
    assert (
        record["final_concept_gate_approved_code_sha256"]
        == record["executed_code_sha256"]
    )


def test_typed_lossy_guard_repairs_without_logical_llm_budget(
    ra, tmp_path: Path, monkeypatch
) -> None:
    _isolate_article_suite_contract(monkeypatch)
    lossy_code = r"""
import json
import os
import numpy as np
import pandas as pd

cohort = pd.read_parquet(os.environ["COHORT_PARQUET"])

def numeric_coercion_audit(frame, column):
    original = frame[column]
    coerced = pd.to_numeric(original, errors="coerce")
    record = {
        "newly_invalid_or_coerced_n": int(
            (original.notna() & coerced.isna()).sum()
        ),
    }
    return coerced, record

stage_numeric, coercion_record = numeric_coercion_audit(
    cohort, "aki_stage_max"
)
out = os.environ["STEP_OUT_DIR"]
table_path = os.path.join(out, "exposure_qc.csv")
pd.DataFrame({"aki_stage_max": stage_numeric}).to_csv(table_path, index=False)
summary = {
    "n": int(len(cohort)),
    "newly_invalid_or_coerced_n": int(
        coercion_record["newly_invalid_or_coerced_n"]
    ),
    "output_files": {"table:exposure_qc": "exposure_qc.csv"},
}
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle)
"""
    plan = json.dumps(
        {
            "research_question": "Audit an ordered numeric exposure.",
            "steps": [
                {
                    "step_id": "02_exposure_qc",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Audit the ordered exposure without silent coercion.",
                    "inputs": ["aki_stage_max"],
                    "expected_outputs": ["table:exposure_qc"],
                    "method": "ordered_exposure_quality_control",
                    "icu_rule_refs": [],
                }
            ],
            "rationale": "typed deterministic repair budget regression",
        }
    )

    llm = PatternScriptedMockLLMClient(
        [
            ("ICU-AWARE RESEARCH PLAN", [plan]),
            ("WRITE THE PYTHON CODE", [lossy_code]),
            (
                "REPAIR THE PYTHON CODE",
                [AssertionError("typed mechanical repair must avoid the coder")],
            ),
            (
                "CONSERVATIVE ICU CONCEPT-USE AUDITOR",
                [json.dumps({"findings": []})],
            ),
            (
                "INTERPRET THE RESULTS",
                ["The exposure quality-control table is available."],
            ),
        ]
    )
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=True,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
        max_step_llm_repair_attempts=0,
        runner_kind="subprocess",
    )
    result = pipeline.run(
        question="Audit an ordered numeric exposure.",
        cohort=pd.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "aki_stage_max": [0.0, 1.0, 2.0],
                "death": [0, 1, 0],
            }
        ),
        cohort_name="lossy_guard_test",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="02_exposure_qc",
        stop_after_analysis=True,
    )

    partial = json.loads(
        (Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8")
    )
    record = [
        item
        for item in partial["per_step_records"]
        if item.get("step_id") == "02_exposure_qc"
    ][-1]
    assert record["status"] == "ok"
    assert record["deterministic_concept_repairs"] == 2
    assert record["applied_concept_repair_names"] == [
        "lossy_numeric_coercion_guard_v1",
        "strict_numeric_nonfinite_guard_v1",
    ]
    assert record.get("step_llm_repair_attempts", 0) == 0
    assert _call_count(llm, "REPAIR THE PYTHON CODE") == 0
    assert _call_count(llm, "CONSERVATIVE ICU CONCEPT-USE AUDITOR") == 1


def test_runtime_repair_transport_retry_does_not_reexecute_known_failure(
    ra, tmp_path: Path, monkeypatch
) -> None:
    _isolate_article_suite_contract(monkeypatch)
    from easyicu.research_agent.execution.runner import CodeRunner

    broken_code = "raise RuntimeError('KNOWN_RUNTIME_FAILURE')\n"

    llm = PatternScriptedMockLLMClient(
        [
            ("ICU-AWARE RESEARCH PLAN", [_plan()]),
            ("WRITE THE PYTHON CODE", [broken_code]),
            (
                "REPAIR THE PYTHON CODE",
                [RuntimeError("provider rate limited")] * 2,
            ),
            ("INTERPRET THE RESULTS", ["The fallback summary is available."]),
        ]
    )

    executed_code: list[str] = []
    original_run = CodeRunner.run

    def recording_run(self, *, step_id, code, resolved_inputs_path=None):
        executed_code.append(code)
        return original_run(
            self,
            step_id=step_id,
            code=code,
            resolved_inputs_path=resolved_inputs_path,
        )

    monkeypatch.setattr(CodeRunner, "run", recording_run)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        enable_deterministic_runner_repair=False,
        max_code_repair_attempts=3,
        runner_kind="subprocess",
    )
    result = pipeline.run(
        question="Summarize the ICU cohort.",
        cohort=pd.DataFrame({"stay_id": [1, 2, 3], "death": [0, 1, 0]}),
        cohort_name="runtime_transport_retry_test",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )

    record = _latest_step_record(Path(result.workdir))
    assert _call_count(llm, "REPAIR THE PYTHON CODE") == 2
    assert sum("KNOWN_RUNTIME_FAILURE" in code for code in executed_code) == 1
    assert record["status"] == "repair_failed"


def test_noop_runtime_repair_is_retried_without_reexecution(
    ra, tmp_path: Path, monkeypatch
) -> None:
    _isolate_article_suite_contract(monkeypatch)
    from easyicu.research_agent.execution.runner import CodeRunner

    broken_code = (
        _summary_script(phase="NOOP_RUNTIME_FAILURE")
        + "\nraise RuntimeError('NOOP_RUNTIME_FAILURE')\n"
    )

    llm = PatternScriptedMockLLMClient(
        [
            ("ICU-AWARE RESEARCH PLAN", [_plan()]),
            ("WRITE THE PYTHON CODE", [broken_code]),
            ("REPAIR THE PYTHON CODE", [broken_code] * 4),
            ("INTERPRET THE RESULTS", ["The fallback summary is available."]),
        ]
    )

    executed_code: list[str] = []
    original_run = CodeRunner.run

    def recording_run(self, *, step_id, code, resolved_inputs_path=None):
        executed_code.append(code)
        return original_run(
            self,
            step_id=step_id,
            code=code,
            resolved_inputs_path=resolved_inputs_path,
        )

    monkeypatch.setattr(CodeRunner, "run", recording_run)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        enable_deterministic_runner_repair=False,
        max_code_repair_attempts=3,
        runner_kind="subprocess",
    )
    result = pipeline.run(
        question="Summarize the ICU cohort.",
        cohort=pd.DataFrame({"stay_id": [1, 2, 3], "death": [0, 1, 0]}),
        cohort_name="noop_runtime_retry_test",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )

    record = _latest_step_record(Path(result.workdir))
    assert record["step_llm_repair_attempts"] == 2
    assert _call_count(llm, "REPAIR THE PYTHON CODE") == 4
    assert sum("NOOP_RUNTIME_FAILURE" in code for code in executed_code) == 1
    assert record["status"] == "repair_failed"


def test_exact_capsule_resume_skips_generation_audit_and_execution_but_reruns_gates(
    ra, tmp_path: Path, monkeypatch
) -> None:
    _isolate_article_suite_contract(monkeypatch)
    from easyicu.research_agent.execution import phase as pipeline_execute
    from easyicu.research_agent.agents.core import RuntimeSupervisor
    from easyicu.research_agent.contracts.runtime import RunResult
    from easyicu.research_agent.schema import ValidationFinding
    from easyicu.research_agent.authority.step_capsule import (
        StepAuthorityCapsuleRef,
        load_verified_step_authority_capsule,
    )

    runner_calls: list[str] = []

    class RecordingRunner:
        network_policy = "none"
        authority_identity_sha256 = "1" * 64

        def __init__(self, *, workdir: Path) -> None:
            self.workdir = Path(workdir)

        @staticmethod
        def validate_runtime_capabilities() -> tuple[str, ...]:
            return ("pandas",)

        def run(self, *, step_id, code, resolved_inputs_path=None):
            del resolved_inputs_path
            runner_calls.append(step_id)
            step_dir = self.workdir / "steps" / step_id
            out_dir = step_dir / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            script_path = step_dir / "analysis.py"
            script_path.write_text(code, encoding="utf-8")
            summary = {
                "phase": "CAPSULE_RESUME",
                "n": 3,
                "output_files": {
                    "table:cohort_summary": "cohort_summary.csv",
                },
            }
            table_path = out_dir / "cohort_summary.csv"
            pd.DataFrame([summary]).to_csv(table_path, index=False)
            summary_path = out_dir / "step_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            log_path = step_dir / "run.log"
            log_path.write_text("controlled runner\n", encoding="utf-8")
            return RunResult(
                step_id=step_id,
                script_path=script_path,
                cwd=step_dir,
                out_dir=out_dir,
                stdout="",
                stderr="",
                returncode=0,
                duration_seconds=0.01,
                artefacts=[table_path, summary_path],
                requested_network_policy="none",
                effective_isolation="controlled_test",
                runtime_provenance={"runner": "controlled_test"},
                runner_log_path=log_path,
            )

    def runner_factory(*, workdir, **_kwargs):
        return RecordingRunner(workdir=Path(workdir))

    real_gate = pipeline_execute._evaluate_final_deterministic_gates
    gate_calls: list[str] = []
    gate_behavior = {"mode": "error"}

    def forced_final_gate(**kwargs):
        gate_calls.append(kwargs["step"].step_id)
        if gate_behavior["mode"] == "interrupt":
            raise KeyboardInterrupt("simulated crash during deterministic review")
        base = real_gate(**kwargs)
        if gate_behavior["mode"] == "pass":
            return base
        forced = pipeline_execute._bind_findings_to_step_attempt(
            [
                ValidationFinding(
                    validator="capsule_resume_test_gate",
                    severity="error",
                    message="Force one durable post-audit deterministic failure.",
                    detail={"step_id": kwargs["step"].step_id},
                )
            ],
            step_id=kwargs["step"].step_id,
            attempt_id=kwargs["attempt_id"],
            checkpoint_id=kwargs["checkpoint_id"],
        )
        return replace(base, stat_findings=(*base.stat_findings, *forced))

    monkeypatch.setattr(
        pipeline_execute,
        "_evaluate_final_deterministic_gates",
        forced_final_gate,
    )
    monkeypatch.setattr(
        RuntimeSupervisor,
        "critique_step",
        lambda self, *, state, **_kwargs: state,
    )
    llm = PatternScriptedMockLLMClient(
        [
            ("ICU-AWARE RESEARCH PLAN", [_plan()] * 16),
            (
                "WRITE THE PYTHON CODE",
                [_summary_script(phase="CAPSULE_RESUME")] * 16,
            ),
            (
                "CONSERVATIVE ICU CONCEPT-USE AUDITOR",
                [json.dumps({"findings": []})] * 16,
            ),
            ("INTERPRET THE RESULTS", ["The cohort summary is available."] * 16),
        ]
    )

    def capsule_auditor(endpoint: str) -> ScriptedMockLLMClient:
        client = ScriptedMockLLMClient(
            [json.dumps({"findings": []})] * 16,
            repeat_last=True,
        )
        client.name = "capsule-auditor"
        client._model = "gpt-5.6-luna"
        client._extra_body = {"reasoning_effort": "high"}
        client._resolved_base_url = endpoint
        return client

    auditor_a = capsule_auditor("http://127.0.0.1:8787/v1")
    auditor_b = capsule_auditor("http://127.0.0.1:8317/v1")

    def build_pipeline(auditor=auditor_a):
        return ra.ResearchAgentPipeline(
            workdir=tmp_path,
            llm=llm,
            llm_concept_auditor_client=auditor,
            runner_factory=runner_factory,
            enable_literature=False,
            enable_visual_qa=False,
            enable_latex=False,
            enable_llm_concept_audit=True,
            enable_deterministic_code_fallback=False,
            enable_deterministic_runner_repair=False,
            max_code_repair_attempts=0,
        )

    run_kwargs = {
        "question": "Summarize the ICU cohort.",
        "cohort": pd.DataFrame({"stay_id": [1, 2, 3], "death": [0, 1, 0]}),
        "cohort_name": "capsule_resume_test",
        "database": "synthetic",
        "target_outcome": "death",
        "stop_after_step_id": "01_summary",
        "stop_after_analysis": True,
    }
    first = build_pipeline().run(**run_kwargs)
    run_dir = Path(first.workdir)
    first_record = _latest_step_record(run_dir)
    first_ref = StepAuthorityCapsuleRef.model_validate(
        first_record["step_authority_capsule_ref"]
    )
    first_capsule = load_verified_step_authority_capsule(run_dir, ref=first_ref)

    assert first_record["status"] == "contract_failed"
    assert first_capsule.capsule.stage == "executed_concept_audited"
    assert first_capsule.capsule.execution is not None
    assert first_capsule.capsule.concept_audit is not None
    assert (
        first_record["step_provider_call_receipt_version"]
        == PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION
    )
    assert first_record["step_provider_call_receipt"].endswith(".json")
    assert _call_count(llm, "WRITE THE PYTHON CODE") == 1
    assert _call_count(llm, "CONSERVATIVE ICU CONCEPT-USE AUDITOR") == 0
    assert len(auditor_a.calls) == 1
    assert runner_calls == ["01_summary"]
    assert gate_calls == ["01_summary"]

    second = build_pipeline().run(
        **run_kwargs,
        resume_run_id=first.run_id,
        resume_from_step_id="01_summary",
    )
    second_record = _latest_step_record(Path(second.workdir))

    assert second.run_id == first.run_id
    assert second_record["status"] == "contract_failed"
    assert second_record["step_authority_capsule_reused"] is True
    assert second_record["capsule_execution_replayed"] is True
    assert second_record["capsule_concept_audit_replayed"] is True
    assert _call_count(llm, "WRITE THE PYTHON CODE") == 1
    assert len(auditor_a.calls) == 1
    assert runner_calls == ["01_summary"]
    assert gate_calls == ["01_summary", "01_summary"]

    third = build_pipeline(auditor_b).run(
        **run_kwargs,
        resume_run_id=first.run_id,
        resume_from_step_id="01_summary",
    )
    third_record = _latest_step_record(Path(third.workdir))
    third_ref = StepAuthorityCapsuleRef.model_validate(
        third_record["step_authority_capsule_ref"]
    )
    third_capsule = load_verified_step_authority_capsule(run_dir, ref=third_ref)

    assert third_record["status"] == "contract_failed"
    assert third_record["step_authority_audit_cache_miss"] == "audit_identity_drift"
    assert third_record["capsule_execution_replayed"] is True
    assert third_record.get("capsule_concept_audit_replayed") is not True
    assert len(auditor_b.calls) == 1
    assert _call_count(llm, "WRITE THE PYTHON CODE") == 1
    assert runner_calls == ["01_summary"]
    assert third_capsule.capsule.execution == first_capsule.capsule.execution
    assert third_capsule.capsule.concept_audit is not None
    assert (
        third_capsule.capsule.concept_audit.auditor_identity_sha256
        != first_capsule.capsule.concept_audit.auditor_identity_sha256
    )

    fourth = build_pipeline(auditor_b).run(
        **run_kwargs,
        resume_run_id=first.run_id,
        resume_from_step_id="01_summary",
    )
    fourth_record = _latest_step_record(Path(fourth.workdir))
    assert fourth_record["capsule_execution_replayed"] is True
    assert fourth_record["capsule_concept_audit_replayed"] is True
    assert len(auditor_b.calls) == 1
    assert _call_count(llm, "WRITE THE PYTHON CODE") == 1
    assert runner_calls == ["01_summary"]
    assert gate_calls == ["01_summary"] * 4

    monkeypatch.setattr(
        pipeline_execute,
        "engine_code_sha256",
        lambda: "f" * 64,
    )
    fifth = build_pipeline(auditor_b).run(
        **run_kwargs,
        resume_run_id=first.run_id,
        resume_from_step_id="01_summary",
    )
    fifth_record = _latest_step_record(Path(fifth.workdir))
    fifth_capsule = load_verified_step_authority_capsule(
        run_dir,
        ref=StepAuthorityCapsuleRef.model_validate(
            fifth_record["step_authority_capsule_ref"]
        ),
    )
    assert fifth_record["step_authority_capsule_cache_miss"] == (
        "control_plane_drift_revalidation"
    )
    assert fifth_record["capsule_concept_audit_replayed"] is True
    assert _call_count(llm, "WRITE THE PYTHON CODE") == 1
    assert len(auditor_b.calls) == 1
    assert runner_calls == ["01_summary", "01_summary"]
    assert fifth_capsule.capsule.candidate_origin.kind == "legacy_adoption"
    assert (
        fifth_capsule.capsule.candidate_origin.adopted_from_capsule_sha256 is not None
    )

    sixth = build_pipeline(auditor_b).run(
        **run_kwargs,
        resume_run_id=first.run_id,
        resume_from_step_id="01_summary",
    )
    sixth_record = _latest_step_record(Path(sixth.workdir))
    assert sixth_record["capsule_execution_replayed"] is True
    assert sixth_record["capsule_concept_audit_replayed"] is True
    assert _call_count(llm, "WRITE THE PYTHON CODE") == 1
    assert len(auditor_b.calls) == 1
    assert runner_calls == ["01_summary", "01_summary"]
    assert gate_calls == ["01_summary"] * 6

    gate_behavior["mode"] = "pass"
    success_kwargs = {
        **run_kwargs,
        "cohort_name": "capsule_revalidation_crash_test",
    }
    successful = build_pipeline(auditor_b).run(**success_kwargs)
    successful_dir = Path(successful.workdir)
    assert _latest_step_record(successful_dir)["status"] == "ok"

    gate_behavior["mode"] = "interrupt"
    with pytest.raises(KeyboardInterrupt, match="simulated crash"):
        build_pipeline(auditor_b).run(
            **success_kwargs,
            resume_run_id=successful.run_id,
            resume_from_step_id="01_summary",
        )
    interrupted_record = _latest_step_record(successful_dir)
    assert interrupted_record["status"] in {
        "capsule_revalidation_pending",
        "executed_pending_review",
    }
    assert interrupted_record["status"] != "ok"
    from easyicu.research_agent.authority.runtime_artifacts import (
        current_successful_step_ids,
    )

    partial = json.loads(
        (successful_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    assert "01_summary" not in current_successful_step_ids(partial["per_step_records"])

    # A current validator may invalidate a previously successful sealed
    # candidate.  Resume may reuse those exact candidate bytes as a cache, but
    # must not trust the prior approval: it has to append an invalid checkpoint
    # and run the current gate again without buying another generation.
    from easyicu.research_agent.gates import concept as concept_gates

    gate_behavior["mode"] = "pass"
    invalid_candidate_kwargs = {
        **run_kwargs,
        "cohort_name": "validator_invalid_candidate_resume_test",
    }
    valid_before_drift = build_pipeline(auditor_b).run(**invalid_candidate_kwargs)
    valid_before_drift_dir = Path(valid_before_drift.workdir)
    assert _latest_step_record(valid_before_drift_dir)["status"] == "ok"
    generation_before_validator_drift = _call_count(llm, "WRITE THE PYTHON CODE")
    gate_calls_before_validator_drift = len(gate_calls)
    monkeypatch.setattr(
        concept_gates,
        "engine_code_sha256",
        lambda: "d" * 64,
    )
    gate_behavior["mode"] = "error"

    invalid_after_drift = build_pipeline(auditor_b).run(
        **invalid_candidate_kwargs,
        resume_run_id=valid_before_drift.run_id,
        resume_from_step_id="01_summary",
    )
    invalid_after_drift_dir = Path(invalid_after_drift.workdir)
    invalid_after_drift_record = _latest_step_record(invalid_after_drift_dir)
    invalid_manifest = json.loads(
        (invalid_after_drift_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    invalid_history = invalid_manifest["step_attempt_history"]

    assert any(
        record.get("step_id") == "01_summary"
        and record.get("status") == "resume_validator_invalid"
        and record.get("resume_revalidation_candidate_capsule_ref")
        and record.get("resume_revalidation_candidate_code_sha256")
        for record in invalid_history
    )
    assert invalid_after_drift_record["status"] == "contract_failed"
    assert (
        invalid_after_drift_record["resume_validator_invalid_candidate_reused"] is True
    )
    assert invalid_after_drift_record["step_authority_capsule_cache_miss"] == (
        "control_plane_drift_revalidation"
    )
    assert (
        _call_count(llm, "WRITE THE PYTHON CODE") == generation_before_validator_drift
    )
    assert len(gate_calls) >= gate_calls_before_validator_drift + 2

    from easyicu.research_agent.agents import agentic_coder

    agentic_calls = {"n": 0}

    def delegated_script(self, context, step):
        del self, context, step
        agentic_calls["n"] += 1
        return _summary_script(phase="AGENTIC_CAPSULE")

    monkeypatch.setenv("EASYICU_AGENTIC_CODER_BACKEND", "codex")
    monkeypatch.setattr(agentic_coder, "cli_backend_available", lambda backend: True)
    monkeypatch.setattr(
        agentic_coder.AgenticCoderAgent,
        "_delegate",
        delegated_script,
    )
    gate_behavior["mode"] = "error"
    generation_before_agentic = _call_count(llm, "WRITE THE PYTHON CODE")
    agentic_kwargs = {
        **run_kwargs,
        "cohort_name": "agentic_capsule_resume_test",
    }
    agentic_first = build_pipeline(auditor_b).run(**agentic_kwargs)
    agentic_first_record = _latest_step_record(Path(agentic_first.workdir))
    agentic_first_capsule = load_verified_step_authority_capsule(
        Path(agentic_first.workdir),
        ref=StepAuthorityCapsuleRef.model_validate(
            agentic_first_record["step_authority_capsule_ref"]
        ),
    )
    assert agentic_calls["n"] == 0
    assert _call_count(llm, "WRITE THE PYTHON CODE") == generation_before_agentic + 1
    assert agentic_first_record["step_authority_initial_transport"] == (
        "fallback_provider_receipt"
    )
    assert agentic_first_capsule.capsule.candidate_origin.kind == ("initial_generation")

    agentic_second = build_pipeline(auditor_b).run(
        **agentic_kwargs,
        resume_run_id=agentic_first.run_id,
        resume_from_step_id="01_summary",
    )
    agentic_second_record = _latest_step_record(Path(agentic_second.workdir))
    assert agentic_second_record["step_authority_capsule_reused"] is True
    assert agentic_calls["n"] == 0
    assert _call_count(llm, "WRITE THE PYTHON CODE") == generation_before_agentic + 1


def test_resume_seals_completed_repair_after_capsule_checkpoint_crash(
    ra, tmp_path: Path, monkeypatch
) -> None:
    _isolate_article_suite_contract(monkeypatch)
    from easyicu.research_agent.execution import phase as pipeline_execute
    from easyicu.research_agent.agents.core import RuntimeSupervisor
    from easyicu.research_agent.audits.validators import PrimaryModelContractValidator
    from easyicu.research_agent.contracts.runtime import RunResult
    from easyicu.research_agent.schema import ValidationFinding
    from easyicu.research_agent.authority.step_runtime import (
        StepAuthorityRuntimeError,
    )

    initial_code = _summary_script(phase="CAPSULE_CRASH_INITIAL")
    repaired_code = _summary_script(phase="CAPSULE_CRASH_REPAIRED")

    runner_phases: list[str] = []

    class RepairCrashRunner:
        network_policy = "none"
        authority_identity_sha256 = "2" * 64

        def __init__(self, *, workdir: Path) -> None:
            self.workdir = Path(workdir)

        @staticmethod
        def validate_runtime_capabilities() -> tuple[str, ...]:
            return ("pandas",)

        def run(self, *, step_id, code, resolved_inputs_path=None):
            del resolved_inputs_path
            phase = (
                "CAPSULE_CRASH_REPAIRED"
                if "CAPSULE_CRASH_REPAIRED" in code
                else "CAPSULE_CRASH_INITIAL"
            )
            runner_phases.append(phase)
            step_dir = self.workdir / "steps" / step_id
            out_dir = step_dir / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            script_path = step_dir / "analysis.py"
            script_path.write_text(code, encoding="utf-8")
            summary = {
                "phase": phase,
                "n": 3,
                "output_files": {
                    "table:cohort_summary": "cohort_summary.csv",
                },
            }
            table_path = out_dir / "cohort_summary.csv"
            pd.DataFrame([summary]).to_csv(table_path, index=False)
            summary_path = out_dir / "step_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            log_path = step_dir / "run.log"
            log_path.write_text("controlled runner\n", encoding="utf-8")
            return RunResult(
                step_id=step_id,
                script_path=script_path,
                cwd=step_dir,
                out_dir=out_dir,
                stdout="",
                stderr="",
                returncode=0,
                duration_seconds=0.01,
                artefacts=[table_path, summary_path],
                requested_network_policy="none",
                effective_isolation="controlled_test",
                runtime_provenance={"runner": "controlled_test"},
                runner_log_path=log_path,
            )

    def contract_audit(self, *, step, step_summary, **kwargs):
        del self, kwargs
        if step_summary.get("phase") != "CAPSULE_CRASH_INITIAL":
            return []
        return [
            ValidationFinding(
                validator="primary_model_contract",
                severity="error",
                message="Force one repair before simulating the seal crash.",
                detail={"step_id": step.step_id},
            )
        ]

    real_seal = pipeline_execute.seal_repair_candidate_from_receipt
    seal_calls = 0

    def crash_once(*args, **kwargs):
        nonlocal seal_calls
        seal_calls += 1
        if seal_calls == 1:
            raise StepAuthorityRuntimeError("simulated post-receipt seal crash")
        return real_seal(*args, **kwargs)

    monkeypatch.setattr(PrimaryModelContractValidator, "audit", contract_audit)
    monkeypatch.setattr(
        pipeline_execute,
        "seal_repair_candidate_from_receipt",
        crash_once,
    )
    monkeypatch.setattr(
        RuntimeSupervisor,
        "critique_step",
        lambda self, *, state, **_kwargs: state,
    )

    def runner_factory(*, workdir, **_kwargs):
        return RepairCrashRunner(workdir=Path(workdir))

    llm = PatternScriptedMockLLMClient(
        [
            ("ICU-AWARE RESEARCH PLAN", [_plan()]),
            ("WRITE THE PYTHON CODE", [initial_code]),
            ("REPAIR THE PYTHON CODE", [repaired_code] * 2),
            (
                "INTERPRET THE RESULTS",
                ["The repaired cohort summary is available."],
            ),
        ]
    )

    def build_pipeline():
        return ra.ResearchAgentPipeline(
            workdir=tmp_path,
            llm=llm,
            runner_factory=runner_factory,
            enable_literature=False,
            enable_visual_qa=False,
            enable_latex=False,
            enable_llm_concept_audit=False,
            enable_deterministic_code_fallback=False,
            enable_deterministic_runner_repair=False,
            max_code_repair_attempts=1,
        )

    run_kwargs = {
        "question": "Summarize the ICU cohort.",
        "cohort": pd.DataFrame({"stay_id": [1, 2, 3], "death": [0, 1, 0]}),
        "cohort_name": "capsule_repair_crash_test",
        "database": "synthetic",
        "target_outcome": "death",
        "stop_after_step_id": "01_summary",
        "stop_after_analysis": True,
    }
    with pytest.raises(StepAuthorityRuntimeError, match="simulated post-receipt"):
        build_pipeline().run(**run_kwargs)
    run_dir = next(tmp_path.glob("run_*"))
    first_record = _latest_step_record(run_dir)
    assert first_record["status"] == "repair_transport_pending"
    assert first_record["capsule_pending_repair_attempt_id"] == 1
    assert _call_count(llm, "WRITE THE PYTHON CODE") == 1
    # This fixture intentionally returns a complete script to the patch route,
    # so one logical repair uses patch + authorized full-rewrite calls.
    assert _call_count(llm, "REPAIR THE PYTHON CODE") == 2
    repair_calls_before_resume = _call_count(llm, "REPAIR THE PYTHON CODE")
    assert runner_phases == ["CAPSULE_CRASH_INITIAL"]

    monkeypatch.setattr(
        pipeline_execute,
        "engine_code_sha256",
        lambda: "e" * 64,
    )
    second = build_pipeline().run(
        **run_kwargs,
        resume_run_id=run_dir.name,
        resume_from_step_id="01_summary",
    )
    second_record = _latest_step_record(Path(second.workdir))
    assert second_record["status"] == "ok"
    assert second_record["step_authority_capsule_cache_miss"] == (
        "control_plane_drift_revalidation"
    )
    assert _call_count(llm, "WRITE THE PYTHON CODE") == 1
    assert _call_count(llm, "REPAIR THE PYTHON CODE") == repair_calls_before_resume
    assert runner_phases == [
        "CAPSULE_CRASH_INITIAL",
        "CAPSULE_CRASH_REPAIRED",
    ]
    assert seal_calls == 2
