from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _summary_script(*, phase: str) -> str:
    return f"""
import json
import os
import pandas as pd

out = os.environ["STEP_OUT_DIR"]
summary = {{
    "phase": {phase!r},
    "n": 3,
    "output_files": [
        {{"kind": "table", "name": "cohort_summary", "path": "cohort_summary.csv"}}
    ],
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


def test_llm_concept_audit_runs_once_only_after_local_contracts_pass(
    ra, tmp_path: Path, monkeypatch
) -> None:
    from easyicu.research_agent.audits.validators import PrimaryModelContractValidator
    from easyicu.research_agent.contracts import ValidationFinding
    from easyicu.research_agent.runner import CodeRunner

    initial_code = _summary_script(phase="INITIAL_CONTRACT_ERROR")
    repaired_code = _summary_script(phase="FINAL_CONTRACT_VALID")

    class DeferredAuditLLM:
        name = "deferred-audit-llm"

        def __init__(self) -> None:
            self.audit_prompts: list[str] = []

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            del max_tokens, temperature
            system = "\n".join(
                str(message.content or "")
                for message in messages
                if message.role == "system"
            )
            user = next(
                (
                    str(message.content or "")
                    for message in reversed(messages)
                    if message.role == "user"
                ),
                "",
            )
            upper = user.upper()
            if "ICU-AWARE RESEARCH PLAN" in upper:
                return _plan()
            if "WRITE THE PYTHON CODE" in upper:
                return initial_code
            if "REPAIR THE PYTHON CODE" in upper:
                return repaired_code
            if "CONSERVATIVE ICU CONCEPT-USE AUDITOR" in system.upper():
                self.audit_prompts.append(user)
                return json.dumps({"findings": []})
            if "INTERPRET THE RESULTS" in upper:
                return "The cohort summary is available."
            return "{}"

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
    llm = DeferredAuditLLM()
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
    assert len(llm.audit_prompts) == 1
    assert "FINAL_CONTRACT_VALID" in llm.audit_prompts[0]
    assert "INITIAL_CONTRACT_ERROR" not in llm.audit_prompts[0]
    assert record["status"] == "ok"
    assert record["llm_concept_audit_status"] == "completed"
    assert record["llm_concept_approved_code_sha256"] == record["executed_code_sha256"]
    assert (
        record["deterministic_contract_approved_code_sha256"]
        == record["executed_code_sha256"]
    )


def test_disabled_llm_audit_records_final_gate_without_claiming_llm_approval(
    ra, tmp_path: Path
) -> None:
    class AuditDisabledLLM:
        name = "audit-disabled-llm"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            del max_tokens, temperature
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            upper = user.upper()
            if "ICU-AWARE RESEARCH PLAN" in upper:
                return _plan()
            if "WRITE THE PYTHON CODE" in upper:
                return _summary_script(phase="AUDIT_DISABLED")
            if "INTERPRET THE RESULTS" in upper:
                return "The cohort summary is available."
            return "{}"

    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=AuditDisabledLLM(),
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
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


def test_runtime_repair_transport_retry_does_not_reexecute_known_failure(
    ra, tmp_path: Path, monkeypatch
) -> None:
    from easyicu.research_agent.runner import CodeRunner

    broken_code = "raise RuntimeError('KNOWN_RUNTIME_FAILURE')\n"

    class FailedRepairLLM:
        name = "failed-runtime-repair-llm"

        def __init__(self) -> None:
            self.repair_calls = 0

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            del max_tokens, temperature
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            upper = user.upper()
            if "ICU-AWARE RESEARCH PLAN" in upper:
                return _plan()
            if "WRITE THE PYTHON CODE" in upper:
                return broken_code
            if "REPAIR THE PYTHON CODE" in upper:
                self.repair_calls += 1
                raise RuntimeError("provider rate limited")
            if "INTERPRET THE RESULTS" in upper:
                return "The fallback summary is available."
            return "{}"

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
    llm = FailedRepairLLM()
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        enable_deterministic_runner_repair=False,
        max_code_repair_attempts=3,
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
    assert llm.repair_calls == 2
    assert sum("KNOWN_RUNTIME_FAILURE" in code for code in executed_code) == 1
    assert record["status"] == "repair_failed"


def test_noop_runtime_repair_is_retried_without_reexecution(
    ra, tmp_path: Path, monkeypatch
) -> None:
    from easyicu.research_agent.runner import CodeRunner

    broken_code = (
        _summary_script(phase="NOOP_RUNTIME_FAILURE")
        + "\nraise RuntimeError('NOOP_RUNTIME_FAILURE')\n"
    )

    class NoopRepairLLM:
        name = "noop-runtime-repair-llm"

        def __init__(self) -> None:
            self.repair_calls = 0

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            del max_tokens, temperature
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            upper = user.upper()
            if "ICU-AWARE RESEARCH PLAN" in upper:
                return _plan()
            if "WRITE THE PYTHON CODE" in upper:
                return broken_code
            if "REPAIR THE PYTHON CODE" in upper:
                self.repair_calls += 1
                return broken_code
            if "INTERPRET THE RESULTS" in upper:
                return "The fallback summary is available."
            return "{}"

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
    llm = NoopRepairLLM()
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        enable_deterministic_runner_repair=False,
        max_code_repair_attempts=3,
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
    assert llm.repair_calls == 2
    assert sum("NOOP_RUNTIME_FAILURE" in code for code in executed_code) == 1
    assert record["status"] == "repair_failed"
