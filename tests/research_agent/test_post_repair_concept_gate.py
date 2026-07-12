from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


_SAFE_CODE = """
import json
import os
import pandas as pd

df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
summary = {"n": int(len(df)), "phase": "initial"}
pd.DataFrame([summary]).to_csv(os.path.join(out, "cohort_summary.csv"), index=False)
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
"""

_UNSAFE_REPAIR_CODE = """
import json
import os
import pandas as pd

# UNSAFE_POST_REPAIR
df = pd.read_parquet(os.environ["COHORT_PARQUET"])
df["value"] = df["value"].fillna(0)
out = os.environ["STEP_OUT_DIR"]
summary = {"n": int(len(df)), "phase": "repaired"}
pd.DataFrame([summary]).to_csv(os.path.join(out, "cohort_summary.csv"), index=False)
with open(os.path.join(out, "unsafe_executed.txt"), "w", encoding="utf-8") as f:
    f.write("unsafe repair reached runner")
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
"""


_SELF_MUTATING_CODE = """
import json
import os
from pathlib import Path
import pandas as pd

df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
summary = {"n": int(len(df)), "phase": "initial"}
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
script = Path(__file__)
script.write_text(script.read_text(encoding="utf-8") + "\\n# SELF_MUTATED\\n", encoding="utf-8")
"""


class _RepairGateLLM:
    name = "post-repair-concept-gate-llm"

    def __init__(self, *, interrupt_repair: bool = False) -> None:
        self.interrupt_repair = interrupt_repair
        self.write_calls = 0
        self.repair_calls = 0

    def complete(self, messages, *, max_tokens=2048, temperature=0.2):
        del max_tokens, temperature
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        upper = user.upper()
        if "ICU-AWARE RESEARCH PLAN" in upper:
            return json.dumps(
                {
                    "research_question": "Summarize the cohort.",
                    "steps": [
                        {
                            "step_id": "01_summary",
                            "intent": "Produce a descriptive cohort summary.",
                            "inputs": ["stay_id", "value"],
                            "expected_outputs": ["table:cohort_summary"],
                            "method": "descriptive_summary",
                            "icu_rule_refs": [],
                        }
                    ],
                    "rationale": "post-repair concept-gate regression",
                }
            )
        if "REPAIR THE PYTHON CODE" in upper:
            self.repair_calls += 1
            if self.interrupt_repair:
                raise KeyboardInterrupt("simulated operator interruption")
            return _UNSAFE_REPAIR_CODE
        if "WRITE THE PYTHON CODE" in upper:
            self.write_calls += 1
            return _SAFE_CODE
        if "INTERPRET THE RESULTS" in upper:
            return "Summary {evidence:cohort_summary}."
        if "MANUSCRIPT SCAFFOLD" in upper:
            return "# Title\n\n## Results\n\nSummary {evidence:cohort_summary}."
        return "{}"


class _SelfMutatingLLM(_RepairGateLLM):
    def complete(self, messages, *, max_tokens=2048, temperature=0.2):
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        if "WRITE THE PYTHON CODE" in user.upper():
            self.write_calls += 1
            return _SELF_MUTATING_CODE
        return super().complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )


def _pipeline(ra, tmp_path: Path, llm: _RepairGateLLM):
    return ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
        max_code_repair_attempts=1,
    )


def _run(pipeline, cohort: pd.DataFrame):
    return pipeline.run(
        question="Summarize the cohort.",
        cohort=cohort,
        cohort_name="post_repair_gate_test",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )


def test_contract_repair_reenters_concept_gate_before_runner(
    ra, tmp_path: Path, monkeypatch
) -> None:
    from easyicu.research_agent.audits.validators import (
        ConceptUsageAuditor,
        PrimaryModelContractValidator,
    )
    from easyicu.research_agent.contracts import ValidationFinding

    audited_scripts: list[str] = []

    def concept_audit(self, *, context, script_text, step):
        del self, context
        audited_scripts.append(script_text)
        if "UNSAFE_POST_REPAIR" not in script_text:
            return []
        return [
            ValidationFinding(
                validator="concept_usage_auditor",
                severity="error",
                message="Post-execution repair introduced an unsafe concept transform.",
                detail={"step_id": step.step_id},
            )
        ]

    def contract_audit(self, *, step, step_summary, **kwargs):
        del self, kwargs
        if step_summary.get("phase") != "initial":
            return []
        return [
            ValidationFinding(
                validator="primary_model_contract",
                severity="error",
                message="Force one contract repair for the orchestration test.",
                detail={"step_id": step.step_id},
            )
        ]

    monkeypatch.setattr(ConceptUsageAuditor, "audit", concept_audit)
    monkeypatch.setattr(PrimaryModelContractValidator, "audit", contract_audit)

    llm = _RepairGateLLM()
    result = _run(
        _pipeline(ra, tmp_path, llm),
        pd.DataFrame(
            {"stay_id": [1, 2, 3], "value": [1.0, None, 3.0], "death": [0, 1, 0]}
        ),
    )
    run_dir = Path(result.workdir)
    partial = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    record = next(
        item for item in partial["per_step_records"] if item["step_id"] == "01_summary"
    )

    assert llm.repair_calls == 1
    assert any("UNSAFE_POST_REPAIR" in script for script in audited_scripts)
    assert record["status"] == "blocked_by_concept_audit"
    assert not (
        run_dir / "steps" / "01_summary" / "outputs" / "unsafe_executed.txt"
    ).exists()
    assert (run_dir / "steps" / "01_summary" / ".quarantine").is_dir()


def test_executed_script_digest_mismatch_blocks_outputs_before_evidence(
    ra, tmp_path: Path
) -> None:
    llm = _SelfMutatingLLM()
    result = _run(
        _pipeline(ra, tmp_path, llm),
        pd.DataFrame(
            {"stay_id": [1, 2, 3], "value": [1.0, 2.0, 3.0], "death": [0, 1, 0]}
        ),
    )
    run_dir = Path(result.workdir)
    partial = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    record = next(
        item for item in partial["per_step_records"] if item["step_id"] == "01_summary"
    )

    assert record["status"] == "blocked_script_integrity"
    assert record["concept_approved_code_sha256"] != record["executed_code_sha256"]
    assert list((run_dir / "steps" / "01_summary" / "outputs").iterdir()) == []
    assert any(
        finding["validator"] == "post_repair_concept_gate"
        for finding in partial["findings"]
    )


def test_keyboard_interrupt_during_concept_repair_saves_draft_and_reraises(
    ra, tmp_path: Path, monkeypatch
) -> None:
    from easyicu.research_agent.audits.validators import ConceptUsageAuditor
    from easyicu.research_agent.contracts import ValidationFinding
    from easyicu.research_agent.pipeline_resume import load_quarantined_concept_draft

    def reject_draft(self, *, context, script_text, step):
        del self, context, script_text
        return [
            ValidationFinding(
                validator="concept_usage_auditor",
                severity="error",
                message="Draft requires repair before execution.",
                detail={"step_id": step.step_id},
            )
        ]

    monkeypatch.setattr(ConceptUsageAuditor, "audit", reject_draft)
    llm = _RepairGateLLM(interrupt_repair=True)

    with pytest.raises(KeyboardInterrupt, match="operator interruption"):
        _run(
            _pipeline(ra, tmp_path, llm),
            pd.DataFrame(
                {
                    "stay_id": [1, 2, 3],
                    "value": [1.0, None, 3.0],
                    "death": [0, 1, 0],
                }
            ),
        )

    run_dir = next(tmp_path.glob("run_*"))
    checkpoint = load_quarantined_concept_draft(
        run_dir=run_dir,
        step_id="01_summary",
    )
    assert llm.repair_calls == 1
    assert checkpoint is not None
    assert checkpoint.code.strip() == _SAFE_CODE.strip()
    assert checkpoint.findings[0]["message"] == "Draft requires repair before execution."
    assert not (run_dir / "steps" / "01_summary" / "analysis.py").exists()
