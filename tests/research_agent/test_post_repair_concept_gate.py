from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.repairs.patch import PATCH_FORMAT

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

_INITIAL_CONCEPT_ERROR_CODE = _SAFE_CODE + "\n# INITIAL_CONCEPT_ERROR\n"
_LATER_CONTRACT_ERROR_CODE = (
    _SAFE_CODE.replace('"phase": "initial"', '"phase": "repaired"')
    + "\n# LATER_CONTRACT_ERROR\n"
)

_INVALID_HELPER_REPAIR_CODE = """
import json
import os
import pandas as pd

def summarize(frame):
    return {"n": int(len(frame)), "phase": "repaired"}

df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
summary = summarize(df, unexpected=True)
pd.DataFrame([summary]).to_csv(os.path.join(out, "cohort_summary.csv"), index=False)
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
"""

_RECOVERED_REPAIR_CODE = _SAFE_CODE.replace(
    'summary = {"n": int(len(df)), "phase": "initial"}',
    'summary = {"n": int(len(df)), "phase": "repaired", "output_files": '
    '[{"kind": "table", "name": "cohort_summary", '
    '"path": "cohort_summary.csv"}]}',
)

_UNSAFE_REPAIR_CODE_AGAIN = _UNSAFE_REPAIR_CODE.replace(
    '"phase": "repaired"',
    '"phase": "repaired_again"',
)


def _script_patch(old: str, new: str) -> str:
    return json.dumps(
        {
            "format": PATCH_FORMAT,
            "edits": [
                {
                    "old": old.strip(),
                    "new": new.strip(),
                    "expected_count": 1,
                }
            ],
        }
    )


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
            if self.repair_calls == 1:
                return _script_patch(_SAFE_CODE, _UNSAFE_REPAIR_CODE)
            return _script_patch(
                _UNSAFE_REPAIR_CODE,
                _UNSAFE_REPAIR_CODE_AGAIN,
            )
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


class _SequentialRepairLLM(_RepairGateLLM):
    def complete(self, messages, *, max_tokens=2048, temperature=0.2):
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        upper = user.upper()
        if "WRITE THE PYTHON CODE" in upper:
            self.write_calls += 1
            return _INITIAL_CONCEPT_ERROR_CODE
        if "REPAIR THE PYTHON CODE" in upper:
            self.repair_calls += 1
            if self.repair_calls == 1:
                return _script_patch(_INITIAL_CONCEPT_ERROR_CODE, _SAFE_CODE)
            return _script_patch(_SAFE_CODE, _LATER_CONTRACT_ERROR_CODE)
        return super().complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )


class _MechanicalRecoveryLLM(_RepairGateLLM):
    def complete(self, messages, *, max_tokens=2048, temperature=0.2):
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        upper = user.upper()
        if "WRITE THE PYTHON CODE" in upper:
            self.write_calls += 1
            return _SAFE_CODE
        if "REPAIR THE PYTHON CODE" in upper:
            self.repair_calls += 1
            if self.repair_calls == 1:
                return _script_patch(_SAFE_CODE, _INVALID_HELPER_REPAIR_CODE)
            return _script_patch(
                _INVALID_HELPER_REPAIR_CODE,
                _RECOVERED_REPAIR_CODE,
            )
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
    from easyicu.research_agent.contracts.runtime import ValidationFinding

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

    assert llm.repair_calls == 2
    assert any("UNSAFE_POST_REPAIR" in script for script in audited_scripts)
    assert record["status"] == "blocked_by_concept_audit"
    assert record["step_llm_repair_attempts"] == 2
    assert record["step_llm_repair_classes"] == [
        "contract",
        "post_mutation_concept",
    ]
    assert record["step_provider_call_categories"] == [
        "initial_generation",
        "contract_repair_patch",
        "post_mutation_concept_repair_patch",
    ]
    assert not (
        run_dir / "steps" / "01_summary" / "outputs" / "unsafe_executed.txt"
    ).exists()
    assert (run_dir / "steps" / "01_summary" / ".quarantine").is_dir()


def test_contract_repair_mechanical_error_uses_remaining_step_budget(
    ra, tmp_path: Path, monkeypatch
) -> None:
    from easyicu.research_agent.audits.validators import PrimaryModelContractValidator
    from easyicu.research_agent.contracts.runtime import ValidationFinding

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

    monkeypatch.setattr(PrimaryModelContractValidator, "audit", contract_audit)
    llm = _MechanicalRecoveryLLM()
    result = _run(
        _pipeline(ra, tmp_path, llm),
        pd.DataFrame(
            {"stay_id": [1, 2, 3], "value": [1.0, 2.0, 3.0], "death": [0, 1, 0]}
        ),
    )
    partial = json.loads(
        (Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8")
    )
    record = next(
        item for item in partial["per_step_records"] if item["step_id"] == "01_summary"
    )

    assert llm.repair_calls == 2
    assert record["status"] == "ok"
    assert record["step_llm_repair_attempts"] == 2
    assert record["step_llm_repair_classes"] == [
        "contract",
        "post_mutation_concept",
    ]
    assert record["step_provider_call_categories"] == [
        "initial_generation",
        "contract_repair_patch",
        "post_mutation_concept_repair_patch",
        "analyzer",
    ]


def test_quarantine_persists_repaired_constraints_across_later_repairs(
    ra, tmp_path: Path, monkeypatch
) -> None:
    from easyicu.research_agent.audits.validators import (
        ConceptUsageAuditor,
        PrimaryModelContractValidator,
    )
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.orchestration.resume import load_quarantined_concept_draft

    def concept_audit(self, *, context, script_text, step):
        del self, context
        findings = []
        if "INITIAL_CONCEPT_ERROR" in script_text:
            findings.append(
                ValidationFinding(
                    validator="concept_usage_auditor",
                    severity="error",
                    message="Earlier repaired constraint must remain binding.",
                    detail={"step_id": step.step_id},
                )
            )
        if "LATER_CONTRACT_ERROR" in script_text:
            findings.append(
                ValidationFinding(
                    validator="concept_usage_auditor",
                    severity="error",
                    message="Later contract repair introduced a new constraint.",
                    detail={"step_id": step.step_id},
                )
            )
        return findings

    def contract_audit(self, *, step, step_summary, **kwargs):
        del self, kwargs
        if step_summary.get("phase") != "initial":
            return []
        return [
            ValidationFinding(
                validator="primary_model_contract",
                severity="error",
                message="Force a later contract repair.",
                detail={"step_id": step.step_id},
            )
        ]

    monkeypatch.setattr(ConceptUsageAuditor, "audit", concept_audit)
    monkeypatch.setattr(PrimaryModelContractValidator, "audit", contract_audit)

    llm = _SequentialRepairLLM()
    result = _run(
        _pipeline(ra, tmp_path, llm),
        pd.DataFrame(
            {"stay_id": [1, 2, 3], "value": [1.0, None, 3.0], "death": [0, 1, 0]}
        ),
    )
    checkpoint = load_quarantined_concept_draft(
        run_dir=Path(result.workdir),
        step_id="01_summary",
    )
    partial = json.loads(
        (Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8")
    )
    record = next(
        item for item in partial["per_step_records"] if item["step_id"] == "01_summary"
    )

    assert llm.repair_calls == 2
    assert checkpoint is not None
    expected_messages = [
        "Earlier repaired constraint must remain binding.",
        "Later contract repair introduced a new constraint.",
    ]
    assert [finding["message"] for finding in checkpoint.findings] == expected_messages
    assert [
        finding["message"] for finding in record["monotonic_concept_constraints"]
    ] == expected_messages
    assert record["step_llm_repair_attempts"] == 2
    assert record["step_llm_repair_classes"] == ["concept", "contract"]
    assert record["step_provider_call_categories"] == [
        "initial_generation",
        "concept_repair_patch",
        "contract_repair_patch",
    ]


def test_unfinished_step_record_restores_only_binding_concept_errors() -> None:
    from easyicu.research_agent.pipeline_execute import (
        _persisted_monotonic_concept_constraints,
    )

    error = {
        "validator": "concept_usage_auditor",
        "severity": "error",
        "message": "Keep this repaired constraint binding.",
        "detail": {"step_id": "01_summary"},
    }
    warning = {
        "validator": "concept_usage_auditor",
        "severity": "warning",
        "message": "Do not persist informational findings.",
        "detail": {"step_id": "01_summary"},
    }

    restored = _persisted_monotonic_concept_constraints(
        {
            "status": "contract_failed",
            "monotonic_concept_constraints": [error, warning, error],
        }
    )

    assert [finding.message for finding in restored] == [error["message"]]
    assert (
        _persisted_monotonic_concept_constraints(
            {"status": "ok", "monotonic_concept_constraints": [error]}
        )
        == []
    )


def test_monotonic_constraints_keep_distinct_locals_and_refresh_line_numbers() -> None:
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.pipeline_execute import (
        _merge_monotonic_concept_constraints,
    )

    def finding(name: str, branch_line: int, first_use_line: int):
        return ValidationFinding(
            validator="mechanical_code_preflight",
            severity="error",
            message="A local may be unbound after a continuing branch.",
            detail={
                "reason": "branch_local_unbound",
                "name": name,
                "branch_line": branch_line,
                "first_use_line": first_use_line,
            },
        )

    merged = _merge_monotonic_concept_constraints(
        [finding("coercion_audit", 588, 596)],
        [
            finding("coercion_audit", 633, 641),
            finding("provenance_audit", 643, 653),
            finding("source_status", 655, 670),
        ],
    )

    assert [item.detail["name"] for item in merged] == [
        "coercion_audit",
        "provenance_audit",
        "source_status",
    ]
    assert merged[0].detail["branch_line"] == 633
    assert merged[0].detail["first_use_line"] == 641


def test_monotonic_constraints_preserve_existing_warning_history() -> None:
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.pipeline_execute import (
        _merge_monotonic_concept_constraints,
    )

    warning = ValidationFinding(
        validator="audit_history",
        severity="warning",
        message="Existing nonblocking audit context.",
    )
    new_warning = warning.model_copy(update={"message": "New transient warning."})

    merged = _merge_monotonic_concept_constraints([warning], [new_warning])

    assert merged == [warning]


def test_monotonic_constraints_keep_same_local_from_distinct_scopes(ra) -> None:
    from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
    from easyicu.research_agent.pipeline_execute import (
        _merge_monotonic_concept_constraints,
    )

    code = """
def first():
    try:
        value = may_fail()
    except ValueError:
        pass
    return value

def second():
    try:
        value = may_fail()
    except ValueError:
        pass
    return value
"""
    step = ra.AnalysisStep(
        step_id="scope_check",
        intent="Exercise mechanical scope identity.",
        expected_outputs=["table:scope_check"],
        method="descriptive_summary",
    )
    findings = [
        finding
        for finding in audit_mechanical_code_contracts(code, step)
        if (finding.detail or {}).get("reason") == "branch_local_unbound"
    ]

    assert {(finding.detail or {}).get("scope") for finding in findings} == {
        "first",
        "second",
    }
    assert len(_merge_monotonic_concept_constraints([], findings)) == 2


def test_branch_local_occurrence_ids_distinguish_identical_sibling_tries(ra) -> None:
    from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
    from easyicu.research_agent.pipeline_execute import (
        _merge_monotonic_concept_constraints,
    )

    code = """
def analyze():
    try:
        result = compute()
    except ValueError:
        pass
    consume(result)
    try:
        result = compute()
    except ValueError:
        pass
    consume(result)
"""
    step = ra.AnalysisStep(
        step_id="sibling_scope_check",
        intent="Exercise mechanical occurrence identity.",
        expected_outputs=["table:scope_check"],
        method="descriptive_summary",
    )
    findings = [
        finding
        for finding in audit_mechanical_code_contracts(code, step)
        if (finding.detail or {}).get("reason") == "branch_local_unbound"
        and (finding.detail or {}).get("name") == "result"
        and "continuing try/except" in finding.message
    ]

    assert len(findings) == 2
    assert len({finding.detail["occurrence_id"] for finding in findings}) == 2
    assert len(_merge_monotonic_concept_constraints([], findings)) == 2


def test_branch_local_occurrence_id_survives_body_edit(ra) -> None:
    from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts

    step = ra.AnalysisStep(
        step_id="body_edit_scope_check",
        intent="Exercise mechanical occurrence identity.",
        expected_outputs=["table:scope_check"],
        method="descriptive_summary",
    )

    def occurrence(call: str) -> str:
        code = f"""
def analyze():
    try:
        result = {call}()
    except ValueError:
        pass
    consume(result)
"""
        finding = next(
            finding
            for finding in audit_mechanical_code_contracts(code, step)
            if (finding.detail or {}).get("reason") == "branch_local_unbound"
            and (finding.detail or {}).get("name") == "result"
            and "continuing try/except" in finding.message
        )
        return finding.detail["occurrence_id"]

    assert occurrence("primary_compute") == occurrence("alternate_compute")


def test_monotonic_constraint_identity_ignores_changing_audit_counts() -> None:
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.pipeline_execute import (
        _merge_monotonic_concept_constraints,
    )

    def finding(invalid_n: int):
        return ValidationFinding(
            validator="row_alignment",
            severity="error",
            message="The selected column failed row alignment.",
            detail={
                "reason": "row_alignment_unverified",
                "column": "selected_value",
                "invalid_n": invalid_n,
            },
        )

    merged = _merge_monotonic_concept_constraints([finding(3)], [finding(1)])

    assert len(merged) == 1
    assert merged[0].detail["invalid_n"] == 1


def test_monotonic_constraint_identity_unions_changing_evidence_support() -> None:
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.pipeline_execute import (
        _merge_monotonic_concept_constraints,
    )

    def finding(evidence_id: str):
        return ValidationFinding(
            validator="row_alignment",
            severity="error",
            message="The selected column failed row alignment.",
            detail={
                "reason": "row_alignment_unverified",
                "occurrence_id": "selected_value_alignment",
            },
            evidence_ids=[evidence_id],
        )

    merged = _merge_monotonic_concept_constraints(
        [finding("evidence-old")],
        [finding("evidence-new")],
    )

    assert len(merged) == 1
    assert merged[0].evidence_ids == ["evidence-old", "evidence-new"]


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
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.orchestration.resume import load_quarantined_concept_draft

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
    assert (
        checkpoint.findings[0]["message"] == "Draft requires repair before execution."
    )
    assert not (run_dir / "steps" / "01_summary" / "analysis.py").exists()
