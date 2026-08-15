from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from easyicu.research_agent.audits.validators import FigureContractQualityValidator
from easyicu.research_agent.execution.runners.audit_panel_executor import (
    audit_panel_executor_owns_step,
    run_audit_panel,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _step(**updates: object) -> AnalysisStep:
    payload: dict[str, object] = {
        "step_id": "04_audit_panel",
        "planned_analysis_role": "auxiliary",
        "intent": "Render the framework audit panel.",
        "inputs": [],
        "expected_outputs": ["figure:audit_panel"],
        "method": "visualization",
        "icu_rule_refs": ["visualization_rule"],
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


def test_exact_framework_contract_selects_audit_owner() -> None:
    step = _step()
    assert audit_panel_executor_owns_step(step)
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Test", steps=[step]),
    )
    assert selection is not None
    assert selection.analysis_kind == "audit_panel"
    assert selection.consumed_input_keys == ()
    assert not audit_panel_executor_owns_step(_step(inputs=["table:other"]))


def test_audit_panel_writes_same_stem_contract_and_nonclaiming_source(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    prior = run_dir / "steps" / "01_prior" / "outputs"
    prior.mkdir(parents=True)
    (prior / "step_summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "missingness_audit": {"missing_n": 2},
                "temporal_validity": {"status": "checked"},
            }
        ),
        encoding="utf-8",
    )
    step = _step()
    out_dir = run_dir / "steps" / step.step_id / "outputs"
    summary = run_audit_panel(out_dir=out_dir, run_dir=run_dir, step_id=step.step_id)
    assert (out_dir / "audit_panel.figure_contract.json").is_file()
    source = pd.read_csv(out_dir / "audit_panel_source_data.csv")
    assert source["eligible_summary_n"].tolist() == [1, 1, 1]
    assert set(source["audit_domain"]) == {
        "Data quality / missingness",
        "Sensitivity / robustness",
        "Leakage / validation",
    }
    assert not [
        finding
        for finding in FigureContractQualityValidator().audit(
            step=step,
            out_dir=out_dir,
            run_dir=run_dir,
            step_summary=summary,
        )
        if finding.severity == "error"
    ]
