from __future__ import annotations

import hashlib
import json
from pathlib import Path

from easyicu.research_agent.contracts.declared_product import (
    declared_product_contract_findings,
)
from easyicu.research_agent.execution.runners.scientific_reporting_executor import (
    SCIENTIFIC_REPORTING_ANALYSIS_KIND,
    run_scientific_reporting,
    scientific_reporting_executor_owns_step,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _step(**updates) -> AnalysisStep:
    payload = {
        "step_id": "08_report",
        "planned_analysis_role": "auxiliary",
        "intent": "Assemble a source-bound index of the completed analysis outputs.",
        "inputs": ["table:primary_results", "statistic:model_performance"],
        "expected_outputs": ["report:analysis_results"],
        "method": "scientific_reporting",
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


def _bound_manifest(tmp_path: Path) -> dict:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results = run_dir / "results.csv"
    results.write_text("effect\n1.5\n", encoding="utf-8")
    performance = run_dir / "performance.json"
    performance.write_text('{"auroc": 0.81}\n', encoding="utf-8")

    def binding(path: Path, key: str, evidence_id: str) -> dict:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "declared_kind": key.split(":", 1)[0],
            "relative_path": path.name,
            "sha256": digest,
            "evidence_id": evidence_id,
            "produced_by_step": "05_primary",
            "identity_row": {"input_key": key, "sha256": digest},
        }

    return {
        "step_id": "08_report",
        "inputs": {
            "table:primary_results": binding(
                results, "table:primary_results", "ev_primary"
            ),
            "statistic:model_performance": binding(
                performance, "statistic:model_performance", "ev_performance"
            ),
        },
    }


def test_completed_report_is_selected_separately_from_feasibility() -> None:
    step = _step()
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Test", steps=[step]),
    )

    assert scientific_reporting_executor_owns_step(step)
    assert selection is not None
    assert selection.analysis_kind == SCIENTIFIC_REPORTING_ANALYSIS_KIND
    assert selection.consumed_input_keys == tuple(step.inputs)


def test_primary_or_empty_result_reports_are_not_claimed() -> None:
    assert not scientific_reporting_executor_owns_step(
        _step(planned_analysis_role="primary")
    )
    assert not scientific_reporting_executor_owns_step(_step(inputs=[]))
    assert not scientific_reporting_executor_owns_step(
        _step(method="feasibility_protocol")
    )


def test_runtime_indexes_results_without_claiming_no_estimate(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    summary = run_scientific_reporting(
        out_dir=out_dir,
        run_dir=tmp_path / "run",
        resolved_inputs=_bound_manifest(tmp_path),
        step_id="08_report",
        planned_analysis_role="auxiliary",
        intent=_step().intent,
        report_product="analysis_results",
        declared_inputs=list(_step().inputs),
    )

    assert summary["status"] == "ok"
    assert summary["analysis_status"] == "results_available"
    assert summary["effect_estimate"] is None
    assert summary["bound_input_count"] == 2
    findings = declared_product_contract_findings(
        step=_step(),
        step_summary=summary,
        effect_method_authorized=False,
        out_dir=out_dir,
    )
    assert not [
        finding
        for finding in findings
        if (finding.detail or {}).get("kind") == "declared_product_missing"
    ]
    report = (out_dir / "analysis_results.md").read_text(encoding="utf-8")
    assert "registered analysis outputs are available" in report
    assert "No estimate" not in report
    assert "ev_primary" in report
    receipt = json.loads(
        (out_dir / "analysis_results.receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["scientific_status"] == "evidence_bound_results_available"


def test_runtime_refuses_unbound_or_extra_inputs(tmp_path: Path) -> None:
    manifest = _bound_manifest(tmp_path)
    manifest["inputs"]["table:unexpected"] = dict(
        manifest["inputs"]["table:primary_results"]
    )

    try:
        run_scientific_reporting(
            out_dir=tmp_path / "out",
            run_dir=tmp_path / "run",
            resolved_inputs=manifest,
            step_id="08_report",
            planned_analysis_role="auxiliary",
            intent=_step().intent,
            report_product="analysis_results",
            declared_inputs=list(_step().inputs),
        )
    except ValueError as error:
        assert "does not match declared inputs" in str(error)
    else:
        raise AssertionError("widened report inputs must fail closed")
