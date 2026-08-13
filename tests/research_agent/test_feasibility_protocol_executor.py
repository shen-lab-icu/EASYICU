from __future__ import annotations

import hashlib
import json
from pathlib import Path

from easyicu.research_agent.contracts.declared_product import (
    declared_product_contract_findings,
)
from easyicu.research_agent.execution.runners.feasibility_protocol_executor import (
    FEASIBILITY_PROTOCOL_ANALYSIS_KIND,
    feasibility_protocol_executor_owns_step,
    run_feasibility_protocol,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _step(**updates) -> AnalysisStep:
    payload = {
        "step_id": "08_future_cohort_protocol",
        "planned_analysis_role": "sensitivity",
        "intent": (
            "Document a non-executable one-row-per-person sensitivity because "
            "the sealed source lacks person identity and encounter chronology."
        ),
        "inputs": ["artifact:analysis_cohort", "table:primary_results"],
        "expected_outputs": ["report:future_cohort_protocol"],
        "method": "feasibility_protocol",
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


def _bound_manifest(tmp_path: Path) -> dict:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cohort = run_dir / "cohort.parquet"
    cohort.write_bytes(b"cohort")
    results = run_dir / "results.csv"
    results.write_text("effect\n1.5\n", encoding="utf-8")

    def binding(path: Path, key: str, evidence_id: str) -> dict:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "absolute_path": str(path),
            "declared_kind": key.split(":", 1)[0],
            "relative_path": path.name,
            "sha256": digest,
            "evidence_id": evidence_id,
            "identity_row": {"input_key": key, "sha256": digest},
        }

    return {
        "step_id": "08_future_cohort_protocol",
        "inputs": {
            "artifact:analysis_cohort": binding(
                cohort, "artifact:analysis_cohort", "ev_cohort"
            ),
            "table:primary_results": binding(
                results, "table:primary_results", "ev_results"
            ),
        },
    }


def test_the_terminal_protocol_is_owned_and_selected() -> None:
    step = _step()
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Test", steps=[step]),
    )

    assert feasibility_protocol_executor_owns_step(step)
    assert selection is not None
    assert selection.analysis_kind == FEASIBILITY_PROTOCOL_ANALYSIS_KIND
    assert selection.consumed_input_keys == tuple(step.inputs)


def test_primary_or_numeric_protocols_are_never_claimed() -> None:
    assert not feasibility_protocol_executor_owns_step(
        _step(planned_analysis_role="primary")
    )
    assert not feasibility_protocol_executor_owns_step(
        _step(expected_outputs=["statistic:invented_effect"])
    )
    assert not feasibility_protocol_executor_owns_step(
        _step(method="logistic_regression")
    )


def test_runtime_writes_a_digest_bound_report_and_no_estimate(tmp_path: Path) -> None:
    manifest = _bound_manifest(tmp_path)
    out_dir = tmp_path / "out"

    summary = run_feasibility_protocol(
        out_dir=out_dir,
        run_dir=tmp_path / "run",
        resolved_inputs=manifest,
        step_id="08_future_cohort_protocol",
        planned_analysis_role="sensitivity",
        intent=_step().intent,
        report_product="future_cohort_protocol",
        declared_inputs=list(_step().inputs),
    )

    assert summary["status"] == "ok"
    assert summary["analysis_status"] == "not_executable"
    assert summary["effect_estimate"] is None
    assert summary["bound_input_count"] == 2
    assert summary["output_files"] == {
        "report:future_cohort_protocol": "future_cohort_protocol.md"
    }
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
    report = (out_dir / "future_cohort_protocol.md").read_text("utf-8")
    assert "No estimate" in report
    assert "one-row-per-person" in report
    receipt = json.loads(
        (out_dir / "future_cohort_protocol.receipt.json").read_text("utf-8")
    )
    assert receipt["effect_estimate"] is None
    assert len(receipt["bound_input_authorities"]) == 2


def test_markdown_is_not_inferred_as_a_report_without_a_typed_registration(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "future_cohort_protocol.md").write_text("# Protocol\n", "utf-8")

    findings = declared_product_contract_findings(
        step=_step(),
        step_summary={"output_files": ["future_cohort_protocol.md"]},
        effect_method_authorized=False,
        out_dir=out_dir,
    )

    assert any(
        (finding.detail or {}).get("kind") == "declared_product_missing"
        for finding in findings
    )


def test_runtime_refuses_a_missing_input_binding(tmp_path: Path) -> None:
    manifest = _bound_manifest(tmp_path)
    del manifest["inputs"]["table:primary_results"]

    try:
        run_feasibility_protocol(
            out_dir=tmp_path / "out",
            run_dir=tmp_path / "run",
            resolved_inputs=manifest,
            step_id="08_future_cohort_protocol",
            planned_analysis_role="sensitivity",
            intent=_step().intent,
            report_product="future_cohort_protocol",
            declared_inputs=list(_step().inputs),
        )
    except ValueError as error:
        assert "authority is incomplete" in str(error)
    else:
        raise AssertionError("missing input authority must fail closed")
