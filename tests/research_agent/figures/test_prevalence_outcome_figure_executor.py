from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from easyicu.research_agent.audits.validators import (
    FigureContractQualityValidator,
    FigureSourceDataValidator,
)
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.execution.runners.prevalence_outcome_figure_executor import (
    PREVALENCE_OUTCOME_FIGURE_INPUT,
    prevalence_outcome_figure_executor_owns_step,
    run_prevalence_outcome_figure,
)
from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
)


def _step(**updates) -> AnalysisStep:
    payload = {
        "step_id": "04_prevalence_outcome_figure",
        "planned_analysis_role": "auxiliary",
        "intent": "Render the registered prevalence and outcome-risk table.",
        "inputs": [PREVALENCE_OUTCOME_FIGURE_INPUT],
        "expected_outputs": ["figure:prevalence_mortality"],
        "method": "visualization",
        "input_consumption_contracts": [
            ArtifactConsumptionContract(
                input_key=PREVALENCE_OUTCOME_FIGURE_INPUT,
                mode="all_rows",
            )
        ],
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


def _frame() -> pd.DataFrame:
    columns = [
        "row_type",
        "variable",
        "category",
        "n",
        "denominator",
        "percentage",
        "estimate",
        "ci_lower",
        "ci_upper",
        "events",
        "missing_n",
        "missing_pct",
    ]
    rows = [
        [
            "cohort_context",
            "analysis_denominator",
            "locked_cohort_n",
            10,
            10,
            1.0,
            10.0,
            None,
            None,
            None,
            None,
            None,
        ],
        [
            "cohort_context",
            "analysis_denominator",
            "exposure_available_n",
            10,
            10,
            1.0,
            10.0,
            None,
            None,
            None,
            None,
            None,
        ],
        [
            "prevalence",
            "binary_exposure",
            0,
            6,
            10,
            0.6,
            0.6,
            None,
            None,
            None,
            None,
            None,
        ],
        [
            "prevalence",
            "binary_exposure",
            1,
            4,
            10,
            0.4,
            0.4,
            None,
            None,
            None,
            None,
            None,
        ],
        [
            "absolute_outcome_risk",
            "binary_outcome",
            0,
            6,
            6,
            1 / 6,
            1 / 6,
            0.03,
            0.56,
            1,
            0,
            0,
        ],
        [
            "absolute_outcome_risk",
            "binary_outcome",
            1,
            4,
            4,
            0.5,
            0.5,
            0.15,
            0.85,
            2,
            0,
            0,
        ],
    ]
    return pd.DataFrame(rows, columns=columns)


def _binding(tmp_path: Path, frame: pd.DataFrame) -> tuple[Path, dict]:
    run_dir = tmp_path / "run"
    parent_step = "04_prevalence_outcome"
    parent_output = (
        run_dir / "steps" / parent_step / "outputs" / "absolute_risk_context.csv"
    )
    parent_output.parent.mkdir(parents=True)
    frame.to_csv(parent_output, index=False)
    record = EvidenceStore(run_dir).register_file(
        kind="table",
        description="Digest-bound absolute-risk context.",
        source_path=parent_output,
        evidence_id="table_absolute_risk_context",
        produced_by_step=parent_step,
        producer="deterministic_test",
        generation_mode="deterministic_standard",
    )
    table = run_dir / record.relative_path
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    columns = list(frame.columns)
    binding = {
        "absolute_path": str(table),
        "relative_path": str(table.relative_to(run_dir)),
        "sha256": digest,
        "declared_kind": "table",
        "evidence_kind": "table",
        "evidence_id": record.evidence_id,
        "produced_by_step": parent_step,
        "product": "absolute_risk_context",
        "identity_row": {
            "declared_kind": "table",
            "evidence_id": record.evidence_id,
            "input_key": PREVALENCE_OUTCOME_FIGURE_INPUT,
            "produced_by_step": parent_step,
            "product": "absolute_risk_context",
            "sha256": digest,
        },
        "product_contract": {
            "schema_version": "easyicu.host_typed_product.v4",
            "tabular_format": "csv",
            "columns": columns,
            "column_count": len(columns),
            "column_dtypes": {name: str(frame[name].dtype) for name in columns},
            "numeric_columns": [
                name for name in columns if pd.api.types.is_numeric_dtype(frame[name])
            ],
            "row_count": len(frame),
        },
        "consumption_contract": {
            "schema_version": "easyicu.verified_artifact_consumption/1",
            "input_key": PREVALENCE_OUTCOME_FIGURE_INPUT,
            "mode": "all_rows",
            "artifact_sha256": digest,
            "verified_row_count": len(frame),
        },
    }
    manifest = {
        "schema_version": "2.1",
        "step_id": "04_prevalence_outcome_figure",
        "inputs": {PREVALENCE_OUTCOME_FIGURE_INPUT: binding},
    }
    return run_dir, manifest


def test_exact_closed_contract_selects_standard_executor() -> None:
    step = _step()
    assert prevalence_outcome_figure_executor_owns_step(step)
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Test", steps=[step]),
    )
    assert selection is not None
    assert selection.analysis_kind == "prevalence_outcome_figure"
    assert selection.consumed_input_keys == (PREVALENCE_OUTCOME_FIGURE_INPUT,)


def test_owner_rejects_widened_or_unbound_contracts() -> None:
    assert not prevalence_outcome_figure_executor_owns_step(
        _step(expected_outputs=["figure:unrelated_panel"])
    )
    assert not prevalence_outcome_figure_executor_owns_step(
        _step(
            inputs=[PREVALENCE_OUTCOME_FIGURE_INPUT, "table:other"],
            input_consumption_contracts=[
                ArtifactConsumptionContract(
                    input_key=PREVALENCE_OUTCOME_FIGURE_INPUT,
                    mode="all_rows",
                ),
                ArtifactConsumptionContract(input_key="table:other", mode="all_rows"),
            ],
        )
    )
    assert not prevalence_outcome_figure_executor_owns_step(
        _step(input_consumption_contracts=[])
    )
    assert not prevalence_outcome_figure_executor_owns_step(
        _step(planned_analysis_role="primary")
    )


def test_runner_renders_complete_source_backed_bundle(tmp_path: Path) -> None:
    frame = _frame()
    run_dir, manifest = _binding(tmp_path, frame)
    out_dir = run_dir / "steps" / "04_prevalence_outcome_figure" / "outputs"
    summary = run_prevalence_outcome_figure(
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id="04_prevalence_outcome_figure",
        figure_product="prevalence_mortality",
    )

    assert summary["status"] == "ok"
    assert summary["source_rows_consumed"] == len(frame)
    for suffix in ("png", "svg", "pdf", "tiff"):
        assert (out_dir / f"prevalence_mortality.{suffix}").is_file()
    full_source = pd.read_csv(out_dir / "prevalence_mortality_input_source_data.csv")
    assert full_source["source_row_index"].tolist() == list(range(len(frame)))
    risk_source = pd.read_csv(out_dir / "prevalence_mortality_outcome_source_data.csv")
    assert risk_source["events"].tolist() == [1, 2]
    contract = json.loads(
        (out_dir / "prevalence_mortality.figure_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert [panel["role"] for panel in contract["panels"]] == [
        "baseline_context",
        "descriptive_result",
    ]
    assert summary["contract_files"] == ["prevalence_mortality.figure_contract.json"]
    source_findings = FigureSourceDataValidator().audit(
        step=_step(),
        out_dir=out_dir,
        run_dir=run_dir,
        step_summary=summary,
    )
    assert not [finding for finding in source_findings if finding.severity == "error"]
    assert not [
        finding
        for finding in FigureContractQualityValidator().audit(
            step=_step(),
            out_dir=out_dir,
            run_dir=run_dir,
            step_summary=summary,
        )
        if finding.severity == "error"
    ]


def test_runner_rejects_nonreconciled_event_count(tmp_path: Path) -> None:
    frame = _frame()
    frame.loc[
        frame["row_type"].eq("absolute_outcome_risk") & frame["category"].eq(1),
        "events",
    ] = 3
    run_dir, manifest = _binding(tmp_path, frame)
    out_dir = run_dir / "steps" / "04_prevalence_outcome_figure" / "outputs"

    try:
        run_prevalence_outcome_figure(
            out_dir=out_dir,
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id="04_prevalence_outcome_figure",
            figure_product="prevalence_mortality",
        )
    except ValueError as exc:
        assert "outcome risk or confidence interval" in str(exc)
    else:
        raise AssertionError("nonreconciled outcome events were accepted")
    assert not (out_dir / "prevalence_mortality.png").exists()
