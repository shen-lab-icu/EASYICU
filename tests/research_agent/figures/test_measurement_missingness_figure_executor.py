from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from easyicu.research_agent.execution.runners.missingness_measurement_figure_executor import (
    MEASUREMENT_MISSINGNESS_FIGURE_INPUT,
    measurement_missingness_figure_executor_owns_step,
    run_measurement_missingness_figure,
)
from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
)


def _step() -> AnalysisStep:
    return AnalysisStep(
        step_id="09_figure_data_quality",
        planned_analysis_role="auxiliary",
        intent="Render every row of the typed measurement-missingness audit.",
        inputs=[MEASUREMENT_MISSINGNESS_FIGURE_INPUT],
        expected_outputs=["figure:data_quality"],
        method="visualization",
        input_consumption_contracts=[
            ArtifactConsumptionContract(
                input_key=MEASUREMENT_MISSINGNESS_FIGURE_INPUT,
                mode="all_rows",
            )
        ],
    )


def _binding(run_dir: Path) -> tuple[dict, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "concept": ["exposure", "outcome", "age", "death_time"],
            "variable": ["exposure", "outcome", "age", "death_time"],
            "label": ["Exposure", "Outcome", "Age", "Death time"],
            "value_column": ["exposure_max", "outcome", "age", "death_time"],
            "n_total": [140, 140, 140, 140],
            "measured_one_n": [112, 140, 133, 14],
            "measured_one_pct": [80.0, 100.0, 95.0, 10.0],
            "value_missing_n": [28, 0, 7, 0],
            "value_missing_pct": [20.0, 0.0, 5.0, 0.0],
            "eligible_n": [140, 140, 140, 14],
            "not_applicable_n": [0, 0, 0, 126],
            "indicator_semantics": [
                "measurement_availability",
                "measurement_availability",
                "measurement_availability",
                "conditional_event_time",
            ],
        }
    )
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    path = evidence_dir / "table_owner__missingness_measurement_audit.csv"
    frame.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {
        "declared_kind": "table",
        "evidence_id": "table_owner",
        "input_key": MEASUREMENT_MISSINGNESS_FIGURE_INPUT,
        "produced_by_step": "03_measurement_missingness_audit",
        "product": "missingness_measurement_audit",
        "sha256": digest,
    }
    binding = {
        "relative_path": str(path.relative_to(run_dir)),
        "sha256": digest,
        "declared_kind": "table",
        "evidence_kind": "table",
        "evidence_id": "table_owner",
        "produced_by_step": "03_measurement_missingness_audit",
        "product": "missingness_measurement_audit",
        "identity_row": identity,
        "product_contract": {
            "schema_version": "easyicu.host_typed_product.v4",
            "columns": list(frame.columns),
            "row_count": len(frame),
        },
        "consumption_contract": {
            "schema_version": "easyicu.verified_artifact_consumption/1",
            "input_key": MEASUREMENT_MISSINGNESS_FIGURE_INPUT,
            "mode": "all_rows",
            "artifact_sha256": digest,
            "verified_row_count": len(frame),
        },
    }
    return binding, frame


def test_single_typed_measurement_audit_selects_deterministic_renderer(tmp_path: Path):
    assert MEASUREMENT_MISSINGNESS_FIGURE_INPUT == (
        "table:missingness_measurement_audit"
    )
    step = _step()
    binding, _frame = _binding(tmp_path)
    bindings = {MEASUREMENT_MISSINGNESS_FIGURE_INPUT: binding}

    assert measurement_missingness_figure_executor_owns_step(
        step,
        resolved_bindings=bindings,
    )
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Audit data quality.", steps=[step]),
        resolved_bindings=bindings,
    )

    assert selection is not None
    assert selection.analysis_kind == "measurement_missingness_figure"
    assert selection.consumed_input_keys == (MEASUREMENT_MISSINGNESS_FIGURE_INPUT,)
    assert selection.host_sealed_renderer is True


def test_legacy_measurement_product_label_selects_the_same_host_renderer(
    tmp_path: Path,
) -> None:
    input_key = "table:measurement_missingness"
    step = _step().model_copy(
        update={
            "inputs": [input_key],
            "input_consumption_contracts": [
                ArtifactConsumptionContract(input_key=input_key, mode="all_rows")
            ],
        }
    )
    binding, _frame = _binding(tmp_path)
    binding["product"] = "measurement_missingness"
    binding["identity_row"] = {
        **binding["identity_row"],
        "input_key": input_key,
        "product": "measurement_missingness",
    }

    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Audit data quality.", steps=[step]),
        resolved_bindings={input_key: binding},
    )

    assert selection is not None
    assert selection.analysis_kind == "measurement_missingness_figure"
    assert selection.consumed_input_keys == (input_key,)
    assert f"input_key={input_key!r}" in selection.code


def test_single_typed_measurement_audit_preserves_physical_parent_lineage(
    tmp_path: Path,
):
    step = _step()
    binding, frame = _binding(tmp_path)
    resolved = {
        "step_id": step.step_id,
        "inputs": {MEASUREMENT_MISSINGNESS_FIGURE_INPUT: binding},
    }
    resolved_path = tmp_path / "resolved.json"
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
    out_dir = tmp_path / "outputs"

    summary = run_measurement_missingness_figure(
        out_dir=out_dir,
        run_dir=tmp_path,
        resolved_inputs=resolved_path,
        step_id=step.step_id,
        figure_product="data_quality",
    )

    source = pd.read_csv(out_dir / "data_quality_source_data.csv")
    assert source["source_row_index"].tolist() == list(range(len(frame)))
    assert source["source_table"].unique().tolist() == [
        "missingness_measurement_audit.csv"
    ]
    assert source.drop(columns=["source_row_index", "source_table"]).equals(frame)
    assert summary["source_rows_consumed"] == {
        MEASUREMENT_MISSINGNESS_FIGURE_INPUT: len(frame)
    }
    assert (out_dir / "data_quality.figure_contract.json").is_file()
    assert (out_dir / "data_quality.png").is_file()
    svg = (out_dir / "data_quality.svg").read_text(encoding="utf-8")
    assert "Death time (conditional)" in svg
    assert "14 applicable; 0/14 missing" in svg
    contract = json.loads(
        (out_dir / "data_quality.figure_contract.json").read_text(encoding="utf-8")
    )
    assert contract["panels"][0]["title"] == "Data availability and applicability"


def test_single_renderer_declines_unreadable_or_widened_binding(tmp_path: Path):
    step = _step()
    binding, _frame = _binding(tmp_path)
    binding["product_contract"]["columns"].remove("value_missing_pct")
    assert not measurement_missingness_figure_executor_owns_step(
        step,
        resolved_bindings={MEASUREMENT_MISSINGNESS_FIGURE_INPUT: binding},
    )
    widened = step.model_copy(update={"inputs": [MEASUREMENT_MISSINGNESS_FIGURE_INPUT, "age"]})
    assert not measurement_missingness_figure_executor_owns_step(
        widened,
        resolved_bindings={MEASUREMENT_MISSINGNESS_FIGURE_INPUT: binding},
    )
