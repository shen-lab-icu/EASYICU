from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.execution.runners.cohort_flow_figure_executor import (
    COHORT_FLOW_INPUT,
    cohort_flow_figure_executor_owns_step,
    run_cohort_flow_figure,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
)


def _step(**updates: object) -> AnalysisStep:
    payload: dict[str, object] = {
        "step_id": "08_cohort_accounting_figure",
        "planned_analysis_role": "auxiliary",
        "intent": "Render exact cohort accounting.",
        "inputs": [COHORT_FLOW_INPUT],
        "expected_outputs": ["figure:cohort_accounting"],
        "method": "visualization",
        "input_consumption_contracts": [
            ArtifactConsumptionContract(input_key=COHORT_FLOW_INPUT, mode="all_rows")
        ],
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [0, "universe", 140, 0, 140],
            [1, "adult", 140, 12, 128],
            [2, "first_icu_stay", 128, 8, 120],
        ],
        columns=[
            "step_order",
            "predicate_kind",
            "n_before",
            "n_excluded",
            "n_remaining",
        ],
    )


def _binding(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, object]]:
    run_dir = tmp_path / "run"
    source = run_dir / "steps" / "01_define_analysis_cohort" / "outputs" / "flow.csv"
    source.parent.mkdir(parents=True)
    frame = _frame()
    frame.to_csv(source, index=False)
    record = EvidenceStore(run_dir).register_file(
        kind="table",
        description="Canonical cohort flow.",
        source_path=source,
        evidence_id="cohort_flow_parent",
        produced_by_step="01_define_analysis_cohort",
        producer="deterministic_test",
        generation_mode="deterministic_standard",
    )
    evidence_path = run_dir / record.relative_path
    digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    identity = {
        "declared_kind": "table",
        "evidence_id": record.evidence_id,
        "input_key": COHORT_FLOW_INPUT,
        "produced_by_step": "01_define_analysis_cohort",
        "product": "cohort_flow",
        "sha256": digest,
    }
    binding: dict[str, object] = {
        "relative_path": str(evidence_path.relative_to(run_dir)),
        "sha256": digest,
        "declared_kind": "table",
        "evidence_kind": "table",
        "evidence_id": record.evidence_id,
        "produced_by_step": "01_define_analysis_cohort",
        "product": "cohort_flow",
        "identity_row": identity,
        "product_contract": {
            "schema_version": "easyicu.host_typed_product.v4",
            "tabular_format": "csv",
            "columns": list(frame.columns),
            "row_count": len(frame),
        },
        "consumption_contract": {
            "schema_version": "easyicu.verified_artifact_consumption/1",
            "input_key": COHORT_FLOW_INPUT,
            "mode": "all_rows",
            "artifact_sha256": digest,
            "verified_row_count": len(frame),
        },
    }
    manifest: dict[str, object] = {
        "schema_version": "2.1",
        "step_id": _step().step_id,
        "inputs": {COHORT_FLOW_INPUT: binding},
    }
    return run_dir, manifest, binding


def test_exact_cohort_flow_selects_and_renders_without_llm(tmp_path: Path) -> None:
    step = _step()
    run_dir, manifest, binding = _binding(tmp_path)
    assert cohort_flow_figure_executor_owns_step(
        step, resolved_bindings={COHORT_FLOW_INPUT: binding}
    )
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Test", steps=[step]),
        resolved_bindings={COHORT_FLOW_INPUT: binding},
    )
    assert selection is not None
    assert selection.analysis_kind == "cohort_flow_figure"

    out_dir = run_dir / "steps" / step.step_id / "outputs"
    summary = run_cohort_flow_figure(
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=step.step_id,
        figure_product="cohort_accounting",
    )

    source = pd.read_csv(out_dir / "cohort_accounting_source_data.csv")
    assert source["n_remaining"].tolist() == [140, 128, 120]
    assert summary["source_rows_consumed"] == 3
    assert (out_dir / "cohort_accounting.figure_contract.json").is_file()


def test_owner_and_runner_fail_closed_on_widening_or_arithmetic_drift(
    tmp_path: Path,
) -> None:
    step = _step()
    run_dir, manifest, binding = _binding(tmp_path)
    assert not cohort_flow_figure_executor_owns_step(
        _step(
            inputs=[COHORT_FLOW_INPUT, "table:other"],
            input_consumption_contracts=[
                ArtifactConsumptionContract(
                    input_key=COHORT_FLOW_INPUT, mode="all_rows"
                ),
                ArtifactConsumptionContract(input_key="table:other", mode="all_rows"),
            ],
        ),
        resolved_bindings={COHORT_FLOW_INPUT: binding},
    )
    path = run_dir / str(binding["relative_path"])
    frame = pd.read_csv(path)
    frame.loc[1, "n_remaining"] = 127
    frame.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    binding["sha256"] = digest
    binding["identity_row"]["sha256"] = digest
    binding["consumption_contract"]["artifact_sha256"] = digest
    with pytest.raises(ValueError, match="denominator arithmetic"):
        run_cohort_flow_figure(
            out_dir=run_dir / "steps" / step.step_id / "outputs",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id=step.step_id,
            figure_product="cohort_accounting",
        )


def test_runner_refuses_digest_drift(tmp_path: Path) -> None:
    step = _step()
    run_dir, manifest, binding = _binding(tmp_path)
    binding["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="authority binding|digest"):
        run_cohort_flow_figure(
            out_dir=run_dir / "steps" / step.step_id / "outputs",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id=step.step_id,
            figure_product="cohort_accounting",
        )
