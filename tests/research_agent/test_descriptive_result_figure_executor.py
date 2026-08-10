from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.audits.validators import (
    FigureContractQualityValidator,
    FigureSourceDataValidator,
)
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.execution.runners.descriptive_result_figure_executor import (
    DESCRIPTIVE_DISTRIBUTION_COLUMNS,
    descriptive_result_figure_executor_owns_step,
    run_descriptive_result_figure,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
)


def _distribution_step(**updates: object) -> AnalysisStep:
    payload: dict[str, object] = {
        "step_id": "06_distribution_figure",
        "planned_analysis_role": "auxiliary",
        "intent": "Render the typed descriptive distribution.",
        "inputs": ["table:descriptive_distribution"],
        "expected_outputs": ["figure:descriptive_distribution"],
        "method": "visualization",
        "input_consumption_contracts": [
            ArtifactConsumptionContract(
                input_key="table:descriptive_distribution",
                mode="all_rows",
            )
        ],
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


def _statistic_step(**updates: object) -> AnalysisStep:
    payload: dict[str, object] = {
        "step_id": "07_statistic_figure",
        "planned_analysis_role": "auxiliary",
        "intent": "Render one typed descriptive statistic.",
        "inputs": ["statistic:rank_correlation"],
        "expected_outputs": ["figure:rank_correlation_display"],
        "method": "visualization",
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


def _distribution_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["length_of_stay", "Overall", "overall", 20, 20, 0, 0.0, 2.0, 1.0, 4.0, 3.0, 2.0],
            ["length_of_stay", "A", "group", 9, 9, 0, 0.0, 1.5, 0.8, 3.0, 2.4, 1.8],
            ["length_of_stay", "B", "group", 11, 11, 0, 0.0, 2.5, 1.2, 5.0, 3.5, 2.2],
        ],
        columns=DESCRIPTIVE_DISTRIBUTION_COLUMNS,
    )


def _canonical_grouped_distribution_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["length_of_stay", "sex", "Female", 9, 9, 0, 0.0, 1.5, 0.8, 3.0, 2.4, 1.8, "days"],
            ["length_of_stay", "sex", "Male", 11, 11, 0, 0.0, 2.5, 1.2, 5.0, 3.5, 2.2, "days"],
        ],
        columns=[
            "variable",
            "group_variable",
            "group",
            "group_n",
            "n_nonmissing",
            "missing_n",
            "missing_pct",
            "median",
            "q25",
            "q75",
            "mean",
            "sd",
            "unit",
        ],
    )


def _binding(
    tmp_path: Path,
    *,
    step: AnalysisStep,
    kind: str,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    run_dir = tmp_path / "run"
    parent_dir = run_dir / "steps" / "05_parent" / "outputs"
    parent_dir.mkdir(parents=True)
    input_key = step.inputs[0]
    product = input_key.partition(":")[2]
    if kind == "table":
        frame = _distribution_frame()
        source_path = parent_dir / "typed_distribution.csv"
        frame.to_csv(source_path, index=False)
        product_contract: dict[str, object] = {
            "schema_version": "easyicu.host_typed_product.v4",
            "tabular_format": "csv",
            "columns": list(frame.columns),
            "row_count": len(frame),
        }
    else:
        payload = {
            "name": product,
            "value": -0.125,
            "effect_scale": "Spearman rank correlation",
            "unit": "unitless",
            "p_value": 0.04,
            "denominator": 20,
        }
        source_path = parent_dir / "typed_statistic.json"
        source_path.write_text(json.dumps(payload), encoding="utf-8")
        product_contract = {
            "schema_version": "easyicu.host_typed_product.v1",
            "json_structure": {
                "root_type": "object",
                "paths": {"": {"type": "object", "keys": list(payload)}},
            },
        }
    record = EvidenceStore(run_dir).register_file(
        kind=kind,
        description=f"Digest-bound {kind} parent.",
        source_path=source_path,
        evidence_id=f"{kind}_typed_parent",
        produced_by_step="05_parent",
        producer="deterministic_test",
        generation_mode="deterministic_standard",
    )
    path = run_dir / record.relative_path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {
        "declared_kind": kind,
        "evidence_id": record.evidence_id,
        "input_key": input_key,
        "produced_by_step": "05_parent",
        "product": product,
        "sha256": digest,
    }
    binding: dict[str, object] = {
        "relative_path": str(path.relative_to(run_dir)),
        "sha256": digest,
        "declared_kind": kind,
        "evidence_kind": kind,
        "evidence_id": record.evidence_id,
        "absolute_path": str(path),
        "produced_by_step": "05_parent",
        "product": product,
        "identity_row": identity,
        "product_contract": product_contract,
    }
    if kind == "table":
        binding["consumption_contract"] = {
            "schema_version": "easyicu.verified_artifact_consumption/1",
            "input_key": input_key,
            "mode": "all_rows",
            "artifact_sha256": digest,
            "verified_row_count": len(_distribution_frame()),
        }
    manifest: dict[str, object] = {
        "schema_version": "2.1",
        "step_id": step.step_id,
        "inputs": {input_key: binding},
    }
    return run_dir, manifest, binding


def _assert_valid_bundle(
    *,
    step: AnalysisStep,
    run_dir: Path,
    out_dir: Path,
    summary: dict[str, object],
    binding: dict[str, object],
) -> None:
    validator_kwargs: dict[str, object] = {}
    if binding["declared_kind"] == "statistic":
        validator_kwargs = {
            "completed_step_records": [
                {
                    "step_id": "05_parent",
                    "status": "ok",
                    "evidence_ids": [binding["evidence_id"]],
                    "step_summary": {"status": "ok"},
                }
            ],
            "resolved_input_bindings": {step.inputs[0]: binding},
        }
    source_findings = FigureSourceDataValidator().audit(
        step=step,
        out_dir=out_dir,
        run_dir=run_dir,
        step_summary=summary,
        **validator_kwargs,
    )
    assert not [finding for finding in source_findings if finding.severity == "error"]
    quality_findings = FigureContractQualityValidator().audit(
        step=step,
        out_dir=out_dir,
        run_dir=run_dir,
        step_summary=summary,
    )
    assert not [finding for finding in quality_findings if finding.severity == "error"]


def test_distribution_contract_selects_and_renders_exact_parent(tmp_path: Path) -> None:
    step = _distribution_step()
    run_dir, manifest, binding = _binding(tmp_path, step=step, kind="table")
    assert descriptive_result_figure_executor_owns_step(
        step,
        resolved_bindings={step.inputs[0]: binding},
    )
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Test", steps=[step]),
        resolved_bindings={step.inputs[0]: binding},
    )
    assert selection is not None
    assert selection.analysis_kind == "descriptive_result_figure"

    out_dir = run_dir / "steps" / step.step_id / "outputs"
    summary = run_descriptive_result_figure(
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=step.step_id,
        input_key=step.inputs[0],
        figure_product="descriptive_distribution",
    )
    source = pd.read_csv(out_dir / "descriptive_distribution_source_data.csv")
    assert source["source_row_index"].tolist() == [0, 1, 2]
    assert source["median"].tolist() == [2.0, 1.5, 2.5]
    _assert_valid_bundle(
        step=step,
        run_dir=run_dir,
        out_dir=out_dir,
        summary=summary,
        binding=binding,
    )


def test_canonical_group_n_distribution_contract_is_rendered_without_llm(
    tmp_path: Path,
) -> None:
    step = _distribution_step()
    run_dir, manifest, binding = _binding(tmp_path, step=step, kind="table")
    source_path = run_dir / str(binding["relative_path"])
    frame = _canonical_grouped_distribution_frame()
    frame.to_csv(source_path, index=False)
    frame.to_csv(
        run_dir / "steps" / "05_parent" / "outputs" / "typed_distribution.csv",
        index=False,
    )
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    binding["sha256"] = digest
    binding["identity_row"]["sha256"] = digest
    binding["product_contract"] = {
        "schema_version": "easyicu.host_typed_product.v4",
        "tabular_format": "csv",
        "columns": list(frame.columns),
        "row_count": len(frame),
    }
    binding["consumption_contract"]["artifact_sha256"] = digest
    binding["consumption_contract"]["verified_row_count"] = len(frame)

    assert descriptive_result_figure_executor_owns_step(
        step,
        resolved_bindings={step.inputs[0]: binding},
    )
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Test", steps=[step]),
        resolved_bindings={step.inputs[0]: binding},
    )
    assert selection is not None
    assert selection.analysis_kind == "descriptive_result_figure"

    out_dir = run_dir / "steps" / step.step_id / "outputs"
    summary = run_descriptive_result_figure(
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=step.step_id,
        input_key=step.inputs[0],
        figure_product="descriptive_distribution",
    )
    source = pd.read_csv(out_dir / "descriptive_distribution_source_data.csv")
    assert source["group_n"].tolist() == [9, 11]
    assert summary["method"] == "deterministic_descriptive_result_figure"
    _assert_valid_bundle(
        step=step,
        run_dir=run_dir,
        out_dir=out_dir,
        summary=summary,
        binding=binding,
    )


def test_scalar_contract_excludes_unbound_numeric_siblings(tmp_path: Path) -> None:
    step = _statistic_step()
    run_dir, manifest, binding = _binding(tmp_path, step=step, kind="statistic")
    out_dir = run_dir / "steps" / step.step_id / "outputs"
    summary = run_descriptive_result_figure(
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=step.step_id,
        input_key=step.inputs[0],
        figure_product="rank_correlation_display",
    )
    source = pd.read_csv(out_dir / "rank_correlation_display_source_data.csv")
    assert source["value"].tolist() == [-0.125]
    assert "p_value" not in source.columns
    assert "denominator" not in source.columns
    _assert_valid_bundle(
        step=step,
        run_dir=run_dir,
        out_dir=out_dir,
        summary=summary,
        binding=binding,
    )


def test_owner_and_runner_fail_closed_on_widening_or_digest_drift(tmp_path: Path) -> None:
    step = _distribution_step()
    run_dir, manifest, binding = _binding(tmp_path, step=step, kind="table")
    assert not descriptive_result_figure_executor_owns_step(
        _distribution_step(
            inputs=[step.inputs[0], "table:other"],
            input_consumption_contracts=[
                ArtifactConsumptionContract(input_key=step.inputs[0], mode="all_rows"),
                ArtifactConsumptionContract(input_key="table:other", mode="all_rows"),
            ],
        ),
        resolved_bindings={step.inputs[0]: binding},
    )
    binding["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="authority binding|digest"):
        run_descriptive_result_figure(
            out_dir=run_dir / "steps" / step.step_id / "outputs",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id=step.step_id,
            input_key=step.inputs[0],
            figure_product="descriptive_distribution",
        )
