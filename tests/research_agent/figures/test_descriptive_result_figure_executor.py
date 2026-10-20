from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.figures.display_labels import scoped_label_lookup

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


def test_empirical_summary_renderer_preserves_frequencies_and_distribution_role(tmp_path, monkeypatch):
    import sys

    frame = pd.DataFrame({
        "variable": ["temperature", "temperature"], "value": [35., 37.],
        "count": [1, 2], "denominator": [3, 3], "cohort_n": [4, 4],
        "proportion": [1 / 3, 2 / 3], "cumulative_proportion": [1 / 3, 1.],
    })
    monkeypatch.setattr(sys.modules[__name__], "_distribution_frame", lambda: frame)
    source = "table:distribution_summary"
    step = _distribution_step(inputs=[source], input_consumption_contracts=[
        ArtifactConsumptionContract(input_key=source, mode="all_rows"),
    ])
    run_dir, manifest, binding = _binding(tmp_path, step=step, kind="table")
    assert descriptive_result_figure_executor_owns_step(step, resolved_bindings={source: binding})
    out = tmp_path / "rendered"
    result = run_descriptive_result_figure(
        out_dir=out, run_dir=run_dir, resolved_inputs=manifest,
        step_id=step.step_id, input_key=source, figure_product="empirical",
        display_labels={"temperature": "Recorded temperature"},
    )
    pd.testing.assert_frame_equal(pd.read_csv(out / "empirical_source_data.csv"), frame)
    contract = json.loads((out / result["figure_contract"]).read_text())
    assert contract["panels"][0]["metadata"]["article_role"] == "distribution"
    assert contract["panels"][0]["metadata"]["chart_type"] == "distribution_plot"
    assert result["source_rows_consumed"] == 2
    # Even correctly digest-bound tables with inconsistent frequencies fail.
    frame.loc[0, "cumulative_proportion"] = .9
    run_dir, manifest, _ = _binding(tmp_path / "corrupt", step=step, kind="table")
    with pytest.raises(ValueError, match="frequencies"):
        run_descriptive_result_figure(
            out_dir=tmp_path / "bad", run_dir=run_dir, resolved_inputs=manifest,
            step_id=step.step_id, input_key=source, figure_product="bad",
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


def _wide_grouped_distribution_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [0, 60, 60.0, 100, 60, 0, 0.0, 63.0, 51.0, 74.0, 62.5, 15.0],
            [1, 40, 40.0, 100, 40, 0, 0.0, 67.0, 56.0, 77.0, 66.8, 14.0],
        ],
        columns=[
            "sepsis3_status",
            "n",
            "percentage",
            "denominator",
            "age_n_nonmissing",
            "age_missing_n",
            "age_missing_pct",
            "age_median",
            "age_q25",
            "age_q75",
            "age_mean",
            "age_sd",
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


def test_e1_wide_grouped_summary_is_rendered_by_the_same_owner(
    tmp_path: Path,
) -> None:
    step = _distribution_step(
        inputs=["table:distribution_prevalence"],
        expected_outputs=["figure:age_distribution"],
        input_consumption_contracts=[
            ArtifactConsumptionContract(
                input_key="table:distribution_prevalence", mode="all_rows"
            )
        ],
        figure_panels=[
            {
                "panel_id": "grouped_distribution",
                "figure_output": "figure:age_distribution",
                "article_role": "distribution",
                "chart_type": "point_range",
                "source_products": ["table:distribution_prevalence"],
            }
        ],
    )
    run_dir, manifest, binding = _binding(tmp_path, step=step, kind="table")
    source_path = run_dir / str(binding["relative_path"])
    frame = _wide_grouped_distribution_frame()
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
        step, resolved_bindings={step.inputs[0]: binding}
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
        figure_product="age_distribution",
    )
    source = pd.read_csv(out_dir / "age_distribution_source_data.csv")
    assert source["source_row_index"].tolist() == [0, 1]
    assert source["age_median"].tolist() == [63.0, 67.0]
    contract = json.loads(
        (out_dir / "age_distribution.figure_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["panels"][0]["metadata"] == {
        "article_role": "distribution",
        "chart_type": "point_range",
        "display_policy_reason_code": "SCIENTIFIC_RESULT_DISPLAY",
        "display_purpose": "scientific_result",
        "placement": "main",
        "source_data": ["age_distribution_source_data.csv"],
        "source_products": ["table:distribution_prevalence"],
    }
    from easyicu.research_agent.figures.publication import audit_publication_exports

    assert not [
        finding
        for finding in audit_publication_exports([out_dir / "age_distribution.svg"])
        if "outside the canvas" in finding.message
    ]
    _assert_valid_bundle(
        step=step,
        run_dir=run_dir,
        out_dir=out_dir,
        summary=summary,
        binding=binding,
    )


def test_wide_grouped_summary_requires_one_complete_metric_stem() -> None:
    step = _distribution_step(
        inputs=["table:distribution_prevalence"],
        input_consumption_contracts=[
            ArtifactConsumptionContract(
                input_key="table:distribution_prevalence", mode="all_rows"
            )
        ],
    )
    columns = list(_wide_grouped_distribution_frame().columns)

    for invalid_columns in (
        [column for column in columns if column != "age_sd"],
        [*columns, "lactate_median"],
    ):
        binding = {
            "declared_kind": "table",
            "evidence_kind": "table",
            "product_contract": {"columns": invalid_columns},
            "consumption_contract": {"mode": "all_rows"},
        }
        assert not descriptive_result_figure_executor_owns_step(
            step, resolved_bindings={step.inputs[0]: binding}
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


def _rebound(run_dir: Path, binding: dict[str, object], frame: pd.DataFrame) -> None:
    """``binding``'s parent table rewritten as ``frame``, its digests with it."""

    source_path = run_dir / str(binding["relative_path"])
    frame.to_csv(source_path, index=False)
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


def test_a_wide_tables_groups_go_by_the_plans_level_labels(tmp_path: Path) -> None:
    """A grouped exposure's level codes, read back as numbers (1.0), are named
    on the figure by the plan's labels for them."""

    step = _distribution_step(
        inputs=["table:distribution_prevalence"],
        expected_outputs=["figure:age_distribution"],
        input_consumption_contracts=[
            ArtifactConsumptionContract(
                input_key="table:distribution_prevalence", mode="all_rows"
            )
        ],
    )
    run_dir, manifest, binding = _binding(tmp_path, step=step, kind="table")
    frame = _wide_grouped_distribution_frame().rename(
        columns={"sepsis3_status": "map_group_x1"}
    )
    frame["map_group_x1"] = [1.0, 2.0]
    _rebound(run_dir, binding, frame)
    labels = {
        "map_group_x1=1": "Hypotension on day one",
        "map_group_x1=2": "No hypotension",
    }

    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Test", steps=[step], display_labels=labels),
        resolved_bindings={step.inputs[0]: binding},
    )
    assert selection is not None
    assert "Hypotension on day one" in selection.code
    out_dir = run_dir / "steps" / step.step_id / "outputs"
    run_descriptive_result_figure(
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=step.step_id,
        input_key=step.inputs[0],
        figure_product="age_distribution",
        display_labels=labels,
    )

    svg = (out_dir / "age_distribution.svg").read_text(encoding="utf-8")
    assert "Hypotension on day one" in svg
    assert "No hypotension" in svg


@pytest.mark.parametrize(
    ("value", "label"),
    [
        (2, "Group two"),
        ("2", "Group two"),
        (2.0, "Group two"),
        (np.int64(2), "Group two"),
        (np.float64(2.0), "Group two"),
        # Text stays as written; no fraction, NaN or boolean names a level.
        ("2.0", None),
        (2.5, None),
        (float("nan"), None),
        (True, None),
    ],
)
def test_a_level_code_read_back_as_a_number_names_its_level(
    value: object, label: str | None
) -> None:
    assert (
        scoped_label_lookup("map_group_x1", value, {"map_group_x1=2": "Group two"})
        == label
    )
