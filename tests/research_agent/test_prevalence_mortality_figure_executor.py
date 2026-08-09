from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.prevalence_mortality_figure_executor import (
    PREVALENCE_MORTALITY_FIGURE_INPUTS,
    binary_level_labels,
    prevalence_mortality_figure_executor_owns_step,
    run_prevalence_mortality_figure,
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
        "step_id": "03_prevalence_and_mortality_context_figure",
        "planned_analysis_role": "auxiliary",
        "intent": "Render the sealed prevalence and mortality tables.",
        "inputs": list(PREVALENCE_MORTALITY_FIGURE_INPUTS),
        "expected_outputs": ["figure:prevalence_mortality"],
        "method": "visualization",
        "input_consumption_contracts": [
            ArtifactConsumptionContract(input_key=input_key, mode="all_rows")
            for input_key in PREVALENCE_MORTALITY_FIGURE_INPUTS
        ],
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


def _cohort_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["locked_cohort", None, 10, 10, 100.0, None, None],
            ["exposure_available", None, 10, 10, 100.0, None, None],
            ["exposure_missing", None, 0, 10, 0.0, None, None],
            ["exposure_prevalence", 1.0, 4, 10, 40.0, 20.0, 65.0],
            ["outcome_available", None, 10, 10, 100.0, None, None],
            ["outcome_missing", None, 0, 10, 0.0, None, None],
        ],
        columns=[
            "summary",
            "exposure_level",
            "numerator",
            "denominator",
            "percentage",
            "ci_low",
            "ci_high",
        ],
    )


def _outcome_incidence() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["overall", 10, 10, 3, 30.0, 10.0, 60.0],
            ["0.0", 6, 6, 1, 100.0 / 6.0, 1.0, 55.0],
            ["1.0", 4, 4, 2, 50.0, 15.0, 85.0],
        ],
        columns=[
            "exposure_level",
            "exposure_count",
            "outcome_observed_n",
            "deaths",
            "mortality_pct",
            "ci_low_pct",
            "ci_high_pct",
        ],
    )


def _binding(
    *,
    run_dir: Path,
    input_key: str,
    product: str,
    frame: pd.DataFrame,
) -> dict:
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    path = evidence_dir / f"{product}.csv"
    frame.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence_id = f"table_{product}"
    return {
        "absolute_path": str(path),
        "relative_path": path.relative_to(run_dir).as_posix(),
        "sha256": digest,
        "declared_kind": "table",
        "evidence_kind": "table",
        "evidence_id": evidence_id,
        "produced_by_step": "03_prevalence_and_mortality_context",
        "product": product,
        "identity_row": {
            "declared_kind": "table",
            "evidence_id": evidence_id,
            "input_key": input_key,
            "produced_by_step": "03_prevalence_and_mortality_context",
            "product": product,
            "sha256": digest,
        },
        "product_contract": {
            "schema_version": "easyicu.host_typed_product.v4",
            "tabular_format": "csv",
            "columns": list(frame.columns),
            "column_count": len(frame.columns),
            "column_dtypes": {
                column: str(frame[column].dtype) for column in frame.columns
            },
            "numeric_columns": [
                column
                for column in frame.columns
                if pd.api.types.is_numeric_dtype(frame[column])
            ],
            "row_count": len(frame),
        },
        "consumption_contract": {
            "schema_version": "easyicu.verified_artifact_consumption/1",
            "input_key": input_key,
            "mode": "all_rows",
            "artifact_sha256": digest,
            "verified_row_count": len(frame),
        },
    }


def _manifest(
    tmp_path: Path,
    *,
    cohort: pd.DataFrame | None = None,
    outcome: pd.DataFrame | None = None,
) -> tuple[Path, dict]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cohort_frame = cohort if cohort is not None else _cohort_summary()
    outcome_frame = outcome if outcome is not None else _outcome_incidence()
    return run_dir, {
        "schema_version": "2.1",
        "step_id": _step().step_id,
        "inputs": {
            PREVALENCE_MORTALITY_FIGURE_INPUTS[0]: _binding(
                run_dir=run_dir,
                input_key=PREVALENCE_MORTALITY_FIGURE_INPUTS[0],
                product="cohort_summary",
                frame=cohort_frame,
            ),
            PREVALENCE_MORTALITY_FIGURE_INPUTS[1]: _binding(
                run_dir=run_dir,
                input_key=PREVALENCE_MORTALITY_FIGURE_INPUTS[1],
                product="outcome_incidence",
                frame=outcome_frame,
            ),
        },
    }


def test_exact_two_table_contract_selects_standard_executor() -> None:
    step = _step()

    assert prevalence_mortality_figure_executor_owns_step(step)
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Test", steps=[step]),
    )

    assert selection is not None
    assert selection.analysis_kind == "prevalence_mortality_figure"
    assert selection.consumed_input_keys == PREVALENCE_MORTALITY_FIGURE_INPUTS
    assert "run_prevalence_mortality_figure" in selection.code
    assert "category_labels=('Level 0', 'Level 1')" in selection.code


def test_planner_owned_binary_level_labels_are_compiled_into_renderer() -> None:
    step = _step()
    labels = {
        "sep3_sofa2_max=0": "Sepsis-3 absent",
        "sep3_sofa2_max=1": "Sepsis-3 present",
    }

    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(
            research_question="Test",
            steps=[step],
            display_labels=labels,
        ),
    )

    assert binary_level_labels(labels) == (
        "Sepsis-3 absent",
        "Sepsis-3 present",
    )
    assert selection is not None
    assert "category_labels=('Sepsis-3 absent', 'Sepsis-3 present')" in (selection.code)


def test_owner_reads_the_same_pair_in_either_declared_order() -> None:
    """Order is a property of the Planner's list, not of the rendering.

    ``run_prevalence_mortality_figure`` looks each binding up by key and
    compares the manifest as a set, so the two tables named the other way
    round are the same request.  Refusing it sent an otherwise complete
    contract to the stochastic Coder over the order of a list.
    """

    reversed_inputs = list(reversed(PREVALENCE_MORTALITY_FIGURE_INPUTS))

    assert prevalence_mortality_figure_executor_owns_step(
        _step(
            inputs=reversed_inputs,
            input_consumption_contracts=[
                ArtifactConsumptionContract(input_key=value, mode="all_rows")
                for value in reversed_inputs
            ],
        )
    )


def test_owner_rejects_a_widened_or_narrowed_contract() -> None:
    """Set membership still decides: an extra or missing table is refused."""

    widened = [*PREVALENCE_MORTALITY_FIGURE_INPUTS, "table:unrelated_parent"]
    assert not prevalence_mortality_figure_executor_owns_step(
        _step(
            inputs=widened,
            input_consumption_contracts=[
                ArtifactConsumptionContract(input_key=value, mode="all_rows")
                for value in widened
            ],
        )
    )
    narrowed = [PREVALENCE_MORTALITY_FIGURE_INPUTS[0]]
    assert not prevalence_mortality_figure_executor_owns_step(
        _step(
            inputs=narrowed,
            input_consumption_contracts=[
                ArtifactConsumptionContract(input_key=value, mode="all_rows")
                for value in narrowed
            ],
        )
    )
    assert not prevalence_mortality_figure_executor_owns_step(
        _step(expected_outputs=["figure:prevalence_mortality", "table:other"])
    )
    assert not prevalence_mortality_figure_executor_owns_step(
        _step(planned_analysis_role="primary")
    )


def test_runner_renders_reconciled_source_backed_bundle(tmp_path: Path) -> None:
    run_dir, manifest = _manifest(tmp_path)
    out_dir = run_dir / "steps" / _step().step_id / "outputs"

    summary = run_prevalence_mortality_figure(
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=_step().step_id,
        category_labels=("Sepsis-3 absent", "Sepsis-3 present"),
    )

    assert summary["status"] == "ok"
    assert summary["rendering_only"] is True
    assert summary["locked_denominator"] == 10
    assert summary["source_rows_consumed"] == 9
    for suffix in ("png", "svg", "pdf", "tiff"):
        assert (out_dir / f"prevalence_mortality.{suffix}").is_file()
    assert (out_dir / "prevalence_mortality.figure_contract.json").is_file()
    cohort_source = pd.read_csv(
        out_dir / "prevalence_mortality_cohort_summary_source_data.csv"
    )
    outcome_source = pd.read_csv(
        out_dir / "prevalence_mortality_outcome_incidence_source_data.csv"
    )
    assert cohort_source.drop(columns=["display_label"]).equals(_cohort_summary())
    assert outcome_source.drop(columns=["display_label"]).equals(_outcome_incidence())
    assert cohort_source.loc[
        cohort_source["exposure_level"].eq(1.0),
        "display_label",
    ].tolist() == ["Sepsis-3 present"]
    assert outcome_source["display_label"].tolist() == [
        "Overall",
        "Sepsis-3 absent",
        "Sepsis-3 present",
    ]
    assert summary["category_labels"] == [
        "Sepsis-3 absent",
        "Sepsis-3 present",
    ]


def test_runner_rejects_cross_table_count_mismatch(tmp_path: Path) -> None:
    outcome = _outcome_incidence()
    outcome.loc[outcome["exposure_level"] == "1.0", "exposure_count"] = 5
    run_dir, manifest = _manifest(tmp_path, outcome=outcome)

    with pytest.raises(ValueError, match="partitions do not share authority"):
        run_prevalence_mortality_figure(
            out_dir=run_dir / "outputs",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id=_step().step_id,
        )


def test_runner_rejects_tampered_bound_table(tmp_path: Path) -> None:
    run_dir, manifest = _manifest(tmp_path)
    binding = manifest["inputs"][PREVALENCE_MORTALITY_FIGURE_INPUTS[0]]
    path = run_dir / binding["relative_path"]
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="digest verification failed"):
        run_prevalence_mortality_figure(
            out_dir=run_dir / "outputs",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id=_step().step_id,
        )
