"""The renderer for the self-contained exposure-outcome distribution product.

The load-bearing claim is that it needs **one** table. These tests build the
table with the real producer, hand only that to the renderer, and check a
figure comes out -- no cohort summary, no second binding.

As with the executor, the case is deliberately not the benchmark item.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
    EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS,
    run_exposure_outcome_distribution_from_env,
)
from easyicu.research_agent.execution.runners.exposure_outcome_distribution_render import (
    exposure_outcome_distribution_figure_owns_step,
    run_exposure_outcome_distribution_figure,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
)

INPUT_KEY = "table:exposure_outcome_distribution"
STEP_ID = "04_drug_readmission_distribution_figure"
PRODUCT = "exposure_overview"
EXPOSURE = "anticoagulant_exposed"
OUTCOME = "readmitted_30d"


def _step(**updates) -> AnalysisStep:
    payload = {
        "step_id": STEP_ID,
        "planned_analysis_role": "auxiliary",
        "method": "visualization",
        "intent": "Render the distribution declared by the parent step.",
        "inputs": [INPUT_KEY],
        "expected_outputs": [f"figure:{PRODUCT}"],
        "input_consumption_contracts": [
            ArtifactConsumptionContract(input_key=INPUT_KEY, mode="all_rows")
        ],
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


def _produced_table(tmp_path: Path, monkeypatch) -> Path:
    """Build the product with the real producer, not a hand-written fixture."""

    frame = pd.DataFrame(
        {
            EXPOSURE: [1] * 10 + [0] * 10,
            OUTCOME: (
                [1, 1, 1, 0, 0, 0, 0, 0, 0, None] + [1, 0, 0, 0, 0, 0, 0, 0, 0, None]
            ),
        }
    )
    parent = tmp_path / "parent"
    parent_out = parent / "steps" / "03_parent" / "outputs"
    parent_out.mkdir(parents=True)
    cohort = parent / "cohort.parquet"
    frame.to_parquet(cohort, index=False)
    (parent / "resolved_inputs.json").write_text(
        json.dumps(
            {
                "inputs": {
                    "artifact:analysis_cohort": {
                        "relative_path": "cohort.parquet",
                        "sha256": hashlib.sha256(cohort.read_bytes()).hexdigest(),
                        "product_contract": {
                            "columns": list(frame.columns),
                            "row_count": int(len(frame)),
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("STEP_OUT_DIR", str(parent_out))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(parent))
    monkeypatch.setenv(
        "EASYICU_RESOLVED_INPUTS_JSON", str(parent / "resolved_inputs.json")
    )
    run_exposure_outcome_distribution_from_env(
        spec_payload={
            "exposure": EXPOSURE,
            "exposure_levels": [0, 1],
            "outcome": OUTCOME,
            "outcome_positive_value": 1,
            "denominator_policy": "all_declared_rows",
        },
        typed_cohort_input="artifact:analysis_cohort",
    )
    return parent_out / "exposure_outcome_distribution.csv"


def _bound(
    tmp_path: Path, table: Path, *, rows: int | None = None
) -> tuple[Path, dict]:
    run_dir = tmp_path / "run"
    (run_dir / "inputs").mkdir(parents=True)
    bound = run_dir / "inputs" / "exposure_outcome_distribution.csv"
    bound.write_bytes(table.read_bytes())
    digest = hashlib.sha256(bound.read_bytes()).hexdigest()
    frame = pd.read_csv(bound)
    manifest = {
        "step_id": STEP_ID,
        "inputs": {
            INPUT_KEY: {
                "relative_path": "inputs/exposure_outcome_distribution.csv",
                "sha256": digest,
                "declared_kind": "table",
                "product_contract": {
                    "columns": list(EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS),
                    "row_count": rows if rows is not None else int(len(frame)),
                },
                "consumption_contract": {
                    "input_key": INPUT_KEY,
                    "mode": "all_rows",
                    "artifact_sha256": digest,
                },
            }
        },
    }
    return run_dir, manifest


# --------------------------------------------------------------------------


def test_the_renderer_is_owned_and_selected() -> None:
    step = _step()
    assert exposure_outcome_distribution_figure_owns_step(step)
    selection = select_standard_executor(
        step, plan=AnalysisPlan(research_question="Test", steps=[step])
    )
    assert selection is not None
    assert selection.analysis_kind == "exposure_outcome_distribution_figure"
    assert selection.consumed_input_keys == (INPUT_KEY,)


def test_any_legal_product_label_is_owned() -> None:
    """L0's rule, held here from the start rather than added later."""

    for product in ("prevalence_mortality", "measurement_overview", "f2"):
        assert exposure_outcome_distribution_figure_owns_step(
            _step(expected_outputs=[f"figure:{product}"])
        )


def test_an_unsafe_label_or_a_widened_input_is_refused() -> None:
    assert not exposure_outcome_distribution_figure_owns_step(
        _step(expected_outputs=["figure:../../escape"])
    )
    assert not exposure_outcome_distribution_figure_owns_step(
        _step(
            inputs=[INPUT_KEY, "table:cohort_summary"],
            input_consumption_contracts=[
                ArtifactConsumptionContract(input_key=key, mode="all_rows")
                for key in (INPUT_KEY, "table:cohort_summary")
            ],
        )
    )
    assert not exposure_outcome_distribution_figure_owns_step(
        _step(planned_analysis_role="primary")
    )


def test_it_renders_from_the_one_table_alone(tmp_path: Path, monkeypatch) -> None:
    """The whole point of the self-contained product."""

    table = _produced_table(tmp_path, monkeypatch)
    run_dir, manifest = _bound(tmp_path, table)
    out_dir = tmp_path / "figure_out"
    summary = run_exposure_outcome_distribution_figure(
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=STEP_ID,
        figure_product=PRODUCT,
        level_labels=("Unexposed", "Exposed"),
    )
    assert summary["status"] == "ok"
    assert summary["cohort_n"] == 20
    assert (out_dir / f"{PRODUCT}.png").exists()
    assert (out_dir / f"{PRODUCT}.figure_contract.json").exists()

    # Source data is emitted for every panel, and the denominators and the
    # unobserved count travel with it -- that is what removes the second table.
    outcome_source = pd.read_csv(out_dir / f"{PRODUCT}_outcome_source_data.csv")
    assert {"outcome_denominator", "outcome_missing_n", "ci_low_pct"} <= set(
        outcome_source.columns
    )
    assert int(outcome_source["outcome_missing_n"].sum()) == 2

    contract = json.loads((out_dir / f"{PRODUCT}.figure_contract.json").read_text())
    assert [panel["panel_id"] for panel in contract["panels"]] == ["A", "B"]


def test_a_tampered_table_fails_closed(tmp_path: Path, monkeypatch) -> None:
    table = _produced_table(tmp_path, monkeypatch)
    run_dir, manifest = _bound(tmp_path, table)
    manifest["inputs"][INPUT_KEY]["sha256"] = "0" * 64
    manifest["inputs"][INPUT_KEY]["consumption_contract"]["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest verification failed"):
        run_exposure_outcome_distribution_figure(
            out_dir=tmp_path / "out",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id=STEP_ID,
            figure_product=PRODUCT,
        )


def test_a_row_count_disagreeing_with_the_contract_fails_closed(
    tmp_path: Path, monkeypatch
) -> None:
    table = _produced_table(tmp_path, monkeypatch)
    run_dir, manifest = _bound(tmp_path, table, rows=99)
    with pytest.raises(ValueError, match="disagree with its product contract"):
        run_exposure_outcome_distribution_figure(
            out_dir=tmp_path / "out",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id=STEP_ID,
            figure_product=PRODUCT,
        )


def test_a_table_whose_levels_do_not_partition_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    """The renderer re-checks the arithmetic instead of drawing it on trust."""

    table = _produced_table(tmp_path, monkeypatch)
    frame = pd.read_csv(table)
    frame.loc[frame["row_role"] == "overall", "n_rows"] = 999
    frame.to_csv(table, index=False)
    run_dir, manifest = _bound(tmp_path, table)
    with pytest.raises(ValueError, match="do not partition the reported cohort"):
        run_exposure_outcome_distribution_figure(
            out_dir=tmp_path / "out",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id=STEP_ID,
            figure_product=PRODUCT,
        )


def test_a_manifest_for_another_step_is_refused(tmp_path: Path, monkeypatch) -> None:
    table = _produced_table(tmp_path, monkeypatch)
    run_dir, manifest = _bound(tmp_path, table)
    manifest["step_id"] = "99_someone_elses_step"
    with pytest.raises(ValueError, match="does not belong to this step"):
        run_exposure_outcome_distribution_figure(
            out_dir=tmp_path / "out",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id=STEP_ID,
            figure_product=PRODUCT,
        )


def test_the_renderer_carries_no_case_specific_branch() -> None:
    import easyicu.research_agent.execution.runners.exposure_outcome_distribution_render as module

    source = Path(module.__file__).read_text().lower()
    for token in ("sepsis", "sep3", "e1_", "icu_readmission", "94,458"):
        assert token not in source, f"case-specific token in production: {token}"
