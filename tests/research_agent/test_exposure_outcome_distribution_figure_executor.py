"""Ownership and rendering contract for the role-tagged joint-distribution figure.

The defect these tests lock: a generated script selected plotted cells by
"exposure/outcome label is non-empty", so two zero-count ``missingness`` rows
were treated as joint cells and the step died on its own guard.  Row role, not
label emptiness, decides what is plotted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.exposure_outcome_distribution_figure_executor import (
    EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUTS,
    exposure_outcome_distribution_figure_executor_code,
    exposure_outcome_distribution_figure_executor_owns_step,
    run_exposure_outcome_distribution_figure,
)
from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
)

STEP_ID = "02_cohort_summary_and_prevalence_mortality_figure"
PRODUCER_ID = "02_cohort_summary_and_prevalence_mortality"
LABELS = {
    "sep3_sofa2_max=0": "Sepsis-3 absent",
    "sep3_sofa2_max=1": "Sepsis-3 present",
    "death": "In-hospital mortality",
}
_LOCKED_N = 100


def _step(**updates) -> AnalysisStep:
    payload = {
        "step_id": STEP_ID,
        "planned_analysis_role": "auxiliary",
        "intent": "Render the parent joint exposure/outcome distribution.",
        "inputs": list(EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUTS),
        "expected_outputs": ["figure:prevalence_mortality"],
        "method": "visualization",
        "input_consumption_contracts": [
            ArtifactConsumptionContract(input_key=input_key, mode="all_rows")
            for input_key in EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUTS
        ],
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


def _cell(row_type: str, exposure: str, outcome: str, count: int) -> list:
    return [
        row_type,
        "sep3_sofa2_max",
        exposure,
        "death",
        outcome,
        count,
        100.0 * count / _LOCKED_N,
        _LOCKED_N,
    ]


def _distribution(rows: list[list] | None = None) -> pd.DataFrame:
    payload = (
        rows
        if rows is not None
        else [
            _cell("joint_distribution", "0.0", "No in-hospital death", 55),
            _cell("joint_distribution", "0.0", "In-hospital death", 5),
            _cell("joint_distribution", "1.0", "No in-hospital death", 30),
            _cell("joint_distribution", "1.0", "In-hospital death", 10),
            _cell("missingness", "Missing", "All", 0),
            _cell("missingness", "Observed", "Missing", 0),
        ]
    )
    return pd.DataFrame(
        payload,
        columns=[
            "row_type",
            "exposure_variable",
            "exposure_category",
            "outcome_variable",
            "outcome_category",
            "count",
            "percentage_of_locked_cohort",
            "denominator_n",
        ],
    )


def _cohort_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["cohort", "icu_stays", float(_LOCKED_N), _LOCKED_N],
            ["sep3_sofa2_max", "valid_observed_n", float(_LOCKED_N), _LOCKED_N],
            ["sep3_sofa2_max", "missing_n", 0.0, _LOCKED_N],
            ["sep3_sofa2_max", "count:0.0", 60.0, _LOCKED_N],
            ["sep3_sofa2_max", "count:1.0", 40.0, _LOCKED_N],
        ],
        columns=["variable", "metric", "value", "denominator_n"],
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
        "produced_by_step": PRODUCER_ID,
        "product": product,
        "identity_row": {
            "declared_kind": "table",
            "evidence_id": evidence_id,
            "input_key": input_key,
            "produced_by_step": PRODUCER_ID,
            "product": product,
            "sha256": digest,
        },
        "product_contract": {
            "schema_version": "easyicu.host_typed_product.v4",
            "tabular_format": "csv",
            "columns": list(frame.columns),
            "column_count": len(frame.columns),
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
    distribution: pd.DataFrame | None = None,
    cohort: pd.DataFrame | None = None,
) -> tuple[Path, dict]:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    return run_dir, {
        "schema_version": "2.1",
        "step_id": STEP_ID,
        "inputs": {
            EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUTS[0]: _binding(
                run_dir=run_dir,
                input_key=EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUTS[0],
                product="cohort_summary",
                frame=cohort if cohort is not None else _cohort_summary(),
            ),
            EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUTS[1]: _binding(
                run_dir=run_dir,
                input_key=EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUTS[1],
                product="exposure_outcome_distribution",
                frame=distribution if distribution is not None else _distribution(),
            ),
        },
    }


def _render(tmp_path: Path, **manifest_kwargs) -> tuple[Path, dict]:
    run_dir, manifest = _manifest(tmp_path, **manifest_kwargs)
    out_dir = run_dir / "steps" / STEP_ID / "outputs"
    summary = run_exposure_outcome_distribution_figure(
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=STEP_ID,
        figure_product="prevalence_mortality",
        display_labels=LABELS,
    )
    return out_dir, dict(summary)


# --------------------------------------------------------------------------
# Selection: the exact typed contract, confirmed against the bound schema
# --------------------------------------------------------------------------


def test_exact_typed_contract_and_bound_schema_select_the_executor(
    tmp_path: Path,
) -> None:
    step = _step()
    _, manifest = _manifest(tmp_path)

    assert exposure_outcome_distribution_figure_executor_owns_step(
        step,
        resolved_bindings=manifest["inputs"],
        display_labels=LABELS,
    )
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(
            research_question="Test",
            steps=[step],
            display_labels=LABELS,
        ),
        resolved_bindings=manifest["inputs"],
    )

    assert selection is not None
    assert selection.analysis_kind == "exposure_outcome_distribution_figure"
    assert selection.consumed_input_keys == (
        EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUTS
    )
    assert "run_exposure_outcome_distribution_figure" in selection.code
    assert "Sepsis-3 present" in selection.code


def test_planner_product_name_alone_never_selects_the_executor() -> None:
    """The Planner names a product; only the host's contract proves its shape."""

    step = _step()

    assert not exposure_outcome_distribution_figure_executor_owns_step(
        step, display_labels=LABELS
    )
    assert (
        select_standard_executor(
            step,
            plan=AnalysisPlan(
                research_question="Test", steps=[step], display_labels=LABELS
            ),
        )
        is None
    )


def test_owner_rejects_widened_or_mistyped_plan_contracts(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path)
    bindings = manifest["inputs"]

    def owns(step: AnalysisStep) -> bool:
        return exposure_outcome_distribution_figure_executor_owns_step(
            step, resolved_bindings=bindings, display_labels=LABELS
        )

    widened = [*EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUTS, "table:extra"]
    assert not owns(
        _step(
            inputs=widened,
            input_consumption_contracts=[
                ArtifactConsumptionContract(input_key=value, mode="all_rows")
                for value in widened
            ],
        )
    )
    narrowed = [EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUTS[0]]
    assert not owns(
        _step(
            inputs=narrowed,
            input_consumption_contracts=[
                ArtifactConsumptionContract(input_key=value, mode="all_rows")
                for value in narrowed
            ],
        )
    )
    assert not owns(_step(expected_outputs=["table:prevalence_mortality"]))
    assert not owns(
        _step(expected_outputs=["figure:prevalence_mortality", "figure:other"])
    )
    assert not owns(_step(planned_analysis_role="primary"))
    assert not owns(_step(method="descriptive_epidemiology"))
    assert not owns(
        _step(
            input_consumption_contracts=[
                ArtifactConsumptionContract(
                    input_key=EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUTS[0],
                    mode="all_rows",
                ),
                ArtifactConsumptionContract(
                    input_key=EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUTS[1],
                    mode="one_per_role",
                    role_column="row_type",
                    expected_roles=["joint_distribution"],
                ),
            ]
        )
    )


def test_owner_reads_the_same_pair_in_either_declared_order(tmp_path: Path) -> None:
    """The bound schema decides, not the order the Planner listed the tables."""

    _, manifest = _manifest(tmp_path)
    reordered = list(reversed(EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUTS))

    assert exposure_outcome_distribution_figure_executor_owns_step(
        _step(
            inputs=reordered,
            input_consumption_contracts=[
                ArtifactConsumptionContract(input_key=value, mode="all_rows")
                for value in reordered
            ],
        ),
        resolved_bindings=manifest["inputs"],
        display_labels=LABELS,
    )


def test_owner_rejects_an_unfamiliar_bound_schema(tmp_path: Path) -> None:
    renamed = _distribution().rename(columns={"count": "n"})
    _, manifest = _manifest(tmp_path, distribution=renamed)

    assert not exposure_outcome_distribution_figure_executor_owns_step(
        _step(), resolved_bindings=manifest["inputs"], display_labels=LABELS
    )

    _, wrong_summary = _manifest(
        tmp_path / "second",
        cohort=_cohort_summary().rename(columns={"metric": "statistic"}),
    )
    assert not exposure_outcome_distribution_figure_executor_owns_step(
        _step(), resolved_bindings=wrong_summary["inputs"], display_labels=LABELS
    )


def test_owner_declines_without_one_planner_label_pair(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path)

    for labels in (
        None,
        {},
        {"sep3_sofa2_max=1": "Sepsis-3 present"},
        {
            "sep3_sofa2_max=0": "Sepsis-3 absent",
            "sep3_sofa2_max=1": "Sepsis-3 present",
            "aki=0": "AKI absent",
            "aki=1": "AKI present",
        },
    ):
        assert not exposure_outcome_distribution_figure_executor_owns_step(
            _step(), resolved_bindings=manifest["inputs"], display_labels=labels
        )


def test_a_missing_outcome_label_renders_a_reader_title_not_a_machine_name() -> None:
    """One module-level name, one definition.

    ``_outcome_title`` was defined twice; the later definition shadowed the
    earlier one for every caller and fell back to the raw variable, so a step
    whose Planner supplied no outcome display label titled its figure
    ``in_hospital_death``.  A duplicate definition is invisible at the call
    site, so lock the behaviour rather than the line count.
    """

    from easyicu.research_agent.execution.runners import (
        exposure_outcome_distribution_figure_executor as mod,
    )

    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert source.count("\ndef _outcome_title(") == 1

    assert mod._outcome_title(None, "in_hospital_death") == "In hospital death"
    assert mod._outcome_title({}, "in_hospital_death") == "In hospital death"
    assert (
        mod._outcome_title(
            {"in_hospital_death": "In-hospital mortality"}, "in_hospital_death"
        )
        == "In-hospital mortality"
    )


def test_code_generation_refuses_an_unowned_step() -> None:
    with pytest.raises(ValueError, match="not owned"):
        exposure_outcome_distribution_figure_executor_code(
            _step(), display_labels=LABELS
        )


# --------------------------------------------------------------------------
# Rendering: role selection, reconciliation and traceable source data
# --------------------------------------------------------------------------


def test_accounting_rows_never_enter_a_plotted_cell(tmp_path: Path) -> None:
    out_dir, summary = _render(tmp_path)

    assert summary["status"] == "ok"
    assert summary["joint_cell_rows"] == 4
    assert summary["accounting_rows_excluded"] == 2
    assert summary["plotted_row_role"] == "joint_distribution"
    assert summary["exposure_category_counts"] == [60, 40]
    assert summary["locked_denominator"] == _LOCKED_N
    # The regression: the two zero-count missingness rows are labelled, so a
    # "non-empty label" filter would have plotted them as exposure categories.
    assert summary["category_labels"] == ["Sepsis-3 absent", "Sepsis-3 present"]
    assert "Missing" not in summary["category_labels"]
    assert "Observed" not in summary["category_labels"]


def test_source_data_keeps_every_upstream_row_and_its_index(tmp_path: Path) -> None:
    out_dir, summary = _render(tmp_path)

    source = pd.read_csv(out_dir / "prevalence_mortality_distribution_source_data.csv")
    assert list(source["source_row_index"]) == [0, 1, 2, 3, 4, 5]
    assert set(source["source_table"]) == {"exposure_outcome_distribution.csv"}
    assert list(source["row_type"]).count("missingness") == 2
    assert int(
        source.loc[source["row_type"].eq("joint_distribution"), "count"].sum()
    ) == (_LOCKED_N)
    assert not any(column.startswith("__") for column in source.columns)


def test_publication_bundle_and_contract_are_complete(tmp_path: Path) -> None:
    out_dir, summary = _render(tmp_path)

    for suffix in ("png", "svg", "pdf", "tiff"):
        assert (out_dir / f"prevalence_mortality.{suffix}").is_file()
    contract = json.loads(
        (out_dir / "prevalence_mortality.figure_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["figure_id"] == "figure:prevalence_mortality"
    assert [panel["panel_id"] for panel in contract["panels"]] == ["A", "B"]
    assert summary["outcome_levels"] == [
        "No in-hospital death",
        "In-hospital death",
    ]
    assert "Category 0" not in json.dumps(contract)
    assert "Level 0" not in json.dumps(contract)


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        pytest.param(
            [
                _cell("joint_distribution", "0.0", "No in-hospital death", 55),
                _cell("joint_distribution", "0.0", "In-hospital death", 5),
                _cell("joint_distribution", "1.0", "No in-hospital death", 30),
                _cell("joint_distribution", "1.0", "In-hospital death", 10),
                _cell("summary_total", "All", "All", 0),
            ],
            "unknown distribution row roles",
            id="unknown_row_role",
        ),
        pytest.param(
            [
                _cell("joint_distribution", "0.0", "No in-hospital death", 55),
                _cell("joint_distribution", "0.0", "In-hospital death", 5),
                _cell("joint_distribution", "1.0", "No in-hospital death", 40),
            ],
            "incomplete or contains duplicate cells",
            id="incomplete_grid",
        ),
        pytest.param(
            [
                _cell("joint_distribution", "0.0", "No in-hospital death", 50),
                _cell("joint_distribution", "0.0", "In-hospital death", 5),
                _cell("joint_distribution", "1.0", "No in-hospital death", 30),
                _cell("joint_distribution", "1.0", "In-hospital death", 10),
                _cell("missingness", "Missing", "All", 5),
                _cell("missingness", "Observed", "Missing", 0),
            ],
            "positive count",
            id="positive_missingness",
        ),
        pytest.param(
            [
                _cell("joint_distribution", "0.0", "No in-hospital death", 55),
                _cell("joint_distribution", "0.0", "In-hospital death", 5),
                _cell("joint_distribution", "1.0", "No in-hospital death", 30),
                _cell("joint_distribution", "1.0", "In-hospital death", 5),
            ],
            "do not sum to the locked denominator",
            id="counts_below_denominator",
        ),
        pytest.param(
            [
                _cell("joint_distribution", "0.0", "No in-hospital death", 55),
                _cell("joint_distribution", "0.0", "In-hospital death", 5),
                _cell("joint_distribution", "1.0", "No in-hospital death", 30),
                [
                    "joint_distribution",
                    "sep3_sofa2_max",
                    "1.0",
                    "death",
                    "In-hospital death",
                    10,
                    99.0,
                    _LOCKED_N,
                ],
            ],
            "percentages do not reconcile",
            id="percentage_drift",
        ),
        pytest.param(
            [
                _cell("joint_distribution", "low", "No in-hospital death", 55),
                _cell("joint_distribution", "low", "In-hospital death", 5),
                _cell("joint_distribution", "high", "No in-hospital death", 30),
                _cell("joint_distribution", "high", "In-hospital death", 10),
            ],
            "does not declare that exposure coding",
            id="non_binary_exposure_coding",
        ),
    ],
)
def test_renderer_fails_closed_on_an_unreadable_grid(
    tmp_path: Path,
    rows: list[list],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _render(tmp_path, distribution=_distribution(rows))


def test_renderer_refuses_labels_bound_to_another_column(tmp_path: Path) -> None:
    run_dir, manifest = _manifest(tmp_path)

    with pytest.raises(ValueError, match="different column"):
        run_exposure_outcome_distribution_figure(
            out_dir=run_dir / "out",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id=STEP_ID,
            figure_product="prevalence_mortality",
            display_labels={"aki_stage=0": "AKI absent", "aki_stage=1": "AKI present"},
        )


def test_renderer_refuses_a_manifest_for_another_step(tmp_path: Path) -> None:
    run_dir, manifest = _manifest(tmp_path)
    manifest["step_id"] = "07_other_step"

    with pytest.raises(ValueError, match="does not belong to this step"):
        run_exposure_outcome_distribution_figure(
            out_dir=run_dir / "out",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id=STEP_ID,
            figure_product="prevalence_mortality",
            display_labels=LABELS,
        )


def test_renderer_refuses_a_table_whose_bytes_changed(tmp_path: Path) -> None:
    run_dir, manifest = _manifest(tmp_path)
    binding = manifest["inputs"][EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUTS[1]]
    (run_dir / binding["relative_path"]).write_text(
        _distribution().to_csv(index=False).replace("55", "56"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="digest verification failed"):
        run_exposure_outcome_distribution_figure(
            out_dir=run_dir / "out",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id=STEP_ID,
            figure_product="prevalence_mortality",
            display_labels=LABELS,
        )
