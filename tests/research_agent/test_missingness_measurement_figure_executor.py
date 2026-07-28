from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from easyicu.research_agent.audits.validators import (
    FigureContractQualityValidator,
    FigureSourceDataValidator,
)
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.execution.runners.missingness_measurement_figure_executor import (
    MEASUREMENT_PROCESS_AUDIT_INPUT,
    MISSINGNESS_MEASUREMENT_AUDIT_INPUT,
    MISSINGNESS_MEASUREMENT_FIGURE_INPUTS,
    _validate_audit_rows,
    _validate_process_rows,
    missingness_measurement_figure_executor_owns_step,
    run_missingness_measurement_figure,
)
from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
)

STEP_ID = "05_missingness_measurement_audit_figure"
PARENT_STEP = "05_missingness_measurement_audit"
PRODUCT = "data_quality"

_AUDIT_COLUMNS = [
    "variable",
    "metric",
    "level",
    "count",
    "percentage",
    "denominator",
    "valid_observed_denominator",
    "raw_nonfinite_n",
    "plausibility_flag_n",
    "summary_value",
    "q1",
    "q3",
    "notes",
]
_PROCESS_COLUMNS = [
    "variable",
    "process_measure",
    "level",
    "count",
    "percentage",
    "denominator",
    "valid_observed_denominator",
    "median",
    "q1",
    "q3",
    "nonfinite_n",
    "notes",
]
_N = 1000


def _step(**updates) -> AnalysisStep:
    payload = {
        "step_id": STEP_ID,
        "planned_analysis_role": "auxiliary",
        "intent": "Render the registered missingness and measurement-process audit.",
        "inputs": list(MISSINGNESS_MEASUREMENT_FIGURE_INPUTS),
        "expected_outputs": [f"figure:{PRODUCT}"],
        "method": "visualization",
        "input_consumption_contracts": [
            ArtifactConsumptionContract(input_key=key, mode="all_rows")
            for key in MISSINGNESS_MEASUREMENT_FIGURE_INPUTS
        ],
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


def _numeric_variable_rows(variable: str, missing: int) -> list[list]:
    """Emit the real four-row shape: two tallies plus two distribution summaries."""

    observed = _N - missing
    return [
        # Counting rows carry count/percentage and no distribution summary.
        [
            variable,
            "missing",
            "",
            float(missing),
            100.0 * missing / _N,
            _N,
            float(_N),
            0.0,
            0.0,
            None,
            None,
            None,
            "source missingness",
        ],
        [
            variable,
            "valid_observed",
            "",
            float(observed),
            100.0 * observed / _N,
            _N,
            float(_N),
            0.0,
            0.0,
            None,
            None,
            None,
            "finite observed",
        ],
        # Distribution rows carry summary_value/q1/q3 and NO count by design.
        [
            variable,
            "median",
            "",
            None,
            None,
            _N,
            float(_N),
            0.0,
            0.0,
            1.0,
            0.0,
            2.0,
            "median among observed",
        ],
        [
            variable,
            "iqr",
            "",
            None,
            None,
            _N,
            float(_N),
            0.0,
            0.0,
            2.0,
            0.0,
            2.0,
            "iqr among observed",
        ],
    ]


def _audit_frame() -> pd.DataFrame:
    rows = _numeric_variable_rows("lact_first", missing=400)
    rows += _numeric_variable_rows("sep3_sofa2_max", missing=0)
    # A categorical variable closes through level_distribution rows instead.
    rows += [
        [
            "sex",
            "missing",
            "",
            0.0,
            0.0,
            _N,
            float(_N),
            None,
            None,
            None,
            None,
            None,
            "source missingness",
        ],
        [
            "sex",
            "level_distribution",
            "Female",
            440.0,
            44.0,
            _N,
            float(_N),
            None,
            None,
            None,
            None,
            None,
            "category share",
        ],
        [
            "sex",
            "level_distribution",
            "Male",
            560.0,
            56.0,
            _N,
            float(_N),
            None,
            None,
            None,
            None,
            None,
            "category share",
        ],
    ]
    return pd.DataFrame(rows, columns=_AUDIT_COLUMNS)


def _process_frame() -> pd.DataFrame:
    rows = [
        [
            "lact_n",
            "count_missing",
            "",
            0,
            0.0,
            _N,
            float(_N),
            0.0,
            0.0,
            1.0,
            0.0,
            "count missingness",
        ],
        [
            "lact_n",
            "count_frequency_summary",
            "",
            _N,
            100.0,
            _N,
            float(_N),
            1.0,
            0.0,
            2.0,
            0.0,
            "count frequency",
        ],
        # One measure split across levels: the levels partition its denominator.
        [
            "lact_measured",
            "measured_status",
            "0.0",
            400,
            40.0,
            _N,
            float(_N),
            None,
            None,
            None,
            0.0,
            "not measured",
        ],
        [
            "lact_measured",
            "measured_status",
            "1.0",
            600,
            60.0,
            _N,
            float(_N),
            None,
            None,
            None,
            0.0,
            "measured",
        ],
        [
            "sep3_sofa2_max",
            "count_positive_rows",
            "",
            360,
            36.0,
            _N,
            None,
            None,
            None,
            None,
            None,
            "positive rows",
        ],
        [
            "sep3_sofa2_max",
            "count_zero_rows",
            "",
            640,
            64.0,
            _N,
            None,
            None,
            None,
            None,
            None,
            "zero rows",
        ],
    ]
    return pd.DataFrame(rows, columns=_PROCESS_COLUMNS)


def _register(
    run_dir: Path,
    frame: pd.DataFrame,
    *,
    input_key: str,
    product: str,
) -> dict:
    parent_output = run_dir / "steps" / PARENT_STEP / "outputs" / f"{product}.csv"
    parent_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(parent_output, index=False)
    record = EvidenceStore(run_dir).register_file(
        kind="table",
        description=f"Digest-bound {product}.",
        source_path=parent_output,
        evidence_id=f"table_{product}",
        produced_by_step=PARENT_STEP,
        producer="deterministic_test",
        generation_mode="deterministic_standard",
    )
    table = run_dir / record.relative_path
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    columns = list(frame.columns)
    return {
        "absolute_path": str(table),
        "relative_path": str(table.relative_to(run_dir)),
        "sha256": digest,
        "declared_kind": "table",
        "evidence_kind": "table",
        "evidence_id": record.evidence_id,
        "produced_by_step": PARENT_STEP,
        "product": product,
        "identity_row": {
            "declared_kind": "table",
            "evidence_id": record.evidence_id,
            "input_key": input_key,
            "produced_by_step": PARENT_STEP,
            "product": product,
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
            "input_key": input_key,
            "mode": "all_rows",
            "artifact_sha256": digest,
            "verified_row_count": len(frame),
        },
    }


def _binding(
    tmp_path: Path,
    audit: pd.DataFrame | None = None,
    process: pd.DataFrame | None = None,
) -> tuple[Path, dict]:
    run_dir = tmp_path / "run"
    manifest = {
        "schema_version": "2.1",
        "step_id": STEP_ID,
        "inputs": {
            MISSINGNESS_MEASUREMENT_AUDIT_INPUT: _register(
                run_dir,
                _audit_frame() if audit is None else audit,
                input_key=MISSINGNESS_MEASUREMENT_AUDIT_INPUT,
                product="missingness_measurement_audit",
            ),
            MEASUREMENT_PROCESS_AUDIT_INPUT: _register(
                run_dir,
                _process_frame() if process is None else process,
                input_key=MEASUREMENT_PROCESS_AUDIT_INPUT,
                product="measurement_process_audit",
            ),
        },
    }
    return run_dir, manifest


def _run(run_dir: Path, manifest: dict) -> tuple[Path, dict]:
    out_dir = run_dir / "steps" / STEP_ID / "outputs"
    summary = run_missingness_measurement_figure(
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=STEP_ID,
        figure_product=PRODUCT,
    )
    return out_dir, summary


def test_exact_closed_contract_selects_standard_executor() -> None:
    step = _step()
    assert missingness_measurement_figure_executor_owns_step(step)
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Test", steps=[step]),
    )
    assert selection is not None
    assert selection.analysis_kind == "missingness_measurement_figure"
    assert selection.consumed_input_keys == MISSINGNESS_MEASUREMENT_FIGURE_INPUTS


def test_owner_is_order_insensitive_but_never_widened() -> None:
    assert missingness_measurement_figure_executor_owns_step(
        _step(inputs=list(reversed(MISSINGNESS_MEASUREMENT_FIGURE_INPUTS)))
    )
    assert not missingness_measurement_figure_executor_owns_step(
        _step(
            inputs=[*MISSINGNESS_MEASUREMENT_FIGURE_INPUTS, "table:other"],
            input_consumption_contracts=[
                ArtifactConsumptionContract(input_key=key, mode="all_rows")
                for key in [*MISSINGNESS_MEASUREMENT_FIGURE_INPUTS, "table:other"]
            ],
        )
    )
    assert not missingness_measurement_figure_executor_owns_step(
        _step(
            inputs=[MISSINGNESS_MEASUREMENT_AUDIT_INPUT],
            input_consumption_contracts=[
                ArtifactConsumptionContract(
                    input_key=MISSINGNESS_MEASUREMENT_AUDIT_INPUT,
                    mode="all_rows",
                )
            ],
        )
    )


def test_owner_rejects_unbound_or_scientific_contracts() -> None:
    assert not missingness_measurement_figure_executor_owns_step(
        _step(planned_analysis_role="primary")
    )
    assert not missingness_measurement_figure_executor_owns_step(
        _step(method="adjusted_association_models")
    )
    assert not missingness_measurement_figure_executor_owns_step(
        _step(expected_outputs=["table:missingness_measurement_audit"])
    )
    assert not missingness_measurement_figure_executor_owns_step(
        _step(expected_outputs=[f"figure:{PRODUCT}", "figure:extra"])
    )
    assert not missingness_measurement_figure_executor_owns_step(
        _step(
            input_consumption_contracts=[
                ArtifactConsumptionContract(
                    input_key=MISSINGNESS_MEASUREMENT_AUDIT_INPUT,
                    mode="all_rows",
                ),
                ArtifactConsumptionContract(
                    input_key=MEASUREMENT_PROCESS_AUDIT_INPUT,
                    mode="one_per_role",
                    role_column="metric",
                    expected_roles=["missing"],
                ),
            ]
        )
    )
    # A model requirement cannot even be attached to a visualization step, so
    # the executor's ``not step.model_requirements`` guard is a second fence
    # behind the schema rather than the only one.
    with pytest.raises(ValidationError, match="model_requirements are currently"):
        _step(
            model_requirements=[
                {
                    "requirement_id": "r1",
                    "outcome": "death",
                    "outcome_type": "binary",
                    "method_family": "logistic_regression",
                    "analysis_role": "primary",
                    "exposure_source": "sep3_sofa2_max",
                    "analysis_set": "source_aware",
                }
            ]
        )


def test_distribution_summary_rows_are_not_read_as_broken_accounting(
    tmp_path: Path,
) -> None:
    """The exact real-run regression: median/IQR rows carry no count by design."""

    audit = _audit_frame()
    summary_rows = audit["metric"].isin({"median", "iqr"})
    assert summary_rows.sum() == 4
    assert audit.loc[summary_rows, "count"].isna().all()

    run_dir, manifest = _binding(tmp_path, audit=audit)
    out_dir, summary = _run(run_dir, manifest)

    assert summary["status"] == "ok"
    assert summary["source_rows_consumed"] == {
        MISSINGNESS_MEASUREMENT_AUDIT_INPUT: len(audit),
        MEASUREMENT_PROCESS_AUDIT_INPUT: len(_process_frame()),
    }
    assert (out_dir / f"{PRODUCT}.png").is_file()


def test_runner_renders_complete_source_backed_bundle(tmp_path: Path) -> None:
    run_dir, manifest = _binding(tmp_path)
    out_dir, summary = _run(run_dir, manifest)

    assert summary["status"] == "ok"
    assert summary["audited_variable_count"] == 3
    assert summary["measurement_process_cell_count"] == 6
    for suffix in ("png", "svg", "pdf", "tiff"):
        assert (out_dir / f"{PRODUCT}.{suffix}").is_file()

    audit_source = pd.read_csv(out_dir / f"{PRODUCT}_missingness_source_data.csv")
    assert audit_source["source_row_index"].tolist() == list(range(11))
    process_source = pd.read_csv(
        out_dir / f"{PRODUCT}_measurement_process_source_data.csv"
    )
    assert process_source["source_row_index"].tolist() == list(range(6))

    # The panel projection is a verbatim row subset of the parent: every value
    # and every row position is the parent's own, so it stays traceable.
    panel = pd.read_csv(out_dir / f"{PRODUCT}_source_missingness_panel_source_data.csv")
    assert panel["variable"].tolist() == ["lact_first", "sep3_sofa2_max", "sex"]
    assert panel["metric"].tolist() == ["missing"] * 3
    assert panel["count"].tolist() == [400.0, 0.0, 0.0]
    assert panel["percentage"].tolist() == [40.0, 0.0, 0.0]
    assert panel["source_row_index"].tolist() == [0, 4, 8]
    parent = pd.read_csv(out_dir / f"{PRODUCT}_missingness_source_data.csv")
    for _, row in panel.iterrows():
        origin = parent.loc[parent["source_row_index"] == row["source_row_index"]]
        assert origin["count"].tolist() == [row["count"]]
        assert origin["percentage"].tolist() == [row["percentage"]]

    contract = json.loads(
        (out_dir / f"{PRODUCT}.figure_contract.json").read_text(encoding="utf-8")
    )
    assert [panel["role"] for panel in contract["panels"]] == [
        "data_quality",
        "data_quality",
    ]
    assert [panel["metadata"]["chart_type"] for panel in contract["panels"]] == [
        "availability_panel",
        "coverage_heatmap",
    ]

    step = _step()
    assert not [
        finding
        for finding in FigureSourceDataValidator().audit(
            step=step,
            out_dir=out_dir,
            run_dir=run_dir,
            step_summary=summary,
        )
        if finding.severity == "error"
    ]
    assert not [
        finding
        for finding in FigureContractQualityValidator().audit(
            step=step,
            out_dir=out_dir,
            run_dir=run_dir,
            step_summary=summary,
        )
        if finding.severity == "error"
    ]


def test_a_summary_row_reported_as_a_tally_is_rejected(tmp_path: Path) -> None:
    audit = _audit_frame()
    audit.loc[audit["metric"].eq("median"), "count"] = 5.0
    run_dir, manifest = _binding(tmp_path, audit=audit)
    with pytest.raises(ValueError, match="as a tally"):
        _run(run_dir, manifest)
    assert not (run_dir / "steps" / STEP_ID / "outputs" / f"{PRODUCT}.png").exists()


def test_a_counting_row_reported_as_a_distribution_is_rejected(
    tmp_path: Path,
) -> None:
    audit = _audit_frame()
    audit.loc[audit["metric"].eq("valid_observed"), "summary_value"] = 1.0
    run_dir, manifest = _binding(tmp_path, audit=audit)
    with pytest.raises(ValueError, match="as a distribution"):
        _run(run_dir, manifest)


def test_counts_that_do_not_partition_the_denominator_are_rejected(
    tmp_path: Path,
) -> None:
    audit = _audit_frame()
    target = audit["variable"].eq("lact_first") & audit["metric"].eq("valid_observed")
    audit.loc[target, "count"] = 500.0
    audit.loc[target, "percentage"] = 50.0
    run_dir, manifest = _binding(tmp_path, audit=audit)
    with pytest.raises(ValueError, match="partition its denominator"):
        _run(run_dir, manifest)


def test_a_percentage_that_does_not_reconcile_is_rejected(tmp_path: Path) -> None:
    audit = _audit_frame()
    audit.loc[audit["variable"].eq("sex") & audit["level"].eq("Male"), "percentage"] = (
        99.0
    )
    run_dir, manifest = _binding(tmp_path, audit=audit)
    with pytest.raises(ValueError, match="percentage does not reconcile"):
        _run(run_dir, manifest)


def test_levels_that_do_not_partition_their_measure_are_rejected(
    tmp_path: Path,
) -> None:
    process = _process_frame()
    target = process["process_measure"].eq("measured_status") & process["level"].eq(
        "1.0"
    )
    process.loc[target, "count"] = 500
    process.loc[target, "percentage"] = 50.0
    run_dir, manifest = _binding(tmp_path, process=process)
    with pytest.raises(ValueError, match="do not partition its denominator"):
        _run(run_dir, manifest)


def test_an_unknown_metric_is_rejected_rather_than_ignored(tmp_path: Path) -> None:
    audit = _audit_frame()
    audit.loc[audit["metric"].eq("iqr"), "metric"] = "invented_metric"
    run_dir, manifest = _binding(tmp_path, audit=audit)
    with pytest.raises(ValueError, match="unsupported metric"):
        _run(run_dir, manifest)


def test_a_tampered_digest_fails_closed(tmp_path: Path) -> None:
    run_dir, manifest = _binding(tmp_path)
    binding = manifest["inputs"][MISSINGNESS_MEASUREMENT_AUDIT_INPUT]
    table = run_dir / binding["relative_path"]
    table.write_text(table.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest verification failed"):
        _run(run_dir, manifest)


def test_a_binding_outside_the_run_directory_fails_closed(tmp_path: Path) -> None:
    run_dir, manifest = _binding(tmp_path)
    manifest["inputs"][MEASUREMENT_PROCESS_AUDIT_INPUT][
        "relative_path"
    ] = "../escaped.csv"
    with pytest.raises(ValueError, match="escapes the run directory"):
        _run(run_dir, manifest)


def test_a_declared_schema_that_disagrees_with_the_bytes_fails_closed(
    tmp_path: Path,
) -> None:
    run_dir, manifest = _binding(tmp_path)
    contract = manifest["inputs"][MISSINGNESS_MEASUREMENT_AUDIT_INPUT][
        "product_contract"
    ]
    contract["columns"] = contract["columns"][:-1]
    with pytest.raises(ValueError, match="product contract is unsupported"):
        _run(run_dir, manifest)


def test_a_declared_row_count_that_disagrees_with_the_bytes_fails_closed(
    tmp_path: Path,
) -> None:
    run_dir, manifest = _binding(tmp_path)
    binding = manifest["inputs"][MEASUREMENT_PROCESS_AUDIT_INPUT]
    binding["product_contract"]["row_count"] = 5
    binding["consumption_contract"]["verified_row_count"] = 5
    with pytest.raises(ValueError, match="disagree with its product contract"):
        _run(run_dir, manifest)


def test_a_widened_or_foreign_manifest_fails_closed(tmp_path: Path) -> None:
    run_dir, manifest = _binding(tmp_path)
    manifest["inputs"]["table:other"] = manifest["inputs"][
        MEASUREMENT_PROCESS_AUDIT_INPUT
    ]
    with pytest.raises(ValueError, match="absent or widened"):
        _run(run_dir, manifest)

    run_dir, manifest = _binding(tmp_path / "second")
    manifest["step_id"] = "07_other_step"
    with pytest.raises(ValueError, match="does not belong to this step"):
        _run(run_dir, manifest)


def test_real_e1_row_kind_mix_is_accepted(tmp_path: Path) -> None:
    """Lock the real E1 shape: 13 numeric variables plus one categorical."""

    rows: list[list] = []
    for index in range(13):
        rows += _numeric_variable_rows(f"concept_{index:02d}", missing=index * 10)
    rows += [
        [
            "sex",
            "missing",
            "",
            0.0,
            0.0,
            _N,
            float(_N),
            None,
            None,
            None,
            None,
            None,
            "source missingness",
        ],
        [
            "sex",
            "level_distribution",
            "Female",
            440.0,
            44.0,
            _N,
            float(_N),
            None,
            None,
            None,
            None,
            None,
            "category share",
        ],
        [
            "sex",
            "level_distribution",
            "Male",
            560.0,
            56.0,
            _N,
            float(_N),
            None,
            None,
            None,
            None,
            None,
            "category share",
        ],
    ]
    audit = pd.DataFrame(rows, columns=_AUDIT_COLUMNS)
    assert len(audit) == 55
    assert audit["metric"].value_counts().to_dict() == {
        "missing": 14,
        "valid_observed": 13,
        "median": 13,
        "iqr": 13,
        "level_distribution": 2,
    }

    run_dir, manifest = _binding(tmp_path, audit=audit)
    _out_dir, summary = _run(run_dir, manifest)
    assert summary["status"] == "ok"
    assert summary["audited_variable_count"] == 14
    assert summary["source_rows_consumed"][MISSINGNESS_MEASUREMENT_AUDIT_INPUT] == 55


# --- Fail-closed gaps in the row validators -------------------------------
#
# The real E1 tables satisfy all of these, so none of them blocked the run.
# A sealed executor that publishes a figure straight from an upstream table is
# the only thing standing between an inconsistent audit and a manuscript
# panel, so each invariant it silently assumed is asserted here instead.


def _audit_rows_error(mutate) -> str:
    frame = _audit_frame()
    mutate(frame)
    with pytest.raises(ValueError) as excinfo:
        _validate_audit_rows(frame)
    return str(excinfo.value)


def _process_rows_error(mutate) -> str:
    frame = _process_frame()
    mutate(frame)
    with pytest.raises(ValueError) as excinfo:
        _validate_process_rows(frame)
    return str(excinfo.value)


def test_real_tables_still_pass_every_tightened_check():
    per_variable = _validate_audit_rows(_audit_frame())
    cells = _validate_process_rows(_process_frame())

    assert set(per_variable) == {"lact_first", "sep3_sofa2_max", "sex"}
    assert cells


def test_a_median_outside_its_own_quartiles_is_rejected():
    """``q1 <= q3`` passes a median copied from a different variable."""

    def mutate(frame: pd.DataFrame) -> None:
        target = (frame["variable"] == "lact_first") & (frame["metric"] == "median")
        frame.loc[target, "summary_value"] = 99.0

    assert "does not reconcile with its own q1-q3 range" in _audit_rows_error(mutate)


def test_an_iqr_that_is_not_the_reported_range_width_is_rejected():
    def mutate(frame: pd.DataFrame) -> None:
        target = (frame["variable"] == "lact_first") & (frame["metric"] == "iqr")
        frame.loc[target, "summary_value"] = 7.0

    assert "does not reconcile with its own q1-q3 range" in _audit_rows_error(mutate)


def test_level_counts_must_agree_with_a_valid_observed_total_reported_beside_them():
    """A variable stating both is stating the same observed rows twice."""

    def mutate(frame: pd.DataFrame) -> None:
        extra = frame.loc[frame["variable"] == "sex"].iloc[[0]].copy()
        extra["metric"] = "valid_observed"
        extra["level"] = ""
        extra["count"] = 900.0
        extra["percentage"] = 100.0 * 900.0 / _N
        frame.loc[len(frame)] = extra.iloc[0]

    message = _audit_rows_error(mutate)
    assert "level counts sum to 1000" in message
    assert "valid-observed total is 900" in message


def test_consistent_level_counts_and_valid_observed_total_are_accepted():
    frame = _audit_frame()
    extra = frame.loc[frame["variable"] == "sex"].iloc[[0]].copy()
    extra["metric"] = "valid_observed"
    extra["level"] = ""
    extra["count"] = 1000.0
    extra["percentage"] = 100.0
    frame.loc[len(frame)] = extra.iloc[0]

    per_variable = _validate_audit_rows(frame)

    assert per_variable["sex"]["valid_observed"] == 1000


def test_a_repeated_unlevelled_process_cell_is_rejected():
    """Levelled rows are covered by the partition check; a lone measure was not."""

    def mutate(frame: pd.DataFrame) -> None:
        duplicate = frame.loc[frame["process_measure"] == "count_missing"].iloc[[0]]
        frame.loc[len(frame)] = duplicate.iloc[0]

    assert "repeats the cell 'count_missing'" in _process_rows_error(mutate)


def test_a_process_median_outside_its_own_quartiles_is_rejected():
    def mutate(frame: pd.DataFrame) -> None:
        target = frame["process_measure"] == "count_frequency_summary"
        frame.loc[target, "median"] = 50.0

    assert "median outside its own q1-q3 range" in _process_rows_error(mutate)


# --------------------------------------------------------------------------
# L0: the renderer claims a step by its typed contract, not by the spelling of
# the figure product.  The id is a Planner-owned label; competence is fixed by
# the two required inputs and their verified schemas.
# --------------------------------------------------------------------------


#: Transcribed verbatim from the real 12-step authority plan
#: (``analysis_plan_revision_3``) of run ``run_20260728T131514_c54ead``.  It is
#: a *fixture*: the production renderer must contain no branch that recognises
#: this benchmark item, and the assertions below are about the shape of the
#: contract, never about the case.
_REAL_PLAN_FIGURE_STEP = {
    "step_id": "05_missingness_measurement_process_audit_figure",
    "planned_analysis_role": "auxiliary",
    "method": "visualization",
    "inputs": [
        "table:missingness_measurement_audit",
        "table:measurement_process_audit",
    ],
    "expected_outputs": ["figure:missingness_event_timing"],
    "intent": "Render the publication figure(s) declared by the parent step.",
    "input_consumption_contracts": [
        ArtifactConsumptionContract(
            input_key="table:missingness_measurement_audit", mode="all_rows"
        ),
        ArtifactConsumptionContract(
            input_key="table:measurement_process_audit", mode="all_rows"
        ),
    ],
}


@pytest.mark.parametrize(
    "product",
    [
        "measurement_process_overview",
        "missingness_event_timing",
        "data_quality",
        "f1",
        "a",
    ],
)
def test_any_legal_product_id_is_owned_when_the_typed_contract_closes(
    product: str,
) -> None:
    """The spelling of the label is not a capability question."""

    assert missingness_measurement_figure_executor_owns_step(
        _step(expected_outputs=[f"figure:{product}"])
    )


def test_the_real_plans_figure_step_is_owned() -> None:
    """The step this renderer was written for, exactly as the Planner wrote it.

    Before L0 every ``owns_step`` clause passed except the product name, so a
    renderer that consumes precisely these two tables refused the work and the
    step fell to the Coder.
    """

    step = AnalysisStep.model_validate(_REAL_PLAN_FIGURE_STEP)
    assert missingness_measurement_figure_executor_owns_step(step)
    selection = select_standard_executor(
        step, plan=AnalysisPlan(research_question="Test", steps=[step])
    )
    assert selection is not None
    assert selection.analysis_kind == "missingness_measurement_figure"


@pytest.mark.parametrize(
    "output",
    [
        "figure:../../etc/passwd",
        "figure:a/b",
        "figure:.hidden",
        "figure:",
        "figure:UPPER",
        "figure:1leading_digit",
        "figure:trailing space",
        "figure:" + "x" * 129,
        "table:missingness_measurement_audit",
        "missingness_event_timing",
    ],
)
def test_an_unsafe_or_malformed_product_id_is_refused_by_the_selector(
    output: str,
) -> None:
    assert not missingness_measurement_figure_executor_owns_step(
        _step(expected_outputs=[output])
    )


def test_two_declared_figures_are_refused_even_when_both_are_legal() -> None:
    assert not missingness_measurement_figure_executor_owns_step(
        _step(expected_outputs=["figure:one", "figure:two"])
    )


@pytest.mark.parametrize(
    "product",
    ["../../escape", "a/b", "", "UPPER", ".hidden", "x" * 129],
)
def test_the_runtime_refuses_an_unsafe_product_id_without_trusting_its_caller(
    tmp_path: Path, product: str
) -> None:
    """``run_...`` is public and interpolates this id into a path.

    The selector already parses it out of ``figure:<id>``; the entry point does
    not take that on trust.
    """

    with pytest.raises(ValueError, match="unsafe or malformed figure product id"):
        run_missingness_measurement_figure(
            out_dir=tmp_path / "out",
            run_dir=tmp_path / "run",
            resolved_inputs={},
            step_id=STEP_ID,
            figure_product=product,
        )


def test_a_renamed_product_still_renders_the_same_verified_bundle(
    tmp_path: Path,
) -> None:
    """Renaming the label changes the filenames and nothing else."""

    run_dir, manifest = _binding(tmp_path)
    result = run_missingness_measurement_figure(
        out_dir=tmp_path / "out",
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=STEP_ID,
        figure_product="measurement_process_overview",
    )
    assert result["figure_path"] == "measurement_process_overview.png"
    assert (tmp_path / "out" / "measurement_process_overview.png").exists()
    assert (
        tmp_path / "out" / "measurement_process_overview_missingness_source_data.csv"
    ).exists()


def test_the_renderer_carries_no_case_specific_branch() -> None:
    """Ownership is decided by the typed contract, not by recognising a case."""

    import easyicu.research_agent.execution.runners.missingness_measurement_figure_executor as module

    source = Path(module.__file__).read_text()
    for token in ("sepsis", "sep3", "e1_", "_e1", "94,458", "missingness_event_timing"):
        assert (
            token not in source.lower()
        ), f"case-specific token in production: {token}"
