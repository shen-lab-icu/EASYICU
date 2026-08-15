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
    missingness_measurement_figure_executor_code,
    measurement_missingness_figure_executor_owns_step,
    _COLUMNS_BY_INPUT,
    _validate_audit_rows,
    _validate_process_rows,
    missingness_measurement_figure_executor_owns_step,
    run_measurement_missingness_figure,
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

# The producer's real wide schemas, verbatim from a real E1 run's step-04
# artifacts.  The renderer reads a subset of these; the extra columns are kept
# in the fixture on purpose, because a consumer that demanded exact equality
# with its own read-set is the defect this file now guards against.
_AUDIT_COLUMNS = [
    "concept",
    "variable",
    "label",
    "value_column",
    "n_total",
    "measured_one_n",
    "measured_one_pct",
    "value_missing_n",
    "value_missing_pct",
    "eligible_n",
    "not_applicable_n",
    "indicator_semantics",
    "missingness_kind",
]
_PROCESS_COLUMNS = [
    "concept",
    "variable",
    "value_column",
    "n_total",
    "measured_one_n",
    "measurement_total_n",
    "measurement_count_max",
    "repeat_measured_n",
    "eligible_n",
    "not_applicable_n",
    "missingness_kind",
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


#: The typed-input binding map the selector really hands this owner. Ownership
#: now requires it: the renderer's loader has always refused a bound table
#: missing the columns it reads, and asking that BEFORE the claim is what stops
#: a claimed step from dying at load. Measured 2026-07-31, 8 of the 10 recorded
#: steps the old predicate claimed carried a table it could not read.
def _readable_bindings(**overrides) -> dict:
    from easyicu.research_agent.execution.runners.missingness_measurement_figure_executor import (
        MISSINGNESS_MEASUREMENT_FIGURE_INPUTS as _KEYS,
        _COLUMNS_BY_INPUT as _COLS,
    )

    bindings = {
        key: {"product_contract": {"columns": list(_COLS[key])}} for key in _KEYS
    }
    bindings.update(overrides)
    return bindings


def _owns(step, *, resolved_bindings=None) -> bool:
    """Ask the real predicate, defaulting to a readable binding map.

    Ownership requires that map as of 2026-07-31: the loader has always refused
    a bound table missing the columns this renderer reads, and asking the same
    question BEFORE the claim is what stops a claimed step from dying at load.
    Measured over the recorded plans, 8 of the 10 steps the old predicate
    claimed carried a table it could not read. Tests about the STEP shape pass
    the readable map so the shape is what they vary; tests about the BINDING
    pass their own.
    """

    return missingness_measurement_figure_executor_owns_step(
        step,
        resolved_bindings=(
            _readable_bindings() if resolved_bindings is None else resolved_bindings
        ),
    )


def _audit_row(
    variable: str,
    *,
    missing: int,
    not_applicable: int = 0,
) -> list:
    """One wide audit row, with both partitions satisfied by construction."""

    eligible = _N - not_applicable
    measured = eligible - missing
    return [
        variable,
        variable,
        variable.replace("_", " "),
        f"{variable}_value",
        _N,
        measured,
        100.0 * measured / _N,
        missing,
        100.0 * missing / _N,
        eligible,
        not_applicable,
        "conditional_event_time" if not_applicable else "measurement_availability",
        "conditional_event_time" if not_applicable else "measurement_missing",
    ]


def _process_row(
    variable: str,
    *,
    measured: int,
    repeat: int,
    not_applicable: int = 0,
) -> list:
    return [
        variable,
        variable,
        f"{variable}_value",
        _N,
        measured,
        measured * 3,
        5,
        repeat,
        _N - not_applicable,
        not_applicable,
        "measurement_missing",
    ]


def _audit_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _audit_row("lact_first", missing=400),
            _audit_row("sep3_sofa2_max", missing=0),
            # A conditional variable: applicable to a tenth of the cohort, and
            # fully observed within it.  Its cohort-stated missing share is 0 %,
            # which is exactly the number a reader can misread.
            _audit_row("death_time", missing=0, not_applicable=900),
        ],
        columns=_AUDIT_COLUMNS,
    )


def _process_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _process_row("lact_first", measured=600, repeat=300),
            _process_row("sep3_sofa2_max", measured=1000, repeat=0),
            _process_row("death_time", measured=100, repeat=0, not_applicable=900),
        ],
        columns=_PROCESS_COLUMNS,
    )


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
    assert _owns(step)
    # The selector must be given the same binding map the predicate needs;
    # without it this owner declines, which is the whole point of the clause.
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Test", steps=[step]),
        resolved_bindings=_readable_bindings(),
    )
    assert selection is not None
    assert selection.analysis_kind == "missingness_measurement_figure"
    assert selection.consumed_input_keys == MISSINGNESS_MEASUREMENT_FIGURE_INPUTS


def test_owner_is_order_insensitive_but_never_widened() -> None:
    assert _owns(_step(inputs=list(reversed(MISSINGNESS_MEASUREMENT_FIGURE_INPUTS))))
    assert not _owns(
        _step(
            inputs=[*MISSINGNESS_MEASUREMENT_FIGURE_INPUTS, "table:other"],
            input_consumption_contracts=[
                ArtifactConsumptionContract(input_key=key, mode="all_rows")
                for key in [*MISSINGNESS_MEASUREMENT_FIGURE_INPUTS, "table:other"]
            ],
        )
    )
    assert not _owns(
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
    assert not _owns(_step(planned_analysis_role="primary"))
    assert not _owns(_step(method="adjusted_association_models"))
    assert not _owns(_step(expected_outputs=["table:missingness_measurement_audit"]))
    assert not _owns(_step(expected_outputs=[f"figure:{PRODUCT}", "figure:extra"]))
    assert not _owns(
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


def test_a_conditional_variable_is_drawn_and_marked_not_rescaled(
    tmp_path: Path,
) -> None:
    """The real trap this panel can set for a reader.

    ``death_time`` applies to a tenth of the cohort and is fully observed
    within it, so its cohort-stated missing share is 0 %.  Drawn alone that
    reads as a completely observed variable.  The renderer keeps the parent's
    own number -- rescaling it to the eligible stays would be the renderer
    choosing a denominator -- and marks the variable instead.
    """

    audit = _audit_frame()
    conditional = audit["variable"].eq("death_time")
    assert audit.loc[conditional, "value_missing_pct"].tolist() == [0.0]
    assert audit.loc[conditional, "not_applicable_n"].tolist() == [900]

    run_dir, manifest = _binding(tmp_path, audit=audit)
    out_dir, summary = _run(run_dir, manifest)

    assert summary["status"] == "ok"
    assert summary["source_rows_consumed"] == {
        MISSINGNESS_MEASUREMENT_AUDIT_INPUT: len(audit),
        MEASUREMENT_PROCESS_AUDIT_INPUT: len(_process_frame()),
    }
    assert (out_dir / f"{PRODUCT}.png").is_file()
    # Panel B is what lets the reader see the 0 % is over a cohort the
    # variable barely applies to, so the eligible share must be drawn.
    process_source = pd.read_csv(
        out_dir / f"{PRODUCT}_measurement_process_source_data.csv"
    )
    assert process_source.loc[
        process_source["variable"].eq("death_time"), "eligible_n"
    ].tolist() == [100]


def test_runner_renders_complete_source_backed_bundle(tmp_path: Path) -> None:
    run_dir, manifest = _binding(tmp_path)
    out_dir, summary = _run(run_dir, manifest)

    assert summary["status"] == "ok"
    assert summary["audited_variable_count"] == 3
    # Three variables x the three declared stay-count measures.
    assert summary["measurement_process_cell_count"] == 9
    for suffix in ("png", "svg", "pdf", "tiff"):
        assert (out_dir / f"{PRODUCT}.{suffix}").is_file()

    audit_source = pd.read_csv(out_dir / f"{PRODUCT}_missingness_source_data.csv")
    assert audit_source["source_row_index"].tolist() == list(range(3))
    process_source = pd.read_csv(
        out_dir / f"{PRODUCT}_measurement_process_source_data.csv"
    )
    assert process_source["source_row_index"].tolist() == list(range(3))

    # The panel projection is a verbatim row subset of the parent: every value
    # and every row position is the parent's own, so it stays traceable.
    panel = pd.read_csv(out_dir / f"{PRODUCT}_source_missingness_panel_source_data.csv")
    assert panel["variable"].tolist() == ["lact_first", "sep3_sofa2_max", "death_time"]
    assert panel["value_missing_n"].tolist() == [400, 0, 0]
    assert panel["value_missing_pct"].tolist() == [40.0, 0.0, 0.0]
    assert panel["source_row_index"].tolist() == [0, 1, 2]
    parent = pd.read_csv(out_dir / f"{PRODUCT}_missingness_source_data.csv")
    for _, row in panel.iterrows():
        origin = parent.loc[parent["source_row_index"] == row["source_row_index"]]
        assert origin["value_missing_n"].tolist() == [row["value_missing_n"]]
        assert origin["value_missing_pct"].tolist() == [row["value_missing_pct"]]

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


def test_all_zero_missingness_renders_explicit_completeness_instead_of_blank_bars(
    tmp_path: Path,
) -> None:
    audit = _audit_frame()
    audit["value_missing_n"] = 0
    audit["value_missing_pct"] = 0.0
    audit["measured_one_n"] = audit["eligible_n"]
    audit["measured_one_pct"] = 100.0 * audit["measured_one_n"] / audit["n_total"]
    process = _process_frame()
    process["measured_one_n"] = process["eligible_n"]
    process["repeat_measured_n"] = 0

    run_dir, manifest = _binding(tmp_path, audit=audit, process=process)
    out_dir, summary = _run(run_dir, manifest)

    assert summary["zero_missing_completeness_display"] is True
    source = pd.read_csv(
        out_dir / f"{PRODUCT}_source_missingness_panel_source_data.csv"
    )
    assert source["value_missing_n"].tolist() == [0, 0, 0]
    assert source["value_missing_pct"].tolist() == [0.0, 0.0, 0.0]

    svg = (out_dir / f"{PRODUCT}.svg").read_text(encoding="utf-8")
    assert "Source completeness" in svg
    assert "0 missing / 100% complete" in svg
    contract = json.loads(
        (out_dir / f"{PRODUCT}.figure_contract.json").read_text(encoding="utf-8")
    )
    panel = contract["panels"][0]
    assert panel["title"] == "Source completeness"
    assert panel["metadata"]["zero_missing_completeness_display"] is True
    assert panel["metadata"]["source_products"] == [MISSINGNESS_MEASUREMENT_AUDIT_INPUT]
    assert "zero missing source values" in panel["claim"]


# The long-format schema these tests were written against (metric / level /
# summary_value rows) was never emitted by the deterministic producer, so the
# row-kind properties -- "a median reported as a tally", "an unknown metric",
# "levels that do not partition their measure" -- describe a table that does
# not exist.  Each is replaced below by the property the real wide schema has
# in its place; nothing is merely deleted.


def test_counts_that_do_not_partition_the_eligible_stays_are_rejected(
    tmp_path: Path,
) -> None:
    """Replaces the long-format missing/valid-observed partition check."""

    audit = _audit_frame()
    audit.loc[audit["variable"].eq("lact_first"), "measured_one_n"] = 500
    run_dir, manifest = _binding(tmp_path, audit=audit)
    with pytest.raises(ValueError, match="partition its eligible stays"):
        _run(run_dir, manifest)
    assert not (run_dir / "steps" / STEP_ID / "outputs" / f"{PRODUCT}.png").exists()


def test_eligibility_that_does_not_partition_the_cohort_is_rejected(
    tmp_path: Path,
) -> None:
    """The second partition: the one a single-denominator schema could not state."""

    audit = _audit_frame()
    audit.loc[audit["variable"].eq("death_time"), "not_applicable_n"] = 800
    run_dir, manifest = _binding(tmp_path, audit=audit)
    with pytest.raises(ValueError, match="partition the cohort"):
        _run(run_dir, manifest)


def test_a_percentage_that_does_not_reconcile_is_rejected(tmp_path: Path) -> None:
    audit = _audit_frame()
    audit.loc[audit["variable"].eq("lact_first"), "value_missing_pct"] = 99.0
    run_dir, manifest = _binding(tmp_path, audit=audit)
    with pytest.raises(ValueError, match="percentage does not reconcile"):
        _run(run_dir, manifest)


def test_a_percentage_restated_over_the_wrong_denominator_is_rejected(
    tmp_path: Path,
) -> None:
    """The conditional variable's share must stay stated over the cohort.

    40 % of ``death_time``'s eligible stays is a defensible number, but it is
    not the number this column holds, and drawing it beside cohort-stated bars
    would put two denominators on one axis.
    """

    audit = _audit_frame()
    conditional = audit["variable"].eq("death_time")
    audit.loc[conditional, "value_missing_n"] = 40
    audit.loc[conditional, "measured_one_n"] = 60
    audit.loc[conditional, "value_missing_pct"] = 40.0  # over eligible_n, not n_total
    run_dir, manifest = _binding(tmp_path, audit=audit)
    with pytest.raises(ValueError, match="against the cohort it is stated over"):
        _run(run_dir, manifest)


def test_measurement_counts_that_do_not_nest_are_rejected(tmp_path: Path) -> None:
    """Replaces the long-format level-partition check."""

    process = _process_frame()
    process.loc[process["variable"].eq("lact_first"), "repeat_measured_n"] = 900
    run_dir, manifest = _binding(tmp_path, process=process)
    with pytest.raises(ValueError, match="do not nest"):
        _run(run_dir, manifest)


def test_a_measure_larger_than_its_cohort_is_rejected(tmp_path: Path) -> None:
    """Replaces the long-format count-vs-denominator check."""

    process = _process_frame()
    process.loc[process["variable"].eq("sep3_sofa2_max"), "measured_one_n"] = _N + 1
    run_dir, manifest = _binding(tmp_path, process=process)
    with pytest.raises(ValueError, match="other than a stay count"):
        _run(run_dir, manifest)


def test_a_repeated_variable_row_is_rejected(tmp_path: Path) -> None:
    """Replaces the long-format repeated-cell check.

    One wide row per variable is the schema; a second row is a second answer
    to the same question, and the panel would silently draw one of them.
    """

    audit = pd.concat([_audit_frame(), _audit_frame().iloc[[0]]], ignore_index=True)
    run_dir, manifest = _binding(tmp_path, audit=audit)
    with pytest.raises(ValueError, match="appears twice in the audit"):
        _run(run_dir, manifest)

    process = pd.concat(
        [_process_frame(), _process_frame().iloc[[0]]], ignore_index=True
    )
    run_dir, manifest = _binding(tmp_path / "second", process=process)
    with pytest.raises(ValueError, match="appears twice in the measurement-process"):
        _run(run_dir, manifest)


def test_a_producer_column_the_figure_reads_cannot_go_missing(
    tmp_path: Path,
) -> None:
    """The W1 regression, stated from the consumer's side.

    A producer that stops emitting a column this renderer reads must fail
    closed and name the column -- the failure that actually happened said only
    "product contract is unsupported" and named nothing.
    """

    audit = _audit_frame().drop(columns=["eligible_n"])
    run_dir, manifest = _binding(tmp_path, audit=audit)
    with pytest.raises(ValueError, match="omits the columns this figure reads"):
        _run(run_dir, manifest)


def test_extra_producer_columns_do_not_break_the_consumer(tmp_path: Path) -> None:
    """The other half of W1: exact equality was the wrong consumer contract.

    The producer emits 27 columns and this renderer reads seven of them.  A
    column added upstream must not turn a working figure into a host crash.
    """

    audit = _audit_frame()
    audit["a_column_added_upstream_later"] = 1
    run_dir, manifest = _binding(tmp_path, audit=audit)
    out_dir, summary = _run(run_dir, manifest)

    assert summary["status"] == "ok"
    assert (out_dir / f"{PRODUCT}.png").is_file()


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
    # Drop a column the renderer never reads: the declared schema must still
    # match the bytes, because the digest pins one and the contract describes
    # the other.  A consumer that only checked its own read-set would accept a
    # contract that has stopped describing the artifact.
    contract["columns"] = [
        name for name in contract["columns"] if name != "missingness_kind"
    ]
    with pytest.raises(ValueError, match="disagree with its product contract"):
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


def test_real_e1_shape_is_accepted(tmp_path: Path) -> None:
    """Lock the real E1 shape: 15 audited variables, one of them conditional."""

    audit = pd.DataFrame(
        [_audit_row(f"concept_{index:02d}", missing=index * 10) for index in range(14)]
        + [_audit_row("death_time", missing=0, not_applicable=898)],
        columns=_AUDIT_COLUMNS,
    )
    process = pd.DataFrame(
        [
            _process_row(
                f"concept_{index:02d}", measured=_N - index * 10, repeat=index * 5
            )
            for index in range(14)
        ]
        + [_process_row("death_time", measured=102, repeat=0, not_applicable=898)],
        columns=_PROCESS_COLUMNS,
    )
    assert len(audit) == 15
    assert (audit["not_applicable_n"] > 0).sum() == 1

    run_dir, manifest = _binding(tmp_path, audit=audit, process=process)
    out_dir, summary = _run(run_dir, manifest)

    assert summary["status"] == "ok"
    assert summary["audited_variable_count"] == 15
    assert summary["measurement_process_cell_count"] == 45
    assert (out_dir / f"{PRODUCT}.png").is_file()


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

    assert set(per_variable) == {"lact_first", "sep3_sofa2_max", "death_time"}
    # The conditional variable is the one the panel must mark.
    assert [name for name, entry in per_variable.items() if entry["conditional"]] == [
        "death_time"
    ]
    assert len(cells) == 9


def test_a_non_integer_count_is_rejected_rather_than_coerced():
    """Replaces the long-format quartile checks.

    The wide schema has no distribution rows, so a median cannot fall outside
    its own quartiles here.  What can still arrive is a count that is not a
    whole number of stays, and reading it as one would silently round patients.
    """

    message = _audit_rows_error(
        lambda frame: frame.__setitem__("measured_one_n", 599.5)
    )
    assert "whole-stay count" in message


def test_an_empty_audit_is_rejected_rather_than_drawn_blank():
    """A figure with no variables is a blank claim, not a valid rendering."""

    with pytest.raises(ValueError, match="audits no variable"):
        _validate_audit_rows(_audit_frame().iloc[0:0])
    with pytest.raises(ValueError, match="audits no variable"):
        _validate_process_rows(_process_frame().iloc[0:0])


def test_a_measurement_total_is_never_treated_as_a_stay_count():
    """``measurement_total_n`` counts measurements, not stays.

    A real run reported 24,179 measurements over 1,000 stays.  It is excluded
    from the panel by construction; this pins that the panel's declared
    measures are the stay counts only, so a future edit cannot quietly put a
    measurement total on a 0-100 % axis.
    """

    from easyicu.research_agent.execution.runners import (
        missingness_measurement_figure_executor as module,
    )

    drawn = {column for column, _label in module._PROCESS_MEASURES}
    assert drawn == {"eligible_n", "measured_one_n", "repeat_measured_n"}
    assert "measurement_total_n" not in drawn
    assert "measurement_count_max" not in drawn
    assert "measurement_count_median_when_measured" not in drawn
    frame = _process_frame()
    cells = _validate_process_rows(frame)
    for cell in cells:
        assert 0.0 <= cell["percentage"] <= 100.0


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

    assert _owns(_step(expected_outputs=[f"figure:{product}"]))


def test_the_real_plans_figure_step_is_owned() -> None:
    """The step this renderer was written for, exactly as the Planner wrote it.

    Before L0 every ``owns_step`` clause passed except the product name, so a
    renderer that consumes precisely these two tables refused the work and the
    step fell to the Coder.
    """

    step = AnalysisStep.model_validate(_REAL_PLAN_FIGURE_STEP)
    assert _owns(step)
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Test", steps=[step]),
        resolved_bindings=_readable_bindings(),
    )
    assert selection is not None
    assert selection.analysis_kind == "missingness_measurement_figure"


def test_typed_measurement_alias_selects_and_renders_the_single_panel_owner(
    tmp_path: Path,
) -> None:
    input_key = "table:missingness_data_quality"
    producer = AnalysisStep(
        step_id=PARENT_STEP,
        planned_analysis_role="auxiliary",
        intent="Audit source availability.",
        method="measurement_audit",
        expected_outputs=[input_key],
        measurement_audit_spec={
            "products": [
                {
                    "product_id": "missingness_data_quality",
                    "audit": "measurement_missingness",
                }
            ]
        },
    )
    figure = AnalysisStep(
        step_id=STEP_ID,
        planned_analysis_role="auxiliary",
        intent="Render the exact source-availability audit.",
        method="visualization",
        inputs=[input_key],
        expected_outputs=["figure:missingness_data_quality"],
        input_consumption_contracts=[
            ArtifactConsumptionContract(input_key=input_key, mode="all_rows")
        ],
        figure_panels=[
            {
                "panel_id": "source_availability",
                "figure_output": "figure:missingness_data_quality",
                "article_role": "data_quality",
                "chart_type": "availability_panel",
                "source_products": [input_key],
            }
        ],
    )
    plan = AnalysisPlan(
        research_question="Audit source availability.",
        steps=[producer, figure],
    )
    run_dir = tmp_path / "run"
    binding = _register(
        run_dir,
        _audit_frame(),
        input_key=input_key,
        product="missingness_data_quality",
    )

    assert measurement_missingness_figure_executor_owns_step(
        figure,
        plan=plan,
        resolved_bindings={input_key: binding},
    )
    selection = select_standard_executor(
        figure,
        plan=plan,
        resolved_bindings={input_key: binding},
    )
    assert selection is not None
    assert selection.analysis_kind == "measurement_missingness_figure"
    assert selection.consumed_input_keys == (input_key,)

    out_dir = run_dir / "steps" / STEP_ID / "outputs"
    summary = run_measurement_missingness_figure(
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs={
            "schema_version": "2.1",
            "step_id": STEP_ID,
            "inputs": {input_key: binding},
        },
        step_id=STEP_ID,
        figure_product="missingness_data_quality",
        input_key=input_key,
    )
    contract = json.loads(
        (out_dir / "missingness_data_quality.figure_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["panels"][0]["metadata"] == {
        "article_role": "data_quality",
        "chart_type": "availability_panel",
        "source_data": ["missingness_data_quality_source_data.csv"],
        "source_products": [input_key],
    }
    assert summary["source_inputs"] == [input_key]


@pytest.mark.parametrize("audit", [None, "event_timing"])
def test_untyped_or_wrong_measurement_alias_is_not_claimed(audit: str | None) -> None:
    input_key = "table:missingness_data_quality"
    figure = AnalysisStep(
        step_id=STEP_ID,
        planned_analysis_role="auxiliary",
        intent="Render a table whose audit meaning is not yet closed.",
        method="visualization",
        inputs=[input_key],
        expected_outputs=["figure:missingness_data_quality"],
        input_consumption_contracts=[
            ArtifactConsumptionContract(input_key=input_key, mode="all_rows")
        ],
    )
    steps = []
    if audit is not None:
        steps.append(
            AnalysisStep(
                step_id=PARENT_STEP,
                planned_analysis_role="auxiliary",
                intent="Produce a different audit.",
                method="measurement_audit",
                expected_outputs=[input_key],
                measurement_audit_spec={
                    "products": [
                        {
                            "product_id": "missingness_data_quality",
                            "audit": audit,
                        }
                    ]
                },
            )
        )
    steps.append(figure)
    plan = AnalysisPlan(research_question="Audit source availability.", steps=steps)
    binding = {
        "product_contract": {"columns": list(_AUDIT_COLUMNS)},
    }

    assert not measurement_missingness_figure_executor_owns_step(
        figure,
        plan=plan,
        resolved_bindings={input_key: binding},
    )


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
    assert not _owns(_step(expected_outputs=[output]))


def test_two_declared_figures_are_refused_even_when_both_are_legal() -> None:
    assert not _owns(_step(expected_outputs=["figure:one", "figure:two"]))


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


# --------------------------------------------------------------------------
# claiming is a promise: a table this renderer cannot read must not be claimed


#: A real recorded header for this product, from a run where the Coder wrote
#: the audit rather than the deterministic producer. It shares `variable` and
#: `n_total` with the contract and is missing everything this renderer indexes.
_CODER_AUDIT_HEADER = [
    "variable",
    "n_total",
    "n_missing",
    "pct_missing",
    "first_measured_hours_from_admission",
    "note",
]


def test_it_declines_a_bound_table_it_cannot_read() -> None:
    """The measured defect: 8 of 10 recorded claims would have died at load.

    The loader has always refused this table. Refusing it HERE instead is the
    difference between the step going to the Coder -- which was drawing it
    successfully -- and the step being dead.
    """

    step = _step()
    assert _owns(step) is True
    unreadable = _readable_bindings(
        **{
            MISSINGNESS_MEASUREMENT_AUDIT_INPUT: {
                "product_contract": {"columns": list(_CODER_AUDIT_HEADER)}
            }
        }
    )
    assert _owns(step, resolved_bindings=unreadable) is False


def test_one_unreadable_input_is_enough_to_decline() -> None:
    """Both panels are drawn, so both bindings have to be readable."""

    step = _step()
    for key in MISSINGNESS_MEASUREMENT_FIGURE_INPUTS:
        bindings = _readable_bindings(
            **{key: {"product_contract": {"columns": ["variable", "n_total"]}}}
        )
        assert _owns(step, resolved_bindings=bindings) is False, key


def test_it_declines_when_the_selector_supplied_no_bindings() -> None:
    """Without the map this owner cannot know whether it can read the table.

    Both shapes matter and they take different branches: an EMPTY map has no
    binding for either input, and NO map at all is the older call signature
    every caller used before 2026-07-31. The real predicate is called directly
    here because the wrapper's default is a readable map -- routing through it
    would have tested the default instead of the absence.
    """

    assert _owns(_step(), resolved_bindings={}) is False
    assert (
        missingness_measurement_figure_executor_owns_step(
            _step(), resolved_bindings=None
        )
        is False
    )


def test_a_producer_may_add_columns_without_losing_its_renderer() -> None:
    """Containment, not equality: a new diagnostic field must not break it."""

    wider = {
        key: {
            "product_contract": {
                "columns": [*_COLUMNS_BY_INPUT[key], "a_future_diagnostic"]
            }
        }
        for key in MISSINGNESS_MEASUREMENT_FIGURE_INPUTS
    }
    assert _owns(_step(), resolved_bindings=wider) is True


def test_the_entrypoint_does_not_re_derive_ownership() -> None:
    """It cannot see the bindings the selector had, so it must not try."""

    code = missingness_measurement_figure_executor_code(_step())
    assert "run_missingness_measurement_figure(" in code
