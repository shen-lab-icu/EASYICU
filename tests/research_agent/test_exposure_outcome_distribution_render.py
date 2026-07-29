"""The renderer for the self-contained exposure-outcome distribution product.

Two load-bearing claims are tested here. The first is that it needs **one**
table: these tests build the table with the real producer, hand only that to
the renderer, and check a figure comes out -- no cohort summary, no second
binding, and no access to the spec.

The second is that "re-checks the arithmetic" is literal. Each published
quantity is recomputed from the counts beside it, using the method and
confidence level the table itself declares, so the negative tests below alter
one number at a time and expect a refusal. A check that only asked whether a
rate fell inside its own interval would pass most of them.

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
    digest = hashlib.sha256(cohort.read_bytes()).hexdigest()
    (parent / "resolved_inputs.json").write_text(
        json.dumps(
            {
                "step_id": "03_parent",
                "inputs": {
                    "artifact:analysis_cohort": {
                        "relative_path": "cohort.parquet",
                        "sha256": digest,
                        "declared_kind": "artifact",
                        "product": "analysis_cohort",
                        "product_contract": {
                            "columns": list(frame.columns),
                            "row_count": int(len(frame)),
                        },
                    }
                },
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
            "outcome_levels": [0, 1],
            "outcome_positive_value": 1,
            "level_match_policy": "exact_typed",
            "denominator_policy": "all_declared_rows",
            "missing_outcome_policy": "structural_absence_is_non_event",
            "confidence_level": 0.95,
        },
        typed_cohort_input="artifact:analysis_cohort",
    )
    return parent_out / "exposure_outcome_distribution.csv"


def _bound(
    tmp_path: Path, table: Path, *, rows: int | None = None
) -> tuple[Path, dict]:
    run_dir = tmp_path / "run"
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
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
                "evidence_kind": "table",
                "product": "exposure_outcome_distribution",
                "evidence_id": "ev-distribution",
                "identity_row": {
                    "input_key": INPUT_KEY,
                    "declared_kind": "table",
                    "product": "exposure_outcome_distribution",
                    "evidence_id": "ev-distribution",
                    "sha256": digest,
                },
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


def _render(run_dir: Path, manifest: dict, out_dir: Path):
    return run_exposure_outcome_distribution_figure(
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=STEP_ID,
        figure_product=PRODUCT,
    )


def _tampered(tmp_path: Path, monkeypatch, mutate) -> tuple[Path, dict]:
    """Rebind a table after ``mutate`` has changed one published number."""

    table = _produced_table(tmp_path, monkeypatch)
    frame = pd.read_csv(table)
    mutate(frame)
    frame.to_csv(table, index=False)
    return _bound(tmp_path, table)


# --------------------------------------------------------------------------
# Ownership
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


# --------------------------------------------------------------------------
# Rendering from the one table
# --------------------------------------------------------------------------


def test_it_renders_from_the_one_table_alone(tmp_path: Path, monkeypatch) -> None:
    """The whole point of the self-contained product."""

    table = _produced_table(tmp_path, monkeypatch)
    run_dir, manifest = _bound(tmp_path, table)
    out_dir = tmp_path / "figure_out"
    summary = _render(run_dir, manifest, out_dir)
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


def test_the_summary_and_note_carry_the_declared_design(
    tmp_path: Path, monkeypatch
) -> None:
    """A reader can see which design the drawing was made under.

    The renderer echoes what the table declares rather than restating a
    convention, so a figure drawn under complete-case cannot be read as one
    drawn over every declared row.
    """

    table = _produced_table(tmp_path, monkeypatch)
    run_dir, manifest = _bound(tmp_path, table)
    out_dir = tmp_path / "figure_out"
    summary = _render(run_dir, manifest, out_dir)
    design = summary["declared_design"]
    assert design["denominator_policy"] == "all_declared_rows"
    assert design["missing_outcome_policy"] == "structural_absence_is_non_event"
    assert design["confidence_level"] == pytest.approx(0.95)

    contract = json.loads((out_dir / f"{PRODUCT}.figure_contract.json").read_text())
    note = contract["statistics_note"]
    assert "all_declared_rows" in note
    assert "wilson" in note


# --------------------------------------------------------------------------
# What must fail closed: the binding
# --------------------------------------------------------------------------


def test_a_tampered_table_fails_closed(tmp_path: Path, monkeypatch) -> None:
    table = _produced_table(tmp_path, monkeypatch)
    run_dir, manifest = _bound(tmp_path, table)
    manifest["inputs"][INPUT_KEY]["sha256"] = "0" * 64
    manifest["inputs"][INPUT_KEY]["consumption_contract"]["artifact_sha256"] = "0" * 64
    manifest["inputs"][INPUT_KEY]["identity_row"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="digest verification failed"):
        _render(run_dir, manifest, tmp_path / "out")


def test_a_row_count_disagreeing_with_the_contract_fails_closed(
    tmp_path: Path, monkeypatch
) -> None:
    table = _produced_table(tmp_path, monkeypatch)
    run_dir, manifest = _bound(tmp_path, table, rows=99)
    with pytest.raises(RuntimeError, match="disagree with its product contract"):
        _render(run_dir, manifest, tmp_path / "out")


def test_a_manifest_for_another_step_is_refused(tmp_path: Path, monkeypatch) -> None:
    table = _produced_table(tmp_path, monkeypatch)
    run_dir, manifest = _bound(tmp_path, table)
    manifest["step_id"] = "99_someone_elses_step"
    with pytest.raises(RuntimeError, match="does not belong to this step"):
        _render(run_dir, manifest, tmp_path / "out")


def test_a_capsule_naming_another_product_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    table = _produced_table(tmp_path, monkeypatch)
    run_dir, manifest = _bound(tmp_path, table)
    manifest["inputs"][INPUT_KEY]["product"] = "cohort_summary"
    manifest["inputs"][INPUT_KEY]["identity_row"]["product"] = "cohort_summary"
    with pytest.raises(RuntimeError, match="does not match the input key"):
        _render(run_dir, manifest, tmp_path / "out")


def test_a_capsule_without_a_consumption_contract_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    """The step declares one, so the capsule that authorises it must carry it."""

    table = _produced_table(tmp_path, monkeypatch)
    run_dir, manifest = _bound(tmp_path, table)
    manifest["inputs"][INPUT_KEY].pop("consumption_contract")
    with pytest.raises(RuntimeError, match="no consumption contract"):
        _render(run_dir, manifest, tmp_path / "out")


def test_a_second_bound_input_is_refused(tmp_path: Path, monkeypatch) -> None:
    """One table, and no other -- the property the whole design rests on."""

    table = _produced_table(tmp_path, monkeypatch)
    run_dir, manifest = _bound(tmp_path, table)
    manifest["inputs"]["table:cohort_summary"] = dict(manifest["inputs"][INPUT_KEY])
    with pytest.raises(RuntimeError, match="is widened by") as excinfo:
        _render(run_dir, manifest, tmp_path / "out")
    # The refusal names what was added, so a reader is not left guessing which
    # of the bound inputs this consumer did not ask for.
    assert "table:cohort_summary" in str(excinfo.value)


# --------------------------------------------------------------------------
# What must fail closed: the arithmetic
# --------------------------------------------------------------------------


def test_a_table_whose_levels_do_not_partition_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    def mutate(frame: pd.DataFrame) -> None:
        frame.loc[frame["row_role"] == "overall", "n_rows"] = 999

    run_dir, manifest = _tampered(tmp_path, monkeypatch, mutate)
    with pytest.raises(ValueError, match="do not partition the reported cohort"):
        _render(run_dir, manifest, tmp_path / "out")


def test_a_rate_that_is_not_its_own_events_over_denominator_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    """The defect a plausibility check misses.

    30.0% over 10 rows becomes 31.0%: still inside its own interval, still
    below 100, still adds up against every total -- and simply not the number
    the counts produce.
    """

    def mutate(frame: pd.DataFrame) -> None:
        target = (frame["row_role"] == "exposure_level") & (
            frame["exposure_level"] == 1
        )
        frame.loc[target, "outcome_rate_pct"] = 31.0

    run_dir, manifest = _tampered(tmp_path, monkeypatch, mutate)
    with pytest.raises(ValueError, match="not its own events over denominator"):
        _render(run_dir, manifest, tmp_path / "out")


def test_an_exposure_percentage_that_is_not_its_own_counts_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    def mutate(frame: pd.DataFrame) -> None:
        target = (frame["row_role"] == "exposure_level") & (
            frame["exposure_level"] == 1
        )
        frame.loc[target, "exposure_pct"] = 55.0

    run_dir, manifest = _tampered(tmp_path, monkeypatch, mutate)
    with pytest.raises(ValueError, match="not its own counts"):
        _render(run_dir, manifest, tmp_path / "out")


def test_an_interval_that_is_not_the_declared_method_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    """Narrowed intervals that still contain their estimate are refused."""

    def mutate(frame: pd.DataFrame) -> None:
        frame["ci_low_pct"] = frame["outcome_rate_pct"] * 0.99
        frame["ci_high_pct"] = frame["outcome_rate_pct"] * 1.01

    run_dir, manifest = _tampered(tmp_path, monkeypatch, mutate)
    with pytest.raises(ValueError, match="not the declared method"):
        _render(run_dir, manifest, tmp_path / "out")


def test_a_confidence_level_changed_after_the_fact_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    """Relabelling 95% intervals as 99% is a claim about coverage."""

    def mutate(frame: pd.DataFrame) -> None:
        frame["confidence_level"] = 0.99

    run_dir, manifest = _tampered(tmp_path, monkeypatch, mutate)
    with pytest.raises(ValueError, match="not the declared method"):
        _render(run_dir, manifest, tmp_path / "out")


def test_a_denominator_contradicting_the_declared_policy_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    """A table cannot say 'over every declared row' and report complete-case."""

    def mutate(frame: pd.DataFrame) -> None:
        frame["denominator_policy"] = "observed_outcome_rows"

    run_dir, manifest = _tampered(tmp_path, monkeypatch, mutate)
    with pytest.raises(ValueError, match="does not follow the declared"):
        _render(run_dir, manifest, tmp_path / "out")


def test_rows_disagreeing_about_the_design_are_refused(
    tmp_path: Path, monkeypatch
) -> None:
    """Two designs in one table is not a table this renderer can draw."""

    def mutate(frame: pd.DataFrame) -> None:
        frame.loc[frame.index[0], "missing_outcome_policy"] = "fail_closed"

    run_dir, manifest = _tampered(tmp_path, monkeypatch, mutate)
    with pytest.raises(ValueError, match="rows disagree on"):
        _render(run_dir, manifest, tmp_path / "out")


def test_missing_counts_that_do_not_sum_are_refused(
    tmp_path: Path, monkeypatch
) -> None:
    def mutate(frame: pd.DataFrame) -> None:
        frame.loc[frame["row_role"] == "overall", "outcome_missing_n"] = 5
        frame.loc[frame["row_role"] == "overall", "outcome_observed_n"] = 15

    run_dir, manifest = _tampered(tmp_path, monkeypatch, mutate)
    with pytest.raises(ValueError, match="does not sum to the overall"):
        _render(run_dir, manifest, tmp_path / "out")


def test_the_product_name_is_shared_with_another_shape_and_that_is_safe(
    tmp_path: Path, monkeypatch
) -> None:
    """A foreign schema under this product name is refused, not rendered.

    A second owner did once publish a completely different long-format table
    under the same ``table:exposure_outcome_distribution`` key; it was deleted
    when this renderer took the name, and deleting it was only safe *because*
    of the property below -- binding happens on the schema the host recorded,
    not on the product name. The shape is written out here rather than imported
    so the guard outlives any particular other producer: the hazard is the name
    being reusable at all, and a collision resolved by structure is exactly the
    kind of thing that starts silently mis-rendering once someone loosens a
    check.
    """

    OTHER_SHAPE = ("variable", "metric", "value", "denominator_n")
    assert set(OTHER_SHAPE) != set(EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS)

    other = tmp_path / "other.csv"
    # Three rows so the row-count floor passes and the *schema* is what refuses.
    pd.DataFrame(
        [{column: index for column in OTHER_SHAPE} for index in range(3)]
    ).to_csv(other, index=False)
    run_dir, manifest = _bound(tmp_path, other)
    manifest["inputs"][INPUT_KEY]["product_contract"]["columns"] = list(OTHER_SHAPE)
    with pytest.raises(RuntimeError, match="different product schema"):
        _render(run_dir, manifest, tmp_path / "out")


def test_the_renderer_carries_no_case_specific_branch() -> None:
    import easyicu.research_agent.execution.runners.exposure_outcome_distribution_render as module

    source = Path(module.__file__).read_text().lower()
    for token in ("sepsis", "sep3", "e1_", "icu_readmission", "94,458"):
        assert token not in source, f"case-specific token in production: {token}"


def test_a_row_the_renderer_does_not_recognise_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    """A third role would be dropped from every sum and still ship in the table.

    Both this and the duplicate-level test below were kept only after a mutation
    pass: with the check deleted, each table validated cleanly. Two sibling
    checks written at the same time (a finite-value guard and a non-negative
    count guard) did NOT survive that pass -- the exact re-derivations already
    caught them -- and were removed rather than kept for the look of the thing.
    """

    def mutate(frame: pd.DataFrame) -> None:
        extra = frame.iloc[0].copy()
        extra["row_role"] = "footnote"
        frame.loc[len(frame)] = extra

    run_dir, manifest = _tampered(tmp_path, monkeypatch, mutate)
    with pytest.raises(ValueError, match="unknown row roles"):
        _render(run_dir, manifest, tmp_path / "out")


def test_the_same_exposure_level_twice_is_refused(tmp_path: Path, monkeypatch) -> None:
    """Two bars under one label still partition the cohort correctly."""

    def mutate(frame: pd.DataFrame) -> None:
        levels = frame.index[frame["row_role"] == "exposure_level"].tolist()
        frame.loc[levels[0], "exposure_level"] = frame.loc[
            levels[1], "exposure_level"
        ]

    run_dir, manifest = _tampered(tmp_path, monkeypatch, mutate)
    with pytest.raises(ValueError, match="appears more than once"):
        _render(run_dir, manifest, tmp_path / "out")


def test_the_source_data_beside_the_figure_holds_every_row_it_drew(
    tmp_path: Path, monkeypatch
) -> None:
    """Ported from the deleted two-table executor's suite, not silently dropped.

    A reader who opens the source data must be able to rebuild the panels. The
    full file therefore carries the whole product, and each panel file carries
    one row per drawn level -- if a level were dropped on the way to disk the
    figure would show a bar with no line to check it against.
    """

    table = _produced_table(tmp_path, monkeypatch)
    run_dir, manifest = _bound(tmp_path, table)
    out = tmp_path / "out"
    _render(run_dir, manifest, out)

    produced = pd.read_csv(table)
    levels = produced[produced["row_role"] == "exposure_level"]

    full = pd.read_csv(out / f"{PRODUCT}_input_source_data.csv")
    assert len(full) == len(produced)

    for panel in ("prevalence", "outcome"):
        panel_rows = pd.read_csv(out / f"{PRODUCT}_{panel}_source_data.csv")
        assert len(panel_rows) == len(levels), panel
        assert set(panel_rows["exposure_level"].astype(str)) == set(
            levels["exposure_level"].astype(str)
        ), panel
