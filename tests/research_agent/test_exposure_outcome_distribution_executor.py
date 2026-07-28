"""The exposure-by-outcome distribution owner.

Every scientific choice is the Planner's. These tests pin that: the executor
must refuse a step that has not declared the design, must never infer the
exposure from names or input order, and must produce a product a renderer can
draw without a second table.

The case used throughout is deliberately *not* the benchmark item -- a drug
exposure and a readmission outcome -- so a production branch that recognised
one study would not be exercised by the suite that guards it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
    EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS,
    exposure_outcome_distribution_executor_code,
    exposure_outcome_distribution_executor_owns_step,
    run_exposure_outcome_distribution_from_env,
)
from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

STEP_ID = "03_drug_readmission_distribution"
EXPOSURE = "anticoagulant_exposed"
OUTCOME = "readmitted_30d"

_SPEC = {
    "exposure": EXPOSURE,
    "exposure_levels": [0, 1],
    "outcome": OUTCOME,
    "outcome_positive_value": 1,
    "denominator_policy": "all_declared_rows",
}


def _step(**updates) -> AnalysisStep:
    spec = {**_SPEC, **updates.pop("spec", {})}
    payload = {
        "step_id": STEP_ID,
        "planned_analysis_role": "auxiliary",
        "method": "descriptive",
        "intent": "Report the exposure-by-outcome distribution.",
        "inputs": ["artifact:analysis_cohort", EXPOSURE, OUTCOME],
        "expected_outputs": ["table:exposure_outcome_distribution"],
        "exposure_outcome_distribution_spec": spec,
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


def _cohort(tmp_path: Path, frame: pd.DataFrame) -> tuple[Path, Path]:
    """Write a digest-bound cohort and its resolved-inputs manifest."""

    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / STEP_ID / "outputs"
    out_dir.mkdir(parents=True)
    cohort_path = run_dir / "cohort.parquet"
    frame.to_parquet(cohort_path, index=False)
    manifest = {
        "inputs": {
            "artifact:analysis_cohort": {
                "relative_path": "cohort.parquet",
                "sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
                "product_contract": {
                    "columns": list(frame.columns),
                    "row_count": int(len(frame)),
                },
            }
        }
    }
    manifest_path = run_dir / "resolved_inputs.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir, out_dir


def _run(monkeypatch, tmp_path: Path, frame: pd.DataFrame, **spec_updates) -> dict:
    run_dir, out_dir = _cohort(tmp_path, frame)
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(run_dir))
    monkeypatch.setenv(
        "EASYICU_RESOLVED_INPUTS_JSON", str(run_dir / "resolved_inputs.json")
    )
    return run_exposure_outcome_distribution_from_env(
        spec_payload={**_SPEC, **spec_updates},
        typed_cohort_input="artifact:analysis_cohort",
    )


# --------------------------------------------------------------------------
# Ownership
# --------------------------------------------------------------------------


def test_a_declared_design_is_owned_and_selected() -> None:
    step = _step()
    assert exposure_outcome_distribution_executor_owns_step(step)
    selection = select_standard_executor(
        step, plan=AnalysisPlan(research_question="Test", steps=[step])
    )
    assert selection is not None
    assert selection.analysis_kind == "exposure_outcome_distribution"


def test_a_step_without_the_spec_is_never_claimed() -> None:
    """The whole point: no spec, no owner -- not a guess from the inputs.

    The columns are right there in ``inputs``; an executor that took the
    first as exposure and the second as outcome would work on this step and
    silently invert the next one.
    """

    step = AnalysisStep.model_validate(
        {
            "step_id": STEP_ID,
            "planned_analysis_role": "auxiliary",
            "method": "descriptive",
            "intent": "Report the exposure-by-outcome distribution.",
            "inputs": ["artifact:analysis_cohort", EXPOSURE, OUTCOME],
            "expected_outputs": ["table:exposure_outcome_distribution"],
        }
    )
    assert not exposure_outcome_distribution_executor_owns_step(step)
    with pytest.raises(ValueError):
        exposure_outcome_distribution_executor_code(step)


def test_a_scientific_or_widened_contract_is_refused() -> None:
    assert not exposure_outcome_distribution_executor_owns_step(
        _step(planned_analysis_role="primary")
    )
    assert not exposure_outcome_distribution_executor_owns_step(
        _step(method="adjusted_association_models")
    )
    assert not exposure_outcome_distribution_executor_owns_step(
        _step(
            expected_outputs=[
                "table:exposure_outcome_distribution",
                "figure:extra",
            ]
        )
    )


def test_the_executor_carries_no_case_specific_branch() -> None:
    import easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor as module

    source = Path(module.__file__).read_text().lower()
    for token in ("sepsis", "sep3", "mortality", "death", "e1_", "icu_readmission"):
        assert token not in source, f"case-specific token in production: {token}"


# --------------------------------------------------------------------------
# The product
# --------------------------------------------------------------------------


def _frame() -> pd.DataFrame:
    # 10 exposed (3 events), 10 unexposed (1 event), 2 outcomes unobserved.
    return pd.DataFrame(
        {
            EXPOSURE: [1] * 10 + [0] * 10,
            OUTCOME: (
                [1, 1, 1, 0, 0, 0, 0, 0, 0, None] + [1, 0, 0, 0, 0, 0, 0, 0, 0, None]
            ),
        }
    )


def test_the_product_is_self_contained(monkeypatch, tmp_path: Path) -> None:
    """A renderer must not need a second table to draw this."""

    summary = _run(monkeypatch, tmp_path, _frame())
    table = pd.read_csv(
        tmp_path
        / "run"
        / "steps"
        / STEP_ID
        / "outputs"
        / summary["output_files"]["table:exposure_outcome_distribution"]
    )
    assert list(table.columns) == list(EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS)

    exposed = table[
        (table["row_role"] == "exposure_level") & (table["exposure_level"] == 1)
    ].iloc[0]
    assert exposed["n_rows"] == 10
    assert exposed["outcome_events"] == 3
    assert exposed["outcome_observed_n"] == 9
    assert exposed["outcome_missing_n"] == 1
    assert exposed["exposure_denominator"] == 20
    assert exposed["exposure_pct"] == pytest.approx(50.0)
    # all_declared_rows: the rate is over all 10, not over the 9 observed.
    assert exposed["outcome_denominator"] == 10
    assert exposed["outcome_rate_pct"] == pytest.approx(30.0)
    assert exposed["ci_low_pct"] < 30.0 < exposed["ci_high_pct"]

    overall = table[table["row_role"] == "overall"].iloc[0]
    assert overall["n_rows"] == 20
    assert overall["outcome_events"] == 4
    assert overall["outcome_missing_n"] == 2


def test_the_denominator_policy_changes_the_reported_rate(
    monkeypatch, tmp_path: Path
) -> None:
    """The field earns its place: the two policies are different quantities."""

    over_all = _run(monkeypatch, tmp_path / "a", _frame())
    over_observed = _run(
        monkeypatch,
        tmp_path / "b",
        _frame(),
        denominator_policy="observed_outcome_rows",
    )

    def _exposed_rate(summary: dict, root: Path) -> float:
        table = pd.read_csv(
            root
            / "run"
            / "steps"
            / STEP_ID
            / "outputs"
            / summary["output_files"]["table:exposure_outcome_distribution"]
        )
        row = table[
            (table["row_role"] == "exposure_level") & (table["exposure_level"] == 1)
        ].iloc[0]
        return float(row["outcome_rate_pct"])

    assert _exposed_rate(over_all, tmp_path / "a") == pytest.approx(30.0)  # 3/10
    assert _exposed_rate(over_observed, tmp_path / "b") == pytest.approx(100.0 * 3 / 9)


def test_the_declared_event_value_is_honoured_not_assumed(
    monkeypatch, tmp_path: Path
) -> None:
    """A binary outcome is not always encoded 1/0."""

    frame = pd.DataFrame({EXPOSURE: [1, 1, 0, 0], OUTCOME: ["yes", "no", "yes", "no"]})
    summary = _run(monkeypatch, tmp_path, frame, outcome_positive_value="yes")
    table = pd.read_csv(
        tmp_path
        / "run"
        / "steps"
        / STEP_ID
        / "outputs"
        / summary["output_files"]["table:exposure_outcome_distribution"]
    )
    assert int(table[table["row_role"] == "overall"].iloc[0]["outcome_events"]) == 2


def test_a_row_matching_no_declared_level_fails_closed(
    monkeypatch, tmp_path: Path
) -> None:
    """Neither dropped nor pooled: the spec says fail closed, so it fails."""

    frame = pd.DataFrame({EXPOSURE: [0, 1, 2], OUTCOME: [0, 1, 0]})
    with pytest.raises(RuntimeError, match="do not match any declared exposure level"):
        _run(monkeypatch, tmp_path, frame)


def test_a_numeric_level_matches_a_numerically_stored_column(
    monkeypatch, tmp_path: Path
) -> None:
    """A declared numeric level matches the same number however it is stored.

    Deliberate, and worth stating: a prepared column may arrive string-typed,
    and refusing it would fail-close a run whose design is entirely correct.
    What must not happen is matching a *different* category, which the next
    test covers.
    """

    frame = pd.DataFrame({EXPOSURE: ["0", "1"], OUTCOME: [0, 1]})
    summary = _run(monkeypatch, tmp_path, frame)
    assert summary["cohort_n"] == 2


def test_a_differently_categorised_column_fails_closed(
    monkeypatch, tmp_path: Path
) -> None:
    """Levels declared 0/1 must not quietly absorb a yes/no column."""

    frame = pd.DataFrame({EXPOSURE: ["yes", "no"], OUTCOME: [0, 1]})
    with pytest.raises(RuntimeError, match="do not match any declared exposure level"):
        _run(monkeypatch, tmp_path, frame)


def test_a_cohort_whose_digest_does_not_match_is_refused(
    monkeypatch, tmp_path: Path
) -> None:
    run_dir, out_dir = _cohort(tmp_path, _frame())
    manifest_path = run_dir / "resolved_inputs.json"
    payload = json.loads(manifest_path.read_text())
    payload["inputs"]["artifact:analysis_cohort"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(run_dir))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(manifest_path))
    with pytest.raises(RuntimeError, match="digest verification failed"):
        run_exposure_outcome_distribution_from_env(
            spec_payload=_SPEC, typed_cohort_input="artifact:analysis_cohort"
        )


def test_a_declared_column_absent_from_the_cohort_is_refused(
    monkeypatch, tmp_path: Path
) -> None:
    frame = pd.DataFrame({EXPOSURE: [0, 1], "something_else": [0, 1]})
    with pytest.raises(RuntimeError, match="absent from the bound cohort"):
        _run(monkeypatch, tmp_path, frame)
