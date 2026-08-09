"""The exposure-by-outcome distribution owner.

Every scientific choice is the Planner's. These tests pin that: the executor
must refuse a step that has not declared the design, must never infer the
exposure from names or input order, must never read an unbound cohort, and must
produce a product a renderer can draw *and re-check* without a second table.

The sharpest of these are the negative tests. A distribution executor fails
quietly rather than loudly -- an undeclared outcome value is observed, is not
the event, and would be counted as a non-event unless something refuses it --
so the tests that matter most are the ones proving it refuses.

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

from pydantic import ValidationError

from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
    EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS,
    _distribution_rows,
    exposure_outcome_distribution_executor_code,
    exposure_outcome_distribution_executor_owns_step,
    run_exposure_outcome_distribution_from_env,
    wilson_interval,
)
from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ExposureOutcomeDistributionSpec,
)

STEP_ID = "03_drug_readmission_distribution"
EXPOSURE = "anticoagulant_exposed"
OUTCOME = "readmitted_30d"

_SPEC = {
    "exposure": EXPOSURE,
    "exposure_levels": [0, 1],
    "outcome": OUTCOME,
    "outcome_levels": [0, 1],
    "outcome_positive_value": 1,
    "level_match_policy": "exact_typed",
    "denominator_policy": "all_declared_rows",
    "missing_outcome_policy": "structural_absence_is_non_event",
    "confidence_level": 0.95,
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
    digest = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
    manifest = {
        "step_id": STEP_ID,
        "inputs": {
            "artifact:analysis_cohort": {
                "relative_path": "cohort.parquet",
                "sha256": digest,
                "declared_kind": "artifact",
                "product": "analysis_cohort",
                "evidence_id": "ev-cohort",
                "identity_row": {
                    "input_key": "artifact:analysis_cohort",
                    "declared_kind": "artifact",
                    "product": "analysis_cohort",
                    "evidence_id": "ev-cohort",
                    "sha256": digest,
                },
                "product_contract": {
                    "columns": list(frame.columns),
                    "row_count": int(len(frame)),
                },
            }
        },
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


def _table(summary: dict, root: Path) -> pd.DataFrame:
    return pd.read_csv(
        root
        / "run"
        / "steps"
        / STEP_ID
        / "outputs"
        / summary["output_files"]["table:exposure_outcome_distribution"]
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


def test_a_step_with_no_typed_cohort_input_is_never_claimed() -> None:
    """No binding, no ownership -- there is no unbound-cohort fallback.

    A cohort handed over as a bare path has no digest, no product contract and
    no named producer, so a table counted from it cannot be bound to the plan
    that asked for it. Claiming the step anyway would put an unverifiable
    number into the run under a deterministic label.
    """

    step = _step(inputs=[EXPOSURE, OUTCOME])
    assert not exposure_outcome_distribution_executor_owns_step(step)
    with pytest.raises(ValueError):
        exposure_outcome_distribution_executor_code(step)


def test_the_runtime_refuses_an_unbound_cohort_too() -> None:
    """The runtime entry point re-checks; it does not trust its caller."""

    with pytest.raises(RuntimeError, match="requires an exact typed cohort binding"):
        run_exposure_outcome_distribution_from_env(
            spec_payload=_SPEC, typed_cohort_input=""
        )


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
# The declaration must be closed before it is executed
# --------------------------------------------------------------------------


def test_a_positive_value_outside_the_closed_outcome_set_is_refused() -> None:
    """Otherwise every remaining level is a non-event by omission."""

    with pytest.raises(ValueError, match="must be one of outcome_levels"):
        ExposureOutcomeDistributionSpec.model_validate(
            {**_SPEC, "outcome_levels": [0, 1], "outcome_positive_value": 2}
        )


def test_a_positive_value_of_the_wrong_type_is_refused() -> None:
    """``1`` and ``"1"`` are different declarations, so the check is typed."""

    with pytest.raises(ValueError, match="must be one of outcome_levels"):
        ExposureOutcomeDistributionSpec.model_validate(
            {**_SPEC, "outcome_levels": ["0", "1"], "outcome_positive_value": 1}
        )


@pytest.mark.parametrize(
    ("missing_policy", "denominator_policy"),
    [
        ("exclude_from_denominator", "all_declared_rows"),
        ("structural_absence_is_non_event", "observed_outcome_rows"),
    ],
)
def test_contradictory_missing_and_denominator_policies_are_refused(
    missing_policy: str, denominator_policy: str
) -> None:
    """Complete-case and carry-the-missing are opposite denominators."""

    with pytest.raises(ValueError, match="requires denominator_policy"):
        ExposureOutcomeDistributionSpec.model_validate(
            {
                **_SPEC,
                "missing_outcome_policy": missing_policy,
                "denominator_policy": denominator_policy,
            }
        )


def test_the_confidence_level_must_be_declared() -> None:
    """No coverage is hard-coded, so the study has to state one."""

    payload = {key: value for key, value in _SPEC.items() if key != "confidence_level"}
    with pytest.raises(ValueError):
        ExposureOutcomeDistributionSpec.model_validate(payload)


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
    """A renderer must not need a second table, nor the spec, to check this."""

    summary = _run(monkeypatch, tmp_path, _frame())
    table = _table(summary, tmp_path)
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

    # The design travels with the numbers, identically on every row.
    assert exposed["outcome_column"] == OUTCOME
    declared_outcome_levels = json.loads(exposed["outcome_levels_declared"])
    assert declared_outcome_levels == [0, 1]
    # The event is identified by position, because a CSV cell cannot tell 1
    # from "1" and that is exactly the distinction the level policy protects.
    assert declared_outcome_levels[int(exposed["outcome_positive_index"])] == 1
    assert exposed["missing_outcome_policy"] == "structural_absence_is_non_event"
    assert exposed["confidence_level"] == pytest.approx(0.95)
    assert table["denominator_policy"].nunique() == 1


def test_the_declared_confidence_level_is_the_one_reported(
    monkeypatch, tmp_path: Path
) -> None:
    """No 95% is baked in: a different declaration produces a wider interval."""

    narrow = _table(_run(monkeypatch, tmp_path / "a", _frame()), tmp_path / "a")
    wide = _table(
        _run(monkeypatch, tmp_path / "b", _frame(), confidence_level=0.99),
        tmp_path / "b",
    )
    narrow_row = narrow[narrow["row_role"] == "overall"].iloc[0]
    wide_row = wide[wide["row_role"] == "overall"].iloc[0]
    assert wide_row["ci_low_pct"] < narrow_row["ci_low_pct"]
    assert wide_row["ci_high_pct"] > narrow_row["ci_high_pct"]
    assert (wide_row["ci_low_pct"], wide_row["ci_high_pct"]) == pytest.approx(
        wilson_interval(4, 20, confidence_level=0.99)
    )


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
        missing_outcome_policy="exclude_from_denominator",
    )

    def _exposed_rate(summary: dict, root: Path) -> float:
        table = _table(summary, root)
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
    summary = _run(
        monkeypatch,
        tmp_path,
        frame,
        outcome_levels=["no", "yes"],
        outcome_positive_value="yes",
        missing_outcome_policy="fail_closed",
    )
    table = _table(summary, tmp_path)
    assert int(table[table["row_role"] == "overall"].iloc[0]["outcome_events"]) == 2


# --------------------------------------------------------------------------
# What must fail closed
# --------------------------------------------------------------------------


def test_an_undeclared_outcome_value_fails_closed(monkeypatch, tmp_path: Path) -> None:
    """The defect this closed set exists for.

    A ``2`` in a column believed to be 0/1 is observed and is not the event, so
    without a closed outcome set it is counted as a non-event: the rate drops,
    the table still balances, and nothing downstream can tell.
    """

    frame = pd.DataFrame({EXPOSURE: [0, 0, 1, 1], OUTCOME: [0, 1, 2, 1]})
    with pytest.raises(RuntimeError, match="not one of the declared outcome levels"):
        _run(monkeypatch, tmp_path, frame, missing_outcome_policy="fail_closed")


def test_the_refusal_does_not_echo_the_undeclared_values(
    monkeypatch, tmp_path: Path
) -> None:
    """Counts make the failure actionable; the values are cohort data.

    A mis-declared column could be a continuous measurement, so the message
    reports how many rows and how many distinct values were undeclared and
    stops there.
    """

    frame = pd.DataFrame({EXPOSURE: [0, 1], OUTCOME: [0, 987654]})
    with pytest.raises(RuntimeError) as excinfo:
        _run(monkeypatch, tmp_path, frame, missing_outcome_policy="fail_closed")
    assert "987654" not in str(excinfo.value)
    assert "1 distinct undeclared" in str(excinfo.value)


def test_a_boolean_column_is_not_absorbed_by_numeric_levels(
    monkeypatch, tmp_path: Path
) -> None:
    """``True == 1`` in Python; a study's variables are not interchangeable.

    Without this, a boolean column answers levels declared ``0``/``1`` and the
    table reports a different variable from the one the plan declared.
    """

    frame = pd.DataFrame({EXPOSURE: [True, False], OUTCOME: [0, 1]})
    with pytest.raises(RuntimeError, match="not one of the declared exposure levels"):
        _run(monkeypatch, tmp_path, frame, missing_outcome_policy="fail_closed")


def test_a_boolean_declaration_matches_only_a_boolean_column(
    monkeypatch, tmp_path: Path
) -> None:
    """And the mirror: numeric rows never answer a boolean level."""

    boolean_frame = pd.DataFrame({EXPOSURE: [True, False], OUTCOME: [0, 1]})
    summary = _run(
        monkeypatch,
        tmp_path / "ok",
        boolean_frame,
        exposure_levels=[False, True],
        missing_outcome_policy="fail_closed",
    )
    assert summary["cohort_n"] == 2

    numeric_frame = pd.DataFrame({EXPOSURE: [1, 0], OUTCOME: [0, 1]})
    with pytest.raises(RuntimeError, match="not one of the declared exposure levels"):
        _run(
            monkeypatch,
            tmp_path / "refused",
            numeric_frame,
            exposure_levels=[False, True],
            missing_outcome_policy="fail_closed",
        )


def test_numeric_string_equivalence_is_declared_not_assumed(
    monkeypatch, tmp_path: Path
) -> None:
    """A prepared column may store codes as text -- but that is a declaration.

    Under ``exact_typed`` the same data fails closed. Silently coercing would
    make the policy field decorative and would mean the host, not the study,
    decided what counts as the same value.
    """

    frame = pd.DataFrame({EXPOSURE: ["0", "1"], OUTCOME: [0, 1]})
    summary = _run(
        monkeypatch,
        tmp_path / "declared",
        frame,
        level_match_policy="numeric_string_equivalent",
        missing_outcome_policy="fail_closed",
    )
    assert summary["cohort_n"] == 2

    with pytest.raises(RuntimeError, match="not one of the declared exposure levels"):
        _run(
            monkeypatch,
            tmp_path / "exact",
            frame,
            level_match_policy="exact_typed",
            missing_outcome_policy="fail_closed",
        )


def test_a_differently_categorised_column_fails_closed(
    monkeypatch, tmp_path: Path
) -> None:
    """Levels declared 0/1 must not quietly absorb a yes/no column."""

    frame = pd.DataFrame({EXPOSURE: ["yes", "no"], OUTCOME: [0, 1]})
    with pytest.raises(RuntimeError, match="not one of the declared exposure levels"):
        _run(
            monkeypatch,
            tmp_path,
            frame,
            level_match_policy="numeric_string_equivalent",
            missing_outcome_policy="fail_closed",
        )


def test_a_missing_exposure_is_diagnosed_as_missing_not_as_an_odd_category(
    monkeypatch, tmp_path: Path
) -> None:
    """Both stop the step, but they send a reader to different places.

    Folding a missing exposure into the undeclared-level message would have
    someone hunting for a stray category code that does not exist.
    """

    frame = pd.DataFrame({EXPOSURE: [0, 1, None], OUTCOME: [0, 1, 0]})
    with pytest.raises(RuntimeError, match="no observed value") as excinfo:
        _run(monkeypatch, tmp_path, frame, missing_outcome_policy="fail_closed")
    assert "not one of the declared" not in str(excinfo.value)
    assert EXPOSURE in str(excinfo.value)


def test_a_missing_outcome_under_fail_closed_stops_the_step(
    monkeypatch, tmp_path: Path
) -> None:
    """A study that declared no missingness must not silently acquire some."""

    with pytest.raises(RuntimeError, match="no observed value"):
        _run(
            monkeypatch,
            tmp_path,
            _frame(),
            missing_outcome_policy="fail_closed",
        )


def test_a_cohort_whose_digest_does_not_match_is_refused(
    monkeypatch, tmp_path: Path
) -> None:
    run_dir, out_dir = _cohort(tmp_path, _frame())
    manifest_path = run_dir / "resolved_inputs.json"
    payload = json.loads(manifest_path.read_text())
    binding = payload["inputs"]["artifact:analysis_cohort"]
    binding["sha256"] = "0" * 64
    binding["identity_row"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(run_dir))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(manifest_path))
    with pytest.raises(RuntimeError, match="digest verification failed"):
        run_exposure_outcome_distribution_from_env(
            spec_payload=_SPEC, typed_cohort_input="artifact:analysis_cohort"
        )


def test_a_capsule_disagreeing_with_its_own_identity_row_is_refused(
    monkeypatch, tmp_path: Path
) -> None:
    """One record, not four fields that happen to sit near each other."""

    run_dir, out_dir = _cohort(tmp_path, _frame())
    manifest_path = run_dir / "resolved_inputs.json"
    payload = json.loads(manifest_path.read_text())
    payload["inputs"]["artifact:analysis_cohort"]["identity_row"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(run_dir))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(manifest_path))
    with pytest.raises(RuntimeError, match="identity_row disagrees"):
        run_exposure_outcome_distribution_from_env(
            spec_payload=_SPEC, typed_cohort_input="artifact:analysis_cohort"
        )


def test_a_capsule_naming_another_product_is_refused(
    monkeypatch, tmp_path: Path
) -> None:
    run_dir, out_dir = _cohort(tmp_path, _frame())
    manifest_path = run_dir / "resolved_inputs.json"
    payload = json.loads(manifest_path.read_text())
    binding = payload["inputs"]["artifact:analysis_cohort"]
    binding["product"] = "some_other_table"
    binding["identity_row"]["product"] = "some_other_table"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(run_dir))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(manifest_path))
    with pytest.raises(RuntimeError, match="does not match the input key"):
        run_exposure_outcome_distribution_from_env(
            spec_payload=_SPEC, typed_cohort_input="artifact:analysis_cohort"
        )


def test_a_declared_column_absent_from_the_cohort_is_refused(
    monkeypatch, tmp_path: Path
) -> None:
    frame = pd.DataFrame({EXPOSURE: [0, 1], "something_else": [0, 1]})
    with pytest.raises(RuntimeError, match="absent from the bound cohort"):
        _run(monkeypatch, tmp_path, frame)


# ---------------------------------------------------------------------------
# canary13: the host demanded a policy choice it did not offer
#
# Step 05 died with "8 rows have no observed value for 'aki_stage_max'; the
# spec declares missing_exposure_policy='fail_closed'". The field was
# Literal["fail_closed"] -- there was no other value the Planner could have
# written. So the owner could not be used on any cohort whose exposure had a
# single missing value, and nothing about the plan was wrong.
# ---------------------------------------------------------------------------


def _frame_with_unobserved_exposure():
    import numpy as np

    frame = pd.DataFrame(
        {
            "sep3": [0.0, 0.0, 1.0, 1.0, 1.0, np.nan, np.nan],
            "death": [0, 1, 0, 1, 1, 1, 0],
        }
    )
    return frame


def _distribution_spec(**overrides):
    payload = dict(
        schema_version="easyicu.exposure_outcome_distribution/2",
        exposure="sep3",
        exposure_levels=[0.0, 1.0],
        outcome="death",
        outcome_levels=[0, 1],
        outcome_positive_value=1,
        level_match_policy="exact_typed",
        denominator_policy="all_declared_rows",
        missing_outcome_policy="fail_closed",
        confidence_level=0.95,
    )
    payload.update(overrides)
    return ExposureOutcomeDistributionSpec.model_validate(payload)


def test_a_missing_exposure_can_now_be_declared_complete_case() -> None:
    """The Planner has a second answer, and it is the standard one."""

    spec = _distribution_spec(missing_exposure_policy="exclude_from_denominator")

    rows = _distribution_rows(_frame_with_unobserved_exposure(), spec=spec)

    overall = next(row for row in rows if row["row_role"] == "overall")
    # Seven stays, two with no exposure: five are analysed, and every
    # denominator is over those same five.
    assert overall["n_rows"] == 5
    assert overall["exposure_denominator"] == 5
    assert (
        sum(row["n_rows"] for row in rows if row["row_role"] == "exposure_level") == 5
    )


def test_the_rows_that_left_are_counted_in_the_product() -> None:
    """A denominator that silently shrank is one a reader cannot check.

    The count travels on every row, the way Table 1 carries
    ``group_missing_excluded_n`` -- so a reader holding only this CSV can tell
    5-of-7 from 5-of-5.
    """

    spec = _distribution_spec(missing_exposure_policy="exclude_from_denominator")

    rows = _distribution_rows(_frame_with_unobserved_exposure(), spec=spec)

    assert {row["missing_exposure_excluded_n"] for row in rows} == {2}
    assert {row["missing_exposure_policy"] for row in rows} == {
        "exclude_from_denominator"
    }
    assert "missing_exposure_excluded_n" in EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS


def test_fail_closed_is_still_the_default_and_still_stops_the_step() -> None:
    """The new option is opt-in; silence still means refuse."""

    assert _distribution_spec().missing_exposure_policy == "fail_closed"

    with pytest.raises(RuntimeError, match="fail_closed"):
        _distribution_rows(_frame_with_unobserved_exposure(), spec=_distribution_spec())


def test_a_fully_observed_exposure_reports_zero_excluded() -> None:
    """Zero is a result: its absence cannot be told from never having looked."""

    frame = _frame_with_unobserved_exposure().dropna(subset=["sep3"])

    rows = _distribution_rows(frame, spec=_distribution_spec())

    assert {row["missing_exposure_excluded_n"] for row in rows} == {0}


def test_there_is_no_policy_that_pools_an_unobserved_exposure() -> None:
    """The third option a reader might expect is the one that must not exist.

    An unobserved exposure is not the reference and not any other category.
    The adjusted-association owner made exactly that mistake by accident --
    treatment coding encodes a missing value identically to the reference --
    and it reported 8 stays under a stage nobody recorded. Offering it here as
    a *declarable* policy would make that a supported feature.
    """

    with pytest.raises(ValidationError):
        _distribution_spec(missing_exposure_policy="structural_absence_is_reference")
    with pytest.raises(ValidationError):
        _distribution_spec(missing_exposure_policy="pool_into_reference")
