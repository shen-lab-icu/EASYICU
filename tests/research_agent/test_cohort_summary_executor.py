from __future__ import annotations

import ast
import hashlib
import json

import pandas as pd
import pytest

from easyicu.research_agent.authority.plausibility import (
    FlagOnlyPlausibilityScope,
)
from easyicu.research_agent.gates.plausibility_obligation import (
    flag_only_plausibility_obligation_findings,
)
from easyicu.research_agent.gates.plausibility_receipt import (
    plausibility_audit_receipt_findings,
)
from easyicu.research_agent.execution.runners.cohort_summary_executor import (
    cohort_summary_executor_code,
    cohort_summary_executor_owns_step,
    run_cohort_summary_from_env,
)
from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _step() -> AnalysisStep:
    return AnalysisStep(
        step_id="02_cohort_summary",
        planned_analysis_role="auxiliary",
        intent="Describe the exact closed cohort columns.",
        inputs=[
            "artifact:analysis_cohort",
            "age",
            "sex",
            "exposure",
            "outcome",
        ],
        expected_outputs=["table:cohort_summary"],
        method="descriptive_cohort_summary",
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [50.0, 60.0, 70.0, 80.0],
            "sex": ["Female", "Male", "Female", "Male"],
            "exposure": [0.0, 1.0, 0.0, 1.0],
            "outcome": [0, 0, 1, 1],
        }
    )


def _context() -> dict:
    return {
        "variables": [
            {
                "name": "age",
                "unit": "years",
                "observed_domain": {
                    "n_unique": 4,
                    "is_binary": False,
                    "min": 50.0,
                    "max": 80.0,
                },
            },
            {
                "name": "sex",
                "unit": None,
                "observed_domain": {
                    "n_unique": 2,
                    "is_binary": False,
                    "levels": ["Female", "Male"],
                },
            },
            {
                "name": "exposure",
                "unit": None,
                "observed_domain": {
                    "n_unique": 2,
                    "is_binary": True,
                    "levels": [0.0, 1.0],
                },
            },
            {
                "name": "outcome",
                "unit": None,
                "observed_domain": {
                    "n_unique": 2,
                    "is_binary": True,
                    "levels": [0, 1],
                },
            },
        ]
    }


def _bind_run(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cohort_path = run_dir / "analysis_cohort.parquet"
    frame = _frame()
    frame.to_parquet(cohort_path, index=False)
    manifest = {
        "inputs": {
            "artifact:analysis_cohort": {
                "relative_path": cohort_path.relative_to(run_dir).as_posix(),
                "sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
                "product_contract": {
                    "columns": list(frame.columns),
                    "row_count": len(frame),
                },
            }
        }
    }
    manifest_path = run_dir / "resolved_inputs.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    context_path = run_dir / "research_context.json"
    context_path.write_text(json.dumps(_context()), encoding="utf-8")
    out_dir = run_dir / "steps" / "02_cohort_summary" / "outputs"
    monkeypatch.setenv("EASYICU_RUN_DIR", str(run_dir))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(manifest_path))
    monkeypatch.setenv("EASYICU_RESEARCH_CONTEXT", str(context_path))
    monkeypatch.setenv("COHORT_PARQUET", str(cohort_path))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    return cohort_path, out_dir


def test_cohort_summary_executor_owns_only_closed_auxiliary_contract():
    step = _step()
    assert cohort_summary_executor_owns_step(step)

    # Canonical9 planners may use the generic method head, but the remaining
    # closed contract still uniquely identifies this mechanical summary.
    assert cohort_summary_executor_owns_step(
        step.model_copy(update={"method": "descriptive"})
    )
    assert not cohort_summary_executor_owns_step(
        step.model_copy(update={"planned_analysis_role": "primary"})
    )
    assert not cohort_summary_executor_owns_step(
        step.model_copy(
            update={
                "expected_outputs": [
                    "table:cohort_summary",
                    "statistic:adjusted_or",
                ]
            }
        )
    )
    assert not cohort_summary_executor_owns_step(
        step.model_copy(update={"inputs": [*step.inputs, "table:unrelated_parent"]})
    )


def test_cohort_summary_is_selected_before_coder_and_declares_consumption():
    step = _step()
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Test", steps=[step]),
    )

    assert selection is not None
    assert selection.analysis_kind == "descriptive_cohort_summary"
    assert selection.selection_reason == "cohort_summary_contract_preflight"
    assert selection.consumed_input_keys == ("artifact:analysis_cohort",)
    assert audit_mechanical_code_contracts(selection.code, step) == []


def test_cohort_summary_executes_exact_metadata_levels_and_numeric_statistics(
    tmp_path,
    monkeypatch,
):
    _, out_dir = _bind_run(tmp_path, monkeypatch)
    step = _step()

    exec(compile(cohort_summary_executor_code(step), "<cohort-summary>", "exec"), {})

    table = pd.read_csv(out_dir / "cohort_summary.csv")
    summary = json.loads((out_dir / "step_summary.json").read_text("utf-8"))
    cohort_row = table[
        (table["variable"] == "__cohort__") & (table["statistic"] == "cohort_n")
    ].iloc[0]
    age_median = table[
        (table["variable"] == "age") & (table["statistic"] == "median")
    ].iloc[0]
    exposed = table[
        (table["variable"] == "exposure")
        & (table["statistic"] == "level_count")
        & (table["level"].astype(str) == "1.0")
    ].iloc[0]

    assert cohort_row["value"] == 4
    assert age_median["value"] == 65.0
    assert exposed["numerator"] == 2
    assert exposed["denominator"] == 4
    assert exposed["percentage"] == 50.0
    assert summary["status"] == "ok"
    assert summary["cohort_n"] == 4
    assert summary["adjusted_effect"] is None
    assert summary["output_files"] == {"table:cohort_summary": "cohort_summary.csv"}
    assert summary["source_row_count_reconciliation"] == {
        "source_rows": 4,
        "analyzed_rows": 4,
        "filtering_performed": False,
    }


def test_cohort_summary_rejects_digest_drift(tmp_path, monkeypatch):
    cohort_path, _out_dir = _bind_run(tmp_path, monkeypatch)
    cohort_path.write_bytes(cohort_path.read_bytes() + b"tamper")

    with pytest.raises(RuntimeError, match="digest verification failed"):
        exec(
            compile(
                cohort_summary_executor_code(_step()),
                "<cohort-summary>",
                "exec",
            ),
            {},
        )


def test_cohort_summary_rejects_missing_structured_metadata(tmp_path, monkeypatch):
    _cohort_path, _out_dir = _bind_run(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    context = _context()
    context["variables"] = [
        item for item in context["variables"] if item["name"] != "outcome"
    ]
    (run_dir / "research_context.json").write_text(
        json.dumps(context),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="lack structured metadata: outcome"):
        exec(
            compile(
                cohort_summary_executor_code(_step()),
                "<cohort-summary>",
                "exec",
            ),
            {},
        )


# --------------------------------------------------------------------------
# The flag-only plausibility receipt.
#
# This executor used to decline every step that owed one, because the
# obligation gate proves the obligation from the source that will run and a
# lone call into an imported helper is not attributable to it.  The real E1
# Step 02 was recorded as ``declined_receipt_required`` and handed to the LLM
# coder -- a step the host can compute exactly.
# --------------------------------------------------------------------------


def _contracts(ranges: dict[str, tuple[float | None, float | None]]) -> dict:
    payload = {
        "contracts": {
            column: {
                "column": column,
                "analysis_plausibility_range": {
                    "minimum": minimum,
                    "maximum": maximum,
                },
                "plausibility_policy": {
                    "range_policy": "flag_only",
                    "out_of_range_action": "retain_and_flag",
                },
            }
            for column, (minimum, maximum) in ranges.items()
        }
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {**payload, "contracts_sha256": digest}


def _bind_run_with_contracts(tmp_path, monkeypatch, *, frame, ranges):
    """Bind a run whose manifest declares sealed flag-only contracts."""

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cohort_path = run_dir / "analysis_cohort.parquet"
    frame.to_parquet(cohort_path, index=False)
    contracts = _contracts(ranges)
    manifest = {
        "inputs": {
            "artifact:analysis_cohort": {
                "relative_path": cohort_path.relative_to(run_dir).as_posix(),
                "sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
                "product_contract": {
                    "columns": list(frame.columns),
                    "row_count": len(frame),
                },
            }
        },
        "raw_input_contracts": contracts,
    }
    manifest_path = run_dir / "resolved_inputs.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    context_path = run_dir / "research_context.json"
    context_path.write_text(json.dumps(_context()), encoding="utf-8")
    out_dir = run_dir / "steps" / "02_cohort_summary" / "outputs"
    monkeypatch.setenv("EASYICU_RUN_DIR", str(run_dir))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(manifest_path))
    monkeypatch.setenv("EASYICU_RESEARCH_CONTEXT", str(context_path))
    monkeypatch.setenv("COHORT_PARQUET", str(cohort_path))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    return contracts["contracts_sha256"], out_dir


def _scope(digest: str, *columns: str) -> FlagOnlyPlausibilityScope:
    return FlagOnlyPlausibilityScope(
        step_id="02_cohort_summary",
        expected_columns=tuple(sorted(columns)),
        source_contracts_sha256=digest,
        authority_kind="resolved_raw_input_contracts",
    )


def test_a_receipt_bearing_cohort_summary_is_owned_not_handed_to_the_coder():
    step = _step()
    scope = _scope("a" * 64, "age")

    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Test", steps=[step]),
        plausibility_scope=scope,
    )

    assert selection is not None
    assert selection.analysis_kind == "descriptive_cohort_summary"


def test_the_generated_receipt_code_satisfies_the_obligation_gate():
    """The gate, not a shape assertion, is what decides the step's fate."""

    step = _step()
    scope = _scope("a" * 64, "age", "exposure")
    code = cohort_summary_executor_code(step, plausibility_scope=scope)

    findings = flag_only_plausibility_obligation_findings(
        ast.parse(code),
        script_text=code,
        step=step,
        scope=scope,
    )

    assert findings == []
    assert audit_mechanical_code_contracts(code, step) == []


def test_the_receipt_reaches_the_step_summary_the_host_reads(
    tmp_path,
    monkeypatch,
):
    frame = _frame()
    frame.loc[0, "age"] = 200.0  # above the declared maximum, retained
    digest, out_dir = _bind_run_with_contracts(
        tmp_path,
        monkeypatch,
        frame=frame,
        ranges={"age": (0.0, 120.0)},
    )
    step = _step()
    scope = _scope(digest, "age")
    code = cohort_summary_executor_code(step, plausibility_scope=scope)

    exec(compile(code, "<cohort-summary>", "exec"), {})

    summary = json.loads((out_dir / "step_summary.json").read_text("utf-8"))
    receipt = summary["plausibility_audit"]["age"]
    assert receipt["policy"] == "retain_and_flag"
    assert receipt["above_maximum_n"] == 1
    assert receipt["below_minimum_n"] == 0
    assert receipt["out_of_range_n"] == 1
    # Every row is retained: flagging is not filtering.
    assert summary["cohort_n"] == 4
    assert receipt["compared_n"] == 4

    # The post-execution half reads the sealed artifact, not the source.
    assert (
        plausibility_audit_receipt_findings(
            step_summary=summary,
            step=step,
            script_text=code,
            scope=scope,
        )
        == []
    )


def test_a_column_with_nothing_to_compare_is_not_a_clean_bill_of_health(
    tmp_path,
    monkeypatch,
):
    """ "None out of range" and "nothing was there" must not look identical.

    Counts alone cannot separate them: an entirely missing column and a fully
    observed, entirely in-range one both report ``out_of_range_n = 0``.  The
    obligation gate already refuses a receipt that appears only when the count
    is nonzero for exactly this reason; a receipt with no denominator loses the
    same distinction one level down.  Partly recorded outcomes -- death being
    the ordinary case -- are where it bites.
    """

    frame = _frame()
    frame["age"] = [float("nan")] * len(frame)
    digest, out_dir = _bind_run_with_contracts(
        tmp_path,
        monkeypatch,
        frame=frame,
        ranges={"age": (0.0, 120.0)},
    )
    step = _step()

    exec(
        compile(
            cohort_summary_executor_code(
                step,
                plausibility_scope=_scope(digest, "age"),
            ),
            "<cohort-summary>",
            "exec",
        ),
        {},
    )

    receipt = json.loads((out_dir / "step_summary.json").read_text("utf-8"))[
        "plausibility_audit"
    ]["age"]

    assert receipt["out_of_range_n"] == 0
    assert receipt["compared_n"] == 0
    assert receipt["observed_n"] == 0


def test_a_partly_recorded_column_reports_what_it_actually_compared(
    tmp_path,
    monkeypatch,
):
    frame = _frame()
    frame.loc[[1, 2], "age"] = float("nan")
    digest, out_dir = _bind_run_with_contracts(
        tmp_path,
        monkeypatch,
        frame=frame,
        ranges={"age": (0.0, 120.0)},
    )

    exec(
        compile(
            cohort_summary_executor_code(
                _step(),
                plausibility_scope=_scope(digest, "age"),
            ),
            "<cohort-summary>",
            "exec",
        ),
        {},
    )

    summary = json.loads((out_dir / "step_summary.json").read_text("utf-8"))
    receipt = summary["plausibility_audit"]["age"]

    assert receipt["compared_n"] == 2
    assert receipt["out_of_range_n"] == 0
    # The missing rows are still in the cohort; the receipt is not a filter.
    assert summary["cohort_n"] == 4


def test_contract_drift_fails_the_step_instead_of_producing_a_receipt(
    tmp_path,
    monkeypatch,
):
    """A receipt computed against contracts other than the sealed ones is void."""

    digest, _out_dir = _bind_run_with_contracts(
        tmp_path,
        monkeypatch,
        frame=_frame(),
        ranges={"age": (0.0, 120.0)},
    )
    assert digest != "b" * 64

    with pytest.raises(RuntimeError, match="do not match the step authority"):
        exec(
            compile(
                cohort_summary_executor_code(
                    _step(),
                    plausibility_scope=_scope("b" * 64, "age"),
                ),
                "<cohort-summary>",
                "exec",
            ),
            {},
        )


def test_a_receipt_that_misses_a_scoped_column_is_refused_by_the_host(
    tmp_path,
    monkeypatch,
):
    """The host decides the receipt is complete, not whoever calls it."""

    _digest, _out_dir = _bind_run_with_contracts(
        tmp_path,
        monkeypatch,
        frame=_frame(),
        ranges={"age": (0.0, 120.0)},
    )

    with pytest.raises(RuntimeError, match="exact sealed scope"):
        run_cohort_summary_from_env(
            declared_columns=("age", "sex", "exposure", "outcome"),
            typed_cohort_input="artifact:analysis_cohort",
            plausibility_expected_columns=("age", "exposure"),
            plausibility_audit={
                "age": {
                    "policy": "retain_and_flag",
                    "below_minimum_n": 0,
                    "above_maximum_n": 0,
                    "out_of_range_n": 0,
                    "compared_n": 4,
                }
            },
        )


def test_a_receipt_flagging_more_than_it_compared_is_refused(
    tmp_path,
    monkeypatch,
):
    _digest, _out_dir = _bind_run_with_contracts(
        tmp_path,
        monkeypatch,
        frame=_frame(),
        ranges={"age": (0.0, 120.0)},
    )

    with pytest.raises(RuntimeError, match="flags more values than it compared"):
        run_cohort_summary_from_env(
            declared_columns=("age", "sex", "exposure", "outcome"),
            typed_cohort_input="artifact:analysis_cohort",
            plausibility_expected_columns=("age",),
            plausibility_audit={
                "age": {
                    "policy": "retain_and_flag",
                    "below_minimum_n": 3,
                    "above_maximum_n": 0,
                    "out_of_range_n": 3,
                    "compared_n": 2,
                }
            },
        )


def test_a_step_with_no_scope_may_not_smuggle_in_a_receipt(
    tmp_path,
    monkeypatch,
):
    _digest, _out_dir = _bind_run_with_contracts(
        tmp_path,
        monkeypatch,
        frame=_frame(),
        ranges={"age": (0.0, 120.0)},
    )

    with pytest.raises(RuntimeError, match="no\\s+flag-only scope"):
        run_cohort_summary_from_env(
            declared_columns=("age", "sex", "exposure", "outcome"),
            typed_cohort_input="artifact:analysis_cohort",
            plausibility_expected_columns=(),
            plausibility_audit={
                "age": {
                    "policy": "retain_and_flag",
                    "below_minimum_n": 0,
                    "above_maximum_n": 0,
                    "out_of_range_n": 0,
                    "compared_n": 4,
                }
            },
        )
