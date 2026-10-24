from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

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


@pytest.mark.parametrize("grouped", [True, False])
def test_saved_demo_summary_preserves_decimal_values_on_csv_readback(tmp_path, monkeypatch, grouped):
    from easyicu.research_agent.contracts.cohort_summary import descriptive_summary_result_receipt_valid
    from easyicu.research_agent.execution.runners import cohort_summary_executor as owner

    fixture = Path(__file__).parents[2] / "fixtures/demo_round_stops/dd3_descriptive_summary.json"
    saved = json.loads(fixture.read_text())
    step = AnalysisStep.model_validate(saved["step"])
    context = saved["context"]
    grouping, outcome = step.inputs[:2]
    levels = next(v["observed_domain"]["levels"] for v in context["variables"] if v["name"] == grouping)
    # Synthetic fractional days exercise CSV parsing precision; no saved rows.
    frame = pd.DataFrame({
        grouping: [levels[0], levels[0], levels[1], levels[1], None],
        outcome: [1 / 1440, 113 / 1440, 1 / 1440, None, 2 / 1440],
    })
    if not grouped:
        step = step.model_copy(update={"inputs": [outcome, "artifact:analysis_cohort"]})
        frame = frame.drop(columns=grouping)
    source = tmp_path / "synthetic.parquet"
    frame.to_parquet(source, index=False)
    (tmp_path / "research_context.json").write_text(json.dumps(context))
    out = tmp_path / "out"
    monkeypatch.setenv("EASYICU_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("EASYICU_RESEARCH_CONTEXT", str(tmp_path / "research_context.json"))
    monkeypatch.setenv("STEP_OUT_DIR", str(out))
    monkeypatch.setattr(owner, "load_cohort_summary_frame", lambda **kwargs: (frame, source))
    selection = select_standard_executor(step, plan=AnalysisPlan(
        research_question="Describe a synthetic duration by category.",
        analysis_type=saved["analysis_type"], steps=[step],
    ))
    assert selection.analysis_kind == saved["stop"]["deterministic_standard_analysis"]
    exec(compile(selection.code, "<saved-demo-summary>", "exec"), {})
    result = json.loads((out / "step_summary.json").read_text())
    assert descriptive_summary_result_receipt_valid(result)
    assert result["grouping_variable"] == (grouping if grouped else None)
    summary = pd.read_csv(out / "cohort_summary.csv", float_precision="round_trip")
    distribution = pd.read_csv(out / "distribution_summary.csv", float_precision="round_trip")
    assert set(distribution["value"]) == set(frame[outcome].dropna())
    kwargs = {"grouping_variable": grouping, "grouping_levels": levels} if grouped else {}
    owner.validate_primary_descriptive_tables(frame, summary, distribution, **kwargs)
    # Precision preservation must not weaken exact support-value/count checks.
    for field in ("value", "count"):
        corrupt = distribution.copy()
        original = corrupt.at[0, field]
        corrupt.at[0, field] = math.nextafter(original, math.inf) if field == "value" else original + 1
        with pytest.raises(ValueError, match=saved["stop"]["exception_message"]):
            owner.validate_primary_descriptive_tables(frame, summary, corrupt, **kwargs)


def test_primary_summary_emits_exact_empirical_counts_and_validated_statistics(tmp_path, monkeypatch):
    import easyicu.research_agent.execution.runners.cohort_summary_executor as owner
    from easyicu.research_agent.contracts.cohort_summary import descriptive_summary_result_receipt_valid

    run_dir = tmp_path / "synthetic"
    run_dir.mkdir()
    context = {"variables": [
        {"name": "temperature", "role": "vital", "dtype": "float64", "unit": "C"},
        {"name": "stay_id", "role": "id", "dtype": "int64"},
    ]}
    (run_dir / "research_context.json").write_text(json.dumps(context))
    frame = pd.DataFrame({"temperature": [35.0, 37.0, 37.0, None], "stay_id": [1, 2, 3, 4]})
    source = run_dir / "synthetic.parquet"
    frame.to_parquet(source, index=False)
    out = run_dir / "outputs"
    monkeypatch.setenv("EASYICU_RUN_DIR", str(run_dir))
    monkeypatch.setenv("STEP_OUT_DIR", str(out))
    result = owner.run_cohort_summary_from_env(
        declared_columns=["temperature", "stay_id"], typed_cohort_input="artifact:analysis_cohort",
        frame=frame, cohort_path=source, primary_summary=True,
    )
    assert descriptive_summary_result_receipt_valid(result)
    assert result["declared_columns"] == ["temperature"]
    distribution = pd.read_csv(out / "distribution_summary.csv")
    assert distribution["value"].tolist() == [35.0, 37.0]
    assert distribution["count"].tolist() == [1, 2]
    assert distribution["cumulative_proportion"].tolist() == pytest.approx([1 / 3, 1])
    summary = pd.read_csv(out / "cohort_summary.csv")
    values = summary.loc[summary["variable"] == "temperature"].set_index("statistic")["value"]
    assert values["mean"] == pytest.approx(109 / 3)
    assert values["median"] == 37
    assert values["missing_n"] == 1
    assert "stay_id" not in summary["variable"].tolist()
    assert result["paper_authorization_allowed"] is False
    owner.validate_primary_descriptive_tables(frame[["temperature"]], summary, distribution)
    for column in ("count", "cumulative_proportion", "denominator", "value"):
        corrupt = distribution.copy()
        corrupt.loc[0, column] += 1
        with pytest.raises(ValueError, match="descriptive"):
            owner.validate_primary_descriptive_tables(frame[["temperature"]], summary, corrupt)
    corrupt = summary.copy()
    corrupt.loc[corrupt["statistic"] == "median", "value"] = 99
    with pytest.raises(ValueError, match="descriptive"):
        owner.validate_primary_descriptive_tables(frame[["temperature"]], corrupt, distribution)
    assert not descriptive_summary_result_receipt_valid({**result, "descriptive_summary_validation": {}})


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


@pytest.mark.parametrize("role", ["id", "index"])
def test_summary_does_not_publish_numeric_identifier_statistics(tmp_path, monkeypatch, role):
    cohort_path, out_dir = _bind_run(tmp_path, monkeypatch)
    context_path = cohort_path.parent / "research_context.json"
    context = json.loads(context_path.read_text())
    context["variables"][0]["role"] = role
    context_path.write_text(json.dumps(context))
    with pytest.raises(RuntimeError, match="Identifiers cannot be described"):
        run_cohort_summary_from_env(declared_columns=["age"], typed_cohort_input="artifact:analysis_cohort")
    assert not (out_dir / "cohort_summary.csv").exists()
