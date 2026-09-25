"""The review counts the rows the primary model would fit, on the cohort execution reads."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.execution.primary_model_retention import (
    measure_primary_model_retention,
    resolve_review_cohort_path,
)
from easyicu.research_agent.schema import AnalysisPlan

from tests.research_agent.planning.scientific_review_fixtures import _context, _plan


def _cohort(tmp_path: Path, *, n: int = 600, seed: int = 3) -> tuple[Path, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "exposure": rng.integers(0, 2, n),
            "age": rng.normal(60.0, 12.0, n),
            "death": (rng.random(n) < 0.25).astype(int),
        }
    )
    # Age is unmeasured in most survivors: dropping it keeps the sicker rows.
    unmeasured = rng.random(n) < np.where(frame["death"].eq(1), 0.2, 0.7)
    frame.loc[unmeasured, "age"] = np.nan
    frame.loc[frame.index[:10], "exposure"] = None
    path = tmp_path / "cohort_analysis.parquet"
    frame.to_parquet(path, index=False)
    return path, frame


def _declared(plan: AnalysisPlan) -> AnalysisPlan:
    step = plan.steps[0]
    requirement = step.model_requirements[0].model_copy(
        update={"baseline_missing_handling": {"covariates": ["age"]}}
    )
    requirement = type(requirement).model_validate(requirement.model_dump())
    return plan.model_copy(
        update={
            "steps": [
                step.model_copy(update={"model_requirements": [requirement]}),
                *plan.steps[1:],
            ]
        }
    )


def test_the_probe_counts_what_the_primary_fit_keeps(tmp_path: Path) -> None:
    path, frame = _cohort(tmp_path)

    evidence = measure_primary_model_retention(
        context=_context(), plan=_plan(), cohort_path=path
    )

    assert evidence.status == "measured"
    assert evidence.cohort_source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    (item,) = evidence.requirements
    evaluable = frame["exposure"].notna() & frame["death"].notna()
    complete = evaluable & frame["age"].notna()
    assert (item.evaluable_n, item.model_n) == (int(evaluable.sum()), int(complete.sum()))
    assert item.retention == pytest.approx(complete.sum() / evaluable.sum(), abs=1e-4)
    assert item.policy == "drop_missing_baseline"
    (age,) = item.covariates
    assert age.handling == "drop_row"
    assert age.n_missing == int((evaluable & frame["age"].isna()).sum())
    # Missingness tracks the outcome, and the review can see it.
    assert age.outcome_rate_missing < age.outcome_rate_observed


def test_a_kept_unmeasured_state_is_counted_as_kept(tmp_path: Path) -> None:
    path, frame = _cohort(tmp_path)

    evidence = measure_primary_model_retention(
        context=_context(), plan=_declared(_plan()), cohort_path=path
    )

    (item,) = evidence.requirements
    assert item.policy == "explicit_missing_category"
    assert item.model_n == item.evaluable_n
    assert item.retention == 1.0
    assert item.complete_case_retention < 0.5
    assert item.covariates[0].handling == "unmeasured_category"


def test_candidate_planning_reads_no_rows(tmp_path: Path, monkeypatch) -> None:
    path, _frame = _cohort(tmp_path)
    context = _context()
    context = context.model_copy(
        update={
            "cohort": context.cohort.model_copy(
                update={"provenance": {"evidence_stage": "metadata_only_planning"}}
            )
        }
    )
    monkeypatch.setattr(pd, "read_parquet", lambda *_a, **_k: pytest.fail("read rows"))

    evidence = measure_primary_model_retention(context=context, plan=_plan(), cohort_path=path)

    assert (evidence.status, evidence.reason_code) == (
        "rows_unavailable",
        "metadata_only_planning",
    )


def test_a_cohort_the_owner_cannot_read_is_evidence_not_an_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.parquet"
    pd.DataFrame({"exposure": [], "age": [], "death": []}).to_parquet(empty)
    corrupt = tmp_path / "corrupt.parquet"
    corrupt.write_bytes(b"not a parquet file")
    narrow = tmp_path / "narrow.parquet"
    pd.DataFrame({"exposure": [0, 1], "death": [0, 1]}).to_parquet(narrow)

    def status(path):
        found = measure_primary_model_retention(context=_context(), plan=_plan(), cohort_path=path)
        return found.status, found.reason_code

    assert status(empty) == ("rows_unavailable", "zero_rows")
    assert status(corrupt)[0] == "probe_failed"
    assert status(narrow) == ("not_evaluable", "primary_model_columns_absent")
    assert status(None) == ("rows_unavailable", "cohort_not_materialized")


def test_only_a_primary_adjusted_model_on_the_analysis_cohort_is_measured(
    tmp_path: Path,
) -> None:
    path, _frame = _cohort(tmp_path)
    plan = _plan()
    without_model = plan.model_copy(
        update={"steps": [plan.steps[0].model_copy(update={"model_requirements": []}), *plan.steps[1:]]}
    )
    derived_input = plan.model_copy(
        update={
            "steps": [
                plan.steps[0].model_copy(
                    update={"inputs": [*plan.steps[0].inputs, "artifact:landmark_cohort"]}
                ),
                *plan.steps[1:],
            ]
        }
    )

    none = measure_primary_model_retention(context=_context(), plan=without_model, cohort_path=path)
    derived = measure_primary_model_retention(context=_context(), plan=derived_input, cohort_path=path)

    assert (none.status, none.reason_code) == ("not_applicable", "no_primary_adjusted_model")
    assert (derived.status, derived.reason_code) == (
        "rows_unavailable",
        "typed_cohort_product_not_materialized",
    )


def test_the_review_reads_the_cohort_execution_binds(tmp_path: Path) -> None:
    universe = tmp_path / "universe.parquet"
    applied = tmp_path / "cohort_analysis.parquet"

    def resolve(materialization):
        return resolve_review_cohort_path(
            run_dir=tmp_path, plan=_plan(), universe_path=universe, materialization=materialization
        )

    assert resolve({"status": "applied", "path": applied}) == applied
    assert resolve({"status": "no_definition"}) == universe
    assert resolve({"status": "error"}) is None
    # A reused plan without a closed materialization reads the universe.
    assert resolve(None) == universe


def test_the_plan_phase_hands_the_review_this_measurement() -> None:
    """The entry surface measures on the bound cohort; the gate only receives evidence."""

    import ast
    import inspect
    import textwrap

    from easyicu.research_agent import pipeline
    from easyicu.research_agent.orchestration import scientific_plan_review_gate

    def called(node, name):
        return isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == name

    def keyword(call, name):
        return next(item.value for item in call.keywords if item.arg == name)

    source = textwrap.dedent(
        inspect.getsource(pipeline.ResearchAgentPipeline._validate_and_persist_plan)
    )
    (gate_call,) = [
        node
        for node in ast.walk(ast.parse(source))
        if called(node, "prepare_scientific_plan_review_gate")
    ]
    measurement = keyword(gate_call, "model_retention")
    assert called(measurement, "measure_primary_model_retention")
    assert called(keyword(measurement, "cohort_path"), "resolve_review_cohort_path")
    assert "primary_model_retention import" not in inspect.getsource(
        scientific_plan_review_gate
    )
