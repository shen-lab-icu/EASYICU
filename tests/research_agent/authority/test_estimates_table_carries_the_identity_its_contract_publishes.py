"""The model's identity was published under two names, and the reader knew one.

``run_adjusted_association_from_env`` writes three things about one fitted
model: a model contract, a term-level coefficient table, and the one-row
estimates table the manuscript quotes.  The contract publishes the model's
identity as ``model_id``.  The coefficient table's first column is ``model_id``.
The estimates table spelled the same value ``requirement_id`` and wrote no
``model_id`` at all.

That matters downstream because of what a generic table name cannot prove.
``FigureSourceDataValidator._contract_scoped_effect_product`` exists so that
``table:adjusted_association_estimates`` -- a name that says nothing about
whether its rows are primary, secondary or sensitivity estimates -- can inherit
an estimand tier from the parent's machine-readable ``model_contracts``, but
only after the figure's source data has value-matched the table.  It performs
that lookup by the ``model_id`` column.  With no such column it returns the
product untouched, the product carries no tier, and
``figure:primary_adjusted_association`` -- whose tier IS ``primary`` -- fails
its ``effect`` obligation with ``missing_figure_family_source``.

Measured over every recorded run: 24 estimates tables carry ``requirement_id``
and no ``model_id``, against 13 written by the older coder path that carry
``model_id`` and pass.  In none of the 24 does ``requirement_id`` differ from
the contract's ``model_id`` -- the value was always right, only the name was
wrong.  On the 2026-08-01 E1 run (canary27) this was the sole finding left on
``08_primary_association_figure``.

Same shape as the four fixes before it, and fixed the same way: at the
producer, by writing the name the contract already publishes.  The validator is
deliberately fail-closed here and is not touched.
"""

from __future__ import annotations

import ast
import csv
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.audits.validators import FigureSourceDataValidator
from easyicu.research_agent.execution.runners import adjusted_association_executor
from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    ADJUSTED_ASSOCIATION_COEFFICIENT_COLUMNS,
    ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS,
)


def test_the_estimates_header_names_the_model() -> None:
    """The property that was false: the table could not say which model it is."""

    assert "model_id" in ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS


def test_both_tables_of_one_model_use_one_name_for_it() -> None:
    """The sibling coefficient table already keyed on ``model_id``.

    Two tables describing the same fit must not identify it differently, or a
    reader joining them has to know which of two spellings each one chose.
    """

    assert "model_id" in ADJUSTED_ASSOCIATION_COEFFICIENT_COLUMNS
    assert "model_id" in ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS


def test_the_requirement_the_plan_asked_for_is_still_reported() -> None:
    """Adding an identity must not cost the plan linkage.

    ``requirement_id`` answers "which planned requirement does this row
    satisfy"; ``model_id`` answers "which fitted model is this".  They hold the
    same string in this owner because its roster is keyed by requirement, but
    they are different questions and the model contract publishes both.
    """

    assert "requirement_id" in ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS


def _shared_row_dict() -> ast.Dict:
    """The literal every emitted estimates row is built from.

    Read from source rather than by running the executor, which needs a fitted
    cohort inside the sandbox.  Both the single-contrast row and
    ``_contrast_rows`` spread this same mapping, so a key present here is
    present on every row the step writes.
    """

    tree = ast.parse(
        inspect.getsource(
            adjusted_association_executor.run_adjusted_association_from_env
        )
    )
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "shared"
                for target in node.targets
            )
            and isinstance(node.value, ast.Dict)
        ):
            return node.value
    raise AssertionError("the shared row mapping is gone")


def test_every_row_carries_the_identity_and_it_is_the_same_value() -> None:
    """The wiring, not just the header.

    A header entry with nothing writing it produces a column of blanks, which
    fails the contract lookup exactly as the missing column did.
    """

    shared = _shared_row_dict()
    written = {
        key.value: ast.unparse(value)
        for key, value in zip(shared.keys, shared.values)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert "model_id" in written, "no row is given a model identity"
    assert written["model_id"] == written.get("requirement_id"), (
        "the model identity is not the value this step's own contract publishes: "
        f"{written['model_id']!r}"
    )


def test_the_contract_publishes_that_same_identity() -> None:
    """What the reader joins against.

    The lookup matches the table's ``model_id`` values against the keys of the
    parent summary's ``model_contracts``; if the contract were keyed on
    anything else the column would be unjoinable.
    """

    source = inspect.getsource(
        adjusted_association_executor.run_adjusted_association_from_env
    )
    tree = ast.parse(source)
    contract_expressions = [
        ast.unparse(value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "model_contract"
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        )
        and isinstance(node.value, ast.Dict)
        for key, value in zip(node.value.keys, node.value.values)
        if isinstance(key, ast.Constant) and key.value == "model_id"
    ]
    assert contract_expressions, "the model contract no longer names the model"
    written = {
        key.value: ast.unparse(value)
        for key, value in zip(_shared_row_dict().keys, _shared_row_dict().values)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert contract_expressions[0] == written["model_id"], (
        "the table and the contract name the model from different expressions: "
        f"{contract_expressions[0]!r} vs {written['model_id']!r}"
    )


# --- the reader this unblocks -------------------------------------------------


_PARENT_STEP_ID = "06_primary_adjusted_association"
_PRODUCT = "table:adjusted_association_estimates"


def _estimates_frame(*, model_id: str | None) -> pd.DataFrame:
    row = {
        "fit_status": "fitted",
        "estimate": 1.57,
        "ci_low": 1.02,
        "ci_high": 2.39,
        "effect_scale": "odds_ratio",
        "exposure": "exposure_source_variable",
        "requirement_id": "declared_primary_model",
        "outcome": "outcome_variable",
        "covariates": "age;sex",
        "estimator_kind": "logistic",
        "analysis_set": "source_aware",
        "n": 1000,
        "n_events": 102,
        "standard_error": 0.21,
        "notes": "",
        "exposure_level": "",
        "reference_level": "",
        "contrast": "",
        "is_primary_contrast": True,
    }
    if model_id is not None:
        row["model_id"] = model_id
    return pd.DataFrame([row])


def _records(*, analysis_role: str = "primary") -> list[dict]:
    return [
        {
            "step_id": _PARENT_STEP_ID,
            "status": "ok",
            "step_summary": {
                "model_contracts": [
                    {
                        "model_id": "declared_primary_model",
                        "analysis_role": analysis_role,
                        "fit_status": "fitted",
                        "exposure_source": "exposure_source_variable",
                    }
                ]
            },
        }
    ]


def _scoped(frame: pd.DataFrame, records: list[dict]) -> str:
    return FigureSourceDataValidator._contract_scoped_effect_product(
        product=_PRODUCT,
        source_frame=frame,
        upstream_frame=frame,
        upstream_step_id=_PARENT_STEP_ID,
        completed_step_records=records,
    )


def test_the_tier_is_inherited_once_the_table_names_its_model() -> None:
    """The unlock, stated as the reader sees it."""

    assert _scoped(_estimates_frame(model_id="declared_primary_model"), _records()) == (
        "table:primary_adjusted_association_estimates"
    )


def test_without_the_identity_the_reader_still_learns_nothing() -> None:
    """Pin the defect itself, so a revert cannot pass silently."""

    assert _scoped(_estimates_frame(model_id=None), _records()) == _PRODUCT


def test_an_identity_no_contract_claims_inherits_nothing() -> None:
    """The gate stays closed.

    The point of the lookup is that the tier comes from a validated contract.
    A row naming a model the parent never fitted must not acquire ``primary``
    just because it filled the column in.
    """

    assert _scoped(_estimates_frame(model_id="a_model_nobody_fitted"), _records()) == (
        _PRODUCT
    )


def test_a_secondary_contract_does_not_become_primary() -> None:
    """The inherited tier is the contract's, not the figure's wish."""

    scoped = _scoped(
        _estimates_frame(model_id="declared_primary_model"),
        _records(analysis_role="secondary"),
    )
    assert scoped == "table:secondary_adjusted_association_estimates"


# --- the recorded corpus ------------------------------------------------------

_CORPUS = Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


def _recorded_estimates_tables() -> list[tuple[Path, dict]]:
    found: list[tuple[Path, dict]] = []
    for summary_path in sorted(
        _CORPUS.glob("batch_*/*/aware/run_*/steps/*/outputs/step_summary.json")
    ):
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(summary, dict):
            continue
        contracts = summary.get("model_contracts")
        if not isinstance(contracts, list) or not contracts:
            continue
        table_path = summary_path.parent / "adjusted_association_estimates.csv"
        if table_path.is_file():
            found.append((table_path, summary))
    return found


@pytest.mark.skipif(
    not _CORPUS.exists(), reason="recorded runs are not on this machine"
)
def test_the_recorded_tables_that_lack_it_hold_the_value_under_the_other_name() -> None:
    """Real bytes: the fix writes a value the producer already had.

    If any recorded table's ``requirement_id`` disagreed with its own
    contract's ``model_id``, copying one into the other would be inventing a
    join key rather than spelling an existing one -- and this fix would be
    wrong.  Scoped to this owner's own file; sibling producers are not claimed.
    """

    disagreements = []
    covered = 0
    for table_path, summary in _recorded_estimates_tables():
        try:
            rows = list(csv.DictReader(table_path.open()))
        except OSError:
            continue
        if not rows or "model_id" in rows[0]:
            continue
        contract_ids = {
            str(item.get("model_id") or "").strip()
            for item in summary["model_contracts"]
            if isinstance(item, dict)
        }
        covered += 1
        for row in rows:
            named = (row.get("requirement_id") or "").strip()
            if named not in contract_ids:
                disagreements.append((table_path.parent.parent.name, named))

    if not covered:
        pytest.skip("no recorded estimates table predates the identity column")
    assert not disagreements, (
        "recorded rows name a requirement their own step never fitted, so the "
        f"identity cannot simply be spelled correctly: {disagreements[:5]}"
    )


@pytest.mark.skipif(
    not _CORPUS.exists(), reason="recorded runs are not on this machine"
)
def test_the_fix_reaches_the_recorded_failures() -> None:
    """Real bytes: adding the column flips the reader on the runs that failed.

    Reproduces the reader against each recorded table as it is, and again with
    the identity spelled as this fix spells it.  Every recorded table without
    the column must go from "no tier" to a tier; otherwise something else is
    also wrong and closing this one would not be enough.
    """

    unflipped = []
    flipped = 0
    for table_path, summary in _recorded_estimates_tables():
        try:
            frame = pd.read_csv(table_path)
        except (OSError, ValueError):
            continue
        if "model_id" in frame.columns or "requirement_id" not in frame.columns:
            continue
        step_id = table_path.parent.parent.name
        records = [{"step_id": step_id, "status": "ok", "step_summary": summary}]
        before = FigureSourceDataValidator._contract_scoped_effect_product(
            product=_PRODUCT,
            source_frame=frame,
            upstream_frame=frame,
            upstream_step_id=step_id,
            completed_step_records=records,
        )
        repaired = frame.assign(model_id=frame["requirement_id"].astype(str))
        after = FigureSourceDataValidator._contract_scoped_effect_product(
            product=_PRODUCT,
            source_frame=repaired,
            upstream_frame=repaired,
            upstream_step_id=step_id,
            completed_step_records=records,
        )
        if before == _PRODUCT and after != _PRODUCT:
            flipped += 1
        else:
            unflipped.append((step_id, before, after))

    if not flipped and not unflipped:
        pytest.skip("no recorded estimates table predates the identity column")
    assert not unflipped, (
        "recorded tables the identity column does not unblock, so the figure "
        f"would still fail its effect obligation: {unflipped[:5]}"
    )
