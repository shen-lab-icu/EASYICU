"""The replay knew which contract was primary and published nothing.

``deterministic_robustness`` resolves its primary model contract with
``_primary_contract_from_summary``: by ``primary_model_id`` when the parent
summary wrote one, otherwise by ``analysis_role == "primary"``.  Its own
fallback means it works fine when the field is absent.

It then published ``primary_model_id`` by copying the parent's field verbatim.
The parent is not obliged to write it, and measured over every recorded run it
never has: 10 of 10 robustness summaries carrying model contracts left this
empty, while each named exactly one primary contract.

The consequence is downstream and has no fallback.  ``figure_source_data``
builds the set of traceable contracts by filtering the parent's
``model_contracts`` on exact equality with ``primary_model_id``.  An empty
value matches nothing, so the ``primary`` row of the robustness figure's
source data is reported as ``ambiguous_model_contract_trace`` with 0 matches --
and on the 2026-08-01 E1 run (canary25) that was one of the findings that kept
the last step of an otherwise complete task from passing.

Same shape as the other fixes in this series: one value, written under one
spelling by a producer that already knows it, read by someone stricter.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from easyicu.research_agent.execution.runners import deterministic_robustness


def _published_primary_model_id(
    *, parent_field: object, contract_model_id: object
) -> object:
    """Evaluate the published expression against the two inputs it reads.

    Extracted from the module source so the test cannot drift from the code:
    the summary is assembled deep inside a runtime function that needs a real
    cohort, so re-running it here is not possible, but the expression itself
    is exactly what that function evaluates.
    """

    tree = ast.parse(inspect.getsource(deterministic_robustness))
    expression = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "primary_model_id":
                expression = ast.unparse(value)
    assert expression, "the summary no longer publishes primary_model_id"
    source_summary = {"primary_model_id": parent_field}
    structured_source = {"primary_contract": {"model_id": contract_model_id}}
    return eval(  # noqa: S307 - the expression comes from our own module
        expression,
        {"str": str},
        {"source_summary": source_summary, "structured_source": structured_source},
    )


def test_the_selected_contract_is_named_when_the_parent_wrote_nothing() -> None:
    """The property that was false in 10 of 10 recorded runs."""

    assert (
        _published_primary_model_id(
            parent_field=None,
            contract_model_id="primary_sepsis3_mortality_association",
        )
        == "primary_sepsis3_mortality_association"
    )


def test_a_parent_that_does_name_it_still_wins() -> None:
    """The parent remains authoritative when it speaks.

    Preferring the contract would silently override an upstream that
    deliberately designated a different primary model.
    """

    assert (
        _published_primary_model_id(
            parent_field="parent_choice", contract_model_id="contract_choice"
        )
        == "parent_choice"
    )


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_a_blank_parent_field_falls_through(blank: object) -> None:
    assert (
        _published_primary_model_id(parent_field=blank, contract_model_id="chosen")
        == "chosen"
    )


def test_nothing_to_name_stays_none_rather_than_an_empty_string() -> None:
    """Absent must stay absent.

    An empty string is a value that matches no contract while looking like an
    answer; ``None`` says the replay could not name one.
    """

    assert (
        _published_primary_model_id(parent_field=None, contract_model_id=None) is None
    )
    assert _published_primary_model_id(parent_field="", contract_model_id="") is None


# --- the recorded corpus ------------------------------------------------------

_CORPUS = Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


@pytest.mark.skipif(not _CORPUS.exists(), reason="recorded runs are not on this machine")
def test_every_recorded_replay_could_have_named_its_primary_contract() -> None:
    """Real bytes: the fallback must actually resolve on recorded runs.

    If a recorded summary carries model contracts but none of them can be
    picked as primary, the fix would publish ``None`` again and the figure
    would still fail -- so this measures the fallback's reach, not the code.
    """

    resolvable = unresolvable = 0
    for path in sorted(
        _CORPUS.glob("batch_*/*/aware/run_*/steps/*/outputs/step_summary.json")
    ):
        try:
            summary = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(summary, dict):
            continue
        if summary.get("analysis_family") != "robustness_sensitivity":
            continue
        contracts = summary.get("model_contracts")
        if not isinstance(contracts, list) or not contracts:
            continue
        chosen = deterministic_robustness._primary_contract_from_summary(summary)
        if isinstance(chosen, dict) and str(chosen.get("model_id") or "").strip():
            resolvable += 1
        else:
            unresolvable += 1

    if not resolvable and not unresolvable:
        pytest.skip("no recorded robustness summary carries model contracts")
    assert unresolvable == 0, (
        f"{unresolvable} of {resolvable + unresolvable} recorded replays carry "
        "model contracts but cannot name a primary one"
    )
