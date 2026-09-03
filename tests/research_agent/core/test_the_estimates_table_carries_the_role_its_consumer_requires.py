"""The estimates table must carry the role column its consumer contracts on.

MEASURED (e2 lactate, 9 of 11 steps): step
``06_lactate_mortality_association_figure`` was blocked
``blocked_dependency_evidence`` with the exact reason

    {"input": "table:adjusted_association_estimates",
     "reason": "artifact_consumption_contract_invalid",
     "message": "role column 'analysis_role' is absent from the verified schema"}

Step 05 produced that table through the host's own deterministic
adjusted-association executor and reported ``ok``: unique producer, single
registration, file present, digest matching. The table simply did not have the
column, and the value sat in the requirement object the executor was already
holding.

The model contract this executor writes into ``step_summary.json`` has
carried ``analysis_role`` all along (``MODEL_CONTRACT_FIELDS``); the TABLE
did not, and the consumption contract is on the table.

``analysis_set`` was present, which is why this looked covered: it is a
different field -- WHICH POPULATION, not which role -- so a reader scanning for
"analysis_*" would see one and assume the other.

This is the second recurrence of the same shape on the same table. The first
was ``model_id`` (see
test_estimates_table_carries_the_identity_its_contract_publishes.py).
"""

from __future__ import annotations

import ast
import inspect

from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    ADJUSTED_ASSOCIATION_COEFFICIENT_COLUMNS,
    ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS,
    MODEL_CONTRACT_FIELDS,
    run_adjusted_association_from_env,
)


def test_the_estimates_table_declares_the_role_column() -> None:
    assert "analysis_role" in ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS


def test_the_role_column_is_not_confused_with_the_population_column() -> None:
    """Both fields exist and mean different things; one cannot stand in."""

    assert "analysis_set" in ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS
    assert "analysis_role" != "analysis_set"


def test_the_model_contract_and_the_table_agree_on_the_role_column() -> None:
    """The divergence that hid this: the contract had it, the table did not.

    An earlier draft of this test asserted the sibling COEFFICIENT table
    carried `analysis_role`. It does not -- that name sits in
    MODEL_CONTRACT_FIELDS, which this executor writes into the step summary.
    The test caught the wrong claim before the commit did.
    """

    assert "analysis_role" in MODEL_CONTRACT_FIELDS
    assert "analysis_role" in ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS
    assert "analysis_role" not in ADJUSTED_ASSOCIATION_COEFFICIENT_COLUMNS


def _shared_row_keys() -> set[str]:
    """The keys of the per-row `shared` dict, located structurally.

    A substring search for `"analysis_role":` over the function source is not
    this check: the same literal appears in the model-contract dict a hundred
    lines below, so deleting it from the ROW dict left the search satisfied and
    the mutation survived. The column would then exist in the frame and be
    silently all-NaN -- present enough for a schema check, empty for every
    role-value verification downstream.
    """

    tree = ast.parse(inspect.getsource(run_adjusted_association_from_env).lstrip())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "shared" not in targets:
            continue
        return {
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    raise AssertionError("the per-row `shared` dict was not found")


def test_every_declared_estimates_column_is_written_by_the_producer() -> None:
    """A column in the tuple that no row populates would be an empty column."""

    keys = _shared_row_keys()
    for column in ("analysis_role", "analysis_set", "model_id", "requirement_id"):
        assert column in keys, f"{column} is declared but never written to a row"


def test_the_role_value_comes_from_the_requirement_not_a_literal() -> None:
    source = inspect.getsource(run_adjusted_association_from_env)
    assert '"analysis_role": analysis_role,' in source
    # Not hard-coded to one role: the table has to be able to say secondary.
    assert '"analysis_role": "primary"' not in source
