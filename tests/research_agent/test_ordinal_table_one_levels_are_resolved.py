"""An ordinal row was allowed to declare its levels, and nobody resolved them.

``test_table_one_ordinal_row_may_declare_its_levels`` lifted the schema ban so
a SOFA component could declare the closed set it really has.  The level
RESOLVER was not told.  ``bind_table_one_execution_spec`` read

    if variable_spec.summary != "count_percent": continue

and an ordinal row summarises as ``median_iqr``, so its declaration was skipped
and the host's own ``__easyicu_level_N__`` placeholders travelled straight
through the execution spec into the host's own generated script.

WHAT THAT COST, m1 on 2026-08-03, timed from its audit log::

    04:41:49  Using planner-specified grouped Table 1 executor
    04:41:49  Running standard executor script
    04:41:53  Repairing failed script
    04:42:23  Repairing post-mutation concept violation
    04:42:56  Concept audit blocked mutated code

The script the host wrote for itself raised in the sandbox::

    TableOneContractError: sofa2_resp_max contains values outside the
    Planner-declared closed levels

-- five declared tokens against a column holding 0/1/2/3/4.  The Coder was then
asked to repair the host's script, changed the spec (the only change that could
make it run), and ``table_one_spec_not_planner_owned`` refused that too: the
spec-ownership gate requires the exact host-owned declaration.  Two host layers
holding one contradiction.  Step 02 died and took ``07_robustness_replay_figure``
with it as ``blocked_dependency_evidence``.

MEASURED over every recorded plan, deduplicated per (run, step): 348 Table 1
specs.  18 (5.2%) carry a level set the filter skipped, 56 variables in all,
and every single one is ordinal x median_iqr.  On the other side, 594 of 594
``count_percent`` variables declare levels -- so widening the clause changes
behaviour for exactly the skipped sets and for nothing else.

The clause is WIDENED, not replaced.  ``summary == "count_percent"`` with an
empty declaration must still reach the resolver, which raises rather than
inventing a domain; dropping that half would turn a refusal into a silent
guess.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from easyicu.research_agent.authority.table_one_binding import (  # noqa: E402
    bind_table_one_execution_spec,
    table_one_execution_spec,
)
from easyicu.research_agent.methods.table_one import (  # noqa: E402
    TableOneContractError,
    build_grouped_table_one,
)
from easyicu.research_agent.research_context.prompt_variables import (  # noqa: E402
    opaque_level_tokens,
)
from easyicu.research_agent.schema import (  # noqa: E402
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
)

#: The real observed domain of every SOFA component on the 94,458-stay cohort.
_STAGES = [0.0, 1.0, 2.0, 3.0, 4.0]
_STAGE_TOKENS = list(opaque_level_tokens(len(_STAGES)))
_BINARY_TOKENS = list(opaque_level_tokens(2))


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Does organ dysfunction grade in-hospital mortality?",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[
            ConceptDescriptor(
                name="death",
                dtype="int64",
                observed_domain={
                    "n_unique": 2,
                    "is_constant": False,
                    "is_binary": True,
                    "min": 0.0,
                    "max": 1.0,
                    "levels": [0, 1],
                },
            ),
            ConceptDescriptor(
                name="sex",
                dtype="str",
                observed_domain={
                    "n_unique": 2,
                    "is_constant": False,
                    "is_binary": False,
                    "levels": ["Female", "Male"],
                },
            ),
            ConceptDescriptor(
                name="sofa2_resp_max",
                dtype="float32",
                observed_domain={
                    "n_unique": 5,
                    "is_constant": False,
                    "is_binary": False,
                    "min": 0.0,
                    "max": 4.0,
                    "levels": list(_STAGES),
                },
            ),
        ],
    )


def _step(*, variables) -> AnalysisStep:
    """m1's step 02, reduced to the rows that carry the defect."""

    return AnalysisStep.model_validate(
        {
            "step_id": "02_table_one_by_mortality",
            "planned_analysis_role": "auxiliary",
            "intent": "Describe the cohort by the outcome.",
            # The schema requires every Table 1 variable to be an explicit
            # step input, as m1's real step declared them.
            "inputs": [
                "artifact:analysis_cohort",
                "death",
                *sorted({row["name"] for row in variables}),
            ],
            "expected_outputs": ["table:table_one"],
            "method": "descriptive_summary",
            "table_one_spec": {
                "group_by": "death",
                "group_levels": list(_BINARY_TOKENS),
                "variables": variables,
            },
            "input_consumption_contracts": [
                {
                    "schema_version": "easyicu.artifact_consumption/1",
                    "input_key": "artifact:analysis_cohort",
                    "mode": "all_rows",
                    "role_column": None,
                    "expected_roles": [],
                }
            ],
        }
    )


_ORDINAL_ROW = {
    "name": "sofa2_resp_max",
    "variable_kind": "ordinal",
    "summary": "median_iqr",
    "test": "mann_whitney_or_kruskal",
    "levels": list(_STAGE_TOKENS),
}
_CATEGORICAL_ROW = {
    "name": "sex",
    "variable_kind": "categorical",
    "summary": "count_percent",
    "test": "chi_square_with_fisher_exact_for_sparse_2x2",
    "levels": list(_BINARY_TOKENS),
}


def _frame() -> pd.DataFrame:
    stages = [_STAGES[index % len(_STAGES)] for index in range(40)]
    return pd.DataFrame(
        {
            "death": [index % 2 for index in range(40)],
            "sex": ["Female" if index % 2 else "Male" for index in range(40)],
            "sofa2_resp_max": pd.Series(stages, dtype="float32"),
        }
    )


def test_an_ordinal_declaration_is_resolved_to_the_observed_stages() -> None:
    """The property that was false for all 56 recorded ordinal declarations."""

    step = _step(variables=[_CATEGORICAL_ROW, _ORDINAL_ROW])
    bind_table_one_execution_spec(step, _context())
    spec = table_one_execution_spec(step)

    ordinal = next(v for v in spec.variables if v.name == "sofa2_resp_max")
    assert ordinal.levels == _STAGES
    # The public plan is untouched: placeholders are what leaves the host.
    planner = next(
        v for v in step.table_one_spec.variables if v.name == "sofa2_resp_max"
    )
    assert planner.levels == _STAGE_TOKENS


def test_the_categorical_row_still_resolves_beside_it() -> None:
    """Widening the clause must not disturb the path that already worked."""

    step = _step(variables=[_CATEGORICAL_ROW, _ORDINAL_ROW])
    bind_table_one_execution_spec(step, _context())
    spec = table_one_execution_spec(step)

    assert spec.group_levels == [0, 1]
    assert next(v for v in spec.variables if v.name == "sex").levels == [
        "Female",
        "Male",
    ]


def test_nothing_the_old_clause_admitted_can_be_lost() -> None:
    """Why ``levels`` alone is safe to key on, checked rather than assumed.

    The replaced clause admitted every ``count_percent`` variable, including
    one declaring nothing.  Keying on ``levels`` would skip such a row -- so
    the first draft kept both halves.  It is unreachable: the schema refuses
    ``count_percent`` with fewer than two declared levels, which makes
    "summarised as counts" a strict subset of "declared levels" and the second
    half a clause that can never fire.

    Locked here because it is the whole safety argument for the simpler
    condition, and it lives in a different module from the one it protects.
    """

    with pytest.raises(ValueError, match="at least two"):
        _step(variables=[{**_CATEGORICAL_ROW, "levels": []}])
    with pytest.raises(ValueError, match="at least two"):
        _step(variables=[{**_CATEGORICAL_ROW, "levels": list(_BINARY_TOKENS[:1])}])


def test_the_resolved_spec_is_the_one_the_sdk_can_execute() -> None:
    """End to end: the placeholder spec raises, the resolved spec builds.

    This is the recorded sandbox failure and its repair, on a frame holding the
    same values the real cohort did.
    """

    frame = _frame()
    step = _step(variables=[_CATEGORICAL_ROW, _ORDINAL_ROW])

    unresolved = step.table_one_spec.model_dump(mode="python")
    unresolved["group_levels"] = [0, 1]
    unresolved["variables"][0]["levels"] = ["Female", "Male"]
    with pytest.raises(TableOneContractError, match="sofa2_resp_max"):
        build_grouped_table_one(frame, unresolved)

    bind_table_one_execution_spec(step, _context())
    table = build_grouped_table_one(
        frame, table_one_execution_spec(step).model_dump(mode="python")
    )
    rows = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
    assert not rows.empty
    assert set(rows["variable"]) == {"sex", "sofa2_resp_max"}
