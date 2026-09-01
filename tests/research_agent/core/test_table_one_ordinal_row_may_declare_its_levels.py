"""An ordinal Table 1 row may declare the closed set it actually has.

MEASURED 2026-07-30 in batch ``..._88d3983_canonical9_full02``.  The
ventilation-survival task never produced a plan at all::

    StructuredResponseFailure[role=planner, n_attempts=5]:
      steps.2.table_one_spec.variables.2
        Value error, numeric ordinal Table 1 summaries must not declare
        category levels

Five consecutive planning attempts, one rejected declaration, and the whole
task dead before a single step ran.  It completes the batch's plan-stage map:
two tasks died on a prompt over its byte budget, two on a stability spec whose
fields had one legal value each, and this one here.

The refusal was arbitrary in the one place it should not have been.  Every
ordinal score in intensive care has a real closed set -- a SOFA component 0-4,
a KDIGO stage 0-3, a GCS -- and on the real 94,458-stay cohort all five SOFA
components hold exactly {0, 1, 2, 3, 4}.  The Planner directive asks each
variable for its "closed levels" without qualifying which kinds may have them,
and the worked example beside it shows only a continuous row with an empty
list, so nothing told the Planner the field it was filling in was forbidden.
The variable catalog now supplies those observed levels itself (``7a70a52``),
which made the plan schema refuse the very list the context had just offered.

Permission alone would not be enough.  A declared set that nothing checks reads
as a guarantee and is not one, so the executor enforces it the way it enforces
``group_levels``: a value outside the declared set stops the step instead of
being summarised as though the study had anticipated it.  Continuous rows keep
the ban, because a measurement has no closed set to declare.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from easyicu.research_agent.methods.table_one import (  # noqa: E402
    TableOneContractError,
    build_grouped_table_one,
)
from easyicu.research_agent.schema import (  # noqa: E402
    TableOneSpec,
    TableOneVariableSpec,
)

# The shape recorded in the dead plan: an organ score summarised as median
# (IQR) -- never as a mean, which is why it is ordinal and not continuous --
# carrying the five stages it can take.
_ORDINAL_ROW = {
    "name": "organ_score",
    "variable_kind": "ordinal",
    "summary": "median_iqr",
    "test": "mann_whitney_or_kruskal",
    "levels": [0, 1, 2, 3, 4],
}


def _spec(row: dict) -> dict:
    return {
        "group_by": "arm",
        "group_levels": [0, 1],
        "variables": [row],
    }


def _frame(scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "arm": [index % 2 for index in range(len(scores))],
            "organ_score": pd.Series(scores, dtype="float64"),
        }
    )


# ---------------------------------------------------------------------------
# The declaration that killed the task
# ---------------------------------------------------------------------------


def test_an_ordinal_row_may_declare_its_closed_levels():
    variable = TableOneVariableSpec.model_validate(_ORDINAL_ROW)

    assert variable.levels == [0, 1, 2, 3, 4]
    assert variable.variable_kind == "ordinal"
    assert variable.summary == "median_iqr"


def test_an_ordinal_row_may_still_omit_them():
    """Declaring the set stays optional; only declaring it wrongly is refused."""

    variable = TableOneVariableSpec.model_validate({**_ORDINAL_ROW, "levels": []})

    assert variable.levels == []


def test_a_continuous_row_still_may_not():
    """A measurement has no closed set, so the ban there is not arbitrary."""

    with pytest.raises(ValueError, match="must not declare categorical levels"):
        TableOneVariableSpec.model_validate(
            {**_ORDINAL_ROW, "variable_kind": "continuous"}
        )


def test_one_declared_level_is_not_a_set():
    with pytest.raises(ValueError, match="one value is not a set"):
        TableOneVariableSpec.model_validate({**_ORDINAL_ROW, "levels": [0]})


def test_a_count_percent_row_keeps_its_own_rule():
    """The categorical branch is untouched: it still requires two levels."""

    with pytest.raises(ValueError, match="at least two"):
        TableOneVariableSpec.model_validate(
            {
                "name": "organ_score",
                "variable_kind": "ordinal",
                "summary": "count_percent",
                "test": "chi_square_with_fisher_exact_for_sparse_2x2",
                "levels": [0],
            }
        )


# ---------------------------------------------------------------------------
# The declaration has to mean something
# ---------------------------------------------------------------------------


def test_the_declared_set_is_enforced_on_the_rows():
    """A score outside the declared stages stops the step.

    Without this, declaring 0-4 and meeting a 7 would summarise the 7 as an
    ordinary observation -- pulling the median and the IQR toward a value the
    study said could not occur, with nothing in the output to show it.
    """

    spec = TableOneSpec.model_validate(_spec(_ORDINAL_ROW))

    with pytest.raises(TableOneContractError, match="outside the Planner-declared"):
        build_grouped_table_one(_frame([0.0, 1.0, 2.0, 3.0, 4.0, 7.0] * 4), spec)


def test_values_inside_the_declared_set_are_summarised_normally():
    table = build_grouped_table_one(
        _frame([0.0, 1.0, 2.0, 3.0, 4.0] * 8),
        TableOneSpec.model_validate(_spec(_ORDINAL_ROW)),
    )

    overall = table[table["group"].eq("Overall")].iloc[0]
    assert overall["median"] == 2.0
    assert int(overall["denominator_n"]) == 40


def test_an_undeclared_ordinal_row_is_not_range_checked():
    """Omitting the set is a real choice, not a silently enforced default."""

    table = build_grouped_table_one(
        _frame([0.0, 1.0, 2.0, 3.0, 4.0, 7.0] * 4),
        TableOneSpec.model_validate(_spec({**_ORDINAL_ROW, "levels": []})),
    )

    assert not table.empty


def test_a_missing_value_is_not_an_undeclared_one():
    """Missingness in a row variable is reported, not refused.

    Only the grouping column may not be missing. Real organ scores are sparse
    -- on the measured cohort one SOFA component was absent for 43,112 of
    94,458 stays -- so treating an absent score as an undeclared level would
    close the door this fix just opened.
    """

    table = build_grouped_table_one(
        _frame([0.0, 1.0, None, 3.0, 4.0] * 8),
        TableOneSpec.model_validate(_spec(_ORDINAL_ROW)),
    )

    overall = table[table["group"].eq("Overall")].iloc[0]
    assert int(overall["missing_n"]) == 8
    assert int(overall["nonmissing_n"]) == 32


# ---------------------------------------------------------------------------
# The Planner is told the rule where it reads it
# ---------------------------------------------------------------------------


def test_the_directive_states_the_rule_per_variable_kind():
    """AA1's lesson: a gate must not forbid what the directive asks for.

    The old sentence listed "closed levels" for every variable without saying
    that two of the three kinds reject them.
    """

    from easyicu.research_agent.agents.core import _build_planner_user_prompt
    from easyicu.research_agent.schema import (
        CohortDescriptor,
        ConceptDescriptor,
        ResearchContext,
        VariableRole,
    )

    directive = _build_planner_user_prompt(
        ResearchContext(
            research_question="Does the Table 1 directive state the levels rule?",
            cohort=CohortDescriptor(
                cohort_name="example",
                database="miiv",
                n_patients=10,
                n_stays=10,
                id_columns=["stay_id"],
                outcome_columns=["outcome_flag"],
            ),
            variables=[
                ConceptDescriptor(
                    name="outcome_flag",
                    role=VariableRole.OUTCOME,
                    source_concept="outcome_flag",
                    dtype="int",
                )
            ],
            target_outcome="outcome_flag",
        )
    )

    assert "Levels follow the variable kind" in directive
    assert "'ordinal' row summarised numerically may declare" in directive
    assert "'continuous' row must leave levels empty" in directive
