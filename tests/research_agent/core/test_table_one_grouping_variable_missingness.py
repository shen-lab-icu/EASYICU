"""A Table 1 may be grouped by a variable that is not observed on every row.

MEASURED 2026-07-30 in ``e3_kdigo_gradient`` of batch
``..._88d3983_canonical9_full02``.  Step ``02_table_one_by_kdigo_stage`` was
claimed by the deterministic owner ``grouped_table_one`` (``contract_matches``
true, ``outcome`` selected), ran, and died inside its container with::

    TableOneContractError: Table 1 grouping variable contains missing values
    under fail_closed policy

The data fact behind it: ``aki_stage_max`` -- the KDIGO stage -- is absent for
696 of the cohort's 94,458 stays, 0.74%.  ``missing_group_policy`` had exactly
one legal value, ``fail_closed``, so no plan could have declared anything else.
A Table 1 grouped on a measurement-derived clinical score was therefore
unbuildable by construction, because such a score is never observed on every
stay.

The control is in the same batch, same owner, same code path: ``e2`` and ``m1``
both grouped by ``death``, which no stay is missing, and both finished ``ok``.
Missingness in the grouping column was the only difference.

Two more things the recorded run got wrong downstream, which is why the fix
belongs at the declaration and not in a repair: the deterministic repair
correctly declined (a data fact is not a code defect), then two LLM repairs
were spent, and the draft they produced changed one line -- reading the spec
out of the manifest instead of inlining it -- without touching the failing
call.  The preflight blocked that draft, and *its* message is what the manifest
records as the step's blocker.

``exclude_and_report`` removes the unplaceable rows from one frame that then
feeds Overall and every group, so Overall stays equal to the sum of its parts,
and the count removed rides on every emitted row.  Splitting the filter -- the
groups filtered, Overall not -- would reproduce the defect fixed in ``0160ddb``
one table over: a denominator and its parts describing two different row sets.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from easyicu.research_agent.contracts.table_one import (  # noqa: E402
    table_one_output_findings,
)
from easyicu.research_agent.methods.table_one import (  # noqa: E402
    TableOneContractError,
    build_grouped_table_one,
)
from easyicu.research_agent.schema import AnalysisStep, TableOneSpec  # noqa: E402


def _spec(policy: str = "fail_closed") -> dict:
    """The shape of the recorded E3 declaration: an ordered stage, 0..3."""

    return {
        "group_by": "stage",
        "group_levels": [0, 1, 2, 3],
        "missing_group_policy": policy,
        "variables": [
            {
                "name": "age",
                "variable_kind": "continuous",
                "summary": "median_iqr",
                "test": "mann_whitney_or_kruskal",
            },
            {
                "name": "sex",
                "variable_kind": "categorical",
                "summary": "count_percent",
                "test": "chi_square_with_fisher_exact_for_sparse_2x2",
                "levels": ["Female", "Male"],
            },
        ],
    }


def _frame(n_unstaged: int = 6) -> pd.DataFrame:
    """40 staged rows plus ``n_unstaged`` the grouping variable cannot place."""

    stage = [0, 1, 2, 3] * 10 + [None] * n_unstaged
    total = len(stage)
    return pd.DataFrame(
        {
            "stage": pd.Series(stage, dtype="float64"),
            "age": pd.Series(
                [50.0 + (index % 37) for index in range(total)], dtype="float64"
            ),
            "sex": ["Female" if index % 2 else "Male" for index in range(total)],
        }
    )


def _step(policy: str = "fail_closed") -> AnalysisStep:
    return AnalysisStep(
        step_id="02_table_one_by_stage",
        intent="Describe the cohort by stage.",
        inputs=["stage", "age", "sex"],
        expected_outputs=["table:table_one"],
        method="descriptive",
        table_one_spec=_spec(policy),
    )


def _write(table: pd.DataFrame, tmp_path) -> None:
    table.to_csv(tmp_path / "table_one.csv", index=False)


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------


def test_the_policy_offers_more_than_one_answer():
    """One legal value is not a policy; it is a constant with a question mark.

    ``ExposureOutcomeDistributionSpec.missing_outcome_policy`` -- the same
    question about an unobserved value, one spec over -- has offered three
    answers all along.
    """

    field = TableOneSpec.model_fields["missing_group_policy"]
    legal = set(field.annotation.__args__)

    assert legal == {"fail_closed", "exclude_and_report"}
    assert field.default == "fail_closed"


def test_a_grouping_variable_with_missing_values_can_be_declared():
    table = build_grouped_table_one(_frame(), _spec("exclude_and_report"))

    assert set(table["group"]) == {"Overall", "0", "1", "2", "3"}
    assert set(table["schema_version"]) == {"easyicu.table_one_result/3"}


def test_fail_closed_still_refuses_and_now_says_how_many():
    """The default is unchanged, but the refusal names what it refused on.

    The recorded message said only that missing values existed. A Planner
    reading "696 of 94,458" can tell a 0.74% exclusion from a broken column.
    """

    with pytest.raises(TableOneContractError) as raised:
        build_grouped_table_one(_frame(n_unstaged=6), _spec("fail_closed"))

    message = str(raised.value)
    assert "6 of 46 rows" in message
    assert "'stage'" in message


# ---------------------------------------------------------------------------
# What the emitted table must say
# ---------------------------------------------------------------------------


def test_overall_is_taken_over_the_same_rows_as_its_groups():
    table = build_grouped_table_one(_frame(), _spec("exclude_and_report"))

    for variable in ("age", "sex"):
        rows = table[table["variable"].eq(variable)]
        per_group = rows.groupby(rows["group"].astype(str))["denominator_n"].first()
        parts = sum(int(per_group[str(level)]) for level in (0, 1, 2, 3))

        assert int(per_group["Overall"]) == parts == 40


def test_the_excluded_count_rides_on_every_row():
    """A count kept anywhere else can be dropped between here and the reader."""

    table = build_grouped_table_one(_frame(n_unstaged=6), _spec("exclude_and_report"))

    assert set(table["group_missing_excluded_n"]) == {6}
    assert len(table) > 1


def test_the_kept_rows_are_summarised_exactly_as_if_they_were_the_cohort():
    """Excluding rows must not perturb the rows that remain."""

    frame = _frame(n_unstaged=6)
    with_policy = build_grouped_table_one(frame, _spec("exclude_and_report"))
    prefiltered = build_grouped_table_one(
        frame[frame["stage"].notna()], _spec("fail_closed")
    )

    shared = [column for column in prefiltered.columns if column != "contract_sha256"]
    left = with_policy[shared].drop(columns=["group_missing_excluded_n"])
    right = prefiltered[shared].drop(columns=["group_missing_excluded_n"])
    pd.testing.assert_frame_equal(left, right)


def test_a_fully_observed_grouping_variable_is_untouched():
    """The e2/m1 control: grouping on a column no row is missing."""

    frame = _frame(n_unstaged=0)

    for policy in ("fail_closed", "exclude_and_report"):
        table = build_grouped_table_one(frame, _spec(policy))
        assert set(table["group_missing_excluded_n"]) == {0}
        overall = table[
            table["variable"].eq("age") & table["group"].eq("Overall")
        ].iloc[0]
        assert int(overall["denominator_n"]) == 40


# ---------------------------------------------------------------------------
# The output gate reads the table, and only the table
# ---------------------------------------------------------------------------


def test_the_gate_accepts_a_declared_exclusion(tmp_path):
    _write(build_grouped_table_one(_frame(), _spec("exclude_and_report")), tmp_path)

    findings = table_one_output_findings(
        step=_step("exclude_and_report"), out_dir=tmp_path
    )

    assert [finding.detail.get("reason") for finding in findings] == []


def test_the_gate_refuses_an_exclusion_the_declaration_did_not_allow(tmp_path):
    """A table claiming exclusions under fail_closed contradicts its own spec."""

    table = build_grouped_table_one(_frame(), _spec("exclude_and_report"))
    _write(table, tmp_path)

    findings = table_one_output_findings(step=_step("fail_closed"), out_dir=tmp_path)

    reasons = {finding.detail.get("reason") for finding in findings}
    assert "table_one_group_exclusion_undeclared" in reasons


def test_the_gate_catches_an_overall_that_kept_the_excluded_rows(tmp_path):
    """The failure mode the exclusion policy makes possible.

    An executor that filtered the groups but left Overall on the unfiltered
    frame would report a denominator larger than the sum of its parts, and
    nothing in the table would say so.
    """

    table = build_grouped_table_one(_frame(), _spec("exclude_and_report"))
    inflated = table.copy()
    overall = inflated["group"].astype(str).eq("Overall")
    inflated.loc[overall, "denominator_n"] = inflated.loc[overall, "denominator_n"] + 6
    _write(inflated, tmp_path)

    findings = table_one_output_findings(
        step=_step("exclude_and_report"), out_dir=tmp_path
    )

    reasons = {finding.detail.get("reason") for finding in findings}
    assert "table_one_overall_denominator_mismatch" in reasons


def test_the_gate_refuses_a_table_that_does_not_report_one_count(tmp_path):
    table = build_grouped_table_one(_frame(), _spec("exclude_and_report"))
    varying = table.copy()
    varying.iloc[0, varying.columns.get_loc("group_missing_excluded_n")] = 99
    _write(varying, tmp_path)

    findings = table_one_output_findings(
        step=_step("exclude_and_report"), out_dir=tmp_path
    )

    reasons = {finding.detail.get("reason") for finding in findings}
    assert "table_one_group_exclusion_inconsistent" in reasons


def test_the_gate_refuses_a_table_that_omits_the_count_entirely(tmp_path):
    table = build_grouped_table_one(_frame(), _spec("exclude_and_report"))
    _write(table.drop(columns=["group_missing_excluded_n"]), tmp_path)

    findings = table_one_output_findings(
        step=_step("exclude_and_report"), out_dir=tmp_path
    )

    reasons = {finding.detail.get("reason") for finding in findings}
    assert "table_one_schema_incomplete" in reasons


# ---------------------------------------------------------------------------
# The Planner is told the choice exists
# ---------------------------------------------------------------------------


def test_the_planner_directive_names_both_answers():
    """A policy the Planner is never told about is a policy it will not use.

    It is also told where to look before choosing: a grouping variable's own
    missingness is the fact that decides which answer is right, and the
    variable catalog already carries it (the recorded E3 context reported
    ``n_missing`` 696 of 94,458 for the column its plan then declared
    ``fail_closed``).
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
            research_question="Does the Table 1 directive still offer both?",
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

    assert "missing_group_policy" in directive
    assert "'exclude_and_report'" in directive
    assert "'fail_closed'" in directive
    assert "missingness" in directive
