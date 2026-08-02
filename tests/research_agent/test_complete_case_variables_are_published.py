"""The host's own worked example produced a spec the host then refused.

canary19's E1 finally reached the robustness step with a validated primary
estimate -- and blocked on::

    alt_missing_complete_case: complete-case equivalence requires explicit
    locked variables

The plan had declared the variables.  It called them ``columns``, because
nothing told it the key.  The planner directive's worked example showed
``"missing_override": {"strategy": "complete_case"}`` with no variable list at
all, while the equivalence proof required ``variables`` and the concept
validator validated ``variables`` -- neither of them published anywhere.

Measured over 94 locked complete-case specs from real runs: 44 wrote
``variables``, **50 wrote no list at all** (the example, copied verbatim), and
3 invented ``columns`` or ``required_variables``.  All 53 were refused at
execution time, after the whole analysis had already run.

Two things are wrong and both are fixed here:

* the key is now published where the Planner reads -- in the sentence and in
  the worked example, rendered from the constant that enforces it;
* the refusal moved to plan validation, where the Planner can still supply the
  list, instead of arriving six steps later as a dead run.

The list is required rather than inferred on purpose.  A model fitted on one
adjustment set and a complete-case restriction taken over a different set are
different analyses, so the host asks.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.agents.core import _build_planner_user_prompt
from easyicu.research_agent.planning.robustness_contract import (
    COMPLETE_CASE_STRATEGY,
    COMPLETE_CASE_VARIABLES_KEY,
    RobustnessPlanError,
    RobustnessSpec,
    complete_case_variables,
    validate_planner_robustness_specs,
    validate_robustness_specs,
)
from easyicu.research_agent.schema import CohortDescriptor, ResearchContext


def _spec(missing_override, *, spec_id="alt_missing_complete_case"):
    return RobustnessSpec(
        spec_id=spec_id,
        axis="missing",
        description="Repeat the primary model on complete cases.",
        missing_override=missing_override,
    )


_DECLARED = {"strategy": "complete_case", "variables": ["exposure", "death", "age"]}


# --- the constants are the ones the run actually uses ------------------------


def test_the_published_key_names_are_the_enforced_ones():
    assert COMPLETE_CASE_STRATEGY == "complete_case"
    assert COMPLETE_CASE_VARIABLES_KEY == "variables"


def test_a_declared_list_is_read():
    assert complete_case_variables(_spec(_DECLARED)) == ["exposure", "death", "age"]


def test_whitespace_is_stripped_but_a_blank_name_is_not_a_name():
    assert complete_case_variables(
        _spec({"strategy": "complete_case", "variables": [" age ", "sex"]})
    ) == ["age", "sex"]
    assert (
        complete_case_variables(
            _spec({"strategy": "complete_case", "variables": ["age", "  "]})
        )
        is None
    )


@pytest.mark.parametrize(
    "override",
    [
        {"strategy": "complete_case"},
        {"strategy": "complete_case", "variables": []},
        {"strategy": "complete_case", "variables": "age"},
        {"strategy": "complete_case", "variables": ["age", 7]},
        {"strategy": "complete_case", "columns": ["age", "sex"]},
        {"strategy": "complete_case", "required_variables": ["age"]},
    ],
    ids=[
        "the-worked-example-as-it-was",
        "empty",
        "a-string-is-not-a-list",
        "a-number-is-not-a-column",
        "the-spelling-canary19-used",
        "another-invented-spelling",
    ],
)
def test_an_undeclared_list_reads_as_undeclared(override):
    assert complete_case_variables(_spec(override)) is None


def test_a_different_strategy_is_not_this_rule():
    assert complete_case_variables(_spec({"strategy": "median_imputation"})) is None
    assert complete_case_variables(_spec(None)) is None


# --- and the refusal happens while the Planner can still act -----------------


def test_the_canary19_spec_is_refused_at_plan_validation():
    """It used to validate here and die at step 07 with six steps already spent."""

    with pytest.raises(RobustnessPlanError) as excinfo:
        validate_planner_robustness_specs(
            [_spec({"strategy": "complete_case", "columns": ["age", "sex"]})]
        )

    message = str(excinfo.value)
    assert COMPLETE_CASE_VARIABLES_KEY in message
    assert "columns" in message, "the refusal must name what the plan did write"
    assert "alt_missing_complete_case" in message


def test_the_old_worked_example_is_refused_too():
    with pytest.raises(RobustnessPlanError) as excinfo:
        validate_planner_robustness_specs([_spec({"strategy": "complete_case"})])

    assert COMPLETE_CASE_VARIABLES_KEY in str(excinfo.value)


def test_a_declared_spec_validates():
    validate_planner_robustness_specs([_spec(_DECLARED)])


def test_a_repeated_variable_is_refused():
    with pytest.raises(RobustnessPlanError, match="repeats a variable"):
        validate_planner_robustness_specs(
            [_spec({"strategy": "complete_case", "variables": ["age", "age"]})]
        )


def test_other_axes_and_strategies_are_untouched():
    validate_planner_robustness_specs(
        [
            _spec({"strategy": "median_imputation"}, spec_id="alt_median"),
            RobustnessSpec(
                spec_id="alt_outcome",
                axis="outcome",
                description="An alternative endpoint.",
                outcome_override={"concept_id": "death", "aggregation": "any"},
            ),
        ]
    )


def test_the_refusal_explains_why_the_host_will_not_infer():
    with pytest.raises(RobustnessPlanError) as excinfo:
        validate_planner_robustness_specs([_spec({"strategy": "complete_case"})])

    assert "different analysis" in str(excinfo.value)


def test_the_shared_structural_validator_still_accepts_it():
    """The population boundary, asserted rather than assumed.

    ``validate_robustness_specs`` is also asked about locks being re-read and
    about case-neutral placeholders. Attaching a Planner-output requirement to
    it judged those populations too, and 26 tests said so.
    """

    validate_robustness_specs([_spec({"strategy": "complete_case"})])


# --- and the plan constructor does NOT, which is the same boundary again -----


def _plan_with(missing_override) -> None:
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    AnalysisPlan(
        research_question="Does severity predict mortality?",
        steps=[
            AnalysisStep(
                step_id="01_model",
                intent="Fit the primary model.",
                expected_outputs=["statistic:primary_or"],
            )
        ],
        robustness_specs=[
            {
                "spec_id": "alt_missing_complete_case",
                "axis": "missing",
                "description": "Complete cases only.",
                "cohort_override": None,
                "missing_override": missing_override,
                "outcome_override": None,
            }
        ],
    )


def test_the_plan_constructor_accepts_it_because_it_also_loads_recorded_plans():
    """The same population argument as the shared validator, one level up.

    This assertion used to demand the opposite -- that constructing an
    ``AnalysisPlan`` raise -- on the reasoning that "without this the wiring can
    revert and only a real run would notice". The worry is right; the object was
    wrong. ``AnalysisPlan.model_validate`` is not only how Planner output
    arrives: it is also how a recorded plan is loaded from disk, how a lock is
    re-read on resume, and how the framework's own case-neutral placeholders are
    built. Judging all of those by what the *Planner must declare* stopped 190 of
    409 recorded plan documents from parsing, and a resume of any of them would
    have failed at load.

    So the rule lives where Planner output is accepted, and the two halves are
    pinned separately: this one says the constructor stays permissive, and
    ``test_planning_contract_boundaries`` says the Planner path still applies the
    rule -- both by identity and at the call site, so an import that is never
    called cannot pass for wiring.
    """

    _plan_with({"strategy": "complete_case", "columns": ["age"]})
    _plan_with({"strategy": "complete_case"})


def test_the_plan_accepts_a_declared_spec():
    _plan_with({"strategy": "complete_case", "variables": ["severity", "mortality"]})


# --- the executor asks the same question the plan was judged by --------------


@pytest.mark.parametrize(
    "override",
    [
        {"strategy": "complete_case"},
        {"strategy": "complete_case", "columns": ["age"]},
        {"strategy": "complete_case", "variables": ["age", "   "]},
        {"strategy": "complete_case", "variables": "age"},
    ],
    ids=["absent", "another-spelling", "a-blank-name", "a-string-not-a-list"],
)
def test_the_equivalence_proof_refuses_exactly_what_the_plan_refuses(override):
    """Same reader on both sides, so neither can accept what the other rejects.

    The blank name and the bare string are the cases that separate the shared
    reader from a hand-rolled ``override.get("variables")``: both of those
    would survive a local re-implementation and neither is a column list.
    """

    from easyicu.research_agent.execution.runners.deterministic_robustness import (
        _verified_complete_case_equivalence,
    )

    row, contracts, replay, error = _verified_complete_case_equivalence(
        spec=_spec(override),
        source={},
        primary_data=None,
    )

    assert error is not None
    # The exact refusal, not merely one containing the word "variables": the
    # very next check reports "variables are absent from the primary cohort",
    # which satisfied a looser assertion and let a hand-rolled reader survive
    # a mutation that accepted a blank column name.
    assert "requires explicit locked variables" in error
    assert f"missing_override.{COMPLETE_CASE_VARIABLES_KEY}" in error
    assert row.converged is False
    assert contracts == []
    assert replay is None


# --- the directive states it, so the Planner is not guessing -----------------


@pytest.fixture(scope="module")
def directive() -> str:
    return _build_planner_user_prompt(
        ResearchContext(
            research_question="Is the exposure associated with death?",
            cohort=CohortDescriptor(
                cohort_name="complete-case",
                database="test",
                n_patients=10,
                n_stays=10,
                id_columns=["stay_id"],
            ),
            variables=[],
        )
    )


def test_the_directive_names_the_key(directive):
    assert f"`{COMPLETE_CASE_VARIABLES_KEY}`" in directive
    assert f"'{COMPLETE_CASE_STRATEGY}' MUST also carry" in directive


def test_the_worked_example_shows_the_list(directive):
    marker = '"spec_id": "alt_missing_complete_case"'
    assert marker in directive
    example = directive[directive.index(marker) :][:400]

    assert f'"{COMPLETE_CASE_VARIABLES_KEY}"' in example
    assert "<each covariate>" in example


def test_following_the_worked_example_produces_a_spec_that_validates(directive):
    """The property that was false: the example itself was refused.

    Reads the shape out of the rendered directive rather than restating it, so
    an example that stops carrying a list fails here instead of in a run.
    """

    marker = '"spec_id": "alt_missing_complete_case"'
    example = directive[directive.index(marker) :][:400]
    assert f'"strategy": "{COMPLETE_CASE_STRATEGY}"' in example
    assert f'"{COMPLETE_CASE_VARIABLES_KEY}": [' in example

    # A Planner following it substitutes real column names for the placeholders.
    validate_planner_robustness_specs(
        [
            _spec(
                {
                    "strategy": COMPLETE_CASE_STRATEGY,
                    COMPLETE_CASE_VARIABLES_KEY: ["exposure", "death", "age"],
                }
            )
        ]
    )
