"""A rule the schema enforces and the directive never states costs an attempt.

canary11 (E3, 2026-07-31) never reached execution. Five planner attempts were
rejected in FOUR distinct ways, one rule each, and the run produced nothing::

    [0] steps.6.model_requirements.0: outcome_type='continuous' is
        incompatible with method_family=...
    [1] steps.6: model_requirements are supported only on
        method='adjusted_association_models' steps declaring
        'table:adjusted_association_estimates'
    [2] steps.7: robustness_replay_spec names products the step does not
        declare as outputs
    [3] steps.1.table_one_spec: Table 1 group_by must not also be a row
        variable

The retry already carries earlier rejections forward -- that was built after
three runs burned every attempt on different violations -- so the Planner was
told all four and still could not satisfy them together. Feedback is the wrong
place to learn a constraint set: by the time it arrives, an attempt is already
spent. The directive is the right place, and two of these four were not in it
in a usable form.

``group_by must not also be a row variable`` was not stated at all. And
``method_family`` was described ("a binary logistic family or a continuous
linear/quantile family") while the validator checks membership of an exact
frozenset -- so the Planner had to guess which strings count, which is the
gate-allowlist failure mode in its prompt-side form.

These tests bind the published text to the enforcing constant, so the two
cannot drift apart again: rendering the set from the frozenset means adding a
family updates the directive, and removing one from the directive fails here.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.schema import (
    ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES,
    ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    TableOneSpec,
)


def _directive() -> str:
    """The prompt the Planner is actually sent, built the way the host builds it.

    Asserted against the rendered prompt rather than the source file: the point
    is that the rule REACHES the model, and a paragraph assembled into a string
    the caller never sends would satisfy a source-level check while teaching the
    Planner nothing.
    """

    from easyicu.research_agent.agents.core import _build_planner_user_prompt

    context = ResearchContext(
        research_question="Does AKI stage grade in-hospital mortality?",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[
            ConceptDescriptor(name="aki_stage_max", dtype="float64"),
            ConceptDescriptor(name="death", dtype="int64"),
        ],
    )
    return _build_planner_user_prompt(context)


@pytest.mark.parametrize(
    "family",
    sorted(ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES)
    + sorted(ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES),
)
def test_every_accepted_method_family_is_named_in_the_directive(family) -> None:
    """The validator checks a set, so the directive must publish the set.

    Describing it ("a continuous linear/quantile family") leaves the Planner
    guessing which exact label passes -- and one guess costs one of five
    attempts.
    """

    assert family in _directive()


def test_the_two_outcome_types_are_kept_apart() -> None:
    """A family from the wrong list is refused, so the split must be visible."""

    text = _directive()
    binary_at = text.index(sorted(ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES)[0])
    continuous_at = text.index(
        sorted(ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES)[0]
    )
    assert "'binary'" in text and "'continuous'" in text
    assert binary_at != continuous_at


def test_the_table_one_grouping_rule_is_published() -> None:
    """The rule that rejected canary11's attempt 4, stated before it is enforced."""

    text = _directive()
    assert "GROUP ON IS NOT ALSO" in text
    assert "never in both" in text


def test_the_grouping_rule_is_really_enforced() -> None:
    """Publishing a rule nobody enforces would be the opposite defect.

    This is the validator canary11 actually hit; if it is ever relaxed, the
    directive paragraph above becomes a false constraint and should go too.
    """

    payload = {
        "group_by": "stage",
        "group_levels": [0, 1],
        "missing_group_policy": "fail_closed",
        "variables": [
            {
                "name": "stage",
                "variable_kind": "categorical",
                "summary": "count_percent",
                "test": "chi_square_with_fisher_exact_for_sparse_2x2",
                "levels": [0, 1],
            }
        ],
    }

    with pytest.raises(ValueError, match="must not also be a row variable"):
        TableOneSpec.model_validate(payload)
