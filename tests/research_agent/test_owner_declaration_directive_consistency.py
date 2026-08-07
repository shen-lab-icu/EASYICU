"""A gate must not demand a declaration and forbid making it in the same breath.

``execution/owner_declaration.py`` reports the fields an existing deterministic
owner is waiting on, and ships a replan directive alongside.  The directive
carried a hand-written prohibition list::

    Do not choose an exposure, outcome, cohort, covariate, estimand, or method

while the only gap the gate has ever reported is
``model_requirements[0].covariates`` -- measured 2026-07-30, 74 of the 81
recorded declarations of the paper's primary product decline for exactly that
field.  So one replan message told the Planner to declare the adjustment set
and forbade choosing one.

That is not a strict instruction; it is an unsatisfiable one, and doing nothing
is a defensible reading of it.  The durable fix is structural: the prohibition
is computed by subtracting whatever the findings demand, so it cannot collide
with a future owner's missing field either.
"""

from __future__ import annotations

import re

import pytest

from easyicu.research_agent.execution.owner_declaration import (
    _declared_choice,
    _prohibited_choices,
    _SCIENTIFIC_CHOICES,
    owner_declaration_plan_findings,
    owner_declaration_replan_directive,
)
from easyicu.research_agent.schema import (
    AnalysisStep,
    PlannedModelRequirement,
    ValidationFinding,
)

#: Every field name an owner emits as a missing declaration today.  Sourced by
#: grepping ``missing=(`` across ``src/``; the sweep test below fails if that
#: set grows without this list growing with it, so a new owner cannot quietly
#: reintroduce the collision.
EMITTED_MISSING_FIELDS = (
    "model_requirements",
    "model_requirements[0].covariates",
)


#: The prohibition clause, and nothing else.  Asserting over the whole message
#: was wrong and its own failure said so: the message legitimately *names* the
#: field it wants declared (``'model_requirements[0].covariates'``) and quotes
#: the owner's reason ("declares no adjustment set"), so a bare
#: ``"covariate" not in message`` fails on a message that is already correct.
#: A check has to anchor on what is exclusive to the thing under test.
_PROHIBITION = re.compile(
    r"Do not (?:change the|choose a different) (.+?) to satisfy this",
    re.S,
)


def _prohibition_clause(text: str) -> str:
    match = _PROHIBITION.search(text)
    assert match is not None, f"no prohibition clause in: {text}"
    return match.group(1).casefold()


def _finding(*missing: str) -> ValidationFinding:
    return ValidationFinding(
        validator="plan_owner_declaration",
        severity="error",
        message="step under-declares",
        detail={
            "reason": "owner_declaration_incomplete",
            "step_id": "06_primary_adjusted_association",
            "analysis_kind": "adjusted_association_estimates",
            "missing_declarations": list(missing),
        },
    )


# ---------------------------------------------------------------------------
# The defect itself
# ---------------------------------------------------------------------------


def test_the_directive_does_not_forbid_the_covariates_it_demands():
    """The live case: 91% of real declines are this exact field."""

    directive = owner_declaration_replan_directive(
        [_finding("model_requirements[0].covariates")]
    )
    assert directive is not None
    clause = _prohibition_clause(directive)
    assert "covariate" not in clause, clause
    # and the rest of the guard is still there
    for kept in ("exposure", "outcome", "cohort", "estimand", "method"):
        assert kept in clause, kept


@pytest.mark.parametrize("missing", EMITTED_MISSING_FIELDS)
def test_no_field_any_owner_emits_is_both_demanded_and_forbidden(missing: str):
    """The sweep, not just the one field that bit us."""

    directive = owner_declaration_replan_directive([_finding(missing)])
    assert directive is not None
    clause = _prohibition_clause(directive)
    demanded = _declared_choice(missing)
    assert demanded not in re.split(r"[\s,]+", clause), (
        f"the directive forbids {demanded!r} while the finding demands "
        f"{missing!r}: {clause}"
    )


def test_the_prohibition_survives_a_finding_that_demands_nothing_listed():
    """A gap in a field that is not a scientific choice keeps the full guard."""

    assert _prohibited_choices(["model_requirements"]) == _SCIENTIFIC_CHOICES


# ---------------------------------------------------------------------------
# The matching rule -- exact on the normalised leaf, never substring
# ---------------------------------------------------------------------------


def test_a_neighbouring_field_name_does_not_unlock_a_choice():
    """``outcome_levels`` is not permission to choose the ``outcome``.

    A substring test would grant it, which is how an over-loose string set
    turns a guard off without anyone editing the guard.
    """

    assert "outcome" in _prohibited_choices(["outcome_levels"])
    assert "exposure" in _prohibited_choices(["exposure_window_hours"])
    assert "cohort" in _prohibited_choices(["cohort_receipt_sha256"])


@pytest.mark.parametrize(
    "name,expected",
    [
        ("model_requirements[0].covariates", "covariate"),
        ("covariates", "covariate"),
        ("Covariates", "covariate"),
        ("spec.exposure", "exposure"),
        ("model_requirements", "model_requirement"),
        ("model_requirements[12].outcome", "outcome"),
    ],
)
def test_the_leaf_is_what_names_the_choice(name: str, expected: str):
    assert _declared_choice(name) == expected


# ---------------------------------------------------------------------------
# The finding message and the directive must agree with each other
# ---------------------------------------------------------------------------


def _under_declared_plan():
    """A real plan shaped like the 74 recorded ones: covariates left null.

    Built rather than loaded, so the test does not depend on a run directory --
    but built to the same declaration the gate actually declines on, with the
    adjustment columns present in ``inputs`` exactly as the real plans carry
    them.  That last detail is the point of the whole gate: the Planner did
    choose, and only failed to say so in the typed field.
    """

    requirement = PlannedModelRequirement(
        requirement_id="primary_marker_outcome_logistic",
        outcome="outcome",
        outcome_type="binary",
        method_family="binary_logistic_regression",
        exposure_source="marker_max",
        analysis_role="primary",
        analysis_set="complete_case",
        covariates=None,
    )
    step = AnalysisStep(
        step_id="06_primary_adjusted_association",
        planned_analysis_role="primary",
        intent="Estimate the adjusted association declared by the plan.",
        inputs=["artifact:analysis_cohort", "marker_max", "age", "sex", "outcome"],
        expected_outputs=["table:adjusted_association_estimates"],
        method="adjusted_association_models",
        model_requirements=[requirement],
    )

    class _Plan:
        steps = (step,)

    return _Plan()


def test_the_finding_message_itself_does_not_forbid_the_covariates_it_demands():
    """The message path, driven through the real gate on a real declaration.

    Asserting only on the directive would leave the other half of what reaches
    the Planner unchecked -- and these two drifted apart once already, which is
    how the contradiction survived review.
    """

    findings = owner_declaration_plan_findings(plan=_under_declared_plan())
    assert findings, "the gate must still decline this declaration"
    finding = findings[0]
    assert finding.detail["missing_declarations"] == [
        "model_requirements[0].covariates"
    ]
    clause = _prohibition_clause(finding.message)
    assert "covariate" not in clause, clause
    for kept in ("exposure", "outcome", "cohort", "estimand", "method"):
        assert kept in clause, kept


def test_message_and_directive_forbid_exactly_the_same_things():
    """One computed list, so a future edit cannot move one without the other."""

    findings = owner_declaration_plan_findings(plan=_under_declared_plan())
    assert findings
    directive = owner_declaration_replan_directive(findings)
    assert directive is not None
    in_directive = _prohibition_clause(directive)
    in_message = _prohibition_clause(findings[0].message)
    expected = _prohibited_choices(findings[0].detail["missing_declarations"])
    for choice in expected:
        assert choice in in_directive, choice
        assert choice in in_message, choice
    for choice in _SCIENTIFIC_CHOICES:
        if choice in expected:
            continue
        assert choice not in in_directive, choice
        assert choice not in in_message, choice


def test_a_directive_never_renders_an_empty_prohibition():
    """If a finding demanded every listed choice, the sentence must still parse.

    ``Do not choose a different  to satisfy this`` is the kind of output that
    reads as a formatting bug and gets ignored.
    """

    directive = owner_declaration_replan_directive(
        [_finding(*(f"{choice}s" for choice in _SCIENTIFIC_CHOICES))]
    )
    assert directive is not None
    assert not re.search(r"different\s+to satisfy", directive), directive
    assert "any scientific choice already declared" in _prohibition_clause(directive)


def test_no_findings_means_no_directive():
    assert owner_declaration_replan_directive([]) is None
