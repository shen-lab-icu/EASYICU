"""A refusal that names a role must name what would satisfy it.

Forcing the trajectory contract on over H3's real recorded plan (full03,
run_20260731T105349) produces four findings, three of them::

    The trajectory plan is missing one structured scientific role.  role=representation
    ... role=candidate_selection
    ... role=stability_freeze

None of those four role names appears anywhere in the Planner prompt -- the
contract was enforced without ever being stated -- so a replan is being asked
to add an owner for a role it has never heard of, with products it cannot
guess.  ``_role_qualifies`` decides ownership from method-family tokens plus
exact typed product sets, and every one of those sets is a literal in the
module, so the refusal can simply render them.

The test that carries the weight is the round trip: a step built from exactly
what a message asks for must satisfy the predicate that produced the message.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ResearchContext,
)
from easyicu.research_agent.trajectory.plan_contract import (
    _ROLE_ORDER,
    _role_qualifies,
    _method_head,
    _step_products,
    evaluate_trajectory_plan_dag,
    role_declaration_requirement,
)

# One method string per role that names a token the requirement publishes.
_QUALIFYING_METHOD = {
    "representation": "trajectory_representation",
    "candidate_selection": "trajectory_clustering",
    "stability_freeze": "trajectory_clustering_stability",
    "characterization": "cluster_characterization",
}


def _owner_step(role: str) -> AnalysisStep:
    """Declare exactly one product from each group the requirement publishes."""

    requirement = role_declaration_requirement(role)
    outputs = [sorted(group)[0] for group in requirement.product_groups]
    return AnalysisStep(
        step_id=f"0x_{role}",
        intent=f"Own the {role} role.",
        method=_QUALIFYING_METHOD[role],
        expected_outputs=outputs,
        inputs=[],
    )


@pytest.mark.parametrize("role", sorted(_ROLE_ORDER))
def test_declaring_exactly_what_the_message_asks_for_satisfies_the_predicate(role):
    """The round trip. If this fails the host is publishing a false promise."""

    step = _owner_step(role)

    assert _role_qualifies(
        role,
        method=_method_head(step.method),
        products=_step_products(step),
    )


@pytest.mark.parametrize("role", sorted(_ROLE_ORDER))
def test_every_published_product_really_is_one_the_predicate_accepts(role):
    """Not just the first of each group -- every published alternative works."""

    requirement = role_declaration_requirement(role)
    method = _method_head(_QUALIFYING_METHOD[role])
    first_group = requirement.product_groups[0]
    rest = [sorted(group)[0] for group in requirement.product_groups[1:]]

    for candidate in sorted(first_group):
        step = AnalysisStep(
            step_id=f"0x_{role}",
            intent="x",
            method=_QUALIFYING_METHOD[role],
            expected_outputs=[candidate, *rest],
            inputs=[],
        )
        assert _role_qualifies(
            role, method=method, products=_step_products(step)
        ), f"{role} publishes {candidate} but the predicate refuses it"


@pytest.mark.parametrize("role", sorted(_ROLE_ORDER))
def test_a_role_with_a_second_required_group_is_not_satisfied_by_one_group(role):
    """A published `and` must really be an and, or the message oversells."""

    requirement = role_declaration_requirement(role)
    if len(requirement.product_groups) < 2:
        pytest.skip(f"{role} publishes a single product group")

    partial = AnalysisStep(
        step_id=f"0x_{role}",
        intent="x",
        method=_QUALIFYING_METHOD[role],
        expected_outputs=[sorted(requirement.product_groups[0])[0]],
        inputs=[],
    )

    assert not _role_qualifies(
        role,
        method=_method_head(partial.method),
        products=_step_products(partial),
    )


@pytest.mark.parametrize("role", sorted(_ROLE_ORDER))
def test_the_requirement_publishes_a_method_token_and_at_least_one_product(role):
    requirement = role_declaration_requirement(role)

    assert requirement.method_tokens
    assert requirement.product_groups
    assert all(group for group in requirement.product_groups)


def test_an_unknown_role_fails_closed_rather_than_returning_an_empty_promise():
    with pytest.raises(ValueError):
        role_declaration_requirement("not_a_trajectory_role")


# --- the refusal actually carries it ----------------------------------------


def _empty_trajectory_plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Do trajectory subgroups differ?",
        analysis_type="trajectory_clustering",
        steps=[
            AnalysisStep(
                step_id="01_define_analysis_cohort",
                intent="x",
                method="cohort_definition",
                expected_outputs=["artifact:analysis_cohort", "table:cohort_flow"],
                inputs=[],
            )
        ],
    )


def _findings(monkeypatch) -> list:
    from easyicu.research_agent.trajectory import plan_contract as module

    monkeypatch.setattr(
        module, "trajectory_plan_contract_applies", lambda **_kwargs: True
    )
    context = ResearchContext(
        research_question="Do trajectory subgroups differ?",
        cohort=CohortDescriptor(
            cohort_name="trajectory-role-requirement",
            database="test",
            n_patients=20,
            n_stays=20,
            id_columns=["stay_id"],
        ),
        variables=[],
    )
    return list(
        evaluate_trajectory_plan_dag(
            plan=_empty_trajectory_plan(), context=context
        ).findings
    )


def test_the_refusal_names_the_role_and_what_would_own_it(monkeypatch):
    messages = [
        finding.message
        for finding in _findings(monkeypatch)
        if (finding.detail or {}).get("kind") == "trajectory_role_missing"
    ]

    assert messages, "no role-missing finding was produced"
    for message in messages:
        assert "expected_outputs containing" in message


@pytest.mark.parametrize("role", sorted(_ROLE_ORDER))
def test_the_refusal_for_each_role_quotes_that_role_s_own_products(monkeypatch, role):
    """A message must not carry another role's set."""

    finding = next(
        item
        for item in _findings(monkeypatch)
        if (item.detail or {}).get("kind") == "trajectory_role_missing"
        and (item.detail or {}).get("role") == role
    )
    published = {
        product
        for group in role_declaration_requirement(role).product_groups
        for product in group
    }

    assert published
    for product in published:
        assert product in finding.message


def test_the_detail_carries_the_sets_as_data_not_only_as_prose(monkeypatch):
    finding = next(
        item
        for item in _findings(monkeypatch)
        if (item.detail or {}).get("kind") == "trajectory_role_missing"
    )
    detail = finding.detail or {}

    assert detail.get("required_method_family_tokens")
    assert detail.get("qualifying_products")
    role = detail["role"]
    expected = [
        sorted(group) for group in role_declaration_requirement(role).product_groups
    ]
    assert detail["qualifying_products"] == expected
