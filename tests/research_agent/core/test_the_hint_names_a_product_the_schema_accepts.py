"""The host told the Planner to declare the one product its schema refuses.

When a plan is missing a required article-contract role, the refusal carries a
remediation hint: "Required typed step examples: <role> -> 'table:<module>'".
The module id is the CONTRACT's display name.  For ``baseline_context`` every
playbook family names it ``baseline_table`` or ``descriptive_table`` -- and
``AnalysisStep`` refuses any step carrying a ``table_one_spec`` unless its
expected outputs include ``table:table_one``.  There is one way to declare a
Table 1 and the hint named the other one.

MEASURED on h1_ventilation_survival, 2026-08-03 (``..._7e98a59_verify05``):
five planner attempts, five DISTINCT rejections, and two of them are this
loop::

    [2] PlannerArticleContractError: missing required article contract
        role(s): baseline_context. Required typed step examples:
        baseline_context -> 'table:baseline_table'.
    [0] ValidationError: table_one_spec requires expected output
        'table:table_one'

The Planner takes the advice, the schema refuses the result, and the attempt is
spent.  h1 has never produced a plan.

``table:table_one`` would have been credited all along -- ``_ROLE_ALIASES``
lists ``table_one`` for this role -- so the acceptance was never the problem.
Only the advice was.

WHY THIS TEST AND NOT A LITERAL.  ``SCHEMA_MANDATED_ROLE_PRODUCTS`` is a
one-entry map and a bare list of names is exactly the shape that rots.  So the
check below does not compare it to another list: it takes every required role
of every playbook family, asks for the hint the Planner would receive, and
feeds that product to the real ``AnalysisStep`` validator.  A hint the schema
cannot accept fails here, whichever role grows one next.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.planning.study_design_playbook import (  # noqa: E402
    _FAMILY_DISPLAY_MODULES,
)
from easyicu.research_agent.reporting.article_contract import (  # noqa: E402
    SCHEMA_MANDATED_ROLE_PRODUCTS,
    _ROLE_ALIASES,
    _plan_outputs_match_requirement,
    hinted_typed_products,
)
from easyicu.research_agent.schema import AnalysisStep  # noqa: E402

_TABLE_ONE_SPEC = {
    "group_by": "death",
    "group_levels": ["__easyicu_level_1__", "__easyicu_level_2__"],
    "variables": [
        {
            "name": "age",
            "variable_kind": "continuous",
            "summary": "median_iqr",
            "test": "mann_whitney_or_kruskal",
            "levels": [],
        }
    ],
}


def _table_one_step(product: str) -> AnalysisStep:
    """The only shape that can serve ``baseline_context``: a Table 1 step."""

    return AnalysisStep.model_validate(
        {
            "step_id": "02_baseline_table",
            "planned_analysis_role": "auxiliary",
            "intent": "Describe who contributed to the estimate.",
            "inputs": ["artifact:analysis_cohort", "death", "age"],
            "expected_outputs": [product],
            "method": "descriptive_summary",
            "table_one_spec": _TABLE_ONE_SPEC,
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


def test_the_hint_for_baseline_context_is_a_product_the_schema_accepts() -> None:
    """The property that was false, on the shape that actually serves the role."""

    hinted = hinted_typed_products("baseline_context", ["baseline_table"])
    assert hinted == ["table:table_one"]
    _table_one_step(hinted[0])  # must not raise


def test_the_display_id_alone_is_what_the_schema_refuses() -> None:
    """The recorded rejection, reproduced -- so this file is about a real wall."""

    with pytest.raises(ValueError, match="table:table_one"):
        _table_one_step("table:baseline_table")


def test_the_hinted_product_would_actually_have_credited_the_role() -> None:
    """Advice that is legal but uncreditable would be the same defect twice."""

    class _Requirement:
        role = "baseline_context"
        module_id = "baseline_table"
        acceptable_outputs = ()
        required = True

    hinted = hinted_typed_products("baseline_context", ["baseline_table"])
    assert _plan_outputs_match_requirement(set(hinted), _Requirement())
    # And the alias that makes it creditable is the reason -- recorded so a
    # future edit to _ROLE_ALIASES cannot quietly break the pairing.
    assert "table_one" in _ROLE_ALIASES["baseline_context"]


@pytest.mark.parametrize("family", sorted(_FAMILY_DISPLAY_MODULES))
def test_every_family_hints_baseline_context_the_same_legal_way(family: str) -> None:
    """All four playbooks name this module something the schema forbids.

    Measured: ``baseline_table`` x3 and ``descriptive_table`` x1, and neither is
    ``table_one``. Whichever family a task lands in, the hint has to survive the
    validator.
    """

    modules = [
        module
        for module in _FAMILY_DISPLAY_MODULES[family]
        if getattr(module, "role", None) == "baseline_context"
    ]
    if not modules:
        pytest.skip(f"{family} does not require baseline_context")
    hinted = hinted_typed_products(
        "baseline_context", [module.module_id for module in modules]
    )
    assert hinted, family
    for product in hinted:
        _table_one_step(product)  # the validator is the assertion


def test_a_role_with_no_schema_law_still_hints_its_own_modules() -> None:
    """The map is an exception list, not a replacement for the display ids."""

    assert "data_quality" not in SCHEMA_MANDATED_ROLE_PRODUCTS
    assert hinted_typed_products(
        "data_quality", ["measurement_process_audit", "missingness_profile"]
    ) == ["table:measurement_process_audit", "table:missingness_profile"]
