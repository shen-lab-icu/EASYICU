"""A renderer claims what it can read, not a remembered input tuple.

Every figure executor used to decide ownership with ``tuple(step.inputs) ==
<its constant>``.  Enumerated shape rather than capability, and it failed both
ways: order-sensitive against renderers that look every binding up by key, and
with no way for a renderer to say an input is dispensable.

The rule that replaced it must not become a loose matcher.  What it may not
do is locked here alongside what it now allows.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.execution.runners.figure_input_capability import (
    TypedInputCapability,
)
from easyicu.research_agent.schema import AnalysisStep, ArtifactConsumptionContract


A = "table:alpha"
B = "table:beta"
C = "table:gamma"


def _step(inputs: list[str], *, contracts: list[str] | None = None) -> AnalysisStep:
    keys = inputs if contracts is None else contracts
    return AnalysisStep(
        step_id="09_render",
        planned_analysis_role="auxiliary",
        intent="Render a declared figure.",
        method="visualization",
        inputs=list(inputs),
        expected_outputs=["figure:rendered"],
        input_consumption_contracts=[
            ArtifactConsumptionContract(input_key=key, mode="all_rows") for key in keys
        ],
    )


# --------------------------------------------------------------------------
# What the capability rule now allows.
# --------------------------------------------------------------------------


def test_the_same_inputs_in_either_order_are_the_same_request():
    capability = TypedInputCapability(required=frozenset({A, B}))

    assert capability.admits_step(_step([A, B]))
    assert capability.admits_step(_step([B, A]))


def test_an_optional_input_may_be_absent():
    capability = TypedInputCapability(required=frozenset({A}), optional=frozenset({B}))

    assert capability.admits_step(_step([A]))
    assert capability.admits_step(_step([A, B]))


# --------------------------------------------------------------------------
# What it still refuses.  Each of these would publish a figure that answers a
# different question than the plan promised.
# --------------------------------------------------------------------------


def test_an_input_the_renderer_cannot_read_is_refused_not_ignored():
    """A step naming an extra table is asking for a figure that reads it."""

    capability = TypedInputCapability(required=frozenset({A, B}))

    assert not capability.admits_step(_step([A, B, C]))


def test_a_missing_required_input_is_refused():
    capability = TypedInputCapability(required=frozenset({A, B}))

    assert not capability.admits_step(_step([A]))


def test_an_optional_set_does_not_make_the_required_one_negotiable():
    capability = TypedInputCapability(required=frozenset({A}), optional=frozenset({B}))

    assert not capability.admits_step(_step([B]))


def test_a_duplicate_input_is_refused_rather_than_deduplicated():
    """Two identical keys cannot both bind; taking the set hides that."""

    capability = TypedInputCapability(required=frozenset({A}))

    assert not capability.admits([A, A])


def test_a_step_declaring_nothing_is_refused():
    capability = TypedInputCapability(required=frozenset({A}))

    assert not capability.admits([])


def test_a_blank_input_is_refused():
    capability = TypedInputCapability(required=frozenset({A}))

    assert not capability.admits([A, "   "])


# --------------------------------------------------------------------------
# Inputs and their consumption contracts must be the same decision.
#
# ``AnalysisStep`` already refuses a mismatch at validation, so these cannot be
# reached through a validated step.  The rule keeps its own check because it is
# also asked about steps assembled in code, and a capability that trusted the
# caller to have validated first would be the weakest link in the chain.
# --------------------------------------------------------------------------


def _contracts(keys):
    return [ArtifactConsumptionContract(input_key=key, mode="all_rows") for key in keys]


class _Step:
    """A step-shaped object that skipped schema validation."""

    def __init__(self, inputs, contracts):
        self.inputs = inputs
        self.input_consumption_contracts = contracts


def test_the_schema_refuses_a_contract_for_an_undeclared_input():
    with pytest.raises(ValueError, match="exact inputs on the same step"):
        _step([A], contracts=[A, B])


def test_the_capability_also_refuses_a_contract_for_an_undeclared_input():
    capability = TypedInputCapability(required=frozenset({A}), optional=frozenset({B}))
    step = _Step(
        [A],
        [
            ArtifactConsumptionContract(input_key=A, mode="all_rows"),
            ArtifactConsumptionContract(input_key=B, mode="all_rows"),
        ],
    )

    assert not capability.admits_step(step)


def test_the_capability_refuses_a_declared_input_with_no_contract():
    capability = TypedInputCapability(required=frozenset({A}), optional=frozenset({B}))
    step = _Step(
        [A, B],
        [ArtifactConsumptionContract(input_key=A, mode="all_rows")],
    )

    assert not capability.admits_step(step)


def test_a_partially_consumed_input_is_refused():
    capability = TypedInputCapability(required=frozenset({A}))
    step = _Step(
        [A],
        [
            ArtifactConsumptionContract(
                input_key=A,
                mode="one_per_role",
                role_column="row_type",
                expected_roles=["primary"],
            )
        ],
    )

    assert not capability.admits_step(step)


# --------------------------------------------------------------------------
# The declaration itself has to be coherent.
# --------------------------------------------------------------------------


def test_a_capability_requiring_nothing_is_rejected_at_construction():
    with pytest.raises(ValueError, match="at least one input"):
        TypedInputCapability(required=frozenset())


def test_an_input_cannot_be_both_required_and_optional():
    with pytest.raises(ValueError, match="both required and optional"):
        TypedInputCapability(required=frozenset({A}), optional=frozenset({A}))


def test_a_bare_column_name_is_not_a_typed_input():
    """Raw columns come from the cohort, not from a parent product binding."""

    with pytest.raises(ValueError, match="typed keys"):
        TypedInputCapability(required=frozenset({"age"}))


# --------------------------------------------------------------------------
# The live renderers.
# --------------------------------------------------------------------------


def test_every_figure_renderer_requires_each_of_its_inputs_today():
    """Optionality is a claim about the code, so it may not be assumed.

    Each renderer indexes every binding it declares while rendering; marking
    one optional would turn a clean decline into a failure inside the sandbox.
    This asserts the current, honest state rather than a target.
    """

    from easyicu.research_agent.execution.runners import (
        exposure_outcome_distribution_render as distribution,
        missingness_measurement_figure_executor as missingness,
        prevalence_mortality_figure_executor as mortality,
        prevalence_outcome_figure_executor as prevalence,
    )

    capabilities = {
        "prevalence_outcome": prevalence.PREVALENCE_OUTCOME_FIGURE_CAPABILITY,
        "prevalence_mortality": mortality.PREVALENCE_MORTALITY_FIGURE_CAPABILITY,
        "missingness_measurement": (
            missingness.MISSINGNESS_MEASUREMENT_FIGURE_CAPABILITY
        ),
        "exposure_outcome_distribution": (
            distribution.EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_CAPABILITY
        ),
    }

    for name, capability in capabilities.items():
        assert capability.required, name
        assert capability.optional == frozenset(), name


# --------------------------------------------------------------------------
# A product with no rows has no row-consumption decision to declare.
#
# Measured 2026-07-30 across today's plans: on visualization steps, 0 of 30
# declared ``table:`` inputs lack a contract and 0 of 19 declared
# ``statistic:`` inputs have one.  The Planner is entirely consistent; the
# host's demand was the incoherent half, and it silently cost 7 of 21 figure
# steps their deterministic owner.
# --------------------------------------------------------------------------

S = "statistic:primary_or"
T = "statistic:complete_case_n"


def test_a_statistic_input_needs_no_row_consumption_contract():
    """``mode="all_rows"`` over one finite number is a category error."""

    capability = TypedInputCapability(required=frozenset({A}), optional=frozenset({S}))

    assert capability.admits_step(_step([A, S], contracts=[A]))


def test_the_schema_is_the_authority_on_which_inputs_owe_a_contract():
    """Not "optional to declare" -- a contract for a statistic is invalid.

    ``AnalysisStep`` refuses it outright, so the capability agreeing with the
    schema is the whole point: it must not accept a step the schema rejects
    any more than it may reject one the schema accepts.
    """

    with pytest.raises(ValueError, match="cover every"):
        _step([A, S], contracts=[A, S])

    capability = TypedInputCapability(required=frozenset({A}), optional=frozenset({S}))
    assert not capability.admits_step(_Step([A, S], _contracts([A, S])))


def test_a_table_beside_an_exempt_statistic_still_needs_its_contract():
    """The exemption is per input, not a licence to drop the whole check."""

    capability = TypedInputCapability(
        required=frozenset({A}), optional=frozenset({B, S})
    )

    assert not capability.admits_step(_Step([A, B, S], _contracts([A])))


def test_the_real_recorded_figure_shape_is_admitted():
    """Two tables with contracts, three statistics without -- 7 real steps."""

    capability = TypedInputCapability(
        required=frozenset({"table:robustness_matrix"}),
        optional=frozenset(
            {"table:robustness_summary", S, T, "statistic:robustness_summary"}
        ),
    )
    step = _step(
        [
            "table:robustness_matrix",
            "table:robustness_summary",
            S,
            T,
            "statistic:robustness_summary",
        ],
        contracts=["table:robustness_matrix", "table:robustness_summary"],
    )

    assert capability.admits_step(step)


def test_an_unfamiliar_kind_owes_no_contract_because_only_tables_do():
    """The rule names what DOES owe one, so a new kind cannot silently owe it.

    Stated positively on purpose: an exemption list would have to grow every
    time a kind appears, and the one that was forgotten is the one that
    silently costs a step its owner.
    """

    capability = TypedInputCapability(
        required=frozenset({A}), optional=frozenset({"artifact:analysis_cohort"})
    )

    assert capability.admits_step(
        _Step([A, "artifact:analysis_cohort"], _contracts([A]))
    )


def test_a_statistic_contract_pointing_at_an_undeclared_input_is_still_refused():
    """Exempting the requirement must not also exempt the coherence check."""

    capability = TypedInputCapability(required=frozenset({A}), optional=frozenset({S}))

    assert not capability.admits_step(_Step([A, S], _contracts([A, "statistic:other"])))


def test_a_duplicate_contract_key_is_still_refused():
    """Two contracts for one input cannot both be the decision."""

    capability = TypedInputCapability(required=frozenset({A}), optional=frozenset({S}))

    assert not capability.admits_step(_Step([A, S], _contracts([A, A])))


def test_a_statistic_carrying_a_contract_is_refused_whatever_its_shape():
    """It owes none, so any contract for it is one contract too many."""

    capability = TypedInputCapability(required=frozenset({A}), optional=frozenset({S}))
    step = _Step(
        [A, S],
        [
            *_contracts([A]),
            ArtifactConsumptionContract(
                input_key=S,
                mode="one_per_role",
                role_column="row_type",
                expected_roles=["primary"],
            ),
        ],
    )

    assert not capability.admits_step(step)


def test_a_table_with_no_contract_at_all_is_refused():
    """The one shape that a "nobody owes a contract" rule would wave through.

    Every other negative here has a non-empty contract set, so an empty
    requirement still trips the equality check and the test passes for the
    wrong reason.  This is the case that fails only if tables really do owe
    one.
    """

    capability = TypedInputCapability(required=frozenset({A}))

    assert not capability.admits_step(_Step([A], []))
