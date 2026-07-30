"""A pre-specified sensitivity grid is declared by the Planner, or not reported.

``deterministic_robustness`` already exists and is already reachable before the
Coder is asked.  What stops it is the missing ``robustness_replay_spec``:
measured 2026-07-30 over the recorded plans (623 distinct step shapes), 20 steps
promise a product this replay is the registered emitter of and declare no spec
at all, 4 declare a spec that does not back every product, and **0** reach the
runner.  Every one of the 20 goes to the Coder to invent a specification grid.

Which sensitivity analyses a paper reports is a pre-specified choice.  An
unspecified grid is not a weaker sensitivity analysis; it is an undeclared one.

The gap is keyed on the products the step promises, never on its ``method``
label -- ``robustness_replay_spec_is_emittable`` settled that and its reasoning
is not reopened here: over the recorded corpus a method allowlist turned away
182 robustness steps, 62 of them for saying ``prespecified_sensitivity_analysis``,
and widening it would hand a differently-scienced analysis to a runner that
replays an already-locked grid.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.execution.runners.deterministic_robustness import (
    ROBUSTNESS_REPLAY_ANALYSIS_KIND,
    ROBUSTNESS_REPLAY_OUTPUT_FILES,
    ROBUSTNESS_REPLAY_OUTPUT_KINDS,
    robustness_replay_declaration_verdict,
)
from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.gates.owner_declaration import (
    execution_declaration_refusal,
    owner_declaration_plan_findings,
)
from easyicu.research_agent.schema import AnalysisStep


def _step(outputs, **kwargs) -> AnalysisStep:
    return AnalysisStep(
        step_id="07_robustness_analysis",
        planned_analysis_role="auxiliary",
        intent="Replay the pre-specified robustness grid.",
        inputs=["artifact:analysis_cohort"],
        expected_outputs=list(outputs),
        method="robustness_sensitivity",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The gap
# ---------------------------------------------------------------------------


def test_a_step_promising_this_replays_products_without_a_spec_is_a_gap():
    verdict = robustness_replay_declaration_verdict(
        _step(["table:robustness_summary", "table:robustness_matrix"])
    )
    assert verdict.missing_declarations == ("robustness_replay_spec",)
    assert verdict.analysis_kind == ROBUSTNESS_REPLAY_ANALYSIS_KIND


def test_the_reason_names_the_products_that_triggered_it():
    """A reason that does not say *why this step* is unactionable.

    The Planner is being asked to declare a grid; it has to be able to see
    which promise put it on the hook.
    """

    verdict = robustness_replay_declaration_verdict(
        _step(["table:robustness_summary", "statistic:complete_case_n"])
    )
    assert "table:robustness_summary" in verdict.reason
    assert "statistic:complete_case_n" in verdict.reason


def test_the_gate_reports_it_and_the_execution_refusal_acts_on_it():
    """The two halves that make the gap cost something, end to end."""

    step = _step(["table:robustness_matrix"])

    class _Plan:
        steps = (step,)

    findings = owner_declaration_plan_findings(plan=_Plan())
    assert [f.detail["analysis_kind"] for f in findings] == [
        ROBUSTNESS_REPLAY_ANALYSIS_KIND
    ]
    assert findings[0].detail["missing_declarations"] == ["robustness_replay_spec"]

    trace: list = []
    chosen = select_standard_executor(step, plan=_Plan(), trace=trace)
    assert chosen is None
    refused = execution_declaration_refusal(claimed_by=chosen, trace=trace)
    assert [c.analysis_kind for c in refused] == [ROBUSTNESS_REPLAY_ANALYSIS_KIND]


# ---------------------------------------------------------------------------
# Negative controls -- each one is a measured over-claim, not a hypothetical
# ---------------------------------------------------------------------------


def test_a_figure_only_step_is_not_a_gap():
    """MEASURED over-claim: ``figure:robustness_summary`` SOLO-triggers 5 steps.

    This replay writes tables, statistics and logs.  Keying the gap on a
    product name without its kind would put five figure steps on the hook for a
    declaration that could not make this runner emit them.
    """

    assert "figure" not in ROBUSTNESS_REPLAY_OUTPUT_KINDS
    verdict = robustness_replay_declaration_verdict(
        _step(["figure:robustness_summary"])
    )
    assert verdict.missing_declarations == ()


def _spec(*pairs: tuple[str, str]) -> dict:
    return {
        "schema_version": "easyicu.robustness_replay/1",
        "products": [
            {"product_id": product_id, "output": output} for product_id, output in pairs
        ],
    }


def test_a_spec_that_backs_every_promised_product_is_not_a_gap():
    """The Planner answered completely; there is nothing to ask it for."""

    step = _step(
        ["table:robustness_matrix"],
        robustness_replay_spec=_spec(("robustness_matrix", "robustness_matrix")),
    )
    verdict = robustness_replay_declaration_verdict(step)

    assert verdict.missing_declarations == ()
    assert "backs every product" in verdict.reason


def test_a_spec_that_names_no_entry_for_a_promised_product_is_a_gap():
    """Measured 2026-07-30: 5 of the 6 robustness steps carrying a spec today.

    The earlier design left this silent on the reasoning that the Planner had
    answered.  It had answered about two products and promised four, and every
    one of those steps went to the Coder with the host never saying why.
    """

    step = _step(
        ["table:robustness_matrix", "statistic:complete_case_n"],
        robustness_replay_spec=_spec(("robustness_matrix", "robustness_matrix")),
    )
    verdict = robustness_replay_declaration_verdict(step)

    assert verdict.missing_declarations == ("robustness_replay_spec.products",)
    assert "'complete_case_n'" in verdict.reason


def test_the_gap_names_every_unmapped_product_not_just_the_first():
    """One replan round only closes a gap it was told the whole of."""

    step = _step(
        [
            "table:robustness_matrix",
            "statistic:complete_case_n",
            "log:missingness_strategy_notes",
        ],
        robustness_replay_spec=_spec(("robustness_matrix", "robustness_matrix")),
    )
    verdict = robustness_replay_declaration_verdict(step)

    assert "'complete_case_n'" in verdict.reason
    assert "'missingness_strategy_notes'" in verdict.reason


def test_a_spec_naming_every_product_is_not_asked_for_more_entries():
    """The duplicate-kind step: adding entries could not close it.

    ``table:robustness_summary`` plus ``statistic:robustness_summary`` resolve
    to one bare name, so the spec already names every promised product and is
    still not emittable.  Demanding another entry would be work that leaves the
    step exactly as unowned; the product-promise gate owns this shape.
    """

    step = _step(
        ["table:robustness_summary", "statistic:robustness_summary"],
        robustness_replay_spec=_spec(("robustness_summary", "robustness_summary")),
    )
    verdict = robustness_replay_declaration_verdict(step)

    assert verdict.missing_declarations == ()
    assert "how the products are promised" in verdict.reason


def test_a_step_promising_none_of_this_replays_products_is_not_a_gap():
    verdict = robustness_replay_declaration_verdict(
        _step(["table:adjusted_association_estimates"])
    )
    assert verdict.missing_declarations == ()


@pytest.mark.parametrize("kind", sorted(ROBUSTNESS_REPLAY_OUTPUT_KINDS))
def test_every_kind_this_replay_writes_can_raise_the_gap(kind: str):
    verdict = robustness_replay_declaration_verdict(
        _step([f"{kind}:robustness_summary"])
    )
    assert verdict.missing_declarations == ("robustness_replay_spec",)


# ---------------------------------------------------------------------------
# The boundary this owner deliberately does not cross
# ---------------------------------------------------------------------------


def test_this_owner_never_claims_a_step():
    """It reports the gap; it does not take over routing.

    No recorded step carries an emittable spec, so a claim path here could not
    be exercised by any real plan.  The runner is already reachable as a
    preflight substitute before the Coder is asked; moving that routing is a
    separate, characterised change.
    """

    for outputs in (
        ["table:robustness_summary"],
        ["table:robustness_matrix", "statistic:complete_case_n"],
        ["figure:robustness_summary"],
    ):
        step = _step(outputs)

        class _Plan:
            steps = (step,)

        assert select_standard_executor(step, plan=_Plan()) is None


def test_the_gap_is_not_keyed_on_the_method_label():
    """Two steps, same promise, different labels -- the same verdict.

    A method allowlist turned away 62 recorded steps for saying
    ``prespecified_sensitivity_analysis``.  Reintroducing one here would put
    the host back to guessing which science a label means.
    """

    verdicts = [
        robustness_replay_declaration_verdict(
            AnalysisStep(
                step_id="07_robustness_analysis",
                planned_analysis_role="auxiliary",
                intent="Replay the pre-specified robustness grid.",
                inputs=["artifact:analysis_cohort"],
                expected_outputs=["table:robustness_summary"],
                method=label,
            )
        )
        for label in (
            "robustness_sensitivity",
            "prespecified_sensitivity_analysis",
            "scientific_sensitivity_analysis",
            "something_nobody_registered",
        )
    ]
    assert {v.missing_declarations for v in verdicts} == {("robustness_replay_spec",)}


def test_the_registered_products_are_read_from_the_runners_own_table():
    """A second copy of this list is how the gap and the runner drift apart."""

    assert "robustness_matrix" in ROBUSTNESS_REPLAY_OUTPUT_FILES
    unregistered = _step(["table:not_a_product_this_replay_emits"])
    assert (
        robustness_replay_declaration_verdict(unregistered).missing_declarations == ()
    )
