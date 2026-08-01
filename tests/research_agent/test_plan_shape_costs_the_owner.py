"""How the Planner packages products decides whether a host owner can run.

fresh20 is the first recorded run whose Planner carried the typed declarations
at all, and it separates the two failure modes cleanly:

* ``04_missingness_event_timing_audit`` declared ``measurement_audit_spec`` and
  the host owned it -- 0 provider calls, ``ok``.  The same step in fresh19 had
  no declaration, no owner, and was blocked by the concept audit.
* ``05_primary_adjusted_association`` declared exactly the covariates the
  association owner needs, and still lost that owner, because the same step
  also declared ``figure:primary_adjusted_association``.  The owner produces
  one table and no figure; claiming a step it cannot finish would fail for a
  missing product, so declining is correct and the plan shape is the defect.
* ``07_robustness_replay`` declared the canonical six-product robustness
  bundle, which names ``robustness_summary`` under both ``table:`` and
  ``statistic:``.  No replay declaration can back that: one output is one
  answer, and the pre-declaration code satisfied both from a single CSV --
  handing a reader who asked for a number a table instead.

Neither is fixed by loosening a host contract.  Both are fixed by telling the
Planner the shape the host can execute, which is what the two prompt sentences
under test here do.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    ADJUSTED_ASSOCIATION_OUTPUT,
    adjusted_association_executor_owns_step,
)
from easyicu.research_agent.execution.runners.deterministic_missingness import (
    missingness_audit_executor_owns_step,
)
from easyicu.research_agent.execution.runners.deterministic_robustness import (
    robustness_replay_spec_is_emittable,
)
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "real_plan_steps_fresh17_fresh19.json"


def _real_step(label: str, step_id: str) -> dict:
    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    plan = next(e for e in document["plans"] if e["label"] == label)["plan"]
    return next(s for s in plan["steps"] if s["step_id"] == step_id)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Is the laboratory signal associated with death?",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_patients=6,
            n_stays=6,
        ),
        variables=[
            ConceptDescriptor(name="lab_max", role="lab", dtype="float64"),
            ConceptDescriptor(name="death", role="outcome", dtype="int64"),
        ],
        primary_exposure="lab",
        target_outcome="death",
    )


# --------------------------------------------------------------------------
# the declaration that did work


def test_the_real_audit_step_the_planner_declared_is_owned() -> None:
    """End-to-end on a real Planner-written declaration, not a hand-built one.

    fresh19 planned this same audit without the spec and no owner claimed it;
    fresh20's Planner wrote ``measurement_audit_spec`` and the host ran it with
    zero provider calls.
    """

    step = AnalysisStep.model_validate(
        _real_step("fresh20", "04_missingness_event_timing_audit")
    )

    assert step.measurement_audit_spec is not None
    assert [item.audit for item in step.measurement_audit_spec.products] == [
        "measurement_missingness",
        "event_timing",
        "measurement_process",
        "component_completeness",
    ]
    # Two of these four product names are absent from the legacy alias table,
    # so the name-driven path would have declined this step exactly as it
    # declined fresh19's.
    assert missingness_audit_executor_owns_step(step) is True


# --------------------------------------------------------------------------
# bundling a figure into the result step


def test_bundling_the_figure_costs_the_association_step_its_owner() -> None:
    """The load-bearing one: same science, same covariates, one extra product.

    Splitting the figure out is the only edit; nothing about the model,
    the adjustment set or the inputs changes.
    """

    planned = _real_step("fresh20", "05_primary_adjusted_association")
    assert planned["expected_outputs"] == [
        ADJUSTED_ASSOCIATION_OUTPUT,
        "figure:primary_adjusted_association",
    ]
    requirement = planned["model_requirements"][0]
    assert requirement["covariates"] == ["age", "sex", "charlson_first"]

    as_planned = AnalysisStep.model_validate(planned)
    assert adjusted_association_executor_owns_step(as_planned) is False

    split = dict(planned)
    split["expected_outputs"] = [ADJUSTED_ASSOCIATION_OUTPUT]
    assert (
        adjusted_association_executor_owns_step(AnalysisStep.model_validate(split))
        is True
    )


def test_the_owner_declines_rather_than_claiming_a_figure_it_cannot_render() -> None:
    """Declining is the correct behaviour, so the fix belongs in the plan.

    An owner that claimed the bundle would fail the step for a missing product
    -- strictly worse than never claiming it.
    """

    planned = _real_step("fresh20", "05_primary_adjusted_association")
    table_only = dict(planned)
    table_only["expected_outputs"] = [ADJUSTED_ASSOCIATION_OUTPUT]

    assert (
        adjusted_association_executor_owns_step(AnalysisStep.model_validate(table_only))
        is True
    )
    for extra in (
        "figure:primary_adjusted_association",
        "figure:anything_else",
    ):
        bundled = dict(planned)
        bundled["expected_outputs"] = [ADJUSTED_ASSOCIATION_OUTPUT, extra]
        assert (
            adjusted_association_executor_owns_step(
                AnalysisStep.model_validate(bundled)
            )
            is False
        )


# --------------------------------------------------------------------------
# one name, two kinds


def test_the_real_robustness_bundle_cannot_carry_a_replay_declaration() -> None:
    """``robustness_summary`` is declared as both a table and a statistic."""

    planned = _real_step("fresh20", "07_robustness_replay")
    assert "table:robustness_summary" in planned["expected_outputs"]
    assert "statistic:robustness_summary" in planned["expected_outputs"]

    payload = dict(planned)
    payload["robustness_replay_spec"] = {
        "products": [
            {"product_id": "primary_or", "output": "primary_effect"},
            {"product_id": "complete_case_n", "output": "complete_case_n"},
            {"product_id": "robustness_summary", "output": "robustness_summary"},
            {
                "product_id": "missingness_strategy_notes",
                "output": "missingness_strategy_notes",
            },
            {"product_id": "robustness_matrix", "output": "robustness_matrix"},
        ]
    }
    # The plan must stay readable: this exact declaration, rejected in the
    # schema, is what killed fresh21 at re-parse of its own sealed artifact.
    step = AnalysisStep.model_validate(payload)
    assert robustness_replay_spec_is_emittable(step) is False


def test_dropping_the_duplicate_kind_makes_the_same_step_emittable() -> None:
    planned = _real_step("fresh20", "07_robustness_replay")
    payload = dict(planned)
    payload["expected_outputs"] = [
        item
        for item in planned["expected_outputs"]
        if item != "statistic:robustness_summary"
    ]
    payload["robustness_replay_spec"] = {
        "products": [
            {"product_id": "primary_or", "output": "primary_effect"},
            {"product_id": "complete_case_n", "output": "complete_case_n"},
            {"product_id": "robustness_summary", "output": "robustness_summary"},
            {
                "product_id": "missingness_strategy_notes",
                "output": "missingness_strategy_notes",
            },
            {"product_id": "robustness_matrix", "output": "robustness_matrix"},
        ]
    }

    step = AnalysisStep.model_validate(payload)
    assert robustness_replay_spec_is_emittable(step) is True


# --------------------------------------------------------------------------
# a plan the host wrote must stay a plan the host can read


def test_the_whole_fresh21_plan_the_host_sealed_can_be_read_back() -> None:
    """The regression for the run that died before its first step.

    fresh21's Planner did carry ``robustness_replay_spec``, on a step that also
    declared ``robustness_summary`` under two kinds.  A schema validator
    refused that, so the plan the host had already written and registered as
    evidence could no longer be parsed: the plan-authority resolver treats an
    unreadable record as absent, found no candidate, and the item ended with
    ``current analysis plan is not bound to immutable EvidenceStore
    authority`` -- 0 steps executed, and a message pointing nowhere near the
    cause.

    Anything that decides whether the plan can be *read* must therefore stay
    independent of whether anything can *execute* it.
    """

    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    payload = next(e for e in document["plans"] if e["label"] == "fresh21")["plan"]

    steps = [AnalysisStep.model_validate(item) for item in payload["steps"]]

    declaring = {
        step.step_id: step
        for step in steps
        if step.robustness_replay_spec is not None
        or step.measurement_audit_spec is not None
    }
    assert set(declaring) == {
        "05_missingness_event_timing_audit",
        "08_standard_robustness_sensitivity",
    }
    # Readable, and each answered on capability alone.
    assert (
        missingness_audit_executor_owns_step(
            declaring["05_missingness_event_timing_audit"]
        )
        is True
    )
    assert (
        robustness_replay_spec_is_emittable(
            declaring["08_standard_robustness_sensitivity"]
        )
        is False
    )


# --------------------------------------------------------------------------
# the Planner has to be told, or nothing above ever happens


def test_the_planner_prompt_states_both_shape_rules() -> None:
    """A contract the prompt never mentions is a contract no plan will meet.

    Built through the real prompt function, so a rule written into a helper
    nobody calls still fails here.
    """

    from easyicu.research_agent.agents.core import _build_planner_user_prompt

    prompt = _build_planner_user_prompt(_context())

    assert "A figure is its own step" in prompt
    assert "loses the deterministic owner" in prompt
    assert "never declare the same name under two kinds" in prompt
    assert "`table:x` together with `statistic:x`" in prompt


# --------------------------------------------------------------------------
# answering half the question must not cost more than saying nothing


def _robustness_step(spec: dict | None) -> AnalysisStep:
    """A real recorded robustness shape, with the spec swapped in or out.

    The products are the six-product bundle the Planner really emits, minus the
    ``statistic:robustness_summary`` half of the collision, so the only thing
    varying between the cases below is the declaration itself.
    """

    payload = {
        "step_id": "09_robustness_replay",
        "planned_analysis_role": "sensitivity",
        "intent": "Re-estimate the locked robustness grid without changing the estimand.",
        "inputs": ["artifact:analysis_cohort"],
        "expected_outputs": [
            "table:robustness_matrix",
            "table:robustness_summary",
            "statistic:primary_or",
            "statistic:complete_case_n",
            "log:missingness_strategy_notes",
        ],
        "method": "robustness_sensitivity",
    }
    if spec is not None:
        payload["robustness_replay_spec"] = spec
    return AnalysisStep.model_validate(payload)


_PARTIAL_SPEC = {
    "schema_version": "easyicu.robustness_replay/1",
    "products": [
        {"product_id": "robustness_matrix", "output": "robustness_matrix"},
        {"product_id": "robustness_summary", "output": "robustness_summary"},
    ],
}


def test_a_partial_declaration_does_not_cost_the_step_its_owner() -> None:
    """Declaring half the products must be no worse than declaring none.

    Until 2026-07-31 the spec branch returned instead of falling through, so a
    step that filled the field partially was refused outright while the very
    same step with the field left empty was claimed. Measured over the recorded
    plans: 10 undeclared steps claimed, 8 partially-declared ones refused, and
    every one of those 8 would have been claimed had the Planner said nothing.
    The host was charging the Planner for trying.
    """

    from easyicu.research_agent.execution.phase import (
        _robustness_sensitivity_runner_owns_step as owns,
    )
    from easyicu.research_agent.execution.runners.deterministic_robustness import (
        robustness_replay_declaration_verdict,
    )

    undeclared = _robustness_step(None)
    partial = _robustness_step(_PARTIAL_SPEC)

    # The partial spec really is unemittable -- otherwise this test would pass
    # for the wrong reason, on a declaration that was complete all along.
    assert robustness_replay_spec_is_emittable(partial) is False

    claimed_undeclared = owns(
        str(undeclared.method),
        undeclared.step_id,
        undeclared.expected_outputs,
        step=undeclared,
    )
    claimed_partial = owns(
        str(partial.method),
        partial.step_id,
        partial.expected_outputs,
        step=partial,
    )
    assert claimed_undeclared is True, "the fallback path must claim the bare shape"
    assert claimed_partial is True, "answering partially must not remove the owner"

    # ...and the gap is still reported, so the Planner is still asked to close
    # it. Not punishing the step is not the same as pretending it is complete.
    verdict = robustness_replay_declaration_verdict(partial)
    assert verdict.claimed is False
    # One entry per unbacked product since 2026-08-01. The field path alone was
    # what the plan already had, so the replan had nothing to act on.
    assert verdict.missing_declarations
    assert all(
        name.startswith("robustness_replay_spec.products[")
        for name in verdict.missing_declarations
    ), verdict.missing_declarations
    assert any("primary_or" in name for name in verdict.missing_declarations)
    assert "primary_or" in verdict.reason


def test_an_emittable_declaration_still_claims_without_the_label() -> None:
    """The declaration path is what the label path exists to stop needing."""

    from easyicu.research_agent.execution.phase import (
        _robustness_sensitivity_runner_owns_step as owns,
    )

    full = _robustness_step(
        {
            "schema_version": "easyicu.robustness_replay/1",
            "products": [
                {"product_id": "robustness_matrix", "output": "robustness_matrix"},
                {"product_id": "robustness_summary", "output": "robustness_summary"},
                {"product_id": "primary_or", "output": "primary_effect"},
                {"product_id": "complete_case_n", "output": "complete_case_n"},
                {
                    "product_id": "missingness_strategy_notes",
                    "output": "missingness_strategy_notes",
                },
            ],
        }
    )
    assert robustness_replay_spec_is_emittable(full) is True
    # A method label no allowlist contains: the declaration alone must carry it.
    assert (
        owns(
            "a_method_no_allowlist_has_ever_seen",
            full.step_id,
            full.expected_outputs,
            step=full,
        )
        is True
    )


def test_a_step_promising_a_figure_alongside_its_tables_is_still_refused() -> None:
    """Falling through must not reopen the one gate that was never about labels.

    There is no longer an explicit `figure:` guard to test -- it was deleted on
    2026-07-31 once measured unreachable: `figure` is not one of the three
    auxiliary output kinds, so the product check refuses any step promising one,
    and no replay output names a figure, so the declaration path refuses it too.
    Deleting a guard is only safe while the property it claimed still holds, so
    this test asserts the PROPERTY over the shapes that reach each path, and
    fails if either structural rule is ever relaxed.

    A first version promised the figure alone and passed on the product check --
    it survived deleting the guard, which is how the guard was found to be dead.
    """

    from easyicu.research_agent.execution.phase import (
        _robustness_sensitivity_runner_owns_step as owns,
    )

    step = AnalysisStep.model_validate(
        {
            "step_id": "07_robustness_replay_and_figure",
            "planned_analysis_role": "sensitivity",
            "intent": "Replay the locked grid and draw it.",
            "inputs": ["artifact:analysis_cohort"],
            "expected_outputs": [
                "table:robustness_matrix",
                "table:robustness_summary",
                # NOT `figure:robustness_plot`: that bare name is not one of
                # the runner's products, so the product check alone would
                # refuse it and this test would pass without ever exercising
                # the kind rule. `figure:robustness_summary` is the shape 5
                # recorded steps really use, and its bare name IS supported --
                # so only `figure` being a non-auxiliary kind refuses it.
                "figure:robustness_summary",
            ],
            "method": "robustness_sensitivity",
        }
    )
    assert (
        owns(str(step.method), step.step_id, step.expected_outputs, step=step) is False
    )
    # ...and it is the FIGURE that refuses it: the same step without one is claimed.
    without_figure = step.model_copy(
        update={
            "expected_outputs": [
                "table:robustness_matrix",
                "table:robustness_summary",
            ]
        }
    )
    assert (
        owns(
            str(without_figure.method),
            without_figure.step_id,
            without_figure.expected_outputs,
            step=without_figure,
        )
        is True
    )

    # The declaration path, which the product check never sees: an otherwise
    # complete spec cannot make a promised figure emittable either.
    declaring = AnalysisStep.model_validate(
        {
            **step.model_dump(),
            "robustness_replay_spec": {
                "schema_version": "easyicu.robustness_replay/1",
                "products": [
                    {"product_id": "robustness_matrix", "output": "robustness_matrix"},
                    {
                        "product_id": "robustness_summary",
                        "output": "robustness_summary",
                    },
                ],
            },
        }
    )
    assert robustness_replay_spec_is_emittable(declaring) is False
    assert (
        owns(
            "a_method_no_allowlist_has_ever_seen",
            declaring.step_id,
            declaring.expected_outputs,
            step=declaring,
        )
        is False
    )


# --------------------------------------------------------------------------
# one model per step -- the rule the host enforced but never published


def test_bundling_a_second_model_costs_the_step_its_owner() -> None:
    """The same shape as the bundled figure, one layer down, and unpublished.

    The association owner refuses any step whose roster carries more than one
    entry -- it writes one estimate row and one contract. Measured 2026-07-31
    over the recorded plans: 8 steps declare two or more, and 7 of the 8 also
    omit their covariates so they were refused twice over. The eighth is real
    and current: E3's ``07_primary_adjusted_association_models`` in the newest
    nine-task run declares its covariates properly and is refused for the
    roster alone. Splitting it recovers the paper's primary estimate -- the
    first entry, on its own, is claimed.
    """

    primary = {
        "requirement_id": "primary_mortality_by_stage",
        "outcome": "death",
        "outcome_type": "binary",
        "method_family": "logistic_regression",
        "exposure_source": "stage_max",
        "analysis_role": "primary",
        "analysis_set": "complete_case",
        "required_for_step_success": True,
        "covariates": ["age", "sex"],
    }
    secondary = {
        **primary,
        "requirement_id": "secondary_stay_length_by_stage",
        "outcome": "stay_length",
        "outcome_type": "continuous",
        "method_family": "linear_regression",
        "analysis_role": "secondary",
    }
    payload = {
        "step_id": "07_primary_adjusted_association_models",
        "planned_analysis_role": "primary",
        "intent": "Estimate the adjusted association across the exposure gradient.",
        "inputs": ["artifact:analysis_cohort"],
        "expected_outputs": [ADJUSTED_ASSOCIATION_OUTPUT],
        "method": "adjusted_association_models",
    }

    bundled = AnalysisStep.model_validate(
        {**payload, "model_requirements": [primary, secondary]}
    )
    assert adjusted_association_executor_owns_step(bundled) is False

    # The only edit is removing the second entry; the science of the first is
    # untouched, and it is the paper's primary estimate.
    split = AnalysisStep.model_validate({**payload, "model_requirements": [primary]})
    assert adjusted_association_executor_owns_step(split) is True


def test_the_planner_prompt_states_the_one_model_per_step_rule() -> None:
    """A rule enforced but never published is a rule no plan can meet.

    The roster paragraph used to say "record each pre-specified estimand/model
    in the roster" while the owner refused every roster with more than one --
    the host asking for exactly what it would then refuse.
    """

    from easyicu.research_agent.agents.core import _build_planner_user_prompt

    prompt = _build_planner_user_prompt(_context())

    assert "ONE MODEL PER STEP" in prompt
    assert "declare exactly one entry" in prompt
    assert "is its own step with its own roster entry" in prompt
    # ...and the sentence that contradicted it is gone.
    assert "record each pre-specified estimand/model in the roster" not in prompt
