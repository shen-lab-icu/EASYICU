"""A step the host can compute deterministically was never asked to say so.

``exposure_outcome_distribution_executor`` computes prevalence and outcome by
exposure level -- n, events, rate, interval -- against a fixed contract, and has
a matching renderer.  It owns a step only when the Planner declares
``exposure_outcome_distribution_spec`` and promises its product.

The Planner does that, and often: measured over every recorded run, 33 steps
promise ``table:exposure_outcome_distribution``, 29 of them declare the spec, 28
are claimed by this owner, and 29 of 33 pass (88%).

It also plans the SAME science under its own label.  28 recorded steps promise
``table:absolute_risk_context``: 0 declare the spec, 0 are claimed, and nothing
ever asked them to.  They pass 82% of the time, so the Coder writes a table --
a DIFFERENT table every run.  Of the 26 recorded ``absolute_risk_context.csv``
files, 25 have distinct headers.  Every downstream figure over them is dead:
14 such steps recorded, 0 ok.

That is the defect: an 82%-passing step emits an artifact with no contract, and
the whole cost lands on its consumer.  Success at a producer is not the same as
a contract.

**The first version of this gate asked those steps to rename their product, and
a real run overturned it.**  Measured over every recorded run it asked 48
distinct step shapes and only 1 promised this product: 27 promised
``table:cohort_summary``, 18 ``table:absolute_risk_context``, one each
``table:stage_stratified_outcome`` and ``table:ordinal_trend_audit``.  Which
table a step promises is a scientific choice, and canary33 showed the price of
taking it at execution: ``04_absolute_risk_context`` executed fine one run
earlier and was refused, taking its figure with it, and after three plan
revisions the Planner never filled the spec.  Across two real runs the wide
version helped 0 steps and cost 4.

So the gate is narrowed to the step that ALREADY promises this product and only
omits the spec -- 4 such records exist, 3 ``ok`` (an uncontracted table the
Coder wrote) and 1 ``coder_failed``.  Getting the same science planned under
this product's name is a real problem, but it belongs to the Planner directive,
where the Planner can act on it before the plan is sealed, not to a refusal
raised at the step.

Two things this deliberately does NOT do, each because the evidence says it
would regress:
  * it does not relax ``prevalence_outcome_figure``'s supported-product set.
    That renderer's ``_REQUIRED_COLUMNS`` is an exact 12-column header, and 25
    of 26 recorded tables do not have it -- so relaxing the name would make it
    claim steps and then raise at load, which is the failure
    ``robustness_figure_executor``'s own docstring records paying for once.
  * it does not let the owner claim a differently-named product. The executor
    writes ``exposure_outcome_distribution.csv`` and registers its own key, so
    claiming a step that promised another name would leave that promise
    unfulfilled.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
    EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT,
    exposure_outcome_distribution_declaration_verdict,
    exposure_outcome_distribution_executor_owns_step,
)
from easyicu.research_agent.execution.owner_declaration import (
    _declared_choice,
    _prohibited_choices,
)
from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

#: The same shape the owner's own test suite uses.
_SPEC = {
    "exposure": "exposure_variable",
    "exposure_levels": [0, 1],
    "outcome": "outcome_variable",
    "outcome_levels": [0, 1],
    "outcome_positive_value": 1,
    "level_match_policy": "exact_typed",
    "denominator_policy": "all_declared_rows",
    "missing_outcome_policy": "structural_absence_is_non_event",
    "confidence_level": 0.95,
}


def _step(*, outputs, spec=None, method="descriptive", **kwargs) -> AnalysisStep:
    return AnalysisStep(
        step_id="04_absolute_risk_context",
        planned_analysis_role="auxiliary",
        intent="Report prevalence and outcome by exposure level on the locked cohort.",
        # The schema requires the spec's exposure and outcome to be explicit
        # step inputs, so the fixture carries them whether or not it declares a spec.
        inputs=["artifact:analysis_cohort", "exposure_variable", "outcome_variable"],
        expected_outputs=list(outputs),
        method=method,
        exposure_outcome_distribution_spec=spec,
        **kwargs,
    )


def _missing(step: AnalysisStep) -> tuple[str, ...]:
    return tuple(
        exposure_outcome_distribution_declaration_verdict(step).missing_declarations
    )


# --- the gap it must report ---------------------------------------------------


def test_a_step_promising_this_product_without_the_spec_is_a_reported_gap() -> None:
    """4 recorded records promise it and declare no spec; 3 shipped a table
    the Coder invented and 1 died outright, and none was ever asked."""

    missing = _missing(_step(outputs=[EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT]))
    assert missing == ("exposure_outcome_distribution_spec",), missing


def test_a_step_with_no_typed_cohort_input_is_not_asked() -> None:
    """A precondition of the owner running, not a guess about intent.

    This executor reads exactly one typed cohort. Asking a step without one to
    declare the spec would demand work that leaves it exactly as unowned --
    the same rule that keeps the robustness gap off steps whose products this
    replay does not emit.
    """

    step = AnalysisStep(
        step_id="04_prevalence_by_exposure",
        planned_analysis_role="auxiliary",
        intent="Report prevalence and outcome by exposure level.",
        inputs=["exposure_variable", "outcome_variable"],
        expected_outputs=[EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT],
        method="descriptive",
    )

    assert _missing(step) == ()


def test_the_method_label_does_not_decide_this_gap() -> None:
    """Measured: the two-string allowlist turned away exactly the step the gap
    is for -- ``descriptive_prevalence_and_mortality``, promising this product
    with no spec. The promised product is the claim; a label allowlist beside
    it is the disease, not the cure."""

    labels = (
        "descriptive",
        "descriptive_prevalence_and_mortality",
        "prevalence_and_absolute_risk_descriptive",
        "something_nobody_registered",
    )
    verdicts = {
        _missing(_step(outputs=[EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT], method=label))
        for label in labels
    }

    assert verdicts == {("exposure_outcome_distribution_spec",)}


def test_a_step_carrying_another_owners_spec_is_not_asked_for_this_one() -> None:
    """Both of these are constructible alongside this product, so both are live.

    A step already declaring another owner's typed contract is that owner's
    step; asking it for this spec would spend a replan on a declaration that
    could not make this executor run.

    Only two. ``table_one_spec`` requires ``table:table_one`` as an expected
    output and ``robustness_replay_spec`` requires its products to be declared
    outputs, so a step promising exactly this product can carry neither -- the
    guards for those were deleted rather than left as clauses no input can
    reach.
    """

    for field, spec in (
        (
            "measurement_audit_spec",
            {
                "schema_version": "easyicu.measurement_audit/1",
                "products": [
                    {
                        "product_id": "exposure_outcome_distribution",
                        "audit": "measurement_missingness",
                    }
                ],
            },
        ),
        ("trajectory_stability_spec", {"n_resamples": 50, "sample_fraction": 0.8}),
    ):
        step = AnalysisStep(
            step_id="04_x",
            planned_analysis_role="auxiliary",
            intent="Report prevalence and outcome by exposure level.",
            inputs=[
                "artifact:analysis_cohort",
                "exposure_variable",
                "outcome_variable",
                "age",
            ],
            expected_outputs=[EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT],
            method="descriptive",
            **{field: spec},
        )

        assert _missing(step) == (), field


def test_a_step_promising_a_DIFFERENT_table_is_not_asked_to_rename_it() -> None:
    """canary33's regression, pinned so it cannot come back.

    ``04_absolute_risk_context`` executed in canary32 and was refused in
    canary33 by the wide version of this gate, which demanded it promise a
    different product. Which table a step promises is the Planner's scientific
    choice; the widest measured cost of taking it here was 27 cohort-summary
    steps, 18 absolute-risk steps and 2 others -- 47 of the 48 shapes asked.
    """

    for other in (
        "table:absolute_risk_context",
        "table:cohort_summary",
        "table:stage_stratified_outcome",
        "table:ordinal_trend_audit",
    ):
        assert _missing(_step(outputs=[other])) == (), other

    reason = exposure_outcome_distribution_declaration_verdict(
        _step(outputs=["table:absolute_risk_context"])
    ).reason
    assert EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT in reason, reason


def test_a_step_that_only_lacks_the_spec_is_asked_only_for_the_spec() -> None:
    """Do not demand a rename that is already right."""

    missing = _missing(_step(outputs=[EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT]))
    assert missing == ("exposure_outcome_distribution_spec",), missing


# --- what it must NOT claim ---------------------------------------------------


def test_a_fully_declared_step_reports_no_gap() -> None:
    """The owner claims it; a gap here would demand work already done."""

    step = _step(outputs=[EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT], spec=_SPEC)
    assert exposure_outcome_distribution_executor_owns_step(step) is True
    assert _missing(step) == ()


def test_a_step_another_typed_owner_would_claim_is_left_alone() -> None:
    """Every guarded spec, each on the output ITS OWN schema requires.

    This is the population the first version of this test got wrong. Probing
    ``table_one_spec`` on a step promising ``table:absolute_risk_context`` only
    shows the schema pairs a Table 1 spec with ``table:table_one`` -- it does
    not show that no step reaching this verdict can carry one. Removing the
    clause on that reading asked every recorded Table 1 step, because this
    verdict is consulted BEFORE ``grouped_table_one`` in selection order.
    """

    table_one = AnalysisStep(
        step_id="03_table_one",
        planned_analysis_role="auxiliary",
        intent="Baseline characteristics by exposure level.",
        inputs=["artifact:analysis_cohort", "exposure_variable", "age"],
        expected_outputs=["table:table_one"],
        method="descriptive",
        table_one_spec={
            "group_by": "exposure_variable",
            "group_levels": [0, 1],
            "variables": [
                {
                    "name": "age",
                    "variable_kind": "continuous",
                    "summary": "mean_sd",
                    "test": "welch_t_or_anova",
                }
            ],
        },
    )
    assert _missing(table_one) == (), _missing(table_one)

    association = AnalysisStep(
        step_id="06_primary_adjusted_association",
        planned_analysis_role="primary",
        intent="Adjusted association between exposure and outcome.",
        inputs=["artifact:analysis_cohort", "exposure_variable", "outcome_variable"],
        expected_outputs=["table:adjusted_association_estimates"],
        method="adjusted_association_models",
        model_requirements=[
            {
                "requirement_id": "primary",
                "outcome": "outcome_variable",
                "outcome_type": "binary",
                "method_family": "logistic_regression",
                "exposure_source": "exposure_variable",
                "analysis_role": "primary",
                "analysis_set": "source_aware",
            }
        ],
    )
    assert _missing(association) == (), _missing(association)


def test_a_step_carrying_another_owners_live_spec_is_left_alone() -> None:
    """The three the schema does allow beside this shape.

    Each means a different typed owner already has the step, so asking it to
    declare this one would demand work that leaves it exactly as unowned.
    """

    for field, value in (
        ("trajectory_stability_spec", {"n_resamples": 20, "sample_fraction": 0.8}),
        (
            "measurement_audit_spec",
            {
                "products": [
                    {
                        "product_id": "absolute_risk_context",
                        "audit": "analytic_denominators",
                    }
                ]
            },
        ),
        (
            "robustness_replay_spec",
            {
                "products": [
                    {
                        "product_id": "absolute_risk_context",
                        "output": "robustness_matrix",
                    }
                ]
            },
        ),
    ):
        step = _step(outputs=["table:absolute_risk_context"], **{field: value})
        assert _missing(step) == (), (field, _missing(step))


def test_a_step_promising_two_tables_is_not_claimed_by_asking() -> None:
    """This owner emits exactly one product; asking a two-product step to
    declare a spec would demand a declaration that still cannot be satisfied."""

    assert (
        _missing(_step(outputs=["table:absolute_risk_context", "table:cohort_summary"]))
        == ()
    )


def test_a_figure_step_is_not_asked_for_a_table_contract() -> None:
    step = AnalysisStep(
        step_id="07_prevalence_mortality_figure",
        planned_analysis_role="auxiliary",
        intent="Draw prevalence and mortality by exposure level.",
        inputs=["table:absolute_risk_context"],
        expected_outputs=["figure:stage_stratified_outcome"],
        method="visualization",
    )
    assert _missing(step) == ()


def test_a_descriptive_step_promising_a_FIGURE_is_not_asked_either() -> None:
    """The case the visualization test does not reach.

    Mutation found it: deleting the ``table:`` clause left every test passing,
    because the only figure case here also had ``method='visualization'``, which
    the method clause already excludes. This owner emits a table, so a
    descriptive step promising a figure could not be claimed however it were
    declared.
    """

    step = AnalysisStep(
        step_id="04_absolute_risk_figure",
        planned_analysis_role="auxiliary",
        intent="Show prevalence and outcome by exposure level.",
        inputs=["artifact:analysis_cohort", "exposure_variable", "outcome_variable"],
        expected_outputs=["figure:absolute_risk_context"],
        method="descriptive",
    )
    assert _missing(step) == (), _missing(step)


# --- the safety property ------------------------------------------------------


def test_asking_for_these_cannot_unforbid_a_scientific_choice() -> None:
    """``_prohibited_choices`` subtracts demanded fields from the "do not change
    the science" list, keyed on the normalised leaf of the path. Neither name
    this reports may land on exposure, outcome or cohort."""

    missing = _missing(_step(outputs=["table:absolute_risk_context"]))
    baseline = _prohibited_choices([])
    assert _prohibited_choices(missing) == baseline, (
        missing,
        _prohibited_choices(missing),
    )
    for name in missing:
        assert _declared_choice(name) not in {"exposure", "outcome", "cohort"}, name


# --- the recorded corpus ------------------------------------------------------

_CORPUS = Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


def _recorded_steps():
    for manifest_path in sorted(_CORPUS.glob("batch_*/*/aware/run_*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            continue
        for record in manifest.get("per_step_records") or []:
            if not isinstance(record, dict):
                continue
            raw = ((record.get("analysis_request") or {}).get("step")) or {}
            if not raw:
                continue
            yield record, raw


def _gaps_as_production_sees_them(raw: dict) -> tuple[str, ...]:
    """What the plan-time gate really reads.

    Not the verdict called directly: ``select_standard_executor`` RETURNS as
    soon as an owner claims, so a step another owner takes never reaches this
    verdict at all. Calling it standalone over every recorded step measures a
    population production does not have -- the mistake this whole series keeps
    paying for. So this asks the real selector and reads its trace, exactly as
    ``owner_declaration._declaration_gaps`` does.
    """

    try:
        step = AnalysisStep(**raw)
    except Exception:
        return ()
    trace: list = []
    try:
        select_standard_executor(
            step,
            plan=AnalysisPlan(research_question="q", steps=[step]),
            trace=trace,
        )
    except Exception:
        return ()
    return tuple(
        name
        for candidate in trace
        if candidate.analysis_kind == "exposure_outcome_distribution"
        for name in candidate.missing_declarations
    )


@pytest.mark.skipif(
    not _CORPUS.exists(), reason="recorded runs are not on this machine"
)
def test_it_fires_on_the_steps_it_was_written_for() -> None:
    """Reachability, on real plans: a check that never fires is worse than none.

    Thin on purpose. 34 distinct recorded shapes promise this product and 30
    already declare the spec; the gap is the remaining 4, of which 1 clears
    every other clause of the contract. Firing once on 596 recorded shapes is
    the honest size of this defect -- the wide version fired on 48 and was
    wrong 47 times.
    """

    fired = 0
    asked_a_rename = 0
    for _record, raw in _recorded_steps():
        gaps = _gaps_as_production_sees_them(raw)
        if not gaps:
            continue
        fired += 1
        outputs = [str(o) for o in (raw.get("expected_outputs") or [])]
        if outputs != [EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT]:
            asked_a_rename += 1

    assert fired >= 1, "the narrowed check fires on no recorded plan at all"
    assert asked_a_rename == 0, (
        f"{asked_a_rename} recorded steps are asked to promise a different "
        "product; that is the Planner's scientific choice, not this owner's"
    )


@pytest.mark.skipif(
    not _CORPUS.exists(), reason="recorded runs are not on this machine"
)
def test_it_never_fires_on_a_step_an_owner_already_claimed() -> None:
    """The false-positive cost, measured where it would be paid.

    Asking a step that already has an owner would spend a forced replan on a
    declaration that changes nothing -- and could cost it the owner it had.
    """

    offenders = []
    for record, raw in _recorded_steps():
        trace = record.get("standard_executor_candidates")
        if not isinstance(trace, dict) or not trace.get("claimed_by"):
            continue
        if _gaps_as_production_sees_them(raw):
            offenders.append((record.get("step_id"), trace.get("claimed_by")))
    assert (
        not offenders
    ), f"steps that already have a deterministic owner would be asked: {offenders[:5]}"
