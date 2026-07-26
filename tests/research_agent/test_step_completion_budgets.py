"""The budgets that decide whether a step is allowed to finish.

Three settings jointly determine whether an analysis completes for a scientific
reason or for an accounting one: the wall clock a generated script is given, the
number of provider calls a step may spend, and the number of steps a plan may
contain. Each has a legitimate guard behind it, and each was previously set
where it bound during ordinary work rather than only on a runaway.

These tests pin the arithmetic, not the taste. A future change to any default
is fine; a change that leaves a step entitled to spend more than it is granted,
or that silently drops declared scientific products, is not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.orchestration.config import (
    PipelineConfig,
    step_provider_call_entitlement,
)
from easyicu.research_agent.plan_utils import _cap_plan_preserving_figure_steps
from easyicu.research_agent.reporting.completion import publication_authorized
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


# --------------------------- provider-call budget ---------------------------


def test_a_step_may_not_be_entitled_to_spend_more_than_it_is_granted() -> None:
    """Entitlement is 1 generation + code repairs + LLM repairs + a reserved
    final concept-audit call. When the grant equals the entitlement exactly,
    a structured-retry on one malformed response is paid for out of the
    science: the step loses a repair it was promised.
    """

    config = PipelineConfig(workdir="./unused")
    entitled = (
        1  # initial generation
        + config.max_code_repair_attempts
        + config.max_step_llm_repair_attempts
        + 1  # execution/phase.py reserves the last call for the concept audit
    )
    assert config.max_step_provider_calls > entitled, (
        "the per-step provider budget must leave headroom above what the step "
        f"is entitled to spend (entitled={entitled}, "
        f"granted={config.max_step_provider_calls})"
    )


def test_a_budget_that_cannot_fund_its_own_repairs_is_refused() -> None:
    """The default being adequate says nothing about a configured one.

    A step that runs out of provider calls mid-repair fails the way a broken
    analysis fails, so the run reports a scientific problem that is really an
    accounting one. Construction is the last point where the two are still
    distinguishable.
    """

    with pytest.raises(ValueError, match="cannot fund the repair policy"):
        PipelineConfig(
            workdir="./unused",
            max_code_repair_attempts=3,
            max_step_llm_repair_attempts=4,
            enable_llm_concept_audit=True,
            max_step_provider_calls=7,
        )


def test_deliberate_under_funding_stays_available_when_it_is_declared() -> None:
    """Capping spend on a throwaway run is a real need; hiding it is not."""

    config = PipelineConfig(
        workdir="./unused",
        max_step_provider_calls=2,
        allow_underfunded_step_provider_calls=True,
    )
    assert config.max_step_provider_calls == 2


def test_declaring_under_funding_changes_the_config_identity() -> None:
    """An escape hatch that leaves no trace is a hole, not an escape hatch.

    `canonical_payload` renders every field, so this holds today by
    construction — which is exactly why it is worth pinning: a future payload
    that filters fields could drop it silently, and a run would then claim a
    repair policy its budget never funded.
    """

    funded = PipelineConfig(workdir="./unused")
    declared = PipelineConfig(
        workdir="./unused", allow_underfunded_step_provider_calls=True
    )

    assert funded.canonical_payload()["allow_underfunded_step_provider_calls"] is False
    assert declared.canonical_payload()["allow_underfunded_step_provider_calls"] is True
    assert funded.canonical_digest() != declared.canonical_digest()


def test_a_paper_run_may_not_declare_an_under_funded_budget(tmp_path: Path) -> None:
    """Declaring the shortfall makes it visible; it does not make it authority.

    Keyed off paper-facing rather than "a profile was supplied", so a `*_dev`
    profile can still exercise exhaustion — the same distinction the two
    existing non-paper-authority guards beside it already draw.
    """

    from easyicu.research_agent import MockLLMClient, ResearchAgentPipeline
    from easyicu.research_agent.orchestration.profiles import is_paper_facing_profile

    with pytest.raises(ValueError, match="non-paper authority"):
        ResearchAgentPipeline(
            workdir=tmp_path / "paper",
            llm=MockLLMClient(),
            submission_profile_name="npj_dm",
            submission_profile_version="20260719",
            allow_underfunded_step_provider_calls=True,
        )

    # The dev arm is checked at the predicate the guard keys off. Constructing
    # a `*_dev` pipeline here would drag in that profile's unrelated pins
    # (know-how, capability workflow) and test those instead of this.
    assert is_paper_facing_profile("npj_dm_know_how_dev") is False
    assert is_paper_facing_profile("npj_dm") is True

    # And no profile at all must not be treated as paper-facing either.
    ResearchAgentPipeline(
        workdir=tmp_path / "plain",
        llm=MockLLMClient(),
        allow_underfunded_step_provider_calls=True,
    )


def test_the_entitlement_is_computed_from_the_policy_not_a_constant() -> None:
    """Each term is edited for its own reason; nothing recomputed the sum.

    Turning the concept auditor off must lower the requirement, or the check
    would demand budget for a call the run will never make.
    """

    with_audit = step_provider_call_entitlement(
        max_code_repair_attempts=3,
        max_step_llm_repair_attempts=2,
        llm_concept_audit_enabled=True,
    )
    without_audit = step_provider_call_entitlement(
        max_code_repair_attempts=3,
        max_step_llm_repair_attempts=2,
        llm_concept_audit_enabled=False,
    )
    assert with_audit == 7
    assert without_audit == with_audit - 1
    assert (
        step_provider_call_entitlement(
            max_code_repair_attempts=5,
            max_step_llm_repair_attempts=2,
            llm_concept_audit_enabled=True,
        )
        == with_audit + 2
    )


def test_the_default_configuration_funds_itself(tmp_path: Path) -> None:
    """The gate must not fire on the shipped defaults — through either door."""

    from easyicu.research_agent import MockLLMClient, ResearchAgentPipeline

    PipelineConfig(workdir="./unused")
    ResearchAgentPipeline(workdir=tmp_path / "wd", llm=MockLLMClient())


# ------------------------------- wall clock --------------------------------


def test_generated_code_is_not_held_to_a_wall_clock_the_work_cannot_meet() -> None:
    """A Cox fit with PH diagnostics, a bootstrap stability sweep, or a
    propensity match on a real ICU cohort does not finish in five minutes.
    A limit below the honest cost of the work makes those steps unreachable
    however the script is written."""

    config = PipelineConfig(workdir="./unused")
    assert config.timeout_seconds >= 900.0


def test_the_deterministic_executor_keeps_its_own_larger_budget() -> None:
    """Raising the coder's wall clock must not be achieved by lowering the
    registered-standard budget to meet it; the two are separate on purpose."""

    config = PipelineConfig(workdir="./unused")
    assert (
        config.standard_executor_timeout_seconds > config.timeout_seconds
    ), "a registered deterministic standard runs a larger bounded workload"


# ------------------------------ plan capacity ------------------------------


def _plan_with_products(n_steps: int) -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Does early vasopressor exposure change 28-day mortality?",
        steps=[
            AnalysisStep(
                step_id=f"{index:02d}_step",
                intent=f"step {index}",
                inputs=[],
                expected_outputs=[f"table:product_{index}"],
                method="descriptive",
                icu_rule_refs=[],
            )
            for index in range(1, n_steps + 1)
        ],
        rationale="probe",
    )


def test_a_four_product_family_plan_is_not_truncated_by_the_default_cap() -> None:
    """Prediction, survival, causal and trajectory each declare four required
    products, and each still needs cohort definition, missingness, the primary
    model and robustness before them. A cap that bites there shrinks the
    science rather than catching a runaway."""

    config = PipelineConfig(workdir="./unused")
    realistic_plan_length = 4 + 4 + 4  # products + prerequisites + replan slack
    assert config.max_total_steps >= realistic_plan_length


def test_truncation_names_the_products_the_analysis_no_longer_has() -> None:
    """Step ids are internal. A reader cannot tell from "dropped: 13_x" that
    the run no longer contains the calibration figure it was asked for."""

    plan = _plan_with_products(6)
    _capped, findings = _cap_plan_preserving_figure_steps(plan=plan, cap=3)
    truncation = [
        finding
        for finding in findings
        if finding.detail and finding.detail.get("plan_truncated")
    ]
    assert truncation, "truncation must be reported as a typed finding"
    detail = truncation[0].detail
    dropped_outputs = detail["dropped_expected_outputs"]
    assert dropped_outputs, "the dropped scientific products must be named"
    step_products = detail["dropped_step_products"]
    assert {contract["step_id"] for contract in step_products} == set(
        detail["dropped_step_ids"]
    )
    assert {
        output
        for contract in step_products
        for output in contract["expected_outputs"]
    } == set(dropped_outputs)
    # Every named product really belongs to a dropped step, and none belongs
    # to a step that survived.
    dropped_ids = set(detail["dropped_step_ids"])
    kept_outputs = {
        str(output)
        for step in _capped.steps
        for output in (step.expected_outputs or ())
    }
    assert not (set(dropped_outputs) & kept_outputs)
    assert all(
        any(
            f"product_{sid.split('_')[0].lstrip('0') or '0'}" in out
            for out in dropped_outputs
        )
        for sid in dropped_ids
    )
    assert "no longer produces" in truncation[0].message


def test_an_untruncated_plan_reports_no_truncation() -> None:
    """The marker must mean something: a plan under the cap must not carry it."""

    plan = _plan_with_products(3)
    _capped, findings = _cap_plan_preserving_figure_steps(plan=plan, cap=8)
    assert not [
        finding
        for finding in findings
        if finding.detail and finding.detail.get("plan_truncated")
    ]


# ------------------- truncation is binding, not just visible -------------------


def _authorized(**overrides: bool) -> bool:
    """Everything a paper needs is satisfied except what the test overrides."""

    terms = {
        "manuscript_ready": True,
        "publication_figure_bundle_ready": True,
        "publication_provenance_ready": True,
        "display_suite_complete": True,
        "article_contract_complete": True,
        "article_figure_strategy_complete": True,
    }
    terms.update(overrides)
    return publication_authorized(**terms)


def test_a_run_that_lost_planned_products_is_not_a_paper() -> None:
    """Naming the dropped products is only a report until something reads it.

    Every other gate asks whether what the run did is sound. None of them can
    see that the run was asked for more: the dropped steps never executed, so
    they have no failed record, no missing evidence and no unbound number.
    """

    assert _authorized() is True
    assert _authorized(plan_not_truncated=False) is False


def test_a_truncated_run_is_still_worth_reading() -> None:
    """The block belongs on paper authorization, not on the manuscript.

    A truncated run is exactly what an operator iterating on a study wants to
    read; refusing to produce it would trade a silent loss for a blind one.
    """

    from easyicu.research_agent.reporting.readiness import _plan_truncation_status

    plan = _plan_with_products(6)
    capped, findings = _cap_plan_preserving_figure_steps(plan=plan, cap=3)
    status = _plan_truncation_status(findings, plan=capped)
    assert status["plan_truncated"] is True
    # The gate carries the reason with it, so the report can say which products
    # are missing rather than only that authorization failed.
    assert status["plan_truncated_dropped_outputs"]


def test_the_cap_and_the_gate_agree_on_an_intact_plan() -> None:
    """Negative control: the gate must not block a plan that fit its cap."""

    from easyicu.research_agent.reporting.readiness import _plan_truncation_status

    plan = _plan_with_products(3)
    capped, findings = _cap_plan_preserving_figure_steps(plan=plan, cap=8)
    status = _plan_truncation_status(findings, plan=capped)
    assert status["plan_truncated"] is False
    assert status["plan_truncation_recorded"] is False
    assert status["plan_truncated_dropped_outputs"] == []
    assert _authorized(plan_not_truncated=not status["plan_truncated"]) is True


def test_a_later_revision_that_restores_the_products_clears_the_block() -> None:
    """The cap runs on every replanner revision and all findings accumulate.

    Asking "was anything ever truncated" would latch: a capped revision
    followed by one that fits and declares every product would be refused
    forever on the strength of a superseded warning. The question is whether
    the plan of record still lacks the products.
    """

    from easyicu.research_agent.reporting.readiness import _plan_truncation_status

    over_cap = _plan_with_products(6)
    _capped, findings = _cap_plan_preserving_figure_steps(plan=over_cap, cap=3)
    assert [f for f in findings if f.detail and f.detail.get("plan_truncated")]

    # The replanner then produces a revision that fits and declares every one
    # of the products the earlier revision had to drop.
    final_plan = _plan_with_products(6)
    status = _plan_truncation_status(findings, plan=final_plan)

    assert status["plan_truncated"] is False
    assert status["plan_truncated_dropped_outputs"] == []
    # The run was still capped once, and the record must say so.
    assert status["plan_truncation_recorded"] is True


def test_a_same_named_product_on_another_step_does_not_clear_the_block() -> None:
    """A product label is not the identity of the scientific step producing it."""

    from easyicu.research_agent.reporting.readiness import _plan_truncation_status

    plan = _plan_with_products(4)
    plan.steps[0] = plan.steps[0].model_copy(
        update={"expected_outputs": ["table:product_4"]}
    )
    capped, findings = _cap_plan_preserving_figure_steps(plan=plan, cap=3)

    status = _plan_truncation_status(findings, plan=capped)

    assert "table:product_4" in {
        output for step in capped.steps for output in step.expected_outputs
    }
    assert status["plan_truncated"] is True
    assert status["plan_truncated_dropped_step_ids"] == ["04_step"]
    assert status["plan_truncated_dropped_outputs"] == ["table:product_4"]
    assert _authorized(plan_not_truncated=not status["plan_truncated"]) is False


def test_reusing_a_step_id_for_another_scientific_role_does_not_clear_the_block() -> None:
    from easyicu.research_agent.reporting.readiness import _plan_truncation_status

    over_cap = _plan_with_products(4)
    _capped, findings = _cap_plan_preserving_figure_steps(plan=over_cap, cap=3)
    final_plan = _plan_with_products(4)
    final_plan.steps[3] = final_plan.steps[3].model_copy(
        update={"planned_analysis_role": "secondary"}
    )

    status = _plan_truncation_status(findings, plan=final_plan)

    assert status["plan_truncated"] is True
    assert status["plan_truncated_dropped_step_ids"] == ["04_step"]
    assert status["plan_truncated_dropped_outputs"] == ["table:product_4"]


def test_restoring_the_step_shell_with_another_method_does_not_clear_the_block() -> (
    None
):
    """Same id, same role, same output name — different science.

    A dropped PH diagnostic (``schoenfeld_residual_test``) reappearing as a
    ``descriptive_summary`` satisfies every part of the step's shell while the
    diagnostic itself never came back. Method is what distinguishes them.
    """

    from easyicu.research_agent.reporting.readiness import _plan_truncation_status

    over_cap = _plan_with_products(4)
    over_cap.steps[3] = over_cap.steps[3].model_copy(
        update={"method": "schoenfeld_residual_test"}
    )
    _capped, findings = _cap_plan_preserving_figure_steps(plan=over_cap, cap=3)

    final_plan = _plan_with_products(4)
    final_plan.steps[3] = final_plan.steps[3].model_copy(
        update={"method": "descriptive_summary"}
    )

    status = _plan_truncation_status(findings, plan=final_plan)

    assert status["plan_truncated"] is True
    assert status["plan_truncated_dropped_step_ids"] == ["04_step"]
    assert _authorized(plan_not_truncated=not status["plan_truncated"]) is False


def test_restoring_the_same_method_does_clear_the_block() -> None:
    """Negative control: the method check must not block genuine recovery.

    Recovery is not tamper detection — the replanner is expected to author a
    fresh step — so matching method, id, role and outputs is enough.
    """

    from easyicu.research_agent.reporting.readiness import _plan_truncation_status

    over_cap = _plan_with_products(4)
    over_cap.steps[3] = over_cap.steps[3].model_copy(
        update={"method": "schoenfeld_residual_test"}
    )
    _capped, findings = _cap_plan_preserving_figure_steps(plan=over_cap, cap=3)

    final_plan = _plan_with_products(4)
    final_plan.steps[3] = final_plan.steps[3].model_copy(
        update={
            "method": "schoenfeld_residual_test",
            "intent": "Reworded intent that means the same thing.",
        }
    )

    status = _plan_truncation_status(findings, plan=final_plan)

    assert status["plan_truncated"] is False
    assert status["plan_truncation_recorded"] is True
    assert _authorized(plan_not_truncated=not status["plan_truncated"]) is True


def test_a_final_plan_that_never_regained_the_products_stays_blocked(
    tmp_path: Path,
) -> None:
    """The other half of the same question — and the one that must fail closed."""

    from easyicu.research_agent.reporting.readiness import _plan_truncation_status

    plan = _plan_with_products(6)
    capped, findings = _cap_plan_preserving_figure_steps(plan=plan, cap=3)
    status = _plan_truncation_status(findings, plan=capped)

    assert status["plan_truncated"] is True
    assert status["plan_truncated_dropped_outputs"]
    assert _authorized(plan_not_truncated=not status["plan_truncated"]) is False


def test_a_truncation_that_named_no_products_is_not_treated_as_repaired() -> None:
    """Being unable to prove a loss was recovered is not evidence that it was.

    Without this the check fails open: a dropped step declaring no outputs
    would produce an empty dropped-product list, which would then look
    identical to "every dropped product came back".
    """

    from easyicu.research_agent.reporting.readiness import _plan_truncation_status
    from easyicu.research_agent.schema import ValidationFinding

    finding = ValidationFinding(
        validator="planner",
        severity="warning",
        message="Initial plan had 6 steps; truncated to max_total_steps=3.",
        detail={
            "plan_truncated": True,
            "dropped_step_ids": ["05_step", "06_step"],
            "dropped_expected_outputs": [],
        },
    )
    status = _plan_truncation_status([finding], plan=_plan_with_products(6))
    assert status["plan_truncated"] is True


def test_pipeline_default_wall_clock_matches_the_config_default(
    tmp_path: Path,
) -> None:
    """PipelineConfig and the ResearchAgentPipeline signature carry the same
    defaults in two places; they must not drift apart."""

    from easyicu.research_agent import MockLLMClient, ResearchAgentPipeline

    config = PipelineConfig(workdir="./unused")
    pipeline = ResearchAgentPipeline(
        workdir=tmp_path / "wd",
        llm=MockLLMClient(),
    )
    assert pipeline._timeout_seconds == config.timeout_seconds
    assert pipeline._max_step_provider_calls == config.max_step_provider_calls
    assert pipeline._max_total_steps == config.max_total_steps
