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

from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.plan_utils import _cap_plan_preserving_figure_steps
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
