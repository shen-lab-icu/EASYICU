"""A deterministic PRIMARY runner owns its step's contract.

Built 2026-07-07 after H2 fix7: the LLM planner handed the causal step a
17-output contract (target_trial_protocol, confounder_set, model_stability_audit
...). The deterministic IPTW runner emits its core estimate + a handful of
tables, so the contract check flagged the *missing* documentation tables as
errors and triggered a repair that replaced the trustworthy estimate
(OR 3.04) with LLM code that then blocked (adjusted_effect=None).

Fix: when a primary deterministic runner (causal IPTW / survival Cox) produced
its core estimate, ``step_contract`` missing-output ERRORS are demoted to
advisory warnings so planner output-bloat cannot repair-away the estimate.
Integrity findings from other validators still block.
"""

from __future__ import annotations

from easyicu.research_agent.pipeline_execute import (
    _demote_step_contract_for_primary_runner,
    _primary_runner_core_estimate_present,
)
from easyicu.research_agent.schema import ValidationFinding


def _contract_error(
    msg: str = "missing table:target_trial_protocol",
) -> ValidationFinding:
    return ValidationFinding(validator="step_contract", severity="error", message=msg)


def _integrity_error() -> ValidationFinding:
    # e.g. overadjustment / leakage / exposure-mismatch: must NEVER be demoted.
    return ValidationFinding(
        validator="primary_exposure_overadjustment",
        severity="error",
        message="conditioned on a constituent of the composite exposure",
    )


# --- core-estimate detection ------------------------------------------------


def test_core_estimate_present_causal_ok():
    assert _primary_runner_core_estimate_present(
        "causal_primary_iptw", {"status": "ok", "adjusted_effect": 3.04}
    )


def test_core_estimate_absent_causal_blocked():
    assert not _primary_runner_core_estimate_present(
        "causal_primary_iptw", {"status": "blocked", "adjusted_effect": None}
    )


def test_core_estimate_present_survival_flat_and_nested():
    assert _primary_runner_core_estimate_present(
        "survival_primary_cox", {"status": "ok", "hazard_ratio": 0.83}
    )
    assert _primary_runner_core_estimate_present(
        "survival_primary_cox",
        {"status": "ok", "primary_model": {"hazard_ratio": 0.83}},
    )


def test_core_estimate_non_primary_runner_is_never_owned():
    assert not _primary_runner_core_estimate_present(
        "cohort_definition_sensitivity", {"status": "ok", "adjusted_effect": 3.0}
    )
    assert not _primary_runner_core_estimate_present(
        None, {"status": "ok", "adjusted_effect": 3.0}
    )


# --- demotion behaviour -----------------------------------------------------


def test_causal_runner_with_estimate_demotes_step_contract_errors():
    step_record = {"deterministic_standard_analysis": "causal_primary_iptw"}
    summary = {"status": "ok", "adjusted_effect": 3.04}
    out = _demote_step_contract_for_primary_runner(
        step_record, summary, [_contract_error()]
    )
    assert out[0].severity == "warning"
    assert "advisory" in out[0].message


def test_causal_runner_blocked_does_not_demote():
    step_record = {"deterministic_standard_analysis": "causal_primary_iptw"}
    summary = {"status": "blocked", "adjusted_effect": None}
    out = _demote_step_contract_for_primary_runner(
        step_record, summary, [_contract_error()]
    )
    assert out[0].severity == "error"


def test_survival_runner_with_hr_demotes():
    step_record = {"deterministic_standard_analysis": "survival_primary_cox"}
    summary = {"status": "ok", "hazard_ratio": 0.83}
    out = _demote_step_contract_for_primary_runner(
        step_record, summary, [_contract_error()]
    )
    assert out[0].severity == "warning"


def test_integrity_findings_are_never_demoted():
    step_record = {"deterministic_standard_analysis": "causal_primary_iptw"}
    summary = {"status": "ok", "adjusted_effect": 3.04}
    out = _demote_step_contract_for_primary_runner(
        step_record, summary, [_contract_error(), _integrity_error()]
    )
    by_validator = {f.validator: f.severity for f in out}
    assert by_validator["step_contract"] == "warning"
    # the overadjustment / leakage integrity error must still block
    assert by_validator["primary_exposure_overadjustment"] == "error"


def test_non_primary_runner_leaves_contract_errors_intact():
    step_record = {"deterministic_standard_analysis": "cohort_definition_sensitivity"}
    summary = {"status": "ok"}
    out = _demote_step_contract_for_primary_runner(
        step_record, summary, [_contract_error()]
    )
    assert out[0].severity == "error"


def test_llm_coded_step_no_runner_leaves_contract_errors_intact():
    # No deterministic runner -> the contract check must still fail-close.
    step_record: dict = {}
    summary = {"status": "ok", "adjusted_effect": 3.0}
    out = _demote_step_contract_for_primary_runner(
        step_record, summary, [_contract_error()]
    )
    assert out[0].severity == "error"
