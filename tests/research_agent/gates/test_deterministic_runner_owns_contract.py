"""Only a receipt-bound registered primary owner may waive its own shape error."""

from __future__ import annotations

from easyicu.research_agent.execution.phase import (
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


def test_historical_causal_runner_never_owns_primary_estimate():
    assert not _primary_runner_core_estimate_present(
        "causal_primary_iptw", {"status": "ok", "adjusted_effect": 3.04}
    )


def test_core_estimate_absent_causal_blocked():
    assert not _primary_runner_core_estimate_present(
        "causal_primary_iptw", {"status": "blocked", "adjusted_effect": None}
    )


def test_historical_survival_runner_never_owns_flat_or_nested_estimate():
    assert not _primary_runner_core_estimate_present(
        "survival_primary_cox", {"status": "ok", "hazard_ratio": 0.83}
    )


def test_current_host_survival_owner_binds_its_primary_estimate():
    assert _primary_runner_core_estimate_present(
        "survival_primary_cox",
        {
            "status": "ok",
            "receipt_issuer": "easyicu.host.survival_primary_cox_v1",
            "hazard_ratio": 0.83,
        },
    )
    assert not _primary_runner_core_estimate_present(
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


def test_typed_descriptive_owner_requires_its_exact_noncausal_receipt() -> None:
    summary = {
        "status": "ok",
        "adjusted_effect": None,
        "interpretation_class": "exposure_outcome_distribution",
        "analysis_role": "primary",
        "analysis_set": "bound_typed_cohort",
        "interpretation_ceiling": "descriptive_unadjusted_not_causal",
        "descriptive_estimates": {
            "schema_version": "easyicu.exposure_outcome_descriptive_estimates/1",
            "analysis_role": "primary",
            "analysis_set": "bound_typed_cohort",
            "interpretation_ceiling": "descriptive_unadjusted_not_causal",
            "exposure_prevalence": [{"level": 0, "estimate_pct": 60.0}],
            "outcome_absolute_risks": [{"level": 0, "estimate_pct": 8.0}],
            "risk_difference": None,
        },
    }

    assert _primary_runner_core_estimate_present(
        "exposure_outcome_distribution", summary
    )
    summary["descriptive_estimates"]["interpretation_ceiling"] = "associated"
    assert not _primary_runner_core_estimate_present(
        "exposure_outcome_distribution", summary
    )


# --- demotion behaviour -----------------------------------------------------


def test_historical_causal_runner_cannot_demote_step_contract_errors():
    step_record = {"deterministic_standard_analysis": "causal_primary_iptw"}
    summary = {"status": "ok", "adjusted_effect": 3.04}
    out = _demote_step_contract_for_primary_runner(
        step_record, summary, [_contract_error()]
    )
    assert out[0].severity == "error"


def test_causal_runner_blocked_does_not_demote():
    step_record = {"deterministic_standard_analysis": "causal_primary_iptw"}
    summary = {"status": "blocked", "adjusted_effect": None}
    out = _demote_step_contract_for_primary_runner(
        step_record, summary, [_contract_error()]
    )
    assert out[0].severity == "error"


def test_historical_survival_runner_cannot_demote_step_contract_errors():
    step_record = {"deterministic_standard_analysis": "survival_primary_cox"}
    summary = {"status": "ok", "hazard_ratio": 0.83}
    out = _demote_step_contract_for_primary_runner(
        step_record, summary, [_contract_error()]
    )
    assert out[0].severity == "error"


def test_current_host_survival_owner_cannot_waive_an_untyped_shape_error():
    step_record = {"deterministic_standard_analysis": "survival_primary_cox"}
    summary = {
        "status": "ok",
        "receipt_issuer": "easyicu.host.survival_primary_cox_v1",
        "hazard_ratio": 0.83,
    }
    out = _demote_step_contract_for_primary_runner(
        step_record, summary, [_contract_error(), _integrity_error()]
    )
    by_validator = {finding.validator: finding.severity for finding in out}
    assert by_validator["step_contract"] == "error"
    assert by_validator["primary_exposure_overadjustment"] == "error"


def test_integrity_findings_are_never_demoted():
    step_record = {"deterministic_standard_analysis": "causal_primary_iptw"}
    summary = {"status": "ok", "adjusted_effect": 3.04}
    out = _demote_step_contract_for_primary_runner(
        step_record, summary, [_contract_error(), _integrity_error()]
    )
    by_validator = {f.validator: f.severity for f in out}
    assert by_validator["step_contract"] == "error"
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


# --- the empirical measurement summary -----------------------------------------


def _empirical_summary(**changes):
    summary = {
        "status": "ok", "analysis_role": "primary", "cohort_n": 12,
        "declared_columns": ["temperature"], "adjusted_effect": None,
        "interpretation_ceiling": "empirical_stay_distribution",
        "paper_authorization_allowed": False,
        "descriptive_summary_validation": {
            "schema_version": "easyicu.descriptive_summary_validation/1",
            "validated": True, "source_rows": 12, "columns": ["temperature"],
            "source_sha256": "a" * 64, "cohort_summary_sha256": "b" * 64,
            "distribution_summary_sha256": "c" * 64,
        },
    }
    summary.update(changes)
    return summary


def test_the_empirical_summary_binds_only_its_validated_receipt():
    from easyicu.research_agent.contracts.cohort_summary import (
        DESCRIPTIVE_SUMMARY_ANALYSIS_KIND,
    )

    kind = DESCRIPTIVE_SUMMARY_ANALYSIS_KIND
    assert _primary_runner_core_estimate_present(kind, _empirical_summary())
    assert not _primary_runner_core_estimate_present(kind, {"status": "ok"})
    # A summary that claims paper authority is not this owner's result.
    assert not _primary_runner_core_estimate_present(
        kind, _empirical_summary(paper_authorization_allowed=True),
    )
    receipt = dict(_empirical_summary()["descriptive_summary_validation"], source_rows=11)
    assert not _primary_runner_core_estimate_present(
        kind, _empirical_summary(descriptive_summary_validation=receipt),
    )
