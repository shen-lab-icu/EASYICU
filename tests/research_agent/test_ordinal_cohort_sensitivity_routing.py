"""A PRIMARY ordinal/association estimand step must not be hijacked by the
cohort-definition-sensitivity runner just because its intent mentions the
primary *cohort* denominator and a pre-specified within-cohort *sensitivity*
sub-analysis.

Regression for E3 (KDIGO stage dose-response): the primary step
``04_primary_stage_outcome_gradient`` was routed to the cohort-definition-
sensitivity deterministic runner, which cleanly skipped for lack of an
alternative-cohort input -> no primary result -> no publication figure ->
replan budget exhausted -> fail-closed. Root cause: a blunt
``"sensitivity" in blob and ("cohort"|"definition" in blob)`` co-occurrence
test misfired on both sides (it vetoed the ordinal runner and let the
cohort-sensitivity runner claim the step). The precise discriminator keys off
the actual alternative-definition signal instead.
"""

from __future__ import annotations

from easyicu.research_agent.pipeline_execute import (
    _is_cohort_definition_sensitivity_step,
    _ordinal_dose_response_step_matches,
)

# The real E3 primary step (verbatim shape): a graded ordinal exposure vs a
# binary outcome whose intent legitimately says "primary cohort denominator"
# and registers a "survivor-only LOS sensitivity" sub-analysis.
_E3_PRIMARY_INTENT = (
    "Estimate and display the stage-response gradient for ICU length of stay "
    "and death using the reconciled primary cohort denominator from step 03b. "
    "Report stage-stratified counts/percentages and confidence intervals, and "
    "separately register the survivor-only descriptive LOS sensitivity already "
    "pre-specified."
)
_E3_PRIMARY_OUTPUTS = [
    "table:stage_stratified_outcomes",
    "table:ordinal_trend_tests",
    "table:adjusted_death_gradient",
    "table:effect_scale_definition",
    "table:sparse_cell_audit",
]

# The real E3 cohort-definition-sensitivity step.
_E3_SENSITIVITY_INTENT = (
    "Re-run the stage distribution, attrition, and outcome-gradient summaries "
    "across the pre-specified alternative eligibility, exposure-availability, "
    "and missing-data definitions."
)
_E3_SENSITIVITY_OUTPUTS = [
    "table:overlap_and_movement_across_cohorts",
    "table:sensitivity_grid",
    "table:exposure_availability_definition_sensitivity",
]


def _blob(step_id, intent, outputs):
    expected_blob = " ".join(outputs).lower()
    blob = " ".join([step_id, intent, "", expected_blob]).lower()
    return blob, expected_blob


# --- the discriminator ------------------------------------------------------


def test_primary_ordinal_step_is_not_a_cohort_sensitivity_step():
    assert not _is_cohort_definition_sensitivity_step(
        "descriptive_plus_association",
        "04_primary_stage_outcome_gradient",
        _E3_PRIMARY_INTENT,
        _E3_PRIMARY_OUTPUTS,
    )


def test_real_cohort_sensitivity_step_matches_by_method():
    assert _is_cohort_definition_sensitivity_step(
        "cohort_definition_sensitivity",
        "05_cohort_definition_sensitivity_comparison",
        _E3_SENSITIVITY_INTENT,
        _E3_SENSITIVITY_OUTPUTS,
    )


def test_cohort_sensitivity_matches_by_alternative_eligibility_phrase():
    # even without the method key, the "alternative eligibility" phrase is a
    # genuine definition-comparison signal
    assert _is_cohort_definition_sensitivity_step(
        "association",
        "07_robustness",
        "Compare outcomes across alternative eligibility windows.",
        ["table:results"],
    )


def test_cohort_sensitivity_matches_by_output_table_token():
    assert _is_cohort_definition_sensitivity_step(
        "association",
        "07_robustness",
        "Robustness comparison.",
        ["table:alternative_cohort_attrition", "table:sensitivity_grid"],
    )


def test_within_cohort_sensitivity_subanalysis_is_not_hijacked():
    # a primary step mentioning both "cohort" and a within-cohort "sensitivity"
    # sub-analysis must NOT be treated as a definition comparison
    assert not _is_cohort_definition_sensitivity_step(
        "logistic_regression",
        "04_primary_effect",
        "Fit the primary model on the reconciled cohort; run a survivor-only "
        "sensitivity check and a complete-case sensitivity check.",
        ["table:primary_effect", "table:adjusted_or"],
    )


def test_plain_cohort_definition_step_is_not_a_sensitivity_step():
    # the primary cohort DEFINITION step (defines, does not vary/compare)
    assert not _is_cohort_definition_sensitivity_step(
        "cohort_definition",
        "01_primary_cohort_definition",
        "Define the adult analysis cohort with complete first-24h exposure "
        "ascertainment; require ICU LOS >= 1 day.",
        ["table:cohort_flow", "table:attrition_by_rule"],
    )


# --- the ordinal matcher still recognises the primary step ------------------


def test_ordinal_matcher_claims_primary_dose_response_step():
    blob, expected_blob = _blob(
        "04_primary_stage_outcome_gradient",
        _E3_PRIMARY_INTENT,
        _E3_PRIMARY_OUTPUTS,
    )
    assert _ordinal_dose_response_step_matches(
        "descriptive_plus_association", blob, expected_blob
    )


def test_ordinal_matcher_rejects_plain_association_without_dose_signal():
    # no explicit ordinal method, no ordinal output token, no dose-response
    # narrative -> not the ordinal primary step
    blob, expected_blob = _blob(
        "04_primary_association",
        "Estimate the adjusted association between vasopressor use and mortality.",
        ["table:adjusted_or"],
    )
    assert not _ordinal_dose_response_step_matches("association", blob, expected_blob)


def test_routing_precedence_primary_ordinal_beats_cohort_sensitivity():
    """The end-to-end invariant: for the E3 primary step the ordinal runner
    wins (matcher True) and the cohort-sensitivity runner declines (discriminator
    False); for the E3 sensitivity step the reverse holds."""
    blob, expected_blob = _blob(
        "04_primary_stage_outcome_gradient",
        _E3_PRIMARY_INTENT,
        _E3_PRIMARY_OUTPUTS,
    )
    primary_is_ordinal = _ordinal_dose_response_step_matches(
        "descriptive_plus_association", blob, expected_blob
    )
    primary_is_cohort_sens = _is_cohort_definition_sensitivity_step(
        "descriptive_plus_association",
        "04_primary_stage_outcome_gradient",
        _E3_PRIMARY_INTENT,
        _E3_PRIMARY_OUTPUTS,
    )
    assert primary_is_ordinal and not primary_is_cohort_sens

    sens_is_cohort = _is_cohort_definition_sensitivity_step(
        "cohort_definition_sensitivity",
        "05_cohort_definition_sensitivity_comparison",
        _E3_SENSITIVITY_INTENT,
        _E3_SENSITIVITY_OUTPUTS,
    )
    assert sens_is_cohort
