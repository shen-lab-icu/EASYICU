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


def test_alternative_eligibility_phrase_without_structure_is_not_claimed():
    assert not _is_cohort_definition_sensitivity_step(
        "association",
        "07_robustness",
        "Compare outcomes across alternative eligibility windows.",
        ["table:results"],
    )


def test_cohort_sensitivity_outputs_without_method_owner_are_not_claimed():
    assert not _is_cohort_definition_sensitivity_step(
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


def test_ordinal_matcher_requires_a_closed_primary_product():
    blob, expected_blob = _blob(
        "04_primary_stage_outcome_gradient",
        _E3_PRIMARY_INTENT,
        _E3_PRIMARY_OUTPUTS,
    )
    assert not _ordinal_dose_response_step_matches(
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


def test_ordinal_matcher_rejects_stage_workflow_prose_and_generic_trend_method():
    assert not _ordinal_dose_response_step_matches(
        "mixed_effects_regression",
        "report records per stage of cohort construction",
        "table:cohort_flow",
    )
    assert not _ordinal_dose_response_step_matches(
        "trend_analysis",
        "summarize temporal records",
        "table:descriptive_summary",
    )


def test_legacy_science_helpers_do_not_claim_an_unclosed_primary_step():
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
    assert not primary_is_ordinal and not primary_is_cohort_sens

    sens_is_cohort = _is_cohort_definition_sensitivity_step(
        "cohort_definition_sensitivity",
        "05_cohort_definition_sensitivity_comparison",
        _E3_SENSITIVITY_INTENT,
        _E3_SENSITIVITY_OUTPUTS,
    )
    assert sens_is_cohort


# --- hybrid "<head>_with_<rider>" method routing (E3 4th blocker) -------------
# A stochastic E3 run merged the primary ordinal result with the sensitivity
# comparison into ONE step whose method was "association_with_cohort_sensitivity"
# and whose outputs included "table:sensitivity_comparison_grid". Two predicate
# gaps mis-routed it to the cohort-sensitivity runner (which blocked): (1) the
# output token "sensitivity_comparison" substring-matched the grid output and
# over-claimed it; (2) the ordinal matcher required exact method membership, so
# the "association" head was not recognised. The fix parses the method HEAD (the
# part before "_with_"): a primary head keeps its primary runner; a
# definition-sensitivity head keeps the cohort-sensitivity runner.

# the real merged step's declared outputs (verbatim shape)
_E3_MERGED_OUTPUTS = [
    "table:absolute_outcome_risk_by_stage",
    "table:adjusted_death_trend_model",
    "table:sensitivity_comparison_grid",
    "statistic:primary_or",
    "table:robustness_summary",
]


def test_hybrid_association_with_cohort_sensitivity_is_not_a_definition_step():
    # head = "association" (a primary head) -> NOT a cohort-definition comparison
    assert not _is_cohort_definition_sensitivity_step(
        "association_with_cohort_sensitivity",
        "05_stage_outcomes_and_sensitivity_comparison",
        "Estimate the per-stage dose-response gradient and compare across "
        "pre-specified sensitivity specifications.",
        _E3_MERGED_OUTPUTS,
    )


def test_hybrid_association_without_closed_ordinal_product_is_not_claimed():
    # head = "association" (in the ordinal primary set) + a dose-response signal
    # present -> the ordinal runner owns the primary part
    blob, expected_blob = _blob(
        "05_stage_outcomes_and_sensitivity_comparison",
        "Estimate the per-stage dose-response gradient (ordinal trend) for "
        "mortality and compare across sensitivity specifications.",
        _E3_MERGED_OUTPUTS,
    )
    assert not _ordinal_dose_response_step_matches(
        "association_with_cohort_sensitivity", blob, expected_blob
    )


def test_definition_sensitivity_head_still_claims_cohort_sensitivity():
    # head = "cohort_definition_sensitivity" (a definition-sensitivity head) ->
    # a hybrid whose PRIMARY intent IS the definition comparison stays with the
    # cohort-sensitivity runner
    assert _is_cohort_definition_sensitivity_step(
        "cohort_definition_sensitivity_with_binomial_glm",
        "05_definition_sensitivity",
        "Re-run across alternative eligibility definitions with a binomial GLM.",
        ["table:sensitivity_grid"],
    )


def test_definition_sensitivity_head_does_not_match_ordinal():
    # even with a dose-response signal, a definition-sensitivity head is not the
    # primary ordinal estimand
    blob, expected_blob = _blob(
        "05_definition_sensitivity",
        "Re-run the per-stage dose-response summaries across alternative "
        "eligibility definitions.",
        ["table:sensitivity_grid"],
    )
    assert not _ordinal_dose_response_step_matches(
        "cohort_definition_sensitivity_with_binomial_glm", blob, expected_blob
    )


def test_sensitivity_comparison_grid_output_no_longer_over_claims():
    # the merged step's "table:sensitivity_comparison_grid" output alone must not
    # make a primary step read as a definition comparison
    assert not _is_cohort_definition_sensitivity_step(
        "association",
        "05_stage_outcomes_and_sensitivity_comparison",
        "Primary per-stage gradient with a sensitivity comparison grid.",
        ["table:sensitivity_comparison_grid", "statistic:primary_or"],
    )


def test_sensitivity_grid_without_exact_method_does_not_signal_ownership():
    assert not _is_cohort_definition_sensitivity_step(
        "robustness_analysis",
        "06_robustness",
        "Compare results across specifications.",
        ["table:sensitivity_grid", "table:overlap_and_movement_across_cohorts"],
    )
