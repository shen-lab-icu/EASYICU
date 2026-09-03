"""Routing test for the deterministic ordinal dose-response runner.

E3's FIRST bench routed its primary step to the LLM coder instead of the ordinal
runner: the planner labelled the step ``method="association_analysis"`` (with an
``ordinal_trend_tests`` output and a "dose-response gradient" question), but the
preflight predicate's method allowlist only had ``"association"``. The routing
logic was trapped in a closure, unverifiable without a 40-minute bench.

The logic is now a pure function, ``_ordinal_dose_response_step_matches``. These
tests lock the two properties that matter: it FIRES on a real dose-response
primary step (however the planner phrases the method), and it does NOT hijack a
plain association / prep step that lacks a dose-response signal.
"""

from __future__ import annotations

from easyicu.research_agent.execution.phase import (
    _ordinal_dose_response_step_matches as _matches,
)


def test_descriptive_ordinal_trend_output_is_not_a_primary_model_contract():
    assert (
        _matches(
            "association_analysis",
            "05_stage_stratified_outcomes_and_trend perform ordinal trend tests "
            "dose-response gradient of kdigo stage against mortality "
            "table:ordinal_trend_tests",
            "stage_stratified_outcomes ordinal_trend_tests adjusted_mortality_trend",
        )
        is False
    )


def test_narrative_dose_signal_cannot_replace_a_closed_product():
    # E3's real hybridfix bench (2026-07-07): the planner labelled the PRIMARY
    # ordinal step method="multivariable_association" for a "dose-response gradient
    # ... ORDERED categorical exposure" question. The dose signal was present (from
    # the research question) and the cohort-sensitivity veto was correctly False,
    # but the exact-match method allowlist missed "multivariable_association", so
    # the deterministic ordinal runner never claimed the step -> no dose_response
    # table -> no traceable primary figure -> fail closed. Token-matching fixes it.
    assert (
        _matches(
            "multivariable_association",
            "04_adjusted_gradient_models estimate adjusted associations of kdigo "
            "stage; characterise the dose-response gradient; ordered categorical "
            "exposure; monotonic-trend modelling choice on an ordinal exposure",
            "adjusted_effect_estimates model_population_and_missingness "
            "primary_vs_expanded_adjustment_comparison model_diagnostics",
        )
        is False
    )


def test_multivariable_association_without_dose_signal_is_not_hijacked():
    # The token fix must NOT relax the anti-hijack guard: a multivariable
    # association with NO dose-response signal is a plain association and must
    # fall through to its own runner, not the ordinal one.
    assert (
        _matches(
            "multivariable_association",
            "estimate the adjusted association between vasopressor exposure and "
            "in-hospital mortality with a multivariable logistic model",
            "adjusted_effect_estimates model_summary forest_plot",
        )
        is False
    )


def test_dose_response_question_without_closed_product_is_not_claimed():
    assert (
        _matches(
            "association_analysis",
            "characterise the dose-response gradient of aki stage against mortality",
            "stage_outcomes model_estimates",
        )
        is False
    )


def test_explicit_dose_response_method_still_requires_a_closed_product():
    assert _matches("dose_response", "a primary step", "some_outputs") is False
    assert _matches("dose_response", "a primary step", "table:dose_response") is True
    # Ordinal logistic regression can model an ordinal *outcome* and therefore
    # is not, by itself, evidence of a graded-exposure/dose-response task.
    assert _matches("ordinal_logistic_regression", "step", "outs") is False


def test_matches_declared_per_stage_output_alone():
    assert _matches("regression", "step", "per_stage_odds_ratios forest") is True


def test_does_not_hijack_a_plain_association_step():
    # a plain association primary with NO dose-response signal must NOT route here
    assert (
        _matches(
            "association_analysis",
            "estimate the association between vasopressor use and mortality",
            "primary_effect model_summary forest_plot",
        )
        is False
    )


def test_ignores_a_bare_trend_mention_without_a_dose_signal():
    assert (
        _matches(
            "descriptive",
            "report the temporal trend of icu admissions over the study period",
            "admissions_by_year",
        )
        is False
    )


def test_does_not_fire_on_a_cohort_definition_step():
    # E3 step 01: a cohort-definition/prep step must fall through to its own runner
    assert (
        _matches(
            "cohort_definition",
            "define the primary adult icu cohort keep exposure aligned aki_stage_max",
            "cohort_flow_attrition exposure_definition_audit",
        )
        is False
    )


def test_narrative_dose_signal_needs_a_primary_estimation_method():
    # A dose-response NARRATIVE signal on a non-estimation method (e.g. a prep
    # verb) must NOT route here — the estimation-method gate still applies. This
    # is the anti-hijack guard for the dose_signal branch.
    assert (
        _matches(
            "cohort_definition",
            "define the cohort for the dose-response gradient study",
            "cohort_flow_attrition",
        )
        is False
    )
    # A compatible method still needs a closed ordinal result product.
    assert (
        _matches(
            "association_analysis",
            "define the cohort for the dose-response gradient study",
            "cohort_flow_attrition",
        )
        is False
    )
    assert _matches(
        "association_analysis",
        "define the cohort for the dose-response gradient study",
        "table:trend_or",
    )


def test_fresh_e3_exposure_qc_is_not_hijacked_as_primary_model():
    assert not _matches(
        "ordinal_exposure_derivation_and_quality_control",
        "04 exposure derivation qc for an ordinal dose-response study",
        "table:kdigo_stage_distribution table:kdigo_component_qc",
    )


def test_fresh_e3_descriptive_trend_is_not_hijacked_as_primary_model():
    assert not _matches(
        "ordinal_stratified_descriptive_analysis",
        "05 report stage-stratified mortality and an ordinal trend test",
        "table:stage_stratified_outcomes test:ordinal_trend",
    )


def test_fresh_e3_supportive_adjusted_association_is_not_primary_model():
    assert not _matches(
        "adjusted_association_models",
        "06_secondary_adjusted_association fit supportive adjusted "
        "associations for an ordered exposure",
        "table:adjusted_association_estimates "
        "table:adjustment_set_and_analytic_population",
    )


def test_generic_adjusted_association_is_not_ordinal_without_closed_product():
    assert not _matches(
        "adjusted_association_models",
        "06_primary_adjusted_association fit the primary adjusted association "
        "for an ordered exposure",
        "table:adjusted_association_estimates "
        "table:adjustment_set_and_analytic_population",
    )
    assert _matches(
        "adjusted_association_models",
        "fit the primary ordered-exposure model",
        "table:trend_or",
    )


def test_secondary_or_sensitivity_prose_cannot_create_ownership():
    for role in ("secondary", "supportive", "sensitivity"):
        assert not _matches(
            "logistic_regression",
            f"{role} model of an ordered exposure severity gradient",
            "table:adjusted_association_estimates",
        )
