"""A target trial claim states one estimate in fixed words, under its assumptions.

An emulated trial answers its question only under assumptions the data cannot
test, so its host claim (``scientific_claim/5``) carries typed terms instead of
free text: the sentence names the assumptions before the estimate, its numbers
come from the claim, and its direction follows the interval.  A strategy's
risk is a percentage that states no direction.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.authority.scientific_claims import (
    ScientificClaim,
    ScientificClaimDraft,
)
from easyicu.research_agent.authority.target_trial_claim_terms import (
    TargetTrialClaimTerms,
    ordinal,
)
from easyicu.research_agent.review.causal_audit import (
    scan_manuscript_for_causal_language,
)

_ASSUMPTIONS = (
    "Under the emulation's assumptions of no unmeasured confounding, positivity, "
    "correctly specified weight models and censoring at ICU exit that the "
    "baseline covariates explain, the emulated target trial"
)


def _terms(**overrides) -> dict:
    terms = {
        "schema_version": "easyicu.target_trial_claim_terms/1",
        "measure": "risk_difference",
        "strategy": None,
        "weighting": "stabilized",
        "initiate_label": "Early vasopressor",
        "defer_label": "No early vasopressor",
        "outcome_label": "Death",
        "horizon_days": 28,
        "analysis_unit_label": "ICU stays",
        "truncation_percentiles": None,
    }
    terms.update(overrides)
    return terms


def _payload(**overrides) -> dict:
    payload = {
        "schema_version": "easyicu.scientific_claim/5",
        "claim_id": "strategy_risk_difference",
        "claim_type": "target_trial_estimate",
        "exposure": "Early vasopressor versus No early vasopressor",
        "outcome": "Death",
        "direction": "negative",
        "estimand": "difference in the risk of death by day 28",
        "population": "the eligible ICU stays",
        "analysis_role": "primary",
        "status": "supported",
        "adjusted_for": ["age", "sex"],
        "point_estimate": -4.0,
        "interval_lower": -7.5,
        "interval_upper": -0.5,
        "confidence_level": 0.95,
        "interval_method": "bootstrap_percentile",
        "effect_scale": "percentage_points",
        "target_trial_terms": _terms(),
    }
    payload.update(overrides)
    return payload


def _claim(**overrides) -> ScientificClaim:
    return ScientificClaim.model_validate(
        {**_payload(**overrides), "step_id": "01_primary", "evidence_id": "summary"}
    )


def test_the_sentence_names_the_assumptions_before_the_estimate() -> None:
    claim = _claim()

    assert claim.render_reader_text() == (
        f"{_ASSUMPTIONS} estimated a difference in the risk of death by day 28 of "
        "-4.00 percentage points (95% CI, -7.50 to -0.50) between the Early "
        "vasopressor and No early vasopressor strategies (Early vasopressor minus "
        "No early vasopressor) in the eligible ICU stays."
    )
    assert claim.render_reader_text(include_estimate=False) == (
        f"{_ASSUMPTIONS} estimated a lower risk of death by day 28 under the Early "
        "vasopressor strategy than under the No early vasopressor strategy in the "
        "eligible ICU stays."
    )
    assert claim.render_text().endswith(
        "(target trial estimate; analysis role: primary)."
    )


def test_truncated_weights_and_ratios_keep_the_same_template() -> None:
    truncated = _claim(
        claim_id="truncated_weight_risk_ratio",
        analysis_role="sensitivity",
        point_estimate=0.82,
        interval_lower=0.70,
        interval_upper=0.97,
        effect_scale="ratio",
        target_trial_terms=_terms(
            measure="risk_ratio",
            weighting="stabilized_truncated",
            truncation_percentiles=[1.0, 99.0],
        ),
    )
    risk = _claim(
        claim_id="initiate_strategy_risk",
        direction="descriptive_only",
        point_estimate=31.2,
        interval_lower=28.0,
        interval_upper=34.5,
        effect_scale="percent",
        target_trial_terms=_terms(measure="strategy_risk", strategy="initiate"),
    )

    assert truncated.render_reader_text().startswith(
        "Under the emulation's assumptions of no unmeasured confounding"
    )
    assert (
        "the emulated target trial, with each strategy's weights truncated at the "
        "1st and 99th percentiles, estimated a risk ratio of 0.820"
    ) in truncated.render_reader_text()
    assert (
        "a risk of death by day 28 of 31.20% (95% CI, 28.00% to 34.50%) under the "
        "Early vasopressor strategy"
    ) in risk.render_reader_text()
    sentences = " ".join(
        claim.render_reader_text(include_estimate=include)
        for claim in (_claim(), truncated, risk)
        for include in (True, False)
    )
    assert (
        scan_manuscript_for_causal_language(
            bound_manuscript=sentences, effect_labels=[]
        )
        == []
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"schema_version": "easyicu.scientific_claim/3"},
            "require scientific_claim/5",
        ),
        ({"claim_type": "association"}, "require scientific_claim/5"),
        ({"target_trial_terms": None}, "require scientific_claim/5"),
        ({"interval_method": "wilson"}, "bootstrap percentile"),
        ({"effect_scale": "ratio"}, "bootstrap percentile interval and its scale"),
        ({"point_estimate": -8.0}, "must contain its finite estimate"),
        ({"analysis_role": "secondary"}, "primary or a sensitivity analysis"),
        ({"direction": "no_clear_association"}, "direction follows its interval"),
        (
            {
                "effect_scale": "ratio",
                "point_estimate": 0.9,
                "interval_lower": 0.0,
                "interval_upper": 1.2,
                "direction": "no_clear_association",
                "target_trial_terms": _terms(measure="risk_ratio"),
            },
            "risk ratio interval is positive",
        ),
        (
            {
                "effect_scale": "percent",
                "direction": "negative",
                "point_estimate": 31.2,
                "interval_lower": 28.0,
                "interval_upper": 34.5,
                "target_trial_terms": _terms(measure="strategy_risk", strategy="defer"),
            },
            "states no direction",
        ),
    ],
)
def test_an_open_or_inconsistent_estimate_is_refused(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        ScientificClaimDraft.model_validate(_payload(**overrides))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"strategy": "initiate"}, "names its strategy, and only it does"),
        ({"weighting": "stabilized_truncated"}, "name their percentiles"),
        (
            {
                "weighting": "stabilized_truncated",
                "truncation_percentiles": [99.0, 1.0],
            },
            "increase inside 0 to 100",
        ),
        (
            {
                "weighting": "stabilized_truncated",
                "truncation_percentiles": [2.5, 97.5],
            },
            "are whole",
        ),
        ({"defer_label": "early vasopressor"}, "different labels"),
        ({"outcome_label": "Death by day 28"}, "String should match pattern"),
    ],
)
def test_the_terms_are_closed(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        TargetTrialClaimTerms.model_validate(_terms(**overrides))


def test_a_percentile_is_named_only_when_whole() -> None:
    assert [ordinal(value) for value in (1.0, 2.0, 3.0, 11.0, 22.0, 99.0)] == [
        "1st",
        "2nd",
        "3rd",
        "11th",
        "22nd",
        "99th",
    ]
    with pytest.raises(ValueError, match="not a whole percentile"):
        ordinal(99.5)
