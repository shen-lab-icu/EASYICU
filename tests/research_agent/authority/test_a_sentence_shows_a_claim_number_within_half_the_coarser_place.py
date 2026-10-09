"""A sentence shows a claim's number only in the claim's unit, within half the
coarser displayed place, and each distinct number by a different number.

Synthetic displays and synthetic adapter summaries; each case names the part
of the rule it holds.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.authority.absolute_risk_scientific_claims import (
    derive_absolute_risk_claim_payloads,
)
from easyicu.research_agent.authority.reported_numbers import (
    FACT_NUMBER_MISSING,
    FACT_PRECISION_TOO_COARSE,
    FACT_UNIT_MISMATCH,
    ReportedNumber,
    bind_reported_numbers,
    sentence_numbers,
)
from easyicu.research_agent.authority.scientific_claims import (
    ScientificClaim,
    bind_scientific_claim_drafts,
    derive_scientific_claim_drafts,
)


def _percent(text: str) -> ReportedNumber:
    return ReportedNumber.displayed(text, "percent")


def _count(text: str) -> ReportedNumber:
    return ReportedNumber.displayed(text, "count")


def _unbound(stated, sentence: str) -> list[tuple[str, str]]:
    binding = bind_reported_numbers(stated, sentence_numbers(sentence))
    return [(str(number.value), reason) for number, reason in binding.unbound]


@pytest.mark.parametrize("sentence", ["(12.35%)", "(12.34%)"])
def test_a_display_half_its_place_away_shows_the_value(sentence: str) -> None:
    # 12.345 rounds to either neighbour at two places: the edge is included.
    assert _unbound([_percent("12.345")], sentence) == []


def test_a_finer_display_shows_a_coarser_value_within_half_the_coarser_place() -> None:
    assert _unbound([_percent("12.3")], "(12.33%)") == []
    assert _unbound([_percent("12.3")], "(12.36%)") == [("12.3", FACT_NUMBER_MISSING)]


def test_a_coarse_display_shows_no_value_it_rounds_away() -> None:
    assert _unbound([_percent("0.004")], "(0.0%)") == [
        ("0.004", FACT_PRECISION_TOO_COARSE)
    ]
    assert _unbound([_percent("0.004")], "(0.00%)") == [
        ("0.004", FACT_PRECISION_TOO_COARSE)
    ]
    assert _unbound([_percent("9.876543")], "(10%)") == [
        ("9.876543", FACT_PRECISION_TOO_COARSE)
    ]


def test_a_zero_shows_an_exact_zero() -> None:
    assert (
        _unbound([_count("0"), _percent("0")], "was 0 of 120 observations (0.00%)")
        == []
    )


def test_a_number_in_another_unit_shows_no_percent() -> None:
    assert _unbound([_percent("12.5")], "中位 12.5 天") == [
        ("12.5", FACT_UNIT_MISMATCH)
    ]
    assert _unbound([_percent("5")], "a difference of 5.00 percentage points") == [
        ("5", FACT_UNIT_MISMATCH)
    ]


def test_a_count_is_shown_only_by_the_equal_integer() -> None:
    assert _unbound([_count("12345")], "was 12,345 observations") == []
    assert _unbound([_count("12")], "for 12.0 days") == [("12", FACT_UNIT_MISMATCH)]
    assert _unbound([_count("12")], "in 12% of stays") == [("12", FACT_UNIT_MISMATCH)]
    assert _unbound([_count("12")], "in 13 stays") == [("12", FACT_NUMBER_MISSING)]


def test_each_distinct_claim_number_needs_its_own_sentence_number() -> None:
    assert _unbound([_percent("12.341"), _percent("12.344")], "(12.34%)") == [
        ("12.344", FACT_NUMBER_MISSING)
    ]
    # A number the claim states twice is shown once.
    assert (
        _unbound(
            [_count("12"), _count("12"), _percent("100")],
            "12 of 12 observations (100.00%)",
        )
        == []
    )


def test_a_hyphen_between_terms_is_not_a_minus_sign() -> None:
    shown = sentence_numbers(
        "95% CI, 10.12-14.57; under Sepsis-3; a change of −0.5 and -1.2"
    )

    assert [(str(number.value), number.unit) for number in shown] == [
        ("95", "percent"),
        ("10.12", ""),
        ("14.57", ""),
        ("3", ""),
        ("-0.5", ""),
        ("-1.2", ""),
    ]


def test_names_and_identifiers_state_no_number() -> None:
    shown = sentence_numbers(
        "Observed “28-day mortality” in the “Sepsis-3” group "
        "(sofa2_max, claim_5) was 4 of 1,200"
    )

    assert [str(number.value) for number in shown] == ["4", "1200"]


def test_a_number_inside_a_quoted_name_never_shows_a_claim_number() -> None:
    stated = [_count("3"), _count("200"), _percent("1.5")]

    assert _unbound(
        stated,
        "Observed “death” in the “Sepsis-3” group was 4 of 200 observations (1.50%)",
    ) == [("3", FACT_NUMBER_MISSING)]


def _distribution_claims() -> list[ScientificClaim]:
    def row(level: int, count: int, denominator: int, *, events: bool = False):
        return {
            "level_index": level,
            "level": level,
            "events" if events else "n": count,
            "denominator": denominator,
            "estimate_pct": 100 * count / denominator,
            "interval_method": "none_counts_only",
            "covariance": "none_counts_only",
            "ci_low_pct": None,
            "ci_high_pct": None,
            "confidence_level": None,
            "standard_error_pct": None,
            "cluster_count": None,
        }

    summary = {
        "status": "ok",
        "interpretation_class": "exposure_outcome_distribution",
        "analysis_role": "primary",
        "analysis_set": "bound_typed_cohort",
        "interpretation_ceiling": "descriptive_unadjusted_not_causal",
        "adjusted_effect": None,
        "interval_method": "none_counts_only",
        "cohort_n": 3001,
        "exposure": "exposure",
        "outcome": "outcome",
        "descriptive_estimates": {
            "schema_version": "easyicu.exposure_outcome_descriptive_estimates/1",
            "analysis_role": "primary",
            "analysis_set": "bound_typed_cohort",
            "interpretation_ceiling": "descriptive_unadjusted_not_causal",
            "exposure_prevalence": [row(0, 2470, 3001), row(1, 531, 3001)],
            "outcome_absolute_risks": [
                row(0, 247, 2470, events=True),
                row(1, 71, 531, events=True),
            ],
            "risk_difference": None,
            "dependence": None,
        },
    }
    drafts = derive_scientific_claim_drafts(summary)
    return bind_scientific_claim_drafts(
        [draft.model_dump() for draft in drafts],
        step_id="distribution",
        evidence_id="summary",
    )


def test_each_counts_only_form_states_its_counts_and_recorded_percent() -> None:
    whole, partial = _distribution_claims()
    frequency = bind_scientific_claim_drafts(
        derive_absolute_risk_claim_payloads(
            {
                "analysis_family": "absolute_risk_context",
                "status": "ok",
                "adjusted_effect": None,
                "outcome": "death",
                "n_total": 120,
                "outcome_missing_n": 0,
                "outcome_nonmissing_n": 120,
                "reportable_descriptive_results": {
                    "schema_version": "easyicu.absolute_risk_reporting/1",
                    "execution_owner": "absolute_risk_context_executor_v1",
                    "interpretation_ceiling": "descriptive_not_causal",
                    "overall_outcome": {
                        "outcome": "death",
                        "n": 120,
                        "event_n": 7,
                        "risk_pct": 100 * 7 / 120,
                    },
                },
            }
        ),
        step_id="risk",
        evidence_id="risk_summary",
    )[0]

    assert [
        (str(number.value), number.unit) for number in partial.reported_numbers()
    ] == [("71", "count"), ("531", "count"), ("13.370998", "percent")]
    assert [
        (str(number.value), number.unit) for number in frequency.reported_numbers()
    ] == [("7", "count"), ("120", "count"), ("5.83333", "percent")]
    # The adapter records 10.000000 as "10": a sentence must show 10.00%.
    assert (
        _unbound(whole.reported_numbers(), "247 of 2,470 observations (10.00%)") == []
    )
    assert _unbound(whole.reported_numbers(), "247 of 2,470 observations (10.40%)") == [
        ("10", FACT_NUMBER_MISSING)
    ]
    assert _unbound(partial.reported_numbers(), "71 of 531 observations (13.37%)") == []
    assert _unbound(frequency.reported_numbers(), "7 of 120 records (5.83%)") == []


def test_a_claim_whose_numbers_are_not_typed_states_none() -> None:
    common = {
        "claim_id": "c",
        "exposure": "lactate",
        "outcome": "death",
        "population": "p",
        "analysis_role": "primary",
        "status": "supported",
        "step_id": "s",
        "evidence_id": "e",
    }
    association = ScientificClaim(
        **common,
        schema_version="easyicu.scientific_claim/1",
        claim_type="association",
        direction="positive",
        estimand="adjusted odds ratio",
        point_estimate=1.2,
        interval_lower=1.1,
        interval_upper=1.3,
    )
    interval = ScientificClaim(
        **common,
        schema_version="easyicu.scientific_claim/3",
        claim_type="descriptive_absolute_risk",
        direction="descriptive_only",
        estimand="observed absolute risk",
        point_estimate=10.0,
        interval_lower=8.0,
        interval_upper=12.0,
        confidence_level=0.95,
        interval_method="wilson",
        effect_scale="percent",
    )
    untyped = ScientificClaim(
        **common,
        schema_version="easyicu.scientific_claim/2",
        claim_type="descriptive_absolute_risk",
        direction="descriptive_only",
        estimand="observed absolute risk was about one in ten",
    )

    assert association.reported_numbers() is None
    assert interval.reported_numbers() is None
    assert untyped.reported_numbers() is None
