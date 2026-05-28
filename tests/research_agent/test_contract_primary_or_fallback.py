from __future__ import annotations

import math


def _association_step(ra, step_id: str = "04_adjusted_association"):
    return ra.AnalysisStep(
        step_id=step_id,
        intent="Estimate the adjusted association between SOFA-2 and mortality.",
        expected_outputs=[
            "model:logistic_regression_sofa_death",
            "statistic:adjusted_odds_ratio",
        ],
    )


def _errors(findings):
    return [finding for finding in findings if finding.severity == "error"]


def test_accepts_statistic_adjusted_sofa2_odds_ratio_dict(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    findings = _step_contract_findings(
        step=_association_step(ra, step_id="03b_event_count_check"),
        step_summary={
            "statistic:adjusted_sofa2_odds_ratio": {
                "adjusted_odds_ratio": 1.3114628301617859,
                "ci_lower": 1.1779552418640442,
                "ci_upper": 1.46010195784202,
                "standard_error": 0.054778092768164525,
                "model_converged": True,
            }
        },
    )

    assert _errors(findings) == []


def test_accepts_primary_estimates_list_first_finite_or(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    findings = _step_contract_findings(
        step=_association_step(ra),
        step_summary={
            "primary_estimates": [
                {
                    "stratum": "overall",
                    "odds_ratio": 1.42,
                    "ci_lower": 1.10,
                    "ci_upper": 1.83,
                }
            ]
        },
    )

    assert _errors(findings) == []


def test_accepts_generic_primary_statistic_value_with_ci(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    findings = _step_contract_findings(
        step=_association_step(ra),
        step_summary={
            "statistic:primary_adjusted_or": {
                "value": 1.18,
                "ci_low": 1.02,
                "ci_high": 1.36,
            }
        },
    )

    assert _errors(findings) == []


def test_prior_successful_primary_effect_satisfies_later_primary_requirement(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    findings = _step_contract_findings(
        step=_association_step(ra),
        step_summary={
            "skipped": [
                {
                    "variables": ["sofa_factor", "weight_missing_indicator"],
                    "reason": "column not present",
                }
            ],
            "notes": [],
        },
        completed_step_records=[
            {
                "step_id": "03b_event_count_check",
                "status": "ok",
                "step_summary": {
                    "statistic:adjusted_sofa2_odds_ratio": {
                        "adjusted_odds_ratio": 1.3114628301617859,
                        "ci_lower": 1.1779552418640442,
                        "ci_upper": 1.46010195784202,
                    }
                },
            }
        ],
    )

    assert _errors(findings) == []
    assert any(
        finding.severity == "warning"
        and finding.detail["fallback_step_id"] == "03b_event_count_check"
        for finding in findings
    )


def test_rejects_nonfinite_primary_effect_values(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    for value in (math.nan, math.inf, -math.inf):
        findings = _step_contract_findings(
            step=_association_step(ra),
            step_summary={
                "statistic:adjusted_sofa2_odds_ratio": {
                    "adjusted_odds_ratio": value,
                    "ci_lower": 1.1779552418640442,
                    "ci_upper": 1.46010195784202,
                },
                "primary_estimates": [
                    {
                        "odds_ratio": value,
                        "ci_lower": 1.10,
                        "ci_upper": 1.83,
                    }
                ],
            },
        )

        assert _errors(findings)

