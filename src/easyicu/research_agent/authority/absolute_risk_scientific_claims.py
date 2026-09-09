"""Compile descriptive frequencies from the sealed absolute-risk result.

This adapter does not promote the source table's binomial intervals into
inferential authority: repeated records may require a different variance
contract. The registered counts, denominator and population support a
descriptive frequency independently of that question.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping


def _count(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"absolute-risk reporting {field} must be a nonnegative integer"
        )
    return value


def derive_absolute_risk_claim_payloads(
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Read only the native /1 reporting contract, never arbitrary summary prose."""

    reporting = summary.get("reportable_descriptive_results")
    if not isinstance(reporting, Mapping):
        raise ValueError("absolute-risk reporting contract is missing")
    if (
        reporting.get("schema_version") != "easyicu.absolute_risk_reporting/1"
        or reporting.get("execution_owner") != "absolute_risk_context_executor_v1"
        or reporting.get("interpretation_ceiling") != "descriptive_not_causal"
        or summary.get("analysis_family") != "absolute_risk_context"
        or summary.get("status") != "ok"
        or summary.get("adjusted_effect") is not None
    ):
        raise ValueError("absolute-risk reporting authority is inconsistent")
    outcome = summary.get("outcome")
    result = reporting.get("overall_outcome")
    if (
        not isinstance(outcome, str)
        or not outcome.strip()
        or not isinstance(result, Mapping)
    ):
        raise ValueError("absolute-risk reporting requires an explicit outcome")
    if result.get("outcome") != outcome:
        raise ValueError("absolute-risk reporting outcome drifted")
    total = _count(summary.get("n_total"), "n_total")
    n = _count(result.get("n"), "overall_outcome.n")
    events = _count(result.get("event_n"), "overall_outcome.event_n")
    missing = _count(summary.get("outcome_missing_n"), "outcome_missing_n")
    nonmissing = _count(summary.get("outcome_nonmissing_n"), "outcome_nonmissing_n")
    if events > n or n + missing != total or nonmissing != n:
        raise ValueError("absolute-risk reporting denominators are inconsistent")

    population = "the bound analysis records with an observed outcome"
    group = "bound analysis records with an observed outcome"
    binding = summary.get("population_binding")
    if binding is not None:
        if not isinstance(binding, Mapping) or (
            binding.get("schema_version") != "easyicu.primary_population_descriptive/1"
            or binding.get("scope") != "primary_model_complete_cases"
            or not re.fullmatch(
                r"scientific_runtime_contract:[a-f0-9]{64}",
                str(binding.get("owner_ref") or ""),
            )
            or not re.fullmatch(
                r"[a-f0-9]{64}", str(binding.get("runtime_projection_sha256") or "")
            )
        ):
            raise ValueError("absolute-risk reporting population authority is invalid")
        population_n = _count(
            binding.get("population_n"), "population_binding.population_n"
        )
        source_n = _count(
            binding.get("source_cohort_n"), "population_binding.source_cohort_n"
        )
        event_n = _count(binding.get("event_n"), "population_binding.event_n")
        if population_n != n or total != n or source_n < n or event_n != events:
            raise ValueError("absolute-risk reporting primary population drifted")
        population = "the primary model complete-case records"
        group = "primary model complete-case records"

    raw_risk = result.get("risk_pct")
    if n == 0:
        if raw_risk is not None:
            raise ValueError(
                "absolute-risk reporting empty denominator cannot have a risk"
            )
        return []
    if isinstance(raw_risk, bool) or not isinstance(raw_risk, (int, float)):
        raise ValueError("absolute-risk reporting risk must be numeric")
    risk = float(raw_risk)
    if not math.isfinite(risk) or not math.isclose(
        risk, 100.0 * events / n, rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError("absolute-risk reporting frequency does not match its counts")
    return [
        {
            "schema_version": "easyicu.scientific_claim/2",
            "claim_id": "observed_outcome_frequency",
            "claim_type": "descriptive_absolute_risk",
            "exposure": group,
            "outcome": outcome,
            "direction": "descriptive_only",
            "estimand": (
                f"observed outcome frequency was {risk:.6g}% "
                f"({events} events among {n} records; counts only, no confidence interval)"
            ),
            "population": population,
            "analysis_role": "auxiliary",
            "status": "supported",
            "adjusted_for": [],
        }
    ]
