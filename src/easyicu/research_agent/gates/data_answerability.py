"""Outcome-blind, pre-Planner answerability checks for declared predictors.

These checks use typed ``ResearchContext`` facts only.  They never turn a
missing value into an event-absent value, choose an exposure definition, or
infer source coverage from a column name.  Their purpose is to stop an
unestimable contrast before any Planner/Coder provider call.
"""

from __future__ import annotations

from ..schema import ResearchContext, ValidationFinding


def primary_exposure_answerability_findings(
    context: ResearchContext,
) -> list[ValidationFinding]:
    """Return a blocking finding for a proven single-level primary contrast.

    A non-null outcome plus ``primary_exposure`` declares a predictor/outcome
    contrast.  When the observed exposure has fewer than two levels, that
    contrast is not estimable.  Missing rows can rescue the contrast only when
    upstream metadata explicitly defines their semantics; the host must never
    guess that missing means event absence.
    """

    exposure_name = str(context.primary_exposure or "").strip()
    if not exposure_name or not str(context.target_outcome or "").strip():
        return []
    descriptor = context.variable(exposure_name)
    if descriptor is None or not isinstance(descriptor.observed_domain, dict):
        return []
    domain = descriptor.observed_domain
    n_unique = domain.get("n_unique")
    is_constant = domain.get("is_constant") is True
    try:
        fewer_than_two = int(n_unique) < 2
    except (TypeError, ValueError):
        fewer_than_two = False
    if not is_constant and not fewer_than_two:
        return []

    missing = descriptor.missingness
    missing_n = int(missing.n_missing) if missing is not None else 0
    missing_fraction = float(missing.fraction_missing) if missing is not None else 0.0
    missingness_semantics = str(descriptor.missingness_semantics or "").strip()
    if missing_n > 0 and not missingness_semantics:
        kind = "scientifically_infeasible_requires_data_contract"
        message = (
            f"Primary exposure `{exposure_name}` has only one observed level and "
            f"{missing_n} missing rows ({missing_fraction:.1%}), but upstream "
            "metadata does not define whether missing means event absence, "
            "unmeasured, or source-unavailable. The requested contrast cannot be "
            "estimated until a host-owned source/absence contract is supplied."
        )
    elif missing_n == 0:
        kind = "scientifically_infeasible_no_exposure_contrast"
        message = (
            f"Primary exposure `{exposure_name}` has only one observed level and "
            "no missing rows; the requested predictor/outcome contrast is not "
            "estimable in this cohort."
        )
    else:
        # Explicit missingness semantics exist.  A later source-status gate may
        # decide whether those rows create a valid second level; this narrow
        # gate does not parse prose or recode them.
        return []
    return [
        ValidationFinding(
            validator="data_answerability_gate",
            severity="error",
            message=message,
            detail={
                "kind": kind,
                "primary_exposure": exposure_name,
                "target_outcome": context.target_outcome,
                "observed_domain": dict(domain),
                "missing_n": missing_n,
                "missing_fraction": missing_fraction,
                "missingness_semantics_present": bool(missingness_semantics),
                "required_action": (
                    "supply_host_owned_source_absence_contract"
                    if missing_n > 0
                    else "revise_question_or_cohort"
                ),
            },
        )
    ]


__all__ = ["primary_exposure_answerability_findings"]
