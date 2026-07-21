"""Outcome-blind, pre-Planner answerability checks for declared predictors.

These checks use typed ``ResearchContext`` facts only.  They never turn a
missing value into an event-absent value, choose an exposure definition, or
infer source coverage from a column name.  Their purpose is to stop an
unestimable contrast before any Planner/Coder provider call.
"""

from __future__ import annotations

from ..authority.source_status import (
    SourceStatusContractError,
    source_status_contract_digest,
    source_status_contract_from_context,
)
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
    try:
        source_contract = source_status_contract_from_context(
            context,
            variable=exposure_name,
        )
    except SourceStatusContractError as exc:
        return [
            ValidationFinding(
                validator="data_answerability_gate",
                severity="error",
                message=(
                    f"Primary exposure `{exposure_name}` has an invalid host-owned "
                    f"source-status contract: {exc}"
                ),
                detail={
                    "kind": "source_status_contract_invalid",
                    "primary_exposure": exposure_name,
                    "validation_error": str(exc),
                },
            )
        ]
    if source_contract is not None:
        counts = source_contract.counts
        expected_nonmissing = source_contract.n_total - missing_n
        missing_partition = (
            counts.verified_absent
            + counts.unmeasured
            + counts.source_missing
            + counts.contradictory
        )
        binding_issues: list[str] = []
        if counts.observed != expected_nonmissing:
            binding_issues.append("observed count disagrees with variable nonmissing_n")
        if missing_partition != missing_n:
            binding_issues.append(
                "non-observed source-status counts disagree with missing_n"
            )
        if counts.contradictory:
            binding_issues.append("contract contains contradictory rows")
        if binding_issues:
            return [
                ValidationFinding(
                    validator="data_answerability_gate",
                    severity="error",
                    message=(
                        f"Primary exposure `{exposure_name}` source-status authority "
                        "does not reconcile with the locked ResearchContext."
                    ),
                    detail={
                        "kind": "source_status_contract_binding_mismatch",
                        "primary_exposure": exposure_name,
                        "issues": binding_issues,
                        "contract_sha256": source_status_contract_digest(
                            source_contract
                        ),
                    },
                )
            ]
        if counts.observed > 0 and counts.verified_absent > 0:
            return [
                ValidationFinding(
                    validator="data_answerability_gate",
                    severity="error",
                    message=(
                        f"Primary exposure `{exposure_name}` has verified-absence "
                        "authority, but the locked exposure column still contains "
                        "a single observed level. The host must materialize the "
                        "verified-absent rows before scientific planning."
                    ),
                    detail={
                        "kind": "source_status_contract_not_materialized",
                        "primary_exposure": exposure_name,
                        "contract_sha256": source_status_contract_digest(
                            source_contract
                        ),
                        "row_status_artifact_sha256": (
                            source_contract.row_status_artifact_sha256
                        ),
                        "row_identity_sha256": source_contract.row_identity_sha256,
                        "verified_absent_n": counts.verified_absent,
                        "required_action": (
                            "host_materialize_verified_absence_into_bound_exposure"
                        ),
                    },
                )
            ]
    if missing_n > 0:
        kind = "scientifically_infeasible_requires_data_contract"
        if source_contract is None:
            message = (
                f"Primary exposure `{exposure_name}` has only one observed level and "
                f"{missing_n} missing rows ({missing_fraction:.1%}), but upstream "
                "metadata does not define whether missing means event absence, "
                "unmeasured, or source-unavailable. The requested contrast cannot "
                "be estimated until a host-owned source/absence contract is "
                "supplied."
            )
        else:
            message = (
                f"Primary exposure `{exposure_name}` has only one observed level. "
                "Its host-owned source-status contract does not establish a usable "
                "verified-absence contrast for the locked cohort."
            )
    elif missing_n == 0:
        kind = "scientifically_infeasible_no_exposure_contrast"
        message = (
            f"Primary exposure `{exposure_name}` has only one observed level and "
            "no missing rows; the requested predictor/outcome contrast is not "
            "estimable in this cohort."
        )
    else:
        raise AssertionError("unreachable answerability state")
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
                "typed_source_status_contract_present": source_contract is not None,
                "required_action": (
                    "supply_host_owned_source_absence_contract"
                    if missing_n > 0 and source_contract is None
                    else (
                        "revise_source_contract_or_materialize_exposure"
                        if missing_n > 0
                        else "revise_question_or_cohort"
                    )
                ),
            },
        )
    ]


__all__ = ["primary_exposure_answerability_findings"]
