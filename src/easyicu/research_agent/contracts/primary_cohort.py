"""Typed ownership and routing contract for the locked primary cohort."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schema import AnalysisStep, ValidationFinding
from .product_identity import (
    normalize_product_token as _normalise,
    typed_product,
)

_PRIMARY_ANALYSIS_COHORT_METHODS = frozenset(
    {
        "cohort_construction",
        "cohort_definition",
        "cohort_definition_and_attrition",
        "cohort_definition_with_attrition",
        "eligibility_definition",
        "primary_cohort_definition",
    }
)
_PRIMARY_ANALYSIS_COHORT_DATA_KINDS = frozenset({"artifact", "dataset", "table"})
_PRIMARY_ANALYSIS_COHORT_FLOW_PRODUCTS = frozenset(
    {
        "attrition",
        "attrition_by_rule",
        "cohort_attrition",
        "cohort_denominator",
        "cohort_denominators",
        "cohort_flow",
        "eligibility_flow",
    }
)


def _is_primary_analysis_cohort_method(value: object) -> bool:
    """Recognise a case-neutral primary cohort construction method family.

    Planner method labels are agent-authored and may use an equivalent phrase
    instead of one host spelling.  Match the method's semantic tokens while
    excluding sensitivity/robustness riders; typed outputs and the locked plan
    still decide whether the step may own the primary cohort.
    """

    method = _normalise(value)
    if method in _PRIMARY_ANALYSIS_COHORT_METHODS:
        return True
    tokens = frozenset(part for part in method.split("_") if part)
    if tokens & {
        "external",
        "matched",
        "overlap",
        "robustness",
        "secondary",
        "sensitivity",
        "subgroup",
        "validation",
    }:
        return False
    return bool(tokens & {"cohort", "eligibility"}) and bool(
        tokens & {"construction", "definition", "filter"}
    )


def _is_primary_analysis_cohort_flow_product(name: object) -> bool:
    """Recognise a case-neutral cohort attrition/flow product.

    Product names are agent-authored for the same reason method labels are, and
    the two halves of one predicate must tolerate the same variation. The method
    half above has taken tokens since it was written; this half required one of
    seven exact spellings, and the asymmetry decided which steps received the
    host's flow contract and which were checked by
    :func:`primary_analysis_cohort_integrity_findings` at all.

    Measured 2026-07-30 over 314 recorded real steps: ``table:attrition_flow``
    was declared five times and matched none of the seven, along with
    ``analytic_set_attrition``, ``complete_case_attrition`` and
    ``primary_model_complete_case_attrition_reconciled``.  Those steps were
    never told the canonical column contract and never reached the 40 arithmetic
    checks that enforce it.  One of them shipped a flow table reporting
    ``n_at_start_rows`` 94,458 and ``n_excluded_rows`` 0 alongside
    ``n_remaining_rows`` 60,461 -- a patient-flow diagram that does not
    subtract -- and the step was recorded ``ok``.

    ``attrition`` alone carries the meaning; ``flow`` and ``denominator`` need a
    population word beside them, because a flow of something else is not this.
    A cohort *summary*, *prevalence*, or *reconciliation audit* is a different
    product and stays out: those three appear 25 times in the same corpus and
    none is an attrition cascade.
    """

    product = _normalise(name)
    tokens = frozenset(part for part in product.split("_") if part)
    if not tokens:
        return False
    if "attrition" in tokens:
        return True
    return bool(tokens & {"flow", "denominator", "denominators"}) and bool(
        tokens & {"cohort", "eligibility", "analytic", "analysis"}
    )


def _primary_analysis_cohort_product(raw: object) -> tuple[str, str] | None:
    """Return one exact primary-cohort product identity.

    ``analysis_cohort`` is the legacy closed product across its supported
    physical aliases. Any non-empty product in the explicit Planner-facing
    ``cohort:`` namespace is also a closed population identity candidate; the
    plan-level cohort name and surrounding method/output contract decide whether
    it owns the primary cohort. A model-specific ``table:analysis_set`` or
    ``artifact:analysis_set`` must not become a primary population merely
    because a cohort/attrition step exists nearby.
    """

    raw_kind, separator, raw_product = str(raw or "").strip().partition(":")
    raw_product = raw_product.strip()
    if (
        not separator
        or not raw_product
        or Path(raw_product).name != raw_product
        or "/" in raw_product
        or "\\" in raw_product
    ):
        return None
    parsed = typed_product(raw)
    if parsed is None:
        return None
    kind, name = parsed
    if kind not in _PRIMARY_ANALYSIS_COHORT_DATA_KINDS:
        return None
    if name == "analysis_cohort":
        return parsed
    if _normalise(raw_kind) == "cohort":
        return parsed
    return None


def reserved_primary_cohort_product(raw: object) -> tuple[str, str] | None:
    """Return the identity of one globally reserved primary-cohort product.

    ``analysis_cohort`` is the legacy closed name and ``cohort:analysis_set``
    is its explicit Planner-facing spelling; both denote the single population
    the run's cohort authority locked.  Every host surface that decides *which
    population a step sees* must ask here instead of comparing against one
    spelling: a surface that recognises only one of them silently hands the
    step a different population than the surface next to it, which is how a
    development sample gets bypassed by the typed-input plane while the
    run-level plane still reports it.
    """

    product = _primary_analysis_cohort_product(raw)
    if product is None:
        return None
    raw_kind, _, _ = str(raw or "").strip().partition(":")
    _, name = product
    if name == "analysis_cohort" or (
        _normalise(raw_kind) == "cohort" and name == "analysis_set"
    ):
        return product
    return None


def _declares_reserved_primary_cohort_product(step: AnalysisStep) -> bool:
    """Return whether a step claims one legacy globally reserved identity."""

    return any(
        reserved_primary_cohort_product(raw) is not None
        for raw in step.expected_outputs or []
    )


def locked_primary_cohort_product(
    raw: object, *, locked_cohort_name: object
) -> tuple[str, str] | None:
    """Return the identity of the run's single locked primary-cohort population.

    :func:`reserved_primary_cohort_product` answers for the two globally
    reserved spellings only.  A plan whose Planner declared the population
    under the plan's own locked cohort name means the *same* population, and
    the host publishes that spelling itself -- ``closed_cohort_product_
    vocabulary`` offers ``cohort:<exact cohort.name>`` to the Planner.

    Surfaces that decide *which population a step sees* must ask here, because
    the reserved reader alone cannot see that third spelling.  Measured over
    819 recorded real plans: of 3,995 typed primary-cohort inputs, 36 were
    written under the plan's own cohort name, and every one of them made the
    typed-input plane bind the full cohort while the run-level plane mounted
    and reported the development sample.  In canary20 that split was what fed
    the primary model 94,425 rows against a contract expecting 1,000, whose
    ``model_denominator_or_event_mismatch`` then spent the step's repairs.

    ``locked_cohort_name`` is the plan's cohort name; an absent or blank name
    narrows this back to the reserved spellings rather than matching anything.
    """

    product = _primary_analysis_cohort_product(raw)
    if product is None:
        return None
    if reserved_primary_cohort_product(raw) is not None:
        return product
    raw_kind, _, _ = str(raw or "").strip().partition(":")
    _, name = product
    cohort_name = _normalise(locked_cohort_name)
    if _normalise(raw_kind) == "cohort" and cohort_name and name == cohort_name:
        return product
    return None


def _primary_analysis_cohort_product_matches_plan(
    raw: object, *, plan: Any
) -> tuple[str, str] | None:
    """Bind a named ``cohort:`` product to the Planner-locked cohort only."""

    return locked_primary_cohort_product(
        raw,
        locked_cohort_name=getattr(getattr(plan, "cohort", None), "name", None),
    )


def _declares_explicit_cohort_namespace(raw: object) -> bool:
    """Recognize an explicit cohort claim even when its product is malformed."""

    raw_kind, separator, raw_product = str(raw or "").strip().partition(":")
    return bool(separator and raw_product.strip() and _normalise(raw_kind) == "cohort")


def _primary_analysis_cohort_attrition_candidate(step: AnalysisStep) -> bool:
    if not _is_primary_analysis_cohort_method(step.method):
        return False
    outputs = list(step.expected_outputs or [])
    parsed = [typed_product(raw) for raw in outputs]
    return any(
        _primary_analysis_cohort_product(raw) is not None
        or _declares_explicit_cohort_namespace(raw)
        for raw in outputs
    ) and any(
        product is not None
        and product[0] == "table"
        and _is_primary_analysis_cohort_flow_product(product[1])
        for product in parsed
    )


def _primary_analysis_cohort_attrition_step(step: AnalysisStep) -> bool:
    """Return whether a step claims primary cohort construction + attrition.

    Unlike ``_primary_analysis_cohort_attrition_candidate``, this predicate
    deliberately does not require a valid closed cohort output.  It lets plan
    preflight reject a definition/log artifact that was incorrectly declared
    in place of the materialised cohort before Coder or cohort execution runs.
    """

    if not _is_primary_analysis_cohort_method(step.method):
        return False
    return any(
        product is not None
        and product[0] == "table"
        and _is_primary_analysis_cohort_flow_product(product[1])
        for product in (typed_product(raw) for raw in step.expected_outputs or [])
    )


def primary_analysis_cohort_producer_uses_universe(
    *, step: AnalysisStep, plan: Any
) -> bool:
    """Bind only one closed primary-cohort producer to the raw universe.

    The host materialises the Planner-locked cohort before ordinary analysis
    steps run.  A mixed cohort+attrition producer must nevertheless see the raw
    universe so it can report truthful exclusions.  This predicate changes only
    the input role: the Agent still owns every cohort criterion and output.
    """

    if not _primary_analysis_cohort_attrition_candidate(step):
        return False

    raw_outputs = list(step.expected_outputs or [])
    parsed_outputs = [typed_product(raw) for raw in raw_outputs]
    if not parsed_outputs or any(product is None for product in parsed_outputs):
        return False
    analysis_cohort_products = 0
    has_attrition = False
    for raw, parsed in zip(raw_outputs, parsed_outputs, strict=True):
        kind, name = parsed  # type: ignore[misc]
        if _primary_analysis_cohort_product_matches_plan(raw, plan=plan) is not None:
            analysis_cohort_products += 1
            continue
        # The canonical vocabulary, not the recognition predicate.  Entering the
        # gate is a question about what a step claims to be; passing it is a
        # question about whether the declaration is one the host will validate.
        # A step naming its cascade something else must be told to rename it,
        # which is what the owner finding below does -- at plan preflight,
        # before the first Coder call.
        if kind == "table" and name in _PRIMARY_ANALYSIS_COHORT_FLOW_PRODUCTS:
            has_attrition = True
            continue
        return False
    if analysis_cohort_products != 1 or not has_attrition:
        return False

    producers = [
        candidate
        for candidate in getattr(plan, "steps", ())
        if _declares_reserved_primary_cohort_product(candidate)
        or any(
            _primary_analysis_cohort_product_matches_plan(raw, plan=plan) is not None
            for raw in (candidate.expected_outputs or [])
        )
    ]
    return len(producers) == 1 and producers[0].step_id == step.step_id


def _primary_analysis_cohort_product_owner_finding(
    *,
    step: AnalysisStep,
    plan: Any,
    validator: str,
) -> ValidationFinding | None:
    """Return the shared structural owner finding for one cohort step.

    This check uses only the Planner-declared method and typed products.  It is
    therefore safe to run before code generation as well as inside the later
    execution-time integrity gate.  Keeping the finding here prevents the two
    stages from drifting into different definitions of a closed owner.
    """

    if not _primary_analysis_cohort_attrition_step(step):
        return None
    if primary_analysis_cohort_producer_uses_universe(step=step, plan=plan):
        return None
    declared_closed_candidates = [
        str(raw or "").strip()
        for raw in step.expected_outputs or []
        if _primary_analysis_cohort_product(raw) is not None
        or _declares_explicit_cohort_namespace(raw)
    ]
    issue = (
        "primary_cohort_product_owner_ambiguous"
        if declared_closed_candidates
        else "primary_cohort_product_missing"
    )
    detail: dict[str, Any] = {
        "issue": issue,
        "step_id": step.step_id,
    }
    if issue == "primary_cohort_product_missing":
        detail["declared_closed_candidates"] = declared_closed_candidates
    return ValidationFinding(
        validator=validator,
        severity="error",
        message=(
            "A primary cohort construction + attrition step must declare exactly "
            "one materialised closed cohort product and be the plan's unique "
            "primary-cohort owner. A definition or protocol artifact is not a "
            "cohort dataset."
        ),
        detail=detail,
    )


def primary_analysis_cohort_plan_findings(*, plan: Any) -> list[ValidationFinding]:
    """Validate primary-cohort typed-product ownership before Coder execution.

    The host does not select a cohort or edit eligibility here.  It only checks
    whether a Planner-declared mixed cohort/attrition step is structurally
    closed and uniquely owns the locked primary cohort.  The probe-aware
    replanner may then repair declarations without changing scientific choices.
    """

    findings: list[ValidationFinding] = []
    for step in getattr(plan, "steps", ()):
        finding = _primary_analysis_cohort_product_owner_finding(
            step=step,
            plan=plan,
            validator="plan_primary_analysis_cohort_integrity",
        )
        if finding is not None:
            findings.append(finding)
    return findings

__all__ = [
    "locked_primary_cohort_product",
    "primary_analysis_cohort_plan_findings",
    "primary_analysis_cohort_producer_uses_universe",
    "reserved_primary_cohort_product",
]
