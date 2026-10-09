"""State in Limitations what the approved plan's scientific review left to the study.

A run whose plan the researcher approved carries the scientific review of that
plan (``scientific_plan_review.json``), bound into the approval by its digest.
A ``major`` finding the review routes to ``study_authority_change`` is a
design limitation the study kept instead of revising: approving the plan does
not repair it, so the manuscript states it.  The Writer is not asked to.  The
host places one fixed sentence per such finding code at the top of
``## Limitations``, after the source method facts, and fails closed after the
provenance filters when one did not survive.

The sentences are manuscript prose, not the reviewer's message, which
addresses the researcher and may quote a criterion with numbers no evidence
registers.  Each sentence holds for every way its code is raised, states no
number and no window, and avoids the vocabulary the claim policy reads as a
scientific assertion, so the unchanged policy keeps it as context prose that
cites the review.  A code that reaches the manuscript without a sentence stops
the manuscript with a typed reason; it is never skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

from pydantic import ValidationError

from ..authority.evidence_store import EvidenceStore
from ..authority.runtime_artifacts import verified_run_evidence_path
from ..orchestration.scientific_plan_review_gate import (
    SCIENTIFIC_PLAN_REVIEW_EVIDENCE_ID,
)
from ..planning.scientific_review import (
    PlanScientificFinding,
    PlanScientificReview,
    remediation_route_for_finding,
)
from ..schema import ValidationFinding
from .manuscript_method_facts import missing_bound_method_facts

#: The sentence that states each finding the manuscript must carry.
PLAN_REVIEW_LIMITATION_SENTENCES: Mapping[str, str] = MappingProxyType(
    {
        "POPULATION_CRITERION_NOT_APPLIED": (
            "A population criterion in the analysis plan was not applied to the "
            "analysed data, so the analysed cohort is broader than the "
            "population the plan describes"
        ),
        "POPULATION_SCOPE_AMENDMENT_DECLARED": (
            "A descriptive result is reported for a population that differs from "
            "the one the study first specified for it; the change was declared "
            "and reviewed before the analysis, so that result does not describe "
            "the originally specified population"
        ),
        "UNADJUSTED_ASSOCIATION_NOT_ARTICLE_GRADE": (
            "The primary comparison was not adjusted for confounding; it "
            "describes crude differences in outcomes between the compared groups "
            "and does not estimate an association independent of other patient "
            "characteristics"
        ),
        "ADJUSTMENT_RATIONALE_OR_TIMING_UNBOUND": (
            "Not every covariate in the adjustment set had a reviewed clinical "
            "rationale and a confirmed baseline timing, so residual confounding, "
            "and adjustment for variables measured after baseline, cannot be "
            "ruled out"
        ),
        "COHORT_SELECTS_ON_OUTCOME": (
            "Inclusion in the cohort depended on the outcome or on a variable "
            "used to define it, so the reported outcomes are conditional on that "
            "selection, are subject to selection bias, and do not describe all "
            "eligible patients"
        ),
    }
)

PLAN_REVIEW_LIMITATION_UNMAPPED = "writer_plan_review_limitation_unmapped"
PLAN_REVIEW_UNVERIFIED = "writer_plan_review_unverified"
PLAN_REVIEW_LIMITATIONS_MISSING = "writer_plan_review_limitations_missing"


class PlanReviewLimitationError(ValueError):
    """The approved review's limitations cannot be stated."""

    def __init__(
        self, reason_code: str, message: str, *, codes: Sequence[str] = ()
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.codes = tuple(codes)


@dataclass(frozen=True)
class PlanReviewLimitation:
    """One limitation the approved review left to the study, as Limitations states it."""

    code: str
    text: str
    #: The digest of the review the sentence was read from.
    source_sha256: str
    section: str = "limitations"

    @property
    def source_field(self) -> str:
        return f"{SCIENTIFIC_PLAN_REVIEW_EVIDENCE_ID}.findings.{self.code}"

    @property
    def scaffold(self) -> str:
        return f"{self.text} {{evidence:{SCIENTIFIC_PLAN_REVIEW_EVIDENCE_ID}}}."


def states_a_manuscript_limitation(finding: PlanScientificFinding) -> bool:
    """Whether the study kept this finding as a limitation of its design."""

    return (
        finding.severity == "major"
        and remediation_route_for_finding(finding) == "study_authority_change"
    )


def load_plan_review_limitations(
    *, root: Path, records: Sequence[Any]
) -> tuple[PlanReviewLimitation, ...]:
    """The limitations of the run's bound plan review; none without a review."""

    record = next(
        (
            item
            for item in records
            if getattr(item, "evidence_id", None) == SCIENTIFIC_PLAN_REVIEW_EVIDENCE_ID
        ),
        None,
    )
    if record is None:
        return ()
    path = verified_run_evidence_path(root, record)
    if path is None:
        raise PlanReviewLimitationError(
            PLAN_REVIEW_UNVERIFIED,
            "The plan review's bytes no longer match the review the plan was "
            "approved with.",
        )
    try:
        review = PlanScientificReview.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise PlanReviewLimitationError(
            PLAN_REVIEW_UNVERIFIED, "The plan review could not be read."
        ) from exc
    codes = tuple(
        dict.fromkeys(
            finding.code
            for finding in review.findings
            if states_a_manuscript_limitation(finding)
        )
    )
    unmapped = [code for code in codes if code not in PLAN_REVIEW_LIMITATION_SENTENCES]
    if unmapped:
        raise PlanReviewLimitationError(
            PLAN_REVIEW_LIMITATION_UNMAPPED,
            "The approved plan review left limitations the manuscript has no "
            f"sentence for: {', '.join(unmapped)}.",
            codes=unmapped,
        )
    return tuple(
        PlanReviewLimitation(
            code=code,
            text=PLAN_REVIEW_LIMITATION_SENTENCES[code],
            source_sha256=str(record.sha256),
        )
        for code in codes
    )


def _finding(error: PlanReviewLimitationError) -> ValidationFinding:
    return ValidationFinding(
        validator="evidence_bound_writer",
        severity="error",
        message=str(error),
        detail={"reason_code": error.reason_code, "finding_codes": list(error.codes)},
    )


def writer_plan_review_limitations(
    evidence: EvidenceStore,
) -> tuple[tuple[PlanReviewLimitation, ...], Optional[ValidationFinding]]:
    """The limitations the Writer's Limitations carries, or why none can be stated."""

    try:
        return (
            load_plan_review_limitations(
                root=evidence.root, records=evidence.records()
            ),
            None,
        )
    except PlanReviewLimitationError as exc:
        return (), _finding(exc)


def audit_bound_plan_review_limitations(
    bound: str,
    *,
    evidence: EvidenceStore,
    per_step_records: Sequence[Mapping[str, Any]],
) -> Optional[ValidationFinding]:
    """Fail closed when a stated limitation did not survive provenance validation.

    A review that cannot be read, or a limitation without a sentence, was
    already reported where the limitations are placed, as an error.
    """

    limitations, failure = writer_plan_review_limitations(evidence)
    if failure is not None:
        return None
    missing = missing_bound_method_facts(
        bound,
        limitations,
        lambda text: evidence.bind_manuscript(
            text, per_step_records=per_step_records, reader_labels=None
        ),
    )
    if not missing:
        return None
    return ValidationFinding(
        validator="evidence_bound_writer",
        severity="error",
        message=(
            "A limitation the approved plan review left to the study did not "
            "survive manuscript provenance validation."
        ),
        detail={
            "reason_code": PLAN_REVIEW_LIMITATIONS_MISSING,
            "source_fields": list(missing),
        },
    )


__all__ = [
    "PLAN_REVIEW_LIMITATIONS_MISSING",
    "PLAN_REVIEW_LIMITATION_SENTENCES",
    "PLAN_REVIEW_LIMITATION_UNMAPPED",
    "PLAN_REVIEW_UNVERIFIED",
    "PlanReviewLimitation",
    "PlanReviewLimitationError",
    "audit_bound_plan_review_limitations",
    "load_plan_review_limitations",
    "states_a_manuscript_limitation",
    "writer_plan_review_limitations",
]
