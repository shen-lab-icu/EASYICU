"""Why a descriptive study's design reports counts only, or does not.

Owner of one record beside the study's analysis design: the basis of its
variance when counts only were on the table.  A descriptive design reports
counts and proportions without any interval (``none_counts_only``) only when
the researcher's own words ask for it (:func:`counts_only_request`); the
host's rule otherwise chooses the variance by the source's patient grouping
(``study_family_design.descriptive_variance``).  The record says which, so
the study card shows why the proportions carry intervals or do not:

* ``basis: user_words`` with ``evidence``: counts only, as the researcher's
  words (a contiguous span of their message) ask;
* ``basis: source_patient_grouping`` / ``source_without_patient_grouping``
  with ``replaced: none_counts_only``: counts only were proposed, nobody
  asked for them, and the host chose the variance by the source;
* ``conflict: counts_only_not_requested``: the study already records counts
  only on no stated request; the design stays, and the card says so.

A record is cleared once the design's variance is no longer the one it
explains (:func:`design_variance_basis_is_stale`).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional

__all__ = [
    "COUNTS_ONLY",
    "COUNTS_ONLY_NOT_REQUESTED",
    "DESIGN_VARIANCE_BASIS_FIELD",
    "SOURCE_BASES",
    "USER_WORDS",
    "counts_only_request",
    "design_variance_basis_is_stale",
    "normalize_design_variance_basis",
]

#: The study field and receipt field that carry the record.
DESIGN_VARIANCE_BASIS_FIELD = "design_variance_basis"
#: The variance of a design that reports counts only.
COUNTS_ONLY = "none_counts_only"
#: The basis of counts only the researcher's words ask for.
USER_WORDS = "user_words"
#: The bases of a variance the host chose by the source's patient grouping.
SOURCE_BASES = ("source_patient_grouping", "source_without_patient_grouping")
#: A recorded counts-only design no stated request explains.
COUNTS_ONLY_NOT_REQUESTED = "counts_only_not_requested"
_MAX_EVIDENCE_CHARS = 300

# Words that ask for counts without any interval or inference.  Declining a
# model ("不需要回归", "no adjustment") does not: a proportion still carries
# its interval.
_COUNTS_ONLY = re.compile(
    r"\bcounts?[\s_-]*only\b|\bonly\s+(?:the\s+)?counts\b"
    r"|\b(?:no|without)\s+(?:confidence\s+)?intervals?\b"
    r"|\b(?:no|without)\s+(?:statistical\s+)?inference\b"
    r"|只计数|(?:只|仅)(?:报告|给出|列出)?(?:计数|人数|例数|频数)"
    r"|(?:不需要|不用|无需|不报告|不要|不给出?)\s*置信区间"
    r"|(?:不做|不进行)\s*(?:统计)?推断",
    re.IGNORECASE,
)


def counts_only_request(message: Any) -> Optional[str]:
    """The words of ``message`` that ask for counts only, or ``None``."""

    match = _COUNTS_ONLY.search(str(message or ""))
    return match.group(0) if match else None


def normalize_design_variance_basis(value: Any) -> Optional[Dict[str, Any]]:
    """The stored shape of a record; ``None`` clears it.

    Raises ``ValueError`` for any other shape: user words carry their
    evidence and explain counts only; a source basis explains another
    variance and names the counts only it replaced; a conflict is counts only
    with no basis.
    """

    if value is None or value == {}:
        return None
    allowed = {"variance_estimator", "basis", "evidence", "replaced", "conflict"}
    if not isinstance(value, Mapping) or not set(value) <= allowed:
        raise ValueError("a variance basis states its variance and why")
    variance = value.get("variance_estimator")
    if not isinstance(variance, str) or not variance.strip():
        raise ValueError("a variance basis names the variance it explains")
    basis = value.get("basis")
    record: Dict[str, Any] = {"variance_estimator": variance}
    if basis == USER_WORDS:
        evidence = value.get("evidence")
        if (
            variance != COUNTS_ONLY
            or set(value) != {"variance_estimator", "basis", "evidence"}
            or not isinstance(evidence, str)
            or not evidence.strip()
            or len(evidence) > _MAX_EVIDENCE_CHARS
        ):
            raise ValueError("the researcher's words explain counts only, with the words")
        record.update(basis=USER_WORDS, evidence=evidence)
    elif basis in SOURCE_BASES:
        if variance == COUNTS_ONLY or set(value) != {
            "variance_estimator", "basis", "replaced"
        } or value.get("replaced") != COUNTS_ONLY:
            raise ValueError("a source basis names the counts only it replaced")
        record.update(basis=basis, replaced=COUNTS_ONLY)
    elif basis is None:
        if variance != COUNTS_ONLY or set(value) != {
            "variance_estimator", "conflict"
        } or value.get("conflict") != COUNTS_ONLY_NOT_REQUESTED:
            raise ValueError("a basis-free record is counts only nobody asked for")
        record.update(conflict=COUNTS_ONLY_NOT_REQUESTED)
    else:
        raise ValueError("a variance basis is the researcher's words or the source")
    return record


def design_variance_basis_is_stale(context: Mapping[str, Any]) -> bool:
    """Whether a stored record no longer explains the study's design."""

    record = context.get(DESIGN_VARIANCE_BASIS_FIELD)
    if not record:
        return False
    design = context.get("analysis_design")
    if not isinstance(design, Mapping):
        return True
    return (
        str(design.get("analysis_family") or "") != "descriptive_epidemiology"
        or design.get("variance_estimator") != record.get("variance_estimator")
    )
