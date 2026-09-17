"""Shared simulated-review disclaimer copy (Owner: reporting).

The three-role reviewer loop is a deterministic checklist, not independent
external peer review.  Both :mod:`reviewer` and :mod:`scientific_maturity`
must cite the same wording so gates cannot drift into reading the checklist
as publication authority.  Import these constants instead of inlining the
strings.
"""

from __future__ import annotations

#: Machine-readable review mode recorded on reviewer summaries.
SIMULATED_REVIEW_MODE = "simulated_deterministic"

#: Claim boundary recorded on reviewer summaries.
SIMULATED_REVIEW_CLAIM_BOUNDARY = (
    "Simulated three-role checklist. Not independent external "
    "review and not publication authority."
)

#: Shared fragment: the checklist is not independent external review.
#: Composed by maturity findings, e.g.
#: ``f"No reviewer receipt is available. The pipeline's {...}."``.
SIMULATED_REVIEW_NOT_INDEPENDENT = (
    "simulated three-role checklist is not independent external review"
)

__all__ = [
    "SIMULATED_REVIEW_MODE",
    "SIMULATED_REVIEW_CLAIM_BOUNDARY",
    "SIMULATED_REVIEW_NOT_INDEPENDENT",
]
