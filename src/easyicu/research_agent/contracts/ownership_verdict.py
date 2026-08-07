"""Why a deterministic owner does, or does not, claim a step.

Every ``*_owns_step`` predicate returned ``bool``.  A bool cannot tell apart
the two declines that call for **opposite** host behaviour:

``wrong shape``
    The step is not this owner's contract at all.  Handing it to the coder is
    correct; there is nothing for anyone to fix.

``incomplete declaration``
    The step **is** this owner's contract, and the only thing standing between
    it and a deterministic result is a field the Planner never filled in.
    Handing that to the coder is a fail-open at a declaration boundary: the
    host silently substitutes a stochastic actor for one it already has, and
    says nothing to the only party who could have fixed it.

Measured over 553 recorded real steps (``tools/measure_executor_ownership.py``,
2026-07-29): the adjusted-association owner -- which computes the paper's
primary result and has been wired since L4 -- claimed **0 of the 59** steps
that declared its method and its product.  The declines split:

* **26** declared exactly one model and no adjustment set.  Second kind: one
  declared ``covariates`` list away from a deterministic primary estimate.
* **28 + 5** declared two and three models in one step.  First kind, and the
  distinction matters in both directions -- ``bind_primary_output`` binds a
  one-row table, so more declaring is not what would let this owner claim
  them, and reporting them as a gap would send the Planner to fix something
  that is not broken (that is task #105's question instead).
* **5** bundled this product with another in one step.  Also first kind.

None declared zero models; an earlier reading said so because the helper that
selects "the single requirement" returns ``None`` for both zero and many.  The
bool could say none of this, so for all 59 nothing did, and the primary
estimate went to the path whose accumulated repair guidance records a script
that dropped the whole cohort by numeric-coercing ``sex``, object-dtype design
matrices handed to statsmodels, and a contract "satisfied" with a null
estimate.

Design notes, both load-bearing:

* **One predicate, not two.**  A separate ``declaration_gap(step)`` beside a
  surviving ``owns_step(step)`` would encode the same clauses twice, and two
  copies of one rule drifting apart is the single most repeated defect shape
  in this package.  An owner returns a verdict; a ``bool`` wrapper delegates
  to it so existing callers keep working without a second source of truth.

* **No ``__bool__``.**  Making a verdict falsy would let every
  ``if owner_owns_step(step):`` keep compiling while silently collapsing the
  distinction this type exists to draw -- the exact failure it is replacing.
  Callers must ask for ``.claimed``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Tuple

__all__ = ["OwnershipVerdict"]


@dataclass(frozen=True)
class OwnershipVerdict:
    """One owner's answer about one step.

    ``missing_declarations`` is the whole point: it names the fields whose
    absence is the only reason the owner declined, in a form the Planner can
    act on.  It is empty for a claim and for a wrong-shape decline, and
    non-empty exactly when the host should refuse the plan instead of falling
    through to the coder.
    """

    analysis_kind: str
    claimed: bool
    reason: str
    missing_declarations: Tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not str(self.analysis_kind or "").strip():
            raise ValueError("an ownership verdict must name its analysis kind")
        if not str(self.reason or "").strip():
            raise ValueError(
                "an ownership verdict must carry a reason: it is the whole "
                "diagnostic value of the type"
            )
        for name in self.missing_declarations:
            if not str(name or "").strip():
                raise ValueError(
                    "a missing-declaration entry must name a field; an empty "
                    "name would report a gap nobody can close"
                )
        if self.claimed and self.missing_declarations:
            raise ValueError("a claimed step cannot also report missing declarations")

    # -- constructors ----------------------------------------------------
    # Named rather than positional so a call site cannot swap the two decline
    # kinds by argument order.

    @classmethod
    def claim(cls, analysis_kind: str, *, reason: str) -> "OwnershipVerdict":
        return cls(analysis_kind=analysis_kind, claimed=True, reason=reason)

    @classmethod
    def wrong_shape(cls, analysis_kind: str, *, reason: str) -> "OwnershipVerdict":
        """Decline because the step is someone else's contract."""

        return cls(analysis_kind=analysis_kind, claimed=False, reason=reason)

    @classmethod
    def incomplete_declaration(
        cls,
        analysis_kind: str,
        *,
        missing: Iterable[str],
        reason: str,
    ) -> "OwnershipVerdict":
        """Decline because a field the Planner owns was never declared.

        ``missing`` must be non-empty.  An "incomplete declaration" naming
        nothing is indistinguishable from a wrong-shape decline at every
        consumer, so accepting one would silently reopen the fail-open this
        type closes.
        """

        names = tuple(str(name).strip() for name in missing)
        if not names:
            raise ValueError(
                "an incomplete-declaration verdict must name at least one "
                "missing field, otherwise it says nothing a bool did not"
            )
        return cls(
            analysis_kind=analysis_kind,
            claimed=False,
            reason=reason,
            missing_declarations=names,
        )

    # -- queries ---------------------------------------------------------

    @property
    def declaration_is_incomplete(self) -> bool:
        """True when the host should refuse rather than fall back to the coder."""

        return not self.claimed and bool(self.missing_declarations)
