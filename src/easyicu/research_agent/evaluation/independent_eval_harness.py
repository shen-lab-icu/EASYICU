"""[Track 4: Independent Evaluation Harness] Held-out skeleton and口径 locks.

Skeleton only: preregistration structures (item/version/eval-method/
denominator/allowed-interventions), failure-retention records, and a report
that counts the three recovery kinds separately. It runs no formal
experiment, executes no evaluation items, and claims no pass rate.

Authoritative references: ``docs/EASYICU_PRODUCT_AGREEMENT.md`` U1-U4 are
unresolved and must not be silently completed here. Coverage proportions
(U1), rate thresholds (U2), and iteration minima (U3) are absent on purpose;
the journal-quality judge (U4) is a pending-user-decision hook only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

ScopeDimension = Literal[
    "descriptive",
    "association",
    "prediction",
    "survival",
    "causal_simulation",
    "subtyping_trajectory",
]

#: Scope matrix for the held-out bank (U1). Membership is fixed; coverage
#: proportions, weights, and thresholds are NOT set here — U1 stays pending
#: a user decision, so no coverage claim may be derived from this constant.
SCOPE_MATRIX: Tuple[ScopeDimension, ...] = (
    "descriptive",
    "association",
    "prediction",
    "survival",
    "causal_simulation",
    "subtyping_trajectory",
)

#: Chinese labels for the six scope dimensions, keyed by ScopeDimension.
SCOPE_LABELS_ZH: Dict[str, str] = {
    "descriptive": "描述",
    "association": "关联",
    "prediction": "预测",
    "survival": "生存",
    "causal_simulation": "因果模拟",
    "subtyping_trajectory": "分型轨迹",
}

FailureKind = Literal["system_defect", "repairable_failure", "data_limitation"]

#: Failure kinds tracked separately per U2 (system defect / repairable
#: failure / substantive data-or-method limitation).
FAILURE_KINDS: Tuple[FailureKind, ...] = (
    "system_defect",
    "repairable_failure",
    "data_limitation",
)

#: Denominator rule (U2): system defects count in the failure denominator
#: and must never be excluded. There are no denominator exclusions.
SYSTEM_DEFECT_COUNTS_IN_DENOMINATOR = True
DENOMINATOR_EXCLUSIONS: Tuple[str, ...] = ()


def is_excluded_from_denominator(kind: str) -> bool:
    """Return whether a failure kind is excluded from the denominator.

    Always ``False``: every preregistered item stays in the denominator and
    every failure kind — including ``system_defect`` — counts. U2 leaves
    rate thresholds undecided but forbids dropping system defects.
    """

    return kind in DENOMINATOR_EXCLUSIONS


RecoveryKind = Literal["in_run_recovery", "post_fix_recovery", "later_reuse"]

#: The three recovery kinds from the product agreement, counted separately:
#: in-run self recovery / recovery after a developer source fix (must not be
#: recorded as autonomous recovery) / reuse of a prior fix on a later run.
RECOVERY_KINDS: Tuple[RecoveryKind, ...] = (
    "in_run_recovery",
    "post_fix_recovery",
    "later_reuse",
)

#: U4 stays pending: no owner has decided who judges journal quality, so the
#: harness records this status and refuses to sign a judgment in code.
U4_STATUS_PENDING = "pending_user_decision"


@dataclass(frozen=True)
class HeldOutItem:
    """One preregistered held-out evaluation item."""

    item_id: str
    version: str
    scope: ScopeDimension
    eval_method: str
    denominator_id: str
    allowed_interventions: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("held-out item_id must be non-empty")
        if not self.version.strip():
            raise ValueError("held-out item version must be non-empty")
        if self.scope not in SCOPE_MATRIX:
            raise ValueError(f"unknown scope dimension {self.scope!r}")
        if not self.eval_method.strip():
            raise ValueError("held-out eval_method must be non-empty")
        if not self.denominator_id.strip():
            raise ValueError("held-out denominator_id must be non-empty")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "version": self.version,
            "scope": self.scope,
            "eval_method": self.eval_method,
            "denominator_id": self.denominator_id,
            "allowed_interventions": list(self.allowed_interventions),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HeldOutItem":
        return cls(
            item_id=str(data["item_id"]),
            version=str(data["version"]),
            scope=data["scope"],
            eval_method=str(data["eval_method"]),
            denominator_id=str(data["denominator_id"]),
            allowed_interventions=tuple(
                str(v) for v in data.get("allowed_interventions") or ()
            ),
        )


@dataclass(frozen=True)
class PreregistrationLock:
    """Frozen preregistration: items, versions, denominators, interventions.

    The digest binds the locked set so a later report can prove it evaluates
    exactly what was preregistered. Repair samples must not re-enter as
    unseen items; that exclusion is enforced by keeping this lock immutable.
    """

    items: Tuple[HeldOutItem, ...]
    digest: str = field(default="")
    locked: bool = True

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("preregistration requires at least one item")
        if not self.locked:
            raise ValueError("preregistration lock must be created locked")
        object.__setattr__(self, "digest", _digest_items(self.items))

    def denominator_size(self) -> int:
        """Number of preregistered items; failures never shrink this."""

        return len(self.items)

    def item_ids(self) -> List[str]:
        return [item.item_id for item in self.items]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "locked": self.locked,
            "digest": self.digest,
            "denominator_size": self.denominator_size(),
            "items": [item.to_dict() for item in self.items],
        }


def lock_preregistration(items: Sequence[HeldOutItem]) -> PreregistrationLock:
    """Freeze a preregistered held-out set and return its lock."""

    return PreregistrationLock(items=tuple(items))


@dataclass(frozen=True)
class FailureRecord:
    """One retained failure attempt. ``retained`` is always True: failures
    are never dropped, and the ledger below offers no removal API."""

    item_id: str
    item_version: str
    kind: FailureKind
    attempt_id: str
    detail: str
    retained: bool = True

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("failure record item_id must be non-empty")
        if self.kind not in FAILURE_KINDS:
            raise ValueError(f"unknown failure kind {self.kind!r}")
        if not self.attempt_id.strip():
            raise ValueError("failure record attempt_id must be non-empty")
        if not self.retained:
            raise ValueError("failure records must be retained; dropping is forbidden")

    def counts_in_denominator(self) -> bool:
        """Always True; see :func:`is_excluded_from_denominator`."""

        return not is_excluded_from_denominator(self.kind)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "item_version": self.item_version,
            "kind": self.kind,
            "attempt_id": self.attempt_id,
            "detail": self.detail,
            "retained": self.retained,
        }


class FailureLedger:
    """Append-only failure store. Deliberately no remove/clear API."""

    def __init__(self) -> None:
        self._records: List[FailureRecord] = []

    def add(self, record: FailureRecord) -> None:
        self._records.append(record)

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> Tuple[FailureRecord, ...]:
        return tuple(self._records)

    def records_for(self, item_id: str) -> Tuple[FailureRecord, ...]:
        return tuple(r for r in self._records if r.item_id == item_id)

    def to_dict(self) -> Dict[str, Any]:
        return {"records": [r.to_dict() for r in self._records]}


@dataclass(frozen=True)
class RecoveryRecord:
    """One recovery event, tagged with exactly one of the three kinds."""

    item_id: str
    kind: RecoveryKind
    human_involved: bool
    note: str = ""

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("recovery record item_id must be non-empty")
        if self.kind not in RECOVERY_KINDS:
            raise ValueError(f"unknown recovery kind {self.kind!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "kind": self.kind,
            "human_involved": self.human_involved,
            "note": self.note,
        }


@dataclass(frozen=True)
class RecoveryCounts:
    """The three recovery kinds counted separately (U3). No merged
    "autonomous recovery" aggregate is provided: post-fix recoveries prove
    the fix, not autonomy."""

    in_run_recovery: int = 0
    post_fix_recovery: int = 0
    later_reuse: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "in_run_recovery": self.in_run_recovery,
            "post_fix_recovery": self.post_fix_recovery,
            "later_reuse": self.later_reuse,
        }


def count_recoveries(records: Sequence[RecoveryRecord]) -> RecoveryCounts:
    """Group recovery records by kind, keeping the three counts separate."""

    counts = {"in_run_recovery": 0, "post_fix_recovery": 0, "later_reuse": 0}
    for record in records:
        counts[record.kind] += 1
    return RecoveryCounts(**counts)


@dataclass(frozen=True)
class IndependentEvalReport:
    """Skeleton report: prereg lock, retained failures, split recovery counts.

    This report carries counts only. It computes no pass rate and makes no
    coverage claim (U1/U2 pending); ``coverage_claim`` stays None until the
    user sets the scope matrix weights and thresholds.
    """

    preregistration: PreregistrationLock
    failures: Tuple[FailureRecord, ...] = ()
    recoveries: Tuple[RecoveryRecord, ...] = ()
    u4_status: str = U4_STATUS_PENDING
    coverage_claim: Optional[str] = None

    def __post_init__(self) -> None:
        known = set(self.preregistration.item_ids())
        for record in list(self.failures) + list(self.recoveries):
            if record.item_id not in known:
                raise ValueError(
                    f"record references unregistered item {record.item_id!r}"
                )

    def denominator_size(self) -> int:
        return self.preregistration.denominator_size()

    def recovery_counts(self) -> RecoveryCounts:
        return count_recoveries(self.recoveries)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preregistration": self.preregistration.to_dict(),
            "denominator_size": self.denominator_size(),
            "failures_retained": [r.to_dict() for r in self.failures],
            "recovery_counts": self.recovery_counts().to_dict(),
            "u4_status": self.u4_status,
            "coverage_claim": self.coverage_claim,
        }


def build_report(
    preregistration: PreregistrationLock,
    failures: Sequence[FailureRecord],
    recoveries: Sequence[RecoveryRecord],
) -> IndependentEvalReport:
    """Assemble a report; rejects records for unregistered items."""

    return IndependentEvalReport(
        preregistration=preregistration,
        failures=tuple(failures),
        recoveries=tuple(recoveries),
    )


class PendingU4Decision(RuntimeError):
    """Raised when journal-quality adjudication is requested before U4 lands.

    U4 (who judges journal quality — target journal, checklist, independent
    roles) is undecided and belongs to the user. This harness must never
    decide it in code nor sign an independent evaluation.
    """


def request_journal_quality_adjudication(
    *, report: IndependentEvalReport, adjudicator: Optional[str] = None
) -> None:
    """Decision hook for U4. Always raises :class:`PendingU4Decision`.

    待用户定：期刊质量由谁判定（拟投期刊/检查表/独立评审角色与是否盲评）。
    在用户拍板前，任何调用都必须以挂起失败，而不是返回一个代替决定。
    """

    _ = (report, adjudicator)
    raise PendingU4Decision(
        "U4 pending user decision: journal-quality adjudicator, checklist, "
        "and blinding are undecided; refusing to sign an independent "
        "evaluation in code."
    )


def _digest_items(items: Tuple[HeldOutItem, ...]) -> str:
    canonical = json.dumps(
        [item.to_dict() for item in sorted(items, key=lambda i: i.item_id)],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "DENOMINATOR_EXCLUSIONS",
    "FAILURE_KINDS",
    "FailureKind",
    "FailureLedger",
    "FailureRecord",
    "HeldOutItem",
    "IndependentEvalReport",
    "PendingU4Decision",
    "PreregistrationLock",
    "RECOVERY_KINDS",
    "RecoveryCounts",
    "RecoveryKind",
    "RecoveryRecord",
    "SCOPE_LABELS_ZH",
    "SCOPE_MATRIX",
    "SYSTEM_DEFECT_COUNTS_IN_DENOMINATOR",
    "ScopeDimension",
    "U4_STATUS_PENDING",
    "build_report",
    "count_recoveries",
    "is_excluded_from_denominator",
    "lock_preregistration",
    "request_journal_quality_adjudication",
]
