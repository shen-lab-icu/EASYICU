"""Measure how much real Planner work the host can execute deterministically.

Every canonical-9 run produces a *different* plan -- fresh17 declared 13 steps,
fresh19 declared 13 with different ids and different product bundles -- so
"did that change help?" had no denominator.  Two consecutive runs of the same
case are not two samples of the same experiment.  This gives a fixed one: point
it at recorded plans and it reports, over the same corpus every time, how many
steps a deterministic owner claims and which declared products no owner can
emit.

It asks ``select_standard_executor`` itself.  It must never re-implement an
ownership predicate: the selector applies gates *after* a contract matches
(a receipt obligation an owner cannot discharge, a typed input scope it does
not support), so a second copy of the predicates eventually reports an owner
the selector declined.  ``StandardExecutorCandidate`` exists for exactly this
reason and this tool consumes it.

What it does not know, it says.  Ownership depends on two run-time facts that
a recorded plan does not carry -- whether the step owes a flag-only
plausibility receipt, and the host's resolved typed-input bindings -- so every
count is reported as a pair:

    upper   no receipt obligation, no resolved bindings: no owner declines for
            a receipt it cannot discharge.  Permissive.
    lower   a receipt obligation is present.  Conservative.

The truth for a given run is between them.  Reporting only the flattering bound
would be answering in the permissive direction for a fact this tool lacks.

This is a measurement, not a gate.  Nothing here may decide whether a run may
start; a plan that scores well is not thereby authorized to execute.

Usage::

    python tools/measure_executor_ownership.py                       # summary
    python tools/measure_executor_ownership.py --json ledger.json
    python tools/measure_executor_ownership.py --root research_output \
        --root /Volumes/外置硬盘/easyicu_data/canonical9_runs
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence


def _ensure_importable() -> None:
    """Let the script run from a checkout without an install.

    Guarded on ``easyicu`` not already resolving, and never applied on import:
    a test that imports this module must not inherit a ``sys.path`` entry that
    could shadow the installed package it is actually testing.
    """

    try:
        import easyicu  # noqa: F401
    except ImportError:
        repo_src = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
        )
        if os.path.isdir(repo_src):
            sys.path.insert(0, repo_src)


if __name__ == "__main__":
    _ensure_importable()

from easyicu.research_agent.authority.plausibility import (  # noqa: E402
    FlagOnlyPlausibilityScope,
)
from easyicu.research_agent.execution.runners.selection import (  # noqa: E402
    select_standard_executor,
)
from easyicu.research_agent.planning.cohort_contract import (  # noqa: E402
    cohort_concept_id_scope,
)
from easyicu.research_agent.schema import AnalysisPlan  # noqa: E402

DEFAULT_ROOTS = ("research_output",)
PLAN_GLOB = "analysis_plan*.json"


@dataclass(frozen=True, slots=True)
class StepOwnership:
    """What the selector answered for one recorded step, under both bounds."""

    key: str
    step_id: str
    method: str
    declared_products: tuple[str, ...]
    upper_owner: Optional[str]
    lower_owner: Optional[str]
    upper_trace: tuple[tuple[str, bool, str], ...]

    @property
    def claimed_upper(self) -> bool:
        return self.upper_owner is not None

    @property
    def claimed_lower(self) -> bool:
        return self.lower_owner is not None


@dataclass
class OwnershipLedger:
    """The corpus-wide answer, with what it could not read kept in view."""

    steps: list[StepOwnership] = field(default_factory=list)
    readable_plans: int = 0
    unreadable_plans: int = 0
    plans_needing_registered_concepts: int = 0
    raised: list[tuple[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "corpus": {
                "readable_plans": self.readable_plans,
                "unreadable_plans": self.unreadable_plans,
                "plans_needing_registered_concepts": (
                    self.plans_needing_registered_concepts
                ),
                "unique_steps": len(self.steps),
                "selector_raised": [
                    {"step_id": step_id, "error": error}
                    for step_id, error in self.raised
                ],
            },
            "bounds": {
                "upper": "no receipt obligation and no resolved bindings (permissive)",
                "lower": "a flag-only receipt obligation is present (conservative)",
                "unknown": (
                    "a recorded plan does not carry the run-time receipt scope or "
                    "the resolved typed-input bindings; the true count for any "
                    "single run lies between the two bounds"
                ),
            },
            "claimed_upper": sum(1 for step in self.steps if step.claimed_upper),
            "claimed_lower": sum(1 for step in self.steps if step.claimed_lower),
            "steps": [
                {
                    "key": step.key,
                    "step_id": step.step_id,
                    "method": step.method,
                    "declared_products": list(step.declared_products),
                    "upper_owner": step.upper_owner,
                    "lower_owner": step.lower_owner,
                    "upper_trace": [list(entry) for entry in step.upper_trace],
                }
                for step in self.steps
            ],
        }


def _step_key(raw: Any) -> str:
    return hashlib.sha256(
        json.dumps(raw, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _receipt_scope(step_id: str) -> FlagOnlyPlausibilityScope:
    # Any non-empty column scope makes the obligation present; which column it
    # names does not change whether an owner can discharge a receipt at all.
    return FlagOnlyPlausibilityScope(
        step_id=step_id,
        expected_columns=("age",),
        source_contracts_sha256="0" * 64,
        authority_kind="resolved_raw_input_contracts",
    )


def iter_plan_paths(roots: Sequence[str]) -> Iterator[str]:
    for root in roots:
        yield from sorted(
            glob.glob(os.path.join(root, "**", PLAN_GLOB), recursive=True)
        )


def referenced_concept_ids(document: Any) -> frozenset[str]:
    """Every ``concept_id`` named anywhere in a recorded plan document."""

    found: set[str] = set()
    stack = [document]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            value = node.get("concept_id")
            if isinstance(value, str) and value.strip():
                found.add(value.strip())
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return frozenset(found)


def load_plan(document: Any) -> tuple[Optional[AnalysisPlan], bool]:
    """Parse a recorded plan, reporting whether its concepts had to be granted.

    Cohort predicates validate ``concept_id`` against the concept dictionary
    plus a registry the pipeline populates *during a run* from the cohort's
    materialised columns.  Re-reading a plan outside its run therefore rejects
    every plan whose robustness cohort override names such a column -- 100 of
    the 110 plans this corpus first dropped, and they were concentrated in the
    robustness family, which is the one this measurement most needs to see.

    So the ids the plan itself names are granted for the parse.  That is an
    assumption ("at run time these existed", which is why the run wrote them),
    and the caller counts it rather than hiding it: a corpus silently missing
    its hardest family would have reported a flattering ownership rate.
    """

    try:
        return AnalysisPlan.model_validate(document), False
    except Exception:  # noqa: BLE001 - fall through to the explicit grant
        pass
    concept_ids = referenced_concept_ids(document)
    if not concept_ids:
        return None, False
    try:
        with cohort_concept_id_scope(concept_ids):
            return AnalysisPlan.model_validate(document), True
    except Exception:  # noqa: BLE001 - a genuinely unreadable plan
        return None, False


def measure(roots: Sequence[str]) -> OwnershipLedger:
    ledger = OwnershipLedger()
    seen: set[str] = set()
    for path in iter_plan_paths(roots):
        try:
            document = json.loads(open(path, encoding="utf-8").read())
        except (OSError, ValueError):
            ledger.unreadable_plans += 1
            continue
        if not isinstance(document, dict) or "steps" not in document:
            continue
        plan, granted = load_plan(document)
        if plan is None:
            ledger.unreadable_plans += 1
            continue
        ledger.readable_plans += 1
        ledger.plans_needing_registered_concepts += int(granted)
        ledger.steps.extend(measure_plan(plan, document.get("steps") or [], seen=seen))
    return ledger


def measure_plan(
    plan: AnalysisPlan,
    raw_steps: Sequence[Any],
    *,
    seen: Optional[set[str]] = None,
    ledger: Optional[OwnershipLedger] = None,
) -> list[StepOwnership]:
    """Ask the selector about every step of one plan, under both bounds."""

    rows: list[StepOwnership] = []
    for raw, step in zip(raw_steps, plan.steps):
        key = _step_key(raw)
        if seen is not None:
            if key in seen:
                continue
            seen.add(key)
        answers: dict[str, Optional[str]] = {}
        trace: list = []
        for label, scope in (
            ("upper", None),
            ("lower", _receipt_scope(step.step_id)),
        ):
            candidates: list = []
            try:
                selection = select_standard_executor(
                    step,
                    plan=plan,
                    plausibility_scope=scope,
                    trace=candidates,
                )
            except Exception as error:  # noqa: BLE001 - report, never assume
                answers[label] = None
                if ledger is not None:
                    ledger.raised.append((step.step_id, f"{type(error).__name__}"))
                continue
            answers[label] = selection.analysis_kind if selection is not None else None
            if label == "upper":
                trace = candidates
        rows.append(
            StepOwnership(
                key=key,
                step_id=step.step_id,
                method=str(step.method or ""),
                declared_products=tuple(
                    str(value or "").strip() for value in step.expected_outputs or []
                ),
                upper_owner=answers.get("upper"),
                lower_owner=answers.get("lower"),
                upper_trace=tuple(
                    (c.analysis_kind, c.contract_matches, c.outcome) for c in trace
                ),
            )
        )
    return rows


def unowned_products(steps: Sequence[StepOwnership]) -> Counter:
    """Declared products of steps no owner claims, by frequency."""

    counter: Counter = Counter()
    for step in steps:
        if step.claimed_upper:
            continue
        counter.update(step.declared_products)
    return counter


def _report(ledger: OwnershipLedger, *, top: int) -> None:
    steps = ledger.steps
    total = len(steps)
    if not total:
        print("no readable plans found -- nothing measured")
        return
    upper = sum(1 for step in steps if step.claimed_upper)
    lower = sum(1 for step in steps if step.claimed_lower)
    print(f"plans read {ledger.readable_plans}   unreadable {ledger.unreadable_plans}")
    if ledger.plans_needing_registered_concepts:
        print(
            f"  {ledger.plans_needing_registered_concepts} parsed only after "
            "granting the concept ids the plan itself names"
        )
    print(f"unique recorded steps: {total}")
    print(f"  claimed, upper bound (no receipt owed): {upper:5d}  {upper / total:6.1%}")
    print(f"  claimed, lower bound (receipt owed)   : {lower:5d}  {lower / total:6.1%}")
    if ledger.raised:
        print(
            f"  selector raised on {len(ledger.raised)} step(s) -- counted as unclaimed"
        )

    print("\nowners, by steps claimed (upper bound):")
    for owner, count in Counter(
        step.upper_owner for step in steps if step.claimed_upper
    ).most_common():
        print(f"  {count:5d}  {owner}")

    unclaimed = [step for step in steps if not step.claimed_upper]
    print(f"\nmethods of the {len(unclaimed)} unclaimed steps:")
    for method, count in Counter(step.method for step in unclaimed).most_common(top):
        print(f"  {count:5d}  {method}")

    counter = unowned_products(steps)
    grand = sum(counter.values()) or 1
    print(f"\ndeclared products no owner can emit ({len(counter)} distinct):")
    running = 0
    for product, count in counter.most_common(top):
        running += count
        print(f"  {count:5d}  {running / grand:6.1%} cum   {product}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        default=None,
        help="directory to search recursively for analysis_plan*.json (repeatable)",
    )
    parser.add_argument("--json", dest="json_path", default=None)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args(argv)

    ledger = measure(tuple(args.roots or DEFAULT_ROOTS))
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(ledger.as_dict(), handle, indent=1, sort_keys=True)
        print(f"wrote {args.json_path}")
    _report(ledger, top=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
