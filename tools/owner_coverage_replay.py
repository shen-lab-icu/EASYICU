"""Report which plan steps a deterministic executor would own. Zero Provider.

Motivation: on the 2026-07-28 E1 run exactly one of nine executed steps got a
deterministic owner, and that step was the only contested one that did not
fail.  Finding that out cost a real run.  Reading it off a recorded plan costs
nothing, so it should be read first.

The one thing this tool must not do is flatter the plan.  A coverage report
that over-states ownership green-lights exactly the run that then falls
through to the coder -- worse than no report, because it was trusted.  Three
distinct ways to over-state are refused here:

* **Scoring a plan that did not validate.**  An earlier version dropped
  ``robustness_specs`` when the plan failed validation and scanned what was
  left.  That changes the plan's semantics and then reports a precise-looking
  number for a plan nobody would run.  A plan that does not validate is
  ``invalid_plan``: no coverage is produced at all.

* **Calling an unknown a coder step.**  Executors whose readable schema is
  fixed by the *producing* step need the host's ``resolved_bindings``, which
  exist only once the parent has run.  Offline they decline by construction --
  which is not evidence that the coder will run them.  Those steps are
  ``unknown_runtime_binding``, reported apart from both owned and coder.

* **Calling a receipt-gated owner an owner.**  ``select_standard_executor``
  declines an owner whose contract matched when the step also owes a
  host-verified plausibility receipt its deterministic code cannot emit.  With
  the real obligation unknown, an owner that survives only the ungated call is
  ``conditional_receipt``, never owned.  That is the shape that made an earlier
  scan disagree with the real E1 Step 02, which recorded
  ``declined_receipt_required``.

Supply what you know through :class:`SelectionContextSnapshot` and the answers
sharpen: real bindings and real obligations turn ``unknown_*`` verdicts into
definite ones.  Obligations are compiled by the production compiler
(:func:`compile_flag_only_plausibility_scope`), never re-derived here.

Coverage alone is not a launch gate.  A gate needs the study protocol to say
which steps it *requires* be deterministic; open-ended scientific steps may
legitimately go to the Coder.  Pass ``deterministic_required_step_ids`` to get
a gate; without it the report says plainly that it is advisory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

__all__ = [
    "PlanNotScannable",
    "SelectionContextSnapshot",
    "StepCoverage",
    "compile_receipt_obligations",
    "load_plan",
    "main",
    "replay_owner_coverage",
]

OWNED = "owned"
CONDITIONAL_RECEIPT = "conditional_receipt"
UNKNOWN_RUNTIME_BINDING = "unknown_runtime_binding"
CODER = "coder"

#: Ordered worst-to-best for reporting; only ``OWNED`` counts as covered.
VERDICTS = (OWNED, CONDITIONAL_RECEIPT, UNKNOWN_RUNTIME_BINDING, CODER)

_ARTIFACT_PREFIX = "artifact:"


class PlanNotScannable(RuntimeError):
    """The plan cannot be scanned, so no coverage number may be reported."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class StepCoverage:
    """What one step's ownership looks like before anything is spent."""

    step_id: str
    verdict: str
    analysis_kind: str | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "verdict": self.verdict,
            "analysis_kind": self.analysis_kind,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class SelectionContextSnapshot:
    """The typed context the production selector is asked to decide against.

    Each optional field is a fact the host holds at run time and an offline
    scan usually does not.  Leaving one unset is not neutral: it is why a
    verdict comes back ``unknown_*`` instead of definite.  Supplying a wrong
    one would be worse than leaving it out, so nothing here is guessed.
    """

    plan: Any  # AnalysisPlan; typed loosely so this module imports cheaply
    resolved_bindings: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict,
    )
    plausibility_scopes: Mapping[str, Any] = field(default_factory=dict)
    deterministic_required_step_ids: frozenset[str] | None = None

    def bindings_for(self, step_id: str) -> Mapping[str, Any] | None:
        return self.resolved_bindings.get(step_id)

    def scope_for(self, step_id: str) -> Any | None:
        return self.plausibility_scopes.get(step_id)

    def produced_products(self) -> frozenset[str]:
        """Typed products some step in this plan promises to emit.

        ``artifact:`` outputs are excluded: the locked cohort's schema is host
        knowledge that does not depend on a parent step having run, so a step
        reading it is not blocked on a runtime binding.
        """

        products: set[str] = set()
        for step in self.plan.steps:
            for raw in step.expected_outputs or ():
                value = str(raw or "").strip()
                if value and ":" in value and not value.startswith(_ARTIFACT_PREFIX):
                    products.add(value)
        return frozenset(products)

    def unresolved_parent_inputs(self, step: Any) -> tuple[str, ...]:
        """Typed inputs of ``step`` produced upstream whose binding is unknown.

        This is the structural reason an offline scan cannot answer for figure
        and other downstream renderers: their readable schema is fixed by the
        producing step, not by the Planner's product name.
        """

        if self.bindings_for(str(step.step_id)) is not None:
            return ()
        produced = self.produced_products()
        return tuple(
            value
            for raw in step.inputs or ()
            if (value := str(raw or "").strip()) in produced
        )


def load_plan(plan_path: Path) -> Any:
    """Return a fully validated ``AnalysisPlan`` or refuse.

    Nothing is dropped, coerced or retried to make a plan validate.  A plan
    the pipeline would reject is one this tool must not describe.
    """

    from easyicu.research_agent.schema import AnalysisPlan

    try:
        raw = plan_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PlanNotScannable("plan_unreadable", f"{plan_path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise PlanNotScannable("plan_not_json", f"{plan_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PlanNotScannable("plan_not_json", f"{plan_path}: not a JSON object")
    if "steps" not in payload and isinstance(payload.get("plan"), Mapping):
        payload = payload["plan"]
    try:
        return AnalysisPlan.model_validate(payload)
    except Exception as exc:
        raise PlanNotScannable(
            "invalid_plan",
            f"{plan_path.name} does not validate as an AnalysisPlan.\n"
            "  No coverage is reported: a plan the pipeline would reject is\n"
            "  not a plan whose ownership means anything.\n\n"
            f"{exc}",
        ) from exc


def compile_receipt_obligations(
    plan: Any,
    *,
    context: Any,
    raw_input_contracts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile each step's real receipt obligation with the production compiler.

    Re-deriving obligations here would be a second authority that drifts from
    the one the run uses, so this delegates entirely.
    """

    from easyicu.research_agent.authority.plausibility import (
        compile_flag_only_plausibility_scope,
    )

    scopes: dict[str, Any] = {}
    for step in plan.steps:
        scopes[str(step.step_id)] = compile_flag_only_plausibility_scope(
            context=context,
            step=step,
            raw_input_contracts=raw_input_contracts,
        )
    return scopes


def _probe_scope(step_id: str) -> Any:
    """A minimal non-empty obligation: does an owner survive owing a receipt?"""

    from easyicu.research_agent.authority.plausibility import (
        FlagOnlyPlausibilityScope,
    )

    return FlagOnlyPlausibilityScope(
        step_id=step_id,
        expected_columns=("__receipt_probe__",),
        source_contracts_sha256="0" * 64,
        authority_kind="owner_coverage_receipt_probe",
    )


def _select(step: Any, *, snapshot: SelectionContextSnapshot, scope: Any) -> Any:
    from easyicu.research_agent.execution.runners.selection import (
        select_standard_executor,
    )

    return select_standard_executor(
        step,
        plan=snapshot.plan,
        plausibility_scope=scope,
        resolved_bindings=snapshot.bindings_for(str(step.step_id)),
    )


def _coverage_for_step(
    step: Any, *, snapshot: SelectionContextSnapshot
) -> StepCoverage:
    step_id = str(step.step_id)
    known_scope = snapshot.scope_for(step_id)

    if known_scope is not None:
        # The real obligation is known, so one call is the whole answer.
        try:
            selected = _select(step, snapshot=snapshot, scope=known_scope)
        except Exception as exc:
            return StepCoverage(step_id, CODER, note=f"{type(exc).__name__}: {exc}")
        if selected is not None:
            return StepCoverage(step_id, OWNED, selected.analysis_kind)
        return _unowned(step, snapshot=snapshot, note_prefix="")

    try:
        ungated = _select(step, snapshot=snapshot, scope=None)
    except Exception as exc:  # a scan must not die on one step
        return StepCoverage(step_id, CODER, note=f"{type(exc).__name__}: {exc}")
    if ungated is None:
        return _unowned(step, snapshot=snapshot, note_prefix="")

    try:
        gated = _select(step, snapshot=snapshot, scope=_probe_scope(step_id))
    except Exception:
        gated = None
    if gated is not None:
        return StepCoverage(step_id, OWNED, ungated.analysis_kind)
    return StepCoverage(
        step_id,
        CONDITIONAL_RECEIPT,
        ungated.analysis_kind,
        note=(
            "owned only when this step owes no plausibility receipt; this "
            "snapshot does not carry the real obligation, so it is unknown"
        ),
    )


def _unowned(
    step: Any,
    *,
    snapshot: SelectionContextSnapshot,
    note_prefix: str,
) -> StepCoverage:
    """No owner claimed. Say whether that is a finding or a missing input."""

    step_id = str(step.step_id)
    pending = snapshot.unresolved_parent_inputs(step)
    if pending:
        return StepCoverage(
            step_id,
            UNKNOWN_RUNTIME_BINDING,
            note=(
                f"{note_prefix}reads {', '.join(pending)} from an upstream step; "
                "an executor bound to that product's schema cannot be resolved "
                "until the parent has run"
            ),
        )
    return StepCoverage(step_id, CODER, note=note_prefix.strip())


def replay_owner_coverage(
    plan_path: Path | None = None,
    *,
    snapshot: SelectionContextSnapshot | None = None,
) -> list[StepCoverage]:
    """Return one verdict per step, erring toward "unknown" when unsure."""

    if snapshot is None:
        if plan_path is None:
            raise TypeError("replay_owner_coverage needs a plan_path or a snapshot")
        snapshot = SelectionContextSnapshot(plan=load_plan(plan_path))
    return [_coverage_for_step(step, snapshot=snapshot) for step in snapshot.plan.steps]


def _tally(rows: Sequence[StepCoverage]) -> dict[str, int]:
    return {
        verdict: sum(1 for row in rows if row.verdict == verdict)
        for verdict in VERDICTS
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="an analysis_plan*.json from a run")
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable rows"
    )
    parser.add_argument(
        "--require-deterministic",
        metavar="STEP_ID",
        action="append",
        default=[],
        help=(
            "a step the study protocol requires be deterministic. Repeatable. "
            "Given any, this becomes a gate: exit 1 unless every named step is "
            "owned outright."
        ),
    )
    args = parser.parse_args(argv)

    try:
        plan = load_plan(args.plan)
    except PlanNotScannable as exc:
        print(f"not scannable [{exc.reason_code}]: {exc}", file=sys.stderr)
        return 2

    required = frozenset(args.require_deterministic) or None
    snapshot = SelectionContextSnapshot(
        plan=plan,
        deterministic_required_step_ids=required,
    )
    rows = replay_owner_coverage(snapshot=snapshot)

    if required:
        unknown_required = sorted(required - {str(s.step_id) for s in plan.steps})
        if unknown_required:
            print(
                "not scannable [required_step_absent]: the protocol requires "
                f"{', '.join(unknown_required)}, which this plan does not contain.",
                file=sys.stderr,
            )
            return 2

    if args.json:
        print(
            json.dumps(
                {
                    "plan": str(args.plan),
                    "steps": [row.as_dict() for row in rows],
                    "tally": _tally(rows),
                    "deterministic_required_step_ids": sorted(required or ()),
                },
                indent=2,
            )
        )
    else:
        _print_report(rows)

    if not required:
        if not args.json:
            print()
            print(
                "Advisory only. No protocol requirement was declared, so this "
                "is a description, not a gate:\n"
                "an open-ended scientific step may legitimately go to the Coder."
            )
        return 0

    shortfall = [
        row for row in rows if row.step_id in required and row.verdict != OWNED
    ]
    if shortfall:
        print(file=sys.stderr)
        print(
            "gate failed: the protocol requires a deterministic owner for "
            f"{len(shortfall)} step(s) that do not have one:",
            file=sys.stderr,
        )
        for row in shortfall:
            print(f"  {row.step_id}: {row.verdict}", file=sys.stderr)
        return 1
    return 0


def _print_report(rows: Sequence[StepCoverage]) -> None:
    width = max((len(row.step_id) for row in rows), default=10)
    label = {
        OWNED: "owned",
        CONDITIONAL_RECEIPT: "CONDITIONAL",
        UNKNOWN_RUNTIME_BINDING: "UNKNOWN",
        CODER: "-- coder --",
    }
    for index, row in enumerate(rows, start=1):
        shown = row.analysis_kind or ""
        print(f"{index:>2}. {row.step_id:<{width}}  {label[row.verdict]:<12} {shown}")
        if row.note:
            print(f"    {' ' * width}  {row.note}")

    tally = _tally(rows)
    total = len(rows)
    print()
    print(f"owned outright          : {tally[OWNED]}/{total}")
    print(f"unknown (receipt)       : {tally[CONDITIONAL_RECEIPT]}/{total}")
    print(f"unknown (parent binding): {tally[UNKNOWN_RUNTIME_BINDING]}/{total}")
    print(f"falls to the coder      : {tally[CODER]}/{total}")
    if tally[CONDITIONAL_RECEIPT] or tally[UNKNOWN_RUNTIME_BINDING]:
        print()
        print(
            "An UNKNOWN step is neither owned nor proven to reach the coder: "
            "this snapshot\nlacks the run-time fact that decides it. Supply "
            "real bindings or obligations to\nresolve it; do not read it as "
            "either answer."
        )


if __name__ == "__main__":  # pragma: no cover - exercised via tests
    raise SystemExit(main())
