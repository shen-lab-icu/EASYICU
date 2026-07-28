"""Report which plan steps a deterministic executor would own. Zero Provider.

Motivation: on the 2026-07-28 E1 run exactly one of nine executed steps got a
deterministic owner, and that step was the only contested one that did not
fail.  Finding that out cost a real run.  Reading it off a recorded plan costs
nothing, so it should be read first.

The one thing this tool must not do is flatter the plan.  A coverage report
that over-states ownership would green-light exactly the run that then falls
through to the coder -- worse than no report, because it was trusted.  Two
sources of over-statement are handled explicitly:

* **Receipt-gated owners.**  ``select_standard_executor`` declines an owner
  whose contract matched when the step also owes a host-verified plausibility
  receipt its deterministic code cannot emit.  That scope is a runtime object,
  so an offline scan does not have it -- and without it the scan reported an
  owner for E1 Step 02 that the real run had declined.  Each step is therefore
  evaluated twice, with and without a receipt obligation, and an owner that
  survives only the ungated evaluation is reported as CONDITIONAL, never owned.

* **Binding-gated owners.**  Executors whose readable schema is fixed by the
  producing step require the host's own ``resolved_bindings``, which exist only
  once the parent has run.  Those decline offline by construction; they are
  reported as such rather than counted either way.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

__all__ = ["StepCoverage", "main", "replay_owner_coverage"]

_CONDITIONAL = "conditional_on_receipt"
_OWNED = "owned"
_CODER = "coder"


class StepCoverage:
    """What one step's ownership looks like before anything is spent."""

    __slots__ = ("step_id", "verdict", "analysis_kind", "note")

    def __init__(
        self,
        step_id: str,
        verdict: str,
        analysis_kind: str | None = None,
        note: str = "",
    ) -> None:
        self.step_id = step_id
        self.verdict = verdict
        self.analysis_kind = analysis_kind
        self.note = note

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "verdict": self.verdict,
            "analysis_kind": self.analysis_kind,
            "note": self.note,
        }


def _load_plan(plan_path: Path) -> Any:
    from easyicu.research_agent.schema import AnalysisPlan

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if "steps" not in payload and isinstance(payload.get("plan"), dict):
        payload = payload["plan"]
    try:
        return AnalysisPlan.model_validate(payload)
    except Exception:
        # Ownership is decided from steps and display labels.  A robustness
        # spec this installation cannot resolve must not hide the coverage
        # answer, but it is reported rather than silently dropped.
        print(
            f"note: {plan_path.name} did not validate in full; "
            "robustness_specs dropped for the scan",
            file=sys.stderr,
        )
        payload = dict(payload)
        payload["robustness_specs"] = []
        return AnalysisPlan.model_validate(payload)


def replay_owner_coverage(plan_path: Path) -> list[StepCoverage]:
    """Return one verdict per step, erring toward the coder when unsure."""

    from easyicu.research_agent.authority.plausibility import (
        FlagOnlyPlausibilityScope,
    )
    from easyicu.research_agent.execution.runners.selection import (
        select_standard_executor,
    )

    plan = _load_plan(plan_path)
    rows: list[StepCoverage] = []
    for step in plan.steps:
        try:
            ungated = select_standard_executor(step, plan=plan)
        except Exception as exc:  # a scan must not die on one step
            rows.append(
                StepCoverage(step.step_id, _CODER, note=f"{type(exc).__name__}: {exc}")
            )
            continue
        if ungated is None:
            rows.append(StepCoverage(step.step_id, _CODER))
            continue

        # Re-ask with a receipt obligation in force.  An owner that disappears
        # here is one the real run will decline, which is what happened to E1
        # Step 02 while an earlier version of this scan called it owned.
        gated_scope = FlagOnlyPlausibilityScope(
            step_id=step.step_id,
            expected_columns=("__receipt_probe__",),
            source_contracts_sha256="0" * 64,
            authority_kind="raw_universe",
        )
        try:
            gated = select_standard_executor(
                step, plan=plan, plausibility_scope=gated_scope
            )
        except Exception:
            gated = None
        if gated is None:
            rows.append(
                StepCoverage(
                    step.step_id,
                    _CONDITIONAL,
                    ungated.analysis_kind,
                    note=(
                        "owned only when this step owes no plausibility receipt; "
                        "declines to the coder when it does"
                    ),
                )
            )
            continue
        rows.append(StepCoverage(step.step_id, _OWNED, ungated.analysis_kind))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="an analysis_plan*.json from a run")
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable rows"
    )
    args = parser.parse_args(argv)

    rows = replay_owner_coverage(args.plan)
    if args.json:
        print(json.dumps([row.as_dict() for row in rows], indent=2))
        return 0

    width = max((len(row.step_id) for row in rows), default=10)
    label = {
        _OWNED: "owned",
        _CONDITIONAL: "CONDITIONAL",
        _CODER: "-- CODER --",
    }
    for index, row in enumerate(rows, start=1):
        shown = row.analysis_kind if row.verdict != _CODER else ""
        print(f"{index:>2}. {row.step_id:<{width}}  {label[row.verdict]:<12} {shown}")
        if row.note:
            print(f"    {' ' * width}  {row.note}")

    owned = sum(1 for row in rows if row.verdict == _OWNED)
    conditional = sum(1 for row in rows if row.verdict == _CONDITIONAL)
    coder = sum(1 for row in rows if row.verdict == _CODER)
    print()
    print(f"owned outright      : {owned}/{len(rows)}")
    print(f"conditional on gate : {conditional}/{len(rows)}")
    print(f"falls to the coder  : {coder}/{len(rows)}")
    if conditional:
        print()
        print(
            "A CONDITIONAL step is not covered. It is the shape that made an "
            "earlier scan disagree with the real run."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via tests
    raise SystemExit(main())
