"""Replay which plan steps a deterministic executor owned. Zero Provider.

**This is a post-run replay, not a preflight.**  It reads a plan a run already
executed together with that run's own sealed context, and reports what the
production selector concluded.  That is useful -- on the 2026-07-28 E1 run only
three of twelve steps had a deterministic owner, which is most of the
explanation for why the run failed -- but it is *not* a guarantee obtainable
before spending money.

The gap is specific and worth stating rather than blurring: steps the run never
reached have no recorded bindings, so their owners cannot be resolved and they
come back ``unknown_runtime_binding``.  In the real E1 plan all three
unexecuted figure steps land there.  Turning this into a genuine preflight
requires **prospective** bindings -- pre-compiling each unexecuted step's
readable schema from the producing step's Planner-declared typed product
contract, rather than from a binding that does not exist yet -- and a preflight
that still reports ``unknown`` for any step has not answered the question it
was asked.  That work is not done here.

The one thing this tool must not do is state something it does not know.  A
coverage report that over-states ownership green-lights exactly the run that
then falls through to the coder -- worse than no report, because it was
trusted.  An over-strict one condemns a plan that ran fine.  Both are the same
defect: answering in whichever direction is available rather than saying which
fact is missing.  Four refusals:

* **Scoring a plan that did not validate.**  An earlier version dropped
  ``robustness_specs`` when the plan failed validation and scanned what was
  left.  That changes the plan's semantics and then reports a precise-looking
  number for a plan nobody would run.  A plan that does not validate is
  ``invalid_plan``: no coverage is produced at all.

* **Calling a missing registry an invalid plan.**  A plan may name a
  ``concept_id`` that is not a packaged concept because the run registered it
  as a pre-materialised cohort column.  Offline that registry does not exist,
  so validation fails for a reason that says nothing about the plan; the E1
  plan trips this on ``icu_readmission`` and the pipeline accepted it.  That
  is ``missing_validation_context``, and ``--run-dir`` resolves it from the
  run's own digest-bound authority.

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
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

__all__ = [
    "PlanNotScannable",
    "RunValidationContext",
    "SelectionContextSnapshot",
    "StepCoverage",
    "compile_receipt_obligations",
    "load_plan",
    "load_run_context",
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


@dataclass(frozen=True, slots=True)
class RunValidationContext:
    """Run-scoped facts an offline scan lacks, taken only from digest-bound records.

    Every field here is read from an artifact the run itself sealed, and the
    cohort column registry is reached through the capsule's
    ``materialized_cohort_authority_ref`` -- file name *and* sha256 -- rather
    than by picking whichever ``cohort_authority.*.json`` happens to be in the
    directory.  A registry assembled the loose way would let this tool validate
    a plan against a cohort the run never used, which is the same class of
    error as scoring a plan it had quietly edited.
    """

    run_dir: Path
    cohort_sha256: str
    cohort_columns: frozenset[str]
    cohort_authority_path: Path
    resolved_bindings: Mapping[str, Mapping[str, Any]]
    raw_input_contracts: Mapping[str, Mapping[str, Any]]
    plan_authority_path: Path
    plan_authority_sha256: str

    @property
    def steps_with_bindings(self) -> frozenset[str]:
        return frozenset(self.resolved_bindings)


_REASON = "missing_validation_context"


def _read_json(path: Path, *, reason: str = _REASON) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PlanNotScannable(reason, f"{path}: {exc}") from exc
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise PlanNotScannable(reason, f"{path}: {exc}") from exc


def _inside(run_dir: Path, relative: str, *, what: str) -> Path:
    """Resolve a manifest-declared path, refusing anything outside the run."""

    if not relative:
        raise PlanNotScannable(_REASON, f"{what}: empty path")
    candidate = (run_dir / relative).resolve()
    root = run_dir.resolve()
    if not candidate.is_relative_to(root):
        raise PlanNotScannable(
            _REASON, f"{what}: {relative!r} escapes the run directory"
        )
    return candidate


def _verified_bytes(path: Path, declared: str, *, what: str) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise PlanNotScannable(_REASON, f"{what}: {path}: {exc}") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if not declared or actual != declared.lower():
        raise PlanNotScannable(
            _REASON,
            f"{what}: {path.name} does not match its declared digest.\n"
            f"  declared {declared or '(none)'}\n  actual   {actual}",
        )
    return payload


def _plan_authority(run_dir: Path, manifest: Mapping[str, Any]) -> tuple[Path, str]:
    """The one plan revision this run executed, agreed by manifest and store.

    ``current_plan_authority`` names it and the EvidenceStore record for the
    same ``evidence_id`` repeats the path and digest.  Both are checked: a
    manifest pointer alone would let an edited authority block redirect the
    scan, and the store record alone would not say which revision was current.
    """

    authority = manifest.get("current_plan_authority")
    if not isinstance(authority, Mapping):
        raise PlanNotScannable(
            _REASON,
            f"{run_dir.name}/manifest.json declares no current_plan_authority, "
            "so there is no\n  authoritative plan revision to scan.",
        )
    evidence_id = str(authority.get("evidence_id") or "")
    relative = str(authority.get("relative_path") or "")
    declared = str(authority.get("sha256") or "")
    path = _inside(run_dir, relative, what="current_plan_authority")
    _verified_bytes(path, declared, what="current_plan_authority")

    records = manifest.get("evidence")
    matches = [
        record
        for record in (records if isinstance(records, list) else ())
        if isinstance(record, Mapping)
        and str(record.get("evidence_id") or "") == evidence_id
    ]
    if len(matches) != 1:
        raise PlanNotScannable(
            _REASON,
            f"the evidence store holds {len(matches)} records for "
            f"{evidence_id!r}; exactly one must back\n  the current plan "
            "authority.",
        )
    record = matches[0]
    if (
        str(record.get("sha256") or "").lower() != declared.lower()
        or str(record.get("relative_path") or "") != relative
    ):
        raise PlanNotScannable(
            _REASON,
            f"the evidence record for {evidence_id!r} disagrees with "
            "current_plan_authority;\n  the run's own two records of which plan "
            "is authoritative do not match.",
        )
    return path, declared.lower()


def _bindings_from_manifest(
    run_dir: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    """Read each step's binding capsule at the digest the manifest recorded.

    Scanning ``resolved_inputs/*.json`` off the directory would accept a file
    the run never sealed -- a leftover from an earlier attempt, or an edited
    one -- and every verdict downstream would inherit it.
    """

    records = manifest.get("per_step_records")
    if not isinstance(records, list):
        raise PlanNotScannable(
            _REASON, f"{run_dir.name}/manifest.json has no per_step_records"
        )
    bindings: dict[str, Mapping[str, Any]] = {}
    contracts: dict[str, Mapping[str, Any]] = {}
    seen: set[str] = set()
    referenced: set[Path] = set()
    for record in records:
        if not isinstance(record, Mapping):
            continue
        step_id = str(record.get("step_id") or "")
        if not step_id:
            continue
        if step_id in seen:
            raise PlanNotScannable(
                _REASON,
                f"manifest.per_step_records names {step_id!r} more than once; "
                "which binding is\n  authoritative is undecidable.",
            )
        seen.add(step_id)
        relative = record.get("resolved_inputs_path")
        declared = record.get("resolved_inputs_sha256")
        if relative is None and declared is None:
            continue  # the step never resolved its inputs; absent, not empty
        if bool(relative) != bool(declared):
            raise PlanNotScannable(
                _REASON,
                f"{step_id}: half a binding receipt (path={relative!r}, "
                f"sha256={declared!r}); a path without\n  a digest is not "
                "evidence of anything.",
            )
        path = _inside(run_dir, str(relative), what=f"{step_id} resolved_inputs")
        payload = _verified_bytes(path, str(declared), what=f"{step_id} bindings")
        referenced.add(path)
        capsule = json.loads(payload.decode("utf-8"))
        if not isinstance(capsule, Mapping):
            raise PlanNotScannable(_REASON, f"{path}: not a JSON object")
        if str(capsule.get("step_id") or "") != step_id:
            raise PlanNotScannable(
                _REASON,
                f"{path.name} names step {capsule.get('step_id')!r} but the "
                f"manifest filed it under {step_id!r}.",
            )
        step_inputs = capsule.get("inputs")
        if isinstance(step_inputs, Mapping):
            bindings[step_id] = dict(step_inputs)
        raw = capsule.get("raw_input_contracts")
        if isinstance(raw, Mapping):
            contracts[step_id] = dict(raw)

    directory = run_dir / "resolved_inputs"
    if directory.is_dir():
        stray = sorted(
            path.name
            for path in directory.glob("*.json")
            if path.resolve() not in referenced
        )
        if stray:
            raise PlanNotScannable(
                _REASON,
                "resolved_inputs/ holds "
                f"{', '.join(stray)}, which no manifest record claims.\n"
                "  An unclaimed binding capsule has no authority and may be a "
                "stale attempt; refusing\n  rather than guessing which files "
                "belong to this run.",
            )
    return bindings, contracts


def load_run_context(run_dir: Path) -> RunValidationContext:
    """Load the digest-bound validation context a run sealed for itself.

    Every fact is verified against the digest the run recorded for it, and
    nothing is discovered by listing a directory.
    """

    manifest = _read_json(run_dir / "manifest.json")
    if not isinstance(manifest, Mapping):
        raise PlanNotScannable(_REASON, f"{run_dir}: manifest.json is not an object")
    plan_path, plan_sha256 = _plan_authority(run_dir, manifest)

    capsule = _read_json(run_dir / "run_input_capsule.json")
    if not isinstance(capsule, Mapping):
        raise PlanNotScannable(
            _REASON, f"{run_dir}: run_input_capsule.json is not an object"
        )
    cohort_sha256 = str(capsule.get("cohort_sha256") or "")
    ref = capsule.get("materialized_cohort_authority_ref")
    if not cohort_sha256 or not isinstance(ref, Mapping):
        raise PlanNotScannable(
            _REASON,
            f"{run_dir.name}/run_input_capsule.json carries no materialised "
            "cohort authority reference,\n"
            "  so there is no digest-bound column registry to validate a plan "
            "against.",
        )
    authority_path = _inside(
        run_dir, str(ref.get("file") or ""), what="cohort authority"
    )
    payload = _verified_bytes(
        authority_path, str(ref.get("sha256") or ""), what="cohort authority"
    )
    authority = json.loads(payload.decode("utf-8"))
    if str(authority.get("cohort_sha256") or "") != cohort_sha256:
        raise PlanNotScannable(
            _REASON,
            f"{authority_path.name} describes a different cohort than the run "
            "capsule names;\n  its column registry is not this run's.",
        )
    columns = frozenset(str(c) for c in (authority.get("cohort_columns") or ()))
    if not columns:
        raise PlanNotScannable(
            _REASON, f"{authority_path.name} declares no cohort columns"
        )

    bindings, contracts = _bindings_from_manifest(run_dir, manifest)
    return RunValidationContext(
        run_dir=run_dir,
        cohort_sha256=cohort_sha256,
        cohort_columns=columns,
        cohort_authority_path=authority_path,
        resolved_bindings=bindings,
        raw_input_contracts=contracts,
        plan_authority_path=plan_path,
        plan_authority_sha256=plan_sha256,
    )


_UNKNOWN_CONCEPT_ID = re.compile(r"unknown concept_id:\s*(\S+)")


def _unknown_concept_ids(exc: BaseException) -> tuple[str, ...]:
    """Concept ids the plan names that static dictionary validation rejects."""

    errors = getattr(exc, "errors", None)
    messages: list[str] = []
    if callable(errors):
        try:
            messages = [str(item.get("msg", "")) for item in errors()]
        except Exception:  # pragma: no cover - pydantic shape change
            messages = []
    if not messages:
        messages = [str(exc)]
    found: list[str] = []
    for message in messages:
        for match in _UNKNOWN_CONCEPT_ID.finditer(message):
            found.append(match.group(1))
    return tuple(dict.fromkeys(found))


def load_plan(
    plan_path: Path, *, run_context: "RunValidationContext | None" = None
) -> Any:
    """Return a fully validated ``AnalysisPlan`` or refuse.

    Nothing is dropped, coerced or retried to make a plan validate.

    Two refusals, and the difference between them matters.  A plan may name a
    ``concept_id`` that is not in the packaged dictionary because the run
    registered it as a pre-materialised cohort column
    (``register_cohort_concept_ids``); the E1 plan does exactly this for
    ``icu_readmission`` and the pipeline accepted it.  Offline that column
    registry does not exist, so validation fails for a reason that says
    nothing about the plan.  Reporting that as ``invalid_plan`` -- "a plan the
    pipeline would reject" -- is a false statement about a run that happened.
    That case is ``missing_validation_context``: supply the run's digest-bound
    context and ask again.
    """

    from easyicu.research_agent.cohort.schema import cohort_concept_id_scope
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

    if run_context is not None:
        actual = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if actual != run_context.plan_authority_sha256:
            raise PlanNotScannable(
                "plan_not_authority",
                f"{plan_path.name} is not the plan {run_context.run_dir.name} "
                "executed.\n"
                f"  scanned    {actual}\n"
                f"  authority  {run_context.plan_authority_sha256} "
                f"({run_context.plan_authority_path.name})\n"
                "  A run's context only explains the plan that run executed; "
                "pairing it with a\n  different revision produces verdicts for "
                "neither.",
            )

    registered = () if run_context is None else run_context.cohort_columns
    try:
        with cohort_concept_id_scope(registered):
            return AnalysisPlan.model_validate(payload)
    except Exception as exc:
        pending = _unknown_concept_ids(exc)
        if pending:
            # Prove the claim instead of pattern-matching it: if registering
            # exactly these ids makes the plan validate, the only thing wrong
            # was the missing registry.
            try:
                with cohort_concept_id_scope([*registered, *pending]):
                    AnalysisPlan.model_validate(payload)
            except Exception:
                pass
            else:
                named = ", ".join(pending)
                if run_context is None:
                    raise PlanNotScannable(
                        "missing_validation_context",
                        f"{plan_path.name} names {named}, which the packaged "
                        "concept dictionary does not contain.\n"
                        "  A run registers its pre-materialised cohort columns "
                        "before planning, so this is\n"
                        "  most likely a real column of that run's cohort and "
                        "NOT an invalid plan.\n"
                        "  Offline this tool cannot tell. Pass --run-dir to "
                        "supply the run's own\n"
                        "  digest-bound column registry and ask again.",
                    ) from exc
                raise PlanNotScannable(
                    "missing_validation_context",
                    f"{plan_path.name} names {named}, which is neither a "
                    "packaged concept nor one of the\n"
                    f"  {len(run_context.cohort_columns)} columns of the cohort "
                    f"bound to {run_context.run_dir.name}.\n"
                    "  Either this plan belongs to a different run or that "
                    "run's authority is incomplete;\n"
                    "  no coverage is reported for a plan whose validation "
                    "context does not match it.",
                ) from exc
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


def receipt_obligations_from_run(
    plan: Any, *, run_context: RunValidationContext
) -> dict[str, Any]:
    """Compile the real obligation for each step whose run recorded its contracts.

    A step with no recorded contracts is left out rather than given an empty
    obligation: "this step owes no receipt" and "this run never got far enough
    to say" are different facts, and only the first one is an answer.
    """

    from easyicu.research_agent.authority.plausibility import (
        compile_flag_only_plausibility_scope,
    )

    scopes: dict[str, Any] = {}
    for step in plan.steps:
        step_id = str(step.step_id)
        recorded = run_context.raw_input_contracts.get(step_id)
        if recorded is None:
            continue
        scopes[step_id] = compile_flag_only_plausibility_scope(
            context=None,
            step=step,
            raw_input_contracts=recorded,
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
    parser.add_argument(
        "plan",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "an analysis_plan*.json from a run. Optional with --run-dir, which "
            "knows which revision that run treated as authoritative; given "
            "both, the file must BE that revision."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "the run directory that produced this plan. Supplies the run's own "
            "digest-bound cohort column registry, typed product bindings and "
            "receipt obligations, turning unknown verdicts into definite ones."
        ),
    )
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

    if args.plan is None and args.run_dir is None:
        parser.error("give a plan path, a --run-dir, or both")
    try:
        run_context = None if args.run_dir is None else load_run_context(args.run_dir)
        plan_path = args.plan
        if plan_path is None:
            assert run_context is not None
            plan_path = run_context.plan_authority_path
        plan = load_plan(plan_path, run_context=run_context)
    except PlanNotScannable as exc:
        print(f"not scannable [{exc.reason_code}]: {exc}", file=sys.stderr)
        return 2

    required = frozenset(args.require_deterministic) or None
    if run_context is None:
        snapshot = SelectionContextSnapshot(
            plan=plan,
            deterministic_required_step_ids=required,
        )
    else:
        snapshot = SelectionContextSnapshot(
            plan=plan,
            resolved_bindings=run_context.resolved_bindings,
            plausibility_scopes=receipt_obligations_from_run(
                plan, run_context=run_context
            ),
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
    print()
    print(
        "This is a post-run REPLAY, not a preflight. A step the run never "
        "reached has no\nrecorded binding, so its owner cannot be resolved "
        "here at all. Using this as a\nlaunch guarantee would need prospective "
        "bindings compiled from the producing step's\ndeclared typed product "
        "contract, and would have to leave no UNKNOWN at all."
    )


if __name__ == "__main__":  # pragma: no cover - exercised via tests
    raise SystemExit(main())
