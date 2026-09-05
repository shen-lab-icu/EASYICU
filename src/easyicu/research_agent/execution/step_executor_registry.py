"""One typed interface and registry for deterministic plan-step executors."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from ..authority.plausibility import FlagOnlyPlausibilityScope
from ..contracts.ownership_verdict import OwnershipContractDetail, OwnershipVerdict
from ..schema import AnalysisPlan, AnalysisStep


@dataclass(frozen=True, slots=True)
class StandardExecutorSelection:
    """One deterministic implementation of already-fixed Planner science."""

    analysis_kind: str
    selection_reason: str
    progress_message: str
    code: str
    consumed_input_keys: tuple[str, ...]
    host_sealed_renderer: bool = False


@dataclass(frozen=True, slots=True)
class StandardExecutorCandidate:
    """One executor's own recorded answer to an ownership query."""

    analysis_kind: str
    contract_matches: bool
    outcome: str
    missing_declarations: tuple[str, ...] = ()
    decline_reason: str = ""
    details: tuple[OwnershipContractDetail, ...] = ()


@dataclass(frozen=True, slots=True)
class StepExecutorContext:
    """The single render shape presented to every deterministic executor."""

    step: AnalysisStep
    plan: AnalysisPlan
    plausibility_scope: Optional[FlagOnlyPlausibilityScope] = None
    resolved_bindings: Optional[Mapping[str, Any]] = None
    trajectory_scientific_runtime_authority: Optional[Mapping[str, Any]] = None
    current_case_scientific_runtime_authority: Any = None
    scientific_runtime_projection_sha256: str = ""

    @property
    def receipt_required(self) -> bool:
        return bool(
            self.plausibility_scope is not None
            and self.plausibility_scope.expected_columns
        )

    def typed_cohort_inputs(self) -> tuple[str, ...]:
        from .runners.typed_input_binding import sole_typed_cohort_input

        value = sole_typed_cohort_input(self.step)
        return (value,) if value else ()


TextValue = Union[str, Callable[[StepExecutorContext], str]]
InputKeys = Callable[[StepExecutorContext], Sequence[str]]
Ownership = Callable[[StepExecutorContext], Union[bool, OwnershipVerdict]]
Renderer = Callable[[StepExecutorContext], str]
Applicability = Callable[[StepExecutorContext], bool]
DeclarationVerdict = Callable[[StepExecutorContext], OwnershipVerdict]
ContractDetails = Callable[[StepExecutorContext], tuple[OwnershipContractDetail, ...]]


def _text(value: TextValue, context: StepExecutorContext) -> str:
    return value(context) if callable(value) else value


@dataclass(frozen=True)
class StepExecutor:
    """One runner-owned declaration behind a uniform context-to-code seam."""

    key: str
    owns: Ownership
    render: Renderer
    analysis_kind: TextValue
    selection_reason: TextValue
    progress_message: TextValue
    consumed_input_keys: InputKeys
    host_sealed_renderer: bool = False
    blocks_on_plausibility_receipt: bool = False
    applicable: Optional[Applicability] = None
    declaration_verdict: Optional[DeclarationVerdict] = None
    accepts_figure_presentation: bool = False
    contract_details: Optional[ContractDetails] = None

    def claim(
        self, context: StepExecutorContext
    ) -> Optional[StandardExecutorCandidate]:
        """Evaluate ownership and refusal gates without invoking a renderer."""
        if self.applicable is not None and not self.applicable(context):
            return None
        if not self.accepts_figure_presentation and any(
            panel.presentation is not None for panel in context.step.figure_panels
        ):
            return StandardExecutorCandidate(
                analysis_kind=self.key,
                contract_matches=False,
                outcome="contract_declined",
                decline_reason="unsupported_figure_presentation: renderer has not declared support for planned display parameters",
            )
        answer = self.owns(context)
        verdict = answer if isinstance(answer, OwnershipVerdict) else None
        claimed = verdict.claimed if verdict is not None else bool(answer)
        if not claimed:
            if verdict is None and self.declaration_verdict is not None:
                verdict = self.declaration_verdict(context)
            return StandardExecutorCandidate(
                analysis_kind=(
                    verdict.analysis_kind if verdict is not None else self.key
                ),
                contract_matches=False,
                outcome="contract_declined",
                missing_declarations=(
                    verdict.missing_declarations if verdict is not None else ()
                ),
                decline_reason=verdict.reason if verdict is not None else "",
            )
        if self.blocks_on_plausibility_receipt and context.receipt_required:
            return StandardExecutorCandidate(
                analysis_kind=self.key,
                contract_matches=True,
                outcome="declined_receipt_required",
            )
        return StandardExecutorCandidate(
            analysis_kind=self.key,
            contract_matches=True,
            outcome="claimed",
        )

    def render_selection(
        self, context: StepExecutorContext
    ) -> StandardExecutorSelection:
        """Render only after the registry has established a unique owner."""
        return StandardExecutorSelection(
            analysis_kind=_text(self.analysis_kind, context),
            selection_reason=_text(self.selection_reason, context),
            progress_message=_text(self.progress_message, context),
            code=self.render(context),
            consumed_input_keys=tuple(self.consumed_input_keys(context)),
            host_sealed_renderer=self.host_sealed_renderer,
        )


STANDARD_EXECUTOR_CANDIDATE_SCHEMA_VERSION = "easyicu.standard_executor_candidates/3"


@dataclass(frozen=True, slots=True)
class StepExecutorDecision:
    """One complete ownership query, independent of code generation.

    This short-lived decision owns its diagnostic projection and render target;
    it is not approval to execute code or to bypass the caller's scientific gates.
    """

    candidates: tuple[StandardExecutorCandidate, ...]
    claimed_by: str | None
    step_id: str
    declared_method: str
    declared_outputs: tuple[str, ...]
    declared_inputs: tuple[str, ...]
    _context: StepExecutorContext = field(repr=False, compare=False)
    _executor: StepExecutor | None = field(repr=False, compare=False)

    def render_selection(self) -> StandardExecutorSelection | None:
        return (
            self._executor.render_selection(self._context) if self._executor else None
        )

    def report(self) -> dict[str, Any]:
        """Project the same query without re-running any owner or classifier."""
        entries: list[dict[str, Any]] = []
        for candidate in self.candidates:
            entries.append(
                {
                    "kind": "owner",
                    "analysis_kind": candidate.analysis_kind,
                    "contract_matches": candidate.contract_matches,
                    "outcome": candidate.outcome,
                    "missing_declarations": list(candidate.missing_declarations),
                    "decline_reason": candidate.decline_reason,
                }
            )
            for detail in candidate.details:
                entries.append(
                    {
                        "kind": "detail",
                        "analysis_kind": detail.analysis_kind,
                        "matches": detail.matches,
                        **({"error": detail.error} if detail.error is not None else {}),
                    }
                )
        return {
            "schema_version": STANDARD_EXECUTOR_CANDIDATE_SCHEMA_VERSION,
            "step_id": self.step_id,
            "claimed_by": self.claimed_by,
            "trace_available": True,
            "owning_candidates": [
                c.analysis_kind for c in self.candidates if c.contract_matches
            ],
            "declined_after_match": [
                c.analysis_kind
                for c in self.candidates
                if c.contract_matches and c.outcome != "selected"
            ],
            "declared_method": self.declared_method,
            "declared_outputs": list(self.declared_outputs),
            "declared_typed_inputs": [
                value for value in self.declared_inputs if ":" in value
            ],
            "declared_raw_input_count": sum(
                ":" not in value for value in self.declared_inputs
            ),
            "candidates": entries,
        }


class AmbiguousExecutorOwnership(RuntimeError):
    """Multiple scientific owners claimed one step; no code may be rendered."""

    code = "ambiguous_executor_ownership"

    def __init__(
        self,
        step_id: str,
        owner_keys: Sequence[str],
        candidates: tuple[StandardExecutorCandidate, ...] = (),
    ) -> None:
        self.step_id = step_id
        self.owner_keys = tuple(sorted(owner_keys))
        self.candidates = candidates
        super().__init__(
            f"{self.code}: step={step_id}; owners={','.join(self.owner_keys)}"
        )


class StepExecutorRegistry:
    """Exactly one applicable owner must claim a step before any rendering."""

    def __init__(self) -> None:
        self._executors: list[StepExecutor] = []
        self._keys: set[str] = set()

    @property
    def executors(self) -> tuple[StepExecutor, ...]:
        return tuple(self._executors)

    def declare(self, executor: StepExecutor) -> None:
        key = executor.key.strip()
        if not key:
            raise ValueError("step executor key is required")
        if key in self._keys:
            raise ValueError(f"step executor already declared: {key}")
        self._keys.add(key)
        self._executors.append(executor)

    def resolve(self, context: StepExecutorContext) -> StepExecutorDecision:
        """Ask all owners once; a read-only query never renders code."""
        # Registration order is diagnostic presentation only, never authority.
        claims = [(executor, executor.claim(context)) for executor in self._executors]
        owners = [
            (executor, claim)
            for executor, claim in claims
            if claim is not None and claim.contract_matches
        ]
        if len(owners) > 1:
            raise AmbiguousExecutorOwnership(
                context.step.step_id,
                [e.key for e, _ in owners],
                tuple(
                    replace(claim, outcome="ambiguous_ownership")
                    if claim.contract_matches
                    else claim
                    for _, claim in claims
                    if claim is not None
                ),
            )
        owner = owners[0][0] if owners and owners[0][1].outcome == "claimed" else None
        candidates = []
        for executor, claim in claims:
            if claim is None:
                continue
            if executor.contract_details is not None:
                try:
                    details = executor.contract_details(context)
                except Exception as exc:  # diagnostics cannot grant or refuse ownership
                    details = (
                        OwnershipContractDetail(
                            f"{executor.key}:details",
                            False,
                            f"{type(exc).__name__}: {exc}"[:200],
                        ),
                    )
                claim = replace(claim, details=details)
            candidates.append(
                replace(claim, outcome="selected") if executor is owner else claim
            )
        return StepExecutorDecision(
            candidates=tuple(candidates),
            claimed_by=_text(owner.analysis_kind, context) if owner else None,
            step_id=str(context.step.step_id),
            declared_method=str(context.step.method or ""),
            declared_outputs=tuple(str(v) for v in context.step.expected_outputs),
            declared_inputs=tuple(str(v) for v in context.step.inputs),
            _context=context,
            _executor=owner,
        )

    def select(
        self,
        context: StepExecutorContext,
        *,
        trace: Optional[list[StandardExecutorCandidate]] = None,
    ) -> Optional[StandardExecutorSelection]:
        """Compatibility entrypoint for callers that explicitly request code."""
        try:
            decision = self.resolve(context)
        except AmbiguousExecutorOwnership as exc:
            if trace is not None:
                trace.extend(exc.candidates)
            raise
        if trace is not None:
            trace.extend(decision.candidates)
        return decision.render_selection()
