"""One typed interface and registry for deterministic plan-step executors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from ..authority.plausibility import FlagOnlyPlausibilityScope
from ..contracts.ownership_verdict import OwnershipVerdict
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

    def claim(
        self, context: StepExecutorContext
    ) -> Optional[StandardExecutorCandidate]:
        """Evaluate ownership and refusal gates without invoking a renderer."""
        if self.applicable is not None and not self.applicable(context):
            return None
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


class AmbiguousExecutorOwnership(RuntimeError):
    """Multiple scientific owners claimed one step; no code may be rendered."""

    code = "ambiguous_executor_ownership"

    def __init__(self, step_id: str, owner_keys: Sequence[str]) -> None:
        self.step_id = step_id
        self.owner_keys = tuple(sorted(owner_keys))
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

    def select(
        self,
        context: StepExecutorContext,
        *,
        trace: Optional[list[StandardExecutorCandidate]] = None,
    ) -> Optional[StandardExecutorSelection]:
        # Registration order is diagnostic presentation only, never authority.
        claims = [(executor, executor.claim(context)) for executor in self._executors]
        owners = [
            (executor, claim)
            for executor, claim in claims
            if claim is not None and claim.contract_matches
        ]
        ambiguous = len(owners) > 1
        trace_start = len(trace) if trace is not None else 0
        if trace is not None:
            trace.extend(
                replace(claim, outcome="ambiguous_ownership")
                if ambiguous and claim.contract_matches
                else claim
                for _, claim in claims
                if claim is not None
            )
        if ambiguous:
            raise AmbiguousExecutorOwnership(
                context.step.step_id, [e.key for e, _ in owners]
            )
        if not owners:
            return None  # The caller still governs unsupported vs bounded Coder.
        owner, claim = owners[0]
        if claim.outcome == "declined_receipt_required":
            return None
        selection = owner.render_selection(context)
        if trace is not None:
            for index in range(trace_start, len(trace)):
                if trace[index] is claim:
                    trace[index] = replace(claim, outcome="selected")
                    break
        return selection
