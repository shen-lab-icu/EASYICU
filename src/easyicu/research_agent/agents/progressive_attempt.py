"""Atomic result contract for one Progressive Planner attempt."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ..planning.progressive_artifacts import ProgressiveCompileReplayAttempt
from ..planning.progressive_contract import (
    ProgressiveFoundationMaterialization,
    ProgressivePlanCompileReceipt,
    ProgressivePlanOutline,
    ProgressivePlanSkeleton,
    ProgressiveStepMaterialization,
)
from ..schema import AnalysisPlan


@dataclass(frozen=True)
class ProgressivePlannerRunFacts:
    """Immutable provenance emitted with one Progressive Planner attempt."""

    prompt_metrics: Mapping[str, Any]
    compile_receipt: Optional[ProgressivePlanCompileReceipt]
    outline: Optional[ProgressivePlanOutline]
    foundation: Optional[ProgressiveFoundationMaterialization]
    materializations: tuple[ProgressiveStepMaterialization, ...]
    compile_failure_attempts: tuple[ProgressiveCompileReplayAttempt, ...]
    skeleton: Optional[ProgressivePlanSkeleton]
    resume_validated: bool
    dropped_plan_keys: Mapping[str, tuple[str, ...]]

    @property
    def complete_for_persistence(self) -> bool:
        return bool(
            self.outline is not None
            and self.foundation is not None
            and self.materializations
            and self.skeleton is not None
            and self.compile_receipt is not None
        )


@dataclass(frozen=True)
class ProgressivePlannerAttemptResult:
    """Planner output and provenance frozen at the same return boundary."""

    output: AnalysisPlan | ProgressivePlanOutline
    facts: ProgressivePlannerRunFacts


@dataclass
class ProgressivePlannerAttemptState:
    """Mutable state private to one in-flight attempt."""

    prompt_metrics: dict[str, Any] = field(default_factory=dict)
    compile_receipt: Optional[ProgressivePlanCompileReceipt] = None
    outline: Optional[ProgressivePlanOutline] = None
    foundation: Optional[ProgressiveFoundationMaterialization] = None
    materializations: list[ProgressiveStepMaterialization] = field(default_factory=list)
    compile_failure_attempts: list[ProgressiveCompileReplayAttempt] = field(
        default_factory=list
    )
    skeleton: Optional[ProgressivePlanSkeleton] = None
    resume_validated: bool = False
    dropped_plan_keys: dict[str, list[str]] = field(
        default_factory=lambda: {"top_level": [], "steps": []}
    )

    def freeze(self) -> ProgressivePlannerRunFacts:
        return ProgressivePlannerRunFacts(
            prompt_metrics=dict(self.prompt_metrics),
            compile_receipt=self.compile_receipt,
            outline=self.outline,
            foundation=self.foundation,
            materializations=tuple(self.materializations),
            compile_failure_attempts=tuple(self.compile_failure_attempts),
            skeleton=self.skeleton,
            resume_validated=self.resume_validated,
            dropped_plan_keys={
                str(key): tuple(str(value) for value in values)
                for key, values in self.dropped_plan_keys.items()
            },
        )


_FAILURE_FACTS_ATTRIBUTE = "_easyicu_progressive_planner_run_facts"


def bind_progressive_planner_failure_facts(
    error: BaseException,
    facts: ProgressivePlannerRunFacts,
) -> None:
    """Bind same-attempt facts without replacing the original exception type."""

    setattr(error, _FAILURE_FACTS_ATTRIBUTE, facts)


def progressive_planner_failure_facts(error: BaseException) -> ProgressivePlannerRunFacts:
    """Return the immutable facts bound to a failed Planner attempt."""

    facts = getattr(error, _FAILURE_FACTS_ATTRIBUTE, None)
    if not isinstance(facts, ProgressivePlannerRunFacts):
        raise RuntimeError(
            "Progressive Planner failure did not carry atomic attempt facts"
        ) from error
    return facts


__all__ = [
    "ProgressivePlannerAttemptResult",
    "ProgressivePlannerAttemptState",
    "ProgressivePlannerRunFacts",
    "bind_progressive_planner_failure_facts",
    "progressive_planner_failure_facts",
]
