"""Bind optional caller-reviewed scientific authorities for one pipeline run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..authority.current_case_scientific_runtime import (
    CurrentCaseScientificRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from ..schema import AnalysisPlan
from ..trajectory.scientific_runtime_authority import (
    TrajectoryScientificRuntimeAuthority,
    load_trajectory_scientific_runtime_authority,
)


@dataclass(frozen=True)
class ScientificRuntimeAuthorities:
    """Immutable pair compiled once and shared with planners and executors."""

    trajectory: TrajectoryScientificRuntimeAuthority | None
    current_case: CurrentCaseScientificRuntimeAuthority | None

    @classmethod
    def load(
        cls,
        *,
        trajectory: Mapping[str, Any] | None,
        current_case: Mapping[str, Any] | None,
    ) -> "ScientificRuntimeAuthorities":
        return cls(
            trajectory=(
                load_trajectory_scientific_runtime_authority(trajectory)
                if trajectory is not None
                else None
            ),
            current_case=(
                load_current_case_scientific_runtime_authority(current_case)
                if current_case is not None
                else None
            ),
        )

    def validate_plan(self, plan: AnalysisPlan) -> None:
        """Preserve the precise authority-owner error for any plan drift."""

        if self.trajectory is not None:
            self.trajectory.validate_plan(plan)
        if self.current_case is not None:
            self.current_case.validate_plan(plan)


__all__ = ["ScientificRuntimeAuthorities"]
