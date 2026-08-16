"""Owner contract for runtime revisions of only the unexecuted plan suffix.

This module owns one dependency-light boundary: an executed plan prefix is
immutable, and a runtime revision must begin at the first unexecuted step.  It
does not read checkpoints, call a Provider, interpret observations, or repeat
scientific action/literature/clinical gates.  Those remain with their existing
runtime and planning owners after this structural merge.
"""

from __future__ import annotations

import re
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..schema import AnalysisPlan, AnalysisStep


class RuntimePlanSuffixError(ValueError):
    """Stable structural finding from the runtime-suffix owner."""

    owner = "easyicu.planning.runtime_suffix_v1"

    def __init__(self, reason_code: str, message: str) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,79}", reason_code):
            raise ValueError(f"invalid runtime suffix reason code {reason_code!r}")
        self.reason_code = reason_code
        self.code = reason_code
        self.details = {
            "owner": self.owner,
            "reason_code": reason_code,
            "message": str(message),
        }
        super().__init__(f"{reason_code}: {message}")


class RuntimePlanSuffixRevision(BaseModel):
    """Provider response for only the next unexecuted materialized step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.runtime_plan_suffix_revision/1"] = (
        "easyicu.runtime_plan_suffix_revision/1"
    )
    replace_from_step_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    replacement_step: AnalysisStep
    rationale: str = Field(min_length=8, max_length=1200)

    @model_validator(mode="after")
    def _closed_coordinate(self) -> "RuntimePlanSuffixRevision":
        if self.replacement_step.step_id != self.replace_from_step_id:
            raise ValueError("replacement_step must match replace_from_step_id")
        return self


def runtime_locked_prefix_count(
    *,
    current_plan: AnalysisPlan,
    completed_step_ids: Sequence[str],
) -> int:
    """Return the contiguous completed prefix or fail closed on a hole."""

    completed = {str(value) for value in completed_step_ids if str(value)}
    locked_count = 0
    found_unexecuted = False
    for step in current_plan.steps:
        if step.step_id in completed:
            if found_unexecuted:
                raise RuntimePlanSuffixError(
                    "runtime_completed_prefix_noncontiguous",
                    "a completed step appears after an unexecuted plan coordinate",
                )
            locked_count += 1
        else:
            found_unexecuted = True
    return locked_count


def merge_runtime_plan_suffix(
    *,
    current_plan: AnalysisPlan,
    completed_step_ids: Sequence[str],
    revision: RuntimePlanSuffixRevision,
) -> AnalysisPlan:
    """Merge one suffix while preserving the executed prefix and root scope."""

    locked_count = runtime_locked_prefix_count(
        current_plan=current_plan,
        completed_step_ids=completed_step_ids,
    )
    if locked_count >= len(current_plan.steps):
        raise RuntimePlanSuffixError(
            "runtime_suffix_already_complete",
            "the current plan has no unexecuted suffix to revise",
        )
    expected_step_id = current_plan.steps[locked_count].step_id
    if revision.replace_from_step_id != expected_step_id:
        raise RuntimePlanSuffixError(
            "runtime_suffix_coordinate_mismatch",
            f"revision must begin at {expected_step_id!r}, not "
            f"{revision.replace_from_step_id!r}",
        )
    locked_ids = {step.step_id for step in current_plan.steps[:locked_count]}
    if revision.replacement_step.step_id in locked_ids:
        raise RuntimePlanSuffixError(
            "runtime_suffix_repeats_locked_step",
            "revision repeated an executed step id",
        )
    payload = current_plan.model_dump(mode="json")
    payload["steps"] = [
        *[
            step.model_dump(mode="json")
            for step in current_plan.steps[:locked_count]
        ],
        revision.replacement_step.model_dump(mode="json"),
        *[
            step.model_dump(mode="json")
            for step in current_plan.steps[locked_count + 1 :]
        ],
    ]
    payload["revision"] = current_plan.revision + 1
    try:
        return AnalysisPlan.model_validate(payload)
    except ValidationError as exc:
        raise RuntimePlanSuffixError(
            "runtime_suffix_merged_plan_invalid",
            str(exc),
        ) from exc


__all__ = [
    "RuntimePlanSuffixError",
    "RuntimePlanSuffixRevision",
    "merge_runtime_plan_suffix",
    "runtime_locked_prefix_count",
]
