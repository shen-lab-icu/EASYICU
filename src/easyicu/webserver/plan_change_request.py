"""A current-user amendment request, not an approval or scientific finding."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PlanChangeRequest(BaseModel):
    """Path-free request bound by the host to the plan being discussed."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal["easyicu.plan-change-request/1"] = (
        "easyicu.plan-change-request/1"
    )
    source_run_id: str = Field(min_length=1, max_length=160)
    user_message: str = Field(min_length=1, max_length=12_000)

    def planner_context(self) -> str:
        """Keep requested amendments distinct from reviewed plan authority."""

        return (
            "Current user request to revise the complete candidate plan. "
            "Address the requested amendments or explain the exact conflict. "
            "This request is not a scientific fact, approved plan, clinical "
            "sign-off, or permission to execute analysis. Preserve the research "
            "question, data source, required outcomes, and host authority gates; "
            "propose changes for a fresh complete-plan review.\n"
            + self.model_dump_json()
        )


__all__ = ["PlanChangeRequest"]
