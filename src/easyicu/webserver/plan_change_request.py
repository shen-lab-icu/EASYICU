"""A user amendment or Host runtime revision, never approval or a finding."""

from __future__ import annotations

import json
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from easyicu.research_agent.planning.baseline_requirements import AcceptedBaselineRequirements
from easyicu.research_agent.planning.population_requirements import PlanPopulationRequirements


class PlanChangeRequirements(BaseModel):
    """Host-verified planning constraints; never a data or execution grant."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline: AcceptedBaselineRequirements | None = None
    population: PlanPopulationRequirements | None = None
    planning_concepts: tuple[str, ...] = ()
    operationalized_columns: tuple[str, ...] = ()


class ReferencedPlan(BaseModel):
    """Host-read, privacy-checked planning content; no execution grant."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str = Field(min_length=1, max_length=160)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan: dict[str, Any]


def reference_plan_content(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Keep scientific choices and exact rosters without result artifacts.

    This consumes the artifact owner's privacy-checked payload. Size is checked
    by PlanChangeRequest; silent list truncation would hide the requested rows.
    """
    projected = {key: plan[key] for key in (
        "research_question", "analysis_type", "cohort", "endpoint", "robustness_specs",
        "subgroup_analysis_spec", "display_labels",
    ) if key in plan}
    selection = plan.get("design_selection")
    if isinstance(selection, Mapping):
        projected["design_selection"] = [
            {key: candidate[key] for key in (
                "design_id", "disposition", "estimand", "time_zero", "observation_window",
                "primary_method", "required_variables",
            ) if key in candidate}
            for candidate in selection.get("candidates", ()) if isinstance(candidate, Mapping)
        ]
    projected["steps"] = [
        {key: step[key] for key in (
            "step_id", "planned_analysis_role", "intent", "method", "inputs", "expected_outputs",
            "table_one_spec", "model_requirements", "cohort_definition_spec", "functional_form_spec",
            "population_scope", "population_scope_change_reason", "scientific_action_id", "literature_citation_keys",
        ) if key in step}
        for step in plan.get("steps", ()) if isinstance(step, Mapping)
    ]
    return projected



class PlanChangeRequest(BaseModel):
    """Path-free request bound by the host to the plan being discussed."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal["easyicu.plan-change-request/1"] = (
        "easyicu.plan-change-request/1"
    )
    source_run_id: str = Field(min_length=1, max_length=160)
    user_message: str = Field(min_length=1, max_length=12_000)
    reference_plans: tuple[ReferencedPlan, ...] = Field(default=(), max_length=4)
    source_scientific_configuration_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    target_scientific_configuration_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_requirements: PlanChangeRequirements | None = None

    @model_validator(mode="after")
    def _bounded_references(self) -> "PlanChangeRequest":
        payload = [reference.model_dump(mode="json") for reference in self.reference_plans]
        if len(json.dumps(payload, ensure_ascii=False).encode()) > 64_000:
            raise ValueError("referenced plan context exceeds its bounded transport")
        if len({reference.run_id for reference in self.reference_plans}) != len(self.reference_plans):
            raise ValueError("referenced plans must be unique")
        if (self.source_scientific_configuration_sha256 is None) != (self.target_scientific_configuration_sha256 is None):
            raise ValueError("revision source and target configurations must be paired")
        if self.source_requirements is not None and self.source_scientific_configuration_sha256 is None:
            raise ValueError("verified requirements need exact source and target configurations")
        return self

    @model_serializer(mode="wrap")
    def _legacy_serialization(self, handler):
        payload = handler(self)
        if not self.reference_plans:
            payload.pop("reference_plans", None)
        for key in ("source_scientific_configuration_sha256", "target_scientific_configuration_sha256", "source_requirements"):
            if payload.get(key) is None:
                payload.pop(key, None)
        return payload

    def reference_concepts(self, catalog_ids: set[str]) -> tuple[str, ...]:
        """Keep historical coordinates available in a zero-row planning menu.

        These are candidates for revision, not required analysis variables or
        permission to read patient rows. The new plan still requires review.
        """
        coordinates = set()
        if self.source_requirements is not None:
            # These source concepts were read from sealed descriptors, never
            # inferred by stripping suffixes from old analysis column names.
            if self.source_scientific_configuration_sha256 != self.target_scientific_configuration_sha256:
                # Only the Host-verified runtime projection carries these
                # requirements across digests. Explicit new scientific scope
                # has an empty roster; old reference inputs cannot widen it.
                return tuple(sorted(set(self.source_requirements.planning_concepts) & catalog_ids))
            coordinates.update(self.source_requirements.planning_concepts)
        for reference in self.reference_plans:
            for step in reference.plan.get("steps") or ():
                if isinstance(step, Mapping):
                    coordinates.update(value for value in step.get("inputs", ()) if isinstance(value, str))
        return tuple(sorted(coordinates & catalog_ids))

    def baseline_requirements(self) -> AcceptedBaselineRequirements | None:
        return self.source_requirements.baseline if self.source_requirements is not None else None

    def population_requirements(self):
        """Bind the discussed current plan, not a guessed historical winner."""
        from easyicu.research_agent.planning.population_requirements import candidate_population_requirements

        if self.source_requirements is not None:
            return self.source_requirements.population
        reference = next((r for r in self.reference_plans if r.run_id == self.source_run_id), None)
        if reference is None:
            if self.reference_plans:
                raise ValueError("current source plan is absent from revision references")
            return None
        return candidate_population_requirements(reference.plan, reference.artifact_sha256, source_digest_kind="artifact_sha256")

    def planner_context(self) -> str:
        """Keep requested amendments distinct from reviewed plan authority."""

        return (
            "Host-bound request to revise the complete candidate plan. "
            "Address the requested amendments or explain the exact conflict. "
            "This request is not a scientific fact, approved plan, clinical "
            "sign-off, or permission to execute analysis. Preserve the research "
            "question, data source, required outcomes, and host authority gates; "
            "propose changes for a fresh complete-plan review. "
            "reference_plans contain the exact saved candidate content; "
            "compare their declared variables, methods and outputs instead of "
            "reconstructing them from run names. Historical plans are context, "
            "not current approval or evidence of scientific correctness.\n"
            + ("source_requirements preserves the accepted baseline and population "
            "within the same scientific configuration or an exact Host-recorded "
            "runtime projection of that candidate; an explicit "
            "new configuration supersedes the old scope. Missing variables in "
            "the latest message do not withdraw accepted requirements.\n"
            if self.source_requirements is not None else "")
            + self.model_dump_json()
        )


__all__ = ["PlanChangeRequest", "PlanChangeRequirements", "ReferencedPlan", "reference_plan_content"]
