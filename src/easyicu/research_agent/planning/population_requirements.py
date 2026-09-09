"""Preserve source-bound descriptive populations across plan revisions.

These are planning constraints, never execution approval. An intentional scope
change remains possible, but must be disclosed in the new complete plan.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..schema import ResearchContext

_KEY = "plan_population_requirements"


class PopulationRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_step_id: str = Field(min_length=1)
    output_product: str = Field(pattern=r"^table:[a-zA-Z0-9_]+$")
    population_scope: Literal["analysis_cohort", "primary_model"]


class PlanPopulationRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["easyicu.plan_population_requirements/1"] = (
        "easyicu.plan_population_requirements/1"
    )
    source_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_digest_kind: Literal["canonical_plan_sha256", "artifact_sha256"] = (
        "canonical_plan_sha256"
    )
    populations: tuple[PopulationRequirement, ...] = Field(min_length=1, max_length=24)

    @model_validator(mode="after")
    def unique_products(self):
        if len({p.output_product for p in self.populations}) != len(self.populations):
            raise ValueError("population requirement products must be unique")
        return self


def candidate_population_requirements(
    plan: Mapping[str, Any],
    source_plan_sha256: str,
    *,
    source_digest_kind: Literal[
        "canonical_plan_sha256", "artifact_sha256"
    ] = "canonical_plan_sha256",
) -> PlanPopulationRequirements | None:
    populations = []
    for step in plan.get("steps") or ():
        if step.get("population_scope") is None:
            continue
        outputs = [
            p for p in step.get("expected_outputs", ()) if p.startswith("table:")
        ]
        if len(outputs) != 1:
            raise ValueError(
                "a descriptive population requires one exact table product"
            )
        populations.append(
            PopulationRequirement(
                source_step_id=step["step_id"],
                output_product=outputs[0],
                population_scope=step["population_scope"],
            )
        )
    return (
        PlanPopulationRequirements(
            source_plan_sha256=source_plan_sha256,
            source_digest_kind=source_digest_kind,
            populations=tuple(populations),
        )
        if populations
        else None
    )


def context_population_requirements(
    context: ResearchContext,
) -> PlanPopulationRequirements | None:
    payload = context.cohort.provenance.get(_KEY)
    return (
        PlanPopulationRequirements.model_validate(payload)
        if payload is not None
        else None
    )


def bind_population_requirements(
    context: ResearchContext,
    payload: Mapping[str, Any] | None,
    *,
    restoring: bool = False,
) -> ResearchContext:
    required = (
        PlanPopulationRequirements.model_validate(payload)
        if payload is not None
        else None
    )
    existing = context_population_requirements(context)
    if restoring or existing is not None:
        if existing != required:
            raise ValueError("plan_population_requirements_binding_drift")
        return context
    if required is None:
        return context
    return context.model_copy(
        update={
            "cohort": context.cohort.model_copy(
                update={
                    "provenance": {
                        **context.cohort.provenance,
                        _KEY: required.model_dump(mode="json"),
                    },
                }
            )
        }
    )


def validate_population_choice(
    context: ResearchContext,
    *,
    product: str,
    scope: str | None,
    change_reason: str | None,
) -> None:
    required = context_population_requirements(context)
    if required is None:
        return
    for population in required.populations:
        if (
            population.output_product == product
            and population.population_scope != scope
        ):
            if scope is None or not change_reason:
                raise ValueError(
                    f"Preserve {product} population_scope={population.population_scope} "
                    f"from plan {required.source_plan_sha256}; received {scope!r}. "
                    "Only an intentional scientific amendment may change this scope: "
                    "declare population_scope_change_reason for complete-plan review. "
                    "A presentation-only amendment does not change its population."
                )
