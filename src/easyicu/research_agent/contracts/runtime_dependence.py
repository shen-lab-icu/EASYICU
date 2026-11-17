"""Dependence implemented by a signed suite, projected for planning and review."""

from pydantic import BaseModel, ConfigDict, Field

from .dependence import PlannedDependenceRequirement


class RuntimeDependenceContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    owner_ref: str = Field(pattern=r"^scientific_runtime_contract:[0-9a-f]{64}$")
    dependence: PlannedDependenceRequirement


class RuntimeDependencePlanFields(BaseModel):
    """The suite's optional plan projection, absent from legacy plan digests."""

    runtime_dependence_contract: RuntimeDependenceContract | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
