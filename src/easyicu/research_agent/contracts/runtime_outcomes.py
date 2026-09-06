"""Public endpoint projection issued by a digest-bound execution owner.

This is a planning/review contract, not a signature or execution permission.
The execution authority must still validate its exact outcome and owner ref.
"""

from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeOutcomeContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    owner_ref: str = Field(pattern=r"^scientific_runtime_contract:[0-9a-f]{64}$")
    outcomes: Tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_outcomes(self):
        if len(set(self.outcomes)) != len(self.outcomes) or any(
            not name or name != name.strip() for name in self.outcomes
        ):
            raise ValueError("runtime outcomes must be unique, nonblank column keys")
        return self
