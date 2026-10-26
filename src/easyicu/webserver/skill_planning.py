"""Bounded HTTP assumptions for the existing scalar planning owners.

These are transport limits, not scientific applicability criteria. Domain
checks (null effects, power versus alpha and compatible R²) belong to sk4.
This entry accepts user assumptions only: no aggregate receipt is verified here.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from easyicu.research_agent import skill_packages

Probability = Annotated[float, Field(ge=1e-6, le=1 - 1e-6)]
Difference = Annotated[float, Field(ge=-1_000_000, le=1_000_000)]
StandardDeviation = Annotated[float, Field(ge=1e-6, le=1_000_000)]
AllocationRatio = Annotated[float, Field(ge=.01, le=100)]
HazardRatio = Annotated[float, Field(ge=.001, le=1000)]
ParameterCount = Annotated[int, Field(ge=1, le=10_000)]


class Assumptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    basis: Literal["user_assumptions"] = "user_assumptions"


class PowerAssumptions(Assumptions):
    alpha: Probability = .05
    power: Probability = .8


class TwoProportionsRequest(PowerAssumptions):
    calculation: Literal["two_proportions"]
    p1: Probability
    p2: Probability
    allocation_ratio: AllocationRatio = 1


class TwoMeansRequest(PowerAssumptions):
    calculation: Literal["two_means"]
    difference: Difference
    standard_deviation: StandardDeviation
    allocation_ratio: AllocationRatio = 1


class SurvivalEventsRequest(PowerAssumptions):
    calculation: Literal["survival_events"]
    hazard_ratio: HazardRatio
    allocation_fraction: Probability = .5
    event_fraction: Probability | None = None


class BinaryPredictionRequest(Assumptions):
    calculation: Literal["prediction_binary"]
    parameters: ParameterCount
    cox_snell_r2: Probability
    prevalence: Probability
    shrinkage: Probability = .9
    r2_optimism: Probability = .05
    prevalence_margin: Probability = .05
    confidence: Probability = .95


PlanningRequest = Annotated[
    TwoProportionsRequest | TwoMeansRequest | SurvivalEventsRequest | BinaryPredictionRequest,
    Field(discriminator="calculation"),
]


def calculate(request: PlanningRequest) -> dict:
    """Run exactly one allowlisted host calculation and preserve its result."""
    calculators = {
        "two_proportions": skill_packages.two_proportions,
        "two_means": skill_packages.two_means,
        "survival_events": skill_packages.survival_events,
        "prediction_binary": skill_packages.prediction_binary,
    }
    return calculators[request.calculation](**request.model_dump(exclude={"calculation"}))
