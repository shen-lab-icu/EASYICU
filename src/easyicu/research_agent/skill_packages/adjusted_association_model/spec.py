"""One explicit model declaration, checked against the existing host owner."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...contracts.association_execution import association_execution_verdict
from ...schema import AnalysisStep, PlannedModelRequirement

SKILL_ID = "adjusted-association-model"


class AdjustedAssociationReferenceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1)
    identity_column: str = Field(min_length=1)
    model: PlannedModelRequirement

    def host_step(self) -> AnalysisStep:
        return AnalysisStep(
            step_id="reference_association", intent=self.title,
            method="adjusted_association_models",
            inputs=["artifact:analysis_cohort"],
            expected_outputs=["table:adjusted_association_estimates"],
            model_requirements=[self.model], planned_analysis_role=self.model.analysis_role,
        )

    @model_validator(mode="after")
    def _host_supported(self):
        verdict = association_execution_verdict(self.host_step())
        if not verdict.claimed:
            raise ValueError(verdict.reason)
        return self

    def required_columns(self) -> list[str]:
        columns = [self.model.outcome, self.model.exposure_source, *(self.model.covariates or [])]
        if self.model.dependence is not None:
            columns.append(self.model.dependence.group_source)
        return list(dict.fromkeys(columns))
