"""The existing primary Cox declaration; no alternate estimator defaults."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...contracts.survival import (
    SURVIVAL_ANALYSIS_RECEIPT_PRODUCT, SURVIVAL_PH_DIAGNOSTIC_PRODUCT,
    survival_supplement_products,
)
from ...contracts.survival_execution import survival_execution_verdict
from ...schema import AnalysisStep, FamilyPrimaryResultRequirement

SKILL_ID = "survival-time-to-event"


class SurvivalReferenceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1)
    identity_column: str = Field(min_length=1)
    primary: FamilyPrimaryResultRequirement

    def host_step(self) -> AnalysisStep:
        return AnalysisStep(
            step_id="reference_survival", intent=self.title,
            method="cox_proportional_hazards", planned_analysis_role="primary",
            inputs=[str(self.primary.input_product)],
            expected_outputs=[self.primary.expected_result_product,
                              SURVIVAL_PH_DIAGNOSTIC_PRODUCT, SURVIVAL_ANALYSIS_RECEIPT_PRODUCT,
                              *sorted(survival_supplement_products(self.primary))],
            family_primary_result_requirement=self.primary,
        )

    @model_validator(mode="after")
    def _host_supported(self):
        step = self.host_step()
        verdict = survival_execution_verdict(
            requirement=self.primary, planned_analysis_role=step.planned_analysis_role,
            inputs=step.inputs, expected_outputs=step.expected_outputs,
        )
        if not verdict.claimed:
            raise ValueError(verdict.reason)
        return self

    def required_columns(self) -> list[str]:
        return [str(self.primary.time_column), str(self.primary.event_column),
                self.primary.exposure_source, *(self.primary.covariates or [])]
