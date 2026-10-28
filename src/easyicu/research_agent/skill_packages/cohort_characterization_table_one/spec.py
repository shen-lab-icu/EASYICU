"""Reviewed Table 1 choices, using the host's existing declaration verbatim."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...schema import AnalysisStep, TableOneSpec
from ...execution.runners.table_one_executor import table_one_executor_owns_step

SKILL_ID = "cohort-characterization-table-one"


class TableOneReferenceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1)
    identity_column: str = Field(min_length=1)
    table_one: TableOneSpec

    def host_step(self) -> AnalysisStep:
        return AnalysisStep(
            step_id="reference_table_one", intent=self.title,
            method="grouped baseline characteristics",
            inputs=["artifact:analysis_cohort", *self.required_columns()],
            expected_outputs=["table:table_one"],
            table_one_spec=self.table_one,
        )

    @model_validator(mode="after")
    def _host_supported(self):
        if not table_one_executor_owns_step(self.host_step()):
            raise ValueError("Table 1 declaration is not supported by the host executor")
        return self

    def required_columns(self) -> list[str]:
        return [self.table_one.group_by, *[item.name for item in self.table_one.variables]]
