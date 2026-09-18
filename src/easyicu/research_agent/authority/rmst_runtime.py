"""Digest-bound, analysis-only execution of a prespecified RMST contrast.

This owner projects one reviewed sensitivity specification into a small
execution plan: a two-group restricted-mean-survival difference at a fixed
horizon.  It never borrows a landmark population, relabels a hazard ratio, or
claims publication readiness.  Cohort construction stays with the bound typed
cohort; the reviewed KM kernel owns the estimate.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..canonical_json import canonical_sha256
from ..contracts.cohort_product_keys import sole_typed_cohort_input
from ..contracts.rmst import RMSTSpec
from ..contracts.runtime_outcomes import RuntimeOutcomeContract
from ..schema import AnalysisPlan, AnalysisStep


RMST_PLAN_METHOD = "rmst"
RMST_PLAN_OUTPUTS = ("table:rmst_summary", "log:rmst_runtime_receipt")
RMST_SCIENTIFIC_ACTION_ID = "time_to_event.rmst"


class RmstRuntimeAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["easyicu.rmst_runtime_authority/1"] = (
        "easyicu.rmst_runtime_authority/1"
    )
    authority_kind: Literal["restricted_mean_survival_difference"] = (
        "restricted_mean_survival_difference"
    )
    protocol_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    specification: RMSTSpec
    sensitivity_spec_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    identity_column: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    primary_cohort_selection_mode: Literal["all_input_rows"] = "all_input_rows"
    development_execution_only_allowed: Literal[True] = True
    plan_method: Literal["rmst"] = "rmst"
    plan_intent: str = Field(min_length=12, max_length=1200)
    plan_outputs: tuple[str, ...] = RMST_PLAN_OUTPUTS

    @model_validator(mode="after")
    def _closed(self) -> "RmstRuntimeAuthority":
        if self.schema_version != "easyicu.rmst_runtime_authority/1":
            raise ValueError("RMST authority schema changed")
        if self.authority_kind != "restricted_mean_survival_difference":
            raise ValueError("RMST authority kind changed")
        if self.plan_method != RMST_PLAN_METHOD:
            raise ValueError("RMST plan method changed")
        if self.plan_outputs != RMST_PLAN_OUTPUTS:
            raise ValueError("RMST output contract changed")
        if self.primary_cohort_selection_mode != "all_input_rows":
            raise ValueError("RMST executor only binds an all-input-row cohort")
        if not self.development_execution_only_allowed:
            raise ValueError("RMST execution is development analysis only")
        if self.specification.event_column == self.specification.time_column:
            raise ValueError("RMST time and event columns must differ")
        if self.identity_column in {
            self.specification.time_column,
            self.specification.event_column,
            self.specification.group_column,
        }:
            raise ValueError("RMST identity column must differ from analysis columns")
        unsigned = self.model_dump(
            mode="json", exclude={"execution_contract_sha256"}
        )
        if canonical_sha256(unsigned) != self.execution_contract_sha256:
            raise ValueError("RMST authority digest mismatch")
        return self

    @property
    def plan_rule_ref(self) -> str:
        return f"scientific_runtime_contract:{self.execution_contract_sha256}"

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (
            self.identity_column,
            self.specification.time_column,
            self.specification.event_column,
            self.specification.group_column,
        )

    def _candidate(self, plan: AnalysisPlan) -> AnalysisStep:
        owners = [
            step
            for step in plan.steps
            if tuple(step.sensitivity_spec_ids) == (self.sensitivity_spec_id,)
        ]
        if len(owners) != 1:
            raise ValueError(
                "RMST plan must contain exactly one step for the reviewed "
                "sensitivity specification"
            )
        return owners[0]

    def governed_step(self, plan: AnalysisPlan) -> AnalysisStep:
        """Return the one step this authority authorizes, or fail closed."""

        step = self._candidate(plan)
        issues: list[str] = []
        if step.planned_analysis_role != "sensitivity":
            issues.append("planned_analysis_role")
        if step.method != self.plan_method:
            issues.append("method")
        if step.scientific_action_id != RMST_SCIENTIFIC_ACTION_ID:
            issues.append("scientific_action_id")
        if step.intent != self.plan_intent:
            issues.append("intent")
        if tuple(step.expected_outputs) != self.plan_outputs:
            issues.append("expected_outputs")
        cohort_input = sole_typed_cohort_input(step)
        if not cohort_input:
            issues.append("typed_cohort_input")
        elif step.inputs != [cohort_input, *self.required_columns]:
            issues.append("required_inputs")
        if self.plan_rule_ref not in set(step.icu_rule_refs):
            issues.append("execution_contract_sha256")
        expected_outcome = RuntimeOutcomeContract(
            owner_ref=self.plan_rule_ref,
            outcomes=(self.specification.event_column,),
        )
        if step.runtime_outcome_contract != expected_outcome:
            issues.append("runtime_outcome_contract")
        if (
            step.model_requirements
            or step.family_primary_result_requirement is not None
            or step.table_one_spec is not None
        ):
            issues.append("nested_model_contract")
        if issues:
            raise ValueError(
                "RMST plan drifted from its bound specification: "
                + ", ".join(issues)
            )
        return step

    def bind_plan(self, plan: AnalysisPlan) -> AnalysisPlan:
        """Compile the sealed executor coordinates into the declared sensitivity."""

        candidate = self._candidate(plan)
        cohort_input = sole_typed_cohort_input(candidate)
        if not cohort_input:
            raise ValueError("RMST sensitivity requires exactly one typed cohort input")
        bound = candidate.model_copy(
            update={
                "planned_analysis_role": "sensitivity",
                "method": self.plan_method,
                "scientific_action_id": RMST_SCIENTIFIC_ACTION_ID,
                "intent": self.plan_intent,
                "inputs": [cohort_input, *self.required_columns],
                "expected_outputs": list(self.plan_outputs),
                "icu_rule_refs": list(
                    dict.fromkeys([*candidate.icu_rule_refs, self.plan_rule_ref])
                ),
                "runtime_outcome_contract": RuntimeOutcomeContract(
                    owner_ref=self.plan_rule_ref,
                    outcomes=(self.specification.event_column,),
                ),
                "model_requirements": [],
                "family_primary_result_requirement": None,
                "table_one_spec": None,
            }
        )
        steps = [bound if step is candidate else step for step in plan.steps]
        compiled = plan.model_copy(update={"steps": steps})
        self.governed_step(compiled)
        return compiled

    def validate_plan(self, plan: AnalysisPlan) -> None:
        self.governed_step(plan)


__all__ = [
    "RMST_PLAN_METHOD",
    "RMST_PLAN_OUTPUTS",
    "RMST_SCIENTIFIC_ACTION_ID",
    "RmstRuntimeAuthority",
]
