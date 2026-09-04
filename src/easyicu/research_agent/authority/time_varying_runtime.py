"""Digest-bound, analysis-only execution of an explicit time-updated design.

This owner projects a complete specification into a small execution plan. It
does not relabel static odds ratios, borrow a landmark population, or claim
article/publication readiness. Source acquisition and model fitting remain
separate owners.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..canonical_json import canonical_sha256

from ..contracts.time_varying_exposure import (
    TIME_VARYING_EXPOSURE_CAPABILITY,
    TIME_VARYING_EXPOSURE_METHOD,
    TimeVaryingExposureSpecification,
)
from ..schema import AnalysisPlan, AnalysisStep


class TimeVaryingRuntimeAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["easyicu.time_varying_runtime_authority/1"]
    authority_kind: Literal["time_varying_exposure_association"]
    protocol_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    specification: TimeVaryingExposureSpecification
    sensitivity_spec_id: str
    exposure_column: str
    outcome_column: Literal["death"]
    identity_column: Literal["patient_stay_id"]
    primary_cohort_selection_mode: Literal["all_input_rows"]
    development_execution_only_allowed: Literal[True]
    plan_method: Literal["time_varying_exposure_model"]
    plan_intent: str
    plan_outputs: tuple[
        Literal[
            "table:time_varying_cox_estimates",
            "table:time_varying_input_audit",
            "log:time_varying_runtime_receipt",
        ],
        ...,
    ]

    @model_validator(mode="after")
    def _closed(self) -> "TimeVaryingRuntimeAuthority":
        if self.plan_outputs != (
            "table:time_varying_cox_estimates",
            "table:time_varying_input_audit",
            "log:time_varying_runtime_receipt",
        ):
            raise ValueError("time-varying output contract changed")
        body = self.model_dump(mode="json", exclude={"execution_contract_sha256"})
        if canonical_sha256(body) != self.execution_contract_sha256:
            raise ValueError("time-varying authority digest mismatch")
        return self

    @property
    def plan_rule_ref(self) -> str:
        return f"scientific_runtime_contract:{self.execution_contract_sha256}"

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (
            self.identity_column,
            self.exposure_column,
            self.outcome_column,
            *self.specification.baseline_columns,
        )

    def development_execution_only_plan(
        self, *, research_question: str
    ) -> AnalysisPlan:
        recommendation = [
            "以每次 ICU 入住为分析单位；保留重复入住和早期死亡，以源数据合同明确记录无法构造随访的排除原因。",
            f"仅使用 ICU 入科后 0–24 小时已发生的 {self.specification.exposure_concept} 直接测量，随时间更新最大值；24 小时后冻结已观测状态。",
            "从 ICU 入科时点随访至院内死亡或出院；使用真实医院出院时间，不能用 ICU 住院时长代替。",
            f"使用 R survival 计数过程 Cox、Efron ties 和按患者聚类的稳健标准误；调整变量固定为 {', '.join(self.specification.baseline_columns)}。",
            "首次测量前保留未测量指示变量，不使用未来值回填；基线变量缺失或分类编码不匹配时停止，不自动填补。",
            "检查时间区间、早期事件、未测量者、患者分组和模型收敛；本次仅为开发分析，比例风险检验、独立科学审阅及投稿资格尚未完成。",
        ]
        selected_design = {
            "design_id": "time_updated_running_max",
            "analysis_type": "association",
            "estimand": "Descriptive hospital-death hazard association with the time-updated observed exposure and an unmeasured-state indicator.",
            "time_zero": "ICU admission (0 hours)",
            "observation_window": "Direct exposure measurements in ICU hours 0–24; hospital follow-up until death or discharge.",
            "primary_method": self.plan_method,
            "required_variables": list(self.required_columns),
            "assumptions": [
                "Only previously observed measurements enter each risk interval.",
                "Cox proportional-hazards assumptions are not yet independently validated; informative measurement and residual confounding remain.",
            ],
            "novelty_positioning": "Development validation of an explicitly bound analysis; no novelty claim.",
            "figure_role": "Inspect aggregate input audit and estimate tables; no publication figure is authorized.",
            "supports": "Source-traceable, descriptive time-updated association and input accounting.",
            "cannot_prove": "A causal effect, absence of informative measurement, proportional hazards or publication readiness.",
            "reviewable_plan": recommendation,
            "disposition": "selected",
            "decision_reason": "The declared time-updated specification retains events from ICU admission and does not condition on surviving to a later landmark.",
        }
        rejected_design = {
            **selected_design,
            "design_id": "fixed_24_hour_landmark",
            "disposition": "rejected",
            "estimand": "Exposure association restricted to patients who remain observable and alive at ICU hour 24.",
            "time_zero": "ICU admission plus 24 hours",
            "primary_method": "landmark_analysis",
            "reviewable_plan": None,
            "decision_reason": "Conditioning on survival to hour 24 removes early events and changes the explicitly declared time-zero population.",
        }
        plan = AnalysisPlan.model_validate(
            {
                "research_question": research_question,
                "analysis_type": "association",
                "cohort": {
                    "name": "source_bound_time_varying_cohort",
                    "selection_mode": self.primary_cohort_selection_mode,
                    "inclusion": [],
                    "exclusion": [],
                },
                "endpoint": {
                    "name": self.outcome_column,
                    "kind": "binary",
                    "absence_semantics": "no_absent_rows",
                    "levels": [0, 1],
                },
                "design_selection": {"candidates": [selected_design, rejected_design]},
                "rationale": "Analysis-only development execution of the explicit source-bound specification. No publication authority is conferred.",
                "steps": [
                    {
                        "step_id": "00_host_bound_analysis_cohort",
                        "planned_analysis_role": "auxiliary",
                        "intent": "Bind the source-verified one-row-per-stay cohort, including early events and unmeasured exposure states.",
                        "method": "host_materialized_locked_cohort",
                        "inputs": [],
                        "expected_outputs": ["table:analysis_cohort"],
                    },
                    {
                        "step_id": "01_time_varying_exposure_cox",
                        "planned_analysis_role": "primary",
                        "intent": self.plan_intent,
                        "method": self.plan_method,
                        "inputs": ["table:analysis_cohort", *self.required_columns],
                        "expected_outputs": list(self.plan_outputs),
                        "scientific_capability": TIME_VARYING_EXPOSURE_CAPABILITY,
                        "sensitivity_spec_ids": [self.sensitivity_spec_id],
                        "icu_rule_refs": [self.plan_rule_ref],
                    },
                ],
            }
        )
        self.validate_plan(plan)
        return plan

    def bind_plan(self, plan: AnalysisPlan) -> AnalysisPlan:
        # This explicitly analysis-only capability has its own whole-plan
        # contract. Static-model sensitivities/OR figures are not compatible
        # downstream consumers and must not survive by renaming their inputs.
        return self.development_execution_only_plan(
            research_question=plan.research_question
        )

    def governed_step(self, plan: AnalysisPlan) -> AnalysisStep:
        primary = [
            step for step in plan.steps if step.planned_analysis_role == "primary"
        ]
        if len(primary) != 1:
            raise ValueError("time-varying authority requires exactly one primary step")
        step = primary[0]
        if (
            step.method != TIME_VARYING_EXPOSURE_METHOD
            or step.intent != self.plan_intent
            or step.scientific_capability != TIME_VARYING_EXPOSURE_CAPABILITY
            or tuple(step.expected_outputs) != self.plan_outputs
            or step.inputs != ["table:analysis_cohort", *self.required_columns]
            or step.sensitivity_spec_ids != [self.sensitivity_spec_id]
            or self.plan_rule_ref not in step.icu_rule_refs
            or step.model_requirements
            or step.family_primary_result_requirement is not None
        ):
            raise ValueError("time-varying plan drifted from its bound specification")
        return step

    def validate_plan(self, plan: AnalysisPlan) -> None:
        self.governed_step(plan)
        if (
            plan.cohort is None
            or plan.cohort.selection_mode != self.primary_cohort_selection_mode
            or plan.cohort.inclusion
            or plan.cohort.exclusion
        ):
            raise ValueError("time-varying plan changed its bound cohort selection")
        if (
            plan.endpoint is None
            or plan.endpoint.name != self.outcome_column
            or plan.endpoint.kind != "binary"
            or plan.endpoint.absence_semantics != "no_absent_rows"
            or plan.endpoint.levels != [0, 1]
        ):
            raise ValueError("time-varying plan changed its bound hospital endpoint")
        if (
            len(plan.steps) != 2
            or plan.steps[0].method != "host_materialized_locked_cohort"
            or plan.steps[0].inputs
            or plan.steps[0].expected_outputs != ["table:analysis_cohort"]
        ):
            raise ValueError(
                "time-varying analysis-only plan has undeclared additional analyses"
            )


__all__ = ["TimeVaryingRuntimeAuthority"]
