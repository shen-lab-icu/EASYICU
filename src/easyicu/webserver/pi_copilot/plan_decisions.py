"""Compile host-rendered plan choices into typed StudyContext updates.

The browser must never turn a choice button into a synthetic paragraph for the
LLM to reinterpret.  This owner accepts only known review codes and option ids,
derives executable coordinates from the reviewed plan, and returns the exact
StudyContext patch that the host may persist after a human click.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

from easyicu.webserver.study_scientific_configuration import ScientificConfiguration


class PlanDecisionError(ValueError):
    """A rendered plan choice cannot be compiled for the bound plan."""

    def __init__(
        self, code: str, message: str, *, details: Mapping[str, Any] | None = None
    ):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True)
class CompiledPlanDecision:
    patch: Dict[str, Any]
    display_label_en: str
    display_label_zh: str
    next_action: str


_AGGREGATION_SUFFIXES = ("_first", "_last", "_min", "_max", "_mean", "_sum")
_DISPLAY_LABELS_ZH = {
    "death": "院内死亡",
    "lact": "最高乳酸水平",
}
_DISPLAY_LABELS_EN = {
    "death": "in-hospital mortality",
    "lact": "maximum lactate level",
}


def _source_concept(materialized: Any, *, field: str) -> str:
    value = str(materialized or "").strip()
    if not value:
        raise PlanDecisionError(
            "plan_decision_coordinate_missing",
            f"The reviewed plan does not declare {field}.",
            details={"field": field},
        )
    for suffix in _AGGREGATION_SUFFIXES:
        if value.endswith(suffix) and len(value) > len(suffix):
            return value[: -len(suffix)]
    return value


def _selected_design(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    selection = plan.get("design_selection")
    candidates = selection.get("candidates") if isinstance(selection, Mapping) else None
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise PlanDecisionError(
            "plan_decision_design_missing",
            "The reviewed plan has no selected design.",
        )
    selected = [
        item
        for item in candidates
        if isinstance(item, Mapping) and item.get("disposition") == "selected"
    ]
    if len(selected) != 1:
        raise PlanDecisionError(
            "plan_decision_design_ambiguous",
            "The reviewed plan must declare exactly one selected design.",
        )
    return selected[0]


def _primary_requirement(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    matches: list[Mapping[str, Any]] = []
    steps = plan.get("steps")
    if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)):
        for step in steps:
            requirements = (
                step.get("model_requirements") if isinstance(step, Mapping) else None
            )
            if not isinstance(requirements, Sequence) or isinstance(
                requirements, (str, bytes)
            ):
                continue
            matches.extend(
                item
                for item in requirements
                if isinstance(item, Mapping) and item.get("analysis_role") == "primary"
            )
    if len(matches) != 1:
        raise PlanDecisionError(
            "plan_decision_primary_model_ambiguous",
            "The reviewed plan must declare exactly one primary model requirement.",
        )
    return matches[0]


def proposed_adjustment_set(plan: Mapping[str, Any]) -> list[str]:
    """Project the exact, bounded covariate roster from the primary model."""

    requirement = _primary_requirement(plan)
    raw = requirement.get("covariates")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    result: list[str] = []
    for value in raw:
        concept = str(value or "").strip()
        if concept and concept not in result:
            result.append(concept)
    return result


def decision_is_resolved(study: Mapping[str, Any], decision_code: str) -> bool:
    """Return whether the StudyContext already carries a typed human choice."""

    return ScientificConfiguration.inspect(study).decision_is_resolved(decision_code)


def pending_authorization_questions(
    study: Mapping[str, Any], questions: Any
) -> list[Dict[str, Any]]:
    """Keep unresolved review questions in their original evidence order."""

    if not isinstance(questions, Sequence) or isinstance(questions, (str, bytes)):
        return []
    result: list[Dict[str, Any]] = []
    for raw in questions:
        if not isinstance(raw, Mapping):
            continue
        code = str(raw.get("code") or "").strip()
        if not code or decision_is_resolved(study, code):
            continue
        result.append(dict(raw))
    return result


def _timing_coordinates(plan: Mapping[str, Any]) -> Dict[str, str]:
    requirement = _primary_requirement(plan)
    _selected_design(plan)
    exposure_materialized = str(requirement.get("exposure_source") or "").strip()
    outcome_materialized = str(requirement.get("outcome") or "").strip()
    exposure = _source_concept(exposure_materialized, field="primary exposure")
    outcome = _source_concept(outcome_materialized, field="outcome")
    event_time = f"{outcome}_time"
    observation_duration = "los_icu"
    return {
        "exposure": exposure,
        "outcome": outcome,
        "exposure_materialized": exposure_materialized,
        "outcome_materialized": outcome_materialized,
        "event_time": event_time,
        "observation_duration": observation_duration,
    }


def compile_plan_decision(
    *,
    decision_code: str,
    option_id: str,
    study: Mapping[str, Any],
    agent_plan: Mapping[str, Any],
) -> CompiledPlanDecision:
    """Return one typed patch for a known review decision and option."""

    code = str(decision_code or "").strip()
    option = str(option_id or "").strip()
    configuration = ScientificConfiguration.inspect(study)
    if code == "REQUIRED_SENSITIVITY_IS_PROTOCOL_ONLY":
        if option != "keep_executable_sensitivities":
            raise PlanDecisionError(
                "plan_decision_option_unknown",
                "The selected option is not available for this scientific review decision.",
                details={"decision_code": code, "option_id": option},
            )
        coordinates = _timing_coordinates(agent_plan)
        landmark = {
            "spec_id": "landmark_24h",
            "axis": "timing",
            "strategy": "landmark",
            "execution_variables": [
                coordinates["event_time"],
                coordinates["observation_duration"],
            ],
            "landmark_hours": 24,
            "require_alive_at_landmark": True,
            "exclude_negative_event_times": True,
            "event_time_variable": coordinates["event_time"],
            "observation_duration_variable": coordinates["observation_duration"],
            "observation_duration_unit": "days",
        }
        return CompiledPlanDecision(
            patch={
                "sensitivity_specs": configuration.replace_sensitivity(
                    axis="timing", replacement=landmark
                ),
                "confirmations": configuration.merge_confirmations(
                    plan_required_sensitivities_executable=True,
                ),
            },
            display_label_en="Keep and execute the confirmed robustness analyses",
            display_label_zh="已保留并执行确认的稳健性分析",
            next_action="replan",
        )

    if code == "REPEATED_STAY_IDENTITY_UNAVAILABLE":
        if option != "all_icu_stays_clustered":
            raise PlanDecisionError(
                "plan_decision_option_unknown",
                "The selected option is not available for this scientific review decision.",
                details={"decision_code": code, "option_id": option},
            )
        sensitivity = {
            "spec_id": "repeated_stays_cluster_robust",
            "axis": "repeated_stays",
            "strategy": "cluster_robust",
            "execution_variables": [],
        }
        return CompiledPlanDecision(
            patch={
                "analysis_design": {
                    "analysis_unit": "icu_stay",
                    "variance_estimator": "cluster_robust",
                    "cluster_unit": "patient",
                },
                "sensitivity_specs": configuration.replace_sensitivity(
                    axis="repeated_stays", replacement=sensitivity
                ),
                "confirmations": configuration.merge_confirmations(
                    plan_repeated_stays_clustered=True,
                ),
            },
            display_label_en="Keep every ICU stay with patient-clustered uncertainty",
            display_label_zh="已选择保留每次 ICU 入住并按患者聚类",
            next_action="continue_review",
        )

    if code == "ADJUSTMENT_SET_NOT_USER_CONFIRMED":
        if option != "accept_proposed_adjustment":
            raise PlanDecisionError(
                "plan_decision_option_unknown",
                "The selected option is not available for this scientific review decision.",
                details={"decision_code": code, "option_id": option},
            )
        covariates = proposed_adjustment_set(agent_plan)
        if not covariates:
            raise PlanDecisionError(
                "plan_decision_adjustment_set_missing",
                "The reviewed plan does not declare a proposed adjustment set.",
            )
        rationales = {
            covariate: (
                "入 ICU 时已确定的基线人口学因素，可能同时关联乳酸水平与院内死亡风险。"
                if covariate in {"age", "sex"}
                else "候选计划提出的基线混杂因素；仅在研究时间零点前可用时纳入调整。"
            )
            for covariate in covariates
        }
        execution = dict(study.get("execution_concepts") or {})
        execution["covariates"] = covariates
        return CompiledPlanDecision(
            patch={
                "covariates": covariates,
                "covariate_selection": "exact",
                "covariate_rationales": rationales,
                "covariate_temporal_roles": {
                    covariate: "baseline_static" for covariate in covariates
                },
                "covariate_operationalizations": {
                    covariate: covariate for covariate in covariates
                },
                "execution_concepts": execution,
                "confirmations": configuration.merge_confirmations(
                    plan_adjustment_set_confirmed=True,
                ),
            },
            display_label_en="Use the proposed adjustment set",
            display_label_zh="已采用计划建议的调整变量",
            next_action="replan",
        )

    if code != "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED":
        raise PlanDecisionError(
            "plan_decision_code_unknown",
            "The scientific review decision is not supported by this host action.",
            details={"decision_code": code},
        )
    coordinates = _timing_coordinates(agent_plan)
    exposure = coordinates["exposure"]
    outcome = coordinates["outcome"]
    exposure_zh = _DISPLAY_LABELS_ZH.get(exposure, exposure)
    outcome_zh = _DISPLAY_LABELS_ZH.get(outcome, outcome)
    exposure_en = _DISPLAY_LABELS_EN.get(exposure, exposure)
    outcome_en = _DISPLAY_LABELS_EN.get(outcome, outcome)
    execution = {
        "outcome": outcome,
        "primary_exposure": exposure,
        "primary_exposure_aggregation": "max",
    }

    if option == "landmark_24h":
        cohort = dict(study.get("cohort") or {})
        cohort.update(
            {
                "label": "入 ICU 后 24 小时仍存活的 ICU stay",
                "review_scope": (
                    "纳入入 ICU 后 24 小时仍存活的 ICU stay；"
                    "以第 24 小时作为 landmark 起点。"
                ),
            }
        )
        sensitivity = {
            "spec_id": "landmark_24h",
            "axis": "timing",
            "strategy": "landmark",
            "execution_variables": [
                coordinates["event_time"],
                coordinates["observation_duration"],
            ],
            "landmark_hours": 24,
            "require_alive_at_landmark": True,
            "exclude_negative_event_times": True,
            "event_time_variable": coordinates["event_time"],
            "observation_duration_variable": coordinates["observation_duration"],
            "observation_duration_unit": "days",
        }
        return CompiledPlanDecision(
            patch={
                "question": (
                    f"在入 ICU 后 24 小时仍存活的 ICU stay 中，评估入 ICU 后 "
                    f"0–24 小时{exposure_zh}与自第 24 小时起至出院的{outcome_zh}之间的关系。"
                ),
                "purpose": (
                    f"采用预先设定的 24 小时 landmark 设计，评估 0–24 小时"
                    f"{exposure_zh}与 landmark 后{outcome_zh}之间的调整后观察性关联。"
                ),
                "cohort": cohort,
                "outcome": f"{outcome_zh}（第 24 小时起至出院）",
                "primary_exposure": f"入 ICU 后 0–24 小时{exposure_zh}",
                "execution_concepts": execution,
                "analysis_goal": "24 小时 landmark 后的调整关联分析",
                "export_format": "parquet",
                "sensitivity_specs": configuration.replace_sensitivity(
                    axis="timing", replacement=sensitivity
                ),
                "confirmations": configuration.merge_confirmations(
                    feature_time_window=True,
                    extraction_completed=True,
                    export_format=True,
                    plan_timing_landmark_24h=True,
                ),
            },
            display_label_en="Use the recommended 24-hour landmark",
            display_label_zh="已采用推荐的 24 小时 landmark",
            next_action="replan",
        )

    if option == "descriptive_only":
        return CompiledPlanDecision(
            patch={
                "outcome": outcome_zh,
                "primary_exposure": f"入 ICU 后 0–24 小时{exposure_zh}",
                "execution_concepts": execution,
                "analysis_goal": "描述暴露与结局分布，不估计时间对齐后的关联",
                "export_format": "parquet",
                "sensitivity_specs": configuration.replace_sensitivity(
                    axis="timing", replacement=None
                ),
                "confirmations": configuration.merge_confirmations(
                    feature_time_window=True,
                    extraction_completed=True,
                    export_format=True,
                    plan_timing_descriptive_only=True,
                ),
            },
            display_label_en=f"Keep {exposure_en} and {outcome_en} descriptive",
            display_label_zh="已选择仅作描述性分析",
            next_action="replan",
        )

    if option == "time_varying_reextract":
        sensitivity = {
            "spec_id": "time_varying_exposure",
            "axis": "timing",
            "strategy": "time_varying",
            "execution_variables": [exposure],
        }
        return CompiledPlanDecision(
            patch={
                "outcome": outcome_zh,
                "primary_exposure": f"带时间戳的{exposure_zh}",
                "execution_concepts": {
                    "outcome": outcome,
                    "primary_exposure": exposure,
                },
                "analysis_goal": f"{exposure_zh}的时变暴露关联分析",
                "sensitivity_specs": configuration.replace_sensitivity(
                    axis="timing", replacement=sensitivity
                ),
                "confirmations": configuration.merge_confirmations(
                    plan_timing_time_varying=True,
                    extraction_completed=False,
                ),
            },
            display_label_en=f"Re-extract timestamped {exposure_en}",
            display_label_zh="已选择重新提取时变暴露数据",
            next_action="reextract",
        )

    raise PlanDecisionError(
        "plan_decision_option_unknown",
        "The selected option is not available for this scientific review decision.",
        details={"decision_code": code, "option_id": option},
    )


__all__ = [
    "CompiledPlanDecision",
    "PlanDecisionError",
    "compile_plan_decision",
    "decision_is_resolved",
    "pending_authorization_questions",
    "proposed_adjustment_set",
]
