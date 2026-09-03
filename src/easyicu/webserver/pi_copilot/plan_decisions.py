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
    "sep3_sofa1": "Sepsis-3 状态",
}
_DISPLAY_LABELS_EN = {
    "death": "in-hospital mortality",
    "lact": "maximum lactate level",
    "sep3_sofa1": "Sepsis-3 status",
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
    event_time = "death_time_hours" if outcome == "death" else f"{outcome}_time"
    # ``los_icu`` ends at ICU discharge.  It cannot stand in for the hospital
    # follow-up axis of an in-hospital mortality analysis: a stay may leave the
    # ICU alive and still die before hospital discharge.  Issue the required
    # owner coordinate here; data readiness will fail closed until a source-bound
    # follow-up contract materializes it.
    observation_duration = (
        "hospital_followup_time_hours"
        if outcome == "death"
        else f"{outcome}_followup_time_hours"
    )
    return {
        "exposure": exposure,
        "outcome": outcome,
        "exposure_materialized": exposure_materialized,
        "outcome_materialized": outcome_materialized,
        "event_time": event_time,
        "observation_duration": observation_duration,
        "observation_duration_unit": "hours",
    }


def plan_decision_context(
    plan: Mapping[str, Any], decision_code: str
) -> Dict[str, Any]:
    """Project plan-bound coordinates needed to render one host decision."""

    code = str(decision_code or "").strip()
    if code != "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED":
        return {}
    coordinates = _timing_coordinates(plan)
    selected = _selected_design(plan)
    exposure = coordinates["exposure"]
    outcome = coordinates["outcome"]
    fixed_24h_lactate = (
        exposure == "lact" and coordinates["exposure_materialized"] == "lact_max"
    )
    return {
        **coordinates,
        "exposure_label_en": _DISPLAY_LABELS_EN.get(exposure, exposure),
        "exposure_label_zh": _DISPLAY_LABELS_ZH.get(exposure, exposure),
        "outcome_label_en": _DISPLAY_LABELS_EN.get(outcome, outcome),
        "outcome_label_zh": _DISPLAY_LABELS_ZH.get(outcome, outcome),
        "time_zero": str(selected.get("time_zero") or "").strip(),
        "observation_window": str(selected.get("observation_window") or "").strip(),
        "timing_profile": (
            "fixed_24h_lactate"
            if fixed_24h_lactate
            else "unspecified_post_baseline"
        ),
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
        # “Keep and execute” authorizes execution of an already confirmed
        # specification.  It must never reinterpret a protocol-only plan and
        # replace (for example) a time-varying design with a landmark design.
        # A missing timing specification is therefore an authority error, not
        # permission to synthesize a convenient fallback from the agent plan.
        raw_specs = study.get("sensitivity_specs")
        timing_rows = (
            [
                dict(spec)
                for spec in raw_specs
                if isinstance(spec, Mapping)
                and str(spec.get("axis") or "") == "timing"
            ]
            if isinstance(raw_specs, Sequence) and not isinstance(raw_specs, (str, bytes))
            else []
        )
        if len(timing_rows) != 1:
            raise PlanDecisionError(
                "plan_decision_confirmed_timing_missing",
                "Keeping a required sensitivity requires exactly one existing typed timing specification.",
                details={"timing_spec_count": len(timing_rows)},
            )
        from easyicu.research_agent.planning.sensitivity_authority import (
            EXECUTABLE_METHODS_BY_STRATEGY,
            PrespecifiedSensitivitySpec,
        )

        try:
            timing = PrespecifiedSensitivitySpec.model_validate(
                timing_rows[0]
            ).model_dump(mode="json")
        except ValueError as exc:
            raise PlanDecisionError(
                "plan_decision_confirmed_timing_invalid",
                "The existing timing specification is not a valid executable sensitivity contract.",
                details={"reason": str(exc)[:500]},
            ) from exc
        if (not EXECUTABLE_METHODS_BY_STRATEGY[timing["strategy"]]
            or (timing["strategy"] == "time_varying" and timing.get("time_varying_execution") is None)):
            raise PlanDecisionError(
                "plan_decision_confirmed_timing_runtime_unavailable",
                "The saved timing strategy has no registered deterministic runtime.",
                details={
                    "spec_id": timing["spec_id"],
                    "strategy": timing["strategy"],
                },
            )
        return CompiledPlanDecision(
            patch={
                "sensitivity_specs": configuration.replace_sensitivity(
                    axis="timing", replacement=timing
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
        current_design = study.get("analysis_design")
        current_family = (
            str(current_design.get("analysis_family") or "").strip()
            if isinstance(current_design, Mapping)
            else ""
        )
        current_cohort = study.get("cohort")
        cohort = dict(current_cohort) if isinstance(current_cohort, Mapping) else {}
        # This option explicitly changes the estimand from a first-stay
        # restriction to every ICU stay.  Persist that population choice with
        # the dependence model; otherwise a prior unverified first-stay flag
        # survives the click and the full launch correctly rejects the
        # contradictory configuration later.
        cohort["exclude_readmissions"] = False
        return CompiledPlanDecision(
            patch={
                "cohort": cohort,
                "analysis_design": {
                    **(
                        {"analysis_family": current_family}
                        if current_family
                        else {}
                    ),
                    "analysis_unit": "icu_stay",
                    "variance_estimator": "cluster_robust",
                    "cluster_unit": "patient",
                },
                # Patient-clustered uncertainty is the primary estimator's
                # dependence model, not a second sensitivity analysis.  Drop
                # any legacy duplicate written by older host versions so the
                # scientific reviewer does not require a redundant replay.
                "sensitivity_specs": configuration.replace_sensitivity(
                    axis="repeated_stays", replacement=None
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
                "入 ICU 时已确定的基线人口学因素，可能同时关联主要暴露与结局。"
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
    fixed_24h_lactate = (
        exposure == "lact" and coordinates["exposure_materialized"] == "lact_max"
    )
    execution = {
        "outcome": outcome,
        "primary_exposure": exposure,
        "primary_exposure_aggregation": "max",
    }

    if option == "landmark_24h":
        if not fixed_24h_lactate:
            raise PlanDecisionError(
                "plan_decision_option_not_applicable",
                "A 24-hour landmark is only available when the reviewed plan "
                "binds the fixed 0-24-hour lactate maximum.",
                details={
                    "decision_code": code,
                    "option_id": option,
                    "primary_exposure": coordinates["exposure_materialized"],
                },
            )
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
            "observation_duration_unit": coordinates["observation_duration_unit"],
        }
        current_design = study.get("analysis_design")
        current_design = (
            current_design if isinstance(current_design, Mapping) else {}
        )
        association_design = {
            "analysis_family": "association_study",
            "analysis_unit": "icu_stay",
            "variance_estimator": "model_based",
        }
        if (
            current_design.get("variance_estimator") == "cluster_robust"
            and current_design.get("cluster_unit") == "patient"
        ):
            association_design.update(
                {
                    "variance_estimator": "cluster_robust",
                    "cluster_unit": "patient",
                }
            )
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
                # The timing choice closes an observational association
                # design, not just its prose description.  Persist the typed
                # launch contract atomically so the next Planner run can
                # consume the user's decision without asking the model to
                # reconstruct it.  A later repeated-stay decision may upgrade
                # the variance estimator while preserving this family.  When
                # repeated-stay dependence was already confirmed, retain that
                # compatible typed choice instead of silently downgrading it
                # while leaving its confirmation receipt behind.
                "analysis_design": association_design,
                "export_format": "parquet",
                "sensitivity_specs": configuration.replace_sensitivity(
                    axis="timing", replacement=sensitivity
                ),
                "confirmations": configuration.merge_confirmations(
                    feature_time_window=True,
                    extraction_completed=True,
                    export_format=True,
                    plan_timing_landmark_24h=True,
                    plan_timing_descriptive_only=False,
                    plan_timing_time_varying=False,
                ),
            },
            display_label_en="Use the recommended 24-hour landmark",
            display_label_zh="已采用推荐的 24 小时 landmark",
            next_action="replan",
        )

    if option == "descriptive_only":
        exposure_display = (
            f"入 ICU 后 0–24 小时{exposure_zh}"
            if fixed_24h_lactate
            else exposure_zh
        )
        descriptive_execution = dict(execution)
        descriptive_execution["covariates"] = []
        return CompiledPlanDecision(
            patch={
                "outcome": outcome_zh,
                "primary_exposure": exposure_display,
                "execution_concepts": descriptive_execution,
                "analysis_goal": "描述暴露与结局分布，不估计时间对齐后的关联",
                # This click changes the scientific family, so it must also
                # close the typed launch contract.  Leaving only prose in
                # ``analysis_goal`` made the next replan fail before the
                # Research Agent could see the user's decision.  Counts-only
                # descriptive work has no independence-sensitive estimator;
                # do not carry a stale cluster/model variance choice forward
                # from the superseded association plan.
                "analysis_design": {
                    "analysis_family": "descriptive_epidemiology",
                    "analysis_unit": "icu_stay",
                    "variance_estimator": "none_counts_only",
                },
                "covariates": [],
                "covariate_selection": "exact",
                "covariate_rationales": {},
                "covariate_temporal_roles": {},
                "covariate_operationalizations": {},
                "export_format": "parquet",
                "sensitivity_specs": configuration.replace_sensitivity(
                    axis="timing", replacement=None
                ),
                "confirmations": configuration.merge_confirmations(
                    feature_time_window=True,
                    extraction_completed=True,
                    export_format=True,
                    plan_timing_landmark_24h=False,
                    plan_timing_descriptive_only=True,
                    plan_timing_time_varying=False,
                    plan_repeated_stays_clustered=False,
                    plan_adjustment_set_confirmed=False,
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
                    plan_timing_landmark_24h=False,
                    plan_timing_descriptive_only=False,
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
    "plan_decision_context",
    "proposed_adjustment_set",
]
