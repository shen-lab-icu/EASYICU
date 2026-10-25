"""Read-only clinical card facts; product-labels.js owns their bilingual wording.

Only declared cohort/model/endpoint coordinates cross this boundary. Candidate
sentences are retained by the complete reader, never used to infer these facts.
This is a browser projection, not a Planner contract or execution authority.
"""

from __future__ import annotations

import math
import re
from types import SimpleNamespace
from typing import Any, Mapping

from easyicu.research_agent.contracts.prediction_execution import (
    PREDICTION_PRIMARY_ACTION,
    static_prediction_execution_verdict,
    static_prediction_features,
    static_prediction_model_columns,
)
from easyicu.research_agent.planning.step_phase import design_analysis_family


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list:
    return value if isinstance(value, list) else []


def _variable(value: Any) -> dict[str, str]:
    name = str(value or "")
    # Identity fields have no clinical display role, even with a supplied label.
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,119}", name) or re.search(
        r"(?:^|_)(?:id|identifier|key)$|^(?:subject|patient|stay|icustay|hadm)_?id$", name, re.I
    ):
        return {}
    return {"concept": name}


def _variables(values: Any) -> list[dict[str, str]]:
    return [term for value in _rows(values) if (term := _variable(value))]


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _timing(plan: Mapping, study: Mapping, steps: list[Mapping]) -> dict | None:
    bound = {key for step in steps for key in _rows(step.get("sensitivity_spec_ids"))}
    if _map(study.get("confirmations")).get("plan_timing_landmark_24h") is True:
        for spec in _rows(study.get("sensitivity_specs")):
            spec = _map(spec)
            hours = spec.get("landmark_hours")
            if spec.get("spec_id") in bound and spec.get("strategy") == "landmark" and _number(hours) and hours > 0:
                return {"anchor": "icu_admit", "hours": hours, "risk_set": spec.get("require_alive_at_landmark") is True}
    origin = _map(plan.get("endpoint")).get("time_origin")
    if origin in {"icu_admit", "hospital_admit"}:
        return {"anchor": origin, "hours": 0, "risk_set": False}
    candidates = _rows(_map(plan.get("design_selection")).get("candidates"))
    declared_start = any(_map(row).get("disposition") == "selected" and _map(row).get("time_zero")
                         for row in candidates)
    descriptive = design_analysis_family(plan.get("analysis_type")) == "descriptive"
    return {"unresolved": True} if origin or (declared_start and not descriptive) else None


def project_clinical_summary(
    plan: Mapping[str, Any], *, study: Any = None, exposure_groups: Any = None,
    review_summary: Any = None,
) -> dict[str, Any]:
    """Five ordered fact records plus optional declared timing, with no prose fallback."""
    steps = [_map(step) for step in _rows(plan.get("steps"))]
    primary = next((step for step in steps if step.get("planned_analysis_role") == "primary"), {})
    model = next((item for item in _rows(primary.get("model_requirements"))
                  if _map(item).get("analysis_role") == "primary"), {})
    model = _map(model) or _map(primary.get("family_primary_result_requirement"))
    family = design_analysis_family(plan.get("analysis_type"))
    descriptive = family == "descriptive"
    prediction = family == "prediction"
    cohort = _map(plan.get("cohort"))
    predicates = [(value, exclude) for key, exclude in (("inclusion", False), ("exclusion", True))
                  for value in _rows(cohort.get(key))]
    criteria = []
    for raw, exclude in predicates:
        pred = _map(raw)
        variable = _variable(pred.get("concept_id"))
        value = pred.get("value")
        if variable and pred.get("op") in {"==", "!=", ">", ">=", "<", "<="} and _number(value):
            criteria.append({**variable, "op": pred["op"], "value": value, "exclude": exclude})
    population = {"key": "population", "criteria": criteria,
                  "all_rows": bool(cohort) and (cohort.get("selection_mode") == "all_input_rows" or not predicates),
                  "unresolved": bool(cohort.get("unapplied_population_criteria")) or len(criteria) != len(predicates) or not cohort}
    endpoint = _map(plan.get("endpoint"))
    distribution = next((_map(step.get("exposure_outcome_distribution_spec")) for step in steps
                         if step.get("exposure_outcome_distribution_spec")), {})
    outcome = model.get("outcome") or endpoint.get("name") or distribution.get("outcome")
    table = next((_map(step.get("table_one_spec")) for step in steps if step.get("table_one_spec")), {})
    exposure = table.get("group_by") if descriptive else model.get("exposure_source")
    exposure = exposure or distribution.get("exposure")
    group = next((item for item in _rows(_map(exposure_groups).get("groupings"))
                  if _map(item).get("variable") == exposure), {})
    group = _map(group)
    compared = (group.get("reference"), group.get("contrast"))
    levels = [_map(level) for level in _rows(group.get("levels"))]
    group_labels = [{"en": level.get("reader_name") or level.get("label", ""), "zh": level.get("label", "")}
                    for code in compared for level in levels if code is not None and level.get("code") == code]
    comparison = {"key": "comparison", "mode": "descriptive" if descriptive else "prediction" if prediction else "association",
                  "variables": _variables([group.get("concept") or exposure]), "groups": group_labels}
    exposure_term = next((_map(term) for term in _rows(model.get("model_terms"))
                          if _map(term).get("role") == "exposure"), {})
    comparison["continuous"] = exposure_term.get("coding") == "continuous"
    comparison["unresolved"] = not comparison["variables"] or bool(group and len(group_labels) != 2)
    covariates = model.get("covariates", _map(plan.get("adjustment_proposal")).get("covariates"))
    missing = model.get("missing_data_policy") or model.get("analysis_set")
    if _map(model.get("baseline_missing_handling")).get("policy") == "explicit_missing_category":
        missing = "explicit_missing_category"
    evaluation = None
    if prediction:
        if primary.get("scientific_action_id") == PREDICTION_PRIMARY_ACTION:
            step = SimpleNamespace(**primary)
            columns = static_prediction_model_columns(step)
            covariates = list(static_prediction_features(columns, outcome=outcome, group_source=None))
            claimed = static_prediction_execution_verdict(step).claimed
            missing = "training_imputation" if claimed else None
            evaluation = "discrimination_calibration" if claimed else None
        comparison.update(variables=_variables(covariates), unresolved=not bool(covariates))
    adjustment = {"key": "adjustment", "mode": "descriptive" if descriptive else "prediction" if prediction else "adjusted",
                  "variables": _variables(covariates), "unresolved": not (descriptive or prediction) and covariates is None}
    if prediction:
        adjustment["evaluation"] = evaluation
    if covariates == [] and not prediction and not descriptive:
        adjustment["mode"] = "unadjusted"
    outcome_row = {"key": "outcome", "variables": _variables([outcome]), "unresolved": not bool(_variable(outcome))}
    checks = {"key": "data_checks", "missing": missing if missing in {
        "complete_case", "explicit_missing_category", "training_imputation",
    } else "unresolved"}
    findings = set(_rows(_map(review_summary).get("finding_codes")))
    for row, codes in (
        (population, {"POPULATION_CRITERION_NOT_APPLIED", "PLAN_POPULATION_REQUIREMENT_DRIFT"}),
        (comparison, {"POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED", "PRIMARY_EXPOSURE_TIME_ANCHOR_UNVERIFIED", "PREDICTION_PREDICTOR_TIMING_UNPROVEN"}),
        (outcome_row, {"OUTCOME_DEFINITION_UNRESOLVED"}),
        (adjustment, {"PLANNER_ADJUSTMENT_PROPOSAL_INCOMPLETE", "STATED_ADJUSTMENT_UNRESOLVED", "ADJUSTMENT_RATIONALE_OR_TIMING_UNBOUND"}),
    ):
        row["unresolved"] = row.get("unresolved", False) or bool(findings & codes)
    lines = [population, comparison, outcome_row, adjustment, checks]
    labels = _map(plan.get("display_labels"))
    for row in lines:
        for term in row.get("variables", row.get("criteria", [])):
            label = labels.get(term["concept"])
            if isinstance(label, str) and 0 < len(label) <= 64 and not re.search(r"[_\n]|<[/A-Za-z!]|/(?:home|Users)/", label):
                term["label"] = label
    return {"lines": lines,
            "timing": _timing(plan, _map(study), steps)}
