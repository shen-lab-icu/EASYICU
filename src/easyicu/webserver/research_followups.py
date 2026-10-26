"""Deterministic next questions grounded in completed design and result evidence."""
from __future__ import annotations

from pathlib import Path

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.contracts.exposure_group_rules import EXPOSURE_GROUP_LABELS_KEY, read_grouping_rules
from easyicu.research_agent.planning.adjustment_authority import context_stated_variables
from easyicu.research_agent.planning.family_spec.baseline_association_template import compile_result_append
from easyicu.webserver.research_plan_append import validate_append_structure
from .research_followup_compile import compile_new_study, require_database_family
from .research_followup_contracts import (
    AppendDelta, CompletedResult, FollowupReading, FollowupSuggestion, NewStudyDelta, ResultFollowups,
)
from .research_followup_source import load_completed_result
from .research_followup_databases import OtherDatabaseOwner, other_database_deltas

_RANK = {
    "prespecified_sensitivity": 0, "named_subgroup": 1, "dose_response": 2,
    "complementary_survival": 3, "functional_form_check": 4, "descriptive_context": 5,
    "complementary_association": 6, "article_extension": 7, "same_study_other_database": 8,
}


def _suggestion(source, kind, reasons, delta):
    return FollowupSuggestion(
        suggestion_id=canonical_sha256([source.source_sha256, kind, delta.model_dump(mode="json")]),
        kind=kind, target_action=delta.action, reason_codes=tuple(reasons), delta=delta,
    )

def _append_suggestions(source: CompletedResult):
    if source.baseline is None:
        return
    used_specs = {name for step in source.completed_steps for name in step.sensitivity_spec_ids}
    for kind in ("sensitivity", "subgroup", "exposure", "article"):
        if kind in source.completed_append_kinds:
            continue
        try:
            compiled = compile_result_append(kind=kind, source=source.plan, context=source.context, authority=source.baseline)
            plan = compiled.plan
            validate_append_structure(source.plan.model_dump(mode="json"), plan.model_dump(mode="json"),
                                      action=kind, append_run=True)
            needed = {value for step in plan.steps for value in step.inputs if ":" in value}
            produced = {value for step in plan.steps for value in step.expected_outputs}
            if not needed - produced <= source.products:
                continue
            code, reason = "article_extension", "first_result_completed"
            if kind == "sensitivity":
                specs = {name for step in plan.steps for name in step.sensitivity_spec_ids}
                if not specs or specs & used_specs:
                    continue
                forms = [step.functional_form_spec for step in plan.steps if step.functional_form_spec is not None]
                if any(step.functional_form_spec in forms for step in source.completed_steps if step.functional_form_spec is not None):
                    continue
                preferences = source.context.user_preferences
                stated = {spec.spec_id for spec in preferences.sensitivity_specs} if preferences else set()
                code = "prespecified_sensitivity" if specs <= stated else "functional_form_check"
                reason = "prespecified_not_run" if specs <= stated else "sealed_linear_covariate"
            elif kind == "subgroup":
                existing = {term.name for step in source.completed_steps for req in step.model_requirements
                            for term in req.effect_modifiers}
                proposed = {term.name for step in plan.steps for req in step.model_requirements for term in req.effect_modifiers}
                named = {item.variable for item in context_stated_variables(source.context, "stated_subgroups")}
                if not proposed - existing or not proposed - existing <= named:
                    continue
                code, reason = "named_subgroup", "design_named_subgroup_not_run"
            elif kind == "exposure":
                groups = source.context.cohort.provenance.get(EXPOSURE_GROUP_LABELS_KEY, ())
                if not any(row.get("variable") == source.baseline.exposure_column
                           and read_grouping_rules(row["parameters"]).scale == "ordinal" for row in groups):
                    continue
                code, reason = "dose_response", "sealed_continuous_exposure_source"
            yield _suggestion(source, code, (reason, "append_owner_compiled"), AppendDelta(kind=kind, plan=plan))
        except ValueError:
            continue


def _new_study_suggestions(source: CompletedResult):
    family = source.plan.analysis_type
    targets = [("survival", "complementary_survival", "paired_event_time_available")] if family == "association_study" else []
    if family in {"association_study", "survival", "prediction_model"}:
        # A descriptive primary or absolute-risk product already answers this
        # question. Do not ask for it again under a new step identity.
        has_description = any(step.descriptive_claim is not None or "table:absolute_risk_context" in step.expected_outputs
                              for step in source.completed_steps)
        if not has_description:
            targets.append(("descriptive_epidemiology", "descriptive_context", "primary_result_denominator_context"))
    elif family == "descriptive_epidemiology":
        targets.append(("association_study", "complementary_association", "description_has_no_adjusted_estimand"))
    for target, kind, reason in targets:
        try:
            exposure = source.context.primary_exposure
            if not exposure and family == "prediction_model":
                # Describe outcome denominators by one declared categorical
                # baseline predictor. Never invent a threshold from results.
                inputs = {name for step in source.plan.steps if step.planned_analysis_role == "primary" for name in step.inputs}
                eligible = sorted(v.name for v in source.context.variables if v.name in inputs
                                  and v.role == "demographic" and (v.observed_domain or {}).get("n_unique") in {2, 3, 4})
                if not eligible:
                    continue
                exposure = eligible[0]
            _, plan = compile_new_study(source.context, family=target, source_plan=source.plan, primary_exposure=exposure)
            yield _suggestion(source, kind, (reason, "family_owner_compiled"), NewStudyDelta(
                source_reading=FollowupReading.from_context(source.context, family=family), analysis_family=target,
                primary_exposure=exposure, plan=plan,
            ))
        except ValueError:
            continue


def suggest_followups(source: CompletedResult, *, database_owner: OtherDatabaseOwner | None = None) -> ResultFollowups:
    """At most four runnable deltas, in fixed scientific priority and hash order."""
    try:
        require_database_family(source.context, source.plan.analysis_type)
    except ValueError:
        return ResultFollowups(run_id=source.run_id, source_run_sha256=source.source_sha256,
                               unavailable_actions=("followup_database_refused", "multi_database_unavailable"))
    candidates = [*_append_suggestions(source), *_new_study_suggestions(source)]
    unavailable = ("multi_database_unavailable",) if database_owner is None else ()
    if database_owner is not None:
        candidates.extend(_suggestion(source, "same_study_other_database",
                                      ("registered_database_" + delta.capability_decision.status, "multi_database_owner_compiled"), delta)
                          for delta in other_database_deltas(source, database_owner))
    unique = {row.suggestion_id: row for row in candidates}
    ordered = sorted(unique.values(), key=lambda row: (_RANK[row.kind], row.suggestion_id))
    return ResultFollowups(run_id=source.run_id, source_run_sha256=source.source_sha256,
                           suggestions=ordered[:4], unavailable_actions=unavailable)


def result_followups(service, *, project_id, run_id):
    """Project-scoped read API. Failure of evidence admission offers no action."""
    row = service._research_run_row(project_id, run_id)
    try:
        source = load_completed_result(Path(row["project_dir"]), run_id=run_id, study_id=row["study_id"])
        result = suggest_followups(source, database_owner=getattr(service, "result_followup_database_owner", None))
    except (ValueError, OSError, KeyError):
        result = ResultFollowups(run_id=run_id, unavailable_actions=("followup_source_unavailable", "multi_database_unavailable"))
    from .run_artifact_disclosure import scan_browser_projection

    payload = result.model_dump(mode="json")
    if not scan_browser_projection(payload)["passed"]:
        return ResultFollowups(run_id=run_id, unavailable_actions=("followup_projection_unavailable", "multi_database_unavailable")).model_dump(mode="json")
    return payload
