"""Synthetic contexts and plans for descriptive interval authority tests."""

import json

from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
from easyicu.research_agent.schema import ConceptDescriptor, UserPreferences

from .progressive_planner_fixtures import _context, _payload


def descriptive_context(variance="model_based", *, grouped=False, repeats=True):
    context = _context()
    design = {
        "analysis_family": "descriptive_epidemiology",
        "analysis_unit": "icu_stay",
        "variance_estimator": variance,
    }
    if variance == "cluster_robust":
        design["cluster_unit"] = "patient"
    return context.model_copy(update={
        "research_question": "Describe exposure and outcome distributions in a synthetic ICU cohort.",
        "variables": [
            *[
                variable.model_copy(update={"description": "Synthetic in-hospital binary outcome."})
                if variable.name == "outcome_flag" else variable
                for variable in context.variables
            ],
            ConceptDescriptor(name="stay_id", dtype="int64"),
        ],
        "endpoint": EndpointSpec(
            name="outcome_flag", kind="binary", absence_semantics="no_absent_rows", levels=[0, 1],
        ),
        "cohort": context.cohort.model_copy(update={
            "n_patients": 80 if repeats else 120,
            "id_columns": ["stay_id", "person_key"] if grouped else ["stay_id"],
            "provenance": {
                "analysis_unit": "icu_stay",
                **({"patient_id_columns": ["person_key"]} if grouped else {}),
            },
        }),
        "user_preferences": UserPreferences(
            inferred_analysis_family="descriptive_epidemiology",
            data_constraints=json.dumps({"analysis_design": design}) if variance else None,
        ),
    })


def descriptive_plan(context):
    payload = _payload()
    payload.update(analysis_type="descriptive_epidemiology", robustness_intents=[])
    payload["steps"] = [payload["steps"][0], payload["steps"][2]]
    payload["steps"][0]["raw_inputs"] = ["stay_id"]
    payload["steps"][1]["planned_analysis_role"] = "primary"
    return compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(payload), context=context,
    )[0]


def counts_only_plan(context, *, role="primary"):
    # Construct a valid Planner declaration independently of host compilation.
    plan = descriptive_plan(descriptive_context(repeats=False))
    step = plan.steps[-1]
    payload = step.exposure_outcome_distribution_spec.model_dump(mode="python")
    payload.update(
        schema_version="easyicu.exposure_outcome_distribution/3",
        interval_method="none_counts_only", repeated_unit_interval_method=None,
        patient_weighted_interval_assumptions=None, confidence_level=None,
        dependence=None, risk_difference_contrast=None,
    )
    spec = type(step.exposure_outcome_distribution_spec).model_validate(payload)
    return plan.model_copy(update={
        "research_question": context.research_question,
        "steps": [plan.steps[0], step.model_copy(update={
            "planned_analysis_role": role, "exposure_outcome_distribution_spec": spec,
        })],
    })


