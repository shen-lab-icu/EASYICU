"""Synthetic grouped contrast plans and records; no source data."""

from easyicu.research_agent.contracts.exposure_group_rules import (
    EXPOSURE_GROUP_CONTRASTS_KEY, EXPOSURE_GROUP_LABELS_KEY,
)
from easyicu.research_agent.planning.exposure_group_compile import CompiledGrouping, CompiledGroupings
from easyicu.research_agent.planning.exposure_group_spec import ExposureGroupSpec
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
from tests.research_agent.planning.progressive_planner_fixtures import _context, _payload


def grouped_context(*, stated=None):
    context = _context()
    variable = context.variables[0].model_copy(update={
        "observed_domain": {"n_unique": 3, "is_binary": False, "levels": [1, 2, 3]},
    })
    contrast = {"variable": "exposure_flag", "reference": 2, "contrast": 3}
    if stated is not None:
        contrast["stated"] = stated
    return context.model_copy(update={
        "variables": [variable, *context.variables[1:]],
        "cohort": context.cohort.model_copy(update={"database": "miiv", "provenance": {
            EXPOSURE_GROUP_CONTRASTS_KEY: [contrast],
            EXPOSURE_GROUP_LABELS_KEY: [{
                "variable": "exposure_flag", "concept": "bmi", "scale": "ordinal",
                "levels": [
                    {"code": i, "group": f"g{i}", "label": f"Band {letter}"}
                    for i, letter in enumerate("ABC", 1)
                ],
            }],
        }}),
    })


def skeleton(reference=2, comparison=3, *, model=False):
    payload = _payload()
    payload["robustness_intents"] = []
    payload["steps"] = [payload["steps"][0], payload["steps"][4 if model else 2]]
    step = payload["steps"][1]
    if model:
        step["model_terms"][0].update(coding="categorical", reference_level_index=reference - 1)
        step["primary_contrast_level_index"] = comparison - 1
    else:
        step.update(reference_exposure_level_index=reference - 1,
                    comparison_exposure_level_index=comparison - 1,
                    planned_analysis_role="primary")
        payload["analysis_type"] = "descriptive_epidemiology"
    return ProgressivePlanSkeleton.model_validate(payload)


def _bmi():
    return {
        "id": "x2", "concept": "bmi", "window": None, "scale": "ordinal",
        "groups": [
            {"id": "g1", "label": "Band A", "rule": {"summary": "value", "op": "<", "value": 19}},
            {"id": "g3", "label": "Band C", "rule": {"summary": "value", "op": ">=", "value": 28}},
            {"id": "g2", "label": "Band B", "rule": "otherwise"},
        ],
        "unmeasured": {"handling": "exclude"}, "reference": "g2", "contrast": "g3",
        "quote": "Compare synthetic body mass bands", "source": "question",
    }


def _record(spec):
    item=CompiledGrouping(grouping=ExposureGroupSpec.model_validate(spec), disposition="applied",
                          reason=None, detail="Synthetic", source_columns={"value": "bmi"})
    compiled=CompiledGroupings((item,))
    return {"schema_version": "easyicu.exposure_groupings_record/1",
            "compiled": compiled.record(), "compiled_sha256": compiled.sha256()}
