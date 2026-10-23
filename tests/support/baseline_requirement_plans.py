"""Baseline-requirement plans and contexts shared by planning and revision tests."""

from __future__ import annotations

from easyicu.research_agent.planning.baseline_requirements import (
    bind_baseline_requirements,
    candidate_baseline_requirements,
)
from easyicu.research_agent.schema import AnalysisPlan, ConceptDescriptor

from tests.research_agent.planning.scientific_review_fixtures import (
    _context,
    _traditional_table_one_step,
)


def _plan(*names: str, group: str = "exposure") -> AnalysisPlan:
    step = _traditional_table_one_step().model_dump(mode="json")
    step["step_id"] = "renamed_baseline"
    step["inputs"] = ["cohort:analysis_set", group, *names]
    spec = step["table_one_spec"]
    spec["group_by"] = group
    spec["variables"] = [dict(spec["variables"][0], name=name) for name in names]
    return AnalysisPlan.model_validate({
        "research_question": _context().research_question, "steps": [step],
    })


def _requirements(*names: str):
    return candidate_baseline_requirements(
        plan=_plan(*names).model_dump(mode="json"),
        source_plan_sha256="a" * 64,
        selected_concepts=["exposure", *names],
        catalog_columns=["exposure", *names],
    )


def _bound_context(*names: str):
    context = _context().model_copy(update={"variables": [
        *_context().variables,
        ConceptDescriptor(name="cci_value", source_concept="charlson", role="other", dtype="float64"),
        ConceptDescriptor(name="charlson_n", source_concept="charlson", role="meta", dtype="int64"),
        ConceptDescriptor(name="charlson_time", source_concept="charlson", role="time", dtype="float64"),
        ConceptDescriptor(name="unrelated", source_concept="other_score", role="other", dtype="float64"),
    ]})
    return bind_baseline_requirements(context, _requirements(*names).model_dump(mode="json"))
