"""A question that states no family is read on the study's typed design.

pilot1 10-26: 「入 ICU 第一天最高乳酸的中位数是多少？男女之间有差别吗？」 had a
descriptive design, but its words state no family. The coordinate reader read
it as non-descriptive: lactate became the exposure, the first-day maximum was
never bound, and the exact grouped summary fell to analysis-only
(SCIENTIFIC_CAPABILITY_NOT_REPORTABLE). Synthetic concept owners only.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from easyicu.concept.schema import ConceptDefinition
from easyicu.research_agent.acquisition import catalog
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.schema import TimeWindow, VariableRole
from easyicu.webserver import research_launch_scientific as launch

_WEBSERVER = Path(__file__).resolve().parents[2] / "src" / "easyicu" / "webserver"
# Its words name a statistic and a grouping but no family (no proportion, no relationship).
QUESTION = "入 ICU 第一天最高 signal 的中位数是多少？各 ward_group 之间有差别吗？"
DESCRIPTIVE = "descriptive_epidemiology"


@pytest.fixture
def owners(monkeypatch):
    from easyicu.concept import loader
    from easyicu.research_agent.research_context import builder
    from easyicu.webserver import study_intent

    definitions = {
        "signal": ConceptDefinition.from_name_and_payload("signal", {
            "unit": "mmol/L", "description": "Synthetic measured concentration",
        }),
        "ward_group": ConceptDefinition.from_name_and_payload("ward_group", {
            "class": "fct_cncpt", "target": "id_tbl",
            "description": "Synthetic ward category",
        }),
    }
    menu = catalog.AvailableCatalog(source="synthetic", concepts=[
        catalog.CatalogConcept(concept_id=name, description=definition.description or "")
        for name, definition in definitions.items()
    ])
    monkeypatch.setattr(loader, "load_dictionary", lambda: definitions)
    monkeypatch.setattr(catalog, "build_database_capability_catalog", lambda db: menu)
    monkeypatch.setattr(builder, "_concept_info_for_wide_column", lambda col:
                        ({"name": col, "description": definitions[col].description}, col)
                        if col in definitions else (None, None))
    intent = study_intent.deterministic_intent

    def read(question):
        # The keyword reader names the measurement in its exposure slot, as it
        # named lactate on pilot1.
        result = intent(question)
        result["slots"]["exposure"] = {"value": "signal", "provenance": "keyword", "evidence": "signal"}
        return result

    monkeypatch.setattr(study_intent, "deterministic_intent", read)
    monkeypatch.setattr(study_intent, "named_study_concepts", lambda q: [(("signal",), "signal")])
    monkeypatch.setattr(study_intent, "catalog_named_concepts", lambda q: ["signal"])
    # The catalog's name readings are cached for the process; the synthetic
    # name pattern must not stay in them for later tests.
    caches = (study_intent._catalog_name_patterns, study_intent._catalog_name_readings)
    for cached in caches:
        cached.cache_clear()
    monkeypatch.setattr(study_intent, "_concept_name_pattern", lambda name: rf"\b{re.escape(name)}\b")
    yield definitions
    for cached in caches:
        cached.cache_clear()


def _coordinates(analysis_family=None):
    return launch._metadata_only_planning_coordinates(
        question=QUESTION, database="synthetic", analysis_family=analysis_family,
    )


@pytest.mark.parametrize("family", [None, "association_study"])
def test_without_a_descriptive_design_the_keyword_reading_is_unchanged(owners, family):
    coordinates = _coordinates(family)
    assert coordinates["primary_exposure"] == "signal"
    assert coordinates["endpoint"] is None
    assert coordinates["target_outcome_window"] is None


def test_a_descriptive_design_binds_the_named_window_summary_as_the_outcome(owners):
    coordinates = _coordinates(DESCRIPTIVE)
    assert coordinates["primary_exposure"] is None
    assert coordinates["target_outcome"] == "signal"
    assert coordinates["endpoint"].name == "signal_max"
    assert coordinates["endpoint"].kind == "continuous"
    # 「中位数」 summarizes the first-day maximum; it is not its reduction.
    assert coordinates["target_outcome_aggregation"] == "max"
    assert coordinates["target_outcome_window"] == {
        "anchor": "icu_admission", "start_hours": 0.0, "end_hours": 24.0,
    }
    assert [row["role"] for row in coordinates["coordinate_readings"]] == ["outcome"]


def _planning_context(coordinates):
    endpoint = coordinates["endpoint"]
    frame = pd.DataFrame({
        "stay_id": pd.Series(dtype="int64"), "signal": pd.Series(dtype="float64"),
        endpoint.name: pd.Series(dtype="float64"), "ward_group": pd.Series(dtype="string"),
    })
    frame.attrs["easyicu_planning_authority"] = {
        "kind": "metadata_only_planning_catalog", "patient_rows_read": False,
    }
    return build_research_context(
        research_question=QUESTION, cohort=frame, cohort_name="synthetic", database="synthetic",
        id_columns=["stay_id"], target_outcome=endpoint.name, endpoint=endpoint,
        primary_exposure=coordinates["primary_exposure"],
        time_windows=[TimeWindow(name="bound", anchor="icu_admission", start_hours=0, end_hours=24)],
        user_preferences={"inferred_analysis_family": DESCRIPTIVE, "data_constraints": json.dumps({
            "materialization_window": {"role": "outer_observation_window", "anchor": "icu_admission", "hours": 24},
        })},
    )


def test_the_planners_grouped_summary_is_reportable_on_the_design_reading(owners):
    from easyicu.research_agent.planning.capability_registry import assess_scientific_capability
    from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review

    context = _planning_context(_coordinates(DESCRIPTIVE))
    assert context.primary_exposure is None
    assert context.variable("signal_max").role == VariableRole.OUTCOME
    # The shape the pilot1 Planner wrote: the raw measurement and its grouping.
    skeleton = ProgressivePlanSkeleton.model_validate({
        "analysis_type": DESCRIPTIVE,
        "cohort": {"name": "synthetic", "selection_mode": "all_input_rows"},
        "display_labels": [{"key": name, "value": name} for name in ("signal", "signal_max", "ward_group")],
        "steps": [
            {"step_id": "cohort", "module_id": "cohort_definition", "planned_analysis_role": "auxiliary",
             "objective": "Account for the declared synthetic stay denominator.", "raw_inputs": ["stay_id"]},
            {"step_id": "summary", "module_id": "custom_analysis", "planned_analysis_role": "primary",
             "scientific_action_id": "descriptive.descriptive_summary", "custom_method": "grouped_summary",
             "objective": "Describe the first-day maximum within each ward.", "depends_on": ["cohort"],
             "raw_inputs": ["signal", "ward_group"],
             "product_inputs": [{"producer_step_id": "cohort", "product_id": "artifact:analysis_cohort"}],
             "outputs": [{"product_id": product, "semantic_role": "custom"}
                         for product in ("table:cohort_summary", "table:distribution_summary")]},
        ],
        "rationale": "Describe the requested synthetic quantity with its original denominator.",
    })
    plan, _ = compile_progressive_plan(skeleton=skeleton, context=context)
    primary = next(step for step in plan.steps if step.planned_analysis_role == "primary")
    assert {name for name in primary.inputs if ":" not in name} == {"signal_max", "ward_group"}
    assert assess_scientific_capability(
        analysis_type=plan.analysis_type, plan=plan, context=context,
    ).claim_ceiling == "reportable"
    review = build_plan_scientific_review(context=context, plan=plan, require_reportable_capability=True)
    assert "SCIENTIFIC_CAPABILITY_NOT_REPORTABLE" not in {finding.code for finding in review.findings}


@pytest.mark.parametrize("study,family", [
    ({"analysis_design": {"analysis_family": DESCRIPTIVE, "analysis_unit": "icu_stay",
                          "variance_estimator": "model_based"}}, DESCRIPTIVE),
    ({"confirmations": {"plan_timing_descriptive_only": True}}, DESCRIPTIVE),
    ({"analysis_design": {"analysis_family": "association_study"}}, "association_study"),
    ({}, None),
    # A design the launch refuses is refused there, not read here.
    ({"analysis_design": "descriptive"}, None),
])
def test_the_study_design_family_is_the_one_the_launch_plans_on(study, family):
    assert launch.study_design_family(study) == family


@pytest.mark.parametrize("module", [
    "research_pipeline_run_preparation.py", "agent_pipeline_runs.py", "pi_copilot/plan_decisions.py",
])
def test_every_planning_reader_receives_the_study_design_family(module):
    tree = ast.parse((_WEBSERVER / module).read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_metadata_only_planning_coordinates"
    ]
    assert calls
    assert all("analysis_family" in {keyword.arg for keyword in call.keywords} for call in calls)
