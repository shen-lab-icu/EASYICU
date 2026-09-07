"""Architecture contract for the Planner-owned scientific signature kernel."""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest

from easyicu.research_agent.execution import phase as execution_phase
from easyicu.research_agent.authority import plan_scope
from easyicu.research_agent.contracts.figure_plan import PlannedFigurePanelSpec
from easyicu.research_agent.contracts.functional_form import FunctionalFormSpec
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, PhenotypeComparisonSpec

_PHASE_PLAN_SCOPE_NAMES = {
    "_normalise_scientific_text",
    "_plan_signature",
    "_plan_scientific_scope_signature",
    "_serializable_plan_scientific_scope_signature",
    "_step_scientific_signature",
}


def test_execution_phase_uses_plan_scope_objects_with_identity() -> None:
    assert plan_scope.__all__
    for name in _PHASE_PLAN_SCOPE_NAMES:
        assert getattr(execution_phase, name) is getattr(plan_scope, name)
    assert _PHASE_PLAN_SCOPE_NAMES < set(plan_scope.__all__)


def test_plan_scope_has_no_orchestration_or_mutation_dependency() -> None:
    tree = ast.parse(inspect.getsource(plan_scope))
    imported_leaves = {
        node.module.rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert imported_leaves.isdisjoint(
        {"pipeline", "pipeline_execute", "gates", "execution", "evidence"}
    )
    assert identifiers.isdisjoint(
        {
            "open",
            "write_text",
            "write_bytes",
            "register",
            "promote",
            "consume",
            "repair",
            "complete",
        }
    )


def test_every_public_plan_field_has_exactly_one_authority_class() -> None:
    classes = (
        plan_scope._ANALYSIS_PLAN_CORE_SCIENTIFIC_AUTHORITY_FIELDS,
        plan_scope._ANALYSIS_PLAN_STRUCTURED_SCIENTIFIC_AUTHORITY_FIELDS,
        plan_scope._ANALYSIS_PLAN_STEP_AUTHORITY_FIELDS,
        plan_scope._ANALYSIS_PLAN_PRESENTATION_ONLY_FIELDS,
        plan_scope._ANALYSIS_PLAN_RUNTIME_ONLY_FIELDS,
    )
    flattened = [field for fields in classes for field in fields]
    assert set(flattened) == set(AnalysisPlan.model_fields)
    assert len(flattened) == len(set(flattened))


def test_every_public_step_field_has_exactly_one_authority_class() -> None:
    classes = (
        plan_scope._ANALYSIS_STEP_CORE_SCIENTIFIC_AUTHORITY_FIELDS,
        plan_scope._ANALYSIS_STEP_STRUCTURED_SCIENTIFIC_AUTHORITY_FIELDS,
        plan_scope._ANALYSIS_STEP_PRESENTATION_ONLY_FIELDS,
        plan_scope._ANALYSIS_STEP_RUNTIME_ONLY_FIELDS,
    )
    flattened = [field for fields in classes for field in fields]
    assert set(flattened) == set(AnalysisStep.model_fields)
    assert len(flattened) == len(set(flattened))


@pytest.mark.parametrize("field, initial, changed", [
    ("phenotyping_feature_columns", ["age"], ["age", "severity"]),
    ("functional_form_spec",
     FunctionalFormSpec(target_column="age", knot_quantiles=(0.1, 0.5, 0.9)),
     FunctionalFormSpec(target_column="age", knot_quantiles=(0.2, 0.5, 0.8))),
    ("phenotype_comparison_spec",
     PhenotypeComparisonSpec(identity_column="stay_id", outcome_columns=["death"], variables=[
         {"name": "death", "variable_kind": "categorical", "summary": "count_percent",
          "test": "none_descriptive_smd_only", "levels": [0, 1]},
     ]),
     PhenotypeComparisonSpec(identity_column="patient_id", outcome_columns=["death"], variables=[
         {"name": "death", "variable_kind": "categorical", "summary": "count_percent",
          "test": "none_descriptive_smd_only", "levels": [0, 1]},
     ])),
])
def test_new_scientific_fields_change_plan_signature(field, initial, changed) -> None:
    # Exercise the signature owner, not the separate executable-step validator.
    base = AnalysisStep(step_id="scope_probe", intent="Test scope identity.", method="custom_analysis")
    before = base.model_copy(update={field: initial})
    after = base.model_copy(update={field: changed})
    assert plan_scope._step_scientific_signature(before) != plan_scope._step_scientific_signature(after)
    assert field in plan_scope._ANALYSIS_STEP_STRUCTURED_SCIENTIFIC_AUTHORITY_FIELDS


def test_typed_figure_panel_is_part_of_scientific_plan_authority() -> None:
    base = AnalysisStep(
        step_id="06_figure",
        intent="Render the prespecified descriptive panel.",
        method="visualization",
        planned_analysis_role="auxiliary",
        inputs=["table:exposure_outcome_distribution"],
        expected_outputs=["figure:descriptive_overview"],
        figure_panels=[
            PlannedFigurePanelSpec(
                panel_id="absolute_risk",
                figure_output="figure:descriptive_overview",
                article_role="distribution",
                chart_type="point_interval",
                source_products=["table:exposure_outcome_distribution"],
            )
        ],
    )
    changed = base.model_copy(
        update={
            "figure_panels": [
                base.figure_panels[0].model_copy(
                    update={"article_role": "data_quality"}
                )
            ]
        }
    )

    assert plan_scope._step_scientific_signature(base) != (
        plan_scope._step_scientific_signature(changed)
    )


def test_runtime_outcome_owner_and_endpoint_are_scientific_plan_authority() -> None:
    def step(owner: str, outcome: str) -> AnalysisStep:
        return AnalysisStep(
            step_id="runtime_analysis", intent="Execute the governed endpoint analysis.",
            method="custom_analysis", expected_outputs=["table:runtime_result"],
            runtime_outcome_contract={
                "owner_ref": "scientific_runtime_contract:" + owner * 64,
                "outcomes": [outcome],
            },
        )

    base = plan_scope._step_scientific_signature(step("a", "death"))
    assert base != plan_scope._step_scientific_signature(step("b", "death"))
    assert base != plan_scope._step_scientific_signature(step("a", "icu_los"))


def test_scientific_signature_uses_typed_role_not_intent_role_words() -> None:
    step = AnalysisStep(
        step_id="01_model",
        intent=(
            "Discuss primary, secondary, sensitivity, and corroborative results "
            "without owning the primary estimand."
        ),
        method="logistic_regression",
        planned_analysis_role="auxiliary",
        expected_outputs=["statistic:adjusted_effect"],
    )

    signature = plan_scope._step_scientific_signature(step)

    assert signature[6] == "auxiliary"
    assert not isinstance(signature[6], tuple)


def test_scientific_signature_changes_only_role_coordinate_for_role_change() -> None:
    primary = AnalysisStep(
        step_id="01_model",
        intent="Fit the prespecified model.",
        method="logistic_regression",
        planned_analysis_role="primary",
        expected_outputs=["statistic:adjusted_effect"],
    )
    secondary = primary.model_copy(update={"planned_analysis_role": "secondary"})

    primary_signature = plan_scope._step_scientific_signature(primary)
    secondary_signature = plan_scope._step_scientific_signature(secondary)

    assert primary_signature != secondary_signature
    assert [
        index
        for index, (left, right) in enumerate(
            zip(primary_signature, secondary_signature, strict=True)
        )
        if left != right
    ] == [6]


def test_plan_display_labels_are_presentation_only() -> None:
    base = AnalysisPlan(
        research_question="Estimate an adjusted association.",
        steps=[],
        display_labels={"death": "In-hospital mortality"},
    )
    changed = base.model_copy(update={"display_labels": {"death": "28-day mortality"}})

    assert plan_scope._plan_scientific_scope_signature(base) == (
        plan_scope._plan_scientific_scope_signature(changed)
    )


def test_plan_endpoint_is_scientific_scope_authority() -> None:
    from easyicu.research_agent.schema import EndpointSpec

    base = AnalysisPlan(research_question="q", steps=[])
    changed = base.model_copy(
        update={
            "endpoint": EndpointSpec(
                name="death",
                kind="binary",
                absence_semantics="no_absent_rows",
                levels=[0, 1],
            )
        }
    )

    assert plan_scope._plan_scientific_scope_signature(base) != (
        plan_scope._plan_scientific_scope_signature(changed)
    )


def test_revision_is_runtime_history_not_scientific_scope() -> None:
    base = AnalysisPlan(research_question="q", steps=[], revision=1)
    changed = base.model_copy(update={"revision": 2})
    assert plan_scope._plan_scientific_scope_signature(base) == (
        plan_scope._plan_scientific_scope_signature(changed)
    )


def test_legacy_missing_empty_literature_decisions_matches_but_nonempty_does_not() -> (
    None
):
    expected = [
        "question",
        "association_study",
        '{"design_selection":{"candidates":['
        '{"design_id":"a","literature_design_decisions":[]}]}}',
        "rationale",
    ]
    legacy = [
        "question",
        "association_study",
        '{"design_selection":{"candidates":[{"design_id":"a"}]}}',
        "rationale",
    ]
    nonempty = [
        "question",
        "association_study",
        '{"design_selection":{"candidates":['
        '{"design_id":"a","literature_design_decisions":['
        '{"dimension":"missing_data"}]}]}}',
        "rationale",
    ]

    assert plan_scope._plan_scope_signatures_match(legacy, expected)
    assert not plan_scope._plan_scope_signatures_match(nonempty, expected)


def test_plan_display_labels_reject_conflicting_normalized_keys() -> None:
    with pytest.raises(ValueError, match="conflicting normalized keys"):
        AnalysisPlan(
            research_question="Estimate an adjusted association.",
            steps=[],
            display_labels={
                "death-hosp": "Hospital mortality",
                "death_hosp": "In-hospital death",
            },
        )

    with pytest.raises(ValueError, match="at least one letter or digit"):
        AnalysisPlan(
            research_question="Estimate an adjusted association.",
            steps=[],
            display_labels={"---": "Invalid identifier"},
        )


@pytest.mark.parametrize("canonical_first", [True, False])
def test_plan_scope_identity_survives_import_order(canonical_first: bool) -> None:
    canonical = "easyicu.research_agent.authority.plan_scope"
    consumer = "easyicu.research_agent.execution.phase"
    first, second = (canonical, consumer) if canonical_first else (consumer, canonical)
    script = f"""
import importlib
first = importlib.import_module({first!r})
second = importlib.import_module({second!r})
canonical = importlib.import_module({canonical!r})
consumer = importlib.import_module({consumer!r})
for name in {_PHASE_PLAN_SCOPE_NAMES!r}:
    assert getattr(consumer, name) is getattr(canonical, name), name
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[3] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)
