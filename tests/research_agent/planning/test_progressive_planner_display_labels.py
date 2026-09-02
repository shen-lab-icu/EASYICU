"""Focused display-label and outline-authority contracts for progressive planning."""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.progressive_planner import (
    ProgressivePlannerAgent,
    _bind_required_outline_method_sources,
)
from easyicu.research_agent.planning.progressive_compiler import (
    required_reader_display_label_keys,
    validate_progressive_foundation,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlanFoundation,
    ProgressivePlanOutline,
)
from easyicu.research_agent.planning.progressive_host_materialization import (
    progressive_module_method_source_keys,
)
from easyicu.research_agent.schema import UserPreferences

from .test_progressive_planner_v2 import _context, _outline_payload, _payload


def test_outline_binds_sealed_method_source_to_applicable_scientific_module() -> None:
    payload = _outline_payload()
    for step in payload["steps"]:
        if step["planned_analysis_role"] == "secondary":
            step["literature_citation_keys"] = []
    outline = _bind_required_outline_method_sources(
        ProgressivePlanOutline.model_validate(payload),
        allowed_literature_citation_keys=["strobe_2007", "record_2015"],
        context_required_method_layers=(),
        continuous_domain_variables=(),
    )

    distribution = next(
        step
        for step in outline.steps
        if step.module_id == "exposure_outcome_distribution"
    )
    assert distribution.literature_citation_keys == ["strobe_2007"]
    assert progressive_module_method_source_keys(
        "exposure_outcome_distribution", ["record_2015", "strobe_2007"]
    ) == ("record_2015", "strobe_2007")


def test_counts_only_outline_rejects_inferential_plan_review_copy() -> None:
    payload = _outline_payload()
    payload["design_selection"]["candidates"][0]["primary_method"] = (
        "按 stay-level 分母计算比例及其不确定性"
    )
    outline = ProgressivePlanOutline.model_validate(payload)
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_family": "association",
                            "analysis_unit": "icu_stay",
                            "cluster_unit": None,
                            "variance_estimator": "none_counts_only",
                        }
                    }
                )
            )
        }
    )

    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent._validate_outline_authority(
            outline,
            analysis_types=("association_study",),
            variable_names=tuple(variable.name for variable in context.variables),
            allowed_literature_citation_keys=(),
            primary_exposure=context.primary_exposure,
            target_outcome=context.target_outcome,
            require_design_selection=True,
            article_context=context,
        )

    assert caught.value.reason_code == (
        "progressive_selected_design_counts_only_claim_exceeded"
    )
    assert caught.value.path.endswith("primary_method")


def test_outline_rejects_unbound_scientific_step_before_materialization() -> None:
    payload = _outline_payload()
    for step in payload["steps"]:
        if step["planned_analysis_role"] in {"primary", "secondary", "sensitivity"}:
            step["literature_citation_keys"] = ["strobe_2007"]
    target = next(
        step
        for step in payload["steps"]
        if step["module_id"] == "exposure_outcome_distribution"
    )
    target["literature_citation_keys"] = []

    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent._validate_outline_authority(
            ProgressivePlanOutline.model_validate(payload),
            analysis_types=["association_study"],
            variable_names=[variable.name for variable in _context().variables],
            allowed_literature_citation_keys=["strobe_2007"],
        )

    assert caught.value.reason_code == (
        "progressive_outline_scientific_citation_missing"
    )
    assert caught.value.step_id == target["step_id"]


def test_outline_rejects_landmark_declined_by_typed_timing_authority() -> None:
    outline_payload = _outline_payload()
    selected = outline_payload["design_selection"]["candidates"][0]
    selected.update(
        {
            "design_id": "landmark_descriptive_design",
            "time_zero": "A prespecified landmark after cohort entry.",
            "observation_window": "From the landmark to discharge.",
        }
    )
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                timing_and_design=json.dumps(
                    {
                        "analysis_scope": "descriptive_distribution_only",
                        "association_timing": "not_authorized",
                        "landmark": "not_authorized",
                    }
                )
            )
        }
    )

    with pytest.raises(
        ProgressivePlanCompileError,
        match="progressive_selected_design_landmark_not_authorized",
    ):
        ProgressivePlannerAgent._validate_outline_authority(
            ProgressivePlanOutline.model_validate(outline_payload),
            analysis_types=["association_study"],
            variable_names=[variable.name for variable in context.variables],
            allowed_literature_citation_keys=[],
            primary_exposure="exposure_flag",
            target_outcome="outcome_flag",
            require_design_selection=True,
            article_context=context,
        )


def test_outline_rejects_time_varying_design_when_association_timing_declined() -> None:
    outline_payload = _outline_payload()
    selected = outline_payload["design_selection"]["candidates"][0]
    selected.update(
        {
            "design_id": "timevarying_descriptive",
            "primary_method": "Time-updated descriptive risk summaries",
        }
    )
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                timing_and_design=json.dumps(
                    {
                        "analysis_scope": "descriptive_distribution_only",
                        "association_timing": "not_authorized",
                        "landmark": "not_authorized",
                    }
                )
            )
        }
    )

    with pytest.raises(
        ProgressivePlanCompileError,
        match="progressive_selected_design_association_timing_not_authorized",
    ):
        ProgressivePlannerAgent._validate_outline_authority(
            ProgressivePlanOutline.model_validate(outline_payload),
            analysis_types=["association_study"],
            variable_names=[variable.name for variable in context.variables],
            allowed_literature_citation_keys=[],
            primary_exposure="exposure_flag",
            target_outcome="outcome_flag",
            require_design_selection=True,
            article_context=context,
        )


@pytest.mark.parametrize(
    "display_labels",
    [
        [],
        [{"key": "exposure_flag=0", "value": "Exposure absent"}],
        [
            {"key": "exposure_flag=0", "value": "Exposure"},
            {"key": "exposure_flag=1", "value": " exposure "},
        ],
    ],
)
def test_foundation_requires_complete_distinct_binary_figure_labels(
    display_labels: list[dict[str, str]],
) -> None:
    foundation = ProgressivePlanFoundation.model_validate(
        {
            "cohort": _payload()["cohort"],
            "display_labels": display_labels,
            "robustness_intents": [],
            "know_how_decisions": [],
        }
    )

    with pytest.raises(ProgressivePlanCompileError) as caught:
        validate_progressive_foundation(
            foundation,
            context=_context(),
            analysis_type="descriptive_epidemiology",
            required_binary_display_label_scopes=("exposure_flag",),
        )

    assert caught.value.reason_code == (
        "progressive_required_binary_display_labels_missing"
    )
    assert caught.value.path == "display_labels"


def test_foundation_accepts_complete_distinct_binary_figure_labels() -> None:
    foundation = ProgressivePlanFoundation.model_validate(
        {
            "cohort": _payload()["cohort"],
            "display_labels": _payload()["display_labels"],
            "robustness_intents": [],
            "know_how_decisions": [],
        }
    )

    validate_progressive_foundation(
        foundation,
        context=_context(),
        analysis_type="descriptive_epidemiology",
        required_binary_display_label_scopes=("exposure_flag",),
    )


def test_foundation_requires_selected_design_reader_labels() -> None:
    outline = ProgressivePlanOutline.model_validate(_outline_payload())
    required_keys = required_reader_display_label_keys(_context(), outline.design_selection)
    foundation_payload = {
        "cohort": _payload()["cohort"],
        "display_labels": [
            {"key": "exposure_flag=0", "value": "Exposure absent"},
            {"key": "exposure_flag=1", "value": "Exposure present"},
        ],
        "robustness_intents": [],
        "know_how_decisions": [],
    }

    with pytest.raises(ProgressivePlanCompileError) as caught:
        validate_progressive_foundation(
            ProgressivePlanFoundation.model_validate(foundation_payload),
            context=_context(),
            analysis_type="descriptive_epidemiology",
            required_reader_display_label_keys=required_keys,
        )

    assert caught.value.reason_code == (
        "progressive_required_reader_display_labels_missing"
    )
    assert caught.value.details["required_key"] == "exposure_flag"

    foundation_payload["display_labels"] = [
        *foundation_payload["display_labels"],
        {"key": "exposure_flag", "value": "Exposure status"},
        {"key": "outcome_flag", "value": "In-hospital outcome"},
    ]
    validate_progressive_foundation(
        ProgressivePlanFoundation.model_validate(foundation_payload),
        context=_context(),
        analysis_type="descriptive_epidemiology",
        required_reader_display_label_keys=required_keys,
    )


def test_reader_display_label_keys_exclude_row_identifiers() -> None:
    from types import SimpleNamespace

    selection = SimpleNamespace(
        selected=SimpleNamespace(
            required_variables=["stay_id", "exposure_flag", "outcome_flag"]
        )
    )

    assert required_reader_display_label_keys(_context(), selection) == (
        "exposure_flag",
        "outcome_flag",
    )


def test_foundation_rejects_mechanically_humanized_reader_label() -> None:
    foundation = ProgressivePlanFoundation.model_validate(
        {
            "cohort": _payload()["cohort"],
            "display_labels": [
                {"key": "exposure_flag", "value": "exposure flag"},
                {"key": "outcome_flag", "value": "Clinical outcome"},
            ],
            "robustness_intents": [],
            "know_how_decisions": [],
        }
    )

    with pytest.raises(ProgressivePlanCompileError) as caught:
        validate_progressive_foundation(
            foundation,
            context=_context(),
            analysis_type="descriptive_epidemiology",
            required_reader_display_label_keys=("exposure_flag", "outcome_flag"),
        )

    assert caught.value.reason_code == (
        "progressive_required_reader_display_labels_missing"
    )
