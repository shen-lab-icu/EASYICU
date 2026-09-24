"""An accepted candidate's primary-analysis inputs survive metadata-to-data planning.

Dev9 M3 (Sepsis-3 phenotypes) was reviewed with ten clustering features
(``hr``, ``wbc``, ...).  After materialization each became eight columns and
the package-bound roster, ranked by the Planner's audit-heavy notes, kept only
measurement-process companions, so the phenotyping family was refused before
any Provider call ("needs selectable feature candidates").
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.agents.progressive_planner import select_progressive_variables
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.planning.accepted_analysis_inputs import (
    analysis_input_value_columns,
    bind_analysis_inputs,
    candidate_analysis_inputs,
    context_analysis_inputs,
)
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.schema import ConceptDescriptor

from .scientific_review_fixtures import _context


_SHA = "a" * 64
_FEATURES = ("hr", "wbc", "crea")
_VALUE_SUFFIXES = ("max", "min", "mean", "first")
_PROCESS_SUFFIXES = (("n", "meta"), ("measured", "meta"), ("first_time", "time"), ("last_time", "time"))


def _plan(*inputs: str) -> dict:
    return {
        "steps": [
            {"step_id": "cohort", "planned_analysis_role": "auxiliary", "inputs": ["age", "lact"]},
            {
                "step_id": "primary_cluster_solution",
                "planned_analysis_role": "primary",
                "inputs": [*inputs, "artifact:analysis_cohort"],
            },
        ]
    }


def _materialized_context(*concepts: str, notes: str = ""):
    context = _context()
    variables = [*context.variables]
    for concept in concepts:
        role = "lab" if concept != "hr" else "vital"
        variables += [
            ConceptDescriptor(name=f"{concept}_{suffix}", source_concept=concept, role=role, dtype="float64")
            for suffix in _VALUE_SUFFIXES
        ]
        variables += [
            ConceptDescriptor(
                name=f"{concept}_{suffix}", source_concept=concept, role=role_name, dtype="float64",
                description=f"{concept} measurement process audit: count, measured status, observation timing",
            )
            for suffix, role_name in _PROCESS_SUFFIXES
        ]
    return context.model_copy(update={"variables": variables, "notes": notes})


def _bound(context, *concepts: str):
    accepted = candidate_analysis_inputs(
        plan=_plan(*concepts), source_plan_sha256=_SHA,
        selected_concepts=concepts, excluded=(),
    )
    return bind_analysis_inputs(context, accepted.model_dump(mode="json"))


def test_candidate_inputs_are_the_primary_steps_catalog_concepts() -> None:
    accepted = candidate_analysis_inputs(
        plan=_plan("sep3", "hr", "wbc", "hr_max", "death", "stay_id"),
        source_plan_sha256=_SHA,
        selected_concepts=("sep3", "hr", "wbc", "death", "age", "lact"),
        excluded=("sep3", "death", "stay_id", ""),
    )

    assert accepted is not None
    assert accepted.concepts == ("hr", "wbc")
    assert accepted.source_step_ids == ("primary_cluster_solution",)


def test_a_candidate_without_primary_catalog_inputs_issues_nothing() -> None:
    assert candidate_analysis_inputs(
        plan=_plan("bili_max"), source_plan_sha256=_SHA,
        selected_concepts=("bili",), excluded=(),
    ) is None


def test_binding_is_sealed_once_and_a_resume_cannot_drift() -> None:
    context = _bound(_materialized_context(*_FEATURES), *_FEATURES)

    assert context_analysis_inputs(context).concepts == _FEATURES
    with pytest.raises(ValueError, match="accepted_analysis_inputs_binding_drift"):
        bind_analysis_inputs(context, None, restoring=True)
    assert bind_analysis_inputs(
        context, context_analysis_inputs(context).model_dump(mode="json"), restoring=True,
    ) is context


def test_value_columns_come_from_source_and_role_metadata_not_suffixes() -> None:
    context = _bound(_materialized_context(*_FEATURES), *_FEATURES)

    columns = analysis_input_value_columns(context)

    assert set(columns["wbc"]) == {f"wbc_{suffix}" for suffix in _VALUE_SUFFIXES}
    assert not {"wbc_n", "wbc_measured", "wbc_first_time", "wbc_last_time"} & set(columns["wbc"])


def test_audit_heavy_notes_cannot_crowd_an_accepted_input_out_of_the_roster() -> None:
    notes = "measurement process audit: count, measured status, observation timing " * 5
    crowded = _materialized_context(*_FEATURES, "na", "k", "cl", "ph", "ast", "alt", notes=notes)
    unbound = set(select_progressive_variables(crowded, max_variables=24))
    assert any(
        not unbound & {f"{concept}_{suffix}" for suffix in _VALUE_SUFFIXES}
        for concept in _FEATURES
    )

    selected = set(select_progressive_variables(_bound(crowded, *_FEATURES), max_variables=24))

    accepted = {f"{concept}_{suffix}" for concept in _FEATURES for suffix in _VALUE_SUFFIXES}
    assert accepted <= selected
    assert len(selected) <= 24 + len(accepted)


def test_a_lost_accepted_input_fails_before_the_provider() -> None:
    context = _bound(_materialized_context("hr"), "hr", "wbc")

    with pytest.raises(ProgressivePlanCompileError) as caught:
        select_progressive_variables(context)

    assert caught.value.reason_code == "progressive_accepted_input_unavailable"
    assert "wbc" in str(caught.value)


def test_accepted_inputs_do_not_consume_the_optional_retrieval_budget() -> None:
    context = _bound(_materialized_context(*_FEATURES), *_FEATURES)
    accepted = {f"{concept}_{suffix}" for concept in _FEATURES for suffix in _VALUE_SUFFIXES}

    selected = set(select_progressive_variables(context, max_variables=6))

    assert accepted <= selected
    assert len(selected) == 6 + len(accepted)
    assert {context.primary_exposure, context.target_outcome} <= selected


def test_pipeline_config_validates_the_bound_inputs_and_keeps_old_digests(tmp_path) -> None:
    accepted = candidate_analysis_inputs(
        plan=_plan("hr"), source_plan_sha256=_SHA, selected_concepts=("hr",), excluded=(),
    ).model_dump(mode="json")

    with pytest.raises(ValueError, match="requires require_human_plan_review"):
        PipelineConfig(workdir=tmp_path, bound_analysis_inputs=accepted)
    bound = PipelineConfig(
        workdir=tmp_path, require_human_plan_review=True, bound_analysis_inputs=accepted,
    )
    unbound = PipelineConfig(workdir=tmp_path, require_human_plan_review=True)

    assert "bound_analysis_inputs" in bound.canonical_payload()
    assert "bound_analysis_inputs" not in unbound.canonical_payload()
    restored = PipelineConfig.from_recovery_payload(
        bound.recovery_payload(), expected_digest=bound.canonical_digest(),
    )
    assert restored.bound_analysis_inputs == bound.bound_analysis_inputs
    with pytest.raises(ValueError):
        PipelineConfig(
            workdir=tmp_path, require_human_plan_review=True,
            bound_analysis_inputs={**accepted, "concepts": ["hr", "hr"]},
        )
