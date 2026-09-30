"""A signed trajectory candidate can prepare its data package.

Accepting a metadata-only candidate grants data preparation only when the
candidate states an explicit population that matches the study's stated
scope.  The host-compiled signed trajectory plan used to state none, so a
reviewed, approvable trajectory candidate was refused before any row was
read ("no valid population authority").  The owner now states the population
its owners analyze.  Synthetic StudyContext, capsule, zero-row catalog and
review only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from easyicu.research_agent.contracts.trajectory_design import (
    load_trajectory_design,
    sealed_trajectory_authority_body,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.planning.scientific_review import PlanScientificReview
from easyicu.research_agent.schema import AnalysisPlan
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    build_trajectory_scientific_runtime_authority,
)
from easyicu.webserver import agent_pipeline_runs, agent_runs
from easyicu.webserver import study_contexts as study_context_owner

QUESTION = (
    "Which organ-dysfunction trajectory classes emerge over the first 72 h of "
    "an ICU stay from SOFA-2 components, and how does 28-day mortality differ "
    "by class?"
)
COORDINATES = ["sofa2_cardio", "sofa2_renal", "sofa2_resp"]
OUTCOME = "mort_28d"
SOURCE_RUN_ID = "run-signed-trajectory-candidate"


def _signed_candidate_plan(*, population: bool = True) -> dict[str, Any]:
    """The host-compiled signed plan with its frozen-class description."""

    design = load_trajectory_design({"coordinate_concepts": COORDINATES})
    authority = build_trajectory_scientific_runtime_authority(
        sealed_trajectory_authority_body(design, protocol_content_sha256="0" * 64)
    )
    owners = authority.development_execution_only_plan(research_question=QUESTION)
    draft = owners.model_dump(mode="json")
    draft["steps"].append(
        {
            "step_id": "outcome_by_class",
            "planned_analysis_role": "secondary",
            "intent": "Describe 28-day mortality by frozen class.",
            "method": "descriptive_outcome_by_cluster",
            "scientific_action_id": "phenotyping.outcome_by_cluster",
            "inputs": [
                "stay_id",
                OUTCOME,
                "artifact:analysis_cohort",
                "table:cluster_assignments",
                "artifact:stability_freeze",
            ],
            "expected_outputs": ["table:outcome_by_cluster"],
            "phenotype_comparison_spec": {
                "identity_column": "stay_id",
                "outcome_columns": [OUTCOME],
                "variables": [
                    {
                        "name": OUTCOME,
                        "variable_kind": "categorical",
                        "summary": "count_percent",
                        "test": "none_descriptive_smd_only",
                        "levels": [0, 1],
                    }
                ],
            },
        }
    )
    bound, findings = ScientificRuntimeAuthorities(
        trajectory=authority, current_case=None
    ).bind_plan(AnalysisPlan.model_validate(draft))
    assert findings[0].detail["frozen_class_description"]["carried"] is True
    payload = bound.model_dump(mode="json")
    if not population:
        # The plan the host compiled before its owner stated a population.
        payload["cohort"] = None
    return payload


def _study(workflow_complete_study: dict, *, cohort: dict) -> dict:
    study = dict(workflow_complete_study)
    study.update(
        {
            "question": QUESTION,
            "primary_exposure": "",
            "covariates": [],
            "covariate_selection": "planner_selectable",
            "execution_concepts": {"outcome": OUTCOME},
            "cohort": cohort,
            "analysis_design": {
                "analysis_family": "trajectory_clustering",
                "analysis_unit": "icu_stay",
                "variance_estimator": "model_based",
            },
            "trajectory_design": study_context_owner.normalize_trajectory_design(
                {"coordinate_concepts": COORDINATES}
            ),
        }
    )
    return study


def _load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    study: dict,
    plan: dict[str, Any],
):
    project_dir = tmp_path / "candidate-wrapper"
    inner_run = project_dir / "pipeline" / SOURCE_RUN_ID
    inner_run.mkdir(parents=True)
    capsule_raw = json.dumps(
        {
            "scientific_identity": {
                "question": QUESTION,
                "database": "miiv",
                "primary_exposure": None,
                "target_outcome": OUTCOME,
                "user_preferences": {},
            }
        }
    ).encode()
    (inner_run / "run_input_capsule.json").write_bytes(capsule_raw)
    (inner_run / "human_review_checkpoint.json").write_text(
        json.dumps(
            {
                "state": "pending",
                "run_input_capsule_sha256": hashlib.sha256(capsule_raw).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    pipeline_input = project_dir / "pipeline_input"
    pipeline_input.mkdir()
    catalog = ["stay_id", OUTCOME, *COORDINATES]
    pd.DataFrame(columns=catalog).to_parquet(
        pipeline_input / "planner_catalog.parquet", index=False
    )
    (pipeline_input / "planner_catalog_receipt.json").write_text(
        json.dumps({"selected_concepts": catalog}), encoding="utf-8"
    )
    review = PlanScientificReview(
        status="analysis_only",
        approval_allowed=True,
        top_journal_candidate=False,
        score=89,
        dimension_scores={"study_design": 89},
        findings=[],
        facts={"requested_outcomes": [OUTCOME]},
        context_sha256="a" * 64,
        plan_sha256="b" * 64,
        literature_sha256="c" * 64,
        figure_strategy_sha256="d" * 64,
        generated_at="2026-09-30T00:00:00Z",
    ).model_dump(mode="json")
    monkeypatch.setattr(
        agent_runs,
        "list_run_history",
        lambda **_kwargs: {
            "runs": [
                {
                    "run_id": SOURCE_RUN_ID,
                    "project_dir": str(project_dir),
                    "scientific_configuration_sha256": (
                        study_context_owner.scientific_configuration_sha256(study)
                    ),
                }
            ]
        },
    )
    monkeypatch.setattr(
        agent_runs,
        "read_run_review",
        lambda _project_dir: {
            "ok": True,
            "artifact_payloads": {
                "scientific_plan_review.json": review,
                "agent_plan.json": plan,
            },
        },
    )
    monkeypatch.setattr(
        agent_pipeline_runs,
        "_metadata_only_planning_coordinates",
        lambda **_kwargs: {"primary_exposure": None, "target_outcome": OUTCOME},
    )
    return agent_pipeline_runs._load_candidate_plan_materialization_authority(
        study=study,
        project_root=str(tmp_path),
        source_run_id=SOURCE_RUN_ID,
        database="miiv",
        covariates=(),
    )


def test_a_reviewed_signed_trajectory_candidate_can_prepare_its_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workflow_complete_study: dict,
) -> None:
    authority = _load(
        tmp_path,
        monkeypatch,
        study=_study(workflow_complete_study, cohort={}),
        plan=_signed_candidate_plan(),
    )

    assert authority is not None
    # The package-bound run is then held to the same all-row population.
    assert authority.primary_cohort_selection_mode == "all_input_rows"
    assert authority.target_outcome == OUTCOME
    assert authority.outcome_concepts == (OUTCOME,)
    assert "05_frozen_class_description" in authority.contract


def test_a_signed_candidate_without_a_population_is_still_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workflow_complete_study: dict,
) -> None:
    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as raised:
        _load(
            tmp_path,
            monkeypatch,
            study=_study(workflow_complete_study, cohort={}),
            plan=_signed_candidate_plan(population=False),
        )
    assert raised.value.code == "candidate_plan_materialization_authority_invalid"
    assert raised.value.details["field"] == "cohort"


def test_a_stated_population_filter_is_not_granted_by_the_all_row_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workflow_complete_study: dict,
) -> None:
    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as raised:
        _load(
            tmp_path,
            monkeypatch,
            study=_study(workflow_complete_study, cohort={"preset": "adult_first"}),
            plan=_signed_candidate_plan(),
        )
    assert raised.value.code == "candidate_plan_materialization_authority_invalid"
    assert "stated population mode" in raised.value.details["cause"]
