"""An approvable candidate whose only gap the bound source closes is compiled.

A candidate planned before the study chose clustered inference is reviewed
without the source's patient grouping, so it carries the repeated-stay
limitation.  When the bound source groups stays, the workflow sends it to the
Host compiler (which persists the clustered design and plans afresh) instead of
offering the limited plan for approval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from easyicu.webserver import source_identity_authority
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.pi_copilot import workflow as workflow_owner
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
from tests.webserver.copilot.research_workflow_fixtures import complete_study

_RUN_ID = "run-candidate-repeated-stays"


@pytest.fixture(autouse=True)
def _fresh_grouping_cache():
    workflow_owner._source_groups_stays_at.cache_clear()
    yield
    workflow_owner._source_groups_stays_at.cache_clear()


def _study(tmp_path: Path, **changes: Any) -> dict[str, Any]:
    export = tmp_path / "export"
    export.mkdir(exist_ok=True)
    study = complete_study()
    study["data_source"] = {"path": str(export), "database": "eicu_demo"}
    study.update(changes)
    return study


def _plan() -> dict[str, Any]:
    return {
        "design_selection": {
            "candidates": [
                {
                    "design_id": "landmark_adjusted_association",
                    "disposition": "selected",
                }
            ]
        },
        "steps": [
            {
                "model_requirements": [
                    {
                        "analysis_role": "primary",
                        "exposure_source": "aki_stage",
                        "outcome": "death",
                        "covariates": ["age", "sex"],
                        "covariate_rationales": {
                            "age": "Baseline age precedes exposure.",
                            "sex": "Baseline sex precedes exposure.",
                        },
                        "covariate_temporal_roles": {
                            "age": "baseline_static",
                            "sex": "baseline_static",
                        },
                    }
                ]
            }
        ],
    }


def _candidate(study: dict[str, Any], runtime_codes: list[str]):
    digest = study_context_owner.scientific_configuration_sha256(study)
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_type": "full",
            "run_id": _RUN_ID,
            "study_id": study["id"],
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "gate_reason": "human_plan_review_required",
            "run_status": "human_review_pending",
            "pending_review_reason_codes": ["operator_plan_approval_required"],
            "scientific_configuration_sha256": digest,
            "artifact_names": [
                "agent_plan.json",
                "scientific_plan_review.json",
                "source_run_manifest.json",
            ],
        },
        plan_review_authority={
            "run_id": _RUN_ID,
            "resumable_here": True,
            "scientific_configuration_sha256": digest,
            "budget_mode": "planner_canary",
            "research_input_state": "metadata_only",
            "plan_approval_allowed": True,
            "scientific_plan_review": {
                "status": "analysis_only",
                "approval_allowed": True,
                "findings": [],
                "facts": {
                    "remediation_buckets": {
                        "agent_plan_revision": [],
                        "runtime_capability": runtime_codes,
                        "study_authority_change": [],
                        "external_evidence": [],
                        "independent_review": [],
                    }
                },
            },
        },
    )
    assert snapshot.next_action_code == "plan_execution_upgrade_required"
    return snapshot


def _enriched(study: dict[str, Any], runtime_codes: list[str]):
    return workflow_owner._enrich_plan_review(
        _candidate(study, runtime_codes),
        study=study,
        review={"artifact_payloads": {"agent_plan.json": _plan()}},
    )


def _grouping(monkeypatch: pytest.MonkeyPatch, result: Any) -> list[dict]:
    calls: list[dict] = []

    def resolve(**kwargs: Any) -> Any:
        calls.append(kwargs)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(
        source_identity_authority, "resolve_study_patient_grouping", resolve
    )
    return calls


def test_a_grouped_source_sends_the_limited_candidate_to_the_host_compiler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path)
    calls = _grouping(monkeypatch, object())

    enriched = _enriched(study, ["REPEATED_STAY_IDENTITY_UNAVAILABLE"])

    assert enriched.next_action_code == "agent_plan_configuration_required"
    assert calls == [
        {"export_path": study["data_source"]["path"], "database": "eicu_demo"}
    ]


@pytest.mark.parametrize(
    "grouping",
    [
        None,
        source_identity_authority.PatientGroupingAuthorityError(
            "patient_grouping_authority_invalid", "broken bridge"
        ),
    ],
    ids=["no_grouping", "grouping_refused"],
)
def test_a_source_without_verified_grouping_keeps_the_approvable_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, grouping: Any
) -> None:
    _grouping(monkeypatch, grouping)

    enriched = _enriched(_study(tmp_path), ["REPEATED_STAY_IDENTITY_UNAVAILABLE"])

    assert enriched.next_action_code == "plan_execution_upgrade_required"


@pytest.mark.parametrize(
    ("changes", "runtime_codes"),
    [
        (
            {
                "analysis_design": {
                    "analysis_unit": "icu_stay",
                    "variance_estimator": "cluster_robust",
                    "cluster_unit": "patient",
                }
            },
            ["REPEATED_STAY_IDENTITY_UNAVAILABLE"],
        ),
        (
            {"cohort": {"exclude_readmissions": True}},
            ["REPEATED_STAY_IDENTITY_UNAVAILABLE"],
        ),
        (
            {
                "analysis_design": {
                    "analysis_family": "survival",
                    "analysis_unit": "icu_stay",
                    "variance_estimator": "model_based",
                }
            },
            ["REPEATED_STAY_IDENTITY_UNAVAILABLE"],
        ),
        (
            {
                "analysis_design": {
                    "analysis_family": "trajectory_clustering",
                    "analysis_unit": "icu_stay",
                    "variance_estimator": "model_based",
                }
            },
            ["REPEATED_STAY_IDENTITY_UNAVAILABLE"],
        ),
        (
            {},
            [
                "REPEATED_STAY_IDENTITY_UNAVAILABLE",
                "PRIMARY_POPULATION_EXECUTION_OWNER_MISSING",
            ],
        ),
    ],
    ids=[
        "already_clustered",
        "first_stay_cohort",
        "survival_suite",
        "trajectory_suite",
        "another_runtime_gap",
    ],
)
def test_a_decided_study_or_an_unowned_gap_keeps_the_approvable_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, Any],
    runtime_codes: list[str],
) -> None:
    _grouping(monkeypatch, object())

    enriched = _enriched(_study(tmp_path, **changes), runtime_codes)

    assert enriched.next_action_code == "plan_execution_upgrade_required"


def test_a_polled_workflow_resolves_the_source_grouping_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path)
    calls = _grouping(monkeypatch, object())

    for _ in range(3):
        enriched = _enriched(study, ["REPEATED_STAY_IDENTITY_UNAVAILABLE"])
        assert enriched.next_action_code == "agent_plan_configuration_required"

    assert len(calls) == 1
