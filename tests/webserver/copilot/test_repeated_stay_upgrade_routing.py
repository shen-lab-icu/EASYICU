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
from tests.support.reviewed_candidate import reviewed_candidate
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


_candidate = reviewed_candidate


def _enriched(study: dict[str, Any], runtime_codes: list[str], run: dict[str, Any] | None = None,
              compiled_source_run_id: str | None = None):
    payloads: dict[str, Any] = {"agent_plan.json": _plan()}
    if compiled_source_run_id is not None:
        payloads["run_context.json"] = {"compiled_configuration_source_run_id": compiled_source_run_id}
    return workflow_owner._enrich_plan_review(
        _candidate(study, runtime_codes),
        study=study,
        review={"artifact_payloads": payloads},
        run=run,
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


@pytest.mark.parametrize("recorded", ["run-compiled-earlier", "", "another-run"],
                         ids=["continues_the_compiled_candidate", "lineage_broken", "another_candidate"])
def test_a_plan_that_continues_the_compiled_candidate_is_not_compiled_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: str
) -> None:
    # B10 (UX leg 3 en/zh, 2026-11-19): every compile rewrote the study and a
    # fresh plan came back with the same runtime gaps, so the host compiled
    # again without end. A plan whose launch recorded that it continues the
    # compiled candidate keeps its review's own action; without that record
    # (its lineage was broken) the compile stays available (Carbon b10 F1).
    from easyicu.webserver.pi_copilot import plan_review_progress

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    _grouping(monkeypatch, object())
    study = _study(tmp_path)
    earlier = {**study, "covariates": ["age"]}
    plan_review_progress.record_choice(
        before=earlier, after=study,
        run={"run_id": "run-compiled-earlier",
             "scientific_configuration_sha256": study_context_owner.scientific_configuration_sha256(earlier)},
        decision_code="agent_plan_configuration",
        option_id="typed_runtime_projection",
    )
    # Two job lifecycle writes since then keep the configuration.
    study = {**study, "revision": int(study.get("revision") or 1) + 2}
    run = {"run_id": _RUN_ID, "scientific_configuration_sha256": study_context_owner.scientific_configuration_sha256(study)}

    enriched = _enriched(study, ["REPEATED_STAY_IDENTITY_UNAVAILABLE"], run, recorded)

    assert enriched.next_action_code == (
        "plan_execution_upgrade_required" if recorded == "run-compiled-earlier" else "agent_plan_configuration_required"
    )
