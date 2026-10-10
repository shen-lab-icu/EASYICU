"""An accepted candidate carries the exposure groups its planning run formed.

Accepting a metadata-only candidate grants data preparation for exactly what
it planned.  When its planning run formed exposure groups
(``research_agent.orchestration.exposure_grouping_phase``), the grant carries
the grouping record, checked against the input the candidate's capsule
sealed: the package-bound run forms the same groups without asking, the
prepared data hold the concepts they read, and a baseline table may be
grouped by them.  A record that does not match the sealed input refuses the
grant.  The run that follows the candidate is configured to form its
groups without asking, and prepares the values they read.  Each stop at a
grouping says what stopped planning and what the researcher changes.
Synthetic StudyContext, capsule, zero-row input, export manifest and review
only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from easyicu.research_agent.contracts.trajectory_design import (
    load_trajectory_design,
    sealed_trajectory_authority_body,
)
import easyicu.research_agent as research_agent
from easyicu.research_agent.acquisition import foundation
from easyicu.research_agent.execution import runner as runner_module
from easyicu.research_agent.orchestration.exposure_grouping_phase import (
    EXPOSURE_GROUPING_STOPS,
    PlanningRunGroupings,
    run_exposure_grouping_phase,
)
from easyicu.research_agent.planning.exposure_group_compile import (
    CandidateExposureGroupings,
)
from easyicu.research_agent.planning.exposure_group_spec import (
    read_stated_exposure_groupings,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
)
from easyicu.research_agent.planning.scientific_review import PlanScientificReview
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.schema import AnalysisPlan
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    build_trajectory_scientific_runtime_authority,
)
from easyicu.webserver import agent_pipeline_runs, agent_runs, provider_adapter
from easyicu.webserver import research_pipeline_run_preparation
from easyicu.webserver import study_contexts as study_context_owner
from tests.webserver.copilot.research_workflow_fixtures import (
    _acquisition_receipt,
    _foundation_profile,
)

QUESTION = (
    "Which organ-dysfunction trajectory classes emerge over the first 72 h of "
    "an ICU stay from SOFA-2 components, and how does 28-day mortality differ "
    "by class and across lactate groups?"
)
COORDINATES = ["sofa2_cardio", "sofa2_renal", "sofa2_resp"]
OUTCOME = "mort_28d"
SOURCE_RUN_ID = "run-grouped-candidate"
_GROUPED = "lact_group_x1"
_ANSWER = json.dumps(
    {
        "groupings": [
            {
                "id": "x1",
                "concept": "lact",
                "window": {"start_hours": 0, "end_hours": 24},
                "scale": "nominal",
                "groups": [
                    {
                        "id": "g1",
                        "label": "lactate below 2.5",
                        "rule": {
                            "summary": "max",
                            "op": "<",
                            "value": 2.5,
                            "unit": "mmol/L",
                        },
                    },
                    {"id": "g2", "label": "lactate 2.5 or above", "rule": "otherwise"},
                ],
                "unmeasured": {"handling": "own_group", "label": "not measured"},
                "reference": "g1",
                "contrast": "g2",
                "quote": "lactate groups",
                "source": "question",
            }
        ]
    }
)


def _candidate_plan() -> dict[str, Any]:
    """A host-compiled approvable candidate with a table grouped by the groups."""

    design = load_trajectory_design({"coordinate_concepts": COORDINATES})
    authority = build_trajectory_scientific_runtime_authority(
        sealed_trajectory_authority_body(design, protocol_content_sha256="0" * 64)
    )
    owners = authority.development_execution_only_plan(research_question=QUESTION)
    bound, _findings = ScientificRuntimeAuthorities(
        trajectory=authority, current_case=None
    ).bind_plan(AnalysisPlan.model_validate(owners.model_dump(mode="json")))
    payload = bound.model_dump(mode="json")
    payload["steps"].append(
        {
            "step_id": "baseline_by_lactate_group",
            "planned_analysis_role": "secondary",
            "intent": "Describe the cardiovascular component by lactate group.",
            "method": "table_one",
            "inputs": [_GROUPED, "sofa2_cardio"],
            "expected_outputs": ["table:table_one"],
            "table_one_spec": {
                "group_by": _GROUPED,
                "group_levels": [1, 2, 3],
                "variables": [
                    {
                        "name": "sofa2_cardio",
                        "variable_kind": "continuous",
                        "summary": "median_iqr",
                        "test": "mann_whitney_or_kruskal",
                    }
                ],
            },
        }
    )
    return payload


def _study(workflow_complete_study: dict) -> dict:
    study = dict(workflow_complete_study)
    study.update(
        {
            "question": QUESTION,
            "primary_exposure": "",
            "covariates": [],
            "covariate_selection": "planner_selectable",
            "execution_concepts": {"outcome": OUTCOME},
            "cohort": {},
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


def _context(cohort: Path):
    return build_research_context(
        research_question=QUESTION,
        cohort=cohort,
        cohort_name="candidate",
        database="miiv",
        target_outcome=OUTCOME,
        id_columns=("stay_id",),
        outcome_columns=(OUTCOME,),
        user_preferences={
            "data_constraints": json.dumps(
                {
                    "materialization_window": {
                        "role": "outer_observation_window",
                        "anchor": "icu_admission",
                        "hours": 24,
                    }
                }
            )
        },
    )


def _planning_run(inner_run: Path, catalog: list[str]) -> str:
    """The candidate's run: its empty input, with the groups it formed declared."""

    frame = pd.DataFrame(
        {
            name: pd.Series(dtype="int64" if name == "stay_id" else "float64")
            for name in catalog
        }
    )
    frame.attrs["easyicu_planning_authority"] = {
        "kind": "metadata_only_planning_catalog",
        "patient_rows_read": False,
    }
    cohort = inner_run / "cohort.parquet"
    frame.to_parquet(cohort, index=False)
    run_exposure_grouping_phase(
        context=_context(cohort),
        cohort_path=cohort,
        run_dir=inner_run,
        planner=ScriptedMockLLMClient([_ANSWER]),
        rebuild_context=_context,
        capability_review_pending=False,
        trajectory_staged=False,
        emit_progress=lambda *_args, **_kwargs: None,
    )
    return hashlib.sha256(cohort.read_bytes()).hexdigest()


def _load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    study: dict,
    tamper: bool = False,
):
    project_dir = tmp_path / "candidate-wrapper"
    inner_run = project_dir / "pipeline" / SOURCE_RUN_ID
    inner_run.mkdir(parents=True)
    catalog = ["stay_id", OUTCOME, *COORDINATES, "lact"]
    cohort_sha256 = _planning_run(inner_run, catalog)
    if tamper:
        record_path = inner_run / "exposure_groupings.json"
        record = json.loads(record_path.read_text())
        record["compiled"]["groupings"][0]["derivation_sha256"] = "1" * 64
        record_path.write_text(json.dumps(record), encoding="utf-8")
    capsule_raw = json.dumps(
        {
            "scientific_identity": {
                "question": QUESTION,
                "database": "miiv",
                "primary_exposure": None,
                "target_outcome": OUTCOME,
                "user_preferences": {},
            },
            "cohort_sha256": cohort_sha256,
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
        generated_at="2026-10-09T00:00:00Z",
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
                "agent_plan.json": _candidate_plan(),
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


def test_the_grant_carries_the_groups_the_candidate_formed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workflow_complete_study: dict,
) -> None:
    authority = _load(tmp_path, monkeypatch, study=_study(workflow_complete_study))

    assert authority is not None
    groupings = authority.exposure_groupings
    assert groupings is not None
    assert (groupings.variables, groupings.concepts) == ((_GROUPED,), ("lact",))
    assert list(groupings.candidate.derivations) == ["x1"]
    # A baseline table grouped by the study's groups is part of the grant.
    (table,) = [
        table
        for table in authority.baseline_requirements.tables
        if table.source_step_id == "baseline_by_lactate_group"
    ]
    assert table.group_by.name == _GROUPED
    assert table.group_by.source_concept is None


def test_a_record_that_does_not_match_the_sealed_input_refuses_the_grant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workflow_complete_study: dict,
) -> None:
    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as raised:
        _load(
            tmp_path,
            monkeypatch,
            study=_study(workflow_complete_study),
            tamper=True,
        )

    assert raised.value.code == "candidate_plan_materialization_authority_invalid"
    assert raised.value.details["field"] == "exposure_groupings"


@pytest.mark.parametrize(
    "code",
    [
        *EXPOSURE_GROUPING_STOPS,
        # A template's own refusal, as the family planner projects it.
        "progressive_family_spec_exposure_group_contrast_unavailable",
    ],
)
def test_a_stop_at_the_groupings_says_what_to_change(code: str) -> None:
    fallback = agent_pipeline_runs._progressive_compile_failure_message(
        ProgressivePlanCompileError("unrelated_planning_stop", "detail")
    )

    message = agent_pipeline_runs._progressive_compile_failure_message(
        ProgressivePlanCompileError(code, "lact_group_x1 reads lact_max")
    )

    assert message != fallback
    assert "no analysis was run" in message
    # The compiler's message may name the study's values; it stays out.
    assert "lact_max" not in message


class _Captured(BaseException):
    """The pipeline was handed its run: nothing past it is this test's."""


_PROVIDER_ENVIRONMENT = {
    "OPENAI_API_KEY": "test-private-provider-key",
    "OPENAI_BASE_URL": "http://127.0.0.1:8317/v1",
    "OPENAI_MODEL": "test-local-model",
    "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
}


def _export(root: Path) -> Path:
    """A prepared export package: one demographics file and its manifest."""

    root.mkdir(parents=True)
    pd.DataFrame({"stay_id": [1], "age": [65]}).to_parquet(
        root / "demographics.parquet", index=False
    )
    (root / "_manifest.json").write_text(
        json.dumps(
            {
                "database": "miiv",
                "format": "parquet",
                "concept_selection": {
                    "mode": "explicit",
                    "modules": {"demographics": ["age"]},
                },
                "feature_definitions": {"included": False},
                "files": [
                    {
                        "file": "demographics.parquet",
                        "module": "demographics",
                        "concepts": 1,
                        "concept_ids": ["age"],
                        "rows": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def _grouped_candidate() -> PlanningRunGroupings:
    stated = read_stated_exposure_groupings(json.loads(_ANSWER))
    return PlanningRunGroupings(
        candidate=CandidateExposureGroupings(
            record_sha256="b" * 64,
            stated=stated.model_dump(mode="json"),
            derivations={"x1": "c" * 64},
        ),
        variables=(_GROUPED,),
        concepts=("lact",),
    )


def test_the_run_that_follows_the_candidate_forms_its_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workflow_complete_study: dict,
) -> None:
    groupings = _grouped_candidate()
    authority = agent_pipeline_runs._CandidatePlanMaterializationAuthority(
        primary_exposure="heart_rate",
        target_outcome="death",
        outcome_concepts=("death",),
        contract="Compare death across the candidate's lactate groups.",
        primary_cohort_selection_mode="all_input_rows",
        exposure_groupings=groupings,
    )
    monkeypatch.setattr(
        agent_pipeline_runs,
        "_load_candidate_plan_materialization_authority",
        lambda **_kwargs: authority,
    )
    profiles: list[dict[str, Any]] = []

    def profile(**kwargs: Any) -> dict[str, Any]:
        profiles.append(kwargs)
        return _foundation_profile()

    monkeypatch.setattr(agent_pipeline_runs, "_data_foundation_profile", profile)
    monkeypatch.setattr(
        research_pipeline_run_preparation,
        "_data_foundation_profile",
        lambda **_kwargs: _foundation_profile(),
    )
    universe = tmp_path / "universe.parquet"
    universe.write_bytes(b"typed-universe-placeholder")
    acquisition = _acquisition_receipt()
    acquisition.blocked = False
    acquisition.universe_path = universe
    acquisition.cohort_authority_path = None
    acquisition.cohort_authority_ref = None
    acquisition.trajectory_path = None
    acquisition.trajectory_authority_path = None
    acquisition.trajectory_authority_ref = None
    monkeypatch.setattr(
        foundation, "acquire_universe_for_question", lambda **_kwargs: acquisition
    )
    monkeypatch.setattr(
        runner_module,
        "probe_runner_availability",
        lambda kind, **_kwargs: runner_module.RunnerAvailability(
            kind=kind, available=True, image="easyicu-research-agent:test"
        ),
    )
    monkeypatch.setattr(
        provider_adapter,
        "build_research_agent_provider_client",
        lambda *_args, **_kwargs: (object(), {"provider": "openai", "model": "test"}),
    )
    captured: dict[str, Any] = {}

    class FakePipeline:
        def run(self, **_kwargs: Any) -> Any:
            raise _Captured()

    def from_config(config: Any, *, services: Any) -> FakePipeline:
        captured["config"] = config
        return FakePipeline()

    monkeypatch.setattr(
        research_agent.ResearchAgentPipeline, "from_config", from_config
    )
    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(_export(tmp_path / "export")),
        study_context=workflow_complete_study,
        project_root=str(tmp_path / "projects"),
        provider={"provider": "openai", "external": True},
        provider_environment=_PROVIDER_ENVIRONMENT,
        budget_mode="full_reviewed",
        plan_revision_source_run_id=SOURCE_RUN_ID,
    )

    with pytest.raises(_Captured):
        runner(
            SimpleNamespace(
                id="job-grouped", cancel_requested=False, emit=lambda _e: None
            )
        )

    # The prepared data hold the value its groups are formed from ...
    (prepared,) = [item for item in profiles if item.get("require_primary_exposure")]
    assert "lact" in prepared["analysis_inputs"]
    # ... and the run forms the candidate's groups without asking.
    config = captured["config"]
    assert config.enable_exposure_grouping is True
    assert (
        CandidateExposureGroupings.model_validate(config.bound_exposure_groupings)
        == groupings.candidate
    )


@pytest.mark.parametrize(
    ("planning", "continuation", "candidate", "asks", "binds"),
    [
        pytest.param(True, False, False, True, False, id="a-planning-run-asks"),
        pytest.param(True, True, False, None, False, id="a-continuation-replays"),
        pytest.param(False, False, True, True, True, id="the-candidates-run-binds"),
        pytest.param(False, False, False, None, False, id="another-run-plans-none"),
    ],
)
def test_a_run_asks_for_groups_only_when_it_plans_them(
    planning: bool,
    continuation: bool,
    candidate: bool,
    asks: Any,
    binds: bool,
) -> None:
    groupings = _grouped_candidate() if candidate else None

    config = agent_pipeline_runs._exposure_grouping_config(
        metadata_only_planning=planning,
        development_continuation=continuation,
        candidate_groupings=groupings,
    )

    assert config["enable_exposure_grouping"] is asks
    assert (config["bound_exposure_groupings"] is not None) is binds
