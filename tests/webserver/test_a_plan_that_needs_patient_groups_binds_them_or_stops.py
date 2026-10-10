"""A plan that needs patient groups binds the source's grouping, or stops typed.

A step needs patient groups when it cannot run without them: the static
prediction owner splits development and validation rows by patient, and a
model requirement that carries within-patient dependence fits patient
clusters (``contracts.patient_grouping_need``, the rule the plan review reads
too).  A study whose declared inference reads no patient groups bound no
grouping at execution, so such a step reached its executor without the
authority and failed there, after approval and spend, with an untyped error.
A package-bound run now binds the source's verified grouping for an accepted
step that needs it, and states within-patient dependence only for a design
whose inference reads the grouping.  A study that also declares a design
read from the long trajectory stops before materialization, because a grouped
materialization omits the trajectory.  An approval whose plan needs groups
that its planned data does not hold stops before any step runs; a source
without a grouping is left to that check, since the materialized rows may
still carry a direct patient identifier.  A metadata-only planning context,
which binds no grouping unless its design reads one, states what the source
can provide, decided by the same resolver and rule as the binding.

Synthetic studies, plans and sources; no benchmark item.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_args

import pandas as pd
import pytest

from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
from easyicu.research_agent.contracts.patient_grouping_need import (
    PATIENT_GROUPING_AUTHORITY_ERROR_KEY,
    PATIENT_GROUPING_STATUS_KEY,
    PATIENT_GROUPING_STATUSES,
    PatientGroupingStatus,
    steps_needing_patient_groups,
)
from easyicu.research_agent.orchestration.workflow import (
    HumanReviewPending,
    HumanReviewRequest,
)
from easyicu.research_agent.planning.scientific_review import PlanScientificReview
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)
from easyicu.webserver import (
    agent_pipeline_runs,
    agent_runs,
    dataio,
    research_launch_scientific,
    run_patient_grouping,
)
from easyicu.webserver import study_contexts as study_context_owner
from tests.support.node import run_node
from tests.webserver.copilot.research_workflow_fixtures import (
    _acquisition_receipt,
    complete_study,
)

_PREDICTION_PRIMARY = {
    "step_id": "primary_performance",
    "intent": "Fit and evaluate the prespecified prediction model.",
    "method": "logistic_prediction",
    "scientific_action_id": "prediction.discrimination_calibration",
    "planned_analysis_role": "primary",
    "inputs": ["hr", "age", "death", "cohort:analysis_set"],
    "expected_outputs": ["table:prediction_scores", "table:model_performance"],
}
_DESCRIPTIVE = {
    "step_id": "baseline_context",
    "intent": "Describe the cohort.",
    "method": "descriptive",
    "planned_analysis_role": "auxiliary",
    "inputs": ["age"],
    "expected_outputs": ["table:baseline"],
}
_GROUPING = SimpleNamespace(output_identity_column="patient_stay_id")


def _refused(call: Any) -> Any:
    with pytest.raises(run_patient_grouping.ResearchPipelineRunError) as caught:
        call()
    return caught.value


# --- which steps need patient groups ---------------------------------------------


def test_a_static_prediction_primary_needs_patient_groups() -> None:
    plan = {"steps": [_DESCRIPTIVE, _PREDICTION_PRIMARY]}

    assert steps_needing_patient_groups(plan) == ("primary_performance",)


def test_a_model_that_carries_patient_dependence_needs_patient_groups() -> None:
    clustered = AnalysisStep.model_construct(
        step_id="clustered_model",
        model_requirements=[
            SimpleNamespace(
                dependence=PlannedDependenceRequirement(
                    group_source="patient_stay_id",
                    group_derivation="prefix_before_delimiter",
                    delimiter=":s",
                )
            )
        ],
    )
    independent = AnalysisStep.model_construct(
        step_id="independent_model",
        model_requirements=[SimpleNamespace(dependence=None)],
    )

    assert steps_needing_patient_groups(
        SimpleNamespace(steps=[independent, clustered])
    ) == ("clustered_model",)


def test_a_step_without_a_patient_split_or_dependence_needs_none() -> None:
    # A secondary prediction step reads the primary's scores; it does not refit.
    secondary = {
        **_PREDICTION_PRIMARY,
        "step_id": "calibration_metrics",
        "scientific_action_id": "prediction.calibration_metrics",
        "planned_analysis_role": "secondary",
        "inputs": ["table:prediction_scores"],
        "expected_outputs": ["table:calibration"],
    }
    malformed = {"step_id": "not_a_step", "inputs": "age"}

    assert (
        steps_needing_patient_groups({"steps": [_DESCRIPTIVE, secondary, malformed]})
        == ()
    )
    assert steps_needing_patient_groups({}) == ()


# --- planning: what the source can provide -------------------------------------


def _forbidden(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("not consulted")


@pytest.mark.parametrize(
    ("grouping", "longitudinal", "expected"),
    [
        (_GROUPING, None, "available_unbound"),
        (_GROUPING, "trajectory", "not_carried_by_trajectory"),
        (None, None, "source_has_none"),
    ],
)
def test_planning_states_what_the_source_can_provide(
    monkeypatch: pytest.MonkeyPatch,
    grouping: Any,
    longitudinal: Any,
    expected: str,
) -> None:
    monkeypatch.setattr(
        run_patient_grouping, "verified_patient_grouping", lambda _study: grouping
    )
    monkeypatch.setattr(
        run_patient_grouping,
        "declared_longitudinal_design",
        lambda _study: longitudinal if grouping is not None else _forbidden(),
    )

    assert run_patient_grouping.planning_patient_grouping_status({}, bound=None) == (
        expected,
        None,
    )


def test_a_context_that_binds_its_grouping_states_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_patient_grouping, "verified_patient_grouping", _forbidden)

    assert run_patient_grouping.planning_patient_grouping_status(
        {}, bound=_GROUPING
    ) == ("bound", None)


def test_an_authority_that_fails_its_checks_does_not_stop_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Planning may need no groups; execution refuses with this code if it binds.
    def broken(_study: Any) -> Any:
        raise run_patient_grouping.ResearchPipelineRunError(
            "patient_grouping_authority_mapping_mismatch", "The bridge changed."
        )

    monkeypatch.setattr(run_patient_grouping, "verified_patient_grouping", broken)

    assert run_patient_grouping.planning_patient_grouping_status({}, bound=None) == (
        "authority_invalid",
        "patient_grouping_authority_mapping_mismatch",
    )


def test_the_statuses_are_one_closed_set() -> None:
    assert PATIENT_GROUPING_STATUSES == get_args(PatientGroupingStatus)


@pytest.fixture
def planning_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    from easyicu.research_agent.acquisition import catalog

    menu = AvailableCatalog(source="canonical", concepts=[CatalogConcept("death")])
    monkeypatch.setattr(catalog, "build_database_capability_catalog", lambda _: menu)
    monkeypatch.setattr(catalog, "build_available_catalog", lambda _: menu)


def _planning_catalog(output_dir: Path, status: Any) -> Any:
    llm = ScriptedMockLLMClient(
        [
            json.dumps(
                {
                    "selected_concepts": ["death"],
                    "rationale": "Mortality only.",
                    "inclusion_exclusion": [],
                }
            )
        ]
    )
    return agent_pipeline_runs._metadata_only_planning_acquisition(
        database="miiv",
        export_path="/metadata",
        question="Describe in-hospital death.",
        llm=llm,
        output_dir=output_dir,
        patient_grouping_status=status,
    )


@pytest.mark.parametrize(
    "status",
    [
        ("available_unbound", None),
        ("authority_invalid", "patient_grouping_authority_mapping_mismatch"),
    ],
)
def test_a_metadata_only_catalog_states_it_beside_its_context(
    planning_menu: None, tmp_path: Path, status: Any
) -> None:
    result = _planning_catalog(tmp_path / "catalog", status)

    authority = pd.read_parquet(result.universe_path).attrs[
        "easyicu_planning_authority"
    ]
    receipt = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    # The authority the context builder reads and the receipt state the same;
    # an invalid authority's code is beside its status, and only there.
    for stated in (authority, receipt):
        assert stated[PATIENT_GROUPING_STATUS_KEY] == status[0]
        assert stated.get(PATIENT_GROUPING_AUTHORITY_ERROR_KEY) == status[1]
    # A status only: no column is bound beside it.
    assert "replacement_row_identity" not in authority
    unstated = _planning_catalog(tmp_path / "unstated", None)
    assert (
        PATIENT_GROUPING_STATUS_KEY
        not in pd.read_parquet(unstated.universe_path).attrs[
            "easyicu_planning_authority"
        ]
    )


def _resume_profile(written: Any) -> Any:
    return SimpleNamespace(
        kind="metadata_only_planning_catalog",
        universe_path=written.universe_path,
        provenance_path=written.provenance_path,
        selected_concepts=["death"],
        universe_sha256=hashlib.sha256(written.universe_path.read_bytes()).hexdigest(),
        provenance_sha256=hashlib.sha256(
            written.provenance_path.read_bytes()
        ).hexdigest(),
    )


def test_a_restored_catalog_states_what_the_launch_decides(
    planning_menu: None, tmp_path: Path
) -> None:
    def restore(written: Any, status: Any, name: str) -> Any:
        return agent_pipeline_runs._restore_metadata_only_planning_acquisition(
            database="miiv",
            export_path="/metadata",
            profile=_resume_profile(written),
            output_dir=tmp_path / name,
            patient_grouping_status=status,
        )

    stated = _planning_catalog(tmp_path / "stated", ("available_unbound", None))
    unstated = _planning_catalog(tmp_path / "unstated", None)

    assert not restore(stated, ("available_unbound", None), "same").blocked
    refused = _refused(lambda: restore(stated, ("source_has_none", None), "changed"))
    assert refused.code == (
        "research_pipeline_development_resume_identity_authority_mismatch"
    )
    # A catalog written before the status existed states none, and restores.
    assert not restore(unstated, ("available_unbound", None), "older").blocked


# --- execution: bind the source's grouping, or stop before materialization -------


def test_no_step_that_needs_groups_binds_nothing_and_resolves_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(_study: Any) -> Any:
        raise AssertionError("resolved a grouping no step needs")

    monkeypatch.setattr(run_patient_grouping, "verified_patient_grouping", forbidden)

    assert run_patient_grouping.execution_patient_grouping({}, step_ids=()) is None


def test_a_step_that_needs_groups_binds_the_sources_verified_grouping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_patient_grouping, "declared_longitudinal_design", lambda _study: None
    )
    monkeypatch.setattr(
        run_patient_grouping, "verified_patient_grouping", lambda _study: _GROUPING
    )

    bound = run_patient_grouping.execution_patient_grouping(
        {}, step_ids=("primary_performance",)
    )

    assert bound is _GROUPING


def test_a_source_without_grouping_binds_nothing_and_leaves_it_to_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The materialized rows may still carry a direct patient identifier; the
    # approval check stops a plan whose planned data holds no grouping.
    def forbidden(_study: Any) -> Any:
        raise AssertionError("a conflict without a grouping to bind")

    monkeypatch.setattr(run_patient_grouping, "declared_longitudinal_design", forbidden)
    monkeypatch.setattr(
        run_patient_grouping, "verified_patient_grouping", lambda _study: None
    )

    assert (
        run_patient_grouping.execution_patient_grouping(
            {}, step_ids=("primary_performance",)
        )
        is None
    )


@pytest.mark.parametrize("design", ["trajectory_design", "landmark"])
def test_a_declared_trajectory_design_stops_before_materialization(
    design: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        run_patient_grouping, "declared_longitudinal_design", lambda _study: design
    )
    monkeypatch.setattr(
        run_patient_grouping, "verified_patient_grouping", lambda _study: _GROUPING
    )

    error = _refused(
        lambda: run_patient_grouping.execution_patient_grouping(
            {}, step_ids=("primary_performance",)
        )
    )

    assert error.code == "research_pipeline_patient_grouping_trajectory_conflict"
    assert error.details == {
        "step_ids": ["primary_performance"],
        "longitudinal_design": design,
    }


@pytest.mark.parametrize(
    ("strategies", "trajectory", "expected"),
    [
        ((), None, None),
        (("landmark",), None, "landmark"),
        ((), object(), "trajectory_design"),
        # The time-varying owner applies the grouping to both row spaces, so
        # the trajectory is still emitted beside it.
        (("time_varying", "landmark"), object(), None),
    ],
)
def test_which_declared_designs_read_the_long_trajectory(
    strategies: tuple[str, ...],
    trajectory: Any,
    expected: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        research_launch_scientific,
        "_configured_sensitivity_specs",
        lambda _study: tuple(SimpleNamespace(strategy=value) for value in strategies),
    )
    monkeypatch.setattr(
        research_launch_scientific, "_validate_trajectory_design", lambda _s: trajectory
    )

    assert research_launch_scientific.declared_longitudinal_design({}) == expected


def test_a_cluster_design_still_resolves_through_the_same_grouping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(
        research_launch_scientific,
        "verified_patient_grouping",
        lambda study: calls.append(study) or _GROUPING,
    )
    study = {
        "analysis_design": {
            "variance_estimator": "cluster_robust",
            "cluster_unit": "patient",
        }
    }

    assert (
        research_launch_scientific._patient_grouping_for_analysis_design(study)
        is _GROUPING
    )
    assert (
        research_launch_scientific._patient_grouping_for_analysis_design(
            {"analysis_design": {"variance_estimator": "model_based"}}
        )
        is None
    )
    assert calls == [study]


@pytest.mark.parametrize(
    "design",
    [
        {"variance_estimator": "model_based"},
        {"variance_estimator": "cluster_robust", "cluster_unit": "patient"},
        {},
    ],
)
def test_the_sources_grouping_is_resolved_whatever_the_design(
    monkeypatch: pytest.MonkeyPatch, design: dict[str, Any]
) -> None:
    # A step that splits by patient needs the groups even when the study's
    # inference does not read them, so the resolver never asks the design.
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        research_launch_scientific.source_identity_authority,
        "resolve_study_patient_grouping",
        lambda *, export_path, database: calls.append((export_path, database)) or _GROUPING,
    )
    study = {
        "data_source": {"path": "/exports/eicu", "database": "eicu"},
        "analysis_design": design,
    }

    assert research_launch_scientific.verified_patient_grouping(study) is _GROUPING
    assert calls == [("/exports/eicu", "eicu")]
    # No bound source, no grouping, and nothing resolved.
    assert research_launch_scientific.verified_patient_grouping({"analysis_design": design}) is None
    assert calls == [("/exports/eicu", "eicu")]


def test_a_grouping_authority_that_fails_its_checks_refuses_with_its_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*, export_path: str, database: str) -> Any:
        raise research_launch_scientific.source_identity_authority.PatientGroupingAuthorityError(
            "patient_grouping_authority_mismatch", "The bridge does not match its source."
        )

    monkeypatch.setattr(
        research_launch_scientific.source_identity_authority,
        "resolve_study_patient_grouping",
        broken,
    )

    refused = _refused(
        lambda: research_launch_scientific.verified_patient_grouping(
            {"data_source": {"path": "/exports/eicu", "database": "eicu"}}
        )
    )

    assert refused.code == "patient_grouping_authority_mismatch"


def test_the_launch_binds_the_accepted_steps_grouping_when_its_design_binds_none() -> None:
    # The package-bound launch of an accepted candidate binds the owner's
    # grouping for the steps that need it, unless the design already bound one.
    tree = ast.parse(Path(agent_pipeline_runs.__file__).read_text(encoding="utf-8"))
    branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(statement, ast.Assign)
            and isinstance(statement.value, ast.Call)
            and ast.unparse(statement.value.func)
            == "run_patient_grouping.execution_patient_grouping"
            for statement in node.body
        )
    ]

    assert len(branches) == 1
    (branch,) = branches
    assert ast.unparse(branch.test) == "patient_grouping is None"
    assert [ast.unparse(statement) for statement in branch.body] == [
        "patient_grouping = run_patient_grouping.execution_patient_grouping("
        "study, step_ids=candidate_authority.patient_grouping_steps)"
    ]


@pytest.mark.parametrize(
    ("design", "states"),
    [
        ({"variance_estimator": "cluster_robust", "cluster_unit": "patient"}, True),
        # A causal suite's bootstrap that resamples patients reads them too.
        ({"variance_estimator": "bootstrap", "cluster_unit": "patient"}, True),
        ({"variance_estimator": "bootstrap"}, False),
        ({"variance_estimator": "model_based"}, False),
        ({}, False),
    ],
)
def test_a_run_states_patient_dependence_only_for_a_design_that_reads_the_grouping(
    design: dict[str, Any], states: bool
) -> None:
    # A grouping bound only for a step that splits by patient leaves the
    # variance the study declares.
    dependence = run_patient_grouping.runtime_patient_dependence(_GROUPING, design)

    if states:
        assert dependence == PlannedDependenceRequirement(
            group_source="patient_stay_id",
            group_derivation="prefix_before_delimiter",
            delimiter=":s",
        )
    else:
        assert dependence is None
    assert run_patient_grouping.runtime_patient_dependence(None, design) is None


def test_the_launch_hands_its_runtime_projection_the_owners_dependence() -> None:
    # The package-bound launch compiles one runtime projection.  Its dependence
    # is the owner's, so a patient bootstrap resamples patients and a grouping
    # bound only for a split states none.
    tree = ast.parse(Path(agent_pipeline_runs.__file__).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None))
        == "compile_web_scientific_runtime_projection"
    ]

    assert len(calls) == 1
    (dependence,) = [kw.value for kw in calls[0].keywords if kw.arg == "dependence"]
    assert ast.unparse(dependence) == (
        "run_patient_grouping.runtime_patient_dependence("
        "patient_grouping, validated_analysis_design)"
    )


# --- the accepted candidate carries the steps -------------------------------------


def test_the_accepted_candidate_names_the_steps_that_need_patient_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = complete_study()
    study.update(
        {
            "question": "Build an in-hospital mortality prediction model from first-day vitals.",
            "primary_exposure": "",
            "covariates": [],
            "covariate_selection": "planner_selectable",
            "execution_concepts": {"outcome": "death"},
            "analysis_design": {
                "analysis_family": "prediction_model",
                "analysis_unit": "icu_stay",
                "variance_estimator": "model_based",
            },
        }
    )
    source_run_id = "run-prediction-candidate"
    project_dir = tmp_path / "candidate-wrapper"
    inner_run = project_dir / "pipeline" / source_run_id
    inner_run.mkdir(parents=True)
    capsule_raw = json.dumps(
        {
            "scientific_identity": {
                "question": study["question"],
                "database": "miiv",
                "primary_exposure": None,
                "target_outcome": "death",
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
    catalog = ["death", "hr", "age"]
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
        score=92,
        dimension_scores={"study_design": 92},
        findings=[],
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
                    "run_id": source_run_id,
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
                "agent_plan.json": {
                    "analysis_type": "prediction_model",
                    "cohort": {"selection_mode": "all_input_rows"},
                    "steps": [_PREDICTION_PRIMARY, _DESCRIPTIVE],
                },
            },
        },
    )
    monkeypatch.setattr(
        agent_pipeline_runs,
        "_metadata_only_planning_coordinates",
        lambda **_kwargs: {"primary_exposure": None, "target_outcome": "death"},
    )

    authority = agent_pipeline_runs._load_candidate_plan_materialization_authority(
        study=study,
        project_root=str(tmp_path),
        source_run_id=source_run_id,
        database="miiv",
        covariates=(),
    )

    assert authority is not None
    assert authority.patient_grouping_steps == ("primary_performance",)


# --- approval: a plan that needs groups its planned data lacks stops first --------


class _UnreachablePipeline:
    has_resumable_human_review = True

    def resume_human_review(self, decisions, *, run_id, progress_callback=None):
        raise RuntimeError("the approved plan resumed")


def _write_export(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
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


def _planned_context(*, grouped: bool) -> ResearchContext:
    provenance: dict[str, Any] = {"database": "miiv", "analysis_unit": "icu_stay"}
    id_columns = ["stay_id"]
    if grouped:
        provenance["replacement_row_identity"] = {
            "output_identity_column": "patient_stay_id",
            "mapping_file_sha256": "e" * 64,
            "patient_group_derivation": {
                "algorithm": "prefix_before_:s",
                "delimiter": ":s",
            },
        }
        id_columns = ["patient_stay_id"]
    return ResearchContext(
        research_question="Build an in-hospital mortality prediction model.",
        cohort=CohortDescriptor(
            cohort_name="synthetic_prediction",
            database="miiv",
            n_stays=10,
            id_columns=id_columns,
            outcome_columns=["death"],
            provenance=provenance,
        ),
        variables=[
            ConceptDescriptor(name=id_columns[0], role=VariableRole.ID, dtype="object"),
            ConceptDescriptor(
                name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64"
            ),
        ],
    )


def _approve_paused_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    context: ResearchContext | None,
) -> agent_pipeline_runs.ResearchPipelineRunError:
    """Pause one approvable prediction plan, then approve it."""

    monkeypatch.setattr(
        agent_pipeline_runs,
        "_load_pending_scientific_review",
        lambda *_args, **_kwargs: {
            "schema_version": agent_pipeline_runs.CURRENT_SCIENTIFIC_REVIEW_SCHEMA_VERSION,
            "approval_allowed": True,
        },
    )
    monkeypatch.setattr(
        agent_pipeline_runs,
        "_plan_has_complete_reviewable_recommendation",
        lambda _plan: True,
    )
    export = _write_export(tmp_path / "export")
    study = {
        **complete_study(),
        "data_source": {"path": str(export), "database": "miiv"},
    }
    binding = dict(
        dataio.validate_research_pipeline_source(str(export), database="miiv")[
            "binding"
        ]
    )
    run_dir = tmp_path / "pipeline" / "run-prediction"
    run_dir.mkdir(parents=True)
    if context is not None:
        (run_dir / "research_context.json").write_text(
            context.model_dump_json(), encoding="utf-8"
        )
    authority = PlanReviewAuthority.create(
        plan=AnalysisPlan(
            research_question="Build an in-hospital mortality prediction model.",
            steps=[AnalysisStep.model_validate(_PREDICTION_PRIMARY)],
        )
    )
    pending = HumanReviewPending(
        run_id="run-prediction",
        thread_id="thread-prediction",
        run_dir=str(run_dir),
        requests=(
            HumanReviewRequest.create(
                kind="scientific_stop",
                summary="Review the digest-bound plan before analysis.",
                authority_sha256="a" * 64,
                payload={
                    "reason": "operator_plan_approval_required",
                    "plan_review_authority": authority.model_dump(mode="json"),
                },
            ),
        ),
    )
    wrapper = tmp_path / "projects" / str(study["id"]) / "run_web_prediction"
    agent_pipeline_runs._write_projection(
        wrapper_dir=wrapper,
        study=study,
        provider={"provider": "openai", "model": "test-model"},
        acquisition=_acquisition_receipt(),
        run_dir=run_dir,
        pending=pending,
    )
    hard_stop = agent_pipeline_runs._start_web_provider_hard_stop(
        wrapper_dir=wrapper,
        job_id="prediction",
        declaration_sha256=study_context_owner.scientific_configuration_sha256(study),
    )
    hard_stop.pause()
    registry = agent_pipeline_runs.PendingReviewRegistry()
    monkeypatch.setattr(agent_pipeline_runs, "_PENDING_REVIEWS", registry)
    registry.register(
        agent_pipeline_runs._PendingRun(
            pipeline=_UnreachablePipeline(),
            pending=pending,
            wrapper_dir=wrapper,
            study=study,
            provider={},
            acquisition=_acquisition_receipt(),
            created_at=1.0,
            prepared_package_binding=binding,
            provider_hard_stop=hard_stop,
        )
    )
    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as caught:
        agent_pipeline_runs.resume_research_pipeline(
            run_id="run-prediction",
            study_context_id=study["id"],
            decision="approved",
            reviewer="local reviewer",
            note="",
            job=SimpleNamespace(emit=lambda _event: None, cancel_requested=False),
            current_study_context=study,
        )
    return caught.value


@pytest.mark.parametrize(
    ("context", "status"),
    [
        (None, "context_unreadable"),
        (_planned_context(grouped=False), "not_in_planned_data"),
    ],
)
def test_an_approval_whose_plan_needs_groups_its_data_lacks_stops_before_any_step(
    context: ResearchContext | None,
    status: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _approve_paused_prediction(tmp_path, monkeypatch, context=context)

    # The paused pipeline was never resumed: its resume raises.
    assert error.code == "research_pipeline_patient_grouping_required"
    assert error.details == {
        "step_ids": ["primary_performance"],
        "grouping_status": status,
        "stage": "approval",
    }


def test_an_approval_whose_planned_data_holds_the_grouping_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = _approve_paused_prediction(
        tmp_path, monkeypatch, context=_planned_context(grouped=True)
    )

    # The defence let the approval through to the pipeline, whose resume fails.
    assert error.code != "research_pipeline_patient_grouping_required"


def test_a_plan_that_needs_no_groups_is_not_read_for_them(tmp_path: Path) -> None:
    run_patient_grouping.require_patient_groups_for_approval(
        {"steps": [_DESCRIPTIVE]}, context_path=tmp_path / "absent.json"
    )


# --- the conversation reads both stops ---------------------------------------------

ERROR_TEXT = (
    Path(agent_pipeline_runs.__file__).resolve().parent
    / "static"
    / "js"
    / "screens-guided-pi-error-text.js"
)


def _stop_line(code: str, lang: str) -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = r"""
let errorText = null;
const lang = process.argv[2];
global.window = {
  EU_LANG: lang,
  EU_HTML: { esc: value => String(value) },
  EasyICU: { guidedPi: { declare: (name, api) => { if (name === 'errorText') errorText = api; } } },
};
require(process.argv[1]);
const owner = errorText.create({ tr: (en, zh) => (lang === 'zh' ? zh : en), staticPreview: () => false });
process.stdout.write(owner.runFailureText(process.argv[3]));
"""
    result = run_node(node, script, str(ERROR_TEXT), lang, code, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


@pytest.mark.parametrize(
    ("code", "en", "zh"),
    [
        (
            run_patient_grouping.PATIENT_GROUPING_REQUIRED,
            "has no verified patient grouping",
            "没有已核实的患者分组",
        ),
        (
            run_patient_grouping.PATIENT_GROUPING_TRAJECTORY_CONFLICT,
            "(a trajectory or landmark design)",
            "（轨迹或 landmark 设计）",
        ),
    ],
)
def test_the_conversation_reads_each_stop_as_its_cause(
    code: str, en: str, zh: str
) -> None:
    assert en in _stop_line(code, "en")
    assert zh in _stop_line(code, "zh")
