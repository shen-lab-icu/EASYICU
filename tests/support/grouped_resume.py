"""Synthetic grouped inputs and sealed Web launch coordinates for resume tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.cohort import materializer
from easyicu.research_agent.intake.materialized_metadata import stage_materialized_cohort_authority
from easyicu.research_agent.orchestration.exposure_grouping_phase import run_exposure_grouping_phase
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.webserver import agent_review_recovery as recovery
from easyicu.webserver import research_launch_resume as owner
from easyicu.webserver import study_contexts
from tests.support.typed_export import typed_export

ANSWER = json.dumps({"groupings": [{
    "id": "x1", "concept": "lact", "window": {"start_hours": 0, "end_hours": 24},
    "scale": "nominal",
    "groups": [
        {"id": "g1", "label": "lactate below 2.5", "rule": {
            "summary": "max", "op": "<", "value": 2.5, "unit": "mmol/L",
        }},
        {"id": "g2", "label": "lactate 2.5 or above", "rule": "otherwise"},
    ],
    "unmeasured": {"handling": "own_group", "label": "not measured"},
    "reference": "g1", "contrast": "g2", "quote": "lactate groups", "source": "question",
}]})


def context(cohort: Path, *, primary_exposure=None):
    return build_research_context(
        research_question="Compare hospital death across lactate groups in the first day.",
        cohort=cohort, cohort_name="synthetic", database="miiv",
        primary_exposure=primary_exposure, target_outcome="death",
        id_columns=("stay_id",), outcome_columns=("death",),
        user_preferences={"data_constraints": json.dumps({"materialization_window": {
            "role": "outer_observation_window", "anchor": "icu_admission", "hours": 24,
        }})},
    )


def catalog(root: Path, *columns: str) -> Path:
    frame = pd.DataFrame({
        "stay_id": pd.Series(dtype="int64"),
        **{name: pd.Series(dtype="float64") for name in columns},
        "death": pd.Series(dtype="float64"),
    })
    frame.attrs["easyicu_planning_authority"] = {
        "kind": "metadata_only_planning_catalog", "patient_rows_read": False,
    }
    path = root / "run/cohort.parquet"
    path.parent.mkdir(parents=True)
    frame.to_parquet(path, index=False)
    return path


def prepared(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    paths = materializer.materialize_to_parquet(
        root / "materialized", stem="universe", data_path=typed_export(root / "export"),
        database="miiv", static_concepts=("age",), feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    copy = root / "run/cohort.parquet"
    stage_materialized_cohort_authority(paths["parquet"], copy, producer_implementation_sha256="a" * 64)
    return copy


def phase(cohort: Path, client):
    return run_exposure_grouping_phase(
        context=context(cohort), cohort_path=cohort, run_dir=cohort.parent,
        planner=client, rebuild_context=context, capability_review_pending=False,
        trajectory_staged=False, emit_progress=lambda *_args, **_kwargs: None,
    )


@pytest.fixture
def source(tmp_path, monkeypatch):
    study = {
        "id": "study-grouped-resume", "question": "Compare first-day lactate groups",
        "data_source": {"path": str(tmp_path / "source"), "database": "miiv"},
    }
    wrapper = tmp_path / study["id"] / "run_prior"
    run = wrapper / "pipeline/run_pipeline"
    run.mkdir(parents=True)
    checkpoint = run / "progressive_planner_checkpoint_002.json"
    checkpoint.write_text("fixture: checkpoint chains have separate real validation tests")
    monkeypatch.setattr(owner, "_development_progressive_resume_binding", lambda **_: (checkpoint, "a" * 64))
    seed = recovery.WebReviewRecoverySeed.create(
        wrapper_dir=str(wrapper.resolve()), study=study,
        scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(study),
        provider_meta={}, provider_public={}, credential_source="pi_verified",
        budget_mode="planner_canary", prepared_package_binding=None,
        pipeline_config={"bound_plan_revision_contract": None}, pipeline_config_sha256="c" * 64,
        acquisition_projection={}, hard_stop_ledger_path="", hard_stop_task_id="web-prior",
        hard_stop_declaration_sha256="d" * 64, created_at=1.0,
    )
    (wrapper / ".runtime").mkdir()
    (wrapper / ".runtime/web_review_recovery_seed.json").write_text(seed.model_dump_json())

    def scope():
        return owner._development_resume_launch_scope(
            project_root=str(tmp_path), study=study, source_job_id="prior",
        )

    return SimpleNamespace(study=study, run=run, scope=scope)
