"""Synthetic grouped inputs and sealed Web launch coordinates for resume tests."""

import json
import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.cohort import materializer
from easyicu.research_agent.authority.run_input import build_scientific_identity, seal_run_input_capsule
from easyicu.research_agent.intake.materialized_metadata import load_verified_materialized_cohort_authority
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


def context(cohort: Path, *, primary_exposure=None, trajectory_binding=None):
    typed = load_verified_materialized_cohort_authority(cohort)
    return build_research_context(
        research_question="Compare hospital death across lactate groups in the first day.",
        cohort=cohort, cohort_name="synthetic", database="miiv",
        primary_exposure=primary_exposure, target_outcome="death",
        trajectory_binding=trajectory_binding,
        id_columns=(typed.authority.identity_column if typed else "stay_id",), outcome_columns=("death",),
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


def prepared(root: Path, *, patient_grouped: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    identity = {}
    if patient_grouped:
        from easyicu.research_agent.acquisition.patient_grouping import PatientGroupingBinding

        mapping = root / "synthetic_grouping.parquet"
        pd.DataFrame({"stay_id": [1, 2], "patient_key": [7, 8]}).to_parquet(mapping, index=False)
        digest = hashlib.sha256(mapping.read_bytes()).hexdigest()
        identity = PatientGroupingBinding(
            mapping_path=mapping, mapping_sha256=digest,
            mapping_stay_column="stay_id", mapping_patient_column="patient_key",
            authority_coordinates={
                "schema_version": "easyicu.patient_grouping_runtime_authority/1",
                "authority_ref": "synthetic/patient_grouping", "mapping_sha256": digest,
                "grouping_derivation": "prefix_before_:s", "provider_visible_values": False,
            },
        ).materializer_kwargs()
    paths = materializer.materialize_to_parquet(
        root / "materialized", stem="universe", data_path=typed_export(root / "export"),
        database="miiv", static_concepts=("age",), feature_concepts=("lact",),
        outcome_concepts=("death",),
        **identity,
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


def seal_prepared_source(source, evidence):
    """Keep the real materializer input and capsule, as a Web preparation does."""
    original_dir = source.original.parent.parent / "materialized"
    input_dir = source.run.parents[1] / "pipeline_input"
    shutil.copytree(original_dir, input_dir)
    universe = input_dir / "universe.parquet"
    typed = load_verified_materialized_cohort_authority(universe)
    initial = context(universe)
    identity = build_scientific_identity(
        cohort=universe, question=initial.research_question, cohort_name="synthetic",
        database="miiv", target_outcome="death", primary_exposure=None,
        cross_database_validation=None, inclusion_criteria=None, exclusion_criteria=None,
        id_columns=initial.cohort.id_columns, time_columns=None, outcome_columns=("death",),
        time_windows=None, concept_descriptions=None,
        user_preferences=initial.user_preferences.model_dump(mode="json"), notes=None,
        skill_key=None, experiment_spec=None, source_files=None, disable_icu_context=False,
        materialized_cohort_authority_ref=typed.reference.to_dict(),
    )
    seal_run_input_capsule(
        run_dir=source.run, evidence=evidence, scientific_identity=identity,
        initial_environment={}, context_path=source.run / "research_context.json",
        cohort_path=source.cohort, experiment_spec_path=None,
    )
    source.universe = universe


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
