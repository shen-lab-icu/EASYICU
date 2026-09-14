"""Shared bounded StudyContext fixtures for Copilot workflow contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.pi_copilot import cohort_eligibility

def confirmed_cohort_decision(
    option_id: str,
    *,
    study_context_id: str,
    study_context_revision: int,
    current_cohort: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = dict(current_cohort or {})
    target = cohort_eligibility.selection_cohort_for_option(
        {"cohort": base}, option_id
    )
    scope = study_context_owner.normalize_primary_cohort_scope(
        {"cohort": target}
    )
    session_id = f"pi-{study_context_id}"
    seed = (
        f"{session_id}:{study_context_id}:{study_context_revision - 1}:"
        f"{option_id}:{scope.sha256}"
    )
    event = cohort_eligibility.build_selection_event(
        option_id=option_id,
        study_context_id=study_context_id,
        expected_revision=study_context_revision - 1,
        session_id=session_id,
        user_turn_id=f"turn-{session_id}",
        event_id=hashlib.sha256(f"event:{seed}".encode()).hexdigest(),
        one_use_grant_id=hashlib.sha256(f"grant:{seed}".encode()).hexdigest(),
        primary_cohort_contract_sha256=scope.sha256,
        actor_id_sha256=hashlib.sha256(f"actor:{session_id}".encode()).hexdigest(),
        selected_at="2026-08-29T12:00:00Z",
    )
    authority = cohort_eligibility.confirmation_authority_for_option(
        option_id,
        study_context_id=study_context_id,
        study_context_revision=study_context_revision,
        current_cohort=base,
        selection_event=event,
        confirmed_at="2026-08-29T12:00:00Z",
    )
    return target, authority


def complete_study() -> dict[str, Any]:
    cohort, authority = confirmed_cohort_decision(
        "no_eligibility_filter",
        study_context_id="study-workflow",
        study_context_revision=4,
        current_cohort={"max_patients": 2000},
    )
    return {
        "id": "study-workflow",
        "revision": 4,
        "question": "Does an aggregate ICU feature predict mortality?",
        "data_source": {
            "path": "/private/prepared/source",
            "database": "mimiciv",
        },
        "cohort": cohort,
        "cohort_eligibility_authority": authority,
        "modules": ["vitals", "outcome"],
        "outcome": "In-hospital mortality",
        "primary_exposure": "heart_rate",
        "covariates": ["age", "sex"],
        "covariate_selection": "exact",
        "covariate_rationales": {
            "age": "Age is a baseline demographic confounder selected before analysis.",
            "sex": "Sex is a baseline demographic confounder selected before analysis.",
        },
        "covariate_temporal_roles": {
            "age": "baseline_static",
            "sex": "baseline_static",
        },
        "execution_concepts": {
            "outcome": "death",
            "primary_exposure": "heart_rate",
            "covariates": ["age", "sex"],
        },
        "analysis_design": {
            "analysis_unit": "icu_stay",
            "variance_estimator": "model_based",
        },
        "time_window": {"hours": 24, "anchor": "ICU admission"},
        "confirmations": {
            "feature_time_window": True,
            "export_format": True,
            "extraction_completed": True,
        },
        "export_format": "parquet",
        "analysis_goal": "Descriptive prognostic association",
    }


def _foundation_profile() -> dict[str, Any]:
    return {
        "allowed_modules": ("demographics", "outcome"),
        "static_concepts": ("age", "sex"),
        "outcome_concepts": ("death",),
        "required_feature_concepts": (),
        "require_outcome": True,
        "primary_exposure_source_concept": "heart_rate",
    }


def _write_real_pipeline_fixture(run_dir: Path, *, manuscript: str) -> None:
    (run_dir / "evidence").mkdir(parents=True)
    (run_dir / "results").mkdir()
    readiness = {
        "execution_complete": True,
        "analysis_validated": True,
        "evidence_complete": True,
        "numeric_verified": True,
        "manuscript_ready": False,
    }
    (run_dir / "run_status.json").write_text("{}", encoding="utf-8")
    plan_payload = {
        "steps": [
            {
                "id": "model",
                "title": "Fit specified model",
                "literature_citation_keys": ["method_paper"],
            }
        ]
    }
    plan_bytes = json.dumps(plan_payload).encode("utf-8")
    (run_dir / "analysis_plan.json").write_bytes(plan_bytes)
    import hashlib

    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "readiness": readiness,
                "current_plan_authority": {
                    "relative_path": "analysis_plan.json",
                    "sha256": hashlib.sha256(plan_bytes).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "preplan_literature_bundle.json").write_text(
        json.dumps(
            {
                "research_question": "Does an ICU exposure predict mortality?",
                "citations": [
                    {
                        "key": "method_paper",
                        "title": "A source-backed method paper",
                        "year": "2024",
                        "venue": "Statistics in Medicine",
                        "pmid": "12345",
                    }
                ],
                "prisma": None,
                "search_provenance": {
                    "curated_seed_count": 1,
                    "sources_enabled": [],
                    "sources_returning": [],
                    "search_conducted": False,
                    "note": "Curated method reference; no retrieval was run.",
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manuscript_scaffold_bound.md").write_text(
        manuscript,
        encoding="utf-8",
    )
    (run_dir / "claim_ledger.csv").write_text(
        "claim_id,claim_text,evidence_refs,status,note\n"
        "c1,The registered aggregate estimate passed validation,ev-table,analysis_only,Review required\n",
        encoding="utf-8",
    )
    (run_dir / "results" / "aggregate.csv").write_text(
        "row_role,exposure_level_index,exposure_level,n_rows,exposure_denominator,exposure_pct,exposure_ci_low_pct,exposure_ci_high_pct,exposure_standard_error_pct,exposure_interval_covariance,exposure_interval_cluster_count,outcome_observed_n,outcome_missing_n,outcome_events,outcome_denominator,outcome_rate_pct,interval_method\n"
        "exposure_level,0,0,60,100,60.0,,,,none_counts_only,,60,0,5,60,8.3,none_counts_only\n",
        encoding="utf-8",
    )
    (run_dir / "results" / "identifier_rows.csv").write_text(
        "metric,a,b,c,d,e,f,g,h,i,j,k,stay_id\n"
        "sensitive,1,2,3,4,5,6,7,8,9,10,11,123\n",
        encoding="utf-8",
    )
    (run_dir / "evidence" / "evidence_index.json").write_text(
        json.dumps(
            [
                {
                    "kind": "table",
                    "evidence_id": "ev-table",
                    "description": "Aggregate model result",
                    "relative_path": "results/aggregate.csv",
                },
                {
                    "kind": "table",
                    "evidence_id": "ev-sensitive",
                    "description": "Identifier rows",
                    "relative_path": "results/identifier_rows.csv",
                },
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "figure_gallery.json").write_text(
        json.dumps({"status": "no_figures", "figures": []}),
        encoding="utf-8",
    )


def _acquisition_receipt() -> SimpleNamespace:
    return SimpleNamespace(
        selection=SimpleNamespace(selected_concepts=["heart_rate", "mortality"]),
        materialized_concepts=["heart_rate", "mortality"],
        materialized_columns=("heart_rate", "mortality"),
        coverage=SimpleNamespace(sufficient=True),
        analysis_columns={
            "heart_rate": "heart_rate",
            "death": "mortality",
        },
        endpoint=None,
    )


def _write_development_resume_literature(
    run_dir: Path,
    *,
    research_question: str,
) -> None:
    (run_dir / "preplan_literature_bundle.json").write_text(
        json.dumps(
            {
                "research_question": research_question,
                "citations": [],
                "screening_decisions": [],
                "design_evidence_cards": [],
            }
        ),
        encoding="utf-8",
    )


def _write_development_resume_planner_catalog(
    run_dir: Path,
    *,
    selected_concepts: tuple[str, ...] = ("lact", "death"),
    patient_identity_column: str | None = None,
    operationalized_columns: tuple[str, ...] = (),
) -> None:
    pipeline_input = run_dir.parent.parent / "pipeline_input"
    pipeline_input.mkdir(parents=True, exist_ok=True)
    universe = pipeline_input / "planner_catalog.parquet"
    columns: dict[str, pd.Series] = {"stay_id": pd.Series(dtype="int64")}
    if patient_identity_column:
        columns[patient_identity_column] = pd.Series(dtype="string")
    columns.update(
        {
            column: pd.Series(dtype="float64")
            for column in operationalized_columns
        }
    )
    columns.update(
        {
            concept: pd.Series(dtype="float64")
            for concept in selected_concepts
        }
    )
    frame = pd.DataFrame(columns)
    replacement_row_identity = (
        {
            "output_identity_column": patient_identity_column,
            "mapping_file_sha256": "a" * 64,
            "mapped_cohort_rows": 0,
            "patient_group_derivation": {
                "algorithm": "prefix_before_:s",
                "delimiter": ":s",
            },
            "authority_coordinates": {
                "schema_version": "easyicu.patient_grouping_runtime_authority/1",
                "authority_ref": "test/identity-bridge/v1",
                "database": "miiv",
                "mapping_sha256": "a" * 64,
                "grouping_derivation": "prefix_before_:s",
                "provider_visible_values": False,
            },
        }
        if patient_identity_column
        else None
    )
    frame.attrs["easyicu_planning_authority"] = {
        "kind": "metadata_only_planning_catalog",
        "patient_rows_read": False,
        **(
            {"replacement_row_identity": replacement_row_identity}
            if replacement_row_identity is not None
            else {}
        ),
    }
    frame.to_parquet(universe, index=False)
    selected_sha256 = hashlib.sha256(
        json.dumps(
            list(selected_concepts),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    (pipeline_input / "planner_catalog_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu.metadata-only-planning-catalog/1",
                "database": "miiv",
                "catalog_source": "easyicu-database-capability:miiv",
                "row_identity_column": "stay_id",
                "patient_identity_column": patient_identity_column,
                "operationalized_columns": list(operationalized_columns),
                "replacement_row_identity": replacement_row_identity,
                "selected_concepts": list(selected_concepts),
                "selected_concepts_sha256": selected_sha256,
                "patient_rows_read": False,
                "patient_rows_written": False,
                "observed_feasibility_claims": False,
                "execution_authorized": False,
                "planning_target_outcome": "death",
                "planning_endpoint": {
                    "name": "death",
                    "kind": "binary",
                    "absence_semantics": "no_absent_rows",
                    "levels": [0, 1],
                    "event_column": None,
                    "time_column": None,
                    "time_origin": None,
                    "censoring_rule": None,
                },
            }
        ),
        encoding="utf-8",
    )
