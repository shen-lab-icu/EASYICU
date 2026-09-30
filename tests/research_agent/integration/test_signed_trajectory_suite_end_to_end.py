"""The signed trajectory suite runs through its figure and class description.

End to end without a Provider: a typed synthetic export is materialized by the
host, the signed owners run on the long panel, the selection figure reads the
owners' published tables, the host draws the owner's cohort flow, and the
frozen classes are described on the run cohort.  Unit fixtures had drifted from the stability owner's real columns, so
the figure failed on every real run while its own tests passed; this module
runs the owners themselves.  The run's evidence then holds each rule's formal
outcome as a host claim and each owner's executed design as a Methods fact.
Synthetic stays only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.concept.export_metadata import build_export_file_metadata_binding
from easyicu.concept.metadata_projection import ConceptColumnRole
from easyicu.concept.metadata_sidecar import (
    EXPORT_PHYSICAL_SCOPE,
    ColumnMetadataFileBinding,
    ColumnMetadataSidecar,
    write_content_addressed_sidecar,
)
from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.cohort import materializer as cohort_materializer
from easyicu.research_agent.contracts.phenotype_comparison import (
    TRAJECTORY_FROZEN_STATUS,
    TRAJECTORY_NO_SOLUTION_REASON,
)
from easyicu.research_agent.intake import export_package as intake
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.orchestration.services import PipelineServices
from easyicu.research_agent.pipeline import ResearchAgentPipeline
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from easyicu.research_agent.schema import AnalysisPlan, TrajectoryStabilitySpec
from easyicu.research_agent.trajectory.plan_contract import DIAG_GMM_BEST_OF_10_ENGINE
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    build_trajectory_scientific_runtime_authority,
)
from easyicu.resources import load_dictionary
from tests.support.typed_export import metadata_binding

pytestmark = [pytest.mark.integration, pytest.mark.slow]

QUESTION = (
    "Which respiratory and cardiovascular SOFA-2 trajectory classes emerge over "
    "the first 24 h, and how does hospital mortality differ by class?"
)
COORDINATES = ("sofa2_resp", "sofa2_cardio")
N_STAYS = 150
#: Ten stays have no window value and ten have one window only; the owner
#: needs two, so twenty stays are counted but never clustered.
N_NOT_CLUSTERED = 20
#: The host draws the owner's cohort flow, so no step needs a generated script.
COHORT_FLOW_FIGURE_STEP = "07_cohort_accounting_figure"
SIGNED_AND_DESCRIPTION_STEPS = (
    "00_authority_compiled_trajectory_representation",
    "01_authority_compiled_trajectory_candidates",
    "02_authority_compiled_trajectory_stability",
    "03_authority_compiled_trajectory_selection_figure",
    "04_host_bound_analysis_cohort",
    "05_frozen_class_description",
)


#: Per class, the (low, high) level of each coordinate.  Three classes sit
#: inside the candidate grid (2-4); five separated classes lie beyond it, so
#: the minimum BIC falls on the grid's upper boundary; noise has no classes.
LAYOUTS = {
    "three_classes": {
        "sofa2_resp": [(0, 1), (2, 2), (3, 4)],
        "sofa2_cardio": [(0, 0), (1, 2), (3, 4)],
    },
    "five_classes": {
        "sofa2_resp": [(0, 0), (0, 0), (4, 4), (4, 4), (2, 2)],
        "sofa2_cardio": [(0, 0), (4, 4), (0, 0), (4, 4), (2, 2)],
    },
    "noise": {"sofa2_resp": [(0, 4)], "sofa2_cardio": [(0, 4)]},
}


def _typed_export(root: Path, *, layout: str) -> Path:
    """Write a native typed export whose SOFA-2 values carry owner receipts."""

    rng = np.random.default_rng(7)
    stays = np.arange(1001, 1001 + N_STAYS)
    levels = LAYOUTS[layout]
    n_classes = len(levels["sofa2_resp"])
    classes = np.repeat(np.arange(n_classes), N_STAYS // n_classes)
    rows = []
    for index, stay in enumerate(stays):
        age = float(np.round(rng.normal(62, 12), 1))
        times: tuple[float, ...] = (2.0, 8.0, 14.0, 20.0)
        if index % 15 == 0:
            times = ()
        elif index % 15 == 1:
            times = (2.0, 8.0)
        for time in times:
            row = {"stay_id": int(stay), "charttime": time, "age": age}
            for concept in COORDINATES:
                low, high = levels[concept][classes[index]]
                spread = 1.2 if layout == "noise" else 0.15
                row[concept] = float(np.clip(rng.normal((low + high) / 2, spread), 0, 4))
                row[f"{concept}_observed"] = 1
                row[f"{concept}_available"] = 1
            rows.append(row)
        if not times:
            # Outside the 0-24 h window: the stay exists but has no window value.
            rows.append(
                {
                    "stay_id": int(stay), "charttime": 30.0, "age": age,
                    "sofa2_resp": 1.0, "sofa2_cardio": 0.0,
                    "sofa2_resp_observed": 1, "sofa2_resp_available": 1,
                    "sofa2_cardio_observed": 1, "sofa2_cardio_available": 1,
                }
            )
    labs = pd.DataFrame(rows)
    outcomes = pd.DataFrame(
        {
            "stay_id": stays.astype(int),
            "death": rng.random(N_STAYS) < np.linspace(0.1, 0.6, n_classes)[classes],
        }
    )
    export = root / "export"
    export.mkdir()
    labs.to_parquet(export / "labs.parquet", index=False)
    outcomes.to_parquet(export / "outcomes.parquet", index=False)
    lab_concepts = ("age", "sofa2_cardio", "sofa2_resp")
    lab_binding = build_export_file_metadata_binding(
        relative_path="labs.parquet",
        module="labs",
        frame=labs,
        concept_ids=lab_concepts,
        database="miiv",
        database_class_prefixes=(),
        dictionary=load_dictionary(include_sofa2=True),
    )
    reference = write_content_addressed_sidecar(
        export,
        ColumnMetadataSidecar(
            source_database="miiv",
            source_database_class_prefixes=(),
            scope=EXPORT_PHYSICAL_SCOPE,
            files=(
                lab_binding,
                ColumnMetadataFileBinding(
                    relative_path="outcomes.parquet",
                    module="outcomes",
                    identity_column="stay_id",
                    time_coordinates=(),
                    columns={
                        "death": metadata_binding(
                            "death", "death", ConceptColumnRole.EVENT_STATUS
                        )
                    },
                ),
            ),
        ),
    )
    (export / intake.NATIVE_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": intake.NATIVE_MANIFEST_SCHEMA_V2,
                "database": "miiv",
                "format": "parquet",
                "concept_selection": {
                    "mode": "explicit",
                    "modules": {"labs": list(lab_concepts), "outcomes": ["death"]},
                },
                "files": [
                    {
                        "file": "labs.parquet", "module": "labs",
                        "concepts": len(lab_concepts), "concept_ids": list(lab_concepts),
                        "rows": len(labs),
                        "column_metadata_columns": sorted(lab_binding.columns),
                    },
                    {
                        "file": "outcomes.parquet", "module": "outcomes",
                        "concepts": 1, "concept_ids": ["death"], "rows": len(outcomes),
                        "column_metadata_columns": ["death"],
                    },
                ],
                "feature_definitions": {"included": False},
                "column_metadata": reference.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    return export


def _authority():
    columns = [f"{c}__h{s}_{s + 12}" for c in COORDINATES for s in (0, 12)]
    stability = TrajectoryStabilitySpec(
        n_resamples=6,
        sample_fraction=0.8,
        base_seed=11,
        minimum_successful_resamples=6,
        refit_max_iter=300,
        refit_tolerance=1e-5,
        refit_regularization=1e-6,
        minimum_mean_stability=0.6,
        decision_mode="minimum_mean_threshold",
        refit_engine=DIAG_GMM_BEST_OF_10_ENGINE,
    )
    return build_trajectory_scientific_runtime_authority(
        {
            "schema_version": "easyicu.trajectory_scientific_runtime_authority/1",
            "protocol_content_sha256": "1" * 64,
            "coordinate_concepts": list(COORDINATES),
            "descriptive_only_concepts": [],
            "window_start_hours": 0,
            "window_end_hours": 24,
            "grid_width_hours": 12,
            "aggregation": "max",
            "representation_columns": columns,
            "minimum_available_windows": 2,
            "coordinate_scaling": {
                "method": "pooled_coordinate_wise_z_score",
                "ddof": 0,
                "observed_value_policy": "direct_or_owner_locf_available",
                "missing_value_policy": "preserve_missing_exclude_from_likelihood",
                "zero_variance_action": "fail_closed",
            },
            "evidence_state_policy": {
                "direct_observed": "include",
                "owner_locf_available": "include_and_audit",
                "unavailable": "exclude",
                "additional_clustering_stage_imputation": "none",
            },
            "representation_plan_method": "signed_fixed_window_trajectory_representation",
            "representation_plan_intent": (
                "Build the digest-bound fixed-window trajectory representation exactly as declared."
            ),
            "representation_plan_inputs": [],
            "representation_required_outputs": [
                "artifact:trajectory_representation",
                "table:trajectory_membership",
                "manifest:trajectory_representation_schema",
            ],
            "model_family": "latent_class_diagonal_gaussian_mixture",
            "fit_method": "observed_data_em_diagonal_gaussian_mixture",
            "covariance_type": "diag",
            "candidate_cluster_counts": [2, 3, 4],
            "selection_criterion": "bic",
            "selection_rule": "minimum",
            "candidate_fit_base_seed": 1729,
            "candidate_fit_max_iter": 500,
            "candidate_fit_tolerance": 1e-5,
            "candidate_fit_regularization": 1e-6,
            "bic_sample_size": "frozen_population_rows",
            "bic_parameter_count": "mixture_weights_k_minus_1_plus_2_k_per_coordinate",
            "bic_tie_break": "smaller_k",
            "upper_boundary_action": "fail_closed_if_selected_at_upper_boundary",
            "upper_boundary_reason_code": "NO_INTERIOR_BIC_OPTIMUM",
            "minimum_cluster_fraction": 0.05,
            "minimum_cluster_fraction_reason_code": "MINIMUM_CLUSTER_FRACTION_NOT_MET",
            "stability_spec": stability.model_dump(mode="json"),
        }
    )


def _run(tmp_path: Path, *, layout: str):
    export = _typed_export(tmp_path, layout=layout)
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        stem="universe",
        data_path=export,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=COORDINATES,
        outcome_concepts=("death",),
        emit_trajectory=True,
        trajectory_concepts=COORDINATES,
        trajectory_window=(0.0, 24.0),
    )
    authority = _authority()
    owners = authority.development_execution_only_plan(research_question=QUESTION)
    description = {
        "step_id": "outcome_by_class",
        "planned_analysis_role": "secondary",
        "intent": "Describe hospital mortality and age by frozen class.",
        "method": "descriptive_outcome_by_cluster",
        "scientific_action_id": "phenotyping.outcome_by_cluster",
        "inputs": ["stay_id", "death", "age", "artifact:analysis_cohort",
                   "table:cluster_assignments", "artifact:stability_freeze"],
        "expected_outputs": ["table:outcome_by_cluster"],
        "phenotype_comparison_spec": {
            "identity_column": "stay_id",
            "outcome_columns": ["death"],
            "variables": [
                {"name": "death", "variable_kind": "categorical", "summary": "count_percent",
                 "test": "none_descriptive_smd_only", "levels": [0, 1]},
                {"name": "age", "variable_kind": "continuous", "summary": "median_iqr",
                 "test": "none_descriptive_smd_only"},
            ],
        },
    }
    draft_payload = owners.model_dump(mode="json")
    draft = AnalysisPlan.model_validate(
        {**draft_payload, "steps": [*draft_payload["steps"], description]}
    )
    bound, findings = ScientificRuntimeAuthorities(
        trajectory=authority, current_case=None
    ).bind_plan(draft)
    carried = findings[0].detail["frozen_class_description"]
    locked = tmp_path / "locked_plan.json"
    locked.write_text(bound.model_dump_json(indent=2), encoding="utf-8")
    pipeline = ResearchAgentPipeline(
        config=PipelineConfig(
            workdir=tmp_path / "pipeline",
            development_diagnostic=True,
            development_locked_analysis_plan_path=locked,
            development_locked_analysis_plan_sha256=hashlib.sha256(
                locked.read_bytes()
            ).hexdigest(),
            trajectory_scientific_runtime_authority=authority.model_dump(mode="json"),
            scientific_runtime_projection_sha256="2" * 64,
            enable_memory=False,
            enable_replanning=False,
        ),
        services=PipelineServices(llm=ScriptedMockLLMClient([])),
    )
    pipeline.run(
        question=QUESTION,
        cohort=paths["parquet"],
        trajectory_path=paths["trajectory"],
        database="miiv",
        target_outcome="death",
        id_columns=["stay_id"],
        stop_after_analysis=True,
    )
    (run_dir,) = sorted((tmp_path / "pipeline").glob("run_*"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return carried, run_dir, manifest


def _records(manifest: dict) -> dict[str, dict]:
    return {record["step_id"]: record for record in manifest["per_step_records"]}


def _assert_the_host_drew_the_cohort_flow(manifest: dict, run_dir: Path) -> None:
    record = _records(manifest)[COHORT_FLOW_FIGURE_STEP]
    assert record["status"] == "ok"
    assert record["step_summary"]["method"] == "deterministic_cohort_flow_figure"
    assert record["step_summary"]["source_rows_consumed"] == 4
    # The figure exports exactly the representation owner's four flow rows.
    owner = pd.read_csv(
        run_dir / "steps" / "00_authority_compiled_trajectory_representation"
        / "outputs" / "cohort_flow.csv"
    )
    (source,) = (run_dir / "steps" / COHORT_FLOW_FIGURE_STEP / "outputs").glob(
        "*_source_data.csv"
    )
    table = pd.read_csv(source)
    assert len(owner) == 4
    assert dict(zip(table.metric, table.n)) == dict(zip(owner.metric, owner.n))


def _assert_rule_outcomes_and_designs_reach_the_report(
    manifest: dict, run_dir: Path, *, claims: dict[str, str],
) -> None:
    """Each rule's outcome is a host claim; each executed design binds in Methods."""

    records = manifest["per_step_records"]
    store = EvidenceStore(run_dir, enforcement_mode=EvidenceEnforcementMode.STRICT)
    found = {
        claim.claim_ref: getattr(claim.rule_outcome, "disposition", claim.rule_outcome.rule)
        for claim in store.authoritative_scientific_claims(records)
    }
    assert found == claims
    facts = store.manuscript_method_facts(records)
    design_facts = [fact for fact in facts if fact.text.startswith("Executed ")]
    assert [fact.source_field for fact in design_facts] == [
        "00_authority_compiled_trajectory_representation.executed_method_design",
        "01_authority_compiled_trajectory_candidates.executed_method_design",
    ]
    assert "12-hour windows from 0 to 24 hours after ICU admission" in design_facts[0].text
    scaffold = "## Methods\n\n### Variables\n\n" + "\n\n".join(
        fact.scaffold for fact in design_facts
    )
    safe, removed = store.enforce_evidence_bound_scaffold(scaffold, per_step_records=records)
    assert not removed
    bound = store.bind_manuscript(safe, per_step_records=records)
    _, _, untraced = bind_numeric_values(bound, evidence=store, per_step_records=records)
    assert not untraced


def _assert_every_step_ran_without_a_script(manifest: dict, run_dir: Path) -> None:
    assert manifest["readiness"]["failed_steps"] == []
    assert [
        finding for finding in manifest["findings"] if finding.get("severity") == "error"
    ] == []
    _assert_the_host_drew_the_cohort_flow(manifest, run_dir)


def test_frozen_classes_are_rendered_and_described_on_the_run_cohort(tmp_path):
    carried, run_dir, manifest = _run(tmp_path, layout="three_classes")

    assert carried["carried"] is True
    records = _records(manifest)
    assert all(records[step]["status"] == "ok" for step in SIGNED_AND_DESCRIPTION_STEPS)
    freeze = records["02_authority_compiled_trajectory_stability"]["step_summary"]
    assert freeze["freeze_status"] == TRAJECTORY_FROZEN_STATUS
    # Every refit keeps the best of ten starts, like the candidate fit it checks.
    stability_out = run_dir / "steps" / "02_authority_compiled_trajectory_stability" / "outputs"
    spec = json.loads((stability_out / "cluster_stability_spec.json").read_text(encoding="utf-8"))
    assert spec["executor_version"] == DIAG_GMM_BEST_OF_10_ENGINE
    attempts = json.loads(
        (stability_out / "cluster_stability_refit_attempts.json").read_text(encoding="utf-8")
    )["attempts"]
    assert attempts and all(
        attempt["executor_version"] == DIAG_GMM_BEST_OF_10_ENGINE
        and attempt["engine_starts"]["n_starts"] == 10
        for attempt in attempts
    )
    description = records["05_frozen_class_description"]["step_summary"]
    assert description["n_not_clustered"] == N_NOT_CLUSTERED
    assert description["n_rows"] == N_STAYS - N_NOT_CLUSTERED
    assert len(description["cluster_counts"]) == freeze["selected_n_clusters"]
    table = pd.read_csv(
        run_dir / "steps" / "05_frozen_class_description" / "outputs" / "outcome_by_cluster.csv"
    )
    assert table.group_missing_excluded_n.eq(N_NOT_CLUSTERED).all()
    stability_source = pd.read_csv(
        run_dir / "steps" / "03_authority_compiled_trajectory_selection_figure"
        / "outputs" / "trajectory_cluster_stability_source_data.csv"
    )
    assert not stability_source.empty
    _assert_every_step_ran_without_a_script(manifest, run_dir)
    _assert_rule_outcomes_and_designs_reach_the_report(manifest, run_dir, claims={
        "00_authority_compiled_trajectory_representation.observed_window_rule": (
            "minimum_observed_windows"
        ),
        "01_authority_compiled_trajectory_candidates.class_count_rule": "minimum_selected",
    })


def test_a_suite_without_an_interior_solution_describes_no_class(tmp_path):
    carried, run_dir, manifest = _run(tmp_path, layout="five_classes")

    assert carried["carried"] is True
    records = _records(manifest)
    assert all(records[step]["status"] == "ok" for step in SIGNED_AND_DESCRIPTION_STEPS)
    candidates = records["01_authority_compiled_trajectory_candidates"]["step_summary"]
    assert candidates["n_clusters"] == 4
    assert candidates["reason_code"] == "NO_INTERIOR_BIC_OPTIMUM"
    freeze = records["02_authority_compiled_trajectory_stability"]["step_summary"]
    assert freeze["freeze_status"] != TRAJECTORY_FROZEN_STATUS
    description = records["05_frozen_class_description"]["step_summary"]
    assert description["scientific_status"] == "failed_closed"
    assert description["reason_code"] == TRAJECTORY_NO_SOLUTION_REASON
    assert pd.read_csv(
        run_dir / "steps" / "05_frozen_class_description" / "outputs" / "outcome_by_cluster.csv"
    ).empty
    figure = records["03_authority_compiled_trajectory_selection_figure"]["step_summary"]
    assert figure["reportable_phenotype_solution"] is False
    _assert_every_step_ran_without_a_script(manifest, run_dir)
    # The formal no-solution result is reportable: each rule states its outcome.
    _assert_rule_outcomes_and_designs_reach_the_report(manifest, run_dir, claims={
        "00_authority_compiled_trajectory_representation.observed_window_rule": (
            "minimum_observed_windows"
        ),
        "01_authority_compiled_trajectory_candidates.class_count_rule": (
            "minimum_at_upper_boundary"
        ),
        "05_frozen_class_description.class_description_rule": "no_frozen_solution",
    })


def test_noise_stops_at_the_stability_gate_and_describes_no_class(tmp_path):
    """Noise can still minimise BIC inside the grid; its refits then disagree."""

    _carried, run_dir, manifest = _run(tmp_path, layout="noise")

    records = _records(manifest)
    candidates = records["01_authority_compiled_trajectory_candidates"]["step_summary"]
    assert candidates["scientific_status"] == "selected"
    freeze = records["02_authority_compiled_trajectory_stability"]["step_summary"]
    assert freeze["reason_code"] == "TRAJECTORY_STABILITY_BELOW_THRESHOLD"
    assert freeze["mean_adjusted_rand_index"] < 0.6
    assert freeze["freeze_status"] == "not_frozen_stability_threshold_failed"
    for step in (
        "03_authority_compiled_trajectory_selection_figure",
        "05_frozen_class_description",
    ):
        assert records[step]["status"] == "skipped_dependency_failed"
    assert not (
        run_dir / "steps" / "05_frozen_class_description" / "outputs" / "outcome_by_cluster.csv"
    ).exists()
    # The flow does not depend on the stability gate; the host still draws it.
    _assert_the_host_drew_the_cohort_flow(manifest, run_dir)
