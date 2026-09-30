"""Signed owners state each prespecified rule's outcome and the design they ran.

The representation owner states its observed-window rule and time grid; the
candidate owner states its class-count rule, whichever way it falls, and its
model; a non-executable planned analysis states that it produced nothing.
Each block is typed at the owner, so an owner cannot publish an outcome its
own numbers contradict.  Synthetic panels: a SOFA-2 renal and lactate
trajectory over 0-48 h on an 8-hour grid, not the benchmark's design.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.scientific_claims import (
    derive_scientific_claim_drafts,
)
from easyicu.research_agent.execution.runners.feasibility_protocol_executor import (
    run_feasibility_protocol,
)
from easyicu.research_agent.execution.runners.trajectory_scientific_candidate_executor import (
    run_trajectory_scientific_candidate_selection,
)
from easyicu.research_agent.execution.runners.trajectory_scientific_representation_executor import (
    run_trajectory_scientific_representation,
)
from easyicu.research_agent.schema import TrajectoryStabilitySpec
from easyicu.research_agent.trajectory.plan_contract import DIAG_GMM_BEST_OF_10_ENGINE
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    build_trajectory_scientific_runtime_authority,
)

CONCEPTS = ("sofa2_renal", "lact")
STARTS = (0, 8, 16, 24, 32, 40)


def _authority(grid=(2, 3, 4, 5)):
    stability = TrajectoryStabilitySpec(
        n_resamples=2, sample_fraction=0.75, base_seed=5, minimum_successful_resamples=2,
        refit_max_iter=300, refit_tolerance=1e-5, refit_regularization=1e-6,
        minimum_mean_stability=0.0, decision_mode="minimum_mean_threshold",
        refit_engine=DIAG_GMM_BEST_OF_10_ENGINE,
    )
    return build_trajectory_scientific_runtime_authority({
        "schema_version": "easyicu.trajectory_scientific_runtime_authority/1",
        "protocol_content_sha256": "3" * 64,
        "coordinate_concepts": list(CONCEPTS),
        "descriptive_only_concepts": [],
        "window_start_hours": 0,
        "window_end_hours": 48,
        "grid_width_hours": 8,
        "aggregation": "max",
        "representation_columns": [f"{c}__h{s}_{s + 8}" for c in CONCEPTS for s in STARTS],
        "minimum_available_windows": 3,
        "coordinate_scaling": {
            "method": "pooled_coordinate_wise_z_score", "ddof": 0,
            "observed_value_policy": "direct_or_owner_locf_available",
            "missing_value_policy": "preserve_missing_exclude_from_likelihood",
            "zero_variance_action": "fail_closed",
        },
        "evidence_state_policy": {
            "direct_observed": "include", "owner_locf_available": "include_and_audit",
            "unavailable": "exclude", "additional_clustering_stage_imputation": "none",
        },
        "representation_plan_method": "signed_fixed_window_trajectory_representation",
        "representation_plan_intent": "Build the declared fixed-window representation.",
        "representation_plan_inputs": [],
        "representation_required_outputs": [
            "artifact:trajectory_representation", "table:trajectory_membership",
            "manifest:trajectory_representation_schema",
        ],
        "model_family": "latent_class_diagonal_gaussian_mixture",
        "fit_method": "observed_data_em_diagonal_gaussian_mixture",
        "covariance_type": "diag",
        "candidate_cluster_counts": list(grid),
        "selection_criterion": "bic",
        "selection_rule": "minimum",
        "candidate_fit_base_seed": 29,
        "candidate_fit_max_iter": 500,
        "candidate_fit_tolerance": 1e-5,
        "candidate_fit_regularization": 1e-6,
        "bic_sample_size": "frozen_population_rows",
        "bic_parameter_count": "mixture_weights_k_minus_1_plus_2_k_per_coordinate",
        "bic_tie_break": "smaller_k",
        "upper_boundary_action": "fail_closed_if_selected_at_upper_boundary",
        "upper_boundary_reason_code": "NO_INTERIOR_OPTIMUM",
        "minimum_cluster_fraction": 0.05,
        "minimum_cluster_fraction_reason_code": "SMALLEST_CLASS_TOO_SMALL",
        "stability_spec": stability.model_dump(mode="json"),
    })


def test_the_panel_owner_states_its_window_rule_and_grid(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    rows = []
    for stay in range(1, 41):
        # Stays 1-6 have renal scores in two windows only; lactate in every
        # window does not make them eligible.
        renal_windows = STARTS[:2] if stay <= 6 else STARTS
        for start in STARTS:
            time = start + 2.0
            if start in renal_windows:
                rows.append({"stay_id": stay, "charttime": time, "concept": "sofa2_renal",
                             "value_num": float(rng.integers(0, 5)),
                             "evidence_state": "direct_observed",
                             "owner_observed": 1, "owner_available": 1})
            rows.append({"stay_id": stay, "charttime": time, "concept": "lact",
                         "value_num": float(rng.gamma(2.0, 1.0)),
                         "evidence_state": "direct_observed",
                         "owner_observed": 1, "owner_available": 1})
    panel = tmp_path / "panel.parquet"
    pd.DataFrame(rows).to_parquet(panel, index=False)

    summary = run_trajectory_scientific_representation(
        authority=_authority(), runtime_projection_sha256="4" * 64,
        trajectory_path=panel, out_dir=tmp_path / "out",
    )

    flow = dict(pd.read_csv(tmp_path / "out" / "cohort_flow.csv").itertuples(index=False))
    [outcome] = summary["reportable_rule_outcomes"]
    assert outcome == {
        "schema_version": "easyicu.prespecified_rule_outcome/1",
        "rule": "minimum_observed_windows",
        "anchor": "icu_admission",
        "window_start_hours": 0,
        "window_end_hours": 48,
        "window_width_hours": 8,
        "n_windows": 6,
        "minimum_observed_windows": 3,
        "input_n": flow["input_cohort"],
        "included_n": flow["included_in_clustering"],
        "excluded_n": flow["excluded_insufficient_windows"],
    }
    assert (outcome["input_n"], outcome["included_n"]) == (40, 34)
    assert summary["executed_method_design"] == {
        "schema_version": "easyicu.executed_method_design/1",
        "design_kind": "fixed_window_representation",
        "anchor": "icu_admission",
        "window_start_hours": 0,
        "window_end_hours": 48,
        "window_width_hours": 8,
        "n_windows": 6,
        "window_aggregation": "max",
        "minimum_observed_windows": 3,
    }
    written = json.loads((tmp_path / "out" / "step_summary.json").read_text("utf-8"))
    [draft] = derive_scientific_claim_drafts(written)
    assert draft.claim_id == "observed_window_rule"


def _candidate_inputs(run_dir: Path, authority, matrix: np.ndarray) -> dict:
    upstream = run_dir / "upstream"
    upstream.mkdir()
    frame = pd.DataFrame(matrix, columns=list(authority.representation_columns))
    frame.insert(0, "stay_id", np.arange(1, len(frame) + 1))
    matrix_path = upstream / "trajectory_representation.parquet"
    frame.to_parquet(matrix_path, index=False)

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    schema = {
        "schema_version": "easyicu.trajectory_representation_schema/2",
        "id_column": "stay_id",
        "observation_family": list(authority.coordinate_concepts),
        "observation_columns": list(authority.representation_columns),
        "min_observed_windows": authority.minimum_available_windows,
        "profile_columns": list(authority.representation_columns),
        "profile_summary_statistic": "mean",
        "time_axis": "relative_hours",
        "anchor": "icu_admission",
        "anchor_provenance": "task_contract",
        "anchor_source": "signed_runtime_scientific_projection",
        "source_window_contract": {
            "start_hours": 0, "end_hours": 48, "grid_width_hours": 8, "aggregation": "max",
        },
        "trailing_na_policy": {
            "zero_imputation": False,
            "eligibility_uses_observed_window_count": True,
            "profile_summaries_ignore_missing": True,
        },
        "coordinate_scaling": authority.scaling_payload,
        "evidence_state_policy": authority.evidence_payload,
        "representation_columns": list(authority.representation_columns),
        "frozen_population_n": len(frame),
        "representation_sha256": sha(matrix_path),
        "scientific_runtime_authority": {
            "schema_version": authority.schema_version,
            "protocol_content_sha256": authority.protocol_content_sha256,
            "execution_contract_sha256": authority.execution_contract_sha256,
        },
        "runtime_projection_sha256": "4" * 64,
    }
    schema_path = upstream / "trajectory_representation_schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    return {"inputs": {
        key: {"relative_path": str(path.relative_to(run_dir)), "sha256": sha(path),
              "evidence_id": key.split(":", 1)[1]}
        for key, path in (
            ("artifact:trajectory_representation", matrix_path),
            ("manifest:trajectory_representation_schema", schema_path),
        )
    }}


def _classes(sizes: list[int], seed: int = 8) -> np.ndarray:
    """Separated classes; each class shifts every coordinate by its own offset."""

    rng = np.random.default_rng(seed)
    blocks = [
        offset * 4.0 + rng.normal(0.0, 0.3, size=(size, len(CONCEPTS) * len(STARTS)))
        for offset, size in enumerate(sizes)
    ]
    matrix = np.vstack(blocks)
    # Each coordinate also carries a class-specific sign so classes differ in shape.
    signs = np.where(np.arange(matrix.shape[1]) % 2 == 0, 1.0, -1.0)
    return matrix * signs


@pytest.mark.parametrize(
    ("sizes", "grid", "disposition", "status"),
    [
        pytest.param([60, 60, 60], (2, 3, 4, 5), "minimum_selected", "selected",
                     id="interior_minimum"),
        pytest.param([40] * 5, (2, 3, 4), "minimum_at_upper_boundary", "failed_closed",
                     id="upper_boundary"),
        pytest.param([100, 94, 6], (2, 3, 4, 5), "smallest_class_below_minimum",
                     "failed_closed", id="smallest_class_too_small"),
    ],
)
def test_the_class_count_owner_states_its_rule_whichever_way_it_falls(
    tmp_path: Path, sizes, grid, disposition, status,
) -> None:
    authority = _authority(grid)

    summary = run_trajectory_scientific_candidate_selection(
        authority=authority, runtime_projection_sha256="4" * 64,
        out_dir=tmp_path / "candidate", run_dir=tmp_path,
        resolved_inputs=_candidate_inputs(tmp_path, authority, _classes(sizes)),
    )

    [outcome] = summary["reportable_rule_outcomes"]
    assert summary["scientific_status"] == status
    assert outcome["disposition"] == disposition
    assert outcome["criterion_minimum_class_count"] == summary["n_clusters"]
    assert outcome["candidate_class_counts"] == list(grid)
    assert outcome["n_records"] == sum(sizes)
    assert outcome["smallest_class_fraction"] == summary["minimum_observed_cluster_fraction"]
    assert summary["executed_method_design"]["candidate_class_counts"] == list(grid)
    written = json.loads((tmp_path / "candidate" / "step_summary.json").read_text("utf-8"))
    [draft] = derive_scientific_claim_drafts(written)
    assert draft.claim_id == "class_count_rule"


@pytest.mark.parametrize(
    ("role", "claimed"),
    [("sensitivity", True), ("secondary", True), ("auxiliary", False)],
)
def test_a_non_executable_planned_analysis_states_that_it_produced_nothing(
    tmp_path: Path, role, claimed,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    summary = run_feasibility_protocol(
        out_dir=tmp_path / "out", run_dir=run_dir,
        resolved_inputs={"step_id": "08_repeat_lactate_protocol", "inputs": {}},
        step_id="08_repeat_lactate_protocol", planned_analysis_role=role,
        intent="Repeat the model on stays with two lactate measurements.",
        report_product="repeat_lactate_protocol", declared_inputs=[],
    )

    drafts = derive_scientific_claim_drafts(summary)
    if not claimed:
        assert "reportable_rule_outcomes" not in summary and drafts == []
        return
    [draft] = drafts
    assert draft.analysis_role == role
    # The Planner's free-text intent never enters the host sentence.
    assert "lactate" not in json.dumps(draft.model_dump(mode="json"))
