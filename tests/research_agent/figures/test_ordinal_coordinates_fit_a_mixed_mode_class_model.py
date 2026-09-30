"""Declared ordinal coordinates are fitted as levels, not as Gaussians.

A SOFA-2 organ score takes whole levels 0-4.  The signed trajectory owner
z-scored every coordinate and fitted a diagonal Gaussian mixture, whose
likelihood keeps rising as narrow components settle on tied levels, so its BIC
can fall at every class count up to the grid's upper boundary and the
prespecified rule can only report that no interior solution exists.  On a
real ICU cohort it did exactly that.

The host now seals a mixed-mode latent class model when a coordinate is
declared ordinal: each ordinal indicator has class-specific level
probabilities, each continuous one a class-specific Gaussian on its pooled
z-score, and a missing indicator leaves the likelihood.  An all-continuous
design and every recorded authority keep the Gaussian contract byte for byte.
Synthetic data only.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.metrics import adjusted_rand_score

from easyicu.research_agent.contracts.executed_method_design import (
    validate_executed_method_design,
)
from easyicu.research_agent.contracts.sealed_suite_robustness import (
    sealed_suite_prespecified_axes,
)
from easyicu.research_agent.contracts.trajectory_design import (
    load_trajectory_design,
    sealed_trajectory_authority_body,
)
from easyicu.research_agent.execution.runners.trajectory_scientific_candidate_executor import (
    run_trajectory_scientific_candidate_selection,
)
from easyicu.research_agent.execution.runners.trajectory_stability_executor import (
    _fit_with_engine,
)
from easyicu.research_agent.trajectory.mixed_mode_latent_class import (
    fit_observed_data_mixed_mode_lca,
    mixed_mode_parameter_count,
)
from easyicu.research_agent.trajectory.plan_contract import (
    DIAG_GMM_BEST_OF_10_ENGINE,
    MIXED_MODE_LCA_BEST_OF_10_ENGINE,
    OBSERVED_DATA_DIAG_GMM_FIT_METHOD,
    OBSERVED_DATA_DIAG_GMM_METHOD,
    OBSERVED_DATA_DIAG_GMM_MODEL_FAMILY,
    OBSERVED_DATA_MIXED_MODE_LCA_METHOD,
    trajectory_role_result_findings,
)
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    TrajectoryScientificAuthorityError,
    build_trajectory_scientific_runtime_authority,
)

FIT = {"max_iter": 500, "tolerance": 1e-5, "regularization": 1e-6}
LEVELS = (0, 1, 2, 3, 4)
N_WINDOWS = 6
#: Execution-contract digests the host sealed before the mixed-mode model
#: existed; a Gaussian contract must keep them.
CONTINUOUS_DESIGN_DIGEST = "4fadcb129b2a5844505cf1cda7622e410735546a0945be287de788f5434d0ac0"
RECORDED_ORDINAL_DESIGN_DIGEST = (
    "6f32d88201e81a6f8db866a70dfb33e79e71daeb37405e253f2cf2059da0188c"
)


def _ordinal_classes(*, seed: int = 5, n: int = 600):
    """Three SOFA-like courses over six windows, as whole levels, 10% missing."""

    rng = np.random.default_rng(seed)
    means = np.asarray(
        [
            [[0.3] * N_WINDOWS, [0.2] * N_WINDOWS],
            [np.linspace(3.2, 1.0, N_WINDOWS), np.linspace(2.8, 0.8, N_WINDOWS)],
            [np.linspace(1.0, 3.5, N_WINDOWS), np.linspace(0.8, 3.2, N_WINDOWS)],
        ]
    )
    truth = rng.integers(0, 3, size=n)
    blocks = []
    for concept in range(2):
        centre = means[truth, concept, :]
        blocks.append(np.clip(np.rint(centre + rng.normal(0.0, 0.6, centre.shape)), 0, 4))
    x = np.concatenate(blocks, axis=1)
    x[rng.random(x.shape) < 0.1] = np.nan
    return x, truth


def _bic(fit: dict, parameters: int, n: int) -> float:
    return -2.0 * fit["final_log_likelihood"] + parameters * math.log(n)


def test_tied_levels_pull_a_gaussian_bic_to_the_boundary_but_not_the_class_model():
    x, truth = _ordinal_classes()
    n, d = x.shape
    z = (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)
    column_levels = [LEVELS] * d
    gaussian, mixed, labels = [], [], {}
    for k in range(2, 7):
        _labels, fit, _starts = _fit_with_engine(
            z, engine=DIAG_GMM_BEST_OF_10_ENGINE, n_components=k, seed=1729, **FIT
        )
        gaussian.append(_bic(fit, (k - 1) + 2 * k * d, n))
        labels[k], fit, starts = _fit_with_engine(
            x,
            engine=MIXED_MODE_LCA_BEST_OF_10_ENGINE,
            n_components=k,
            seed=1729,
            column_levels=column_levels,
            **FIT,
        )
        assert starts["n_starts"] == 10
        mixed.append(_bic(fit, mixed_mode_parameter_count(column_levels, k), n))

    # The Gaussian BIC keeps falling with the class count; three classes exist.
    assert all(later < earlier for earlier, later in zip(gaussian, gaussian[1:]))
    assert int(np.argmin(mixed)) + 2 == 3
    assert adjusted_rand_score(truth, labels[3]) > 0.8


def test_each_indicator_counts_its_own_free_parameters():
    # Weights 2; per class: 4 level probabilities, a mean and a variance, 2 levels.
    assert mixed_mode_parameter_count([LEVELS, None, (0, 1, 2)], 3) == 2 + 3 * (4 + 2 + 2)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda x, levels: (np.where(np.arange(x.size).reshape(x.shape) == 0, 2.5, x), levels),
         "outside its declared levels"),
        (lambda x, levels: (np.where(np.arange(x.shape[1]) == 0, np.nan, x), levels),
         "no observed values"),
        (lambda x, levels: (np.where(np.arange(x.shape[0])[:, None] == 0, np.nan, x), levels),
         "no observed indicator"),
        (lambda x, levels: (x, levels[:-1]), "one scale per indicator"),
        (lambda x, levels: (x, [(2, 1, 0)] + levels[1:]), "increasing"),
    ],
    ids=["between_levels", "empty_indicator", "empty_row", "unscaled_column", "unordered_levels"],
)
def test_the_engine_refuses_data_it_cannot_model(change, message):
    x, _truth = _ordinal_classes(n=60)
    changed, levels = change(x, [LEVELS] * x.shape[1])

    with pytest.raises(ValueError, match=message):
        fit_observed_data_mixed_mode_lca(
            changed, column_levels=levels, n_components=2, seed=1, **FIT
        )


def _design_body(concepts, *, clusters=(2, 4), window_end=24) -> dict:
    design = load_trajectory_design(
        {
            "coordinate_concepts": list(concepts),
            "window_start_hours": 0,
            "window_end_hours": window_end,
            "grid_width_hours": 12,
            "candidate_cluster_min": clusters[0],
            "candidate_cluster_max": clusters[1],
            "stability_resamples": 10,
        }
    )
    return sealed_trajectory_authority_body(design, protocol_content_sha256="a" * 64)


def _recorded_gaussian_body(body: dict) -> dict:
    """The body the host sealed for the same design before the class model."""

    recorded = {key: value for key, value in body.items() if key != "coordinate_measurement"}
    recorded.update(
        schema_version="easyicu.trajectory_scientific_runtime_authority/1",
        coordinate_scaling={
            **body["coordinate_scaling"],
            "method": "pooled_coordinate_wise_z_score",
        },
        model_family=OBSERVED_DATA_DIAG_GMM_MODEL_FAMILY,
        fit_method=OBSERVED_DATA_DIAG_GMM_FIT_METHOD,
        bic_parameter_count="mixture_weights_k_minus_1_plus_2_k_per_coordinate",
        stability_spec={**body["stability_spec"], "refit_engine": DIAG_GMM_BEST_OF_10_ENGINE},
    )
    return recorded


def test_the_host_seals_the_class_model_for_declared_ordinal_coordinates():
    body = _design_body(["sofa2_resp", "lactate"])
    authority = build_trajectory_scientific_runtime_authority(body)

    assert body["schema_version"] == "easyicu.trajectory_scientific_runtime_authority/2"
    assert authority.measurement_payload == [
        {"concept": "sofa2_resp", "scale": "ordinal", "levels": list(LEVELS)},
        {"concept": "lactate", "scale": "continuous"},
    ]
    assert authority.coordinate_scaling.method == "continuous_coordinate_wise_z_score"
    assert authority.model_family == "latent_class_mixed_mode"
    assert authority.stability_spec.refit_engine == MIXED_MODE_LCA_BEST_OF_10_ENGINE
    plan = authority.development_execution_only_plan(research_question="Which courses emerge?")
    methods = [step.method for step in plan.steps]
    assert OBSERVED_DATA_MIXED_MODE_LCA_METHOD in methods
    assert OBSERVED_DATA_DIAG_GMM_METHOD not in methods
    assert OBSERVED_DATA_MIXED_MODE_LCA_METHOD in authority.planning_contract_context()


def test_an_all_continuous_design_keeps_the_gaussian_contract_byte_for_byte():
    body = _design_body(["lactate", "map"])
    authority = build_trajectory_scientific_runtime_authority(body)

    assert "coordinate_measurement" not in body
    assert "coordinate_measurement" not in authority.model_dump(mode="json")
    assert authority.execution_contract_sha256 == CONTINUOUS_DESIGN_DIGEST
    assert authority.candidate_plan_method == OBSERVED_DATA_DIAG_GMM_METHOD


def test_a_recorded_gaussian_authority_for_ordinal_coordinates_still_verifies():
    recorded = _recorded_gaussian_body(_design_body(["sofa2_resp", "sofa2_cardio"]))

    authority = build_trajectory_scientific_runtime_authority(recorded)

    assert authority.execution_contract_sha256 == RECORDED_ORDINAL_DESIGN_DIGEST
    assert authority.model_dump(mode="json", exclude={"execution_contract_sha256"}) == recorded


def _without_measurement(body):
    body.pop("coordinate_measurement")


def _reordered_measurement(body):
    body["coordinate_measurement"] = list(reversed(body["coordinate_measurement"]))


def _continuous_measurement(body):
    body["coordinate_measurement"] = [
        {"concept": entry["concept"], "scale": "continuous"}
        for entry in body["coordinate_measurement"]
    ]


def _gaussian_engine(body):
    body["stability_spec"] = {**body["stability_spec"], "refit_engine": DIAG_GMM_BEST_OF_10_ENGINE}


def _pooled_scaling(body):
    body["coordinate_scaling"] = {
        **body["coordinate_scaling"], "method": "pooled_coordinate_wise_z_score"
    }


def _gaussian_rule(body):
    body["bic_parameter_count"] = "mixture_weights_k_minus_1_plus_2_k_per_coordinate"


def _one_level(body):
    body["coordinate_measurement"][0] = {**body["coordinate_measurement"][0], "levels": [0]}


def _continuous_with_levels(body):
    body["coordinate_measurement"][1] = {
        "concept": body["coordinate_measurement"][1]["concept"],
        "scale": "continuous",
        "levels": [0, 1],
    }


def _gaussian_version(body):
    body["schema_version"] = "easyicu.trajectory_scientific_runtime_authority/1"


@pytest.mark.parametrize(
    "mutate",
    [
        _without_measurement,
        _reordered_measurement,
        _continuous_measurement,
        _gaussian_engine,
        _pooled_scaling,
        _gaussian_rule,
        _one_level,
        _continuous_with_levels,
        _gaussian_version,
    ],
)
def test_the_class_model_contract_refuses_a_mismatched_body(mutate):
    body = _design_body(["sofa2_resp", "lactate"])
    mutate(body)

    with pytest.raises(ValidationError):
        build_trajectory_scientific_runtime_authority(body)


def test_a_gaussian_contract_declares_no_measurement():
    body = _design_body(["lactate", "map"])
    body["coordinate_measurement"] = [
        {"concept": "lactate", "scale": "continuous"},
        {"concept": "map", "scale": "continuous"},
    ]

    with pytest.raises(ValidationError, match="only the mixed-mode model"):
        build_trajectory_scientific_runtime_authority(body)


@pytest.mark.parametrize(
    ("model_family", "coordinate_scaling", "valid"),
    [
        ("latent_class_mixed_mode", "continuous_coordinate_wise_z_score", True),
        ("latent_class_diagonal_gaussian_mixture", "pooled_coordinate_wise_z_score", True),
        ("latent_class_mixed_mode", "pooled_coordinate_wise_z_score", False),
        ("latent_class_diagonal_gaussian_mixture", "continuous_coordinate_wise_z_score", False),
    ],
)
def test_an_executed_class_model_states_the_scaling_it_used(model_family, coordinate_scaling, valid):
    receipt = {
        "schema_version": "easyicu.executed_method_design/1",
        "design_kind": "latent_class_model",
        "model_family": model_family,
        "coordinate_scaling": coordinate_scaling,
        "candidate_class_counts": [2, 3, 4],
        "selection_criterion": "bic",
        "minimum_class_fraction": 0.05,
    }

    if valid:
        assert validate_executed_method_design(receipt).model_family == model_family
    else:
        with pytest.raises(ValidationError, match="disagree"):
            validate_executed_method_design(receipt)


def _bind(run_dir: Path, path: Path, evidence_id: str) -> dict[str, str]:
    return {
        "relative_path": str(path.relative_to(run_dir)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "evidence_id": evidence_id,
    }


def _run_candidates(run_dir: Path, body: dict, matrix: np.ndarray, *, schema_changes=None):
    """Run the signed candidate owner on a representation of ``matrix``."""

    authority = build_trajectory_scientific_runtime_authority(body)
    upstream = run_dir / "upstream"
    upstream.mkdir(parents=True)
    representation = pd.DataFrame(matrix, columns=list(authority.representation_columns))
    representation.insert(0, "stay_id", np.arange(1, len(matrix) + 1))
    representation_path = upstream / "trajectory_representation.parquet"
    representation.to_parquet(representation_path, index=False)
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
            "start_hours": authority.window_start_hours,
            "end_hours": authority.window_end_hours,
            "grid_width_hours": authority.grid_width_hours,
            "aggregation": "max",
        },
        "trailing_na_policy": {
            "zero_imputation": False,
            "eligibility_uses_observed_window_count": True,
            "profile_summaries_ignore_missing": True,
        },
        "coordinate_scaling": authority.scaling_payload,
        **(
            {"coordinate_measurement": authority.measurement_payload}
            if authority.measurement_payload is not None
            else {}
        ),
        "evidence_state_policy": authority.evidence_payload,
        "representation_columns": list(authority.representation_columns),
        "frozen_population_n": len(representation),
        "representation_sha256": hashlib.sha256(representation_path.read_bytes()).hexdigest(),
        "scientific_runtime_authority": {
            "schema_version": authority.schema_version,
            "protocol_content_sha256": authority.protocol_content_sha256,
            "execution_contract_sha256": authority.execution_contract_sha256,
        },
        "runtime_projection_sha256": "2" * 64,
    }
    if schema_changes is not None:
        schema_changes(schema)
    schema_path = upstream / "trajectory_representation_schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    out_dir = run_dir / "candidate"
    summary = run_trajectory_scientific_candidate_selection(
        authority=authority,
        runtime_projection_sha256="2" * 64,
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs={
            "inputs": {
                "artifact:trajectory_representation": _bind(
                    run_dir, representation_path, "signed-representation"
                ),
                "manifest:trajectory_representation_schema": _bind(
                    run_dir, schema_path, "signed-representation-schema"
                ),
            }
        },
    )
    models = json.loads((out_dir / "candidate_cluster_models.json").read_text(encoding="utf-8"))
    return summary, models


def test_the_candidate_owner_finds_the_interior_class_count_the_gaussian_misses(tmp_path):
    x, truth = _ordinal_classes()
    body = _design_body(["sofa2_resp", "sofa2_cardio"], clusters=(2, 6), window_end=72)

    mixed, mixed_models = _run_candidates(tmp_path / "mixed", body, x)
    gaussian, _models = _run_candidates(
        tmp_path / "gaussian", _recorded_gaussian_body(body), x
    )

    assert mixed["status"] == "ok"
    assert mixed["n_clusters"] == 3
    assert mixed["clustering_method"] == "observed_data_mixed_mode_latent_class"
    assert mixed_models["fit_engine"] == MIXED_MODE_LCA_BEST_OF_10_ENGINE
    assert [row["model_id"] for row in mixed_models["candidates"]] == [
        f"signed-observed-data-mixed-mode-lca-k{k}" for k in range(2, 7)
    ]
    assert [row["parameter_count"] for row in mixed_models["candidates"]] == [
        (k - 1) + k * 12 * 4 for k in range(2, 7)
    ]
    assignments = pd.read_csv(tmp_path / "mixed" / "candidate" / "candidate_cluster_assignments.csv")
    assert adjusted_rand_score(truth, assignments["candidate_cluster"]) > 0.8
    # The Gaussian contract on the same levels runs to the grid's upper boundary.
    assert gaussian["n_clusters"] == 6
    assert gaussian["reason_code"] == body["upper_boundary_reason_code"]


def test_a_value_between_levels_fails_the_candidate_owner(tmp_path):
    x, _truth = _ordinal_classes(n=120)
    x[0, 0] = 2.5
    body = _design_body(["sofa2_resp", "sofa2_cardio"], window_end=72)

    with pytest.raises(ValueError, match="outside its declared levels"):
        _run_candidates(tmp_path, body, x)


def test_a_representation_schema_without_the_measurement_is_refused(tmp_path):
    x, _truth = _ordinal_classes(n=120)
    body = _design_body(["sofa2_resp", "sofa2_cardio"], window_end=72)

    with pytest.raises(TrajectoryScientificAuthorityError, match="coordinate_measurement"):
        _run_candidates(
            tmp_path, body, x, schema_changes=lambda schema: schema.pop("coordinate_measurement")
        )


def test_the_class_model_owner_is_the_same_robustness_axis_as_the_gaussian_owner():
    axes = {}
    for concepts in (["sofa2_resp", "sofa2_cardio"], ["lactate", "map"]):
        authority = build_trajectory_scientific_runtime_authority(_design_body(concepts))
        plan = authority.development_execution_only_plan(research_question="Which courses emerge?")
        axes[authority.candidate_plan_method] = [
            sealed_suite_prespecified_axes(method=step.method, rule_refs=step.icu_rule_refs)
            for step in plan.steps
        ]

    assert axes[OBSERVED_DATA_MIXED_MODE_LCA_METHOD] == axes[OBSERVED_DATA_DIAG_GMM_METHOD]
    assert ("model_specification",) in axes[OBSERVED_DATA_MIXED_MODE_LCA_METHOD]


def test_a_class_model_candidate_schema_must_state_the_class_model(tmp_path):
    x, _truth = _ordinal_classes(n=120)
    body = _design_body(["sofa2_resp", "sofa2_cardio"], window_end=72)
    summary, _models = _run_candidates(tmp_path, body, x)
    plan = build_trajectory_scientific_runtime_authority(body).development_execution_only_plan(
        research_question="Which courses emerge?"
    )
    (step,) = [step for step in plan.steps if step.method == OBSERVED_DATA_MIXED_MODE_LCA_METHOD]
    out_dir = tmp_path / "candidate"

    def schema_issues() -> list[str]:
        return [
            issue
            for finding in trajectory_role_result_findings(
                step=step, step_summary=summary, out_dir=out_dir
            )
            if finding.detail.get("kind") == "trajectory_candidate_schema_incomplete"
            for issue in finding.detail["issues"]
        ]

    assert schema_issues() == []
    path = out_dir / "candidate_cluster_solution_schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(
        json.dumps(
            {
                **schema,
                "model_family": OBSERVED_DATA_DIAG_GMM_MODEL_FAMILY,
                "fit_method": OBSERVED_DATA_DIAG_GMM_FIT_METHOD,
            }
        ),
        encoding="utf-8",
    )
    assert schema_issues() == [
        "model_family does not match the declared observed-data method",
        "fit_method does not match the declared observed-data method",
    ]
