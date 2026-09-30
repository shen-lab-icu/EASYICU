"""Signed trajectory fits keep the best of ten EM starts.

The observed-data diagonal-GMM engine ran one EM from one random balanced
start.  On a real ICU cohort its six-class candidate fit ended about 168,000
log-likelihood units below the five-class fit, which a converged optimum
cannot do, and the stability refits -- the same single start -- agreed with
the frozen solution at a mean adjusted Rand index of 0.377.  The stability
gate measured the optimizer as well as the data.

The best-of-10 engine runs ten deterministic starts, the first exactly the
single-start fit, and keeps the highest observed-data likelihood.  The host's
signed policy names it for the candidate fit and every stability refit; a spec
that names no engine keeps its recorded contract.  Synthetic data only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.metrics import adjusted_rand_score

from easyicu.research_agent.contracts.trajectory_design import (
    TRAJECTORY_HOST_POLICY,
    load_trajectory_design,
    sealed_trajectory_authority_body,
)
from easyicu.research_agent.execution.runners.trajectory_scientific_candidate_executor import (
    run_trajectory_scientific_candidate_selection,
)
from easyicu.research_agent.execution.runners.trajectory_stability_executor import (
    _fit_with_engine,
)
from easyicu.research_agent.schema import TrajectoryStabilitySpec
from easyicu.research_agent.trajectory.plan_contract import (
    DIAG_GMM_BEST_OF_10_ENGINE,
    DIAG_GMM_SINGLE_START_ENGINE,
)
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    build_trajectory_scientific_runtime_authority,
)

FIT = {"max_iter": 200, "tolerance": 1e-6, "regularization": 1e-6}


def _mixture(*, seed: int, n: int, d: int, k: int, separation: float):
    """Separated diagonal-Gaussian classes, 10% missing, scaled like the owner."""

    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, separation, size=(k, d))
    truth = rng.integers(0, k, size=n)
    x = centers[truth] + rng.normal(0.0, 1.0, size=(n, d))
    x[rng.random((n, d)) < 0.1] = np.nan
    return (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0), truth


def _five_classes():
    return _mixture(seed=1, n=300, d=8, k=5, separation=3.0)


def test_the_first_start_is_the_single_start_fit_and_the_best_is_never_worse():
    x, truth = _five_classes()

    single_labels, single, none = _fit_with_engine(
        x, engine=DIAG_GMM_SINGLE_START_ENGINE, n_components=5, seed=2, **FIT
    )
    labels, best, starts = _fit_with_engine(
        x, engine=DIAG_GMM_BEST_OF_10_ENGINE, n_components=5, seed=2, **FIT
    )

    assert none is None
    assert starts["n_starts"] == 10 and len(starts["starts"]) == 10
    assert starts["starts"][0] == {
        "seed": 2,
        "final_log_likelihood": single["final_log_likelihood"],
    }
    assert set(best) == set(single)
    # This single start stops in a poor local optimum; another start does not.
    assert best["final_log_likelihood"] > single["final_log_likelihood"] + 50
    assert starts["best_start_index"] != 0
    assert best["final_log_likelihood"] == max(
        start["final_log_likelihood"] for start in starts["starts"]
    )
    assert adjusted_rand_score(truth, labels) > 0.95
    assert adjusted_rand_score(truth, single_labels) < 0.8


def test_refits_agree_with_the_frozen_fit_once_the_optimum_is_reached():
    """The stability gate's mechanism: 80% refits against the full-data labels."""

    x, _truth = _five_classes()

    def refit_agreement(engine: str) -> list[float]:
        full, _trace, _starts = _fit_with_engine(
            x, engine=engine, n_components=5, seed=2, **FIT
        )
        scores = []
        for child in np.random.SeedSequence(11).spawn(10):
            seed = int(child.generate_state(1, dtype=np.uint32)[0])
            rows = np.sort(np.random.default_rng(seed).choice(300, size=240, replace=False))
            labels, _trace, _starts = _fit_with_engine(
                x[rows], engine=engine, n_components=5, seed=seed, **FIT
            )
            scores.append(adjusted_rand_score(full[rows], labels))
        return scores

    assert min(refit_agreement(DIAG_GMM_BEST_OF_10_ENGINE)) > 0.95
    assert min(refit_agreement(DIAG_GMM_SINGLE_START_ENGINE)) < 0.8


def test_the_same_seed_gives_the_same_fit():
    x, _truth = _five_classes()

    first = _fit_with_engine(
        x, engine=DIAG_GMM_BEST_OF_10_ENGINE, n_components=5, seed=7, **FIT
    )
    second = _fit_with_engine(
        x, engine=DIAG_GMM_BEST_OF_10_ENGINE, n_components=5, seed=7, **FIT
    )

    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1]
    assert first[2] == second[2]


def test_a_failed_start_is_recorded_and_only_every_start_failing_fails_the_fit():
    x, _truth = _mixture(seed=3, n=400, d=6, k=4, separation=1.6)

    _labels, best, starts = _fit_with_engine(
        x,
        engine=DIAG_GMM_BEST_OF_10_ENGINE,
        n_components=4,
        seed=0,
        max_iter=15,
        tolerance=1e-6,
        regularization=1e-6,
    )

    failed = [start for start in starts["starts"] if "error" in start]
    assert 1 <= starts["n_successful_starts"] <= 9
    assert len(failed) == 10 - starts["n_successful_starts"]
    assert all("did not converge" in start["error"] for start in failed)
    chosen = starts["starts"][starts["best_start_index"]]
    assert chosen["final_log_likelihood"] == best["final_log_likelihood"]

    with pytest.raises(ValueError, match=DIAG_GMM_BEST_OF_10_ENGINE) as caught:
        _fit_with_engine(
            x[:3], engine=DIAG_GMM_BEST_OF_10_ENGINE, n_components=4, seed=1, **FIT
        )
    assert "not larger than selected_n_clusters" in str(caught.value)


def test_an_unknown_engine_is_refused():
    x, _truth = _five_classes()

    with pytest.raises(ValueError, match="unsupported observed-data GMM engine"):
        _fit_with_engine(
            x, engine="easyicu_observed_data_diag_gmm_v3", n_components=5, seed=2, **FIT
        )


def test_a_spec_naming_no_engine_keeps_its_recorded_contract():
    """Recorded, reviewed and Planner-written specs name no engine; their
    digests must not move.  The host's signed policy names best of ten."""

    unnamed = TrajectoryStabilitySpec(n_resamples=10, sample_fraction=0.8)
    assert unnamed.refit_engine == DIAG_GMM_SINGLE_START_ENGINE
    named = TrajectoryStabilitySpec(
        n_resamples=10,
        sample_fraction=0.8,
        refit_engine=DIAG_GMM_BEST_OF_10_ENGINE,
    ).model_dump(mode="json")
    assert named["refit_engine"] == DIAG_GMM_BEST_OF_10_ENGINE
    assert TrajectoryStabilitySpec.model_validate(named).model_dump(mode="json") == named

    with pytest.raises(ValidationError):
        TrajectoryStabilitySpec.model_validate(
            {**named, "refit_engine": "easyicu_observed_data_diag_gmm_v3"}
        )


def _signed_body() -> dict:
    # Continuous coordinates keep the Gaussian mixture; declared ordinal ones
    # (SOFA-2 components) seal the mixed-mode latent class model instead.
    design = load_trajectory_design(
        {
            "coordinate_concepts": ["lactate", "map"],
            "window_start_hours": 0,
            "window_end_hours": 24,
            "grid_width_hours": 12,
            "candidate_cluster_min": 2,
            "candidate_cluster_max": 4,
            "stability_resamples": 10,
        }
    )
    return sealed_trajectory_authority_body(design, protocol_content_sha256="a" * 64)


def test_the_signed_authority_fits_and_refits_with_best_of_ten():
    authority = build_trajectory_scientific_runtime_authority(_signed_body())

    assert TRAJECTORY_HOST_POLICY["fit_engine"] == DIAG_GMM_BEST_OF_10_ENGINE
    assert authority.stability_spec.refit_engine == DIAG_GMM_BEST_OF_10_ENGINE


def test_a_recorded_single_start_authority_still_verifies():
    """An authority sealed before the engine existed keeps its exact digest."""

    body = _signed_body()
    body["stability_spec"] = {
        **body["stability_spec"],
        "refit_engine": DIAG_GMM_SINGLE_START_ENGINE,
    }

    authority = build_trajectory_scientific_runtime_authority(body)

    assert authority.stability_spec.refit_engine == DIAG_GMM_SINGLE_START_ENGINE
    assert authority.model_dump(mode="json", exclude={"execution_contract_sha256"}) == body


def _bind(run_dir: Path, path: Path, evidence_id: str) -> dict[str, str]:
    return {
        "relative_path": str(path.relative_to(run_dir)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "evidence_id": evidence_id,
    }


def _candidate_models(tmp_path: Path, body: dict) -> dict:
    """Run the signed candidate owner on three separated synthetic classes."""

    authority = build_trajectory_scientific_runtime_authority(body)
    run_dir = tmp_path
    upstream = run_dir / "upstream"
    upstream.mkdir(parents=True)
    rng = np.random.default_rng(44)
    labels = np.repeat([0, 1, 2], 40)
    centers = np.asarray(
        [[-6.0, -4.0, -2.0, -1.0], [0.0, 0.0, 0.0, 0.0], [6.0, 4.0, 2.0, 1.0]]
    )
    matrix = centers[labels] + rng.normal(0.0, 0.2, size=(len(labels), 4))
    representation = pd.DataFrame(matrix, columns=list(authority.representation_columns))
    representation.insert(0, "stay_id", np.arange(1, len(labels) + 1))
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
            "start_hours": 0,
            "end_hours": 24,
            "grid_width_hours": 12,
            "aggregation": "max",
        },
        "trailing_na_policy": {
            "zero_imputation": False,
            "eligibility_uses_observed_window_count": True,
            "profile_summaries_ignore_missing": True,
        },
        "coordinate_scaling": authority.scaling_payload,
        "evidence_state_policy": authority.evidence_payload,
        "representation_columns": list(authority.representation_columns),
        "frozen_population_n": len(representation),
        "representation_sha256": hashlib.sha256(
            representation_path.read_bytes()
        ).hexdigest(),
        "scientific_runtime_authority": {
            "schema_version": authority.schema_version,
            "protocol_content_sha256": authority.protocol_content_sha256,
            "execution_contract_sha256": authority.execution_contract_sha256,
        },
        "runtime_projection_sha256": "2" * 64,
    }
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
    assert summary["status"] == "ok", summary
    return json.loads((out_dir / "candidate_cluster_models.json").read_text(encoding="utf-8"))


def test_the_signed_candidate_fit_records_every_start(tmp_path: Path):
    best = _candidate_models(tmp_path / "best", _signed_body())
    recorded_body = _signed_body()
    recorded_body["stability_spec"] = {
        **recorded_body["stability_spec"],
        "refit_engine": DIAG_GMM_SINGLE_START_ENGINE,
    }
    single = _candidate_models(tmp_path / "single", recorded_body)

    assert best["fit_engine"] == DIAG_GMM_BEST_OF_10_ENGINE
    for candidate in best["candidates"]:
        starts = candidate["engine_starts"]
        assert starts["n_starts"] == 10
        assert candidate["final_log_likelihood"] == max(
            start["final_log_likelihood"]
            for start in starts["starts"]
            if "final_log_likelihood" in start
        )
    # The single-start record is the one a recorded authority always wrote.
    assert "fit_engine" not in single
    assert all("engine_starts" not in candidate for candidate in single["candidates"])
    for with_best, with_single in zip(best["candidates"], single["candidates"], strict=True):
        assert with_best["n_clusters"] == with_single["n_clusters"]
        assert with_best["final_log_likelihood"] >= with_single["final_log_likelihood"]
