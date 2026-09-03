"""Deterministic analysis-only adapter for cross-sectional phenotyping.

The Planner owns the feature roster.  This owner excludes typed outcomes,
identifiers and time coordinates, then fixes only median imputation,
standardisation, candidate-k scoring, clustering and resampling mechanics.
The assignment product retains the exact standardised matrix so downstream
selection and stability steps never reopen raw cohort bytes.
"""

from __future__ import annotations

import json
from pathlib import Path
import textwrap
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ...contracts.capability_ids import PHENOTYPING_ANALYSIS_KIND
from ...contracts.phenotyping_validation import (
    PhenotypingCompleteCaseReceipt,
    PhenotypingRuntimeReceipt,
)
from ...research_context.typed import parse_research_context_json
from ...robustness.panel import load_locked_robustness_specs
from ...schema import AnalysisStep
from .typed_input_binding import (
    load_typed_input,
    sha256_file,
    sole_typed_cohort_input,
)

PHENOTYPE_PROFILES_PRODUCT = "table:phenotype_profiles"
PHENOTYPE_ASSIGNMENTS_PRODUCT = "table:phenotype_assignments"
CLUSTER_SELECTION_PRODUCT = "table:cluster_selection"
CLUSTER_STABILITY_PRODUCT = "table:cluster_stability"

_PRIMARY_ACTION = "phenotyping.cluster_solution"
_ACTION_OUTPUTS = {
    _PRIMARY_ACTION: (PHENOTYPE_PROFILES_PRODUCT, PHENOTYPE_ASSIGNMENTS_PRODUCT),
    "phenotyping.k_selection": (CLUSTER_SELECTION_PRODUCT,),
    "phenotyping.cluster_stability": (CLUSTER_STABILITY_PRODUCT,),
}
_SEED = 1729
_COMPLETE_CASE_BOOTSTRAPS = 200
_FEATURE_PREFIX = "feature__"


def _raw_columns(step: AnalysisStep) -> tuple[str, ...]:
    return tuple(
        value
        for item in step.inputs
        if (value := str(item or "").strip()) and ":" not in value
    )


def cross_sectional_phenotyping_executor_owns_step(step: AnalysisStep) -> bool:
    action = str(step.scientific_action_id or "")
    expected = _ACTION_OUTPUTS.get(action)
    if expected is None or tuple(step.expected_outputs) != expected:
        return False
    typed = tuple(value for value in step.inputs if ":" in value)
    if action == _PRIMARY_ACTION:
        if (
            step.planned_analysis_role != "primary"
            or sole_typed_cohort_input(step) is None
            or len(_raw_columns(step)) < 2
        ):
            return False
    elif (
        step.planned_analysis_role not in {"secondary", "sensitivity", "auxiliary"}
        or typed != (PHENOTYPE_ASSIGNMENTS_PRODUCT,)
    ):
        return False
    return bool(
        step.table_one_spec is None
        and step.cohort_definition_spec is None
        and step.measurement_audit_spec is None
        and step.robustness_replay_spec is None
        and step.trajectory_stability_spec is None
        and not step.model_requirements
    )


def cross_sectional_phenotyping_consumed_input_keys(
    step: AnalysisStep,
) -> tuple[str, ...]:
    if step.scientific_action_id == _PRIMARY_ACTION:
        cohort = sole_typed_cohort_input(step)
        return (cohort,) if cohort else ()
    return (PHENOTYPE_ASSIGNMENTS_PRODUCT,)


def cross_sectional_phenotyping_executor_code(step: AnalysisStep) -> str:
    if not cross_sectional_phenotyping_executor_owns_step(step):
        raise ValueError("step is not owned by cross-sectional phenotyping")
    action = str(step.scientific_action_id)
    if action == _PRIMARY_ACTION:
        cohort = sole_typed_cohort_input(step)
        return textwrap.dedent(
            f"""
            import json
            import os
            from pathlib import Path
            from easyicu.research_agent.execution.runners.cross_sectional_phenotyping_executor import run_primary_phenotyping
            from easyicu.research_agent.execution.runners.typed_input_binding import load_step_cohort_frame

            frame, cohort_path = load_step_cohort_frame(typed_cohort_input={cohort!r})
            summary = run_primary_phenotyping(
                frame=frame,
                declared_columns={_raw_columns(step)!r},
                typed_cohort_input={cohort!r},
                source_cohort=cohort_path,
                out_dir=Path(os.environ["STEP_OUT_DIR"]),
                run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
                step_id={step.step_id!r},
            )
            print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
            """
        ).strip()
    return textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path
        from easyicu.research_agent.execution.runners.cross_sectional_phenotyping_executor import run_phenotyping_diagnostic

        summary = run_phenotyping_diagnostic(
            action_id={action!r},
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
        )
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        """
    ).strip()


def _candidate_scores(matrix: np.ndarray) -> tuple[list[dict[str, Any]], int]:
    if len(matrix) < 20:
        raise RuntimeError("phenotyping requires at least 20 rows")
    maximum = min(6, len(matrix) - 1)
    sample_size = min(10_000, len(matrix))
    rng = np.random.default_rng(_SEED)
    sample = np.sort(rng.choice(len(matrix), size=sample_size, replace=False))
    rows: list[dict[str, Any]] = []
    for k in range(2, maximum + 1):
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=_SEED,
            n_init=10,
            batch_size=min(2048, len(matrix)),
        )
        labels = model.fit_predict(matrix)
        score = float(silhouette_score(matrix[sample], labels[sample]))
        rows.append({"candidate_k": k, "silhouette": score, "inertia": float(model.inertia_)})
    selected = int(max(rows, key=lambda row: (row["silhouette"], -row["candidate_k"]))["candidate_k"])
    for row in rows:
        row["selected"] = bool(row["candidate_k"] == selected)
        row["selection_rule"] = "maximum_silhouette_then_lower_k"
    return rows, selected


def _paired_assignment_interval(
    reference: np.ndarray,
    sensitivity: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return ARI and a deterministic paired-assignment bootstrap interval."""

    point = float(adjusted_rand_score(reference, sensitivity))
    rng = np.random.default_rng(_SEED)
    bootstrap = np.empty(_COMPLETE_CASE_BOOTSTRAPS, dtype=float)
    for replicate in range(_COMPLETE_CASE_BOOTSTRAPS):
        indices = rng.integers(0, len(reference), size=len(reference))
        bootstrap[replicate] = adjusted_rand_score(
            reference[indices], sensitivity[indices]
        )
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return point, float(low), float(high), float(bootstrap.std(ddof=1))


def _fit_locked_complete_case_sensitivities(
    *,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    primary_labels: np.ndarray,
    primary_selected_k: int,
    out_dir: Path,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute only locked complete-case specs supported by this method owner.

    The lock decides which variables define completeness.  The primary feature
    representation stays fixed: features not named by the lock retain the
    primary median-imputation policy, so the owner neither widens the locked
    complete-case set nor silently changes the phenotype definition.
    """

    supported = []
    for spec in load_locked_robustness_specs(Path(run_dir)):
        override = spec.missing_override or {}
        if spec.axis == "missing" and override.get("strategy") == "complete_case":
            supported.append((spec, override))
    if not supported:
        return [], []

    feature_set = set(features)
    calculations: list[dict[str, Any]] = []
    for spec, override in supported:
        variables = tuple(str(value or "").strip() for value in override.get("variables", []))
        if not variables or any(not value for value in variables):
            raise RuntimeError(
                f"phenotyping complete-case spec {spec.spec_id!r} has no exact variable roster"
            )
        if len(set(variables)) != len(variables):
            raise RuntimeError(
                f"phenotyping complete-case spec {spec.spec_id!r} repeats variables"
            )
        outside = sorted(set(variables) - feature_set)
        if outside:
            raise RuntimeError(
                f"phenotyping complete-case spec {spec.spec_id!r} names variables "
                "outside the primary feature roster: " + ", ".join(outside)
            )
        complete_mask = frame.loc[:, variables].notna().all(axis=1).to_numpy()
        complete = frame.loc[complete_mask, features]
        if len(complete) < 20:
            raise RuntimeError(
                f"phenotyping complete-case spec {spec.spec_id!r} retains fewer than 20 rows"
            )
        # The lock applies complete-case deletion only to its exact variable
        # list. Any remaining primary feature follows the unchanged primary
        # median-imputation policy before scaling.
        imputed = SimpleImputer(strategy="median").fit_transform(complete)
        matrix = StandardScaler().fit_transform(imputed)
        candidates, selected_k = _candidate_scores(matrix)
        labels = MiniBatchKMeans(
            n_clusters=selected_k,
            random_state=_SEED,
            n_init=10,
            batch_size=min(2048, len(matrix)),
        ).fit_predict(matrix)
        point, low, high, standard_error = _paired_assignment_interval(
            np.asarray(primary_labels)[complete_mask], labels
        )
        calculations.append(
            {
                "spec_id": spec.spec_id,
                "axis": "missing",
                "missing_strategy": "complete_case",
                "complete_case_variables": list(variables),
                "primary_feature_roster": list(features),
                "n_total": int(len(frame)),
                "n_complete": int(len(complete)),
                "primary_selected_n_clusters": int(primary_selected_k),
                "complete_case_selected_n_clusters": int(selected_k),
                "complete_case_candidates": candidates,
                "comparison_metric": "adjusted_rand_index",
                "point_estimate": point,
                "ci_low": low,
                "ci_high": high,
                "standard_error": standard_error,
                "interval_method": "paired_assignment_bootstrap_percentile_95",
                "n_bootstrap": _COMPLETE_CASE_BOOTSTRAPS,
                "random_seed": _SEED,
                "primary_preprocessing": "median_imputation_then_standard_scaling",
                "sensitivity_preprocessing": "complete_case_then_standard_scaling",
                "clustering_method": "minibatch_kmeans",
                "outcome_used_for_fit": False,
                "causal_entity_claim_authorized": False,
                "paper_authorization_allowed": False,
            }
        )

    table_path = Path(out_dir) / "phenotyping_complete_case_sensitivity.csv"
    pd.DataFrame(
        [
            {
                key: value
                for key, value in calculation.items()
                if key
                not in {
                    "complete_case_candidates",
                    "complete_case_variables",
                    "primary_feature_roster",
                }
            }
            | {
                "complete_case_variables": json.dumps(
                    calculation["complete_case_variables"], separators=(",", ":")
                ),
                "primary_feature_roster": json.dumps(
                    calculation["primary_feature_roster"], separators=(",", ":")
                ),
                "complete_case_candidates": json.dumps(
                    calculation["complete_case_candidates"], separators=(",", ":")
                ),
            }
            for calculation in calculations
        ]
    ).to_csv(table_path, index=False)
    table_sha256 = sha256_file(table_path)
    receipts = [
        PhenotypingCompleteCaseReceipt(
            schema_version=(
                "easyicu.cross_sectional_phenotyping_complete_case_receipt/1"
            ),
            table_sha256=table_sha256,
            **calculation,
        ).model_dump(mode="json")
        for calculation in calculations
    ]
    rows = [
        {
            "spec_id": receipt["spec_id"],
            "axis": "missing",
            "n": receipt["n_complete"],
            "point_estimate": receipt["point_estimate"],
            "ci_low": receipt["ci_low"],
            "ci_high": receipt["ci_high"],
            "se": receipt["standard_error"],
            "evidence_id": "",
            "converged": True,
            "notes": (
                "Adjusted Rand index comparing primary assignments with a "
                "complete-case refit; interval is a paired assignment "
                "bootstrap and does not establish external reproducibility."
            ),
        }
        for receipt in receipts
    ]
    return receipts, rows


def _feature_roster(run_dir: Path, declared: tuple[str, ...], frame: pd.DataFrame) -> tuple[str, ...]:
    context = parse_research_context_json((Path(run_dir) / "research_context.json").read_text("utf-8"))
    descriptors = {item.name: item for item in context.variables}
    allowed_roles = {
        "demographic",
        "vital",
        "lab",
        "intervention",
        "ordinal_score",
        "composite_score",
        "other",
    }
    features = tuple(
        name
        for name in declared
        if name in frame.columns
        and name in descriptors
        and str(descriptors[name].role.value) in allowed_roles
        and pd.api.types.is_numeric_dtype(frame[name])
        and frame[name].notna().sum() >= 20
        and frame[name].nunique(dropna=True) >= 2
    )
    if len(features) < 2:
        raise RuntimeError("phenotyping requires at least two eligible numeric features")
    return features


def run_primary_phenotyping(
    *,
    frame: pd.DataFrame,
    declared_columns: tuple[str, ...],
    typed_cohort_input: str,
    source_cohort: Path,
    out_dir: Path,
    run_dir: Path,
    step_id: str,
) -> dict[str, Any]:
    features = _feature_roster(Path(run_dir), declared_columns, frame)
    imputed = SimpleImputer(strategy="median").fit_transform(frame.loc[:, features])
    matrix = StandardScaler().fit_transform(imputed)
    scores, selected_k = _candidate_scores(matrix)
    model = MiniBatchKMeans(
        n_clusters=selected_k,
        random_state=_SEED,
        n_init=10,
        batch_size=min(2048, len(matrix)),
    )
    labels = model.fit_predict(matrix)
    identity = str(parse_research_context_json((Path(run_dir) / "research_context.json").read_text("utf-8")).cohort.id_columns[0])
    if identity not in frame or frame[identity].isna().any():
        raise RuntimeError("phenotyping requires a complete typed row identity")
    assignments = pd.DataFrame({"unit_id": frame[identity].astype(str), "cluster": labels.astype(int)})
    for index, feature in enumerate(features):
        assignments[f"{_FEATURE_PREFIX}{feature}"] = matrix[:, index]
    profile_rows = []
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        for index, feature in enumerate(features):
            raw = pd.to_numeric(frame.loc[mask, feature], errors="coerce")
            profile_rows.append(
                {
                    "cluster": int(cluster),
                    "variable": feature,
                    "mean": float(raw.mean()),
                    "median": float(raw.median()),
                    "sd": float(raw.std(ddof=1)),
                    "standardised_centroid": float(matrix[mask, index].mean()),
                    "n": int(mask.sum()),
                    "missing_pct": float(raw.isna().mean() * 100.0),
                }
            )
    profiles = pd.DataFrame(profile_rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = out_dir / "phenotype_profiles.csv"
    assignments_path = out_dir / "phenotype_assignments.csv"
    profiles.to_csv(profiles_path, index=False)
    assignments.to_csv(assignments_path, index=False)
    complete_case_receipts, robustness_rows = _fit_locked_complete_case_sensitivities(
        frame=frame,
        features=features,
        primary_labels=labels,
        primary_selected_k=selected_k,
        out_dir=out_dir,
        run_dir=Path(run_dir),
    )
    selected_silhouette = next(
        row["silhouette"] for row in scores if row["selected"]
    )
    receipt = PhenotypingRuntimeReceipt(
        schema_version="easyicu.cross_sectional_phenotyping_runtime_receipt/1",
        analysis_kind=PHENOTYPING_ANALYSIS_KIND,
        owner=(
            "easyicu.research_agent.execution.runners."
            "cross_sectional_phenotyping_executor"
        ),
        n_rows=len(assignments),
        feature_roster=list(features),
        preprocessing="median_imputation_then_standard_scaling",
        clustering_method="minibatch_kmeans",
        random_seed=_SEED,
        candidates=scores,
        selected_n_clusters=selected_k,
        selected_silhouette_score=selected_silhouette,
        cluster_counts={
            str(int(cluster)): int(count)
            for cluster, count in assignments["cluster"].value_counts().items()
        },
        source_cohort_sha256=sha256_file(Path(source_cohort)),
        phenotype_profiles_sha256=sha256_file(profiles_path),
        phenotype_assignments_sha256=sha256_file(assignments_path),
        complete_case_sensitivities=complete_case_receipts,
        outcome_used_for_fit=False,
        downstream_outcome_use="descriptive_only",
        causal_entity_claim_authorized=False,
        external_reproducibility_established=False,
        paper_authorization_allowed=False,
    ).model_dump(mode="json")
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "method": "deterministic_cross_sectional_minibatch_kmeans",
        "analysis_family": "phenotyping",
        "deterministic_standard_analysis": PHENOTYPING_ANALYSIS_KIND,
        "authority_scope": "analysis_only",
        "paper_authorization_allowed": False,
        "selected_n_clusters": selected_k,
        "selected_silhouette_score": selected_silhouette,
        "cluster_selection": {
            "criterion": "silhouette_score",
            "selection_rule": "maximum_silhouette_then_lower_k",
            "direction": "maximize",
            "selected_n_clusters": selected_k,
            "candidates": scores,
        },
        "scientific_runtime_receipt": receipt,
        "robustness_rows": robustness_rows,
        "feature_roster": list(features),
        "source_cohort": str(Path(source_cohort).resolve()),
        "source_cohort_sha256": sha256_file(Path(source_cohort)),
        "source_inputs": [typed_cohort_input],
        "input_bindings": [{"input_key": typed_cohort_input, "loaded": True}],
        "output_files": {
            PHENOTYPE_PROFILES_PRODUCT: "phenotype_profiles.csv",
            PHENOTYPE_ASSIGNMENTS_PRODUCT: "phenotype_assignments.csv",
        },
    }
    (out_dir / "step_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def run_phenotyping_diagnostic(
    *,
    action_id: str,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
) -> dict[str, Any]:
    bound = load_typed_input(
        input_key=PHENOTYPE_ASSIGNMENTS_PRODUCT,
        run_dir=Path(run_dir),
        resolved_inputs=resolved_inputs,
        step_id=step_id,
        expected_declared_kind="table",
        expected_evidence_kind="table",
        require_consumption_contract=True,
        minimum_row_count=20,
    )
    feature_columns = tuple(column for column in bound.frame.columns if column.startswith(_FEATURE_PREFIX))
    if len(feature_columns) < 2 or "cluster" not in bound.frame:
        raise RuntimeError("phenotype assignments lack the sealed feature matrix or cluster labels")
    matrix = bound.frame.loc[:, feature_columns].to_numpy(dtype=float)
    reference = pd.to_numeric(bound.frame["cluster"], errors="coerce").to_numpy()
    if not np.isfinite(matrix).all() or not np.isfinite(reference).all():
        raise RuntimeError("phenotype assignments contain non-finite values")
    scores, selected_k = _candidate_scores(matrix)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if action_id == "phenotyping.k_selection":
        result = pd.DataFrame(scores)
        filename = "cluster_selection.csv"
        product = CLUSTER_SELECTION_PRODUCT
        result.to_csv(out_dir / filename, index=False)
        details = {
            "selected_n_clusters": selected_k,
            "cluster_selection": {
                "criterion": "silhouette_score",
                "selection_rule": "maximum_silhouette_then_lower_k",
                "selected_n_clusters": selected_k,
                "candidates": scores,
            },
        }
    elif action_id == "phenotyping.cluster_stability":
        selected_k = int(pd.Series(reference).nunique())
        alternative = GaussianMixture(
            n_components=selected_k,
            covariance_type="diag",
            random_state=_SEED,
            n_init=5,
            max_iter=500,
            reg_covar=1e-6,
        )
        alternative_labels = alternative.fit_predict(matrix)
        if not alternative.converged_:
            raise RuntimeError("alternative diagonal GMM did not converge")
        algorithm_agreement_ari = float(
            adjusted_rand_score(reference, alternative_labels)
        )
        rows = []
        for replicate in range(5):
            rng = np.random.default_rng(_SEED + replicate + 1)
            indices = np.sort(rng.choice(len(matrix), size=max(20, int(0.8 * len(matrix))), replace=False))
            labels = MiniBatchKMeans(n_clusters=selected_k, random_state=_SEED + replicate + 1, n_init=10, batch_size=min(2048, len(indices))).fit_predict(matrix[indices])
            rows.append(
                {
                    "replicate": replicate + 1,
                    "n": len(indices),
                    "adjusted_rand_index": float(
                        adjusted_rand_score(reference[indices], labels)
                    ),
                    "primary_algorithm": "minibatch_kmeans",
                    "alternative_algorithm": "diagonal_gaussian_mixture",
                    "algorithm_agreement_metric": "adjusted_rand_index",
                    "algorithm_agreement_ari": algorithm_agreement_ari,
                    "alternative_algorithm_converged": True,
                    "alternative_algorithm_seed": _SEED,
                }
            )
        mean_ari = float(np.mean([row["adjusted_rand_index"] for row in rows]))
        for row in rows:
            row["selected_n_clusters"] = selected_k
            row["mean_adjusted_rand_index"] = mean_ari
        result = pd.DataFrame(rows)
        filename = "cluster_stability_with_algorithm_agreement.csv"
        product = CLUSTER_STABILITY_PRODUCT
        result.to_csv(out_dir / filename, index=False)
        details = {
            "cluster_stability": {
                "selected_n_clusters": selected_k,
                "n_resamples": len(rows),
                "mean_adjusted_rand_index": mean_ari,
                "replicates": rows,
            },
            "algorithm_agreement": {
                "primary_algorithm": "minibatch_kmeans",
                "alternative_algorithm": "diagonal_gaussian_mixture",
                "selected_n_clusters": selected_k,
                "n": len(matrix),
                "metric": "adjusted_rand_index",
                "adjusted_rand_index": algorithm_agreement_ari,
                "alternative_algorithm_converged": True,
                "random_seed": _SEED,
                "outcome_used_for_fit": False,
                "authority_scope": "analysis_only",
                "external_reproducibility_established": False,
            },
        }
    else:
        raise RuntimeError("unsupported cross-sectional phenotyping action")
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "method": "deterministic_cross_sectional_phenotyping_diagnostic",
        "analysis_family": "phenotyping",
        "deterministic_standard_analysis": PHENOTYPING_ANALYSIS_KIND,
        "authority_scope": "analysis_only",
        "paper_authorization_allowed": False,
        "source_inputs": [PHENOTYPE_ASSIGNMENTS_PRODUCT],
        "input_bindings": [{"input_key": PHENOTYPE_ASSIGNMENTS_PRODUCT, "evidence_id": bound.evidence_id, "sha256": bound.sha256, "loaded": True}],
        "output_files": {product: filename},
        **details,
    }
    (out_dir / "step_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary


__all__ = [
    "CLUSTER_SELECTION_PRODUCT",
    "CLUSTER_STABILITY_PRODUCT",
    "PHENOTYPE_ASSIGNMENTS_PRODUCT",
    "PHENOTYPE_PROFILES_PRODUCT",
    "PHENOTYPING_ANALYSIS_KIND",
    "cross_sectional_phenotyping_consumed_input_keys",
    "cross_sectional_phenotyping_executor_code",
    "cross_sectional_phenotyping_executor_owns_step",
    "run_phenotyping_diagnostic",
    "run_primary_phenotyping",
]
