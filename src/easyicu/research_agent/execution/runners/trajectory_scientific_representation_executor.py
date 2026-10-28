"""Deterministic fixed-grid representation owned by a signed trajectory contract."""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from patsy import dmatrix

from ...canonical_json import canonical_sha256
from ...contracts.trajectory_design import (DEATH_LIMITATION, RepeatedMeasureSpec, repeated_measure_spec_for_step, repeated_measure_step_matches)
from ...methods.gee import fit_gee
from ...methods.mixed_effects import fit_mixed_effects
from ...trajectory.panel import FixedWindowGrid
from .typed_input_binding import sole_typed_cohort_input

from ...authority.prespecified_rule_outcomes import (
    RULE_OUTCOME_SCHEMA_VERSION,
    RULE_OUTCOMES_KEY,
    rule_outcome_payload,
    validate_rule_outcome,
)
from ...contracts.executed_method_design import (
    EXECUTED_METHOD_DESIGN_KEY,
    EXECUTED_METHOD_DESIGN_SCHEMA_VERSION,
    executed_method_design_payload,
    validate_executed_method_design,
)
from ...contracts.trajectory_design import trajectory_eligibility_basis
from ...schema import AnalysisPlan, AnalysisStep
from ...trajectory.plan_contract import trajectory_step_roles
from ...trajectory.scientific_runtime_authority import (
    TrajectoryScientificRuntimeAuthority,
    load_trajectory_scientific_runtime_authority,
)
from ._shared import write_json as _write_json

_IDENTITY_COLUMN = "stay_id"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _study_cohort_ids(cohort_path: Path) -> pd.Series:
    """Return the locked study cohort's stay identities in cohort order.

    The panel holds rows only for stays with a value inside its window, and
    its bound universe may be wider than the analysis cohort, so the cohort,
    not the panel, decides who is counted.
    """

    if _IDENTITY_COLUMN not in pq.read_schema(cohort_path).names:
        raise ValueError(
            f"study cohort lacks the trajectory identity column {_IDENTITY_COLUMN}"
        )
    ids = pd.read_parquet(cohort_path, columns=[_IDENTITY_COLUMN])[_IDENTITY_COLUMN]
    if ids.empty or bool(ids.isna().any()) or bool(ids.duplicated().any()):
        raise ValueError("study cohort stay identities must be present and unique")
    return ids.reset_index(drop=True)


def trajectory_scientific_representation_executor_owns_step(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    authority: TrajectoryScientificRuntimeAuthority | Mapping[str, Any] | None,
) -> bool:
    if authority is None or trajectory_step_roles(step) != frozenset(
        {"representation"}
    ):
        return False
    owners = [
        item for item in plan.steps if "representation" in trajectory_step_roles(item)
    ]
    return owners == [step]


def trajectory_scientific_representation_executor_code(
    *,
    authority: TrajectoryScientificRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
) -> str:
    sealed = load_trajectory_scientific_runtime_authority(authority)
    authority_json = json.dumps(sealed.model_dump(mode="json"), sort_keys=True)
    return (
        "import json, os\n"
        "from pathlib import Path\n"
        "from easyicu.research_agent.execution.runners."
        "trajectory_scientific_representation_executor import "
        "run_trajectory_scientific_representation\n"
        f"authority = json.loads({json.dumps(authority_json)})\n"
        "run_trajectory_scientific_representation("
        "authority=authority, "
        f"runtime_projection_sha256={runtime_projection_sha256!r}, "
        "trajectory_path=Path(os.environ['TRAJECTORY_PARQUET']), "
        "cohort_path=Path(os.environ['COHORT_PARQUET']), "
        "out_dir=Path(os.environ['STEP_OUT_DIR']))\n"
    )


def run_trajectory_scientific_representation(
    *,
    authority: TrajectoryScientificRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    trajectory_path: Path,
    cohort_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Build the exact signed concept-by-window matrix without model imputation.

    Every study-cohort stay is counted: a stay without an owner-available
    value in the window has no observed window and is excluded by the
    window rule, and a panel stay outside the cohort is not in the study.
    """

    sealed = load_trajectory_scientific_runtime_authority(authority)
    if len(str(runtime_projection_sha256)) != 64:
        raise ValueError("runtime projection digest is required")
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectory = pd.read_parquet(trajectory_path)
    required = {
        "stay_id",
        "charttime",
        "concept",
        "value_num",
        "evidence_state",
        "owner_observed",
        "owner_available",
    }
    missing = sorted(required - set(trajectory.columns))
    if missing:
        raise ValueError(f"signed trajectory input lacks columns: {missing}")
    cohort_ids = _study_cohort_ids(cohort_path)
    # Match on the identity text, as the frozen class description does.
    cohort_keys = pd.Index(cohort_ids.astype(str), name=_IDENTITY_COLUMN)
    stay_keys = trajectory[_IDENTITY_COLUMN].astype(str)
    in_cohort = stay_keys.isin(cohort_keys)
    if not bool(in_cohort.any()):
        raise ValueError("signed trajectory panel shares no stay with the study cohort")
    panel_stays_outside_cohort_n = int(stay_keys.loc[~in_cohort].nunique())
    trajectory = trajectory.loc[in_cohort].assign(stay_key=stay_keys.loc[in_cohort])
    frame = trajectory.loc[
        trajectory["concept"].isin(sealed.coordinate_concepts)
        & trajectory["charttime"].ge(sealed.window_start_hours)
        & trajectory["charttime"].le(sealed.window_end_hours)
    ].copy()
    if frame.empty:
        raise ValueError("signed trajectory coordinates have no rows")
    frame["value_num"] = pd.to_numeric(frame["value_num"], errors="coerce")
    frame["owner_available"] = pd.to_numeric(frame["owner_available"], errors="coerce")
    frame["owner_observed"] = pd.to_numeric(frame["owner_observed"], errors="coerce")
    invalid_state = ~frame["evidence_state"].isin(
        ["direct_observed", "owner_locf_available", "unavailable"]
    )
    if bool(invalid_state.any()):
        raise ValueError("trajectory input contains an unknown owner evidence state")
    sofa_rows = frame["concept"].astype(str).str.startswith("sofa2")
    if bool(
        (
            sofa_rows
            & (
                ~frame["owner_available"].isin([0, 1])
                | ~frame["owner_observed"].isin([0, 1])
            )
        ).any()
    ):
        raise ValueError("SOFA-2 coordinate rows lack binary owner receipts")
    usable = (
        frame["owner_available"].eq(1)
        & frame["value_num"].notna()
        & ~frame["evidence_state"].eq("unavailable")
    )
    frame = frame.loc[usable].copy()
    if frame.empty:
        raise ValueError("no owner-available coordinate values remain")
    relative = frame["charttime"] - sealed.window_start_hours
    frame["window_index"] = np.floor(relative / sealed.grid_width_hours).astype(int)
    n_windows = (sealed.window_end_hours - sealed.window_start_hours) // (
        sealed.grid_width_hours
    )
    frame = frame.loc[frame["window_index"].between(0, n_windows - 1)].copy()
    frame["representation_column"] = [
        f"{concept}__h{sealed.window_start_hours + int(index) * sealed.grid_width_hours}_"
        f"{sealed.window_start_hours + (int(index) + 1) * sealed.grid_width_hours}"
        for concept, index in zip(frame["concept"], frame["window_index"], strict=True)
    ]
    matrix = (
        frame.groupby(["stay_key", "representation_column"], sort=False)["value_num"]
        .max()
        .unstack("representation_column")
        .reindex(
            index=cohort_keys,
            columns=list(sealed.representation_columns),
        )
    )
    matrix.index = pd.Index(cohort_ids.to_numpy(), name=_IDENTITY_COLUMN)
    # A window counts on owner-available SOFA-2 evidence when the design has
    # a SOFA-2 coordinate; a design without one counts the recorded values of
    # its coordinates, each a charted measurement by the design owner's rule.
    sofa_concepts = tuple(
        concept for concept in sealed.coordinate_concepts if concept.startswith("sofa2")
    )
    basis_concepts = trajectory_eligibility_basis(sealed.coordinate_concepts)
    sofa_columns = [
        column
        for column in sealed.representation_columns
        if any(column.startswith(f"{concept}__h") for concept in sofa_concepts)
    ]
    available_windows = pd.DataFrame(
        {
            index: matrix[
                [
                    f"{concept}__h{sealed.window_start_hours + index * sealed.grid_width_hours}_"
                    f"{sealed.window_start_hours + (index + 1) * sealed.grid_width_hours}"
                    for concept in basis_concepts
                ]
            ]
            .notna()
            .any(axis=1)
            for index in range(n_windows)
        }
    )
    observed_window_count = available_windows.sum(axis=1).astype(int)
    eligible = observed_window_count.ge(sealed.minimum_available_windows)
    membership = pd.DataFrame(
        {
            "stay_id": matrix.index,
            "observed_window_count": observed_window_count.to_numpy(),
            "meets_min_observed_windows": eligible.to_numpy(),
            "included_in_clustering": eligible.to_numpy(),
            "exclusion_reason": np.where(
                eligible,
                "",
                "fewer_than_signed_minimum_owner_available_sofa2_windows"
                if sofa_concepts
                else "fewer_than_signed_minimum_recorded_coordinate_windows",
            ),
        }
    )
    membership.to_csv(out_dir / "trajectory_membership.csv", index=False)
    pd.DataFrame(
        {
            "metric": [
                "input_cohort",
                "meets_min_observed_windows",
                "excluded_insufficient_windows",
                "included_in_clustering",
            ],
            "n": [
                len(matrix),
                int(eligible.sum()),
                int((~eligible).sum()),
                int(eligible.sum()),
            ],
        }
    ).to_csv(out_dir / "cohort_flow.csv", index=False)
    model_matrix = matrix.loc[eligible].reset_index()
    if len(model_matrix) <= max(sealed.candidate_cluster_counts):
        raise ValueError(
            "eligible trajectory population is too small for candidate grid"
        )
    if model_matrix[list(sealed.representation_columns)].isna().all(axis=1).any():
        raise ValueError("an eligible stay has no model coordinate")
    representation_path = out_dir / "trajectory_representation.parquet"
    model_matrix.to_parquet(representation_path, index=False)
    feature_rows = []
    for column in sealed.representation_columns:
        observed_n = int(model_matrix[column].notna().sum())
        feature_rows.append(
            {
                "feature": column,
                "observed_n": observed_n,
                "missing_n": int(len(model_matrix) - observed_n),
                "missing_fraction": float(1.0 - observed_n / len(model_matrix)),
            }
        )
    feature_frame = pd.DataFrame(feature_rows)
    feature_frame.to_csv(out_dir / "feature_availability.csv", index=False)
    feature_frame.to_csv(out_dir / "feature_missingness_heatmap.csv", index=False)
    authority_binding = {
        "schema_version": sealed.schema_version,
        "protocol_content_sha256": sealed.protocol_content_sha256,
        "execution_contract_sha256": sealed.execution_contract_sha256,
    }
    schema = {
        "schema_version": "easyicu.trajectory_representation_schema/2",
        "id_column": "stay_id",
        "observation_family": list(sealed.coordinate_concepts),
        "observation_columns": list(sealed.representation_columns),
        "min_observed_windows": sealed.minimum_available_windows,
        "profile_columns": list(sealed.representation_columns),
        "profile_summary_statistic": "mean",
        "time_axis": "relative_hours",
        "anchor": "icu_admission",
        "anchor_provenance": "task_contract",
        "anchor_source": "signed_runtime_scientific_projection",
        "source_window_contract": {
            "start_hours": sealed.window_start_hours,
            "end_hours": sealed.window_end_hours,
            "grid_width_hours": sealed.grid_width_hours,
            "aggregation": sealed.aggregation,
        },
        "trailing_na_policy": {
            "zero_imputation": False,
            "eligibility_uses_observed_window_count": True,
            "profile_summaries_ignore_missing": True,
        },
        "coordinate_scaling": sealed.scaling_payload,
        # Only the mixed-mode contract declares measurement; a Gaussian
        # contract's schema keeps its exact fields.
        **(
            {"coordinate_measurement": sealed.measurement_payload}
            if sealed.measurement_payload is not None
            else {}
        ),
        "evidence_state_policy": sealed.evidence_payload,
        "representation_columns": list(sealed.representation_columns),
        "frozen_population_n": len(model_matrix),
        "representation_sha256": _sha256(representation_path),
        "scientific_runtime_authority": authority_binding,
        "runtime_projection_sha256": runtime_projection_sha256,
        "descriptive_only_concepts": list(sealed.descriptive_only_concepts),
    }
    sealed.validate_representation_schema(schema)
    _write_json(out_dir / "trajectory_representation_schema.json", schema)
    window_manifest = {
        "schema_version": "easyicu.trajectory_window_manifest/1",
        "panel_product": "artifact:trajectory_representation",
        "families": [
            {
                "family": concept,
                "ordered_source_columns": [
                    {
                        "name": (
                            f"{concept}__h{start}_{start + sealed.grid_width_hours}"
                        ),
                        "window_start_hours": start,
                        "window_end_hours": start + sealed.grid_width_hours,
                    }
                    for start in range(
                        sealed.window_start_hours,
                        sealed.window_end_hours,
                        sealed.grid_width_hours,
                    )
                ],
            }
            for concept in sealed.coordinate_concepts
        ],
        "scientific_runtime_authority": authority_binding,
        "runtime_projection_sha256": runtime_projection_sha256,
    }
    _write_json(out_dir / "trajectory_window_manifest.json", window_manifest)
    state_counts = {
        str(key): int(value)
        for key, value in trajectory["evidence_state"].value_counts().items()
    }
    # The executed grid and its eligibility rule, as the report must state
    # them; the run context's window describes the cohort, not this panel.
    time_design = {
        "anchor": schema["anchor"],
        "window_start_hours": sealed.window_start_hours,
        "window_end_hours": sealed.window_end_hours,
        "window_width_hours": sealed.grid_width_hours,
        "n_windows": n_windows,
        "minimum_observed_windows": sealed.minimum_available_windows,
        # A window counts on SOFA-2 evidence only when the design has a
        # SOFA-2 coordinate, else on any recorded coordinate value
        # (``available_windows``).
        "window_evidence": (
            "any_available_sofa2_score" if sofa_concepts else "any_recorded_coordinate_value"
        ),
    }
    eligibility_outcome = validate_rule_outcome(
        {
            "schema_version": RULE_OUTCOME_SCHEMA_VERSION,
            "rule": "minimum_observed_windows",
            **time_design,
            "input_n": len(matrix),
            "included_n": int(eligible.sum()),
            "excluded_n": int((~eligible).sum()),
        }
    )
    representation_design = validate_executed_method_design(
        {
            "schema_version": EXECUTED_METHOD_DESIGN_SCHEMA_VERSION,
            "design_kind": "fixed_window_representation",
            **time_design,
            "window_aggregation": sealed.aggregation,
        }
    )
    summary = {
        "status": "ok",
        "id_column": "stay_id",
        "observation_family": list(sealed.coordinate_concepts),
        "observation_columns": list(sealed.representation_columns),
        "min_observed_windows": sealed.minimum_available_windows,
        "profile_columns": list(sealed.representation_columns),
        "profile_summary_statistic": "mean",
        "time_axis": "relative_hours",
        "anchor": "icu_admission",
        "anchor_provenance": "task_contract",
        "anchor_source": "signed_runtime_scientific_projection",
        "trailing_na_policy": schema["trailing_na_policy"],
        "coordinate_scaling": sealed.scaling_payload,
        "evidence_state_policy": sealed.evidence_payload,
        "representation_columns": list(sealed.representation_columns),
        "frozen_population_n": len(model_matrix),
        "eligible_n": int(eligible.sum()),
        "excluded_n": int((~eligible).sum()),
        "panel_stays_outside_study_cohort_n": panel_stays_outside_cohort_n,
        "owner_evidence_state_counts": state_counts,
        "scientific_runtime_authority": authority_binding,
        "runtime_projection_sha256": runtime_projection_sha256,
        "representation_sha256": schema["representation_sha256"],
        "sofa_coordinate_columns": sofa_columns,
        RULE_OUTCOMES_KEY: [rule_outcome_payload(eligibility_outcome)],
        EXECUTED_METHOD_DESIGN_KEY: executed_method_design_payload(
            representation_design
        ),
        "output_files": {
            "artifact:trajectory_representation": (
                "trajectory_representation.parquet"
            ),
            "table:trajectory_membership": "trajectory_membership.csv",
            "table:feature_availability": "feature_availability.csv",
            "table:cohort_flow": "cohort_flow.csv",
            "manifest:trajectory_representation_schema": (
                "trajectory_representation_schema.json"
            ),
            "manifest:trajectory_window_manifest": (
                "trajectory_window_manifest.json"
            ),
        },
    }
    _write_json(out_dir / "step_summary.json", summary)
    return summary


__all__ = [
    "run_trajectory_scientific_representation",
    "trajectory_scientific_representation_executor_code",
    "trajectory_scientific_representation_executor_owns_step",
]


# Repeated-measure growth and marginal models share the reviewed long-grid owner.
def _numeric(values: pd.Series, *, missing: bool = False) -> pd.Series:
    result = pd.to_numeric(values, errors="raise").astype(float)
    if (not missing and result.isna().any()) or np.isinf(result.to_numpy()).any():
        raise ValueError("repeated-measure coordinates must be finite")
    return result


def prepare_repeated_panel(frame: pd.DataFrame, spec: RepeatedMeasureSpec) -> pd.DataFrame:
    """Aggregate raw observations on the existing ICU grid, without filling gaps.

    The grid's final edge is closed. Bins are labelled by their left edge in
    the reviewed time unit; AR1 subsequently uses those bins as unit steps.
    Baseline covariates and the binary death indicator must be constant per
    analysis unit, so a time-varying confounder cannot become a baseline mean.
    """
    spec = RepeatedMeasureSpec.model_validate(spec.model_dump())
    if set(spec.required_columns) - set(frame.columns) or frame.columns.duplicated().any():
        raise ValueError("long panel is missing declared columns or has duplicate names")
    identities = frame[spec.identity_column]
    if identities.isna().any() or identities.astype(str).str.strip().eq("").any():
        raise ValueError("repeated-measure identity must be nonmissing")
    source = pd.DataFrame({
        "unit": identities, "time": _numeric(frame[spec.time_column]),
        "value": _numeric(frame[spec.outcome_column], missing=True),
    })
    if spec.family == "binomial" and not source.value.dropna().isin([0.0, 1.0]).all():
        raise ValueError("binomial repeated outcomes must be binary before aggregation")
    static = []
    for i, name in enumerate(spec.covariates):
        target = f"covariate_{i}"
        source[target] = _numeric(frame[name])
        static.append(target)
    if spec.death_column:
        source["death"] = _numeric(frame[spec.death_column])
        if not source.death.isin([0.0, 1.0]).all():
            raise ValueError("death indicator must be binary")
        static.append("death")
    if static and source.groupby("unit", sort=False)[static].nunique().gt(1).any().any():
        raise ValueError("baseline covariates and death status must be constant per unit")
    multiplier = 24.0 if spec.time_unit == "days" else 1.0
    # Reuse the grid's geometry; the reviewed value aggregation is applied below.
    grid = FixedWindowGrid(
        width_hours=spec.bin_width * multiplier,
        horizon_hours=(spec.window_end - spec.window_start) * multiplier,
    )
    hours = (source.time - spec.window_start) * multiplier
    source = source.loc[hours.between(0, grid.horizon_hours) & source.value.notna()].copy()
    source["bin"] = (hours.loc[source.index] // grid.width_hours).astype(int).clip(upper=len(grid.edges) - 1)
    source = source.sort_values("time", kind="stable")
    grouped = source.groupby(["unit", "bin"], sort=True, observed=True)
    prepared = grouped.value.agg(spec.aggregation).rename("value").to_frame()
    for name in static:
        prepared[name] = grouped[name].first()
    prepared = prepared.reset_index()
    prepared["time"] = spec.window_start + prepared.bin * spec.bin_width
    return prepared


def _design(panel: pd.DataFrame, spec: RepeatedMeasureSpec) -> pd.DataFrame:
    if spec.time_basis == "linear":
        design = pd.DataFrame({"time": panel.time.to_numpy()})
    else:
        design = dmatrix(
            "cr(time, knots=knots, lower_bound=lower, upper_bound=upper, constraints='center') - 1",
            {"time": panel.time.to_numpy(), "knots": spec.spline_knots,
             "lower": spec.window_start, "upper": spec.window_end}, return_type="dataframe",
        )
        design.columns = [f"time_spline_{i + 1}" for i in range(design.shape[1])]
    for i, _name in enumerate(spec.covariates):
        design[f"covariate_{i}"] = panel[f"covariate_{i}"].to_numpy()
    if np.linalg.matrix_rank(np.column_stack([np.ones(len(design)), design])) < design.shape[1] + 1:
        raise ValueError("repeated-measure fixed design is rank deficient")
    return design


def run_repeated_measures(*, frame: pd.DataFrame, spec: RepeatedMeasureSpec | Mapping, out_dir: Path) -> dict:
    spec = RepeatedMeasureSpec.model_validate(spec)
    panel = prepare_repeated_panel(frame, spec)
    populations = [("all_observed", panel)]
    if spec.dropout_sensitivity == "survivors":
        populations.append(("survivors", panel.loc[panel.death.eq(0)].copy()))
    effects, diagnostics, profiles, fits = [], [], [], []
    for population, selected in populations:
        if selected.empty or selected.groupby("unit", sort=False).size().max() < 2:
            raise ValueError(f"{population} has no repeated observed time bins")
        selected = selected.reset_index(drop=True)
        x = _design(selected, spec)
        y, groups = selected.value.to_numpy(), selected.unit.to_numpy()
        if spec.model == "gee":
            result = fit_gee(y, x, groups, family=spec.family, cov_struct=spec.working_correlation,
                             time=selected.bin.to_numpy() if spec.working_correlation == "ar1" else None,
                             ci_level=spec.ci_level)
        else:
            result = fit_mixed_effects(y, x, groups, ci_level=spec.ci_level,
                                      random_slope=selected.time.to_numpy() if spec.model == "mixed_slope" else None)
        for i, term in enumerate(result.var_names):
            if term.startswith("covariate_"):
                term = spec.covariates[int(term.removeprefix("covariate_"))]
            effects.append(dict(population=population, term=term, estimate=result.coef[i], se=result.se[i],
                                ci_low=result.ci_low[i], ci_high=result.ci_high[i], p=result.p[i]))
        stats = result.to_json()
        fits.append({"population": population, "statistics": stats})
        diagnostics.append(dict(
            population=population, model=spec.model, family=spec.family, n_obs=len(selected),
            n_units=selected.unit.nunique(), source_rows=len(frame),
            time_origin=spec.time_origin, time_unit=spec.time_unit, time_basis=spec.time_basis,
            window_start=spec.window_start, window_end=spec.window_end, bin_width=spec.bin_width,
            aggregation=spec.aggregation, working_correlation=spec.working_correlation,
            limitation=DEATH_LIMITATION + (
                " Survivor restriction changes the estimand and does not correct informative dropout."
                if spec.dropout_sensitivity == "survivors" else
                " Consider a joint longitudinal-survival model as a sensitivity; it is not fitted here."
            ),
            missing_measurements="Observed bins only; no imputation or post-death extrapolation.",
            qic_comparison="Compare QIC only on identical outcomes/rows using a common scale.",
            **{key: stats.get(key) for key in ("group_var", "random_slope_var", "intercept_slope_cov", "resid_scale",
                                              "icc", "icc_reference", "loglike", "qic", "qicu", "qic_scale",
                                              "qic_unavailable_reason", "cov_type", "converged")},
        ))
        profile = selected.groupby("time").value.agg(["count", "mean", "std"]).reset_index()
        profile.insert(0, "population", population)
        profiles.append(profile)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(effects).to_csv(out_dir / "repeated_measure_effects.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out_dir / "repeated_measure_diagnostics.csv", index=False)
    pd.concat(profiles, ignore_index=True).to_csv(out_dir / "repeated_measure_profile.csv", index=False)
    return {"status": "ok", "claim_ceiling": "analysis_only", "n_populations": len(populations), "limitation": DEATH_LIMITATION,
            "repeated_measure_receipt": {"schema_version": "easyicu.repeated_measure_result/1",
                "specification": spec.model_dump(mode="json"),
                "specification_sha256": canonical_sha256(spec.model_dump(mode="json")), "fits": fits}}


def repeated_measure_executor_code(step) -> str:
    if not repeated_measure_step_matches(step):
        raise ValueError("repeated-measure step needs its exact reviewed specification and products")
    spec = repeated_measure_spec_for_step(step)
    return textwrap.dedent(f"""
        import json
        import os
        from pathlib import Path
        from easyicu.research_agent.execution.runners.typed_input_binding import load_step_cohort_frame
        from easyicu.research_agent.execution.runners.trajectory_scientific_representation_executor import run_repeated_measures
        frame, source = load_step_cohort_frame(typed_cohort_input={sole_typed_cohort_input(step)!r})
        result = run_repeated_measures(frame=frame, spec={spec.model_dump(mode="json")!r}, out_dir=Path(os.environ["STEP_OUT_DIR"]))
        print(json.dumps(result, allow_nan=False))
    """).strip()
