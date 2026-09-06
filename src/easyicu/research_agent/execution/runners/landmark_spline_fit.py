"""Shared population, coding and covariance for landmark spline model fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...authority.current_case_scientific_runtime import LandmarkSplineRuntimeAuthority
from ...contracts.dependence import resolve_patient_groups

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class LandmarkModelPopulation:
    working: pd.DataFrame
    model_frame: pd.DataFrame
    alive_at_landmark: pd.Series
    under_observation: pd.Series
    valid_exposure: pd.Series
    primary_mask: pd.Series


def adjustment_design(frame: pd.DataFrame, authority: LandmarkSplineRuntimeAuthority) -> pd.DataFrame:
    import pandas as pd

    pieces = []
    for column in authority.required_adjustment_columns:
        source = frame[column]
        if column in authority.categorical_adjustment_columns:
            encoded = pd.get_dummies(
                source.astype("string"), prefix=column, drop_first=True, dtype=float
            )
            encoded.loc[source.isna(), :] = float("nan")
            if encoded.empty:
                raise ValueError(f"categorical adjustment {column!r} has fewer than two levels")
            pieces.append(encoded)
        else:
            pieces.append(pd.DataFrame({column: pd.to_numeric(source, errors="coerce")}, index=frame.index))
    return pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=frame.index)


def prepare_landmark_model_population(
    frame: pd.DataFrame, authority: LandmarkSplineRuntimeAuthority,
) -> LandmarkModelPopulation:
    import pandas as pd

    missing = sorted(set(authority.required_columns) - set(frame.columns))
    if missing:
        raise ValueError("signed landmark input lacks columns: " + ", ".join(missing))
    working = frame[list(authority.required_columns)].copy()
    for column in (authority.exposure_column, authority.outcome_column, authority.outcome_time_column, authority.observation_duration_column):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    outcome_values = set(working[authority.outcome_column].dropna().unique().tolist())
    if not outcome_values.issubset({0, 1}):
        raise ValueError("signed landmark outcome is not binary")
    if bool((working[authority.outcome_column].eq(1) & working[authority.outcome_time_column].isna()).any()):
        raise ValueError("signed landmark population cannot verify event timing for every death")
    alive = working[authority.outcome_column].eq(0) | working[authority.outcome_time_column].gt(authority.landmark_hours)
    observed = working[authority.observation_duration_column].ge(authority.observation_threshold)
    exposed = working[authority.exposure_column].notna()
    mask = alive & observed & exposed
    primary = working.loc[mask].copy()
    model_frame = pd.concat([
        primary[authority.exposure_column].rename("__exposure"),
        primary[authority.outcome_column].rename("__outcome"),
        adjustment_design(primary, authority),
    ], axis=1).dropna()
    if len(model_frame) < 30 or model_frame["__outcome"].nunique() != 2:
        raise ValueError("signed landmark primary population is not estimable")
    return LandmarkModelPopulation(working, model_frame, alive, observed, exposed, mask)


def fit_binomial_model(*, sm, outcome, design, source_frame, authority):
    """Fit with the exact declared patient-dependence covariance policy."""

    model = sm.GLM(outcome.astype(float), design, family=sm.families.Binomial())
    dependence = authority.dependence
    if dependence is None:
        return model.fit(maxiter=200, disp=0), None
    group_values = source_frame.loc[design.index, dependence.group_source]
    if bool(group_values.isna().any()):
        raise ValueError("signed landmark cluster group contains missing values")
    resolved = resolve_patient_groups(group_values.tolist(), requirement=dependence)
    fit = model.fit(maxiter=200, disp=0, cov_type="cluster", cov_kwds={"groups": list(resolved.groups)})
    return fit, resolved.cluster_count


def restricted_cubic_spline_basis(values: pd.Series, *, knots: tuple[float, float, float], prefix: str) -> pd.DataFrame:
    import numpy as np
    import patsy

    if not np.isfinite(knots).all() or not np.all(np.diff(knots) > 0):
        raise ValueError("signed landmark spline knots are not distinct")
    lower, middle, upper = knots
    basis = patsy.dmatrix(
        "cr(x, knots=(middle,), lower_bound=lower, upper_bound=upper, constraints='center') - 1",
        {"x": values.to_numpy(dtype=float), "middle": middle, "lower": lower, "upper": upper},
        return_type="dataframe",
    )
    basis.columns = [f"{prefix}_rcs_{index + 1}" for index in range(basis.shape[1])]
    basis.index = values.index
    return basis


def compare_covariate_functional_form(*, frame, authority, form, primary_diagnostics) -> dict[str, Any]:
    """Refit one covariate form, holding the primary RCS and rows constant."""

    import math
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy.stats import chi2

    from .nested_model_comparison import cluster_robust_nested_wald

    if form.target_column not in authority.required_adjustment_columns or form.target_column in authority.categorical_adjustment_columns:
        raise ValueError("functional-form refit target is not a continuous adjustment term")
    population = prepare_landmark_model_population(frame, authority)
    data = population.model_frame
    target = form.target_column
    exposure_knots = tuple(data["__exposure"].quantile(list(authority.spline_knot_quantiles)))
    target_knots = tuple(data[target].quantile(list(form.knot_quantiles)))
    exposure = restricted_cubic_spline_basis(data["__exposure"], knots=exposure_knots, prefix="exposure")
    covariate = restricted_cubic_spline_basis(data[target], knots=target_knots, prefix="target")
    other = data.drop(columns=["__exposure", "__outcome"])
    restricted = sm.add_constant(pd.concat([exposure, other], axis=1), has_constant="add")
    unrestricted = sm.add_constant(pd.concat([exposure, other.drop(columns=[target]), covariate], axis=1), has_constant="add")
    fits = []
    cluster_counts = []
    for design in (restricted, unrestricted):
        if not np.isfinite(design.to_numpy()).all() or np.linalg.matrix_rank(design) != design.shape[1]:
            raise ValueError("functional-form model design is non-finite or rank deficient")
        fit, clusters = fit_binomial_model(sm=sm, outcome=data["__outcome"], design=design, source_frame=population.working, authority=authority)
        if not bool(getattr(fit, "converged", False)):
            raise ValueError("functional-form model did not converge")
        fits.append(fit)
        cluster_counts.append(clusters)
    base_fit, full_fit = fits
    if cluster_counts[0] != cluster_counts[1]:
        raise ValueError("functional-form cluster populations drifted")
    n, events = len(data), int(data["__outcome"].sum())
    if (
        float(primary_diagnostics["n"]) != n or float(primary_diagnostics["events"]) != events
        or not math.isclose(float(primary_diagnostics["spline_aic"]), float(base_fit.aic), rel_tol=1e-8, abs_tol=1e-6)
    ):
        raise ValueError("functional-form refit does not reproduce the source primary model")
    df = int(full_fit.df_model - base_fit.df_model)
    if df <= 0:
        raise ValueError("functional-form alternative does not extend the linear covariate")
    if authority.dependence is not None:
        test = cluster_robust_nested_wald(fit=full_fit, restricted_design=restricted)
        statistic, p_value = test["statistic"], test["p_value"]
        if test["degrees_of_freedom"] != df:
            raise ValueError("functional-form restriction rank differs from model rank")
        method = test["method"]
        ic_basis = "working_independence_loglikelihood_descriptive_only"
    else:
        statistic = 2.0 * float(full_fit.llf - base_fit.llf)
        if statistic < -1e-7:
            raise ValueError("functional-form unrestricted likelihood is below its nested model")
        statistic = max(statistic, 0.0)
        p_value = float(chi2.sf(statistic, df))
        method, ic_basis = "nested_logistic_likelihood_ratio_test", "independent_loglikelihood"
    return {
        "method": method, "target_column": target, "statistic": float(statistic),
        "information_criteria_basis": ic_basis, "n_complete_case": n, "event_n": events,
        "linear_aic": float(base_fit.aic), "spline_aic": float(full_fit.aic),
        "linear_bic": float(-2 * base_fit.llf + len(base_fit.params) * math.log(n)),
        "spline_bic": float(-2 * full_fit.llf + len(full_fit.params) * math.log(n)),
        "additional_spline_parameters": df, "nonlinearity_p_value": p_value,
        "execution_mode": "covariate_refit_same_primary_population",
        "unchanged_primary_exposure": authority.exposure_column,
        "target_knots": _json_knots(target_knots),
        "primary_exposure_knots": _json_knots(exposure_knots),
        "cluster_count": cluster_counts[0],
    }


def _json_knots(knots: tuple[float, ...]) -> str:
    import json

    return json.dumps([float(value) for value in knots], allow_nan=False)
