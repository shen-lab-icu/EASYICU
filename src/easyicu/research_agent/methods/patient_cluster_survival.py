"""Patient grouping shared by the signed survival estimators.

The signed owner supplies grouping. It is never inferred from a value or a
column name. Model frames keep that vector through complete-case selection
and sensitivity restrictions; it is not a model covariate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..contracts.dependence import resolve_patient_groups

_GROUPS = "easyicu_patient_groups"


def bind_groups(model, source, dependence):
    if dependence is None:
        return
    values = source.loc[model.index, dependence.group_source]
    if values.isna().any():
        raise ValueError("patient grouping contains missing identities")
    resolved = resolve_patient_groups(values, requirement=dependence)
    model.attrs[_GROUPS] = dict(zip(model.index, resolved.groups, strict=True))


def groups_for(frame):
    mapping = frame.attrs.get(_GROUPS)
    if mapping is None:
        return None
    groups = np.asarray([mapping[index] for index in frame.index], dtype=object)
    if len(set(groups)) < 2:
        raise ValueError("patient-clustered inference requires at least two patients")
    return groups


def cox_fit_arguments(frame):
    """Lifelines' patient sandwich covariance, retaining the same Cox fit."""
    groups = groups_for(frame)
    if groups is None or len(pd.unique(groups)) == len(groups):
        return frame, {}
    key = "__host_patient_cluster"
    if key in frame.columns:
        raise ValueError("patient cluster column collides with a model term")
    return frame.assign(**{key: groups}), {"cluster_col": key, "robust": True}


def group_structure(frame):
    groups = groups_for(frame)
    if groups is None:
        return None
    counts = pd.Series(groups).value_counts()
    return {
        "rule": "keep_each_stay_cluster_by_patient",
        "n_patients": int(len(counts)),
        "n_stays": int(len(frame)),
        "patients_with_multiple_stays": int(counts.gt(1).sum()),
        "stay_count_distribution": {
            str(n): int(count) for n, count in counts.value_counts().items()
        },
    }


def has_repeated_patients(frame):
    """Use the original independent-stay estimator when every patient occurs once.

    A sandwich or delete-patient variance need not numerically equal the
    original model variance even with singleton clusters. Preserve that
    estimator on the actual analysis subset when there is no dependence.
    """
    groups = groups_for(frame)
    return groups is not None and len(pd.unique(groups)) < len(groups)


def jackknife_covariance(frame, estimate):
    """Delete one whole patient at a time; never delete individual stays.

    This fallback serves estimators without an analytic patient sandwich.
    Every deletion must be estimable. There is no successful-only variance.
    """
    groups = groups_for(frame)
    if groups is None:
        raise ValueError("patient jackknife requires bound grouping")
    unique = pd.unique(groups)
    values = np.asarray(
        [estimate(frame.loc[groups != group]) for group in unique], dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError("patient jackknife has a non-finite estimate")
    values = values.reshape(len(unique), -1)
    centered = values - values.mean(axis=0)
    return (len(unique) - 1) / len(unique) * centered.T @ centered


def clustered_ph_test(
    frame, *, duration_col, event_col, covariates, time_transform="km"
):
    """Scaled Schoenfeld residual trends with patient sandwich covariance.

    Regress each event residual on transformed event time with an intercept,
    using statsmodels' cluster covariance. The global test retains the
    existing conservative Bonferroni rule. Event-free patients remain in the
    fitted Cox risk sets; the residual regression uses patients with events.
    """
    from lifelines import CoxPHFitter, KaplanMeierFitter
    import statsmodels.api as sm

    groups = groups_for(frame)
    if groups is None:
        raise ValueError("clustered PH test requires grouping")
    if not has_repeated_patients(frame):
        from .ph_schoenfeld import ph_test

        return ph_test(
            frame,
            duration_col=duration_col,
            event_col=event_col,
            covariates=covariates,
            time_transform=time_transform,
        )
    columns = [duration_col, event_col, *covariates]
    source = frame[columns].copy()
    fitter = CoxPHFitter().fit(source, duration_col=duration_col, event_col=event_col)
    residuals = fitter.compute_residuals(source, kind="scaled_schoenfeld")
    event_times = source.loc[residuals.index, duration_col]
    if time_transform != "km":
        raise ValueError(
            "clustered PH diagnostic requires the sealed KM time transform"
        )
    km = KaplanMeierFitter().fit(source[duration_col], source[event_col])
    times = 1 - km.predict(event_times).to_numpy()
    if np.ptp(times) <= 0:
        raise ValueError("PH diagnostic has no event-time contrast")
    labels = pd.Series(groups, index=frame.index).loc[residuals.index].to_numpy()
    if len(set(labels)) < 2:
        raise ValueError("PH diagnostic needs events in at least two patients")
    rows = []
    for column in covariates:
        fit = sm.OLS(residuals[column].to_numpy(), sm.add_constant(times)).fit(
            cov_type="cluster",
            cov_kwds={"groups": labels},
            use_t=False,
        )
        p = float(fit.pvalues[1])
        if not np.isfinite(p):
            raise ValueError("patient-clustered PH trend is not estimable")
        rows.append(
            {
                "covariate": column,
                "test_statistic": float(fit.tvalues[1] ** 2),
                "p_value": p,
            }
        )
    best = min(rows, key=lambda row: row["p_value"])
    rows.append(
        {
            "covariate": "global",
            "test_statistic": best["test_statistic"],
            "p_value": min(1.0, len(rows) * best["p_value"]),
        }
    )
    return pd.DataFrame(rows).assign(
        variance_method="patient_cluster_sandwich_residual_trend"
    )


def clustered_interval_table(frame, table, estimate):
    covariance = jackknife_covariance(frame, estimate)
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    result = table.copy()
    beta = result["coefficient"].to_numpy(dtype=float)
    if len(se) != len(result) or np.any(se <= 0):
        raise ValueError("patient interval covariance is not estimable")
    result["standard_error"] = se
    result["ci_low"] = np.exp(beta - 1.96 * se)
    result["ci_high"] = np.exp(beta + 1.96 * se)
    result["p_value"] = 2 * norm.sf(np.abs(beta / se))
    result["variance_method"] = "delete_patient_jackknife"
    return result
