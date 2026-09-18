"""Covariance-consistent tests of a nested linear predictor space.

Scientific callers choose and fit the two designs on the same ordered rows.
This adapter derives restrictions in the full model's actual coordinates;
it never assumes the last spline coefficient alone represents nonlinearity.
"""

from __future__ import annotations

from typing import Any


def cluster_robust_nested_wald(*, fit: Any, restricted_design: Any) -> dict[str, Any]:
    import numpy as np
    from scipy.linalg import null_space

    if getattr(fit, "cov_type", None) != "cluster":
        raise ValueError("nested robust Wald requires a cluster covariance fit")
    full = np.asarray(fit.model.exog, dtype=float)
    restricted = np.asarray(restricted_design, dtype=float)
    if (
        full.ndim != 2
        or restricted.ndim != 2
        or full.shape[0] != restricted.shape[0]
        or not 0 < restricted.shape[1] < full.shape[1]
        or not np.isfinite(full).all()
        or not np.isfinite(restricted).all()
    ):
        raise ValueError("nested Wald designs must be finite with aligned dimensions")
    row_labels = getattr(fit.model.data, "row_labels", None)
    restricted_labels = getattr(restricted_design, "index", None)
    if row_labels is not None and restricted_labels is not None:
        if not np.array_equal(np.asarray(row_labels), np.asarray(restricted_labels)):
            raise ValueError(
                "nested Wald designs have different ordered row identities"
            )
    if (
        np.linalg.matrix_rank(full) != full.shape[1]
        or np.linalg.matrix_rank(restricted) != restricted.shape[1]
    ):
        raise ValueError("nested Wald designs must have full column rank")
    # Under the null, beta_full = embedding @ beta_restricted. The
    # orthogonal complement of this embedding is the complete joint null.
    embedding = np.linalg.lstsq(full, restricted, rcond=None)[0]
    if not np.allclose(full @ embedding, restricted, rtol=1e-9, atol=1e-10):
        raise ValueError("restricted predictor space is not nested in the full model")
    restriction = null_space(embedding.T).T
    degrees_of_freedom = full.shape[1] - restricted.shape[1]
    if restriction.shape != (degrees_of_freedom, full.shape[1]):
        raise ValueError("nested Wald restriction rank is inconsistent")
    covariance = np.asarray(fit.cov_params(), dtype=float)
    if (
        covariance.shape != (full.shape[1], full.shape[1])
        or not np.isfinite(covariance).all()
        or not np.allclose(covariance, covariance.T, rtol=1e-9, atol=1e-12)
    ):
        raise ValueError("nested Wald covariance must be finite and symmetric")
    restricted_covariance = restriction @ covariance @ restriction.T
    eigenvalues = np.linalg.eigvalsh(restricted_covariance)
    tolerance = np.finfo(float).eps * degrees_of_freedom * max(abs(eigenvalues))
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= tolerance):
        raise ValueError(
            "nested Wald restriction covariance is singular or nonpositive"
        )
    result = fit.wald_test(
        restriction,
        cov_p=covariance,
        use_f=False,
        scalar=True,
    )
    statistic, p_value = float(result.statistic), float(result.pvalue)
    if (
        not np.isfinite([statistic, p_value]).all()
        or statistic < 0
        or not 0 <= p_value <= 1
    ):
        raise ValueError("nested Wald returned an invalid statistic or probability")
    return {
        "method": "cluster_robust_nested_wald_chi2",
        "statistic": statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
    }
