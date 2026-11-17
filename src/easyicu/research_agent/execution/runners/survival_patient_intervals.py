"""Patient-level uncertainty for the signed suite's Kaplan-Meier RMST contrast."""

import numpy as np
from scipy.stats import norm

from ...methods.patient_cluster_survival import (
    has_repeated_patients,
    jackknife_covariance,
)
from ...methods.rmst import rmst


def clustered_rmst_table(frame, *, sealed, table, tau):
    if not has_repeated_patients(frame):
        return table.copy()

    def estimates(subset):
        values = []
        for level in (1, 0):
            group = subset.loc[subset[sealed.derived_exposure_column].eq(level)]
            if group.empty:
                raise ValueError("patient-deletion RMST loses an exposure group")
            values.append(
                rmst(
                    group[sealed.derived_time_column],
                    group[sealed.derived_event_column],
                    tau,
                ).rmst
            )
        return [*values, values[0] - values[1]]

    covariance = jackknife_covariance(frame, estimates)
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    result = table.copy()
    for index, prefix in enumerate(
        ("exposed_rmst", "comparator_rmst", "rmst_difference")
    ):
        value = float(result.loc[0, prefix + "_days"])
        low, high = (
            ("ci_low", "ci_high")
            if index == 2
            else (prefix + "_ci_low", prefix + "_ci_high")
        )
        result.loc[0, low] = value - 1.96 * se[index]
        result.loc[0, high] = value + 1.96 * se[index]
    if se[2] <= 0:
        raise ValueError("patient-deletion RMST contrast variance is not estimable")
    result.loc[0, "standard_error"] = se[2]
    result.loc[0, "p_value"] = 2 * norm.sf(
        abs(result.loc[0, "rmst_difference_days"] / se[2])
    )
    result["variance_method"] = "delete_patient_jackknife"
    return result
