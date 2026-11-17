"""Without-replacement subsamples of patients, retaining all their stays."""

import numpy as np
import pandas as pd

from ..contracts.dependence import resolve_patient_groups


def patient_units(ids, dependence):
    resolved = resolve_patient_groups(ids, requirement=dependence)
    groups = np.asarray(resolved.groups, dtype=object)
    return groups, pd.unique(groups)


def patient_subsample_positions(groups, units, *, sample_n, rng):
    selected = rng.choice(units, size=sample_n, replace=False)
    return np.flatnonzero(np.isin(groups, selected))
