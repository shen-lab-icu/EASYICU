"""SICdb's own ``KDIGO_AKI_168`` field as a case-level native AKI label.

SICdb ships the result of its maintainers' KDIGO script for every case:
``data_ref`` rows with ``FieldID`` 3113 (d_references ``ProcessedFields``
``KDIGO_AKI_168``) whose ``RefID`` is one of the ``KDIGO_AKI`` references
3115-3118, the values ``0``-``3``.  It is the maximum stage over the first
168 hours of the ICU stay, so it uses information up to that point.

The label is therefore placed once per case, on the hour in which its window
closes: 168 hours after ICU admission, or the ICU discharge hour for a shorter
stay (``cases.TimeOfStay``).  It is never repeated onto earlier hourly rows,
where it would become a predictor built from the future.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

#: d_references ProcessedFields ``KDIGO_AKI_168``.
KDIGO_AKI_168_FIELD_ID = 3113
#: d_references ``KDIGO_AKI`` group: ReferenceValue "0".."3".
KDIGO_AKI_168_STAGE_BY_REF_ID = {3115: 0, 3116: 1, 3117: 2, 3118: 3}
#: The field's observation window, in hours after ICU admission.
KDIGO_AKI_168_WINDOW_HOURS = 168.0

NATIVE_STAGE = "aki_stage_source_native"
WINDOW_CLOSE_HOURS = "window_close_hours"


def sicdb_native_case_labels(
    data_ref: pd.DataFrame,
    cases: pd.DataFrame,
    *,
    patient_ids: Optional[Iterable[Any]] = None,
) -> pd.DataFrame:
    """Return ``CaseID``, the native stage and the hour its window closes.

    A ``RefID`` outside the documented four values, or two different stages
    for one case, leaves that case without a label rather than picking one.
    """
    ref = data_ref.loc[
        pd.to_numeric(data_ref["FieldID"], errors="coerce") == KDIGO_AKI_168_FIELD_ID,
        ["CaseID", "RefID"],
    ]
    stage = pd.to_numeric(ref["RefID"], errors="coerce").map(KDIGO_AKI_168_STAGE_BY_REF_ID)
    labels = pd.DataFrame({"CaseID": ref["CaseID"].to_numpy(), NATIVE_STAGE: stage.to_numpy()})
    labels = labels.dropna(subset=[NATIVE_STAGE]).drop_duplicates()
    conflicting = labels["CaseID"].duplicated(keep=False)
    labels = labels.loc[~conflicting]
    stays = cases[["CaseID", "TimeOfStay"]].drop_duplicates("CaseID")
    labels = labels.merge(stays, on="CaseID", how="inner", validate="one_to_one")
    los_hours = pd.to_numeric(labels.pop("TimeOfStay"), errors="coerce") / 3600.0
    close = np.floor(np.minimum(los_hours, KDIGO_AKI_168_WINDOW_HOURS))
    labels[WINDOW_CLOSE_HOURS] = close
    labels = labels.loc[labels[WINDOW_CLOSE_HOURS].ge(0)]
    if patient_ids is not None:
        labels = labels.loc[labels["CaseID"].isin(set(patient_ids))]
    labels[NATIVE_STAGE] = labels[NATIVE_STAGE].astype("Int64")
    return labels.reset_index(drop=True)


def load_sicdb_native_case_labels(
    data_path: Any,
    *,
    patient_ids: Optional[Iterable[Any]] = None,
) -> pd.DataFrame:
    """Read ``data_ref`` and ``cases`` of a prepared SICdb source."""
    from .outcomes import _raw_table

    data_ref = _raw_table("sic", data_path, "data_ref")
    cases = _raw_table("sic", data_path, "cases")
    return sicdb_native_case_labels(data_ref, cases, patient_ids=patient_ids)


__all__ = [
    "KDIGO_AKI_168_FIELD_ID",
    "KDIGO_AKI_168_STAGE_BY_REF_ID",
    "load_sicdb_native_case_labels",
    "sicdb_native_case_labels",
]
