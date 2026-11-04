"""Admission type from the hospital service active when an ICU stay starts.

MIMIC-IV ``services`` records each change of the responsible hospital service
with its ``transfertime``.  The service in charge at ICU admission is the last
change at or before ``icustays.intime`` within the same hospitalisation; a stay
that starts before the first recorded change takes that first service.

The generic hospital-table path assigned a service row to the stay whose
``outtime`` follows it and kept rows within 24 h before ``intime``: a service
chosen more than a day before the ICU stay, or a second ICU stay in the same
hospitalisation, had no admission type (17.6% of 3,000 stays; 2026-11-04
extraction audit round 3, MIIV adm).
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

SERVICE_AT_ADMISSION_CALLBACK = "service_at_admission"


def map_service_codes(values: pd.Series, service_map: Mapping[str, str]) -> pd.Series:
    """Map hospital service codes to admission types; an unmapped code is missing."""
    mapping = {str(k).upper(): v for k, v in service_map.items()}
    code = values.astype("string").str.strip().str.upper()
    return code.map(mapping).astype("object").where(lambda v: v.notna(), None)


def service_at_admission(
    stays: pd.DataFrame,
    services: pd.DataFrame,
    *,
    service_map: Mapping[str, str],
    stay_column: str,
    value_column: str,
) -> pd.DataFrame:
    """Return one admission type per stay from the service active at intime."""
    if not service_map:
        raise ValueError("service_at_admission requires a declared service_map")
    out = stays[[stay_column, "hadm_id", "intime"]].dropna(subset=["hadm_id", "intime"]).copy()
    out["intime"] = pd.to_datetime(out["intime"], errors="coerce").astype("datetime64[ns]")
    out = out.dropna(subset=["intime"])
    changes = services[["hadm_id", "transfertime", "curr_service"]].dropna(subset=["hadm_id", "transfertime"]).copy()
    changes["transfertime"] = pd.to_datetime(changes["transfertime"], errors="coerce").astype("datetime64[ns]")
    changes = changes.dropna(subset=["transfertime"])
    if out.empty or changes.empty:
        return pd.DataFrame({stay_column: pd.Series(dtype=out[stay_column].dtype), value_column: pd.Series(dtype="object")})
    out["hadm_id"] = out["hadm_id"].astype("int64")
    changes["hadm_id"] = changes["hadm_id"].astype("int64")
    active = pd.merge_asof(
        out.sort_values("intime"),
        changes.sort_values("transfertime"),
        left_on="intime", right_on="transfertime", by="hadm_id", direction="backward",
    )
    first = changes.sort_values("transfertime").drop_duplicates("hadm_id")[["hadm_id", "curr_service"]]
    active = active.merge(first.rename(columns={"curr_service": "_first_service"}), on="hadm_id", how="left")
    service = active["curr_service"].fillna(active["_first_service"])
    active[value_column] = map_service_codes(service, service_map)
    result = active.loc[active[value_column].notna(), [stay_column, value_column]]
    return result.sort_values(stay_column).reset_index(drop=True)


__all__ = ["SERVICE_AT_ADMISSION_CALLBACK", "map_service_codes", "service_at_admission"]
