"""HiRID admission type from the APACHE diagnostic group.

HiRID has no admission service or specialty field.  It stores an APACHE II
group (observation 9990002, codes 98-114) or an APACHE IV group (9990004,
codes 190-206) for most stays (2026-11-03 extraction audit round 2, S8).  The
code table is Table 7 of the HiRID-ICU-Benchmark (Yèche et al., NeurIPS 2021
Datasets and Benchmarks, arXiv:2111.08536): APACHE II 107-114 and APACHE IV
199-205 are the surgical groups, every other listed code is non-surgical.

A stay can carry several codes, because the group is corrected during the
stay.  As in the benchmark, the APACHE II group is used when it maps and the
APACHE IV group otherwise, each taken as the last code charted within
``WINDOW`` of the stay's first APACHE record.  An unlisted code is missing.

Dictionary use::

    "ids": [9990002, 9990004], "table": "observations",
    "sub_var": "variableid", "callback": "hirid_apache_admission"
"""

from __future__ import annotations

from typing import Any

import pandas as pd

CALLBACK = "hirid_apache_admission"
APACHE_II = 9990002
APACHE_IV = 9990004
#: Code -> admission type (HiRID-ICU-Benchmark Table 7).
ADMISSION_TYPE = {
    APACHE_II: {**{code: "med" for code in range(98, 107)},
                **{code: "surg" for code in range(107, 115)}},
    APACHE_IV: {**{code: "med" for code in range(190, 199)},
                **{code: "surg" for code in range(199, 206)},
                206: "med"},
}
WINDOW = pd.Timedelta(hours=24)


def _elapsed(times: pd.Series, groups: pd.Series) -> pd.Series:
    """Time since the stay's first APACHE record, as a Timedelta."""
    if pd.api.types.is_numeric_dtype(times):  # relative hours
        times = pd.to_timedelta(times, unit="h")
    elif not pd.api.types.is_timedelta64_dtype(times):
        times = pd.to_datetime(times)
    return times - times.groupby(groups).transform("min")


def apache_admission_type(
    frame: pd.DataFrame, *, id_col: str, time_col: str, item_col: str, value_col: str
) -> pd.Series:
    """Per stay: ``'med'`` or ``'surg'`` (stays without a listed code absent)."""
    rows = frame[[id_col, time_col, item_col, value_col]].copy()
    rows["_item"] = pd.to_numeric(rows[item_col], errors="coerce")
    rows["_code"] = pd.to_numeric(rows[value_col], errors="coerce").round()
    rows["_type"] = [
        ADMISSION_TYPE.get(item, {}).get(code) if pd.notna(item) and pd.notna(code) else None
        for item, code in zip(rows["_item"], rows["_code"])
    ]
    rows["_elapsed"] = _elapsed(rows[time_col], rows[id_col])
    rows = rows.loc[rows["_type"].notna() & (rows["_elapsed"] <= WINDOW)]
    rows = rows.sort_values([id_col, "_elapsed"], kind="stable")
    last = rows.groupby([id_col, "_item"])["_type"].last().unstack()
    preferred = last.get(APACHE_II, pd.Series(index=last.index, dtype=object))
    fallback = last.get(APACHE_IV, pd.Series(index=last.index, dtype=object))
    return preferred.where(preferred.notna(), fallback).dropna().rename(None)


def apply_apache_admission(
    frame: pd.DataFrame, *, source: Any, concept_name: str, id_col: str, time_col: str
) -> pd.DataFrame:
    """Source callback: one admission-type row per stay, at its first APACHE record."""
    item_col = getattr(source, "sub_var", None) or "variableid"
    value_col = getattr(source, "val_var", None) or "value"
    if value_col not in frame.columns:
        value_col = concept_name
    if frame.empty:
        return frame.loc[:, [c for c in (id_col, time_col) if c in frame.columns]].assign(
            **{concept_name: pd.Series(dtype=object)}
        )
    types = apache_admission_type(
        frame, id_col=id_col, time_col=time_col, item_col=item_col, value_col=value_col
    )
    first = frame.groupby(id_col)[time_col].min()
    return pd.DataFrame({
        id_col: types.index.to_numpy(),
        time_col: first.reindex(types.index).to_numpy(),
        concept_name: types.to_numpy(),
    })


__all__ = [
    "ADMISSION_TYPE",
    "CALLBACK",
    "WINDOW",
    "apache_admission_type",
    "apply_apache_admission",
]
