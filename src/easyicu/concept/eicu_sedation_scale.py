"""Keep eICU ``Sedation Score`` rows that were charted on the RASS scale.

eICU ``nursecharting`` stores the sedation assessment as two rows charted at
the same ``nursingchartoffset``: ``Sedation Scale`` (the instrument: ``RASS``,
``SAS``, ``MAAS`` or ``Ramsay``) and ``Sedation Score`` (its number).  The
generic ``rass`` concept means the Richmond Agitation-Sedation Scale
(-5..+4), so a score is RASS only when the scale charted for the same stay
and offset is ``RASS``.  SAS (1-7), MAAS (0-6) and Ramsay (1-6) values read
as RASS would turn deep sedation into agitation.

Full eICU v2.0 (2026-11-03): 1,377,665 scores pair with RASS, 51,827 with
SAS, 14,682 with MAAS, 6,454 with Ramsay and one score has no scale row.  A
score without a scale row, or with conflicting scales at the same offset, is
dropped: the instrument is unknown.

The dictionary opts a source in with the callback :data:`RASS_SCALE_CALLBACK`.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

RASS_SCALE_CALLBACK = "eicu_rass_scale_only"
AUDIT_ATTR = "eicu_rass_scale_audit"

TABLE = "nursecharting"
STAY_COLUMN = "patientunitstayid"
OFFSET_COLUMN = "nursingchartoffset"
NAME_COLUMN = "nursingchartcelltypevalname"
VALUE_COLUMN = "nursingchartvalue"
SCALE_NAME = "Sedation Scale"
RASS_TOKEN = "RASS"
KEYS = (STAY_COLUMN, OFFSET_COLUMN)


class SedationScaleContractError(ValueError):
    """A source opted into the RASS-scale rule but cannot be evaluated."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


def scale_status(scales: pd.DataFrame) -> pd.DataFrame:
    """One row per (stay, offset) with ``_rass`` True when RASS is the only scale."""

    tokens = scales[VALUE_COLUMN].map(
        lambda v: "" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v).strip().upper()
    )
    frame = scales.loc[:, list(KEYS)].copy()
    frame["_recorded"] = tokens.ne("").to_numpy()
    frame["_other"] = (tokens.ne("") & tokens.ne(RASS_TOKEN)).to_numpy()
    grouped = frame.groupby(list(KEYS), dropna=True, sort=False)
    status = grouped[["_recorded", "_other"]].any().reset_index()
    status["_rass"] = status["_recorded"] & ~status["_other"]
    return status.loc[:, [*KEYS, "_rass"]]


def _load_scales(frame: pd.DataFrame, data_source) -> pd.DataFrame:
    from ..datasource import FilterOp, FilterSpec

    stays = pd.unique(frame[STAY_COLUMN].dropna())
    columns = [STAY_COLUMN, OFFSET_COLUMN, NAME_COLUMN, VALUE_COLUMN]
    if len(stays) == 0:
        return pd.DataFrame(columns=columns)
    table = data_source.load_table(
        TABLE,
        columns=columns,
        filters=[
            FilterSpec(NAME_COLUMN, FilterOp.IN, [SCALE_NAME]),
            FilterSpec(STAY_COLUMN, FilterOp.IN, list(stays)),
        ],
        verbose=False,
    )
    records = table.data if hasattr(table, "data") else table
    missing = [c for c in columns if c not in records.columns]
    if missing:
        raise SedationScaleContractError(
            "eicu_sedation_scale_columns_missing", f"{TABLE} lacks {missing}"
        )
    return records.loc[records[NAME_COLUMN].eq(SCALE_NAME)]


def _align(frame: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    status = status.copy()
    for key in KEYS:
        left = frame[key]
        if pd.api.types.is_timedelta64_dtype(left):
            status[key] = pd.to_timedelta(status[key], errors="coerce")
            if not pd.api.types.is_timedelta64_dtype(status[key]):
                status[key] = pd.to_timedelta(pd.to_numeric(status[key]), unit="m")
        elif pd.api.types.is_numeric_dtype(left):
            status[key] = pd.to_numeric(status[key], errors="coerce").astype(left.dtype)
    return status


def restrict_to_rass_scale(frame: pd.DataFrame, *, data_source, concept_name: str) -> pd.DataFrame:
    """Keep only sedation scores whose same-offset scale row is RASS."""

    missing = [column for column in KEYS if column not in frame.columns]
    if missing:
        raise SedationScaleContractError(
            "eicu_sedation_scale_keys_missing",
            f"concept={concept_name} missing={missing}",
        )
    if frame.empty:
        return frame
    if data_source is None:
        raise SedationScaleContractError("eicu_sedation_scale_data_source_missing")
    attrs = dict(frame.attrs)
    status = _align(frame, scale_status(_load_scales(frame, data_source)))
    keyed = frame.loc[:, list(KEYS)].reset_index(drop=True)
    joined = keyed.merge(status, on=list(KEYS), how="left", sort=False)
    if len(joined) != len(frame):
        raise SedationScaleContractError("eicu_sedation_scale_join_not_one_to_one")
    rass = joined["_rass"].fillna(False).astype(bool).to_numpy()
    kept = frame.loc[rass].copy()
    kept.attrs.update(attrs)
    audit = {
        "rows": int(len(frame)),
        "rass_rows": int(rass.sum()),
        "other_scale_rows_dropped": int(joined["_rass"].eq(False).sum()),
        "unpaired_rows_dropped": int(joined["_rass"].isna().sum()),
    }
    kept.attrs[AUDIT_ATTR] = audit
    logger.info("eICU RASS scale restriction concept=%s audit=%s", concept_name, audit)
    return kept


__all__ = [
    "RASS_SCALE_CALLBACK",
    "SedationScaleContractError",
    "restrict_to_rass_scale",
    "scale_status",
]
