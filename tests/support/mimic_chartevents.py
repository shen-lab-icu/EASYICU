"""Synthetic MIMIC-III chartevents rows shared by the storage-integrity tests."""

from __future__ import annotations

import pandas as pd


def chart(rows, anchor: pd.Timestamp) -> pd.DataFrame:
    """Rows ``(row_id, subject_id, hadm_id, icustay_id, valuenum)`` charted ``row_id`` hours after ``anchor``."""
    return pd.DataFrame(
        {
            "row_id": [r[0] for r in rows], "subject_id": [r[1] for r in rows],
            "hadm_id": [r[2] for r in rows], "icustay_id": [r[3] for r in rows],
            "itemid": [580] * len(rows), "charttime": [anchor + pd.Timedelta(hours=r[0]) for r in rows],
            "storetime": [anchor] * len(rows), "cgid": [1] * len(rows), "value": [None] * len(rows),
            "valuenum": [r[4] for r in rows], "valueuom": ["kg"] * len(rows),
            "warning": [0] * len(rows), "error": [0] * len(rows),
            "resultstatus": [None] * len(rows), "stopped": [None] * len(rows),
        }
    )
