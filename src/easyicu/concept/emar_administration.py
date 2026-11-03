"""MIMIC-IV eMAR rows that record a dose actually given.

``emar.event_txt`` records what happened to a scheduled dose.  A drug-exposure
concept read from eMAR keeps only the events that give the drug; "Not Given",
"Hold Dose", "Stopped", "Not Started" and similar rows are not exposure.

Dictionary use::

    "table": "emar", "sub_var": "medication", "regex": "...",
    "class_name": "rgx_itm", "extra_vars": ["event_txt"],
    "callback": "emar_administered"
"""

from __future__ import annotations

import pandas as pd

CALLBACK = "emar_administered"
EVENT_COLUMN = "event_txt"
#: MIMIC-IV emar.event_txt values that give the drug (full source 2026-11-03).
ADMINISTERED_EVENTS = frozenset(
    {
        "Administered",
        "Delayed Administered",
        "Partial Administered",
        "Administered in Other Location",
        "Started",
        "Delayed Started",
        "Restarted",
    }
)


def keep_administered(frame: pd.DataFrame, *, concept_name: str) -> pd.DataFrame:
    """Keep administered eMAR events and mark them ``TRUE``."""
    if EVENT_COLUMN not in frame.columns:
        raise ValueError("emar_event_txt_missing: declare extra_vars ['event_txt']")
    given = frame[EVENT_COLUMN].astype("string").str.strip().isin(ADMINISTERED_EVENTS)
    out = frame.loc[given.fillna(False).to_numpy()].copy()
    out[concept_name] = True
    return out


__all__ = ["ADMINISTERED_EVENTS", "CALLBACK", "keep_administered"]
