"""What a native export's death time is, and whether an hour can be read from it.

The native export issues ``death_time`` beside the ``death`` status: hours from
ICU admission, derived per source.  Each derivation is a different physical
claim, so the export labels it in the outcome file's time-axis audit
(``files[].time_axis_audit.event_time_semantics``):

* MIMIC (III and IV): the death time the admission records;
* SICdb: the recorded offset of death, for a death at hospital discharge;
* AmsterdamUMCdb: the recorded date of death, for a death up to 72 h after
  ICU discharge, as a proxy;
* HiRID: kept, but labelled a proxy: its ricu-compatible death callback places
  a recorded death at the last observation of variables 110/200;
* eICU: indexed by the ICU discharge offset while its status is
  hospital-discharge mortality.  That coordinate is no time of death, so the
  export issues none (``structurally_unavailable``).

This module owns that vocabulary and the one reading a consumer needs from it:
whether a label's time is recorded to the hour or finer, so a window of hours
can be read by it, or is a date, a proxy, of unrecorded resolution or absent,
so no such window can.  Every label the producer writes is classified exactly
once; a new label needs its classification here, and has none by default.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Optional

__all__ = [
    "DEATH_STATUS",
    "DEATH_TIME_COMPANION",
    "DEATH_TIME_NOT_READ_TO_THE_HOUR",
    "DEATH_TIME_READ_TO_THE_HOUR",
    "NATIVE_EXPORT_DEATH_TIME_SEMANTICS",
    "SOURCE_EVENT_TIME",
    "STRUCTURALLY_UNAVAILABLE",
    "death_time_read_to_the_hour",
    "native_export_death_time_semantics",
]

#: The death status the native export issues, and the time it issues beside it.
DEATH_STATUS = "death"
DEATH_TIME_COMPANION = "death_time"

_RECORDED_DEATHTIME = "recorded_deathtime"
_RECORDED_OFFSET_OF_DEATH = "recorded_offset_of_death_for_hospital_discharge_death"
_DATE_OF_DEATH_PROXY = "recorded_dateofdeath_for_72h_post_icu_discharge_death_proxy"
_LAST_OBSERVATION_PROXY = "last_recorded_observation_proxy_for_dead_discharge"

#: The label for an export that names no database: the source's own event
#: time, whose resolution nothing records.
SOURCE_EVENT_TIME = "source_event_time"
#: The label for a named database whose export issues no death time.
STRUCTURALLY_UNAVAILABLE = "structurally_unavailable"

#: Per database, the death time the native export issues.
NATIVE_EXPORT_DEATH_TIME_SEMANTICS: Mapping[str, str] = MappingProxyType(
    {
        "miiv": _RECORDED_DEATHTIME,
        "miiv_demo": _RECORDED_DEATHTIME,
        "mimic": _RECORDED_DEATHTIME,
        "mimic_demo": _RECORDED_DEATHTIME,
        "aumc": _DATE_OF_DEATH_PROXY,
        "hirid": _LAST_OBSERVATION_PROXY,
        "sic": _RECORDED_OFFSET_OF_DEATH,
        "sic_demo": _RECORDED_OFFSET_OF_DEATH,
    }
)

#: Labels whose time is recorded to the hour or finer: a recorded timestamp,
#: or a recorded offset in seconds.
DEATH_TIME_READ_TO_THE_HOUR = frozenset(
    {_RECORDED_DEATHTIME, _RECORDED_OFFSET_OF_DEATH}
)
#: Labels whose time no window of hours can be read by: a date, a proxy, a
#: time of unrecorded resolution, or none.
DEATH_TIME_NOT_READ_TO_THE_HOUR = frozenset(
    {
        _DATE_OF_DEATH_PROXY,
        _LAST_OBSERVATION_PROXY,
        SOURCE_EVENT_TIME,
        STRUCTURALLY_UNAVAILABLE,
    }
)


def native_export_death_time_semantics(database: str) -> str:
    """The label the native export writes for ``database``."""

    normalized = str(database or "").strip().lower()
    if not normalized:
        return SOURCE_EVENT_TIME
    return NATIVE_EXPORT_DEATH_TIME_SEMANTICS.get(normalized, STRUCTURALLY_UNAVAILABLE)


def death_time_read_to_the_hour(label: str) -> Optional[bool]:
    """Whether a label's death time can be read to the hour; ``None`` for no label of ours."""

    if label in DEATH_TIME_READ_TO_THE_HOUR:
        return True
    if label in DEATH_TIME_NOT_READ_TO_THE_HOUR:
        return False
    return None
