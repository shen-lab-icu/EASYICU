"""Every death-time label the native export writes says whether an hour can be read from it.

The native export labels the death time it issues beside the death status by
how its source records it (``easyicu.utils.death_time_semantics``): a recorded
time, a recorded offset, a date, a proxy, or none.  A consumer reads a window
of hours by that time only when the label says it was recorded to the hour, so
every label the producer can write must be classified, exactly once, and a new
label needs its classification before any consumer can read it.
"""

from __future__ import annotations

import pytest

from easyicu.api import extraction
from easyicu.utils import death_time_semantics as owner


def _labels_the_producer_writes() -> set[str]:
    named = {
        owner.native_export_death_time_semantics(database)
        for database in [
            *owner.NATIVE_EXPORT_DEATH_TIME_SEMANTICS,
            "eicu",
            "eicu_demo",
            "unknown_db",
        ]
    }
    return {
        *owner.NATIVE_EXPORT_DEATH_TIME_SEMANTICS.values(),
        *named,
        owner.SOURCE_EVENT_TIME,
    }


def test_every_label_is_read_to_the_hour_or_not_exactly_once() -> None:
    written = _labels_the_producer_writes()

    assert not owner.DEATH_TIME_READ_TO_THE_HOUR & owner.DEATH_TIME_NOT_READ_TO_THE_HOUR
    # Every label written is classified, and nothing else is.
    assert (
        owner.DEATH_TIME_READ_TO_THE_HOUR | owner.DEATH_TIME_NOT_READ_TO_THE_HOUR
        == written
    )
    for label in written:
        assert owner.death_time_read_to_the_hour(label) is (
            label in owner.DEATH_TIME_READ_TO_THE_HOUR
        )
    assert owner.death_time_read_to_the_hour("a_label_nobody_wrote") is None


@pytest.mark.parametrize(
    ("database", "read_to_the_hour"),
    [
        ("miiv", True),  # the admission's recorded death time
        ("mimic_demo", True),
        ("sic", True),  # a recorded offset of death, in seconds
        ("aumc", False),  # a date of death, as a proxy
        ("hirid", False),  # the last observation, as a proxy
        ("eicu", False),  # no death time is issued
        ("", False),  # no database named: the source's own time, resolution unknown
    ],
)
def test_each_source_reads_as_its_death_time_is_recorded(
    database: str, read_to_the_hour: bool
) -> None:
    label = owner.native_export_death_time_semantics(database)

    assert owner.death_time_read_to_the_hour(label) is read_to_the_hour


def test_the_export_writes_the_owners_labels_beside_the_owners_death_status() -> None:
    assert (
        extraction._NATIVE_EXPORT_DEATH_TIME_SEMANTICS
        is owner.NATIVE_EXPORT_DEATH_TIME_SEMANTICS
    )
    assert extraction._NATIVE_EXPORT_EVENT_TIME_COMPANIONS == {
        owner.DEATH_STATUS: owner.DEATH_TIME_COMPANION
    }
