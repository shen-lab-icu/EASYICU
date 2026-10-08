"""Independent synthetic scalar oracle for support-window record evidence."""

import random

import pandas as pd

from easyicu.io.scoped_support_events import _decorate, support_window_evidence


def test_root_scalar_windows_and_future_endpoint_invariance():
    origin = pd.Timestamp("2150-01-01")
    rng = random.Random(174)

    def stamp(hours):
        return None if hours is None else origin + pd.Timedelta(hours=hours)

    specs = []
    rows = []
    for i in range(160):
        start = rng.choice([None, -1, 0, 1, 2, 3, 5])
        end = rng.choice([None, 0, 1, 2, 3, 4, 10])
        stored = rng.choice([None, -1, 0, 1, 2, 3, 4, 8, 11])
        status = rng.choice(
            [None, "FinishedRunning", "Paused", "Stopped", "Cancelled", "unknown"]
        )
        specs.append((start, end, stored, status))
        rows.append(
            dict(
                source_file="synthetic",
                source_row_number=i,
                source_item_id=225792,
                subject_id=1,
                hadm_id=2,
                stay_id=3,
                intime=origin,
                outtime=stamp(12),
                raw_starttime=stamp(start),
                raw_endtime=stamp(end),
                raw_storetime=stamp(stored),
                raw_statusdescription=status,
                raw_value=None,
                raw_valueuom=None,
            )
        )
    raw = pd.DataFrame(rows)
    events = _decorate(raw, {"synthetic": "synthetic-hash"}, "synthetic-dictionary")
    checks = 0
    for left, right, cutoff in [(0, 1, 1), (0, 1, 2), (1, 3, 4), (2, 3, 8), (0, 4, 12)]:
        got = support_window_evidence(
            events,
            window_start=stamp(left),
            window_end=stamp(right),
            decision_time=stamp(cutoff),
        )
        for row, (start, end, stored, status) in zip(got.itertuples(), specs):
            positive = start is not None and end is not None and end > start
            overlap = max(0, min(end, right) - max(start, left)) if positive else None
            covers = positive and start <= left and end >= right
            claim = (
                start is not None
                and stored is not None
                and stored >= start
                and stored < cutoff
            )
            completed = (
                positive and stored is not None and stored >= end and stored < cutoff
            )
            evidence = (
                covers
                and completed
                and status in {"FinishedRunning", "Stopped", "Paused"}
            )
            assert (
                pd.isna(row.retrospective_overlap_hours)
                if overlap is None
                else row.retrospective_overlap_hours == overlap
            )
            assert row.retrospective_covers_window == covers
            assert row.recorded_start_claim_visible == claim
            assert row.completed_interval_record_visible == completed
            assert row.completed_window_record_evidence == evidence
            assert row.support_evidence_status == (
                "positive_completed_window_record" if evidence else "unknown"
            )
            checks += 6
        # Holding start/store fixed, changing ultimate end/status cannot revise a start claim.
        changed = raw.copy()
        changed["raw_endtime"] = stamp(100)
        changed["raw_statusdescription"] = "Cancelled"
        mutated = _decorate(
            changed, {"synthetic": "synthetic-hash"}, "synthetic-dictionary"
        )
        other = support_window_evidence(
            mutated,
            window_start=stamp(left),
            window_end=stamp(right),
            decision_time=stamp(cutoff),
        )
        assert got.recorded_start_claim_visible.equals(
            other.recorded_start_claim_visible
        )
        assert not other.completed_window_record_evidence.any()
        checks += 2
    assert checks == 4810
