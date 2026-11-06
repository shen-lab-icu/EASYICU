"""#25 MIMIC-III epoch-µs charttime guard: each value is judged alone.

Some MIMIC-III callback/prescription-derived concepts arrive with absolute
microsecond-epoch charttimes (~6e15) instead of ICU-relative hours. The guard in
``ConceptResolver._load_single_concept`` converts a value only when its own
magnitude exceeds 1e10, about nine orders above any relative hour. Because the
test is per value, a stay's hours cannot depend on which other stays share its
batch: the union of two stays converts exactly as each stay does alone.

This is the lead-r13 magnitude sweep's site #25 (LOW, per-row), the one LOW
site the partition pins in test_per_row_unit_rules_pinned.py did not map
(Codex delta4v C25). Synthetic values only.
"""

from __future__ import annotations

import pandas as pd

from easyicu.concept import ConceptDefinition, ConceptDictionary, ConceptResolver, ConceptSource
from easyicu.config import DataSourceConfig
from easyicu.table import ICUTable

_INTIME = pd.Timestamp("2150-01-01 00:00:00")


def _epoch_us(ts: str) -> float:
    return float(pd.Timestamp(ts).value // 1000)  # ns -> µs, as a float64 column


# stay 1: absolute epoch-µs times (05:30 and 30:10 after admission)
# stay 2: already ICU-relative hours
_EVENTS = pd.DataFrame({
    "icustay_id": [1, 1, 2, 2],
    "charttime": [_epoch_us("2150-01-01 05:30"), _epoch_us("2150-01-02 06:10"), 3.0, 20.0],
    "value": [1.0, 2.0, 3.0, 4.0],
})
_ICUSTAYS = pd.DataFrame({
    "icustay_id": [1, 2],
    "intime": [_INTIME, _INTIME],
    "outtime": [_INTIME + pd.Timedelta(days=10)] * 2,
})


class _Source:
    config = DataSourceConfig(
        name="mimic",
        tables={
            "events": {"defaults": {"id_var": "icustay_id", "index_var": "charttime", "val_var": "value"}},
            "icustays": {"defaults": {"id_var": "icustay_id"}},
        },
    )
    base_path = None

    def load_table(self, table_name, columns=None, filters=None, verbose=False):
        del verbose
        frame = {"events": _EVENTS, "icustays": _ICUSTAYS}[table_name].copy()
        for spec in filters or []:
            frame = spec.apply(frame)
        if columns:
            frame = frame[[c for c in frame.columns if c in set(columns) | {"icustay_id"}]]
        defaults = self.config.get_table(table_name).defaults
        return ICUTable(
            data=frame, id_columns=["icustay_id"], index_column=defaults.index_var,
            value_column=defaults.val_var,
        )


def _load(ids):
    dictionary = ConceptDictionary({
        "probe": ConceptDefinition(
            name="probe",
            sources={"mimic": [ConceptSource(table="events", value_var="value", index_var="charttime")]},
        )
    })
    frame = ConceptResolver(dictionary).load_concepts(
        ["probe"], _Source(), merge=False, interval=pd.Timedelta(hours=1),
        patient_ids={"icustay_id": list(ids)}, r_compatible=False, verbose=False, concept_workers=1,
    )["probe"].data
    out = frame[["icustay_id", "charttime", "probe"]].copy()
    out["charttime"] = out["charttime"].astype(float)
    return out.sort_values(["icustay_id", "charttime"], kind="mergesort").reset_index(drop=True)


def test_epoch_values_convert_per_row_and_relative_hours_are_untouched():
    both = _load([1, 2])
    by_stay = {sid: g["charttime"].tolist() for sid, g in both.groupby("icustay_id")}
    assert by_stay[1] == [5.0, 30.0]  # floor of 05:30 and 30:10 after admission
    assert by_stay[2] == [3.0, 20.0]  # already relative: left alone


def test_a_stay_converts_the_same_alone_and_in_a_shared_batch():
    both = _load([1, 2])
    parts = pd.concat([_load([1]), _load([2])], ignore_index=True)
    pd.testing.assert_frame_equal(both, parts, check_exact=True)
