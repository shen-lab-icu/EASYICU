"""A streamed module's produced concepts follow the loader, not Arrow's nulls.

2026-11-06 r13mergeidx: published concept_status differed between a default
(one-shot) and a batch-50 extraction on all five formal databases.  A
whole-module load counts a concept as produced when the loader returned its
column; the streamed path counted every column of a part schema, and Arrow
adds a structural null column for every requested concept.  A concept the
loader never returned was therefore "produced_all_null" when batched and
"structurally_unavailable_placeholder" by default.  Produced is now the union
of the loader's columns over all batches, so the three states are the same
at every batch size: values somewhere -> available; returned but null in
every batch -> produced_all_null; never returned -> placeholder.
"""
from __future__ import annotations

import pandas as pd
import pytest

import easyicu
from easyicu.api import extraction as api

STAYS = list(range(1, 11))
CONCEPTS = ["sparse_signal", "null_signal", "late_signal", "never_signal"]


def _loader(**kwargs):
    ids = list(kwargs["patient_ids"]["stay_id"])
    frame = pd.DataFrame({"stay_id": ids, "charttime": [0.0] * len(ids)})
    # values only for stay 9: null in every other batch
    frame["sparse_signal"] = [1.0 if i == 9 else None for i in ids]
    # returned in every batch, never a value
    frame["null_signal"] = pd.Series([None] * len(ids), dtype="float64")
    # the loader returns this column only when the batch holds stay 10
    if 10 in ids:
        frame["late_signal"] = [2.0 if i == 10 else None for i in ids]
    # never_signal: never returned (an unmapped concept)
    return frame


@pytest.mark.parametrize("batch_size", [10, 4, 1])
def test_produced_concepts_are_the_loader_union_at_every_batch_size(tmp_path, monkeypatch, batch_size):
    monkeypatch.setattr(easyicu, "load_concepts", _loader)
    info = api._stream_module_batches_to_parquet(
        "test_module", CONCEPTS, {"database": "miiv", "concepts": CONCEPTS},
        {"stay_id": STAYS}, batch_size, str(tmp_path),
    )
    assert info["concepts"] == ["sparse_signal", "null_signal", "late_signal"]
    per_batch = [entry["loader_concepts"] for entry in info["stream_batches"]]
    assert all("never_signal" not in names for names in per_batch)
    assert sum("late_signal" in names for names in per_batch) == 1
