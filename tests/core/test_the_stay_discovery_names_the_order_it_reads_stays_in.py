"""The stay discovery names the order its IDs come in, which a cap keeps the first of.

An export capped at K stays keeps the first K IDs the discovery returns.  It
reads them as the source's stay table lists them, or, for a source without
such a table, from the loader's ID-sorted sample.  Neither is a random
sample, and they are not the same order, so the discovery now says which one
it read; the IDs it returns are unchanged.  Synthetic tables only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu import api


def test_a_stay_table_is_read_in_its_own_order(tmp_path) -> None:
    pd.DataFrame({"stay_id": [30, 10, 20, 10]}).to_parquet(
        tmp_path / "icustays.parquet", index=False
    )
    listing: dict[str, str] = {}

    ids, id_col = api.get_all_patient_ids(
        tmp_path, database="miiv", max_patients=2, listing=listing
    )

    assert (ids, id_col) == ([30, 10], "stay_id")
    assert listing == {"order": "source_file_order"}
    # Without the record, the discovery returns what it always did.
    assert api.get_all_patient_ids(tmp_path, database="miiv", max_patients=2) == (
        [30, 10],
        "stay_id",
    )


def test_a_source_without_a_stay_table_is_read_in_identifier_order(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Loader:
        def __init__(self, **_kwargs) -> None:
            pass

    sampled: list[tuple[int, str]] = []

    def sample(loader, max_patients, verbose=False, sample_strategy="random"):
        sampled.append((max_patients, sample_strategy))
        return [1, 2]

    monkeypatch.setattr(api, "BaseICULoader", Loader)
    monkeypatch.setattr(api, "_sample_patient_ids", sample)
    listing: dict[str, str] = {}

    ids, _ = api.get_all_patient_ids(
        tmp_path, database="miiv", max_patients=2, listing=listing
    )

    assert ids == [1, 2]
    # The sample is asked for in identifier order, which is the order recorded.
    assert sampled == [(2, "sorted")]
    assert listing == {"order": "identifier_order"}


def test_a_failed_sample_names_no_order(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "BaseICULoader", lambda **_kwargs: object())
    monkeypatch.setattr(api, "_sample_patient_ids", lambda *_args, **_kwargs: None)
    listing: dict[str, str] = {}

    assert api.get_all_patient_ids(tmp_path, database="miiv", listing=listing) == (
        [],
        "stay_id",
    )
    assert listing == {}
