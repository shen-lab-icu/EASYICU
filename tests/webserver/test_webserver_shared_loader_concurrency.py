"""Two overlapping web requests on one database both load (B18 resolver pool).

The dataio cohort prefilter and the cross-database raw loader call
``easyicu.load_concepts``, which shares one cached loader per database. Each
call holds its own pooled resolver, so an overlapping request is neither
refused (``ResolverBusyError``) nor different from a serial one. Rows are
synthetic.
"""

from __future__ import annotations

import pytest

from easyicu.api import concepts as api
from easyicu.webserver import crossdb_review, dataio
from tests.support.stay_call_sources import (
    SYNTHETIC_EICU_IDS as _IDS,
    gate_first_read as _gate_first_read,
    run_threads as _run_threads,
    synthetic_eicu as _eicu,
    wait_for as _wait_for,
)

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(autouse=True)
def _fresh_global_loader():
    api.clear_global_loader()
    yield
    api.clear_global_loader()


def _overlap(monkeypatch, request):
    inside, release = _gate_first_read(monkeypatch)
    threads, results = _run_threads(lambda _i: request(), 2)
    try:
        assert _wait_for(lambda: len(inside) == 2), "the two requests never overlapped"
    finally:
        release.set()
    for thread in threads:
        thread.join(timeout=120)
    assert not any(thread.is_alive() for thread in threads)
    return results


def test_dataio_concept_cohort_requests_overlap_on_one_database(tmp_path, monkeypatch):
    root = tmp_path / "eicu"
    _eicu(root)
    monkeypatch.setitem(
        dataio._CONCEPT_DERIVED_COHORTS, "b18_probe",
        {"concepts": ["pap_mean"], "positive": ["pap_mean"]},
    )
    import easyicu.api as easyicu_api

    def request():
        return dataio._match_concept_derived_cohort_ids(
            easyicu_api, str(root), "eicu", "patientunitstayid",
            {int(i) for i in _IDS}, "b18_probe", 24,
        )

    serial = request()
    assert serial  # the probe cohort is non-empty, so equality means something
    results = _overlap(monkeypatch, request)
    for index in range(2):
        assert isinstance(results[index], set), results[index]
        assert results[index] == serial


def test_crossdb_raw_feature_requests_overlap_on_one_database(tmp_path, monkeypatch):
    root = tmp_path / "databases"
    _eicu(root / "eicu")

    def request():
        return crossdb_review._load_raw_feature_data(
            data_root=str(root), concepts=["pap_mean"], databases=["eicu"],
            max_patients=len(_IDS), sample_size=len(_IDS),
        )

    serial = request()
    assert not serial["eicu"].empty
    results = _overlap(monkeypatch, request)
    for index in range(2):
        assert isinstance(results[index], dict), results[index]
        assert list(results[index]) == list(serial) == ["eicu"]
        assert results[index]["eicu"].equals(serial["eicu"])
