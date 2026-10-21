"""A hosted instance summarizes its prepared data once, after it starts."""

from __future__ import annotations

import threading

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from easyicu.webserver import cohort_review, crossdb_parquet, crossdb_review, dataio, hosted_warmup
from easyicu.webserver import sources as source_store


@pytest.fixture(autouse=True)
def fresh_thread(monkeypatch):
    monkeypatch.setattr(hosted_warmup, "_thread", None)


@pytest.mark.parametrize(("hosted", "flag"), [("0", "1"), ("1", ""), ("1", "0")])
def test_only_a_hosted_instance_whose_launcher_asked_warms(monkeypatch, hosted, flag):
    monkeypatch.setenv("EASYICU_HOSTED", hosted)
    monkeypatch.setenv("EASYICU_HOSTED_WARMUP", flag)
    monkeypatch.setattr(hosted_warmup, "warm", lambda: pytest.fail("must not warm"))
    assert hosted_warmup.start() is None


def test_the_warm_up_runs_once_per_process_in_the_background(monkeypatch):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_HOSTED_WARMUP", "1")
    calls = []
    release = threading.Event()
    monkeypatch.setattr(hosted_warmup, "warm", lambda: calls.append(1) or release.wait(5))
    thread = hosted_warmup.start()
    assert thread is not None and thread.daemon
    assert hosted_warmup.start() is thread
    release.set()
    thread.join(5)
    assert calls == [1]


def test_every_valid_export_is_summarized_active_first_and_a_failure_is_not_fatal(monkeypatch, caplog):
    monkeypatch.setattr(source_store, "load_registry", lambda: {
        "active_path": "/data/b",
        "sources": [
            {"id": "a", "path": "/data/a", "ok": True},
            {"id": "b", "path": "/data/b", "ok": True},
            {"id": "broken", "path": "/data/broken", "ok": False},
        ],
    })
    calls = []

    def cohort(body):
        calls.append(("cohort", body["source_path"]))
        if body["source_path"] == "/data/b":
            raise cohort_review.CohortReviewError({"error": "no_entity_denominator"})

    monkeypatch.setattr(cohort_review, "cohort_review_summary", cohort)
    monkeypatch.setattr(crossdb_review, "precompute_feature_distributions",
                        lambda path: calls.append(("crossdb", path)) or 3)
    finished = hosted_warmup.warm()

    assert calls == [("cohort", "/data/b"), ("crossdb", "/data/b"), ("cohort", "/data/a"), ("crossdb", "/data/a")]
    assert finished == [("b", "crossdb"), ("a", "cohort"), ("a", "crossdb")]
    assert "cohort summary of source b failed: CohortReviewError" in caplog.text


def test_precomputed_features_are_what_a_later_comparison_reads(tmp_path, monkeypatch):
    path = tmp_path / "vitals.parquet"
    pq.write_table(pa.table({"stay_id": np.arange(50), "hr": np.linspace(40, 140, 50)}), path)
    desc = {"ok": True, "path": str(tmp_path),
            "files": [{"file": path.name, "module": "vitals", "columns": ["stay_id", "hr"]}]}
    monkeypatch.setattr(dataio, "describe_export_source", lambda raw: desc)
    assert crossdb_review.precompute_feature_distributions(str(tmp_path)) == 1

    def no_scan(*args, **kwargs):
        raise AssertionError("the comparison must read the precomputed summary")

    monkeypatch.setattr(crossdb_parquet, "_connection", no_scan)
    later = crossdb_review._source_feature_distributions(desc)
    assert [item["feature"] for item in later["vitals"]["features"]] == ["hr"]

    monkeypatch.setattr(dataio, "describe_export_source", lambda raw: {"ok": False})
    assert crossdb_review.precompute_feature_distributions(str(tmp_path)) == 0
