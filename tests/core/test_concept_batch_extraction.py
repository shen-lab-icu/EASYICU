"""Correctness of concept batch routing, retries, and typed refusals."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu import api
from easyicu.api import concepts as concept_api
from easyicu.concept.availability_signal import TypedAbsenceSink
from easyicu.concept.errors import ConceptExtractionUnavailable


@pytest.mark.parametrize(("failure", "detail"), [
    ("crash", "worker exited with code 9"),
    ("timeout", "worker timed out"),
    ("missing-receipt", "worker left no completion receipt"),
    ("missing-shard", "worker output shards do not match"),
    ("missing-absence", "worker left no valid typed absence receipt"),
])
@pytest.mark.parametrize("failed_stay", [48, 73], ids=["first-batch", "second-batch"])
def test_subprocess_batch_failure_refuses_partial_rows_and_absences(
    monkeypatch, failure, detail, failed_stay,
):
    from easyicu.runtime import memory_manager

    sink = TypedAbsenceSink()
    attempts = []

    def run_batch(args, *_):
        stay = args["patient_ids"]["stay_id"][0]
        attempts.append(stay)
        prefix = Path(args["output_prefix"])
        shard = prefix.with_suffix(".parquet")
        receipt = {"shards": [], "absences": {"fio2_set": "carevue_view_required"}}
        if stay == failed_stay:
            if failure == "crash":
                return 9
            if failure == "timeout":
                raise TimeoutError("worker timed out")
            if failure == "missing-receipt":
                return 0
            if failure == "missing-shard":
                receipt["shards"] = [shard.name]
            if failure == "missing-absence":
                del receipt["absences"]
        else:
            pd.DataFrame({"stay_id": [stay], "hr": [42.0]}).to_parquet(shard, index=False)
            receipt["shards"] = [shard.name]
        prefix.with_suffix(".receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        return 0

    monkeypatch.setattr(memory_manager, "_run_batch_process", run_batch)
    with pytest.raises(ConceptExtractionUnavailable, match=detail) as caught:
        memory_manager.subprocess_batch_load(
            ["hr", "fio2_set"], "mimic", {"stay_id": [48, 73]}, batch_size=1, _typed_absence_sink=sink,
        )

    assert caught.value.stage == "subprocess_batch_load"
    assert attempts == ([48, 48] if failed_stay == 48 else [48, 73, 73])
    assert sink.codes() == {}


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs a forked worker to inject a hang")
@pytest.mark.parametrize("daemon", [False, True], ids=["multiprocessing", "raw-fork"])
def test_subprocess_batch_timeout_is_typed_after_repair(tmp_path, monkeypatch, daemon):
    import multiprocessing as mp
    import time

    from easyicu.runtime import memory_manager

    def hang(_args):
        (tmp_path / f"worker-{os.getpid()}").touch()
        time.sleep(10)

    monkeypatch.setenv("EASYICU_BATCH_TIMEOUT_SEC", "0.5")
    if daemon:
        monkeypatch.setattr(mp, "current_process", lambda: SimpleNamespace(daemon=True))
    monkeypatch.setattr(memory_manager, "_subprocess_load_worker", hang)

    with pytest.raises(ConceptExtractionUnavailable, match=r"batch 1: worker timed out after 0\.5s"):
        memory_manager.subprocess_batch_load(["hr"], "miiv", {"stay_id": [48]}, batch_size=1)
    assert len(list(tmp_path.glob("worker-*"))) == 2


def test_subprocess_batch_accepts_empty_output_with_a_receipt(monkeypatch):
    from easyicu.runtime import memory_manager

    def load_empty(**kwargs):
        kwargs["_typed_absence_sink"].observe(["fio2_set"], {"fio2_set": "carevue_view_required"})
        return pd.DataFrame()

    def run_batch(args, *_):
        memory_manager._subprocess_load_worker(args)
        return 0

    monkeypatch.setattr(api, "load_concepts", load_empty)
    monkeypatch.setattr(memory_manager, "_run_batch_process", run_batch)
    sink = TypedAbsenceSink()
    result = memory_manager.subprocess_batch_load(
        ["fio2_set"], "mimic", {"stay_id": [48, 73]}, batch_size=1, _typed_absence_sink=sink,
    )

    assert result.empty
    assert sink.codes() == {"fio2_set": "carevue_view_required"}


@pytest.mark.parametrize("merge", [True, False])
def test_subprocess_batch_retry_matches_clean_rows(monkeypatch, merge):
    from easyicu.runtime import memory_manager

    fail_second_batch = False

    def load_rows(**kwargs):
        ids = kwargs["patient_ids"]["stay_id"]
        frame = pd.DataFrame({"stay_id": ids, "hr": [60.0 + stay for stay in ids]})
        if merge:
            return frame
        result = {"hr": frame}
        if fail_second_batch and ids == [73, 19]:
            result["uncommitted"] = frame
        return result

    attempts = []

    def run_batch(args, *_):
        ids = args["patient_ids"]["stay_id"]
        attempts.append(ids)
        memory_manager._subprocess_load_worker(args)
        # A crashed attempt may already have written its shards and receipt.
        return 9 if fail_second_batch and ids == [73, 19] else 0

    monkeypatch.setattr(api, "load_concepts", load_rows)
    monkeypatch.setattr(memory_manager, "_run_batch_process", run_batch)
    args = (["hr"], "miiv", {"stay_id": [48, 25, 73, 19]})
    clean = memory_manager.subprocess_batch_load(*args, batch_size=2, merge=merge)
    assert attempts == [[48, 25], [73, 19]]

    attempts.clear()
    fail_second_batch = True
    retried = memory_manager.subprocess_batch_load(*args, batch_size=2, merge=merge)

    assert attempts == [[48, 25], [73, 19], [73], [19]]
    clean_rows = clean if merge else clean["hr"]
    assert clean_rows["stay_id"].tolist() == [48, 25, 73, 19]
    if not merge:
        assert retried.keys() == clean.keys()
    pd.testing.assert_frame_equal(retried if merge else retried["hr"], clean_rows)


@pytest.mark.parametrize("reasons", [{}, {"fio2_set": "required_source_table_missing"}])
@pytest.mark.parametrize("reverse", [False, True])
def test_disagreeing_absence_receipts_clear_only_the_disputed_concept(reasons, reverse):
    sink = TypedAbsenceSink()
    agreed = {"fio2_set": "carevue_view_required", "renal_excl": "carevue_view_required"}
    disputed = {**reasons, "renal_excl": "carevue_view_required"}
    receipts = [disputed, agreed] if reverse else [agreed, disputed]

    for receipt in [*receipts, agreed]:
        sink.observe(agreed, receipt)

    assert sink.codes() == {"renal_excl": "carevue_view_required"}


def test_special_loaders_receive_resolved_source_and_patient_filter(
    monkeypatch, tmp_path
):
    resolved_path = tmp_path / "resolved"
    resolved_path.mkdir()
    loader = SimpleNamespace(database="miiv", data_path=resolved_path)
    monkeypatch.setattr(concept_api, "_get_global_loader", lambda **kwargs: loader)

    calls = {}

    def record(name, frame):
        def _loader(database, data_path=None, **kwargs):
            calls[name] = (database, data_path, kwargs.get("patient_ids"))
            return frame

        return _loader

    import easyicu.scores.circ_failure as circ
    import easyicu.scores.comorbidity as comorbidity
    import easyicu.scores.kdigo_aki as kdigo
    import easyicu.scores.microbiology as microbiology
    import easyicu.scores.outcomes as outcomes

    monkeypatch.setattr(
        kdigo,
        "load_kdigo_aki",
        record(
            "aki",
            pd.DataFrame(
                {
                    "stay_id": [7],
                    "aki": [1],
                    "aki_severe": [True],
                    "aki_severe_ascertainment": ["positive"],
                }
            ),
        ),
    )
    monkeypatch.setattr(
        circ,
        "load_circ_failure",
        record("circ", pd.DataFrame({"stay_id": [7], "circ_failure": [1]})),
    )
    monkeypatch.setattr(
        comorbidity,
        "load_comorbidity",
        record("charlson", pd.DataFrame({"stay_id": [7], "charlson_index": [2]})),
    )
    monkeypatch.setattr(
        outcomes,
        "load_outcomes",
        record(
            "outcome",
            pd.DataFrame(
                {
                    "stay_id": [7],
                    "mort_28d": [False],
                    "followup_days_28d": [28.0],
                }
            ),
        ),
    )
    monkeypatch.setattr(
        microbiology,
        "load_microbiology",
        record("micro", pd.DataFrame({"stay_id": [7], "culture_positive": [True]})),
    )

    result = api.load_concepts(
        [
            "aki",
            "aki_severe",
            "aki_severe_ascertainment",
            "circ_failure",
            "charlson",
            "mort_28d",
            "followup_days_28d",
            "culture_positive",
        ],
        patient_ids={"stay_id": [7]},
        merge=False,
        concept_workers=1,
        parallel_workers=1,
        verbose=False,
    )

    assert set(result) == {
        "aki",
        "aki_severe",
        "aki_severe_ascertainment",
        "circ_failure",
        "charlson",
        "mort_28d",
        "followup_days_28d",
        "culture_positive",
    }
    assert set(calls) == {"aki", "circ", "charlson", "outcome", "micro"}
    assert all(call == ("miiv", str(resolved_path), [7]) for call in calls.values())


def test_batched_load_routes_special_concepts_through_subprocess_with_full_list(
    monkeypatch, tmp_path
):
    """Regression for the 2026-07 batching data-loss bug.

    When a request mixes special concepts (KDIGO/CIRC/COMORB/OUTCOME/MICRO) with a
    base concept AND batching triggers, the batched branch used to return before the
    special re-attach block, silently dropping the whole special group. The fix forces
    the subprocess batch path and passes the FULL concept list (specials included) so
    each per-batch child re-runs the dedicated special loaders. This test pins that
    invariant without forking: it captures what concept list reaches subprocess_batch_load.
    """
    import easyicu.runtime.memory_manager as mm

    loader = SimpleNamespace(database="miiv", data_path=tmp_path)
    monkeypatch.setattr(concept_api, "_get_global_loader", lambda **kwargs: loader)

    captured = {}

    def fake_subprocess_batch_load(concepts, **kwargs):
        captured["concepts"] = list(concepts)
        return {c: pd.DataFrame({c: [1]}) for c in concepts}

    monkeypatch.setattr(mm, "subprocess_batch_load", fake_subprocess_batch_load)

    # 200 patients with batch_size 50 -> 4 chunks -> batching triggers.
    result = api.load_concepts(
        ["aki", "hr"],
        patient_ids={"stay_id": list(range(200))},
        batch_size=50,
        merge=False,
        verbose=False,
    )

    assert "concepts" in captured, "batching must route through subprocess_batch_load"
    # The special concept 'aki' must survive into the batched subprocess call.
    assert "aki" in captured["concepts"], (
        "special concept 'aki' was stripped from the batched path — the KDIGO/CIRC/"
        "COMORB/OUTCOME/MICRO drop regression is back"
    )
    assert "hr" in captured["concepts"]
    assert "aki" in result


@pytest.mark.parametrize("ram_gib", [16, 8, 32], ids=["desktop-16GiB", "hosted-8GiB", "32GiB"])
@pytest.mark.parametrize("force_inprocess", [False, True], ids=["unset", "inprocess-1"])
@pytest.mark.parametrize(("reason", "count", "batch_size"), [
    ("fits-one-batch", 7, 7),
    ("ram-small-cohort", 7, 3),
    ("ram-large-cohort", 60_001, 30_001),
    ("ram-very-large-cohort", 120_001, 60_001),
    ("force-subprocess", 7, 3),
    ("special-concept", 7, 3),
    ("all-stays-explicit-batch", 7, 3),
    ("filtered-over-budget", 5_003, None),
    ("all-stays-over-budget", 5_003, None),
    ("all-stays-prefetch-retry", 5_003, None),
])
def test_concept_batch_routes_under_inprocess_override(
    monkeypatch, tmp_path, ram_gib, force_inprocess, reason, count, batch_size,
):
    import psutil

    from easyicu.runtime import memory_manager

    for key in (
        "EASYICU_FORCE_INPROCESS_BATCH", "EASYICU_FORCE_SUBPROCESS_BATCH",
        "EASYICU_AUTO_CHUNK_SIZE", "EASYICU_DISABLE_AUTO_CHUNK", "EASYICU_ONESHOT_BUDGET_MB",
    ):
        monkeypatch.delenv(key, raising=False)
    if force_inprocess:
        monkeypatch.setenv("EASYICU_FORCE_INPROCESS_BATCH", "1")
    if reason == "force-subprocess":
        monkeypatch.setenv("EASYICU_FORCE_SUBPROCESS_BATCH", "1")

    total = ram_gib * 1024**3
    memory = psutil.virtual_memory()._replace(total=total, available=total)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: memory)
    # The RAM rule as Linux/Windows run it; macOS keeps batches in-process (S1458).
    monkeypatch.setattr(concept_api, "_platform_keeps_batches_in_process", lambda: False)
    ids = list(range(1, count + 1))
    all_stays = reason.startswith("all-stays")
    concepts = ["hr", "aki"] if reason == "special-concept" else ["hr"]
    calls = []
    sample_attempts = []

    def sample(_loader, total, *_args, sample_strategy, **_kwargs):
        sample_attempts.append(total)
        assert sample_strategy == "sorted"
        if reason == "all-stays-prefetch-retry" and len(sample_attempts) == 1:
            return None
        return ids

    def record(route):
        def load(**kwargs):
            calls.append((route, kwargs))
            return pd.DataFrame({"stay_id": [1], "hr": [61.0]})
        return load

    loader = SimpleNamespace(database="miiv", data_path=tmp_path, load_concepts=record("one-shot"))
    monkeypatch.setattr(concept_api, "_get_global_loader", lambda **kwargs: loader)
    monkeypatch.setattr(concept_api, "_get_total_patient_count", lambda _loader: count)
    monkeypatch.setattr(concept_api, "_sample_patient_ids", sample)
    monkeypatch.setattr(memory_manager, "estimate_memory_mb", lambda *args: 24_000)
    monkeypatch.setattr(memory_manager, "auto_batch_size", lambda *args, **kwargs: 2_501)
    monkeypatch.setattr(memory_manager, "inprocess_batch_load", record("in-process"))
    monkeypatch.setattr(memory_manager, "subprocess_batch_load", record("subprocess"))

    if reason == "fits-one-batch":
        expected = "one-shot"
    elif reason == "special-concept" or batch_size is None:
        expected = "subprocess"
    elif force_inprocess or (
        ram_gib >= 16 and reason in {"ram-small-cohort", "all-stays-explicit-batch"}
    ) or (ram_gib == 32 and reason == "ram-large-cohort"):
        expected = "in-process"
    else:
        expected = "subprocess"

    result = api.load_concepts(
        concepts, database="miiv", data_path=tmp_path,
        patient_ids=None if all_stays else {"stay_id": ids},
        batch_size=batch_size, concept_workers=1, parallel_workers=1,
        merge=True, r_compatible=False, verbose=False,
    )
    (route, options), = calls
    assert route == expected
    assert options["concepts"] == concepts
    assert options["merge"] is True
    assert options["r_compatible"] is False
    scope_key = "all_patient_ids" if route == "subprocess" else "patient_ids"
    assert options[scope_key] == {"stay_id": ids}
    if route != "one-shot":
        assert options["batch_size"] == (2_501 if batch_size is None else batch_size)
    expected_samples = 2 if reason == "all-stays-prefetch-retry" else int(all_stays)
    assert sample_attempts == [count] * expected_samples
    assert result["hr"].tolist() == [61.0]
