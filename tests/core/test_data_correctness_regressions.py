from __future__ import annotations

from easyicu.datasource import ICUDataSource

import json
import os
import sys
from collections import UserDict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu import api
from easyicu.api import concepts as concept_api
from easyicu.api.cache import data_path_fingerprint
from easyicu.api import extraction as extraction_api
from easyicu.concept.availability_signal import TypedAbsenceSink
from easyicu.concept.errors import ConceptExtractionUnavailable
from easyicu.config import load_src_cfg
from easyicu.content_identity import file_content_receipt
from easyicu.datasource import (
    FilterOp,
    FilterSpec,
    _resolve_wide_column_batch_size,
    load_bucketed_table_aggregated,
    load_wide_table_aggregated,
)
from easyicu.io.data_converter import ConversionStatus, DataConverter
from easyicu.runtime.parallel_config import get_parallel_config
from easyicu.table import ICUTable


def test_align_to_icu_admission_fails_instead_of_returning_input_unchanged():
    frame = pd.DataFrame(
        {"stay_id": [1], "charttime": [pd.Timestamp("2026-01-01T00:00:00")]}
    )

    with pytest.raises(NotImplementedError, match="no longer returns unaligned data"):
        api.align_to_icu_admission(frame, verbose=False)

    with pytest.raises(NotImplementedError, match="load_concepts"):
        api.align_to_icu_admission({"hr": frame}, verbose=False)


def test_explicit_empty_cohort_returns_all_requested_special_outputs(monkeypatch):
    monkeypatch.setattr(
        concept_api,
        "_get_global_loader",
        lambda **kwargs: pytest.fail("empty cohort must not construct a loader"),
    )

    result = api.load_concepts(
        ["aki", "mort_28d"], patient_ids=[], merge=False, verbose=False
    )

    assert set(result) == {"aki", "mort_28d"}
    assert all(frame.empty for frame in result.values())

    via_alias = api.load_concepts(["hr"], stay_id=[], merge=False, verbose=False)
    assert via_alias["hr"].empty


@pytest.mark.parametrize("filters", [
    {"patientunitstayid": [], "hospitalid": [31]},
    UserDict({"patientunitstayid": [18], "PatientUnitStayID": [27]}),
])
def test_public_load_rejects_ambiguous_id_mappings(tmp_path, filters):
    with pytest.raises(ValueError, match="Expected one patientunitstayid identifier list"):
        api.load_concepts("fio2_set", database="eicu", data_path=tmp_path, patient_ids=filters)


def test_public_load_preserves_a_user_mapping_with_a_non_miiv_id(tmp_path):
    pd.DataFrame({
        "patientunitstayid": [38, 62],
        "patienthealthsystemstayid": [413, 716],
    }).to_parquet(tmp_path / "patient.parquet", index=False)
    pd.DataFrame({
        "patientunitstayid": [38, 62],
        "respchartoffset": [23, 23],
        "respchartvaluelabel": "FiO2",
        "respcharttypecat": "respFlowSettings",
        "respchartvalue": ["43", "66"],
    }).to_parquet(tmp_path / "respiratorycharting.parquet", index=False)

    empty = api.load_concepts("hr", patient_ids=UserDict())
    result = api.load_concepts(
        "fio2_set", database="eicu", data_path=tmp_path,
        patient_ids=UserDict({"patientunitstayid": [62]}), r_compatible=False,
    )

    assert empty.empty
    assert result[["patientunitstayid", "fio2_set"]].values.tolist() == [[62, 66.0]]


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


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs a forked worker to inject an exit")
def test_second_nested_worker_exit_blocks_native_publication(tmp_path, monkeypatch):
    from easyicu.runtime import memory_manager
    from tests.support.bucket_storage import b6_source

    source = b6_source(tmp_path / "source", "miiv")
    output = tmp_path / "module"
    output.mkdir()
    monkeypatch.setenv("EASYICU_FORCE_SUBPROCESS_BATCH", "1")
    monkeypatch.delenv("EASYICU_FORCE_INPROCESS_BATCH", raising=False)
    original_worker = memory_manager._subprocess_load_worker

    def exit_second_worker(args):
        if args["patient_ids"] == {"stay_id": [1]}:
            raise SystemExit(9)
        original_worker(args)

    monkeypatch.setattr(memory_manager, "_subprocess_load_worker", exit_second_worker)
    extraction_api._run_module_extraction(
        module_name="vitals", concepts=["hr", "map"], database="miiv",
        data_path=str(source.base_path), patient_ids_filter={"stay_id": [2, 1]},
        batch_size=1, output_dir=str(output),
    )
    module = json.loads((output / "_manifest.json").read_text(encoding="utf-8"))
    assert module["saved"] == {}
    assert "batch 2: worker exited with code 9" in module["errors"][0]
    assert not (output / "vitals.parquet").exists()

    native = tmp_path / "native"
    native.mkdir()
    with pytest.raises(ValueError, match="requires every requested module to finish"):
        extraction_api._publish_native_export_v2(
            database="miiv", data_path=str(source.base_path), output_dir=str(native),
            modules=["vitals"], max_patients=None, result={"modules": {"vitals": module}},
        )
    assert not (native / "_manifest.json").exists()


@pytest.fixture
def desktop_renal_cohort(tmp_path, monkeypatch):
    import psutil

    from easyicu.runtime import memory_manager
    from easyicu.webserver.hosted import deployment_kind
    from tests.support.renal_export import write_renal_source

    monkeypatch.setenv("EASYICU_DEPLOYMENT", "desktop")
    monkeypatch.setenv("EASYICU_MP_START_METHOD", "fork")
    for key in (
        "EASYICU_HOSTED", "EASYICU_ONESHOT_BUDGET_MB", "EASYICU_AUTO_CHUNK_SIZE",
        "EASYICU_DISABLE_AUTO_CHUNK", "EASYICU_FORCE_INPROCESS_BATCH",
        "EASYICU_FORCE_SUBPROCESS_BATCH",
    ):
        monkeypatch.delenv(key, raising=False)
    assert deployment_kind() == "desktop"

    ram = 16 * 1024**3
    memory = psutil.virtual_memory()._replace(total=ram, available=ram)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: memory)
    monkeypatch.setattr(memory_manager, "_read_cgroup_v2_memory_mb", lambda root: (None, None))

    # Put synthetic observations at both ends of a large logical cohort.
    source = tmp_path / "source"
    write_renal_source(source, "eicu", stays=24, hours=6, urine_rows_per_hour=1)
    id_col = "patientunitstayid"
    cohort_size = 135_003
    for path in source.glob("*.parquet"):
        frame = pd.read_parquet(path)
        if id_col in frame and not frame.empty:
            ids = frame[id_col]
            frame[id_col] = ids.where(ids <= 12, ids + cohort_size - 24)
            if path.name == "patient.parquet":
                # Every fixture stay is the first in its own hospitalization.
                frame["unitvisitnumber"] = 1
            frame.to_parquet(path, index=False)
    return source, {id_col: list(range(1, cohort_size + 1))}


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs a forked worker to inject an exit")
@pytest.mark.parametrize("worker_exit", [0, 9], ids=["clean", "exit-9"])
def test_desktop_renal_automatically_uses_nested_batches(
    tmp_path, monkeypatch, desktop_renal_cohort, worker_exit,
):
    import multiprocessing as mp

    from easyicu.runtime import memory_manager

    source, patient_ids = desktop_renal_cohort
    id_col, ids = next(iter(patient_ids.items()))
    concepts = extraction_api.EXTRACT_MODULES["renal"]
    original_worker = memory_manager._subprocess_load_worker
    run_batch = memory_manager._run_batch_process
    batch_log = tmp_path / "nested-batches.jsonl"

    def worker(args):
        if worker_exit and ids[-1] in args["patient_ids"][id_col]:
            raise SystemExit(worker_exit)
        original_worker(args)

    def observe_batch(args, attempt_dir, batch_num, mode):
        assert os.environ["EASYICU_FORCE_INPROCESS_BATCH"] == "1"
        batch_ids = args["patient_ids"][id_col]
        exitcode = run_batch(args, attempt_dir, batch_num, mode)
        with batch_log.open("a", encoding="utf-8") as log:
            log.write(json.dumps([batch_num, batch_ids[0], batch_ids[-1], exitcode]) + "\n")
        return exitcode

    def extract(name, batch_size):
        output = tmp_path / name
        worker = mp.get_context("fork").Process(
            target=extraction_api._extract_module_group_worker,
            args=([("renal", concepts)], [], "eicu", str(source), patient_ids, batch_size, str(output), False),
            daemon=True,
        )
        worker.start()
        worker.join()
        assert worker.exitcode == 0
        return output / "renal"

    monkeypatch.setattr(memory_manager, "_subprocess_load_worker", worker)
    monkeypatch.setattr(memory_manager, "_run_batch_process", observe_batch)
    output = extract("automatic", None)
    attempts = [json.loads(line) for line in batch_log.read_text(encoding="utf-8").splitlines()]
    first, second, *repairs = attempts
    assert first == [1, ids[0], second[1] - 1, 0]
    assert second == [2, first[2] + 1, ids[-1], worker_exit]
    manifest = json.loads((output / "_manifest.json").read_text(encoding="utf-8"))
    if worker_exit:
        assert [(number, code) for number, _, _, code in repairs] == [(2, 0), (2, 9)]
        assert manifest["saved"] == {}
        assert "stage 'subprocess_batch_load': batch 2: worker exited with code 9" in manifest["errors"][0]
        assert not (output / "renal.parquet").exists()
    else:
        assert not repairs
        assert manifest["errors"] == []
        one_shot = extract("one-shot", len(ids))
        assert len(batch_log.read_text(encoding="utf-8").splitlines()) == 2
        baseline_manifest = json.loads((one_shot / "_manifest.json").read_text(encoding="utf-8"))
        assert baseline_manifest["errors"] == []
        actual = pd.read_parquet(output / "renal.parquet")
        expected = pd.read_parquet(one_shot / "renal.parquet")
        assert {ids[0], ids[-1]} <= set(expected[id_col])
        sort_keys = [id_col, "charttime"]
        pd.testing.assert_frame_equal(
            actual.sort_values(sort_keys).reset_index(drop=True),
            expected.sort_values(sort_keys).reset_index(drop=True),
            check_exact=True,
        )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs a forked worker to inject an exit")
@pytest.mark.parametrize("worker_exit", [0, 9], ids=["clean", "exit-9"])
def test_desktop_native_export_reports_stream_batch_exit(
    tmp_path, monkeypatch, desktop_renal_cohort, worker_exit,
):
    import psutil

    from tools import formal_run_receipt
    from tools.formal import v2_bench_local

    source, patient_ids = desktop_renal_cohort
    id_col, ids = next(iter(patient_ids.items()))
    stream_worker = extraction_api._write_isolated_stream_batch
    batch_log = tmp_path / "stream-batches.jsonl"

    def run_stream_batch(module, concepts, load_kwargs, batch_ids, *args):
        assert os.environ["EASYICU_FORCE_INPROCESS_BATCH"] == "1"
        if module == "renal":
            # Record the actual inner load size and patient partition in the child.
            positions = batch_ids[id_col]
            with batch_log.open("a", encoding="utf-8") as log:
                log.write(json.dumps([load_kwargs["batch_size"], positions[0], positions[-1]]) + "\n")
            if worker_exit and ids[-1] in positions:
                raise SystemExit(worker_exit)
        stream_worker(module, concepts, load_kwargs, batch_ids, *args)

    monkeypatch.setattr(extraction_api, "_write_isolated_stream_batch", run_stream_batch)
    monkeypatch.setattr(extraction_api, "_get_all_patient_ids", lambda *args: (ids, id_col))
    summary_dir = tmp_path / "eicu"
    summary_dir.mkdir()
    monkeypatch.setattr(sys, "argv", ["v2_bench_local.py", "eicu", str(source), str(summary_dir), "full"])
    monkeypatch.setitem(sys.modules, "formal_boot_live", SimpleNamespace(record={
        "tools": str(tmp_path), "src": str(tmp_path), "sha": "", "site_packages": "",
    }))
    cpu_count = os.cpu_count() or 1
    monkeypatch.setattr(v2_bench_local, "_affinity_selfcheck", lambda: cpu_count)
    if not hasattr(psutil.Process, "cpu_affinity"):
        monkeypatch.setattr(psutil.Process, "cpu_affinity", lambda self: range(cpu_count), raising=False)
    monkeypatch.setattr(v2_bench_local, "subprocess", SimpleNamespace(run=lambda *args, **kwargs: None))
    monkeypatch.setattr(v2_bench_local, "_cgroup_dir", lambda: tmp_path / "synthetic-cgroup")
    # Native longitudinal publication requires the outcome module's LOS bounds.
    monkeypatch.setenv("BENCH_MODULES", "renal,outcome")
    monkeypatch.delenv("BENCH_BATCH", raising=False)
    # Register the environment slot so pytest restores it after main sets it.
    monkeypatch.setenv("FORMAL_WORKER_ORIGIN_DIR", str(summary_dir / "worker_origin"))
    v2_bench_local.main()

    summary = json.loads((summary_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["batch_size"] is None
    attempts = [json.loads(line) for line in batch_log.read_text(encoding="utf-8").splitlines()]
    assert len(attempts) > 1
    batch_size = attempts[0][0]
    assert attempts == [
        [min(batch_size, len(ids) - start), ids[start], ids[min(start + batch_size, len(ids)) - 1]]
        for start in range(0, len(ids), batch_size)
    ]
    output = summary_dir / "export"
    if worker_exit:
        with pytest.raises(formal_run_receipt.ReceiptError) as caught:
            formal_run_receipt._database_section(tmp_path, "eicu", str(source), "", 0, "")
        assert caught.value.code == "database_run_reported_errors"
        assert caught.value.details == {
            "database": "eicu",
            "root_type": "builtins.ValueError",
            "root_reason": None,
            "chain": ["NativeExportPublicationError", "ValueError"],
        }
        assert "failures=['renal']" in summary["exception"]
        manifest_path, = (output / ".easyicu-unpublished").glob("*/renal.manifest.json")
        module = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert module["saved"] == {}
        assert len(module["errors"]) == 1
        assert module["resource_plan"]["batch_size"] == batch_size
        failure = module["stream_failure"]
        expected_failure = {
            "exception_type": "StreamBatchWorkerKilled",
            "failure_class": "worker_failed",
            "memory_evidence": None,
            "worker_exit_code": worker_exit,
            "batch_index": len(attempts),
            "start_offset": batch_size * (len(attempts) - 1),
            "batch_size": batch_size,
            "retries_used": 0,
        }
        assert {key: failure[key] for key in expected_failure} == expected_failure
        assert not (output / "_manifest.json").exists()
        assert not list(output.glob("*.parquet"))
    else:
        assert summary.get("errors") == [], summary.get("exception")
        assert summary["worker_failures"] == 0
        assert "_manifest.json" in summary["files"]
        assert "renal.parquet" in summary["files"]
        published = json.loads((output / "_manifest.json").read_text(encoding="utf-8"))
        assert published["module_resource_plans"]["renal"]["batch_size"] == batch_size
        assert published["time_window_authority"]["source"] == "outcome.los_icu"
        native = pd.read_parquet(output / "renal.parquet", columns=["stay_id"])
        assert set(native["stay_id"]) == set(range(1, 13)) | set(ids[-12:])


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


def test_concept_cache_isolates_cohort_source_and_data_fingerprint(
    monkeypatch, tmp_path
):
    data_path = tmp_path / "data"
    cache_path = tmp_path / "cache"
    data_path.mkdir()
    source_file = data_path / "stays.parquet"
    source_file.write_bytes(b"v1")
    calls = []

    def fake_load_concepts(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"stay_id": kwargs["patient_ids"]})

    monkeypatch.setattr(api, "load_concepts", fake_load_concepts)
    monkeypatch.setattr(
        api,
        "align_to_icu_admission",
        lambda frame, **kwargs: frame.assign(aligned=True),
    )

    first = api.load_concept_cached(
        "hr",
        "miiv",
        data_path,
        cache_dir=cache_path,
        patient_ids=[1],
        n_patients=3,
        align_time=True,
        verbose=False,
    )
    api.load_concept_cached(
        "hr",
        "miiv",
        data_path,
        cache_dir=cache_path,
        patient_ids=[2],
        n_patients=3,
        align_time=True,
        verbose=False,
    )
    cached = api.load_concept_cached(
        "hr",
        "miiv",
        data_path,
        cache_dir=cache_path,
        patient_ids=[1],
        n_patients=3,
        align_time=True,
        verbose=False,
    )
    source_file.write_bytes(b"version-two")
    os.utime(source_file, None)
    api.load_concept_cached(
        "hr",
        "miiv",
        data_path,
        cache_dir=cache_path,
        patient_ids=[1],
        n_patients=3,
        align_time=True,
        verbose=False,
    )

    assert len(calls) == 3
    assert calls[0]["patient_ids"] == [1]
    assert calls[0]["n_patients"] == 3
    assert first["aligned"].all() and cached["aligned"].all()


def test_concept_cache_invalidates_same_size_content_with_restored_mtime(
    monkeypatch, tmp_path
):
    data_path = tmp_path / "data"
    cache_path = tmp_path / "cache"
    data_path.mkdir()
    source_file = data_path / "stays.csv"
    source_file.write_bytes(b"id,value\n1,old\n")
    original_stat = source_file.stat()
    calls = []

    def fake_load_concepts(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"stay_id": [len(calls)]})

    monkeypatch.setattr(api, "load_concepts", fake_load_concepts)
    api.load_concept_cached(
        "hr",
        "miiv",
        data_path,
        cache_dir=cache_path,
        patient_ids=[1],
        verbose=False,
    )

    source_file.write_bytes(b"id,value\n1,new\n")
    os.utime(
        source_file,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    api.load_concept_cached(
        "hr",
        "miiv",
        data_path,
        cache_dir=cache_path,
        patient_ids=[1],
        verbose=False,
    )

    assert len(calls) == 2


def test_content_fingerprint_does_not_exclude_data_when_cache_is_an_ancestor(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "data"
    data_path.mkdir()
    source_file = data_path / "stays.csv"
    source_file.write_bytes(b"id,value\n1,old\n")
    original_stat = source_file.stat()

    before = data_path_fingerprint(data_path, exclude_dir=tmp_path)
    source_file.write_bytes(b"id,value\n1,new\n")
    os.utime(
        source_file,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    after = data_path_fingerprint(data_path, exclude_dir=tmp_path)

    assert after != before


def test_concept_cache_does_not_truncate_digest_to_collision_prone_prefix(
    monkeypatch, tmp_path
):
    data_path = tmp_path / "data"
    cache_path = tmp_path / "cache"
    data_path.mkdir()
    calls = []

    def fake_cache_key(concepts, source, **kwargs):
        cohort = kwargs["patient_ids"][0]
        suffix = "1" * 24 if cohort == 1 else "2" * 24
        return "deadbeef" + suffix + "0" * 32

    def fake_load_concepts(**kwargs):
        calls.append(kwargs["patient_ids"])
        return pd.DataFrame({"stay_id": kwargs["patient_ids"]})

    monkeypatch.setattr(api, "_get_cache_key", fake_cache_key)
    monkeypatch.setattr(api, "load_concepts", fake_load_concepts)

    first = api.load_concept_cached(
        "hr",
        "miiv",
        data_path,
        cache_dir=cache_path,
        patient_ids=[1],
        verbose=False,
    )
    second = api.load_concept_cached(
        "hr",
        "miiv",
        data_path,
        cache_dir=cache_path,
        patient_ids=[2],
        verbose=False,
    )

    assert calls == [[1], [2]]
    assert first["stay_id"].tolist() == [1]
    assert second["stay_id"].tolist() == [2]


def test_concept_cache_uses_opaque_parquet_name_by_default(monkeypatch, tmp_path):
    data_path = tmp_path / "data"
    cache_path = tmp_path / "cache"
    data_path.mkdir()
    calls = []

    def fake_load_concepts(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"stay_id": [1], "charttime": [pd.Timestamp("2020-01-01")]})

    monkeypatch.setattr(api, "load_concepts", fake_load_concepts)

    first = api.load_concept_cached(
        "../heart/rate",
        "../../miiv",
        data_path,
        cache_dir=cache_path,
        verbose=False,
    )
    second = api.load_concept_cached(
        "../heart/rate",
        "../../miiv",
        data_path,
        cache_dir=cache_path,
        verbose=False,
    )

    receipt_index = cache_path / ".easyicu_content_receipts.json"
    cache_files = [path for path in cache_path.iterdir() if path != receipt_index]
    assert len(calls) == 1
    assert first.equals(second)
    assert receipt_index.is_file()
    assert len(cache_files) == 1
    assert cache_files[0].parent == cache_path
    assert cache_files[0].suffix == ".parquet"
    assert len(cache_files[0].stem) == 64
    int(cache_files[0].stem, 16)


def test_pickle_cache_is_explicit_and_uses_an_opaque_name(monkeypatch, tmp_path):
    data_path = tmp_path / "data"
    cache_path = tmp_path / "cache"
    data_path.mkdir()
    monkeypatch.setenv("EASYICU_CACHE_HMAC_KEY", "test-hmac-key")
    monkeypatch.setattr(
        api,
        "load_concepts",
        lambda **_kwargs: pd.DataFrame({"stay_id": [1]}),
    )

    api.load_concept_cached(
        "../heart/rate",
        "../../miiv",
        data_path,
        cache_dir=cache_path,
        use_pickle=True,
        verbose=False,
    )

    receipt_index = cache_path / ".easyicu_content_receipts.json"
    cache_files = [path for path in cache_path.iterdir() if path != receipt_index]
    assert receipt_index.is_file()
    assert len(cache_files) == 2
    payloads = [path for path in cache_files if path.name.endswith(".trusted.pkl")]
    sidecars = [path for path in cache_files if path.name.endswith(".trusted.pkl.hmac")]
    assert len(payloads) == 1
    assert len(sidecars) == 1
    assert sidecars[0].name == payloads[0].name + ".hmac"
    assert len(payloads[0].name.removesuffix(".trusted.pkl")) == 64


def test_transformed_bounds_are_applied_before_hourly_aggregation(tmp_path):
    pd.DataFrame(
        {
            "patientunitstayid": [1, 1],
            "labresultoffset": [0, 10],
            "labname": ["x", "x"],
            "value": [80.0, 10_000.0],
        }
    ).to_parquet(tmp_path / "labs.parquet", index=False)

    defaults = SimpleNamespace(
        index_var="labresultoffset", sub_var="labname", unit_var=None
    )
    config = SimpleNamespace(
        name="eicu", get_table=lambda name: SimpleNamespace(defaults=defaults)
    )
    source = SimpleNamespace(
        config=config,
        base_path=tmp_path,
        resolve_bucket_directory=lambda name: None,
        resolve_flat_parquet_directory=lambda name: tmp_path,
        resolve_target_files=lambda name, *_selector, **_stay: tuple(sorted(tmp_path.glob("*.parquet"))),
        _get_parquet_columns_for_files=lambda files: {
            "patientunitstayid",
            "labresultoffset",
            "labname",
            "value",
        },
    )

    result = load_bucketed_table_aggregated(
        source,
        "labs",
        "value",
        ["x"],
        patient_ids=[1],
        value_min=0,
        value_max=100,
        value_transform='TRY_CAST("value" AS DOUBLE)',
    )

    assert result["value"].tolist() == [80.0]
    assert result.attrs["easyicu_bounds_loader"] == {
        "bounds_raw_transformed_non_null": 2,
        "bounds_bounded_transformed_non_null": 1,
        "bounds_bounded_aggregate_non_null": 1,
    }


def test_bucket_loader_honours_source_sub_var_over_table_default(tmp_path):
    official_path = (
        "flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine"
    )
    pd.DataFrame(
        {
            "patientunitstayid": [1, 1],
            "intakeoutputoffset": [0, 60],
            "celllabel": ["Urine", "Urine"],
            "cellpath": [official_path, "unrelated|Urine"],
            "cellvaluenumeric": [50.0, 999.0],
        }
    ).to_parquet(tmp_path / "intakeoutput.parquet", index=False)

    defaults = SimpleNamespace(
        index_var="intakeoutputoffset",
        sub_var="celllabel",
        unit_var=None,
    )
    config = SimpleNamespace(
        name="eicu", get_table=lambda name: SimpleNamespace(defaults=defaults)
    )
    source = SimpleNamespace(
        config=config,
        base_path=tmp_path,
        resolve_bucket_directory=lambda name: None,
        resolve_flat_parquet_directory=lambda name: tmp_path,
        resolve_target_files=lambda name, *_selector, **_stay: tuple(sorted(tmp_path.glob("*.parquet"))),
        _get_parquet_columns_for_files=lambda files: set(
            pd.read_parquet(files[0]).columns
        ),
    )

    result = load_bucketed_table_aggregated(
        source,
        "intakeoutput",
        "cellvaluenumeric",
        [official_path],
        patient_ids=[1],
        agg_func="sum",
        itemid_col="cellpath",
    )

    assert result["charttime"].tolist() == [0.0]
    assert result["cellvaluenumeric"].tolist() == [50.0]


def test_inline_unit_conversion_resolves_configured_column_case_insensitively(
    tmp_path,
):
    """MIMIC-III parquet columns are uppercase while config names are lowercase."""
    pd.DataFrame(
        {
            "patientunitstayid": [1, 1],
            "labresultoffset": [0, 60],
            "labname": ["CRP", "CRP"],
            "value": [30.0, 4.0],
            "VALUEUOM": ["mg/L", "mg/dL"],
        }
    ).to_parquet(tmp_path / "lab.parquet", index=False)

    defaults = SimpleNamespace(
        index_var="labresultoffset", sub_var="labname", unit_var="valueuom"
    )
    config = SimpleNamespace(
        name="eicu", get_table=lambda name: SimpleNamespace(defaults=defaults)
    )

    source = SimpleNamespace(
        config=config,
        base_path=tmp_path,
        resolve_bucket_directory=lambda name: None,
        resolve_flat_parquet_directory=lambda name: tmp_path,
        resolve_target_files=lambda name, *_selector, **_stay: tuple(sorted(tmp_path.glob("*.parquet"))),
        _get_parquet_columns_for_files=lambda files: {
            "patientunitstayid",
            "labresultoffset",
            "labname",
            "value",
            "VALUEUOM",
        },
    )

    result = load_bucketed_table_aggregated(
        source,
        "lab",
        "value",
        ["CRP"],
        patient_ids=[1],
        convert_unit_op="*",
        convert_unit_factor=10,
        convert_unit_filter="mg/dl",
    )

    assert result["value"].tolist() == pytest.approx([30.0, 40.0])


def test_partitioned_arrow_projection_resolves_mimic_uppercase_columns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pyarrow as pa

    partition = tmp_path / "inputevents_cv"
    partition.mkdir()
    pd.DataFrame(
        {
            "ICUSTAY_ID": [1, 2, 2],
            "ITEMID": [10, 10, 11],
            "VALUENUM": [1.0, 2.0, 3.0],
        }
    ).to_parquet(partition / "part-0.parquet", index=False)
    source = ICUDataSource(
        load_src_cfg("mimic"),
        base_path=tmp_path,
        enable_cache=False,
    )
    configured_thread_counts = []
    monkeypatch.setenv("EASYICU_ARROW_THREADS", "3")
    monkeypatch.setattr(pa, "cpu_count", lambda: 384)
    monkeypatch.setattr(
        pa,
        "set_cpu_count",
        lambda value: configured_thread_counts.append(value),
    )

    frame = source._read_partitioned_data_optimized(
        partition,
        columns=["icustay_id", "itemid", "valuenum"],
        patient_ids_filter=FilterSpec(
            column="icustay_id",
            op=FilterOp.IN,
            value=[2],
        ),
        itemid_filter_config=("itemid", {10}),
    )

    assert list(frame.columns) == ["icustay_id", "itemid", "valuenum"]
    assert configured_thread_counts == [3]
    assert frame.to_dict("records") == [
        {"icustay_id": 2, "itemid": 10, "valuenum": 2.0}
    ]


def test_partitioned_arrow_scan_ignores_invalid_thread_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pyarrow as pa

    partition = tmp_path / "inputevents_cv"
    partition.mkdir()
    pd.DataFrame(
        {
            "ICUSTAY_ID": [1],
            "ITEMID": [10],
            "VALUENUM": [1.0],
        }
    ).to_parquet(partition / "part-0.parquet", index=False)
    source = ICUDataSource(
        load_src_cfg("mimic"),
        base_path=tmp_path,
        enable_cache=False,
    )
    configured_thread_counts = []
    monkeypatch.setenv("EASYICU_ARROW_THREADS", "not-an-integer")
    monkeypatch.setattr("os.cpu_count", lambda: 16)
    monkeypatch.setattr(
        "easyicu.runtime.parallel_config.get_global_config",
        lambda: SimpleNamespace(max_workers=4),
    )
    monkeypatch.setattr(pa, "cpu_count", lambda: 384)
    monkeypatch.setattr(
        pa,
        "set_cpu_count",
        lambda value: configured_thread_counts.append(value),
    )

    frame = source._read_partitioned_data_optimized(
        partition,
        columns=["icustay_id", "itemid", "valuenum"],
    )

    assert configured_thread_counts == [4]
    assert frame.to_dict("records") == [
        {"icustay_id": 1, "itemid": 10, "valuenum": 1.0}
    ]


def test_parallel_config_uses_two_worker_safe_default_at_16gb(
    monkeypatch,
) -> None:
    monkeypatch.delenv("EASYICU_PARALLEL_MAX_WORKERS", raising=False)
    monkeypatch.setattr(
        "easyicu.runtime.parallel_config.get_cpu_count",
        lambda: 16,
    )

    config = get_parallel_config(override_memory_gb=16)

    assert config.max_workers == 2
    assert get_parallel_config(
        override_memory_gb=16,
        override_workers=6,
    ).max_workers == 6


def test_wide_column_batch_size_is_bounded_on_16gb(monkeypatch) -> None:
    monkeypatch.delenv("EASYICU_WIDE_COLUMN_BATCH_SIZE", raising=False)
    monkeypatch.setattr(
        "easyicu.runtime.parallel_config.get_global_config",
        lambda: SimpleNamespace(total_memory_gb=16),
    )

    assert _resolve_wide_column_batch_size(6) == 2

    monkeypatch.setenv("EASYICU_WIDE_COLUMN_BATCH_SIZE", "3")
    assert _resolve_wide_column_batch_size(6) == 3


def test_wide_table_column_chunks_reassemble_identical_grid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    parquet = tmp_path / "vitalperiodic.parquet"
    pd.DataFrame(
        {
            "patientunitstayid": [1, 1, 2],
            "observationoffset": [0, 30, 60],
            "a": [1.0, 3.0, 5.0],
            "b": [2.0, 4.0, 6.0],
            "c": [10.0, 14.0, 18.0],
        }
    ).to_parquet(parquet, index=False)
    defaults = SimpleNamespace(
        id_var="patientunitstayid",
        index_var="observationoffset",
    )
    source = SimpleNamespace(
        config=SimpleNamespace(
            get_table=lambda name: SimpleNamespace(defaults=defaults),
        ),
        resolve_loader_from_disk=lambda name: parquet,
        # The storage policy's validated set for that loader.
        resolve_target_files=lambda name, loader=None: (parquet,),
    )
    monkeypatch.setenv("EASYICU_WIDE_COLUMN_BATCH_SIZE", "2")

    result = load_wide_table_aggregated(
        source,
        "vitalperiodic",
        ["a", "b", "c"],
        patient_ids=[1, 2],
    ).sort_values(["patientunitstayid", "charttime"]).reset_index(drop=True)

    assert result.columns.tolist() == [
        "patientunitstayid",
        "charttime",
        "a",
        "b",
        "c",
    ]
    assert result[["a", "b", "c"]].to_dict("records") == [
        {"a": 2.0, "b": 3.0, "c": 12.0},
        {"a": 5.0, "b": 6.0, "c": 18.0},
    ]


@pytest.mark.parametrize("column_batch_size", [1, 8])
def test_wide_table_aggregation_orders_stays_and_hours(tmp_path, monkeypatch, column_batch_size):
    parquet = tmp_path / "vitalperiodic.parquet"
    pd.DataFrame({
        "patientunitstayid": [102, 101, 101, 102],
        "observationoffset": [180, 120, 60, 0],
        "a": [4.0, 2.0, 1.0, 3.0],
        "b": [40.0, 20.0, 10.0, 30.0],
    }).to_parquet(parquet, index=False)
    source = SimpleNamespace(
        config=SimpleNamespace(get_table=lambda name: SimpleNamespace(defaults=SimpleNamespace(
            id_var="patientunitstayid", index_var="observationoffset",
        ))),
        resolve_loader_from_disk=lambda name: parquet,
        # The storage policy's validated set for that loader.
        resolve_target_files=lambda name, loader=None: (parquet,),
    )
    monkeypatch.setenv("EASYICU_WIDE_COLUMN_BATCH_SIZE", str(column_batch_size))
    result = load_wide_table_aggregated(source, "vitalperiodic", ["a", "b"], patient_ids=[102, 101])
    # Physical key order must agree across column chunks and the unsplit
    # GROUP BY. Source-file and requested-ID order are deliberately reversed.
    assert result.to_dict("list") == {
        "patientunitstayid": [101, 101, 102, 102], "charttime": [1.0, 2.0, 0.0, 3.0],
        "a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0],
    }


def test_mimic_hospital_value_transform_carries_unit_through_rolling_cte(
    tmp_path,
):
    """FEU exclusion must still see valueuom after hadm-to-stay assignment."""
    labevents_dir = tmp_path / "labevents"
    labevents_dir.mkdir()
    pd.DataFrame(
        {
            "subject_id": [1, 1],
            "hadm_id": [10, 10],
            "itemid": [50915, 50915],
            "charttime": [
                pd.Timestamp("2026-01-01T00:00:00"),
                pd.Timestamp("2026-01-01T01:00:00"),
            ],
            "valuenum": [100.0, 200.0],
            "valueuom": ["ng/mL", "ng/mL FEU"],
        }
    ).to_parquet(labevents_dir / "part.parquet", index=False)
    stays = pd.DataFrame(
        {
            "icustay_id": [7],
            "hadm_id": [10],
            "intime": [pd.Timestamp("2026-01-01T00:00:00")],
            "outtime": [pd.Timestamp("2026-01-02T00:00:00")],
        }
    )
    stays.to_parquet(tmp_path / "icustays.parquet", index=False)

    defaults = SimpleNamespace(
        index_var="charttime", sub_var="itemid", unit_var="valueuom"
    )
    config = SimpleNamespace(
        name="mimic", get_table=lambda name: SimpleNamespace(defaults=defaults)
    )

    def load_table(name, columns=None, filters=None, verbose=False):
        assert name == "icustays"
        return SimpleNamespace(dataframe=lambda: stays)

    source = SimpleNamespace(
        config=config,
        base_path=tmp_path,
        load_table=load_table,
        resolve_bucket_directory=lambda name: None,
        resolve_flat_parquet_directory=lambda name: labevents_dir,
        resolve_target_files=lambda name, *_selector, **_stay: tuple(sorted(labevents_dir.glob("*.parquet"))),
        _get_parquet_columns_for_files=lambda files: set(
            pd.read_parquet(files[0]).columns
        ),
    )
    value_transform = (
        'CASE WHEN regexp_matches(COALESCE(CAST("valueuom" AS VARCHAR), \'\'), '
        "'(?i)FEU') THEN NULL ELSE TRY_CAST(\"valuenum\" AS DOUBLE) END"
    )

    result = load_bucketed_table_aggregated(
        source,
        "labevents",
        "valuenum",
        [50915],
        patient_ids=[7],
        value_transform=value_transform,
    )

    assert result.loc[result["valuenum"].notna(), "valuenum"].tolist() == [100.0]


def test_transformed_bounds_drop_and_count_when_all_values_are_out_of_bounds(tmp_path):
    # lead-r13 bounds ruling: no unbounded retry and no refusal; the values
    # are dropped per row and the read's counts are recorded.
    pd.DataFrame(
        {
            "patientunitstayid": [1] * 100,
            "labresultoffset": [0] * 100,
            "labname": ["x"] * 100,
            "value": [98.6] * 100,
        }
    ).to_parquet(tmp_path / "labs.parquet", index=False)

    defaults = SimpleNamespace(
        index_var="labresultoffset", sub_var="labname", unit_var=None
    )
    config = SimpleNamespace(
        name="eicu", get_table=lambda name: SimpleNamespace(defaults=defaults)
    )
    source = SimpleNamespace(
        config=config,
        base_path=tmp_path,
        resolve_bucket_directory=lambda name: None,
        resolve_flat_parquet_directory=lambda name: tmp_path,
        resolve_target_files=lambda name, *_selector, **_stay: tuple(sorted(tmp_path.glob("*.parquet"))),
        _get_parquet_columns_for_files=lambda files: {
            "patientunitstayid",
            "labresultoffset",
            "labname",
            "value",
        },
    )

    from easyicu.source_bounds import collect_bounds_drops

    with collect_bounds_drops(tmp_path) as drops:
        result = load_bucketed_table_aggregated(
            source,
            "labs",
            "value",
            ["x"],
            patient_ids=[1],
            value_min=32,
            value_max=42,
            value_transform='TRY_CAST("value" AS DOUBLE)',
        )
    assert int(result["value"].notna().sum()) == 0
    assert drops == {"value": {"labs": {
        "dropped_values": 100, "stays_all_out_of_bounds": 1, "stays_read": 1,
    }}}


def test_untransformed_bounds_drop_and_count_when_all_values_are_out_of_bounds(
    tmp_path,
):
    pd.DataFrame(
        {
            "patientunitstayid": [1] * 100,
            "labresultoffset": [0] * 100,
            "labname": ["x"] * 100,
            "value": [98.6] * 100,
        }
    ).to_parquet(tmp_path / "labs.parquet", index=False)

    defaults = SimpleNamespace(
        index_var="labresultoffset", sub_var="labname", unit_var=None
    )
    config = SimpleNamespace(
        name="eicu", get_table=lambda name: SimpleNamespace(defaults=defaults)
    )
    source = SimpleNamespace(
        config=config,
        base_path=tmp_path,
        resolve_bucket_directory=lambda name: None,
        resolve_flat_parquet_directory=lambda name: tmp_path,
        resolve_target_files=lambda name, *_selector, **_stay: tuple(sorted(tmp_path.glob("*.parquet"))),
        _get_parquet_columns_for_files=lambda files: {
            "patientunitstayid",
            "labresultoffset",
            "labname",
            "value",
        },
    )

    from easyicu.source_bounds import collect_bounds_drops

    with collect_bounds_drops(tmp_path) as drops:
        result = load_bucketed_table_aggregated(
            source,
            "labs",
            "value",
            ["x"],
            patient_ids=[1],
            value_min=32,
            value_max=42,
            source_bounds_concept="temp",
        )
    assert result.empty
    assert drops == {"temp": {"labs": {
        "dropped_values": 100, "stays_all_out_of_bounds": 1, "stays_read": 1,
    }}}


def test_untransformed_bounds_keep_empty_out_of_bounds_batch(tmp_path):
    pd.DataFrame(
        {
            "patientunitstayid": [1] * 99,
            "labresultoffset": [0] * 99,
            "labname": ["x"] * 99,
            "value": [98.6] * 99,
        }
    ).to_parquet(tmp_path / "labs.parquet", index=False)

    defaults = SimpleNamespace(
        index_var="labresultoffset", sub_var="labname", unit_var=None
    )
    config = SimpleNamespace(
        name="eicu", get_table=lambda name: SimpleNamespace(defaults=defaults)
    )
    source = SimpleNamespace(
        config=config,
        base_path=tmp_path,
        resolve_bucket_directory=lambda name: None,
        resolve_flat_parquet_directory=lambda name: tmp_path,
        resolve_target_files=lambda name, *_selector, **_stay: tuple(sorted(tmp_path.glob("*.parquet"))),
        _get_parquet_columns_for_files=lambda files: {
            "patientunitstayid",
            "labresultoffset",
            "labname",
            "value",
        },
    )

    result = load_bucketed_table_aggregated(
        source,
        "labs",
        "value",
        ["x"],
        patient_ids=[1],
        value_min=32,
        value_max=42,
    )

    assert result.empty
    assert result.attrs["easyicu_bounds_loader"] == {
        "bounds_raw_transformed_non_null": 99,
        "bounds_bounded_transformed_non_null": 0,
        "bounds_bounded_aggregate_non_null": 0,
    }


def test_sofa2_overlay_bounds_are_included(monkeypatch):
    monkeypatch.setattr(extraction_api, "_CONCEPT_BOUNDS_CACHE", None)
    assert extraction_api._load_concept_bounds_map()["motor_response"] == (1.0, 6.0)


def test_converter_rejects_partial_shards_after_interrupted_status(tmp_path):
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("id,value\n1,2\n", encoding="utf-8")
    shard_dir = tmp_path / "events"
    shard_dir.mkdir()
    pd.DataFrame({"id": [1], "value": [2]}).to_parquet(shard_dir / "1.parquet")
    converter = DataConverter(tmp_path, database="miiv", verbose=False)
    converter._status[csv_path.name] = {
        "status": ConversionStatus.CONVERTING,
        "shards": 2,
    }
    converter._invalidate_dir_caches()

    needed, reason = converter._is_conversion_needed(csv_path)

    assert needed is True
    assert "converting" in reason


def test_converter_rejects_valid_looking_orphan_shard_without_completion_status(
    tmp_path,
):
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("id,value\n1,2\n", encoding="utf-8")
    shard_dir = tmp_path / "events"
    shard_dir.mkdir()
    pd.DataFrame({"id": [1], "value": [2]}).to_parquet(shard_dir / "1.parquet")
    converter = DataConverter(tmp_path, database="miiv", verbose=False)
    converter._status.clear()
    converter._invalidate_dir_caches()

    needed, reason = converter._is_conversion_needed(csv_path)

    assert needed is True
    assert "without completed conversion status" in reason


def test_converter_rejects_valid_looking_single_file_without_completion_status(
    tmp_path,
):
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("id,value\n1,2\n", encoding="utf-8")
    pd.DataFrame({"id": [1], "value": [2]}).to_parquet(
        tmp_path / "events.parquet", index=False
    )
    converter = DataConverter(tmp_path, database="miiv", verbose=False)
    converter._status.clear()

    needed, reason = converter._is_conversion_needed(csv_path)

    assert needed is True
    assert "without completed conversion status" in reason


@pytest.mark.parametrize(
    "status",
    [
        ConversionStatus.PENDING,
        ConversionStatus.CONVERTING,
        ConversionStatus.FAILED,
        "in_progress",
    ],
)
def test_converter_rejects_single_file_with_non_completed_status(tmp_path, status):
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("id,value\n1,2\n", encoding="utf-8")
    pd.DataFrame({"id": [1], "value": [2]}).to_parquet(
        tmp_path / "events.parquet", index=False
    )
    converter = DataConverter(tmp_path, database="miiv", verbose=False)
    converter._status[csv_path.name] = {
        "status": status,
        "row_count": 1,
        "shards": 0,
    }

    needed, _ = converter._is_conversion_needed(csv_path)

    assert needed is True


def test_converter_rejects_completed_single_file_with_row_count_mismatch(tmp_path):
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("id,value\n1,2\n", encoding="utf-8")
    pd.DataFrame({"id": [1], "value": [2]}).to_parquet(
        tmp_path / "events.parquet", index=False
    )
    converter = DataConverter(tmp_path, database="miiv", verbose=False)
    converter._status[csv_path.name] = {
        "status": ConversionStatus.COMPLETED,
        "row_count": 2,
        "shards": 0,
        "source_content_receipt": file_content_receipt(csv_path),
    }

    needed, reason = converter._is_conversion_needed(csv_path)

    assert needed is True
    assert "row-count mismatch" in reason

    converter._status[csv_path.name]["row_count"] = 1
    needed, reason = converter._is_conversion_needed(csv_path)

    assert needed is False
    assert reason == "already converted and verified"


def test_converter_rejects_completed_shards_with_row_count_mismatch(tmp_path):
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("id,value\n1,2\n2,3\n", encoding="utf-8")
    shard_dir = tmp_path / "events"
    shard_dir.mkdir()
    pd.DataFrame({"id": [1], "value": [2]}).to_parquet(
        shard_dir / "1.parquet", index=False
    )
    pd.DataFrame({"id": [2], "value": [3]}).to_parquet(
        shard_dir / "2.parquet", index=False
    )
    converter = DataConverter(tmp_path, database="miiv", verbose=False)
    converter._status[csv_path.name] = {
        "status": ConversionStatus.COMPLETED,
        "row_count": 3,
        "shards": 2,
        "source_content_receipt": file_content_receipt(csv_path),
    }
    converter._invalidate_dir_caches()

    needed, reason = converter._is_conversion_needed(csv_path)

    assert needed is True
    assert "row-count mismatch" in reason

    converter._status[csv_path.name]["row_count"] = 2
    needed, reason = converter._is_conversion_needed(csv_path)

    assert needed is False
    assert reason == "sharded (2 files)"


def test_converter_rebuilds_same_size_source_with_restored_mtime(tmp_path):
    csv_path = tmp_path / "events.csv"
    csv_path.write_bytes(b"id,value\n1,old\n")
    converter = DataConverter(tmp_path, database="miiv", verbose=False)
    first = converter.convert_all(force=True)
    assert first[csv_path.name]["status"] == ConversionStatus.COMPLETED
    original_stat = csv_path.stat()

    csv_path.write_bytes(b"id,value\n1,new\n")
    os.utime(csv_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    needed, reason = converter._is_conversion_needed(csv_path)
    assert needed is True
    assert reason == "source content changed since completed conversion"

    second = converter.convert_all()
    assert second[csv_path.name]["status"] == ConversionStatus.COMPLETED
    converted = pd.read_parquet(tmp_path / "events.parquet")
    assert converted["value"].tolist() == ["new"]


def test_converter_manifest_keeps_existing_tables_and_counts_bad_rows(tmp_path):
    good = tmp_path / "good.csv"
    bad = tmp_path / "bad.csv"
    good.write_text("id,value\n1,2\n", encoding="utf-8")
    bad.write_text("id,value\n1,2\n3,4,5\n6,7\n", encoding="utf-8")
    converter = DataConverter(
        tmp_path, database="miiv", parallel_workers=1, verbose=False
    )
    first = converter.convert_all(force=True)
    assert first["bad.csv"]["bad_rows_skipped"] == 1

    good.write_text("id,value\n1,2\n2,3\n", encoding="utf-8")
    os.utime(good, None)
    converter.convert_all()
    manifest = json.loads((tmp_path / "conversion_manifest.json").read_text())
    tables = {row["file"]: row for row in manifest["tables"]}

    assert set(tables) == {"good.csv", "bad.csv"}
    assert tables["bad.csv"]["bad_rows_skipped"] == 1


def test_converter_distinguishes_mimic_generations_and_rejects_ambiguity(tmp_path):
    mimic3 = tmp_path / "mimic-iii"
    mimic4 = tmp_path / "mimic-iv"
    ambiguous = tmp_path / "mimic-dataset"
    for path in (mimic3, mimic4, ambiguous):
        path.mkdir()

    assert DataConverter(mimic3, verbose=False).database == "mimic"
    assert DataConverter(mimic4, verbose=False).database == "miiv"
    with pytest.raises(ValueError, match="Ambiguous MIMIC"):
        DataConverter(ambiguous, verbose=False)


def _mimic_iv_tables(icu, patients):
    """The MIMIC-IV tables the outcome loader reads; no admission records a
    death time."""

    admissions = pd.DataFrame({"hadm_id": icu["hadm_id"].unique(), "deathtime": None})
    tables = {"icustays": icu, "patients": patients, "admissions": admissions}
    return lambda database, data_path, table: tables[table]


def test_mimic_same_calendar_day_death_is_not_lost_to_midnight_precision(
    monkeypatch,
):
    import easyicu.scores.outcomes as outcomes

    icu = pd.DataFrame(
        {
            "subject_id": [1],
            "hadm_id": [10],
            "stay_id": [1],
            "intime": ["2020-01-01 12:00:00"],
            "los": [2.0],
        }
    )
    # MIMIC patients.dod is a DATE, represented at midnight after loading.
    patients = pd.DataFrame({"subject_id": [1], "dod": ["2020-01-01"]})
    monkeypatch.setattr(outcomes, "_raw_table", _mimic_iv_tables(icu, patients))

    result = outcomes.load_outcomes("miiv").set_index("stay_id")

    # Subtracting exact intime from a midnight DATE produces -0.5 days and
    # previously censored a genuine same-day death. Calendar-day semantics are
    # the strongest resolution the source can support.
    assert bool(result.loc[1, "mort_28d"]) is True


def test_mimic_absent_death_date_has_documented_one_year_followup(monkeypatch):
    import easyicu.scores.outcomes as outcomes

    icu = pd.DataFrame(
        {
            "subject_id": [1],
            "hadm_id": [10],
            "stay_id": [1],
            "intime": ["2020-01-01 12:00:00"],
            "los": [2.0],
        }
    )
    patients = pd.DataFrame({"subject_id": [1], "dod": [None]})
    monkeypatch.setattr(outcomes, "_raw_table", _mimic_iv_tables(icu, patients))

    result = outcomes.load_outcomes("miiv").set_index("stay_id")

    assert bool(result.loc[1, "mort_28d"]) is False
    assert bool(result.loc[1, "mort_365d"]) is False
    assert result.loc[1, "followup_days_28d"] == 28.0
    assert result.loc[1, "followup_days_365d"] == 365.0


def test_fixed_horizon_followup_time_pairs_events_and_censoring(monkeypatch):
    import easyicu.scores.outcomes as outcomes

    icu = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "hadm_id": [10, 20, 30],
            "stay_id": [1, 2, 3],
            "intime": ["2020-01-01", "2020-01-01", "2020-01-01"],
            "los": [2.0, 2.0, 2.0],
        }
    )
    patients = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "dod": ["2020-01-06", "2020-02-10", None],
        }
    )
    monkeypatch.setattr(outcomes, "_raw_table", _mimic_iv_tables(icu, patients))

    result = outcomes.load_outcomes("miiv").set_index("stay_id")

    assert bool(result.loc[1, "mort_28d"]) is True
    assert result.loc[1, "followup_days_28d"] == 5.0
    assert bool(result.loc[2, "mort_28d"]) is False
    assert result.loc[2, "followup_days_28d"] == 28.0
    assert bool(result.loc[3, "mort_28d"]) is False
    assert result.loc[3, "followup_days_28d"] == 28.0


def test_an_icu_stay_without_a_discharge_time_leaves_free_days_unknown(
    monkeypatch,
):
    # The current-extract LOS is never the ICU time: a stay in the window
    # without a discharge time leaves every overlapping window unknown.
    import easyicu.scores.outcomes as outcomes

    icu = pd.DataFrame(
        {
            "subject_id": [1, 1],
            "hadm_id": [10, 10],
            "stay_id": [2, 1],
            "intime": ["2020-01-04", "2020-01-01"],
            "outtime": [None, "2020-01-03"],
            "los": [None, 2.0],
        }
    )
    patients = pd.DataFrame({"subject_id": [1], "dod": [None]})
    monkeypatch.setattr(outcomes, "_raw_table", _mimic_iv_tables(icu, patients))

    result = outcomes.load_outcomes("miiv")
    assert result["stay_id"].tolist() == [2, 1]
    assert result["icu_free_days_28"].isna().all()
    assert "icu_readmission" not in result.columns


def test_sic_outcomes_use_icu_origin_and_censor_incomplete_365d_followup(
    monkeypatch,
):
    import easyicu.scores.outcomes as outcomes

    cases = pd.DataFrame(
        {
            "CaseID": [1, 2, 3],
            "TimeOfStay": [90_000, 90_000, 90_000],
            "ICUOffset": [3_600, 3_600, 3_600],
            "OffsetOfDeath": [None, None, 3_600 + 300 * 86_400],
            "EstimatedSurvivalObservationTime": [3077, 3076, 3076],
        }
    )
    from tests.support.sic_references import sic_raw_tables

    monkeypatch.setattr(outcomes, "_raw_table", sic_raw_tables(cases))

    result = outcomes.load_outcomes("sic").set_index("CaseID")

    assert "icu_free_days_28" not in result.columns
    # Registry follow-up is bounded from primary admission, not the later
    # intensive-or-intermediate care transfer.  A one-year category an hour
    # short of the horizon keeps its survivor (audit round 2, R2-03: censoring
    # every such survivor defined the horizon for the dead only); a six-month
    # category cannot reach 365 days.
    assert bool(result.loc[1, "mort_365d"]) is False
    assert pd.isna(result.loc[2, "mort_365d"])
    assert bool(result.loc[2, "mort_90d"]) is False
    # The same six-month category leaves the dead as missing at 365 days too:
    # defining the horizon for its deaths only is the selection bias of
    # R3-08 (2026-11-04 audit round 3).  The 90-day horizon is reached.
    assert pd.isna(result.loc[3, "mort_365d"])
    assert bool(result.loc[3, "mort_90d"]) is False


def test_sic_outcomes_accept_the_public_sicdb_alias(monkeypatch):
    import easyicu.scores.outcomes as outcomes

    observed = {}

    def fake_sic(database, _data_path):
        observed["database"] = database
        return pd.DataFrame(
            {
                "_stay": [1],
                "days_to_death": [400.0],
                "los_days": [1.0],
                "followup_days": [365.0],
                "hadm_id": [pd.NA],
            }
        )

    monkeypatch.setattr(outcomes, "_sic_stay_death_days", fake_sic)

    result = outcomes.load_outcomes("sicdb")

    assert observed["database"] == "sic"
    assert result["CaseID"].tolist() == [1]


def test_eicu_hospital_mortality_does_not_publish_vfd28(monkeypatch):
    import easyicu.scores.outcomes as outcomes

    apache = pd.DataFrame(
        {
            "apacheversion": ["IVa", "IVa", "IVa"],
            "patientunitstayid": [1, 2, 3],
            "actualventdays": [-1, None, 4],
            "actualhospitalmortality": ["ALIVE", "EXPIRED", "ALIVE"],
        }
    )
    monkeypatch.setattr(outcomes, "_raw_table", lambda *args: apache)

    result = outcomes.load_outcomes("eicu")

    assert result.empty
    assert "vent_free_days_28" not in result.columns


def test_eicu_microbiology_uses_all_stays_as_denominator(monkeypatch):
    import easyicu.scores.microbiology as microbiology

    patient = pd.DataFrame({"patientunitstayid": [1, 2]})
    microlab = pd.DataFrame(
        {
            "patientunitstayid": [1],
            "organism": ["E. coli"],
            "culturesite": ["Blood"],
        }
    )
    monkeypatch.setattr(microbiology, "build_datasource", lambda *args: object())
    monkeypatch.setattr(
        microbiology,
        "table_df",
        lambda ds, table: patient if table == "patient" else microlab,
    )

    result = microbiology.load_microbiology("eicu").set_index("patientunitstayid")

    assert bool(result.loc[1, "culture_positive"]) is True
    assert bool(result.loc[2, "culture_positive"]) is False
    assert bool(result.loc[2, "bld_culture_positive"]) is False


def test_miiv_microbiology_pushes_stay_subset_to_hospital_table(monkeypatch):
    import easyicu.scores.microbiology as microbiology

    captured = []
    stays = pd.DataFrame({"stay_id": [10, 20], "hadm_id": [100, 200]})
    microbiology_events = pd.DataFrame(
        {
            "hadm_id": [100, 200],
            "org_name": ["E. coli", ""],
            "spec_type_desc": ["Blood", "Urine"],
        }
    )

    monkeypatch.setattr(microbiology, "build_datasource", lambda *args: object())

    def fake_table(_ds, table, columns=None, filters=None):
        if table == "icustays":
            return stays
        captured.extend(filters or [])
        assert table == "microbiologyevents"
        return microbiology_events[microbiology_events["hadm_id"].isin([100])]

    monkeypatch.setattr(microbiology, "table_df", fake_table)

    result = microbiology.load_microbiology("miiv", patient_ids=[10])

    assert result["stay_id"].tolist() == [10]
    assert bool(result["culture_positive"].iloc[0]) is True
    assert len(captured) == 1
    assert captured[0].column == "hadm_id"
    assert captured[0].value == [100]


def test_icu_table_to_wide_sets_column_axis_name():
    table = ICUTable(
        pd.DataFrame({"stay_id": [1, 1], "hour": [0, 1], "value": [2.0, 3.0]}),
        id_columns=["stay_id"],
        index_column="hour",
        value_column="value",
    )

    wide = table.to_wide("hr")

    assert wide.columns.name == "hr"
    assert wide.loc[1].tolist() == [2.0, 3.0]
