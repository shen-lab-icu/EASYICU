"""Correctness of native export across nested worker batches."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.api import extraction as extraction_api
from tests.support.desktop_renal_cohort import desktop_renal_cohort as desktop_renal_cohort


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
