"""Persist extraction worker failures before temporary directories disappear."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from easyicu.api import extraction as api


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _inventory_sha256(entries: list[dict]) -> str:
    canonical = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256(canonical)


def _failure_records(output_dir: Path) -> list[dict]:
    paths = sorted((output_dir / api._EXTRACTION_WORKER_FAILURE_DIRNAME).glob(
        "worker-failure-*.json"
    ))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def test_in_worker_exception_is_persisted_before_temp_cleanup(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "output"
    worker_roots: list[Path] = []

    def failing_module(*args, **_kwargs):
        module_dir = Path(args[6])
        module_dir.mkdir(parents=True, exist_ok=True)
        (module_dir / "partial.parquet").write_bytes(b"partial output")
        raise RuntimeError("synthetic worker stop")

    class InlineProcess:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            self.exitcode = None

        def start(self):
            worker_roots.append(Path(self.args[6]))
            try:
                self.target(*self.args)
            except BaseException:
                self.exitcode = 1
            else:
                self.exitcode = 0

        def join(self):
            return None

    monkeypatch.setattr(api, "_extract_worker_env_setup", lambda *_args: None)
    monkeypatch.setattr(
        api,
        "_get_extraction_mp_context",
        lambda _mp: SimpleNamespace(Process=InlineProcess),
    )
    monkeypatch.setattr(
        "easyicu.api.keep_cache",
        lambda **_kwargs: nullcontext(object()),
    )
    monkeypatch.setattr(api, "_run_module_extraction", failing_module)
    monkeypatch.setattr(
        api,
        "_native_export_runtime_provenance",
        lambda: {"fixture": True},
    )

    result = api.extract_database(
        "miiv",
        data_path=str(tmp_path / "data"),
        output_dir=str(output_dir),
        modules=["demographics"],
        patient_ids={"stay_id": [1]},
        batch_size=1,
        group_modules=False,
        native_export_v2=False,
        verbose=False,
    )

    records = _failure_records(output_dir)
    assert len(records) == 1
    record = records[0]
    assert record["phase"] == "module"
    assert record["modules"] == ["demographics"]
    assert record["worker_exit_code"] == 0
    assert record["exception_type"] == "RuntimeError"
    assert "synthetic worker stop" in record["traceback"]
    assert record["traceback_sha256"] == _sha256(
        record["traceback"].encode("utf-8")
    )
    assert record["partial_outputs"] == [
        {
            "file": "demographics/partial.parquet",
            "bytes": len(b"partial output"),
            "sha256": _sha256(b"partial output"),
        }
    ]
    assert record["partial_outputs_sha256"] == _inventory_sha256(
        record["partial_outputs"]
    )
    assert record["runtime_provenance"] == {"fixture": True}
    assert record["private"] is True
    assert record["external_llm_allowed"] is False
    assert result["worker_failures"][0]["exception_type"] == "RuntimeError"
    assert result["worker_failures"][0]["file"] == (
        f"{api._EXTRACTION_WORKER_FAILURE_DIRNAME}/"
        f"worker-failure-{record['failure_id']}.json"
    )
    assert all(not worker_root.exists() for worker_root in worker_roots)


def test_hard_crash_fallback_records_exit_and_partial_outputs(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    worker_roots: list[Path] = []

    class HardCrashProcess:
        def __init__(self, *, target, args, daemon):
            self.args = args
            self.exitcode = None

        def start(self):
            worker_root = Path(self.args[6])
            worker_roots.append(worker_root)
            module_dir = worker_root / "demographics"
            module_dir.mkdir(parents=True)
            (module_dir / "partial.parquet").write_bytes(b"crash residue")
            self.exitcode = 1

        def join(self):
            return None

    monkeypatch.setattr(
        api,
        "_get_extraction_mp_context",
        lambda _mp: SimpleNamespace(Process=HardCrashProcess),
    )

    result = api.extract_database(
        "miiv",
        data_path=str(tmp_path / "data"),
        output_dir=str(output_dir),
        modules=["demographics"],
        patient_ids={"stay_id": [1]},
        batch_size=1,
        group_modules=False,
        native_export_v2=False,
        verbose=False,
    )

    records = _failure_records(output_dir)
    assert len(records) == 1
    record = records[0]
    assert record["failure_kind"] == "worker_exit_without_failure_record"
    assert record["worker_exit_code"] == 1
    assert record["failed_modules"] == ["demographics"]
    assert record["partial_outputs"] == [
        {
            "file": "demographics/partial.parquet",
            "bytes": len(b"crash residue"),
            "sha256": _sha256(b"crash residue"),
        }
    ]
    assert record["partial_outputs_sha256"] == _inventory_sha256(
        record["partial_outputs"]
    )
    assert record["private"] is True
    assert record["external_llm_allowed"] is False
    assert result["worker_failures"][0]["failure_id"] == record["failure_id"]
    assert all(not worker_root.exists() for worker_root in worker_roots)


def test_special_worker_exception_records_special_phase(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    worker_roots: list[Path] = []

    def failing_special(*_args, **_kwargs):
        raise RuntimeError("synthetic special worker stop")

    class InlineProcess:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            self.exitcode = None

        def start(self):
            worker_roots.append(Path(self.args[6]))
            try:
                self.target(*self.args)
            except BaseException:
                self.exitcode = 1
            else:
                self.exitcode = 0

        def join(self):
            return None

    monkeypatch.setattr(api, "_extract_worker_env_setup", lambda *_args: None)
    monkeypatch.setattr(
        api,
        "_get_extraction_mp_context",
        lambda _mp: SimpleNamespace(Process=InlineProcess),
    )
    monkeypatch.setattr(
        "easyicu.api.keep_cache",
        lambda **_kwargs: nullcontext(object()),
    )
    monkeypatch.setattr(api, "_run_special_extraction", failing_special)
    monkeypatch.setattr(
        api,
        "_native_export_runtime_provenance",
        lambda: {"fixture": True},
    )

    result = api.extract_database(
        "miiv",
        data_path=str(tmp_path / "data"),
        output_dir=str(output_dir),
        modules=["sepsis3_sofa1"],
        patient_ids={"stay_id": [1]},
        batch_size=1,
        group_modules=False,
        native_export_v2=False,
        verbose=False,
    )

    records = _failure_records(output_dir)
    assert len(records) == 1
    record = records[0]
    assert record["phase"] == "special_extraction"
    assert record["modules"] == []
    assert record["special_modules"] == ["sepsis3_sofa1"]
    assert record["exception_type"] == "RuntimeError"
    assert "synthetic special worker stop" in record["traceback"]
    assert result["worker_failures"][0]["phase"] == "special_extraction"
    assert all(not worker_root.exists() for worker_root in worker_roots)
