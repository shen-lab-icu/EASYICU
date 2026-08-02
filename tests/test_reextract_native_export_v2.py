"""Contract tests for the sequential native-v2 re-export launcher."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "reextract_native_export_v2.py"
SPEC = importlib.util.spec_from_file_location("reextract_native_export_v2", TOOL)
assert SPEC and SPEC.loader
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def _memory_info(
    *,
    total_gb: float,
    available_gb: float,
    source: str = "host",
    cgroup_limit_gb: float | None = None,
    cgroup_current_gb: float | None = None,
):
    return launcher.EffectiveMemoryInfo(
        host_total_mb=1536 * 1024,
        host_available_mb=1200 * 1024,
        effective_total_mb=total_gb * 1024,
        effective_available_mb=available_gb * 1024,
        source=source,
        cgroup_limit_mb=(
            None if cgroup_limit_gb is None else cgroup_limit_gb * 1024
        ),
        cgroup_current_mb=(
            None if cgroup_current_gb is None else cgroup_current_gb * 1024
        ),
    )


class _Package:
    column_metadata_sha256 = "a" * 64
    concept_index = {"age": object()}
    missing_selected_concepts = ()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


@pytest.mark.parametrize(
    ("available_memory_mb", "expected_budget_mb"),
    [
        (8 * 1024, 2560),
        (12 * 1024, 4 * 1024),
        (20 * 1024, 6656),
        (24 * 1024, 8 * 1024),
        (256 * 1024, 8 * 1024),
    ],
)
def test_nested_workset_budget_scales_with_current_available_memory(
    available_memory_mb: float,
    expected_budget_mb: int,
) -> None:
    assert (
        launcher._adaptive_oneshot_budget_mb(available_memory_mb)
        == expected_budget_mb
    )


@pytest.mark.parametrize(
    ("available_memory_mb", "cpu_count", "expected"),
    [
        (
            16 * 1024,
            8,
            {
                "duckdb_threads": "1",
                "duckdb_memory_limit": "1GB",
                "parallel_max_workers": "1",
                "cache_budget_mb": "256",
            },
        ),
        (
            24 * 1024,
            8,
            {
                "duckdb_threads": "2",
                "duckdb_memory_limit": "2GB",
                "parallel_max_workers": "2",
                "cache_budget_mb": "2048",
            },
        ),
        (
            32 * 1024,
            8,
            {
                "duckdb_threads": "4",
                "duckdb_memory_limit": "4GB",
                "parallel_max_workers": "4",
                "cache_budget_mb": "6144",
            },
        ),
        (
            256 * 1024,
            384,
            {
                "duckdb_threads": "8",
                "duckdb_memory_limit": "8GB",
                "parallel_max_workers": "8",
                "cache_budget_mb": "8192",
            },
        ),
    ],
)
def test_one_shot_runtime_scales_without_weakening_laptop_floor(
    available_memory_mb: float,
    cpu_count: int,
    expected: dict[str, str],
) -> None:
    assert (
        launcher._one_shot_runtime_limits(available_memory_mb, cpu_count)
        == expected
    )


def test_launcher_is_sequential_private_and_uses_adaptive_external_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(launcher, "DEFAULT_DATA_PATHS", {"miiv": str(source)})
    monkeypatch.setattr(
        launcher, "_adaptive_oneshot_budget_mb", lambda _available: 4 * 1024
    )
    monkeypatch.setattr(
        launcher,
        "get_effective_memory_info",
        lambda: _memory_info(total_gb=16, available_gb=8),
    )

    def fake_extract(database, **kwargs):
        calls.append({"database": database, **kwargs})
        assert os.environ["TMPDIR"] == str(tmp_path / "out" / ".runtime_tmp")
        assert os.environ["TMP"] == str(tmp_path / "out" / ".runtime_tmp")
        assert os.environ["TEMP"] == str(tmp_path / "out" / ".runtime_tmp")
        assert os.environ["EASYICU_DUCKDB_TEMP_DIR"] == str(
            tmp_path / "out" / ".runtime_spill"
        )
        assert os.environ["EASYICU_DUCKDB_THREADS"] == "1"
        assert os.environ["EASYICU_DUCKDB_MEMORY_LIMIT"] == "1GB"
        assert os.environ["EASYICU_PARALLEL_MAX_WORKERS"] == "1"
        assert os.environ["EASYICU_ONESHOT_BUDGET_MB"] == "4096"
        assert os.environ["EASYICU_OVERRIDE_MEMORY_GB"] == "8.000000"
        out = Path(kwargs["output_dir"])
        out.mkdir(mode=0o700)
        spill = out / ".easyicu_spill"
        spill.mkdir()
        (spill / "duckdb_temp_storage.tmp").write_text("temporary")
        (out / "_manifest.json").write_text(
            json.dumps({"unavailable_modules": []}), encoding="utf-8"
        )
        return {
            "num_patients": 10,
            "native_export_v2": {"output_validation_reads": 19},
        }

    monkeypatch.setattr(launcher, "extract_database", fake_extract)
    monkeypatch.setattr(launcher, "open_export_package", lambda _path: _Package())
    args = launcher._parse_args(
        ["--output-root", str(tmp_path / "out"), "--databases", "miiv"]
    )

    manifest = launcher.run(args)

    assert manifest["status"] == "verified"
    assert calls == [
        {
            "database": "miiv",
            "data_path": str(source),
            "output_dir": tmp_path / "out" / "miiv",
            "batch_size": None,
            "native_export_v2": True,
            "stream_output_batches": True,
            "verbose": True,
        }
    ]
    persisted = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert persisted["sources"]["miiv"]["status"] == "verified"
    assert persisted["runtime_limits"]["nested_workset_budget_mb"] == 4096
    assert persisted["runtime_detection"] == {
        "source": "host",
        "host_total_mb": 1572864,
        "host_available_mb": 1228800,
        "effective_total_mb": 16384,
        "effective_available_mb": 8192,
        "cgroup_limit_mb": None,
        "cgroup_current_mb": None,
        "selection_basis": "effective_available_memory",
        "logical_cpu_count": os.cpu_count() or 1,
        "selection_tier": "portable_lt24gib",
        "parallel_config_override_memory_gb": 8.0,
    }
    assert persisted["sources"]["miiv"]["spill_directory_removed"] is True
    assert not (tmp_path / "out" / "miiv" / ".easyicu_spill").exists()
    assert not (tmp_path / "out" / ".runtime_tmp").exists()
    assert not (tmp_path / "out" / ".runtime_spill").exists()
    assert oct((tmp_path / "out").stat().st_mode & 0o777) == "0o700"


def test_launcher_rejects_an_existing_output_root(tmp_path: Path) -> None:
    out = tmp_path / "existing"
    out.mkdir()
    args = launcher._parse_args(["--output-root", str(out), "--databases", "miiv"])

    with pytest.raises(ValueError, match="must be new"):
        launcher.run(args)


def test_one_shot_launcher_keeps_external_runtime_but_disables_auto_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(launcher, "DEFAULT_DATA_PATHS", {"miiv": str(source)})
    monkeypatch.setattr(
        launcher,
        "_one_shot_runtime_limits",
        lambda *_args: {
            "duckdb_threads": "8",
            "duckdb_memory_limit": "8GB",
            "parallel_max_workers": "8",
            "cache_budget_mb": "8192",
        },
    )

    def fake_extract(database, **kwargs):
        calls.append({"database": database, **kwargs})
        assert os.environ["TMPDIR"] == str(tmp_path / "out" / ".runtime_tmp")
        assert os.environ["EASYICU_DUCKDB_TEMP_DIR"] == str(
            tmp_path / "out" / ".runtime_spill"
        )
        assert "EASYICU_ONESHOT_BUDGET_MB" not in os.environ
        assert os.environ["EASYICU_DUCKDB_THREADS"] == "8"
        assert os.environ["EASYICU_DUCKDB_MEMORY_LIMIT"] == "8GB"
        assert os.environ["EASYICU_PARALLEL_MAX_WORKERS"] == "8"
        assert os.environ["EASYICU_CACHE_BUDGET_MB"] == "8192"
        out = Path(kwargs["output_dir"])
        out.mkdir(mode=0o700)
        (out / "_manifest.json").write_text(
            json.dumps({"unavailable_modules": []}), encoding="utf-8"
        )
        return {
            "num_patients": 10,
            "native_export_v2": {"output_validation_reads": 19},
        }

    monkeypatch.setattr(launcher, "extract_database", fake_extract)
    monkeypatch.setattr(launcher, "open_export_package", lambda _path: _Package())
    args = launcher._parse_args(
        ["--output-root", str(tmp_path / "out"), "--databases", "miiv", "--one-shot"]
    )

    manifest = launcher.run(args)

    assert manifest["extraction_mode"] == "one_shot_all_patients"
    assert manifest["runtime_limits"] == {
        "duckdb_threads": 8,
        "duckdb_memory_limit": "8GB",
        "parallel_max_workers": 8,
        "cache_budget_mb": 8192,
        "nested_workset_budget_mb": None,
    }
    assert calls == [
        {
            "database": "miiv",
            "data_path": str(source),
            "output_dir": tmp_path / "out" / "miiv",
            "batch_size": None,
            "native_export_v2": True,
            "stream_output_batches": False,
            "verbose": True,
        }
    ]


@pytest.mark.parametrize(
    ("available_gb", "expected_workers", "expected_memory", "expected_cache"),
    [
        (8, "1", "1GB", "256"),
        (16, "1", "1GB", "256"),
        (24, "2", "2GB", "2048"),
        (32, "4", "4GB", "6144"),
        (64, "8", "8GB", "8192"),
    ],
)
def test_streamed_runtime_uses_same_bounded_server_tiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    available_gb: int,
    expected_workers: str,
    expected_memory: str,
    expected_cache: str,
) -> None:
    monkeypatch.setattr(
        launcher,
        "get_effective_memory_info",
        lambda: _memory_info(total_gb=available_gb, available_gb=available_gb),
    )
    monkeypatch.setattr(launcher.os, "cpu_count", lambda: 32)

    (tmp_path / "runtime").mkdir()
    prior, prior_tempdir, detection = launcher._configure_external_runtime(
        tmp_path / "runtime",
        one_shot=False,
    )
    try:
        assert os.environ["EASYICU_DUCKDB_THREADS"] == expected_workers
        assert os.environ["EASYICU_DUCKDB_MEMORY_LIMIT"] == expected_memory
        assert os.environ["EASYICU_PARALLEL_MAX_WORKERS"] == expected_workers
        assert os.environ["EASYICU_CACHE_BUDGET_MB"] == expected_cache
        assert detection["effective_available_mb"] == available_gb * 1024
    finally:
        launcher._restore_runtime(prior, prior_tempdir)


def test_runtime_restores_all_environment_including_memory_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_keys = (
        "TMPDIR",
        "TMP",
        "TEMP",
        "EASYICU_DUCKDB_TEMP_DIR",
        "EASYICU_DUCKDB_THREADS",
        "EASYICU_DUCKDB_MEMORY_LIMIT",
        "EASYICU_PARALLEL_MAX_WORKERS",
        "EASYICU_CACHE_BUDGET_MB",
        "EASYICU_ONESHOT_BUDGET_MB",
        "EASYICU_OVERRIDE_MEMORY_GB",
    )
    expected = {key: f"prior-{index}" for index, key in enumerate(runtime_keys)}
    for key, value in expected.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        launcher,
        "get_effective_memory_info",
        lambda: _memory_info(
            total_gb=13,
            available_gb=8,
            source="cgroup_v2",
            cgroup_limit_gb=13,
            cgroup_current_gb=5,
        ),
    )

    (tmp_path / "runtime").mkdir()
    prior, prior_tempdir, _detection = launcher._configure_external_runtime(
        tmp_path / "runtime",
        one_shot=False,
    )
    launcher._restore_runtime(prior, prior_tempdir)

    assert {key: os.environ.get(key) for key in runtime_keys} == expected


def test_failed_export_also_restores_runtime_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(launcher, "DEFAULT_DATA_PATHS", {"miiv": str(source)})
    monkeypatch.setattr(
        launcher,
        "get_effective_memory_info",
        lambda: _memory_info(total_gb=64, available_gb=64),
    )
    monkeypatch.setenv("TMPDIR", "prior-tmpdir")
    monkeypatch.setenv("EASYICU_OVERRIDE_MEMORY_GB", "prior-memory")
    monkeypatch.setattr(
        launcher,
        "extract_database",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    args = launcher._parse_args(
        ["--output-root", str(tmp_path / "out"), "--databases", "miiv"]
    )

    with pytest.raises(RuntimeError, match="boom"):
        launcher.run(args)

    assert os.environ["TMPDIR"] == "prior-tmpdir"
    assert os.environ["EASYICU_OVERRIDE_MEMORY_GB"] == "prior-memory"
    failed = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert failed["status"] == "failed"
    assert failed["sources"]["miiv"]["status"] == "failed"


@pytest.mark.parametrize(
    ("requested_batch_size", "expected_batch_size"),
    [(None, None), (50_000, 50_000)],
)
def test_streamed_server_keeps_eicu_outer_batches_and_explicit_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested_batch_size: int | None,
    expected_batch_size: int | None,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(launcher, "DEFAULT_DATA_PATHS", {"eicu": str(source)})
    monkeypatch.setattr(
        launcher,
        "get_effective_memory_info",
        lambda: _memory_info(total_gb=128, available_gb=96),
    )
    monkeypatch.setattr(launcher.os, "cpu_count", lambda: 32)
    calls: list[dict[str, object]] = []

    def fake_extract(database, **kwargs):
        calls.append({"database": database, **kwargs})
        assert os.environ["EASYICU_PARALLEL_MAX_WORKERS"] == "8"
        out = Path(kwargs["output_dir"])
        out.mkdir(mode=0o700)
        (out / "_manifest.json").write_text(
            json.dumps({"unavailable_modules": []}), encoding="utf-8"
        )
        return {
            "num_patients": 200_859,
            "batch_size": 67_000 if requested_batch_size is None else 50_000,
            "native_export_v2": {"output_validation_reads": 19},
        }

    monkeypatch.setattr(launcher, "extract_database", fake_extract)
    monkeypatch.setattr(launcher, "open_export_package", lambda _path: _Package())
    argv = ["--output-root", str(tmp_path / "out"), "--databases", "eicu"]
    if requested_batch_size is not None:
        argv.extend(["--batch-size", str(requested_batch_size)])

    manifest = launcher.run(launcher._parse_args(argv))

    assert calls[0]["stream_output_batches"] is True
    assert calls[0]["batch_size"] == expected_batch_size
    assert manifest["sources"]["eicu"]["effective_batch_size"] == (
        67_000 if requested_batch_size is None else 50_000
    )
    assert manifest["runtime_limits"]["parallel_max_workers"] == 8
    assert manifest["runtime_detection"]["selection_tier"] == "server_ge64gib"
