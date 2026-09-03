"""Portable host/cgroup memory detection contracts."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from easyicu import datasource
from easyicu.runtime import memory_manager, parallel_config
from easyicu.runtime.memory_manager import get_effective_memory_info


def _write_cgroup(root: Path, *, maximum: str, current_bytes: int) -> None:
    root.mkdir()
    (root / "memory.max").write_text(maximum, encoding="utf-8")
    (root / "memory.current").write_text(str(current_bytes), encoding="utf-8")


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="cgroup v2 is a Linux contract",
)
def test_cgroup_v2_limit_and_current_bound_effective_memory(tmp_path: Path) -> None:
    cgroup = tmp_path / "cgroup"
    _write_cgroup(
        cgroup,
        maximum=str(13 * 1024**3),
        current_bytes=5 * 1024**3,
    )

    info = get_effective_memory_info(
        cgroup_root=cgroup,
        host_total_mb=1536 * 1024,
        host_available_mb=1200 * 1024,
    )

    assert info.source == "cgroup_v2"
    assert info.host_total_mb == 1536 * 1024
    assert info.effective_total_mb == 13 * 1024
    assert info.effective_available_mb == 8 * 1024
    assert info.cgroup_limit_mb == 13 * 1024
    assert info.cgroup_current_mb == 5 * 1024


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="cgroup v2 is a Linux contract",
)
def test_parallel_config_uses_the_same_cgroup_memory_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cgroup = tmp_path / "cgroup"
    _write_cgroup(
        cgroup,
        maximum=str(14 * 1024**3),
        current_bytes=2 * 1024**3,
    )
    monkeypatch.setattr(
        memory_manager,
        "_host_memory_mb",
        lambda: (1536 * 1024.0, 1200 * 1024.0),
    )

    total_gb, available_gb = parallel_config.get_system_memory(
        cgroup_root=cgroup
    )

    assert total_gb == 14.0
    assert available_gb == 12.0

    monkeypatch.setattr(
        parallel_config,
        "get_system_memory",
        lambda: (total_gb, available_gb),
    )
    config = parallel_config.get_parallel_config()
    assert config.performance_tier == "limited"
    assert config.max_workers == 2
    assert config.buckets_per_batch == 1


def test_duckdb_defaults_follow_effective_parallel_and_memory_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EASYICU_DUCKDB_THREADS", raising=False)
    monkeypatch.setattr(
        parallel_config,
        "get_global_config",
        lambda: parallel_config.ParallelConfig(
            total_memory_gb=14.0,
            available_memory_gb=12.0,
            cpu_count=384,
            max_workers=2,
            buckets_per_batch=1,
            memory_per_concept_mb=200,
            use_duckdb_aggregation=True,
            enable_concept_cache=True,
        ),
    )
    monkeypatch.setattr(
        memory_manager,
        "get_effective_memory_info",
        lambda: memory_manager.EffectiveMemoryInfo(
            host_total_mb=1536 * 1024.0,
            host_available_mb=1200 * 1024.0,
            effective_total_mb=14 * 1024.0,
            effective_available_mb=12 * 1024.0,
            source="cgroup_v2",
            cgroup_limit_mb=14 * 1024.0,
            cgroup_current_mb=2 * 1024.0,
        ),
    )

    assert datasource._default_duckdb_threads() == 2
    assert datasource._default_duckdb_memory_limit_gb() == pytest.approx(1.8)

    monkeypatch.setenv("EASYICU_DUCKDB_THREADS", "5")
    assert datasource._default_duckdb_threads() == 5


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="cgroup v2 is a Linux contract",
)
def test_unlimited_cgroup_falls_back_to_host_memory(tmp_path: Path) -> None:
    cgroup = tmp_path / "cgroup"
    _write_cgroup(
        cgroup,
        maximum="max",
        current_bytes=99 * 1024**3,
    )

    info = get_effective_memory_info(
        cgroup_root=cgroup,
        host_total_mb=64 * 1024,
        host_available_mb=40 * 1024,
    )

    assert info.source == "host"
    assert info.effective_total_mb == 64 * 1024
    assert info.effective_available_mb == 40 * 1024
    assert info.cgroup_limit_mb is None
    assert info.cgroup_current_mb is None


@pytest.mark.parametrize("platform_name", ["darwin", "win32"])
def test_non_linux_platforms_use_host_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
) -> None:
    cgroup = tmp_path / "cgroup"
    _write_cgroup(
        cgroup,
        maximum=str(4 * 1024**3),
        current_bytes=3 * 1024**3,
    )
    monkeypatch.setattr(memory_manager.sys, "platform", platform_name)

    info = get_effective_memory_info(
        cgroup_root=cgroup,
        host_total_mb=16 * 1024,
        host_available_mb=9 * 1024,
    )

    assert info.source == "host"
    assert info.effective_total_mb == 16 * 1024
    assert info.effective_available_mb == 9 * 1024
