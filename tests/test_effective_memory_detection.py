"""Portable host/cgroup memory detection contracts."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from easyicu.runtime import memory_manager
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
