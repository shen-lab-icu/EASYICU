"""Characterization tests for the research-agent module-graph gate."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))
try:
    import research_agent_module_graph as graph  # type: ignore[import-not-found]
finally:
    sys.path.pop(0)


def _write_package(root: Path) -> Path:
    package = root / "demo"
    sub = package / "sub"
    sub.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "a.py").write_text(
        "from . import b\n"
        "from .sub import leaf\n"
        "__all__ = ['PublicA', 'Shared']\n"
        "class PublicA: pass\n",
        encoding="utf-8",
    )
    (package / "b.py").write_text(
        "from demo.a import PublicA\n"
        "import importlib as il\n"
        "il.import_module('demo.dynamic')\n",
        encoding="utf-8",
    )
    (package / "dynamic.py").write_text(
        "__import__('demo.sub.leaf')\n", encoding="utf-8"
    )
    (sub / "__init__.py").write_text("from .. import a\n", encoding="utf-8")
    (sub / "leaf.py").write_text("from ..dynamic import thing\n", encoding="utf-8")
    return package


def test_snapshot_resolves_relative_absolute_dynamic_imports_and_sccs(
    tmp_path: Path,
) -> None:
    package = _write_package(tmp_path)
    snapshot = graph.build_snapshot(package, "demo", legacy_targets=("demo.a",))

    assert snapshot["metrics"] == {
        "module_count": 6,
        "top_level_module_count": 4,
        "package_count": 2,
        "edge_count": 7,
        "cyclic_scc_count": 2,
        "cyclic_module_count": 4,
        "largest_scc_size": 2,
    }
    assert snapshot["cyclic_sccs"] == [
        ["demo.a", "demo.b"],
        ["demo.dynamic", "demo.sub.leaf"],
    ]
    assert ["demo.a", "demo.b"] in snapshot["edges"]
    assert ["demo.a", "demo.sub.leaf"] in snapshot["edges"]
    assert ["demo.sub.leaf", "demo.dynamic"] in snapshot["edges"]

    identities = {
        (item["source"], item["kind"], item["target"], item["resolved"])
        for item in snapshot["dynamic_literal_imports"]
    }
    assert (
        "demo.b",
        "importlib.import_module",
        "demo.dynamic",
        "demo.dynamic",
    ) in identities
    assert (
        "demo.dynamic",
        "__import__",
        "demo.sub.leaf",
        "demo.sub.leaf",
    ) in identities
    assert snapshot["legacy_surfaces"]["demo.a"]["literal_all"] == [
        "PublicA",
        "Shared",
    ]


def test_nonliteral_all_is_recorded_as_unavailable(tmp_path: Path) -> None:
    package = _write_package(tmp_path)
    (package / "a.py").write_text(
        "BASE = ['A']\n__all__ = BASE + ['B']\n", encoding="utf-8"
    )
    snapshot = graph.build_snapshot(package, "demo", legacy_targets=("demo.a",))
    assert snapshot["legacy_surfaces"]["demo.a"]["literal_all"] is None


@pytest.mark.parametrize("metric", ["cyclic_module_count", "largest_scc_size"])
def test_diff_rejects_increased_cycle_metric(tmp_path: Path, metric: str) -> None:
    baseline = graph.build_snapshot(
        _write_package(tmp_path), "demo", legacy_targets=("demo.a",)
    )
    current = copy.deepcopy(baseline)
    current["metrics"][metric] += 1
    assert any(metric in error for error in graph.compare_snapshots(current, baseline))


def test_diff_accepts_one_large_scc_split_into_more_smaller_sccs(
    tmp_path: Path,
) -> None:
    baseline = graph.build_snapshot(
        _write_package(tmp_path), "demo", legacy_targets=("demo.a",)
    )
    current = copy.deepcopy(baseline)
    current["metrics"].update(
        {
            "cyclic_scc_count": baseline["metrics"]["cyclic_scc_count"] + 1,
            "cyclic_module_count": baseline["metrics"]["cyclic_module_count"] - 1,
            "largest_scc_size": baseline["metrics"]["largest_scc_size"] - 1,
        }
    )

    assert graph.compare_snapshots(current, baseline) == []


def test_diff_rejects_missing_legacy_module_and_all_symbol(tmp_path: Path) -> None:
    baseline = graph.build_snapshot(
        _write_package(tmp_path), "demo", legacy_targets=("demo.a",)
    )

    missing_module = copy.deepcopy(baseline)
    missing_module["legacy_surfaces"]["demo.a"]["exists"] = False
    assert any(
        "legacy target module disappeared" in error
        for error in graph.compare_snapshots(missing_module, baseline)
    )

    missing_symbol = copy.deepcopy(baseline)
    missing_symbol["legacy_surfaces"]["demo.a"]["literal_all"] = ["PublicA"]
    assert any(
        "lost __all__ symbols: Shared" in error
        for error in graph.compare_snapshots(missing_symbol, baseline)
    )


def test_diff_rejects_disappearing_dynamic_literal_import(tmp_path: Path) -> None:
    baseline = graph.build_snapshot(
        _write_package(tmp_path), "demo", legacy_targets=("demo.a",)
    )
    current = copy.deepcopy(baseline)
    current["dynamic_literal_imports"] = current["dynamic_literal_imports"][1:]
    assert any(
        "dynamic literal import disappeared" in error
        for error in graph.compare_snapshots(current, baseline)
    )


def test_cli_emit_and_diff_round_trip(tmp_path: Path) -> None:
    package = _write_package(tmp_path)
    baseline_path = tmp_path / "baseline.json"
    common = [
        "--package-dir",
        str(package),
        "--package-name",
        "demo",
        "--legacy-target",
        "demo.a",
    ]
    assert graph.main([*common, "--emit", str(baseline_path)]) == 0
    assert (
        json.loads(baseline_path.read_text(encoding="utf-8"))["package_name"] == "demo"
    )
    assert graph.main([*common, "--diff", str(baseline_path)]) == 0


def test_production_snapshot_keeps_all_named_legacy_targets() -> None:
    snapshot = graph.build_snapshot()
    assert set(snapshot["legacy_surfaces"]) == set(graph.LEGACY_TARGET_MODULES)
    assert all(surface["exists"] for surface in snapshot["legacy_surfaces"].values())
