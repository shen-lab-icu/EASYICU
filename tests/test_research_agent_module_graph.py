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


@pytest.mark.parametrize("metric", ["module_count", "top_level_module_count"])
def test_diff_rejects_module_inventory_growth(tmp_path: Path, metric: str) -> None:
    baseline = graph.build_snapshot(
        _write_package(tmp_path), "demo", legacy_targets=("demo.a",)
    )
    current = copy.deepcopy(baseline)
    current["metrics"][metric] += 1

    errors = graph.compare_snapshots(current, baseline)

    assert any(f"{metric} increased" in error for error in errors)


def test_diff_rejects_malformed_or_incomplete_metrics(tmp_path: Path) -> None:
    baseline = graph.build_snapshot(
        _write_package(tmp_path), "demo", legacy_targets=("demo.a",)
    )
    malformed = copy.deepcopy(baseline)
    malformed["metrics"] = []
    assert graph.compare_snapshots(baseline, malformed) == [
        "baseline snapshot lacks a valid metrics object"
    ]

    incomplete = copy.deepcopy(baseline)
    del incomplete["metrics"]["module_count"]
    assert "required metric missing: module_count" in graph.compare_snapshots(
        baseline, incomplete
    )


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


def test_production_snapshot_has_no_legacy_surfaces() -> None:
    snapshot = graph.build_snapshot()
    assert snapshot["legacy_surfaces"] == {}
    assert graph.LEGACY_TARGET_MODULES == ()


def test_production_snapshot_contains_supported_canonical_modules() -> None:
    snapshot = graph.build_snapshot()
    assert set(graph.SUPPORTED_CANONICAL_MODULES).issubset(snapshot["modules"])


def test_production_research_agent_import_graph_is_acyclic() -> None:
    snapshot = graph.build_snapshot()
    assert snapshot["cyclic_sccs"] == []
    assert snapshot["metrics"]["cyclic_scc_count"] == 0
    assert snapshot["metrics"]["cyclic_module_count"] == 0
    assert snapshot["metrics"]["largest_scc_size"] == 0


def test_checked_in_module_graph_baseline_has_no_regression() -> None:
    """The counterpart of ``test_arch_measure``'s baseline lock.

    Added 2026-08-22. Until then nothing in ``tests/`` or ``.github/``
    referenced ``arch_baselines/research_agent_module_graph.json``: the tool
    could emit and diff, but only the acyclicity assertion above ever ran, so
    the module-count and canonical-surface ratchet drifted 8 days and 54
    modules (519 -> 573) unnoticed. Keeping the baseline honest is the whole
    reason it is checked in.
    """

    baseline = json.loads(
        (TOOLS_DIR / "arch_baselines" / "research_agent_module_graph.json").read_text(
            encoding="utf-8"
        )
    )

    assert graph.compare_snapshots(graph.build_snapshot(), baseline) == []
