"""Explicit full-CI routing: historical sources and current V4 contracts.

Only the four declared historical owners are routed away from the current
execution. Every original parametrized contract has a mandatory current V4
node; all other tests are unaffected. Plain pytest does not activate routing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.figure2_successor import ROOT, policy


def pytest_addoption(parser):
    group = parser.getgroup("figure2-versioned-ci")
    group.addoption("--figure2-current-v4", action="store_true", help="Run the complete explicit V4 contract mapping; legacy owners run in the history job")
    group.addoption("--figure2-collection-receipt", help="Write the explicit current/historical node mapping")


def pytest_ignore_collect(collection_path, config):
    if not config.getoption("--figure2-current-v4"):
        return None
    try:
        relative = Path(collection_path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return None
    if relative in policy()["test_sources"]:
        return True
    return None


def pytest_collection_finish(session):
    config = session.config
    if not config.getoption("--figure2-current-v4"):
        return
    declaration = policy()
    nodeids = {item.nodeid for item in session.items}
    expected = set(declaration["expected_successor_nodes"])
    missing = sorted(expected - nodeids)
    wrappers = {row["successor_test"] for row in declaration["test_sources"].values()}
    observed = {node for node in nodeids if node.split("::", 1)[0] in wrappers}
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise pytest.UsageError(f"Figure2 V4 contract mapping incomplete: missing={missing}, unexpected={unexpected}")
    legacy = [node for node in nodeids if node.split("::", 1)[0] in declaration["test_sources"]]
    if legacy:
        raise pytest.UsageError("Figure2 current CI accidentally collected raw historical owners")
    for item in session.items:
        if item.nodeid not in expected:
            continue
        original = next(entry for entry in declaration["test_sources"].values() if entry["successor_test"] == item.nodeid.split("::", 1)[0])
        expected_module = original["successor_module"]
        if item.module.__name__ != expected_module or item.obj.__module__ != expected_module:
            raise pytest.UsageError("Figure2 V4 contract did not execute in its declared namespace")
        if Path(item.module.__file__).resolve() != (ROOT / original["successor_test"]).resolve():
            raise pytest.UsageError("Figure2 V4 module file differs from its declared successor")
        if getattr(item.module, "__figure2_contract_projection__", None) != "current-v4-authority-only":
            raise pytest.UsageError("Figure2 V4 contract lacks explicit source provenance")
    receipt = config.getoption("--figure2-collection-receipt")
    worker = getattr(config, "workerinput", {}).get("workerid")
    if receipt and worker in (None, "gw0"):
        target = Path(receipt)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "schema_version": "easyicu.figure2_current_collection/1",
            "authority_status": declaration["authority_status"],
            "legacy_commit": declaration["legacy_commit"],
            "current_contract": "current-v4",
            "legacy_raw_nodes_executed_as_current": False,
            "original_nodes": declaration["expected_original_nodes"],
            "successor_nodes": declaration["expected_successor_nodes"],
            "node_mapping": list(zip(declaration["expected_original_nodes"], declaration["expected_successor_nodes"])),
            "all_other_collected_nodes": sorted(nodeids - expected),
            "collected_count": len(nodeids),
        }, indent=2) + "\n", encoding="utf-8")
