"""Smoke test for the baselines registry (O14).

Keeps the registry and the fetcher in sync. Verifies:

* the YAML block in REGISTRY.md is parseable;
* every entry has ``name``, ``repo``, ``ref``, ``category``, and
  ``axis``;
* names are unique;
* every repo URL is an https GitHub URL (we do not fetch anything in
  this test — that is an opt-in CLI action).
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
LOCK_PATH = REPO_ROOT / "baselines" / "LOCK.json"
PAPER_CITED_BASELINES = {
    "data-to-paper",
    "healthflow",
    "ai-scientist-v2",
    "openlens-ai",
    "m4",
}


@pytest.fixture(scope="module")
def fetch_module():
    sys.path.insert(0, str(TOOLS_DIR))
    try:
        import fetch_baselines  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return fetch_baselines


def test_registry_parses(fetch_module):
    entries = fetch_module.load_registry()
    assert len(entries) >= 15  # we ship more than 15 entries


def test_registry_entry_fields(fetch_module):
    for e in fetch_module.load_registry():
        assert e.name
        assert e.repo.startswith("https://github.com/")
        assert e.category
        assert isinstance(e.axis, list)


def test_registry_names_are_unique(fetch_module):
    names = [e.name for e in fetch_module.load_registry()]
    assert len(names) == len(set(names))


def test_registry_covers_expected_baselines(fetch_module):
    names = {e.name for e in fetch_module.load_registry()}
    # A minimal set the paper's Methods section expects.
    required = {
        "healthflow",
        "openlens-ai",
        "m4",
        "ai-scientist-v2",
        "data-to-paper",
        "science-agent-bench",
        "dowhy",
        "lifelines",
    }
    missing = required - names
    assert not missing, f"missing expected baselines: {missing}"


def test_paper_cited_baselines_are_locked(fetch_module):
    entries = {e.name: e for e in fetch_module.load_registry()}
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    locked = {row["name"]: row for row in lock.get("paper_cited", [])}
    missing_lock = PAPER_CITED_BASELINES - locked.keys()
    assert not missing_lock, f"paper-cited baselines missing from LOCK.json: {missing_lock}"
    for name in PAPER_CITED_BASELINES:
        entry = entries[name]
        row = locked[name]
        assert re.fullmatch(r"[0-9a-f]{40}", row["commit"])
        assert entry.ref == row["commit"], (
            f"{name} must use the exact locked commit in REGISTRY.md; "
            "mutable refs such as main are not allowed for paper-cited baselines"
        )
        assert row["repo"].rstrip(".git") == entry.repo.rstrip(".git")
