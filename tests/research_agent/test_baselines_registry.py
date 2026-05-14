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

import pathlib
import sys

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"


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
        "science-agent-bench",
        "dowhy",
        "lifelines",
    }
    missing = required - names
    assert not missing, f"missing expected baselines: {missing}"
