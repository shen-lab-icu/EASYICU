from __future__ import annotations

import os
from functools import lru_cache
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

# Several WebApp owners resolve their default persistence path while their
# module is imported during pytest collection.  A fixture is therefore too
# late to protect the user's real ~/.easyicu state.  Give the entire pytest
# process one disposable EasyICU home before any test module can import those
# owners; individual tests may still override the variable with monkeypatch.
_PYTEST_EASYICU_HOME = tempfile.TemporaryDirectory(prefix="easyicu-pytest-state-")
os.environ["EASYICU_HOME"] = _PYTEST_EASYICU_HOME.name

for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

def pytest_addoption(parser):
    parser.addoption(
        "--run-real",
        action="store_true",
        default=False,
        help="Run tests that require a local real ICU database path.",
    )
    parser.addoption(
        "--run-packaging",
        action="store_true",
        default=False,
        help="Run tests that build and install a real wheel (slow, ~2 min).",
    )

# --- 慢测试自动打标 (2026-08-17) -------------------------------------------
# tests/slow_tests.txt 里的节点在收集时自动获得 @pytest.mark.slow, 于是
# pytest.ini 的开发默认 -m "not slow" 能跳过它们。用 pytest -m "" 跑全套。
# 用 pytest_itemcollected 而不是 pytest_collection_modifyitems: marker 必须
# 在 -m 表达式做去选之前就挂上去。
_SLOW_LIST_PATH = Path(__file__).parent / "slow_tests.txt"


@lru_cache(maxsize=1)
def _slow_node_ids() -> frozenset[str]:
    if not _SLOW_LIST_PATH.exists():
        return frozenset()
    return frozenset(
        line.strip()
        for line in _SLOW_LIST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def pytest_itemcollected(item):
    if item.nodeid in _slow_node_ids():
        item.add_marker(pytest.mark.slow)


def pytest_collection_modifyitems(config, items):
    run_real = config.getoption("--run-real", default=False)
    real_data_path = os.environ.get("EASYICU_DATA_PATH", "")
    real_data_ready = bool(real_data_path) and Path(real_data_path).exists()
    skip_real_data = pytest.mark.skip(
        reason="Need --run-real and an existing EASYICU_DATA_PATH"
    )

    for item in items:
        if "needs_real_data" in item.keywords and not (run_real and real_data_ready):
            item.add_marker(skip_real_data)
