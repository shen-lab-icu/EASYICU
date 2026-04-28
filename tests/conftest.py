from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def pytest_addoption(parser):
    parser.addoption(
        "--run-real",
        action="store_true",
        default=False,
        help="Run tests that require a local real ICU database path.",
    )


def pytest_collection_modifyitems(config, items):
    run_real = config.getoption("--run-real", default=False)
    real_data_path = os.environ.get("EASYICU_DATA_PATH", "")
    real_data_ready = bool(real_data_path) and Path(real_data_path).exists()
    if run_real and real_data_ready:
        return

    reason = "Need --run-real and an existing EASYICU_DATA_PATH"
    skip_marker = pytest.mark.skip(reason=reason)
    for item in items:
        if "needs_real_data" in item.keywords:
            item.add_marker(skip_marker)
