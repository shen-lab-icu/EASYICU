from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

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
        "--run-real-llm",
        action="store_true",
        default=False,
        help="Run tests that hit a real LLM provider (OpenAI / OpenRouter / Anthropic).",
    )


_REAL_LLM_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
)


def pytest_collection_modifyitems(config, items):
    run_real = config.getoption("--run-real", default=False)
    real_data_path = os.environ.get("EASYICU_DATA_PATH", "")
    real_data_ready = bool(real_data_path) and Path(real_data_path).exists()
    skip_real_data = pytest.mark.skip(
        reason="Need --run-real and an existing EASYICU_DATA_PATH"
    )

    run_real_llm = config.getoption("--run-real-llm", default=False)
    real_llm_ready = any(os.environ.get(k) for k in _REAL_LLM_KEY_ENV_VARS)
    skip_real_llm = pytest.mark.skip(
        reason="Need --run-real-llm and at least one of "
        + ", ".join(_REAL_LLM_KEY_ENV_VARS)
    )

    for item in items:
        if "needs_real_data" in item.keywords and not (run_real and real_data_ready):
            item.add_marker(skip_real_data)
        if "needs_real_llm" in item.keywords and not (run_real_llm and real_llm_ready):
            item.add_marker(skip_real_llm)
