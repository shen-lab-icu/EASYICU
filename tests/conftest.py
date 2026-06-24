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

_LEGACY_STREAMLIT_TESTS = {
    "tests/test_app_rendering.py",
    "tests/test_cohort_workspace_bundle.py",
    "tests/test_llm_chat.py",
    "tests/test_mock_data_catalog_coverage.py",
    "tests/test_real_ui_smoke.py",
    "tests/test_research_agent_web_helpers.py",
    "tests/test_shared_webapp_helper_migration.py",
    "tests/test_webapp_launch.py",
    "tests/test_webapp_resume_panel.py",
}


def _relative_test_path(path: os.PathLike[str] | str) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _is_legacy_streamlit_path(path: os.PathLike[str] | str) -> bool:
    rel_path = _relative_test_path(path)
    return (
        rel_path in _LEGACY_STREAMLIT_TESTS
        or rel_path == "tests/webapp"
        or rel_path.startswith("tests/webapp/")
    )


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
    parser.addoption(
        "--run-legacy-streamlit",
        action="store_true",
        default=False,
        help="Run deprecated Streamlit WebApp UI tests.",
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

    run_legacy_streamlit = config.getoption("--run-legacy-streamlit", default=False)
    skip_legacy_streamlit = pytest.mark.skip(
        reason="Need --run-legacy-streamlit for deprecated Streamlit UI tests"
    )

    for item in items:
        item_path = getattr(item, "path", None) or getattr(item, "fspath", "")
        if _is_legacy_streamlit_path(item_path):
            item.add_marker(pytest.mark.legacy_streamlit)
            if not run_legacy_streamlit:
                item.add_marker(skip_legacy_streamlit)
        if "needs_real_data" in item.keywords and not (run_real and real_data_ready):
            item.add_marker(skip_real_data)
        if "needs_real_llm" in item.keywords and not (run_real_llm and real_llm_ready):
            item.add_marker(skip_real_llm)


def pytest_ignore_collect(collection_path, config):
    if config.getoption("--run-legacy-streamlit", default=False):
        return None
    if _is_legacy_streamlit_path(collection_path):
        return True
    return None
