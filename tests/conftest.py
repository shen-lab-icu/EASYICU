from __future__ import annotations

import os
from functools import lru_cache
from importlib.util import find_spec
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

_OPTIONAL_EXTRA_PACKAGES = {
    "webapp": ("fastapi", "starlette", "psutil"),
    "mcp": ("anyio", "httpx", "mcp"),
    "methods": ("lifelines",),
}
_OPTIONAL_EXTRA_TEST_FILES = {
    "webapp": frozenset(
        {
            "tests/core/test_data_package_review.py",
            "tests/core/test_database_profiles.py",
            "tests/core/test_hosted_llm_server_security.py",
            "tests/core/test_idea_prior_art_receipt.py",
            "tests/core/test_memory_evidence_runner.py",
            "tests/governance/test_demo_release_pack.py",
            "tests/research_agent/providers/test_literature_concepts.py",
        }
    ),
    "mcp": frozenset(
        {
            "tests/core/test_extension_registry.py",
            "tests/research_agent/providers/test_mcp_transport.py",
        }
    ),
    "methods": frozenset(
        {"tests/research_agent/test_competing_risks_kernel.py"}
    ),
}

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


@lru_cache(maxsize=None)
def _missing_optional_packages(extra: str) -> tuple[str, ...]:
    return tuple(
        package
        for package in _OPTIONAL_EXTRA_PACKAGES[extra]
        if find_spec(package) is None
    )


def _optional_extra_for_test(module_path: Path) -> str | None:
    try:
        relative = module_path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None
    if relative == "tests/webserver" or relative.startswith("tests/webserver/"):
        return "webapp"
    for extra, paths in _OPTIONAL_EXTRA_TEST_FILES.items():
        if relative in paths:
            return extra
    return None


def pytest_ignore_collect(collection_path: Path, config) -> bool | None:
    """Do not import test owners whose declared optional runtime is absent."""

    extra = _optional_extra_for_test(collection_path)
    if extra is None:
        return None
    missing = _missing_optional_packages(extra)
    if missing:
        omitted = getattr(config, "_easyicu_omitted_optional_extras", {})
        omitted[extra] = missing
        config._easyicu_omitted_optional_extras = omitted
        return True
    return None


def pytest_report_header(config) -> str | None:
    omitted = getattr(config, "_easyicu_omitted_optional_extras", {})
    if not omitted:
        return None
    summary = "; ".join(
        f"easyicu[{extra}] missing {', '.join(packages)}"
        for extra, packages in sorted(omitted.items())
    )
    return f"optional test owners not collected: {summary}"

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
    # E-P2-4: resource-gated markers skip with an explicit reason so the
    # skip is counted (visible in -rs) instead of silently passing.  The
    # governance gate test_corpus_and_node_skips_are_counted pins that these
    # markers exist and that this hook handles them; coverage tooling must
    # treat these skips as uncovered, not as passes.
    corpus_ready = corpus_root().exists()
    node_ready = shutil.which("node") is not None
    docker_ready = shutil.which("docker") is not None
    skip_no_corpus = pytest.mark.skip(
        reason=f"Recorded run corpus is not mounted at {corpus_root()}"
    )
    skip_no_node = pytest.mark.skip(reason="Node.js is unavailable")
    skip_no_docker = pytest.mark.skip(reason="Docker is unavailable")

    for item in items:
        if "needs_real_data" in item.keywords and not (run_real and real_data_ready):
            item.add_marker(skip_real_data)
        if "requires_corpus" in item.keywords and not corpus_ready:
            item.add_marker(skip_no_corpus)
        if "requires_node" in item.keywords and not node_ready:
            item.add_marker(skip_no_node)
        if "requires_docker" in item.keywords and not docker_ready:
            item.add_marker(skip_no_docker)


def corpus_root() -> Path:
    """Recorded-run corpus root (E-P2-4/E-P2-5 shared helper).

    Honors ``EASYICU_CORPUS_ROOT`` so CI and developers without the
    historical ``/Volumes`` mount can point at a local copy; falls back to
    the historical default which the ``requires_corpus`` marker skips on.
    """

    return Path(
        os.environ.get(
            "EASYICU_CORPUS_ROOT", "/Volumes/外置硬盘/easyicu_data/canonical9_runs"
        )
    )


def node_binary() -> str | None:
    """Node.js binary for JS contract tests (E-P2-4 shared helper)."""

    direct = shutil.which("node")
    if direct:
        return direct
    candidates = sorted((Path.home() / ".nvm" / "versions" / "node").glob("*/bin/node"))
    return str(candidates[-1]) if candidates else None
