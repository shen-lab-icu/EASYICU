from __future__ import annotations

import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

from benchmarks.figure2_canonical9.evaluator.rubric_v1 import FIGURE2_TASK_IDS

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_installed_easyicu_source_never_imports_repository_benchmarks() -> None:
    violations: list[str] = []
    source_root = REPO_ROOT / "src" / "easyicu"
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported_roots: tuple[str, ...]
            if isinstance(node, ast.Import):
                imported_roots = tuple(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots = (node.module.split(".", 1)[0],)
            elif (
                isinstance(node, ast.Call)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "__import__"
                    or isinstance(node.func, ast.Attribute)
                    and node.func.attr == "import_module"
                )
            ):
                imported_roots = (node.args[0].value.split(".", 1)[0],)
            else:
                continue
            if "benchmarks" in imported_roots:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not violations, (
        "installed easyicu source must not import repository-only benchmark code: "
        f"{violations}"
    )


def test_installed_easyicu_source_contains_no_canonical9_case_material() -> None:
    source_root = REPO_ROOT / "src" / "easyicu"
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(source_root.rglob("*.py"))
    )
    assert "easyicu_evaluation_protocol_suite" not in combined
    assert all(task_id not in combined for task_id in FIGURE2_TASK_IDS)


def _strip_sdist_root(name: str) -> str:
    parts = Path(name).parts
    if len(parts) <= 1:
        return name
    return str(Path(*parts[1:]))


def _contains_debris(path: str) -> bool:
    parts = Path(path).parts
    return (
        "__pycache__" in parts
        or "__MACOSX" in parts
        or path.endswith(".pyc")
        or path.endswith(".pyo")
        or path.endswith(".DS_Store")
        or any(part.startswith("._") for part in parts)
    )


def test_release_archives_preserve_reviewer_contract_and_package_data(
    tmp_path: Path,
) -> None:
    pytest.importorskip("build.__main__")

    sdist_out_dir = tmp_path / "sdist-dist"
    wheel_out_dir = tmp_path / "wheel-dist"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    try:
        sdist_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--sdist",
                "--outdir",
                str(sdist_out_dir),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert sdist_result.returncode == 0, sdist_result.stdout + sdist_result.stderr

        sdist_path = next(sdist_out_dir.glob("easyicu-*.tar.gz"))
        unpacked_dir = tmp_path / "unpacked-sdist"

        with tarfile.open(sdist_path, "r:gz") as archive:
            sdist_names = {
                _strip_sdist_root(member.name) for member in archive.getmembers()
            }
            archive.extractall(unpacked_dir, filter="data")

        unpacked_roots = [path for path in unpacked_dir.iterdir() if path.is_dir()]
        assert len(unpacked_roots) == 1, (
            "sdist must contain exactly one project root before the wheel build: "
            f"{unpacked_roots}"
        )
        unpacked_root = unpacked_roots[0]
        assert (unpacked_root / "pyproject.toml").is_file()

        wheel_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(wheel_out_dir),
            ],
            cwd=unpacked_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert wheel_result.returncode == 0, wheel_result.stdout + wheel_result.stderr

        wheel_path = next(wheel_out_dir.glob("easyicu-*.whl"))

        with zipfile.ZipFile(wheel_path) as archive:
            wheel_names = set(archive.namelist())

        required_sdist_files = {
            ".github/workflows/ci.yml",
            ".github/workflows/research_agent_ci.yml",
            "CITATION.cff",
            "CONTRIBUTING.md",
            "LICENSE",
            "MANIFEST.in",
            "README.md",
            "README_zh.md",
            "benchmarks/__init__.py",
            "benchmarks/figure2_canonical9/canonical_run_input_bindings_v2.json",
            "benchmarks/figure2_canonical9/evaluator/scoring.py",
            "benchmarks/figure2_canonical9/figure2_paper_rubric_v2.json",
            "benchmarks/meta_generalization/meta_benchmark.jsonl",
            "docs/images/01_mode_selection.jpg",
            "docs/images/06_cross_db_benchmark.jpg",
            "examples/research_agent_mortality_sofa.py",
            "pytest.ini",
            "src/easyicu/data/concept-dict.json",
            "src/easyicu/data/data-sources.json",
            "src/easyicu/research_agent/README.md",
            "src/easyicu/research_agent/planning/cohort_contract.py",
            "src/easyicu/research_agent/planning/robustness_contract.py",
            "src/easyicu/research_agent/providers/protocol.py",
            "src/easyicu/research_agent/replication/metrics.py",
            "src/easyicu/webserver/static/index.html",
            "src/easyicu/webserver/static/css/app.css",
            "src/easyicu/webserver/static/js/app.js",
            "tests/test_release_archive_contract.py",
            "tests/test_release_hardening_p0.py",
            "tests/test_repository_contract.py",
        }
        missing_sdist = sorted(required_sdist_files - sdist_names)
        assert (
            not missing_sdist
        ), f"sdist is missing reviewer/release files: {missing_sdist}"
        benchmark_sources = {
            str(path.relative_to(REPO_ROOT))
            for path in (REPO_ROOT / "benchmarks").rglob("*")
            if path.is_file() and path.suffix in {".py", ".json", ".jsonl", ".md"}
        }
        missing_benchmark_sources = sorted(benchmark_sources - sdist_names)
        assert not missing_benchmark_sources, (
            "sdist is missing repository benchmark authority files: "
            f"{missing_benchmark_sources}"
        )

        assert "easyicu/__init__.py" in wheel_names
        assert "easyicu/data/concept-dict.json" in wheel_names
        assert "easyicu/data/data-sources.json" in wheel_names
        assert "easyicu/webserver/static/index.html" in wheel_names
        assert "easyicu/webserver/static/css/app.css" in wheel_names
        assert "easyicu/webserver/static/js/app.js" in wheel_names
        assert any(name.endswith(".dist-info/entry_points.txt") for name in wheel_names)

        required_canonical_modules = {
            "easyicu/research_agent/planning/cohort_contract.py",
            "easyicu/research_agent/planning/robustness_contract.py",
            "easyicu/research_agent/providers/protocol.py",
            "easyicu/research_agent/replication/metrics.py",
        }
        missing_canonical_modules = sorted(required_canonical_modules - wheel_names)
        assert not missing_canonical_modules, (
            "wheel built from the sdist is missing canonical research-agent modules: "
            f"{missing_canonical_modules}"
        )

        assert "src/easyicu/research_agent/projection.py" not in sdist_names
        assert "easyicu/research_agent/projection.py" not in wheel_names
        assert not any(
            name.startswith(("tests/", "examples/", "docs/", "benchmarks/"))
            for name in wheel_names
        )
        assert not any(
            name.startswith("easyicu/research_agent/figure2_") for name in wheel_names
        )
        assert "easyicu/research_agent/canonical_input_freeze.py" not in wheel_names
        assert not [name for name in sdist_names if _contains_debris(name)]
        assert not [name for name in wheel_names if _contains_debris(name)]
    finally:
        shutil.rmtree(REPO_ROOT / "build", ignore_errors=True)
        shutil.rmtree(REPO_ROOT / "easyicu.egg-info", ignore_errors=True)
        shutil.rmtree(REPO_ROOT / "src" / "easyicu.egg-info", ignore_errors=True)
