from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_release_archives_preserve_reviewer_contract_and_package_data(tmp_path: Path) -> None:
    pytest.importorskip("build")

    out_dir = tmp_path / "dist"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--sdist",
                "--wheel",
                "--outdir",
                str(out_dir),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

        sdist_path = next(out_dir.glob("easyicu-*.tar.gz"))
        wheel_path = next(out_dir.glob("easyicu-*.whl"))

        with tarfile.open(sdist_path, "r:gz") as archive:
            sdist_names = {_strip_sdist_root(member.name) for member in archive.getmembers()}

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
            "docs/images/01_mode_selection.jpg",
            "docs/images/06_cross_db_benchmark.jpg",
            "examples/research_agent_mortality_sofa.py",
            "examples/research_agent_real_llm_smoke.py",
            "pytest.ini",
            "src/easyicu/data/concept-dict.json",
            "src/easyicu/data/data-sources.json",
            "src/easyicu/research_agent/README.md",
            "tests/test_release_archive_contract.py",
            "tests/test_release_hardening_p0.py",
            "tests/test_repository_contract.py",
        }
        missing_sdist = sorted(required_sdist_files - sdist_names)
        assert not missing_sdist, f"sdist is missing reviewer/release files: {missing_sdist}"

        assert "easyicu/__init__.py" in wheel_names
        assert "easyicu/data/concept-dict.json" in wheel_names
        assert "easyicu/data/data-sources.json" in wheel_names
        assert any(name.endswith(".dist-info/entry_points.txt") for name in wheel_names)

        assert not any(name.startswith(("tests/", "examples/", "docs/")) for name in wheel_names)
        assert not [name for name in sdist_names if _contains_debris(name)]
        assert not [name for name in wheel_names if _contains_debris(name)]
    finally:
        shutil.rmtree(REPO_ROOT / "build", ignore_errors=True)
        shutil.rmtree(REPO_ROOT / "easyicu.egg-info", ignore_errors=True)
        shutil.rmtree(REPO_ROOT / "src" / "easyicu.egg-info", ignore_errors=True)
