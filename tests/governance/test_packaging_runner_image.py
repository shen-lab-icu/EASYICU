"""Prove the Docker runner build context survives a real wheel build.

Asserting that pyproject.toml *mentions* runner_image only tests the config
text. setuptools ships non-Python files only when package-data actually matches
them, so the claim "a wheel install can build the sandbox image" is only
verified by building a wheel and looking inside it.

Opt-in because a build takes ~1 minute:

    pytest tests/governance/test_packaging_runner_image.py --run-packaging
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import sysconfig
import zipfile

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_RUNNER_IMAGE_ASSETS = (
    "easyicu/research_agent/runner_image/Dockerfile",
    "easyicu/research_agent/runner_image/requirements.lock",
    "easyicu/research_agent/runner_image/README.md",
)


def _build_wheel(destination: Path) -> Path:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "-w",
            str(destination),
            str(REPO_ROOT),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"wheel build failed:\n{result.stdout}\n{result.stderr}")
    wheels = sorted(destination.glob("easyicu-*.whl"))
    assert wheels, f"no wheel produced in {destination}"
    return wheels[-1]


@pytest.mark.packaging
def test_built_wheel_ships_the_runner_image_context(tmp_path, request):
    if not request.config.getoption("--run-packaging"):
        pytest.skip("needs --run-packaging (builds a wheel, ~1 min)")

    wheel = _build_wheel(tmp_path / "dist")
    names = set(zipfile.ZipFile(wheel).namelist())

    missing = [asset for asset in REQUIRED_RUNNER_IMAGE_ASSETS if asset not in names]
    assert not missing, (
        "wheel does not ship the Docker build context DockerRunner documents: "
        f"{missing}"
    )


@pytest.mark.packaging
def test_installed_wheel_exposes_runner_image_via_importlib(tmp_path, request):
    """Install into a clean venv and read the files the way callers would."""

    if not request.config.getoption("--run-packaging"):
        pytest.skip("needs --run-packaging (builds and installs a wheel, ~2 min)")

    wheel = _build_wheel(tmp_path / "dist")
    venv_dir = tmp_path / "clean-env"
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)], check=True, capture_output=True
    )
    scheme = "nt" if sys.platform == "win32" else "posix_prefix"
    bin_dir = Path(
        sysconfig.get_path("scripts", scheme=scheme, vars={"base": str(venv_dir)})
    )
    python = bin_dir / ("python.exe" if sys.platform == "win32" else "python")

    install = subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", str(wheel)],
        capture_output=True,
        text=True,
    )
    if install.returncode != 0:
        pytest.fail(f"wheel install failed:\n{install.stdout}\n{install.stderr}")

    probe = (
        "from importlib.resources import files;"
        "root = files('easyicu.research_agent') / 'runner_image';"
        "names = sorted(p.name for p in root.iterdir());"
        "assert 'Dockerfile' in names, names;"
        "assert 'requirements.lock' in names, names;"
        "print(','.join(names))"
    )
    check = subprocess.run([str(python), "-c", probe], capture_output=True, text=True)
    if check.returncode != 0:
        pytest.fail(
            "installed wheel cannot read runner_image via importlib.resources:\n"
            f"{check.stdout}\n{check.stderr}"
        )

    pip_check = subprocess.run(
        [str(python), "-m", "pip", "check"], capture_output=True, text=True
    )
    assert (
        pip_check.returncode == 0
    ), f"pip check failed on the installed wheel:\n{pip_check.stdout}"
