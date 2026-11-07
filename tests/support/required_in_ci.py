"""Test files that must run in CI rather than skip for a missing optional import.

A privacy test that skips itself because CI lacks an optional package proves
nothing, and pytest still reports the shard green. Every test file whose name
contains ``privacy`` is required, plus the files listed here.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Extras the main CI test job installs: `pip install -e ".[dev,webapp]"`.
CI_INSTALLED_EXTRAS = ("dev", "webapp")

# Files outside the *privacy* name pattern that must not skip in CI.
REQUIRED_IN_CI: tuple[str, ...] = (
    "tests/core/test_patient_report_export_privacy.py",
)

# Import names whose installed distribution has a different name.
IMPORT_TO_DISTRIBUTION = {
    "fitz": "pymupdf",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
}


def required_test_files() -> list[Path]:
    found = {path.resolve() for path in (REPO_ROOT / "tests").rglob("test_*privacy*.py")}
    found.update((REPO_ROOT / rel).resolve() for rel in REQUIRED_IN_CI)
    return sorted(found)
