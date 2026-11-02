"""A Windows checkout keeps the committed bytes that runtime identities hash."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("path", [
    "src/easyicu/data/sic-case-duration-overlay.v1.json",
    "src/easyicu/data/concept-dict.json",
    "src/easyicu/research_agent/execution/phase.py",
    "benchmarks/figure2_canonical9/successor_v4/scoring.py",
])
def test_byte_identified_sources_are_exempt_from_line_ending_conversion(path: str) -> None:
    git = shutil.which("git")
    if not git or not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    completed = subprocess.run(
        [git, "check-attr", "text", "--", path],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    # core.autocrlf would otherwise turn LF into CRLF and break the pinned digests.
    assert completed.stdout.strip().endswith(": text: unset"), completed.stdout
