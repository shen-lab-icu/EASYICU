"""Byte-level retirement lock for the historical Figure 2 paper rubric v2."""

from __future__ import annotations

import hashlib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_V2_MODULE_SHA256 = "bd9b19ae328a48d156fb42d6b660795928e09f529acfea070f6813e11f00b9da"
_V2_MANIFEST_SHA256 = "92823dded6fffe49ae85feb6fe9fd7e26883882d64b745907fdf03905c28acb3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_retired_paper_rubric_v2_bytes_remain_frozen() -> None:
    assert (
        _sha256(
            _REPO_ROOT / "benchmarks/figure2_canonical9/evaluator/paper_rubric_v2.py"
        )
        == _V2_MODULE_SHA256
    )
    assert (
        _sha256(
            _REPO_ROOT / "benchmarks/figure2_canonical9/figure2_paper_rubric_v2.json"
        )
        == _V2_MANIFEST_SHA256
    )
