"""Empty safety-adjudication guard (E-P2-1).

``_score_safety_receipt`` divides by ``len(hazards)`` and ``len(forbidden)``.
An empty adjudication list would previously raise ``ZeroDivisionError``; it
must fail closed with ``ValueError`` instead.  The frozen rubric/request
models already enforce ``min_length=1``, so this test bypasses validation
with ``model_construct`` to exercise the scorer's own defense-in-depth guard.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.figure2_canonical9.evaluator.scoring import _score_safety_receipt


def _stub_task_rubric() -> SimpleNamespace:
    return SimpleNamespace(
        hazard_codes=("H1",),
        forbidden_claim_codes=("F1",),
    )


def _stub_receipt(hazards, forbidden) -> SimpleNamespace:
    return SimpleNamespace(
        hazard_adjudications=tuple(hazards),
        forbidden_claim_adjudications=tuple(forbidden),
        provider_ref="provider",
        model_ref="model",
        request_sha256="0" * 64,
        response_sha256="1" * 64,
    )


def _thresholds() -> SimpleNamespace:
    return SimpleNamespace(full=0.9, partial=0.6, marginal=0.3)


def test_empty_hazards_raise_value_error() -> None:
    receipt = _stub_receipt([], [{"code": "F1", "status": "absent"}])
    with pytest.raises(ValueError, match="non-empty"):
        _score_safety_receipt(receipt, _stub_task_rubric(), _thresholds())  # type: ignore[arg-type]


def test_empty_forbidden_claims_raise_value_error() -> None:
    receipt = _stub_receipt([{"code": "H1", "status": "addressed"}], [])
    with pytest.raises(ValueError, match="non-empty"):
        _score_safety_receipt(receipt, _stub_task_rubric(), _thresholds())  # type: ignore[arg-type]


def test_both_empty_raise_value_error() -> None:
    receipt = _stub_receipt([], [])
    with pytest.raises(ValueError, match="non-empty"):
        _score_safety_receipt(receipt, _stub_task_rubric(), _thresholds())  # type: ignore[arg-type]
