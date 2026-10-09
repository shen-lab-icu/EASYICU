"""A step's result envelope is rebound to its terminal status, and nothing else.

The shadow envelope is rebound without re-reading its source: only the status
and the content digest change, so the source and raw-summary digests,
artifacts and step identity are preserved.  A tampered or forged envelope is
refused rather than laundered into a fresh, internally consistent digest
(M7 lifecycle).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.contracts.result_envelope import (
    StepResultEnvelope,
    normalize_step_result_shadow,
    rebind_step_result_status,
    verify_step_result_envelope,
)


def _ready_envelope(tmp_path: Path, *, status: str) -> StepResultEnvelope:
    envelope = normalize_step_result_shadow(
        step_id="05_primary_result",
        step_summary={"status": "completed", "primary_or": "1.5"},
        output_dir=tmp_path,
        status=status,
    )
    assert verify_step_result_envelope(envelope) is True
    return envelope


def test_rebind_step_result_status_rebinds_status_and_recomputes_digest(
    tmp_path: Path,
) -> None:
    original = _ready_envelope(tmp_path, status="executed_pending_review")

    rebound = rebind_step_result_status(original, status="ok")

    # Only status + digest change; the rebound envelope self-verifies.
    assert rebound.status == "ok"
    assert verify_step_result_envelope(rebound) is True
    assert rebound.content_sha256 != original.content_sha256
    # The source view is untouched: no cohort/CSV/JSON re-read means the
    # source and raw-summary digests, artifacts, and step identity are
    # preserved byte-for-byte.
    assert rebound.step_id == original.step_id
    assert rebound.source_summary_sha256 == original.source_summary_sha256
    assert rebound.raw_summary_artifact_sha256 == original.raw_summary_artifact_sha256
    assert rebound.observed_scalars == original.observed_scalars
    assert rebound.schema_version == original.schema_version
    assert rebound.shadow is True
    assert rebound.paper_authorized is False
    # The input envelope is frozen and unchanged.
    assert original.status == "executed_pending_review"


@pytest.mark.parametrize("status", ["", "   ", "\t\n"])
def test_rebind_step_result_status_rejects_empty_status(
    tmp_path: Path, status: str
) -> None:
    original = _ready_envelope(tmp_path, status="executed_pending_review")

    with pytest.raises(ValueError, match="non-empty status"):
        rebind_step_result_status(original, status=status)


def test_rebind_step_result_status_rejects_invalid_original_digest(
    tmp_path: Path,
) -> None:
    original = _ready_envelope(tmp_path, status="executed_pending_review")
    # A well-formed but wrong digest: the pattern still matches, so only the
    # self-verification catches it -- a tampered envelope must not be laundered
    # into a fresh, internally consistent digest.
    tampered = original.model_copy(update={"content_sha256": "0" * 64})
    assert verify_step_result_envelope(tampered) is False

    with pytest.raises(ValueError, match="invalid content digest"):
        rebind_step_result_status(tampered, status="ok")


def test_rebind_step_result_status_rejects_hand_tampered_status(
    tmp_path: Path,
) -> None:
    original = _ready_envelope(tmp_path, status="executed_pending_review")
    # Flip the status WITHOUT recomputing the digest, as a forger would.
    forged = original.model_copy(update={"status": "ok"})
    assert verify_step_result_envelope(forged) is False

    with pytest.raises(ValueError, match="invalid content digest"):
        rebind_step_result_status(forged, status="ok")
