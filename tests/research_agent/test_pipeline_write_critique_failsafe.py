from __future__ import annotations

import json
from pathlib import Path

from easyicu.research_agent.pipeline_write import (
    _persist_manuscript_critique,
    _review_manuscript_with_fail_safe,
)


class _FailingCritic:
    def review_manuscript(self, *, scaffold, available_evidence_ids):
        raise RuntimeError("critic endpoint unavailable")


class _EvidenceStub:
    def __init__(self) -> None:
        self.registered: list[dict] = []

    def get(self, evidence_id: str):
        return None

    def register_file(self, **kwargs):
        self.registered.append(kwargs)


def test_critic_exception_persists_explicit_blocked_failsafe(tmp_path: Path):
    critique, exception_type = _review_manuscript_with_fail_safe(
        _FailingCritic(),
        scaffold="Evidence-bound manuscript text.",
        available_evidence_ids=["result_table"],
    )
    evidence = _EvidenceStub()

    critique_path = _persist_manuscript_critique(
        critique=critique,
        run_dir=tmp_path,
        evidence=evidence,
        producer="pipeline",
    )

    payload = json.loads(critique_path.read_text(encoding="utf-8"))
    assert exception_type == "RuntimeError"
    assert payload["status"] == "blocked"
    assert payload["reviewer"] == "PipelineCritiqueFailSafe"
    assert "no passing review decision" in payload["concerns"][0]
    assert evidence.registered[0]["evidence_id"] == "manuscript_critique"
    assert evidence.registered[0]["producer"] == "pipeline"
