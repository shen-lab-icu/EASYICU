"""Tests for the C1 / C3 / E2 fixes derived from pilot run 20260515.

* C1 — replanner/planner step-count cap (``max_total_steps``)
* C3 — byte-identical validator finding dedupe (``dedupe_findings``)
* E2 — per-step numeric-claim cap (``max_leaves`` param)

The pilot triage is at ``pilot_runs/run_20260515T144753_597802/TRIAGE.md``;
each test below maps 1-to-1 to a problem documented there.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# -------------------------- C3: dedupe findings --------------------------

def test_dedupe_findings_collapses_identical(ra):
    from easyicu.research_agent.audits.validators import dedupe_findings
    schema = ra.schema
    raw = [
        schema.ValidationFinding(validator="v", severity="error", message="X"),
        schema.ValidationFinding(validator="v", severity="error", message="X"),
        schema.ValidationFinding(validator="v", severity="error", message="X"),
    ]
    out = dedupe_findings(raw)
    assert len(out) == 1
    assert out[0].detail["duplicate_count"] == 3


def test_dedupe_findings_keeps_different_severities_separate(ra):
    from easyicu.research_agent.audits.validators import dedupe_findings
    schema = ra.schema
    raw = [
        schema.ValidationFinding(validator="v", severity="error", message="X"),
        schema.ValidationFinding(validator="v", severity="warning", message="X"),
    ]
    out = dedupe_findings(raw)
    # Same validator + message but different severity → 2 distinct entries.
    assert len(out) == 2


def test_dedupe_findings_merges_evidence_ids(ra):
    from easyicu.research_agent.audits.validators import dedupe_findings
    schema = ra.schema
    raw = [
        schema.ValidationFinding(
            validator="v", severity="error", message="X",
            evidence_ids=["a", "b"],
        ),
        schema.ValidationFinding(
            validator="v", severity="error", message="X",
            evidence_ids=["b", "c"],
        ),
    ]
    out = dedupe_findings(raw)
    assert len(out) == 1
    assert out[0].evidence_ids == ["a", "b", "c"]


def test_dedupe_findings_preserves_order(ra):
    from easyicu.research_agent.audits.validators import dedupe_findings
    schema = ra.schema
    raw = [
        schema.ValidationFinding(validator="a", severity="error", message="first"),
        schema.ValidationFinding(validator="b", severity="error", message="second"),
        schema.ValidationFinding(validator="a", severity="error", message="first"),
    ]
    out = dedupe_findings(raw)
    assert [f.message for f in out] == ["first", "second"]


# -------------------------- E2: claim cap --------------------------

def test_register_step_summary_numerics_respects_max_leaves(ra, tmp_path: Path):
    store = ra.EvidenceStore(tmp_path)
    summary = {f"x_{i}": float(i) * 0.1 for i in range(20)}
    claims = store.register_step_summary_numerics(
        step_id="s1", evidence_id="evid_a", summary=summary, max_leaves=5,
    )
    # First 5 keep, 15 dropped (plus 1 overflow sentinel).
    assert len(claims) == 5
    overflows = [
        c for c in store.numeric_claims()
        if "overflow" in c.source_field
    ]
    assert len(overflows) == 1
    assert overflows[0].canonical == 15.0  # dropped count


def test_register_step_summary_numerics_no_cap_when_none(ra, tmp_path: Path):
    store = ra.EvidenceStore(tmp_path)
    summary = {f"x_{i}": float(i) for i in range(10)}
    claims = store.register_step_summary_numerics(
        step_id="s1", evidence_id="evid_a", summary=summary, max_leaves=None,
    )
    assert len(claims) == 10
    overflows = [
        c for c in store.numeric_claims()
        if "overflow" in c.source_field
    ]
    assert overflows == []


def test_register_step_summary_numerics_no_cap_when_zero(ra, tmp_path: Path):
    """Cap of 0 is treated as 'disabled' (matches PipelineConfig semantics)."""
    store = ra.EvidenceStore(tmp_path)
    summary = {f"x_{i}": float(i) for i in range(10)}
    claims = store.register_step_summary_numerics(
        step_id="s1", evidence_id="evid_a", summary=summary, max_leaves=0,
    )
    assert len(claims) == 10


# -------------------------- C1: pipeline max_total_steps --------------------------

def test_pipeline_max_total_steps_default(ra, tmp_path: Path):
    """The default cap on plan size is exposed and non-zero."""
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "wd",
        llm=ra.MockLLMClient(),
    )
    assert pipeline._max_total_steps > 0
    assert pipeline._max_total_steps == 12  # documented default in pipeline_config


def test_pipeline_max_total_steps_can_be_disabled(ra, tmp_path: Path):
    """Setting max_total_steps=0 turns off the cap (back-compat)."""
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "wd",
        llm=ra.MockLLMClient(),
        max_total_steps=0,
    )
    assert pipeline._max_total_steps == 0


def test_pipeline_max_numeric_claims_per_step_default(ra, tmp_path: Path):
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "wd",
        llm=ra.MockLLMClient(),
    )
    assert pipeline._max_numeric_claims_per_step == 100
