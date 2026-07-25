"""Canaries for manuscript numeric hallucination blocking.

These tests make the submission-facing contract explicit: a writer may
only print numbers that the value-level NumericClaim registry can trace,
with derived values handled through the same binder rather than prose
arithmetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_strict_writer_numeric_hallucination_is_blocked(ra, tmp_path: Path):
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(root=tmp_path, enforcement_mode="strict")
    store.register_step_summary_numerics(
        step_id="assoc",
        evidence_id="evid_assoc",
        summary={"primary_or": 1.42},
    )

    manuscript = "The observed odds ratio was 1.42, but the writer invented 999."
    with pytest.raises(ra.EvidenceEnforcementError) as exc_info:
        bind_numeric_values(manuscript, evidence=store)

    assert exc_info.value.detail["untraced"] == ["999"]


def test_soft_writer_numeric_hallucination_is_annotated(ra, tmp_path: Path):
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(root=tmp_path, enforcement_mode="soft")
    store.register_step_summary_numerics(
        step_id="assoc",
        evidence_id="evid_assoc",
        summary={"primary_or": 1.42},
    )

    bound, binding_map, untraced = bind_numeric_values(
        "The observed odds ratio was 1.42, but the writer invented 999.",
        evidence=store,
    )

    assert untraced == ["999"]
    assert "<!-- UNTRACED:999 -->" in bound
    assert len(binding_map) == 1
    assert binding_map["claim_1"].source_field == "primary_or"


def test_registered_and_derived_writer_numbers_pass_strict(ra, tmp_path: Path):
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(root=tmp_path, enforcement_mode="strict")
    store.register_numeric_claim(
        value="1.42",
        canonical=1.42,
        evidence_id="evid_assoc",
        step_id="assoc",
        source_field="primary_or",
    )
    store.register_numeric_claim(
        value="0.13",
        canonical=0.13,
        evidence_id="evid_assoc",
        step_id="assoc",
        source_field="primary_or_se",
    )
    derived = store.register_derived_claim(
        name="primary_or_ci_low",
        formula="exp(log(primary_or) - 1.96 * primary_or_se)",
        explanation="Lower 95% CI for primary OR, log-normal approximation",
        sources={
            "primary_or": ("assoc", "primary_or"),
            "primary_or_se": ("assoc", "primary_or_se"),
        },
        evidence_id="evid_assoc",
        step_id="assoc",
    )

    manuscript = (
        "The model registered OR=1.42 with SE=0.13 and a derived lower "
        f"confidence bound of {derived.value}."
    )
    bound, binding_map, untraced = bind_numeric_values(manuscript, evidence=store)

    assert untraced == []
    assert {claim.source_field for claim in binding_map.values()} == {
        "primary_or",
        "primary_or_se",
        "primary_or_ci_low",
    }
    assert "formula=exp(log(primary_or) - 1.96 * primary_or_se)" in bound
    assert "derived_from=assoc.primary_or, assoc.primary_or_se" in bound
