"""Value-level provenance (A-track, inspired by data-to-paper NEJM AI 2024).

These tests pin the contract that every numeric value in a manuscript
can be reverse-linked to the exact step / field / evidence id that
produced it:

* :meth:`EvidenceStore.register_step_summary_numerics` walks a nested
  ``step_summary`` dict, registers a :class:`NumericClaim` per numeric
  leaf, and is idempotent on re-run.
* :meth:`EvidenceStore.find_claim_for_value` matches a numeric literal
  by exact string, exact float, or relative tolerance.
* :func:`bind_numeric_values` rewrites a manuscript to attach
  per-value footnotes, and respects STRICT/SOFT enforcement modes for
  numbers that cannot be traced.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _store(ra, tmp_path):
    return ra.EvidenceStore(tmp_path)


def test_register_step_summary_numerics_collects_leaves(ra, tmp_path: Path):
    store = _store(ra, tmp_path)
    summary = {
        "primary_or": 1.42,
        "or_ci_lower": 1.18,
        "or_ci_upper": 1.71,
        "p_value": "0.003",  # string-encoded number must be picked up
        "n_rows": 1234,
        "subgroup": {"male": {"auc": 0.78}, "female": {"auc": 0.81}},
        "is_significant": True,  # bool: deliberately rejected
        "note": "analysis ran",  # non-numeric: skipped
    }
    claims = store.register_step_summary_numerics(
        step_id="03_assoc", evidence_id="evid_assoc", summary=summary,
    )
    fields = {c.source_field for c in claims}
    assert "primary_or" in fields
    assert "subgroup.male.auc" in fields
    assert "p_value" in fields
    assert "n_rows" in fields
    # Booleans and strings must NOT be registered as numeric claims.
    assert "is_significant" not in fields
    assert "note" not in fields


def test_register_step_summary_numerics_is_idempotent(ra, tmp_path: Path):
    store = _store(ra, tmp_path)
    summary = {"primary_or": 1.42, "p_value": 0.003}
    a = store.register_step_summary_numerics(
        step_id="s1", evidence_id="evid_a", summary=summary,
    )
    b = store.register_step_summary_numerics(
        step_id="s1", evidence_id="evid_a", summary=summary,
    )
    assert len(a) == len(b)
    assert len(store.numeric_claims()) == len(a), (
        "re-running registration on the same step must not duplicate claims"
    )


def test_find_claim_for_value_supports_tolerance(ra, tmp_path: Path):
    store = _store(ra, tmp_path)
    store.register_step_summary_numerics(
        step_id="s1", evidence_id="evid_a",
        summary={"primary_or": 1.42345},
    )
    # Exact literal hit (registry literal happens to be 1.42345)
    assert store.find_claim_for_value("1.42345").source_field == "primary_or"
    # Canonical float hit when the manuscript prints a rounded form
    hit = store.find_claim_for_value("1.4235", tolerance=1e-2)
    assert hit is not None and hit.source_field == "primary_or"
    # Out of tolerance → no match
    assert store.find_claim_for_value("2.0", tolerance=1e-3) is None


def test_bind_numeric_values_attaches_footnotes(ra, tmp_path: Path):
    from easyicu.research_agent.manuscript_post import bind_numeric_values

    store = _store(ra, tmp_path)
    store.register_step_summary_numerics(
        step_id="03_assoc", evidence_id="evid_assoc",
        summary={"primary_or": 1.42, "or_ci_lower": 1.18,
                 "or_ci_upper": 1.71, "p_value": 0.003, "n_rows": 1234},
    )
    manuscript = (
        "Higher SOFA-2 was associated with mortality "
        "(OR=1.42, 95% CI 1.18-1.71, p=0.003) in 1,234 patients.\n"
    )
    bound, binding_map, untraced = bind_numeric_values(
        manuscript, evidence=store,
    )
    assert "[^claim_1]" in bound  # OR=1.42 got a footnote
    # Both CI bounds bound (the hyphen between them must NOT block the
    # second number from matching).
    assert "1.18[^" in bound and "1.71[^" in bound
    # The "SOFA-2" suffix '2' is one digit → tightened regex skips it.
    assert "SOFA-2 was associated" in bound  # unchanged
    # All five canonical claims should be bound.
    assert len(binding_map) == 5
    assert untraced == []
    # Footnote definitions point back to step / field / evidence.
    assert "step=03_assoc" in bound
    assert "field=primary_or" in bound
    assert "evidence=evid_assoc" in bound


def test_bind_numeric_values_strict_raises_on_untraced(ra, tmp_path: Path):
    from easyicu.research_agent.manuscript_post import bind_numeric_values

    store = _store(ra, tmp_path, )
    store.register_step_summary_numerics(
        step_id="s1", evidence_id="evid_a",
        summary={"primary_or": 1.42},
    )
    manuscript = "The OR was 1.42, but a stray 999 appeared.\n"
    # SOFT mode → annotates and returns list.
    bound, _, untraced = bind_numeric_values(
        manuscript, evidence=store,
        enforcement_mode=ra.EvidenceEnforcementMode.SOFT,
    )
    assert "999" in untraced
    assert "<!-- UNTRACED:999 -->" in bound
    # STRICT mode → raises with detail.
    with pytest.raises(ra.EvidenceEnforcementError) as exc_info:
        bind_numeric_values(
            manuscript, evidence=store,
            enforcement_mode=ra.EvidenceEnforcementMode.STRICT,
        )
    assert "999" in exc_info.value.detail["untraced"]


def test_bind_numeric_values_skips_existing_evidence_placeholders(
    ra, tmp_path: Path,
):
    from easyicu.research_agent.manuscript_post import bind_numeric_values

    store = _store(ra, tmp_path)
    store.register_step_summary_numerics(
        step_id="s1", evidence_id="evid_a",
        summary={"primary_or": 1.42},
    )
    # The "1.42" inside the placeholder must NOT be re-bound (it would
    # produce a malformed nested footnote and confuse downstream
    # parsers). Outside-the-placeholder copies still bind.
    manuscript = "See {evidence:primary_or_1.42} for the OR (1.42)."
    bound, binding_map, _ = bind_numeric_values(manuscript, evidence=store)
    assert "{evidence:primary_or_1.42}" in bound
    # Exactly one footnote should be created — for the parenthesised
    # 1.42 outside the placeholder.
    assert len(binding_map) == 1


def test_bind_numeric_values_skips_sha256_in_link_targets(ra, tmp_path: Path):
    """D1 (pilot 20260515 fix). After sentence-level binding, every
    ``{evidence:foo}`` becomes ``[label](evidence/foo.json
    "sha256=273e4341")``. The hex sha256 prefix would otherwise match
    the numeric regex's exponent branch and emit ``UNTRACED:273e4341``.
    The Markdown link target must be in the skip-span set.
    """
    from easyicu.research_agent.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    store.register_step_summary_numerics(
        step_id="s1", evidence_id="evid_assoc",
        summary={"primary_or": 1.42},
    )
    manuscript = (
        "Higher SOFA-2 (OR=1.42) per "
        '[outcome_rate](evidence/statistic_outcome_273e4341__outcome.json "sha256=273e4341")'
        " and a stray genuine 999 should be untraced.\n"
    )
    _, _, untraced = bind_numeric_values(manuscript, evidence=store)
    # The hex inside the link target must not surface as untraced.
    assert "273e4341" not in untraced
    # Real numeric outside the link target still surfaces untraced.
    assert "999" in untraced


def test_numeric_claims_persist_across_store_reload(ra, tmp_path: Path):
    store_a = _store(ra, tmp_path)
    store_a.register_step_summary_numerics(
        step_id="s1", evidence_id="evid_a",
        summary={"primary_or": 1.42},
    )
    store_b = _store(ra, tmp_path)
    claims = store_b.numeric_claims()
    assert len(claims) == 1
    assert claims[0].source_field == "primary_or"
