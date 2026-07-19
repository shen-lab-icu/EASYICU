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
        step_id="03_assoc",
        evidence_id="evid_assoc",
        summary=summary,
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
        step_id="s1",
        evidence_id="evid_a",
        summary=summary,
    )
    b = store.register_step_summary_numerics(
        step_id="s1",
        evidence_id="evid_a",
        summary=summary,
    )
    assert len(a) == len(b)
    assert len(store.numeric_claims()) == len(
        a
    ), "re-running registration on the same step must not duplicate claims"


def test_later_successful_attempt_keeps_its_own_numeric_claim_authority(
    ra, tmp_path: Path
):
    store = _store(ra, tmp_path)
    summary = {"primary_or": 1.42, "p_value": 0.003}

    store.register_step_summary_numerics(
        step_id="s1",
        evidence_id="evid_attempt_1",
        summary=summary,
    )
    second = store.register_step_summary_numerics(
        step_id="s1",
        evidence_id="evid_attempt_2",
        summary=summary,
    )

    assert {claim.evidence_id for claim in second} == {"evid_attempt_2"}
    assert len(store.numeric_claims()) == 4


def test_find_claim_for_value_supports_tolerance(ra, tmp_path: Path):
    store = _store(ra, tmp_path)
    store.register_step_summary_numerics(
        step_id="s1",
        evidence_id="evid_a",
        summary={"primary_or": 1.42345},
    )
    # Exact literal hit (registry literal happens to be 1.42345)
    assert store.find_claim_for_value("1.42345").source_field == "primary_or"
    # Canonical float hit when the manuscript prints a rounded form
    hit = store.find_claim_for_value("1.4235", tolerance=1e-2)
    assert hit is not None and hit.source_field == "primary_or"
    # Out of tolerance → no match
    assert store.find_claim_for_value("2.0", tolerance=1e-3) is None


def test_find_claim_for_value_supports_percent_and_rounding(ra, tmp_path: Path):
    store = _store(ra, tmp_path)
    store.register_step_summary_numerics(
        step_id="s1",
        evidence_id="evid_a",
        summary={
            "event_rate": 0.03769230769230769,
            "primary_or": 1.2247797141430332,
        },
    )
    percent_hit = store.find_claim_for_value("3.8%")
    assert percent_hit is not None and percent_hit.source_field == "event_rate"
    rounded_hit = store.find_claim_for_value("1.22")
    assert rounded_hit is not None and rounded_hit.source_field == "primary_or"


def test_near_zero_literal_does_not_match_zero_claim(ra, tmp_path: Path):
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store = _store(ra, tmp_path)
    store.register_numeric_claim(
        value="0",
        canonical=0.0,
        evidence_id="e_zero",
        step_id="s1",
        source_field="zero_count",
    )
    store.register_numeric_claim(
        value="0.00108315",
        canonical=0.001083146182081867,
        evidence_id="e_rd",
        step_id="s2",
        source_field="risk_difference.ci_low",
    )

    assert store.find_claim_for_value("0.001").source_field == "risk_difference.ci_low"
    bound, binding_map, untraced = bind_numeric_values(
        "The lower risk-difference bound was 0.001.",
        evidence=store,
    )

    assert untraced == []
    assert "<!-- AMBIGUOUS:" not in bound
    assert binding_map["claim_1"].source_field == "risk_difference.ci_low"


def test_numeric_binder_rejects_claim_from_latest_failed_attempt(
    ra, tmp_path: Path
) -> None:
    from easyicu.research_agent.authority.evidence_store import (
        EvidenceEnforcementError,
        EvidenceEnforcementMode,
    )
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store = _store(ra, tmp_path)
    store.register_numeric_claim(
        value="0.999",
        canonical=0.999,
        evidence_id="retired_summary",
        step_id="03_model",
        source_field="auroc",
    )
    records = [
        {
            "step_id": "03_model",
            "status": "ok",
            "evidence_ids": ["retired_summary"],
            "step_summary": {"auroc": 0.999},
        },
        {
            "step_id": "03_model",
            "status": "contract_failed",
            "evidence_ids": [],
            "step_summary": {},
        },
    ]

    with pytest.raises(EvidenceEnforcementError):
        bind_numeric_values(
            "The AUROC was 0.999.",
            evidence=store,
            enforcement_mode=EvidenceEnforcementMode.STRICT,
            per_step_records=records,
        )


def test_manuscript_numeric_audit_ignores_retired_summary(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.audits.manuscript_claims import (
        audit_manuscript_numeric_claims,
    )

    records = [
        {
            "step_id": "03_model",
            "status": "ok",
            "evidence_ids": ["retired_summary"],
            "step_summary": {"auroc": 0.999},
        },
        {
            "step_id": "03_model",
            "status": "contract_failed",
            "evidence_ids": [],
            "step_summary": {},
        },
        {
            "step_id": "04_current",
            "status": "ok",
            "evidence_ids": ["current_summary"],
            "step_summary": {"auroc": 0.8},
        },
    ]

    findings = audit_manuscript_numeric_claims(
        "The AUROC was 0.999.",
        per_step_records=records,
    )

    assert any("AUROC claim" in finding.message for finding in findings)


def test_numeric_binder_keeps_registered_run_level_context_claim(
    ra, tmp_path: Path
) -> None:
    from easyicu.research_agent.authority.evidence_store import (
        EvidenceEnforcementError,
        EvidenceEnforcementMode,
    )
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    context_path = tmp_path / "research_context.json"
    context_path.write_text('{"n_stays": 94458}', encoding="utf-8")
    store = _store(ra, tmp_path)
    store.register_file(
        kind="log",
        description="Run-level research context.",
        source_path=context_path,
        evidence_id="research_context",
        producer="pipeline",
    )
    store.register_numeric_claim(
        value="94458",
        canonical=94458.0,
        evidence_id="research_context",
        step_id="research_context",
        source_field="cohort.n_stays",
    )
    store.register_numeric_claim(
        value="777",
        canonical=777.0,
        evidence_id="research_context",
        step_id="99_orphan",
        source_field="invented",
    )

    bound, binding_map, untraced = bind_numeric_values(
        "The source export contained 94,458 stays.",
        evidence=store,
        per_step_records=[],
    )

    assert untraced == []
    assert binding_map["claim_1"].evidence_id == "research_context"
    assert "evidence=research_context" in bound

    with pytest.raises(EvidenceEnforcementError):
        bind_numeric_values(
            "The invented value was 777.",
            evidence=store,
            enforcement_mode=EvidenceEnforcementMode.STRICT,
            per_step_records=[],
        )


def test_numeric_binder_rejects_tampered_current_evidence_blob(
    ra, tmp_path: Path
) -> None:
    from easyicu.research_agent.authority.evidence_store import (
        EvidenceEnforcementError,
        EvidenceEnforcementMode,
    )
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    summary_path = tmp_path / "step_summary.json"
    summary_path.write_text('{"auroc": 0.81}', encoding="utf-8")
    store = _store(ra, tmp_path)
    record = store.register_file(
        kind="statistic",
        description="Current model summary.",
        source_path=summary_path,
        produced_by_step="03_model",
        evidence_id="current_summary",
        producer="runner",
    )
    store.register_numeric_claim(
        value="0.81",
        canonical=0.81,
        evidence_id=record.evidence_id,
        step_id="03_model",
        source_field="auroc",
    )
    records = [
        {
            "step_id": "03_model",
            "status": "ok",
            "evidence_ids": [record.evidence_id],
            "step_summary": {"auroc": 0.81},
        }
    ]
    evidence_blob = tmp_path / record.relative_path
    evidence_blob.write_text('{"auroc": 0.99}', encoding="utf-8")

    with pytest.raises(EvidenceEnforcementError):
        bind_numeric_values(
            "The AUROC was 0.81.",
            evidence=store,
            enforcement_mode=EvidenceEnforcementMode.STRICT,
            per_step_records=records,
        )


def test_numeric_binder_rejects_arbitrary_run_level_self_owner(
    ra, tmp_path: Path
) -> None:
    from easyicu.research_agent.authority.evidence_store import (
        EvidenceEnforcementError,
        EvidenceEnforcementMode,
    )
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    source = tmp_path / "fabricated.json"
    source.write_text('{"value": 777}', encoding="utf-8")
    store = _store(ra, tmp_path)
    store.register_file(
        kind="log",
        description="Unrecognised run-level payload.",
        source_path=source,
        evidence_id="fabricated",
        producer="pipeline",
    )
    store.register_numeric_claim(
        value="777",
        canonical=777.0,
        evidence_id="fabricated",
        step_id="fabricated",
        source_field="value",
    )

    with pytest.raises(EvidenceEnforcementError):
        bind_numeric_values(
            "The fabricated value was 777.",
            evidence=store,
            enforcement_mode=EvidenceEnforcementMode.STRICT,
            per_step_records=[],
        )


def test_context_numeric_claims_cover_source_counts_and_missingness(ra, tmp_path: Path):
    from easyicu.research_agent.research_context.builder_numeric import register_context_numeric_claims
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    schema = ra.schema
    context = schema.ResearchContext(
        research_question="x",
        cohort=schema.CohortDescriptor(
            cohort_name="c",
            database="synthetic",
            n_patients=94458,
            n_stays=94458,
        ),
        variables=[
            schema.ConceptDescriptor(
                name="lact_first",
                role="lab",
                dtype="float64",
                missingness=schema.MissingnessProfile(
                    fraction_missing=0.46388871244362573,
                    n_missing=43818,
                    n_total=94458,
                ),
            ),
            schema.ConceptDescriptor(
                name="lact_measured",
                role="lab",
                dtype="float64",
                missingness=schema.MissingnessProfile(
                    fraction_missing=0.0,
                    n_missing=0,
                    n_total=94458,
                ),
            ),
            schema.ConceptDescriptor(
                name="temp_first",
                role="vital",
                dtype="float64",
                missingness=schema.MissingnessProfile(
                    fraction_missing=0.0293781363145525,
                    n_missing=2775,
                    n_total=94458,
                ),
            ),
            schema.ConceptDescriptor(
                name="temp_measured",
                role="vital",
                dtype="float64",
                missingness=schema.MissingnessProfile(
                    fraction_missing=0.0,
                    n_missing=0,
                    n_total=94458,
                ),
            ),
        ],
    )
    store = _store(ra, tmp_path)
    claims = register_context_numeric_claims(store, context=context)

    fields = {claim.source_field for claim in claims}
    assert "cohort.n_stays_and_patients" in fields
    assert "variable_groups.lact.missingness.max_fraction_missing" in fields
    assert "variable_groups.temp.missingness.max_fraction_missing" in fields
    assert not any(field.endswith(".n_total") for field in fields)
    assert not any("fraction_missing[" in field for field in fields)

    manuscript = (
        "The source export contained 94,458 ICU stays. "
        "Lactate missingness was 46.4%, while temperature missingness was 2.9%."
    )
    bound, binding_map, untraced = bind_numeric_values(manuscript, evidence=store)

    assert untraced == []
    assert "<!-- AMBIGUOUS:" not in bound
    assert {claim.source_field for claim in binding_map.values()} == {
        "cohort.n_stays_and_patients",
        "variable_groups.lact.missingness.max_fraction_missing",
        "variable_groups.temp.missingness.max_fraction_missing",
    }


def test_bind_numeric_values_attaches_footnotes(ra, tmp_path: Path):
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store = _store(ra, tmp_path)
    store.register_step_summary_numerics(
        step_id="03_assoc",
        evidence_id="evid_assoc",
        summary={
            "primary_or": 1.42,
            "or_ci_lower": 1.18,
            "or_ci_upper": 1.71,
            "p_value": 0.003,
            "n_rows": 1234,
        },
    )
    manuscript = (
        "Higher SOFA-2 was associated with mortality "
        "(OR=1.42, 95% CI 1.18-1.71, p=0.003) in 1,234 patients.\n"
    )
    bound, binding_map, untraced = bind_numeric_values(
        manuscript,
        evidence=store,
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


def test_bind_numeric_values_handles_percent_and_rounding(ra, tmp_path: Path):
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store = _store(ra, tmp_path)
    store.register_step_summary_numerics(
        step_id="s1",
        evidence_id="evid_a",
        summary={
            "event_rate": 0.03769230769230769,
            "primary_or": 1.2247797141430332,
        },
    )
    manuscript = "Mortality was 3.8% and the OR was 1.22, supporting stability."
    bound, binding_map, untraced = bind_numeric_values(manuscript, evidence=store)
    assert "3.8%[^" in bound
    assert "1.22" in bound
    assert len(binding_map) == 2
    assert untraced == []
    assert "display=3.8%" in bound
    assert "match=rounded_or_transformed" in bound


def test_bind_numeric_values_footnote_exposes_derived_provenance(ra, tmp_path: Path):
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store = _store(ra, tmp_path)
    store.register_numeric_claim(
        value="1.42",
        canonical=1.42,
        evidence_id="evid_assoc",
        step_id="03_assoc",
        source_field="primary_or",
    )
    store.register_numeric_claim(
        value="0.13",
        canonical=0.13,
        evidence_id="evid_assoc",
        step_id="03_assoc",
        source_field="primary_or_se",
    )
    claim = store.register_derived_claim(
        name="primary_or_ci_low",
        formula="exp(log(primary_or) - 1.96 * primary_or_se)",
        explanation="Lower 95% CI for primary OR, log-normal approx",
        sources={
            "primary_or": ("03_assoc", "primary_or"),
            "primary_or_se": ("03_assoc", "primary_or_se"),
        },
        evidence_id="evid_assoc",
        step_id="03_assoc",
    )
    manuscript = f"The lower confidence bound was {claim.value}."
    bound, binding_map, untraced = bind_numeric_values(manuscript, evidence=store)
    assert len(binding_map) == 1
    assert untraced == []
    assert "formula=exp(log(primary_or) - 1.96 * primary_or_se)" in bound
    assert "explanation=Lower 95% CI for primary OR, log-normal approx" in bound
    assert "derived_from=03_assoc.primary_or, 03_assoc.primary_or_se" in bound


def test_bind_numeric_values_strict_raises_on_untraced(ra, tmp_path: Path):
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store = _store(
        ra,
        tmp_path,
    )
    store.register_step_summary_numerics(
        step_id="s1",
        evidence_id="evid_a",
        summary={"primary_or": 1.42},
    )
    manuscript = "The OR was 1.42, but a stray 999 appeared.\n"
    # SOFT mode → annotates and returns list.
    bound, _, untraced = bind_numeric_values(
        manuscript,
        evidence=store,
        enforcement_mode=ra.EvidenceEnforcementMode.SOFT,
    )
    assert "999" in untraced
    assert "<!-- UNTRACED:999 -->" in bound
    # STRICT mode → raises with detail.
    with pytest.raises(ra.EvidenceEnforcementError) as exc_info:
        bind_numeric_values(
            manuscript,
            evidence=store,
            enforcement_mode=ra.EvidenceEnforcementMode.STRICT,
        )
    assert "999" in exc_info.value.detail["untraced"]


def test_bind_numeric_values_skips_bibliographic_years(ra, tmp_path: Path):
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store = _store(ra, tmp_path)
    manuscript = (
        "Prior SOFA work was reported by Vincent et al., 1996; "
        "external validation followed in Ricu et al., 2023; "
        "and an earlier citation used (Vincent, 1996)."
    )

    bound, binding_map, untraced = bind_numeric_values(
        manuscript,
        evidence=store,
        enforcement_mode=ra.EvidenceEnforcementMode.STRICT,
    )

    assert bound == manuscript
    assert binding_map == {}
    assert untraced == []


def test_bind_numeric_values_does_not_skip_result_years(ra, tmp_path: Path):
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store = _store(ra, tmp_path)

    with pytest.raises(ra.EvidenceEnforcementError) as exc_info:
        bind_numeric_values(
            "In 2023, the primary outcome was observed during follow-up.",
            evidence=store,
            enforcement_mode=ra.EvidenceEnforcementMode.STRICT,
        )

    assert "2023" in exc_info.value.detail["untraced"]


def test_bind_numeric_values_skips_existing_evidence_placeholders(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store = _store(ra, tmp_path)
    store.register_step_summary_numerics(
        step_id="s1",
        evidence_id="evid_a",
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
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    store.register_step_summary_numerics(
        step_id="s1",
        evidence_id="evid_assoc",
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


def test_bind_numeric_values_skips_sha256_in_prose(ra, tmp_path: Path):
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store = _store(ra, tmp_path)
    store.register_step_summary_numerics(
        step_id="s1",
        evidence_id="evid_assoc",
        summary={"primary_or": 1.42},
    )
    manuscript = "The reproducibility envelope includes SHA-256 hashes and OR=1.42."
    _, _, untraced = bind_numeric_values(manuscript, evidence=store)
    assert "256" not in untraced
    assert "1.42" not in untraced


def test_numeric_claims_persist_across_store_reload(ra, tmp_path: Path):
    store_a = _store(ra, tmp_path)
    store_a.register_step_summary_numerics(
        step_id="s1",
        evidence_id="evid_a",
        summary={"primary_or": 1.42},
    )
    store_b = _store(ra, tmp_path)
    claims = store_b.numeric_claims()
    assert len(claims) == 1
    assert claims[0].source_field == "primary_or"
