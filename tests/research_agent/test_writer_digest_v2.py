"""Tests for ``_render_writer_evidence_digest_v2`` (Phase-1 widening).

The v2 digest adds a "secondary numbers" block on top of the v1
primary block. The v1 path is unchanged and tested in
``test_pipeline.py``. These tests pin:

1. v2 with no per-step records returns identically to v1.
2. v2 without evidence falls back to walking step_summary directly.
3. v2 with evidence reads from the full NumericClaim registry and
   excludes fields already covered by the primary block.
4. The per-step secondary cap truncates with a clear marker.
5. statistic:<name> / <name> are treated as the same key when
   determining primary coverage (mirrors the v1 flatten behaviour).
6. The numeric-claim-overflow sentinel field is never echoed to the
   writer.
7. v2 is a strict superset of v1 — the v1 output appears at the top
   of the v2 output byte-for-byte.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementError,
    EvidenceStore,
)
from easyicu.research_agent.reporting import write_phase
from easyicu.research_agent.reporting.writer_evidence import (
    WRITER_DIGEST_PREFERRED_KEYS,
    _preferred_writer_evidence_names,
    _render_writer_evidence_digest,
    _render_writer_evidence_digest_v2,
)


def _record(step_id: str, status: str, summary: dict) -> dict:
    return {"step_id": step_id, "status": status, "step_summary": summary}


def _register_step_evidence(
    evidence: EvidenceStore,
    tmp_path: Path,
    *,
    step_id: str,
    evidence_id: str,
    payload: dict | None = None,
):
    source = tmp_path / f"{evidence_id}.json"
    source.write_text(json.dumps(payload or {}), encoding="utf-8")
    return evidence.register_file(
        kind="statistic",
        description=f"Summary for {step_id}.",
        source_path=source,
        produced_by_step=step_id,
        evidence_id=evidence_id,
        producer="runner",
    )


def test_write_phase_never_reads_append_only_evidence_without_current_ledger() -> None:
    """Every writer-side reader must share the current verified snapshot."""

    source = inspect.getsource(write_phase)

    assert "evidence.current_verified_records(" in source
    assert "evidence.records()" not in source


def test_live_writer_digest_uses_verified_result_envelope_records() -> None:
    source = inspect.getsource(write_phase._draft_manuscript)

    authority_call = source.index("authoritative_writer_records(")
    digest_call = source.index("_render_writer_evidence_digest_v2(")
    writer_prompt_call = source.index("writer.run(")
    assert authority_call < digest_call < writer_prompt_call
    assert source.count("per_step_records=writer_authority_records") == 1
    assert source.count("writer_authority_records,") == 2
    assert source.count("evidence=evidence") >= 2


def test_preferred_writer_evidence_names_excludes_records_with_active_findings(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    clean = tmp_path / "table_one.csv"
    clean.write_text("variable,value\nage,64\n", encoding="utf-8")
    flagged = tmp_path / "primary_association.csv"
    flagged.write_text("term,or\nexposure,1.2\n", encoding="utf-8")
    store.register_file(
        kind="table",
        description="Clean table one.",
        source_path=clean,
        evidence_id="table_table_one",
        aliases=["table_one"],
    )
    record = store.register_file(
        kind="table",
        description="Flagged primary association.",
        source_path=flagged,
        evidence_id="primary_association_table",
        aliases=["primary_association"],
    )
    store.update_record(
        record.evidence_id,
        finding_severity="warning",
        finding_messages=["visual or analysis caveat"],
    )

    names = _preferred_writer_evidence_names(store)

    assert "table_one" in names
    assert "primary_association" not in names
    assert "primary_association_table" not in names


def test_preferred_writer_evidence_names_excludes_retired_step_alias(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    retired_path = tmp_path / "primary_association.csv"
    current_path = tmp_path / "current_table.csv"
    retired_path.write_text("term,or\nexposure,999\n", encoding="utf-8")
    current_path.write_text("term,or\nexposure,1.2\n", encoding="utf-8")
    retired = store.register_file(
        kind="table",
        description="Retired association.",
        source_path=retired_path,
        produced_by_step="03_model",
        aliases=["primary_association"],
    )
    current = store.register_file(
        kind="table",
        description="Current result.",
        source_path=current_path,
        produced_by_step="04_current",
    )
    records = [
        {
            **_record("03_model", "ok", {}),
            "evidence_ids": [retired.evidence_id],
        },
        {
            **_record("03_model", "contract_failed", {}),
            "evidence_ids": [],
        },
        {
            **_record("04_current", "ok", {}),
            "evidence_ids": [current.evidence_id],
        },
    ]

    names = _preferred_writer_evidence_names(store, records)

    assert "primary_association" not in names
    assert retired.evidence_id not in names
    assert current.evidence_id in names


def test_bind_manuscript_rejects_retired_first_write_alias(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    retired_path = tmp_path / "retired.csv"
    current_path = tmp_path / "current.csv"
    retired_path.write_text("term,value\nold,999\n", encoding="utf-8")
    current_path.write_text("term,value\nnew,1\n", encoding="utf-8")
    retired = store.register_file(
        kind="table",
        description="Retired result.",
        source_path=retired_path,
        produced_by_step="03_model",
        evidence_id="retired_result",
        aliases=["primary_association"],
    )
    current = store.register_file(
        kind="table",
        description="Current result.",
        source_path=current_path,
        produced_by_step="04_current",
        evidence_id="current_result",
    )
    records = [
        {
            **_record("03_model", "ok", {}),
            "evidence_ids": [retired.evidence_id],
        },
        {
            **_record("03_model", "contract_failed", {}),
            "evidence_ids": [],
        },
        {
            **_record("04_current", "ok", {}),
            "evidence_ids": [current.evidence_id],
        },
    ]

    with pytest.raises(EvidenceEnforcementError) as exc_info:
        store.bind_manuscript(
            "Result {evidence:primary_association}.",
            per_step_records=records,
        )

    assert "primary_association" in str(exc_info.value)


def test_v2_empty_records_returns_v1_output() -> None:
    out_v1 = _render_writer_evidence_digest([])
    out_v2 = _render_writer_evidence_digest_v2([])
    assert out_v1 == out_v2


def test_v2_without_evidence_falls_back_to_summary_walk() -> None:
    records = [
        _record(
            "03_primary",
            "ok",
            {
                # In primary block:
                "sample_size": 785,
                "primary_or": 1.42,
                # Generic outcome summaries now belong in the primary block:
                "median_los_icu": 3.2,
                "median_los_hospital": 6.1,
                # Outside primary keys — should appear in secondary block:
                "cohort_male_fraction": 0.56,
            },
        )
    ]
    out = _render_writer_evidence_digest_v2(records)
    # primary block intact
    assert '"sample_size": 785' in out
    assert '"primary_or": 1.42' in out
    assert '"median_los_icu": 3.2' in out
    assert '"median_los_hospital": 6.1' in out
    # secondary header + entries
    assert "## secondary numbers" in out
    assert "cohort_male_fraction=0.56" in out


def test_v2_without_evidence_skips_when_nothing_outside_primary() -> None:
    records = [
        _record(
            "03_primary",
            "ok",
            {
                # everything is in primary keys
                "sample_size": 785,
                "primary_or": 1.42,
                "ci_lower": 1.10,
                "ci_upper": 1.83,
                "p_value": 0.001,
            },
        )
    ]
    out = _render_writer_evidence_digest_v2(records)
    # no secondary header should appear when nothing was added
    assert "## secondary numbers" not in out


def test_v2_with_evidence_reads_claim_registry(tmp_path: Path) -> None:
    evidence = EvidenceStore(root=tmp_path)
    _register_step_evidence(
        evidence,
        tmp_path,
        step_id="03_primary",
        evidence_id="03_primary_summary",
    )
    # Register one primary leaf and two secondary leaves into the
    # claim registry. We bypass the source-step_summary path on purpose
    # to assert v2 reads the *registry*, not just summary.
    evidence.register_numeric_claim(
        value="1.42",
        canonical=1.42,
        evidence_id="03_primary_summary",
        step_id="03_primary",
        source_field="primary_or",  # in primary keys
    )
    evidence.register_numeric_claim(
        value="0.86",
        canonical=0.86,
        evidence_id="03_primary_summary",
        step_id="03_primary",
        source_field="cohort_male_fraction",  # not in primary keys
    )
    records = [
        {
            **_record("03_primary", "ok", {"sample_size": 785, "primary_or": 1.42}),
            "evidence_ids": ["03_primary_summary"],
        }
    ]
    out = _render_writer_evidence_digest_v2(records, evidence=evidence)
    assert "## secondary numbers" in out
    assert "cohort_male_fraction=0.86" in out
    assert "## numeric citation authority" in out
    assert "- 03_primary: {evidence:03_primary_summary}" in out
    assert "cite={evidence:03_primary_summary}" in out
    # The primary-keys field MUST NOT appear in the secondary block.
    secondary = out.split("## secondary numbers", 1)[1]
    assert "primary_or=" not in secondary
    assert "median_los_icu=" not in secondary


def test_v2_excludes_claims_from_retired_failed_attempt(tmp_path: Path) -> None:
    evidence = EvidenceStore(root=tmp_path)
    _register_step_evidence(
        evidence,
        tmp_path,
        step_id="03_primary",
        evidence_id="failed_summary",
    )
    _register_step_evidence(
        evidence,
        tmp_path,
        step_id="03_primary",
        evidence_id="current_summary",
    )
    evidence.register_numeric_claim(
        value="999",
        canonical=999.0,
        evidence_id="failed_summary",
        step_id="03_primary",
        source_field="secondary_metric",
    )
    evidence.register_numeric_claim(
        value="1.25",
        canonical=1.25,
        evidence_id="current_summary",
        step_id="03_primary",
        source_field="secondary_metric",
    )
    records = [
        {
            **_record("03_primary", "contract_failed", {"secondary_metric": 999}),
            "evidence_ids": ["failed_summary"],
        },
        {
            **_record("03_primary", "ok", {"secondary_metric": 1.25}),
            "evidence_ids": ["current_summary"],
        },
    ]

    out = _render_writer_evidence_digest_v2(records, evidence=evidence)

    assert "secondary_metric=1.25" in out
    assert "secondary_metric=999" not in out


def test_v2_does_not_let_failed_claim_borrow_another_steps_active_id(
    tmp_path: Path,
) -> None:
    evidence = EvidenceStore(root=tmp_path)
    _register_step_evidence(
        evidence,
        tmp_path,
        step_id="04_current",
        evidence_id="current_other_step",
    )
    evidence.register_numeric_claim(
        value="999",
        canonical=999.0,
        evidence_id="current_other_step",
        step_id="03_failed",
        source_field="secondary_metric",
    )
    evidence.register_numeric_claim(
        value="777",
        canonical=777.0,
        evidence_id="orphan",
        step_id="99_orphan",
        source_field="secondary_metric",
    )
    run_path = tmp_path / "research_context.json"
    run_path.write_text('{"n": 1}', encoding="utf-8")
    evidence.register_file(
        kind="log",
        description="Run-level context.",
        source_path=run_path,
        evidence_id="research_context",
        producer="pipeline",
    )
    evidence.register_numeric_claim(
        value="666",
        canonical=666.0,
        evidence_id="research_context",
        step_id="99_orphan",
        source_field="secondary_metric",
    )
    records = [
        {
            **_record("03_failed", "contract_failed", {}),
            "evidence_ids": ["retired_failed"],
        },
        {
            **_record("04_current", "ok", {}),
            "evidence_ids": ["current_other_step"],
        },
    ]

    out = _render_writer_evidence_digest_v2(records, evidence=evidence)

    assert "secondary_metric=999" not in out
    assert "secondary_metric=777" not in out
    assert "secondary_metric=666" not in out
    assert "99_orphan" not in out


def test_writer_digest_hides_values_from_current_failed_attempt() -> None:
    records = [
        _record("03_primary", "ok", {"secondary_metric": 1.25}),
        _record("03_primary", "contract_failed", {"secondary_metric": 999}),
    ]

    out = _render_writer_evidence_digest_v2(records)

    assert "03_primary [contract_failed]" in out
    assert "secondary_metric=999" not in out
    assert "secondary_metric=1.25" not in out


def test_v2_secondary_cap_truncates_with_marker(tmp_path: Path) -> None:
    evidence = EvidenceStore(root=tmp_path)
    _register_step_evidence(
        evidence,
        tmp_path,
        step_id="03_primary",
        evidence_id="03_primary_summary",
    )
    # Twenty-five secondary leaves; cap is 3 → expect 3 visible + a
    # "more leaves omitted" marker mentioning 22.
    for i in range(25):
        evidence.register_numeric_claim(
            value=str(i + 100),
            canonical=float(i + 100),
            evidence_id="03_primary_summary",
            step_id="03_primary",
            source_field=f"leaf_{i:02d}",
        )
    records = [
        {
            **_record("03_primary", "ok", {}),
            "evidence_ids": ["03_primary_summary"],
        }
    ]
    out = _render_writer_evidence_digest_v2(
        records, evidence=evidence, secondary_cap_per_step=3
    )
    # Three visible entries
    visible_count = sum(1 for line in out.splitlines() if line.startswith("  leaf_"))
    assert visible_count == 3
    # Truncation marker mentions remaining count
    assert "22 more leaves omitted" in out


def test_v2_secondary_cap_counts_only_uncovered_claims(tmp_path: Path) -> None:
    evidence = EvidenceStore(root=tmp_path)
    _register_step_evidence(
        evidence,
        tmp_path,
        step_id="03_primary",
        evidence_id="03_primary_summary",
    )
    evidence.register_numeric_claim(
        value="1.42",
        canonical=1.42,
        evidence_id="03_primary_summary",
        step_id="03_primary",
        source_field="primary_or",
    )
    for i in range(5):
        evidence.register_numeric_claim(
            value=str(i + 100),
            canonical=float(i + 100),
            evidence_id="03_primary_summary",
            step_id="03_primary",
            source_field=f"leaf_{i:02d}",
        )
    records = [
        {
            **_record("03_primary", "ok", {"primary_or": 1.42}),
            "evidence_ids": ["03_primary_summary"],
        }
    ]
    out = _render_writer_evidence_digest_v2(
        records, evidence=evidence, secondary_cap_per_step=2
    )
    assert "3 more leaves omitted" in out


def test_v2_reserves_cap_for_typed_reportable_results(tmp_path: Path) -> None:
    evidence = EvidenceStore(root=tmp_path)
    _register_step_evidence(
        evidence,
        tmp_path,
        step_id="04_descriptive",
        evidence_id="04_descriptive_summary",
    )
    evidence.register_numeric_claim(
        value="999",
        canonical=999.0,
        evidence_id="04_descriptive_summary",
        step_id="04_descriptive",
        source_field="diagnostic_first",
    )
    for source_field, value in (
        ("reportable_descriptive_results.groups[0].risk_pct", "14.7"),
        ("reportable_descriptive_results.groups[1].risk_pct", "6.3"),
        ("reportable_descriptive_results.overall_outcome.risk_pct", "10.0"),
    ):
        evidence.register_numeric_claim(
            value=value,
            canonical=float(value),
            evidence_id="04_descriptive_summary",
            step_id="04_descriptive",
            source_field=source_field,
        )
    records = [
        {
            **_record("04_descriptive", "ok", {}),
            "evidence_ids": ["04_descriptive_summary"],
        }
    ]

    out = _render_writer_evidence_digest_v2(
        records,
        evidence=evidence,
        secondary_cap_per_step=1,
    )

    assert "groups[0].risk_pct=14.7" in out
    assert "groups[1].risk_pct=6.3" in out
    assert "overall_outcome.risk_pct=10.0" in out
    assert "diagnostic_first=999" not in out
    assert "1 more leaves omitted" in out


def test_v2_shows_derived_numbers_in_separate_block(tmp_path: Path) -> None:
    evidence = EvidenceStore(root=tmp_path)
    _register_step_evidence(
        evidence,
        tmp_path,
        step_id="03_primary",
        evidence_id="03_primary_summary",
    )
    evidence.register_numeric_claim(
        value="1.42",
        canonical=1.42,
        evidence_id="03_primary_summary",
        step_id="03_primary",
        source_field="primary_or",
    )
    evidence.register_numeric_claim(
        value="0.13",
        canonical=0.13,
        evidence_id="03_primary_summary",
        step_id="03_primary",
        source_field="primary_or_se",
    )
    evidence.register_derived_claim(
        name="primary_or_ci_low",
        formula="exp(log(primary_or) - 1.96 * primary_or_se)",
        explanation="Lower 95% CI for primary OR, log-normal approx",
        sources={
            "primary_or": ("03_primary", "primary_or"),
            "primary_or_se": ("03_primary", "primary_or_se"),
        },
        evidence_id="03_primary_summary",
        step_id="03_primary",
    )
    records = [
        {
            **_record(
                "03_primary",
                "ok",
                {"primary_or": 1.42, "primary_or_se": 0.13},
            ),
            "evidence_ids": ["03_primary_summary"],
        }
    ]
    out = _render_writer_evidence_digest_v2(records, evidence=evidence)
    assert "## derived numbers" in out
    assert "primary_or_ci_low=" in out
    assert "formula=exp(log(primary_or) - 1.96 * primary_or_se)" in out
    assert "sources=03_primary.primary_or, 03_primary.primary_or_se" in out
    secondary = out.split("## secondary numbers", 1)[1]
    assert "primary_or_ci_low=" not in secondary


def test_v2_treats_statistic_prefix_as_same_key(tmp_path: Path) -> None:
    """``statistic:auroc`` and ``auroc`` should be treated as the same
    primary key when deciding what's already covered. v1's
    ``_first_present_scalar`` already flattens these (see
    ``test_render_writer_evidence_digest_flattens_nested_statistics``),
    and v2 should preserve that semantics so a claim emitted with
    source_field=``statistic:auroc`` isn't shown in the secondary
    block when v1 already cited ``auroc`` for the same step."""
    evidence = EvidenceStore(root=tmp_path)
    _register_step_evidence(
        evidence,
        tmp_path,
        step_id="04_model",
        evidence_id="04_model",
    )
    evidence.register_numeric_claim(
        value="0.79",
        canonical=0.79,
        evidence_id="04_model",
        step_id="04_model",
        source_field="statistic:auroc",
    )
    records = [
        {
            **_record(
                "04_model",
                "ok",
                {"statistic": {"auroc": 0.79}},  # v1 cites auroc=0.79
            ),
            "evidence_ids": ["04_model"],
        }
    ]
    out = _render_writer_evidence_digest_v2(records, evidence=evidence)
    # v1 primary block emitted auroc; v2 must NOT echo statistic:auroc=0.79
    # as a "secondary" entry.
    if "## secondary numbers" in out:
        secondary = out.split("## secondary numbers", 1)[1]
        assert "statistic:auroc=" not in secondary
        assert "auroc=" not in secondary


def test_v2_skips_overflow_sentinel(tmp_path: Path) -> None:
    """``register_step_summary_numerics`` registers a sentinel
    ``__easyicu_numeric_claim_overflow__`` claim when the per-step
    cap is exceeded. The writer must never see this — it's an audit
    marker, not a citable value."""
    evidence = EvidenceStore(root=tmp_path)
    _register_step_evidence(
        evidence,
        tmp_path,
        step_id="03_step",
        evidence_id="03_step",
    )
    evidence.register_numeric_claim(
        value="42",
        canonical=42.0,
        evidence_id="03_step",
        step_id="03_step",
        source_field="__easyicu_numeric_claim_overflow__",
    )
    evidence.register_numeric_claim(
        value="3.2",
        canonical=3.2,
        evidence_id="03_step",
        step_id="03_step",
        source_field="cohort_male_fraction",
    )
    records = [
        {
            **_record("03_step", "ok", {}),
            "evidence_ids": ["03_step"],
        }
    ]
    out = _render_writer_evidence_digest_v2(records, evidence=evidence)
    assert "__easyicu_numeric_claim_overflow__" not in out
    assert "cohort_male_fraction=3.2" in out


def test_v2_output_strictly_contains_v1_output() -> None:
    records = [
        _record(
            "03_primary",
            "ok",
            {
                "sample_size": 785,
                "primary_or": 1.42,
                "median_los_icu": 3.2,
            },
        )
    ]
    out_v1 = _render_writer_evidence_digest(records)
    out_v2 = _render_writer_evidence_digest_v2(records)
    # v1 output is the prefix of v2 output. This is the strongest
    # backwards-compat guarantee: turning on the flag never deletes
    # information the writer was seeing before.
    assert out_v2.startswith(out_v1)


def test_preferred_writer_names_include_exact_numeric_evidence_owner(
    tmp_path: Path,
) -> None:
    evidence = EvidenceStore(root=tmp_path)
    _register_step_evidence(
        evidence,
        tmp_path,
        step_id="03_distribution",
        evidence_id="distribution_summary_exact",
    )
    evidence.register_numeric_claim(
        value="64.01",
        canonical=64.01,
        evidence_id="distribution_summary_exact",
        step_id="03_distribution",
        source_field="prevalence_pct",
    )
    records = [
        {
            **_record("03_distribution", "ok", {"prevalence_pct": 64.01}),
            "evidence_ids": ["distribution_summary_exact"],
        }
    ]

    names = _preferred_writer_evidence_names(evidence, records)

    assert "distribution_summary_exact" in names


def test_writer_digest_preferred_keys_is_tuple_and_nonempty() -> None:
    # Symbolic guard: the exported constant is iterable and unchanged
    # in shape. Useful for downstream consumers that may want to
    # extend the primary set.
    assert isinstance(WRITER_DIGEST_PREFERRED_KEYS, tuple)
    assert len(WRITER_DIGEST_PREFERRED_KEYS) > 0
    assert "primary_or" in WRITER_DIGEST_PREFERRED_KEYS
    assert "hazard_ratio" in WRITER_DIGEST_PREFERRED_KEYS
    assert "average_treatment_effect" in WRITER_DIGEST_PREFERRED_KEYS
    assert "median_los_icu" in WRITER_DIGEST_PREFERRED_KEYS
    assert "auroc" in WRITER_DIGEST_PREFERRED_KEYS


def test_primary_digest_flattens_unique_nested_p_value_beside_effect(
    tmp_path: Path,
) -> None:
    digest = _render_writer_evidence_digest(
        per_step_records=[
            _record(
                "primary_model",
                "ok",
                {
                    "primary_or": 1.96,
                    "scientific_runtime_receipt": {
                        "functional_form_comparison": {"p_value": 0.619}
                    },
                },
            )
        ],
        run_dir=tmp_path,
        include_robustness_panel=False,
    )

    assert '"primary_or": 1.96' in digest
    assert '"p_value": 0.619' in digest
