"""Accepted baseline content must reach Writer and the final Methods audit."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from easyicu.research_agent.research_context.prompt_scope import scoped_reporting_context
from easyicu.research_agent.reporting import manuscript_quality, manuscript_sections
from easyicu.research_agent.reporting.writer_evidence import _executed_method_boundary_rows

from ..planning.test_baseline_requirements import _bound_context, _context
from .test_manuscript_quality import _valid_manuscript


def test_reporting_projection_preserves_bound_baseline_even_under_small_cap() -> None:
    context = _bound_context("age", "charlson")
    before = context.model_dump(mode="json")

    projected = scoped_reporting_context(context, max_variables=1)

    assert "cci_value" in {variable.name for variable in projected.variables}
    assert context.model_dump(mode="json") == before


def test_available_unrequested_score_does_not_expand_reporting_scope() -> None:
    projected = scoped_reporting_context(_bound_context("age"), max_variables=1)

    assert "cci_value" not in {variable.name for variable in projected.variables}
    assert "unrelated" not in {variable.name for variable in projected.variables}


def test_verified_executed_baseline_roster_reaches_method_digest() -> None:
    record = {
        "step_id": "baseline", "status": "ok",
        "writer_result_envelope_evidence_id": "sealed_envelope",
        "deterministic_standard_analysis": "grouped_table_one",
        "step_summary": {
            "analysis_family": "grouped_table_one", "group_by": "exposure",
            "variables": ["age", "cci_value"],
        },
    }

    rows = _executed_method_boundary_rows([record], evidence=SimpleNamespace())

    assert rows[0]["baseline_variables"] == ["age", "cci_value"]
    assert rows[0]["group_by"] == "exposure"
    assert _executed_method_boundary_rows([record], evidence=None) == []
    assert _executed_method_boundary_rows(
        [dict(record, status="failed")], evidence=SimpleNamespace(),
    ) == []


def _mentions():
    from easyicu.research_agent.reporting.manuscript_baseline import baseline_reporting_mentions

    return baseline_reporting_mentions(
        _bound_context("age", "charlson"),
        {"age": "Patient age in years", "cci_value": "Charlson Comorbidity Index"},
    )


def test_methods_coverage_is_not_satisfied_by_introduction_or_table_mentions() -> None:
    text = _valid_manuscript().replace(
        "Sepsis definitions and transparent cohort accounting matter for reproducible ICU research.",
        "Patient age in years and Charlson Comorbidity Index are discussed here.",
    )
    text += "\n\n## Table 1\n\nPatient age in years; Charlson Comorbidity Index.\n"
    audit = manuscript_quality.audit_manuscript_quality(
        text, expected_baseline_mentions=_mentions(),
    )

    missing = next(f for f in audit.findings if f.code == "MANUSCRIPT_BASELINE_METHODS_INCOMPLETE")
    assert missing.section == "Methods" and missing.severity == "error"
    assert missing.excerpts == ("age", "charlson")
    assert "methods" in manuscript_sections.quality_repair_section_keys(
        text, expected_baseline_mentions=_mentions(),
    )


def test_methods_accepts_authorized_label_without_forcing_raw_column_id() -> None:
    text = _valid_manuscript().replace(
        "### Variables\n",
        "### Variables\nPatient age in years and Charlson Comorbidity Index were baseline variables.\n",
    )
    audit = manuscript_quality.audit_manuscript_quality(
        text, expected_baseline_mentions=_mentions(),
    )

    assert not any(f.code == "MANUSCRIPT_BASELINE_METHODS_INCOMPLETE" for f in audit.findings)


def test_audit_comments_and_similar_names_do_not_complete_baseline_methods() -> None:
    text = _valid_manuscript().replace(
        "### Variables\n",
        "### Variables\n<!-- Charlson Comorbidity Index -->\nPatient age in years and charlson_like were used.\n",
    )
    audit = manuscript_quality.audit_manuscript_quality(
        text, expected_baseline_mentions=_mentions(),
    )

    missing = next(f for f in audit.findings if f.code == "MANUSCRIPT_BASELINE_METHODS_INCOMPLETE")
    assert missing.excerpts == ("charlson",)


def test_unbound_context_does_not_invent_baseline_reporting_requirements() -> None:
    from easyicu.research_agent.reporting.manuscript_baseline import baseline_reporting_mentions

    assert baseline_reporting_mentions(_context(), {"age": "Patient age"}) == {}


def test_sealed_baseline_roster_survives_scalar_projection_and_rejects_drift(tmp_path) -> None:
    from easyicu.research_agent.audits.envelope_consumers import (
        RegisteredOutputAuthorityError, RegisteredOutputEnvelopeConsumer,
    )
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.contracts.result_envelope import normalize_step_result_shadow
    from ..execution.test_step_result_envelope import (
        _UPSTREAM_STEP, _commit_upstream_sidecar, _modern_upstream_record,
        _register_upstream_table,
    )

    table = tmp_path / "exposure_outcome_summary.csv"
    table.write_text("group,n\n0,40\n1,60\n")
    summary = {
        "status": "ok", "analysis_family": "grouped_table_one",
        "group_by": "exposure", "variables": ["age", "cci_value"],
        "output_files": {"table:exposure_outcome_summary": table.name},
    }
    envelope = normalize_step_result_shadow(
        step_id=_UPSTREAM_STEP, step_summary=summary, output_dir=tmp_path, status="ok",
    )
    store = EvidenceStore(tmp_path / "run")
    artifact = _register_upstream_table(store, table)
    sidecar = _commit_upstream_sidecar(store, envelope)
    record = _modern_upstream_record(sidecar_evidence_id=sidecar)
    record.update(
        step_summary=summary, evidence_ids=[artifact.evidence_id, sidecar],
        deterministic_standard_analysis="grouped_table_one",
    )
    consumer = RegisteredOutputEnvelopeConsumer()

    projected = consumer.authoritative_writer_records([record], evidence_store=store)

    assert projected[0]["step_summary"]["variables"] == ["age", "cci_value"]
    assert _executed_method_boundary_rows(projected, evidence=store)[0]["baseline_variables"] == ["age", "cci_value"]
    record["step_summary"] = dict(summary, variables=["age", "wrong_score"])
    with pytest.raises(RegisteredOutputAuthorityError, match="canonical_source_digest_mismatch"):
        consumer.authoritative_writer_records([record], evidence_store=store)
