from __future__ import annotations

import json
from pathlib import Path

import pytest


def _registered_store(ra, tmp_path: Path):
    store = ra.EvidenceStore(root=tmp_path, enforcement_mode="strict")
    cohort = tmp_path / "cohort.parquet"
    cohort.write_bytes(b"aggregate-cohort-authority")
    store.register_file(
        kind="table",
        description="Materialized analysis cohort.",
        source_path=cohort,
        evidence_id="analysis_cohort",
        produced_by_step="cohort_definition",
        producer="pipeline",
        generation_mode="system",
    )
    code = tmp_path / "analysis.py"
    code.write_text("print('deterministic')\n", encoding="utf-8")
    store.register_file(
        kind="code",
        description="Deterministic association adapter.",
        source_path=code,
        evidence_id="association_code",
        produced_by_step="primary_association",
        inputs=["analysis_cohort"],
        producer="standard_executor",
        generation_mode="deterministic_standard",
    )
    summary = tmp_path / "step_summary.json"
    summary.write_text(
        json.dumps({"spline_knot_quantiles": [0.1, 0.5, 0.9]}),
        encoding="utf-8",
    )
    store.register_file(
        kind="statistic",
        description="Machine-readable primary association summary.",
        source_path=summary,
        evidence_id="primary_summary",
        produced_by_step="primary_association",
        inputs=["analysis_cohort"],
        script_evidence_id="association_code",
        producer="runner",
        generation_mode="deterministic_standard",
    )
    store.register_numeric_claim(
        value="0.5",
        canonical=0.5,
        evidence_id="primary_summary",
        step_id="primary_association",
        source_field="scientific_runtime_receipt.spline_knot_quantiles[1]",
    )
    return store


def test_manuscript_provenance_links_number_json_code_and_data(
    ra, tmp_path: Path
) -> None:
    from easyicu.research_agent.reporting.manuscript_provenance import (
        build_manuscript_provenance,
        strip_numeric_provenance,
    )

    store = _registered_store(ra, tmp_path)
    manuscript = (
        "# Evidence-bound draft\n\n"
        "The middle spline knot quantile was 0.5[^claim_1] "
        "[primary](evidence/primary_summary__step_summary.json).\n\n"
        "[^claim_1]: value=0.5; step=primary_association; "
        "field=scientific_runtime_receipt.spline_knot_quantiles[1]; "
        "evidence=primary_summary\n"
    )

    payload = build_manuscript_provenance(manuscript=manuscript, evidence=store)

    assert payload["schema_version"] == "easyicu.manuscript-provenance/1"
    assert payload["claim_ceiling"] == "analysis_only"
    assert payload["publication_authorized"] is False
    assert payload["claim_count"] == 1
    claim = payload["claims"][0]
    assert claim["display_value"] == "0.5"
    assert claim["source_json_pointer"] == (
        "/scientific_runtime_receipt/spline_knot_quantiles/1"
    )
    assert claim["evidence"]["evidence_id"] == "primary_summary"
    roles = {row["role"] for row in claim["related_artifacts"]}
    assert {"source_json", "analysis_code", "input_data"} <= roles
    serialized = json.dumps(payload)
    assert str(tmp_path) not in serialized
    assert "relative_path" not in serialized
    assert payload["integrity"]["patient_rows_returned"] is False
    claim_segments = [
        segment
        for block in payload["article_blocks"]
        for segment in block["segments"]
        if segment["kind"] == "claim"
    ]
    assert claim_segments == [{"kind": "claim", "text": "0.5", "claim_id": "claim_1"}]
    stripped = strip_numeric_provenance(manuscript)
    assert "0.5 [primary]" in stripped
    assert "[^claim_1]" not in stripped
    assert "field=scientific_runtime" not in stripped


def test_manuscript_provenance_fails_closed_on_tampered_field(
    ra, tmp_path: Path
) -> None:
    from easyicu.research_agent.reporting.manuscript_provenance import (
        ManuscriptProvenanceError,
        build_manuscript_provenance,
    )

    store = _registered_store(ra, tmp_path)
    manuscript = (
        "The spline knot was 0.5[^claim_1].\n\n"
        "[^claim_1]: value=0.5; step=primary_association; "
        "field=variable_groups.lact.missingness.max_fraction_missing; "
        "evidence=primary_summary\n"
    )

    with pytest.raises(ManuscriptProvenanceError, match="exactly one"):
        build_manuscript_provenance(manuscript=manuscript, evidence=store)


def test_manuscript_provenance_fails_closed_on_stale_evidence(
    ra, tmp_path: Path
) -> None:
    from easyicu.research_agent.reporting.manuscript_provenance import (
        ManuscriptProvenanceError,
        build_manuscript_provenance,
    )

    store = _registered_store(ra, tmp_path)
    record = next(
        item for item in store.records() if item.evidence_id == "primary_summary"
    )
    (tmp_path / record.relative_path).write_text("{}", encoding="utf-8")
    manuscript = (
        "The spline knot was 0.5[^claim_1].\n\n"
        "[^claim_1]: value=0.5; step=primary_association; "
        "field=scientific_runtime_receipt.spline_knot_quantiles[1]; "
        "evidence=primary_summary\n"
    )

    with pytest.raises(ManuscriptProvenanceError, match="digest is stale"):
        build_manuscript_provenance(manuscript=manuscript, evidence=store)


def _stale_manuscript() -> str:
    return (
        "The spline knot was 0.5[^claim_1].\n\n"
        "[^claim_1]: value=0.5; step=primary_association; "
        "field=scientific_runtime_receipt.spline_knot_quantiles[1]; "
        "evidence=primary_summary\n"
    )


def test_manuscript_provenance_marks_stale_reader_links_without_raising(
    ra, tmp_path: Path
) -> None:
    from easyicu.research_agent.reporting.manuscript_provenance import (
        build_manuscript_provenance,
    )

    store = _registered_store(ra, tmp_path)
    record = next(
        item for item in store.records() if item.evidence_id == "primary_summary"
    )
    (tmp_path / record.relative_path).write_text("{}", encoding="utf-8")

    payload = build_manuscript_provenance(
        manuscript=_stale_manuscript(),
        evidence=store,
        verify="mark",
    )

    claim = payload["claims"][0]
    assert claim["status"] == "stale"
    assert claim["evidence"]["status"] == "stale"
    assert claim["related_artifacts"] == []
    assert payload["integrity"]["numeric_claims_verified"] is False


def test_manuscript_provenance_attaches_display_identity_and_reader_context(
    ra, tmp_path: Path
) -> None:
    import hashlib

    from easyicu.research_agent.reporting.manuscript_provenance import (
        build_manuscript_provenance,
    )

    store = _registered_store(ra, tmp_path)
    table = tmp_path / "table_one.csv"
    table.write_text("variable,value\nlact,2\n", encoding="utf-8")
    store.register_file(
        kind="table",
        description="Registered Table 1.",
        source_path=table,
        evidence_id="table_one",
        produced_by_step="primary_association",
        producer="pipeline",
        generation_mode="system",
    )
    table_sha = hashlib.sha256(table.read_bytes()).hexdigest()
    contract_sha = "b" * 64

    payload = build_manuscript_provenance(
        manuscript=_stale_manuscript().replace("{}", "{}"),
        evidence=store,
        binding_map=None,
        display_index={
            table_sha: {"display_id": "Table 1", "contract_sha256": contract_sha}
        },
        method_summaries={
            "primary_association": {
                "intent": "Estimate the primary association.",
                "relative_path": "must/not/leak.csv",
            }
        },
        reader_notes=[
            {
                "code": "strict_untraceable_numeric_sentence_removed",
                "severity": "warning",
                "text": "A numeric sentence without a registered evidence source "
                "was removed from the reader text.",
            }
        ],
    )

    claim = payload["claims"][0]
    linked = [
        row
        for row in claim["related_artifacts"]
        if row.get("display_id") == "Table 1"
    ]
    assert linked
    assert linked[0]["display_contract_sha256"] == contract_sha
    assert claim["method_summary"] == {"intent": "Estimate the primary association."}
    notes_blocks = [
        block
        for block in payload["article_blocks"]
        if block.get("kind") == "verification_notes"
    ]
    assert notes_blocks
    assert notes_blocks[0]["notes"][0]["code"] == (
        "strict_untraceable_numeric_sentence_removed"
    )
    serialized = json.dumps(payload)
    assert "relative_path" not in serialized
    assert "must/not/leak.csv" not in serialized


@pytest.mark.parametrize("status", ["stale", "missing"])
def test_reader_marks_unavailable_related_code_but_strict_binding_rejects(ra, tmp_path, status):
    from easyicu.research_agent.reporting.manuscript_provenance import (
        ManuscriptProvenanceError, build_manuscript_provenance,
    )
    store = _registered_store(ra, tmp_path)
    record = next(item for item in store.records() if item.evidence_id == "association_code")
    path = tmp_path / record.relative_path
    if status == "stale":
        path.write_text("changed code")
    else:
        path.unlink()
    with pytest.raises(ManuscriptProvenanceError):
        build_manuscript_provenance(manuscript=_stale_manuscript(), evidence=store)
    payload = build_manuscript_provenance(manuscript=_stale_manuscript(), evidence=store, verify="mark")
    claim = payload["claims"][0]
    code = next(row for row in claim["related_artifacts"] if row["evidence_id"] == "association_code")
    assert code["status"] == status
    assert claim["status"] == status
    assert payload["integrity"]["numeric_claims_verified"] is False
