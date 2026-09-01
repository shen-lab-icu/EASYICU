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
