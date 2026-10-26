"""Synthetic, row-free receipts shared by source and claim-lineage tests."""

import json

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.reporting.manuscript_provenance import build_manuscript_provenance
from easyicu.webserver.demo_source_contracts import get_source


MANUSCRIPT = (
    "The synthetic mean was 2.5[^value].\n\n"
    "[^value]: value=2.5; step=estimate; field=mean; evidence=summary\n"
)


def register(store, evidence_id, payload, *, kind="log", inputs=(), step=None):
    path = store.root / f"{evidence_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return store.register_file(
        source_path=path, evidence_id=evidence_id, kind=kind,
        description="Synthetic aggregate receipt", inputs=list(inputs),
        produced_by_step=step, producer="pipeline", generation_mode="system",
    )


def source_store(tmp_path, *, change=None, capsule_change=None, derived=False):
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    table = tmp_path / "cohort.parquet"
    # No reader may decode this as a table; its registered bytes identify it.
    table.write_bytes(b"synthetic-table-identity-only")
    cohort = store.register_file(
        source_path=table, evidence_id="cohort", kind="table", description="Synthetic cohort",
        producer="pipeline", generation_mode="system",
    )
    analysis_input = cohort
    if derived:
        table = tmp_path / "derived.parquet"
        table.write_bytes(b"synthetic-derived-table-identity")
        analysis_input = store.register_file(
            source_path=table, evidence_id="derived_cohort", kind="table", description="Synthetic subset",
            inputs=[cohort.evidence_id], producer="pipeline", generation_mode="system",
        )
    official = get_source("eicu_demo_v2_0_1")
    source = {
        "source_id": official.id, "database": official.database,
        "version": official.version, "archive_sha256": official.archive_sha256,
        "registration_id": "synthetic_registration",
        "path": "/private/not-for-the-browser", "credential": "not-for-the-browser",
    }
    context = {
        "cohort": {"database": official.database, "n_stays": 8, "provenance": {
            "materialized_cohort_window_hours": [0, 24], "source_identity": source,
            "cohort_path": "/home/synthetic/cohort.parquet",
        }},
        "materialized_inputs": {"cohort": {
            "cohort_sha256": cohort.sha256, "cohort_rows": 8,
            "column_bindings": {"synthetic_measure": {}, "synthetic_outcome": {}},
        }},
    }
    fingerprint = {"concept_dict_sha": "a" * 64, "sofa2_dict_sha": "b" * 64,
                   "concept_count": 3, "concept_dict_path": "/home/private/dictionary.json"}
    summary = {"mean": 2.5, "method": "synthetic_mean", "input_bindings": [{
        "input_key": "artifact:analysis_cohort", "evidence_id": analysis_input.evidence_id,
        "sha256": analysis_input.sha256, "loaded": True, "row_count": 8,
    }]}
    if change:
        change(context, fingerprint, summary)
    context_record = register(store, "research_context", context)
    capsule = {
        "schema_version": "easyicu.run_input_capsule/2",
        "context_evidence_id": context_record.evidence_id, "context_sha256": context_record.sha256,
        "cohort_sha256": cohort.sha256, "scientific_identity": {"database": official.database},
    }
    if capsule_change:
        capsule_change(capsule)
    register(store, "run_input_capsule", capsule)
    register(store, "concept_dict_fingerprint", fingerprint)
    register(store, "summary", summary, kind="statistic", inputs=[analysis_input.evidence_id], step="estimate")
    store.register_numeric_claim(value="2.5", canonical=2.5, evidence_id="summary", step_id="estimate", source_field="mean")
    return store


def project(store):
    return build_manuscript_provenance(manuscript=MANUSCRIPT, evidence=store, verify="mark")["claims"][0]


