"""A bound aggregate walks only the selected run's registered receipts."""

import json

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.reporting.manuscript_provenance import build_manuscript_provenance
from easyicu.webserver.demo_source_contracts import get_source


KINDS = ["statistic", "analysis_input", "cohort", "column_definitions", "concept_dictionary", "data_source"]
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


def test_every_claim_has_an_ordered_openable_source_chain(tmp_path):
    store = source_store(tmp_path)
    claim = project(store)
    chain = claim["lineage"]
    assert [node["kind"] for node in chain] == KINDS
    assert [node["status"] for node in chain] == ["current"] * 6
    assert [node["facts"] for node in chain] == [
        {"step_id": "estimate", "method": "synthetic_mean"}, {"rows": 8},
        {"rows": 8, "window_start_hours": 0, "window_end_hours": 24}, {"count": 2},
        {"base_sha256": "a" * 64, "overlay_sha256": "b" * 64, "concept_count": 3},
        {"database": "eicu_demo", "version": "2.0.1", "source_id": "eicu_demo_v2_0_1",
         "registration_id": "synthetic_registration", "official_sha256_matches": True,
         "doi": "10.13026/4mxk-na84", "physionet_slug": "eicu-crd-demo"},
    ]
    for node in chain:
        assert set(node) == {"kind", "status", "sha256", "evidence_id", "facts", "missing_reason"}
        assert node["missing_reason"] is None
        assert store.get(node["evidence_id"]).sha256 == node["sha256"]
    assert {row["evidence_id"] for row in claim["related_artifacts"]} == {"summary", "cohort"}
    text = json.dumps(chain)
    assert "not-for-the-browser" not in text
    assert "/home/" not in text and "/private/" not in text


@pytest.mark.parametrize("evidence_id,index", [("summary", 0), ("cohort", 1), ("research_context", 2), ("run_input_capsule", 2), ("concept_dict_fingerprint", 4)])
@pytest.mark.parametrize("state", ["stale", "missing"])
def test_unavailable_receipts_never_supply_facts(tmp_path, evidence_id, index, state):
    store = source_store(tmp_path)
    record = store.get(evidence_id)
    path = tmp_path / record.relative_path
    if state == "stale":
        path.write_text('{"rows":999}')
    else:
        path.unlink()
    chain = project(store)["lineage"]
    assert [node["kind"] for node in chain] == KINDS
    assert chain[index]["status"] == state
    assert chain[index]["missing_reason"] == ("artifact_digest_mismatch" if state == "stale" else "artifact_unavailable")
    assert chain[index]["facts"] == {}
    if index in {0, 1, 2}:
        assert all(node["status"] == "missing" for node in chain[index + 1:])


def test_absent_receipts_keep_all_six_nodes_without_ambient_fallback(tmp_path):
    store = EvidenceStore(tmp_path)
    register(store, "summary", {"mean": 2.5}, kind="statistic", step="estimate")
    store.register_numeric_claim(value="2.5", canonical=2.5, evidence_id="summary", step_id="estimate", source_field="mean")
    (tmp_path / "research_context.json").write_text('{"cohort":{"n_stays":999}}')
    chain = project(store)["lineage"]
    assert [node["kind"] for node in chain] == KINDS
    for node in chain[1:]:
        assert node["status"] == "missing" and node["missing_reason"]
        assert node["facts"] == {} and node["evidence_id"] is None


@pytest.mark.parametrize("alter", ["digest", "database", "version", "id"])
def test_demo_label_alone_never_proves_official_release(tmp_path, alter):
    def change(context, fingerprint, summary):
        source = context["cohort"]["provenance"]["source_identity"]
        source[{"digest": "archive_sha256", "database": "database", "version": "version", "id": "source_id"}[alter]] = "c" * 64 if alter == "digest" else "unverified"
    chain = project(source_store(tmp_path, change=change))["lineage"]
    facts = chain[-1]["facts"]
    assert "doi" not in facts and "physionet_slug" not in facts
    assert facts.get("official_sha256_matches") is not True


def test_ambiguous_or_unloaded_input_cannot_borrow_the_run_cohort(tmp_path):
    def change(context, fingerprint, summary):
        summary["input_bindings"].append(dict(summary["input_bindings"][0], evidence_id="other"))
    chain = project(source_store(tmp_path, change=change))["lineage"]
    assert chain[1]["status"] == "missing"
    assert chain[1]["missing_reason"] == "analysis_input_ambiguous"
    assert all(node["status"] == "missing" for node in chain[2:])


def test_input_receipt_digest_must_match_selected_registered_bytes(tmp_path):
    def change(context, fingerprint, summary):
        summary["input_bindings"][0]["sha256"] = "c" * 64
    chain = project(source_store(tmp_path, change=change))["lineage"]
    assert chain[1]["status"] == "stale"
    assert chain[1]["missing_reason"] == "input_digest_mismatch"
    assert chain[1]["facts"] == {}
    assert all(node["status"] == "missing" for node in chain[2:])


@pytest.mark.parametrize("field", ["input_sha256", "analysis_input_sha256", "counting_process_input_sha256"])
def test_digest_only_summaries_resolve_one_exact_registered_table(tmp_path, field):
    def change(context, fingerprint, summary):
        summary[field] = summary.pop("input_bindings")[0]["sha256"]
    store = source_store(tmp_path, change=change)
    chain = project(store)["lineage"]
    assert [node["status"] for node in chain] == ["current"] * 6
    assert chain[1]["evidence_id"] == "cohort"
    assert chain[1]["facts"] == {}  # No consumed row count was recorded.


@pytest.mark.parametrize("field,value,reason", [
    ("loaded", False, "analysis_input_not_loaded"),
    ("sha256", None, "input_digest_not_recorded"),
    ("evidence_id", "absent", "artifact_unregistered"),
])
def test_an_incomplete_input_receipt_is_explicitly_missing(tmp_path, field, value, reason):
    def change(context, fingerprint, summary):
        summary["input_bindings"][0][field] = value
    chain = project(source_store(tmp_path, change=change))["lineage"]
    assert chain[1]["status"] == "missing"
    assert chain[1]["missing_reason"] == reason
    assert all(node["status"] == "missing" for node in chain[2:])


def test_unrecorded_facts_and_unsafe_scalars_are_omitted(tmp_path):
    def change(context, fingerprint, summary):
        summary["method"] = "/home/private/method.py"
        summary["input_bindings"][0]["row_count"] = True
        typed = context["materialized_inputs"]["cohort"]
        typed.pop("cohort_rows")
        typed.pop("column_bindings")
        context["cohort"].pop("n_stays")
        provenance = context["cohort"]["provenance"]
        provenance["materialized_cohort_window_hours"] = [float("nan"), True]
        source = provenance["source_identity"]
        source["version"] = "C:\\private\\version"
        source["registration_id"] = "postgres://secret@local/db"
        fingerprint.pop("concept_count")
        fingerprint.pop("sofa2_dict_sha")
    chain = project(source_store(tmp_path, change=change))["lineage"]
    assert chain[0]["facts"] == {"step_id": "estimate"}
    assert chain[1]["facts"] == chain[2]["facts"] == {}
    assert chain[3]["status"] == "missing"
    assert chain[3]["missing_reason"] == "column_definitions_not_recorded"
    assert chain[4]["facts"] == {"base_sha256": "a" * 64}
    assert "version" not in chain[5]["facts"] and "registration_id" not in chain[5]["facts"]
    json.dumps(chain, allow_nan=False)


def test_other_run_cohort_digest_cannot_attach_context(tmp_path):
    store = source_store(tmp_path, capsule_change=lambda capsule: capsule.update(cohort_sha256="d" * 64))
    chain = project(store)["lineage"]
    assert chain[2]["status"] == "missing"
    assert chain[2]["missing_reason"] == "cohort_input_unlinked"
    assert all(node["facts"] == {} for node in chain[2:])


def test_capsule_context_digest_is_required(tmp_path):
    store = source_store(tmp_path, capsule_change=lambda capsule: capsule.update(context_sha256="d" * 64))
    chain = project(store)["lineage"]
    assert chain[2]["status"] == "stale"
    assert chain[2]["missing_reason"] == "context_digest_mismatch"
    assert all(node["facts"] == {} for node in chain[2:])


def test_context_digest_and_typed_cohort_digest_are_both_checked(tmp_path):
    def change(context, fingerprint, summary):
        context["materialized_inputs"]["cohort"]["cohort_sha256"] = "c" * 64
    chain = project(source_store(tmp_path, change=change))["lineage"]
    assert chain[2]["status"] == "stale"
    assert chain[2]["missing_reason"] == "cohort_digest_mismatch"
    assert all(node["facts"] == {} for node in chain[2:])


def test_reader_assembly_carries_lineage_without_changing_number(tmp_path):
    from easyicu.research_agent.reporting.manuscript_reader import build_manuscript_reader
    store = source_store(tmp_path)
    payload = build_manuscript_reader(manuscript=MANUSCRIPT, evidence=store)
    claim = payload["claims"][0]
    assert claim["source_value"] == "2.5"
    assert [node["kind"] for node in claim["lineage"]] == KINDS


def test_method_summary_name_is_retained_when_the_statistic_omits_method(tmp_path):
    def change(context, fingerprint, summary):
        summary.pop("method")
    store = source_store(tmp_path, change=change)
    payload = build_manuscript_provenance(
        manuscript=MANUSCRIPT, evidence=store,
        method_summaries={"estimate": {"intent": "Synthetic mean"}},
    )
    assert payload["claims"][0]["lineage"][0]["facts"]["method"] == "Synthetic mean"


def test_derived_analysis_input_walks_registered_parent_edges(tmp_path):
    store = source_store(tmp_path, derived=True)
    chain = project(store)["lineage"]
    assert chain[1]["evidence_id"] == "derived_cohort"
    assert chain[2]["evidence_id"] == "research_context"
    assert all(node["status"] == "current" for node in chain)
    # A missing ancestor prevents use of the run-level context even when the
    # immediate analysis input remains current.
    (tmp_path / store.get("cohort").relative_path).unlink()
    chain = project(store)["lineage"]
    assert chain[1]["status"] == "current"
    assert chain[2]["status"] == "missing"
    assert chain[2]["missing_reason"] == "cohort_input_unlinked"


def test_digest_only_summary_does_not_choose_between_equal_byte_records(tmp_path):
    def change(context, fingerprint, summary):
        summary["input_sha256"] = summary.pop("input_bindings")[0]["sha256"]
    store = source_store(tmp_path, change=change)
    store.register_file(
        source_path=tmp_path / "cohort.parquet", evidence_id="other_table", kind="table",
        description="Other identity with the same bytes", producer="pipeline", generation_mode="system",
    )
    chain = project(store)["lineage"]
    assert chain[1]["status"] == "missing"
    assert chain[1]["missing_reason"] == "analysis_input_ambiguous"


def test_registered_symlink_never_supplies_lineage_facts(tmp_path):
    store = source_store(tmp_path)
    path = tmp_path / store.get("concept_dict_fingerprint").relative_path
    target = tmp_path / "identical_copy.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    node = project(store)["lineage"][4]
    assert node["status"] == "missing"
    assert node["missing_reason"] == "artifact_unavailable"
    assert node["facts"] == {}


def test_non_dictionary_json_is_not_a_dictionary_link(tmp_path):
    def change(context, fingerprint, summary):
        fingerprint.clear()
        fingerprint["patient_rows"] = [{"identifier": "must-not-leak"}]
    node = project(source_store(tmp_path, change=change))["lineage"][4]
    assert node["status"] == "missing"
    assert node["missing_reason"] == "dictionary_digest_not_recorded"
    assert node["facts"] == {}


def test_counts_and_windows_fit_browser_numeric_scalars(tmp_path):
    def change(context, fingerprint, summary):
        summary["input_bindings"][0]["row_count"] = 2 ** 60
        context["materialized_inputs"]["cohort"]["cohort_rows"] = 2 ** 60
        context["cohort"]["provenance"]["materialized_cohort_window_hours"] = [10 ** 400, float("inf")]
        fingerprint["concept_count"] = 2 ** 60
    chain = project(source_store(tmp_path, change=change))["lineage"]
    assert chain[1]["facts"] == chain[2]["facts"] == {}
    assert "concept_count" not in chain[4]["facts"]


def test_database_name_without_a_source_identity_keeps_the_source_link_missing(tmp_path):
    def change(context, fingerprint, summary):
        context["cohort"]["provenance"].pop("source_identity")
    node = project(source_store(tmp_path, change=change))["lineage"][-1]
    assert node["status"] == "missing"
    assert node["missing_reason"] == "source_identity_not_recorded"
    assert node["facts"] == {"database": "eicu_demo"}
