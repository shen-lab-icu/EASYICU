"""MCP server protocol surface."""

from __future__ import annotations

import json

import pandas as pd

from easyicu.concept.availability_signal import ConceptAvailabilityRecord


def test_mcp_initialize_and_tools_list(ra):
    from easyicu.research_agent.mcp_server import handle_jsonrpc

    init = handle_jsonrpc({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {},
    })
    assert init["result"]["capabilities"]["tools"] == {}
    assert init["result"]["serverInfo"]["name"] == "easyicu-research-agent"

    listed = handle_jsonrpc({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    })
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert {
        "research_agent.run",
        "research_agent.list_skills",
        "research_agent.read_manifest",
    } <= names


def test_mcp_tools_call_wraps_tool_result_as_content(ra):
    from easyicu.research_agent.mcp_server import handle_jsonrpc

    resp = handle_jsonrpc({
        "jsonrpc": "2.0",
        "id": "skills",
        "method": "tools/call",
        "params": {
            "name": "research_agent.list_skills",
            "arguments": {},
        },
    })
    assert resp["id"] == "skills"
    assert resp["result"]["isError"] is False
    text = resp["result"]["content"][0]["text"]
    data = json.loads(text)
    keys = {skill["key"] for skill in data["skills"]}
    assert {
        "association_analysis",
        "prediction_model",
        "data_quality_audit",
    } <= keys


def test_mcp_legacy_tool_shape_still_dispatches(ra):
    from easyicu.research_agent.mcp_server import handle_jsonrpc

    resp = handle_jsonrpc({
        "id": 7,
        "tool": "research_agent.list_skills",
        "arguments": {},
    })
    assert resp["id"] == 7
    assert "skills" in resp["result"]


def test_mcp_exposes_atomic_context_and_validator_tools(ra, tmp_path):
    from easyicu.research_agent.mcp_server import dispatch, handle_jsonrpc

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({
        "stay_id": [1, 2, 3],
        "sofa2": [0, 2, 4],
        "death": [0, 1, 1],
    }).to_parquet(cohort_path)

    listed = handle_jsonrpc({
        "jsonrpc": "2.0",
        "id": "tools",
        "method": "tools/list",
        "params": {},
    })
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert {
        "research_agent.build_context",
        "research_agent.list_concepts",
        "research_agent.describe_concept",
        "research_agent.audit_cohort",
        "research_agent.run_validator",
        "research_agent.load_concepts",
        "research_agent.extract_concept",
        "research_agent.cross_database_concept_availability",
        "research_agent.bind_evidence",
    } <= names

    concepts = dispatch("research_agent.list_concepts", {
        "cohort_path": str(cohort_path),
        "question": "Is SOFA associated with mortality?",
        "database": "synthetic",
        "target_outcome": "death",
    })
    assert any(c["name"] == "sofa2" for c in concepts["concepts"])

    described = dispatch("research_agent.describe_concept", {
        "cohort_path": str(cohort_path),
        "concept_name": "sofa2",
        "database": "synthetic",
        "target_outcome": "death",
    })
    assert described["concept"]["role"] in {"composite_score", "ordinal_score", "other"}

    audited = dispatch("research_agent.run_validator", {
        "cohort_path": str(cohort_path),
        "validator": "concept_usage_auditor",
        "database": "synthetic",
        "target_outcome": "death",
        "script_text": 'df["sofa2"].mean()',
    })
    assert audited["validator"] == "concept_usage_auditor"
    # Mean of an ordinal score is an advisory caution (impartiality contract),
    # surfaced as a warning rather than a blocking error.
    assert any(
        f["severity"] == "warning"
        and ("ordinal" in f["message"].lower() or "sofa" in f["message"].lower())
        for f in audited["findings"]
    )

    availability = dispatch("research_agent.cross_database_concept_availability", {
        "concepts": ["sofa2", "creatinine"],
        "databases": ["miiv", "eicu"],
    })
    assert availability["availability"]["creatinine"]["miiv"]["concept"] == "crea"
    assert availability["availability"]["sofa2"]["miiv"]["available"] is True


def test_mcp_load_concepts_calls_standardized_easyicu_api(ra, tmp_path, monkeypatch):
    import easyicu
    from easyicu.research_agent.mcp_server import dispatch

    calls = {}

    def fake_load_concepts(**kwargs):
        calls.update(kwargs)
        return pd.DataFrame({
            "stay_id": [1, 2],
            "hr": [80.0, 92.0],
            "map": [70.0, 76.0],
        })

    monkeypatch.setattr(easyicu, "load_concepts", fake_load_concepts, raising=False)
    out_path = tmp_path / "vitals.parquet"
    workdir = tmp_path / "run"
    result = dispatch("research_agent.load_concepts", {
        "concepts": ["hr", "map"],
        "database": "miiv",
        "data_path": str(tmp_path / "miiv"),
        "patient_ids": [1, 2],
        "interval": "1h",
        "aggregate": "mean",
        "output_path": str(out_path),
        "register_evidence": True,
        "workdir": str(workdir),
        "evidence_id": "extracted_vitals",
    })

    assert calls["concepts"] == ["hr", "map"]
    assert calls["database"] == "miiv"
    assert calls["interval"] == "1h"
    assert calls["aggregate"] == "mean"
    assert result["api"] == "easyicu.load_concepts"
    assert result["summary"]["frame"]["rows"] == 2
    assert result["summary"]["frame"]["columns"] == ["stay_id", "hr", "map"]
    assert result["output_paths"] == [str(out_path)]
    assert out_path.exists()
    assert result["evidence"][0]["evidence_id"] == "extracted_vitals"
    assert (workdir / result["evidence"][0]["relative_path"]).exists()


def test_mcp_load_concepts_returns_runtime_availability(ra, tmp_path, monkeypatch):
    import easyicu
    from easyicu.research_agent.mcp_server import dispatch

    def fake_load_concepts(**kwargs):
        sink = kwargs["availability_sink"]
        sink["norepi_rate"] = ConceptAvailabilityRecord(
            concept="norepi_rate",
            database="mimic",
            reason="source_unavailable",
            n_rows=0,
            sources_defined=("inputevents",),
            missing_tables=("inputevents",),
        )
        return pd.DataFrame(columns=["icustay_id", "charttime", "norepi_rate"])

    monkeypatch.setattr(easyicu, "load_concepts", fake_load_concepts, raising=False)

    result = dispatch(
        "research_agent.load_concepts",
        {
            "concepts": ["norepi_rate"],
            "database": "mimic",
            "data_path": str(tmp_path / "mimic"),
        },
    )

    cell = result["availability"]["norepi_rate"]
    assert cell["status"] == "blocked"
    assert cell["available"] is False
    assert cell["reason"] == "source_unavailable"
    assert cell["source_missing_tables"] == ["inputevents"]
    assert cell["structural_unavailable"] is True


def test_mcp_extract_concept_registers_evidence_by_default(ra, tmp_path, monkeypatch):
    import easyicu
    from easyicu.research_agent.mcp_server import dispatch

    calls = {}

    def fake_load_concepts(**kwargs):
        calls.update(kwargs)
        return pd.DataFrame({
            "stay_id": [1, 2],
            "sofa2": [3, 7],
        })

    monkeypatch.setattr(easyicu, "load_concepts", fake_load_concepts, raising=False)
    workdir = tmp_path / "run"
    result = dispatch("research_agent.extract_concept", {
        "concept": "sofa2",
        "database": "miiv",
        "patient_ids": [1, 2],
        "workdir": str(workdir),
        "evidence_id": "sofa2_first24h_extract",
        "aliases": ["sofa2_extract"],
    })

    assert calls["concepts"] == ["sofa2"]
    assert result["api"] == "easyicu.load_concepts"
    assert result["evidence"][0]["evidence_id"] == "sofa2_first24h_extract"
    assert result["output_paths"]
    assert (workdir / result["evidence"][0]["relative_path"]).exists()


def test_mcp_bind_evidence_registers_external_artifact(ra, tmp_path):
    from easyicu.research_agent.mcp_server import dispatch

    workdir = tmp_path / "run"
    result = dispatch("research_agent.bind_evidence", {
        "workdir": str(workdir),
        "kind": "log",
        "text": "external validation summary",
        "filename": "external_validation.txt",
        "evidence_id": "external_validation_summary",
        "aliases": ["external_validation"],
        "metadata": {"agent": "external"},
    })

    record = result["evidence"]
    assert record["evidence_id"] == "external_validation_summary"
    assert record["sha256"]
    assert record["producer"] == "mcp_external_agent"
    assert (workdir / record["relative_path"]).exists()

    aliases = json.loads(
        (workdir / "evidence" / "evidence_aliases.json").read_text(encoding="utf-8")
    )
    assert aliases["external_validation"] == "external_validation_summary"
