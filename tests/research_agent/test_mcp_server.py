"""MCP server protocol surface."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from easyicu.concept.availability_signal import ConceptAvailabilityRecord


def test_mcp_initialize_and_tools_list(ra):
    from easyicu.research_agent.mcp_server import handle_jsonrpc

    init = handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        }
    )
    assert init["result"]["capabilities"]["tools"] == {}
    assert init["result"]["serverInfo"]["name"] == "easyicu-research-agent"

    listed = handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
    )
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert {
        "research_agent.run",
        "research_agent.list_skills",
        "research_agent.read_manifest",
    } <= names


def test_mcp_tools_call_wraps_tool_result_as_content(ra):
    from easyicu.research_agent.mcp_server import handle_jsonrpc

    resp = handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": "skills",
            "method": "tools/call",
            "params": {
                "name": "research_agent.list_skills",
                "arguments": {},
            },
        }
    )
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

    resp = handle_jsonrpc(
        {
            "id": 7,
            "tool": "research_agent.list_skills",
            "arguments": {},
        }
    )
    assert resp["id"] == 7
    assert "skills" in resp["result"]


def test_mcp_exposes_atomic_context_and_validator_tools(ra, tmp_path):
    from easyicu.research_agent.mcp_server import dispatch, handle_jsonrpc

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "sofa2": [0, 2, 4],
            "death": [0, 1, 1],
        }
    ).to_parquet(cohort_path)

    listed = handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": "tools",
            "method": "tools/list",
            "params": {},
        }
    )
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

    concepts = dispatch(
        "research_agent.list_concepts",
        {
            "cohort_path": str(cohort_path),
            "question": "Is SOFA associated with mortality?",
            "database": "synthetic",
            "target_outcome": "death",
        },
    )
    assert any(c["name"] == "sofa2" for c in concepts["concepts"])

    described = dispatch(
        "research_agent.describe_concept",
        {
            "cohort_path": str(cohort_path),
            "concept_name": "sofa2",
            "database": "synthetic",
            "target_outcome": "death",
        },
    )
    assert described["concept"]["role"] in {"composite_score", "ordinal_score", "other"}

    audited = dispatch(
        "research_agent.run_validator",
        {
            "cohort_path": str(cohort_path),
            "validator": "concept_usage_auditor",
            "database": "synthetic",
            "target_outcome": "death",
            "script_text": 'df["sofa2"].mean()',
        },
    )
    assert audited["validator"] == "concept_usage_auditor"
    # Mean of an ordinal score is an advisory caution (impartiality contract),
    # surfaced as a warning rather than a blocking error.
    assert any(
        f["severity"] == "warning"
        and ("ordinal" in f["message"].lower() or "sofa" in f["message"].lower())
        for f in audited["findings"]
    )

    availability = dispatch(
        "research_agent.cross_database_concept_availability",
        {
            "concepts": ["sofa2", "creatinine"],
            "databases": ["miiv", "eicu"],
        },
    )
    assert availability["availability"]["creatinine"]["miiv"]["concept"] == "crea"
    assert availability["availability"]["sofa2"]["miiv"]["available"] is True


def test_mcp_load_concepts_calls_standardized_easyicu_api(ra, tmp_path, monkeypatch):
    import easyicu
    from easyicu.research_agent.mcp_server import dispatch

    calls = {}

    def fake_load_concepts(**kwargs):
        calls.update(kwargs)
        return pd.DataFrame(
            {
                "stay_id": [1, 2],
                "hr": [80.0, 92.0],
                "map": [70.0, 76.0],
            }
        )

    monkeypatch.setattr(easyicu, "load_concepts", fake_load_concepts, raising=False)
    out_path = tmp_path / "vitals.parquet"
    workdir = tmp_path / "run"
    result = dispatch(
        "research_agent.load_concepts",
        {
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
        },
    )

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
        return pd.DataFrame(
            {
                "stay_id": [1, 2],
                "sofa2": [3, 7],
            }
        )

    monkeypatch.setattr(easyicu, "load_concepts", fake_load_concepts, raising=False)
    workdir = tmp_path / "run"
    result = dispatch(
        "research_agent.extract_concept",
        {
            "concept": "sofa2",
            "database": "miiv",
            "patient_ids": [1, 2],
            "workdir": str(workdir),
            "evidence_id": "sofa2_first24h_extract",
            "aliases": ["sofa2_extract"],
        },
    )

    assert calls["concepts"] == ["sofa2"]
    assert result["api"] == "easyicu.load_concepts"
    assert result["evidence"][0]["evidence_id"] == "sofa2_first24h_extract"
    assert result["output_paths"]
    assert (workdir / result["evidence"][0]["relative_path"]).exists()


def test_mcp_bind_evidence_registers_external_artifact(ra, tmp_path):
    from easyicu.research_agent.mcp_server import dispatch

    workdir = tmp_path / "run"
    result = dispatch(
        "research_agent.bind_evidence",
        {
            "workdir": str(workdir),
            "kind": "log",
            "text": "external validation summary",
            "filename": "external_validation.txt",
            "evidence_id": "external_validation_summary",
            "aliases": ["external_validation"],
            "metadata": {"agent": "external"},
        },
    )

    record = result["evidence"]
    assert record["evidence_id"] == "external_validation_summary"
    assert record["sha256"]
    assert record["producer"] == "mcp_external_agent"
    assert (workdir / record["relative_path"]).exists()

    aliases = json.loads(
        (workdir / "evidence" / "evidence_aliases.json").read_text(encoding="utf-8")
    )
    assert aliases["external_validation"] == "external_validation_summary"


def test_mcp_run_returns_clear_error_without_llm_configuration(
    ra, tmp_path, monkeypatch
):
    from easyicu.research_agent.mcp_server import dispatch

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    result = dispatch(
        "research_agent.run",
        {
            "question": "Inspect the cohort.",
            "cohort_path": str(tmp_path / "cohort.parquet"),
        },
    )

    assert result["error_code"] == "llm_configuration_required"
    assert "explicit model" in result["error"]
    assert "requires an explicit `llm=`" not in result["error"]


def test_mcp_run_constructs_explicit_llm(ra, tmp_path, monkeypatch):
    import easyicu.research_agent.mcp_server as mcp

    cohort = tmp_path / "cohort.parquet"
    cohort.write_bytes(b"fixture")
    seen = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen["llm_kwargs"] = kwargs

    class FakePipeline:
        def __init__(self, *, workdir, llm):
            seen["workdir"] = workdir
            seen["llm"] = llm

        def run(self, *, cohort, **kwargs):
            seen["cohort"] = cohort
            seen["run_kwargs"] = kwargs
            return SimpleNamespace(model_dump=lambda: {"status": "ok"})

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        mcp,
        "build_provider_client",
        lambda **kwargs: FakeClient(**kwargs),
    )
    monkeypatch.setattr(mcp, "ResearchAgentPipeline", FakePipeline)

    result = mcp.dispatch(
        "research_agent.run",
        {
            "question": "Inspect the cohort.",
            "cohort_path": str(cohort),
            "workdir": str(tmp_path / "run"),
            "provider": "openai",
            "model": "test-model",
        },
    )

    assert result == {"status": "ok"}
    assert seen["llm_kwargs"]["model"] == "test-model"
    assert seen["cohort"] == str(cohort)
    assert seen["run_kwargs"]["question"] == "Inspect the cohort."


def test_mcp_run_rejects_key_exfiltration_base_url(ra, tmp_path, monkeypatch):
    import easyicu.research_agent.mcp_server as mcp

    constructed = []
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leave-server")
    monkeypatch.setattr(
        mcp,
        "OpenAIClient",
        lambda **kwargs: constructed.append(kwargs),
    )

    result = mcp.dispatch(
        "research_agent.run",
        {
            "question": "Inspect the cohort.",
            "cohort_path": str(tmp_path / "cohort.parquet"),
            "provider": "openai",
            "model": "test-model",
            "base_url": "https://attacker.example/collect?next=localhost",
        },
    )

    assert result["error_code"] == "llm_configuration_invalid"
    assert "loopback" in result["error"]
    assert constructed == []


def test_mcp_run_loopback_override_does_not_forward_environment_key(
    ra, tmp_path, monkeypatch
):
    import easyicu.research_agent.mcp_server as mcp

    seen = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    class FakePipeline:
        def __init__(self, *, workdir, llm):
            pass

        def run(self, *, cohort, **kwargs):
            return SimpleNamespace(model_dump=lambda: {"status": "ok"})

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-loopback")
    monkeypatch.setattr(mcp, "OpenAIClient", FakeClient)
    monkeypatch.setattr(mcp, "ResearchAgentPipeline", FakePipeline)

    result = mcp.dispatch(
        "research_agent.run",
        {
            "question": "Inspect the cohort.",
            "cohort_path": str(tmp_path / "cohort.parquet"),
            "provider": "openai",
            "model": "test-model",
            "base_url": "http://127.0.0.1:8787/v1",
        },
    )

    assert result == {"status": "ok"}
    assert seen["api_key"] == "easyicu-local-noauth"
    assert seen["api_key"] != "must-not-reach-loopback"


def test_mcp_read_manifest_rejects_path_traversal(ra, tmp_path):
    from easyicu.research_agent.mcp_server import dispatch

    workdir = tmp_path / "runs"
    outside = tmp_path / "outside"
    outside.mkdir(parents=True)
    (outside / "manifest.json").write_text(
        json.dumps({"sentinel": True}), encoding="utf-8"
    )

    result = dispatch(
        "research_agent.read_manifest",
        {"workdir": str(workdir), "run_id": "../outside"},
    )

    assert "single safe path component" in result["error"]
    assert "sentinel" not in result


def test_mcp_bind_evidence_rejects_path_escape(ra, tmp_path):
    from easyicu.research_agent.mcp_server import dispatch

    result = dispatch(
        "research_agent.bind_evidence",
        {
            "workdir": str(tmp_path / "run"),
            "kind": "log",
            "text": "probe",
            "filename": "../../escaped.txt",
            "evidence_id": "../../escaped",
        },
    )

    assert "single safe path component" in result["error"]
    assert not (tmp_path / "escaped.txt").exists()
