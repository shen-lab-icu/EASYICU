"""MCP server protocol surface."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept.availability_signal import ConceptAvailabilityRecord
from easyicu.research_agent.mcp_policy import (
    MCP_ALLOWED_ROOTS_ENV,
    MCP_SCOPES_ENV,
    SCOPE_READ_PATIENT_DATA,
)


@pytest.fixture(autouse=True)
def _mcp_roots(tmp_path, monkeypatch):
    """Grant the test's tmp_path as an MCP root, and the working scopes.

    Every filesystem argument is confined to a root configured at startup, so
    without this an operator-style declaration the tools would refuse to touch
    ``tmp_path`` at all. Declaring it here keeps each test exercising the guard
    it is actually about rather than the outer confinement.

    The process default is ``metadata`` only: running a pipeline, writing
    artefacts and binding evidence are authorities an operator grants
    explicitly, so these tests grant them the same way a deployment would.
    ``read_patient_data`` / ``read_internal_context`` stay ungranted — the
    tests that need those opt in individually.
    """

    monkeypatch.setenv(MCP_ALLOWED_ROOTS_ENV, str(tmp_path))
    monkeypatch.setenv(
        MCP_SCOPES_ENV, "metadata,run_pipeline,write_artifacts,bind_evidence"
    )


def _prepared_export(root):
    root.mkdir(parents=True)
    pd.DataFrame(
        {
            "stay_id": [101, 102],
            "charttime": [0.0, 1.0],
            "lact": [1.2, 2.4],
            "sofa2": [3, 6],
        }
    ).to_parquet(root / "labs.parquet", index=False)
    pd.DataFrame(
        {"stay_id": [101, 102], "death": [0, 1]}
    ).to_parquet(root / "outcome.parquet", index=False)
    (root / "_manifest.json").write_text(
        json.dumps(
            {
                "database": "miiv",
                "format": "parquet",
                "concept_selection": {
                    "modules": {
                        "labs": ["lact", "sofa2"],
                        "outcome": ["death"],
                    }
                },
                "files": [
                    {
                        "file": "labs.parquet",
                        "module": "labs",
                        "concepts": 2,
                        "concept_ids": ["lact", "sofa2"],
                        "rows": 2,
                    },
                    {
                        "file": "outcome.parquet",
                        "module": "outcome",
                        "concepts": 1,
                        "concept_ids": ["death"],
                        "rows": 2,
                    },
                ],
                "feature_definitions": {"included": False},
            }
        ),
        encoding="utf-8",
    )
    return root


def test_mcp_python_dispatch_remains_available(ra):
    from easyicu.research_agent.mcp_server import dispatch

    result = dispatch("research_agent.list_skills", {})

    assert "skills" in result


def test_mcp_run_schema_and_server_reject_additional_properties(
    ra, tmp_path, monkeypatch
):
    import easyicu.research_agent.mcp_server as mcp

    schema = next(
        item for item in mcp.TOOL_SCHEMAS if item["name"] == "research_agent.run"
    )["inputSchema"]
    assert schema["additionalProperties"] is False

    constructed = []
    monkeypatch.setattr(
        mcp,
        "_build_run_llm",
        lambda _args: constructed.append(True) or (object(), None),
    )
    result = mcp.dispatch(
        "research_agent.run",
        {
            "question": "q",
            "cohort_path": str(tmp_path / "cohort.parquet"),
            "model": "m",
            "trajectory_path": "/outside/forwarded.parquet",
        },
    )

    assert result["error_code"] == "invalid_argument"
    assert "additional properties" in result["error"]
    assert constructed == []


def test_every_mcp_tool_schema_is_closed_and_direct_dispatch_rejects_unknown_args(
    ra,
) -> None:
    from easyicu.research_agent.mcp_server import TOOL_SCHEMAS, dispatch

    assert TOOL_SCHEMAS
    for tool in TOOL_SCHEMAS:
        assert tool["inputSchema"]["additionalProperties"] is False
        result = dispatch(tool["name"], {"__undeclared__": True})
        assert result["error_code"] == "invalid_argument", tool["name"]
        assert "additional properties" in result["error"]


def test_extraction_requires_explicit_patient_data_scope(
    ra, monkeypatch
) -> None:
    import easyicu
    from easyicu.research_agent.mcp_server import dispatch

    calls = []
    monkeypatch.setenv(MCP_SCOPES_ENV, "metadata")
    monkeypatch.setattr(
        easyicu,
        "load_concepts",
        lambda **_kwargs: calls.append(True),
        raising=False,
    )

    for tool, arguments in (
        ("research_agent.load_concepts", {"concepts": ["hr"]}),
        ("research_agent.extract_concept", {"concept": "hr"}),
    ):
        result = dispatch(tool, arguments)
        assert result["error_code"] == "scope_not_granted"
    assert calls == []


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("chunk_size", 0),
        ("chunk_size", 10_000_001),
        ("batch_size", 0),
        ("batch_size", 1_000_001),
        ("parallel_workers", 0),
        ("parallel_workers", 65),
        ("concept_workers", 0),
        ("concept_workers", 65),
    ],
)
def test_extraction_worker_and_chunk_arguments_are_bounded(
    ra, monkeypatch, argument, value
) -> None:
    import easyicu
    from easyicu.research_agent.mcp_server import dispatch

    calls = []
    monkeypatch.setenv(MCP_SCOPES_ENV, f"metadata,{SCOPE_READ_PATIENT_DATA}")
    monkeypatch.setattr(
        easyicu,
        "load_concepts",
        lambda **_kwargs: calls.append(True),
        raising=False,
    )

    result = dispatch(
        "research_agent.load_concepts",
        {"concepts": ["hr"], argument: value},
    )

    assert result["error_code"] == "invalid_argument"
    assert calls == []


def test_mcp_exposes_atomic_context_and_validator_tools(ra, tmp_path):
    from easyicu.research_agent.mcp_server import TOOL_SCHEMAS, dispatch

    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "sofa2": [0, 2, 4],
            "death": [0, 1, 1],
        }
    ).to_parquet(cohort_path)

    names = {tool["name"] for tool in TOOL_SCHEMAS}
    assert {
        "research_agent.build_context",
        "research_agent.list_concepts",
        "research_agent.describe_concept",
        "research_agent.list_export_concepts",
        "research_agent.assess_export_coverage",
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
    # surfaced as a warning rather than a blocking error. The caller identifies
    # it from the projected detail: ``message`` is an interpolated sentence and
    # is not disclosed over MCP.
    assert not any("message" in f for f in audited["findings"])
    assert any(
        f["severity"] == "warning"
        and f.get("detail", {}).get("column") == "sofa2"
        and f.get("detail", {}).get("function") == "mean"
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


def test_mcp_projects_pre_materialization_export_catalog_and_coverage(
    ra, tmp_path, monkeypatch
):
    from easyicu.research_agent.mcp_server import dispatch

    export_dir = _prepared_export(tmp_path / "prepared_export")
    monkeypatch.setenv(MCP_SCOPES_ENV, "metadata")

    catalog = dispatch(
        "research_agent.list_export_concepts",
        {
            "export_dir": str(export_dir),
            "modules": ["labs"],
            "query": "lact",
            "limit": 10,
        },
    )

    assert catalog["schema_version"] == "easyicu.mcp-export-catalog/1"
    assert catalog["catalog_concept_count"] == 3
    assert [row["concept_id"] for row in catalog["concepts"]] == ["lact"]
    assert catalog["concepts"][0]["module"] == "labs"
    assert catalog["source"]["path_returned"] is False
    assert catalog["privacy"] == {
        "patient_rows_returned": False,
        "host_path_returned": False,
        "raw_sql_returned": False,
    }

    coverage = dispatch(
        "research_agent.assess_export_coverage",
        {
            "export_dir": str(export_dir),
            "concepts": ["lact", "death", "troponin"],
        },
    )

    assert coverage["schema_version"] == "easyicu.mcp-export-coverage/1"
    assert coverage["available"] == ["lact", "death"]
    assert coverage["missing"] == ["troponin"]
    assert coverage["sufficient"] is False
    assert "re-extract" in coverage["advice"][0].casefold()
    assert "does not establish" in coverage["claim_boundary"]
    rendered = json.dumps({"catalog": catalog, "coverage": coverage})
    assert str(export_dir) not in rendered
    assert '"stay_id"' not in rendered
    assert '"charttime"' not in rendered
    assert '"rows"' not in rendered


def test_mcp_export_coverage_refuses_empty_requests(ra, tmp_path):
    from easyicu.research_agent.mcp_server import dispatch

    export_dir = _prepared_export(tmp_path / "prepared_export")
    result = dispatch(
        "research_agent.assess_export_coverage",
        {"export_dir": str(export_dir), "concepts": []},
    )

    assert result["error_code"] == "invalid_argument"
    assert "at least one" in result["error"]


def test_mcp_export_catalog_remains_confined_to_allowed_roots(
    ra, tmp_path
):
    from easyicu.research_agent.mcp_server import dispatch

    result = dispatch(
        "research_agent.list_export_concepts",
        {"export_dir": str(tmp_path.parent / "outside_export")},
    )

    assert result["error_code"] == "path_not_allowed"


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
    monkeypatch.setenv(
        MCP_SCOPES_ENV,
        f"metadata,write_artifacts,bind_evidence,{SCOPE_READ_PATIENT_DATA}",
    )
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
    monkeypatch.setenv(MCP_SCOPES_ENV, f"metadata,{SCOPE_READ_PATIENT_DATA}")

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
    monkeypatch.setenv(
        MCP_SCOPES_ENV,
        f"metadata,write_artifacts,bind_evidence,{SCOPE_READ_PATIENT_DATA}",
    )
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
            "workdir": str(tmp_path / "run"),
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
        @classmethod
        def from_config(cls, config, *, services):
            seen["workdir"] = config.workdir
            seen["llm"] = services.llm
            seen["hard_stop"] = services.provider_hard_stop
            seen["hard_stop_limit"] = config.max_provider_attempts_per_run
            return cls()

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
    assert seen["hard_stop"].ledger.path.is_file()
    assert seen["hard_stop_limit"] == 96


def test_mcp_run_accepts_account_cli_default_without_api_credentials(
    ra,
    tmp_path,
    monkeypatch,
):
    import easyicu.research_agent.mcp_server as mcp

    cohort = tmp_path / "cohort.parquet"
    cohort.write_bytes(b"fixture")
    seen = {}
    account_client = object()

    def build_account(**kwargs):
        seen["builder"] = kwargs
        return SimpleNamespace(client=account_client)

    class FakeHardStop:
        def finish(self, **_kwargs):
            return None

    def hard_stop(*, workdir, provider_identity):
        seen["provider_identity"] = provider_identity
        return FakeHardStop()

    class FakePipeline:
        @classmethod
        def from_config(cls, _config, *, services):
            seen["llm"] = services.llm
            return cls()

        def run(self, **_kwargs):
            return SimpleNamespace(model_dump=lambda: {"status": "ok"})

    monkeypatch.setattr(mcp, "build_llm_client", build_account)
    monkeypatch.setattr(mcp, "_mcp_provider_hard_stop", hard_stop)
    monkeypatch.setattr(mcp, "ResearchAgentPipeline", FakePipeline)

    result = mcp.dispatch(
        "research_agent.run",
        {
            "question": "Inspect the synthetic cohort.",
            "cohort_path": str(cohort),
            "workdir": str(tmp_path / "run"),
            "provider": "codex",
        },
    )

    assert result == {"status": "ok"}
    assert seen["builder"]["prefer"] == "codex"
    assert seen["builder"]["model"] is None
    assert seen["builder"]["ladder"] == ["codex"]
    assert seen["builder"]["allow_mock"] is False
    assert seen["llm"] is account_client
    assert seen["provider_identity"] == {
        "provider": "codex-cli",
        "model": "cli-default",
        "base_url": "cli://codex",
    }


def test_mcp_write_scope_is_checked_before_concept_extraction(
    ra, tmp_path, monkeypatch
):
    import easyicu
    import easyicu.research_agent.mcp_server as mcp

    calls = []
    monkeypatch.setenv(MCP_SCOPES_ENV, "metadata")
    monkeypatch.setattr(
        easyicu,
        "load_concepts",
        lambda **kwargs: calls.append(kwargs) or pd.DataFrame({"x": [1]}),
        raising=False,
    )

    result = mcp.dispatch(
        "research_agent.load_concepts",
        {
            "concepts": ["lact"],
            "output_path": str(tmp_path / "must-not-exist.parquet"),
        },
    )

    assert result["error_code"] == "scope_not_granted"
    assert calls == []
    assert not (tmp_path / "must-not-exist.parquet").exists()


def test_mcp_rejects_forwarded_dictionary_paths_outside_roots_before_extraction(
    ra, tmp_path, monkeypatch
):
    import easyicu
    import easyicu.research_agent.mcp_server as mcp

    calls = []
    monkeypatch.setattr(
        easyicu,
        "load_concepts",
        lambda **kwargs: calls.append(kwargs) or pd.DataFrame({"x": [1]}),
        raising=False,
    )
    monkeypatch.setenv(MCP_SCOPES_ENV, f"metadata,{SCOPE_READ_PATIENT_DATA}")

    result = mcp.dispatch(
        "research_agent.load_concepts",
        {
            "concepts": ["lact"],
            "dict_path": [str(tmp_path / "allowed.json"), "/outside/dict.json"],
        },
    )

    assert result["error_code"] == "path_not_allowed"
    assert calls == []


def test_mcp_run_returns_explicit_unresumable_pending(ra, tmp_path, monkeypatch):
    import easyicu.research_agent.mcp_server as mcp
    from easyicu.research_agent.orchestration.workflow import (
        HumanReviewPending,
        HumanReviewRequest,
    )

    cohort = tmp_path / "cohort.parquet"
    cohort.write_bytes(b"fixture")
    request = HumanReviewRequest.create(
        kind="capability_request",
        summary="Approve the exact capability-bound plan.",
        authority_sha256="a" * 64,
        payload={"reason": "capability_review_required"},
    )
    pending = HumanReviewPending(
        run_id="run_mcp_review",
        thread_id="run_mcp_review",
        run_dir=str(tmp_path / "run_mcp_review"),
        requests=(request,),
    )

    class FakePipeline:
        @classmethod
        def from_config(cls, _config, *, services):
            assert services.provider_hard_stop is not None
            return cls()

        def run(self, **_kwargs):
            return pending

    monkeypatch.setattr(mcp, "_build_run_llm", lambda _args: (object(), None))
    monkeypatch.setattr(mcp, "ResearchAgentPipeline", FakePipeline)

    result = mcp.dispatch(
        "research_agent.run",
        {
            "question": "Inspect the cohort.",
            "cohort_path": str(cohort),
            "workdir": str(tmp_path / "runs"),
            "provider": "openai",
            "model": "test-model",
        },
    )

    assert result["status"] == "human_review_pending"
    assert result["terminal"] is False
    assert result["resume_scope"] == "same_process"
    assert result["resumable_via_mcp"] is False
    assert result["external_resume_supported"] is False
    assert "does not retain the Pipeline instance" in result["message"]
    assert result["requests"][0]["review_id"] == request.review_id


def test_mcp_run_rejects_key_exfiltration_base_url(ra, tmp_path, monkeypatch):
    import easyicu.research_agent.mcp_server as mcp

    constructed = []
    real_builder = mcp.build_provider_client

    class RecordingClient:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leave-server")
    monkeypatch.setattr(
        mcp,
        "build_provider_client",
        lambda **kwargs: real_builder(client_cls=RecordingClient, **kwargs),
    )

    result = mcp.dispatch(
        "research_agent.run",
        {
            "question": "Inspect the cohort.",
            "cohort_path": str(tmp_path / "cohort.parquet"),
            "workdir": str(tmp_path / "run"),
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
    real_builder = mcp.build_provider_client

    class FakeClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    class FakePipeline:
        @classmethod
        def from_config(cls, _config, *, services):
            assert services.provider_hard_stop is not None
            return cls()

        def run(self, *, cohort, **kwargs):
            return SimpleNamespace(model_dump=lambda: {"status": "ok"})

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-loopback")
    monkeypatch.setattr(
        mcp,
        "build_provider_client",
        lambda **kwargs: real_builder(client_cls=FakeClient, **kwargs),
    )
    monkeypatch.setattr(mcp, "ResearchAgentPipeline", FakePipeline)

    result = mcp.dispatch(
        "research_agent.run",
        {
            "question": "Inspect the cohort.",
            "cohort_path": str(tmp_path / "cohort.parquet"),
            "workdir": str(tmp_path / "run"),
            "provider": "openai",
            "model": "test-model",
            "base_url": "http://127.0.0.1:8787/v1",
        },
    )

    assert result == {"status": "ok"}
    assert seen["api_key"] == "easyicu-local-noauth"
    assert seen["api_key"] != "must-not-reach-loopback"


def test_mcp_multi_concept_outputs_are_collision_safe_and_mapped(
    ra, tmp_path, monkeypatch
):
    import easyicu
    import easyicu.research_agent.mcp_server as mcp

    frames = {
        "a/b": pd.DataFrame({"value": [1]}),
        "a:b": pd.DataFrame({"value": [2]}),
        "a b": pd.DataFrame({"value": [3]}),
    }
    monkeypatch.setattr(
        easyicu,
        "load_concepts",
        lambda **_kwargs: frames,
        raising=False,
    )
    monkeypatch.setenv(
        MCP_SCOPES_ENV,
        f"metadata,write_artifacts,bind_evidence,{SCOPE_READ_PATIENT_DATA}",
    )

    result = mcp.dispatch(
        "research_agent.load_concepts",
        {
            "concepts": list(frames),
            "database": "synthetic",
            "output_path": str(tmp_path / "exports"),
            "register_evidence": True,
            "workdir": str(tmp_path / "run"),
            "metadata": {
                "logical_concept_name": "forged",
                "physical_filename": "forged.parquet",
            },
        },
    )

    assert len(result["output_paths"]) == 3
    assert len(set(result["output_paths"])) == 3
    output_by_name = {
        item["logical_concept_names"][0]: item for item in result["outputs"]
    }
    for logical_name, frame in frames.items():
        item = output_by_name[logical_name]
        assert item["physical_filename"] == mcp._concept_output_filename(logical_name)
        assert pd.read_parquet(item["path"]).equals(frame)
    evidence_by_name = {
        item["metadata"]["logical_concept_name"]: item for item in result["evidence"]
    }
    assert set(evidence_by_name) == set(frames)
    for logical_name, record in evidence_by_name.items():
        assert (
            record["metadata"]["physical_filename"]
            == output_by_name[logical_name]["physical_filename"]
        )


def test_mcp_concept_output_refuses_to_overwrite_existing_file(
    ra, tmp_path, monkeypatch
):
    import easyicu
    import easyicu.research_agent.mcp_server as mcp

    monkeypatch.setattr(
        easyicu,
        "load_concepts",
        lambda **_kwargs: pd.DataFrame({"value": [1]}),
        raising=False,
    )
    monkeypatch.setenv(
        MCP_SCOPES_ENV,
        f"metadata,write_artifacts,{SCOPE_READ_PATIENT_DATA}",
    )
    target = tmp_path / "existing.parquet"
    sentinel = b"do-not-overwrite"
    target.write_bytes(sentinel)

    result = mcp.dispatch(
        "research_agent.load_concepts",
        {
            "concepts": ["lactate"],
            "database": "synthetic",
            "output_path": str(target),
        },
    )

    assert result["error_code"] == "output_exists"
    assert "choose a new output_path" in result["error"]
    assert target.read_bytes() == sentinel


def test_mcp_multi_output_preflights_every_destination_before_writing(
    ra, tmp_path, monkeypatch
):
    import easyicu
    import easyicu.research_agent.mcp_server as mcp

    frames = {
        "first/concept": pd.DataFrame({"value": [1]}),
        "second/concept": pd.DataFrame({"value": [2]}),
    }
    monkeypatch.setattr(
        easyicu,
        "load_concepts",
        lambda **_kwargs: frames,
        raising=False,
    )
    monkeypatch.setenv(
        MCP_SCOPES_ENV,
        f"metadata,write_artifacts,{SCOPE_READ_PATIENT_DATA}",
    )
    output_dir = tmp_path / "exports"
    output_dir.mkdir()
    occupied = output_dir / mcp._concept_output_filename("second/concept")
    occupied.write_bytes(b"occupied")
    first = output_dir / mcp._concept_output_filename("first/concept")

    result = mcp.dispatch(
        "research_agent.load_concepts",
        {
            "concepts": list(frames),
            "database": "synthetic",
            "output_path": str(output_dir),
        },
    )

    assert result["error_code"] == "output_exists"
    assert occupied.read_bytes() == b"occupied"
    assert not first.exists()


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
