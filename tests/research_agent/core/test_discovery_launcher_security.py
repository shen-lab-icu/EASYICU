"""Security and provenance regressions for MCP HTTP and discovery launcher."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.mcp_policy import MCP_ALLOWED_ROOTS_ENV, MCP_SCOPES_ENV


def test_mcp_loopback_environment_url_never_forwards_provider_secrets(
    ra, tmp_path, monkeypatch
):
    import easyicu.research_agent.mcp_server as mcp
    from easyicu.research_agent.providers.factory import (
        build_provider_client as real_builder,
    )

    # Every MCP filesystem argument is confined to a root configured at
    # startup; declare tmp_path so this test still exercises the credential
    # boundary it is about rather than the outer path confinement. Running a
    # pipeline is likewise an explicitly-granted scope now, so grant it here
    # the way a deployment would.
    monkeypatch.setenv(MCP_ALLOWED_ROOTS_ENV, str(tmp_path))
    monkeypatch.setenv(MCP_SCOPES_ENV, "metadata,run_pipeline")

    seen = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    class FakePipeline:
        def __init__(self, *, workdir, llm):
            pass

        @classmethod
        def from_config(cls, config, *, services):
            return cls(workdir=config.workdir, llm=services.llm)

        def run(self, *, cohort, **kwargs):
            return SimpleNamespace(model_dump=lambda: {"status": "ok"})

    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8787/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "paid-openai-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "paid-openrouter-secret")
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
            "model": "local-model",
        },
    )

    assert result == {"status": "ok"}
    assert seen["base_url"] == "http://127.0.0.1:8787/v1"
    assert seen["api_key"] == "easyicu-local-noauth"
    assert seen["api_key"] not in {"paid-openai-secret", "paid-openrouter-secret"}


def test_discovery_launcher_loopback_never_forwards_provider_secrets(monkeypatch):
    import tools.run_discovery_to_manuscript as launcher
    from easyicu.research_agent.providers.factory import (
        build_provider_client as real_builder,
    )

    seen = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8787/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "paid-openai-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "paid-openrouter-secret")
    monkeypatch.setattr(
        launcher,
        "build_provider_client",
        lambda **kwargs: real_builder(client_cls=FakeClient, **kwargs),
    )

    launcher._build_data_foundation_llm(
        provider="openai",
        model="local-model",
        request_timeout=12.0,
    )

    assert seen["base_url"] == "http://localhost:8787/v1"
    assert seen["api_key"] == "easyicu-local-noauth"
    assert seen["api_key"] not in {"paid-openai-secret", "paid-openrouter-secret"}


def test_discovery_outcome_materialisation_uses_only_frozen_handoff_target():
    import tools.run_discovery_to_manuscript as launcher

    assert launcher._outcome_concepts_for_handoff(
        handoff_target="aki", requested=None
    ) == ("aki",)
    assert launcher._outcome_concepts_for_handoff(
        handoff_target="aki", requested="AKI"
    ) == ("aki",)
    with pytest.raises(SystemExit, match="frozen handoff target 'aki'"):
        launcher._outcome_concepts_for_handoff(handoff_target="aki", requested="death")
    with pytest.raises(SystemExit, match="exactly"):
        launcher._outcome_concepts_for_handoff(
            handoff_target="aki", requested="aki,death"
        )
    assert (
        launcher._outcome_concepts_for_handoff(handoff_target=None, requested=None)
        == ()
    )
    with pytest.raises(SystemExit, match="outcome-free concept-set handoff"):
        launcher._outcome_concepts_for_handoff(
            handoff_target=None,
            requested="death",
        )


def test_discovery_handoff_registration_blocks_existing_id_hash_mismatch(tmp_path):
    import tools.run_discovery_to_manuscript as launcher
    from easyicu.research_agent.authority.evidence_store import EvidenceStore

    store = EvidenceStore(tmp_path / "run")
    source = tmp_path / "handoff.json"
    source.write_text('{"target_outcome":"aki"}', encoding="utf-8")
    original = launcher._register_file_exact(
        store,
        source_path=source,
        kind="log",
        description="handoff",
        evidence_id="discovery_handoff",
        producer="discovery_launcher",
        generation_mode="human_confirmed",
    )
    copied_path = store.root / original.relative_path
    source.write_text('{"target_outcome":"death"}', encoding="utf-8")

    with pytest.raises(ValueError, match="Evidence id collision"):
        launcher._register_file_exact(
            store,
            source_path=source,
            kind="log",
            description="handoff",
            evidence_id="discovery_handoff",
            producer="discovery_launcher",
            generation_mode="human_confirmed",
        )

    assert copied_path.read_text(encoding="utf-8") == '{"target_outcome":"aki"}'


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_discovery_story_exports_receive_closed_provenance(tmp_path):
    import tools.run_discovery_to_manuscript as launcher
    from easyicu.research_agent.authority.evidence_store import EvidenceStore

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for filename in ("run_status.json", "evidence_audit.json", "numeric_audit.json"):
        _write_json(run_dir / filename, {"source": filename})
    evidence = EvidenceStore(run_dir)
    launcher._register_story_source_records(evidence=evidence, run_dir=run_dir)

    handoff = run_dir / "discovery_handoff.json"
    _write_json(handoff, {"target_outcome": "aki"})
    launcher._register_file_exact(
        evidence,
        source_path=handoff,
        kind="log",
        description="handoff",
        evidence_id="discovery_handoff",
        producer="discovery_launcher",
        generation_mode="human_confirmed",
    )

    figure_dir = run_dir / "publication_figures"
    figure_dir.mkdir()
    contract_path = figure_dir / "easyicu_discovery_story.figure_contract.json"
    _write_json(
        contract_path,
        {
            "figure_id": "easyicu_discovery_story",
            "source_data": [
                "discovery_handoff",
                "run_status",
                "evidence_audit",
                "numeric_audit",
            ],
            "panels": [
                {
                    "panel_id": "C",
                    "metadata": {"story_role": "primary_result"},
                    "evidence_ids": ["run_status"],
                }
            ],
        },
    )
    svg_path = figure_dir / "easyicu_discovery_story.svg"
    png_path = figure_dir / "easyicu_discovery_story.png"
    svg_path.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8"
    )
    png_path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
            "+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
    )

    records = launcher._register_story_figure_provenance(
        evidence=evidence,
        run_dir=run_dir,
        paths={"contract": contract_path, "svg": svg_path, "png": png_path},
    )

    assert records["script"].kind == "code"
    assert records["contract"].metadata["figure_id"] == "easyicu_discovery_story"
    for extension in ("svg", "png"):
        record = records[extension]
        assert record.kind == "figure"
        assert record.script_evidence_id == records["script"].evidence_id
        assert evidence.get(record.script_evidence_id).kind == "code"
        assert (
            record.metadata["contract_evidence_id"] == records["contract"].evidence_id
        )
        assert set(record.metadata["source_evidence_ids"]) <= set(record.inputs)
        assert record.metadata["inputs"] == record.inputs
