"""Synthetic regression checks for the desktop DUA egress boundary."""

from types import SimpleNamespace
import json
import re
import sys

import pytest

from easyicu.research_agent.gates.publication_disclosure import SMALL_CELL_BELOW
from easyicu.research_agent.research_context.outbound import (
    outbound_safe_context_payload,
    project_outbound_step_summary,
)
from easyicu.research_agent.reporting.writer_evidence import _summarise_table_one_rows


def _context_and_step(ra):
    return ra.ResearchContext(
        research_question="Describe a synthetic exposure.",
        cohort=ra.CohortDescriptor(cohort_name="fixture", database="synthetic", n_stays=20, n_patients=20),
        variables=[ra.ConceptDescriptor(name="exposure", role="lab", dtype="float64"),
                   ra.ConceptDescriptor(name="outcome", role="outcome", dtype="int64")],
        primary_exposure="exposure", target_outcome="outcome",
    ), ra.AnalysisStep(step_id="summary", intent="Describe synthetic aggregates.",
                       inputs=["exposure", "outcome"], expected_outputs=["table:result"], method="descriptive_summary")


def _authority():
    from easyicu.research_agent.repairs.reasons import RepairPromptAuthority, RepairReason

    return RepairPromptAuthority.create(typed_ticket=[{
        "validator": "mechanical_code_preflight", "reason": RepairReason.UNBOUND_LOCAL.value,
        "occurrence_count": 1, "detail": {"reason": "unbound_local", "line": 2},
    }])


def _all_prompt_text(client):
    return "\n".join(message.content for messages, _ in client.calls for message in messages)


@pytest.mark.parametrize("count", range(SMALL_CELL_BELOW + 1))
def test_patient_counts_are_projected_without_changing_measurements(ra, count):
    expected = "suppressed" if 0 < count < SMALL_CELL_BELOW else count
    context = ra.ResearchContext(
        research_question="Describe a synthetic exposure.",
        cohort=ra.CohortDescriptor(
            cohort_name="fixture", database="synthetic", n_stays=count, n_patients=count,
        ),
        variables=[],
    )
    cohort = outbound_safe_context_payload(context)["cohort"]
    assert cohort["n_stays"] == expected
    assert cohort["n_patients"] == expected
    summary = project_outbound_step_summary({
        "count": count, "event_count": count, "n": count,
        "estimate": 2.5, "p_value": 0.04,
    })
    assert summary == {
        "count": expected, "event_count": expected, "n": expected,
        "estimate": 2.5, "p_value": 0.04,
    }


def test_writer_drops_category_literals_and_suppresses_counts():
    row = _summarise_table_one_rows([
        {"variable": "age", "n": 4, "most_common": "PRIVATE_CATEGORY", "most_common_n": 3},
    ])
    assert "PRIVATE_CATEGORY" not in str(row)
    assert row["age_n"] == "suppressed"
    assert "age_most_common_n" not in row
    assert "age_most_common_share" not in row


def test_literature_tallies_do_not_exempt_cohort_counts_with_the_same_name():
    from easyicu.research_agent.research_context.outbound import project_outbound_counts

    assert project_outbound_counts({
        "cohort": {"result_count": 3}, "study_literature_authority": {"result_count": 3},
    }) == {
        "cohort": {"result_count": "suppressed"}, "study_literature_authority": {"result_count": 3},
    }


def test_count_projection_preserves_planning_and_artifact_metadata():
    from easyicu.research_agent.research_context.outbound import project_outbound_counts

    metadata = {"n_unique": 3, "plan_step_count": 3, "figure_count": 3, "withheld_review_claim_count": 1}
    assert project_outbound_counts({**metadata, "count": 3}) == {**metadata, "count": "suppressed"}


def test_writer_retains_large_category_share_and_drops_unknown_summary_text(tmp_path):
    from easyicu.research_agent.reporting.writer_evidence import _render_writer_evidence_digest_v2

    row = _summarise_table_one_rows([
        {"variable": "age", "n": 100, "most_common": "PRIVATE_CATEGORY", "most_common_n": 20},
    ])
    assert row == {"age_n": 100, "age_most_common_share": 0.2}
    digest = _render_writer_evidence_digest_v2([
        {"step_id": "synthetic", "status": "ok", "step_summary": {
            "n_total": 3, "count": 4, "unknown_notes": "PRIVATE_TEXT",
            "most_common": "PRIVATE_CATEGORY", "most_common_n": 2,
        }},
    ], run_dir=tmp_path)
    assert "PRIVATE_TEXT" not in digest
    assert "PRIVATE_CATEGORY" not in digest
    assert '"n_total": "suppressed"' in digest
    assert "count=suppressed" in digest


def test_help_lists_egress_content_without_absolute_patient_row_promise():
    from pathlib import Path

    help_owner = Path(__file__).resolve().parents[3] / "src/easyicu/webserver/static/js/screens-help.js"
    source = help_owner.read_text()
    assert "Patient-level rows are never sent to the model" not in source
    assert "physionet_llm_data_handling_declaration" in source
    assert "aggregate_statistics_with_small_cell_suppression" in source


@pytest.mark.parametrize("file", ["sample.csv", "data/notes.md", "outputs/notes.md"])
def test_desktop_workspace_refuses_data_reads(tmp_path, monkeypatch, file):
    from easyicu.webserver.pi_copilot.contracts import PiCopilotError
    from easyicu.webserver.pi_copilot.workspace import ProjectWorkspace

    monkeypatch.delenv("EASYICU_HOSTED", raising=False)
    workspace = ProjectWorkspace(tmp_path / "workspace")
    workspace.write_file("synthetic-project", file, "synthetic fixture")
    with pytest.raises(PiCopilotError):
        workspace.read_file("synthetic-project", file)


@pytest.mark.parametrize("database", ["miiv", "mimic", "eicu", "hirid", "aumc", "sic"])
def test_agentic_cli_refuses_credentialed_sources_before_delegation(monkeypatch, database):
    from easyicu.research_agent.agents.agentic_coder import AgenticCoderAgent

    monkeypatch.delenv("EASYICU_HOSTED", raising=False)
    agent = AgenticCoderAgent(SimpleNamespace(run=lambda **kwargs: "fallback"))
    with pytest.raises(ValueError, match="agentic_coder_credentialed_source_forbidden"):
        agent.run(context=SimpleNamespace(cohort=SimpleNamespace(database=database)), step=None)


def test_agentic_cli_direct_delegation_cannot_bypass_source_policy(monkeypatch):
    from easyicu.research_agent.agents.agentic_coder import AgenticCoderAgent

    monkeypatch.delenv("EASYICU_HOSTED", raising=False)
    agent = AgenticCoderAgent(None)
    with pytest.raises(ValueError, match="agentic_coder_credentialed_source_forbidden"):
        agent._delegate(SimpleNamespace(cohort=SimpleNamespace(database="miiv")), None)


def _loopback_client(monkeypatch, *, declaration="unspecified", response="unused"):
    from easyicu.research_agent.providers.factory import build_provider_client

    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=response), finish_reason="stop",
        )])

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(
        OpenAI=lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=complete),
        )),
    ))
    client = build_provider_client(
        provider="openai", model="synthetic", request_timeout=1, title="fixture",
        stream_enabled=False, environment={
            "OPENAI_BASE_URL": "http://127.0.0.1:8317/v1",
            "EASYICU_PROVIDER_DATA_HANDLING": declaration,
        },
    )
    return client, calls


@pytest.mark.parametrize("declaration", ["unspecified", "zero_retention_declared", "on_device"])
@pytest.mark.parametrize("database", ["miiv", "eicu", "hirid", "aumc", "sic", "mimic_demo", "eicu_demo", "synthetic"])
def test_clearance_uses_source_and_explicit_declaration(monkeypatch, declaration, database):
    from easyicu.research_agent.providers.client_trust import (
        ProviderConfigurationError, provider_transport_destination,
        require_credentialed_data_clearance,
    )

    monkeypatch.delenv("EASYICU_HOSTED", raising=False)
    client, calls = _loopback_client(monkeypatch, declaration=declaration)
    assert provider_transport_destination(client) == ("local" if declaration == "on_device" else "external")
    if declaration == "unspecified" and database in {"miiv", "eicu", "hirid", "aumc", "sic"}:
        with pytest.raises(ProviderConfigurationError) as error:
            require_credentialed_data_clearance(client, database)
        assert error.value.reason_code == "provider_not_cleared_for_credentialed_data"
    else:
        require_credentialed_data_clearance(client, database)
    assert calls == []


@pytest.mark.parametrize("role", ["planner", "coder", "repair", "writer"])
def test_agents_refuse_before_a_credentialed_prompt_is_sent(ra, monkeypatch, role):
    from easyicu.research_agent.agents.core import PlannerAgent, CoderAgent, WriterAgent
    from easyicu.research_agent.providers.client_trust import ProviderConfigurationError

    context, step = _context_and_step(ra)
    context.cohort.database = "miiv"
    client, calls = _loopback_client(monkeypatch)
    with pytest.raises(ProviderConfigurationError, match="provider_not_cleared_for_credentialed_data"):
        if role == "planner":
            PlannerAgent(client).run(context)
        elif role == "coder":
            CoderAgent(client).run(context=context, step=step)
        elif role == "repair":
            CoderAgent(client).repair(context=context, step=step, code="value = 0", run_log="fixture")
        else:
            WriterAgent(client)._call_section(
                context=context, section_name="Results", instruction="Describe results.",
                evidence_ids=[], evidence_digest="fixture",
            )
    assert calls == []


def test_pipeline_refuses_before_reading_a_credentialed_cohort(monkeypatch, tmp_path):
    import inspect
    from easyicu.research_agent.pipeline import ResearchAgentPipeline
    from easyicu.research_agent.providers.client_trust import ProviderConfigurationError

    pipeline = ResearchAgentPipeline.__new__(ResearchAgentPipeline)
    pipeline._llm, calls = _loopback_client(monkeypatch)
    pipeline._llm_concept_auditor_client = None
    with pytest.raises(ProviderConfigurationError, match="provider_not_cleared_for_credentialed_data"):
        inspect.unwrap(ResearchAgentPipeline.run)(pipeline, question="Describe synthetic records.", database="miiv", cohort=tmp_path / "absent.parquet")
    assert calls == []


@pytest.mark.parametrize("database", ["mimic_demo", "eicu_demo"])
def test_agentic_cli_keeps_demo_delegation(monkeypatch, database):
    from easyicu.research_agent.agents import agentic_coder

    monkeypatch.delenv("EASYICU_HOSTED", raising=False)
    monkeypatch.setenv("EASYICU_ALLOW_EXTERNAL_LLM", "1")
    monkeypatch.setattr(agentic_coder, "cli_backend_available", lambda _: True)
    agent = agentic_coder.AgenticCoderAgent(None)
    monkeypatch.setattr(agent, "_delegate", lambda context, step: "synthetic script")
    monkeypatch.setattr(agent, "_enforce_compatibility", lambda context, step, code: code)
    assert agent.run(context=SimpleNamespace(cohort=SimpleNamespace(database=database)), step=None) == "synthetic script"


def test_account_mode_cannot_inherit_api_provider_declaration():
    from easyicu.webserver.pi_copilot.service import PiCopilotService

    service = PiCopilotService.__new__(PiCopilotService)
    service.gateway = SimpleNamespace(environ={"EASYICU_PI_DATA_HANDLING": "zero_retention_declared"})
    session = SimpleNamespace(research_provider=SimpleNamespace(authentication_mode="chatgpt_account"))
    assert service._data_handling_for_session(session) == "unspecified"


def test_loopback_repair_redacts_script_and_cannot_exempt_figure_upload(ra, monkeypatch, tmp_path):
    from easyicu.research_agent.agents.core import CoderAgent
    from easyicu.research_agent.providers.client_trust import provider_transport_destination
    from easyicu.research_agent.gates.figure_egress import authorize_figure_upload, FigureEgressError
    from easyicu.research_agent.repairs.patch import PATCH_FORMAT
    from easyicu.research_agent.authority.table_one_binding import bind_table_one_execution_spec

    context, _ = _context_and_step(ra)
    labels = ["SYNTHETIC_ALPHA", "SYNTHETIC_BETA", "SYNTHETIC_GAMMA", "SYNTHETIC_DELTA"]
    for i, variable in enumerate(context.variables):
        variable.dtype = "object"
        variable.observed_domain = {"levels": labels[i * 2:i * 2 + 2], "n_unique": 2}
    tokens = ["__easyicu_level_1__", "__easyicu_level_2__"]
    step = ra.AnalysisStep.model_validate({
        "step_id": "table_one", "planned_analysis_role": "auxiliary",
        "intent": "Describe the synthetic categories.", "inputs": ["exposure", "outcome"],
        "expected_outputs": ["table:table_one"], "method": "table_one",
        "table_one_spec": {"group_by": "exposure", "group_levels": tokens, "variables": [{
            "name": "outcome", "variable_kind": "categorical", "summary": "count_percent",
            "test": "chi_square_with_fisher_exact_for_sparse_2x2", "levels": tokens,
        }]},
    })
    assert bind_table_one_execution_spec(step, context) is not None
    response = json.dumps({"format": PATCH_FORMAT, "edits": [{
        "old": "value = 1", "new": "value = 4", "expected_count": 1,
    }]})
    client, calls = _loopback_client(monkeypatch, response=response)
    authority = _authority()
    CoderAgent(client).repair(
        context=context, step=step, code=f"levels = {labels!r}\nvalue = 1\n", run_log="fixture",
        repair_authority=authority, current_repair_authority=authority,
    )
    assert all(label not in json.dumps(calls) for label in labels)
    assert "__easyicu_table1_label_" in json.dumps(calls)
    with pytest.raises(FigureEgressError):
        authorize_figure_upload([tmp_path / "unused.png"], policy=None,
                                destination=provider_transport_destination(client))


@pytest.mark.parametrize("count", range(SMALL_CELL_BELOW + 1))
def test_no_small_counts_reach_agent_messages_or_copilot_results(ra, count, tmp_path):
    from easyicu.research_agent.agents.core import PlannerAgent, CoderAgent, WriterAgent
    from easyicu.research_agent.providers.mocks import ExternalCaptureMockLLMClient
    from easyicu.research_agent.reporting.writer_evidence import _render_writer_evidence_digest
    from easyicu.research_agent.schema import MissingnessProfile
    from easyicu.webserver.pi_copilot.contracts import PiSessionRecord, ToolExecutionContext
    from easyicu.webserver.pi_copilot.tools import _result

    context, step = _context_and_step(ra)
    context.cohort.n_stays = context.cohort.n_patients = count
    context.variables[0].missingness = MissingnessProfile(
        fraction_missing=0.5, n_missing=count, n_total=count,
        raw_n_missing=count, eligible_n=count, not_applicable_n=count,
    )
    planner = "\n".join(m.content for m in PlannerAgent.request_messages(context))
    coder = ExternalCaptureMockLLMClient(["value = 0"])
    CoderAgent(coder).run(context=context, step=step)
    repair = ExternalCaptureMockLLMClient(["value = 0", "value = 0"])
    CoderAgent(repair).repair(context=context, step=step, code="value = 0", run_log="fixture",
                             repair_authority=_authority(), current_repair_authority=_authority())
    digest = _render_writer_evidence_digest([
        {"step_id": "summary", "status": "ok", "step_summary": {"n_total": count}},
    ], context=context, run_dir=tmp_path)
    writer = ExternalCaptureMockLLMClient(["No numeric claim."])
    WriterAgent(writer)._call_section(
        context=context, section_name="Results", instruction="Describe the evidence.",
        evidence_ids=[], evidence_digest=digest + '\n' + json.dumps({"n": count}),
    )
    tool_context = ToolExecutionContext(session=PiSessionRecord(
        session_id="fixture", project_id="fixture",
    ))
    copilot = _result(tool_context, status="ok", code="fixture", summary="Aggregate result.",
                      owner="fixture", details={
                          "cohort_size": count, "icd_counts": {"A": count},
                          "aggregate": {"stays": count, "source_total": count, "selected_before_icd": count},
                      })
    texts = [planner, *(_all_prompt_text(c) for c in (coder, repair, writer)), json.dumps(copilot)]
    for text in texts:
        counts = re.findall(r'"(?:n|n_total|n_stays|n_patients|n_missing|raw_n_missing|eligible_n|not_applicable_n|cohort_size|A|stays|source_total|selected_before_icd)"\s*:\s*(\d+)', text)
        assert all(not 0 < int(value) < SMALL_CELL_BELOW for value in counts), text
        assert '"suppressed"' in text if 0 < count < SMALL_CELL_BELOW else True


def test_copilot_and_run_gate_refuse_uncleared_source_with_settings_action(monkeypatch):
    from easyicu.webserver.pi_copilot import tools
    from easyicu.webserver.pi_copilot.contracts import PiSessionRecord, ToolExecutionContext, PiCopilotError
    from easyicu.webserver.provider_gate import resolve_provider_gate, ProviderGateError

    context = ToolExecutionContext(session=PiSessionRecord(session_id="fixture", project_id="fixture"))
    monkeypatch.setattr(tools, "_bound_context", lambda _: {"data_source": {"database": "miiv", "path": "/synthetic-unused-export"}})
    calls = []
    monkeypatch.setitem(tools._DISPATCH, "easyicu_inspect_context", lambda *_: calls.append(True))
    with pytest.raises(PiCopilotError) as error:
        tools.execute_tool("easyicu_inspect_context", {}, context)
    assert error.value.code == "provider_not_cleared_for_credentialed_data"
    assert error.value.details["next_actions"][0]["kind"] == "open_provider_settings"
    assert error.value.details["next_actions"][0]["params"] == {"route": "provider_settings"}
    assert calls == []
    with pytest.raises(ProviderGateError) as error:
        resolve_provider_gate(run_type="full", llm_provider="openai", external_llm_opt_in=True,
                              ai_enabled=True, database="miiv")
    assert error.value.detail["error"] == "provider_not_cleared_for_credentialed_data"
    assert error.value.detail["next_actions"][0]["kind"] == "open_provider_settings"
    assert error.value.detail["next_actions"][0]["params"] == {"route": "provider_settings"}


def test_pinned_public_demo_is_scoped_without_trusting_a_source_label(tmp_path, monkeypatch):
    from easyicu.webserver import demo_source_storage as storage
    from easyicu.webserver.demo_source_contracts import SOURCES
    from easyicu.research_agent.providers.client_trust import (
        ProviderConfigurationError, require_data_handling, source_requires_clearance,
    )

    monkeypatch.setenv("EASYICU_DEMO_CACHE_DIR", str(tmp_path / "demo-cache"))
    source = SOURCES[0]
    paths = storage.source_paths(source)
    paths.export.mkdir(parents=True)
    (paths.export / "_manifest.json").write_text("{}")
    storage.write_export_marker(paths, source, archive_sha256=source.archive_sha256, export={})
    assert storage.egress_source_database(str(paths.export), "miiv") == source.id
    assert storage.egress_source_database(str(tmp_path / source.title), "miiv") == "miiv"
    with storage.public_demo_egress_scope(str(paths.export)):
        require_data_handling("miiv", "unspecified")
        assert not source_requires_clearance("miiv")
        with pytest.raises(ProviderConfigurationError):
            require_data_handling("eicu", "unspecified")
    with pytest.raises(ProviderConfigurationError):
        require_data_handling("miiv", "unspecified")
    storage.write_export_marker(paths, source, archive_sha256="unverified", export={})
    assert storage.egress_source_database(str(paths.export), "miiv") == "miiv"


def test_copilot_retains_credentialed_history_after_switching_to_demo(monkeypatch):
    from easyicu.webserver.pi_copilot import tools
    from easyicu.webserver.pi_copilot.contracts import PiSessionRecord, ToolExecutionContext, PiCopilotError

    study = {"data_source": {"database": "miiv", "path": "/synthetic-unused-export"}}
    monkeypatch.setattr(tools, "_bound_context", lambda _: study)
    saved = []
    context = ToolExecutionContext(
        session=PiSessionRecord(session_id="fixture", project_id="fixture"),
        data_handling="zero_retention_declared", note_credentialed_history=lambda: saved.append(True),
    )
    tools.require_copilot_data_clearance(context)
    assert saved and context.session.credentialed_data_in_history is True
    study["data_source"]["database"] = "eicu_demo"
    context.data_handling = "unspecified"
    with pytest.raises(PiCopilotError, match="declare data handling"):
        tools.require_copilot_data_clearance(context)


def test_source_listing_checks_the_source_inferred_from_the_user_message(monkeypatch):
    from easyicu.webserver.pi_copilot import tools
    from easyicu.webserver.pi_copilot.contracts import PiSessionRecord, ToolExecutionContext, PiCopilotError

    monkeypatch.setattr(tools, "_bound_context", lambda _: None)
    monkeypatch.setattr(tools.sources, "load_registry", lambda: {"sources": [{
        "id": "synthetic-source", "database": "miiv", "path": "/synthetic-unused-export",
        "ok": True, "modules": ["fixture"], "summary": {"stays": 4},
    }]})
    context = ToolExecutionContext(
        session=PiSessionRecord(session_id="fixture", project_id="fixture"),
        user_message="Show all registered MIMIC-IV sources.",
    )
    with pytest.raises(PiCopilotError, match="declare data handling"):
        tools.execute_tool("easyicu_list_data_sources", {}, context)


def test_selected_export_catalog_is_checked_before_reading(monkeypatch):
    from easyicu.webserver.pi_copilot import tools
    from easyicu.webserver.pi_copilot.contracts import PiSessionRecord, ToolExecutionContext, PiCopilotError
    from easyicu.research_agent.acquisition import catalog

    monkeypatch.setattr(tools, "_bound_context", lambda _: {})
    monkeypatch.setattr(tools.sources, "load_registry", lambda: {"sources": [{
        "id": "synthetic-source", "database": "miiv", "path": "/synthetic-unused-export", "ok": True,
    }]})
    calls = []
    monkeypatch.setattr(catalog, "build_available_catalog", lambda *_: calls.append(True))
    context = ToolExecutionContext(session=PiSessionRecord(session_id="fixture", project_id="fixture"))
    with pytest.raises(PiCopilotError, match="declare data handling"):
        tools.execute_tool("easyicu_list_source_concepts", {"source_id": "synthetic-source"}, context)
    assert calls == []
