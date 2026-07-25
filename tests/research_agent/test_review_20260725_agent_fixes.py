"""Gates for the 2026-07-25 external research-agent review.

One test per finding, named with the finding id so a reviewer can map a
report line to the check that closes it.

The theme of the report is that *new interfaces bypassed old boundaries*:
MCP returned patient rows the text outbound projection would have stripped;
the VLM uploaded whole figures that same projection never saw; the write
phase demoted every visual error before the readiness classifier that
distinguishes cosmetic from fatal could run.
"""

from __future__ import annotations

import dataclasses
import json
import os
import queue
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.gates.figure_egress import (
    AGGREGATE_ONLY_METADATA_KEY,
    FigureEgressError,
    FigureEgressPolicy,
    authorize_figure_upload,
)
from easyicu.research_agent.gates.visual_qa import VLMVisualQAAdapter
from easyicu.research_agent.learning.experience import (
    ExperienceBank,
    ExperienceBankCorruptError,
    ExperienceRecord,
)
from easyicu.research_agent.mcp_policy import (
    MAX_PREVIEW_ROWS,
    MCP_ALLOWED_ROOTS_ENV,
    MCP_ALLOW_IDENTIFIER_COLUMNS_ENV,
    MCP_ALLOW_PATIENT_DATA_ENV,
    MCP_SCOPES_ENV,
    MIN_ROWS_FOR_AGGREGATE_STATS,
    SCOPE_READ_PATIENT_DATA,
    DisclosurePolicy,
    MCPPathError,
    granted_scopes,
    resolve_within_roots,
    scope_override,
    summarise_frame,
)
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.reporting.write_phase import (
    demote_cosmetic_publication_visual_findings,
)
from easyicu.research_agent.schema import ValidationFinding


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _cohort_frame(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": range(1, rows + 1),
            "charttime": pd.date_range("2130-01-01", periods=rows, freq="h"),
            "sofa2": [i % 12 for i in range(rows)],
            "lactate": [1.0 + (i % 7) * 0.4 for i in range(rows)],
        }
    )


@pytest.fixture
def roots(tmp_path, monkeypatch):
    monkeypatch.setenv(MCP_ALLOWED_ROOTS_ENV, str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# P0.1 — MCP bypassed the patient-data outbound boundary
# ---------------------------------------------------------------------------


def test_p0_1_patient_rows_are_not_disclosed_by_default(monkeypatch):
    monkeypatch.delenv(MCP_SCOPES_ENV, raising=False)
    monkeypatch.delenv(MCP_ALLOW_PATIENT_DATA_ENV, raising=False)

    assert SCOPE_READ_PATIENT_DATA not in granted_scopes()
    summary = summarise_frame(_cohort_frame(), policy=DisclosurePolicy.current(500))

    assert summary["preview"] == []
    assert "preview_withheld_reason" in summary
    assert summary["rows"] == 40
    assert summary["columns"] == ["stay_id", "charttime", "sofa2", "lactate"]


def test_p0_1_preview_rows_are_hard_capped_even_when_granted(monkeypatch):
    monkeypatch.setenv(MCP_ALLOW_PATIENT_DATA_ENV, "1")

    policy = DisclosurePolicy.current(10_000)
    assert policy.preview_rows == MAX_PREVIEW_ROWS

    summary = summarise_frame(_cohort_frame(rows=100), policy=policy)
    assert len(summary["preview"]) == MAX_PREVIEW_ROWS


def test_p0_1_identifier_and_time_columns_are_redacted_from_preview(monkeypatch):
    monkeypatch.setenv(MCP_ALLOW_PATIENT_DATA_ENV, "1")
    monkeypatch.delenv(MCP_ALLOW_IDENTIFIER_COLUMNS_ENV, raising=False)

    summary = summarise_frame(_cohort_frame(), policy=DisclosurePolicy.current(5))

    assert summary["preview"], "a granted scope should still return rows"
    for row in summary["preview"]:
        assert "stay_id" not in row
        assert "charttime" not in row
        assert "sofa2" in row
    assert summary["preview_redacted_columns"] == ["charttime", "stay_id"]


def test_p0_1_aggregate_statistics_withheld_below_the_small_cell_floor(monkeypatch):
    monkeypatch.delenv(MCP_ALLOW_PATIENT_DATA_ENV, raising=False)

    small = summarise_frame(
        _cohort_frame(rows=MIN_ROWS_FOR_AGGREGATE_STATS - 1),
        policy=DisclosurePolicy.current(),
    )
    assert small["aggregate_statistics"]["withheld"] is True

    large = summarise_frame(
        _cohort_frame(rows=MIN_ROWS_FOR_AGGREGATE_STATS + 5),
        policy=DisclosurePolicy.current(),
    )
    assert "lactate" in large["aggregate_statistics"]
    # Identifier and time columns are never described, even in aggregate.
    assert "stay_id" not in large["aggregate_statistics"]
    assert "charttime" not in large["aggregate_statistics"]


def test_p0_1_paths_outside_the_configured_roots_are_refused(roots):
    inside = resolve_within_roots(roots / "run" / "out.parquet", field="output_path")
    assert str(inside).startswith(str(roots))

    with pytest.raises(MCPPathError) as excinfo:
        resolve_within_roots("/etc/passwd", field="data_path")
    assert "outside the configured MCP roots" in str(excinfo.value)


def test_p0_1_symlinked_parent_cannot_hop_outside_a_root(roots, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    link = roots / "escape"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(MCPPathError):
        resolve_within_roots(link / "loot.parquet", field="output_path")


def test_p0_1_load_concepts_response_carries_no_rows_and_no_raw_data_path(
    roots, monkeypatch
):
    import easyicu

    from easyicu.research_agent.mcp_server import dispatch

    monkeypatch.delenv(MCP_ALLOW_PATIENT_DATA_ENV, raising=False)
    monkeypatch.setattr(
        easyicu, "load_concepts", lambda **kw: _cohort_frame(), raising=False
    )

    data_path = roots / "miiv"
    result = dispatch(
        "research_agent.load_concepts",
        {
            "concepts": ["sofa2"],
            "database": "miiv",
            "data_path": str(data_path),
            "preview_rows": 500,
            "workdir": str(roots / "run"),
        },
    )

    assert result["summary"]["frame"]["preview"] == []
    # The raw host path is replaced by its digest.
    assert "data_path" not in result
    assert len(result["data_path_sha256"]) == 64
    assert str(data_path) not in json.dumps(result)


def test_p0_1_patient_data_access_is_recorded_as_evidence(roots, monkeypatch):
    import easyicu

    from easyicu.research_agent.mcp_server import dispatch

    monkeypatch.setattr(
        easyicu, "load_concepts", lambda **kw: _cohort_frame(), raising=False
    )
    workdir = roots / "run"

    dispatch(
        "research_agent.load_concepts",
        {
            "concepts": ["sofa2"],
            "database": "miiv",
            "data_path": str(roots / "miiv"),
            "patient_ids": [1, 2, 3],
            "workdir": str(workdir),
        },
    )

    records = EvidenceStore(workdir).records()
    audits = [r for r in records if "mcp_patient_data_access" in str(r.relative_path)]
    assert audits, [r.relative_path for r in records]

    payload = json.loads((workdir / audits[0].relative_path).read_text())
    assert payload["schema"] == "easyicu.mcp_patient_data_access/1"
    assert payload["requested_patient_ids"] == 3
    assert payload["disclosed_patient_rows"] == 0
    assert len(payload["data_path_sha256"]) == 64
    # The database name is legitimate provenance; the host path is not.
    assert payload["database"] == "miiv"
    assert str(roots) not in json.dumps(payload)


def test_p0_1_scope_override_narrows_a_single_request(monkeypatch):
    monkeypatch.setenv(MCP_ALLOW_PATIENT_DATA_ENV, "1")
    assert SCOPE_READ_PATIENT_DATA in granted_scopes()

    with scope_override(granted_scopes() - {SCOPE_READ_PATIENT_DATA}):
        assert SCOPE_READ_PATIENT_DATA not in granted_scopes()
        assert DisclosurePolicy.current(5).preview_rows == 0

    assert SCOPE_READ_PATIENT_DATA in granted_scopes()


def test_p0_1_scope_override_cannot_widen_beyond_the_process_grant(monkeypatch):
    monkeypatch.delenv(MCP_ALLOW_PATIENT_DATA_ENV, raising=False)
    monkeypatch.delenv(MCP_SCOPES_ENV, raising=False)

    with scope_override(frozenset({SCOPE_READ_PATIENT_DATA})):
        assert SCOPE_READ_PATIENT_DATA not in granted_scopes()


def test_p0_1_run_pipeline_scope_can_be_withheld(roots, monkeypatch):
    from easyicu.research_agent.mcp_server import dispatch

    monkeypatch.setenv(MCP_SCOPES_ENV, "metadata")

    result = dispatch(
        "research_agent.run",
        {"question": "q", "cohort_path": str(roots / "c.parquet")},
    )
    assert result["error_code"] == "scope_not_granted"


def test_p0_1_sse_patient_data_needs_its_own_token_even_on_loopback():
    """A loopback bind does not authenticate; the separate token does."""

    from easyicu.research_agent import mcp_server

    handler_cls = mcp_server._make_sse_handler(bind_host="127.0.0.1", port=8765)
    probe = handler_cls.__new__(handler_cls)

    probe.headers = {}
    assert probe._patient_data_authorized() is False

    os.environ[mcp_server.MCP_PATIENT_DATA_TOKEN_ENV] = "s3cret"
    try:
        probe.headers = {"X-EasyICU-Patient-Data": "wrong"}
        assert probe._patient_data_authorized() is False
        probe.headers = {"X-EasyICU-Patient-Data": "s3cret"}
        assert probe._patient_data_authorized() is True
    finally:
        del os.environ[mcp_server.MCP_PATIENT_DATA_TOKEN_ENV]


# ---------------------------------------------------------------------------
# P0.2 — VLM uploaded whole figures with no image-level authorization
# ---------------------------------------------------------------------------


def _registered_figure(tmp_path: Path, *, aggregate_only: bool = True):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    evidence = EvidenceStore(run_dir)
    record = evidence.register_text(
        kind="figure",
        description="fixture figure",
        text="<svg/>",
        filename="fig1.svg",
        producer="publication_figure_skill",
        generation_mode="deterministic_skill",
        metadata={AGGREGATE_ONLY_METADATA_KEY: aggregate_only},
    )
    return run_dir, evidence, run_dir / record.relative_path, record


def test_p0_2_external_upload_is_denied_by_default(tmp_path):
    run_dir, evidence, path, _ = _registered_figure(tmp_path)
    policy = FigureEgressPolicy(evidence=evidence, run_dir=run_dir)

    with pytest.raises(FigureEgressError) as excinfo:
        authorize_figure_upload([path], policy=policy, destination="external")
    assert "external figure upload is disabled" in str(excinfo.value)


def test_p0_2_local_and_mock_destinations_stay_exempt(tmp_path):
    _, _, path, _ = _registered_figure(tmp_path)

    for destination in ("local", "mock"):
        entries = authorize_figure_upload([path], policy=None, destination=destination)
        assert len(entries) == 1
        assert entries[0]["destination"] == destination


def test_p0_2_authorized_upload_requires_a_registered_matching_artefact(tmp_path):
    run_dir, evidence, path, record = _registered_figure(tmp_path)
    policy = FigureEgressPolicy(
        allow_external_upload=True, evidence=evidence, run_dir=run_dir
    )

    entries = authorize_figure_upload([path], policy=policy, destination="external")
    assert entries[0]["evidence_id"] == record.evidence_id
    assert entries[0]["sha256"] == record.sha256
    assert policy.uploaded == entries

    # An unregistered sibling in the same run directory is refused.
    stray = run_dir / "stray.svg"
    stray.write_text("<svg/>", encoding="utf-8")
    with pytest.raises(FigureEgressError) as excinfo:
        authorize_figure_upload([stray], policy=policy, destination="external")
    assert "not a registered evidence artefact" in str(excinfo.value)


def test_p0_2_upload_refuses_bytes_that_no_longer_match_the_digest(tmp_path):
    run_dir, evidence, path, _ = _registered_figure(tmp_path)
    policy = FigureEgressPolicy(
        allow_external_upload=True, evidence=evidence, run_dir=run_dir
    )
    path.write_text("<svg>tampered</svg>", encoding="utf-8")

    with pytest.raises(FigureEgressError) as excinfo:
        authorize_figure_upload([path], policy=policy, destination="external")
    assert "no longer matches its registered digest" in str(excinfo.value)


def test_p0_2_upload_requires_an_aggregate_only_declaration(tmp_path):
    run_dir, evidence, path, _ = _registered_figure(tmp_path, aggregate_only=False)
    policy = FigureEgressPolicy(
        allow_external_upload=True, evidence=evidence, run_dir=run_dir
    )

    with pytest.raises(FigureEgressError) as excinfo:
        authorize_figure_upload([path], policy=policy, destination="external")
    assert AGGREGATE_ONLY_METADATA_KEY in str(excinfo.value)


def test_p0_2_upload_refuses_a_path_outside_the_run_directory(tmp_path):
    run_dir, evidence, _, _ = _registered_figure(tmp_path)
    outside = tmp_path / "elsewhere.png"
    outside.write_bytes(b"\x89PNG")
    policy = FigureEgressPolicy(
        allow_external_upload=True, evidence=evidence, run_dir=run_dir
    )

    with pytest.raises(FigureEgressError) as excinfo:
        authorize_figure_upload([outside], policy=policy, destination="external")
    assert "outside the run directory" in str(excinfo.value)


def test_p0_2_adapter_degrades_to_metadata_review_and_says_so(tmp_path):
    """A denied upload must not silently look like a clean visual review."""

    run_dir, evidence, path, _ = _registered_figure(tmp_path)
    sent = {}

    class _ExternalVisionClient:
        # Not registered in the provider factory, so it classifies as
        # "external" — the deliberately conservative default.
        def complete_with_images(self, **kwargs):
            sent["images"] = kwargs.get("image_paths")
            return '{"findings": []}'

        def complete(self, messages, **kwargs):
            sent["text_only"] = True
            return '{"findings": []}'

    adapter = VLMVisualQAAdapter(
        _ExternalVisionClient(),
        egress_policy=FigureEgressPolicy(evidence=evidence, run_dir=run_dir),
    )
    findings = adapter.audit(figure_paths=[path])

    assert "images" not in sent, "figure bytes must not reach an external provider"
    denied = [
        f for f in findings if (f.detail or {}).get("reason") == "figure_egress_denied"
    ]
    assert denied, [f.message for f in findings]
    assert denied[0].severity == "warning"
    assert "external figure upload is disabled" in denied[0].message
    # The audit never silently reports "no visual problems": either the text
    # fallback ran, or the provider gate blocked that too and said so.
    assert len(findings) >= 1


def test_p0_2_pipeline_default_denies_external_figure_upload(tmp_path):
    from easyicu.research_agent.pipeline import ResearchAgentPipeline

    config = PipelineConfig(workdir=tmp_path / "wd")
    assert config.allow_external_figure_upload is False

    pipeline = ResearchAgentPipeline.from_config(config)
    assert pipeline._figure_egress_policy().allow_external_upload is False

    opted_in = ResearchAgentPipeline.from_config(
        config.with_overrides(allow_external_figure_upload=True)
    )
    assert opted_in._figure_egress_policy().allow_external_upload is True


# ---------------------------------------------------------------------------
# P0.3 — publication visual QA demoted every error to a warning
# ---------------------------------------------------------------------------


def _visual_error(message: str) -> ValidationFinding:
    return ValidationFinding(validator="visual_qa", severity="error", message=message)


@pytest.mark.parametrize(
    "message",
    [
        "SVG figure 'x.svg' contains no data-backed marks (blank figure).",
        "PNG figure 'x.png' is cropped; the y axis label is cut off.",
        "Figure 'x.svg' is missing its axis labels entirely.",
        "Figure 'x.svg' reports AUROC 0.81 but the evidence numeric is 0.77 (mismatch).",
        "Figure 'x.svg' axis text is unreadable at publication size.",
    ],
)
def test_p0_3_hard_visual_errors_survive_the_publication_pass(message):
    [out] = demote_cosmetic_publication_visual_findings([_visual_error(message)])
    assert out.severity == "error", message


def test_p0_3_only_the_spacing_nit_is_demoted():
    overlap = _visual_error(
        "SVG figure 'y.svg' has overlapping text elements; multi-panel labels, "
        "annotations or axis text need more spacing."
    )
    [out] = demote_cosmetic_publication_visual_findings([overlap])
    assert out.severity == "warning"


def test_p0_3_a_blank_publication_figure_still_blocks_readiness(tmp_path):
    """End to end through the same path the write phase now takes."""

    from easyicu.research_agent.reporting.readiness import _compute_readiness_gates
    from easyicu.research_agent.schema import ResearchContext

    blank = _visual_error(
        "SVG figure 'x.svg' contains no data-backed marks (blank figure)."
    )
    demoted = demote_cosmetic_publication_visual_findings([blank])

    gates = _compute_readiness_gates(
        context=ResearchContext(
            research_question="Does SOFA-2 predict mortality?",
            cohort={
                "cohort_name": "c",
                "database": "miiv",
                "n_patients": 10,
                "n_stays": 10,
            },
            variables=[],
        ),
        plan=None,
        per_step_records=[],
        findings=demoted,
        evidence=EvidenceStore(tmp_path),
        run_dir=tmp_path,
        manuscript_path=tmp_path / "m.md",
        stop_after_analysis=False,
    )
    assert any("blank figure" in message for message in gates["analysis_errors"])


# ---------------------------------------------------------------------------
# P1.1 — Docker runner had no default resource ceilings
# ---------------------------------------------------------------------------


def _built_command(runner, tmp_path) -> list[str]:
    """Render the real ``docker run`` argv without a live daemon."""

    script = runner.workdir / "step.py"
    script.write_text("print(1)", encoding="utf-8")
    out_dir = runner.workdir / "out"
    out_dir.mkdir(exist_ok=True)
    return runner.build_command(step_id="01_probe", script_path=script, out_dir=out_dir)


def _runner(tmp_path, monkeypatch, **kwargs):
    from easyicu.research_agent.execution import runner as runner_module

    monkeypatch.setattr(runner_module.shutil, "which", lambda name: "/usr/bin/docker")
    tmp_path.mkdir(parents=True, exist_ok=True)
    cohort = tmp_path / "cohort.parquet"
    cohort.write_bytes(b"fixture")
    return runner_module.DockerRunner(
        workdir=tmp_path / "wd", cohort_parquet=cohort, **kwargs
    )


def test_p1_1_docker_runner_defaults_declare_resource_limits():
    from easyicu.research_agent.execution.runner import DockerRunner

    assert DockerRunner.DEFAULT_CPU_LIMIT
    assert DockerRunner.DEFAULT_MEMORY_LIMIT
    assert DockerRunner.DEFAULT_PIDS_LIMIT > 0
    assert DockerRunner.DEFAULT_OPEN_FILES_LIMIT > 0


def test_p1_1_docker_command_carries_cpu_memory_swap_pids_and_nofile(
    tmp_path, monkeypatch
):
    """The timeout alone does not bound a runaway step; these flags do."""

    from easyicu.research_agent.execution.runner import DockerRunner

    command = _built_command(_runner(tmp_path, monkeypatch), tmp_path)

    assert f"--cpus={DockerRunner.DEFAULT_CPU_LIMIT}" in command
    assert f"--memory={DockerRunner.DEFAULT_MEMORY_LIMIT}" in command
    # Without an equal swap cap the memory limit only makes the host thrash.
    assert f"--memory-swap={DockerRunner.DEFAULT_MEMORY_LIMIT}" in command
    assert f"--pids-limit={DockerRunner.DEFAULT_PIDS_LIMIT}" in command
    nofile = DockerRunner.DEFAULT_OPEN_FILES_LIMIT
    assert f"--ulimit=nofile={nofile}:{nofile}" in command


def test_p1_1_narrower_limits_reach_the_command(tmp_path, monkeypatch):
    command = _built_command(
        _runner(
            tmp_path,
            monkeypatch,
            cpu_limit="1",
            memory_limit="512m",
            pids_limit=32,
            open_files_limit=256,
        ),
        tmp_path,
    )

    assert "--cpus=1" in command
    assert "--memory=512m" in command
    assert "--memory-swap=512m" in command
    assert "--pids-limit=32" in command
    assert "--ulimit=nofile=256:256" in command


def test_p1_1_limits_are_bound_into_the_replay_authority(tmp_path, monkeypatch):
    """A limit change must change the runner authority digest, not be invisible."""

    from easyicu.research_agent.execution.runner import DockerRunner

    monkeypatch.setattr(
        DockerRunner,
        "_inspect_image_identity",
        lambda self: ("sha256:" + "0" * 64, ()),
    )

    baseline = _runner(tmp_path / "a", monkeypatch).authority_identity_sha256
    same = _runner(tmp_path / "b", monkeypatch).authority_identity_sha256
    narrowed = _runner(
        tmp_path / "c", monkeypatch, pids_limit=32
    ).authority_identity_sha256

    assert baseline == same
    assert narrowed != baseline


def test_p1_1_limits_can_be_opted_out_explicitly(tmp_path, monkeypatch):
    from easyicu.research_agent.execution import runner as runner_module

    monkeypatch.setattr(runner_module.shutil, "which", lambda name: "/usr/bin/docker")
    cohort = tmp_path / "cohort.parquet"
    cohort.write_bytes(b"fixture")

    runner = runner_module.DockerRunner(
        workdir=tmp_path / "wd",
        cohort_parquet=cohort,
        pids_limit=0,
        open_files_limit=0,
        cpu_limit="",
        memory_limit="",
    )
    assert runner.pids_limit == 0
    assert runner.open_files_limit == 0
    assert runner.cpu_limit == ""
    assert runner.memory_limit == ""


def test_p1_1_defaults_apply_when_the_caller_says_nothing(tmp_path, monkeypatch):
    from easyicu.research_agent.execution import runner as runner_module

    monkeypatch.setattr(runner_module.shutil, "which", lambda name: "/usr/bin/docker")
    cohort = tmp_path / "cohort.parquet"
    cohort.write_bytes(b"fixture")

    runner = runner_module.DockerRunner(workdir=tmp_path / "wd", cohort_parquet=cohort)
    assert runner.pids_limit == runner_module.DockerRunner.DEFAULT_PIDS_LIMIT
    assert runner.memory_limit == runner_module.DockerRunner.DEFAULT_MEMORY_LIMIT


# ---------------------------------------------------------------------------
# P1.2 — human review decisions never reached the run authority
# ---------------------------------------------------------------------------


def test_p1_2_decision_record_binds_authority_and_stamps_server_time():
    from easyicu.research_agent.graph import (
        HumanReviewDecision,
        HumanReviewRequest,
        _human_review_decision_record,
    )

    digest = "a" * 64
    request = HumanReviewRequest.create(
        kind="protocol_claim",
        summary="Approve the analysis plan.",
        authority_sha256=digest,
        payload={"step": "01_primary"},
    )
    decision = HumanReviewDecision(
        review_id=request.review_id,
        authority_sha256=digest,
        decision="approved",
        reviewer="whoever-typed-this",
        decided_at="1999-01-01T00:00:00Z",
    )

    record = _human_review_decision_record(
        request=request, decision=decision, reviewer_identity=None
    )

    assert record["schema"] == "easyicu.human_review_decision/1"
    assert record["authority_sha256"] == digest
    assert len(record["request_sha256"]) == 64
    assert len(record["decision_sha256"]) == 64
    # The client's claim is kept, but flagged as a claim.
    assert record["claimed_reviewer"] == "whoever-typed-this"
    assert record["reviewer_identity"] is None
    assert record["reviewer_identity_source"] == "unauthenticated_client_claim"
    assert record["server_decided_at"] != decision.decided_at

    authenticated = _human_review_decision_record(
        request=request, decision=decision, reviewer_identity="alex@hospital"
    )
    assert authenticated["reviewer_identity_source"] == "authenticated"


def test_p1_2_graph_hands_decisions_to_the_recorder():
    """Drive the real interrupt/resume cycle, not just the record builder."""

    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from easyicu.research_agent.graph import HumanReviewRequest, build_pipeline_graph

    digest = "b" * 64
    request = HumanReviewRequest.create(
        kind="protocol_claim",
        summary="Approve the analysis plan.",
        authority_sha256=digest,
        payload={"step": "01_primary"},
    )
    recorded: list = []

    graph = build_pipeline_graph(
        plan_invoker=lambda: {"aborted_result": None},
        execute_invoker=lambda plan: {"ok": True},
        write_invoker=lambda plan, execute: {"ok": True},
        finalise_invoker=lambda plan, execute, write: {"final": True},
        human_review_invoker=lambda plan: [request],
        human_review_recorder=recorded.extend,
        reviewer_identity_resolver=lambda: "alex@hospital",
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "review-test"}}

    paused = graph.invoke({}, config)
    assert "__interrupt__" in paused
    assert not recorded, "nothing may be recorded before the operator decides"

    final = graph.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "review_id": request.review_id,
                        "authority_sha256": digest,
                        "decision": "approved",
                        "reviewer": "typed-by-client",
                        "decided_at": "1999-01-01T00:00:00Z",
                    }
                ]
            }
        ),
        config,
    )

    assert final["final_result"] == {"final": True}
    assert len(recorded) == 1
    record = recorded[0]
    assert record["review_id"] == request.review_id
    assert record["authority_sha256"] == digest
    assert record["reviewer_identity"] == "alex@hospital"
    assert record["reviewer_identity_source"] == "authenticated"
    # The client's backdated claim is kept but does not become the timestamp.
    assert record["claimed_decided_at"] == "1999-01-01T00:00:00Z"
    assert record["server_decided_at"] != record["claimed_decided_at"]


def test_p1_2_a_mismatched_authority_digest_is_rejected():
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from easyicu.research_agent.graph import HumanReviewRequest, build_pipeline_graph

    request = HumanReviewRequest.create(
        kind="protocol_claim",
        summary="Approve the analysis plan.",
        authority_sha256="c" * 64,
        payload={"step": "01_primary"},
    )
    recorded: list = []
    graph = build_pipeline_graph(
        plan_invoker=lambda: {"aborted_result": None},
        execute_invoker=lambda plan: {"ok": True},
        write_invoker=lambda plan, execute: {"ok": True},
        finalise_invoker=lambda plan, execute, write: {"final": True},
        human_review_invoker=lambda plan: [request],
        human_review_recorder=recorded.extend,
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "review-mismatch"}}
    graph.invoke({}, config)

    with pytest.raises(ValueError, match="authority digest mismatch"):
        graph.invoke(
            Command(
                resume={
                    "decisions": [
                        {
                            "review_id": request.review_id,
                            "authority_sha256": "d" * 64,
                            "decision": "approved",
                            "reviewer": "r",
                            "decided_at": "2026-07-25T00:00:00Z",
                        }
                    ]
                }
            ),
            config,
        )
    assert not recorded


# ---------------------------------------------------------------------------
# P1.3 — ExperienceBank claimed a cross-process lock it did not have
# ---------------------------------------------------------------------------


def _record(n: int) -> ExperienceRecord:
    return ExperienceRecord(
        kind="concept_usage_hint",
        research_question=f"question {n}",
        database="miiv",
        cohort_name="c",
        summary=f"lesson {n}",
    )


def test_p1_3_concurrent_processes_do_not_lose_records(tmp_path):
    bank_path = tmp_path / "bank.jsonl"
    script = textwrap.dedent(
        """
        import sys
        from easyicu.research_agent.learning.experience import (
            ExperienceBank, ExperienceRecord,
        )
        path, tag, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
        bank = ExperienceBank(path)
        for i in range(count):
            bank.add(
                ExperienceRecord(
                    kind="concept_usage_hint",
                    research_question=f"{tag} question {i}",
                    database="miiv",
                    cohort_name="c",
                    summary=f"{tag} lesson {i}",
                )
            )
        """
    )
    runner = tmp_path / "writer.py"
    runner.write_text(script, encoding="utf-8")

    procs = [
        subprocess.Popen(
            [sys.executable, str(runner), str(bank_path), tag, "20"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for tag in ("a", "b")
    ]
    for proc in procs:
        _, err = proc.communicate(timeout=180)
        assert proc.returncode == 0, err.decode()

    reloaded = ExperienceBank(bank_path)
    summaries = {r.summary for r in reloaded.records()}
    assert len(summaries) == 40, sorted(summaries)


def test_p1_3_save_is_atomic_so_a_reader_never_sees_a_half_file(tmp_path):
    bank_path = tmp_path / "bank.jsonl"
    bank = ExperienceBank(bank_path)
    for i in range(50):
        bank.add(_record(i))

    torn: list[str] = []
    stop = threading.Event()

    def _reader():
        while not stop.is_set():
            try:
                text = bank_path.read_text(encoding="utf-8")
            except OSError:
                continue
            for line in text.splitlines():
                if line.strip():
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        torn.append(line)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    try:
        for i in range(50, 150):
            bank.add(_record(i))
    finally:
        stop.set()
        thread.join(timeout=10)

    assert not torn, torn[:3]


def test_p1_3_mutation_refuses_to_rewrite_over_unparseable_lines(tmp_path):
    bank_path = tmp_path / "bank.jsonl"
    bank_path.write_text(
        json.dumps(_record(1).to_dict()) + "\nthis is not JSON\n", encoding="utf-8"
    )

    bank = ExperienceBank(bank_path)
    assert bank.corrupt_lines == 1
    # Read-only retrieval still works on the parseable remainder.
    assert len(bank.records()) == 1

    with pytest.raises(ExperienceBankCorruptError):
        bank.add(_record(2))
    # The unparseable line is still on disk, not silently dropped.
    assert "this is not JSON" in bank_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# P1.4 — SSE sessions and queues were unbounded
# ---------------------------------------------------------------------------


def test_p1_4_sse_sessions_are_capped_and_released():
    from easyicu.research_agent import mcp_server

    opened = []
    try:
        for _ in range(mcp_server.MAX_SSE_SESSIONS):
            got = mcp_server._open_sse_session()
            assert got is not None
            opened.append(got[0])
        assert mcp_server._open_sse_session() is None
    finally:
        for session_id in opened:
            mcp_server._close_sse_session(session_id)

    assert mcp_server._open_sse_session() is not None
    for session_id in list(mcp_server._SSE_SESSIONS):
        mcp_server._close_sse_session(session_id)


def test_p1_4_sse_queue_is_bounded():
    from easyicu.research_agent import mcp_server

    session = mcp_server._SSESession()
    assert session.queue.maxsize == mcp_server.SSE_QUEUE_MAXSIZE

    for i in range(mcp_server.SSE_QUEUE_MAXSIZE):
        session.queue.put_nowait({"i": i})
    with pytest.raises(queue.Full):
        session.queue.put_nowait({"overflow": True})


def test_p1_4_session_table_is_lock_guarded_under_threads():
    from easyicu.research_agent import mcp_server

    for session_id in list(mcp_server._SSE_SESSIONS):
        mcp_server._close_sse_session(session_id)

    created: list[str] = []
    lock = threading.Lock()

    def _worker():
        got = mcp_server._open_sse_session()
        if got is not None:
            with lock:
                created.append(got[0])

    threads = [threading.Thread(target=_worker) for _ in range(64)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    try:
        # The cap holds even when 64 threads race for it.
        assert len(created) == mcp_server.MAX_SSE_SESSIONS
        assert len(set(created)) == len(created)
    finally:
        for session_id in list(mcp_server._SSE_SESSIONS):
            mcp_server._close_sse_session(session_id)


# ---------------------------------------------------------------------------
# P1.5 — LLM debug dump wrote full prompts world-readable and unbounded
# ---------------------------------------------------------------------------


def test_p1_5_debug_messages_are_truncated():
    from easyicu.research_agent.providers.llm import (
        LLM_DEBUG_FIELD_CHARS,
        _truncated_debug_messages,
    )

    long = "x" * (LLM_DEBUG_FIELD_CHARS * 3)
    [out] = _truncated_debug_messages([{"role": "user", "content": long}])
    assert len(out["content"]) < len(long)
    assert "chars total" in out["content"]
    assert out["role"] == "user"


def test_p1_5_debug_dump_is_owner_only(tmp_path, monkeypatch):
    from easyicu.research_agent.providers import llm as llm_module

    monkeypatch.setenv("EASYICU_LLM_DEBUG", "1")
    monkeypatch.setenv("EASYICU_LLM_DEBUG_DIR", str(tmp_path / "dbg"))

    source = Path(llm_module.__file__).read_text(encoding="utf-8")
    assert "os.chmod(log_dir, 0o700)" in source
    assert "os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600" in source
    # The raw message is bounded rather than dumped whole.
    assert '"raw_message":' not in source


# ---------------------------------------------------------------------------
# P1.6 / P2.1 / P2.2 — config immutability, version, error surface
# ---------------------------------------------------------------------------


def test_p1_6_pipeline_config_is_frozen(tmp_path):
    config = PipelineConfig(workdir=tmp_path)

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.workdir = tmp_path / "elsewhere"

    changed = config.with_overrides(enable_latex=not config.enable_latex)
    assert changed is not config
    assert changed.canonical_digest() != config.canonical_digest()


def test_p1_6_canonical_digest_is_stable_and_json_safe(tmp_path):
    class _Client:
        pass

    config = PipelineConfig(workdir=tmp_path, llm=_Client())
    payload = config.canonical_payload()

    json.dumps(payload)  # must not raise
    assert payload["llm"].startswith("<")
    assert (
        config.canonical_digest()
        == PipelineConfig(workdir=tmp_path, llm=_Client()).canonical_digest()
    )


def test_p2_1_mcp_server_reports_the_installed_package_version():
    from importlib.metadata import version

    from easyicu.research_agent.mcp_server import SERVER_INFO

    assert SERVER_INFO["version"] == version("easyicu")
    assert SERVER_INFO["version"] != "0.1.0"


def test_p2_2_dispatch_does_not_echo_internal_exception_text(roots, monkeypatch):
    import easyicu

    from easyicu.research_agent import mcp_server

    def _boom(**kwargs):
        raise RuntimeError(
            "no such table 'chartevents' at /Volumes/secret/db/miiv/chartevents.parquet"
        )

    monkeypatch.setattr(easyicu, "load_concepts", _boom, raising=False)

    result = mcp_server.dispatch(
        "research_agent.load_concepts",
        {"concepts": ["hr"], "data_path": str(roots / "miiv")},
    )

    assert result["error_code"] == "tool_failed"
    assert result["error_type"] == "RuntimeError"
    assert "chartevents" not in json.dumps(result)
    assert "/Volumes/secret" not in json.dumps(result)


def test_p2_2_unknown_tool_and_bad_argument_have_stable_codes(roots):
    from easyicu.research_agent.mcp_server import dispatch

    assert dispatch("research_agent.nope", {})["error_code"] == "unknown_tool"

    escaped = dispatch(
        "research_agent.bind_evidence",
        {"workdir": "/etc", "kind": "log", "text": "x"},
    )
    assert escaped["error_code"] == "path_not_allowed"
