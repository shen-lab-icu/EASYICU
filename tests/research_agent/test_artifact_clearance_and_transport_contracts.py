"""Artifact clearance, resume and transport-authority contract tests.

Origin: 2026-07-28 external review (seventh pass).

The through-line of the last three rounds, in order: the gate was right but
nothing called it; the test called the gate but fed it a shape production never
emits; and now — the gate is called, with the real shape, but it reads a field
*the caller controls*. ``metadata["aggregate_only"] is True`` is a string in a
dict, and every path that can write a record can write that string.

So the figure tests here build a **real** ``EvidenceStore``, register real
artefacts through it, and run the real audit; nothing is a fake with a
convenient attribute. Where a fake is unavoidable (a provider that must raise
mid-upload) it stands in for the network, never for an authority.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared: a real evidence store holding a real, cleared figure
# ---------------------------------------------------------------------------


def _store(tmp_path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore

    return EvidenceStore(tmp_path / "run")


def _contract(figure_id="Figure2", roles=("primary_estimand",), sources=()):
    return SimpleNamespace(
        figure_id=figure_id,
        core_claim="SOFA-2 predicts in-hospital mortality.",
        statistics_note=None,
        image_integrity_note=None,
        panels=[
            SimpleNamespace(role=role, title=f"Panel {i}", claim="A claim.")
            for i, role in enumerate(roles)
        ],
        source_data=list(sources),
    )


def _aggregate_source(store, tmp_path, *, evidence_id="summary"):
    """A source table that a correct host audit should clear."""

    payload = {
        "n_patients": 9421,
        "auroc": 0.81,
        "p_value": 0.000001,
        "strata": [{"label": "high", "n": 4200}, {"label": "low", "n": 5221}],
    }
    return store.register_json(
        kind="statistic",
        description="Aggregate summary for the figure.",
        payload=payload,
        filename="summary.json",
        evidence_id=evidence_id,
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
    )


def _cleared_figure(tmp_path, *, figure_id="Figure2"):
    """Register a source, audit it, and register the figure the way skill.py does.

    Returns ``(store, run_dir, figure_record, audit)``.
    """

    from easyicu.research_agent.gates.figure_privacy import audit_figure_privacy
    from easyicu.research_agent.figures.skill import AGGREGATE_ONLY_PANEL_ROLES

    store = _store(tmp_path)
    run_dir = Path(store.root)
    source = _aggregate_source(store, tmp_path)
    contract = _contract(figure_id=figure_id, sources=(source.evidence_id,))

    audit = audit_figure_privacy(
        contract=contract,
        evidence=store,
        run_dir=run_dir,
        source_evidence_ids=[source.evidence_id],
        allowed_panel_roles=AGGREGATE_ONLY_PANEL_ROLES,
    )
    assert audit.aggregate_only is True, audit.reasons

    audit_record = store.register_json(
        kind="log",
        description="Host privacy audit.",
        payload=audit.as_receipt(),
        filename=f"figure_privacy_audit_{figure_id}.json",
        evidence_id=f"figure_privacy_audit_{figure_id}",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
    )

    image = tmp_path / "figure.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"pixels")
    figure = store.register_file(
        kind="figure",
        description="Publication figure export.",
        source_path=image,
        evidence_id="publication_figure_png",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
        metadata={
            "figure_id": figure_id,
            "source_evidence_ids": [source.evidence_id],
            "source_evidence_sha256": {source.evidence_id: source.sha256},
            "figure_privacy_audit_evidence_id": audit_record.evidence_id,
            **audit.as_metadata(),
        },
    )
    return store, run_dir, figure, audit


def _policy(store, run_dir):
    from easyicu.research_agent.gates.figure_egress import FigureEgressPolicy

    return FigureEgressPolicy(
        allow_external_upload=True, evidence=store, run_dir=run_dir
    )


def _authorize(store, run_dir, figure):
    from easyicu.research_agent.gates.figure_egress import authorize_figure_upload

    path = run_dir / figure.relative_path
    return authorize_figure_upload(
        [path], policy=_policy(store, run_dir), destination="external"
    )


# ---------------------------------------------------------------------------
# P0-1 — the flag is an index into the audit, not the authorization
# ---------------------------------------------------------------------------


def test_a_host_audited_figure_still_uploads(tmp_path):
    """The production path must survive the tightening."""

    store, run_dir, figure, _ = _cleared_figure(tmp_path)
    entries = _authorize(store, run_dir, figure)

    assert len(entries) == 1
    assert entries[0]["privacy_audit_evidence_id"] == "figure_privacy_audit_Figure2"
    assert entries[0]["privacy_audit_version"]
    assert entries[0]["transport"] == "authorized"


def test_a_self_declared_aggregate_only_flag_does_not_authorize(tmp_path):
    """The bypass the review named: register a PNG, claim the flag, ship it.

    Nothing here is malformed. The record is a real ``kind="figure"`` record in
    a real store, inside the run directory, hashing to its registered digest —
    every check the gate had before this. The only thing it lacks is an audit.
    """

    from easyicu.research_agent.gates.figure_egress import (
        FigureEgressError,
        authorize_figure_upload,
    )

    store = _store(tmp_path)
    run_dir = Path(store.root)
    image = tmp_path / "sneaky.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"one marker per stay")
    record = store.register_file(
        kind="figure",
        description="An unaudited figure.",
        source_path=image,
        evidence_id="external_figure",
        producer="mcp_external_agent",
        generation_mode="external",
        metadata={"aggregate_only": True},
    )

    with pytest.raises(FigureEgressError, match="not cleared for external upload"):
        authorize_figure_upload(
            [run_dir / record.relative_path],
            policy=_policy(store, run_dir),
            destination="external",
        )


@pytest.mark.parametrize(
    "drop_key",
    [
        "aggregate_only_basis",
        "aggregate_only_audit_version",
        "figure_privacy_audit_evidence_id",
    ],
)
def test_every_link_of_the_audit_chain_is_required(tmp_path, drop_key):
    from easyicu.research_agent.gates.figure_egress import (
        FigureEgressError,
        authorize_figure_upload,
    )

    store, run_dir, figure, audit = _cleared_figure(tmp_path)
    metadata = dict(figure.metadata)
    metadata.pop(drop_key, None)
    image = tmp_path / "copy.png"
    image.write_bytes((run_dir / figure.relative_path).read_bytes() + b"x")
    weakened = store.register_file(
        kind="figure",
        description="Same figure, one link removed.",
        source_path=image,
        evidence_id="weakened_figure",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
        metadata=metadata,
    )

    with pytest.raises(FigureEgressError):
        authorize_figure_upload(
            [run_dir / weakened.relative_path],
            policy=_policy(store, run_dir),
            destination="external",
        )


def test_an_untrusted_producer_cannot_carry_a_real_audit(tmp_path):
    """Copying a genuine audit's metadata onto an externally-produced record."""

    from easyicu.research_agent.gates.figure_egress import (
        FigureEgressError,
        authorize_figure_upload,
    )

    store, run_dir, figure, _ = _cleared_figure(tmp_path)
    image = tmp_path / "borrowed.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"borrowed")
    borrowed = store.register_file(
        kind="figure",
        description="An external artefact wearing the host audit's metadata.",
        source_path=image,
        evidence_id="borrowed_figure",
        producer="mcp_external_agent",
        generation_mode="external",
        metadata=dict(figure.metadata),
    )

    with pytest.raises(FigureEgressError, match="host figure producers"):
        authorize_figure_upload(
            [run_dir / borrowed.relative_path],
            policy=_policy(store, run_dir),
            destination="external",
        )


def test_a_rewritten_audit_receipt_is_caught(tmp_path):
    """Flipping the verdict on disk must not flip the authorization."""

    from easyicu.research_agent.gates.figure_egress import FigureEgressError

    store, run_dir, figure, _ = _cleared_figure(tmp_path)
    audit_record = store.get("figure_privacy_audit_Figure2")
    path = run_dir / audit_record.relative_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["reasons"] = []
    payload["figure_id"] = "SomeOtherFigure"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FigureEgressError, match="registered digest"):
        _authorize(store, run_dir, figure)


def _figure_with_metadata(store, tmp_path, metadata, *, evidence_id, marker):
    image = tmp_path / f"{evidence_id}.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + marker)
    return store.register_file(
        kind="figure",
        description="Publication figure export.",
        source_path=image,
        evidence_id=evidence_id,
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
        metadata=metadata,
    )


def test_a_source_the_audit_never_inspected_is_caught(tmp_path):
    """A clearance describes the artefacts that were read, not their ids."""

    from easyicu.research_agent.gates.figure_egress import (
        FigureEgressError,
        authorize_figure_upload,
    )

    store, run_dir, figure, _ = _cleared_figure(tmp_path)
    metadata = dict(figure.metadata)
    metadata["source_evidence_ids"] = ["summary", "a_source_added_later"]
    metadata["source_evidence_sha256"] = {
        **dict(metadata["source_evidence_sha256"]),
        "a_source_added_later": "f" * 64,
    }
    widened = _figure_with_metadata(
        store, tmp_path, metadata, evidence_id="widened_figure", marker=b"widened"
    )

    with pytest.raises(FigureEgressError, match="did not inspect source"):
        authorize_figure_upload(
            [run_dir / widened.relative_path],
            policy=_policy(store, run_dir),
            destination="external",
        )


def test_metadata_disagreeing_with_the_audit_is_caught(tmp_path):
    """The figure's fingerprints and the audit's must be the same fingerprints."""

    from easyicu.research_agent.gates.figure_egress import (
        FigureEgressError,
        authorize_figure_upload,
    )

    store, run_dir, figure, _ = _cleared_figure(tmp_path)
    metadata = dict(figure.metadata)
    metadata["source_evidence_sha256"] = {"summary": "d" * 64}
    forged = _figure_with_metadata(
        store, tmp_path, metadata, evidence_id="forged_figure", marker=b"forged"
    )

    with pytest.raises(FigureEgressError, match="disagree about source"):
        authorize_figure_upload(
            [run_dir / forged.relative_path],
            policy=_policy(store, run_dir),
            destination="external",
        )


def test_mcp_bind_evidence_refuses_the_reserved_metadata_keys(tmp_path, monkeypatch):
    """Close the door the bypass came through."""

    from easyicu.research_agent import mcp_server
    from easyicu.research_agent.mcp_policy import MCP_SCOPES_ENV

    monkeypatch.setenv(MCP_SCOPES_ENV, "metadata,bind_evidence")
    monkeypatch.setenv("EASYICU_MCP_ALLOWED_ROOTS", str(tmp_path))

    result = mcp_server._tool_bind_evidence(
        {
            "workdir": str(tmp_path / "wd"),
            "kind": "figure",
            "text": "not really a figure",
            "metadata": {"aggregate_only": True, "aggregate_only_basis": "whatever"},
        }
    )
    assert "error" in result
    assert "host privacy audit" in result["error"]


def test_mcp_bind_evidence_refuses_to_impersonate_the_host_producer(
    tmp_path, monkeypatch
):
    from easyicu.research_agent import mcp_server
    from easyicu.research_agent.mcp_policy import MCP_SCOPES_ENV

    monkeypatch.setenv(MCP_SCOPES_ENV, "metadata,bind_evidence")
    monkeypatch.setenv("EASYICU_MCP_ALLOWED_ROOTS", str(tmp_path))

    result = mcp_server._tool_bind_evidence(
        {
            "workdir": str(tmp_path / "wd"),
            "kind": "figure",
            "text": "not really a figure",
            "producer": "publication_figure_skill",
        }
    )
    assert "error" in result
    assert "in-process host producer" in result["error"]


# ---------------------------------------------------------------------------
# P0-2 — the audit reads values, not only names
# ---------------------------------------------------------------------------


def _audit_one_source(tmp_path, filename, writer):
    from easyicu.research_agent.gates.figure_privacy import audit_figure_privacy

    store = _store(tmp_path)
    run_dir = Path(store.root)
    staged = tmp_path / filename
    writer(staged)
    record = store.register_file(
        kind="table",
        description="Figure source.",
        source_path=staged,
        evidence_id="src",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
    )
    return (
        audit_figure_privacy(
            contract=_contract(sources=("src",)),
            evidence=store,
            run_dir=run_dir,
            source_evidence_ids=["src"],
        ),
        store,
        run_dir,
        record,
    )


def test_csv_identifier_values_are_caught_though_the_columns_look_aggregate(tmp_path):
    """``label,predicted`` is a clean header over one row per stay."""

    def write(path):
        path.write_text(
            "label,predicted\npatient_30042318,0.82\npatient_30042319,0.44\n",
            encoding="utf-8",
        )

    audit, *_ = _audit_one_source(tmp_path, "scores.csv", write)
    assert audit.aggregate_only is False
    assert any("identifier-shaped value" in reason for reason in audit.reasons)


def test_json_identifier_values_inside_a_list_are_caught(tmp_path):
    def write(path):
        path.write_text(
            json.dumps({"labels": ["patient 30042318", "patient 30042319"]}),
            encoding="utf-8",
        )

    audit, *_ = _audit_one_source(tmp_path, "labels.json", write)
    assert audit.aggregate_only is False
    assert any("identifier-shaped value" in reason for reason in audit.reasons)


def test_parquet_small_cells_are_caught_not_just_its_schema(tmp_path):
    """A schema-only read clears ``subgroup, n`` whatever ``n`` says."""

    def write(path):
        pd.DataFrame({"subgroup": ["a", "b"], "n": [3, 41]}).to_parquet(path)

    audit, *_ = _audit_one_source(tmp_path, "strata.parquet", write)
    assert audit.aggregate_only is False
    assert any("group size" in reason for reason in audit.reasons)


def test_parquet_identifier_values_are_caught(tmp_path):
    def write(path):
        pd.DataFrame(
            {"label": ["30042318", "30042319"], "risk": [0.2, 0.4]}
        ).to_parquet(path)

    audit, *_ = _audit_one_source(tmp_path, "scores.parquet", write)
    assert audit.aggregate_only is False
    assert any("identifier-shaped value" in reason for reason in audit.reasons)


def test_a_genuine_aggregate_source_is_not_flagged(tmp_path):
    """The failure direction is deliberate, but it must not be everything.

    Large cohort totals and a six-decimal p-value are ordinary content in a
    summary table; flagging them would refuse every real figure and the gate
    would be turned off within a week.
    """

    def write(path):
        path.write_text(
            "stratum,n,estimate,p_value\nhigh,4200,0.81,0.000001\n"
            "low,5221,0.77,0.000123\n",
            encoding="utf-8",
        )

    audit, *_ = _audit_one_source(tmp_path, "aggregate.csv", write)
    assert audit.aggregate_only is True, audit.reasons


def test_the_audit_rehashes_before_it_reads(tmp_path):
    """Inspecting the current bytes under an older digest proves nothing."""

    from easyicu.research_agent.gates.figure_privacy import audit_figure_privacy

    audit, store, run_dir, record = _audit_one_source(
        tmp_path,
        "aggregate.csv",
        lambda p: p.write_text("stratum,n\nhigh,4200\n", encoding="utf-8"),
    )
    assert audit.aggregate_only is True

    (run_dir / record.relative_path).write_text(
        "stay_id,risk\n30042318,0.5\n", encoding="utf-8"
    )
    after = audit_figure_privacy(
        contract=_contract(sources=("src",)),
        evidence=store,
        run_dir=run_dir,
        source_evidence_ids=["src"],
    )
    assert after.aggregate_only is False
    assert any("registered digest" in reason for reason in after.reasons)


def test_the_audit_does_not_echo_the_identifier_it_found(tmp_path):
    """The receipt is read by humans and stored; it must not carry the value."""

    from easyicu.research_agent.gates.figure_privacy import audit_figure_privacy

    store = _store(tmp_path)
    run_dir = Path(store.root)
    source = store.register_json(
        kind="statistic",
        description="Aggregate summary.",
        payload={"n_patients": 4200},
        filename="summary.json",
        evidence_id="summary",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
    )
    contract = _contract(sources=(source.evidence_id,))
    contract.panels[0].title = "Cohort 30042318 trajectory"

    audit = audit_figure_privacy(
        contract=contract,
        evidence=store,
        run_dir=run_dir,
        source_evidence_ids=[source.evidence_id],
    )
    assert audit.aggregate_only is False
    blob = json.dumps(audit.as_receipt()) + json.dumps(audit.as_metadata())
    assert "30042318" not in blob
    assert "8-digit token" in blob


# ---------------------------------------------------------------------------
# P0-3 — a cited step that owns no such metric is an error, not a fallback
# ---------------------------------------------------------------------------


def test_a_sensitivity_step_cannot_vouch_for_the_primary_model(tmp_path):
    """The exact scenario the review described.

    The sentence cites the primary step. The primary step registers no Brier
    score. The sensitivity step does, and it happens to match. Falling back to
    match-any let the wrong step's number back the claim.
    """

    from easyicu.research_agent.audits.manuscript_claims import (
        _cited_step_lacks_metric_finding,
        _scoped_registered_values,
    )

    scope = _scoped_registered_values(
        summaries=[{"brier_score": 0.11}],
        summary_owners=["step_sensitivity"],
        keys=("brier_score",),
        footnote_steps={"f1": "step_primary"},
        footnote_id="f1",
    )
    assert scope.cited_step == "step_primary"
    assert scope.values == []
    assert scope.cited_step_lacks_metric is True

    finding = _cited_step_lacks_metric_finding(
        metric="brier_score",
        label="Brier score",
        claimed=0.11,
        cited_step=scope.cited_step,
    )
    assert finding.severity == "error"
    assert finding.detail["reason"] == "cited_step_does_not_register_metric"
    assert "step_primary" in finding.message


def test_an_unfootnoted_claim_still_falls_back_to_match_any():
    """The other failure keeps its own finding; do not merge the two."""

    from easyicu.research_agent.audits.manuscript_claims import (
        _scoped_registered_values,
    )

    scope = _scoped_registered_values(
        summaries=[{"brier_score": 0.11}],
        summary_owners=["step_sensitivity"],
        keys=("brier_score",),
        footnote_steps={},
        footnote_id=None,
    )
    assert scope.cited_step is None
    assert scope.cited_step_lacks_metric is False


def test_a_cited_step_that_owns_the_metric_scopes_to_it():
    from easyicu.research_agent.audits.manuscript_claims import (
        _scoped_registered_values,
    )

    scope = _scoped_registered_values(
        summaries=[{"brier_score": 0.11}, {"brier_score": 0.30}],
        summary_owners=["step_primary", "step_sensitivity"],
        keys=("brier_score",),
        footnote_steps={"f1": "step_primary"},
        footnote_id="f1",
    )
    assert scope.values == [0.11]
    assert scope.step_id == "step_primary"
    assert scope.cited_step_lacks_metric is False


def test_the_ci_check_no_longer_falls_back_to_every_step():
    """Source-level: the ``or list(summaries)`` fallback is gone."""

    import inspect

    from easyicu.research_agent.audits import manuscript_claims

    source = inspect.getsource(manuscript_claims.audit_manuscript_numeric_claims)
    assert "] or list(summaries)" not in source
    assert "if ci_steps:" in source


# ---------------------------------------------------------------------------
# P0-4 / P0-5 — the human-review boundary, declared and fail-closed
# ---------------------------------------------------------------------------


def test_the_pause_declares_its_own_resume_scope():
    from easyicu.research_agent.orchestration.workflow import (
        HUMAN_REVIEW_RESUME_SCOPE,
        HumanReviewPending,
        HumanReviewRequest,
    )

    request = HumanReviewRequest.create(
        kind="capability_request",
        summary="capability review",
        authority_sha256="a" * 64,
        payload={},
    )
    pending = HumanReviewPending(
        run_id="run-1", thread_id="run-1", run_dir="/tmp/run-1", requests=(request,)
    )

    assert pending.resume_scope == HUMAN_REVIEW_RESUME_SCOPE == "same_process"
    assert pending.resume_pid == os.getpid()
    assert pending.resumable_here is True
    # Machine-readable: a UI can refuse before prompting a human.
    assert "resume_scope" in pending.model_dump()


def test_a_pause_from_another_process_is_refused_not_mis_resumed():
    from easyicu.research_agent.orchestration.workflow import (
        HumanReviewPending,
        HumanReviewRequest,
    )

    request = HumanReviewRequest.create(
        kind="capability_request",
        summary="capability review",
        authority_sha256="a" * 64,
        payload={},
    )
    pending = HumanReviewPending(
        run_id="run-1",
        thread_id="run-1",
        run_dir="/tmp/run-1",
        requests=(request,),
        resume_pid=os.getpid() + 1,
    )
    assert pending.resumable_here is False


def test_resume_on_a_fresh_instance_names_the_boundary(tmp_path):
    from easyicu.research_agent.pipeline import ResearchAgentPipeline
    from easyicu.research_agent.providers.mocks import MockLLMClient

    agent = ResearchAgentPipeline(workdir=tmp_path / "wd", llm=MockLLMClient())
    with pytest.raises(RuntimeError, match="same_process"):
        agent.resume_human_review([])


def test_unreadable_review_evidence_fails_closed():
    """No evidence binding, no review request — an approval must bind something."""

    from easyicu.research_agent.orchestration.workflow import (
        HumanReviewAuthorityError,
        human_review_requests_for_plan,
    )
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        ValidationFinding,
    )

    class _BrokenEvidence:
        def records(self):
            raise OSError("evidence.json is corrupt")

    finding = ValidationFinding(
        validator="capability_gate",
        severity="error",
        message="This plan requests an unregistered capability.",
        detail={"reason": "capability_review_required"},
    )
    with pytest.raises(HumanReviewAuthorityError, match="binds no evidence"):
        human_review_requests_for_plan(
            findings=[finding],
            plan=AnalysisPlan(
                research_question="Can this capability be used?",
                steps=[
                    AnalysisStep(
                        step_id="s1",
                        intent="Run the requested registered analysis.",
                    )
                ],
            ),
            evidence=_BrokenEvidence(),
        )


# ---------------------------------------------------------------------------
# P0-6 — the whole window contract, not one column name
# ---------------------------------------------------------------------------


def _win(dur_col, unit, *, ids=("stay_id",), index="t"):
    from easyicu.table import WinTbl
    from easyicu.table.duration import set_dur_var_unit

    frame = pd.DataFrame({ids[0]: [1, 2], index: [0.0, 1.0], dur_col: [10.0, 20.0]})
    set_dur_var_unit(frame, unit)
    return WinTbl(frame, list(ids), index, dur_col, dur_unit=unit)


def test_a_differently_named_duration_no_longer_slips_past(tmp_path):
    """The review's scenario: the unit check has no column to check.

    ``left.dur_var = duration_hours``, ``right.dur_var = duration_minutes``.
    The right frame has no ``duration_hours``, so it was skipped, its declared
    minutes were dropped, and its duration became an ordinary column under a
    table claiming to be windowed over the other one.
    """

    from easyicu.table import rbind_tbl
    from easyicu.table.duration import WindowContractError

    with pytest.raises(WindowContractError, match="different dur_var"):
        rbind_tbl(_win("duration_hours", "hours"), _win("duration_minutes", "minutes"))


def test_a_different_index_var_is_refused():
    # Via rbind: on the column-bind path the 2026-07-29 key-alignment check
    # now refuses this pair one step earlier, because the second table does not
    # carry the first's index column at all. The window contract itself is what
    # this test is about, so it exercises the path that still reaches it.
    from easyicu.table import rbind_tbl
    from easyicu.table.duration import WindowContractError

    with pytest.raises(WindowContractError, match="different index_var"):
        rbind_tbl(_win("dur", "hours"), _win("dur", "hours", index="charttime"))


def test_a_different_id_var_is_refused():
    from easyicu.table import rbind_tbl
    from easyicu.table.duration import WindowContractError

    with pytest.raises(WindowContractError, match="different id_vars"):
        rbind_tbl(_win("dur", "hours"), _win("dur", "hours", ids=("subject_id",)))


def test_an_agreeing_window_combine_still_works():
    from easyicu.table import rbind_tbl

    out = rbind_tbl(_win("dur", "hours"), _win("dur", "hours"))
    assert out.dur_unit == "hours"
    assert len(out.data) == 4


def test_a_covariate_frame_is_still_not_a_window_input():
    """Only window tables are party to the window contract."""

    from easyicu.table import cbind_tbl

    covariates = pd.DataFrame({"age": [60, 71], "sex": ["F", "M"]})
    out = cbind_tbl(_win("dur", "hours"), covariates)
    assert out.dur_unit == "hours"


# ---------------------------------------------------------------------------
# P1 — authorization is not transport; detail values are not just keys
# ---------------------------------------------------------------------------


def test_a_failed_send_is_recorded_as_failed_not_as_sent(tmp_path):
    """Through the real adapter: authorized, then the send path raises."""

    from easyicu.research_agent.gates.visual_qa import VLMVisualQAAdapter

    store, run_dir, figure, _ = _cleared_figure(tmp_path)
    policy = _policy(store, run_dir)

    class _UnauthorizedVLM:
        # No factory authorization, so the send raises before any byte moves —
        # which is precisely a send that did not happen and must not be
        # counted as one.
        def complete(self, *a, **k):
            return "{}"

        def complete_with_images(self, *a, **k):  # pragma: no cover - never reached
            return "{}"

    findings = VLMVisualQAAdapter(_UnauthorizedVLM(), egress_policy=policy).audit(
        figure_paths=[run_dir / figure.relative_path]
    )

    assert policy.transport_summary() == {"transport_failed": 1}
    assert any("visual QA failed" in f.message for f in findings)


def test_the_adapter_closes_the_transport_loop_on_both_paths():
    """Source-level: neither branch may leave an entry at ``authorized``.

    There is no built-in provider mock that both carries factory authorization
    *and* implements ``complete_with_images``, so the success branch cannot be
    driven end to end without minting an authority a test has no business
    minting. What is checkable is that the success branch exists and closes the
    loop — asserted here, with the policy-level behaviour covered separately.
    """

    import inspect

    from easyicu.research_agent.gates import visual_qa

    source = inspect.getsource(visual_qa.VLMVisualQAAdapter.audit)
    assert "_close_transport(authorized, TRANSPORT_FAILED)" in source
    assert "_close_transport(authorized, TRANSPORT_COMPLETED)" in source
    assert source.index("TRANSPORT_FAILED") < source.index("TRANSPORT_COMPLETED"), (
        "the failure path must be installed before the call it guards"
    )


def test_the_policy_records_a_completed_send(tmp_path):
    from easyicu.research_agent.gates.figure_egress import (
        TRANSPORT_COMPLETED,
        FigureEgressPolicy,
    )

    policy = FigureEgressPolicy(allow_external_upload=True)
    # The rows the policy hands back are the attempts; closing an attempt means
    # naming it, not describing an image that resembles it (2026-07-29).
    recorded = policy.record_upload([{"path": "a.png", "sha256": "a" * 64}])
    assert policy.transport_summary() == {"authorized": 1}

    policy.record_transport_outcome(recorded, TRANSPORT_COMPLETED)
    assert policy.transport_summary() == {"transport_completed": 1}


def test_the_receipt_distinguishes_the_five_states(tmp_path):
    from easyicu.research_agent.gates.figure_egress import (
        TRANSPORT_COMPLETED,
        FigureEgressPolicy,
        register_figure_egress_receipt,
    )

    class _Store:
        def register_file(self, **kwargs):
            return SimpleNamespace(evidence_id=kwargs["evidence_id"])

    policy = FigureEgressPolicy(allow_external_upload=True)
    recorded = policy.record_upload(
        [{"path": "a.png", "sha256": "a" * 64}, {"path": "b.png", "sha256": "b" * 64}]
    )
    policy.record_transport_outcome(recorded[:1], TRANSPORT_COMPLETED)

    register_figure_egress_receipt(
        policy=policy, evidence=_Store(), run_dir=tmp_path, phase="completed"
    )
    payload = json.loads(
        (tmp_path / "figure_egress_receipt.json").read_text(encoding="utf-8")
    )
    assert payload["schema"] == "easyicu.figure_egress_receipt/3"
    assert payload["transport_counts"] == {
        "transport_completed": 1,
        "transport_unknown": 1,
    }


def test_an_unknown_transport_outcome_is_rejected():
    from easyicu.research_agent.gates.figure_egress import FigureEgressPolicy

    policy = FigureEgressPolicy(allow_external_upload=True)
    with pytest.raises(ValueError, match="unknown figure transport outcome"):
        policy.record_transport_outcome([], "probably_fine")


@pytest.mark.parametrize(
    "detail, expected",
    [
        ({"duplicate_count": 3}, {"duplicate_count": "<20"}),
        ({"duplicate_count": 412}, {"duplicate_count": 412}),
        (
            {"reason": "missing /Users/clinician/phi/cohort_v3.parquet"},
            {"reason": "missing cohort_v3.parquet"},
        ),
        (
            {"fallback": "dropped stay 30042318 from the cohort"},
            {"fallback": "dropped stay <id> from the cohort"},
        ),
        ({"reason": {"nested": "shape"}}, {"reason": "<withheld>"}),
    ],
)
def test_mcp_finding_detail_values_are_checked_not_just_their_keys(detail, expected):
    """An allow-listed key says nothing about what a validator wrote into it."""

    from easyicu.research_agent.mcp_server import _safe_finding_payload

    payload = _safe_finding_payload(
        {"validator": "cohort_auditor", "severity": "warning", "detail": detail}
    )
    assert payload["detail"] == expected
