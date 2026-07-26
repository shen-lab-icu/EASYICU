"""Pipeline projection, review, evidence and profile-authority regressions.

Origin: 2026-07-26 deep re-review.

The theme of that review was not "this component is wrong" but "this component
is right and nothing calls it": a gate that only a hand-built test fixture can
satisfy, a graph primitive the production entrypoint never wires, a numeric
binder that proves a value exists somewhere rather than in the step the
sentence names.

Every test here therefore drives the *production* path — the real dispatcher,
the real ``ResearchAgentPipeline.run()``, the real ``build_command()`` — and
asserts on observable behaviour rather than on the presence of a line of
source. One test in the previous round did the latter and was called out for
it; ``test_p1_5_debug_dump_is_owner_only_on_disk`` replaces it.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.mcp_policy import (
    MCP_ALLOWED_ROOTS_ENV,
    MCP_AUDIT_ROOT_ENV,
    MCP_SCOPES_ENV,
    SCOPE_METADATA,
    SCOPE_READ_INTERNAL_CONTEXT,
    SCOPE_READ_PATIENT_DATA,
    DisclosurePolicy,
    summarise_frame,
)
from easyicu.research_agent.mcp_server import dispatch
from easyicu.research_agent.providers.mocks import MockLLMClient

pd = pytest.importorskip("pandas")


@pytest.fixture(autouse=True)
def _mcp_roots(tmp_path, monkeypatch):
    """Declare tmp_path as an allowed root and keep audits out of the repo."""

    monkeypatch.setenv(MCP_ALLOWED_ROOTS_ENV, str(tmp_path))
    monkeypatch.setenv(MCP_AUDIT_ROOT_ENV, str(tmp_path / "audit"))


# ---------------------------------------------------------------------------
# P0-1 — MCP metadata tools bypassed the outbound projection
# ---------------------------------------------------------------------------


def _cohort_parquet(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "stay_id": range(1, 41),
            "subject_id": range(101, 141),
            "intime": pd.date_range("2026-01-01", periods=40, freq="h"),
            "sofa2": [3.0 + (i % 5) for i in range(40)],
            "death": [i % 4 == 0 for i in range(40)],
        }
    )
    path = tmp_path / "cohort.parquet"
    frame.to_parquet(path)
    return path


def _build_context_args(tmp_path: Path) -> dict:
    return {
        "question": "Does SOFA-2 predict ICU death?",
        "cohort_path": str(_cohort_parquet(tmp_path)),
        "database": "miiv",
        "target_outcome": "death",
    }


def test_p0_1_build_context_returns_the_outbound_safe_projection(tmp_path):
    payload = dispatch("research_agent.build_context", _build_context_args(tmp_path))

    assert payload.get("schema") == "easyicu.outbound_safe_context/1"
    assert "projection" in payload
    # The internal shape's own fields are the tell: the raw dump carries the
    # cohort parquet path and the free-text preference notes.
    encoded = json.dumps(payload)
    assert "cohort_parquet" not in encoded
    assert "extra_notes" not in encoded
    assert str(tmp_path) not in encoded


def test_p0_1_internal_context_scope_restores_the_raw_shape(tmp_path, monkeypatch):
    monkeypatch.setenv(
        MCP_SCOPES_ENV, f"{SCOPE_METADATA},{SCOPE_READ_INTERNAL_CONTEXT}"
    )

    payload = dispatch("research_agent.build_context", _build_context_args(tmp_path))

    assert payload.get("schema") != "easyicu.outbound_safe_context/1"
    assert "cohort" in payload


def test_p0_1_list_and_describe_concepts_use_the_same_projection(tmp_path):
    args = _build_context_args(tmp_path)
    listed = dispatch("research_agent.list_concepts", args)

    assert listed["projection"]
    encoded = json.dumps(listed)
    assert str(tmp_path) not in encoded
    # ConceptDescriptor's source tables / item ids / clinical caveats are
    # internal provenance and must not ride along with the concept list.
    assert "source_tables" not in encoded
    assert "clinical_caveats" not in encoded

    named = listed["concepts"][0]["name"]
    described = dispatch(
        "research_agent.describe_concept", {**args, "concept_name": named}
    )
    assert described["concept"]["name"] == named
    assert "source_tables" not in json.dumps(described)


def test_p0_1_read_manifest_projects_evidence_index_not_prose(tmp_path):
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu.research_manifest/1",
                "run_id": "run-1",
                "research_question": "Does SOFA-2 predict ICU death?",
                "started_at": "2026-07-26T00:00:00Z",
                "context_path": str(run_dir / "research_context.json"),
                "plan_path": str(run_dir / "analysis_plan.json"),
                "readiness": {
                    "publication_ready": True,
                    "publication_artifacts_ready": True,
                    "execution_paper_eligible": False,
                    "paper_authorized": False,
                },
                "evidence": [
                    {
                        "evidence_id": "primary_model",
                        "kind": "statistic",
                        "description": "Primary model for stay 4023 in ward 7",
                        "relative_path": "evidence/primary_model.json",
                        "sha256": "a" * 64,
                        "metadata": {"private_label": "ward 7"},
                    }
                ],
                "findings": [
                    {
                        "validator": "cohort_auditor",
                        "severity": "warning",
                        "message": "Only 3 stays in the ward-7 stratum",
                    }
                ],
                "per_step_records": [
                    {
                        "step_id": "01",
                        "status": "ok",
                        "stdout": "cohort head: subject_id 10023 ...",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = dispatch(
        "research_agent.read_manifest",
        {"workdir": str(tmp_path / "runs"), "run_id": "run-1"},
    )

    assert payload["run_id"] == "run-1"
    assert payload["evidence"][0]["evidence_id"] == "primary_model"
    assert payload["evidence"][0]["sha256"] == "a" * 64
    # Content readiness is not paper authority. The public MCP projection
    # exposes the explicit execution-bound verdict and must not invite clients
    # to infer authorization from ``publication_ready``.
    assert payload["readiness"] == {
        "publication_artifacts_ready": True,
        "execution_paper_eligible": False,
        "paper_authorized": False,
    }
    assert "publication_ready" not in payload["readiness"]
    encoded = json.dumps(payload)
    # Identity and digests survive; prose, host paths and step stdout do not.
    assert "ward 7" not in encoded
    assert "ward-7" not in encoded
    assert "10023" not in encoded
    assert str(run_dir) not in encoded
    assert len(payload["context_path_sha256"]) == 64


def test_p0_1_aggregate_statistics_honour_a_per_column_cell_floor():
    frame = pd.DataFrame(
        {
            "common_lab": [float(i) for i in range(100)],
            # 100 rows, but one contributing patient: describe() would report
            # count=1 and mean=min=max=that patient's value.
            "rare_assay": [7.25] + [None] * 99,
        }
    )

    summary = summarise_frame(
        frame,
        policy=DisclosurePolicy(
            patient_data=False, preview_rows=0, include_identifier_columns=False
        ),
    )
    stats = summary["aggregate_statistics"]

    assert stats["common_lab"]["count"] == 100.0
    assert stats["rare_assay"]["withheld"] is True
    # Tightened 2026-07-27: the exact size of a suppressed cell is itself a
    # small-cell disclosure, so the bound is reported instead of the count.
    assert stats["rare_assay"]["non_missing_count"] == "<20"
    assert "7.25" not in json.dumps(stats)


# ---------------------------------------------------------------------------
# P1-1 — the patient-data audit was optional and failed open
# ---------------------------------------------------------------------------


def _patch_load_concepts(monkeypatch, frame):
    import easyicu

    monkeypatch.setattr(easyicu, "load_concepts", lambda **kw: frame, raising=False)


def test_p1_1_audit_is_written_without_a_caller_supplied_workdir(tmp_path, monkeypatch):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore

    _patch_load_concepts(monkeypatch, pd.DataFrame({"sofa2": range(30)}))

    dispatch(
        "research_agent.load_concepts",
        {
            "concepts": ["sofa2"],
            "database": "miiv",
            "data_path": str(tmp_path / "miiv"),
        },
    )

    audit_root = tmp_path / "audit" / ".easyicu_mcp_audit"
    records = EvidenceStore(audit_root).records()
    payloads = [
        json.loads((audit_root / record.relative_path).read_text())
        for record in records
        if "mcp_patient_data_access" in str(record.relative_path)
    ]
    assert {payload["event"] for payload in payloads} == {
        "patient_access_requested",
        "patient_access_completed",
    }


def test_p1_1_rows_are_withheld_when_the_audit_cannot_be_written(tmp_path, monkeypatch):
    monkeypatch.setenv(MCP_SCOPES_ENV, f"{SCOPE_METADATA},{SCOPE_READ_PATIENT_DATA}")
    # A file where the audit root must go: mkdir fails, so the trail cannot
    # be written.
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv(MCP_AUDIT_ROOT_ENV, str(blocker))

    called = False

    def _load(**kwargs):
        nonlocal called
        called = True
        return pd.DataFrame({"sofa2": range(30)})

    import easyicu

    monkeypatch.setattr(easyicu, "load_concepts", _load, raising=False)

    result = dispatch(
        "research_agent.load_concepts",
        {
            "concepts": ["sofa2"],
            "database": "miiv",
            "data_path": str(tmp_path / "miiv"),
            "preview_rows": 5,
        },
    )

    assert result.get("error_code") == "scope_not_granted"
    assert "audit" in result["error"]
    assert "summary" not in result
    assert called is False


def test_p1_1_access_intent_exists_before_the_loader_runs(tmp_path, monkeypatch):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore

    audit_root = tmp_path / "audit" / ".easyicu_mcp_audit"

    def _load(**kwargs):
        store = EvidenceStore(audit_root)
        payloads = [
            json.loads((store.root / record.relative_path).read_text())
            for record in store.records()
        ]
        assert [item["event"] for item in payloads] == ["patient_access_requested"]
        return pd.DataFrame({"sofa2": range(30)})

    import easyicu

    monkeypatch.setattr(easyicu, "load_concepts", _load, raising=False)

    result = dispatch(
        "research_agent.load_concepts",
        {
            "concepts": ["sofa2"],
            "database": "miiv",
            "data_path": str(tmp_path / "miiv"),
        },
    )

    assert result["api"] == "easyicu.load_concepts"


def test_p1_1_audit_records_rows_actually_returned_not_the_cap(tmp_path, monkeypatch):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore

    monkeypatch.setenv(MCP_SCOPES_ENV, f"{SCOPE_METADATA},{SCOPE_READ_PATIENT_DATA}")
    _patch_load_concepts(monkeypatch, pd.DataFrame({"sofa2": [1.0, 2.0, 3.0]}))

    dispatch(
        "research_agent.load_concepts",
        {
            "concepts": ["sofa2"],
            "database": "miiv",
            "data_path": str(tmp_path / "miiv"),
            "preview_rows": 20,
        },
    )

    store = EvidenceStore(tmp_path / "audit" / ".easyicu_mcp_audit")
    record = next(
        r for r in store.records() if "patient_access_completed" in str(r.relative_path)
    )
    payload = json.loads(
        (store.root / record.relative_path).read_text(encoding="utf-8")
    )

    assert payload["disclosed_patient_rows_cap"] == 20
    assert payload["disclosed_patient_rows"] == 3


# ---------------------------------------------------------------------------
# P0-2 — human review was implemented but never wired into run()
# ---------------------------------------------------------------------------


def test_p0_2_plan_findings_become_typed_review_requests():
    from easyicu.research_agent.orchestration.workflow import (
        human_review_requests_for_plan,
    )
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        ValidationFinding,
    )

    requests = human_review_requests_for_plan(
        findings=[
            ValidationFinding(
                validator="capability_workflow",
                severity="error",
                message="Required analytical software is unavailable.",
                detail={"reason": "capability_review_required"},
            ),
            ValidationFinding(
                validator="cohort_auditor",
                severity="warning",
                message="Nothing here needs a human.",
            ),
        ],
        plan=AnalysisPlan(
            research_question="Can this capability be used?",
            steps=[
                AnalysisStep(
                    step_id="01",
                    intent="Run the requested registered analysis.",
                )
            ],
        ),
    )

    assert [item.kind for item in requests] == ["capability_request"]
    # The id binds the payload, so a decision cannot be re-pointed at a
    # different request after the fact.
    assert requests[0].review_id.startswith("review-")
    assert requests[0].payload["plan_step_ids"] == ["01"]


def test_p0_2_run_stops_when_a_review_is_due_and_no_gate_is_configured(tmp_path):
    from easyicu.research_agent.orchestration.workflow import (
        build_pipeline_workflow,
    )
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        ValidationFinding,
    )

    blocking = ValidationFinding(
        validator="capability_workflow",
        severity="error",
        message="Required analytical software is unavailable.",
        detail={"reason": "capability_review_required"},
    )
    executed = []

    def _plan_invoker():
        return SimpleNamespace(
            aborted_result=None,
            findings=[blocking],
            plan=AnalysisPlan(
                research_question="Can this capability be used?",
                steps=[
                    AnalysisStep(
                        step_id="01",
                        intent="Run the requested registered analysis.",
                    )
                ],
            ),
        )

    def _human_review_invoker(plan_result):
        from easyicu.research_agent.orchestration.workflow import (
            human_review_requests_for_plan,
        )

        requests = human_review_requests_for_plan(
            findings=plan_result.findings, plan=plan_result.plan
        )
        if requests:
            raise RuntimeError("no human_review_gate is configured")
        return requests

    workflow = build_pipeline_workflow(
        plan_invoker=_plan_invoker,
        execute_invoker=lambda plan: executed.append("execute"),
        write_invoker=lambda plan, ex: None,
        finalise_invoker=lambda plan, ex, wr: None,
        human_review_invoker=_human_review_invoker,
    )

    with pytest.raises(RuntimeError, match="human_review_gate"):
        workflow.start()

    assert executed == []


def test_p0_2_pipeline_run_passes_a_recorder_to_the_workflow(monkeypatch, tmp_path):
    """The production entrypoint must supply the recorder, not just accept one.

    Asserted through ``ResearchAgentPipeline.run()`` rather than by calling
    ``build_pipeline_workflow`` directly: the previous round proved the primitive
    worked while ``run()`` still passed ``human_review_recorder=None``.
    """

    from easyicu.research_agent import pipeline as pipeline_module

    captured = {}

    def _fake_build(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after workflow construction")

    monkeypatch.setattr(
        pipeline_module, "build_pipeline_workflow", _fake_build, raising=False
    )
    import easyicu.research_agent.orchestration.workflow as workflow_module

    monkeypatch.setattr(workflow_module, "build_pipeline_workflow", _fake_build)

    frame = pd.DataFrame(
        {
            "stay_id": range(1, 31),
            "sofa2": [3.0 + (i % 4) for i in range(30)],
            "death": [i % 3 == 0 for i in range(30)],
        }
    )
    cohort = tmp_path / "cohort.parquet"
    frame.to_parquet(cohort)

    agent = pipeline_module.ResearchAgentPipeline(
        workdir=tmp_path / "wd", llm=MockLLMClient()
    )
    with pytest.raises(RuntimeError, match="stop after workflow construction"):
        agent.run(question="Does SOFA-2 predict death?", cohort=cohort)

    assert captured["human_review_invoker"] is not None
    assert captured["human_review_recorder"] is not None


# ---------------------------------------------------------------------------
# P0-3 — numeric values could be bound to another step's evidence
# ---------------------------------------------------------------------------


def _two_model_store(ra, tmp_path: Path):
    store = ra.EvidenceStore(tmp_path)
    for evidence_id, step_id, auroc in (
        ("primary_model", "01_primary", 0.80),
        ("sensitivity_model", "02_sensitivity", 0.85),
    ):
        (tmp_path / f"{evidence_id}.json").write_text(
            json.dumps({"auroc": auroc}), encoding="utf-8"
        )
        store.register_file(
            kind="statistic",
            description=f"{evidence_id} summary",
            source_path=tmp_path / f"{evidence_id}.json",
            evidence_id=evidence_id,
            producer="coder",
            generation_mode="llm",
        )
        store.register_step_summary_numerics(
            step_id=step_id, evidence_id=evidence_id, summary={"auroc": auroc}
        )
    return store


def test_p0_3_a_number_is_not_bound_to_a_step_the_sentence_did_not_cite(ra, tmp_path):
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store = _two_model_store(ra, tmp_path)

    # The number belongs to the sensitivity step; the citation names the
    # primary one. Both are registered, so match-any would happily bind it.
    bound, binding_map, untraced = bind_numeric_values(
        "The primary model achieved an AUROC of 0.85 {evidence:primary_model}.",
        evidence=store,
    )

    assert "0.85" in untraced
    assert not binding_map
    assert "AMBIGUOUS:0.85" in bound


def test_p0_3_the_cited_step_own_value_still_binds(ra, tmp_path):
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store = _two_model_store(ra, tmp_path)

    _bound, binding_map, untraced = bind_numeric_values(
        "The primary model achieved an AUROC of 0.80 {evidence:primary_model}.",
        evidence=store,
    )

    assert not untraced
    claim = next(iter(binding_map.values()))
    assert claim.evidence_id == "primary_model"
    assert claim.step_id == "01_primary"


def test_p0_3_lineage_lets_a_sentence_cite_a_derived_record(ra, tmp_path):
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    (tmp_path / "table.json").write_text(json.dumps({"n": 1234}), encoding="utf-8")
    store.register_file(
        kind="table",
        description="source table",
        source_path=tmp_path / "table.json",
        evidence_id="cohort_table",
        producer="coder",
        generation_mode="llm",
    )
    (tmp_path / "summary.json").write_text(json.dumps({"n": 1234}), encoding="utf-8")
    store.register_file(
        kind="statistic",
        description="derived summary",
        source_path=tmp_path / "summary.json",
        evidence_id="cohort_summary",
        inputs=["cohort_table"],
        producer="coder",
        generation_mode="llm",
    )
    store.register_step_summary_numerics(
        step_id="01", evidence_id="cohort_table", summary={"n": 1234}
    )

    _bound, binding_map, untraced = bind_numeric_values(
        "The cohort comprised 1234 stays {evidence:cohort_summary}.",
        evidence=store,
    )

    assert not untraced
    assert next(iter(binding_map.values())).evidence_id == "cohort_table"


def test_p0_3_auditor_compares_within_the_step_the_binder_resolved():
    from easyicu.research_agent.audits.manuscript_claims import (
        audit_manuscript_numeric_claims,
    )

    records = [
        {
            "step_id": "01_primary",
            "status": "ok",
            "step_summary": {"auroc": 0.80},
        },
        {
            "step_id": "02_sensitivity",
            "status": "ok",
            "step_summary": {"auroc": 0.85},
        },
    ]
    bound = (
        "The primary model achieved an AUROC of 0.85[^claim_1].\n\n"
        "[^claim_1]: value=0.85; step=01_primary; field=auroc; "
        "evidence=primary_model\n"
    )

    findings = audit_manuscript_numeric_claims(bound, per_step_records=records)

    assert [f.detail["scoped_to_step"] for f in findings] == ["01_primary"]
    assert findings[0].severity == "error"


def test_p0_3_auditor_still_accepts_a_correctly_scoped_claim():
    from easyicu.research_agent.audits.manuscript_claims import (
        audit_manuscript_numeric_claims,
    )

    records = [
        {"step_id": "01_primary", "status": "ok", "step_summary": {"auroc": 0.80}},
        {"step_id": "02_sensitivity", "status": "ok", "step_summary": {"auroc": 0.85}},
    ]
    bound = (
        "The sensitivity model achieved an AUROC of 0.85[^claim_1].\n\n"
        "[^claim_1]: value=0.85; step=02_sensitivity; field=auroc; "
        "evidence=sensitivity_model\n"
    )

    assert audit_manuscript_numeric_claims(bound, per_step_records=records) == []


# ---------------------------------------------------------------------------
# P0-4 — 1-2 digit counts were never value-checked
# ---------------------------------------------------------------------------


def test_p0_4_a_wrong_subgroup_count_is_caught(ra, tmp_path):
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    (tmp_path / "subgroup.json").write_text(json.dumps({"n": 41}), encoding="utf-8")
    store.register_file(
        kind="table",
        description="subgroup table",
        source_path=tmp_path / "subgroup.json",
        evidence_id="subgroup_table",
        producer="coder",
        generation_mode="llm",
    )
    store.register_step_summary_numerics(
        step_id="01", evidence_id="subgroup_table", summary={"n": 41}
    )

    with pytest.raises(ra.EvidenceEnforcementError) as excinfo:
        bind_numeric_values(
            "The subgroup included 42 patients {evidence:subgroup_table}.",
            evidence=store,
            enforcement_mode=ra.EvidenceEnforcementMode.STRICT,
        )

    assert "42" in excinfo.value.detail["untraced"]


def test_p0_4_the_true_count_binds(ra, tmp_path):
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    (tmp_path / "subgroup.json").write_text(json.dumps({"n": 41}), encoding="utf-8")
    store.register_file(
        kind="table",
        description="subgroup table",
        source_path=tmp_path / "subgroup.json",
        evidence_id="subgroup_table",
        producer="coder",
        generation_mode="llm",
    )
    store.register_step_summary_numerics(
        step_id="01", evidence_id="subgroup_table", summary={"n": 41}
    )

    _bound, binding_map, untraced = bind_numeric_values(
        "The subgroup included 41 patients {evidence:subgroup_table}.",
        evidence=store,
    )

    assert not untraced
    assert next(iter(binding_map.values())).value == "41"


def test_p0_4_identifiers_and_section_numbers_are_not_treated_as_counts(ra, tmp_path):
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)

    _bound, _map, untraced = bind_numeric_values(
        "We applied SOFA-2 and Sepsis-3 as described in Section 4 (Figure 2).",
        evidence=store,
    )

    assert untraced == []


# ---------------------------------------------------------------------------
# P0-5 — a failed cohort materialisation fell back to the full universe
# ---------------------------------------------------------------------------


def test_p0_5_failed_materialisation_raises_instead_of_using_the_universe(
    monkeypatch, tmp_path
):
    from easyicu.research_agent import pipeline as pipeline_module
    from easyicu.research_agent.cohort.schema import CohortAuthorityError

    monkeypatch.setattr(
        pipeline_module,
        "materialize_locked_analysis_cohort",
        lambda **kwargs: {
            "status": "error",
            "error": "predicate concept 'lactate' is not in the universe",
            "cohort_definition_sha256": "b" * 64,
        },
    )

    frame = pd.DataFrame(
        {
            "stay_id": range(1, 31),
            "sofa2": [3.0 + (i % 4) for i in range(30)],
            "death": [i % 3 == 0 for i in range(30)],
        }
    )
    cohort = tmp_path / "cohort.parquet"
    frame.to_parquet(cohort)

    agent = pipeline_module.ResearchAgentPipeline(
        workdir=tmp_path / "wd", llm=MockLLMClient()
    )
    with pytest.raises(CohortAuthorityError, match="unfiltered universe"):
        agent.run(
            question="Does SOFA-2 predict death in septic patients?",
            cohort=cohort,
            inclusion_criteria=["lactate > 2"],
        )


# ---------------------------------------------------------------------------
# P1-2 — the figure-egress gate could never pass in production
# ---------------------------------------------------------------------------


def test_p1_2_per_subject_panel_roles_do_not_get_the_flag():
    """Superseded by the 2026-07-27 host-owned privacy audit.

    Panel role is now one *input* to the audit rather than the authorization;
    the role condition itself is covered by
    ``test_review_resume_and_egress_authority.py`` together with the source-artefact
    inspection that a role check cannot perform. See that module for the
    replacement.
    """

    pytest.skip("replaced by the host privacy audit in review-resume regressions")


def test_p1_2_metadata_only_fallback_does_not_forward_host_paths(tmp_path):
    from easyicu.research_agent.gates.visual_qa import _figure_metadata

    nested = tmp_path / "runs" / "haibo-study-2026" / "figures"
    nested.mkdir(parents=True)
    figure = nested / "figure2.png"
    figure.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)

    metadata = _figure_metadata(figure)

    assert metadata["path"] == "figure2.png"
    assert "haibo-study-2026" not in json.dumps(metadata)


def test_p1_2_egress_receipt_is_persisted_even_when_nothing_was_uploaded(ra, tmp_path):
    from easyicu.research_agent.gates.figure_egress import (
        FigureEgressPolicy,
        register_figure_egress_receipt,
    )

    store = ra.EvidenceStore(tmp_path)
    policy = FigureEgressPolicy(allow_external_upload=False)

    record = register_figure_egress_receipt(
        policy=policy, evidence=store, run_dir=tmp_path
    )

    assert record is not None
    payload = json.loads(
        (tmp_path / "figure_egress_receipt.json").read_text(encoding="utf-8")
    )
    # Schema moved to /2 on 2026-07-27 when the receipt became two-phase (an
    # intent record before upload, a completed record after), and to /3 on
    # 2026-07-28 when each entry gained its own transport outcome — an
    # authorized upload and a delivered one are different facts.
    assert payload["schema"] == "easyicu.figure_egress_receipt/3"
    assert payload["phase"] == "completed"
    assert payload["authorized_count"] == 0
    assert payload["transport_counts"] == {}
    assert store.get("figure_egress_receipt") is not None


def test_p1_2_profile_can_pin_external_figure_upload():
    from easyicu.research_agent.orchestration.profiles import SubmissionProfile

    pinned = SubmissionProfile(
        name="npj_dm_test_dev",
        version="1",
        locked_at="2026-07-26T00:00:00Z",
        evidence_enforcement_mode="strict",
        writer_digest_widened=True,
        enable_reproducibility_envelope=True,
        requires_arm="aware",
        allow_external_figure_upload=False,
    )
    unpinned = SubmissionProfile(
        name="npj_dm_test_dev",
        version="1",
        locked_at="2026-07-26T00:00:00Z",
        evidence_enforcement_mode="strict",
        writer_digest_widened=True,
        enable_reproducibility_envelope=True,
        requires_arm="aware",
    )

    assert pinned.as_pipeline_options()["allow_external_figure_upload"] is False
    # Historical profiles that do not pin it stay byte-identical.
    assert "allow_external_figure_upload" not in unpinned.as_pipeline_options()


# ---------------------------------------------------------------------------
# P1-3 — as_kwargs() deep-copied live objects
# ---------------------------------------------------------------------------


def test_p1_3_services_preserve_live_object_identity(tmp_path):
    import threading

    from easyicu.research_agent.orchestration.config import PipelineConfig
    from easyicu.research_agent.orchestration.services import PipelineServices

    class _LiveClient:
        def __init__(self) -> None:
            # A lock is not deep-copyable; an httpx pool is, but the clone is
            # not the object the provider factory authorised.
            self._lock = threading.Lock()

    client = _LiveClient()
    config = PipelineConfig(workdir=tmp_path)
    services = PipelineServices(llm=client)

    kwargs = config.as_kwargs()

    assert "llm" not in kwargs
    assert services.llm is client


def test_p1_3_from_config_survives_a_client_holding_a_lock(tmp_path):
    import threading

    from easyicu.research_agent.orchestration.config import PipelineConfig
    from easyicu.research_agent.orchestration.services import PipelineServices
    from easyicu.research_agent.pipeline import ResearchAgentPipeline

    class _LiveClient:
        def __init__(self) -> None:
            self._lock = threading.Lock()

        def complete(self, *_args, **_kwargs):  # pragma: no cover - not called
            return ""

    client = _LiveClient()
    agent = ResearchAgentPipeline.from_config(
        PipelineConfig(workdir=tmp_path),
        PipelineServices(llm=client),
    )

    assert agent._services.llm is client
    assert agent._llm is client


def test_p1_3_canonical_payload_redacts_secret_valued_fields(tmp_path):
    from easyicu.research_agent.orchestration.config import PipelineConfig

    config = PipelineConfig(
        workdir=tmp_path,
        pubmed_api_key="pubmed-secret-value",
        tavily_api_key="tavily-secret-value",
    )

    payload = config.canonical_payload()
    encoded = json.dumps(payload)

    assert "pubmed-secret-value" not in encoded
    assert "tavily-secret-value" not in encoded
    assert payload["pubmed_api_key"].startswith("sha256:")
    # Still a coordinate: rotating the key changes the digest.
    rotated = config.with_overrides(pubmed_api_key="different")
    assert rotated.canonical_digest() != config.canonical_digest()


# ---------------------------------------------------------------------------
# P1-4 — paper runs could disable every container resource ceiling
# ---------------------------------------------------------------------------


def _docker_runner(tmp_path, **kwargs):
    from easyicu.research_agent.execution.runner import DockerRunner

    cohort = tmp_path / "cohort.parquet"
    cohort.write_bytes(b"parquet")
    return DockerRunner(workdir=tmp_path / "wd", cohort_parquet=cohort, **kwargs)


def test_p1_4_paper_profile_rejects_a_disabled_ceiling(tmp_path):
    (tmp_path / "wd").mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="memory_limit"):
        _docker_runner(tmp_path, memory_limit="", submission_profile_name="npj_dm")


def test_p1_4_development_profile_keeps_the_opt_out(tmp_path):
    (tmp_path / "wd").mkdir(parents=True, exist_ok=True)

    runner = _docker_runner(
        tmp_path, memory_limit="", submission_profile_name="npj_dm_framework_v2_dev"
    )

    assert runner.memory_limit == ""


def test_p1_4_paper_profile_defaults_still_construct(tmp_path):
    (tmp_path / "wd").mkdir(parents=True, exist_ok=True)

    runner = _docker_runner(tmp_path, submission_profile_name="npj_dm")

    assert runner.memory_limit == runner.DEFAULT_MEMORY_LIMIT
    assert runner.pids_limit == runner.DEFAULT_PIDS_LIMIT


# ---------------------------------------------------------------------------
# P1-5 — a corrupt experience bank still fed the Planner
# ---------------------------------------------------------------------------


def test_p1_5_corrupt_bank_refuses_to_serve_a_partial_retrieval(tmp_path):
    from easyicu.research_agent.learning.experience import (
        ExperienceBank,
        ExperienceBankCorruptError,
        ExperienceRecord,
    )

    path = tmp_path / "bank.jsonl"
    good = ExperienceRecord(
        kind="concept_usage_hint",
        research_question="Does SOFA-2 predict ICU death?",
        database="miiv",
        cohort_name="sepsis3",
        summary="Use easyicu.api.load_sepsis3 rather than an ICD regex.",
    )
    path.write_text(
        json.dumps(good.to_dict()) + "\nthis line is not JSON\n", encoding="utf-8"
    )

    bank = ExperienceBank(path=path)
    assert bank.corrupt_lines == 1

    with pytest.raises(ExperienceBankCorruptError):
        bank.retrieve(research_question="Does SOFA-2 predict ICU death?")


def test_p1_5_pipeline_plans_without_a_bank_rather_than_with_a_partial_one(
    tmp_path,
):
    from easyicu.research_agent.learning.experience import ExperienceRecord
    from easyicu.research_agent.pipeline import ResearchAgentPipeline

    path = tmp_path / "bank.jsonl"
    record = ExperienceRecord(
        kind="concept_usage_hint",
        research_question="Does SOFA-2 predict ICU death?",
        database="miiv",
        cohort_name="sepsis3",
        summary="Use easyicu.api.load_sepsis3 rather than an ICD regex.",
    )
    path.write_text(json.dumps(record.to_dict()) + "\nnot JSON\n", encoding="utf-8")

    agent = ResearchAgentPipeline(
        workdir=tmp_path / "wd",
        enable_experience_bank=True,
        experience_bank_path=path,
    )

    assert (
        agent.retrieve_experience_hints(
            research_question="Does SOFA-2 predict ICU death?", database="miiv"
        )
        == []
    )


def test_p1_5_unreadable_bank_raises_instead_of_silently_rewriting(tmp_path):
    from easyicu.research_agent.learning.experience import (
        ExperienceBank,
        ExperienceBankCorruptError,
    )

    path = tmp_path / "bank_dir"
    path.mkdir()  # a directory where the JSONL should be: read() raises OSError

    with pytest.raises(ExperienceBankCorruptError):
        ExperienceBank(path=path)


# ---------------------------------------------------------------------------
# P1-9 — raw-EHR provenance failure was a warning under a paper profile
# ---------------------------------------------------------------------------


def test_p1_9_paper_facing_profile_predicate_excludes_dev_profiles():
    from easyicu.research_agent.orchestration.profiles import is_paper_facing_profile

    assert is_paper_facing_profile("npj_dm") is True
    assert is_paper_facing_profile("npj_dm_framework_v2_dev") is False
    assert is_paper_facing_profile(None) is False


def test_p1_9_provenance_failure_stops_a_paper_run(monkeypatch, tmp_path):
    from easyicu.research_agent import pipeline as pipeline_module

    def _explode(**_kwargs):
        raise OSError("cohort parquet vanished under the mount")

    monkeypatch.setattr(pipeline_module, "build_provenance_bundle", _explode)

    frame = pd.DataFrame(
        {
            "stay_id": range(1, 31),
            "sofa2": [3.0 + (i % 4) for i in range(30)],
            "death": [i % 3 == 0 for i in range(30)],
        }
    )
    cohort = tmp_path / "cohort.parquet"
    frame.to_parquet(cohort)

    agent = pipeline_module.ResearchAgentPipeline(
        workdir=tmp_path / "paper",
        submission_profile_name="npj_dm",
        submission_profile_version="20260527",
        llm=MockLLMClient(),
    )
    with pytest.raises(RuntimeError, match="provenance chain would be incomplete"):
        agent.run(question="Does SOFA-2 predict death?", cohort=cohort)


# ---------------------------------------------------------------------------
# Review correction — the LLM debug permission test read source, not disk
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes only")
def test_p1_5_debug_dump_is_owner_only_on_disk(tmp_path, monkeypatch):
    """Stat the file the dump actually wrote.

    The previous round asserted that ``os.chmod(..., 0o700)`` appeared in the
    module source. That passes whether or not the call is reached, applies to
    the right path, or survives a refactor — the reviewer was right to reject
    it. This drives a real completion and reads the mode off disk.
    """

    from easyicu.research_agent.providers.llm import LLMMessage, OpenAIClient

    debug_dir = tmp_path / "dbg"
    monkeypatch.setenv("EASYICU_LLM_DEBUG", "1")
    monkeypatch.setenv("EASYICU_LLM_DEBUG_DIR", str(debug_dir))

    message = SimpleNamespace(
        content="the model answer",
        model_dump_json=lambda: json.dumps({"content": "the model answer"}),
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=None,
    )

    class _Completions:
        def create(self, **_kwargs):
            return response

    client = OpenAIClient.__new__(OpenAIClient)
    client._resolved_base_url = "http://127.0.0.1:8787/v1"
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    client._model = "gpt-test"
    client._timeout = 1.0
    client._extra_body = {}
    client._max_retries = 0
    monkeypatch.setattr(
        OpenAIClient, "_require_outbound_authorization", lambda self: None
    )

    client.complete([LLMMessage(role="user", content="a" * 20_000)])

    dumps = list(debug_dir.glob("*.json"))
    assert dumps, "the debug dump did not run"
    assert stat.S_IMODE(dumps[0].stat().st_mode) == 0o600
    assert stat.S_IMODE(debug_dir.stat().st_mode) == 0o700

    payload = json.loads(dumps[0].read_text(encoding="utf-8"))
    assert "raw_message" not in payload
    assert len(json.dumps(payload["prompt_messages"])) < 20_000


if sys.version_info < (3, 10):  # pragma: no cover - repo targets 3.10+
    raise RuntimeError("these tests assume Python 3.10+")


def test_p0_2_the_wired_recorder_binds_the_decision_into_run_evidence(
    monkeypatch, tmp_path
):
    """Exercise the recorder closure ``run()`` actually installs.

    Capturing the callable and driving it is the only way to reach this code
    without a full provider-backed run, and it still uses the real ``run_dir``,
    ``run_id`` and evidence store the closure captured.
    """

    import easyicu.research_agent.orchestration.workflow as workflow_module
    from easyicu.research_agent import pipeline as pipeline_module
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        ValidationFinding,
    )

    captured = {}

    def _fake_build(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after workflow construction")

    monkeypatch.setattr(workflow_module, "build_pipeline_workflow", _fake_build)

    frame = pd.DataFrame(
        {
            "stay_id": range(1, 31),
            "sofa2": [3.0 + (i % 4) for i in range(30)],
            "death": [i % 3 == 0 for i in range(30)],
        }
    )
    cohort = tmp_path / "cohort.parquet"
    frame.to_parquet(cohort)

    agent = pipeline_module.ResearchAgentPipeline(
        workdir=tmp_path / "wd",
        llm=MockLLMClient(),
        human_review_gate=lambda *_args, **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="stop after workflow construction"):
        agent.run(question="Does SOFA-2 predict death?", cohort=cohort)

    run_dir = next((tmp_path / "wd").glob("run_*"))
    store = EvidenceStore(run_dir)
    capsule = run_dir / "captured_run_input_capsule.json"
    capsule.write_text('{"schema_version":"test"}\n', encoding="utf-8")
    capsule_record = store.register_file(
        kind="log",
        description="test run input capsule",
        source_path=capsule,
        evidence_id="run_input_capsule",
        producer="pipeline",
        generation_mode="system",
    )
    requests = captured["human_review_invoker"](
        SimpleNamespace(
            findings=[
                ValidationFinding(
                    validator="capability_gate",
                    severity="error",
                    message="This plan requests an unregistered capability.",
                    detail={"reason": "capability_review_required"},
                )
            ],
            plan=AnalysisPlan(
                research_question="Can this capability be used?",
                steps=[
                    AnalysisStep(
                        step_id="01",
                        intent="Run the requested registered analysis.",
                    )
                ],
            ),
            evidence=store,
        )
    )
    execution = requests[0].payload["plan_review_authority"]["execution"]
    assert execution["pipeline_config_sha256"] == agent._config.canonical_digest()
    assert execution["capability_activation_sha256"]
    assert execution["run_input_capsule_sha256"] == capsule_record.sha256
    captured["human_review_recorder"](
        [
            {
                "schema": "easyicu.human_review_decision/1",
                "request": {"review_id": "review-0123456789abcdef"},
                "reviewer_identity": "operator@example.invalid",
                "reviewer_identity_source": "authenticated",
                "server_decided_at": "2026-07-26T00:00:00Z",
            }
        ]
    )

    assert store.get("human_review_decisions") is not None
    payload = json.loads(
        (run_dir / "human_review_decisions.json").read_text(encoding="utf-8")
    )
    assert payload["schema"] == "easyicu.human_review_decisions/1"
    assert payload["decisions"][0]["reviewer_identity_source"] == "authenticated"


def test_p0_2_paper_profile_rejects_a_client_claimed_reviewer(monkeypatch, tmp_path):
    import easyicu.research_agent.orchestration.workflow as workflow_module
    from easyicu.research_agent import pipeline as pipeline_module
    from easyicu.research_agent.authority.evidence_store import EvidenceStore

    captured = {}

    def _fake_build(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after workflow construction")

    monkeypatch.setattr(workflow_module, "build_pipeline_workflow", _fake_build)

    frame = pd.DataFrame(
        {
            "stay_id": range(1, 31),
            "sofa2": [3.0 + (i % 4) for i in range(30)],
            "death": [i % 3 == 0 for i in range(30)],
        }
    )
    cohort = tmp_path / "cohort.parquet"
    frame.to_parquet(cohort)

    agent = pipeline_module.ResearchAgentPipeline(
        workdir=tmp_path / "paper",
        submission_profile_name="npj_dm",
        submission_profile_version="20260527",
        llm=MockLLMClient(),
    )
    with pytest.raises(RuntimeError, match="stop after workflow construction"):
        agent.run(question="Does SOFA-2 predict death?", cohort=cohort)

    run_dir = next((tmp_path / "paper").glob("run_*"))
    captured["human_review_invoker"](
        SimpleNamespace(
            findings=[],
            plan=SimpleNamespace(steps=()),
            evidence=EvidenceStore(run_dir),
        )
    )

    with pytest.raises(RuntimeError, match="authenticated reviewer identity"):
        captured["human_review_recorder"](
            [
                {
                    "schema": "easyicu.human_review_decision/1",
                    "request": {"review_id": "review-0123456789abcdef"},
                    "reviewer_identity": None,
                    "reviewer_identity_source": "unauthenticated_client_claim",
                    "server_decided_at": "2026-07-26T00:00:00Z",
                }
            ]
        )


def test_p0_2_a_warning_with_the_same_reason_does_not_halt_a_run():
    """Severity is the run's own statement about whether the state blocks.

    A development profile records a provenance failure as a warning because it
    makes no provenance claim. If the review gate keyed on the reason code
    alone, that warning would stop every such run waiting for a signature.
    """

    from easyicu.research_agent.orchestration.workflow import (
        human_review_requests_for_plan,
    )
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        ValidationFinding,
    )

    plan = AnalysisPlan(
        research_question="Is raw-EHR provenance available?",
        steps=[
            AnalysisStep(
                step_id="01",
                intent="Assess the requested provenance.",
            )
        ],
    )
    warning = ValidationFinding(
        validator="provenance",
        severity="warning",
        message="Failed to compute raw-EHR provenance bundle.",
        detail={"reason": "raw_ehr_provenance_unavailable"},
    )
    error = warning.model_copy(update={"severity": "error"})

    assert human_review_requests_for_plan(findings=[warning], plan=plan) == ()
    assert len(human_review_requests_for_plan(findings=[error], plan=plan)) == 1
