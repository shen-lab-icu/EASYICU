from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from easyicu.webserver.research_run_submission import ResearchRunSubmissionRequest
from easyicu.webserver import manuscript_repair


def test_report_only_limits_cap_and_never_expand_approved_budget():
    approved = manuscript_repair.provider_adapter.web_research_agent_hard_stop_limits(
        "full_reviewed"
    )
    caps = {
        "max_provider_attempts_per_run": 32,
        "max_provider_attempts_per_batch": 32,
        "max_wall_clock_seconds_per_task": 600,
    }
    narrowed = manuscript_repair._report_only_limits(approved)
    for field, maximum in caps.items():
        assert getattr(narrowed, field) == min(maximum, getattr(approved, field))
    assert narrowed.max_total_tokens_per_run == approved.max_total_tokens_per_run
    assert narrowed.max_total_tokens_per_batch == approved.max_total_tokens_per_batch
    smaller = replace(approved, **{field: 1 for field in caps})
    assert manuscript_repair._report_only_limits(smaller) == smaller


def test_report_only_rejects_unfundable_budget_before_resolving_source(monkeypatch):
    approved = manuscript_repair.provider_adapter.web_research_agent_hard_stop_limits("full_reviewed")
    impossible = replace(approved, max_total_tokens_per_run=100_000, max_total_tokens_per_batch=100_000)
    monkeypatch.setattr(manuscript_repair.provider_adapter, "web_research_agent_hard_stop_limits", lambda _: impossible)
    monkeypatch.setattr(manuscript_repair.pipeline_owner, "_resolve_execution_resume_wrapper",
                        lambda **_: pytest.fail("Invalid budget must fail before source/job creation"))
    with pytest.raises(ValueError, match="cannot fund one minimum Provider"):
        manuscript_repair.make_report_only_run_runner(
            study_context={}, project_root=None, provider={}, provider_environment={},
            credential_source="pi_verified", execution_resume_source_run_id="source",
            budget_mode="full_reviewed", export_path=None,
        )


def _request(**overrides):
    return ResearchRunSubmissionRequest(
        **{
            "study_context_id": "study",
            "provider": "openai",
            "credential_source": "pi_verified",
            "external_llm_opt_in": True,
            "execution_resume_source_run_id": "run-original",
            "report_only": True,
            **overrides,
        }
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"execution_resume_source_run_id": ""},
        {"intent": "candidate_plan"},
        {"planner_start_mode": "fresh"},
        {"planner_start_mode": "resume_checkpoint"},
        {"plan_revision_source_run_id": "different-plan"},
        {"literature_search_authorized": True},
    ],
)
def test_report_repair_cannot_promote_to_new_plan_or_new_search(changes):
    with pytest.raises(
        ValidationError, match="report_only_requires_exact_completed_run"
    ):
        _request(**changes)


def test_report_repair_has_explicit_path_free_submission_intent():
    assert _request().report_only is True
    assert _request(report_only=False).report_only is False


def test_revision_projection_does_not_promote_or_rewrite_original_analysis(tmp_path):
    wrapper = tmp_path / "wrapper"
    wrapper.mkdir()
    original = wrapper / "pipeline" / "run-original"
    original.mkdir(parents=True)
    (original / "run_status.json").write_text('{"manuscript_ready":false}')
    gate = {
        "status": "blocked",
        "reason": "research_agent_pipeline_failed_closed",
        "checks": {
            "execution_complete": True,
            "manuscript_ready": False,
            "paper_authorized": False,
        },
    }
    (wrapper / "quality_gate.json").write_text(json.dumps({"gate": gate}))
    (wrapper / "manuscript_draft.json").write_text(
        json.dumps(
            {
                "run_id": "run-original",
                "question": "Question",
                "markdown_preview": "Old draft",
            }
        )
    )
    artifact = manuscript_repair.pipeline_owner._artifact_record(
        wrapper / "manuscript_draft.json"
    )
    (wrapper / "evidence_ledger.json").write_text(json.dumps({"artifacts": [artifact]}))
    old = (original / "run_status.json").read_bytes()
    revision = {
        "revision_id": "revision",
        "status": "pass",
        "publication_authorized": False,
        "output_sha256": "a" * 64,
    }
    provenance = {
        "schema_version": "easyicu.manuscript-provenance/1",
        "manuscript_sha256": "a" * 64, "claim_ceiling": "analysis_only",
        "publication_authorized": False, "article_blocks": [], "claims": [],
    }
    result = manuscript_repair._project_revision(
        SimpleNamespace(wrapper_dir=wrapper, pipeline_run_id="run-original"),
        {"id": "study"},
        "## Methods\nA revised report.",
        revision,
        {"provider": "fake"},
        provenance=provenance,
    )
    assert result["gate"] == gate
    assert (original / "run_status.json").read_bytes() == old
    draft = json.loads((wrapper / "manuscript_draft.json").read_text())
    assert draft["report_revision"] == revision
    public = manuscript_repair.pipeline_owner.agent_runs._public_review_payloads(
        {"manuscript_draft.json": draft}
    )
    assert public["manuscript_draft.json"]["report_revision"] == revision
    assert public["manuscript_draft.json"]["reader"]["manuscript_sha256"] == "a" * 64
    assert draft["claims"] == []
    assert json.loads((wrapper / "manuscript_provenance.json").read_text())["report_revision"] == revision
    ledger = json.loads((wrapper / "evidence_ledger.json").read_text())
    assert ledger["artifacts"][0]["sha256"] != artifact["sha256"]


def test_source_drift_is_detected_even_when_original_input_names_are_unchanged(
    tmp_path,
):
    source = tmp_path / "run"
    source.mkdir()
    (source / "registered_result.json").write_text('{"n":120}')
    before = manuscript_repair._source_fingerprint(source)
    (source / "registered_result.json").write_text('{"n":121}')
    assert manuscript_repair._source_fingerprint(source) != before


@pytest.mark.parametrize("mutation", ["digest", "ceiling", "authorization"])
def test_revision_reader_mismatch_is_rejected_before_any_wrapper_write(tmp_path, mutation):
    wrapper = tmp_path / "wrapper"
    wrapper.mkdir()
    sentinel = wrapper / "manuscript_draft.json"
    sentinel.write_text('{"markdown_preview":"Prior manuscript"}')
    before = sentinel.read_bytes()
    provenance = {
        "schema_version": "easyicu.manuscript-provenance/1",
        "manuscript_sha256": "a" * 64, "claim_ceiling": "analysis_only",
        "publication_authorized": False,
    }
    if mutation == "digest":
        provenance["manuscript_sha256"] = "b" * 64
    elif mutation == "ceiling":
        provenance["claim_ceiling"] = "reportable"
    else:
        provenance["publication_authorized"] = True
    with pytest.raises(manuscript_repair.WriterOnlyMigrationError, match="READER_BINDING_FAILED"):
        manuscript_repair._project_revision(
            SimpleNamespace(wrapper_dir=wrapper), {"id": "study"}, "New text",
            {"output_sha256": "a" * 64}, {}, provenance=provenance,
        )
    assert sentinel.read_bytes() == before
    assert list(wrapper.iterdir()) == [sentinel]


@pytest.mark.parametrize('mismatch', ['hash', 'revision', 'manuscript', 'missing'])
def test_pdf_binding_failure_preserves_previous_reader_and_historical_pdf(tmp_path, mismatch):
    import hashlib
    wrapper = tmp_path / 'wrapper'
    wrapper.mkdir()
    for name, content in [('evidence_ledger.json', '{"artifacts":[]}'),
                          ('manuscript_draft.json', '{"markdown_preview":"Previous report"}'),
                          ('manuscript_scaffold.pdf', '%PDF-original')]:
        (wrapper / name).write_text(content)
    before = {p.name: p.read_bytes() for p in wrapper.iterdir()}
    path = tmp_path / 'revision.pdf'
    path.write_bytes(b'%PDF-current')
    pdf = {'name': 'manuscript_revision.pdf', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
           'revision_id': 'new', 'manuscript_sha256': 'a' * 64}
    pdf[{'hash': 'sha256', 'revision': 'revision_id', 'manuscript': 'manuscript_sha256', 'missing': 'name'}[mismatch]] = 'wrong'
    revision = {'revision_id': 'new', 'output_sha256': 'a' * 64, 'pdf_artifact': pdf}
    provenance = {'schema_version': 'easyicu.manuscript-provenance/1', 'manuscript_sha256': 'a' * 64,
                  'claim_ceiling': 'analysis_only', 'publication_authorized': False}
    with pytest.raises(manuscript_repair.WriterOnlyMigrationError, match='PDF_BINDING_FAILED'):
        manuscript_repair._project_revision(SimpleNamespace(wrapper_dir=wrapper), {'id': 'study'},
            'New report', revision, {}, provenance=provenance, pdf_path=None if mismatch == 'missing' else path)
    assert {p.name: p.read_bytes() for p in wrapper.iterdir()} == before
