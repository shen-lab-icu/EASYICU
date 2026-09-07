from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from easyicu.webserver.research_run_submission import ResearchRunSubmissionRequest
from easyicu.webserver import manuscript_repair


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
    }
    result = manuscript_repair._project_revision(
        SimpleNamespace(wrapper_dir=wrapper, pipeline_run_id="run-original"),
        {"id": "study"},
        "## Methods\nA revised report.",
        revision,
        {"provider": "fake"},
    )
    assert result["gate"] == gate
    assert (original / "run_status.json").read_bytes() == old
    draft = json.loads((wrapper / "manuscript_draft.json").read_text())
    assert draft["report_revision"] == revision
    public = manuscript_repair.pipeline_owner.agent_runs._public_review_payloads(
        {"manuscript_draft.json": draft}
    )
    assert public["manuscript_draft.json"]["report_revision"] == revision
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
