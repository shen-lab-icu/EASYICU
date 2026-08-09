"""CLI ownership of same-process human-review pauses."""

from __future__ import annotations

import json
from types import SimpleNamespace


def _pending(run_dir):
    from easyicu.research_agent.orchestration.workflow import (
        HumanReviewPending,
        HumanReviewRequest,
    )

    request = HumanReviewRequest.create(
        kind="capability_request",
        summary="Approve the exact capability-bound plan.",
        authority_sha256="a" * 64,
        payload={"reason": "capability_review_required"},
    )
    return HumanReviewPending(
        run_id="run_cli_review",
        thread_id="run_cli_review",
        run_dir=str(run_dir),
        requests=(request,),
    )


def test_noninteractive_cli_emits_structured_pending_and_dedicated_exit(
    tmp_path, monkeypatch, capsys
):
    from easyicu.research_agent import cli
    from easyicu.research_agent import pipeline as pipeline_module

    pending = _pending(tmp_path / "run_cli_review")

    class FakePipeline:
        def __init__(self, **_kwargs):
            pass

        def run(self, **_kwargs):
            return pending

    monkeypatch.setattr(pipeline_module, "ResearchAgentPipeline", FakePipeline)
    monkeypatch.setattr(cli, "_is_interactive_terminal", lambda: False)

    exit_code = cli.main(
        [
            "--llm",
            "mock",
            "--question",
            "Inspect this cohort.",
            "--cohort",
            str(tmp_path / "cohort.parquet"),
            "--workdir",
            str(tmp_path / "runs"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == cli.HUMAN_REVIEW_PENDING_EXIT_CODE
    assert payload["status"] == "human_review_pending"
    assert payload["terminal"] is False
    assert payload["resume_scope"] == "same_process"
    assert payload["resumable_via_cli"] is False
    assert payload["external_resume_supported"] is False
    assert payload["requests"][0]["review_id"] == pending.requests[0].review_id


def test_interactive_cli_resumes_on_the_same_pipeline_instance(
    tmp_path, monkeypatch, capsys
):
    from easyicu.research_agent import cli
    from easyicu.research_agent import pipeline as pipeline_module

    pending = _pending(tmp_path / "run_cli_review")
    seen = {}

    class FakePipeline:
        def __init__(self, **_kwargs):
            seen["pipeline"] = self

        def run(self, **_kwargs):
            return pending

        def resume_human_review(self, decisions, *, run_id=None):
            seen["decisions"] = decisions
            seen["run_id"] = run_id
            return SimpleNamespace(
                run_id="run_cli_review",
                workdir=str(tmp_path / "runs"),
                context_path="context.json",
                plan_path="analysis_plan.json",
                manifest_path="manifest.json",
                report_path="report.md",
                manuscript_path="manuscript.md",
                evidence_count=4,
                findings_count=1,
            )

    answers = iter(["approve", "Dr Reviewer", "approved after inspection"])
    monkeypatch.setattr(pipeline_module, "ResearchAgentPipeline", FakePipeline)
    monkeypatch.setattr(cli, "_is_interactive_terminal", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    exit_code = cli.main(
        [
            "--llm",
            "mock",
            "--question",
            "Inspect this cohort.",
            "--cohort",
            str(tmp_path / "cohort.parquet"),
            "--workdir",
            str(tmp_path / "runs"),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "manifest:     manifest.json" in output
    assert seen["run_id"] == pending.run_id
    assert len(seen["decisions"]) == 1
    assert seen["decisions"][0].decision == "approved"
    assert seen["decisions"][0].reviewer == "Dr Reviewer"
    assert seen["decisions"][0].authority_sha256 == pending.requests[0].authority_sha256
