"""A resumed accepted outline must retain foundation retry feedback."""
from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.planning import outline_retry_journal as owner
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context, _foundation_payload, _outline_payload,
)
from tests.research_agent.planning.test_outline_retry_journal import _setup


@pytest.mark.parametrize("rejected_first", [False, True])
def test_foundation_preserves_its_budget_after_accepted_outline(tmp_path, monkeypatch, rejected_first):
    monkeypatch.setattr(owner, "_verify_pending_guard", lambda *_args: None)
    _inner, book, task = _setup(tmp_path, monkeypatch)
    inner = ScriptedMockLLMClient([])
    inner.supports_strict_json_schema = True
    calls = []

    def fake(_inner, messages, **_kwargs):
        calls.append([m.content for m in messages])
        rows = book.snapshot()["tasks"][0]["calls"]
        prior = rows[-1]["attempt_id"] if rows and rows[-1]["state"] == "in_progress" else None
        attempt = book.reserve_transport_attempt(
            task_id="task", role="planner", model="synthetic", messages=messages,
            max_tokens=16, prior_attempt_id=prior,
        )
        interrupted_call = 3 if rejected_first else 2
        if len(calls) == interrupted_call:
            raise KeyboardInterrupt()
        book.finish_transport_attempt(
            task_id="task", attempt_id=attempt,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            error_type=None,
        )
        if len(calls) == 1:
            return json.dumps(_outline_payload())
        if rejected_first and len(calls) == 2:
            return '{"bad":"not-a-foundation"}'
        return json.dumps(_foundation_payload())

    monkeypatch.setattr(owner, "authorized_complete", fake)
    checkpoints = []
    dependency = {"cohort_file_sha256": "b" * 64, "llm_signature": "mock:test", "prompt_version": "v1"}
    source = tmp_path / "source"
    journal = {"task": task, "run_dir": source, "literature_sha256": "e" * 64}
    with pytest.raises(KeyboardInterrupt):
        ProgressivePlannerAgent(inner).run(
            _context(), checkpoint_callback=checkpoints.append,
            resume_dependency_context=dependency, outline_retry_journal=journal,
        )
    assert checkpoints and checkpoints[-1].stage == "outline"
    signed = source / ".runtime" / owner.FILE_NAME
    assert owner.load_retry_journal(signed)["binding"]["stage"] == "foundation"
    checkpoint_path = source / "progressive_planner_checkpoint_001.json"
    checkpoint_path.write_text(checkpoints[-1].model_dump_json())

    if not rejected_first:
        # A second interruption after verified outline persistence but before
        # another provider transport must still leave the original reservation
        # attributable on the next restart.
        from easyicu.research_agent.planning.progressive_artifacts import ProgressivePlannerCheckpointRecorder
        from tests.research_agent.planning.test_progressive_planner_checkpoints import _RecordingEvidence

        continued = tmp_path / "continued"
        continued.mkdir()
        recorder = ProgressivePlannerCheckpointRecorder(
            run_dir=continued, evidence=_RecordingEvidence(), prompt_pack_version="v1",
            source_chain=(checkpoints[-1],),
        )
        original_complete = owner.OutlineRetryJournal.complete
        def interrupted_before_transport(self, messages, **kwargs):
            if self._data["binding"]["stage"] == "foundation":
                raise KeyboardInterrupt()
            return original_complete(self, messages, **kwargs)
        monkeypatch.setattr(owner.OutlineRetryJournal, "complete", interrupted_before_transport)
        with pytest.raises(KeyboardInterrupt):
            ProgressivePlannerAgent(inner).run(
                _context(), checkpoint_callback=recorder.record,
                resume_checkpoint=checkpoints[-1], resume_dependency_context=dependency,
                resume_outline_validated_callback=recorder.persist_validated_outline,
                outline_retry_journal={"task": task, "run_dir": continued,
                    "literature_sha256": "e" * 64, "accepted_checkpoint_path": checkpoint_path},
            )
        copied = continued / "progressive_planner_checkpoint_000.json"
        assert copied.exists()
        resumed = owner.load_retry_journal(continued / ".runtime" / owner.FILE_NAME)
        assert resumed["expected_retry_guarded"] is True
        monkeypatch.setattr(owner.OutlineRetryJournal, "complete", original_complete)
        checkpoint_path = copied

    class FoundationAccepted(Exception):
        pass

    def capture(checkpoint):
        if checkpoint.stage == "foundation":
            raise FoundationAccepted

    expected = ValueError if rejected_first else FoundationAccepted
    with pytest.raises(expected) as raised:
        ProgressivePlannerAgent(inner).run(
            _context(), checkpoint_callback=capture, resume_checkpoint=checkpoints[-1],
            resume_dependency_context=dependency,
            outline_retry_journal={"task": task, "run_dir": tmp_path / "continued_final",
                "literature_sha256": "e" * 64, "accepted_checkpoint_path": checkpoint_path},
        )
    if rejected_first:
        assert "retry_journal_attempt_limit" in str(raised.value)
        assert len(calls) == 3  # The interrupted second attempt consumed the last slot.
    else:
        assert len(calls) == 3
        assert calls[1] == calls[2]
    assert owner.load_retry_journal(tmp_path / "continued_final" / ".runtime" / owner.FILE_NAME)["abandoned"] == 1


def test_web_resume_requires_pending_journal_for_accepted_checkpoint(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.webserver import metadata_provider_concurrency as slots, research_auto_resume
    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

    inner, _book, task = _setup(tmp_path, monkeypatch)
    wrapper = tmp_path / "wrapper"
    run = wrapper / "pipeline" / "run_a"
    run.mkdir(parents=True)
    checkpoint = run / "progressive_planner_checkpoint_001.json"
    checkpoint.write_text("{}")
    active = {"role": "planner", "state": "in_progress", "attempt_id": "attempt"}
    task_row = {"calls": [active]}
    location = {"path": checkpoint.relative_to(wrapper).as_posix()}
    with pytest.raises(ResearchPipelineRunError, match="original authority") as missing:
        research_auto_resume._verify_checkpoint_inflight_retry(wrapper, location, task_row)
    assert missing.value.code == "research_auto_resume_planner_retry_journal_missing"

    journal = owner.OutlineRetryJournal(
        inner=inner, task=task, run_dir=run, request_sha256="b" * 64,
        dependency_sha256="c" * 64, literature_sha256="e" * 64,
        stage="foundation", parse_attempt_limit=2,
    )
    monkeypatch.setattr(owner, "authorized_complete", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        journal.complete([LLMMessage("user", "synthetic")], max_tokens=16, temperature=0.2)
    pending = owner.load_retry_journal(run / ".runtime" / owner.FILE_NAME)["pending"]
    monkeypatch.setattr(slots, "get", lambda _attempt: {"request_sha256": "a" * 64})
    with pytest.raises(ResearchPipelineRunError) as mismatched:
        research_auto_resume._verify_checkpoint_inflight_retry(wrapper, location, task_row)
    assert mismatched.value.code == "research_auto_resume_planner_retry_journal_invalid"
    monkeypatch.setattr(slots, "get", lambda _attempt: {"request_sha256": pending["request_sha256"]})
    research_auto_resume._verify_checkpoint_inflight_retry(wrapper, location, task_row)

    # A second crash between replay admission and the next reservation leaves
    # the old guarded reservation accounted, but no longer pending.
    signed = owner.load_retry_journal(run / ".runtime" / owner.FILE_NAME)
    signed.update(pending=None, expected_retry_guarded=True, abandoned=1,
                  accounted_planner_calls=1,
                  expected_retry_request_sha256=pending["request_sha256"])
    owner._atomic_write(run / ".runtime" / owner.FILE_NAME, signed)
    research_auto_resume._verify_checkpoint_inflight_retry(wrapper, location, task_row)
    signed["accounted_planner_calls"] = 0
    owner._atomic_write(run / ".runtime" / owner.FILE_NAME, signed)
    with pytest.raises(ResearchPipelineRunError) as inconsistent:
        research_auto_resume._verify_checkpoint_inflight_retry(wrapper, location, task_row)
    assert inconsistent.value.code == "research_auto_resume_planner_retry_journal_invalid"
