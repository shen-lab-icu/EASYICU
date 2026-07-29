"""Tests for the LLM reproducibility envelope (O20).

Two layers exercised:

* ``ReproEnvelope`` + ``ReproRecordingClient`` in isolation —
  deterministic prompt/response sha256 across calls, seed forwarding
  to clients that accept it, and graceful degradation for clients
  that don't.
* End-to-end: ``ResearchAgentPipeline(enable_reproducibility_envelope=True)``
  populates ``manifest.reproducibility`` and writes a
  ``reproducibility_envelope.json`` artefact registered in the
  EvidenceStore.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# ReproEnvelope unit tests
# ---------------------------------------------------------------------------


def test_sha256_messages_is_deterministic_across_calls(ra):
    from easyicu.research_agent.providers.llm import LLMMessage
    from easyicu.research_agent.replication.envelope import sha256_messages

    msgs = [
        LLMMessage(role="system", content="you are careful"),
        LLMMessage(role="user", content="plan the analysis"),
    ]
    h1 = sha256_messages(msgs)
    h2 = sha256_messages(msgs)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_sha256_messages_changes_on_content_change(ra):
    from easyicu.research_agent.providers.llm import LLMMessage
    from easyicu.research_agent.replication.envelope import sha256_messages

    base = [LLMMessage(role="user", content="plan the analysis")]
    other = [LLMMessage(role="user", content="plan the analysis.")]
    assert sha256_messages(base) != sha256_messages(other)


def test_recording_client_records_prompt_and_response_hashes(ra):
    from easyicu.research_agent.providers.llm import LLMMessage

    class _Stub:
        name = "stub"
        _model = "stub-model"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            return "OK: " + (messages[-1].content or "")

    env = ra.ReproEnvelope(run_id="test-run")
    recorder = ra.ReproRecordingClient(_Stub(), role="planner", envelope=env)
    msg = LLMMessage(role="user", content="hello")
    out = recorder.complete([msg], max_tokens=32, temperature=0.1)

    assert out.startswith("OK: hello")
    assert len(env.calls) == 1
    rec = env.calls[0]
    assert rec.role == "planner"
    assert rec.client_name == "stub"
    assert rec.model == "stub-model"
    assert rec.temperature == 0.1
    assert rec.max_tokens == 32
    assert rec.requested_seed is None
    assert (
        rec.prompt_sha256
        == hashlib.sha256(f"<<<user>>>\n{msg.content}".encode("utf-8")).hexdigest()
    )
    assert rec.response_sha256 == hashlib.sha256(out.encode("utf-8")).hexdigest()


def test_recording_client_forwards_seed_when_inner_accepts_it(ra):
    from easyicu.research_agent.providers.llm import LLMMessage

    observed = {}

    class _WithSeed:
        name = "with-seed"
        _model = "seedable"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2, seed=None):
            observed["seed"] = seed
            return "seed=" + repr(seed)

    env = ra.ReproEnvelope(run_id="run-seed")
    recorder = ra.ReproRecordingClient(
        _WithSeed(),
        role="coder",
        envelope=env,
        seed=1234,
    )
    out = recorder.complete([LLMMessage(role="user", content="go")])
    assert observed["seed"] == 1234
    assert "1234" in out
    assert env.calls[0].requested_seed == 1234


def test_recording_client_records_top_p_when_caller_sets_it(ra):
    from easyicu.research_agent.providers.llm import LLMMessage

    observed = {}

    class _WithTopP:
        name = "with-top-p"
        _model = "topp-aware"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2, top_p=None):
            observed["top_p"] = top_p
            return "ok"

    env = ra.ReproEnvelope(run_id="run-top-p")
    recorder = ra.ReproRecordingClient(_WithTopP(), role="planner", envelope=env)
    recorder.complete(
        [LLMMessage(role="user", content="x")],
        max_tokens=16,
        temperature=0.1,
        top_p=0.9,
    )
    assert observed["top_p"] == 0.9
    rec = env.calls[0]
    assert rec.requested_top_p == 0.9
    payload = rec.to_json()
    assert payload["requested_top_p"] == 0.9


def test_recording_client_records_provider_default_when_top_p_unset(ra):
    from easyicu.research_agent.providers.llm import LLMMessage

    class _NoTopP:
        name = "no-top-p"
        _model = "topp-default"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            return "ok"

    env = ra.ReproEnvelope(run_id="run-top-p-default")
    recorder = ra.ReproRecordingClient(_NoTopP(), role="coder", envelope=env)
    # Caller doesn't set top_p, so the envelope must record None to
    # mean "we did not override the provider default".
    recorder.complete([LLMMessage(role="user", content="x")])
    rec = env.calls[0]
    assert rec.requested_top_p is None
    summary = env.to_manifest_summary()
    assert summary["top_p_used_provider_default"] is True
    assert summary["requested_top_ps"] == []


def test_recording_client_records_reasoning_effort_and_elapsed_time(ra):
    from easyicu.research_agent.providers.llm import LLMMessage

    class _EffortClient:
        name = "effort-client"
        _model = "effort-model"
        _extra_body = {"reasoning": {"effort": "medium"}}

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            return "ok"

    env = ra.ReproEnvelope(run_id="run-effort")
    recorder = ra.ReproRecordingClient(
        _EffortClient(),
        role="planner",
        envelope=env,
    )
    recorder.complete([LLMMessage(role="user", content="x")])

    record = env.calls[0]
    assert record.reasoning_effort == "medium"
    assert record.elapsed_ms is not None and record.elapsed_ms >= 0
    assert record.to_json()["reasoning_effort"] == "medium"
    summary = env.to_manifest_summary()
    assert summary["reasoning_efforts"] == ["medium"]
    assert summary["recorded_elapsed_ms_total"] >= 0


def test_recording_client_degrades_gracefully_for_clients_without_seed(ra):
    from easyicu.research_agent.providers.llm import LLMMessage

    class _NoSeed:
        name = "no-seed"
        _model = "rigid"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            return "fine"

    env = ra.ReproEnvelope(run_id="run-no-seed")
    recorder = ra.ReproRecordingClient(
        _NoSeed(),
        role="writer",
        envelope=env,
        seed=42,
    )
    # Should not raise even though the inner client has no seed kwarg.
    out = recorder.complete([LLMMessage(role="user", content="go")])
    assert out == "fine"
    # Requested seed is still recorded as user intent even though the
    # provider cannot honour it.
    assert env.calls[0].requested_seed == 42


def test_recording_client_returns_usage_owned_by_the_same_call(ra):
    from easyicu.research_agent.providers.llm import LLMMessage

    class _WithCallUsage:
        name = "call-usage"
        _model = "usage-model"

        def complete_with_usage(
            self, messages, *, max_tokens=2048, temperature=0.2, seed=None
        ):
            assert seed == 77
            return "ok", {
                "prompt_tokens": 13,
                "completion_tokens": 5,
                "total_tokens": 18,
            }

    env = ra.ReproEnvelope(run_id="run-call-usage")
    recorder = ra.ReproRecordingClient(
        _WithCallUsage(), role="writer", envelope=env, seed=77
    )
    response, usage = recorder.complete_with_usage(
        [LLMMessage(role="user", content="go")]
    )

    assert response == "ok"
    assert usage == {
        "prompt_tokens": 13,
        "completion_tokens": 5,
        "total_tokens": 18,
    }
    assert recorder.last_usage == usage
    assert len(env.calls) == 1


def test_manifest_summary_aggregates_by_role_and_model(ra):
    from easyicu.research_agent.providers.llm import LLMMessage

    class _A:
        name = "a"
        _model = "m1"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            return "ra"

    class _B:
        name = "b"
        _model = "m2"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            return "rb"

    env = ra.ReproEnvelope(run_id="agg")
    ra.ReproRecordingClient(_A(), role="planner", envelope=env).complete(
        [LLMMessage(role="user", content="x")]
    )
    ra.ReproRecordingClient(_A(), role="coder", envelope=env).complete(
        [LLMMessage(role="user", content="y")]
    )
    ra.ReproRecordingClient(_B(), role="writer", envelope=env).complete(
        [LLMMessage(role="user", content="z")]
    )

    summary = env.to_manifest_summary()
    assert summary["n_calls"] == 3
    assert set(summary["by_role"].keys()) == {"planner", "coder", "writer"}
    assert set(summary["by_model"].keys()) == {"m1", "m2"}
    assert summary["by_model"]["m1"]["n_calls"] == 2
    assert summary["by_model"]["m2"]["n_calls"] == 1
    assert summary["schema_version"] == ra.replication.envelope.ENVELOPE_SCHEMA_VERSION  # type: ignore[attr-defined]


def test_envelope_to_disk_roundtrips(ra, tmp_path):
    from easyicu.research_agent.providers.llm import LLMMessage

    class _C:
        name = "c"
        _model = "m"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            return "ok"

    env = ra.ReproEnvelope(run_id="disk", seed=7)
    ra.ReproRecordingClient(_C(), role="planner", envelope=env).complete(
        [LLMMessage(role="user", content="go")]
    )
    out_path = tmp_path / "envelope.json"
    env.to_disk(out_path)
    payload = json.loads(out_path.read_text())
    assert payload["run_id"] == "disk"
    assert payload["seed"] == 7
    assert len(payload["calls"]) == 1
    assert payload["calls"][0]["prompt_sha256"]


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def _write_cohort(df, tmp_path):
    path = tmp_path / "cohort.parquet"
    df.to_parquet(path)
    return path


def test_pipeline_envelope_populates_manifest_and_writes_artifact(
    ra, synthetic_cohort, tmp_path
):
    cohort_path = _write_cohort(synthetic_cohort, tmp_path)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "out",
        llm=ra.MockLLMClient(),
        enable_reproducibility_envelope=True,
        llm_seed=2026,
        envelope_include_previews=False,
    )
    result = pipeline.run(
        skill="association_analysis",
        cohort=cohort_path,
        database="miiv",
    )

    run_dir = Path(result.manifest_path).parent
    envelope_path = run_dir / "reproducibility_envelope.json"
    assert envelope_path.exists()
    payload = json.loads(envelope_path.read_text())
    assert payload["run_id"] == result.run_id
    assert payload["seed"] == 2026
    assert len(payload["calls"]) > 0
    for call in payload["calls"]:
        assert call["prompt_sha256"]
        assert call["response_sha256"]
        assert call["requested_seed"] == 2026

    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["reproducibility"] is not None
    assert manifest["reproducibility"]["run_id"] == result.run_id
    assert manifest["reproducibility"]["n_calls"] == len(payload["calls"])
    # Envelope is registered in evidence.
    ev_ids = [r["evidence_id"] for r in manifest["evidence"]]
    assert "reproducibility_envelope" in ev_ids


def test_pipeline_without_envelope_stays_bit_identical_for_manifest_field(
    ra, synthetic_cohort, tmp_path
):
    cohort_path = _write_cohort(synthetic_cohort, tmp_path)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "out",
        llm=ra.MockLLMClient(),
    )
    result = pipeline.run(
        skill="association_analysis",
        cohort=cohort_path,
        database="miiv",
    )
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["reproducibility"] is None
    run_dir = Path(result.manifest_path).parent
    assert not (run_dir / "reproducibility_envelope.json").exists()


def test_pipeline_envelope_composes_with_cost_tracking(ra, synthetic_cohort, tmp_path):
    cohort_path = _write_cohort(synthetic_cohort, tmp_path)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "out",
        llm=ra.MockLLMClient(),
        enable_reproducibility_envelope=True,
        enable_cost_tracking=True,
        llm_seed=101,
    )
    result = pipeline.run(
        skill="association_analysis",
        cohort=cohort_path,
        database="miiv",
    )
    manifest = json.loads(Path(result.manifest_path).read_text())
    # Both layers populated independently.
    assert manifest["reproducibility"] is not None
    assert manifest["reproducibility"]["n_calls"] > 0
    assert len(manifest["cost_records"]) > 0
    # Every envelope call has a matching cost record (same n_calls).
    assert manifest["reproducibility"]["n_calls"] == len(manifest["cost_records"])


def test_a_seed_with_no_path_to_the_provider_is_refused(ra, tmp_path) -> None:
    """A stamped seed that no request carried is a false provenance claim.

    The seed reaches a provider only through the envelope's recording client.
    With the envelope off the execution identity still recorded ``llm_seed``,
    so a run advertised a reproducibility guarantee its transport never
    delivered. Submission profiles enable the envelope, which is why this hid
    on the development path rather than the paper one.
    """

    with pytest.raises(ValueError, match="llm_seed is set but"):
        ra.ResearchAgentPipeline(
            workdir=tmp_path / "off",
            llm=ra.MockLLMClient(),
            llm_seed=7,
            enable_reproducibility_envelope=False,
        )

    # The honest combination still constructs.
    ra.ResearchAgentPipeline(
        workdir=tmp_path / "on",
        llm=ra.MockLLMClient(),
        llm_seed=7,
        enable_reproducibility_envelope=True,
    )
