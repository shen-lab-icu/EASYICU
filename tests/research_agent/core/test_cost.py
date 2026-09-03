"""Tests for the LLM cost-tracking layer (T3.2).

Two layers exercised:

* ``CostMeter`` + ``MeteredClient`` in isolation — token capture from
  a call-scoped ``complete_with_usage`` result (authoritative path) and from
  the ``chars/4`` fallback when the inner client doesn't expose usage.
* End-to-end: ``ResearchAgentPipeline(enable_cost_tracking=True)``
  populates ``manifest.cost_records`` with multiple roles, and writes
  ``cost_summary.md`` + ``cost_records.json`` to the run directory.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# CostMeter unit tests
# ---------------------------------------------------------------------------


def test_meter_records_authoritative_usage_when_inner_exposes_it(ra):
    LLMMessage = ra.LLMRouter  # ensure llm module is loaded for type
    from easyicu.research_agent.providers.llm import LLMMessage  # noqa: F401

    class _ClientWithUsage:
        name = "stub-with-usage"

        def complete_with_usage(self, messages, *, max_tokens=2048, temperature=0.2):
            return "OK", {
                "prompt_tokens": 123,
                "completion_tokens": 45,
                "total_tokens": 168,
            }

    inner = _ClientWithUsage()
    meter = ra.CostMeter()
    metered = ra.MeteredClient(inner, role="planner", meter=meter)

    from easyicu.research_agent.providers.llm import LLMMessage as _Msg

    metered.complete([_Msg(role="user", content="hi")])
    assert len(meter.records) == 1
    rec = meter.records[0]
    assert rec.role == "planner"
    assert rec.prompt_tokens == 123
    assert rec.completion_tokens == 45
    assert rec.total_tokens == 168
    assert rec.is_heuristic is False
    assert rec.model == "stub-with-usage"


def test_meter_prices_the_actual_model_and_persists_the_requested_model(ra, tmp_path):
    from easyicu.research_agent.providers.llm import LLMMessage

    class _HostedFallback:
        name = "hosted-configured"

        def complete_with_usage(self, messages, **_kwargs):  # noqa: ANN003
            return "OK", {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "actual_model": "provider/served-model",
            }

    meter = ra.CostMeter(
        runtime_dir=tmp_path / ".runtime",
        price_table={"provider/served-model": (1.0, 2.0)},
    )
    ra.MeteredClient(_HostedFallback(), role="planner", meter=meter).complete(
        [LLMMessage(role="user", content="hi")]
    )

    assert meter.records[0].model == "provider/served-model"
    receipt = next((tmp_path / ".runtime" / "provider_transport_receipts").glob("*.json"))
    payload = json.loads(receipt.read_text())
    assert payload["model"] == "hosted-configured"
    assert payload["executed_model"] == "provider/served-model"


def test_meter_falls_back_to_heuristic_when_no_last_usage(ra):
    from easyicu.research_agent.providers.llm import LLMMessage

    class _ClientNoUsage:
        name = "stub-no-usage"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            return "thirty-two character response right"

    meter = ra.CostMeter()
    metered = ra.MeteredClient(_ClientNoUsage(), role="coder", meter=meter)

    metered.complete([LLMMessage(role="user", content="hello world hello world")])
    assert len(meter.records) == 1
    rec = meter.records[0]
    assert rec.is_heuristic is True
    assert rec.role == "coder"
    assert rec.prompt_tokens >= 1
    assert rec.completion_tokens >= 1


def test_meter_recovers_all_completed_receipts_across_resume(ra, tmp_path):
    from easyicu.research_agent.providers.llm import LLMMessage

    class _ClientWithUsage:
        name = "priced"

        def complete_with_usage(self, messages, **_kwargs):
            return "OK", {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
            }

    runtime_dir = tmp_path / ".runtime"
    first = ra.CostMeter(
        runtime_dir=runtime_dir,
        price_table={"priced": (1.0, 2.0)},
    )
    ra.MeteredClient(
        _ClientWithUsage(),
        role="planner",
        meter=first,
    ).complete([LLMMessage(role="user", content="first")])

    resumed = ra.CostMeter(
        runtime_dir=runtime_dir,
        price_table={"priced": (1.0, 2.0)},
    )
    assert len(resumed.records) == 1
    assert resumed.summary()["total_tokens"] == 150

    ra.MeteredClient(
        _ClientWithUsage(),
        role="coder",
        meter=resumed,
    ).complete([LLMMessage(role="user", content="second")])
    summary = resumed.summary()
    assert len(resumed.records) == 2
    assert summary["n_calls"] == 2
    assert summary["total_tokens"] == 300
    assert summary["usage_accounting"]["provider_reported"]["n_calls"] == 2
    assert summary["usage_accounting"]["provider_reported"]["total_tokens"] == 300


def test_cost_summary_separates_unknown_usage_from_upper_bound(ra, tmp_path):
    from easyicu.research_agent.providers.llm import LLMMessage

    meter = ra.CostMeter(
        runtime_dir=tmp_path / ".runtime",
        price_table={"priced": (1.0, 2.0)},
    )
    receipt = meter.begin_transport(
        role="repair",
        model="priced",
        messages=[LLMMessage(role="user", content="interrupted")],
        max_tokens=200,
        temperature=0.0,
    )
    meter.finish_transport(
        receipt,
        state="cancelled",
        error_type="KeyboardInterrupt",
    )

    summary = meter.summary()
    accounting = summary["usage_accounting"]
    assert accounting["provider_reported"]["n_calls"] == 0
    assert accounting["usage_unknown"] == {
        "n_calls": 1,
        "states": {"cancelled": 1},
    }
    assert accounting["conservative_upper_bound"]["total_tokens"] > 200


def test_estimated_cost_uses_price_table(ra):
    meter = ra.CostMeter(price_table={"toy-model": (1.0, 2.0)})
    rec = meter.record(
        role="planner",
        model="toy-model",
        prompt_tokens=1_000_000,
        completion_tokens=500_000,
    )
    # 1M @ $1 prompt + 0.5M @ $2 completion = $1 + $1 = $2
    assert rec.estimated_cost_usd == pytest.approx(2.0)


def test_deepseek_models_are_in_default_price_table(ra):
    # The evaluation-protocol reliability-baseline / discovery models must
    # estimate a cost out of the box (token counts are exact regardless).
    meter = ra.CostMeter()
    for model in ("deepseek-chat", "deepseek-reasoner"):
        rec = meter.record(
            role="coder",
            model=model,
            prompt_tokens=1_000_000,
            completion_tokens=0,
        )
        assert rec.estimated_cost_usd is not None
        assert rec.estimated_cost_usd > 0


def test_free_models_record_zero_cost_not_none(ra):
    # Free OpenRouter rows are kept at (0,0) so the meter still records a
    # row (cost == 0.0) rather than dropping to ``None``.
    meter = ra.CostMeter()
    rec = meter.record(
        role="analyzer",
        model="openai/gpt-oss-120b:free",
        prompt_tokens=200_000,
        completion_tokens=20_000,
    )
    assert rec.estimated_cost_usd == pytest.approx(0.0)


def test_estimated_cost_none_for_unknown_model(ra):
    meter = ra.CostMeter()
    rec = meter.record(
        role="planner",
        model="some-unknown-model-9000",
        prompt_tokens=10,
        completion_tokens=20,
    )
    assert rec.estimated_cost_usd is None


def test_summary_aggregates_by_role_and_model(ra):
    meter = ra.CostMeter(price_table={"m1": (1.0, 2.0), "m2": (4.0, 8.0)})
    meter.record(role="planner", model="m1", prompt_tokens=1000, completion_tokens=500)
    meter.record(role="coder", model="m1", prompt_tokens=2000, completion_tokens=1000)
    meter.record(role="coder", model="m2", prompt_tokens=500, completion_tokens=100)

    s = meter.summary()
    assert s["n_calls"] == 3
    assert s["total_prompt_tokens"] == 3500
    assert s["total_completion_tokens"] == 1600

    # by_role
    assert set(s["by_role"]) == {"planner", "coder"}
    assert s["by_role"]["coder"]["n_calls"] == 2

    # by_model
    assert set(s["by_model"]) == {"m1", "m2"}
    assert s["by_model"]["m1"]["n_calls"] == 2

    # cost: m1 row1 = 1000*1 + 500*2 = 2000 / 1e6 = $0.002
    #        m1 row2 = 2000*1 + 1000*2 = 4000 / 1e6 = $0.004
    #        m2 row3 = 500*4 + 100*8 = 2800 / 1e6 = $0.0028
    assert s["total_cost_usd"] == pytest.approx(0.002 + 0.004 + 0.0028)


def test_summary_handles_empty_meter(ra):
    s = ra.CostMeter().summary()
    assert s["n_calls"] == 0
    assert s["total_cost_usd"] == 0.0
    assert s["by_role"] == {}
    assert s["by_model"] == {}
    assert s["any_heuristic"] is False


def test_metered_client_never_trusts_shared_stale_usage(ra):
    """Legacy shared usage is ignored rather than misattributed."""
    from easyicu.research_agent.providers.llm import LLMMessage

    class _ClientStale:
        name = "stub-stale"

        def __init__(self) -> None:
            # Pre-populate as if from a previous call.
            self.last_usage = {
                "prompt_tokens": 9999,
                "completion_tokens": 9999,
                "total_tokens": 19998,
            }

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            self.last_usage = {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
            return "x"

    meter = ra.CostMeter()
    metered = ra.MeteredClient(_ClientStale(), role="writer", meter=meter)
    metered.complete([LLMMessage(role="user", content="x")])
    assert meter.records[-1].is_heuristic is True
    assert meter.records[-1].prompt_tokens != 10


def test_transport_receipt_terminalizes_keyboard_interrupt_without_content(
    ra, tmp_path
):
    from easyicu.research_agent.providers.llm import LLMMessage

    class _InterruptedClient:
        name = "interrupted-provider"

        def complete_with_usage(self, messages, **kwargs):
            raise KeyboardInterrupt

    meter = ra.CostMeter(runtime_dir=tmp_path / ".runtime")
    metered = ra.MeteredClient(_InterruptedClient(), role="planner", meter=meter)

    with pytest.raises(KeyboardInterrupt):
        metered.complete([LLMMessage(role="user", content="private prompt sentinel")])

    receipts = list((tmp_path / ".runtime/provider_transport_receipts").glob("*.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert payload["state"] == "cancelled"
    assert payload["error_type"] == "KeyboardInterrupt"
    assert payload["role"] == "planner"
    assert payload["request_sha256"]
    assert "private prompt sentinel" not in receipts[0].read_text(encoding="utf-8")
    assert meter.records == []


def test_transport_receipt_records_completed_call_usage(ra, tmp_path):
    from easyicu.research_agent.providers.llm import LLMMessage

    class _CompletedClient:
        name = "completed-provider"

        def complete_with_usage(self, messages, **kwargs):
            return "private response sentinel", {
                "prompt_tokens": 7,
                "completion_tokens": 3,
            }

    meter = ra.CostMeter(runtime_dir=tmp_path / ".runtime")
    metered = ra.MeteredClient(_CompletedClient(), role="coder", meter=meter)
    assert metered.complete([LLMMessage(role="user", content="prompt")]) == (
        "private response sentinel"
    )

    receipt = next((tmp_path / ".runtime/provider_transport_receipts").glob("*.json"))
    raw = receipt.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert payload["state"] == "completed"
    assert payload["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
        "is_heuristic": False,
    }
    assert payload["response_sha256"]
    assert "private response sentinel" not in raw


def test_concurrent_writer_usage_cannot_be_charged_to_another_role(ra):
    from easyicu.research_agent.providers.cost import metered_role_resolver
    from easyicu.research_agent.providers.llm import LLMMessage

    class SharedUsageClient:
        name = "shared-provider"

        def __init__(self) -> None:
            self.calls_entered = threading.Barrier(2)

        def complete_with_usage(self, messages, **_kwargs):  # noqa: ANN003
            role = messages[0].content
            self.calls_entered.wait(timeout=2)
            time.sleep(0.15)
            if role == "writer":
                return "writer result", {
                    "prompt_tokens": 11,
                    "completion_tokens": 3,
                }
            return "analyzer result", {
                "prompt_tokens": 29,
                "completion_tokens": 7,
            }

    shared = SharedUsageClient()
    meter = ra.CostMeter()
    resolver = metered_role_resolver(shared, meter)
    writer = resolver("writer")
    analyzer = resolver("analyzer")
    writer_thread = threading.Thread(
        target=lambda: writer.complete([LLMMessage(role="user", content="writer")])
    )
    analyzer_thread = threading.Thread(
        target=lambda: analyzer.complete([LLMMessage(role="user", content="analyzer")])
    )
    started = time.monotonic()
    writer_thread.start()
    analyzer_thread.start()
    writer_thread.join(timeout=2)
    analyzer_thread.join(timeout=2)
    elapsed = time.monotonic() - started

    by_role = {record.role: record for record in meter.records}
    assert by_role["writer"].prompt_tokens == 11
    assert by_role["writer"].completion_tokens == 3
    assert by_role["analyzer"].prompt_tokens == 29
    assert by_role["analyzer"].completion_tokens == 7
    assert elapsed < 0.27


def test_reproducibility_wrapper_keeps_concurrent_provider_calls_parallel(ra):
    from easyicu.research_agent.providers.llm import LLMMessage
    from easyicu.research_agent.replication.envelope import envelope_role_resolver

    class SharedUsageClient:
        name = "shared-provider"

        def __init__(self) -> None:
            self.calls_entered = threading.Barrier(2)

        def complete_with_usage(self, messages, **_kwargs):  # noqa: ANN003
            role = messages[0].content
            self.calls_entered.wait(timeout=2)
            time.sleep(0.15)
            if role == "writer":
                return "writer result", {
                    "prompt_tokens": 31,
                    "completion_tokens": 5,
                }
            return "analyzer result", {
                "prompt_tokens": 41,
                "completion_tokens": 9,
            }

    shared = SharedUsageClient()
    envelope = ra.ReproEnvelope(run_id="parallel-usage")
    envelope_resolver = envelope_role_resolver(shared, envelope, seed=11)
    meter = ra.CostMeter()
    writer = ra.MeteredClient(envelope_resolver("writer"), role="writer", meter=meter)
    analyzer = ra.MeteredClient(
        envelope_resolver("analyzer"), role="analyzer", meter=meter
    )
    writer_thread = threading.Thread(
        target=lambda: writer.complete([LLMMessage(role="user", content="writer")])
    )
    analyzer_thread = threading.Thread(
        target=lambda: analyzer.complete([LLMMessage(role="user", content="analyzer")])
    )
    started = time.monotonic()
    writer_thread.start()
    analyzer_thread.start()
    writer_thread.join(timeout=2)
    analyzer_thread.join(timeout=2)
    elapsed = time.monotonic() - started

    by_role = {record.role: record for record in meter.records}
    assert by_role["writer"].prompt_tokens == 31
    assert by_role["analyzer"].prompt_tokens == 41
    assert all(not record.is_heuristic for record in meter.records)
    assert len(envelope.calls) == 2
    assert elapsed < 0.27


def test_meter_uses_reproducibility_wrappers_resolved_model_identity(ra):
    from easyicu.research_agent.providers.llm import LLMMessage

    class _OpenAIStyleClient:
        name = "openai"
        _model = "gpt-4o"

        def complete_with_usage(self, messages, **_kwargs):  # noqa: ANN003
            return "ok", {"prompt_tokens": 1_000, "completion_tokens": 100}

    envelope = ra.ReproEnvelope(run_id="cost-model-identity")
    recording = ra.ReproRecordingClient(
        _OpenAIStyleClient(),
        role="planner",
        envelope=envelope,
    )
    meter = ra.CostMeter(price_table={"gpt-4o": (1.0, 2.0)})
    metered = ra.MeteredClient(recording, role="planner", meter=meter)

    metered.complete([LLMMessage(role="user", content="plan")])

    assert meter.records[0].model == "gpt-4o"
    assert meter.records[0].estimated_cost_usd is not None


# ---------------------------------------------------------------------------
# End-to-end: pipeline writes cost summary + manifest carries records
# ---------------------------------------------------------------------------


def test_pipeline_with_cost_tracking_records_per_role_calls(
    ra, synthetic_cohort, tmp_path
):
    """A full pipeline run with ``enable_cost_tracking=True`` must:
    1. populate ``manifest.cost_records`` with at least one entry per
       agent role that actually ran;
    2. write ``cost_summary.md`` + ``cost_records.json`` artefacts to
       the run directory and register both in the evidence store.

    The built-in contextual mock now produces valid deterministic scripts, so
    the clean path reaches Analyzer and Writer without spending a repair call.
    """
    pipeline = ra.ResearchAgentPipeline(
        workdir=str(tmp_path),
        llm=ra.MockLLMClient(),
        enable_cost_tracking=True,
        # Deterministic and fast: skip the long extras.
        enable_literature=False,
        enable_visual_qa=False,
        enable_memory=False,
        enable_latex=False,
    )
    result = pipeline.run(
        skill="association_analysis", cohort=synthetic_cohort, database="synthetic"
    )

    run_dir = Path(result.workdir)
    assert (run_dir / "cost_summary.md").exists()
    assert (run_dir / "cost_records.json").exists()

    # Machine-readable aggregate consumed by the bench scorer / Fig.3 builder.
    assert (run_dir / "cost_summary.json").exists()
    summary = json.loads((run_dir / "cost_summary.json").read_text(encoding="utf-8"))
    for key in (
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_tokens",
        "total_cost_usd",
    ):
        assert key in summary

    records = json.loads((run_dir / "cost_records.json").read_text(encoding="utf-8"))
    assert isinstance(records, list)
    assert len(records) >= 2

    roles = {r["role"] for r in records if r.get("role")}
    assert "coder" in roles
    assert "repair" not in roles
    assert "analyzer" in roles
    assert "writer" in roles

    # Manifest must carry the same records.
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert "cost_records" in manifest
    assert len(manifest["cost_records"]) == len(records)


def test_pipeline_without_cost_tracking_writes_no_cost_files(
    ra, synthetic_cohort, tmp_path
):
    """Default pipeline behaviour: no cost_summary.md, no cost_records.json,
    empty cost_records in the manifest."""
    pipeline = ra.ResearchAgentPipeline(
        workdir=str(tmp_path),
        llm=ra.MockLLMClient(),
        # default: enable_cost_tracking=False
        enable_literature=False,
        enable_visual_qa=False,
        enable_memory=False,
        enable_latex=False,
    )
    result = pipeline.run(
        skill="association_analysis", cohort=synthetic_cohort, database="synthetic"
    )

    run_dir = Path(result.workdir)
    assert not (run_dir / "cost_summary.md").exists()
    assert not (run_dir / "cost_records.json").exists()

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest.get("cost_records") == []
