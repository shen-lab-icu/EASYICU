from __future__ import annotations

import json
from pathlib import Path

from benchmarks.figure2_canonical9 import provider_protocol_probe as probe


class _FakeClient:
    name = "fake"
    provider_attempt_budget_aware = False

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.last_finish_reason = "stop"

    def complete_with_usage(self, messages, **kwargs):  # noqa: ANN001, ANN003
        del messages, kwargs
        response = self.responses.pop(0)
        self.last_finish_reason = "length" if response.startswith("ALPHA ") else "stop"
        return response, {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }


def _fake_clients() -> dict[str, _FakeClient]:
    responses = [
        "READY",
        '{"steps":[{"id":"step_1","method":"mean"}]}',
        "def summarize(values):\n    return {'n': len(values), 'mean': 0.0}",
        "def add(a, b):\n    return a + b",
        '{"claim":"Toy result.","evidence_ids":["toy-evidence-1"]}',
        "ALPHA " * 8,
    ]
    shared = _FakeClient(responses)
    return {"low": shared, "medium": shared}


def test_protocol_probe_is_six_call_bounded_and_stores_no_response_text(
    tmp_path: Path,
    monkeypatch,
) -> None:
    clients = _fake_clients()
    monkeypatch.setattr(
        probe,
        "_git_identity",
        lambda _root: ("a" * 40, True),
    )

    report = probe.run_provider_protocol_probe(
        output_dir=tmp_path / "out",
        model="gpt-5.6-luna",
        base_url="http://127.0.0.1:8317/v1",
        client_for_effort=lambda effort: clients[effort],
        repo_root=tmp_path,
    )

    assert report["status"] == "passed"
    assert report["call_count"] == 6
    assert report["transport_attempts"] == 6
    assert report["transport_retries"] == 0
    assert report["truncation_finish_reason"] == "length"
    serialized = json.dumps(report)
    assert "Toy result." not in serialized
    assert "def summarize" not in serialized
    assert all(item["response_sha256"] for item in report["probes"])
    assert (tmp_path / "out" / probe.PROVIDER_PROTOCOL_REPORT_FILENAME).is_file()
    assert (tmp_path / "out" / probe.PROVIDER_PROTOCOL_LEDGER_FILENAME).is_file()


def test_protocol_probe_rejects_non_frozen_destination(tmp_path: Path) -> None:
    clients = _fake_clients()

    try:
        probe.run_provider_protocol_probe(
            output_dir=tmp_path,
            model="gpt-5.6-luna",
            base_url="https://api.openai.com/v1",
            client_for_effort=lambda effort: clients[effort],
            repo_root=tmp_path,
        )
    except probe.ProviderProtocolProbeError as exc:
        assert "restricted" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("external Provider destination was accepted")
