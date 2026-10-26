"""Synthetic, payload-free trust receipts; no provider or patient data access."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from easyicu.webserver.model_request_records import conversation_egress, run_egress
from tests.support.trust_help import call, ledger


def test_every_attempt_is_projected_without_patient_derived_payloads(tmp_path):
    secret = "SYNTHETIC_PRIVATE_VALUE"
    paths = [tmp_path / ".runtime/provider_hard_stop_ledger.json",
             tmp_path / ".runtime/provider_hard_stop_retry_a.json",
             tmp_path / "report_revisions/a/runtime/provider_hard_stop.json"]
    for path in paths:
        ledger(path, [call(payload=secret, prompt=secret, aggregate={secret: 2},
                           model=secret, task_id=secret),
                      call(role=secret, provider=secret, data_classes=[secret],
                           redaction_status=secret)])
    result = run_egress(tmp_path)
    assert result["status"] == "available"
    assert len(result["requests"]) == 6
    assert secret not in json.dumps(result)
    first, second = result["requests"][:2]
    assert first["purpose"] == "planner"
    assert first["provider"] == "openrouter"
    assert first["byte_size"] == 42
    assert first["data_classes"] == ["text", "schema"]
    assert first["payload_status"] == "withheld"
    assert second["provider"] == second["purpose"] == "unknown"
    assert second["redaction_status"] == "not_recorded"


@pytest.mark.parametrize("damage", ["digest", "symlink", "parent_symlink"])
def test_untrusted_ledgers_fail_closed(tmp_path, damage):
    path = tmp_path / ".runtime/provider_hard_stop_ledger.json"
    ledger(path, [call()])
    if damage == "digest":
        path.write_text(path.read_text().replace('"planner"', '"writer"'))
    elif damage == "symlink":
        saved = tmp_path / "other.json"
        path.rename(saved)
        path.symlink_to(saved)
    else:
        saved = tmp_path / "other"
        path.parent.rename(saved)
        path.parent.symlink_to(saved, target_is_directory=True)
    result = run_egress(tmp_path)
    assert result["status"] == "unavailable"
    assert result["requests"] == []


def test_absent_and_historical_records_do_not_claim_zero_egress(tmp_path):
    assert run_egress(tmp_path)["status"] == "not_recorded"
    ledger(tmp_path / ".runtime/provider_hard_stop_ledger.json",
           [call(provider=None, data_classes=None, prompt_payload_bytes=None)])
    row = run_egress(tmp_path)["requests"][0]
    assert row["provider"] == "unknown" and row["byte_size"] is None
    assert row["data_classes"] == []
    assert row["redaction_status"] == "not_recorded"


def test_transport_failure_state_is_not_lost_when_usage_is_unknown(tmp_path):
    ledger(tmp_path / ".runtime/provider_hard_stop_ledger.json", [
        call(state="failed_usage_unknown", error_type="SyntheticFailure"),
        call(state="completed_usage_overflow"),
    ])
    first, second = run_egress(tmp_path)["requests"]
    assert first["state"] == "failed_usage_unknown" and first["failed"] is True
    assert second["state"] == "completed_usage_overflow" and second["failed"] is False


@pytest.mark.parametrize("provider", ["custom", "claude"])
def test_reviewed_provider_and_repair_role_names_are_visible(tmp_path, provider):
    ledger(tmp_path / ".runtime/provider_hard_stop_ledger.json", [call(provider=provider, role="repair")])
    row = run_egress(tmp_path)["requests"][0]
    assert row["provider"] == provider and row["purpose"] == "repair"


def test_host_visual_review_purpose_remains_visible(tmp_path):
    ledger(tmp_path / ".runtime/provider_hard_stop_ledger.json", [call(role="visual_qa")])
    assert run_egress(tmp_path)["requests"][0]["purpose"] == "visual_qa"


@pytest.mark.parametrize("size", [True, -1, "SYNTHETIC_PRIVATE_VALUE", 2**53])
def test_invalid_size_never_becomes_payload_text(tmp_path, size):
    ledger(tmp_path / ".runtime/provider_hard_stop_ledger.json", [call(prompt_payload_bytes=size)])
    assert run_egress(tmp_path)["requests"][0]["byte_size"] is None


def test_conversation_receipts_never_open_transcripts(tmp_path):
    session_id = "synthetic-conversation"
    record = SimpleNamespace(session_id=session_id, pi_session_file="DO_NOT_READ")
    service = SimpleNamespace(_get_record=lambda _: record,
                              gateway=SimpleNamespace(session_dir=tmp_path))
    assert conversation_egress(service, session_id)["status"] == "not_recorded"
    path = tmp_path / ".egress" / (hashlib.sha256(session_id.encode()).hexdigest() + ".jsonl")
    path.parent.mkdir()
    receipt = {"schema_version": "easyicu.model-request/1", "purpose": "conversation",
               "provider": "openai", "byte_size": 23, "byte_size_basis": "context_json_utf8",
               "data_classes": ["text", "schema"], "redaction_status": "not_recorded",
               "payload": "SYNTHETIC_PRIVATE_VALUE"}
    path.write_text(json.dumps(receipt) + "\n")
    result = conversation_egress(service, session_id)
    assert result["requests"][0]["byte_size"] == 23
    assert "SYNTHETIC_PRIVATE_VALUE" not in json.dumps(result)
    path.write_text(path.read_text() + "{incomplete")
    assert conversation_egress(service, session_id)["status"] == "unavailable"


@pytest.mark.parametrize("damage", ["symlink", "oversize", "schema"])
def test_conversation_receipts_reject_unsafe_files(tmp_path, damage):
    identity = hashlib.sha256(b"synthetic").hexdigest()
    service = SimpleNamespace(_get_record=lambda _: SimpleNamespace(session_id="synthetic"),
                              gateway=SimpleNamespace(session_dir=tmp_path))
    path = tmp_path / ".egress" / f"{identity}.jsonl"
    path.parent.mkdir()
    if damage == "symlink":
        other = tmp_path / "other"
        other.write_text('{"schema_version":"easyicu.model-request/1"}\n')
        path.symlink_to(other)
    elif damage == "oversize":
        with path.open("wb") as stream:
            stream.truncate(8 * 1024 * 1024 + 1)
    else:
        path.write_text('{"schema_version":"future"}\n')
    assert conversation_egress(service, "synthetic")["status"] == "unavailable"
