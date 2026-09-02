from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

import benchmarks.figure2_icu_agent_v2.formal_provider_gate as formal_provider_gate
import benchmarks.figure2_icu_agent_v2.formal_authority as formal_authority
import benchmarks.figure2_icu_agent_v2.review_bundle_normalizer as review_bundle_normalizer
from benchmarks.figure2_icu_agent_v2.design_v2_1 import DesignContractError
from benchmarks.figure2_icu_agent_v2.formal_provider_gate import (
    FormalCallCoordinate,
    FormalProviderBudgetMissingError,
    complete_formal_provider_call,
)
from benchmarks.figure2_icu_agent_v2.formal_easyicu_runner import (
    FormalEasyICUModelRouter,
)
from easyicu.research_agent.authority.provider_hard_stop import (
    ProviderHardStopExceeded,
    ProviderHardStopLedger,
    ProviderHardStopLimits,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from benchmarks.figure2_icu_agent_v2.review_bundle_normalizer import (
    CANONICAL_FILES,
    ReviewBundleNormalizationError,
    normalize_review_bundle,
)
from easyicu.research_agent.providers.protocol import LLMMessage
from easyicu.research_agent.providers.client_trust import ProviderConfigurationError
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


_ACTION_SPACE = json.loads(
    review_bundle_normalizer.ACTION_SPACE_PATH.read_text(encoding="utf-8")
)
_INTERNAL_MARKERS = sorted(
    {stage["stage_id"] for stage in _ACTION_SPACE["stages"]}
    | {
        reason
        for stage in _ACTION_SPACE["stages"]
        for reason in stage["failure_reason_codes"]
    }
)


class _TransportSpy:
    name = "transport-spy"

    def __init__(self) -> None:
        self.called = False

    def complete(self, messages, **kwargs):
        del messages, kwargs
        self.called = True
        return "unexpected"


def _write_bundle(root: Path) -> None:
    payloads = {
        "01_plan.json": {"population": "adult ICU stays", "method": "logistic"},
        "02_cohort.json": {"denominator": 42, "source": "EasyICU"},
        "03_results.json": {"estimate": 1.25, "arm": "easyicu_full"},
        "04_diagnostics.json": {"complete": True},
        "05_evidence_manifest.json": {
            "artifact": "/Users/example/project/result.csv",
            "sha256": "a" * 64,
        },
        "07_run_receipt.json": {
            "terminal_status": "completed",
            "within_frozen_budget": True,
            "failure_category": None,
            "agent_asserted_mandatory_artifact_presence": {
                "01_plan.json": True
            },
            "substantive_output_files": {
                "02_cohort.json": True,
                "03_results.json": True,
                "04_diagnostics.json": True,
                "06_report.md": True,
            },
            "provider_tokens": 1234,
            "model_turns": 3,
            "tool_call_sequence": ["python"],
        },
    }
    for name, payload in payloads.items():
        (root / name).write_text(json.dumps(payload), encoding="utf-8")
    (root / "06_report.md").write_text(
        "EasyICU produced the estimate for easyicu_full.\n",
        encoding="utf-8",
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _signed_qualification_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    arm: str = "generic_code_agent",
    call_id: str = "generic_0001",
) -> tuple[dict[str, object], dict[str, str]]:
    launch = json.loads(formal_authority.LAUNCH_CONTRACT_PATH.read_text())
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    launch["signature_verification"]["trusted_signer_id"] = "external-custodian-1"
    public_key_text = base64.b64encode(public_key).decode("ascii")
    launch["signature_verification"]["trusted_public_key_base64"] = public_key_text
    launch_path = tmp_path / "launch.json"
    launch_path.write_text(json.dumps(launch), encoding="utf-8")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text('{"review":"candidate"}', encoding="utf-8")
    monkeypatch.setattr(formal_authority, "LAUNCH_CONTRACT_PATH", launch_path)
    monkeypatch.setattr(formal_authority, "PROTOCOL_PATH", protocol_path)

    coordinate = {
        "scope": "qualification12",
        "task_id": "qualification12_a_01",
        "arm": arm,
        "call_id": call_id,
    }
    binding = {
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "design_commit": "a" * 40,
        "annotated_tag": "figure2-v2.1-test",
    }
    receipt_count = len(launch["required_receipts"]["qualification_preconditions"])
    receipt_payloads: dict[str, dict[str, object]] = {
        f"qualification_preconditions:{index:02d}": {
            "schema_version": "easyicu.figure2_launch_receipt/1",
            "receipt_id": f"qualification_preconditions:{index:02d}",
            "status": "passed",
            "binding": binding,
            "evidence_sha256": hashlib.sha256(
                f"qualification-evidence-{index}".encode()
            ).hexdigest(),
            "issuer": "independent-custodian",
            "issued_at_utc": "2026-09-01T12:00:00Z",
        }
        for index in range(1, receipt_count + 1)
    }
    receipt_payloads["qualification_preconditions:01"]["details"] = {
        "registry_name": "test-registry",
        "immutable_registration_id": "offline-test-only",
        "registration_timestamp_utc": "2026-09-01T11:00:00Z",
        "embargo_or_public_status": "embargoed",
        "package_sha256": "1" * 64,
        "protocol_sha256": binding["protocol_sha256"],
        **{
            field: hashlib.sha256(path.read_bytes()).hexdigest()
            for field, path in formal_authority.REGISTERED_SOURCE_PATHS.items()
        },
        "design_commit": binding["design_commit"],
        "annotated_tag": binding["annotated_tag"],
        "registrant_identity": "test-registrant",
        "trusted_authority_signer_identity": "external-custodian-1",
        "trusted_authority_ed25519_public_key_base64": public_key_text,
        "amendment_policy_acknowledged": True,
    }
    declaration = {
        "schema_version": "easyicu.figure2_atomic_declaration/1",
        "signer_id": "external-custodian-1",
        "scope": "qualification12",
        **binding,
        "receipt_sha256": {
            receipt_id: hashlib.sha256(_canonical_json(payload)).hexdigest()
            for receipt_id, payload in receipt_payloads.items()
        },
        "authorized_call_coordinates": [coordinate],
    }
    envelope: dict[str, object] = {
        "atomic_declaration": declaration,
        "atomic_declaration_signature_base64": base64.b64encode(
            private_key.sign(_canonical_json(declaration))
        ).decode("ascii"),
        "receipt_payloads": receipt_payloads,
    }
    return envelope, coordinate


def test_formal_provider_gate_denies_before_transport() -> None:
    client = _TransportSpy()
    coordinate = FormalCallCoordinate(
        scope="qualification12",
        task_id="qualification12_a_01",
        arm="generic_code_agent",
        call_id="call_001",
    )

    with pytest.raises(DesignContractError) as exc_info:
        complete_formal_provider_call(
            client,
            [LLMMessage(role="user", content="do not send")],
            receipts={},
            coordinate=coordinate,
        )

    assert exc_info.value.reason_code == "FORMAL_AUTHORITY_SIGNER_NOT_REGISTERED"
    assert client.called is False


def test_formal_authority_accepts_only_exact_signed_coordinate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope, coordinate = _signed_qualification_authority(tmp_path, monkeypatch)

    receipt = formal_authority.authorize_formal_provider_call(
        {"receipts": envelope, "call_coordinate": coordinate}
    )

    assert receipt["authorized"] is True
    assert receipt["call_coordinate"] == coordinate
    with pytest.raises(DesignContractError) as exc_info:
        formal_authority.authorize_formal_provider_call(
            {
                "receipts": envelope,
                "call_coordinate": {**coordinate, "call_id": "generic_0002"},
            }
        )
    assert exc_info.value.reason_code == "FORMAL_AUTHORITY_COORDINATE_NOT_DECLARED"


def test_signed_authority_and_budget_gate_reach_only_offline_mock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope, coordinate = _signed_qualification_authority(tmp_path, monkeypatch)
    limits = ProviderHardStopLimits(
        max_provider_attempts_per_run=1,
        max_provider_attempts_per_batch=1,
        max_total_tokens_per_run=200_000,
        max_total_tokens_per_batch=200_000,
        max_estimated_cost_usd_per_batch=1.0,
        max_wall_clock_seconds_per_task=60.0,
        input_cost_usd_per_million_tokens=0.1,
        output_cost_usd_per_million_tokens=0.1,
    )
    task_budget = ProviderHardStopLedger(
        path=tmp_path / "integrated-hard-stop.json",
        task_ids=(coordinate["task_id"],),
        limits=limits,
        batch_id="offline-signed-test",
    ).start_task(coordinate["task_id"])
    client = ScriptedMockLLMClient(["authorized offline response"])

    response = complete_formal_provider_call(
        client,
        [LLMMessage(role="user", content="offline only")],
        receipts=envelope,
        coordinate=FormalCallCoordinate(**coordinate),
        provider_hard_stop=task_budget,
        max_tokens=16,
    )

    assert response == "authorized offline response"
    assert len(client.calls) == 1


def test_easyicu_router_authorizes_pipeline_role_before_offline_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope, coordinate = _signed_qualification_authority(
        tmp_path,
        monkeypatch,
        arm="easyicu_full",
        call_id="easyicu_0001",
    )
    limits = ProviderHardStopLimits(
        max_provider_attempts_per_run=1,
        max_provider_attempts_per_batch=1,
        max_total_tokens_per_run=200_000,
        max_total_tokens_per_batch=200_000,
        max_estimated_cost_usd_per_batch=1.0,
        max_wall_clock_seconds_per_task=60.0,
        input_cost_usd_per_million_tokens=0.1,
        output_cost_usd_per_million_tokens=0.1,
    )
    task_budget = ProviderHardStopLedger(
        path=tmp_path / "easyicu-hard-stop.json",
        task_ids=(coordinate["task_id"],),
        limits=limits,
        batch_id="offline-easyicu-test",
    ).start_task(coordinate["task_id"])
    client = ScriptedMockLLMClient(["authorized EasyICU offline response"])
    router = FormalEasyICUModelRouter(
        client,
        receipts=envelope,
        scope=coordinate["scope"],
        task_id=coordinate["task_id"],
        provider_hard_stop=task_budget,
    )

    response = router.for_role("planner").complete(
        [LLMMessage(role="user", content="offline only")],
        max_tokens=16,
    )

    assert response == "authorized EasyICU offline response"
    assert len(client.calls) == 1


def test_easyicu_router_denies_before_transport_without_registered_signer(
    tmp_path: Path,
) -> None:
    task_id = "qualification12_a_01"
    limits = ProviderHardStopLimits(
        max_provider_attempts_per_run=1,
        max_provider_attempts_per_batch=1,
        max_total_tokens_per_run=200_000,
        max_total_tokens_per_batch=200_000,
        max_estimated_cost_usd_per_batch=1.0,
        max_wall_clock_seconds_per_task=60.0,
        input_cost_usd_per_million_tokens=0.1,
        output_cost_usd_per_million_tokens=0.1,
    )
    task_budget = ProviderHardStopLedger(
        path=tmp_path / "denied-easyicu-hard-stop.json",
        task_ids=(task_id,),
        limits=limits,
        batch_id="denied-easyicu-test",
    ).start_task(task_id)
    client = _TransportSpy()
    router = FormalEasyICUModelRouter(
        client,
        receipts={},
        scope="qualification12",
        task_id=task_id,
        provider_hard_stop=task_budget,
    )

    with pytest.raises(DesignContractError) as exc_info:
        router.for_role("planner").complete(
            [LLMMessage(role="user", content="do not send")]
        )

    assert exc_info.value.reason_code == "FORMAL_AUTHORITY_SIGNER_NOT_REGISTERED"
    assert client.called is False


def test_formal_authority_rejects_tampered_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope, coordinate = _signed_qualification_authority(tmp_path, monkeypatch)
    receipt_payloads = envelope["receipt_payloads"]
    assert isinstance(receipt_payloads, dict)
    first_receipt = receipt_payloads["qualification_preconditions:01"]
    assert isinstance(first_receipt, dict)
    first_receipt["issuer"] = "tampered-issuer"

    with pytest.raises(DesignContractError) as exc_info:
        formal_authority.authorize_formal_provider_call(
            {"receipts": envelope, "call_coordinate": coordinate}
        )

    assert exc_info.value.reason_code == "FORMAL_AUTHORITY_RECEIPT_DIGEST_MISMATCH"


def test_formal_authority_rejects_invalid_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope, coordinate = _signed_qualification_authority(tmp_path, monkeypatch)
    signature = envelope["atomic_declaration_signature_base64"]
    assert isinstance(signature, str)
    raw_signature = bytearray(base64.b64decode(signature))
    raw_signature[0] ^= 1
    envelope["atomic_declaration_signature_base64"] = base64.b64encode(
        raw_signature
    ).decode("ascii")

    with pytest.raises(DesignContractError) as exc_info:
        formal_authority.authorize_formal_provider_call(
            {"receipts": envelope, "call_coordinate": coordinate}
        )

    assert exc_info.value.reason_code == "FORMAL_AUTHORITY_SIGNATURE_INVALID"


def test_formal_provider_gate_preserves_production_transport_trust_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _TransportSpy()
    monkeypatch.setattr(
        formal_provider_gate,
        "authorize_formal_provider_call",
        lambda receipts: None,
    )

    with pytest.raises(ProviderConfigurationError):
        complete_formal_provider_call(
            client,
            [LLMMessage(role="user", content="still do not send")],
            receipts={"future": "registered"},
            coordinate=FormalCallCoordinate(
                scope="core_wp2_wp3",
                task_id="icu27_t01",
                arm="easyicu_full",
                call_id="call_001",
            ),
            provider_hard_stop=object(),  # type: ignore[arg-type]
        )

    assert client.called is False


def test_formal_provider_gate_requires_shared_budget_after_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _TransportSpy()
    monkeypatch.setattr(
        formal_provider_gate,
        "authorize_formal_provider_call",
        lambda receipts: None,
    )

    with pytest.raises(FormalProviderBudgetMissingError):
        complete_formal_provider_call(
            client,
            [LLMMessage(role="user", content="do not send")],
            receipts={"future": "registered"},
            coordinate=FormalCallCoordinate(
                scope="qualification12",
                task_id="qualification12_a_01",
                arm="generic_code_agent",
                call_id="generic_0001",
            ),
        )

    assert client.called is False


def test_formal_call_coordinate_rejects_unknown_scope_or_arm() -> None:
    with pytest.raises(ValueError, match="unsupported formal scope"):
        FormalCallCoordinate(
            scope="dev9",
            task_id="dev_01",
            arm="generic_code_agent",
            call_id="call_001",
        )


def test_formal_provider_gate_uses_shared_durable_hard_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        formal_provider_gate,
        "authorize_formal_provider_call",
        lambda receipts: None,
    )
    limits = ProviderHardStopLimits(
        max_provider_attempts_per_run=1,
        max_provider_attempts_per_batch=1,
        max_total_tokens_per_run=200_000,
        max_total_tokens_per_batch=200_000,
        max_estimated_cost_usd_per_batch=1.0,
        max_wall_clock_seconds_per_task=60.0,
        input_cost_usd_per_million_tokens=0.1,
        output_cost_usd_per_million_tokens=0.1,
    )
    task_budget = ProviderHardStopLedger(
        path=tmp_path / "hard_stop.json",
        task_ids=("qualification12_a_01",),
        limits=limits,
        batch_id="offline-test",
    ).start_task("qualification12_a_01")
    client = ScriptedMockLLMClient(["first", "must-not-run"])
    coordinate = FormalCallCoordinate(
        scope="qualification12",
        task_id="qualification12_a_01",
        arm="generic_code_agent",
        call_id="generic_0001",
    )

    assert complete_formal_provider_call(
        client,
        [LLMMessage(role="user", content="offline")],
        receipts={"future": "registered"},
        coordinate=coordinate,
        provider_hard_stop=task_budget,
        max_tokens=16,
    ) == "first"

    with pytest.raises(ProviderHardStopExceeded) as exc_info:
        complete_formal_provider_call(
            client,
            [LLMMessage(role="user", content="offline again")],
            receipts={"future": "registered"},
            coordinate=FormalCallCoordinate(
                scope="qualification12",
                task_id="qualification12_a_01",
                arm="generic_code_agent",
                call_id="generic_0002",
            ),
            provider_hard_stop=task_budget,
            max_tokens=16,
        )

    assert exc_info.value.code == "RUN_PROVIDER_ATTEMPT_LIMIT"
    assert len(client.calls) == 1
    with pytest.raises(ValueError, match="unsupported formal arm"):
        FormalCallCoordinate(
            scope="qualification12",
            task_id="qualification12_a_01",
            arm="other",
            call_id="call_001",
        )


def test_normalizer_preserves_science_and_hides_arm_resource_fingerprints(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path)

    result = normalize_review_bundle(tmp_path)

    assert tuple(result.files) == CANONICAL_FILES
    normalized_results = json.loads(result.files["03_results.json"])
    assert normalized_results["estimate"] == 1.25
    assert normalized_results["arm"] == "the producing workflow"
    normalized_receipt = json.loads(result.files["07_run_receipt.json"])
    assert set(normalized_receipt) == {
        "terminal_status",
        "within_frozen_budget",
        "failure_category",
        "agent_asserted_mandatory_artifact_presence",
        "substantive_output_files",
    }
    assert "provider_tokens" not in normalized_receipt
    assert "EasyICU" not in result.files["06_report.md"].decode("utf-8")
    assert result.pre_normalization_sha256.keys() == result.post_normalization_sha256.keys()
    assert {finding["rule"] for finding in result.redaction_log} == {
        "easyicu_name",
        "arm_label",
        "repository_path",
    }


@pytest.mark.parametrize("marker", _INTERNAL_MARKERS)
def test_normalizer_rejects_every_frozen_internal_marker(
    tmp_path: Path,
    marker: str,
) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "06_report.md").write_text(
        f"Internal marker: {marker}\n",
        encoding="utf-8",
    )

    with pytest.raises(ReviewBundleNormalizationError) as exc_info:
        normalize_review_bundle(tmp_path)

    assert exc_info.value.reason_code == "REVIEW_BUNDLE_UNSAFE_MARKER"


def test_normalizer_redacts_container_and_repository_paths(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    manifest = {
        "paths": [
            "/workspace/run/out.csv",
            "/home/agent/plan.py",
            "benchmarks/figure2_icu_agent_v2/heldout27_taskbank_v1.jsonl",
            "/Users/example/out.csv.",
        ]
    }
    (tmp_path / "05_evidence_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    result = normalize_review_bundle(tmp_path)
    normalized = json.loads(result.files["05_evidence_manifest.json"])

    assert normalized["paths"] == [
        "<redacted-path>",
        "<redacted-path>",
        "<redacted-path>",
        "<redacted-path>.",
    ]


@pytest.mark.parametrize(
    "fingerprint",
    [
        "analysis completed after 37 model turns",
        "the workflow made 52 tool calls",
        "provider_calls=4",
        "provider tokens: 1200",
        "per-tool latency was retained",
    ],
)
def test_normalizer_rejects_resource_fingerprints_outside_raw_receipt(
    tmp_path: Path,
    fingerprint: str,
) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "03_results.json").write_text(
        json.dumps({"estimate": 1.25, "note": fingerprint}),
        encoding="utf-8",
    )

    with pytest.raises(ReviewBundleNormalizationError) as exc_info:
        normalize_review_bundle(tmp_path)

    assert exc_info.value.reason_code == "REVIEW_BUNDLE_RESOURCE_FINGERPRINT"


def test_normalizer_detects_numeric_content_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_bundle(tmp_path)
    original = review_bundle_normalizer._normalize_value

    def corrupt_numeric_content(value, *, file_name, location):
        normalized, findings = original(
            value,
            file_name=file_name,
            location=location,
        )
        if file_name == "03_results.json" and location == "":
            normalized["estimate"] = 9.99
        return normalized, findings

    monkeypatch.setattr(
        review_bundle_normalizer,
        "_normalize_value",
        corrupt_numeric_content,
    )

    with pytest.raises(ReviewBundleNormalizationError) as exc_info:
        normalize_review_bundle(tmp_path)

    assert exc_info.value.reason_code == "REVIEW_BUNDLE_NUMERIC_CONTENT_CHANGED"


def test_normalizer_rejects_overflowed_json_number_with_typed_reason(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "03_results.json").write_text(
        '{"estimate": 1e400}',
        encoding="utf-8",
    )

    with pytest.raises(ReviewBundleNormalizationError) as exc_info:
        normalize_review_bundle(tmp_path)

    assert exc_info.value.reason_code == "REVIEW_BUNDLE_NONFINITE_JSON"


def test_normalizer_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "03_results.json").write_text(
        '{"estimate": 1.25, "estimate": 9.99}',
        encoding="utf-8",
    )

    with pytest.raises(ReviewBundleNormalizationError) as exc_info:
        normalize_review_bundle(tmp_path)

    assert exc_info.value.reason_code == "REVIEW_BUNDLE_JSON_KEY_DUPLICATE"


def test_normalizer_rejects_unknown_internal_marker(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "06_report.md").write_text(
        "Terminal reason: FORMAL_PROVIDER_CALL_NOT_AUTHORIZED\n",
        encoding="utf-8",
    )

    with pytest.raises(ReviewBundleNormalizationError) as exc_info:
        normalize_review_bundle(tmp_path)

    assert exc_info.value.reason_code == "REVIEW_BUNDLE_UNSAFE_MARKER"


def test_normalizer_preserves_legitimate_screaming_case_clinical_field(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path)
    result_path = tmp_path / "03_results.json"
    result_path.write_text(
        json.dumps({"field": "ICU_FREE_DAYS_28", "estimate": 1.25}),
        encoding="utf-8",
    )

    result = normalize_review_bundle(tmp_path)

    assert json.loads(result.files["03_results.json"])["field"] == (
        "ICU_FREE_DAYS_28"
    )


def test_normalizer_rejects_noncanonical_file_set(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "debug.log").write_text("hidden fingerprint", encoding="utf-8")

    with pytest.raises(ReviewBundleNormalizationError) as exc_info:
        normalize_review_bundle(tmp_path)

    assert exc_info.value.reason_code == "REVIEW_BUNDLE_FILE_SET_INVALID"


def test_normalizer_rejects_nonboolean_artifact_presence(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    receipt_path = tmp_path / "07_run_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["agent_asserted_mandatory_artifact_presence"] = {
        "result table": "yes"
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ReviewBundleNormalizationError) as exc_info:
        normalize_review_bundle(tmp_path)

    assert exc_info.value.reason_code == "REVIEW_BUNDLE_RECEIPT_FIELD_INVALID"
