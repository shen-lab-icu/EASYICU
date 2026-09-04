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
from benchmarks.figure2_icu_agent_v2.formal_release_identity import (
    FormalReleaseIdentityError,
    REGISTERED_SOURCE_PATHS,
    registered_source_digests,
    required_registration_fields,
    validate_registered_source_identity,
)
from benchmarks.figure2_icu_agent_v2.formal_provider_gate import (
    FormalCallCoordinate,
    FormalProviderBudgetMissingError,
    complete_formal_provider_call,
)
from benchmarks.figure2_icu_agent_v2.formal_scheduler import (
    expected_site_assignment,
    expected_site_assignment_sha256,
    signed_output_root,
    signed_site_assignment,
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
    ReviewBlindingContext,
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
_BLINDING_CONTEXT = ReviewBlindingContext(
    host_markers=("srv-01", "MacBook"),
    output_roots=("/Volumes/ext/easyicu_data/fig2_core/server", r"D:\figure2\laptop"),
)


def _normalize(source_dir: Path):
    return normalize_review_bundle(
        source_dir,
        blinding_context=_BLINDING_CONTEXT,
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
    wrong_site_for_requested_task: bool = False,
    wrong_assignment_digest: bool = False,
    duplicate_output_roots: bool = False,
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

    task_ids = tuple(f"qualification12_a_{index:02d}" for index in range(1, 13))
    assignment = expected_site_assignment("qualification12", task_ids=task_ids)
    site_by_task = {item["task_id"]: item["execution_site"] for item in assignment}
    requested_task = task_ids[0]
    requested_site = site_by_task[requested_task]
    if wrong_site_for_requested_task:
        requested_site = "laptop" if requested_site == "server" else "server"
    coordinate = {
        "scope": "qualification12",
        "task_id": requested_task,
        "arm": arm,
        "execution_site": requested_site,
        "call_id": call_id,
    }
    coordinates = []
    for task_id in task_ids:
        for task_arm in ("easyicu_full", "generic_code_agent"):
            item = {
                "scope": "qualification12",
                "task_id": task_id,
                "arm": task_arm,
                "execution_site": site_by_task[task_id],
                "call_id": f"{task_arm}_{task_id}",
            }
            if task_id == requested_task and task_arm == arm:
                item = coordinate
            coordinates.append(item)
    site_assignment_sha256 = expected_site_assignment_sha256(
        "qualification12",
        task_ids=task_ids,
    )
    if wrong_assignment_digest:
        site_assignment_sha256 = "0" * 64
    binding = {
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "site_assignment_sha256": site_assignment_sha256,
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
            for field, path in REGISTERED_SOURCE_PATHS.items()
        },
        "design_commit": binding["design_commit"],
        "annotated_tag": binding["annotated_tag"],
        "registrant_identity": "test-registrant",
        "trusted_authority_signer_identity": "external-custodian-1",
        "trusted_authority_ed25519_public_key_base64": public_key_text,
        "amendment_policy_acknowledged": True,
    }
    output_root_by_site = {
        "server": str((tmp_path / "server-output").resolve()),
        "laptop": str((tmp_path / "laptop-output").resolve()),
    }
    if duplicate_output_roots:
        output_root_by_site["laptop"] = output_root_by_site["server"]
    declaration = {
        "schema_version": "easyicu.figure2_atomic_declaration/3",
        "signer_id": "external-custodian-1",
        "scope": "qualification12",
        **binding,
        "output_root_by_site": output_root_by_site,
        "receipt_sha256": {
            receipt_id: hashlib.sha256(_canonical_json(payload)).hexdigest()
            for receipt_id, payload in receipt_payloads.items()
        },
        "authorized_call_coordinates": coordinates,
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
        execution_site="server",
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
    declaration = envelope["atomic_declaration"]
    assert isinstance(declaration, dict)
    assert receipt["output_root"] == declaration["output_root_by_site"][
        coordinate["execution_site"]
    ]
    assert len(signed_site_assignment(envelope, scope="qualification12")) == 12
    assert signed_output_root(
        envelope,
        execution_site=coordinate["execution_site"],
    ) == receipt["output_root"]
    with pytest.raises(DesignContractError) as exc_info:
        formal_authority.authorize_formal_provider_call(
            {
                "receipts": envelope,
                "call_coordinate": {**coordinate, "call_id": "generic_0002"},
            }
        )
    assert exc_info.value.reason_code == "FORMAL_AUTHORITY_COORDINATE_NOT_DECLARED"
    with pytest.raises(DesignContractError) as site_exc_info:
        formal_authority.authorize_formal_provider_call(
            {
                "receipts": envelope,
                "call_coordinate": {**coordinate, "execution_site": "laptop"},
            }
        )
    assert site_exc_info.value.reason_code == "FORMAL_AUTHORITY_COORDINATE_NOT_DECLARED"


def test_formal_authority_rejects_signed_coordinate_on_wrong_frozen_site(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope, coordinate = _signed_qualification_authority(
        tmp_path,
        monkeypatch,
        wrong_site_for_requested_task=True,
    )

    with pytest.raises(DesignContractError) as exc_info:
        formal_authority.authorize_formal_provider_call(
            {"receipts": envelope, "call_coordinate": coordinate}
        )

    assert exc_info.value.reason_code == "FORMAL_AUTHORITY_SITE_ASSIGNMENT_INVALID"


def test_formal_authority_rejects_signed_wrong_assignment_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope, coordinate = _signed_qualification_authority(
        tmp_path,
        monkeypatch,
        wrong_assignment_digest=True,
    )

    with pytest.raises(DesignContractError) as exc_info:
        formal_authority.authorize_formal_provider_call(
            {"receipts": envelope, "call_coordinate": coordinate}
        )

    assert exc_info.value.reason_code == "FORMAL_AUTHORITY_SITE_ASSIGNMENT_INVALID"


def test_formal_authority_rejects_invalid_signed_output_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope, coordinate = _signed_qualification_authority(
        tmp_path,
        monkeypatch,
        duplicate_output_roots=True,
    )

    with pytest.raises(DesignContractError) as exc_info:
        formal_authority.authorize_formal_provider_call(
            {"receipts": envelope, "call_coordinate": coordinate}
        )

    assert exc_info.value.reason_code == "FORMAL_AUTHORITY_OUTPUT_ROOT_INVALID"


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
        execution_site=coordinate["execution_site"],
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
        execution_site="server",
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


def test_release_identity_owns_complete_registration_field_set() -> None:
    preregistration = json.loads(
        formal_authority.PREREGISTRATION_PLAN_PATH.read_text(encoding="utf-8")
    )

    assert tuple(preregistration["required_receipt_fields"]) == (
        required_registration_fields()
    )
    assert all(path.is_file() for path in REGISTERED_SOURCE_PATHS.values())
    assert {
        "review_bundle_writer_sha256",
        "formal_trajectory_lifecycle_sha256",
        "formal_release_identity_sha256",
    } <= set(REGISTERED_SOURCE_PATHS)


def test_release_identity_rejects_tampered_registered_source_digest() -> None:
    registration = registered_source_digests()
    registration["formal_release_identity_sha256"] = "0" * 64

    with pytest.raises(FormalReleaseIdentityError) as exc_info:
        validate_registered_source_identity(registration)

    assert exc_info.value.reason_code == (
        "FORMAL_AUTHORITY_REGISTERED_SOURCE_MISMATCH"
    )


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
                execution_site="server",
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
                execution_site="server",
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
            execution_site="server",
            call_id="call_001",
        )
    with pytest.raises(ValueError, match="unsupported execution site"):
        FormalCallCoordinate(
            scope="qualification12",
            task_id="qualification12_a_01",
            arm="generic_code_agent",
            call_id="call_001",
            execution_site="desktop",
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
        execution_site="server",
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
                execution_site="server",
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
            execution_site="server",
            call_id="call_001",
        )


def test_normalizer_preserves_science_and_hides_arm_resource_fingerprints(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path)

    result = _normalize(tmp_path)

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


def test_normalizer_requires_explicit_runtime_blinding_context(tmp_path: Path) -> None:
    _write_bundle(tmp_path)

    with pytest.raises(TypeError, match="blinding_context"):
        normalize_review_bundle(tmp_path)  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="host markers"):
        ReviewBlindingContext(
            host_markers=(" srv-01", "MacBook"),
            output_roots=_BLINDING_CONTEXT.output_roots,
        )
    with pytest.raises(ValueError, match="absolute roots"):
        ReviewBlindingContext(
            host_markers=_BLINDING_CONTEXT.host_markers,
            output_roots=("/formal/server ", "/formal/laptop"),
        )


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
        _normalize(tmp_path)

    assert exc_info.value.reason_code == "REVIEW_BUNDLE_UNSAFE_MARKER"


def test_normalizer_redacts_container_and_repository_paths(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    manifest = {
        "paths": [
            "/workspace/run/out.csv",
            "/home/agent/plan.py",
            "benchmarks/figure2_icu_agent_v2/heldout27_taskbank_v1.jsonl",
            "/Users/example/out.csv.",
            "/Volumes/ext/easyicu_data/fig2_core/server/task/result.csv",
            r"D:\figure2\laptop\task\result.csv",
        ]
    }
    (tmp_path / "05_evidence_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    result = _normalize(tmp_path)
    normalized = json.loads(result.files["05_evidence_manifest.json"])

    assert normalized["paths"] == [
        "<redacted-path>",
        "<redacted-path>",
        "<redacted-path>",
        "<redacted-path>.",
        "<redacted-path>",
        "<redacted-path>",
    ]


def test_normalizer_preserves_clinical_units_and_redacts_full_runtime_path(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path)
    report = (
        "Dose 5 mg /kg/day and rate 90 /min/m2. "
        "Wrote /srv/fig2core/server/icu27_t09/easyicu_full/03_results.json."
    )
    (tmp_path / "06_report.md").write_text(report, encoding="utf-8")

    result = _normalize(tmp_path)

    normalized = result.files["06_report.md"].decode("utf-8")
    assert "5 mg /kg/day" in normalized
    assert "90 /min/m2" in normalized
    assert normalized.endswith("Wrote <redacted-path>.")
    assert "03_results.json" not in normalized


@pytest.mark.parametrize(
    "site_marker",
    (
        "Executed on srv-01 with the registered limits.",
        "Executed on the MacBook overnight.",
        "The run_host was retained.",
        "laptop_runtime_seconds was 20.",
        "Both servers were idle.",
        "Both laptops were idle.",
    ),
)
def test_normalizer_rejects_registered_site_and_host_markers(
    tmp_path: Path,
    site_marker: str,
) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "06_report.md").write_text(site_marker, encoding="utf-8")

    with pytest.raises(ReviewBundleNormalizationError) as exc_info:
        _normalize(tmp_path)

    assert exc_info.value.reason_code in {
        "REVIEW_BUNDLE_EXECUTION_SITE_MARKER",
        "REVIEW_BUNDLE_RESOURCE_FINGERPRINT",
    }


@pytest.mark.parametrize(
    "fingerprint",
    [
        "analysis completed after 37 model turns",
        "the workflow made 52 tool calls",
        "provider_calls=4",
        "provider tokens: 1200",
        "per-tool latency was retained",
        "execution_site=server",
        "host fingerprint was retained",
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
        _normalize(tmp_path)

    assert exc_info.value.reason_code == "REVIEW_BUNDLE_RESOURCE_FINGERPRINT"


def test_normalizer_detects_numeric_content_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_bundle(tmp_path)
    original = review_bundle_normalizer._normalize_value

    def corrupt_numeric_content(value, *, file_name, location, blinding_context):
        normalized, findings = original(
            value,
            file_name=file_name,
            location=location,
            blinding_context=blinding_context,
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
        _normalize(tmp_path)

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
        _normalize(tmp_path)

    assert exc_info.value.reason_code == "REVIEW_BUNDLE_NONFINITE_JSON"


def test_normalizer_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "03_results.json").write_text(
        '{"estimate": 1.25, "estimate": 9.99}',
        encoding="utf-8",
    )

    with pytest.raises(ReviewBundleNormalizationError) as exc_info:
        _normalize(tmp_path)

    assert exc_info.value.reason_code == "REVIEW_BUNDLE_JSON_KEY_DUPLICATE"


def test_normalizer_rejects_unknown_internal_marker(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "06_report.md").write_text(
        "Terminal reason: FORMAL_PROVIDER_CALL_NOT_AUTHORIZED\n",
        encoding="utf-8",
    )

    with pytest.raises(ReviewBundleNormalizationError) as exc_info:
        _normalize(tmp_path)

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

    result = _normalize(tmp_path)

    assert json.loads(result.files["03_results.json"])["field"] == (
        "ICU_FREE_DAYS_28"
    )


def test_normalizer_rejects_noncanonical_file_set(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "debug.log").write_text("hidden fingerprint", encoding="utf-8")

    with pytest.raises(ReviewBundleNormalizationError) as exc_info:
        _normalize(tmp_path)

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
        _normalize(tmp_path)

    assert exc_info.value.reason_code == "REVIEW_BUNDLE_RECEIPT_FIELD_INVALID"
