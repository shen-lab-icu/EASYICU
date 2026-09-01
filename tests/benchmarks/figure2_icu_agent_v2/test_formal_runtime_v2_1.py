from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchmarks.figure2_icu_agent_v2.formal_provider_gate as formal_provider_gate
import benchmarks.figure2_icu_agent_v2.review_bundle_normalizer as review_bundle_normalizer
from benchmarks.figure2_icu_agent_v2.design_v2_1 import DesignContractError
from benchmarks.figure2_icu_agent_v2.formal_provider_gate import (
    FormalCallCoordinate,
    complete_formal_provider_call,
)
from benchmarks.figure2_icu_agent_v2.review_bundle_normalizer import (
    CANONICAL_FILES,
    ReviewBundleNormalizationError,
    normalize_review_bundle,
)
from easyicu.research_agent.providers.protocol import LLMMessage
from easyicu.research_agent.providers.client_trust import ProviderConfigurationError


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
            "mandatory_artifact_presence": {"01_plan.json": True},
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

    assert exc_info.value.reason_code == "FORMAL_PROVIDER_CALL_NOT_AUTHORIZED"
    assert client.called is False


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
        "mandatory_artifact_presence",
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


def test_normalizer_rejects_noncanonical_file_set(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "debug.log").write_text("hidden fingerprint", encoding="utf-8")

    with pytest.raises(ReviewBundleNormalizationError) as exc_info:
        normalize_review_bundle(tmp_path)

    assert exc_info.value.reason_code == "REVIEW_BUNDLE_FILE_SET_INVALID"
