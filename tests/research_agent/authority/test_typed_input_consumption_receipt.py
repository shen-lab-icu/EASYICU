"""Typed-input consumption receipts bind actual bytes to one consumer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.typed_input_receipt import (
    TYPED_INPUT_CONSUMPTION_RECEIPT_SCHEMA,
    TypedInputConsumptionReceipt,
    TypedInputReceiptError,
    load_verified_typed_input_table,
    seal_typed_input_consumption,
    typed_input_receipt_sha256,
    verify_step_typed_input_receipts,
    verify_typed_input_consumption_receipt,
)
from easyicu.research_agent.authority.run_input import canonical_sha256

INPUT_A = "artifact:reference_table"
INPUT_B = "artifact:comparison_table"
STEP_ID = "consumer_step"
CODE_SHA = "c" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_sha256(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.astype("string"):
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _binding(
    *,
    run_root: Path,
    input_key: str,
    evidence_id: str,
    artifact: Path,
    frame: pd.DataFrame,
) -> dict[str, object]:
    digest = _sha256(artifact)
    product = input_key.split(":", 1)[1]
    identity_row = {
        "input_key": input_key,
        "declared_kind": "artifact",
        "product": product,
        "evidence_id": evidence_id,
        "sha256": digest,
        "produced_by_step": "producer_step",
    }
    return {
        "evidence_id": evidence_id,
        "declared_kind": "artifact",
        "product": product,
        "evidence_kind": "table",
        "relative_path": artifact.relative_to(run_root).as_posix(),
        "absolute_path": str(artifact),
        "sha256": digest,
        "produced_by_step": "producer_step",
        "identity_row": identity_row,
        "product_contract": {
            "schema_version": "easyicu.host_typed_product.v2",
            "identity_row": identity_row,
            "tabular_format": "parquet",
            "column_count": len(frame.columns),
            "columns": list(frame.columns),
            "row_identity_column": "record_id",
            "row_count": len(frame),
            "row_identity_sha256": _identity_sha256(frame["record_id"]),
        },
    }


def _resolved_manifest(
    tmp_path: Path,
    *,
    two_inputs: bool = False,
) -> tuple[Path, str, Path, Path | None]:
    run_root = tmp_path / "run"
    evidence = run_root / "evidence"
    resolved = run_root / "resolved_inputs"
    evidence.mkdir(parents=True)
    resolved.mkdir()

    frame_a = pd.DataFrame(
        {"record_id": [101, 102, 103], "measurement": [1.0, 2.0, 3.0]}
    )
    artifact_a = evidence / "ev_a__shared.parquet"
    frame_a.to_parquet(artifact_a, index=False)
    bindings = {
        INPUT_A: _binding(
            run_root=run_root,
            input_key=INPUT_A,
            evidence_id="ev_a",
            artifact=artifact_a,
            frame=frame_a,
        )
    }
    declared = [INPUT_A]
    artifact_b: Path | None = None
    if two_inputs:
        frame_b = pd.DataFrame({"record_id": [201, 202], "measurement": [10.0, 20.0]})
        other_dir = run_root / "other_evidence"
        other_dir.mkdir()
        artifact_b = other_dir / "ev_b__shared.parquet"
        frame_b.to_parquet(artifact_b, index=False)
        bindings[INPUT_B] = _binding(
            run_root=run_root,
            input_key=INPUT_B,
            evidence_id="ev_b",
            artifact=artifact_b,
            frame=frame_b,
        )
        declared.append(INPUT_B)

    manifest = resolved / f"{STEP_ID}.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "2.1",
                "step_id": STEP_ID,
                "planner_declared_inputs": declared,
                "inputs": bindings,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, _sha256(manifest), artifact_a, artifact_b


def _load(manifest: Path, manifest_sha: str, *, input_key: str = INPUT_A):
    return load_verified_typed_input_table(
        resolved_inputs_path=manifest,
        expected_resolved_inputs_sha256=manifest_sha,
        run_root=manifest.parents[1],
        input_key=input_key,
        consumer_step_id=STEP_ID,
        consumer_code_sha256=CODE_SHA,
    )


def _verify(receipt, manifest: Path, manifest_sha: str, *, input_key=INPUT_A):
    return verify_typed_input_consumption_receipt(
        receipt,
        resolved_inputs_path=manifest,
        expected_resolved_inputs_sha256=manifest_sha,
        run_root=manifest.parents[1],
        input_key=input_key,
        consumer_step_id=STEP_ID,
        consumer_code_sha256=CODE_SHA,
    )


def _bindings(manifest: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return {str(key): dict(value) for key, value in payload["inputs"].items()}


def _declared(manifest: Path) -> list[str]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return list(payload["planner_declared_inputs"])


def _receipt(manifest: Path, manifest_sha: str, *, input_key: str = INPUT_A):
    loaded = _load(manifest, manifest_sha, input_key=input_key)
    return seal_typed_input_consumption(loaded, consumed_frame=loaded.frame)


def _step_verify(
    manifest: Path,
    manifest_sha: str,
    receipts,
    *,
    declared: list[str] | None = None,
    bindings: dict[str, dict[str, object]] | None = None,
    step_id: str = STEP_ID,
    code_sha: str = CODE_SHA,
    row_identity_not_applicable: list[str] | None = None,
):
    return verify_step_typed_input_receipts(
        planner_declared_inputs=(
            declared if declared is not None else _declared(manifest)
        ),
        resolved_input_bindings=(
            bindings if bindings is not None else _bindings(manifest)
        ),
        resolved_inputs_sha256=manifest_sha,
        consumer_step_id=step_id,
        consumer_code_sha256=code_sha,
        receipts=receipts,
        row_identity_not_applicable=row_identity_not_applicable or [],
    )


def _issue_codes(result) -> set[str]:
    return {
        str((finding.detail or {}).get("issue_code") or "")
        for finding in result.findings
    }


def _not_applicable_receipt(
    receipt: TypedInputConsumptionReceipt,
    *,
    reason: str = "dictionary lookup input has no row alignment semantics",
) -> dict[str, object]:
    payload = receipt.model_dump(mode="json")
    payload["row_identity"] = {"not_applicable": True, "reason": reason}
    payload["receipt_sha256"] = typed_input_receipt_sha256(payload)
    return payload


def test_correct_artifact_seals_and_reverifies(tmp_path: Path) -> None:
    manifest, manifest_sha, artifact, _ = _resolved_manifest(tmp_path)
    loaded = _load(manifest, manifest_sha)

    receipt = seal_typed_input_consumption(
        loaded,
        consumed_frame=loaded.frame,
    )
    verified = _verify(receipt.model_dump(mode="json"), manifest, manifest_sha)

    assert verified == receipt
    assert receipt.input_key == INPUT_A
    assert receipt.evidence_id == "ev_a"
    assert receipt.artifact_sha256 == _sha256(artifact)
    assert receipt.opened_file_sha256 == receipt.artifact_sha256
    assert receipt.resolved_inputs_sha256 == manifest_sha
    assert receipt.row_identity.row_count == 3
    assert receipt.row_identity.unique is True


def test_same_filename_with_changed_bytes_fails_closed(tmp_path: Path) -> None:
    manifest, manifest_sha, artifact, _ = _resolved_manifest(tmp_path)
    loaded = _load(manifest, manifest_sha)
    receipt = seal_typed_input_consumption(loaded, consumed_frame=loaded.frame)
    pd.DataFrame(
        {"record_id": [101, 102, 103], "measurement": [9.0, 9.0, 9.0]}
    ).to_parquet(artifact, index=False)

    with pytest.raises(TypedInputReceiptError, match="artifact SHA-256"):
        _verify(receipt, manifest, manifest_sha)


def test_symlink_selected_as_artifact_fails_closed(tmp_path: Path) -> None:
    manifest, _, artifact, _ = _resolved_manifest(tmp_path)
    replacement = artifact.with_name("replacement.parquet")
    artifact.replace(replacement)
    artifact.symlink_to(replacement.name)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["inputs"][INPUT_A]["sha256"] = _sha256(replacement)
    payload["inputs"][INPUT_A]["identity_row"]["sha256"] = _sha256(replacement)
    payload["inputs"][INPUT_A]["product_contract"]["identity_row"]["sha256"] = _sha256(
        replacement
    )
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TypedInputReceiptError, match="regular table"):
        _load(manifest, _sha256(manifest))


def test_correct_evidence_sha_cannot_authorize_opening_another_file(
    tmp_path: Path,
) -> None:
    manifest, _, artifact_a, artifact_b = _resolved_manifest(
        tmp_path,
        two_inputs=True,
    )
    assert artifact_b is not None
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    binding = payload["inputs"][INPUT_A]
    binding["absolute_path"] = str(artifact_b)
    binding["relative_path"] = artifact_b.relative_to(manifest.parents[1]).as_posix()
    # Keep evidence identity and SHA bound to artifact A while selecting B.
    assert binding["sha256"] == _sha256(artifact_a)
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TypedInputReceiptError, match="artifact SHA-256"):
        _load(manifest, _sha256(manifest))


def test_receipt_row_count_cannot_be_rewritten(tmp_path: Path) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path)
    loaded = _load(manifest, manifest_sha)
    receipt = seal_typed_input_consumption(loaded, consumed_frame=loaded.frame)
    payload = receipt.model_dump(mode="json")
    payload["row_identity"]["row_count"] = 99
    payload["receipt_sha256"] = typed_input_receipt_sha256(payload)

    with pytest.raises(TypedInputReceiptError, match="row count"):
        _verify(payload, manifest, manifest_sha)


def test_two_typed_input_identities_cannot_be_interchanged(tmp_path: Path) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path, two_inputs=True)
    loaded = _load(manifest, manifest_sha, input_key=INPUT_A)
    receipt = seal_typed_input_consumption(loaded, consumed_frame=loaded.frame)

    with pytest.raises(TypedInputReceiptError, match="input_key"):
        _verify(receipt, manifest, manifest_sha, input_key=INPUT_B)


@pytest.mark.parametrize(
    ("step_id", "code_sha", "message"),
    [
        ("other_step", CODE_SHA, "consumer step"),
        (STEP_ID, "d" * 64, "consumer code"),
    ],
)
def test_receipt_cannot_replay_to_another_step_or_code(
    tmp_path: Path,
    step_id: str,
    code_sha: str,
    message: str,
) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path)
    loaded = _load(manifest, manifest_sha)
    receipt = seal_typed_input_consumption(loaded, consumed_frame=loaded.frame)

    with pytest.raises(TypedInputReceiptError, match=message):
        verify_typed_input_consumption_receipt(
            receipt,
            resolved_inputs_path=manifest,
            expected_resolved_inputs_sha256=manifest_sha,
            run_root=manifest.parents[1],
            input_key=INPUT_A,
            consumer_step_id=step_id,
            consumer_code_sha256=code_sha,
        )


def test_duplicate_row_identity_fails_closed(tmp_path: Path) -> None:
    manifest, _, artifact, _ = _resolved_manifest(tmp_path)
    duplicate = pd.DataFrame(
        {"record_id": [101, 101, 103], "measurement": [1.0, 2.0, 3.0]}
    )
    duplicate.to_parquet(artifact, index=False)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["inputs"][INPUT_A] = _binding(
        run_root=manifest.parents[1],
        input_key=INPUT_A,
        evidence_id="ev_a",
        artifact=artifact,
        frame=duplicate,
    )
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TypedInputReceiptError, match="duplicate row identity"):
        _load(manifest, _sha256(manifest))


def test_only_printing_a_path_cannot_construct_a_receipt(tmp_path: Path) -> None:
    manifest, _, artifact, _ = _resolved_manifest(tmp_path)
    path_only = {
        "input_key": INPUT_A,
        "absolute_path": str(artifact),
        "loaded": True,
    }

    with pytest.raises(Exception):
        TypedInputConsumptionReceipt.model_validate(path_only)
    with pytest.raises(TypedInputReceiptError, match="verified typed-input load"):
        seal_typed_input_consumption(path_only, consumed_frame=pd.DataFrame())


def test_loaded_input_cannot_authorize_another_dataframe(tmp_path: Path) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path)
    loaded = _load(manifest, manifest_sha)
    unrelated = loaded.frame.copy()

    with pytest.raises(TypedInputReceiptError, match="same loaded DataFrame"):
        seal_typed_input_consumption(loaded, consumed_frame=unrelated)


def test_loaded_frame_mutation_invalidates_consumption(tmp_path: Path) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path)
    loaded = _load(manifest, manifest_sha)
    loaded.frame.loc[0, "measurement"] = 999.0

    with pytest.raises(TypedInputReceiptError, match="changed after verified load"):
        seal_typed_input_consumption(loaded, consumed_frame=loaded.frame)


def test_missing_or_extra_receipt_fields_fail_closed(tmp_path: Path) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path)
    loaded = _load(manifest, manifest_sha)
    payload = seal_typed_input_consumption(
        loaded,
        consumed_frame=loaded.frame,
    ).model_dump(mode="json")

    missing = dict(payload)
    missing.pop("evidence_id")
    with pytest.raises(TypedInputReceiptError, match="invalid receipt schema"):
        _verify(missing, manifest, manifest_sha)

    extra = dict(payload)
    extra["trusted"] = True
    with pytest.raises(TypedInputReceiptError, match="invalid receipt schema"):
        _verify(extra, manifest, manifest_sha)


def test_resolved_manifest_sha_and_binding_identity_are_enforced(
    tmp_path: Path,
) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path)
    with pytest.raises(TypedInputReceiptError, match="resolved-input manifest"):
        _load(manifest, "f" * 64)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["inputs"][INPUT_A]["identity_row"]["input_key"] = INPUT_B
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(TypedInputReceiptError, match="identity row"):
        _load(manifest, _sha256(manifest))


def test_product_contract_row_identity_must_match_opened_table(
    tmp_path: Path,
) -> None:
    manifest, _, _, _ = _resolved_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    contract = payload["inputs"][INPUT_A]["product_contract"]
    contract["row_count"] = 4
    contract["row_identity_sha256"] = "e" * 64
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TypedInputReceiptError, match="product contract"):
        _load(manifest, _sha256(manifest))


def test_step_receipt_completeness_accepts_exact_declared_receipt_set(
    tmp_path: Path,
) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path, two_inputs=True)
    receipt_a = _receipt(manifest, manifest_sha, input_key=INPUT_A)
    receipt_b = _receipt(manifest, manifest_sha, input_key=INPUT_B)

    result = _step_verify(manifest, manifest_sha, [receipt_a, receipt_b])

    assert result.findings == ()
    assert set(result.verified_inputs) == {INPUT_A, INPUT_B}
    assert result.verified_inputs[INPUT_A] == receipt_a
    with pytest.raises(TypeError):
        result.verified_inputs[INPUT_A] = receipt_b  # type: ignore[index]


def test_step_receipt_completeness_reports_missing_receipt(tmp_path: Path) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path, two_inputs=True)
    receipt_a = _receipt(manifest, manifest_sha, input_key=INPUT_A)

    result = _step_verify(manifest, manifest_sha, [receipt_a])

    assert _issue_codes(result) == {"missing_receipt"}
    assert result.verified_inputs == {}


def test_step_receipt_completeness_reports_duplicate_receipt(tmp_path: Path) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path)
    receipt = _receipt(manifest, manifest_sha)

    result = _step_verify(manifest, manifest_sha, [receipt, receipt])

    assert _issue_codes(result) == {"duplicate_receipt"}
    assert result.verified_inputs == {}


def test_step_receipt_completeness_reports_extra_receipt(tmp_path: Path) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path, two_inputs=True)
    receipt_a = _receipt(manifest, manifest_sha, input_key=INPUT_A)
    receipt_b = _receipt(manifest, manifest_sha, input_key=INPUT_B)

    result = _step_verify(
        manifest,
        manifest_sha,
        [receipt_a, receipt_b],
        declared=[INPUT_A],
        bindings={INPUT_A: _bindings(manifest)[INPUT_A]},
    )

    assert _issue_codes(result) == {"extra_receipt"}
    assert result.verified_inputs == {}


def test_step_receipt_completeness_rejects_identity_interchange(
    tmp_path: Path,
) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path, two_inputs=True)
    receipt_a = _receipt(manifest, manifest_sha, input_key=INPUT_A)
    payload = receipt_a.model_dump(mode="json")
    payload["input_key"] = INPUT_B
    payload["receipt_sha256"] = typed_input_receipt_sha256(payload)

    result = _step_verify(manifest, manifest_sha, [payload])

    assert _issue_codes(result) == {"missing_receipt", "receipt_binding_mismatch"}
    assert result.verified_inputs == {}


@pytest.mark.parametrize(
    ("step_id", "code_sha", "expected_field"),
    [
        ("old_step", CODE_SHA, "consumer_step_id"),
        (STEP_ID, "d" * 64, "consumer_code_sha256"),
    ],
)
def test_step_receipt_completeness_rejects_old_step_or_code_replay(
    tmp_path: Path,
    step_id: str,
    code_sha: str,
    expected_field: str,
) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path)
    receipt = _receipt(manifest, manifest_sha)

    result = _step_verify(
        manifest,
        manifest_sha,
        [receipt],
        step_id=step_id,
        code_sha=code_sha,
    )

    assert _issue_codes(result) == {"receipt_binding_mismatch"}
    assert (result.findings[0].detail or {})["fields"] == [expected_field]
    assert result.verified_inputs == {}


def test_step_receipt_completeness_rejects_same_file_for_two_logical_inputs(
    tmp_path: Path,
) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path, two_inputs=True)
    bindings = _bindings(manifest)
    bindings[INPUT_B] = dict(bindings[INPUT_A])
    bindings[INPUT_B]["declared_kind"] = "artifact"
    bindings[INPUT_B]["product"] = "comparison_table"
    bindings[INPUT_B]["identity_row"] = {
        **dict(bindings[INPUT_A]["identity_row"]),
        "input_key": INPUT_B,
        "product": "comparison_table",
    }
    bindings[INPUT_B]["product_contract"] = {
        **dict(bindings[INPUT_A]["product_contract"]),
        "identity_row": bindings[INPUT_B]["identity_row"],
    }
    receipt_a = _receipt(manifest, manifest_sha, input_key=INPUT_A)
    payload_b = receipt_a.model_dump(mode="json")
    payload_b["input_key"] = INPUT_B
    payload_b["resolved_input_binding_sha256"] = canonical_sha256(bindings[INPUT_B])
    payload_b["receipt_sha256"] = typed_input_receipt_sha256(payload_b)

    result = _step_verify(
        manifest, manifest_sha, [receipt_a, payload_b], bindings=bindings
    )

    assert "shared_file_identity" in _issue_codes(result)
    assert result.verified_inputs == {}


def test_step_receipt_completeness_rejects_missing_binding(tmp_path: Path) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path, two_inputs=True)
    receipt_a = _receipt(manifest, manifest_sha, input_key=INPUT_A)
    bindings = _bindings(manifest)
    bindings.pop(INPUT_B)

    result = _step_verify(manifest, manifest_sha, [receipt_a], bindings=bindings)

    assert _issue_codes(result) == {"missing_resolved_binding", "missing_receipt"}
    assert result.verified_inputs == {}


def test_step_receipt_completeness_rejects_extra_binding(tmp_path: Path) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path)
    receipt = _receipt(manifest, manifest_sha)
    bindings = _bindings(manifest)
    bindings[INPUT_B] = dict(bindings[INPUT_A])

    result = _step_verify(manifest, manifest_sha, [receipt], bindings=bindings)

    assert _issue_codes(result) == {"extra_resolved_binding"}
    assert result.verified_inputs == {}


def test_step_receipt_completeness_rejects_missing_row_identity_contract(
    tmp_path: Path,
) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path)
    receipt = _receipt(manifest, manifest_sha)
    bindings = _bindings(manifest)
    bindings[INPUT_A]["product_contract"] = {
        "schema_version": "easyicu.host_typed_product.v2",
        "columns": ["record_id", "measurement"],
        "column_count": 2,
    }
    payload = receipt.model_dump(mode="json")
    payload["resolved_input_binding_sha256"] = canonical_sha256(bindings[INPUT_A])
    payload["receipt_sha256"] = typed_input_receipt_sha256(payload)

    result = _step_verify(manifest, manifest_sha, [payload], bindings=bindings)

    assert _issue_codes(result) == {"missing_row_identity_contract"}
    assert result.verified_inputs == {}


def test_step_receipt_completeness_accepts_host_declared_row_identity_not_applicable(
    tmp_path: Path,
) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path)
    receipt = _receipt(manifest, manifest_sha)
    bindings = _bindings(manifest)
    bindings[INPUT_A]["product_contract"] = {
        "schema_version": "easyicu.host_typed_product.v2",
        "columns": ["record_id", "measurement"],
        "column_count": 2,
    }
    payload = _not_applicable_receipt(receipt)
    payload["resolved_input_binding_sha256"] = canonical_sha256(bindings[INPUT_A])
    payload["receipt_sha256"] = typed_input_receipt_sha256(payload)

    result = _step_verify(
        manifest,
        manifest_sha,
        [payload],
        bindings=bindings,
        row_identity_not_applicable=[INPUT_A],
    )

    assert result.findings == ()
    assert set(result.verified_inputs) == {INPUT_A}


def test_step_receipt_completeness_requires_explicit_not_applicable_receipt_marker(
    tmp_path: Path,
) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path)
    receipt = _receipt(manifest, manifest_sha)
    bindings = _bindings(manifest)
    bindings[INPUT_A]["product_contract"] = {
        "schema_version": "easyicu.host_typed_product.v2",
        "columns": ["record_id", "measurement"],
        "column_count": 2,
    }
    payload = receipt.model_dump(mode="json")
    payload["resolved_input_binding_sha256"] = canonical_sha256(bindings[INPUT_A])
    payload["receipt_sha256"] = typed_input_receipt_sha256(payload)

    result = _step_verify(
        manifest,
        manifest_sha,
        [payload],
        bindings=bindings,
        row_identity_not_applicable=[INPUT_A],
    )

    assert _issue_codes(result) == {"row_identity_not_applicable_receipt_missing"}
    assert result.verified_inputs == {}


def test_step_receipt_completeness_rejects_unknown_not_applicable_input(
    tmp_path: Path,
) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path)
    receipt = _receipt(manifest, manifest_sha)

    result = _step_verify(
        manifest,
        manifest_sha,
        [receipt],
        row_identity_not_applicable=[INPUT_B],
    )

    assert _issue_codes(result) == {"row_identity_not_applicable_for_unknown_input"}
    assert result.verified_inputs == {}


def test_step_receipt_completeness_rejects_row_identity_not_applicable_when_contract_exists(
    tmp_path: Path,
) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path)
    receipt = _receipt(manifest, manifest_sha)

    result = _step_verify(
        manifest,
        manifest_sha,
        [receipt],
        row_identity_not_applicable=[INPUT_A],
    )

    assert _issue_codes(result) == {
        "row_identity_not_applicable_conflicts_with_contract"
    }
    assert result.verified_inputs == {}


def test_step_receipt_completeness_rejects_invalid_receipt_schema(
    tmp_path: Path,
) -> None:
    manifest, manifest_sha, _, _ = _resolved_manifest(tmp_path)

    result = _step_verify(
        manifest,
        manifest_sha,
        [
            {
                "schema_version": TYPED_INPUT_CONSUMPTION_RECEIPT_SCHEMA,
                "input_key": INPUT_A,
            }
        ],
    )

    assert _issue_codes(result) == {"invalid_receipt", "missing_receipt"}
    assert result.verified_inputs == {}
