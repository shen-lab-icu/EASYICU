"""Host-owned typed-input SDK authority and adversarial tests."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from easyicu.research_agent.authority import typed_input_sdk as sdk
from easyicu.research_agent.authority.typed_input_receipt import (
    TypedInputReceiptError,
)
from easyicu.research_agent.authority.typed_input_sdk import (
    LoadedTypedInput,
    TypedInputSDKError,
    load_typed_input,
)

INPUT_KEY = "artifact:analysis_table"
STEP_ID = "model_step"
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


def _write_table(path: Path, frame: pd.DataFrame, *, format_name: str) -> None:
    if format_name == "parquet":
        frame.to_parquet(path, index=False)
    elif format_name == "csv":
        frame.to_csv(path, index=False)
    elif format_name == "tsv":
        frame.to_csv(path, index=False, sep="\t")
    else:  # pragma: no cover - test factory contract
        raise AssertionError(format_name)


def _case(
    tmp_path: Path,
    *,
    format_name: str = "parquet",
    frame: pd.DataFrame | None = None,
) -> tuple[Path, Path, Path, dict[str, object]]:
    run_root = tmp_path / "run"
    evidence_dir = run_root / "evidence"
    manifest_dir = run_root / "resolved_inputs"
    evidence_dir.mkdir(parents=True)
    manifest_dir.mkdir()
    source = (
        frame.copy(deep=True)
        if frame is not None
        else pd.DataFrame(
            {
                "record_id": [101, 102, 103],
                "measurement": [1.0, 2.0, 3.0],
            }
        )
    )
    suffix = {"parquet": ".parquet", "csv": ".csv", "tsv": ".tsv"}[format_name]
    artifact = evidence_dir / f"ev_analysis{suffix}"
    _write_table(artifact, source, format_name=format_name)
    artifact_sha = _sha256(artifact)
    identity_row = {
        "input_key": INPUT_KEY,
        "declared_kind": "artifact",
        "product": "analysis_table",
        "evidence_id": "ev_analysis",
        "sha256": artifact_sha,
        "produced_by_step": "cohort_step",
    }
    binding: dict[str, object] = {
        "evidence_id": "ev_analysis",
        "declared_kind": "artifact",
        "product": "analysis_table",
        "evidence_kind": "table",
        "relative_path": artifact.relative_to(run_root).as_posix(),
        "absolute_path": str(artifact),
        "sha256": artifact_sha,
        "produced_by_step": "cohort_step",
        "identity_row": identity_row,
        "product_contract": {
            "schema_version": "easyicu.host_typed_product.v2",
            "identity_row": identity_row,
            "tabular_format": format_name,
            "column_count": len(source.columns),
            "columns": list(source.columns),
            "row_identity_column": "record_id",
            "row_count": len(source),
            "row_identity_sha256": _identity_sha256(source["record_id"]),
        },
    }
    manifest_payload: dict[str, object] = {
        "schema_version": "2.1",
        "step_id": STEP_ID,
        "planner_declared_inputs": [INPUT_KEY],
        "inputs": {INPUT_KEY: binding},
    }
    manifest = manifest_dir / f"{STEP_ID}.json"
    _write_manifest(manifest, manifest_payload)
    return run_root, manifest, artifact, manifest_payload


def _write_manifest(path: Path, payload: dict[str, object]) -> str:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _sha256(path)


def _load(run_root: Path, manifest: Path) -> LoadedTypedInput:
    return load_typed_input(
        resolved_inputs_path=manifest,
        expected_resolved_inputs_sha256=_sha256(manifest),
        run_root=run_root,
        input_key=INPUT_KEY,
        consumer_step_id=STEP_ID,
        consumer_code_sha256=CODE_SHA,
    )


def _binding(payload: dict[str, object]) -> dict[str, object]:
    inputs = payload["inputs"]
    assert isinstance(inputs, dict)
    binding = inputs[INPUT_KEY]
    assert isinstance(binding, dict)
    return binding


def test_loads_parquet_as_one_immutable_payload_receipt_pair(tmp_path: Path) -> None:
    run_root, manifest, artifact, _ = _case(tmp_path)

    loaded = _load(run_root, manifest)

    assert isinstance(loaded.payload, pa.Table)
    assert loaded.payload.num_rows == 3
    assert loaded.input_key == INPUT_KEY
    assert loaded.receipt.artifact_sha256 == _sha256(artifact)
    assert loaded.receipt.resolved_inputs_sha256 == _sha256(manifest)
    assert loaded.receipt.consumer_step_id == STEP_ID
    assert loaded.receipt.consumer_code_sha256 == CODE_SHA


def test_loads_csv_with_the_same_explicit_identity_contract(tmp_path: Path) -> None:
    run_root, manifest, _, _ = _case(tmp_path, format_name="csv")

    loaded = _load(run_root, manifest)

    assert loaded.payload.column_names == ["record_id", "measurement"]
    assert loaded.to_pandas()["record_id"].tolist() == [101, 102, 103]
    assert loaded.receipt.row_identity.column == "record_id"


def test_api_accepts_no_caller_artifact_path_or_dataframe_declaration() -> None:
    parameters = set(inspect.signature(load_typed_input).parameters)

    assert "artifact_path" not in parameters
    assert "path" not in parameters
    assert "dataframe" not in parameters
    assert "frame" not in parameters
    assert "consumed_frame" not in parameters


def test_caller_cannot_construct_payload_receipt_pair_with_another_table(
    tmp_path: Path,
) -> None:
    run_root, manifest, _, _ = _case(tmp_path)
    loaded = _load(run_root, manifest)
    other = pa.Table.from_pydict({"record_id": [999], "measurement": [999.0]})

    with pytest.raises(TypedInputSDKError, match="only be constructed"):
        LoadedTypedInput(
            payload=other,
            receipt=loaded.receipt,
            _construction_token=object(),
        )


def test_mutating_a_pandas_copy_cannot_change_the_authority_payload(
    tmp_path: Path,
) -> None:
    run_root, manifest, _, _ = _case(tmp_path)
    loaded = _load(run_root, manifest)
    caller_frame = loaded.to_pandas()
    caller_frame.loc[:, "measurement"] = 999.0

    fresh = loaded.to_pandas()

    assert fresh["measurement"].tolist() == [1.0, 2.0, 3.0]
    assert loaded.receipt.loaded_frame_sha256 != sdk._frame_digest(caller_frame)


def test_arrow_transform_returns_a_new_table_without_rebinding_receipt(
    tmp_path: Path,
) -> None:
    run_root, manifest, _, _ = _case(tmp_path)
    loaded = _load(run_root, manifest)

    changed = loaded.payload.set_column(
        1,
        "measurement",
        pa.array([9.0, 9.0, 9.0]),
    )

    assert changed is not loaded.payload
    assert loaded.to_pandas()["measurement"].tolist() == [1.0, 2.0, 3.0]
    assert loaded.receipt.loaded_frame_sha256 != sdk._frame_digest(changed.to_pandas())


def test_loaded_typed_input_properties_are_immutable(tmp_path: Path) -> None:
    run_root, manifest, _, _ = _case(tmp_path)
    loaded = _load(run_root, manifest)

    with pytest.raises(AttributeError, match="immutable"):
        loaded.receipt = loaded.receipt  # type: ignore[misc]


def test_wrong_manifest_sha_fails_closed(tmp_path: Path) -> None:
    run_root, manifest, _, _ = _case(tmp_path)

    with pytest.raises(TypedInputReceiptError, match="manifest SHA-256"):
        load_typed_input(
            resolved_inputs_path=manifest,
            expected_resolved_inputs_sha256="0" * 64,
            run_root=run_root,
            input_key=INPUT_KEY,
            consumer_step_id=STEP_ID,
            consumer_code_sha256=CODE_SHA,
        )


def test_artifact_bytes_must_match_resolved_evidence_sha(tmp_path: Path) -> None:
    run_root, manifest, artifact, _ = _case(tmp_path)
    pd.DataFrame(
        {"record_id": [101, 102, 103], "measurement": [9.0, 9.0, 9.0]}
    ).to_parquet(artifact, index=False)

    with pytest.raises(TypedInputReceiptError, match="artifact SHA-256"):
        _load(run_root, manifest)


def test_evidence_identity_row_must_match_the_selected_binding(tmp_path: Path) -> None:
    run_root, manifest, _, payload = _case(tmp_path)
    binding = _binding(payload)
    identity = dict(binding["identity_row"])
    identity["evidence_id"] = "ev_other"
    binding["identity_row"] = identity
    contract = dict(binding["product_contract"])
    contract["identity_row"] = identity
    binding["product_contract"] = contract
    _write_manifest(manifest, payload)

    with pytest.raises(TypedInputReceiptError, match="identity row mismatch"):
        _load(run_root, manifest)


def test_relative_path_escape_is_rejected(tmp_path: Path) -> None:
    run_root, manifest, _, payload = _case(tmp_path)
    _binding(payload)["relative_path"] = "../outside.parquet"
    _write_manifest(manifest, payload)

    with pytest.raises(TypedInputReceiptError, match="relative path is unsafe"):
        _load(run_root, manifest)


def test_absolute_path_cannot_override_the_resolved_relative_artifact(
    tmp_path: Path,
) -> None:
    run_root, manifest, _, payload = _case(tmp_path)
    other = run_root / "evidence" / "other.parquet"
    pd.DataFrame(
        {"record_id": [101, 102, 103], "measurement": [1.0, 2.0, 3.0]}
    ).to_parquet(other, index=False)
    _binding(payload)["absolute_path"] = str(other)
    _write_manifest(manifest, payload)

    with pytest.raises(TypedInputReceiptError, match="paths disagree"):
        _load(run_root, manifest)


def test_symlink_artifact_is_rejected(tmp_path: Path) -> None:
    run_root, manifest, artifact, payload = _case(tmp_path)
    target = artifact.with_name("real.parquet")
    artifact.replace(target)
    artifact.symlink_to(target.name)
    binding = _binding(payload)
    binding["sha256"] = _sha256(target)
    identity = dict(binding["identity_row"])
    identity["sha256"] = _sha256(target)
    binding["identity_row"] = identity
    contract = dict(binding["product_contract"])
    contract["identity_row"] = identity
    binding["product_contract"] = contract
    _write_manifest(manifest, payload)

    with pytest.raises(TypedInputReceiptError, match="regular table"):
        _load(run_root, manifest)


def test_symlink_parent_directory_is_rejected(tmp_path: Path) -> None:
    run_root, manifest, artifact, payload = _case(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / artifact.name
    artifact.replace(target)
    artifact.parent.rmdir()
    artifact.parent.symlink_to(outside, target_is_directory=True)
    binding = _binding(payload)
    binding["sha256"] = _sha256(target)
    identity = dict(binding["identity_row"])
    identity["sha256"] = _sha256(target)
    binding["identity_row"] = identity
    contract = dict(binding["product_contract"])
    contract["identity_row"] = identity
    binding["product_contract"] = contract
    _write_manifest(manifest, payload)

    with pytest.raises(TypedInputReceiptError, match="regular table"):
        _load(run_root, manifest)


def test_duplicate_row_identity_is_rejected_even_if_digest_matches(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame({"record_id": [101, 101, 103], "measurement": [1.0, 2.0, 3.0]})
    run_root, manifest, _, _ = _case(tmp_path, frame=frame)

    with pytest.raises(TypedInputReceiptError, match="duplicate row identity"):
        _load(run_root, manifest)


def test_missing_row_identity_is_rejected(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"record_id": [101, None, 103], "measurement": [1.0, 2.0, 3.0]}
    )
    run_root, manifest, _, _ = _case(tmp_path, frame=frame)

    with pytest.raises(TypedInputReceiptError, match="missing row identity"):
        _load(run_root, manifest)


def test_sdk_never_guesses_an_identity_column(tmp_path: Path) -> None:
    run_root, manifest, _, payload = _case(tmp_path)
    binding = _binding(payload)
    contract = dict(binding["product_contract"])
    contract.pop("row_identity_column")
    contract.pop("row_identity_sha256")
    binding["product_contract"] = contract
    _write_manifest(manifest, payload)

    with pytest.raises(TypedInputReceiptError, match="row identity authority"):
        _load(run_root, manifest)


def test_contract_columns_must_match_loaded_table_exactly(tmp_path: Path) -> None:
    run_root, manifest, _, payload = _case(tmp_path)
    binding = _binding(payload)
    contract = dict(binding["product_contract"])
    contract["columns"] = ["record_id", "other"]
    binding["product_contract"] = contract
    _write_manifest(manifest, payload)

    with pytest.raises(TypedInputReceiptError, match="contract columns"):
        _load(run_root, manifest)


def test_manifest_consumer_step_must_match_current_step(tmp_path: Path) -> None:
    run_root, manifest, _, _ = _case(tmp_path)

    with pytest.raises(TypedInputReceiptError, match="consumer step mismatch"):
        load_typed_input(
            resolved_inputs_path=manifest,
            expected_resolved_inputs_sha256=_sha256(manifest),
            run_root=run_root,
            input_key=INPUT_KEY,
            consumer_step_id="old_step",
            consumer_code_sha256=CODE_SHA,
        )


def test_receipt_is_bound_to_the_host_supplied_current_code_sha(tmp_path: Path) -> None:
    run_root, manifest, _, _ = _case(tmp_path)
    current_code_sha = "d" * 64

    loaded = load_typed_input(
        resolved_inputs_path=manifest,
        expected_resolved_inputs_sha256=_sha256(manifest),
        run_root=run_root,
        input_key=INPUT_KEY,
        consumer_step_id=STEP_ID,
        consumer_code_sha256=current_code_sha,
    )

    assert loaded.receipt.consumer_code_sha256 == current_code_sha
    assert loaded.receipt.consumer_code_sha256 != CODE_SHA


def test_file_changed_after_initial_load_is_rejected_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, manifest, artifact, _ = _case(tmp_path)
    original_verify = sdk.verify_typed_input_consumption_receipt

    def mutate_then_verify(*args, **kwargs):
        pd.DataFrame(
            {"record_id": [101, 102, 103], "measurement": [8.0, 8.0, 8.0]}
        ).to_parquet(artifact, index=False)
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(
        sdk,
        "verify_typed_input_consumption_receipt",
        mutate_then_verify,
    )

    with pytest.raises(TypedInputReceiptError, match="artifact SHA-256"):
        _load(run_root, manifest)


def test_manifest_changed_after_initial_load_is_rejected_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, manifest, _, _ = _case(tmp_path)
    original_verify = sdk.verify_typed_input_consumption_receipt

    def mutate_then_verify(*args, **kwargs):
        manifest.write_text(
            manifest.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(
        sdk,
        "verify_typed_input_consumption_receipt",
        mutate_then_verify,
    )

    with pytest.raises(TypedInputReceiptError, match="manifest SHA-256"):
        _load(run_root, manifest)


def test_non_csv_non_parquet_transport_is_rejected(tmp_path: Path) -> None:
    run_root, manifest, _, _ = _case(tmp_path, format_name="tsv")

    with pytest.raises(TypedInputSDKError, match="unsupported SDK"):
        _load(run_root, manifest)


def test_other_planner_input_cannot_be_substituted(tmp_path: Path) -> None:
    run_root, manifest, _, _ = _case(tmp_path)

    with pytest.raises(TypedInputReceiptError, match="uniquely Planner-declared"):
        load_typed_input(
            resolved_inputs_path=manifest,
            expected_resolved_inputs_sha256=_sha256(manifest),
            run_root=run_root,
            input_key="artifact:other_table",
            consumer_step_id=STEP_ID,
            consumer_code_sha256=CODE_SHA,
        )
