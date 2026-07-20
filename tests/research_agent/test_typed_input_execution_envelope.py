"""Typed-input execution envelope prototype and sink-proof tests."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import time

import pandas as pd
import pyarrow as pa
import pytest

from easyicu.research_agent.authority import typed_input_execution as execution
from easyicu.research_agent.authority.typed_input_execution import (
    TypedInputExecutionEnvelope,
    TypedInputExecutionError,
)
from easyicu.research_agent.authority.typed_input_sdk import load_typed_input

INPUT_A = "artifact:analysis_table"
INPUT_B = "artifact:comparison_table"
STEP_ID = "candidate_step"
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


def _write_source(path: Path, frame: pd.DataFrame, *, format_name: str) -> None:
    if format_name == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)


def _host_loaded_inputs(
    tmp_path: Path,
    *,
    format_name: str = "parquet",
    second_input: bool = False,
    row_count: int = 3,
) -> tuple[Path, Path, dict[str, object]]:
    run_root = tmp_path / "run"
    evidence = run_root / "evidence"
    resolved = run_root / "resolved_inputs"
    evidence.mkdir(parents=True)
    resolved.mkdir()
    inputs: dict[str, dict[str, object]] = {}
    declared: list[str] = []
    for offset, input_key in enumerate(
        [INPUT_A, INPUT_B] if second_input else [INPUT_A]
    ):
        frame = pd.DataFrame(
            {
                "record_id": list(
                    range(1000 + offset * 10000, 1000 + offset * 10000 + row_count)
                ),
                "exposure": [float(index % 4) for index in range(row_count)],
                "outcome": [float(index % 2) for index in range(row_count)],
            }
        )
        suffix = ".csv" if format_name == "csv" else ".parquet"
        artifact = evidence / f"source_{offset}{suffix}"
        _write_source(artifact, frame, format_name=format_name)
        artifact_sha = _sha256(artifact)
        product = input_key.split(":", 1)[1]
        identity = {
            "input_key": input_key,
            "declared_kind": "artifact",
            "product": product,
            "evidence_id": f"ev_{offset}",
            "sha256": artifact_sha,
            "produced_by_step": "producer_step",
        }
        inputs[input_key] = {
            "evidence_id": f"ev_{offset}",
            "declared_kind": "artifact",
            "product": product,
            "evidence_kind": "table",
            "relative_path": artifact.relative_to(run_root).as_posix(),
            "absolute_path": str(artifact),
            "sha256": artifact_sha,
            "produced_by_step": "producer_step",
            "identity_row": identity,
            "product_contract": {
                "schema_version": "easyicu.host_typed_product.v2",
                "identity_row": identity,
                "tabular_format": format_name,
                "column_count": len(frame.columns),
                "columns": list(frame.columns),
                "row_identity_column": "record_id",
                "row_count": len(frame),
                "row_identity_sha256": _identity_sha256(frame["record_id"]),
            },
        }
        declared.append(input_key)
    manifest_payload = {
        "schema_version": "2.1",
        "step_id": STEP_ID,
        "planner_declared_inputs": declared,
        "inputs": inputs,
    }
    manifest = resolved / f"{STEP_ID}.json"
    manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded = {
        input_key: load_typed_input(
            resolved_inputs_path=manifest,
            expected_resolved_inputs_sha256=_sha256(manifest),
            run_root=run_root,
            input_key=input_key,
            consumer_step_id=STEP_ID,
            consumer_code_sha256=CODE_SHA,
        )
        for input_key in declared
    }
    return run_root, manifest, loaded


def _envelope(
    tmp_path: Path,
    *,
    format_name: str = "parquet",
    second_input: bool = False,
    row_count: int = 3,
) -> tuple[Path, Path, TypedInputExecutionEnvelope]:
    run_root, manifest, loaded = _host_loaded_inputs(
        tmp_path,
        format_name=format_name,
        second_input=second_input,
        row_count=row_count,
    )
    envelope = TypedInputExecutionEnvelope(
        root=tmp_path / "candidate_envelope",
        inputs=loaded,
        consumer_step_id=STEP_ID,
        consumer_code_sha256=CODE_SHA,
    )
    return run_root, manifest, envelope


def _model_sink(table: pa.Table) -> bytes:
    frame = table.to_pandas()
    design = frame[["exposure", "outcome"]].astype(float)
    return json.dumps(
        {
            "rows": len(design),
            "cross_product": float((design["exposure"] * design["outcome"]).sum()),
        },
        sort_keys=True,
    ).encode("utf-8")


def _table_sink(table: pa.Table) -> bytes:
    return table.to_pandas().to_csv(index=False).encode("utf-8")


def _figure_sink(table: pa.Table) -> bytes:
    frame = table.to_pandas()
    return json.dumps(
        {
            "x": frame["exposure"].astype(float).tolist(),
            "y": frame["outcome"].astype(float).tolist(),
        },
        sort_keys=True,
    ).encode("utf-8")


def _empty_sink(_table: pa.Table) -> bytes:
    return b""


def _raising_sink(_table: pa.Table) -> bytes:
    raise RuntimeError("sink failed")


def _adapter(kind: str, callback):
    return execution._host_sink_adapter(
        kind=kind,
        adapter_id=f"test_{kind}_sink_v1",
        callback=callback,
    )


def _issue_codes(result) -> set[str]:
    return {
        str((finding.detail or {}).get("issue_code") or "")
        for finding in result.findings
    }


def test_candidate_sees_only_content_addressed_materialization(tmp_path: Path) -> None:
    run_root, _, envelope = _envelope(tmp_path)
    binding = envelope.candidate_bindings()[INPUT_A]
    candidate_path = envelope.candidate_path(INPUT_A)
    manifest_bytes = envelope.candidate_manifest_bytes()

    assert binding.relative_path == f"objects/{binding.sha256}.parquet"
    assert candidate_path.is_file()
    assert _sha256(candidate_path) == binding.sha256
    assert str(run_root).encode("utf-8") not in manifest_bytes
    assert b"ev_0" not in manifest_bytes
    assert b"source_0.parquet" not in manifest_bytes


def test_execution_receipt_binds_materialized_bytes_step_code_and_identity(
    tmp_path: Path,
) -> None:
    _, _, envelope = _envelope(tmp_path)
    receipt = envelope.execution_receipts[INPUT_A]

    assert receipt.materialized_sha256 == _sha256(envelope.candidate_path(INPUT_A))
    assert receipt.consumer_step_id == STEP_ID
    assert receipt.consumer_code_sha256 == CODE_SHA
    assert receipt.row_identity.column == "record_id"
    assert receipt.row_identity.row_count == 3
    assert receipt.row_identity.unique is True
    assert receipt.receipt_sha256 == execution._self_digest(
        receipt,
        digest_field="receipt_sha256",
    )


def test_csv_source_is_normalized_to_candidate_parquet(tmp_path: Path) -> None:
    _, _, envelope = _envelope(tmp_path, format_name="csv")
    binding = envelope.candidate_bindings()[INPUT_A]

    assert binding.format == "parquet"
    assert binding.relative_path.endswith(".parquet")
    assert pd.read_parquet(envelope.candidate_path(INPUT_A))["record_id"].tolist() == [
        1000,
        1001,
        1002,
    ]


def test_printing_candidate_path_does_not_prove_consumption(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, _, envelope = _envelope(tmp_path)

    print(envelope.candidate_path(INPUT_A))
    assert "objects" in capsys.readouterr().out

    result = envelope.verify_required_sinks({INPUT_A: ["model"]})
    assert _issue_codes(result) == {"unproven_downstream_sink"}
    assert not result.verified_proofs


def test_reading_and_one_subscript_access_do_not_prove_consumption(
    tmp_path: Path,
) -> None:
    _, _, envelope = _envelope(tmp_path)
    candidate_frame = pd.read_parquet(envelope.candidate_path(INPUT_A))
    _ = candidate_frame["exposure"]

    result = envelope.verify_required_sinks({INPUT_A: ["table"]})

    assert _issue_codes(result) == {"unproven_downstream_sink"}


@pytest.mark.parametrize(
    ("kind", "callback"),
    [("model", _model_sink), ("table", _table_sink), ("figure", _figure_sink)],
)
def test_host_owned_model_table_and_figure_sinks_issue_proof(
    tmp_path: Path,
    kind: str,
    callback,
) -> None:
    _, _, envelope = _envelope(tmp_path)

    proof = envelope.execute_host_sink(
        input_key=INPUT_A,
        adapter=_adapter(kind, callback),
    )
    result = envelope.verify_required_sinks({INPUT_A: [kind]})

    assert proof.sink_kind == kind
    assert (
        proof.materialized_sha256
        == envelope.execution_receipts[INPUT_A].materialized_sha256
    )
    assert proof.consumer_step_id == STEP_ID
    assert proof.consumer_code_sha256 == CODE_SHA
    assert not result.findings
    assert result.verified_proofs[(INPUT_A, kind)] == proof


def test_sink_proof_binds_output_and_adapter_implementation(tmp_path: Path) -> None:
    _, _, envelope = _envelope(tmp_path)
    adapter = _adapter("model", _model_sink)

    proof = envelope.execute_host_sink(input_key=INPUT_A, adapter=adapter)

    expected_output = _model_sink(
        pq_table := pa.parquet.read_table(envelope.candidate_path(INPUT_A))
    )
    assert pq_table.num_rows == 3
    assert proof.sink_output_sha256 == hashlib.sha256(expected_output).hexdigest()
    assert proof.sink_adapter_sha256 == adapter.implementation_sha256
    assert proof.proof_sha256 == execution._self_digest(
        proof,
        digest_field="proof_sha256",
    )


def test_candidate_cannot_inject_a_proof_collection() -> None:
    parameters = set(
        inspect.signature(TypedInputExecutionEnvelope.verify_required_sinks).parameters
    )

    assert parameters == {"self", "requirements"}
    assert not hasattr(TypedInputExecutionEnvelope, "add_candidate_proof")
    assert not hasattr(TypedInputExecutionEnvelope, "mark_consumed")


@pytest.mark.parametrize("requirements", [{}, {INPUT_A: []}])
def test_host_cannot_omit_sink_requirements(
    tmp_path: Path,
    requirements,
) -> None:
    _, _, envelope = _envelope(tmp_path)

    result = envelope.verify_required_sinks(requirements)

    assert _issue_codes(result) == {"missing_sink_requirement"}
    assert not result.verified_proofs


def test_arbitrary_dataframe_is_not_an_envelope_input(tmp_path: Path) -> None:
    with pytest.raises(TypedInputExecutionError, match="LoadedTypedInput"):
        TypedInputExecutionEnvelope(
            root=tmp_path / "envelope",
            inputs={INPUT_A: pd.DataFrame({"record_id": [1]})},  # type: ignore[dict-item]
            consumer_step_id=STEP_ID,
            consumer_code_sha256=CODE_SHA,
        )


def test_logical_input_cannot_be_relabelled(tmp_path: Path) -> None:
    _, _, loaded = _host_loaded_inputs(tmp_path)

    with pytest.raises(TypedInputExecutionError, match="logical input"):
        TypedInputExecutionEnvelope(
            root=tmp_path / "envelope",
            inputs={"artifact:other": loaded[INPUT_A]},
            consumer_step_id=STEP_ID,
            consumer_code_sha256=CODE_SHA,
        )


@pytest.mark.parametrize(
    ("step_id", "code_sha"),
    [("old_step", CODE_SHA), (STEP_ID, "d" * 64)],
)
def test_loaded_capability_cannot_replay_to_another_step_or_code(
    tmp_path: Path,
    step_id: str,
    code_sha: str,
) -> None:
    _, _, loaded = _host_loaded_inputs(tmp_path)

    with pytest.raises(TypedInputExecutionError, match="another step or code"):
        TypedInputExecutionEnvelope(
            root=tmp_path / "envelope",
            inputs=loaded,
            consumer_step_id=step_id,
            consumer_code_sha256=code_sha,
        )


def test_materialized_bytes_changed_before_sink_fail_closed(tmp_path: Path) -> None:
    _, _, envelope = _envelope(tmp_path)
    envelope.candidate_path(INPUT_A).write_bytes(b"not parquet")

    with pytest.raises(TypedInputExecutionError, match="changed before sink"):
        envelope.execute_host_sink(
            input_key=INPUT_A,
            adapter=_adapter("model", _model_sink),
        )


def test_materialized_symlink_swap_before_sink_fail_closed(tmp_path: Path) -> None:
    _, _, envelope = _envelope(tmp_path)
    selected = envelope.candidate_path(INPUT_A)
    target = selected.with_name("other.parquet")
    target.write_bytes(selected.read_bytes())
    selected.unlink()
    selected.symlink_to(target.name)

    with pytest.raises(TypedInputExecutionError, match="changed before sink"):
        envelope.execute_host_sink(
            input_key=INPUT_A,
            adapter=_adapter("model", _model_sink),
        )


@pytest.mark.parametrize("callback", [_empty_sink, _raising_sink])
def test_sink_without_bound_output_fails_closed(
    tmp_path: Path,
    callback,
) -> None:
    _, _, envelope = _envelope(tmp_path)

    with pytest.raises(TypedInputExecutionError, match="sink"):
        envelope.execute_host_sink(
            input_key=INPUT_A,
            adapter=_adapter("model", callback),
        )

    assert _issue_codes(envelope.verify_required_sinks({INPUT_A: ["model"]})) == {
        "unproven_downstream_sink"
    }


def test_proof_for_one_input_cannot_satisfy_another_input(tmp_path: Path) -> None:
    _, _, envelope = _envelope(tmp_path, second_input=True)
    envelope.execute_host_sink(
        input_key=INPUT_A,
        adapter=_adapter("model", _model_sink),
    )

    result = envelope.verify_required_sinks({INPUT_A: ["model"], INPUT_B: ["model"]})

    assert _issue_codes(result) == {"unproven_downstream_sink"}
    assert not result.verified_proofs


def test_duplicate_sink_proofs_are_ambiguous(tmp_path: Path) -> None:
    _, _, envelope = _envelope(tmp_path)
    adapter = _adapter("model", _model_sink)
    envelope.execute_host_sink(input_key=INPUT_A, adapter=adapter)
    envelope.execute_host_sink(input_key=INPUT_A, adapter=adapter)

    result = envelope.verify_required_sinks({INPUT_A: ["model"]})

    assert _issue_codes(result) == {"ambiguous_downstream_sink_proof"}


def test_unknown_input_and_duplicate_requirement_fail_closed(tmp_path: Path) -> None:
    _, _, envelope = _envelope(tmp_path)

    result = envelope.verify_required_sinks(
        {"artifact:unknown": ["model"], INPUT_A: ["table", "table"]}
    )

    assert _issue_codes(result) == {
        "duplicate_sink_requirement",
        "unknown_required_input",
        "unproven_downstream_sink",
    }


def test_1000_row_materialization_and_model_sink_overhead_is_bounded(
    tmp_path: Path,
) -> None:
    _, _, loaded = _host_loaded_inputs(tmp_path, row_count=1000)
    started = time.perf_counter()
    envelope = TypedInputExecutionEnvelope(
        root=tmp_path / "envelope",
        inputs=loaded,
        consumer_step_id=STEP_ID,
        consumer_code_sha256=CODE_SHA,
    )
    envelope.execute_host_sink(
        input_key=INPUT_A,
        adapter=_adapter("model", _model_sink),
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0
    assert not envelope.verify_required_sinks({INPUT_A: ["model"]}).findings
