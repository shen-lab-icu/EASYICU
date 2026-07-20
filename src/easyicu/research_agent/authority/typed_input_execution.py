"""Prototype host envelope for typed-input execution and sink proof.

The candidate-visible surface contains only host-verified, content-addressed
materializations.  Source evidence paths and the mutable SDK load handle never
cross into the candidate process.  Downstream consumption authority is issued
only when a host-owned sink adapter is executed with the verified
materialization; path mentions, printing, reads, and isolated indexing do not
create a sink proof.

This module is deliberately not wired into the execution orchestrator yet.  It
models the boundary and its fail-closed invariants without selecting scientific
methods, inputs, estimands, or publication authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import marshal
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Annotated, Callable, Literal, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .filesystem import AnchoredDirectory, AuthorityFilesystemError
from .typed_input_receipt import (
    TypedInputReceiptError,
    TypedInputRowIdentity,
    _frame_digest,
    _identity_digest,
)
from .typed_input_sdk import LoadedTypedInput
from ..schema import ValidationFinding

TYPED_INPUT_EXECUTION_RECEIPT_SCHEMA = "easyicu.typed_input_execution_receipt/1"
TYPED_INPUT_SINK_PROOF_SCHEMA = "easyicu.typed_input_sink_proof/1"
TYPED_INPUT_CANDIDATE_MANIFEST_SCHEMA = "easyicu.typed_input_candidate_manifest/1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_SinkKind = Literal["model", "table", "figure"]
_HOST_SINK_TOKEN = object()


class TypedInputExecutionError(TypedInputReceiptError):
    """The execution envelope could not preserve typed-input authority."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TypedInputExecutionReceipt(_StrictFrozenModel):
    """Receipt for one content-addressed candidate materialization."""

    schema_version: Literal["easyicu.typed_input_execution_receipt/1"]
    input_key: str = Field(min_length=1)
    source_consumption_receipt_sha256: _Sha256
    source_artifact_sha256: _Sha256
    materialized_sha256: _Sha256
    materialized_size_bytes: int = Field(ge=1)
    candidate_relative_path: str = Field(min_length=1)
    payload_frame_sha256: _Sha256
    row_identity: TypedInputRowIdentity
    consumer_step_id: str = Field(min_length=1)
    consumer_code_sha256: _Sha256
    receipt_sha256: _Sha256

    @model_validator(mode="after")
    def _verify_self_digest(self) -> "TypedInputExecutionReceipt":
        if _self_digest(self, digest_field="receipt_sha256") != self.receipt_sha256:
            raise ValueError("typed-input execution receipt SHA-256 mismatch")
        return self


class TypedInputCandidateBinding(_StrictFrozenModel):
    """The complete typed-input surface exposed to candidate code."""

    input_key: str = Field(min_length=1)
    format: Literal["parquet"]
    relative_path: str = Field(min_length=1)
    sha256: _Sha256
    size_bytes: int = Field(ge=1)
    row_identity: TypedInputRowIdentity


class TypedInputSinkProof(_StrictFrozenModel):
    """Host-issued proof that a materialization entered one real sink."""

    schema_version: Literal["easyicu.typed_input_sink_proof/1"]
    input_key: str = Field(min_length=1)
    execution_receipt_sha256: _Sha256
    materialized_sha256: _Sha256
    sink_kind: _SinkKind
    sink_adapter_id: str = Field(min_length=1)
    sink_adapter_sha256: _Sha256
    sink_output_sha256: _Sha256
    row_identity_sha256: _Sha256
    consumer_step_id: str = Field(min_length=1)
    consumer_code_sha256: _Sha256
    proof_sha256: _Sha256

    @model_validator(mode="after")
    def _verify_self_digest(self) -> "TypedInputSinkProof":
        if _self_digest(self, digest_field="proof_sha256") != self.proof_sha256:
            raise ValueError("typed-input sink proof SHA-256 mismatch")
        return self


@dataclass(frozen=True, slots=True)
class TypedInputSinkVerification:
    """Structured, non-boolean verification of required sink consumption."""

    findings: tuple[ValidationFinding, ...]
    verified_proofs: Mapping[tuple[str, str], TypedInputSinkProof]


class _HostSinkAdapter:
    """Host-only adapter capability; never include it in candidate globals."""

    __slots__ = ("kind", "adapter_id", "implementation_sha256", "callback")

    def __init__(
        self,
        *,
        kind: _SinkKind,
        adapter_id: str,
        implementation_sha256: str,
        callback: Callable[[pa.Table], bytes],
        _token: object,
    ) -> None:
        if _token is not _HOST_SINK_TOKEN:
            raise TypedInputExecutionError("sink adapters are host-owned capabilities")
        self.kind = kind
        self.adapter_id = adapter_id
        self.implementation_sha256 = implementation_sha256
        self.callback = callback


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _self_digest(value: BaseModel | Mapping[str, object], *, digest_field: str) -> str:
    payload = (
        value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
    )
    payload.pop(digest_field, None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _validated_sha256(value: object, *, name: str) -> str:
    candidate = str(value or "")
    if _SHA256_RE.fullmatch(candidate) is None:
        raise TypedInputExecutionError(f"{name} must be a SHA-256 digest")
    return candidate


def _validated_nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypedInputExecutionError(f"{name} must be a non-empty string")
    return value.strip()


def _ensure_objects_directory(root: Path) -> Path:
    candidate = Path(os.path.abspath(os.fspath(root.expanduser())))
    candidate.mkdir(parents=True, exist_ok=True)
    try:
        with AnchoredDirectory.open(candidate) as anchored:
            try:
                os.mkdir("objects", mode=0o700, dir_fd=anchored.fd)
                os.fsync(anchored.fd)
            except FileExistsError:
                info = anchored.stat("objects")
                if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                    raise TypedInputExecutionError(
                        "execution-envelope objects selector is unsafe"
                    )
    except (AuthorityFilesystemError, OSError) as exc:
        raise TypedInputExecutionError(
            "cannot create an anchored execution-envelope directory"
        ) from exc
    objects = candidate / "objects"
    try:
        with AnchoredDirectory.open(objects):
            pass
    except AuthorityFilesystemError as exc:
        raise TypedInputExecutionError(
            "execution-envelope objects directory is unsafe"
        ) from exc
    return objects


def _parquet_bytes(table: pa.Table) -> bytes:
    output = pa.BufferOutputStream()
    try:
        pq.write_table(
            table,
            output,
            compression="NONE",
            use_dictionary=False,
            write_statistics=True,
            version="2.6",
        )
        return output.getvalue().to_pybytes()
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise TypedInputExecutionError(
            "typed input cannot be materialized as canonical Parquet"
        ) from exc


def _host_sink_adapter(
    *,
    kind: _SinkKind,
    adapter_id: str,
    callback: Callable[[pa.Table], bytes],
) -> _HostSinkAdapter:
    """Build a host-only sink adapter (prototype/internal, not candidate API)."""

    if kind not in {"model", "table", "figure"}:
        raise TypedInputExecutionError("unsupported typed-input sink kind")
    adapter_id = _validated_nonempty(adapter_id, name="sink adapter id")
    if not callable(callback):
        raise TypedInputExecutionError("sink adapter callback must be callable")
    code = getattr(callback, "__code__", None)
    if code is None:
        raise TypedInputExecutionError("sink adapter must expose immutable code")
    if getattr(callback, "__closure__", None):
        raise TypedInputExecutionError(
            "sink adapter must not capture mutable closure state"
        )
    try:
        source = inspect.getsource(callback).encode("utf-8")
    except (OSError, TypeError) as exc:
        raise TypedInputExecutionError(
            "sink adapter source is not inspectable"
        ) from exc
    implementation_sha256 = hashlib.sha256(
        _canonical_json_bytes(
            {
                "module": str(getattr(callback, "__module__", "")),
                "qualname": str(getattr(callback, "__qualname__", "")),
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "code_sha256": hashlib.sha256(marshal.dumps(code)).hexdigest(),
            }
        )
    ).hexdigest()
    return _HostSinkAdapter(
        kind=kind,
        adapter_id=adapter_id,
        implementation_sha256=implementation_sha256,
        callback=callback,
        _token=_HOST_SINK_TOKEN,
    )


def _finding(
    *,
    issue_code: str,
    message: str,
    input_key: str | None = None,
    sink_kind: str | None = None,
) -> ValidationFinding:
    detail: dict[str, object] = {"issue_code": issue_code}
    if input_key is not None:
        detail["input_key"] = input_key
    if sink_kind is not None:
        detail["sink_kind"] = sink_kind
    return ValidationFinding(
        validator="typed_input_execution_envelope",
        severity="error",
        message=message,
        detail=detail,
    )


class TypedInputExecutionEnvelope:
    """Host-side owner of candidate materializations and sink proofs."""

    __slots__ = (
        "__root",
        "__objects",
        "__step_id",
        "__code_sha256",
        "__receipts",
        "__proofs",
    )

    def __init__(
        self,
        *,
        root: Path,
        inputs: Mapping[str, LoadedTypedInput],
        consumer_step_id: str,
        consumer_code_sha256: str,
    ) -> None:
        self.__step_id = _validated_nonempty(
            consumer_step_id,
            name="consumer step",
        )
        self.__code_sha256 = _validated_sha256(
            consumer_code_sha256,
            name="consumer code SHA-256",
        )
        if not isinstance(inputs, Mapping) or not inputs:
            raise TypedInputExecutionError(
                "execution envelope requires host-loaded typed inputs"
            )
        self.__root = Path(os.path.abspath(os.fspath(Path(root).expanduser())))
        self.__objects = _ensure_objects_directory(self.__root)
        receipts: dict[str, TypedInputExecutionReceipt] = {}
        for input_key, loaded in sorted(inputs.items()):
            if not isinstance(input_key, str) or not input_key:
                raise TypedInputExecutionError("envelope input key is invalid")
            if not isinstance(loaded, LoadedTypedInput):
                raise TypedInputExecutionError(
                    "execution envelope accepts only LoadedTypedInput capabilities"
                )
            source = loaded.receipt
            if source.input_key != input_key:
                raise TypedInputExecutionError(
                    "logical input does not match its host-loaded capability"
                )
            if (
                source.consumer_step_id != self.__step_id
                or source.consumer_code_sha256 != self.__code_sha256
            ):
                raise TypedInputExecutionError(
                    "host-loaded capability belongs to another step or code"
                )
            if not isinstance(source.row_identity, TypedInputRowIdentity):
                raise TypedInputExecutionError(
                    "execution envelope requires exact row-identity authority"
                )
            payload_frame_sha256 = _frame_digest(loaded.payload.to_pandas())
            if payload_frame_sha256 != source.loaded_frame_sha256:
                raise TypedInputExecutionError(
                    "host-loaded payload no longer matches its source receipt"
                )
            materialized = _parquet_bytes(loaded.payload)
            materialized_sha256 = hashlib.sha256(materialized).hexdigest()
            object_name = f"{materialized_sha256}.parquet"
            try:
                with AnchoredDirectory.open(self.__objects) as objects:
                    objects.publish_immutable_bytes(object_name, materialized)
                    verified = objects.read_bytes(
                        object_name,
                        max_bytes=len(materialized),
                        expected_size=len(materialized),
                        expected_sha256=materialized_sha256,
                    )
            except AuthorityFilesystemError as exc:
                raise TypedInputExecutionError(
                    "cannot publish typed input materialization"
                ) from exc
            if verified != materialized:  # pragma: no cover - digest already checks
                raise TypedInputExecutionError(
                    "published typed input materialization changed"
                )
            receipt_payload: dict[str, object] = {
                "schema_version": TYPED_INPUT_EXECUTION_RECEIPT_SCHEMA,
                "input_key": input_key,
                "source_consumption_receipt_sha256": source.receipt_sha256,
                "source_artifact_sha256": source.artifact_sha256,
                "materialized_sha256": materialized_sha256,
                "materialized_size_bytes": len(materialized),
                "candidate_relative_path": f"objects/{object_name}",
                "payload_frame_sha256": payload_frame_sha256,
                "row_identity": source.row_identity.model_dump(mode="json"),
                "consumer_step_id": self.__step_id,
                "consumer_code_sha256": self.__code_sha256,
            }
            receipt_payload["receipt_sha256"] = _self_digest(
                receipt_payload,
                digest_field="receipt_sha256",
            )
            receipts[input_key] = TypedInputExecutionReceipt.model_validate(
                receipt_payload
            )
        self.__receipts = MappingProxyType(receipts)
        self.__proofs: dict[tuple[str, str], list[TypedInputSinkProof]] = {}

    @property
    def execution_receipts(self) -> Mapping[str, TypedInputExecutionReceipt]:
        """Return immutable host materialization receipts."""

        return self.__receipts

    def candidate_bindings(self) -> Mapping[str, TypedInputCandidateBinding]:
        """Return the complete path surface that may be exposed to candidate code."""

        return MappingProxyType(
            {
                input_key: TypedInputCandidateBinding(
                    input_key=input_key,
                    format="parquet",
                    relative_path=receipt.candidate_relative_path,
                    sha256=receipt.materialized_sha256,
                    size_bytes=receipt.materialized_size_bytes,
                    row_identity=receipt.row_identity,
                )
                for input_key, receipt in self.__receipts.items()
            }
        )

    def candidate_manifest_bytes(self) -> bytes:
        """Return a source-path-free manifest for the candidate sandbox mount."""

        return _canonical_json_bytes(
            {
                "schema_version": TYPED_INPUT_CANDIDATE_MANIFEST_SCHEMA,
                "consumer_step_id": self.__step_id,
                "consumer_code_sha256": self.__code_sha256,
                "inputs": {
                    key: binding.model_dump(mode="json")
                    for key, binding in self.candidate_bindings().items()
                },
            }
        )

    def candidate_path(self, input_key: str) -> Path:
        """Resolve one candidate binding inside the anchored envelope root."""

        receipt = self.__receipts.get(input_key)
        if receipt is None:
            raise TypedInputExecutionError("unknown typed input")
        return self.__root / receipt.candidate_relative_path

    def execute_host_sink(
        self,
        *,
        input_key: str,
        adapter: _HostSinkAdapter,
    ) -> TypedInputSinkProof:
        """Execute one trusted sink with the exact materialized table."""

        if not isinstance(adapter, _HostSinkAdapter):
            raise TypedInputExecutionError(
                "downstream consumption requires a host-owned sink adapter"
            )
        receipt = self.__receipts.get(input_key)
        if receipt is None:
            raise TypedInputExecutionError("sink requested an unknown typed input")
        object_name = Path(receipt.candidate_relative_path).name
        try:
            with AnchoredDirectory.open(self.__objects) as objects:
                payload = objects.read_bytes(
                    object_name,
                    max_bytes=receipt.materialized_size_bytes,
                    expected_size=receipt.materialized_size_bytes,
                    expected_sha256=receipt.materialized_sha256,
                )
        except AuthorityFilesystemError as exc:
            raise TypedInputExecutionError(
                "candidate materialization changed before sink execution"
            ) from exc
        try:
            table = pq.read_table(pa.BufferReader(payload))
        except (pa.ArrowException, TypeError, ValueError) as exc:
            raise TypedInputExecutionError(
                "candidate materialization is not valid Parquet"
            ) from exc
        frame = table.to_pandas()
        if _frame_digest(frame) != receipt.payload_frame_sha256:
            raise TypedInputExecutionError(
                "candidate materialization does not match host payload"
            )
        row_identity = frame[receipt.row_identity.column]
        if (
            row_identity.isna().any()
            or row_identity.astype("string").duplicated().any()
            or len(frame) != receipt.row_identity.row_count
            or _identity_digest(row_identity) != receipt.row_identity.sha256
        ):
            raise TypedInputExecutionError(
                "candidate materialization row identity changed"
            )
        try:
            output = adapter.callback(table)
        except Exception as exc:
            raise TypedInputExecutionError("host sink execution failed") from exc
        if not isinstance(output, bytes) or not output:
            raise TypedInputExecutionError(
                "host sink must return non-empty output bytes for binding"
            )
        proof_payload: dict[str, object] = {
            "schema_version": TYPED_INPUT_SINK_PROOF_SCHEMA,
            "input_key": input_key,
            "execution_receipt_sha256": receipt.receipt_sha256,
            "materialized_sha256": receipt.materialized_sha256,
            "sink_kind": adapter.kind,
            "sink_adapter_id": adapter.adapter_id,
            "sink_adapter_sha256": adapter.implementation_sha256,
            "sink_output_sha256": hashlib.sha256(output).hexdigest(),
            "row_identity_sha256": receipt.row_identity.sha256,
            "consumer_step_id": self.__step_id,
            "consumer_code_sha256": self.__code_sha256,
        }
        proof_payload["proof_sha256"] = _self_digest(
            proof_payload,
            digest_field="proof_sha256",
        )
        proof = TypedInputSinkProof.model_validate(proof_payload)
        self.__proofs.setdefault((input_key, adapter.kind), []).append(proof)
        return proof

    def verify_required_sinks(
        self,
        requirements: Mapping[str, Sequence[_SinkKind]],
    ) -> TypedInputSinkVerification:
        """Fail-close unless every required logical input has one real sink proof."""

        findings: list[ValidationFinding] = []
        verified: dict[tuple[str, str], TypedInputSinkProof] = {}
        for input_key in sorted(set(self.__receipts) - set(requirements)):
            findings.append(
                _finding(
                    issue_code="missing_sink_requirement",
                    input_key=input_key,
                    message=(
                        "Host sink requirements omit a materialized typed input; "
                        "consumption cannot be proven."
                    ),
                )
            )
        for input_key in sorted(set(requirements) - set(self.__receipts)):
            findings.append(
                _finding(
                    issue_code="unknown_required_input",
                    input_key=input_key,
                    message="Sink requirement refers to an unknown typed input.",
                )
            )
        for input_key, kinds in sorted(requirements.items()):
            if isinstance(kinds, (str, bytes)) or not kinds:
                findings.append(
                    _finding(
                        issue_code="missing_sink_requirement",
                        input_key=input_key,
                        message="Typed input has no explicit downstream sink requirement.",
                    )
                )
                continue
            seen_kinds: set[str] = set()
            for kind in kinds:
                if kind not in {"model", "table", "figure"}:
                    findings.append(
                        _finding(
                            issue_code="invalid_sink_kind",
                            input_key=input_key,
                            sink_kind=str(kind),
                            message="Typed input requires an unsupported sink kind.",
                        )
                    )
                    continue
                if kind in seen_kinds:
                    findings.append(
                        _finding(
                            issue_code="duplicate_sink_requirement",
                            input_key=input_key,
                            sink_kind=kind,
                            message="Typed input sink requirement is duplicated.",
                        )
                    )
                    continue
                seen_kinds.add(kind)
                proofs = self.__proofs.get((input_key, kind), [])
                if not proofs:
                    findings.append(
                        _finding(
                            issue_code="unproven_downstream_sink",
                            input_key=input_key,
                            sink_kind=kind,
                            message=(
                                "No host-owned downstream sink consumed the exact "
                                "typed-input materialization."
                            ),
                        )
                    )
                elif len(proofs) > 1:
                    findings.append(
                        _finding(
                            issue_code="ambiguous_downstream_sink_proof",
                            input_key=input_key,
                            sink_kind=kind,
                            message="Multiple sink proofs claim the same required consumption.",
                        )
                    )
                else:
                    verified[(input_key, kind)] = proofs[0]
        if findings:
            verified = {}
        return TypedInputSinkVerification(
            findings=tuple(findings),
            verified_proofs=MappingProxyType(verified),
        )


__all__ = [
    "TYPED_INPUT_CANDIDATE_MANIFEST_SCHEMA",
    "TYPED_INPUT_EXECUTION_RECEIPT_SCHEMA",
    "TYPED_INPUT_SINK_PROOF_SCHEMA",
    "TypedInputCandidateBinding",
    "TypedInputExecutionEnvelope",
    "TypedInputExecutionError",
    "TypedInputExecutionReceipt",
    "TypedInputSinkProof",
    "TypedInputSinkVerification",
]
