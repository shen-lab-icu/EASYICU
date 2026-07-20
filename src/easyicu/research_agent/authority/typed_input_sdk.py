"""Host-owned SDK for loading one immutable typed tabular input.

Generated analysis code must not select an artifact path, attest that it read a
file, or construct its own consumption receipt.  The host supplies the
checkpoint-selected resolved-input manifest and consumer identity to
``load_typed_input``.  This module opens the selected artifact, verifies the
manifest/binding/artifact/row-identity authority, and returns one immutable
tabular payload paired with the receipt produced in the same call.

This is a leaf authority primitive.  It does not choose Planner inputs, execute
candidate code, or promote result/figure evidence.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa

from .typed_input_receipt import (
    TypedInputConsumptionReceipt,
    TypedInputReceiptError,
    _frame_digest,
    load_verified_typed_input_table,
    seal_typed_input_consumption,
    verify_typed_input_consumption_receipt,
)

_SUPPORTED_SUFFIXES = frozenset({".csv", ".parquet", ".pq"})
_SDK_CONSTRUCTION_TOKEN = object()


class TypedInputSDKError(TypedInputReceiptError):
    """The host could not produce one payload/receipt authority pair."""


class LoadedTypedInput:
    """One immutable tabular payload and its host-issued consumption receipt.

    The payload is a :class:`pyarrow.Table`, whose mutation-style operations
    return new tables rather than changing this authority object.  Consumers
    that need pandas receive a fresh copy from :meth:`to_pandas`; the host must
    pass this ``LoadedTypedInput`` object through the execution boundary rather
    than accepting a separately supplied DataFrame and receipt.

    Construction is deliberately SDK-private so callers cannot pair an
    arbitrary table with a valid receipt.
    """

    __slots__ = ("__payload", "__receipt", "__sealed")

    def __init__(
        self,
        *,
        payload: pa.Table,
        receipt: TypedInputConsumptionReceipt,
        _construction_token: object,
    ) -> None:
        if _construction_token is not _SDK_CONSTRUCTION_TOKEN:
            raise TypedInputSDKError(
                "LoadedTypedInput may only be constructed by load_typed_input"
            )
        if not isinstance(payload, pa.Table):
            raise TypedInputSDKError("typed-input payload must be a pyarrow Table")
        if not isinstance(receipt, TypedInputConsumptionReceipt):
            raise TypedInputSDKError(
                "typed-input payload requires a verified consumption receipt"
            )
        payload_frame_sha256 = _frame_digest(payload.to_pandas())
        if payload_frame_sha256 != receipt.loaded_frame_sha256:
            raise TypedInputSDKError(
                "typed-input payload does not match the consumption receipt"
            )
        object.__setattr__(self, "_LoadedTypedInput__payload", payload)
        object.__setattr__(self, "_LoadedTypedInput__receipt", receipt)
        object.__setattr__(self, "_LoadedTypedInput__sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_LoadedTypedInput__sealed", False):
            raise AttributeError("LoadedTypedInput is immutable")
        object.__setattr__(self, name, value)

    @property
    def payload(self) -> pa.Table:
        """Return the immutable Arrow table selected by host authority."""

        return self.__payload

    @property
    def receipt(self) -> TypedInputConsumptionReceipt:
        """Return the receipt bound to this exact payload and consumer."""

        return self.__receipt

    @property
    def input_key(self) -> str:
        """Return the Planner-declared logical input identity."""

        return self.__receipt.input_key

    def to_pandas(self) -> pd.DataFrame:
        """Materialize a fresh DataFrame copy of the immutable payload."""

        return self.__payload.to_pandas()


def load_typed_input(
    *,
    resolved_inputs_path: Path,
    expected_resolved_inputs_sha256: str,
    run_root: Path,
    input_key: str,
    consumer_step_id: str,
    consumer_code_sha256: str,
) -> LoadedTypedInput:
    """Atomically load one host-selected typed input and issue its receipt.

    The function accepts no artifact-path or caller-supplied DataFrame
    override.  CSV and Parquet are the only supported transport formats.  The
    receipt is reverified against current manifest and artifact bytes before
    the immutable payload is released, closing the load-to-receipt mutation
    window.  Later seal boundaries must reverify the receipt again to detect a
    file changed after this call returned.
    """

    loaded = load_verified_typed_input_table(
        resolved_inputs_path=resolved_inputs_path,
        expected_resolved_inputs_sha256=expected_resolved_inputs_sha256,
        run_root=run_root,
        input_key=input_key,
        consumer_step_id=consumer_step_id,
        consumer_code_sha256=consumer_code_sha256,
    )
    suffix = Path(loaded.artifact_relative_path).suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise TypedInputSDKError(
            f"unsupported SDK typed-input format: {suffix or '<none>'}"
        )

    receipt = seal_typed_input_consumption(
        loaded,
        consumed_frame=loaded.frame,
    )

    try:
        payload = pa.Table.from_pandas(
            loaded.frame,
            preserve_index=False,
            safe=True,
        )
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise TypedInputSDKError(
            "verified typed input cannot be represented as a read-only payload"
        ) from exc

    result = LoadedTypedInput(
        payload=payload,
        receipt=receipt,
        _construction_token=_SDK_CONSTRUCTION_TOKEN,
    )
    verified_receipt = verify_typed_input_consumption_receipt(
        result.receipt,
        resolved_inputs_path=resolved_inputs_path,
        expected_resolved_inputs_sha256=expected_resolved_inputs_sha256,
        run_root=run_root,
        input_key=input_key,
        consumer_step_id=consumer_step_id,
        consumer_code_sha256=consumer_code_sha256,
    )
    if verified_receipt != result.receipt:  # pragma: no cover - defensive
        raise TypedInputSDKError("typed-input receipt changed during SDK verification")
    return result


__all__ = [
    "LoadedTypedInput",
    "TypedInputSDKError",
    "load_typed_input",
]
