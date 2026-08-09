"""Shared canonical JSON and digest contracts."""

from __future__ import annotations

import math

import pytest

from easyicu.research_agent.canonical_json import (
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    sha256_bytes,
)


PAYLOAD = {"z": "é", "a": [1, True]}
CANONICAL_BYTES = b'{"a":[1,true],"z":"\xc3\xa9"}'
CANONICAL_SHA256 = "79be0936a21b307d847835501e81f912ee9fce7b992c6f43dd588e8539da8510"


def test_canonical_representation_is_byte_stable() -> None:
    assert canonical_json(PAYLOAD) == CANONICAL_BYTES.decode("utf-8")
    assert canonical_json_bytes(PAYLOAD) == CANONICAL_BYTES
    assert sha256_bytes(CANONICAL_BYTES) == CANONICAL_SHA256
    assert canonical_sha256(PAYLOAD) == CANONICAL_SHA256


def test_trailing_newline_is_an_explicit_protocol_choice() -> None:
    assert canonical_json_bytes(PAYLOAD, trailing_newline=True) == (
        CANONICAL_BYTES + b"\n"
    )
    assert canonical_sha256(PAYLOAD, trailing_newline=True) == (
        "325ec2d58161618818a05dcbff80f9b52b45f86f1bb0e904e2aec28d48c6fd51"
    )


def test_non_finite_values_and_non_json_objects_fail_closed() -> None:
    with pytest.raises(ValueError):
        canonical_json({"value": math.nan})
    with pytest.raises(TypeError):
        canonical_json({"value": object()})
