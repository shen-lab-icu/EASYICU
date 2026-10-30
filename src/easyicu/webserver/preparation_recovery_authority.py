"""Narrow authorization check for a consumed preparation decision.

This module is not connected to an execution route. A valid decision alone
does not establish a safe recovery: the host must also verify its durable
intent, runtime, immutable inputs, cancellation, and previous worker cleanup.
"""
from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Mapping

from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding, verify_consumed_host_review


def verify_recovery_authority(*, checkpoint: RawPreparationCheckpoint,
                              binding: RawPreparationHostBinding,
                              scope: Mapping[str, Any], job_id: str) -> Mapping[str, Any]:
    """Verify that an exact, already consumed host decision is still in date.

    The read-only completed-result verifier deliberately permits expired
    decisions. Calling this function instead never changes that behavior.
    """
    if not isinstance(job_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', job_id):
        raise ValueError('preparation_job_invalid')
    if canonical_sha256(checkpoint.request.payload.get('raw_preparation_scope')) != canonical_sha256(dict(scope)):
        raise ValueError('preparation_recovery_scope_mismatch')
    receipt = verify_consumed_host_review(checkpoint=checkpoint, binding=binding, scope=scope)
    if receipt.get('job_id') != job_id:
        raise ValueError('preparation_job_binding_mismatch')
    try:
        expires = datetime.fromisoformat(receipt['expires_at'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('preparation_continuation_expiry_invalid') from exc
    if expires.tzinfo is None or expires.utcoffset() is None:
        raise ValueError('preparation_continuation_expiry_invalid')
    if datetime.now(timezone.utc) >= expires:
        raise ValueError('preparation_continuation_expired')
    return receipt
