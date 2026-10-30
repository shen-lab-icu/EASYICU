"""One supervised transport attempt at a time for metadata-only acquisition."""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from easyicu.research_agent.authority.provider_hard_stop import (
    consume_active_provider_hard_stop_attempt, record_active_provider_retry_wait,
    record_active_provider_transport_failure, record_active_metadata_capacity_rejection,
    record_active_metadata_coordination_rejection,
)
from easyicu.research_agent.providers.transport_retry import TransportRetrySchedule
from easyicu.webserver import metadata_provider_concurrency as slots
from easyicu.webserver.metadata_provider_worker import guarded_call, identity


class MetadataProviderError(RuntimeError):
    def __init__(self, reason: str, *, status_code: int | None = None) -> None:
        super().__init__(reason)
        self.status_code = status_code


class MetadataProviderCoordinationUnavailable(RuntimeError):
    """Shared scope verification failed before a metadata worker was started."""


def guarded_acquisition_supported() -> bool:
    """The fixed subprocess entry point is presently available in source runs."""
    return not getattr(sys, "frozen", False)


def guarded_metadata_resume_scope(*, has_checkpoint: bool, scope: Any, acquisition: Any) -> bool:
    """Permit metadata-only continuation, never a prepared-patient grouping."""
    if scope is None:
        return not has_checkpoint and acquisition is None
    if (getattr(scope, "budget_mode", None) != "planner_canary"
            or getattr(scope, "prepared_grouping", None) is not None):
        return False
    if has_checkpoint:
        return getattr(acquisition, "kind", None) == "metadata_only_planning_catalog"
    # The launch owner may explicitly restart metadata planning while retaining
    # its verified literature and scientific requirements, with no input yet.
    return acquisition is None


class MetadataProviderClient:
    provider_attempt_budget_aware = True

    def __init__(self, *, inner: Any, task: Any, provider: Mapping[str, Any],
                 environment: Mapping[str, str], public: Mapping[str, Any], timeout: float) -> None:
        self._inner = inner
        self._task = task
        self._provider = dict(provider)
        self._environment = dict(environment)
        self._public = {key: public.get(key) for key in (
            "provider", "model", "credential_fingerprint", "endpoint_fingerprint",
            "credential_header", "base_url_endpoint", "session_binding_sha256",
        )}
        self._account = slots.account_identity(self._public)
        self._timeout = float(timeout)
        self._model = str(public.get("model") or "metadata-provider")
        self.name = "metadata_provider_guarded"
        self.last_finish_reason = None
        self.last_usage = None
        self.last_transport_attempts = 0
        from easyicu.research_agent.providers.factory import _register_provider_wrapper

        _register_provider_wrapper(self, children_getter=lambda: (self._inner,))

    @property
    def supports_strict_json_schema(self) -> bool:
        from easyicu.research_agent.providers.capabilities import llm_supports_strict_json_schema

        return llm_supports_strict_json_schema(self._inner)

    def complete(self, messages: Sequence[Any], *, max_tokens: int = 2048,
                 temperature: float = 0.2, structured_output: Any = None) -> str:
        response, _ = self.complete_with_usage(
            messages, max_tokens=max_tokens, temperature=temperature,
            structured_output=structured_output,
        )
        return response

    def complete_with_usage(self, messages: Sequence[Any], *, max_tokens: int = 2048,
                            temperature: float = 0.2, structured_output: Any = None) -> tuple[str, Any]:
        from easyicu.research_agent.providers.protocol import StructuredOutputRequest

        if structured_output is not None and not isinstance(structured_output, StructuredOutputRequest):
            raise ValueError("metadata_provider_structured_output_invalid")
        self.last_finish_reason = None
        self.last_usage = None
        self.last_transport_attempts = 0
        request = {
            "provider": self._provider, "environment": self._environment,
            "provider_identity": self._public, "timeout": self._timeout,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens, "temperature": temperature,
            "strict_json_schema_enabled": self.supports_strict_json_schema,
            "structured_output": ({"name": structured_output.name, "schema_json": structured_output.schema_json,
                                   "strict": structured_output.strict} if structured_output is not None else None),
        }
        schedule = TransportRetrySchedule(delays_seconds=(20.0, 180.0), jitter_fraction=0.2, window_seconds=300.0)
        start = time.monotonic()
        for attempt in range(3):
            remaining = consume_active_provider_hard_stop_attempt()
            if remaining is None:
                raise RuntimeError("metadata_provider_budget_scope_missing")
            snapshot = self._task.ledger.snapshot()
            rows = [row for row in snapshot["tasks"] if row.get("task_id") == self._task.task_id]
            calls = rows[0]["calls"] if len(rows) == 1 else []
            if not calls or calls[-1].get("state") != "in_progress" or calls[-1].get("role") not in {"acquisition", "planner", "provider_recovery_probe"}:
                raise RuntimeError("metadata_provider_reservation_missing")
            # Recovery controls its own probe interval and deadline. Do not
            # hide additional transport retries inside a single health probe.
            is_probe = calls[-1].get("role") == "provider_recovery_probe"
            attempt_id = calls[-1]["attempt_id"]
            from easyicu.research_agent.providers.metadata_request_binding import current_metadata_request_sha256

            guard = Path(self._task.ledger.path).parent / ".metadata-provider-execution" / attempt_id
            from easyicu.webserver.metadata_provider_shared_scope import SharedScopeError

            try:
                slots.reserve(attempt_id=attempt_id, account=self._account, task_id=self._task.task_id,
                              ledger_path=self._task.ledger.path, guard_path=guard, worker_sha256=identity(),
                              request_sha256=current_metadata_request_sha256())
            except slots.MetadataProviderCapacityExhausted:
                # This branch is owned by the local slot admission itself and
                # runs before the supervised worker can receive a request.
                record_active_metadata_capacity_rejection()
                raise
            except SharedScopeError as exc:
                # This is distinct from a later scope failure during release:
                # the worker has not been started and no request was sent.
                record_active_metadata_coordination_rejection()
                raise MetadataProviderCoordinationUnavailable(
                    "metadata_provider_shared_scope_unavailable"
                ) from exc
            value = guarded_call(
                guard, request, timeout=max(0.001, min(self._timeout, remaining)),
                on_prepared=lambda nonce: slots.bind_guard(attempt_id, nonce),
            )
            self.last_transport_attempts = attempt + 1
            code = value.get("status_code")
            if value.get("ok") is True:
                slots.release(attempt_id)
                self.last_finish_reason = value.get("finish_reason")
                self.last_usage = value.get("usage")
                return str(value["text"]), value.get("usage")
            if type(code) is int:
                slots.release(attempt_id)
                record_active_provider_transport_failure(str(value.get("type") or "ProviderHTTPError"))
            # A connection timeout can leave remote usage unknown. Keep its
            # slot and reservation; retry only when the configured capacity
            # explicitly allows another simultaneous metadata request.
            unknown_transport = (code is None and value.get("type") in {
                "APITimeoutError", "APIConnectionError", "ReadTimeout", "ConnectTimeout", "ConnectionError",
            })
            retryable = (type(code) is int and 500 <= code <= 599) or (
                unknown_transport and slots.has_capacity(self._account)
            )
            if not retryable or is_probe or attempt == 2:
                raise MetadataProviderError(str(value.get("reason") or value.get("type") or "metadata_provider_failed"), status_code=code if type(code) is int else None)
            wait, stopped = schedule.next_wait(retry_index=attempt, since_start=time.monotonic() - start,
                                               since_reservation=0.0, wall_clock_remaining=self._task.assert_active(),
                                               retry_after=None, draw=random.random)
            if stopped or wait is None:
                raise MetadataProviderError("metadata_provider_retry_window_exhausted", status_code=code)
            record_active_provider_retry_wait(wait)
            time.sleep(wait)
        raise MetadataProviderError("metadata_provider_retry_exhausted")
