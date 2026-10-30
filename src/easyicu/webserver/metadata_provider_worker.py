"""Fixed, guarded metadata-only Provider request subprocess.

The supervisor owns a Host execution lease. The worker accepts one request on
stdin and returns one bounded result on stdout; neither credentials nor prompt
are written to disk. This module never accepts a script or arbitrary command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Mapping

from easyicu.runtime.host_execution_lease import (
    acknowledge_host_execution_parent,
    assert_host_execution_quiescent,
    claim_host_execution,
    complete_host_worker_cleanup,
    prepare_host_execution,
    read_host_execution_receipt,
    request_host_execution_stop,
    verify_host_worker,
)

_MAX_MESSAGE = 2 * 1024 * 1024


class MetadataProviderWorkerTimeout(RuntimeError):
    """The local worker timed out; remote usage and completion are unknown.

    This is deliberately not a TimeoutError: the remote call may still have
    consumed resources and must not acquire automatic recovery authority.
    """


class MetadataProviderWorkerResponseError(RuntimeError):
    """The supervised worker did not return a valid IPC response."""


def _decode_worker_response(output: bytes) -> dict[str, Any]:
    try:
        value = json.loads(output)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise MetadataProviderWorkerResponseError("metadata_provider_worker_response_invalid") from None
    if not isinstance(value, dict):
        raise MetadataProviderWorkerResponseError("metadata_provider_worker_response_invalid")
    return value


def identity() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _worker(lease_dir: Path, nonce: str, expected: str) -> int:
    if verify_host_worker(lease_dir, nonce) != expected or identity() != expected:
        return 125
    status = 0
    try:
        raw = sys.stdin.buffer.read(_MAX_MESSAGE + 1)
        if len(raw) > _MAX_MESSAGE:
            raise ValueError("provider_request_too_large")
        request = json.loads(raw)
        # The trust owner pins reviewed adapters to their public llm module.
        import easyicu.research_agent.providers.llm  # noqa: F401

        from easyicu.webserver.provider_adapter import build_research_agent_provider_client
        from easyicu.research_agent.providers.protocol import LLMMessage, StructuredOutputRequest

        client, public = build_research_agent_provider_client(
            request["provider"], environ=request["environment"],
            request_timeout=request["timeout"], request_hard_timeout=request["timeout"],
            single_transport_attempt=True,
            supports_strict_json_schema=request.get("strict_json_schema_enabled") is True,
        )
        for key, value in request["provider_identity"].items():
            if public.get(key) != value:
                raise ValueError("provider_identity_changed")
        # The parent owns the pre-transport reservation. One child invocation
        # must issue at most one request, including for SDK-managed transports.
        if public.get("transport_max_attempts") != 1:
            raise ValueError("provider_single_attempt_unsupported")
        messages = [LLMMessage(role=row["role"], content=row["content"]) for row in request["messages"]]
        schema = request.get("structured_output")
        structured = None
        if schema is not None:
            if not isinstance(schema, dict) or type(schema.get("strict")) is not bool:
                raise ValueError("provider_schema_invalid")
            structured = StructuredOutputRequest.from_schema(
                name=schema.get("name"), schema=json.loads(schema.get("schema_json") or ""),
                strict=schema["strict"],
            )
            if structured.schema_json != schema.get("schema_json"):
                raise ValueError("provider_schema_not_canonical")
        kwargs = {"max_tokens": request["max_tokens"], "temperature": request["temperature"]}
        if structured is not None:
            kwargs["structured_output"] = structured
        text, usage = client.complete_with_usage(messages, **kwargs)
        from easyicu.research_agent.providers.clients import safe_provider_finish_reason

        result: dict[str, Any] = {"ok": True, "text": text, "usage": usage,
                                  "finish_reason": safe_provider_finish_reason(getattr(client, "last_finish_reason", None))}
    except BaseException as exc:
        from easyicu.research_agent.providers.clients import safe_provider_http_status_code

        code = safe_provider_http_status_code(exc)
        detail = getattr(exc, "detail", {})
        reason = detail.get("error") if isinstance(detail, dict) else None
        result = {"ok": False, "type": type(exc).__name__, "status_code": code,
                  "reason": reason if isinstance(reason, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,99}", reason) else None,
                  "cause_type": detail.get("reason") if isinstance(detail, dict) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,99}", str(detail.get("reason") or "")) else None}
        status = 1
    encoded = json.dumps(result, allow_nan=False, separators=(",", ":")).encode()
    if len(encoded) > _MAX_MESSAGE:
        encoded = b'{"ok":false,"type":"ProviderResponseTooLarge","status_code":null}'
        status = 1
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()
    complete_host_worker_cleanup(lease_dir, nonce, exitcode=status)
    while True:
        time.sleep(60)


def _supervise(lease_dir: Path, nonce: str, expected: str, timeout: float) -> int:
    if identity() != expected or timeout <= 0:
        return 125
    lease = claim_host_execution(lease_dir, nonce)
    worker = None
    try:
        if lease.script_sha256 != expected:
            lease.mark_quiesced("worker_identity_mismatch", cleanup_proven=True)
            return 125
        if not lease.parent_alive() or lease.stop_requested():
            lease.mark_quiesced("cancelled_before_worker", cleanup_proven=True)
            return 125
        worker = subprocess.Popen(
            [sys.executable, "-m", "easyicu.webserver.metadata_provider_worker", "worker",
             str(lease_dir), nonce, expected, str(timeout)],
            start_new_session=os.name == "posix",
        )
        lease.record_worker(worker.pid)
        deadline = time.monotonic() + timeout
        while True:
            if worker.poll() is not None:
                lease.mark_unresolved("worker_exit_without_cleanup")
                return 125
            completion = lease.worker_completion_status()
            reason = (("worker_completed" if completion == 0 else "worker_failed") if completion is not None
                      else "parent_lost" if not lease.parent_alive()
                      else "stop_requested" if lease.stop_requested()
                      else "timeout" if time.monotonic() >= deadline else None)
            if reason is not None:
                stopped = lease.terminate_worker_tree()
                try:
                    worker.wait(timeout=3)
                except (OSError, subprocess.TimeoutExpired):
                    stopped = False
                if stopped:
                    lease.mark_quiesced(reason, cleanup_proven=True)
                else:
                    lease.mark_unresolved("worker_cleanup_unconfirmed")
                return completion if completion is not None and stopped else 125
            time.sleep(0.05)
    except BaseException:
        if worker is not None:
            try:
                stopped = lease.terminate_worker_tree()
                worker.wait(timeout=3)
            except BaseException:
                stopped = False
        else:
            stopped = True
        try:
            if stopped:
                lease.mark_quiesced("guardian_failed", cleanup_proven=True)
            else:
                lease.mark_unresolved("guardian_cleanup_unconfirmed")
        except BaseException:
            pass
        return 125
    finally:
        lease.release()


def guarded_call(lease_dir: Path, request: Mapping[str, Any], *, timeout: float,
                 on_prepared: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Run one already-reserved request with a verifiable local child lifecycle."""
    if getattr(sys, "frozen", False):
        raise RuntimeError("metadata_provider_frozen_worker_unsupported")
    encoded = json.dumps(dict(request), allow_nan=False, separators=(",", ":")).encode()
    if len(encoded) > _MAX_MESSAGE:
        raise ValueError("provider_request_too_large")
    lease = prepare_host_execution(lease_dir, script_sha256=identity())
    spawned = False
    process = None
    try:
        if on_prepared is not None:
            on_prepared(lease.nonce)
        lease.mark_spawning()
        process = subprocess.Popen(
            [sys.executable, "-m", "easyicu.webserver.metadata_provider_worker", "supervise",
             str(lease.path), lease.nonce, lease.script_sha256, str(timeout)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=os.name == "posix",
        )
        lease.bind_guardian(process.pid)
        spawned = True
        lease.release()
        try:
            output, _ = process.communicate(encoded, timeout=timeout + 15)
        except subprocess.TimeoutExpired:
            request_host_execution_stop(lease.path, lease.nonce)
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise MetadataProviderWorkerTimeout("metadata_provider_supervisor_timeout") from None
        assert_host_execution_quiescent(lease.path, allow_missing=False, allow_current_parent=True, wait_timeout=2)
        receipt = read_host_execution_receipt(lease.path)
        if receipt is not None and receipt.get("nonce") == lease.nonce and receipt.get("reason") == "timeout":
            raise MetadataProviderWorkerTimeout("metadata_provider_worker_timeout")
        if len(output) > _MAX_MESSAGE:
            raise ValueError("provider_response_too_large")
        return _decode_worker_response(output)
    finally:
        if not spawned:
            try:
                if process is None:
                    lease.abort()
            finally:
                lease.release()
        else:
            acknowledge_host_execution_parent(lease.path, lease.nonce)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("supervise", "worker"))
    parser.add_argument("lease_dir", type=Path)
    parser.add_argument("nonce")
    parser.add_argument("expected")
    parser.add_argument("timeout", type=float)
    args = parser.parse_args()
    if args.action == "worker":
        return _worker(args.lease_dir, args.nonce, args.expected)
    return _supervise(args.lease_dir, args.nonce, args.expected, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
