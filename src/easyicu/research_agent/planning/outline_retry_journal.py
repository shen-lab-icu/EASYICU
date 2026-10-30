"""Private, host-authenticated replay of metadata-only planner attempts.

Only the Web metadata canary may construct this journal.  It is not a plan,
review, execution authority, or a source of patient data.  Responses remain
private and are revalidated by the original planner in their original order.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any, Mapping, Sequence

from easyicu import state_paths
from easyicu.research_agent.canonical_json import canonical_json, canonical_sha256
from easyicu.research_agent.providers.factory import _register_provider_wrapper, authorized_complete
from easyicu.research_agent.providers.llm import current_provider_call_receipt, safe_provider_finish_reason

SCHEMA = "easyicu.metadata-outline-retry/1"
FILE_NAME = "metadata_outline_retry.json"
MAX_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 128 * 1024
MAX_CALLS = 5  # Maximum across the explicitly supported Planner stages.


def source_run_dir(path: Path) -> Path:
    """Map a private retry journal or a public accepted checkpoint to its run."""
    path = Path(path)
    return path.parent.parent if path.name == FILE_NAME and path.parent.name == ".runtime" else path.parent


def _code_digest() -> str:
    """Bind all Python source under EasyICU, not a hand-picked subset."""
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    paths = list(root.rglob("*.py"))
    paths.extend(path for path in (root / "data").rglob("*.json"))
    for path in sorted(paths):
        if path.is_symlink():
            raise ValueError("retry_journal_code_symlink")
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _key() -> bytes:
    directory = state_paths.state_root() / "private-recovery"
    if directory.is_symlink():
        raise ValueError("retry_journal_key_directory")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.stat().st_mode & 0o077:
        raise ValueError("retry_journal_key_permissions")
    path = directory / "outline-journal.key"
    if path.is_symlink():
        raise ValueError("retry_journal_key_symlink")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "wb") as target:
            target.write(secrets.token_bytes(32))
            target.flush()
            os.fsync(target.fileno())
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise ValueError("retry_journal_key_permissions")
        value = source.read(33)
    if len(value) != 32:
        raise ValueError("retry_journal_key_invalid")
    return value


def _signed(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("signature", None)
    body["signature"] = hmac.new(_key(), canonical_json(body).encode(), hashlib.sha256).hexdigest()
    return body


def load_retry_journal(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    path = Path(path)
    if path.name != FILE_NAME or path.is_symlink() or path.parent.is_symlink() or not path.is_file():
        raise ValueError("retry_journal_path_invalid")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_size > MAX_BYTES:
            raise ValueError("retry_journal_permissions")
        raw = source.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES or (expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256):
        raise ValueError("retry_journal_digest_invalid")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
        raise ValueError("retry_journal_schema_invalid")
    signature = value.get("signature")
    if not isinstance(signature, str) or not hmac.compare_digest(_signed(value)["signature"], signature):
        raise ValueError("retry_journal_signature_invalid")
    records = value.get("records")
    if (not isinstance(records, list) or len(records) > MAX_CALLS
            or type(value.get("abandoned")) is not int
            or not 0 <= value["abandoned"] <= MAX_CALLS
            or len(records) + value["abandoned"] > MAX_CALLS
            or type(value.get("initial_planner_calls")) is not int
            or value["initial_planner_calls"] < 0):
        raise ValueError("retry_journal_records_invalid")
    previous = value["initial_planner_calls"]
    for record in records:
        if (not isinstance(record, dict) or not isinstance(record.get("response"), str)
                or len(record["response"].encode()) > MAX_RESPONSE_BYTES
                or not isinstance(record.get("request_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", record["request_sha256"]) is None
                or type(record.get("planner_calls_after")) is not int
                or record["planner_calls_after"] < previous):
            raise ValueError("retry_journal_records_invalid")
        previous = record["planner_calls_after"]
    return value


def foundation_retry_source(checkpoint_path: Path) -> tuple[Path, str] | None:
    """Find a signed foundation retry beside the accepted source checkpoint.

    An earlier outline/family journal is expected after an accepted outline;
    it is not a foundation response and must never be replayed as one.
    A present malformed journal is rejected rather than treated as absent.
    """
    checkpoint_path = Path(checkpoint_path)
    journal = checkpoint_path.parent / ".runtime" / FILE_NAME
    if not journal.exists() and not journal.is_symlink():
        return None
    value = load_retry_journal(journal)
    stage = value.get("binding", {}).get("stage")
    if stage in {"outline", "family_spec"}:
        return None
    if stage != "foundation":
        raise ValueError("retry_journal_stage_invalid")
    return journal, hashlib.sha256(journal.read_bytes()).hexdigest()


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    parent = path.parent
    if parent.is_symlink():
        raise ValueError("retry_journal_directory_invalid")
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if parent.stat().st_mode & 0o077 or path.is_symlink():
        raise ValueError("retry_journal_permissions")
    data = (canonical_json(_signed(value)) + "\n").encode()
    if len(data) > MAX_BYTES:
        raise ValueError("retry_journal_too_large")
    temporary = parent / f".{FILE_NAME}.{secrets.token_hex(8)}.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _planner_rows(task: Any) -> list[Mapping[str, Any]]:
    rows = [row for row in task.ledger.snapshot()["tasks"] if row.get("task_id") == task.task_id]
    if len(rows) != 1:
        raise ValueError("retry_journal_ledger_invalid")
    return [call for call in rows[0]["calls"] if call.get("role") == "planner"]


def _planner_calls(task: Any) -> int:
    return len(_planner_rows(task))


def _verify_pending_guard(task: Any, call: Mapping[str, Any], request_sha: str) -> None:
    """The unknown reservation must be the journal's exact guarded request."""
    from easyicu.runtime.host_execution_lease import assert_host_execution_quiescent, read_host_execution_receipt
    from easyicu.webserver import metadata_provider_concurrency as slots
    from easyicu.webserver.metadata_provider_worker import identity as worker_identity

    attempt_id = str(call.get("attempt_id") or "")
    held = slots.get(attempt_id)
    expected = Path(task.ledger.path).parent / ".metadata-provider-execution" / attempt_id
    if (not isinstance(held, dict) or call.get("state") != "in_progress"
            or held.get("task_id") != task.task_id or held.get("ledger_path") != str(task.ledger.path)
            or held.get("guard_path") != str(expected) or held.get("request_sha256") != request_sha
            or held.get("worker_sha256") != worker_identity() or not held.get("guard_nonce")):
        raise ValueError("retry_journal_guard_unverified")
    assert_host_execution_quiescent(expected, allow_missing=False, wait_timeout=15.0)
    receipt = read_host_execution_receipt(expected)
    if (receipt is None or receipt.get("nonce") != held["guard_nonce"]
            or receipt.get("script_sha256") != held["worker_sha256"]
            or not slots.has_capacity(held["account"])):
        raise ValueError("retry_journal_guard_unverified")


class OutlineRetryJournal:
    """Reconstruct retry feedback through the original parser, without recharging replay."""

    def __init__(self, *, inner: Any, task: Any, run_dir: Path, request_sha256: str,
                 dependency_sha256: str, literature_sha256: str, source_path: Path | None = None,
                 source_sha256: str | None = None, stage: str = "outline",
                 parse_attempt_limit: int = 4) -> None:
        self._inner, self._task = inner, task
        self._path = Path(run_dir) / ".runtime" / FILE_NAME
        self._index = 0
        self.last_finish_reason: str | None = None
        self.last_usage: Any = None
        self.last_transport_attempts = 0
        if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
               for value in (request_sha256, dependency_sha256, literature_sha256)):
            raise ValueError("retry_journal_authority_invalid")
        if stage not in {"outline", "family_spec", "foundation"} or type(parse_attempt_limit) is not int or not 1 <= parse_attempt_limit < MAX_CALLS:
            raise ValueError("retry_journal_stage_invalid")
        self._parse_attempt_limit = parse_attempt_limit
        binding = {"request_sha256": request_sha256, "dependency_sha256": dependency_sha256,
                   "literature_sha256": literature_sha256,
                   "stage": stage, "parse_attempt_limit": parse_attempt_limit,
                   "code_sha256": _code_digest(), "task_id": task.task_id,
                   "ledger_path": str(Path(task.ledger.path).resolve())}
        if source_path is not None:
            if source_sha256 is None:
                raise ValueError("retry_journal_source_digest_missing")
            value = load_retry_journal(source_path, expected_sha256=source_sha256)
            if value.get("binding") != binding:
                raise ValueError("retry_journal_authority_mismatch")
            self._data = value
            pending = self._data.get("pending")
            current = _planner_calls(task)
            if pending is not None:
                if (not isinstance(pending, dict) or not isinstance(pending.get("planner_calls"), int)
                        or not isinstance(pending.get("request_sha256"), str)):
                    raise ValueError("retry_journal_pending_invalid")
                if current < pending["planner_calls"]:
                    raise ValueError("retry_journal_ledger_changed")
                unrecorded = current - pending["planner_calls"]
                if unrecorded:
                    # Multiple or already-terminal calls cannot be attributed
                    # to this pending response without inventing evidence.
                    if unrecorded != 1:
                        raise ValueError("retry_journal_ledger_changed")
                    _verify_pending_guard(task, _planner_rows(task)[-1], pending["request_sha256"])
                    self._data["abandoned"] += 1
                self._data["expected_retry_request_sha256"] = pending["request_sha256"]
                self._data["expected_retry_guarded"] = bool(unrecorded)
                self._data["pending"] = None
                self._data["accounted_planner_calls"] = current
            else:
                records = self._data["records"]
                last_count = self._data.get("accounted_planner_calls")
                if last_count is None:
                    last_count = (records[-1].get("planner_calls_after") if records
                                  else self._data.get("initial_planner_calls"))
                if not isinstance(last_count, int) or current != last_count:
                    raise ValueError("retry_journal_ledger_changed")
                expected = self._data.get("expected_retry_request_sha256")
                if expected is not None and self._data.get("expected_retry_guarded"):
                    rows = _planner_rows(task)
                    if not rows:
                        raise ValueError("retry_journal_ledger_changed")
                    _verify_pending_guard(task, rows[-1], expected)
        else:
            self._data = {"schema_version": SCHEMA, "binding": binding, "records": [],
                          "pending": None, "abandoned": 0,
                          "initial_planner_calls": _planner_calls(task),
                          "expected_retry_request_sha256": None}
        if not 0 <= self._data["abandoned"] <= MAX_CALLS:
            raise ValueError("retry_journal_attempt_limit")
        _atomic_write(self._path, self._data)
        _register_provider_wrapper(self, children_getter=lambda: (self._inner,))

    @property
    def supports_strict_json_schema(self) -> bool:
        from easyicu.research_agent.providers.capabilities import llm_supports_strict_json_schema
        return llm_supports_strict_json_schema(self._inner)

    def complete(self, messages: Sequence[Any], **kwargs: Any) -> str:
        schema = kwargs.get("structured_output")
        request = {"messages": [{"role": item.role, "content": item.content} for item in messages],
                   "max_tokens": kwargs.get("max_tokens"), "temperature": kwargs.get("temperature"),
                   "schema": getattr(schema, "canonical_payload_json", None)}
        request_sha = canonical_sha256(request)
        records = self._data["records"]
        if self._index < len(records):
            record = records[self._index]
            if not isinstance(record, dict) or record.get("request_sha256") != request_sha or not isinstance(record.get("response"), str):
                raise ValueError("retry_journal_request_mismatch")
            self._index += 1
            self.last_finish_reason = record.get("finish_reason")
            self.last_usage = record.get("usage")
            self.last_transport_attempts = record.get("transport_attempts", 1)
            return record["response"]
        # The fifth call is normally reserved for a proven truncated response.
        # An interrupted call cannot prove that condition, so after any lost
        # attempt use the ordinary four-call limit rather than widening it.
        call_limit = self._parse_attempt_limit + (0 if self._data["abandoned"] else 1)
        if self._index != len(records) or len(records) + self._data["abandoned"] >= call_limit:
            raise ValueError("retry_journal_attempt_limit")
        expected = self._data.get("expected_retry_request_sha256")
        if expected is not None and expected != request_sha:
            raise ValueError("retry_journal_request_mismatch")
        self._data["expected_retry_request_sha256"] = None
        self._data.pop("expected_retry_guarded", None)
        self._data["pending"] = {"request_sha256": request_sha, "planner_calls": _planner_calls(self._task)}
        _atomic_write(self._path, self._data)
        from easyicu.research_agent.providers.metadata_request_binding import metadata_request_binding

        with metadata_request_binding(request_sha):
            result = authorized_complete(self._inner, messages, **kwargs)
        if not isinstance(result, str) or len(result.encode()) > MAX_RESPONSE_BYTES:
            raise ValueError("retry_journal_response_invalid")
        receipt = current_provider_call_receipt()
        self.last_finish_reason = safe_provider_finish_reason(
            receipt.finish_reason if receipt is not None else getattr(self._inner, "last_finish_reason", None)
        )
        usage = receipt.usage_summary if receipt is not None else getattr(self._inner, "last_usage", None)
        self.last_usage = ({key: value for key, value in usage.items()
                            if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
                            and type(value) is int and value >= 0}
                           if isinstance(usage, Mapping) else None)
        self.last_transport_attempts = receipt.transport_attempts if receipt is not None else 1
        self._data["records"].append({"request_sha256": request_sha, "response": result,
                                      "finish_reason": self.last_finish_reason, "usage": self.last_usage,
                                      "transport_attempts": self.last_transport_attempts,
                                      "planner_calls_after": _planner_calls(self._task)})
        self._data.pop("accounted_planner_calls", None)
        self._data["pending"] = None
        _atomic_write(self._path, self._data)
        self._index += 1
        return result
