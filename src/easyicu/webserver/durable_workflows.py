"""Local DBOS execution of opaque, host-owned recovery descriptors.

The callback must revalidate the descriptor's authority, original budget and
checkpoint on every invocation. A process crash may repeat an uncheckpointed
step: DBOS does not make external scientific effects exactly-once. No callback,
provider configuration, research arguments or result payload enters this DB.
The Web deployment lease must be held throughout this dispatcher's lifetime.
"""

from __future__ import annotations

import os
import re
import sqlite3
import stat
import threading
import time
from pathlib import Path
from typing import Any, Callable

APPLICATION_VERSION = "easyicu-durable-descriptors-v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_RECEIPT_STATES = {"done", "failed", "cancelled", "blocked"}
_owner_lock = threading.RLock()
_owner: DurableWorkflowDispatcher | None = None
_workflow: Any = None


class DurableWorkflowError(RuntimeError):
    """A stable, non-sensitive local durable-execution failure code."""


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise DurableWorkflowError("durable_descriptor_invalid")
    return value


def _receipt(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or not isinstance(value.get("status"), str):
        return {"status": "failed", "reason_code": "durable_receipt_invalid"}
    if value["status"] not in _RECEIPT_STATES:
        return {"status": "failed", "reason_code": "durable_receipt_invalid"}
    result = {"status": value["status"]}
    for key in ("reason_code", "job_id"):
        if key in value:
            try:
                result[key] = _identifier(value[key])
            except DurableWorkflowError:
                return {"status": "failed", "reason_code": "durable_receipt_invalid"}
    return result


def _execute_descriptor(descriptor_id: str) -> dict[str, str]:
    with _owner_lock:
        owner = _owner
    if owner is None:
        return {"status": "blocked", "reason_code": "durable_owner_unavailable"}
    return owner._execute(_identifier(descriptor_id))


def _register_workflow() -> Any:
    # The SDK is an optional webapp dependency. Importing core packages and
    # inspecting app configuration must not initialize a DBOS executor.
    global _workflow
    from dbos import DBOS, WorkflowSerializationFormat

    if _workflow is None:
        execute = DBOS.step(
            name="easyicu.execute_recovery_descriptor.v1", retries_allowed=False,
        )(_execute_descriptor)

        @DBOS.workflow(
            name="easyicu.recovery_descriptor.v1", max_recovery_attempts=3,
            serialization_type=WorkflowSerializationFormat.PORTABLE,
        )
        def workflow(descriptor_id: str) -> dict[str, str]:
            return execute(descriptor_id)

        _workflow = workflow
    return _workflow


def _private_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode) or path.is_symlink():
            raise DurableWorkflowError("durable_database_invalid")
        if os.name != "nt":
            os.fchmod(fd, 0o600)
    finally:
        os.close(fd)
    # DBOS supports pickle when a stored row explicitly requests it, even with
    # a JSON default. Refuse such a database before startup recovery deserializes
    # anything. These are all DBOS 3.3 tables with serializer-bearing payloads.
    with sqlite3.connect(path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in (
            "workflow_status", "operation_outputs", "notifications",
            "workflow_events", "workflow_events_history", "streams",
        ):
            if table in tables:
                row = conn.execute(
                    f'SELECT 1 FROM "{table}" WHERE serialization IS NULL OR serialization != ? LIMIT 1',
                    ("portable_json",),
                ).fetchone()
                if row:
                    raise DurableWorkflowError("durable_database_serialization_invalid")


class DurableWorkflowDispatcher:
    """One local executor, with stable IDs and bounded crash recovery.

    ``close`` returns False if work/SDK shutdown has not drained by the deadline;
    the caller must retain the process lease in that case. It does not kill a
    synchronous host callback. Explicit cancellation remains host-owned and
    must be durably recorded before calling ``cancel``.
    """

    def __init__(
        self, database_path: Path, execute_descriptor: Callable[[str], dict[str, Any]],
        application_version: str = APPLICATION_VERSION,
    ) -> None:
        self.database_path = Path(database_path).absolute()
        self.application_version = _identifier(application_version)
        self._callback = execute_descriptor
        self._condition = threading.Condition(threading.RLock())
        self._active = 0
        self._launched = False
        self._closing = False
        self._closed = threading.Event()
        self._close_started = False
        self._shutdown_failed = False

    @staticmethod
    def workflow_id(descriptor_id: str) -> str:
        return "easyicu-recovery:" + _identifier(descriptor_id)

    def launch(self) -> DurableWorkflowDispatcher:
        global _owner
        with _owner_lock:
            if _owner is self and self._launched and not self._closing:
                return self
            if _owner is not None or self._closing:
                raise DurableWorkflowError("durable_executor_already_owned")
            # Cloud mode can override the explicit local DB/config in the SDK.
            if os.environ.get("DBOS__CLOUD") == "true":
                raise DurableWorkflowError("durable_cloud_mode_not_allowed")
            try:
                from dbos import DBOS, DBOSPortableJSONSerializer
                from sqlalchemy.engine import URL
            except ImportError:
                raise DurableWorkflowError("durable_dependency_unavailable") from None
            _private_database(self.database_path)
            _register_workflow()
            DBOS(config={
                "name": "easyicu-local-recovery",
                "application_version": self.application_version,
                "executor_id": "easyicu-local-recovery",
                "system_database_url": URL.create("sqlite", database=str(self.database_path)).render_as_string(),
                "serializer": DBOSPortableJSONSerializer(),
                "enable_otlp": False,
                "use_listen_notify": False,
                "max_executor_threads": 4,
                "notification_listener_polling_interval_sec": 0.1,
                "scheduler_polling_interval_sec": 0.1,
                "log_level": "WARNING",
            })
            # DBOS 3.3 has no admin HTTP server. No Conductor/OTLP endpoint or
            # key is configured; only the explicit private SQLite URL is used.
            _owner = self
            try:
                DBOS.launch()
                self._launched = True
            except Exception:
                self.close()
                raise DurableWorkflowError("durable_launch_failed") from None
        return self

    def _execute(self, descriptor_id: str) -> dict[str, str]:
        with self._condition:
            if self._closing:
                return {"status": "cancelled", "reason_code": "durable_executor_closing"}
            self._active += 1
        try:
            try:
                return _receipt(self._callback(descriptor_id))
            except Exception:
                # Do not let DBOS persist exceptions containing credentials,
                # raw provider messages, scientific inputs or patient data.
                return {"status": "failed", "reason_code": "durable_callback_failed"}
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def _require_running(self) -> None:
        if not self._launched or self._closing:
            raise DurableWorkflowError("durable_executor_unavailable")

    def submit(self, descriptor_id: str) -> str:
        from dbos import DBOS, SetWorkflowID

        workflow_id = self.workflow_id(descriptor_id)
        with self._condition:
            self._require_running()
            with SetWorkflowID(workflow_id):
                DBOS.start_workflow(_workflow, descriptor_id)
        return workflow_id

    def status(self, descriptor_id: str) -> dict[str, Any] | None:
        from dbos import DBOS

        with self._condition:
            self._require_running()
            row = DBOS.get_workflow_status(self.workflow_id(descriptor_id))
        if row is None:
            return None
        return {
            "workflow_id": row.workflow_id, "status": row.status,
            "receipt": _receipt(row.output) if row.output is not None else None,
        }

    def cancel(self, descriptor_id: str) -> None:
        from dbos import DBOS

        with self._condition:
            self._require_running()
            DBOS.cancel_workflow(self.workflow_id(descriptor_id))

    def close(self, timeout: float = 4.0) -> bool:
        with _owner_lock:
            with self._condition:
                self._closing = True
                if not self._close_started:
                    self._close_started = True
                    if _owner is not self:
                        self._closed.set()
                    else:
                        threading.Thread(target=self._drain, name="easyicu-durable-shutdown", daemon=True).start()
        return self._closed.wait(max(0.0, min(timeout, 4.0))) and not self._shutdown_failed

    def _drain(self) -> None:
        global _owner
        try:
            with self._condition:
                self._condition.wait_for(lambda: self._active == 0)
            from dbos import DBOS

            # A callback returning is earlier than DBOS committing its step and
            # workflow receipt. Drain pending dispatch/commit work too, before
            # the SDK's non-waiting ThreadPoolExecutor shutdown. New callbacks
            # see _closing and cannot invoke the host owner.
            if self._launched:
                while DBOS.list_workflows(
                    status=["PENDING", "ENQUEUED", "DELAYED"],
                    name="easyicu.recovery_descriptor.v1",
                    app_version=self.application_version, limit=1,
                    load_input=False, load_output=False,
                ):
                    time.sleep(0.02)
            DBOS.destroy(workflow_completion_timeout_sec=0)
            with _owner_lock:
                if _owner is self:
                    _owner = None
            self._launched = False
        except Exception:
            self._shutdown_failed = True
        finally:
            self._closed.set()
