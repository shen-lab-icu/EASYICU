"""Connect durable scheduling to the existing research and JobManager owners.

Only newly admitted commands participate. Existing studies and chat messages
are never scanned for work to launch. Stored commands are references, not
credentials, approval decisions, scientific results, or serialized callables.
"""

from __future__ import annotations

import threading
import logging
import os
import re
import stat
from pathlib import Path
from typing import Any, Callable

from easyicu.webserver import jobs, study_contexts
from easyicu.webserver.durable_research_store import (
    DurableResearchError, ResearchCommandStore, runtime_digest,
)

_SERVICE: DurableResearchService | None = None
_LOG = logging.getLogger(__name__)

_PREPARATION_SCHEMA = "easyicu.durable-preparation/1"
_SEALED_PREPARATION_SCHEMA = "easyicu.durable-preparation/2"
_HANDOFF_PREPARATION_SCHEMA = "easyicu.durable-preparation/3"


def _command_kind(payload: dict[str, Any]) -> str:
    return "raw-preparation-review" if payload.get("schema_version") in {
        _PREPARATION_SCHEMA, _SEALED_PREPARATION_SCHEMA, _HANDOFF_PREPARATION_SCHEMA,
    } else "agent-run"


def _preparation_result(path: Path) -> dict[str, Any]:
    """Project completion without exposing the private coordinate or a grant."""
    status = ("first_observed_coordinate_complete" if Path(path).parent.name == "private-first-observed"
              else "multi_table_projection_complete")
    return {"preparation_status": status, "cohort_complete": False, "analysis_authorized": False}


class DurableResearchService:
    def __init__(self, root: Path, *, manager: jobs.JobManager, dispatcher_factory=None,
                 recover_runner=None) -> None:
        from easyicu.webserver.durable_workflows import DurableWorkflowDispatcher
        from easyicu.webserver.research_auto_resume import recover_runner as scientific_recover_runner

        self.manager = manager
        self.store = ResearchCommandStore(root / "research-commands.sqlite3")
        self.recover_runner = recover_runner or scientific_recover_runner
        self._live: dict[str, Callable] = {}
        self._lock = threading.Lock()
        self._admission_lock = threading.Lock()
        self._execution_locks: dict[str, threading.Lock] = {}
        self._stopping = threading.Event()
        factory = dispatcher_factory or DurableWorkflowDispatcher
        self.dispatcher = factory(root / "research-workflows.sqlite3", self.execute,
                                  application_version="easyicu-research-durable-v1")

    def launch(self) -> None:
        self._recover_handoffs()
        # Reattach transport synchronously: no browser can consume a synthetic
        # interrupted terminal event for a command about to resume.
        active = self.store.active_ids()
        for identity in active:
            prior = self.manager.get(identity)
            if prior is None:
                self.store.finish(identity, "research_command_job_receipt_missing")
                continue
            if self.store.get(identity)["payload"].get("_cancel_requested"):
                if prior.snapshot().get("interrupted") and not prior.is_cancel_requested():
                    prior = self.manager.reserve(_command_kind(self.store.get(identity)["payload"]), job_id=identity, resume_interrupted=True)
                prior.request_cancel("durable_cancel_receipt")
                prior.complete_from_runner()
                self.store.finish(identity, "cancelled")
                self._clear_binding(identity)
                continue
            state = prior.snapshot()
            if state["cancel_requested"] or (state["status"] != "running" and not state.get("interrupted")):
                self.store.finish(identity, "cancelled" if state["cancel_requested"] else state["status"])
                if state["cancel_requested"] and state["status"] == "running":
                    prior.complete_from_runner()
                self._clear_binding(identity)
                continue
            if prior is not None and prior.snapshot().get("interrupted") and not prior.is_cancel_requested():
                self.manager.reserve(_command_kind(self.store.get(identity)["payload"]), job_id=identity, resume_interrupted=True)
            self.manager.get(identity).register_cancel_callback(lambda identity=identity: self.store.request_cancel(identity))
        self.dispatcher.launch()
        # Transactional outbox: a crash between activation and DBOS submission
        # cannot strand an authorized command. Workflow IDs deduplicate enqueue.
        for identity in self.store.active_ids():
            self._dispatch(identity)

    def reserve(self, payload: dict[str, Any]) -> jobs.Job:
        job = self.manager.reserve("agent-run")
        try:
            self.store.prepare(job.id, payload)
            job.register_cancel_callback(lambda: self.store.request_cancel(job.id))
        except BaseException:
            job.fail_from_runner("research_command_persistence_failed")
            raise
        return job

    def activate_preparation(self, job_id: str, *, checkpoint: Any, binding: Any, scope: dict[str, Any],
                             timeout: float, sealed_recovery: bool = False) -> None:
        """Register a host-approved preparation with the existing DBOS owner.

        Admission must be called after the conversation host action and
        decided checkpoint exist. The private intent is never an approval.
        A crash before command activation remains unstarted, not auto-approved.
        """
        from easyicu.webserver.raw_preparation_durable import (
            admit_preparation_intent, recover_preparation_before_consumption, recover_preparation_with_sealed_snapshot,
        )
        job = self.manager.get(job_id)
        import math
        if (job is None or job.kind != "raw-preparation-review" or job.is_cancel_requested()
                or type(sealed_recovery) is not bool
                or not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
                or not math.isfinite(timeout) or timeout <= 0 or timeout > 86400):
            raise DurableResearchError("preparation_command_job_invalid")
        admit_preparation_intent(job_id=job_id, checkpoint=checkpoint, binding=binding, scope=scope)
        payload = {"schema_version": _SEALED_PREPARATION_SCHEMA if sealed_recovery else _PREPARATION_SCHEMA,
                   "study_context_id": binding.study_context_id,
                   "timeout_seconds": timeout}
        self.store.prepare(job_id, payload)

        def run(bound_job: jobs.Job) -> dict[str, Any]:
            recover = recover_preparation_with_sealed_snapshot if sealed_recovery else recover_preparation_before_consumption
            output = recover(job_id=job_id, job=bound_job, timeout=timeout)
            return _preparation_result(output)

        self.activate(job_id, run)

    @staticmethod
    def _require_handoff_deployment_lease() -> None:
        """Require this process to own the selected Web deployment lock."""
        from easyicu.webserver import deployment_lease
        handle = deployment_lease._HANDLE
        path = deployment_lease.state_paths.state_root() / 'webserver-single-process.lock'
        try:
            if handle is None or handle.closed or path.is_symlink():
                raise ValueError
            opened, selected = os.fstat(handle.fileno()), path.stat()
            if (not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (selected.st_dev, selected.st_ino)
                    or os.pread(handle.fileno(), 64, 0).decode().strip() != str(os.getpid())):
                raise ValueError
        except (OSError, ValueError, UnicodeError) as exc:
            raise DurableResearchError('preparation_handoff_deployment_lease_missing') from exc

    def _recover_handoffs(self) -> None:
        """Adopt only exact /3 handoffs with an intact unstarted lifecycle."""
        from easyicu.research_agent.authority.filesystem import AnchoredDirectory
        from easyicu.webserver import state_paths
        private = state_paths.state_root() / 'private-preparation-decisions'
        if not private.exists() and not private.is_symlink():
            return
        with AnchoredDirectory.open(private) as directory:
            names = sorted(name for name in os.listdir(private)
                           if re.fullmatch(r'[A-Za-z0-9_-]{1,128}\.preparation-handoff\.json', name))
            directory.assert_still_selected()
        known = self.store.known_ids()
        names = [name for name in names if name.removesuffix('.preparation-handoff.json') not in known]
        if not names:
            return
        self._require_handoff_deployment_lease()
        if self.manager._journal is None or len(names) > 1000:
            raise DurableResearchError('preparation_handoff_inventory_invalid')
        with self._admission_lock:
            for name in names:
                identity = name.removesuffix('.preparation-handoff.json')
                if self.store.get(identity) is not None:
                    continue
                try:
                    self._adopt_unstarted_handoff(identity)
                except Exception:
                    # Never manufacture a command from incomplete or stale
                    # authority. The original job/receipt remains visible.
                    _LOG.warning('Preparation handoff could not be safely recovered')

    def _adopt_unstarted_handoff(self, job_id: str) -> None:
        from easyicu.webserver.raw_preparation_durable import verify_unstarted_preparation_handoff
        self._require_handoff_deployment_lease()
        if self.manager._journal is None:
            raise DurableResearchError('preparation_handoff_journal_missing')
        job = self.manager.get(job_id)
        payload = verify_unstarted_preparation_handoff(job_id=job_id, job=job, store_path=self.store.path.absolute(),
                                                       journal_path=self.manager._journal.path.absolute())
        if job.is_cancel_requested() or self.store.get(job_id) is not None:
            raise DurableResearchError('preparation_handoff_state_changed')
        self._require_handoff_deployment_lease()
        self.store.prepare_active_handoff(job_id, payload)

    def activate_preparation_handoff(self, job_id: str, *, checkpoint: Any, binding: Any,
                                     scope: dict[str, Any], timeout: float) -> None:
        """Durably accept a /3 handoff, then create its outbox atomically.

        This owner is not exposed by the product route. An accepted handoff
        remains recoverable if the process dies before outbox creation.
        """
        from easyicu.webserver.raw_preparation_durable import admit_preparation_handoff
        self._require_handoff_deployment_lease()
        if self.manager._journal is None or not self.dispatcher._launched:
            raise DurableResearchError('preparation_handoff_service_unavailable')
        with self._admission_lock:
            job = self.manager.get(job_id)
            if (job is None or job.kind != 'raw-preparation-review' or job.status != 'running'
                    or job.is_cancel_requested() or self.store.get(job_id) is not None):
                raise DurableResearchError('preparation_handoff_job_invalid')
            admit_preparation_handoff(job_id=job_id, checkpoint=checkpoint, binding=binding,
                                       scope=scope, timeout=timeout, store_path=self.store.path.absolute(),
                                       journal_path=self.manager._journal.path.absolute(), job_created=job.created)
            self._adopt_unstarted_handoff(job_id)
            job.register_cancel_callback(lambda: self.store.request_cancel(job_id))
        try:
            self._dispatch(job_id)
        except Exception:
            _LOG.warning('Durable research dispatch pending for job %s', job_id)
            threading.Thread(target=self._retry_dispatch, args=(job_id,), daemon=True).start()

    def recover_approved_preparation_handoff(self, job_id: str, *, checkpoint: Any, binding: Any,
                                             scope: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Reattach the exact approved but unstarted first-observed job.

        This internal bridge does not offer a review action. The caller must
        already hold the original signed host decision; it never creates a
        new job or resets a command's attempt ledger.
        """
        from easyicu.webserver.raw_preparation_durable import recover_approved_unstarted_handoff
        self._require_handoff_deployment_lease()
        if self.manager._journal is None or not self.dispatcher._launched:
            raise DurableResearchError('preparation_handoff_service_unavailable')
        with self._admission_lock:
            job = self.manager.get(job_id)
            if self.store.get(job_id) is not None:
                # The existing outbox owns every later transition. Do not reset
                # or dispatch it again through this admission path.
                raise DurableResearchError('preparation_handoff_already_admitted')
            payload = recover_approved_unstarted_handoff(job_id=job_id, job=job, checkpoint=checkpoint,
                binding=binding, scope=scope, timeout=timeout, store_path=self.store.path.absolute(),
                journal_path=self.manager._journal.path.absolute())
            if self.store.get(job_id) is not None or job.is_cancel_requested():
                raise DurableResearchError('preparation_handoff_state_changed')
            self._require_handoff_deployment_lease()
            if job.snapshot().get('interrupted'):
                job = self.manager.reserve('raw-preparation-review', job_id=job_id, resume_interrupted=True)
            # The signed handoff already exists. If reopening capacity or the
            # outbox write fails, a later adoption can still find that exact
            # unstarted handoff instead of stranding an active command.
            self.store.prepare_active_handoff(job_id, payload)
            job.register_cancel_callback(lambda: self.store.request_cancel(job_id))
        try:
            self._dispatch(job_id)
        except Exception:
            _LOG.warning('Durable research dispatch pending for job %s', job_id)
            threading.Thread(target=self._retry_dispatch, args=(job_id,), daemon=True).start()
        return {'job_id': job_id, 'preparation_status': 'preparing', 'analysis_authorized': False}

    def activate(self, job_id: str, runner: Callable) -> None:
        with self._lock:
            self._live[job_id] = runner
        try:
            self.store.activate(job_id)
        except BaseException:
            with self._lock:
                self._live.pop(job_id, None)
            raise
        # The active outbox is acceptance. An ambiguous enqueue response cannot
        # revoke it or publish a terminal job while its runner may be executing.
        try:
            self._dispatch(job_id)
        except Exception:
            _LOG.warning("Durable research dispatch pending for job %s", job_id)
            threading.Thread(target=self._retry_dispatch, args=(job_id,), daemon=True).start()

    def _retry_dispatch(self, job_id: str) -> None:
        while not self._stopping.wait(5.0):
            try:
                command = self.store.get(job_id)
                if command is None or command["status"] != "active":
                    return
                self._dispatch(job_id)
                return
            except Exception:
                continue  # Re-enqueue only; never retry the scientific runner.

    def _dispatch(self, job_id: str) -> None:
        self.dispatcher.submit(job_id)
        state = self.dispatcher.status(job_id)
        if state and state["status"] in {"SUCCESS", "ERROR", "CANCELLED", "MAX_RECOVERY_ATTEMPTS_EXCEEDED"}:
            command = self.store.get(job_id)
            if command is not None and command["status"] == "active":
                # A completed DBOS step will not run again, including when its
                # final outbox write failed. Expose that fact instead of hanging.
                job = self.manager.get(job_id)
                if job is not None and job.snapshot()["status"] == "running":
                    job.fail_from_runner("research_auto_resume_durable_terminal")
                self.store.finish(job_id, "research_auto_resume_durable_terminal")
                self._clear_binding(job_id)

    def abandon(self, job_id: str) -> None:
        self.store.finish(job_id, "research_command_not_authorized")
        job = self.manager.get(job_id)
        if job is not None:
            job.fail_from_runner("research_command_not_authorized")
        self._clear_binding(job_id)

    def _clear_binding(self, job_id: str) -> None:
        command = self.store.get(job_id)
        job = self.manager.get(job_id)
        if command is None or job is None or job.snapshot()["status"] == "running":
            return
        if _command_kind(command["payload"]) != "agent-run":
            return
        state = job.snapshot()
        stage = ("agent_cancelled" if state["status"] == "cancelled" else
                 "agent_failed" if state["status"] == "failed" else
                 "review_blocked" if isinstance(state["result"], dict) and (state["result"].get("gate") or {}).get("status") == "blocked"
                 else "review")
        try:
            study_contexts.clear_active_job_if(command["payload"]["study_context_id"], job_id,
                                              current_stage=stage, last_route="agent")
        except Exception:
            pass

    def execute(self, job_id: str) -> dict[str, str]:
        with self._lock:
            claim = self._execution_locks.setdefault(job_id, threading.Lock())
        with claim:
            try:
                return self._execute_owned(job_id)
            except Exception:
                # DBOS persists only a safe receipt. Settle host state too so a
                # completed scheduler step cannot strand a running Web job.
                job = self.manager.get(job_id)
                if job is not None:
                    job.fail_from_runner("research_auto_resume_dispatch_failed")
                self.store.finish(job_id, "research_auto_resume_dispatch_failed")
                return {"status": "failed", "reason_code": "research_auto_resume_dispatch_failed"}
            finally:
                self._clear_binding(job_id)

    def _execute_owned(self, job_id: str) -> dict[str, str]:
        command = self.store.get(job_id)
        if command is None or command["status"] != "active":
            return {"status": "blocked", "reason_code": "research_command_not_active"}
        prior = self.manager.get(job_id)
        if prior is None:
            self.store.finish(job_id, "research_command_job_receipt_missing")
            return {"status": "blocked", "reason_code": "research_command_job_receipt_missing"}
        snapshot = prior.snapshot()
        if command["payload"].get("_cancel_requested") and not snapshot["cancel_requested"]:
            prior.request_cancel("durable_cancel_receipt")
            snapshot = prior.snapshot()
        if snapshot["cancel_requested"] or (snapshot["status"] != "running" and not snapshot.get("interrupted")):
            reason = "cancelled" if snapshot["cancel_requested"] else snapshot["status"]
            if snapshot["cancel_requested"] and snapshot["status"] == "running":
                prior.complete_from_runner()
            self.store.finish(job_id, reason)
            return {"status": reason, "job_id": job_id}

        with self._lock:
            initial_runner = self._live.pop(job_id, None)
        job = self.manager.reserve(_command_kind(command["payload"]), job_id=job_id,
                                   resume_interrupted=bool(snapshot.get("interrupted")))
        payload = command["payload"]

        def governed_run(bound_job: jobs.Job):
            terminal_stage = "agent_failed"
            try:
                payload["_resume_attempt"] = self.store.begin_attempt(job_id)
                if bound_job.is_cancel_requested():
                    raise DurableResearchError("research_auto_resume_cancelled")
                if initial_runner is not None:
                    return initial_runner(bound_job)
                if command["code_sha256"] != runtime_digest():
                    raise DurableResearchError("research_auto_resume_runtime_changed")
                if _command_kind(payload) == "raw-preparation-review":
                    from easyicu.webserver.raw_preparation_durable import (
                        recover_preparation_before_consumption, recover_preparation_with_sealed_snapshot,
                        verify_preparation_handoff_command,
                    )
                    try:
                        if payload.get('schema_version') == _HANDOFF_PREPARATION_SCHEMA:
                            self._require_handoff_deployment_lease()
                            verify_preparation_handoff_command(job_id=job_id, job=bound_job, payload=payload,
                                store_path=self.store.path.absolute(), journal_path=self.manager._journal.path.absolute())
                        recover = (recover_preparation_with_sealed_snapshot
                                   if payload.get("schema_version") in {_SEALED_PREPARATION_SCHEMA, _HANDOFF_PREPARATION_SCHEMA}
                                   else recover_preparation_before_consumption)
                        output = recover(job_id=job_id, job=bound_job, timeout=payload["timeout_seconds"])
                    except Exception as exc:
                        raise DurableResearchError("preparation_auto_resume_refused") from exc
                    return _preparation_result(output)
                runner = self.recover_runner(payload, bound_job)
                result = runner(bound_job)
                terminal_stage = ("agent_cancelled" if bound_job.is_cancel_requested()
                                  else "review_blocked" if (result.get("gate") or {}).get("status") == "blocked"
                                  else "review")
                return result
            finally:
                # The initial submission runner already performs this handoff.
                # Recovery must release the same study binding even if a guard
                # refuses it; it never clears another job's active binding.
                if initial_runner is None and _command_kind(payload) == "agent-run":
                    try:
                        study_contexts.clear_active_job_if(
                            payload["study_context_id"], job_id,
                            current_stage=terminal_stage, last_route="agent",
                        )
                    except Exception:
                        pass

        self.manager.run_reserved(job_id, governed_run)
        final = job.snapshot()
        status = str(final["status"])
        self.store.finish(job_id, status)
        return {"status": status, "job_id": job_id}

    def close(self) -> bool:
        self._stopping.set()
        return self.dispatcher.close(timeout=4.0)


def current_service() -> DurableResearchService | None:
    return _SERVICE


def bind_provider_task(job: Any, *, wrapper_dir: Path, task: Any) -> None:
    """Attach the paid-task identity without persisting clients or secrets."""
    service = current_service()
    if service is None:
        return
    recovery = getattr(job, "provider_recovery", None)
    identity = str(getattr(getattr(recovery, "job", job), "id", ""))
    ledger = task.ledger.snapshot()
    service.store.bind_provider(identity, {
        "wrapper_dir": str(Path(wrapper_dir).resolve()),
        "hard_stop_ledger_path": str(task.ledger.path.resolve()),
        "hard_stop_task_id": task.task_id,
        "hard_stop_declaration_sha256": ledger["declaration_sha256"],
    })


def start_service(root: Path) -> None:
    global _SERVICE
    if _SERVICE is not None:
        return
    service = DurableResearchService(root, manager=jobs.MANAGER)
    _SERVICE = service
    try:
        service.launch()
    except BaseException:
        if service.close():
            _SERVICE = None
        raise


def stop_service() -> bool:
    global _SERVICE
    service = _SERVICE
    if service is None:
        return True
    if not service.close():
        return False
    _SERVICE = None
    return True
