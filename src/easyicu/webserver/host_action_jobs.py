"""Owner of a host decision that starts a governed job.

Six Copilot buttons start a governed job: generate or revise a plan, prepare
its data, approve or reject its review, retry its execution.  The browser used
to start the job and record the click in the conversation afterwards, about 50
seconds later once the start checks passed: a page closed in between lost the
record, and a repeated click met a 409 or started a second job.

A submission from one of those buttons now carries the decision it answers
(``host_action``), and the host owns the rest:

* one job per decision of one study, whichever conversation or tab asks; a
  repeat while it runs gets the same job (``reused``);
* while the start checks run, the decision is registered as starting, and a
  repeat is told so instead of starting a second check;
* a decision the study no longer offers is refused with its stale fields and
  the job that answered it, instead of being run again;
* the conversation row is written in the request that starts the job; a write
  that fails is backfilled from the job's tag when the conversation is read.

The submission owners keep every check they make; a decision is reconciliation
input, never authority.  A submission without ``host_action`` (a cached page, a
script) keeps the study-revision CAS and nothing here.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
import hmac
import json
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Tuple

from pydantic import ValidationError

from easyicu.webserver import jobs
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver.host_action_contracts import (
    ACTION_DECISION_FAMILIES,
    HostActionRequest,
    HostActionTag,
    HostDecisionOffers,
    StartingEntry,
    decision_key,
    host_action_id,
    mismatched_fields,
)
from easyicu.webserver.host_action_starting import StudyStarting, study_lock
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.service import get_pi_copilot_service
from easyicu.webserver.pi_copilot.workflow import build_project_workflow_projection

logger = logging.getLogger(__name__)

#: The job each recent decision started, so a repeat from a stale page can be
#: told what became of it.  Bounded; a restart forgets it with the jobs.
_STARTED: "OrderedDict[Tuple[str, str], str]" = OrderedDict()
_STARTED_LIMIT = 256
_STARTED_LOCK = threading.Lock()
# A raw-preparation job cannot be forgotten while its conversation record is
# being written. Entries that fail to record stay pinned and fail closed.
_RAW_PREPARATION_PINS: Dict[Tuple[str, str], str] = {}


class HostActionJobError(RuntimeError):
    """A refused host decision, with the HTTP status its adapter returns."""

    def __init__(self, detail: Mapping[str, Any], *, status_code: int) -> None:
        self.detail = dict(detail)
        self.status_code = int(status_code)
        self.code = str(self.detail.get("error") or "host_action_refused")
        super().__init__(self.code)


class HostActionConversations(Protocol):
    """The conversation owner's part (``PiCopilotService``)."""

    def host_action_session(
        self, session_id: str, *, project_id: str, study_context_id: str
    ) -> None: ...

    def record_job_host_action(
        self,
        session_id: str,
        *,
        project_id: str,
        action_code: str,
        action_key: str,
        child_job_id: str,
    ) -> Mapping[str, Any]: ...

    def host_action_child_job(
        self, session_id: str, *, project_id: str, action_key: str
    ) -> Optional[Mapping[str, Any]]: ...


def _refuse(code: str, *, status_code: int, **details: Any) -> HostActionJobError:
    return HostActionJobError(
        {"error": code, **({"details": details} if details else {})},
        status_code=status_code,
    )


def parse_host_action(raw: Any) -> Optional[HostActionRequest]:
    """Validate a submission's ``host_action``; ``None`` when it carries none."""

    if raw is None:
        return None
    try:
        return HostActionRequest.model_validate(raw)
    except ValidationError as exc:
        fields = sorted(
            {".".join(str(part) for part in error["loc"]) for error in exc.errors()}
        )
        raise _refuse(
            "host_action_invalid", status_code=400, fields=fields[:16]
        ) from exc


def check_agent_run_request(
    host_action: HostActionRequest, body: Mapping[str, Any]
) -> None:
    """Refuse an ``agent-run`` submission whose shape answers another decision."""

    family = ACTION_DECISION_FAMILIES[host_action.action_code]
    engine = str(body.get("engine") or "").strip().lower()
    resume = str(body.get("execution_resume_source_run_id") or "").strip()
    decision = host_action.decision
    if engine != "research_agent_pipeline" or family not in {"plan_transition", "execution_retry"}:
        raise _refuse("host_action_request_mismatch", status_code=400, field="engine")
    if family == "execution_retry" and resume != getattr(
        decision, "source_run_id", None
    ):
        raise _refuse(
            "host_action_request_mismatch",
            status_code=400,
            field="execution_resume_source_run_id",
        )
    if family == "plan_transition" and resume:
        raise _refuse(
            "host_action_request_mismatch",
            status_code=400,
            field="execution_resume_source_run_id",
        )
    if host_action.action_code == "prepare_analysis_data":
        # This button accepts one displayed candidate for data preparation.
        # Never let a client pair that decision with another plan's roster.
        if (getattr(decision, "next_action_code", "") != "plan_execution_upgrade_required"
                or not getattr(decision, "source_run_id", "")
                or str(body.get("plan_revision_source_run_id") or "").strip()
                != decision.source_run_id):
            raise _refuse(
                "host_action_request_mismatch",
                status_code=400,
                field="plan_revision_source_run_id",
            )


def check_review_request(
    host_action: HostActionRequest, body: Mapping[str, Any]
) -> None:
    """Refuse an ``agent-run-review`` submission for another review."""

    decision = host_action.decision
    if decision.family != "plan_review" or str(body.get("run_id") or "").strip() != (
        decision.run_id
    ):
        raise _refuse("host_action_request_mismatch", status_code=400, field="run_id")


def offered_decisions(study_context_id: str) -> HostDecisionOffers:
    """The decisions the study offers now, as its workflow projection names them."""

    return build_project_workflow_projection(
        study_context_id=study_context_id,
    ).host_decisions


@dataclass(frozen=True)
class _Occupied:
    """What already answers the study when a decision arrives."""

    same_decision: bool
    entry: Optional[StartingEntry] = None
    job: Optional[jobs.Job] = None


def _running_study_job(study_context_id: str) -> Optional[jobs.Job]:
    try:
        study = context_store.get_context(study_context_id) or {}
    except context_store.StudyContextError:
        return None
    job_id = str(study.get("active_job_id") or "").strip()
    job = jobs.MANAGER.get(job_id) if job_id else None
    # The pointer outlives the process; the job manager does not.
    return job if job is not None and job.status == "running" else None


def _occupied(
    starting: StudyStarting, study_context_id: str, key: str, *, raw_opposite_key: str = ""
) -> Optional[_Occupied]:
    entry = starting.current()
    if entry is not None:
        return _Occupied(same_decision=entry.decision_key == key, entry=entry)
    job = _running_study_job(study_context_id)
    if job is not None:
        tag = job.host_action
        return _Occupied(same_decision=tag is not None and tag.decision_key == key, job=job)
    if raw_opposite_key:
        with _STARTED_LOCK:
            same = _RAW_PREPARATION_PINS.get((study_context_id, key)) or _STARTED.get((study_context_id, key), "")
            opposite = (_RAW_PREPARATION_PINS.get((study_context_id, raw_opposite_key))
                        or _STARTED.get((study_context_id, raw_opposite_key), ""))
        if same and opposite and same != opposite:
            raise _refuse("raw_preparation_decision_conflict", status_code=409)
        remembered = same or opposite
        if remembered:
            job = jobs.MANAGER.get(remembered)
            if job is None:
                raise _refuse("raw_preparation_original_job_unavailable", status_code=409, job_id=remembered)
            return _Occupied(same_decision=bool(same), job=job)
    return None


def _remember_started(study_context_id: str, key: str, job_id: str) -> None:
    with _STARTED_LOCK:
        _STARTED[(study_context_id, key)] = job_id
        _STARTED.move_to_end((study_context_id, key))
        while len(_STARTED) > _STARTED_LIMIT:
            _STARTED.popitem(last=False)


def clear_started_for_tests() -> None:
    with _STARTED_LOCK:
        _STARTED.clear()
        _RAW_PREPARATION_PINS.clear()


@dataclass(frozen=True)
class _Ask:
    """One host decision as this request asks it."""

    request: HostActionRequest
    study_context_id: str
    key: str
    action_id: str
    conversations: HostActionConversations

    def receipt(self, recorded: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "decision_key": self.key,
            "action_code": self.request.action_code,
            **recorded,
        }


def _record(ask: _Ask, job_id: str) -> Dict[str, Any]:
    """Write the requester's conversation row; the job stands either way."""

    error = "host_action_record_failed"
    for attempt in range(2):
        try:
            ask.conversations.record_job_host_action(
                ask.request.session_id,
                project_id=ask.request.project_id,
                action_code=ask.request.action_code,
                action_key=ask.key,
                child_job_id=job_id,
            )
            return {"recorded": True}
        except PiCopilotError as exc:
            error = exc.code
        except Exception:  # noqa: BLE001 -- the job and its pointer are the authority
            logger.exception("host action row for job %s raised", job_id)
            error = "host_action_record_failed"
        if attempt == 0:
            time.sleep(0.2)
    logger.warning(
        "host action %s for job %s is not in its conversation yet: %s",
        ask.request.action_code,
        job_id,
        error,
    )
    return {"recorded": False, "record_error": error}


def _reused(ask: _Ask, job: jobs.Job) -> Dict[str, Any]:
    return {
        "job_id": job.id,
        "kind": job.kind,
        "status": job.status,
        "study_context_id": ask.study_context_id,
        "reused": True,
        "host_action": ask.receipt(_record(ask, job.id)),
    }


def _answer_occupied(ask: _Ask, occupied: _Occupied) -> Dict[str, Any]:
    if occupied.entry is not None:
        if occupied.same_decision:
            raise _refuse(
                "host_action_in_progress",
                status_code=409,
                action_code=occupied.entry.action_code,
                started_at=occupied.entry.started_at,
            )
        raise _refuse(
            "study_job_running",
            status_code=409,
            action_code=occupied.entry.action_code,
            job_started=False,
        )
    job = occupied.job
    if ask.request.action_code == "review_raw_preparation" and job is not None:
        if not occupied.same_decision and getattr(job.host_action, "action_code", None) == "review_raw_preparation":
            keys = {decision_key(ask.request.decision, review_decision=answer) for answer in ("approved", "rejected")}
            if job.host_action.decision_key in keys:
                raise _refuse("raw_preparation_decision_already_answered", status_code=409, job_id=job.id)
        if occupied.same_decision:
            return _reuse_raw_preparation(ask, job)
    if job is not None and occupied.same_decision:
        return _reused(ask, job)
    tag = job.host_action if job is not None else None
    raise _refuse(
        "study_job_running",
        status_code=409,
        job_id=job.id if job is not None else None,
        action_code=tag.action_code if tag is not None else None,
    )


def _reuse_raw_preparation(ask: _Ask, job: jobs.Job) -> Dict[str, Any]:
    """Reuse only the exact conversation's original job, never retry a failure."""
    tag = job.host_action
    expected = {"session_id": ask.request.session_id, "project_id": ask.request.project_id,
                "study_context_id": ask.study_context_id, "action_code": ask.request.action_code,
                "decision_key": ask.key, "action_id": ask.action_id}
    if (tag is None or any(getattr(tag, name, None) != value for name, value in expected.items())
            or job.kind != "raw-preparation-review"):
        raise _refuse("raw_preparation_original_job_mismatch", status_code=409, job_id=job.id)
    if job.status not in {"running", "done"}:
        raise _refuse("raw_preparation_original_job_terminal", status_code=409,
                      job_id=job.id, job_status=job.status)
    result = _reused(ask, job)
    if result.get("host_action", {}).get("recorded"):
        with _STARTED_LOCK:
            _RAW_PREPARATION_PINS.pop((ask.study_context_id, ask.key), None)
    return result


def _raw_preparation_original(ask: _Ask, opposite_key: str) -> Optional[Dict[str, Any]]:
    """Consult durable conversation records before accepting the same offer again."""
    opposite = _Ask(request=ask.request, study_context_id=ask.study_context_id, key=opposite_key,
                    action_id=host_action_id(ask.request.session_id, ask.request.action_code, opposite_key),
                    conversations=ask.conversations)
    same, other = _original_job(ask), _original_job(opposite)
    if same and other:
        raise _refuse("raw_preparation_decision_conflict", status_code=409)
    if other:
        raise _refuse("raw_preparation_decision_already_answered", status_code=409, job_id=other[0])
    if same:
        # A conversation row alone does not prove that the job binding was
        # durably committed; never revive it without the signed host claim.
        raise _refuse("raw_preparation_original_job_unavailable", status_code=409, job_id=same[0])
    return None


def _raw_recorded_occupied(ask: _Ask, opposite_key: str) -> Optional[_Occupied]:
    """Repeat the durable lookup while admission is serialized for the study."""
    opposite = _Ask(request=ask.request, study_context_id=ask.study_context_id, key=opposite_key,
                    action_id=host_action_id(ask.request.session_id, ask.request.action_code, opposite_key),
                    conversations=ask.conversations)
    same, other = _original_job(ask), _original_job(opposite)
    if same and other:
        raise _refuse("raw_preparation_decision_conflict", status_code=409)
    selected = same or other
    if selected is None:
        return None
    raise _refuse("raw_preparation_original_job_unavailable", status_code=409, job_id=selected[0])


def _raw_claim_occupied(ask: _Ask, opposite_key: str) -> Optional[_Occupied]:
    """Read the private job binding; this is not an approval or a grant."""
    from easyicu.research_agent.authority.filesystem import AnchoredDirectory, AuthorityFilesystemError
    from easyicu.webserver.raw_preparation_decisions import _private_directory, _signature

    from easyicu.research_agent.canonical_json import canonical_sha256

    nonce = str(ask.request.decision.nonce)
    name = nonce + ".host-job.json"
    with AnchoredDirectory.open(_private_directory()) as directory:
        if directory.is_absent(name):
            if (not directory.is_absent(nonce + ".host-job-pending.json")
                    or not directory.is_absent(nonce + ".json") or not directory.is_absent(nonce + ".used")):
                raise _refuse("raw_preparation_original_job_unavailable", status_code=409)
            return None
        try:
            pending = json.loads(directory.read_bytes(nonce + ".host-job-pending.json", max_bytes=64 * 1024))
            pending_signature = pending.pop("signature")
            if not isinstance(pending_signature, str) or not hmac.compare_digest(_signature(pending), pending_signature):
                raise ValueError
            receipt = json.loads(directory.read_bytes(name, max_bytes=64 * 1024))
            signature = receipt.pop("signature")
            if not isinstance(signature, str) or not hmac.compare_digest(_signature(receipt), signature):
                raise ValueError
            if (receipt.get("schema_version") != "easyicu.raw-preparation-host-job/1"
                    or receipt.get("nonce") != ask.request.decision.nonce
                    or receipt.get("study_context_id") != ask.study_context_id
                    or receipt.get("project_id") != ask.request.project_id
                    or receipt.get("session_id") != ask.request.session_id
                    or receipt.get("action_code") != ask.request.action_code
                    or receipt.get("pending_sha256") != canonical_sha256(pending)):
                raise ValueError
            job_id = receipt["job_id"]
            stored_key = receipt["decision_key"]
            if stored_key not in {ask.key, opposite_key} or not isinstance(job_id, str) or not job_id:
                raise ValueError
            expected_pending = _raw_pending_body(ask, key=stored_key)
            if pending != expected_pending:
                raise ValueError
        except (AuthorityFilesystemError, OSError, AttributeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise _refuse("raw_preparation_original_job_mismatch", status_code=409) from exc
        directory.assert_still_selected()
    job = jobs.MANAGER.get(job_id)
    if job is None:
        raise _refuse("raw_preparation_original_job_unavailable", status_code=409, job_id=job_id)
    if job.host_action is None and job.kind == "raw-preparation-review" and job.snapshot().get("recovered") is True:
        # Job journals do not persist the conversational tag. Restore it only
        # from this host-signed binding after the session itself was checked.
        job.tag_host_action(HostActionTag(session_id=ask.request.session_id, project_id=ask.request.project_id,
            study_context_id=ask.study_context_id, action_code=ask.request.action_code, decision_key=stored_key,
            action_id=host_action_id(ask.request.session_id, ask.request.action_code, stored_key)))
    return _Occupied(same_decision=stored_key == ask.key, job=job)


def _raw_pending_body(ask: _Ask, *, key: str) -> dict:
    return {"schema_version": "easyicu.raw-preparation-host-job-pending/1", "nonce": ask.request.decision.nonce,
            "study_context_id": ask.study_context_id, "project_id": ask.request.project_id,
            "session_id": ask.request.session_id, "action_code": ask.request.action_code, "decision_key": key}


def _write_raw_pending(ask: _Ask) -> None:
    """Reserve this exact offer before a job can be created; grants nothing."""
    from easyicu.research_agent.authority.filesystem import AnchoredDirectory
    from easyicu.research_agent.canonical_json import canonical_json
    from easyicu.webserver.raw_preparation_decisions import _private_directory, _publish_once, _signature

    body = _raw_pending_body(ask, key=ask.key)
    with AnchoredDirectory.open(_private_directory()) as directory:
        try:
            _publish_once(directory, body["nonce"] + ".host-job-pending.json",
                          canonical_json({**body, "signature": _signature(body)}).encode())
        except FileExistsError as exc:
            raise _refuse("raw_preparation_original_job_conflict", status_code=409) from exc
        directory.assert_still_selected()


def _write_raw_claim(ask: _Ask, job: jobs.Job) -> None:
    """Durably bind one nonce to its host job without recording a decision."""
    from easyicu.research_agent.authority.filesystem import AnchoredDirectory
    from easyicu.research_agent.canonical_json import canonical_json, canonical_sha256
    from easyicu.webserver.raw_preparation_decisions import _private_directory, _publish_once, _signature

    pending = _raw_pending_body(ask, key=ask.key)
    body = {"schema_version": "easyicu.raw-preparation-host-job/1", "nonce": ask.request.decision.nonce,
            "study_context_id": ask.study_context_id, "project_id": ask.request.project_id,
            "session_id": ask.request.session_id, "action_code": ask.request.action_code,
            "decision_key": ask.key, "job_id": job.id, "pending_sha256": canonical_sha256(pending)}
    with AnchoredDirectory.open(_private_directory()) as directory:
        value = json.loads(directory.read_bytes(body["nonce"] + ".host-job-pending.json", max_bytes=64 * 1024))
        signature = value.pop("signature", None)
        if value != pending or not isinstance(signature, str) or not hmac.compare_digest(_signature(value), signature):
            raise _refuse("raw_preparation_original_job_mismatch", status_code=409)
        try:
            _publish_once(directory, body["nonce"] + ".host-job.json",
                          canonical_json({**body, "signature": _signature(body)}).encode())
        except FileExistsError as exc:
            raise _refuse("raw_preparation_original_job_conflict", status_code=409) from exc
        directory.assert_still_selected()


def _original_job(ask: _Ask) -> Optional[Tuple[str, str]]:
    """The job that answered this decision earlier, and its status, if known."""

    with _STARTED_LOCK:
        remembered = _STARTED.get((ask.study_context_id, ask.key), "")
    job = jobs.MANAGER.get(remembered) if remembered else None
    if job is not None:
        return job.id, job.status
    try:
        row = ask.conversations.host_action_child_job(
            ask.request.session_id,
            project_id=ask.request.project_id,
            action_key=ask.key,
        )
    except PiCopilotError:
        row = None
    if row:
        child_id = str(row.get("child_job_id") or "")
        child = jobs.MANAGER.get(child_id)
        if child is not None:
            return child.id, child.status
        status = str(row.get("status") or "")
        # A row still running names a job this process no longer has.
        return child_id, "interrupted" if status in {"", "running"} else status
    if remembered:
        return remembered, "unavailable"
    return None


def _answer_stale(ask: _Ask, stale: Tuple[str, ...]) -> Dict[str, Any]:
    original = _original_job(ask)
    if original is not None and original[1] in {"done", "running"}:
        job = jobs.MANAGER.get(original[0])
        if job is not None and job.status == original[1]:
            # A page opened before the job finished asks again: the decision
            # was answered, and its job is the answer.
            return _reused(ask, job)
    raise _refuse(
        "host_action_decision_stale",
        status_code=409,
        mismatched_fields=list(stale),
        job_id=original[0] if original else None,
        job_status=original[1] if original else None,
    )


def submit_with_host_action(
    host_action: HostActionRequest,
    *,
    study_context_id: str,
    submit: Callable[[], Mapping[str, Any]],
    review_decision: str = "",
    retry_options: str = "",
    offered: Callable[[str], HostDecisionOffers] = offered_decisions,
    conversations: Optional[HostActionConversations] = None,
) -> Dict[str, Any]:
    """Start the job of one host decision once, and record it in its conversation.

    ``submit`` is the existing submission (``submit_research_run`` or the plan
    review resume) with every check it makes; its errors propagate unchanged.
    """

    study_id = str(study_context_id or "").strip()
    if not study_id:
        raise _refuse("host_action_study_required", status_code=400)
    owner = conversations if conversations is not None else get_pi_copilot_service()
    try:
        owner.host_action_session(
            host_action.session_id,
            project_id=host_action.project_id,
            study_context_id=study_id,
        )
    except PiCopilotError as exc:
        raise HostActionJobError(exc.detail, status_code=exc.status_code) from exc
    try:
        key = decision_key(
            host_action.decision,
            review_decision=review_decision,
            retry_options=retry_options,
        )
    except ValueError as exc:
        raise _refuse(str(exc), status_code=400) from exc
    ask = _Ask(
        request=host_action,
        study_context_id=study_id,
        key=key,
        action_id=host_action_id(host_action.session_id, host_action.action_code, key),
        conversations=owner,
    )

    raw_opposite_key = ""
    if host_action.action_code == "review_raw_preparation":
        raw_opposite_key = decision_key(host_action.decision,
            review_decision="rejected" if review_decision == "approved" else "approved")

    with study_lock(study_id) as starting:
        occupied = _occupied(starting, study_id, key, raw_opposite_key=raw_opposite_key)
        if raw_opposite_key and (occupied is None or occupied.same_decision and occupied.job is not None):
            claimed = _raw_claim_occupied(ask, raw_opposite_key)
            if claimed is not None:
                if occupied is not None and occupied.job.id != claimed.job.id:
                    raise _refuse("raw_preparation_original_job_conflict", status_code=409)
                occupied = claimed
            elif occupied is not None:
                raise _refuse("raw_preparation_original_job_unavailable", status_code=409, job_id=occupied.job.id)
    if occupied is not None:
        return _answer_occupied(ask, occupied)
    if raw_opposite_key:
        original = _raw_preparation_original(ask, raw_opposite_key)
        if original is not None:
            return original
    # The projection is read outside the lock: it reads run history and the
    # paused review, and it reads the registry itself.
    family = ACTION_DECISION_FAMILIES[host_action.action_code]
    try:
        offers = offered(study_id)
    except Exception as exc:  # noqa: BLE001 -- a decision that cannot be checked is not run
        raise _refuse(
            "host_action_state_unavailable",
            status_code=500,
            reason=type(exc).__name__,
        ) from exc
    stale = mismatched_fields(host_action.decision, offers.for_family(family))
    entry = StartingEntry(
        study_context_id=study_id,
        session_id=host_action.session_id,
        action_code=host_action.action_code,
        decision_key=key,
        action_id=ask.action_id,
        started_at=time.time(),
    )
    with study_lock(study_id) as starting:
        # Another request may have started this study while the projection
        # was read.
        occupied = _occupied(starting, study_id, key, raw_opposite_key=raw_opposite_key)
        if raw_opposite_key and (occupied is None or occupied.same_decision and occupied.job is not None):
            claimed = _raw_claim_occupied(ask, raw_opposite_key)
            if claimed is not None:
                if occupied is not None and occupied.job.id != claimed.job.id:
                    raise _refuse("raw_preparation_original_job_conflict", status_code=409)
                occupied = claimed
            elif occupied is not None:
                raise _refuse("raw_preparation_original_job_unavailable", status_code=409, job_id=occupied.job.id)
        if occupied is None and raw_opposite_key:
            occupied = _raw_recorded_occupied(ask, raw_opposite_key)
        if occupied is None and not stale:
            if raw_opposite_key:
                with _STARTED_LOCK:
                    if len(_RAW_PREPARATION_PINS) >= _STARTED_LIMIT:
                        raise _refuse("raw_preparation_record_capacity", status_code=409)
                    _RAW_PREPARATION_PINS[(study_id, key)] = ""
                _write_raw_pending(ask)
            starting.register(entry)
    if occupied is not None:
        return _answer_occupied(ask, occupied)
    if stale:
        return _answer_stale(ask, stale)

    try:
        receipt = dict(submit())
    except BaseException:
        # No job: nothing to record, and the decision is free to be asked again.
        with study_lock(study_id) as starting:
            starting.clear(entry)
            if raw_opposite_key:
                with _STARTED_LOCK:
                    _RAW_PREPARATION_PINS.pop((study_id, key), None)
        raise
    job_id = str(receipt.get("job_id") or "").strip()
    job = jobs.MANAGER.get(job_id) if job_id else None
    with study_lock(study_id) as starting:
        # Tag before clearing: a repeat that no longer finds the entry must
        # find the job's decision, or it would be taken for another decision.
        if job is not None:
            job.tag_host_action(
                HostActionTag(
                    session_id=host_action.session_id,
                    project_id=host_action.project_id,
                    study_context_id=study_id,
                    action_code=host_action.action_code,
                    decision_key=key,
                    action_id=ask.action_id,
                )
            )
            if raw_opposite_key:
                _remember_started(study_id, key, job.id)
                with _STARTED_LOCK:
                    _RAW_PREPARATION_PINS[(study_id, key)] = job.id
                try:
                    _write_raw_claim(ask, job)
                except BaseException:
                    starting.clear(entry)
                    raise
        elif raw_opposite_key:
            with _STARTED_LOCK:
                _RAW_PREPARATION_PINS[(study_id, key)] = job_id or "unavailable"
        starting.clear(entry)
    if job is None:
        logger.warning(
            "host action %s started job %r, which is not running here", key, job_id
        )
        recorded: Dict[str, Any] = {
            "recorded": False,
            "record_error": "host_action_job_unavailable",
        }
    else:
        _remember_started(study_id, key, job.id)
        recorded = _record(ask, job.id)
        if raw_opposite_key and recorded.get("recorded"):
            with _STARTED_LOCK:
                _RAW_PREPARATION_PINS.pop((study_id, key), None)
    return {**receipt, "reused": False, "host_action": ask.receipt(recorded)}


__all__ = [
    "HostActionConversations",
    "HostActionJobError",
    "check_agent_run_request",
    "check_review_request",
    "clear_started_for_tests",
    "offered_decisions",
    "parse_host_action",
    "submit_with_host_action",
]
