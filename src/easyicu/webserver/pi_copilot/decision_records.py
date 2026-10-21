"""The conversation's record of each decision, read from the owners that keep it.

A gate card leaves the conversation once it is answered.  What stays is one
past-tense line per decision, and each line comes from a record a host owner
persisted when the decision was made -- never from the page that clicked:

* a job-starting decision: the host action row ``host_action_jobs`` writes in
  the request that starts its job, under the key ``decision_key`` made (a row
  under any other key proves nothing and is not shown);
* the data choice: the conversation's data-source authorization;
* an automatic continuation after a provider outage: the job's
  ``provider_recovery.json`` receipt, which records a system action.

Records carry closed codes, times and display facts only (the source label is
the one the session already shows); the browser owns the wording (``screens-guided-pi-decision-records.js``).  System actions are a
separate actor and are never projected as the researcher's.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from easyicu.webserver.host_action_contracts import decision_key_facts

#: The researcher's job-starting decisions a record names, by the action the
#: host recorded and what its key says was decided.
_PLAN_CONFIRMED = "plan_confirmed"
_PLAN_APPROVED = "plan_approved"
_PLAN_REJECTED = "plan_rejected"
_DATA_CHOSEN = "data_chosen"
_PROVIDER_RESUMED = "provider_resumed"
#: Data-source confirmations the researcher made; ``agent_default_study_required``
#: is the system's policy and ``legacy_session`` predates the choice.
_CHOSEN_SOURCE_MODES = frozenset(
    {"reuse_project_source", "select_local_source", "select_registered_source"}
)
_MAX_RECORDS = 64


def _utc(value: Any) -> Optional[str]:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def user_decision_records(
    host_action_turns: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """The plan decisions the researcher's clicks recorded, oldest first."""

    records: List[Dict[str, Any]] = []
    for turn in host_action_turns:
        if not isinstance(turn, Mapping) or turn.get("kind") != "host_action":
            continue
        facts = decision_key_facts(turn.get("action_key"))
        at = _utc(turn.get("started_at"))
        if facts is None or at is None:
            continue
        action = str(turn.get("action_code") or "")
        code = ""
        if (
            action == "prepare_analysis_data"
            and facts.family == "plan_transition"
            and facts.next_action_code == "plan_execution_upgrade_required"
        ):
            code = _PLAN_CONFIRMED
        elif action == "execute_plan" and facts.family == "plan_review":
            code = _PLAN_APPROVED if facts.review_decision == "approved" else _PLAN_REJECTED
        if code:
            records.append(
                {"code": code, "actor": "user", "at": at, "run_id": facts.run_id}
            )
    return records


def data_choice_record(authorization: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The data source the researcher chose for this conversation, if one."""

    if not isinstance(authorization, Mapping):
        return None
    if (
        authorization.get("status") != "confirmed"
        or authorization.get("confirmation_mode") not in _CHOSEN_SOURCE_MODES
    ):
        return None
    at = _utc(authorization.get("confirmed_at"))
    source = authorization.get("source")
    source = source if isinstance(source, Mapping) else {}
    label = str(source.get("label") or "").strip()
    if at is None or not label:
        return None
    return {
        "code": _DATA_CHOSEN,
        "actor": "user",
        "at": at,
        "source_label": label[:160],
        "source_release": str(source.get("reference_release") or "").strip()[:80],
    }


def system_action_records(
    recovery_receipts: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Each automatic continuation a provider-recovery receipt records."""

    records: List[Dict[str, Any]] = []
    for receipt in recovery_receipts:
        if not isinstance(receipt, Mapping):
            continue
        limit = receipt.get("max_automatic_resumes")
        for row in receipt.get("automatic_resumes") or []:
            at = _utc(row.get("at")) if isinstance(row, Mapping) else None
            if at is None or type(limit) is not int:
                continue
            records.append(
                {
                    "code": _PROVIDER_RESUMED,
                    "actor": "system",
                    "at": at,
                    "job_id": str(receipt.get("job_id") or ""),
                    "number": int(row.get("number") or 0),
                    "limit": limit,
                    "cause_code": str(row.get("cause_code") or ""),
                }
            )
    return records


def project_decision_records(
    *,
    host_action_turns: Iterable[Mapping[str, Any]],
    data_source_authorization: Mapping[str, Any],
    recovery_receipts: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Every decision record of one conversation, oldest first."""

    data_choice = data_choice_record(data_source_authorization)
    records = [
        *([data_choice] if data_choice else []),
        *user_decision_records(host_action_turns),
        *system_action_records(recovery_receipts),
    ]
    records.sort(key=lambda row: (row["at"], row["actor"] == "system"))
    return records[-_MAX_RECORDS:]


def conversation_job_ids(
    host_action_turns: Iterable[Mapping[str, Any]],
    child_jobs: Sequence[Mapping[str, Any]],
) -> List[str]:
    """The jobs this conversation started or submitted, in first-seen order."""

    ids = [str(turn.get("child_job_id") or "") for turn in host_action_turns]
    ids += [str(job.get("job_id") or "") for job in child_jobs if isinstance(job, Mapping)]
    return [job_id for job_id in dict.fromkeys(ids) if job_id]


__all__ = [
    "conversation_job_ids",
    "data_choice_record",
    "project_decision_records",
    "system_action_records",
    "user_decision_records",
]
