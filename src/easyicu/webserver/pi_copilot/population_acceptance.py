"""Persist explicit omission choices, separately from full-plan approval."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.planning.population_review import (
    NOT_APPLIED_STOP,
    request_allows_approval,
    sealed_population,
    unapplied_inclusions,
)
from easyicu.research_agent.planning.population_spec import STUDY_WORDING_SOURCES
from easyicu.webserver import state_paths, study_contexts

_LOCK = threading.RLock()


def _path(basis: Mapping[str, Any]) -> Path:
    return (
        state_paths.state_root()
        / "population-acceptances"
        / f"{canonical_sha256(basis)}.json"
    )


def _read_choices(basis: Mapping[str, Any], required: set[str]) -> dict | None:
    """Missing means unanswered; corrupt means unavailable, never accepted."""
    path = _path(basis)
    if not path.exists():
        return {}
    try:
        if path.is_symlink() or path.stat().st_size > 131_072:
            return None
        data = json.loads(path.read_text())
        choices = data["choices"]
        if (
            data.get("schema_version") != "easyicu.population_acceptance_choices/1"
            or data.get("basis") != basis
            or data.get("sha256")
            != canonical_sha256({k: v for k, v in data.items() if k != "sha256"})
            or not isinstance(choices, dict)
            or not set(choices) <= required
            or any(
                not isinstance(value, dict)
                or not value.get("session_id")
                or not value.get("decided_at")
                for value in choices.values()
            )
        ):
            return None
        return choices
    except (OSError, ValueError, KeyError, TypeError):
        return None


def current_review_study(planned: Mapping[str, Any]) -> dict | None:
    """Lifecycle edits may advance revision without changing the reviewed science.

    New proposals bind the current revision. Stored choices still match only
    the revision on which the researcher made them.
    """
    try:
        current = study_contexts.get_context(str(planned.get("id") or ""))
    except study_contexts.StudyContextError:
        return None
    if current is None or study_contexts.scientific_configuration_sha256(
        current
    ) != study_contexts.scientific_configuration_sha256(planned):
        return None
    return current


def population_acceptance_proposal(
    *, study: Mapping[str, Any], pending: Any
) -> dict | None:
    selected = [
        r for r in pending.requests if r.payload.get("reason") == NOT_APPLIED_STOP
    ]
    if len(selected) != 1:
        return None
    request = selected[0]
    record = sealed_population(request, reason=NOT_APPLIED_STOP)
    if record is None or any(
        row["disposition"] == "not_applied" and row["criterion"]["role"] == "exclude"
        for row in record["criteria"]
    ):
        # The existing acceptance flow only describes omitted inclusions.
        # Never present it as a way to waive an omitted exclusion.
        return None
    rows = unapplied_inclusions(request)
    if not rows:
        return None
    basis = {
        "study_id": study["id"],
        "study_revision": study["revision"],
        "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(
            study
        ),
        "run_id": pending.run_id,
        "review_id": request.review_id,
        "review_authority_sha256": request.authority_sha256,
        "plan_sha256": request.payload["plan_review_authority"]["plan_sha256"],
        "population_compile_sha256": request.payload["population_compile_sha256"],
    }
    choices = _read_choices(basis, {row["criterion"]["id"] for row in rows})
    accepted = sorted(choices or {})
    complete = choices is not None and request_allows_approval(request, accepted)
    body = {
        "schema_version": "easyicu.population_acceptance_proposal/1",
        **basis,
        "basis": basis,
        "available": choices is not None and not complete,
        "complete": complete,
        "record_invalid": choices is None,
        "accepted_criterion_ids": accepted,
        "criteria": [
            {
                "id": row["criterion"]["id"],
                "quote": row["criterion"]["quote"],
                "source": row["criterion"]["source"],
                "proposed_by_system": row["criterion"]["source"]
                not in STUDY_WORDING_SOURCES,
                **row.get("provenance", {}),
                "disposition": row["disposition"],
                "reason": row["reason"],
                "accepted": row["criterion"]["id"] in accepted,
            }
            for row in rows
        ],
        "applied_criteria": [
            row["criterion"]["quote"]
            for row in record["criteria"]
            if row["disposition"] in {"applied_by_plan", "applied_by_source"}
        ],
        "actual_cohort": record["cohort"],
    }
    return {**body, "proposal_sha256": canonical_sha256(body)}


def current_population_acceptance(
    study: Mapping[str, Any], review: Mapping[str, Any]
) -> dict | None:
    proposal = review.get("population_acceptance")
    if not isinstance(proposal, dict) or (
        proposal.get("study_id") != study.get("id")
        or proposal.get("study_revision") != study.get("revision")
        or proposal.get("run_id") != review.get("run_id")
        or proposal.get("scientific_configuration_sha256")
        != study_contexts.scientific_configuration_sha256(study)
        or proposal.get("proposal_sha256")
        != canonical_sha256(
            {k: v for k, v in proposal.items() if k != "proposal_sha256"}
        )
    ):
        return None
    return proposal


def accepted_ids_for_request(
    proposal: Mapping[str, Any] | None, request: Any
) -> tuple[str, ...]:
    if (
        not proposal
        or proposal.get("record_invalid")
        or (
            proposal.get("review_id") != request.review_id
            or proposal.get("review_authority_sha256") != request.authority_sha256
        )
    ):
        return ()
    return tuple(proposal.get("accepted_criterion_ids") or ())


def save_criterion(
    proposal: Mapping[str, Any], *, criterion_id: str, session_id: str
) -> None:
    """Save one explicit UI answer. No study, plan, grants or jobs are changed."""
    if not proposal.get("available") or criterion_id not in {
        row["id"] for row in proposal["criteria"]
    }:
        raise ValueError("population acceptance is unavailable")
    with _LOCK:
        basis = proposal["basis"]
        choices = _read_choices(basis, {row["id"] for row in proposal["criteria"]})
        if choices is None or sorted(choices) != proposal["accepted_criterion_ids"]:
            raise ValueError("population acceptance changed before saving")
        current = study_contexts.get_context(basis["study_id"])
        if (
            current is None
            or current.get("active_job_id")
            or (
                current["revision"] != basis["study_revision"]
                or study_contexts.scientific_configuration_sha256(current)
                != basis["scientific_configuration_sha256"]
            )
        ):
            raise ValueError("study changed before saving population acceptance")
        if criterion_id in choices:
            return
        body = {
            "schema_version": "easyicu.population_acceptance_choices/1",
            "basis": basis,
            "choices": {
                **choices,
                criterion_id: {
                    "session_id": session_id,
                    "decided_at": datetime.now(timezone.utc).isoformat(),
                    "proposal_sha256": proposal["proposal_sha256"],
                },
            },
        }
        path = _path(basis)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=path.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                json.dump(
                    {**body, "sha256": canonical_sha256(body)},
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
