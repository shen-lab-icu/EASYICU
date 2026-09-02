"""Durable continuity for sibling choices from one immutable candidate plan.

This receipt permits displaying the remaining choices after a host-owned edit.
It never makes the old plan executable or confers cohort/analysis approval.
Unrelated edits, another run, and missing/corrupt receipts fail closed.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from easyicu.webserver import state_paths, study_contexts

from .plan_decisions import PlanDecisionError, pending_authorization_questions


class ReviewChoiceProgress(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.plan-review-choices/1"] = (
        "easyicu.plan-review-choices/1"
    )
    study_id: str = Field(min_length=1, max_length=160)
    run_id: str = Field(min_length=1, max_length=160)
    source_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    current_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    revision: int = Field(ge=1)
    choices: dict[str, str] = Field(min_length=1, max_length=16)


def _path(study_id: str) -> Path:
    key = hashlib.sha256(study_id.encode()).hexdigest()
    return state_paths.state_root() / "plan-review-choices" / f"{key}.json"


def matching_progress(
    study: Mapping[str, Any], run: Mapping[str, Any]
) -> ReviewChoiceProgress | None:
    """Read only; a receipt matches one exact current revision and source run."""
    path = _path(str(study.get("id") or ""))
    try:
        if path.stat().st_size > 16_384:
            return None
        progress = ReviewChoiceProgress.model_validate_json(path.read_text())
    except (OSError, ValidationError):
        return None
    if (
        progress.study_id != study.get("id")
        or progress.run_id != run.get("run_id")
        or progress.source_digest != run.get("scientific_configuration_sha256")
        or progress.revision != study.get("revision")
        or progress.current_digest
        != study_contexts.scientific_configuration_sha256(study)
    ):
        return None
    return progress


def validate_choice_source(study: Mapping[str, Any], run: Mapping[str, Any]) -> None:
    """Do not apply a stale candidate's choices to an unrelated configuration."""
    digest = str(run.get("scientific_configuration_sha256") or "")
    if len(digest) != 64 or (
        digest != study_contexts.scientific_configuration_sha256(study)
        and matching_progress(study, run) is None
    ):
        raise PlanDecisionError(
            "plan_decision_source_superseded",
            "The study changed outside this plan review; generate a fresh candidate before applying its choices.",
        )


def has_pending_choices(
    study: Mapping[str, Any], run: Mapping[str, Any], review: Mapping[str, Any]
) -> bool:
    """Return only a display-continuity fact, never execution authorization."""
    if matching_progress(study, run) is None:
        return False
    payloads = review.get("artifact_payloads")
    scientific = (
        payloads.get("scientific_plan_review.json")
        if isinstance(payloads, Mapping)
        else None
    )
    findings = scientific.get("findings") if isinstance(scientific, Mapping) else None
    if not isinstance(findings, list):
        return False
    return bool(
        pending_authorization_questions(
            study,
            [
                row
                for row in findings
                if isinstance(row, Mapping)
                and row.get("requires_user_authorization") is True
            ],
        )
    )


def record_choice(
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    run: Mapping[str, Any],
    decision_code: str,
    option_id: str,
) -> None:
    """Called only after the typed host selection succeeds at its CAS boundary."""
    validate_choice_source(before, run)
    previous = matching_progress(before, run)
    progress = ReviewChoiceProgress(
        study_id=str(after["id"]),
        run_id=str(run["run_id"]),
        source_digest=str(run["scientific_configuration_sha256"]),
        current_digest=study_contexts.scientific_configuration_sha256(after),
        revision=int(after["revision"]),
        choices={**(previous.choices if previous else {}), decision_code: option_id},
    )
    path = _path(progress.study_id)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(progress.model_dump_json(indent=2))
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
