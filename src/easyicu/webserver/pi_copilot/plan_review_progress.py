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


def _receipt(study: Mapping[str, Any]) -> ReviewChoiceProgress | None:
    path = _path(str(study.get("id") or ""))
    try:
        if path.stat().st_size > 16_384:
            return None
        return ReviewChoiceProgress.model_validate_json(path.read_text())
    except (OSError, ValidationError):
        return None


def matching_progress(
    study: Mapping[str, Any], run: Mapping[str, Any]
) -> ReviewChoiceProgress | None:
    """Read only; a receipt matches one exact current revision and source run."""
    progress = _receipt(study)
    if progress is None:
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


AGENT_PLAN_CONFIGURATION_CHOICE = ("agent_plan_configuration", "typed_runtime_projection")


# What a raw official-Demo reconfirmation writes besides the signed choice
# (demo_planning_source.bind_demo_planning_source): the raw source is to be
# prepared again, so extraction is no longer complete. It carries no exposure
# group, plan revision or analysis coordinate. Anything else it changes (a
# drafted cohort scope, another source) is outside this allowance.
_RECONFIRMATION_CONFIRMATIONS = {"extraction_completed": (True, False)}


def _one_reconfirmation_after(
    study: Mapping[str, Any], progress: ReviewChoiceProgress
) -> bool:
    """Exactly one write followed the compile: the signed raw-Demo reconfirmation.

    The study keeps no per-revision field history, so this accepts one case
    only: the study is one revision after the receipt, its current choice
    resolves and was signed at exactly that revision, and undoing the
    reconfirmation's allowed fields gives back the digest the compile wrote.
    """

    from .demo_planning_authority import signed_choice_revision

    revision = study.get("revision")
    if type(revision) is not int or revision != progress.revision + 1:
        return False
    if signed_choice_revision(study) != revision:
        return False
    confirmations = study.get("confirmations")
    if not isinstance(confirmations, Mapping):
        return False
    before = dict(confirmations)
    for key, (compiled, reconfirmed) in _RECONFIRMATION_CONFIRMATIONS.items():
        if confirmations.get(key) is not reconfirmed:
            return False
        before[key] = compiled
    compiled_study = {
        **{key: value for key, value in study.items() if key != "demo_planning_authority_ref"},
        "confirmations": before,
        "revision": progress.revision,
    }
    return (
        study_contexts.scientific_configuration_sha256(compiled_study)
        == progress.current_digest
    )


def _compile_receipt(study: Mapping[str, Any]) -> ReviewChoiceProgress | None:
    progress = _receipt(study)
    name, option = AGENT_PLAN_CONFIGURATION_CHOICE
    if progress is None or progress.study_id != study.get("id") or progress.choices.get(name) != option:
        return None
    return progress


def compiled_configuration_progress(
    study: Mapping[str, Any], run: Mapping[str, Any]
) -> ReviewChoiceProgress | None:
    """The compile receipt of ``run`` while the study keeps what it wrote.

    The fresh plan after a host compile inherits that candidate's groups and
    revision contract through this receipt. It holds for the exact revision
    the compile wrote (``matching_progress``), or after exactly one signed
    raw-Demo reconfirmation of that configuration (B10, UX leg 3, 2026-11-19:
    the reconfirmation lost the candidate's groups, the grouped landmark
    authority was released and the same runtime gaps returned). Any other
    write, run or study is not this receipt.
    """

    progress = _compile_receipt(study)
    if progress is None or progress.run_id != run.get("run_id") or (
        progress.source_digest != run.get("scientific_configuration_sha256")
    ):
        return None
    if matching_progress(study, run) is not None or _one_reconfirmation_after(study, progress):
        return progress
    return None


def planned_on_compiled_configuration(
    study: Mapping[str, Any], run: Mapping[str, Any]
) -> bool:
    """Whether ``run`` was planned on the configuration the host compiler wrote.

    The compiler projects one reviewed plan's runtime gaps into the study and
    a fresh plan follows. When that fresh plan is reviewed with runtime gaps
    again, compiling once more only rewrites the study from another plan's
    wording and plans again (UX leg 3 en/zh, 2026-11-19, B10: each round also
    ended a signed raw-Demo choice, so the loop never converged). Such a run
    goes to its review instead. A run planned after exactly one signed
    reconfirmation of that configuration is the same lineage. Read only; a
    missing or unrelated receipt is not this fact.
    """

    progress = _compile_receipt(study)
    if progress is None or progress.run_id == run.get("run_id"):
        return False
    digest = run.get("scientific_configuration_sha256")
    return digest == progress.current_digest or (
        _one_reconfirmation_after(study, progress)
        and digest == study_contexts.scientific_configuration_sha256(study)
    )


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
