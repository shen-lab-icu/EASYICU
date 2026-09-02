"""Frozen-sheet generation and pre-unblinding score locking for Heldout27."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


RUBRIC_PATH = Path(__file__).with_name("heldout27_evaluation_rubric_v2.json")
TASKBANK_PATH = Path(__file__).with_name("heldout27_taskbank_v1.jsonl")
_TASK_FIELDS = (
    "question",
    "target_outcome",
    "exposure_or_index",
    "time_origin",
    "measurement_policy",
    "expected_outputs",
    "semantic_guardrails",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BlindedEvaluationError(ValueError):
    reason_code = "BLINDED_EVALUATION_INVALID"


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _load_rubric() -> dict[str, Any]:
    value = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BlindedEvaluationError("rubric must be a JSON object")
    return value


def load_heldout_tasks() -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for line in TASKBANK_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or task_id in tasks:
            raise BlindedEvaluationError("taskbank identities are invalid")
        tasks[task_id] = task
    if len(tasks) != 27:
        raise BlindedEvaluationError("Heldout27 must contain exactly 27 tasks")
    return tasks


def instantiate_review_sheet(task_id: str) -> dict[str, Any]:
    """Create an arm-neutral sheet from only the preregistered task fields."""

    task = load_heldout_tasks().get(task_id)
    if task is None:
        raise BlindedEvaluationError(f"unknown Heldout27 task: {task_id}")
    rubric = _load_rubric()
    task_criteria = {field: task[field] for field in _TASK_FIELDS}
    sheet = {
        "schema_version": "easyicu.figure2_blinded_review_sheet/1",
        "task_id": task_id,
        "primary_endpoint": rubric["primary_endpoint"],
        "hard_gates": rubric["hard_gates"],
        "secondary_dimensions": rubric["secondary_dimensions"],
        "dimension_scale": rubric["dimension_scale"],
        "task_criteria": task_criteria,
        "arm_identity_visible": False,
    }
    sheet["sheet_sha256"] = hashlib.sha256(_canonical_json(sheet)).hexdigest()
    return sheet


def _validate_score(
    score: Mapping[str, Any],
    *,
    task_id: str,
    gate_ids: set[str],
    dimensions: set[str],
) -> None:
    required = {
        "reviewer_id",
        "task_id",
        "bundle_id",
        "primary_success",
        "hard_gates_passed",
        "dimension_scores",
        "arm_guess",
        "rationale",
    }
    if set(score) != required:
        raise BlindedEvaluationError("review score fields do not match the lock schema")
    if score["task_id"] != task_id:
        raise BlindedEvaluationError("review score task identity mismatch")
    if not isinstance(score["reviewer_id"], str) or not score["reviewer_id"].strip():
        raise BlindedEvaluationError("reviewer_id must be non-empty")
    if score["bundle_id"] not in {"bundle_1", "bundle_2"}:
        raise BlindedEvaluationError("bundle_id must be bundle_1 or bundle_2")
    if not isinstance(score["primary_success"], bool):
        raise BlindedEvaluationError("primary_success must be boolean")
    gates = score["hard_gates_passed"]
    if not isinstance(gates, Mapping) or set(gates) != gate_ids or any(
        not isinstance(value, bool) for value in gates.values()
    ):
        raise BlindedEvaluationError("hard-gate scores are incomplete or invalid")
    dimension_scores = score["dimension_scores"]
    if not isinstance(dimension_scores, Mapping) or set(dimension_scores) != dimensions or any(
        type(value) is not int or value not in {0, 1, 2}
        for value in dimension_scores.values()
    ):
        raise BlindedEvaluationError("dimension scores are incomplete or invalid")
    if score["primary_success"] != all(gates.values()):
        raise BlindedEvaluationError("primary score must equal the conjunction of hard gates")
    if score["arm_guess"] not in {"bundle_1", "bundle_2", "cannot_tell"}:
        raise BlindedEvaluationError("arm_guess is invalid")
    if not isinstance(score["rationale"], str) or not score["rationale"].strip():
        raise BlindedEvaluationError("review rationale must be non-empty")


def lock_blinded_scores(
    *,
    task_id: str,
    sheet_sha256: str,
    reviews: Sequence[Mapping[str, Any]],
    eligible_reviewer_ids: Sequence[str],
    reviewer_eligibility_receipt_sha256: str,
    destination: Path,
) -> dict[str, Any]:
    """Atomically lock two independent reviews before any arm mapping is supplied."""

    sheet = instantiate_review_sheet(task_id)
    if sheet_sha256 != sheet["sheet_sha256"]:
        raise BlindedEvaluationError("review sheet digest mismatch")
    if len(reviews) != 4:
        raise BlindedEvaluationError("two reviewers must score both blinded bundles")
    if not _SHA256_RE.fullmatch(reviewer_eligibility_receipt_sha256):
        raise BlindedEvaluationError("reviewer eligibility receipt digest is invalid")
    expected_reviewers = set(eligible_reviewer_ids)
    if len(expected_reviewers) != 2 or any(
        not isinstance(reviewer, str) or not reviewer.strip()
        for reviewer in expected_reviewers
    ):
        raise BlindedEvaluationError("exactly two eligible reviewer IDs are required")
    rubric = _load_rubric()
    gate_ids = {item["gate_id"] for item in rubric["hard_gates"]}
    dimensions = set(rubric["secondary_dimensions"])
    identities: set[tuple[str, str]] = set()
    for score in reviews:
        _validate_score(
            score,
            task_id=task_id,
            gate_ids=gate_ids,
            dimensions=dimensions,
        )
        identity = (score["reviewer_id"], score["bundle_id"])
        if identity in identities:
            raise BlindedEvaluationError("duplicate reviewer-bundle score")
        identities.add(identity)
    reviewers = {reviewer for reviewer, _ in identities}
    if reviewers != expected_reviewers:
        raise BlindedEvaluationError(
            "score reviewers do not match the sealed eligibility receipt"
        )
    if len(reviewers) != 2 or any(
        {(reviewer, "bundle_1"), (reviewer, "bundle_2")} - identities
        for reviewer in reviewers
    ):
        raise BlindedEvaluationError("each of two reviewers must score both bundles")
    for reviewer in reviewers:
        guesses = {
            score["arm_guess"]
            for score in reviews
            if score["reviewer_id"] == reviewer
        }
        if len(guesses) != 1:
            raise BlindedEvaluationError(
                "each reviewer must lock one consistent arm guess"
            )

    disagreements = {
        bundle_id: len(
            {
                score["primary_success"]
                for score in reviews
                if score["bundle_id"] == bundle_id
            }
        )
        > 1
        for bundle_id in ("bundle_1", "bundle_2")
    }
    receipt = {
        "schema_version": "easyicu.figure2_blinded_score_lock/1",
        "task_id": task_id,
        "sheet_sha256": sheet_sha256,
        "reviewer_eligibility_receipt_sha256": (
            reviewer_eligibility_receipt_sha256
        ),
        "arm_mapping_present": False,
        "reviews": [dict(score) for score in reviews],
        "adjudication_required": disagreements,
    }
    payload = _canonical_json(receipt)
    receipt["score_lock_sha256"] = hashlib.sha256(payload).hexdigest()
    final_payload = _canonical_json(receipt)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(final_payload)
    return receipt


__all__ = [
    "BlindedEvaluationError",
    "instantiate_review_sheet",
    "load_heldout_tasks",
    "lock_blinded_scores",
]
