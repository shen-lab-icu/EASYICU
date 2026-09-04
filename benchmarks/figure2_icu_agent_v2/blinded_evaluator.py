"""Frozen-sheet generation and pre-unblinding score locking for Figure 2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .review_bundle_normalizer import (
    CANONICAL_FILES,
    NormalizedReviewBundle,
    ReviewBlindingContext,
    normalize_review_bundle,
)


RUBRIC_PATH = Path(__file__).with_name("heldout27_evaluation_rubric_v2.json")
TASKBANK_PATH = Path(__file__).with_name("heldout27_taskbank_v1.jsonl")
QUALIFICATION_TASKBANK_PATH = (
    Path(__file__).resolve().parents[1]
    / "meta_generalization"
    / "meta_benchmark.jsonl"
)
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
_BLINDED_BUNDLE_IDS = ("bundle_1", "bundle_2")
_REDACTION_LOG_FIELDS = ("file", "location", "rule", "replacement_count")


class BlindedEvaluationError(ValueError):
    reason_code = "BLINDED_EVALUATION_INVALID"


@dataclass(frozen=True)
class BlindedReviewBundle:
    """Immutable normalized bytes and audit identity for one blinded label."""

    bundle_id: str
    _files: tuple[tuple[str, bytes], ...]
    pre_normalization_sha256: tuple[tuple[str, str], ...]
    post_normalization_sha256: tuple[tuple[str, str], ...]
    _redaction_log: tuple[tuple[tuple[str, str], ...], ...]
    redaction_log_sha256: str
    normalized_bundle_sha256: str

    def files_for_review(self) -> dict[str, bytes]:
        """Return a copy of the exact bytes identified by the digest receipt."""

        return dict(self._files)

    def digest_receipt(self) -> dict[str, Any]:
        return {
            "pre_normalization_sha256": dict(self.pre_normalization_sha256),
            "post_normalization_sha256": dict(self.post_normalization_sha256),
            "redaction_log_sha256": self.redaction_log_sha256,
            "normalized_bundle_sha256": self.normalized_bundle_sha256,
        }

    def redaction_log_for_audit(self) -> tuple[dict[str, str], ...]:
        """Return an audit copy kept outside the blinded reviewer surface."""

        return tuple(dict(entry) for entry in self._redaction_log)


@dataclass(frozen=True)
class BlindedReviewPackage:
    """One task's exact normalized pair, sheet, and digest seal."""

    task_id: str
    task_split: str
    _sheet: bytes
    sheet_sha256: str
    bundles: tuple[BlindedReviewBundle, BlindedReviewBundle]
    review_package_sha256: str

    def files_for_review(self, bundle_id: str) -> dict[str, bytes]:
        for bundle in self.bundles:
            if bundle.bundle_id == bundle_id:
                return bundle.files_for_review()
        raise BlindedEvaluationError(f"unknown blinded bundle: {bundle_id}")

    def sheet_for_review(self) -> bytes:
        """Return the exact arm-neutral sheet bytes sealed by this package."""

        return self._sheet

    def digest_receipt(self) -> dict[str, dict[str, Any]]:
        return {
            bundle.bundle_id: bundle.digest_receipt()
            for bundle in self.bundles
        }


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


def _source_task_id(files: Mapping[str, bytes]) -> str:
    try:
        receipt = json.loads(files["07_run_receipt.json"])
        task_id = receipt["task_id"]
    except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise BlindedEvaluationError(
            "normalized source receipt is invalid"
        ) from exc
    if not isinstance(task_id, str) or not task_id:
        raise BlindedEvaluationError("normalized source receipt task is invalid")
    return task_id


def _write_score_lock_stage(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _publish_score_lock(payload: bytes, destination: Path) -> None:
    """Publish complete bytes once; failures never expose a partial target."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    staging = target.parent / f".{target.name}.{uuid4().hex}.stage"
    try:
        _write_score_lock_stage(staging, payload)
        if staging.read_bytes() != payload:
            raise BlindedEvaluationError("score-lock staging verification failed")
        os.link(staging, target, follow_symlinks=False)
        if target.read_bytes() != payload:
            raise BlindedEvaluationError("score-lock publication verification failed")
        staging.unlink()
        parent_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        staging.unlink(missing_ok=True)


def _bundle_identity_payload(bundle: BlindedReviewBundle) -> dict[str, Any]:
    return {
        "bundle_id": bundle.bundle_id,
        **bundle.digest_receipt(),
    }


def _package_identity_payload(
    *,
    task_id: str,
    task_split: str,
    sheet_sha256: str,
    bundles: Sequence[BlindedReviewBundle],
) -> dict[str, Any]:
    return {
        "schema_version": "easyicu.figure2_blinded_review_package/1",
        "task_id": task_id,
        "task_split": task_split,
        "sheet_sha256": sheet_sha256,
        "arm_mapping_present": False,
        "bundles": [_bundle_identity_payload(bundle) for bundle in bundles],
    }


def _seal_normalized_bundle(
    bundle_id: str,
    normalized: NormalizedReviewBundle,
) -> BlindedReviewBundle:
    if bundle_id not in _BLINDED_BUNDLE_IDS:
        raise BlindedEvaluationError(f"unknown blinded bundle: {bundle_id}")
    if not isinstance(normalized, NormalizedReviewBundle):
        raise BlindedEvaluationError("normalizer returned an invalid bundle")
    if (
        set(normalized.files) != set(CANONICAL_FILES)
        or set(normalized.pre_normalization_sha256) != set(CANONICAL_FILES)
        or set(normalized.post_normalization_sha256) != set(CANONICAL_FILES)
    ):
        raise BlindedEvaluationError("normalized bundle file identity is incomplete")
    files = tuple((name, normalized.files[name]) for name in CANONICAL_FILES)
    if any(not isinstance(payload, bytes) for _name, payload in files):
        raise BlindedEvaluationError("normalized review files must be bytes")
    pre_digests = tuple(
        (name, normalized.pre_normalization_sha256[name])
        for name in CANONICAL_FILES
    )
    post_digests = tuple(
        (name, normalized.post_normalization_sha256[name])
        for name in CANONICAL_FILES
    )
    if any(
        not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest)
        for _name, digest in (*pre_digests, *post_digests)
    ):
        raise BlindedEvaluationError("normalized bundle digest is invalid")
    mismatches = [
        name
        for name, payload in files
        if hashlib.sha256(payload).hexdigest()
        != normalized.post_normalization_sha256[name]
    ]
    if mismatches:
        raise BlindedEvaluationError(
            "normalized review bytes do not match post-normalization digests: "
            + ", ".join(mismatches)
        )
    redaction_log: list[tuple[tuple[str, str], ...]] = []
    for item in normalized.redaction_log:
        if not isinstance(item, Mapping) or set(item) != set(
            _REDACTION_LOG_FIELDS
        ):
            raise BlindedEvaluationError("redaction log entry is invalid")
        if any(not isinstance(item[field], str) for field in _REDACTION_LOG_FIELDS):
            raise BlindedEvaluationError("redaction log values must be strings")
        if (
            item["file"] not in CANONICAL_FILES
            or not item["rule"]
            or not item["replacement_count"].isdigit()
            or int(item["replacement_count"]) <= 0
        ):
            raise BlindedEvaluationError("redaction log entry is invalid")
        redaction_log.append(
            tuple((field, item[field]) for field in _REDACTION_LOG_FIELDS)
        )
    frozen_redaction_log = tuple(redaction_log)
    redaction_log_sha256 = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": "easyicu.figure2_redaction_log/1",
                "entries": [dict(entry) for entry in frozen_redaction_log],
            }
        )
    ).hexdigest()
    bundle_digest = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": "easyicu.figure2_normalized_review_bundle/1",
                "bundle_id": bundle_id,
                "post_normalization_sha256": dict(post_digests),
                "redaction_log_sha256": redaction_log_sha256,
            }
        )
    ).hexdigest()
    return BlindedReviewBundle(
        bundle_id=bundle_id,
        _files=files,
        pre_normalization_sha256=pre_digests,
        post_normalization_sha256=post_digests,
        _redaction_log=frozen_redaction_log,
        redaction_log_sha256=redaction_log_sha256,
        normalized_bundle_sha256=bundle_digest,
    )


def _validate_review_package(review_package: BlindedReviewPackage) -> None:
    if not isinstance(review_package, BlindedReviewPackage):
        raise BlindedEvaluationError(
            "score lock requires a BlindedReviewPackage"
        )
    sheet = _instantiate_review_sheet(
        review_package.task_id,
        task_split=review_package.task_split,
    )
    expected_sheet_sha256 = sheet.pop("sheet_sha256")
    expected_sheet = _canonical_json(sheet)
    if (
        review_package.sheet_sha256 != expected_sheet_sha256
        or review_package._sheet != expected_sheet
        or hashlib.sha256(review_package._sheet).hexdigest()
        != review_package.sheet_sha256
    ):
        raise BlindedEvaluationError("review package sheet digest mismatch")
    if tuple(bundle.bundle_id for bundle in review_package.bundles) != (
        _BLINDED_BUNDLE_IDS
    ):
        raise BlindedEvaluationError("review package bundle identities are invalid")
    resealed = tuple(
        _seal_normalized_bundle(
            bundle.bundle_id,
            NormalizedReviewBundle(
                files=bundle.files_for_review(),
                pre_normalization_sha256=dict(
                    bundle.pre_normalization_sha256
                ),
                post_normalization_sha256=dict(
                    bundle.post_normalization_sha256
                ),
                redaction_log=bundle.redaction_log_for_audit(),
            ),
        )
        for bundle in review_package.bundles
    )
    if resealed != review_package.bundles:
        raise BlindedEvaluationError("review package bundle seal is invalid")
    if any(
        _source_task_id(bundle.files_for_review()) != review_package.task_id
        for bundle in review_package.bundles
    ):
        raise BlindedEvaluationError("review package source task identity mismatch")
    identity = _package_identity_payload(
        task_id=review_package.task_id,
        task_split=review_package.task_split,
        sheet_sha256=review_package.sheet_sha256,
        bundles=review_package.bundles,
    )
    if hashlib.sha256(_canonical_json(identity)).hexdigest() != (
        review_package.review_package_sha256
    ):
        raise BlindedEvaluationError("review package digest is invalid")


def prepare_blinded_review_package(
    *,
    task_id: str,
    task_split: str = "heldout27",
    source_dirs: Mapping[str, Path],
    blinding_context: ReviewBlindingContext,
) -> BlindedReviewPackage:
    """Normalize and seal the exact pair of bytes supplied to reviewers."""

    if set(source_dirs) != set(_BLINDED_BUNDLE_IDS):
        raise BlindedEvaluationError(
            "review package must map bundle_1 and bundle_2 exactly"
        )
    resolved_sources = tuple(
        Path(source_dirs[bundle_id]).resolve()
        for bundle_id in _BLINDED_BUNDLE_IDS
    )
    if len(set(resolved_sources)) != 2:
        raise BlindedEvaluationError("blinded bundle sources must be distinct")
    sheet = _instantiate_review_sheet(task_id, task_split=task_split)
    sheet_sha256 = sheet.pop("sheet_sha256")
    sheet_payload = _canonical_json(sheet)
    sealed_bundles: list[BlindedReviewBundle] = []
    for bundle_id in _BLINDED_BUNDLE_IDS:
        normalized = normalize_review_bundle(
            source_dirs[bundle_id],
            blinding_context=blinding_context,
        )
        if _source_task_id(normalized.files) != task_id:
            raise BlindedEvaluationError(
                "source bundle task identity mismatch"
            )
        sealed_bundles.append(_seal_normalized_bundle(bundle_id, normalized))
    bundles = tuple(sealed_bundles)
    identity = _package_identity_payload(
        task_id=task_id,
        task_split=task_split,
        sheet_sha256=sheet_sha256,
        bundles=bundles,
    )
    package = BlindedReviewPackage(
        task_id=task_id,
        task_split=task_split,
        _sheet=sheet_payload,
        sheet_sha256=sheet_sha256,
        bundles=bundles,  # type: ignore[arg-type]
        review_package_sha256=hashlib.sha256(
            _canonical_json(identity)
        ).hexdigest(),
    )
    _validate_review_package(package)
    return package


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


def load_qualification_tasks() -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for line in QUALIFICATION_TASKBANK_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        task_id = task.get("id")
        if not isinstance(task_id, str) or task_id in tasks:
            raise BlindedEvaluationError(
                "Qualification12 taskbank identities are invalid"
            )
        tasks[task_id] = task
    if len(tasks) != 12:
        raise BlindedEvaluationError(
            "Qualification12 must contain exactly 12 tasks"
        )
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
        "task_split": "heldout27",
        "primary_endpoint": rubric["primary_endpoint"],
        "hard_gates": rubric["hard_gates"],
        "secondary_dimensions": rubric["secondary_dimensions"],
        "dimension_scale": rubric["dimension_scale"],
        "task_criteria": task_criteria,
        "arm_identity_visible": False,
    }
    sheet["sheet_sha256"] = hashlib.sha256(_canonical_json(sheet)).hexdigest()
    return sheet


def instantiate_qualification_review_sheet(task_id: str) -> dict[str, Any]:
    """Create the neutral task-contract sheet used by the blinding pilot."""

    task = load_qualification_tasks().get(task_id)
    if task is None:
        raise BlindedEvaluationError(f"unknown Qualification12 task: {task_id}")
    rubric = _load_rubric()
    expected_behavior = task["expected_behavior"]
    if expected_behavior == "bound_result":
        task_contract = (
            "The terminal bundle answers the frozen question with a valid bound "
            "result, explicit denominator, required scientific output, diagnostic, "
            "and no critical semantic error."
        )
    elif expected_behavior == "fail_closed":
        task_contract = (
            "The terminal bundle fails closed without an unsupported result and "
            "identifies the frozen unresolvable capability or data gap."
        )
    else:
        raise BlindedEvaluationError(
            "Qualification12 expected behavior is invalid"
        )
    task_criteria = {
        field: task[field]
        for field in (
            "question",
            "analysis_family",
            "exposure",
            "outcome",
            "time_origin",
            "missingness",
            "expected_behavior",
            "expected_gap_reason",
        )
    }
    sheet = {
        "schema_version": "easyicu.figure2_blinded_review_sheet/1",
        "task_id": task_id,
        "task_split": "qualification12",
        "primary_endpoint": {
            "name": "qualification_task_contract_success",
            "definition": task_contract,
        },
        "hard_gates": [
            {
                "gate_id": "QG01_QUALIFICATION_TASK_CONTRACT",
                "criterion": task_contract,
                "failure_is_critical": True,
            }
        ],
        "secondary_dimensions": rubric["secondary_dimensions"],
        "dimension_scale": rubric["dimension_scale"],
        "task_criteria": task_criteria,
        "arm_identity_visible": False,
    }
    sheet["sheet_sha256"] = hashlib.sha256(_canonical_json(sheet)).hexdigest()
    return sheet


def _instantiate_review_sheet(
    task_id: str,
    *,
    task_split: str,
) -> dict[str, Any]:
    if task_split == "heldout27":
        return instantiate_review_sheet(task_id)
    if task_split == "qualification12":
        return instantiate_qualification_review_sheet(task_id)
    raise BlindedEvaluationError(f"unknown review task split: {task_split}")


def _validate_score(
    score: Mapping[str, Any],
    *,
    review_package: BlindedReviewPackage,
    gate_ids: set[str],
    dimensions: set[str],
) -> None:
    required = {
        "reviewer_id",
        "task_id",
        "bundle_id",
        "review_package_sha256",
        "blinded_bundle_sha256",
        "primary_success",
        "hard_gates_passed",
        "dimension_scores",
        "arm_guess",
        "arm_guess_confidence",
        "rationale",
    }
    if set(score) != required:
        raise BlindedEvaluationError("review score fields do not match the lock schema")
    if score["task_id"] != review_package.task_id:
        raise BlindedEvaluationError("review score task identity mismatch")
    if not isinstance(score["reviewer_id"], str) or not score["reviewer_id"].strip():
        raise BlindedEvaluationError("reviewer_id must be non-empty")
    if score["bundle_id"] not in {"bundle_1", "bundle_2"}:
        raise BlindedEvaluationError("bundle_id must be bundle_1 or bundle_2")
    if score["review_package_sha256"] != review_package.review_package_sha256:
        raise BlindedEvaluationError("review score package identity mismatch")
    bundle_digest = next(
        bundle.normalized_bundle_sha256
        for bundle in review_package.bundles
        if bundle.bundle_id == score["bundle_id"]
    )
    if score["blinded_bundle_sha256"] != bundle_digest:
        raise BlindedEvaluationError("review score bundle identity mismatch")
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
    confidence = score["arm_guess_confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or confidence < 0.0
        or confidence > 1.0
        or not math.isfinite(confidence)
    ):
        raise BlindedEvaluationError(
            "arm_guess_confidence must be a finite number from 0 to 1"
        )
    if not isinstance(score["rationale"], str) or not score["rationale"].strip():
        raise BlindedEvaluationError("review rationale must be non-empty")


def lock_blinded_scores(
    *,
    review_package: BlindedReviewPackage,
    reviews: Sequence[Mapping[str, Any]],
    eligible_reviewer_ids: Sequence[str],
    reviewer_eligibility_receipt_sha256: str,
    destination: Path,
) -> dict[str, Any]:
    """Atomically lock two independent reviews before any arm mapping is supplied."""

    _validate_review_package(review_package)
    task_id = review_package.task_id
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
    sheet = json.loads(review_package.sheet_for_review())
    gate_ids = {item["gate_id"] for item in sheet["hard_gates"]}
    dimensions = set(sheet["secondary_dimensions"])
    identities: set[tuple[str, str]] = set()
    for score in reviews:
        _validate_score(
            score,
            review_package=review_package,
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
            (score["arm_guess"], float(score["arm_guess_confidence"]))
            for score in reviews
            if score["reviewer_id"] == reviewer
        }
        if len(guesses) != 1:
            raise BlindedEvaluationError(
                "each reviewer must lock one consistent arm guess and confidence"
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
        "schema_version": "easyicu.figure2_blinded_score_lock/2",
        "task_id": task_id,
        "task_split": review_package.task_split,
        "sheet_sha256": review_package.sheet_sha256,
        "review_package_sha256": review_package.review_package_sha256,
        "blinded_bundle_digests": review_package.digest_receipt(),
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
    _publish_score_lock(final_payload, destination)
    return receipt


__all__ = [
    "BlindedReviewBundle",
    "BlindedReviewPackage",
    "BlindedEvaluationError",
    "instantiate_review_sheet",
    "instantiate_qualification_review_sheet",
    "load_heldout_tasks",
    "load_qualification_tasks",
    "lock_blinded_scores",
    "prepare_blinded_review_package",
]
