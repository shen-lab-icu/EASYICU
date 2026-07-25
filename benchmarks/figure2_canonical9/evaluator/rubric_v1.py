"""Versioned, evaluator-only rubric authority for the Figure 2 Canonical9 panel.

This module deliberately sits outside the Planner/Coder control plane.  It
describes how completed runs are *evaluated*; it must never be included in an
agent prompt, used to choose an estimand, or exposed as an answer key.  The
manifest contains coded hazards and forbidden-claim classes only.  It contains
no numeric gold values and no expected effect direction.

The manifest does not self-embed its own digest.  Callers compute that digest
over the strict model's canonical JSON with :func:`rubric_manifest_sha256` and
store it in the paper-facing scorecard envelope.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from easyicu.research_agent.evaluation_scorecard import FiveDimensionScorecard
from .suite import easyicu_evaluation_protocol_suite

FIGURE2_RUBRIC_REF = "easyicu.figure2_rubric/20260718-v1"
FIGURE2_RUBRIC_SCHEMA = "easyicu.figure2_rubric_manifest/1"
FIGURE2_SCORECARD_ENVELOPE_SCHEMA = "easyicu.figure2_scorecard_envelope/1"

FIGURE2_DIMENSIONS = (
    "plan",
    "code",
    "result_validity",
    "evidence_binding",
    "audit_conclusion_safety",
)

FIGURE2_TASK_IDS = (
    "e1_sepsis3_prevalence_mortality",
    "e2_lactate_mortality",
    "e3_kdigo_gradient",
    "m1_hepatobiliary_missingness",
    "m2_mortality_prediction",
    "m3_sepsis_subphenotype",
    "h1_ventilation_survival",
    "h2_vasopressor_causal",
    "h3_trajectory_clustering",
)

# Logical paths are part of the scorer authority.  The bundle digest hashes a
# canonical list of {path, sha256} rows, so both file membership and bytes are
# reproducible.  Keep this evaluator-only list explicit rather than allowing a
# manifest to choose a weaker subset.
SCORER_BUNDLE_FILES = (
    "src/easyicu/research_agent/evaluation_scorecard.py",
    "src/easyicu/research_agent/icu_agent_bench.py",
    "src/easyicu/research_agent/icu_rules.py",
    "src/easyicu/research_agent/plan_utils.py",
    "src/easyicu/research_agent/runtime_artifacts.py",
    "src/easyicu/research_agent/validity_signals.py",
    "src/easyicu/research_agent/viability.py",
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
HazardCode = Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]{2,127}$")]
DimensionName = Literal[
    "plan",
    "code",
    "result_validity",
    "evidence_binding",
    "audit_conclusion_safety",
]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Figure2Thresholds(_StrictFrozenModel):
    full: float
    partial: float
    marginal: float

    @model_validator(mode="after")
    def _validate_frozen_values(self) -> "Figure2Thresholds":
        if (self.full, self.partial, self.marginal) != (0.85, 0.55, 0.25):
            raise ValueError("Figure 2 thresholds must remain 0.85/0.55/0.25")
        return self


class Figure2DimensionApplicability(_StrictFrozenModel):
    plan: Literal["required"]
    code: Literal["required"]
    result_validity: Literal["conditional"]
    evidence_binding: Literal["required"]
    audit_conclusion_safety: Literal["required"]
    result_validity_condition_code: Literal["LOCKED_REFERENCE_REQUIRED"]


class Figure2TaskRubric(_StrictFrozenModel):
    task_id: str = Field(min_length=1, max_length=128)
    dimension_applicability: Figure2DimensionApplicability
    hazard_codes: tuple[HazardCode, ...] = Field(min_length=1)
    forbidden_claim_codes: tuple[HazardCode, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_codes(self) -> "Figure2TaskRubric":
        if len(self.hazard_codes) != len(set(self.hazard_codes)):
            raise ValueError(f"duplicate hazard code for {self.task_id}")
        if len(self.forbidden_claim_codes) != len(set(self.forbidden_claim_codes)):
            raise ValueError(f"duplicate forbidden-claim code for {self.task_id}")
        return self


class Figure2RubricManifest(_StrictFrozenModel):
    schema_version: Literal["easyicu.figure2_rubric_manifest/1"]
    rubric_ref: Literal["easyicu.figure2_rubric/20260718-v1"]
    audience: Literal["evaluator_only"]
    agent_visibility: Literal["forbidden"]
    suite_ref: Literal["easyicu_evaluation_protocol_suite/v2"]
    suite_projection_sha256: Sha256
    scorer_bundle_sha256: Sha256
    scorer_files: tuple[str, ...]
    dimensions: tuple[DimensionName, ...]
    thresholds: Figure2Thresholds
    na_policy: Literal["preserve"]
    aggregation_policy: Literal["none"]
    tasks: tuple[Figure2TaskRubric, ...]

    @model_validator(mode="after")
    def _validate_frozen_shape(self) -> "Figure2RubricManifest":
        if tuple(self.dimensions) != FIGURE2_DIMENSIONS:
            raise ValueError("Figure 2 dimensions or their order drifted")
        if tuple(self.scorer_files) != SCORER_BUNDLE_FILES:
            raise ValueError("Figure 2 scorer bundle membership drifted")
        task_ids = tuple(task.task_id for task in self.tasks)
        if task_ids != FIGURE2_TASK_IDS:
            raise ValueError("Figure 2 task IDs or their order drifted")
        return self


class Figure2ScorecardEnvelope(_StrictFrozenModel):
    """Paper-facing authority wrapper for one task's five-dimension scorecard.

    There is intentionally no suite aggregate here.  NA cells remain NA and no
    averaging or imputation is authorised by this schema.
    """

    schema_version: Literal["easyicu.figure2_scorecard_envelope/1"]
    rubric_ref: Literal["easyicu.figure2_rubric/20260718-v1"]
    rubric_manifest_sha256: Sha256
    suite_projection_sha256: Sha256
    scorer_bundle_sha256: Sha256
    task_id: str
    aggregation_policy: Literal["none"]
    na_policy: Literal["preserve"]
    scorecard_sha256: Sha256
    scorecard_canonical_json: str = Field(min_length=2)

    @model_validator(mode="after")
    def _validate_scorecard_identity(self) -> "Figure2ScorecardEnvelope":
        try:
            payload = _strict_json_loads(self.scorecard_canonical_json.encode("utf-8"))
            scorecard = FiveDimensionScorecard.model_validate(payload, strict=True)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("scorecard_canonical_json is invalid") from exc
        canonical = _canonical_json_bytes(scorecard.model_dump(mode="json"))
        if canonical.decode("utf-8") != self.scorecard_canonical_json:
            raise ValueError("scorecard payload is not canonical JSON")
        if _sha256_bytes(canonical) != self.scorecard_sha256:
            raise ValueError("scorecard payload digest mismatch")
        if self.task_id not in FIGURE2_TASK_IDS:
            raise ValueError("scorecard task is outside the frozen Figure 2 suite")
        if scorecard.task_id != self.task_id:
            raise ValueError("scorecard task_id does not match envelope task_id")
        names = tuple(dimension.name for dimension in scorecard.dimensions())
        if names != FIGURE2_DIMENSIONS:
            raise ValueError("scorecard dimension names/order do not match the rubric")
        return self

    def validated_scorecard(self) -> FiveDimensionScorecard:
        """Return a newly validated view without exposing mutable authority state."""

        return FiveDimensionScorecard.model_validate_json(
            self.scorecard_canonical_json, strict=True
        )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json_loads(payload: bytes) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )


def figure2_suite_projection() -> dict[str, object]:
    """Return the evaluator-relevant, answer-free projection of Canonical9.

    Numeric gold answers are deliberately excluded.  ``has_gold_answer`` and
    ``gold_answer_status`` still make a future reference freeze change this
    authority digest without exposing the reference value or effect direction.
    """

    suite = easyicu_evaluation_protocol_suite()
    return {
        "schema_version": suite.schema_version,
        "name": suite.name,
        "maturity": suite.maturity,
        "metrics": suite.metrics.model_dump(mode="json"),
        "tasks": [
            {
                "task_id": task.task_id,
                "kind": task.kind,
                "title": task.title,
                "objective": task.objective,
                "expected_outputs": list(task.expected_outputs),
                "semantic_guardrails": list(task.semantic_guardrails),
                "evaluation_notes": list(task.evaluation_notes),
                "target_databases": list(task.target_databases),
                "gold_answer_status": task.gold_answer_status,
                "has_gold_answer": task.gold_answer is not None,
                "difficulty": task.difficulty,
                "category": task.category,
            }
            for task in suite.tasks
        ],
    }


def figure2_suite_projection_sha256() -> str:
    return _sha256_bytes(_canonical_json_bytes(figure2_suite_projection()))


def scorer_bundle_rows() -> list[dict[str, str]]:
    """Return the immutable historical v1 scorer source-digest rows.

    Version 1 bound the then-current installed source paths.  The active tool
    has since evolved and the paper-only Canonical9 factory moved out of the
    wheel.  Keep the historical logical paths and digests byte-for-byte frozen
    in a repository benchmark asset instead of either mutating the v1 manifest
    or pretending current source bytes still implement that archived scorer.
    """

    rows_path = (
        Path(__file__).resolve().parents[1]
        / "frozen"
        / "v1"
        / "scorer_bundle_rows.json"
    )
    raw = rows_path.read_bytes()
    payload = _strict_json_loads(raw)
    if raw != _canonical_json_bytes(payload) + b"\n":
        raise ValueError("historical v1 scorer rows must use canonical JSON bytes")
    if not isinstance(payload, list):
        raise ValueError("historical v1 scorer rows must be a list")
    rows: list[dict[str, str]] = []
    for row in payload:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError("historical v1 scorer row has an invalid shape")
        path = row.get("path")
        digest = row.get("sha256")
        if (
            not isinstance(path, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("historical v1 scorer row has invalid coordinates")
        rows.append({"path": path, "sha256": digest})
    if tuple(row["path"] for row in rows) != SCORER_BUNDLE_FILES:
        raise ValueError("historical v1 scorer row membership/order drifted")
    return rows


def scorer_bundle_sha256() -> str:
    return _sha256_bytes(_canonical_json_bytes(scorer_bundle_rows()))


def default_figure2_rubric_path() -> Path:
    return Path(__file__).resolve().parents[1] / "figure2_rubric_v1.json"


def verify_figure2_rubric(manifest: Figure2RubricManifest) -> None:
    """Verify the manifest against the live suite and scorer source authority."""

    runtime_task_ids = tuple(
        task.task_id for task in easyicu_evaluation_protocol_suite().tasks
    )
    if runtime_task_ids != FIGURE2_TASK_IDS:
        raise ValueError("runtime Canonical9 task identity/order drifted")
    observed_suite_sha = figure2_suite_projection_sha256()
    if manifest.suite_projection_sha256 != observed_suite_sha:
        raise ValueError(
            "Figure 2 suite projection digest mismatch: "
            f"manifest={manifest.suite_projection_sha256} runtime={observed_suite_sha}"
        )
    observed_scorer_sha = scorer_bundle_sha256()
    if manifest.scorer_bundle_sha256 != observed_scorer_sha:
        raise ValueError(
            "Figure 2 scorer bundle digest mismatch: "
            f"manifest={manifest.scorer_bundle_sha256} runtime={observed_scorer_sha}"
        )


def load_figure2_rubric(
    path: Path | str | None = None,
) -> Figure2RubricManifest:
    """Strictly load and verify the evaluator-only rubric manifest."""

    manifest_path = Path(path) if path is not None else default_figure2_rubric_path()
    raw = manifest_path.read_bytes()
    payload = _strict_json_loads(raw)
    # JSON arrays are valid immutable tuple inputs under Pydantic's JSON-mode
    # strict validation.  Canonicalising the duplicate-checked Python value back
    # to JSON gives us both duplicate-key rejection and deeply immutable tuples.
    manifest = Figure2RubricManifest.model_validate_json(
        _canonical_json_bytes(payload), strict=True
    )
    verify_figure2_rubric(manifest)
    return manifest


def rubric_manifest_sha256(manifest: Figure2RubricManifest) -> str:
    """Compute the external digest of a validated manifest's canonical JSON."""

    return _sha256_bytes(_canonical_json_bytes(manifest.model_dump(mode="json")))


def build_figure2_scorecard_envelope(
    scorecard: FiveDimensionScorecard,
    *,
    rubric_path: Path | str | None = None,
) -> Figure2ScorecardEnvelope:
    """Bind one completed scorecard to the frozen rubric/scorer authorities."""

    manifest = load_figure2_rubric(rubric_path)
    task_ids = {task.task_id for task in manifest.tasks}
    if scorecard.task_id not in task_ids:
        raise ValueError("scorecard task is outside the loaded Figure 2 rubric")
    scorecard_bytes = _canonical_json_bytes(scorecard.model_dump(mode="json"))
    return Figure2ScorecardEnvelope(
        schema_version=FIGURE2_SCORECARD_ENVELOPE_SCHEMA,
        rubric_ref=FIGURE2_RUBRIC_REF,
        rubric_manifest_sha256=rubric_manifest_sha256(manifest),
        suite_projection_sha256=manifest.suite_projection_sha256,
        scorer_bundle_sha256=manifest.scorer_bundle_sha256,
        task_id=scorecard.task_id,
        aggregation_policy=manifest.aggregation_policy,
        na_policy=manifest.na_policy,
        scorecard_sha256=_sha256_bytes(scorecard_bytes),
        scorecard_canonical_json=scorecard_bytes.decode("utf-8"),
    )


__all__ = [
    "FIGURE2_DIMENSIONS",
    "FIGURE2_RUBRIC_REF",
    "FIGURE2_RUBRIC_SCHEMA",
    "FIGURE2_SCORECARD_ENVELOPE_SCHEMA",
    "FIGURE2_TASK_IDS",
    "SCORER_BUNDLE_FILES",
    "Figure2DimensionApplicability",
    "Figure2RubricManifest",
    "Figure2ScorecardEnvelope",
    "Figure2TaskRubric",
    "Figure2Thresholds",
    "build_figure2_scorecard_envelope",
    "default_figure2_rubric_path",
    "figure2_suite_projection",
    "figure2_suite_projection_sha256",
    "load_figure2_rubric",
    "rubric_manifest_sha256",
    "scorer_bundle_rows",
    "scorer_bundle_sha256",
    "verify_figure2_rubric",
]
