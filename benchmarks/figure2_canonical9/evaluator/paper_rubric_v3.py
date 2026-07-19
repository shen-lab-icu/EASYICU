"""Versioned paper authority for the exact-five Figure 2 matrix.

The previously frozen v1 and v2 authorities remain byte-compatible.  This v3
contract is additive: it introduces the paper-only exact-five payload, a full
research-agent source-tree coordinate for the pre-v1 authority package, and a
frozen evaluator protocol without rewriting historical decoder or manifest
bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from easyicu.research_agent.evaluation_scorecard import (
    DimensionScore,
    FiveDimensionScorecard,
    Tristate,
)

from .rubric_v1 import (
    FIGURE2_DIMENSIONS,
    FIGURE2_TASK_IDS,
    SCORER_BUNDLE_FILES,
    Figure2Thresholds,
    figure2_suite_projection_sha256,
)
from .safety_protocol_v1 import (
    FIGURE2_SAFETY_PROTOCOL_REF,
    safety_protocol_sha256,
)

FIGURE2_PAPER_RUBRIC_REF = "easyicu.figure2_paper_rubric/20260719-v3"
FIGURE2_PAPER_RUBRIC_SCHEMA = "easyicu.figure2_paper_rubric_manifest/3"
FIGURE2_PAPER_SCORECARD_SCHEMA = "easyicu.figure2_scorecard_envelope/3"
SCORER_EVALUATOR_ROOT = "benchmarks/figure2_canonical9/evaluator"
PAPER_SCORER_CORE_FILES = tuple(
    sorted(
        (set(SCORER_BUNDLE_FILES) - {"src/easyicu/research_agent/runtime_artifacts.py"})
        | {
            "src/easyicu/research_agent/authority/evidence_snapshot.py",
            "src/easyicu/research_agent/authority/evidence_store.py",
            "src/easyicu/research_agent/authority/run_input.py",
            "src/easyicu/research_agent/authority/run_lock.py",
            "src/easyicu/research_agent/authority/runtime_artifacts.py",
            "src/easyicu/research_agent/schema.py",
        }
    )
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
DimensionName = Literal[
    "plan",
    "code",
    "result_validity",
    "evidence_binding",
    "audit_conclusion_safety",
]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Figure2PaperDimensionApplicability(_StrictFrozenModel):
    plan: Literal["required"]
    code: Literal["required"]
    result_validity: Literal["conditional"]
    evidence_binding: Literal["required"]
    audit_conclusion_safety: Literal["required"]
    result_validity_condition_code: Literal[
        "GOLD_FREE_VALUE_SIGNAL_OR_LOCKED_REFERENCE"
    ]


class Figure2ValidityBinding(_StrictFrozenModel):
    """Evaluator-only concepts used by the gold-free validity backstop.

    ``None`` is permitted only when the rubric explicitly declares that a
    single exposure or outcome concept is not applicable.  Keeping the
    applicability code separate from the value distinguishes a deliberate N/A
    (for example a predictor-set or clustering task) from an omitted authority
    coordinate.
    """

    exposure_applicability: Literal["required", "not_applicable"]
    exposure_concept: str | None
    outcome_applicability: Literal["required", "not_applicable"]
    outcome_concept: str | None

    @model_validator(mode="after")
    def _validate_applicability(self) -> "Figure2ValidityBinding":
        for role in ("exposure", "outcome"):
            applicability = getattr(self, f"{role}_applicability")
            concept = getattr(self, f"{role}_concept")
            if applicability == "required":
                if not isinstance(concept, str) or not concept.strip():
                    raise ValueError(
                        f"required Figure 2 {role} concept must be nonblank"
                    )
                if concept != concept.strip():
                    raise ValueError(
                        f"Figure 2 {role} concept must not contain edge whitespace"
                    )
            elif concept is not None:
                raise ValueError(
                    f"not-applicable Figure 2 {role} concept must be explicit null"
                )
        return self


class Figure2PaperTaskRubric(_StrictFrozenModel):
    task_id: str = Field(min_length=1, max_length=128)
    dimension_applicability: Figure2PaperDimensionApplicability
    validity_binding: Figure2ValidityBinding
    hazard_codes: tuple[str, ...] = Field(min_length=1)
    forbidden_claim_codes: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_codes(self) -> "Figure2PaperTaskRubric":
        if len(self.hazard_codes) != len(set(self.hazard_codes)):
            raise ValueError("duplicate paper hazard code")
        if len(self.forbidden_claim_codes) != len(set(self.forbidden_claim_codes)):
            raise ValueError("duplicate paper forbidden-claim code")
        return self


class Figure2PaperRubricManifest(_StrictFrozenModel):
    schema_version: Literal["easyicu.figure2_paper_rubric_manifest/3"]
    rubric_ref: Literal["easyicu.figure2_paper_rubric/20260719-v3"]
    audience: Literal["evaluator_only"]
    agent_visibility: Literal["forbidden"]
    suite_ref: Literal["easyicu_evaluation_protocol_suite/v2"]
    suite_projection_sha256: Sha256
    scorer_tree_sha256: Sha256
    scorer_tree_root: Literal["benchmarks/figure2_canonical9/evaluator"]
    scorer_core_files: tuple[str, ...]
    safety_protocol_ref: Literal[
        "easyicu.figure2_safety_adjudicator_protocol/20260718-v1"
    ]
    safety_protocol_sha256: Sha256
    safety_provider_ref: str = Field(min_length=1, max_length=256)
    safety_model_ref: str = Field(min_length=1, max_length=256)
    dimensions: tuple[DimensionName, ...]
    thresholds: Figure2Thresholds
    na_policy: Literal["preserve"]
    aggregation_policy: Literal["none"]
    tasks: tuple[Figure2PaperTaskRubric, ...]

    @model_validator(mode="after")
    def _validate_frozen_shape(self) -> "Figure2PaperRubricManifest":
        if tuple(self.dimensions) != FIGURE2_DIMENSIONS:
            raise ValueError("Figure 2 paper dimensions or order drifted")
        if tuple(task.task_id for task in self.tasks) != FIGURE2_TASK_IDS:
            raise ValueError("Figure 2 paper task IDs or order drifted")
        if self.scorer_core_files != PAPER_SCORER_CORE_FILES:
            raise ValueError("Figure 2 paper scorer core membership drifted")
        return self


class Figure2ExactFiveScorecard(_StrictFrozenModel):
    task_id: str
    run_id: str | None = None
    plan: DimensionScore
    code: DimensionScore
    result_validity: DimensionScore
    evidence_binding: DimensionScore
    audit_conclusion_safety: DimensionScore
    tristate: Tristate

    @model_validator(mode="after")
    def _validate_dimension_values(self) -> "Figure2ExactFiveScorecard":
        dimensions = self.dimensions()
        if self.task_id not in FIGURE2_TASK_IDS:
            raise ValueError("paper scorecard task is outside the frozen suite")
        if tuple(dimension.name for dimension in dimensions) != FIGURE2_DIMENSIONS:
            raise ValueError("paper scorecard dimension identity/order mismatch")
        required = (
            self.plan,
            self.code,
            self.evidence_binding,
            self.audit_conclusion_safety,
        )
        if any(dimension.subscore is None for dimension in required):
            raise ValueError("required paper dimensions cannot be NA")

        level_rank = {"Fail": 0, "Marginal": 1, "Partial": 2, "Full": 3}
        for dimension in dimensions:
            if (dimension.subscore is None) != (dimension.level is None):
                raise ValueError(
                    f"{dimension.name} subscore and level must both be present or both be NA"
                )
            if dimension.subscore is not None and (
                not math.isfinite(dimension.subscore)
                or not 0.0 <= dimension.subscore <= 1.0
            ):
                raise ValueError(f"{dimension.name} subscore is outside [0, 1]")
            if dimension.subscore is not None and dimension.level is not None:
                maximum_level = (
                    "Full"
                    if dimension.subscore >= 0.85
                    else (
                        "Partial"
                        if dimension.subscore >= 0.55
                        else "Marginal" if dimension.subscore >= 0.25 else "Fail"
                    )
                )
                if level_rank[dimension.level] > level_rank[maximum_level]:
                    raise ValueError(
                        f"{dimension.name} level is more favorable than its subscore"
                    )
        blocking_dimensions = (
            self.plan,
            self.code,
            self.evidence_binding,
            self.audit_conclusion_safety,
        )
        result_validity_blocks = (
            self.result_validity.subscore is not None
            and self.result_validity.level == "Fail"
        )
        if self.tristate == "gate_reportable" and (
            any(dimension.level == "Fail" for dimension in blocking_dimensions)
            or result_validity_blocks
        ):
            raise ValueError("gate_reportable contradicts a blocking paper dimension")
        return self

    def dimensions(self) -> tuple[DimensionScore, ...]:
        return (
            self.plan,
            self.code,
            self.result_validity,
            self.evidence_binding,
            self.audit_conclusion_safety,
        )


class Figure2PaperScorecard(_StrictFrozenModel):
    """Structurally valid exact-five payload; authenticity is externally replayed."""

    schema_version: Literal["easyicu.figure2_scorecard_envelope/3"]
    rubric_ref: Literal["easyicu.figure2_paper_rubric/20260719-v3"]
    rubric_manifest_sha256: Sha256
    suite_projection_sha256: Sha256
    scorer_tree_sha256: Sha256
    task_id: str
    aggregation_policy: Literal["none"]
    na_policy: Literal["preserve"]
    scorecard_sha256: Sha256
    scorecard_canonical_json: str = Field(min_length=2)

    @model_validator(mode="after")
    def _validate_payload(self) -> "Figure2PaperScorecard":
        payload = _strict_json_loads(self.scorecard_canonical_json.encode("utf-8"))
        scorecard = Figure2ExactFiveScorecard.model_validate(payload, strict=True)
        canonical = _canonical_json_bytes(scorecard.model_dump(mode="json"))
        if canonical.decode("utf-8") != self.scorecard_canonical_json:
            raise ValueError("paper scorecard payload is not canonical JSON")
        if _sha256_bytes(canonical) != self.scorecard_sha256:
            raise ValueError("paper scorecard payload digest mismatch")
        if self.task_id not in FIGURE2_TASK_IDS or scorecard.task_id != self.task_id:
            raise ValueError("paper scorecard task identity mismatch")
        if tuple(item.name for item in scorecard.dimensions()) != FIGURE2_DIMENSIONS:
            raise ValueError("paper scorecard dimension identity/order mismatch")
        manifest = load_figure2_paper_rubric()
        if self.rubric_manifest_sha256 != paper_rubric_manifest_sha256(manifest):
            raise ValueError("paper scorecard rubric authority mismatch")
        if self.suite_projection_sha256 != manifest.suite_projection_sha256:
            raise ValueError("paper scorecard suite authority mismatch")
        if self.scorer_tree_sha256 != manifest.scorer_tree_sha256:
            raise ValueError("paper scorecard scorer-tree authority mismatch")
        return self

    def parsed_scorecard(self) -> Figure2ExactFiveScorecard:
        return Figure2ExactFiveScorecard.model_validate_json(
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


def default_figure2_paper_rubric_path() -> Path:
    return Path(__file__).resolve().parents[1] / "figure2_paper_rubric_v3.json"


def scorer_tree_rows() -> list[dict[str, str]]:
    repo_root = Path(__file__).resolve().parents[3]
    evaluator_root = repo_root / SCORER_EVALUATOR_ROOT
    paths = [
        path for path in evaluator_root.rglob("*.py") if "__pycache__" not in path.parts
    ]
    paths.extend(repo_root / logical_path for logical_path in PAPER_SCORER_CORE_FILES)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in sorted(paths, key=lambda item: item.relative_to(repo_root).as_posix()):
        logical_path = path.relative_to(repo_root).as_posix()
        if logical_path in seen:
            raise ValueError(f"duplicate Figure 2 scorer source: {logical_path}")
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"invalid Figure 2 scorer source: {path}")
        seen.add(logical_path)
        rows.append(
            {
                "path": logical_path,
                "sha256": _sha256_bytes(path.read_bytes()),
            }
        )
    return rows


def scorer_tree_sha256() -> str:
    return _sha256_bytes(_canonical_json_bytes(scorer_tree_rows()))


def verify_figure2_paper_rubric(manifest: Figure2PaperRubricManifest) -> None:
    if manifest.suite_projection_sha256 != figure2_suite_projection_sha256():
        raise ValueError("Figure 2 paper suite projection digest mismatch")
    if manifest.scorer_tree_sha256 != scorer_tree_sha256():
        raise ValueError("Figure 2 paper scorer tree digest mismatch")
    if manifest.safety_protocol_ref != FIGURE2_SAFETY_PROTOCOL_REF:
        raise ValueError("Figure 2 paper safety protocol reference mismatch")
    if manifest.safety_protocol_sha256 != safety_protocol_sha256():
        raise ValueError("Figure 2 paper safety protocol digest mismatch")


def load_figure2_paper_rubric(
    path: Path | str | None = None,
) -> Figure2PaperRubricManifest:
    rubric_path = (
        Path(path) if path is not None else default_figure2_paper_rubric_path()
    )
    parsed = _strict_json_loads(rubric_path.read_bytes())
    manifest = Figure2PaperRubricManifest.model_validate_json(
        _canonical_json_bytes(parsed), strict=True
    )
    verify_figure2_paper_rubric(manifest)
    return manifest


def paper_rubric_manifest_sha256(manifest: Figure2PaperRubricManifest) -> str:
    return _sha256_bytes(_canonical_json_bytes(manifest.model_dump(mode="json")))


def build_figure2_paper_scorecard(
    scorecard: FiveDimensionScorecard,
) -> Figure2PaperScorecard:
    manifest = load_figure2_paper_rubric()
    exact = Figure2ExactFiveScorecard(
        task_id=scorecard.task_id,
        run_id=scorecard.run_id,
        plan=scorecard.plan,
        code=scorecard.code,
        result_validity=scorecard.result_validity,
        evidence_binding=scorecard.evidence_binding,
        audit_conclusion_safety=scorecard.audit_conclusion_safety,
        tristate=scorecard.tristate,
    )
    payload = _canonical_json_bytes(exact.model_dump(mode="json"))
    return Figure2PaperScorecard(
        schema_version=FIGURE2_PAPER_SCORECARD_SCHEMA,
        rubric_ref=FIGURE2_PAPER_RUBRIC_REF,
        rubric_manifest_sha256=paper_rubric_manifest_sha256(manifest),
        suite_projection_sha256=manifest.suite_projection_sha256,
        scorer_tree_sha256=manifest.scorer_tree_sha256,
        task_id=scorecard.task_id,
        aggregation_policy="none",
        na_policy="preserve",
        scorecard_sha256=_sha256_bytes(payload),
        scorecard_canonical_json=payload.decode("utf-8"),
    )


__all__ = [
    "FIGURE2_PAPER_RUBRIC_REF",
    "FIGURE2_PAPER_RUBRIC_SCHEMA",
    "FIGURE2_PAPER_SCORECARD_SCHEMA",
    "PAPER_SCORER_CORE_FILES",
    "SCORER_EVALUATOR_ROOT",
    "Figure2ExactFiveScorecard",
    "Figure2PaperDimensionApplicability",
    "Figure2PaperRubricManifest",
    "Figure2PaperScorecard",
    "Figure2PaperTaskRubric",
    "Figure2ValidityBinding",
    "build_figure2_paper_scorecard",
    "default_figure2_paper_rubric_path",
    "load_figure2_paper_rubric",
    "paper_rubric_manifest_sha256",
    "scorer_tree_rows",
    "scorer_tree_sha256",
]
