# V4 engineering candidate; original v3 remains unchanged and rejects this tree.
"""Independent current-source Figure 2 engineering authority.

The original v3 manifests and owners remain unchanged. V4 uses distinct
artifact schemas, binds the full current implementation, and exposes no
Provider transport or formal-experiment acceptance route.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from easyicu.research_agent.evaluation_scorecard import (
    FiveDimensionScorecard,
)

from ..evaluator.paper_rubric_v3 import (
    Figure2ExactFiveScorecard,
    Figure2PaperDimensionApplicability,
    Figure2PaperTaskRubric,
    Figure2ValidityBinding,
)

from ..evaluator.rubric_v1 import (
    FIGURE2_DIMENSIONS,
    FIGURE2_TASK_IDS,
    Figure2Thresholds,
    figure2_suite_projection_sha256,
)
from ..evaluator.safety_protocol_v1 import (
    FIGURE2_SAFETY_PROTOCOL_REF,
    safety_protocol_sha256,
)

FIGURE2_PAPER_RUBRIC_REF = "easyicu.figure2_paper_rubric/20261101-v4"
FIGURE2_PAPER_RUBRIC_SCHEMA = "easyicu.figure2_paper_rubric_manifest/4"
FIGURE2_PAPER_SCORECARD_SCHEMA = "easyicu.figure2_scorecard_envelope/4"
SCORER_EVALUATOR_ROOT = "benchmarks/figure2_canonical9/successor_v4"
# The v3 12-core facade list is insufficient after owner relocation.  This
# candidate binds every EasyICU Python owner, packaged authority data, and the
# complete Canonical9 implementation/configuration.  The manifest itself is
# excluded to avoid a self-digest; its canonical bytes are bound by envelopes.
_MANIFEST_RELATIVE = "benchmarks/figure2_canonical9/figure2_paper_rubric_v4.json"
_EXCLUDED_DIRECTORIES = {"__pycache__", "node_modules"}


def _source_paths(root: Path, *, suffixes: tuple[str, ...]) -> set[Path]:
    repository = Path(__file__).resolve().parents[3]
    try:
        relative = root.relative_to(repository)
    except ValueError as error:
        raise ValueError("Figure 2 v4 source root escapes the repository") from error
    if ".." in relative.parts:
        raise ValueError("Figure 2 v4 source root escapes the repository")
    cursor = repository
    for part in relative.parts:
        cursor = cursor / part
        if not stat.S_ISDIR(cursor.lstat().st_mode):
            raise ValueError("Figure 2 v4 source root must be a real directory")
    result: set[Path] = set()
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(name for name in directories if name not in _EXCLUDED_DIRECTORIES)
        for name in directories:
            if (Path(current) / name).is_symlink():
                raise ValueError("Figure 2 v4 source directory cannot be a symlink")
        for name in files:
            path = Path(current) / name
            if path.suffix not in suffixes:
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError("Figure 2 v4 source must be a regular file")
            result.add(path)
    return result


def current_core_files() -> tuple[str, ...]:
    root = Path(__file__).resolve().parents[3]
    package = root / "src/easyicu"
    benchmark = root / "benchmarks/figure2_canonical9"
    paths = _source_paths(package, suffixes=(".py",))
    paths |= _source_paths(package / "data", suffixes=(".json",))
    paths |= _source_paths(benchmark, suffixes=(".py", ".json"))
    return tuple(sorted(
        path.relative_to(root).as_posix() for path in paths
        if path.relative_to(root).as_posix() != _MANIFEST_RELATIVE
        and not (path.suffix == ".py" and path.is_relative_to(root / SCORER_EVALUATOR_ROOT))
    ))


PAPER_SCORER_CORE_FILES = current_core_files()


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


class Figure2PaperRubricManifest(_StrictFrozenModel):
    schema_version: Literal["easyicu.figure2_paper_rubric_manifest/4"]
    authority_status: Literal["engineering_candidate_no_formal_authority"]
    rubric_ref: Literal["easyicu.figure2_paper_rubric/20261101-v4"]
    audience: Literal["evaluator_only"]
    agent_visibility: Literal["forbidden"]
    suite_ref: Literal["easyicu_evaluation_protocol_suite/v2"]
    suite_projection_sha256: Sha256
    scorer_tree_sha256: Sha256
    scorer_tree_root: Literal["benchmarks/figure2_canonical9/successor_v4"]
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
        if self.scorer_core_files != current_core_files():
            raise ValueError("Figure 2 paper scorer core membership drifted")
        return self


class Figure2PaperScorecard(_StrictFrozenModel):
    """Structurally valid exact-five payload; authenticity is externally replayed."""

    schema_version: Literal["easyicu.figure2_scorecard_envelope/4"]
    rubric_ref: Literal["easyicu.figure2_paper_rubric/20261101-v4"]
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
    return Path(__file__).resolve().parents[1] / "figure2_paper_rubric_v4.json"


def scorer_tree_rows() -> list[dict[str, str]]:
    repo_root = Path(__file__).resolve().parents[3]
    evaluator_root = repo_root / SCORER_EVALUATOR_ROOT
    paths = [
        path for path in evaluator_root.rglob("*.py") if "__pycache__" not in path.parts
    ]
    paths.extend(repo_root / logical_path for logical_path in current_core_files())
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
    "current_core_files",
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
