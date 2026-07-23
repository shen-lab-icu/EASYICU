"""Fail-closed input authority for paper-facing Figure 2 evaluation.

The paper evaluator must not trust mutable run-root summaries.  This module
binds one selected run checkpoint to one selected EvidenceStore generation,
verifies the complete evidence-coordinate multiset, and materializes every
scorer input from digest-verified current evidence.  Scoring itself is
read-only; the small posthoc sealer at the bottom publishes an immutable,
content-addressed evaluator sidecar without mutating Agent checkpoints.
"""

from __future__ import annotations

import csv
import copy
import hashlib
import io
import json
import os
import re
import secrets
import stat
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from easyicu.research_agent.authority.evidence_snapshot import (
    load_current_evidence_snapshot,
)
from easyicu.research_agent.authority.runtime_artifacts import (
    current_evidence_records,
    current_successful_step_records,
    load_run_artifact_authority,
    verified_run_evidence_path,
)
from easyicu.research_agent.authority.run_input import (
    RUN_INPUT_CAPSULE_EVIDENCE_ID,
    RUN_INPUT_CAPSULE_SCHEMA_VERSION_V2,
    RUN_INPUT_CAPSULE_SCHEMA_VERSION_V3,
    RunInputCapsule,
    RunInputCapsuleV2,
    RunInputCapsuleV3,
    canonical_sha256,
    load_verified_run_input_capsule,
)
from easyicu.research_agent.authority.run_lock import acquire_run_execution_lock
from easyicu.research_agent.schema import AnalysisManifest, AnalysisPlan

from .paper_rubric_v3 import (
    FIGURE2_PAPER_RUBRIC_REF,
    load_figure2_paper_rubric,
    paper_rubric_manifest_sha256,
)
from .rubric_v1 import (
    FIGURE2_TASK_IDS,
    figure2_suite_projection,
    figure2_suite_projection_sha256,
)
from .input_binding_v2 import (
    CANONICAL_RUN_INPUT_BINDING_REF,
    FrozenTypedAuthorityRef,
    ReadyCanonicalTaskBinding,
    require_ready_task_binding,
)

FIGURE2_SCORING_INPUT_AUTHORITY_SCHEMA = "easyicu.figure2_scoring_input_authority/5"
FIGURE2_RUN_TASK_AUTHORITY_SCHEMA = "easyicu.figure2_run_task_authority/4"
FIGURE2_SUITE_REF = "easyicu_evaluation_protocol_suite/v2"
FIGURE2_TASK_AUTHORITY_MAX_BYTES = 64 * 1024
FIGURE2_SCORING_ARTIFACT_ROLES = (
    "run_status",
    "analysis_plan",
    "evidence_audit",
    "numeric_audit",
    "claim_ledger",
    "manuscript_ready",
)
FIGURE2_REVIEW_DOCUMENT_MAX_BYTES = 1 * 1024 * 1024
FIGURE2_REVIEW_CORPUS_MAX_BYTES = 8 * 1024 * 1024

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Figure2ArtifactRole = Literal[
    "run_status",
    "analysis_plan",
    "evidence_audit",
    "numeric_audit",
    "claim_ledger",
    "manuscript_ready",
]
ClaimStatus = Literal[
    "bound",
    "ok",
    "verified",
    "demoted",
    "flagged",
    "downgraded",
    "missing_evidence",
    "not_generated",
    "diagnostic_only",
    "empty",
    "blocked_outcome_leak",
]

_PLAN_ID_RE = re.compile(r"analysis_plan_revision_(\d+)(?:_[0-9a-f]{8})?")
_TEXT_SUFFIXES = frozenset({".json", ".csv", ".md", ".txt", ".py"})
_RECORD_COORDINATE_FIELDS = (
    "evidence_id",
    "relative_path",
    "sha256",
    "kind",
    "producer",
    "generation_mode",
    "produced_by_step",
)
_PLAN_PRODUCERS = frozenset({"planner", "replanner", "clinical_skill"})
_PLAN_GENERATION_MODES = frozenset(
    {
        "llm",
        "system",
        "resumed",
        "resumed_planner_migration",
        "deterministic_skill",
        "fallback",
    }
)
_ROLE_CONTRACTS: dict[str, tuple[str, str, str, str]] = {
    "run_status": ("run_status", "log", "pipeline", "system"),
    "evidence_audit": ("evidence_audit", "statistic", "pipeline", "system"),
    "numeric_audit": ("numeric_audit", "statistic", "pipeline", "system"),
    "claim_ledger": ("claim_ledger", "table", "pipeline", "system"),
    "manuscript_ready": ("manuscript_ready", "log", "pipeline", "system"),
}


class Figure2TaskAuthorityMismatch(PermissionError):
    """Requested paper task differs from checkpoint-sealed run authority."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _validate_contained_relative_path(value: str) -> str:
    if not value or "\x00" in value or "\\" in value:
        raise ValueError("evidence relative path is empty or malformed")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("evidence relative path is not contained")
    if len(path.parts) < 2 or path.parts[0] != "evidence":
        raise ValueError("evidence relative path must be rooted under evidence/")
    return value


class Figure2ArtifactAuthority(_StrictFrozenModel):
    """One digest-bound current artifact consumed by the paper scorer."""

    role: Figure2ArtifactRole
    evidence_id: str = Field(min_length=1, max_length=256)
    relative_path: str = Field(min_length=1, max_length=2048)
    sha256: Sha256
    byte_count: int = Field(ge=1)
    kind: str = Field(min_length=1, max_length=64)
    producer: str = Field(min_length=1, max_length=128)
    generation_mode: str = Field(min_length=1, max_length=128)
    produced_by_step: None = None

    @field_validator("relative_path")
    @classmethod
    def _contained_path(cls, value: str) -> str:
        return _validate_contained_relative_path(value)

    @model_validator(mode="after")
    def _validate_role_contract(self) -> "Figure2ArtifactAuthority":
        if self.role == "analysis_plan":
            if (
                self.evidence_id != "analysis_plan"
                and _PLAN_ID_RE.fullmatch(self.evidence_id) is None
            ):
                raise ValueError("analysis-plan evidence identity is invalid")
            if self.kind != "log" or self.producer not in _PLAN_PRODUCERS:
                raise ValueError("analysis-plan evidence metadata is invalid")
            if self.generation_mode not in _PLAN_GENERATION_MODES:
                raise ValueError("analysis-plan generation mode is invalid")
            return self
        expected = _ROLE_CONTRACTS[self.role]
        observed = (
            self.evidence_id,
            self.kind,
            self.producer,
            self.generation_mode,
        )
        if observed != expected:
            raise ValueError(f"{self.role} evidence metadata is invalid")
        return self


class Figure2RunTaskAuthority(_StrictFrozenModel):
    """Host-owned task/rubric/EvidenceStore coordinate sealed in checkpoint."""

    schema_version: Literal["easyicu.figure2_run_task_authority/4"]
    task_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=256)
    checkpoint_sequence: int = Field(ge=1)
    checkpoint_payload_sha256: Sha256
    suite_ref: Literal["easyicu_evaluation_protocol_suite/v2"]
    suite_projection_sha256: Sha256
    paper_rubric_ref: Literal["easyicu.figure2_paper_rubric/20260719-v3"]
    paper_rubric_sha256: Sha256
    research_question_sha256: Sha256
    exposure_concept: str | None = Field(max_length=256)
    outcome_concept: str = Field(min_length=1, max_length=256)
    benchmark_input_binding_sha256: Sha256
    run_input_capsule_sha256: Sha256
    run_scientific_identity_sha256: Sha256
    run_input_capsule_schema_version: Literal[
        "easyicu.run_input_capsule/2",
        "easyicu.run_input_capsule/3",
    ]
    run_primary_exposure: str | None = Field(max_length=256)
    run_target_outcome: str = Field(min_length=1, max_length=256)
    canonical_input_manifest_ref: Literal[
        "figure2_canonical9/run_input_bindings/20260718-v2"
    ]
    canonical_input_manifest_sha256: Sha256
    canonical_input_case_sha256: Sha256
    submission_profile_ref: str = Field(min_length=1, max_length=128)
    concept_dict_sha256: Sha256
    sofa2_dict_sha256: Sha256
    source_cohort_authority_sha256: Sha256
    source_trajectory_authority_sha256: Sha256 | None
    staged_cohort_authority_sha256: Sha256
    staged_trajectory_authority_sha256: Sha256 | None
    run_status_evidence_sha256: Sha256
    readiness_sha256: Sha256
    evidence_generation: int = Field(ge=1)
    evidence_payload_sha256: Sha256

    @field_validator(
        "exposure_concept",
        "outcome_concept",
        "run_primary_exposure",
        "run_target_outcome",
    )
    @classmethod
    def _nonblank_concept(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("task concept coordinates must be canonical nonblank text")
        return value

    @model_validator(mode="after")
    def _validate_frozen_task_surface(self) -> "Figure2RunTaskAuthority":
        if self.task_id not in FIGURE2_TASK_IDS:
            raise ValueError("run task is outside the frozen Figure 2 suite")
        return self


class Figure2ScoringInputAuthority(_StrictFrozenModel):
    """Complete checkpoint/EvidenceStore coordinate for five-dimension scoring."""

    schema_version: Literal["easyicu.figure2_scoring_input_authority/5"]
    task_id: str = Field(min_length=1, max_length=128)
    suite_ref: Literal["easyicu_evaluation_protocol_suite/v2"]
    suite_projection_sha256: Sha256
    paper_rubric_ref: Literal["easyicu.figure2_paper_rubric/20260719-v3"]
    paper_rubric_sha256: Sha256
    research_question_sha256: Sha256
    exposure_concept: str | None = Field(max_length=256)
    outcome_concept: str = Field(min_length=1, max_length=256)
    benchmark_input_binding_sha256: Sha256
    run_input_capsule_sha256: Sha256
    run_scientific_identity_sha256: Sha256
    run_input_capsule_schema_version: Literal[
        "easyicu.run_input_capsule/2",
        "easyicu.run_input_capsule/3",
    ]
    run_primary_exposure: str | None = Field(max_length=256)
    run_target_outcome: str = Field(min_length=1, max_length=256)
    canonical_input_manifest_ref: Literal[
        "figure2_canonical9/run_input_bindings/20260718-v2"
    ]
    canonical_input_manifest_sha256: Sha256
    canonical_input_case_sha256: Sha256
    submission_profile_ref: str = Field(min_length=1, max_length=128)
    concept_dict_sha256: Sha256
    sofa2_dict_sha256: Sha256
    source_cohort_authority_sha256: Sha256
    source_trajectory_authority_sha256: Sha256 | None
    staged_cohort_authority_sha256: Sha256
    staged_trajectory_authority_sha256: Sha256 | None
    run_status_evidence_sha256: Sha256
    readiness_sha256: Sha256
    task_authority_sha256: Sha256
    task_authority_byte_count: int = Field(ge=1, le=FIGURE2_TASK_AUTHORITY_MAX_BYTES)
    run_id: str = Field(min_length=1, max_length=256)
    checkpoint_sequence: int = Field(ge=1)
    checkpoint_payload_sha256: Sha256
    evidence_generation: int = Field(ge=1)
    evidence_payload_sha256: Sha256
    artifacts: tuple[Figure2ArtifactAuthority, ...]

    @field_validator(
        "exposure_concept",
        "outcome_concept",
        "run_primary_exposure",
        "run_target_outcome",
    )
    @classmethod
    def _nonblank_concept(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("task concept coordinates must be canonical nonblank text")
        return value

    @model_validator(mode="after")
    def _validate_exact_roles(self) -> "Figure2ScoringInputAuthority":
        if self.task_id not in FIGURE2_TASK_IDS:
            raise ValueError("scoring task is outside the frozen Figure 2 suite")
        if tuple(artifact.role for artifact in self.artifacts) != (
            FIGURE2_SCORING_ARTIFACT_ROLES
        ):
            raise ValueError("scoring authority must contain the exact ordered roles")
        return self

    def artifact(self, role: Figure2ArtifactRole) -> Figure2ArtifactAuthority:
        return next(item for item in self.artifacts if item.role == role)


class Figure2ReviewDocument(_StrictFrozenModel):
    """One current textual evidence document exposed to the adjudicator."""

    evidence_id: str = Field(min_length=1, max_length=256)
    relative_path: str = Field(min_length=1, max_length=2048)
    sha256: Sha256
    byte_count: int = Field(ge=0, le=FIGURE2_REVIEW_DOCUMENT_MAX_BYTES)
    kind: str = Field(min_length=1, max_length=64)
    producer: str = Field(min_length=1, max_length=128)
    generation_mode: str = Field(min_length=1, max_length=128)
    produced_by_step: str | None = None
    text: str

    @field_validator("relative_path")
    @classmethod
    def _contained_path(cls, value: str) -> str:
        value = _validate_contained_relative_path(value)
        if PurePosixPath(value).suffix.lower() not in _TEXT_SUFFIXES:
            raise ValueError("review document is not a supported textual artifact")
        return value

    @model_validator(mode="after")
    def _validate_text_bytes(self) -> "Figure2ReviewDocument":
        payload = self.text.encode("utf-8")
        if len(payload) != self.byte_count or _sha256_bytes(payload) != self.sha256:
            raise ValueError("review document text does not match its authority")
        return self


@dataclass(frozen=True, slots=True)
class LoadedFigure2ScoringInputs:
    """Verified scorer values and evaluator review corpus for one checkpoint."""

    authority: Figure2ScoringInputAuthority
    gates: dict[str, Any]
    plan_steps: list[dict[str, Any]]
    evidence_audit: dict[str, Any]
    numeric_audit: dict[str, Any]
    claim_rows: list[dict[str, str]]
    claim_reference_sets: tuple[tuple[str, tuple[str, ...]], ...]
    current_step_summaries: tuple[dict[str, Any], ...]
    manuscript_bytes: bytes
    review_documents: tuple[Figure2ReviewDocument, ...]


class _RunStatusDocument(_StrictFrozenModel):
    schema_version: Literal["easyicu.run_status/1"]
    status: Literal[
        "diagnostic_only", "analysis_only", "manuscript_ready", "publication_ready"
    ]
    strict_fail_closed: bool
    writer_probe_mode: bool
    writer_probe_failed_steps: list[str]
    research_question: str = Field(min_length=1)
    code_version: dict[str, Any]
    gates: dict[str, Any]
    canonical_outputs: dict[str, str]

    @field_validator("strict_fail_closed")
    @classmethod
    def _requires_strict_mode(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("run status is not strict fail-closed")
        return value

    @field_validator("gates")
    @classmethod
    def _validate_gates(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _strict_gate_mapping(value)


class _EvidenceAuditDocument(_StrictFrozenModel):
    schema_version: Literal["easyicu.evidence_audit/1"]
    evidence_count: int = Field(ge=0)
    kinds: dict[str, int]
    missing_evidence_count: int = Field(ge=0)
    evidence_complete: bool
    manuscript_path: str | None

    @model_validator(mode="after")
    def _validate_complete_audit(self) -> "_EvidenceAuditDocument":
        if not self.evidence_complete or self.missing_evidence_count != 0:
            raise ValueError("Figure 2 scoring requires a complete evidence audit")
        if any(type(count) is not int or count < 0 for count in self.kinds.values()):
            raise ValueError("evidence audit kind counts must be non-negative integers")
        if sum(self.kinds.values()) != self.evidence_count:
            raise ValueError("evidence audit kind counts are internally inconsistent")
        return self


class _NumericAuditDocument(_StrictFrozenModel):
    schema_version: Literal["easyicu.numeric_audit/1"]
    numeric_verified: bool
    numeric_error_count: int = Field(ge=0)
    numeric_errors: list[str]

    @model_validator(mode="after")
    def _validate_error_count(self) -> "_NumericAuditDocument":
        if self.numeric_error_count != len(self.numeric_errors):
            raise ValueError("numeric audit error count is inconsistent")
        if self.numeric_verified != (self.numeric_error_count == 0):
            raise ValueError("numeric audit verification state is inconsistent")
        return self


_BOOL_GATE_NAMES = frozenset(
    {
        "analysis_validated",
        "article_contract_complete",
        "article_figure_strategy_complete",
        "blocked_outcome_not_leaked",
        "display_absolute_risk_visual_present",
        "display_audit_context_present",
        "display_primary_publication_absolute_risk_visual_present",
        "display_supporting_absolute_risk_visual_present",
        "display_suite_complete",
        "display_table_one_expected",
        "display_table_one_present",
        "evidence_complete",
        "execution_complete",
        "execution_ok",
        "artifact_valid",
        "manuscript_generated",
        "manuscript_ready",
        "manuscript_text_ready",
        "numeric_verified",
        "paper_authorized",
        "publication_figure_bundle_ready",
        "publication_figure_contract_ready",
        "publication_figure_source_data_ready",
        "publication_figure_visual_qa_passed",
        "publication_provenance_ready",
        "publication_ready",
        "replan_budget_advisory",
        "replan_budget_exhausted",
        "replan_budget_hit",
        "scientific_requirement_complete",
        "step_scientific_requirements_complete",
        "writer_probe_mode",
    }
)
_INT_GATE_NAMES = frozenset(
    {
        "analysis_error_count",
        "article_figure_strategy_minimum_distinct_chart_types",
        "article_figure_strategy_primary_publication_minimum_required_role_count",
        "article_figure_strategy_primary_publication_panel_count",
        "blocked_outcome_leak_count",
        "completed_step_count",
        "display_contract_panel_count",
        "display_contract_role_count",
        "display_figure_contract_count",
        "display_other_figure_contract_count",
        "display_primary_publication_figure_contract_count",
        "display_primary_publication_panel_count",
        "display_primary_publication_result_figure_contract_count",
        "display_primary_publication_role_count",
        "display_result_figure_contract_count",
        "display_supporting_figure_contract_count",
        "display_supporting_panel_count",
        "display_supporting_result_figure_contract_count",
        "display_supporting_role_count",
        "display_table_count",
        "evidence_error_count",
        "manuscript_manifest_error_count",
        "manuscript_manifest_warning_count",
        "manuscript_word_count",
        "missing_evidence_count",
        "numeric_error_count",
        "publication_figure_visual_qa_error_count",
        "required_step_count",
        "superseded_error_count",
    }
)
_LIST_GATE_NAMES = frozenset(
    {
        "analysis_errors",
        "article_artifact_roles",
        "article_contract_errors",
        "article_figure_strategy_chart_types",
        "article_figure_strategy_covered_roles",
        "article_figure_strategy_errors",
        "article_figure_strategy_missing_roles",
        "article_figure_strategy_primary_publication_chart_types",
        "article_figure_strategy_primary_publication_roles",
        "article_figure_strategy_required_roles",
        "article_missing_artifact_modules",
        "article_missing_artifact_roles",
        "article_missing_plan_roles",
        "article_plan_roles",
        "article_required_roles",
        "blocked_outcome_leaks",
        "blocked_outcome_step_ids",
        "display_categories",
        "display_chart_types",
        "display_other_figure_contract_paths",
        "display_primary_publication_chart_types",
        "display_primary_publication_contract_paths",
        "display_suite_errors",
        "display_supporting_chart_types",
        "display_supporting_figure_contract_paths",
        "evidence_errors",
        "failed_steps",
        "manuscript_text_errors",
        "missing_steps",
        "numeric_errors",
        "publication_figure_stems",
        "publication_figure_visual_qa_errors",
        "publication_provenance_invalid_sources",
        "publication_ready_stems",
        "scientific_incomplete_steps",
        "step_completion_states",
        "superseded_errors",
        "writer_probe_failed_steps",
    }
)
_DICT_GATE_NAMES = frozenset(
    {
        "article_contract",
        "article_figure_strategy",
        "article_figure_strategy_primary_publication_role_panels",
        "article_figure_strategy_role_panels",
    }
)
_STR_GATE_NAMES = frozenset(
    {
        "article_contract_audit_schema_version",
        "article_contract_family",
        "article_figure_strategy_archetype",
        "article_figure_strategy_audit_schema_version",
        "article_figure_strategy_family",
        "article_figure_strategy_hero_role",
        "completion_schema_version",
    }
)
_NULLABLE_STR_GATE_NAMES = frozenset({"publication_provenance_error"})
_NULLABLE_INT_GATE_NAMES = frozenset({"writer_attempt_count"})
_ALL_GATE_NAMES = (
    _BOOL_GATE_NAMES
    | _INT_GATE_NAMES
    | _LIST_GATE_NAMES
    | _DICT_GATE_NAMES
    | _STR_GATE_NAMES
    | _NULLABLE_STR_GATE_NAMES
    | _NULLABLE_INT_GATE_NAMES
)
_REQUIRED_SCORING_GATES = frozenset(
    {
        "execution_complete",
        "execution_ok",
        "artifact_valid",
        "required_step_count",
        "completed_step_count",
        "failed_steps",
        "missing_steps",
        "manuscript_ready",
        "paper_authorized",
        "publication_figure_bundle_ready",
        "publication_figure_stems",
        "replan_budget_exhausted",
        "scientific_requirement_complete",
        "scientific_incomplete_steps",
        "step_completion_states",
        "step_scientific_requirements_complete",
        "completion_schema_version",
    }
)


def _strict_gate_mapping(raw: dict[str, Any]) -> dict[str, Any]:
    if set(raw) - _ALL_GATE_NAMES:
        raise ValueError(
            f"run status contains unknown gate fields: {sorted(set(raw) - _ALL_GATE_NAMES)}"
        )
    missing = _REQUIRED_SCORING_GATES - set(raw)
    if missing:
        raise ValueError(f"run status lacks required scoring gates: {sorted(missing)}")
    for name, value in raw.items():
        if name in _BOOL_GATE_NAMES and type(value) is not bool:
            raise ValueError(f"gate {name} must be a boolean")
        if name in _INT_GATE_NAMES and (type(value) is not int or value < 0):
            raise ValueError(f"gate {name} must be a non-negative integer")
        if name in _LIST_GATE_NAMES and type(value) is not list:
            raise ValueError(f"gate {name} must be a list")
        if name in _DICT_GATE_NAMES and type(value) is not dict:
            raise ValueError(f"gate {name} must be an object")
        if name in _STR_GATE_NAMES and type(value) is not str:
            raise ValueError(f"gate {name} must be a string")
        if (
            name in _NULLABLE_STR_GATE_NAMES
            and value is not None
            and type(value) is not str
        ):
            raise ValueError(f"gate {name} must be null or a string")
        if (
            name in _NULLABLE_INT_GATE_NAMES
            and value is not None
            and (type(value) is not int or value < 0)
        ):
            raise ValueError(f"gate {name} must be null or a non-negative integer")
    if raw["completed_step_count"] > raw["required_step_count"]:
        raise ValueError("completed step count exceeds required step count")
    if raw["completion_schema_version"] != "easyicu.run_completion_axes/1":
        raise ValueError("unsupported run completion axes schema")
    return dict(raw)


def require_completed_figure2_gates(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a strict, semantically complete paper-scoring gate snapshot.

    Completion is not established by two booleans alone.  The selected run
    must have completed every required step and must expose neither failed nor
    missing steps.  This predicate is shared by sealing, loading, and bench
    reuse so no path can suppress execution using a weaker definition.
    """

    if type(raw) is not dict:
        raise ValueError("paper-scoring gates must be a JSON object")
    gates = _strict_gate_mapping(dict(raw))
    if gates["execution_complete"] is not True:
        raise PermissionError("paper scoring requires execution_complete")
    if gates["execution_ok"] is not True:
        raise PermissionError("paper scoring requires execution_ok")
    if gates["artifact_valid"] is not True:
        raise PermissionError("paper scoring requires artifact_valid")
    if gates["scientific_requirement_complete"] is not True:
        raise PermissionError("paper scoring requires scientific_requirement_complete")
    if gates["step_scientific_requirements_complete"] is not True:
        raise PermissionError(
            "paper scoring requires step_scientific_requirements_complete"
        )
    if gates["scientific_incomplete_steps"]:
        raise PermissionError("paper scoring rejects scientifically incomplete steps")
    if gates["paper_authorized"] is not True:
        raise PermissionError("paper scoring requires paper_authorized")
    if gates["manuscript_ready"] is not True:
        raise PermissionError("paper scoring requires manuscript_ready")
    if gates["completed_step_count"] != gates["required_step_count"]:
        raise PermissionError("paper scoring requires every planned step to complete")
    if gates["failed_steps"]:
        raise PermissionError("paper scoring rejects runs with failed steps")
    if gates["missing_steps"]:
        raise PermissionError("paper scoring rejects runs with missing steps")
    states = gates["step_completion_states"]
    if len(states) != gates["required_step_count"]:
        raise ValueError("step completion state count differs from required steps")
    seen_step_ids: set[str] = set()
    for state in states:
        if type(state) is not dict:
            raise ValueError("step completion states must contain objects")
        if set(state) != {
            "schema_version",
            "step_id",
            "execution_ok",
            "outer_status",
            "summary_status",
            "scientific_requirement_complete",
        }:
            raise ValueError("step completion state has an invalid schema")
        if state["schema_version"] != "easyicu.step_completion_state/1":
            raise ValueError("unsupported step completion state schema")
        step_id = state["step_id"]
        if type(step_id) is not str or not step_id:
            raise ValueError("step completion state lacks a valid step_id")
        if step_id in seen_step_ids:
            raise ValueError("step completion states contain a duplicate step_id")
        seen_step_ids.add(step_id)
        if state["execution_ok"] is not True:
            raise PermissionError("paper scoring rejects an execution-incomplete step")
        if state["scientific_requirement_complete"] is not True:
            raise PermissionError(
                "paper scoring rejects a scientifically incomplete step"
            )
        if state["outer_status"] is not None and type(state["outer_status"]) is not str:
            raise ValueError("step completion outer_status must be null or a string")
        if (
            state["summary_status"] is not None
            and type(state["summary_status"]) is not str
        ):
            raise ValueError("step completion summary_status must be null or a string")
    return gates


def _require_run_submission_authority(
    checkpoint: Mapping[str, Any],
    *,
    profile_ref: str,
    concept_dict_sha256: str,
    sofa2_dict_sha256: str,
) -> None:
    """Join the owner selector to the submission authority used by the run."""

    try:
        expected_name, expected_version = profile_ref.split("/", 1)
    except ValueError as exc:
        raise ValueError("canonical submission-profile ref is malformed") from exc
    fingerprint = checkpoint.get("concept_dict_fingerprint")
    if type(fingerprint) is not dict:
        raise PermissionError(
            "paper-facing Canonical9 requires a recorded dictionary fingerprint"
        )
    observed = (
        checkpoint.get("submission_profile_name"),
        checkpoint.get("submission_profile_version"),
        checkpoint.get("concept_dict_sha"),
        checkpoint.get("sofa2_dict_sha"),
        fingerprint.get("concept_dict_sha"),
        fingerprint.get("sofa2_dict_sha"),
    )
    expected = (
        expected_name,
        expected_version,
        concept_dict_sha256,
        sofa2_dict_sha256,
        concept_dict_sha256,
        sofa2_dict_sha256,
    )
    if observed != expected:
        raise PermissionError(
            "run submission profile or dictionary authority does not match "
            "the owner-frozen Canonical9 selector"
        )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_loads(payload: bytes) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=_reject_json_constant,
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _research_question_sha256(question: object) -> str:
    if type(question) is not str or not question.strip():
        raise ValueError("current checkpoint lacks a valid research_question")
    return _sha256_bytes(question.encode("utf-8"))


def _frozen_task_objective_sha256(task_id: str) -> str:
    tasks = figure2_suite_projection()["tasks"]
    matches = [
        task
        for task in tasks
        if isinstance(task, Mapping) and task.get("task_id") == task_id
    ]
    if len(matches) != 1:
        raise ValueError("expected task is not unique in the frozen Figure 2 suite")
    return _research_question_sha256(matches[0].get("objective"))


def _paper_task_validity_coordinates(
    paper_rubric: object,
    task_id: str,
) -> tuple[str | None, str | None]:
    tasks = getattr(paper_rubric, "tasks", ())
    matches = [task for task in tasks if getattr(task, "task_id", None) == task_id]
    if len(matches) != 1:
        raise ValueError("paper rubric lacks a unique Figure 2 task binding")
    binding = matches[0].validity_binding
    return binding.exposure_concept, binding.outcome_concept


def _require_paper_task_validity_coordinates(
    *,
    paper_rubric: object,
    task_id: str,
    exposure_concept: str | None,
    outcome_concept: str | None,
) -> tuple[str | None, str]:
    expected_exposure, expected_outcome = _paper_task_validity_coordinates(
        paper_rubric,
        task_id,
    )
    if expected_exposure is None:
        exposure_matches = exposure_concept is None
    else:
        exposure_matches = (
            type(exposure_concept) is str and exposure_concept == expected_exposure
        )
    if not exposure_matches:
        raise ValueError(
            "benchmark exposure concept does not match the frozen Figure 2 task"
        )
    if (
        expected_outcome is None
        or type(outcome_concept) is not str
        or outcome_concept != expected_outcome
    ):
        raise ValueError(
            "benchmark outcome concept does not match the frozen Figure 2 task"
        )
    return expected_exposure, expected_outcome


@dataclass(frozen=True)
class _VerifiedFigure2RunInput:
    capsule_sha256: str
    capsule_schema_version: str
    scientific_identity_sha256: str
    question: str
    database: str
    primary_exposure: str | None
    target_outcome: str
    source_cohort_authority_ref: dict[str, Any]
    source_trajectory_authority_ref: dict[str, Any] | None
    staged_cohort_authority_ref: dict[str, Any]
    staged_trajectory_authority_ref: dict[str, Any] | None


def _canonical_optional_text(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"run input {label} must be explicit null or canonical text")
    return value


def _benchmark_input_binding_sha256(
    *,
    task_id: str,
    research_question: str,
    manifest_sha256: str,
    case_sha256: str,
    binding: ReadyCanonicalTaskBinding,
) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "schema_version": "easyicu.figure2_benchmark_input_binding/2",
                "task_id": task_id,
                "research_question_sha256": _research_question_sha256(
                    research_question
                ),
                "canonical_input_manifest_ref": CANONICAL_RUN_INPUT_BINDING_REF,
                "canonical_input_manifest_sha256": manifest_sha256,
                "canonical_input_case_sha256": case_sha256,
                "binding": binding.model_dump(mode="json"),
            }
        )
    )


def _require_run_matches_ready_binding(
    *,
    run_input: _VerifiedFigure2RunInput,
    research_question: str,
    binding: ReadyCanonicalTaskBinding,
) -> None:
    if (
        binding.research_question_sha256 != _research_question_sha256(research_question)
        or run_input.question != research_question
        or run_input.database != binding.database
        or run_input.primary_exposure != binding.operational_exposure
        or run_input.target_outcome != binding.target_outcome
        or run_input.capsule_schema_version
        != binding.expected_run_input_capsule_schema_version
        or run_input.scientific_identity_sha256 != binding.scientific_identity_sha256
        or _canonical_json_bytes(run_input.source_cohort_authority_ref)
        != _canonical_json_bytes(
            binding.source_materialized_cohort_authority_ref.model_dump(mode="json")
        )
        or _canonical_json_bytes(run_input.source_trajectory_authority_ref)
        != _canonical_json_bytes(
            (
                binding.source_materialized_trajectory_authority_ref.model_dump(
                    mode="json"
                )
                if binding.source_materialized_trajectory_authority_ref is not None
                else None
            )
        )
    ):
        raise PermissionError(
            "run-input capsule does not match the owner-frozen Canonical9 input"
        )


def _record_coordinate(record: Mapping[str, Any]) -> tuple[object, ...]:
    evidence_id = record.get("evidence_id")
    relative_path = record.get("relative_path")
    digest = record.get("sha256")
    kind = record.get("kind")
    producer = record.get("producer")
    generation_mode = record.get("generation_mode")
    produced_by_step = record.get("produced_by_step")
    if not isinstance(evidence_id, str) or not evidence_id.strip():
        raise ValueError("evidence record lacks a valid evidence_id")
    if not isinstance(relative_path, str):
        raise ValueError(f"evidence {evidence_id!r} lacks a relative path")
    _validate_contained_relative_path(relative_path)
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"evidence {evidence_id!r} lacks a valid digest")
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"evidence {evidence_id!r} lacks a valid kind")
    if producer is not None and (not isinstance(producer, str) or not producer):
        raise ValueError(f"evidence {evidence_id!r} has invalid producer metadata")
    if generation_mode is not None and (
        not isinstance(generation_mode, str) or not generation_mode
    ):
        raise ValueError(f"evidence {evidence_id!r} has invalid generation metadata")
    if produced_by_step is not None and (
        not isinstance(produced_by_step, str) or not produced_by_step
    ):
        raise ValueError(f"evidence {evidence_id!r} has invalid step metadata")
    return tuple(record.get(field) for field in _RECORD_COORDINATE_FIELDS)


def _records_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        evidence_id = str(record["evidence_id"])
        if evidence_id in result:
            raise ValueError(f"current evidence identity {evidence_id!r} is ambiguous")
        result[evidence_id] = record
    return result


def _selected_plan_id(records: Mapping[str, Mapping[str, Any]]) -> tuple[str, int]:
    revisions: list[tuple[int, str]] = []
    for evidence_id in records:
        match = _PLAN_ID_RE.fullmatch(evidence_id)
        if match is not None:
            revisions.append((int(match.group(1)), evidence_id))
    if revisions:
        highest = max(revision for revision, _ in revisions)
        winners = sorted(
            evidence_id for revision, evidence_id in revisions if revision == highest
        )
        if len(winners) != 1:
            raise ValueError("current analysis-plan revision is ambiguous")
        return winners[0], highest
    if "analysis_plan" not in records:
        raise FileNotFoundError("current analysis_plan evidence is missing")
    return "analysis_plan", 1


def _read_verified_record(run_dir: Path, record: Mapping[str, Any]) -> bytes:
    path = verified_run_evidence_path(run_dir, record)
    if path is None:
        raise OSError(
            f"current evidence {record.get('evidence_id')!r} failed verification"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("current evidence is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    if _sha256_bytes(payload) != record["sha256"]:
        raise OSError(
            f"current evidence {record.get('evidence_id')!r} changed during read"
        )
    return payload


def _verified_figure2_run_input(
    *,
    run_dir: Path,
    current_records: list[dict[str, Any]],
) -> _VerifiedFigure2RunInput:
    matches = [
        record
        for record in current_records
        if record.get("evidence_id") == RUN_INPUT_CAPSULE_EVIDENCE_ID
    ]
    if len(matches) != 1:
        raise ValueError("current evidence lacks a unique run-input capsule")
    record = matches[0]
    if (
        record.get("kind") != "log"
        or record.get("producer") != "pipeline"
        or record.get("generation_mode") != "system"
        or record.get("produced_by_step") not in (None, "")
    ):
        raise ValueError("run-input capsule evidence metadata is invalid")
    payload = _read_verified_record(run_dir, record)
    decoded = _strict_json_loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("run-input capsule must be a JSON object")
    schema_version = decoded.get("schema_version")
    capsule_type: type[RunInputCapsule]
    if schema_version == RUN_INPUT_CAPSULE_SCHEMA_VERSION_V2:
        capsule_type = RunInputCapsuleV2
    elif schema_version == RUN_INPUT_CAPSULE_SCHEMA_VERSION_V3:
        capsule_type = RunInputCapsuleV3
    else:
        raise PermissionError(
            "paper-facing Canonical9 scoring requires a typed V2/V3 run-input "
            "capsule"
        )
    capsule = capsule_type.model_validate_json(payload, strict=True)
    if capsule.legacy_adopted is not False:
        raise PermissionError("paper-facing Canonical9 scoring rejects legacy input")
    if canonical_sha256(capsule.scientific_identity) != (
        capsule.scientific_identity_sha256
    ):
        raise ValueError("run-input capsule scientific identity digest is invalid")
    verified = load_verified_run_input_capsule(
        run_dir=run_dir,
        scientific_identity=dict(capsule.scientific_identity),
    )
    verified_record = verified.evidence_records.get(RUN_INPUT_CAPSULE_EVIDENCE_ID)
    if (
        verified.capsule != capsule
        or not isinstance(verified_record, Mapping)
        or _record_coordinate(verified_record) != _record_coordinate(record)
    ):
        raise OSError(
            "run-input capsule evidence disagrees with verified input authority"
        )
    identity = capsule.scientific_identity
    required_identity_keys = {
        "question",
        "database",
        "primary_exposure",
        "target_outcome",
        "materialized_cohort_authority_ref",
    }
    if not required_identity_keys.issubset(identity):
        raise ValueError("run-input capsule omits required scientific coordinates")
    question = identity["question"]
    if type(question) is not str or not question.strip():
        raise ValueError("run-input capsule question is invalid")
    database = _canonical_optional_text(identity["database"], label="database")
    if database is None:
        raise ValueError("run-input capsule database is required")
    primary_exposure = _canonical_optional_text(
        identity["primary_exposure"],
        label="primary exposure",
    )
    target_outcome = _canonical_optional_text(
        identity["target_outcome"],
        label="target outcome",
    )
    if target_outcome is None:
        raise ValueError("run-input capsule target outcome is required")
    source_cohort = FrozenTypedAuthorityRef.model_validate(
        identity["materialized_cohort_authority_ref"], strict=True
    ).model_dump(mode="json")
    source_trajectory_value = identity.get("materialized_trajectory_authority_ref")
    source_trajectory = (
        FrozenTypedAuthorityRef.model_validate(
            source_trajectory_value, strict=True
        ).model_dump(mode="json")
        if source_trajectory_value is not None
        else None
    )
    staged_cohort = FrozenTypedAuthorityRef.model_validate(
        capsule.materialized_cohort_authority_ref, strict=True
    ).model_dump(mode="json")
    staged_trajectory = None
    if type(capsule) is RunInputCapsuleV3:
        staged_trajectory = FrozenTypedAuthorityRef.model_validate(
            capsule.materialized_trajectory_authority_ref, strict=True
        ).model_dump(mode="json")
        if source_trajectory is None:
            raise ValueError("typed trajectory capsule lacks its source authority")
    elif source_trajectory is not None:
        raise ValueError("V2 capsule cannot claim a source trajectory authority")
    return _VerifiedFigure2RunInput(
        capsule_sha256=str(record["sha256"]),
        capsule_schema_version=capsule.schema_version,
        scientific_identity_sha256=capsule.scientific_identity_sha256,
        question=question,
        database=database,
        primary_exposure=primary_exposure,
        target_outcome=target_outcome,
        source_cohort_authority_ref=source_cohort,
        source_trajectory_authority_ref=source_trajectory,
        staged_cohort_authority_ref=staged_cohort,
        staged_trajectory_authority_ref=staged_trajectory,
    )


def _read_bounded_regular_file(
    path: Path,
    *,
    label: str,
    max_bytes: int,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> bytes:
    """Read one stable non-symlink regular file through its descriptor."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OSError(f"{label} is not a regular file")
        if before.st_size > max_bytes or (
            expected_size is not None and before.st_size != expected_size
        ):
            raise OSError(f"{label} size is invalid")
        chunks: list[bytes] = []
        remaining = int(before.st_size)
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                raise OSError(f"{label} ended before its stat size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise OSError(f"{label} changed while being read")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise OSError(f"{label} changed while being read")
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    if expected_sha256 is not None and _sha256_bytes(payload) != expected_sha256:
        raise OSError(f"{label} digest is invalid")
    return payload


def _read_final_manifest(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Read one small non-symlink final manifest with strict JSON semantics."""

    payload = _read_bounded_regular_file(
        path,
        label="final run manifest",
        max_bytes=16 * 1024 * 1024,
    )
    decoded = _strict_json_loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("final run manifest must be a JSON object")
    AnalysisManifest.model_validate_json(payload, strict=True)
    return payload, decoded


def _task_authority_bytes(authority: Figure2RunTaskAuthority) -> bytes:
    return _canonical_json_bytes(authority.model_dump(mode="json")) + b"\n"


def _task_authority_path(run_dir: Path, digest: str) -> Path:
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("task-authority digest is invalid")
    return run_dir / f"figure2_task_authority.sha256-{digest}.json"


def _publish_task_authority_once(path: Path, payload: bytes) -> None:
    """Publish one immutable task-authority blob without a mutable selector."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor: int | None = None
    linked = False
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(temporary, flags, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("task-authority write made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(temporary, path, follow_symlinks=False)
            linked = True
        except FileExistsError:
            existing = _read_bounded_regular_file(
                path,
                label="Figure 2 task authority",
                max_bytes=FIGURE2_TASK_AUTHORITY_MAX_BYTES,
                expected_size=len(payload),
                expected_sha256=_sha256_bytes(payload),
            )
            if existing != payload:
                raise PermissionError("existing Figure 2 task authority is immutable")
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if linked and not path.exists():
        raise OSError("task-authority publication did not become visible")


def _load_exact_task_authority(
    run_dir: Path,
    expected: Figure2RunTaskAuthority,
) -> tuple[Figure2RunTaskAuthority, str, int]:
    payload = _task_authority_bytes(expected)
    digest = _sha256_bytes(payload)
    path = _task_authority_path(run_dir, digest)
    observed = _read_bounded_regular_file(
        path,
        label="Figure 2 task authority",
        max_bytes=FIGURE2_TASK_AUTHORITY_MAX_BYTES,
        expected_size=len(payload),
        expected_sha256=digest,
    )
    if observed != payload:
        raise OSError("Figure 2 task authority bytes are not canonical")
    parsed = Figure2RunTaskAuthority.model_validate_json(observed, strict=True)
    if parsed != expected:
        raise OSError("Figure 2 task authority does not match current run authority")
    return parsed, digest, len(payload)


@dataclass(frozen=True, slots=True)
class _VerifiedFigure2RunAuthority:
    checkpoint: dict[str, Any]
    checkpoint_bytes: bytes
    final_manifest_bytes: bytes
    readiness: dict[str, Any]
    raw_steps: list[dict[str, Any]]
    current_records: list[dict[str, Any]]
    snapshot: Any
    run_status: _RunStatusDocument
    run_input: _VerifiedFigure2RunInput
    task_authority: Figure2RunTaskAuthority


def _build_expected_task_authority_locked(
    root: Path,
    *,
    task_id: str,
) -> _VerifiedFigure2RunAuthority:
    """Recompute the only task-authority blob valid for the current run."""

    if task_id not in FIGURE2_TASK_IDS:
        raise ValueError("task is outside the frozen Figure 2 suite")
    final_bytes, final_payload = _read_final_manifest(root / "manifest.json")
    selected = load_run_artifact_authority(root)
    if selected is None or _canonical_json_bytes(selected) != _canonical_json_bytes(
        final_payload
    ):
        raise OSError("final manifest is not the current run checkpoint authority")
    checkpoint_bytes = _canonical_json_bytes(selected)
    run_id = selected.get("run_id")
    sequence = selected.get("checkpoint_sequence")
    research_question = selected.get("research_question")
    if run_id != root.name or type(sequence) is not int or sequence < 1:
        raise ValueError("current checkpoint run coordinate is invalid")
    if not final_bytes:
        raise OSError("final manifest bytes are empty")
    if _research_question_sha256(research_question) != _frozen_task_objective_sha256(
        task_id
    ):
        raise ValueError("run question does not match the frozen Figure 2 task")

    paper_rubric = load_figure2_paper_rubric()
    exposure_concept, outcome_concept = _paper_task_validity_coordinates(
        paper_rubric, task_id
    )
    if outcome_concept is None:
        raise ValueError("paper task lacks a required outcome validity coordinate")
    manifest, ready_binding, manifest_sha256, case_sha256 = require_ready_task_binding(
        task_id
    )
    _require_run_submission_authority(
        selected,
        profile_ref=manifest.submission_profile.ref,
        concept_dict_sha256=manifest.submission_profile.concept_dict_sha256,
        sofa2_dict_sha256=manifest.submission_profile.sofa2_dict_sha256,
    )
    if ready_binding.research_question_sha256 != _research_question_sha256(
        research_question
    ):
        raise PermissionError("ready input binding targets a different question")

    readiness_raw = selected.get("readiness")
    raw_steps = selected.get("per_step_records")
    raw_records = selected.get("evidence")
    readiness = require_completed_figure2_gates(readiness_raw)
    if not isinstance(raw_steps, list) or any(
        not isinstance(item, dict) for item in raw_steps
    ):
        raise ValueError("current checkpoint has a malformed step ledger")
    if not isinstance(raw_records, list) or any(
        not isinstance(item, dict) for item in raw_records
    ):
        raise ValueError("current checkpoint has malformed evidence authority")

    snapshot = load_current_evidence_snapshot(root)
    if (
        snapshot.source != "authority"
        or type(snapshot.generation) is not int
        or snapshot.generation < 1
        or not isinstance(snapshot.payload_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", snapshot.payload_sha256) is None
    ):
        raise ValueError("run lacks a modern selected EvidenceStore generation")
    checkpoint_coordinates = Counter(_record_coordinate(item) for item in raw_records)
    evidence_records = [dict(item) for item in snapshot.records]
    if checkpoint_coordinates != Counter(
        _record_coordinate(item) for item in evidence_records
    ):
        raise OSError("checkpoint and current EvidenceStore generation disagree")
    current_records = [
        dict(item)
        for item in current_evidence_records(evidence_records, raw_steps)
        if isinstance(item, Mapping)
    ]
    current_by_id = _records_by_id(current_records)
    run_status_record = current_by_id.get("run_status")
    if run_status_record is None:
        raise FileNotFoundError("current run_status evidence is missing")
    run_status = _RunStatusDocument.model_validate_json(
        _read_verified_record(root, run_status_record), strict=True
    )
    if run_status.status not in {"manuscript_ready", "publication_ready"}:
        raise PermissionError("current run_status is not manuscript-ready")
    status_gates = require_completed_figure2_gates(run_status.gates)
    if run_status.research_question != research_question or _canonical_json_bytes(
        status_gates
    ) != _canonical_json_bytes(readiness):
        raise OSError("sealed run_status disagrees with the current checkpoint")

    run_input = _verified_figure2_run_input(
        run_dir=root,
        current_records=current_records,
    )
    _require_run_matches_ready_binding(
        run_input=run_input,
        research_question=research_question,
        binding=ready_binding,
    )
    benchmark_binding_sha256 = _benchmark_input_binding_sha256(
        task_id=task_id,
        research_question=research_question,
        manifest_sha256=manifest_sha256,
        case_sha256=case_sha256,
        binding=ready_binding,
    )
    task_authority = Figure2RunTaskAuthority(
        schema_version=FIGURE2_RUN_TASK_AUTHORITY_SCHEMA,
        task_id=task_id,
        run_id=run_id,
        checkpoint_sequence=sequence,
        checkpoint_payload_sha256=_sha256_bytes(checkpoint_bytes),
        suite_ref=FIGURE2_SUITE_REF,
        suite_projection_sha256=figure2_suite_projection_sha256(),
        paper_rubric_ref=FIGURE2_PAPER_RUBRIC_REF,
        paper_rubric_sha256=paper_rubric_manifest_sha256(paper_rubric),
        research_question_sha256=_research_question_sha256(research_question),
        exposure_concept=exposure_concept,
        outcome_concept=outcome_concept,
        benchmark_input_binding_sha256=benchmark_binding_sha256,
        run_input_capsule_sha256=run_input.capsule_sha256,
        run_scientific_identity_sha256=run_input.scientific_identity_sha256,
        run_input_capsule_schema_version=run_input.capsule_schema_version,
        run_primary_exposure=run_input.primary_exposure,
        run_target_outcome=run_input.target_outcome,
        canonical_input_manifest_ref=CANONICAL_RUN_INPUT_BINDING_REF,
        canonical_input_manifest_sha256=manifest_sha256,
        canonical_input_case_sha256=case_sha256,
        submission_profile_ref=manifest.submission_profile.ref,
        concept_dict_sha256=manifest.submission_profile.concept_dict_sha256,
        sofa2_dict_sha256=manifest.submission_profile.sofa2_dict_sha256,
        source_cohort_authority_sha256=(
            run_input.source_cohort_authority_ref["sha256"]
        ),
        source_trajectory_authority_sha256=(
            run_input.source_trajectory_authority_ref["sha256"]
            if run_input.source_trajectory_authority_ref is not None
            else None
        ),
        staged_cohort_authority_sha256=(
            run_input.staged_cohort_authority_ref["sha256"]
        ),
        staged_trajectory_authority_sha256=(
            run_input.staged_trajectory_authority_ref["sha256"]
            if run_input.staged_trajectory_authority_ref is not None
            else None
        ),
        run_status_evidence_sha256=run_status_record["sha256"],
        readiness_sha256=_sha256_bytes(_canonical_json_bytes(readiness)),
        evidence_generation=snapshot.generation,
        evidence_payload_sha256=snapshot.payload_sha256,
    )
    return _VerifiedFigure2RunAuthority(
        checkpoint=dict(selected),
        checkpoint_bytes=checkpoint_bytes,
        final_manifest_bytes=final_bytes,
        readiness=readiness,
        raw_steps=[dict(item) for item in raw_steps],
        current_records=current_records,
        snapshot=snapshot,
        run_status=run_status,
        run_input=run_input,
        task_authority=task_authority,
    )


def _parse_claim_rows(
    payload: bytes,
    *,
    current_evidence_ids: set[str],
    aliases: Mapping[str, str],
) -> tuple[list[dict[str, str]], tuple[tuple[str, tuple[str, ...]], ...]]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""))
    expected = ["claim_id", "claim_text", "evidence_refs", "status", "note"]
    if reader.fieldnames != expected:
        raise ValueError("claim ledger has an unsupported column contract")
    raw_rows = list(reader)
    if not raw_rows:
        raise ValueError("claim ledger must contain at least one claim row")
    rows: list[dict[str, str]] = []
    reference_sets: list[tuple[str, tuple[str, ...]]] = []
    claim_ids: set[str] = set()
    allowed = set(ClaimStatus.__args__)
    for raw in raw_rows:
        if (
            None in raw
            or set(raw) != set(expected)
            or any(value is None for value in raw.values())
        ):
            raise ValueError("claim ledger contains a malformed row")
        row = {key: str(raw[key]) for key in expected}
        if not row["claim_id"].strip():
            raise ValueError("claim ledger contains an empty claim_id")
        if row["claim_id"] in claim_ids:
            raise ValueError("claim ledger contains a duplicate claim_id")
        claim_ids.add(row["claim_id"])
        if not row["claim_text"].strip():
            raise ValueError("claim ledger contains empty claim_text")
        if row["status"] not in allowed:
            raise ValueError("claim ledger contains an unsupported status")
        raw_refs = row["evidence_refs"]
        if raw_refs.strip():
            tokens = raw_refs.split(";")
            if any(not token.strip() for token in tokens):
                raise ValueError("claim ledger contains an empty evidence reference")
        else:
            tokens = []
        resolved: set[str] = set()
        for token in tokens:
            reference = token.strip()
            direct_evidence_id = (
                reference if reference in current_evidence_ids else None
            )
            alias_evidence_id = aliases.get(reference)
            if (
                direct_evidence_id is not None
                and alias_evidence_id is not None
                and alias_evidence_id != direct_evidence_id
            ):
                raise ValueError(
                    f"claim {row['claim_id']!r} has ambiguous evidence reference "
                    f"{reference!r}"
                )
            evidence_id = direct_evidence_id or alias_evidence_id
            if evidence_id not in current_evidence_ids:
                raise ValueError(
                    f"claim {row['claim_id']!r} references unresolved or stale "
                    f"evidence {reference!r}"
                )
            resolved.add(evidence_id)
        normalized = tuple(sorted(resolved))
        if row["status"] in {"bound", "ok", "verified"} and not normalized:
            raise ValueError("bound claim lacks evidence references")
        row["evidence_refs"] = ";".join(normalized)
        rows.append(row)
        reference_sets.append((row["claim_id"], normalized))
    return rows, tuple(reference_sets)


def _artifact_from_record(
    role: Figure2ArtifactRole,
    record: Mapping[str, Any],
    payload: bytes,
) -> Figure2ArtifactAuthority:
    return Figure2ArtifactAuthority(
        role=role,
        evidence_id=record["evidence_id"],
        relative_path=record["relative_path"],
        sha256=record["sha256"],
        byte_count=len(payload),
        kind=record["kind"],
        producer=record["producer"],
        generation_mode=record["generation_mode"],
        produced_by_step=record.get("produced_by_step"),
    )


def _load_figure2_scoring_inputs_locked(
    run_dir: Path | str,
    *,
    expected_task_id: str,
) -> LoadedFigure2ScoringInputs:
    """Load bound scorer inputs while the caller owns the run lease."""

    if type(expected_task_id) is not str or expected_task_id not in FIGURE2_TASK_IDS:
        raise ValueError("expected task is outside the frozen Figure 2 suite")

    root = Path(run_dir).expanduser().resolve()
    verified_run = _build_expected_task_authority_locked(
        root,
        task_id=expected_task_id,
    )
    checkpoint = verified_run.checkpoint
    checkpoint_bytes = verified_run.checkpoint_bytes
    run_id = verified_run.task_authority.run_id
    sequence = verified_run.task_authority.checkpoint_sequence
    research_question = checkpoint["research_question"]
    task_authority, task_authority_sha256, task_authority_byte_count = (
        _load_exact_task_authority(root, verified_run.task_authority)
    )
    raw_steps = verified_run.raw_steps
    current_step_summaries = tuple(
        copy.deepcopy(summary)
        for record in current_successful_step_records(raw_steps)
        if isinstance((summary := record.get("step_summary")), dict)
    )
    snapshot = verified_run.snapshot
    readiness = verified_run.readiness
    current_records = verified_run.current_records
    current_by_id = _records_by_id(current_records)
    plan_id, expected_revision = _selected_plan_id(current_by_id)
    role_to_id: dict[Figure2ArtifactRole, str] = {
        "run_status": "run_status",
        "analysis_plan": plan_id,
        "evidence_audit": "evidence_audit",
        "numeric_audit": "numeric_audit",
        "claim_ledger": "claim_ledger",
        "manuscript_ready": "manuscript_ready",
    }

    payload_cache: dict[tuple[object, ...], bytes] = {}
    artifacts: list[Figure2ArtifactAuthority] = []
    role_payloads: dict[str, bytes] = {}
    for role in FIGURE2_SCORING_ARTIFACT_ROLES:
        evidence_id = role_to_id[role]  # type: ignore[index]
        record = current_by_id.get(evidence_id)
        if record is None:
            raise FileNotFoundError(f"current evidence {evidence_id!r} is missing")
        coordinate = _record_coordinate(record)
        payload = payload_cache.setdefault(
            coordinate, _read_verified_record(root, record)
        )
        role_payloads[role] = payload
        artifacts.append(_artifact_from_record(role, record, payload))  # type: ignore[arg-type]

    run_status = _RunStatusDocument.model_validate_json(
        role_payloads["run_status"], strict=True
    )
    if run_status.research_question != research_question:
        raise OSError("sealed run-status question disagrees with current checkpoint")
    if _canonical_json_bytes(run_status.gates) != _canonical_json_bytes(readiness):
        raise OSError("sealed run-status gates disagree with current checkpoint")
    evidence_audit = _EvidenceAuditDocument.model_validate_json(
        role_payloads["evidence_audit"], strict=True
    )
    observed_kinds = dict(
        sorted(Counter(str(record["kind"]) for record in current_records).items())
    )
    if evidence_audit.evidence_count != len(current_records):
        raise ValueError(
            "evidence audit count disagrees with current evidence authority"
        )
    if dict(sorted(evidence_audit.kinds.items())) != observed_kinds:
        raise ValueError(
            "evidence audit kinds disagree with current evidence authority"
        )
    numeric_audit = _NumericAuditDocument.model_validate_json(
        role_payloads["numeric_audit"], strict=True
    )
    plan = AnalysisPlan.model_validate_json(role_payloads["analysis_plan"], strict=True)
    if plan.research_question != research_question:
        raise OSError("sealed analysis-plan question disagrees with current checkpoint")
    if plan.revision != expected_revision:
        raise ValueError("analysis-plan evidence revision does not match plan.revision")
    plan_steps = [dict(item) for item in plan.model_dump(mode="json")["steps"]]
    claim_rows, claim_reference_sets = _parse_claim_rows(
        role_payloads["claim_ledger"],
        current_evidence_ids=set(current_by_id),
        aliases=snapshot.aliases,
    )
    referenced_evidence_ids = sorted(
        {
            evidence_id
            for _, evidence_ids in claim_reference_sets
            for evidence_id in evidence_ids
        }
    )
    for evidence_id in referenced_evidence_ids:
        record = current_by_id[evidence_id]
        coordinate = _record_coordinate(record)
        if coordinate not in payload_cache:
            payload_cache[coordinate] = _read_verified_record(root, record)
    manuscript_bytes = role_payloads["manuscript_ready"]
    if not manuscript_bytes.decode("utf-8").strip():
        raise ValueError("manuscript_ready evidence is empty")

    review_documents: list[Figure2ReviewDocument] = []
    total_review_bytes = 0
    review_records = sorted(
        (
            record
            for record in current_records
            if PurePosixPath(record["relative_path"]).suffix.lower() in _TEXT_SUFFIXES
        ),
        key=lambda item: (
            str(item["evidence_id"]),
            str(item["relative_path"]),
            str(item["sha256"]),
        ),
    )
    for record in review_records:
        coordinate = _record_coordinate(record)
        payload = payload_cache.get(coordinate)
        if payload is None:
            payload = _read_verified_record(root, record)
            payload_cache[coordinate] = payload
        if len(payload) > FIGURE2_REVIEW_DOCUMENT_MAX_BYTES:
            raise ValueError(f"review document {record['evidence_id']!r} exceeds 1 MiB")
        total_review_bytes += len(payload)
        if total_review_bytes > FIGURE2_REVIEW_CORPUS_MAX_BYTES:
            raise ValueError("Figure 2 review corpus exceeds 8 MiB")
        review_documents.append(
            Figure2ReviewDocument(
                evidence_id=record["evidence_id"],
                relative_path=record["relative_path"],
                sha256=record["sha256"],
                byte_count=len(payload),
                kind=record["kind"],
                producer=record["producer"],
                generation_mode=record["generation_mode"],
                produced_by_step=record.get("produced_by_step"),
                text=payload.decode("utf-8"),
            )
        )
    if not any(item.evidence_id == "manuscript_ready" for item in review_documents):
        raise ValueError("review corpus lacks manuscript_ready evidence")

    authority = Figure2ScoringInputAuthority(
        schema_version=FIGURE2_SCORING_INPUT_AUTHORITY_SCHEMA,
        task_id=task_authority.task_id,
        suite_ref=task_authority.suite_ref,
        suite_projection_sha256=task_authority.suite_projection_sha256,
        paper_rubric_ref=task_authority.paper_rubric_ref,
        paper_rubric_sha256=task_authority.paper_rubric_sha256,
        research_question_sha256=task_authority.research_question_sha256,
        exposure_concept=task_authority.exposure_concept,
        outcome_concept=task_authority.outcome_concept,
        benchmark_input_binding_sha256=(task_authority.benchmark_input_binding_sha256),
        run_input_capsule_sha256=task_authority.run_input_capsule_sha256,
        run_scientific_identity_sha256=(task_authority.run_scientific_identity_sha256),
        run_input_capsule_schema_version=(
            task_authority.run_input_capsule_schema_version
        ),
        run_primary_exposure=task_authority.run_primary_exposure,
        run_target_outcome=task_authority.run_target_outcome,
        canonical_input_manifest_ref=task_authority.canonical_input_manifest_ref,
        canonical_input_manifest_sha256=(
            task_authority.canonical_input_manifest_sha256
        ),
        canonical_input_case_sha256=task_authority.canonical_input_case_sha256,
        submission_profile_ref=task_authority.submission_profile_ref,
        concept_dict_sha256=task_authority.concept_dict_sha256,
        sofa2_dict_sha256=task_authority.sofa2_dict_sha256,
        source_cohort_authority_sha256=(task_authority.source_cohort_authority_sha256),
        source_trajectory_authority_sha256=(
            task_authority.source_trajectory_authority_sha256
        ),
        staged_cohort_authority_sha256=(task_authority.staged_cohort_authority_sha256),
        staged_trajectory_authority_sha256=(
            task_authority.staged_trajectory_authority_sha256
        ),
        run_status_evidence_sha256=task_authority.run_status_evidence_sha256,
        readiness_sha256=task_authority.readiness_sha256,
        task_authority_sha256=task_authority_sha256,
        task_authority_byte_count=task_authority_byte_count,
        run_id=run_id,
        checkpoint_sequence=sequence,
        checkpoint_payload_sha256=_sha256_bytes(checkpoint_bytes),
        evidence_generation=snapshot.generation,
        evidence_payload_sha256=snapshot.payload_sha256,
        artifacts=tuple(artifacts),
    )
    return LoadedFigure2ScoringInputs(
        authority=authority,
        gates=dict(run_status.gates),
        plan_steps=plan_steps,
        evidence_audit=evidence_audit.model_dump(mode="json"),
        numeric_audit=numeric_audit.model_dump(mode="json"),
        claim_rows=claim_rows,
        claim_reference_sets=claim_reference_sets,
        current_step_summaries=current_step_summaries,
        manuscript_bytes=manuscript_bytes,
        review_documents=tuple(review_documents),
    )


def load_figure2_scoring_inputs(
    run_dir: Path | str,
    *,
    expected_task_id: str,
) -> LoadedFigure2ScoringInputs:
    """Load one fully bound scorer input set under the run writer lease."""

    root = Path(run_dir).expanduser().resolve()
    with acquire_run_execution_lock(workdir=root.parent, run_id=root.name):
        return _load_figure2_scoring_inputs_locked(
            root,
            expected_task_id=expected_task_id,
        )


def seal_figure2_run_task_authority(
    run_dir: Path | str,
    *,
    task_id: str,
    research_question: str,
    exposure_concept: str | None,
    outcome_concept: str,
    operational_exposure: str | None,
) -> Figure2RunTaskAuthority:
    """Publish evaluator coordinates as an immutable repository-local sidecar.

    The benchmark adapter supplies all scientific coordinates explicitly.  The
    sealer never parses prose, never falls back to an operational exposure, and
    never mutates the Agent manifest, checkpoint sequence, or EvidenceStore.
    """

    if task_id not in FIGURE2_TASK_IDS:
        raise ValueError("task is outside the frozen Figure 2 suite")
    if type(research_question) is not str or (
        _research_question_sha256(research_question)
        != _frozen_task_objective_sha256(task_id)
    ):
        raise ValueError("benchmark question does not match the frozen task objective")
    normalized_operational_exposure = _canonical_optional_text(
        operational_exposure,
        label="operational exposure",
    )
    paper_rubric = load_figure2_paper_rubric()
    normalized_exposure, normalized_outcome = _require_paper_task_validity_coordinates(
        paper_rubric=paper_rubric,
        task_id=task_id,
        exposure_concept=exposure_concept,
        outcome_concept=outcome_concept,
    )

    root = Path(run_dir).expanduser().resolve()
    with acquire_run_execution_lock(workdir=root.parent, run_id=root.name):
        verified = _build_expected_task_authority_locked(root, task_id=task_id)
        candidate = verified.task_authority
        if verified.checkpoint.get("research_question") != research_question:
            raise ValueError(
                "current checkpoint question disagrees with benchmark item"
            )
        if (
            candidate.exposure_concept != normalized_exposure
            or candidate.outcome_concept != normalized_outcome
            or candidate.run_primary_exposure != normalized_operational_exposure
        ):
            raise ValueError(
                "benchmark execution coordinates disagree with frozen evaluator "
                "authority"
            )

        payload = _task_authority_bytes(candidate)
        digest = _sha256_bytes(payload)
        _publish_task_authority_once(_task_authority_path(root, digest), payload)
        sealed, _, _ = _load_exact_task_authority(root, candidate)

        final_bytes_after, final_payload_after = _read_final_manifest(
            root / "manifest.json"
        )
        selected_after = load_run_artifact_authority(root)
        if (
            final_bytes_after != verified.final_manifest_bytes
            or selected_after is None
            or _canonical_json_bytes(selected_after)
            != _canonical_json_bytes(final_payload_after)
            or _sha256_bytes(_canonical_json_bytes(selected_after))
            != candidate.checkpoint_payload_sha256
            or selected_after.get("checkpoint_sequence")
            != candidate.checkpoint_sequence
        ):
            raise OSError("Figure 2 sidecar publication changed run authority")
        return sealed


__all__ = [
    "FIGURE2_RUN_TASK_AUTHORITY_SCHEMA",
    "FIGURE2_REVIEW_CORPUS_MAX_BYTES",
    "FIGURE2_REVIEW_DOCUMENT_MAX_BYTES",
    "FIGURE2_SCORING_ARTIFACT_ROLES",
    "FIGURE2_SCORING_INPUT_AUTHORITY_SCHEMA",
    "FIGURE2_SUITE_REF",
    "Figure2ArtifactAuthority",
    "Figure2ReviewDocument",
    "Figure2RunTaskAuthority",
    "Figure2ScoringInputAuthority",
    "Figure2TaskAuthorityMismatch",
    "LoadedFigure2ScoringInputs",
    "require_completed_figure2_gates",
    "load_figure2_scoring_inputs",
    "seal_figure2_run_task_authority",
]
