"""Verification of an explicit, preparation-only SIC raw-source review.

This module does not create review decisions, open a raw file, prepare data, or
provide an analysis grant. A caller must reverify the checkpoint immediately
before any future producer side effect. There is not yet a production UI or
producer consuming this capability.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from easyicu.databases.time_origin import SIC_CARE_UNIT_ORIGIN
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint, verify_raw_preparation_review


def _require_sic_file_roster(files: tuple[str, ...]) -> None:
    if (not isinstance(files, tuple) or not files or any(not isinstance(item, str) for item in files)
            or len(files) > 8 or tuple(sorted(set(files))) != files
            or "cases.parquet" not in files):
        raise ValueError("sic_raw_preparation_scope_roster_invalid")
    labs = []
    for raw in files:
        if not isinstance(raw, str) or len(raw) > 70:
            raise ValueError("sic_raw_preparation_scope_roster_invalid")
        path = PurePosixPath(raw)
        if (not raw or "\\" in raw or "\x00" in raw or path.is_absolute()
                or path.as_posix() != raw or not path.parts or ".." in path.parts):
            raise ValueError("sic_raw_preparation_scope_roster_invalid")
        if raw == "cases.parquet":
            continue
        valid_lab = (raw == "laboratory.parquet" or (
            path.parts[0] in {"laboratory", "laboratory_bucket"} and path.suffix == ".parquet"
            and len(path.parts) >= 2
        ))
        if not valid_lab:
            raise ValueError("sic_raw_preparation_scope_roster_invalid")
        labs.append(raw)
    if not labs:
        raise ValueError("sic_raw_preparation_scope_roster_invalid")


class SICRawPreparationScope(BaseModel):
    """Exact words the human must see and approve, not a production receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal["easyicu.sic-raw-preparation-scope/1"] = "easyicu.sic-raw-preparation-scope/1"
    purpose: Literal["prepare_only"] = "prepare_only"
    database: Literal["sic"] = "sic"
    source_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_origin: Literal["first_intensive_or_intermediate_care_transfer"] = SIC_CARE_UNIT_ORIGIN
    output_concepts: tuple[Literal["lact", "care_unit_to_case_closure_days", "death_during_case",
                                   "death_time_during_case_hours"], ...]
    files: tuple[str, ...] = Field(min_length=2, max_length=8)
    case_columns: tuple[Literal["CaseID", "ICUOffset", "TimeOfStay", "OffsetOfDeath", "HospitalDischargeType"], ...]
    laboratory_columns: tuple[Literal["CaseID", "Offset", "LaboratoryID", "LaboratoryValue"], ...]
    laboratory_ids: tuple[Literal[657, 454, 465], ...] = (454, 465, 657)
    window_start_hours: Literal[0] = 0
    window_end_hours: Literal[24] = 24
    window_end_exclusive: Literal[True] = True
    file_read_scope: Literal["whole_listed_files"] = "whole_listed_files"
    analysis_authorized: Literal[False] = False

    @model_validator(mode="after")
    def _exact_scope(self):
        concepts = ("care_unit_to_case_closure_days", "death_during_case", "death_time_during_case_hours", "lact")
        case_fields = ("CaseID", "HospitalDischargeType", "ICUOffset", "OffsetOfDeath", "TimeOfStay")
        lab_fields = ("CaseID", "LaboratoryID", "LaboratoryValue", "Offset")
        if (self.output_concepts != concepts or self.case_columns != case_fields
                or self.laboratory_columns != lab_fields or self.laboratory_ids != (454, 465, 657)):
            raise ValueError("sic_raw_preparation_scope_columns_mismatch")
        _require_sic_file_roster(self.files)
        return self


def render_sic_raw_preparation_scope(scope: SICRawPreparationScope) -> str:
    """The approved review summary must disclose the exact scope, including full-file reads."""
    return (
        "仅批准 SICdb 数据准备，不批准分析。"
        f"来源标识 {scope.source_identity_sha256}；读取完整文件：{', '.join(scope.files)}；"
        f"源字段结构摘要 {scope.source_schema_sha256}；"
        f"病例字段：{', '.join(scope.case_columns)}；乳酸字段：{', '.join(scope.laboratory_columns)}；"
        f"输出概念：{', '.join(scope.output_concepts)}；"
        "乳酸 ID：454、465、657；乳酸分析窗：ICU 或中间监护首次转入后 [0,24) 小时；"
        "人群为所有病例，未应用成人或其他筛选；病例持续时长和死亡状态覆盖整个病例；"
        "病例闭合不代表纯 ICU 出科。"
    )


@dataclass(frozen=True)
class VerifiedSICRawPreparationGrant:
    scope: SICRawPreparationScope
    checkpoint_sha256: str
    review_id: str
    expires_at: str


class SICRawPreparationCheckpoint(RawPreparationCheckpoint):
    """Legacy SIC checkpoint schema backed by the shared host state machine."""

    error_prefix: ClassVar[str] = "sic_raw_preparation"
    schema_version: Literal["easyicu.sic-raw-preparation-checkpoint/1"] = "easyicu.sic-raw-preparation-checkpoint/1"


def sic_preparation_schema_sha256(source_path: Path, files: tuple[str, ...]) -> str:
    """Read only approved Parquet footers, not data rows, for the review scope."""
    import pyarrow.parquet as pq
    from easyicu.research_agent.authority.filesystem import AnchoredDirectory

    root_path = Path(source_path)
    _require_sic_file_roster(files)
    fields = {
        "cases": ("CaseID", "HospitalDischargeType", "ICUOffset", "OffsetOfDeath", "TimeOfStay"),
        "laboratory": ("CaseID", "LaboratoryID", "LaboratoryValue", "Offset"),
    }
    entries = []
    with AnchoredDirectory.open(root_path) as root:
        for raw_file in files:
            relative = PurePosixPath(raw_file)
            expected = fields["cases" if raw_file == "cases.parquet" else "laboratory"]
            with AnchoredDirectory.open(root_path.joinpath(*relative.parts[:-1])) as directory:
                with directory.open_regular(relative.name) as handle:
                    schema = pq.read_schema(handle)
                if not set(expected) <= set(schema.names):
                    raise ValueError("sic_raw_preparation_source_fields_missing")
                entries.append({"file": raw_file,
                                "fields": [(name, str(schema.field(name).type)) for name in expected]})
                directory.assert_still_selected()
        root.assert_still_selected()
    return canonical_sha256(entries)


def verify_sic_raw_preparation_grant(*, plan_result, run_dir: Path, source_path: Path,
                                     requested_scope: SICRawPreparationScope,
                                     now: datetime | None = None) -> VerifiedSICRawPreparationGrant:
    """Verify one explicit host checkpoint and exact requested preparation scope.

    A successful return authorizes *only* this preparation scope. It does not
    establish immutable source bytes, complete read dependencies, or permission
    to analyze the data. The current product does not yet issue this request.
    """
    from easyicu.databases.research_substitutions import case_duration_selection, time_origin_selection
    from easyicu.research_agent.authority.filesystem import AnchoredDirectory
    from easyicu.research_agent.planning.scientific_review import PlanScientificReview
    from easyicu.research_agent.research_context.export_selection import SourceIdentity
    from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
    from easyicu.databases.research_plan_requirements import _constraints
    from easyicu.webserver import sources
    from easyicu.webserver.time_origin_source_authority import (
        CaseDurationMetadataPlanReceipt, prepare_case_duration_metadata_plan,
    )

    if not isinstance(requested_scope, SICRawPreparationScope):
        raise ValueError("sic_raw_preparation_scope_invalid")
    requested_scope = SICRawPreparationScope.model_validate(requested_scope.model_dump(mode="python"))
    source_path = Path(source_path)
    if not source_path.is_absolute() or source_path.is_symlink() or not source_path.is_dir():
        raise ValueError("sic_raw_preparation_source_mismatch")
    context, plan = plan_result.context, plan_result.plan
    plan_sha = canonical_sha256(plan.model_dump(mode="json"))
    context_sha = canonical_sha256(context.model_dump(mode="json"))
    origin, outcome = time_origin_selection(context), case_duration_selection(context)
    identity = SourceIdentity.from_record(_constraints(context).get("source_identity"))
    if (identity is None or identity.database != "sic" or origin is None or outcome is None
            or origin.selected_origin != SIC_CARE_UNIT_ORIGIN
            or origin.source_identity_sha256 != canonical_sha256(identity.to_dict())
            or sources.registered_source_identity(str(source_path)) != identity):
        raise ValueError("sic_raw_preparation_source_mismatch")
    if (requested_scope.source_identity_sha256 != origin.source_identity_sha256
            or requested_scope.plan_sha256 != plan_sha or requested_scope.context_sha256 != context_sha):
        raise ValueError("sic_raw_preparation_scope_binding_mismatch")
    # This producer candidate is currently limited to one lactate exposure and
    # no population criteria or covariates; never silently discard a plan item.
    exposures = [v for v in context.variables if v.name == context.primary_exposure]
    primary = [step for step in plan.steps if step.planned_analysis_role == "primary"]
    risk = [step for step in plan.steps if step.method == "signed_care_unit_landmark_cohort"]
    windows = [w for w in context.time_windows
               if len(exposures) == 1 and w.name == exposures[0].analysis_window]
    if (plan.analysis_type not in {"association", "association_study"} or len(plan.steps) != 2
            or len(risk) != 1 or len(exposures) != 1 or exposures[0].source_concept != "lact" or len(primary) != 1
            or len(primary[0].model_requirements) != 1 or primary[0].model_requirements[0].covariates
            or primary[0].model_requirements[0].exposure_source != context.primary_exposure
            or len(windows) != 1 or windows[0].anchor != SIC_CARE_UNIT_ORIGIN
            or windows[0].start_hours != 0 or windows[0].end_hours != 24
            or exposures[0].analysis_window_role != "exposure_definition"
            or plan.steps.index(risk[0]) >= plan.steps.index(primary[0])
            or set(risk[0].inputs) != {context.primary_exposure, "care_unit_to_case_closure_days",
                                       "death_during_case", "death_time_during_case_hours"}
            or "artifact:analysis_cohort" not in primary[0].inputs
            or plan.cohort is None or plan.cohort.inclusion or plan.cohort.exclusion
            or plan.cohort.selection_mode != "all_input_rows" or plan.cohort.unapplied_population_criteria):
        raise ValueError("sic_raw_preparation_plan_scope_unsupported")
    record = plan_result.evidence.get("scientific_plan_review")
    review_path = verified_run_evidence_path(run_dir, record) if record is not None else None
    if review_path is None:
        raise ValueError("sic_raw_preparation_review_missing")
    review = PlanScientificReview.model_validate_json(review_path.read_bytes())
    fact = review.facts.get("database_case_duration_review") or {}
    metadata = fact.get("metadata_plan_receipt")
    try:
        metadata_is_canonical = isinstance(metadata, dict) and CaseDurationMetadataPlanReceipt(**metadata).to_dict() == metadata
    except (ValueError, TypeError):
        metadata_is_canonical = False
    if (review.plan_sha256 != plan_sha or review.context_sha256 != context_sha
            or any(item.severity == "blocker" or item.requires_user_authorization for item in review.findings)
            or fact.get("selection") != outcome.model_dump(mode="json")
            or fact.get("metadata_status") != "source_metadata_verified_not_executable"
            or not metadata_is_canonical or canonical_sha256(metadata) != requested_scope.metadata_receipt_sha256
            or metadata.get("source_identity_sha256") != origin.source_identity_sha256):
        raise ValueError("sic_raw_preparation_review_mismatch")
    with AnchoredDirectory.open(Path(run_dir)) as directory:
        payload = directory.read_bytes("sic_raw_preparation_checkpoint.json", max_bytes=1024 * 1024)
        directory.assert_still_selected()
    checkpoint = SICRawPreparationCheckpoint.model_validate_json(payload)
    verify_raw_preparation_review(
        checkpoint=checkpoint, requested_scope=requested_scope.model_dump(mode="json"),
        scope_key="sic_raw_preparation_scope", expected_summary=render_sic_raw_preparation_scope(requested_scope),
        registered_database=identity.database, registered_identity_sha256=canonical_sha256(identity.to_dict()),
        plan_sha256=plan_sha, context_sha256=context_sha, scientific_review_sha256=record.sha256, now=now,
    )
    # Metadata-only recheck: a valid review cannot be reused after the source
    # registration, raw schema, resolver code or configured semantics drift.
    if prepare_case_duration_metadata_plan(source_path, expected_identity=identity).to_dict() != metadata:
        raise ValueError("sic_raw_preparation_source_metadata_changed")
    if sic_preparation_schema_sha256(source_path, requested_scope.files) != requested_scope.source_schema_sha256:
        raise ValueError("sic_raw_preparation_source_schema_changed")
    return VerifiedSICRawPreparationGrant(requested_scope, checkpoint.checkpoint_sha256, checkpoint.request.review_id, checkpoint.expires_at)
