"""Restore a prepared plan's input scope without restoring plan approval.

Budget names are not input identities. A scientific repair of a prepared plan
must retain its sealed cohort, while a fresh candidate remains metadata-only.
This owner joins the existing review, package, checkpoint and input authorities;
it never consumes a review or starts an extraction, Provider or executor.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from easyicu.research_agent.authority.run_input import load_verified_run_input_capsule
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.orchestration.human_review_checkpoint import load_checkpoint
from easyicu.research_agent.planning.scientific_review import (
    PlanScientificReview,
    plan_revision_blocker_codes,
)
from easyicu.webserver import agent_runs, dataio, study_contexts
from easyicu.webserver.agent_review_recovery import load_recovery_seed
from easyicu.webserver.research_input_progress import research_input_state
from easyicu.webserver.research_launch_resume import _slug
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from easyicu.webserver.run_record import RunRecordReadError


@dataclass(frozen=True)
class PreparedPlanRevision:
    run_dir: Path
    pipeline_config_sha256: str
    prepared_package_binding: Mapping[str, Any]
    prior_plan_contract: str | None
    required_primary_cohort_selection_mode: str | None
    input_capsule_sha256: str = ""
    budget_mode: str = "full_reviewed"


def load_prepared_plan_revision(
    *,
    study: Mapping[str, Any],
    project_root: str | None,
    source_run_id: str,
) -> PreparedPlanRevision | None:
    """Select only an exact, unapproved prepared plan needing Planner repair.

    A metadata candidate (or an absent history row) supplies no prepared-input
    authority. The existing candidate/revision owner still validates it later.
    Once a source claims prepared input, missing or drifted seals fail closed;
    they must not silently fall back to a new zero-row catalogue.
    """

    if not source_run_id:
        return None
    rows = agent_runs.list_run_history(
        study_id=str(study.get("id") or ""),
        project_root=project_root,
        limit=100,
    ).get("runs", ())
    row = next((item for item in rows if item.get("run_id") == source_run_id), None)
    if row is None:
        return None
    record = agent_runs.read_run_record(str(row.get("project_dir") or ""))
    if isinstance(record, RunRecordReadError):
        if row.get("research_input_state") != "prepared":
            return None
        raise ResearchPipelineRunError(
            "prepared_plan_revision_source_invalid",
            "The prepared plan record failed integrity validation.",
        )
    manifest = record.artifact_payloads.get("source_run_manifest.json") or {}
    if manifest.get("research_input_state") != "prepared":
        if row.get("research_input_state") == "prepared":
            raise ResearchPipelineRunError(
                "prepared_plan_revision_source_invalid",
                "The prepared input receipt is missing or inconsistent.",
            )
        return None
    try:
        digest = study_contexts.scientific_configuration_sha256(study)
        if row.get("scientific_configuration_sha256") != digest:
            raise ValueError("study configuration changed")
        review = PlanScientificReview.model_validate(
            record.artifact_payloads.get("scientific_plan_review.json")
        )
        if review.approval_allowed or plan_revision_blocker_codes(review.findings):
            raise ValueError("source is not a Planner-owned repair")
        root = Path(str(project_root or "")).expanduser().resolve()
        wrapper = Path(str(row.get("project_dir") or "")).expanduser()
        study_root = root / _slug(study.get("id"))
        if (
            wrapper.is_symlink()
            or study_root.is_symlink()
            or wrapper.parent.resolve() != study_root.resolve()
        ):
            raise ValueError("source is outside the exact study")
        if Path(source_run_id).name != source_run_id:
            raise ValueError("invalid run id")
        run_dir = wrapper / "pipeline" / source_run_id
        if run_dir.is_symlink() or run_dir.parent.is_symlink():
            raise ValueError("source is not a regular owned run")
        seed = load_recovery_seed(wrapper)
        if (
            seed is None
            or seed.schema_version
            not in {
                "easyicu.web-review-recovery-seed/3",
                "easyicu.web-review-recovery-seed/4",
            }
            or seed.budget_mode != "full_reviewed"
            or not seed.prepared_package_binding
            or seed.scientific_configuration_sha256 != digest
            or seed.study.get("id") != study.get("id")
            or study_contexts.scientific_configuration_sha256(seed.study) != digest
        ):
            raise ValueError("prepared launch scope is missing or changed")
        config = PipelineConfig.from_recovery_payload(
            seed.pipeline_config,
            expected_digest=seed.pipeline_config_sha256,
        )
        if (
            not config.require_human_plan_review
            or Path(config.workdir).resolve() != run_dir.parent.resolve()
        ):
            raise ValueError("source configuration does not require plan review")
        checkpoint = load_checkpoint(run_dir / "human_review_checkpoint.json")
        if (
            checkpoint.run_id != source_run_id
            or checkpoint.pipeline_config_sha256 != seed.pipeline_config_sha256
            or checkpoint.approved_decisions
            or checkpoint.execution_start_receipt is not None
        ):
            raise ValueError("source is not an unconsumed plan review")
        capsule_path = run_dir / "run_input_capsule.json"
        if capsule_path.is_symlink() or capsule_path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("invalid input capsule")
        raw = capsule_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != checkpoint.run_input_capsule_sha256:
            raise ValueError("checkpoint input identity changed")
        capsule = json.loads(raw)
        load_verified_run_input_capsule(
            run_dir=run_dir,
            scientific_identity=capsule["scientific_identity"],
        )
        if research_input_state(run_dir) != "prepared":
            raise ValueError("source is not physically prepared")
        source = study.get("data_source") or {}
        dataio.validate_research_pipeline_source(
            str(source.get("path") or ""),
            database=source.get("database"),
            expected_binding=seed.prepared_package_binding,
        )
        return PreparedPlanRevision(
            run_dir=run_dir.resolve(),
            pipeline_config_sha256=seed.pipeline_config_sha256,
            prepared_package_binding=dict(seed.prepared_package_binding),
            prior_plan_contract=config.bound_plan_revision_contract,
            required_primary_cohort_selection_mode=config.required_primary_cohort_selection_mode,
            input_capsule_sha256=checkpoint.run_input_capsule_sha256,
        )
    except Exception as exc:
        # No raw paths, source contents or old Provider environment in errors.
        raise ResearchPipelineRunError(
            "prepared_plan_revision_source_invalid",
            "The prepared plan's study, package, review or sealed inputs changed; "
            "no extraction, planning or execution was started.",
        ) from exc


__all__ = ["PreparedPlanRevision", "load_prepared_plan_revision"]
